"""Bench hardware server — run this ON THE BENCH PC (the one with the ports).

    py -3.10 onn_server.py             # connect all devices, serve on port 8765
    py -3.10 onn_server.py --no-laser  # devices up, emission stays off

It is THE single process that owns the laser/DMD/SLM/camera. Everything else
(cluster notebooks, scripts, GUIs) talks to it over HTTP:

    GET  /status     -> per-device status + the config actually loaded
    POST /forward    {"X": [[..]] or [[[..]]], "keep_frames": bool}
                     -> {"Y": [...], "t_wall": [...]}
    GET  /frame      -> one camera frame
    POST /laser      {"power_mw": 12.0} and/or {"emission": true/false}
    POST /slm        {"phase": [[..]], "vmin":?, "vmax":?, "wrap":?} | {"blank": true}
    POST /metrics    {"n": 20, "warmup": 6}
                     -> interleaved ON/OFF contrast: mean, std, sem, snr
    POST /calibrate  {"target_peak": 200, "n": 20, "warmup": 6, "write_yaml": true}
                     -> sweeps exposure, finds the ROI from the ON/OFF diff map,
                        verifies contrast, optionally writes both back to the yaml

Optional auth: set the ONN_TOKEN environment variable before starting, and every
request must then carry the header  X-ONN-Token: <value>.  Unset = open (default).

Ctrl+C shuts the bench down safely; one dead device can no longer prevent the
others from being released.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from hardware.base import load_profile
from onn.forward import ONNForward, pool_to_grid
from onn.tensor_bitmap import to_slm_phase

PORT = 8765
PROFILE_PATH = "config/onn_nico.yaml"
TOKEN = os.environ.get("ONN_TOKEN")            # None = auth disabled

EXPOSURE_MIN_US, EXPOSURE_MAX_US = 20.0, 30000.0


# --------------------------------------------------------------- bench setup
def build_bench(profile, use_laser=True):
    from hardware.laser import LaserOBIS
    from hardware.dmd import DmdViALUX
    from hardware.slm import SlmMeadowlark
    from hardware.camera import CameraFLIR
    laser = LaserOBIS(port=profile["laser"]["port"],
                      default_power_mw=profile["laser"]["default_power_mw"],
                      max_power_mw=profile["laser"]["max_power_mw"])
    dmd = DmdViALUX(macropixel=profile["dmd"]["macropixel"],
                    invert=profile["dmd"]["invert"],
                    picture_time_us=profile["dmd"]["picture_time_us"])
    slm = SlmMeadowlark(lut_file=profile["slm"]["lut_file"],
                        wfc_file=profile["slm"]["wfc_file"],
                        resolution=profile["slm"]["resolution"],
                        coverglass_threshold_v=profile["slm"]["coverglass"]["threshold_v"])
    camera = CameraFLIR(preset=profile["camera"])
    for dev in (laser, dmd, slm, camera):
        dev.connect()
        print(f"  {dev.name}: ready")
    if use_laser:
        laser.emission_on()
        time.sleep(1.0)
        print(f"  laser emitting at {laser.get_power_mw():.1f} mW")
    return laser, dmd, slm, camera


def shutdown_bench(laser, dmd, slm, camera, use_laser=True):
    """Release everything. Each step is independent: a dead device must never
    prevent the others from being freed (that cascade orphans the DMD)."""
    print("\nshutting bench down safely...")
    for label, action in (("dmd free",  lambda: dmd.free()),
                          ("slm blank", lambda: slm.blank()),
                          ("laser off", lambda: laser.emission_off() if use_laser else None),
                          ("camera disconnect", lambda: camera.disconnect()),
                          ("slm disconnect",    lambda: slm.disconnect()),
                          ("dmd disconnect",    lambda: dmd.disconnect()),
                          ("laser disconnect",  lambda: laser.disconnect())):
        try:
            action()
        except Exception as e:
            print(f"  [warn] {label}: {e}")
    print("done")


# ----------------------------------------------------------- camera exposure
def get_exposure_us(camera):
    if hasattr(camera, "exposure_us"):
        return float(camera.exposure_us)
    cam = getattr(camera, "_cam", None)
    if cam is not None:
        try:
            return float(cam.ExposureTime.GetValue())
        except Exception:
            pass
    return float("nan")


def set_exposure_us(camera, us):
    """Live exposure change — no server restart. Returns True if it took."""
    us = float(np.clip(us, EXPOSURE_MIN_US, EXPOSURE_MAX_US))
    if hasattr(camera, "set_exposure_us"):
        camera.set_exposure_us(us)
        return True
    cam = getattr(camera, "_cam", None)
    if cam is not None:
        try:
            cam.ExposureTime.SetValue(us)
            return True
        except Exception:
            return False
    return False


# ------------------------------------------------------------ measurements
def settled_grab(camera, flush=6):
    """Grab a frame after discarding queued ones.

    GenICam pipelines buffer several frames, so the first grabs after any
    change (exposure, DMD pattern) can still show the OLD state. Every
    measurement here goes through this or it reads the past.
    """
    f = None
    for _ in range(max(1, int(flush))):
        f = camera.grab()
    return f


def contrast_metric(onn, n=20, warmup=6):
    """Interleaved ON/OFF paired contrast with warm-up discard.

    Alternating ON,OFF,ON,OFF... so slow drift cancels within each pair; the
    first `warmup` pairs are dropped (a cold bench needs a few measurements to
    settle) and the statistics come from the steady state.
    """
    n = int(max(2, n))
    warmup = int(np.clip(warmup, 0, n - 2))
    stack = np.zeros((2 * n, 4, 4), dtype=int)
    stack[0::2] = 1
    Y = onn.forward(stack, progress=False).Y
    on = Y[0::2].reshape(n, -1).mean(axis=1)
    off = Y[1::2].reshape(n, -1).mean(axis=1)
    d = on - off
    d_ss = d[warmup:]
    std = float(d_ss.std(ddof=1)) if len(d_ss) > 1 else 0.0
    mean = float(d_ss.mean())
    return {
        "n": n, "warmup_discarded": warmup,
        "contrast_mean": mean,
        "contrast_std": std,
        "sem": (std / np.sqrt(len(d_ss))) if len(d_ss) > 1 else 0.0,
        "snr": (abs(mean) / std) if std > 0 else float("inf"),
        "on_mean": float(on[warmup:].mean()), "off_mean": float(off[warmup:].mean()),
        "per_run": [float(v) for v in d],
    }


def sweep_exposure(onn, dmd, camera, target_peak=200.0, max_iter=14, flush=6):
    """Bisect exposure until the all-ON frame peaks near target_peak.

    Guards against the classic trap: if the peak barely moves while exposure
    changes a lot, the frames are stale — the flush depth is raised and the
    point re-measured rather than believing the reading.
    """
    trace, notes = [], []
    dmd.project_input(np.ones((4, 4), dtype=int))
    time.sleep(max(onn.settle_s, 0.05))
    settled_grab(camera, flush)

    exposure = get_exposure_us(camera)
    if not np.isfinite(exposure) or exposure <= 0:
        exposure = 300.0
    lo_known = hi_known = None
    best = None

    def probe(exp_us, fl):
        if not set_exposure_us(camera, exp_us):
            return None, None
        time.sleep(0.2 + 3e-6 * exp_us)
        peak = float(settled_grab(camera, fl).max())
        return float(get_exposure_us(camera)), peak

    for _ in range(max_iter):
        readback, peak = probe(exposure, flush)
        if readback is None:
            return {"ok": False, "reason": "camera exposure not settable from Python",
                    "trace": trace}
        # stale-frame detector: big exposure change, negligible peak change
        if trace:
            de = abs(readback - trace[-1]["exposure_us"]) / max(trace[-1]["exposure_us"], 1)
            dp = abs(peak - trace[-1]["peak"])
            if de > 0.3 and dp < 3 and peak < 250:
                flush = min(24, flush * 2)
                notes.append(f"stale frames suspected -> flush raised to {flush}")
                readback, peak = probe(exposure, flush)
        trace.append({"exposure_us": round(readback, 1), "peak": peak})
        if best is None or abs(peak - target_peak) < abs(best[1] - target_peak):
            best = (exposure, peak)
        if 0.85 * target_peak <= peak <= 1.12 * target_peak:
            break
        if peak >= 250:
            # saturated: the reading carries no gradient, so step down hard
            hi_known = exposure
            exposure = (0.5 * (lo_known + exposure)) if lo_known is not None else exposure * 0.35
        elif peak < target_peak:
            lo_known = exposure
            exposure = (0.5 * (lo_known + hi_known)) if hi_known is not None else \
                exposure * float(np.clip(target_peak / max(peak, 1.0), 1.2, 3.0))
        else:
            hi_known = exposure
            exposure = (0.5 * (lo_known + exposure)) if lo_known is not None else \
                exposure * float(np.clip(target_peak / max(peak, 1.0), 0.3, 0.85))
        exposure = float(np.clip(exposure, EXPOSURE_MIN_US, EXPOSURE_MAX_US))

    if best is not None:
        set_exposure_us(camera, best[0])
        time.sleep(0.2)
        settled_grab(camera, flush)
    dmd.stop()
    final_peak = trace[-1]["peak"] if trace else None
    ok = best is not None and best[1] < 250
    return {"ok": ok, "exposure_us": round(float(get_exposure_us(camera)), 1),
            "final_peak": best[1] if best else final_peak, "flush_used": flush,
            "notes": notes, "trace": trace,
            "reason": None if ok else "could not find an unsaturated exposure"}


def find_roi(frame_on, frame_off, grid=(4, 4), pad=20, max_area_frac=0.5):
    """Locate the region where the DMD's light actually lands.

    Subtracting the two frames cancels everything static; whichever sign
    dominates (ON-bright or OFF-bright) is the region we read. The threshold
    is tightened until the bounding box is compact — a box covering most of
    the sensor means "found nothing", not "signal everywhere".
    """
    if float(frame_on.max()) >= 254 or float(frame_off.max()) >= 254:
        return {"ok": False, "reason": "frames are saturated - fix exposure before ROI"}
    diff = frame_on.astype(np.float64) - frame_off.astype(np.float64)
    sign = 1 if abs(diff.max()) >= abs(diff.min()) else -1
    s = diff * sign
    peak = float(s.max())
    if peak < 5.0:
        return {"ok": False, "reason": f"no modulated light found (peak diff {peak:.1f})"}

    H, W = frame_on.shape
    gh, gw = grid
    chosen = None
    for thr in (0.25, 0.4, 0.55, 0.7, 0.82):
        mask = s > thr * peak
        rows, cols = mask.mean(axis=1), mask.mean(axis=0)
        if rows.max() <= 0 or cols.max() <= 0:
            continue
        r_idx = np.where(rows > 0.25 * rows.max())[0]
        c_idx = np.where(cols > 0.25 * cols.max())[0]
        if len(r_idx) == 0 or len(c_idx) == 0:
            continue
        y0, y1 = max(0, int(r_idx[0]) - pad), min(H, int(r_idx[-1]) + pad + 1)
        x0, x1 = max(0, int(c_idx[0]) - pad), min(W, int(c_idx[-1]) + pad + 1)
        if ((y1 - y0) * (x1 - x0)) / float(H * W) <= max_area_frac:
            chosen = (x0, y0, x1, y1, thr)
            break
    if chosen is None:
        return {"ok": False, "reason": "modulated region too diffuse to bound "
                                       "(is the camera imaging the DMD plane?)"}

    x0, y0, x1, y1, thr = chosen
    h = max(gh, ((y1 - y0) // gh) * gh)
    w = max(gw, ((x1 - x0) // gw) * gw)
    roi = [int(x0), int(y0), int(w), int(h)]
    on_roi = float(pool_to_grid(frame_on, grid, roi).mean())
    off_roi = float(pool_to_grid(frame_off, grid, roi).mean())
    delta = abs(on_roi - off_roi)
    if delta < 3.0:
        return {"ok": False, "reason": f"candidate ROI has almost no contrast "
                                       f"({delta:.1f} counts)"}
    return {"ok": True, "roi": roi, "sign": "ON-bright" if sign > 0 else "OFF-bright",
            "threshold_used": thr, "peak_diff": peak, "roi_on": on_roi,
            "roi_off": off_roi, "roi_delta": delta,
            "area_frac": round((w * h) / float(H * W), 3)}


def write_yaml_values(path, roi=None, exposure_us=None):
    """Patch the yaml in place, preserving comments and layout."""
    try:
        with open(path) as f:
            text = f.read()
        stamp = time.strftime('%Y-%m-%d %H:%M')
        if roi is not None:
            text = re.sub(r"(?m)^(\s*detector_roi:).*$",
                          rf"\1 [{roi[0]}, {roi[1]}, {roi[2]}, {roi[3]}]"
                          f"   # set by /calibrate {stamp}", text)
        if exposure_us is not None:
            text = re.sub(r"(?m)^(\s*exposure_time_us:).*$",
                          rf"\1 {exposure_us}   # set by /calibrate {stamp}", text)
        with open(path, "w") as f:
            f.write(text)
        return {"written": True, "path": path}
    except Exception as e:
        return {"written": False, "error": str(e)}


def calibrate(onn, dmd, camera, profile, target_peak=200.0, n=20, warmup=6,
              write_yaml=True, flush=6):
    """The full tuning ritual in one call: exposure -> ROI -> verify.

    Safety: the pre-tuning contrast is measured first, and if the new settings
    come out WORSE the previous ROI/exposure are restored. Calibration must
    never leave a healthy bench in a worse state than it found it.
    """
    report = {}
    prev_roi = list(onn.detector_roi) if onn.detector_roi else None
    prev_exposure = get_exposure_us(camera)

    baseline = contrast_metric(onn, n=max(6, n // 2), warmup=min(warmup, max(2, n // 4)))
    report["baseline_metrics"] = {k: baseline[k] for k in
                                  ("contrast_mean", "contrast_std", "snr")}

    report["exposure"] = sweep_exposure(onn, dmd, camera, target_peak, flush=flush)
    flush = report["exposure"].get("flush_used", flush)

    dmd.project_input(np.ones((4, 4), dtype=int))
    time.sleep(max(onn.settle_s, 0.05))
    f_on = settled_grab(camera, flush).astype(np.float64)
    dmd.project_input(np.zeros((4, 4), dtype=int))
    time.sleep(max(onn.settle_s, 0.05))
    f_off = settled_grab(camera, flush).astype(np.float64)
    dmd.stop()

    roi_res = find_roi(f_on, f_off, grid=onn.detector_grid)
    report["roi"] = roi_res
    if roi_res.get("ok"):
        onn.detector_roi = tuple(roi_res["roi"])      # live, no restart needed
        profile["onn"]["detector_roi"] = roi_res["roi"]

    after = contrast_metric(onn, n=n, warmup=warmup)
    report["metrics"] = after

    worse = (baseline["snr"] > 3 and after["snr"] < baseline["snr"])
    if worse or not roi_res.get("ok"):
        if prev_roi is not None:
            onn.detector_roi = tuple(prev_roi)
            profile["onn"]["detector_roi"] = prev_roi
        if np.isfinite(prev_exposure):
            set_exposure_us(camera, prev_exposure)
            try:
                settled_grab(camera, flush)      # flush so the restored setting is live
            except Exception:
                pass
        report["reverted"] = True
        report["revert_reason"] = ("new settings scored worse than the previous ones"
                                   if worse else roi_res.get("reason"))
        report["restored"] = {"detector_roi": prev_roi,
                              "exposure_us": round(float(prev_exposure), 1)}
        report["ok"] = False
        return report

    report["reverted"] = False
    if write_yaml:
        report["yaml"] = write_yaml_values(PROFILE_PATH, roi=roi_res["roi"],
                                           exposure_us=report["exposure"].get("exposure_us"))
    report["ok"] = after["snr"] > 3
    return report


# ------------------------------------------------------------------- server
def main():
    use_laser = "--no-laser" not in sys.argv
    profile = load_profile(PROFILE_PATH)
    print("connecting bench...")
    laser, dmd, slm, camera = build_bench(profile, use_laser)
    onn = ONNForward.from_profile(dmd, camera, profile)
    if TOKEN:
        print("  auth: ON (X-ONN-Token header required)")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            if not TOKEN:
                return True
            if self.headers.get("X-ONN-Token") == TOKEN:
                return True
            self._send({"error": "missing or bad X-ONN-Token"}, 401)
            return False

        def _config_echo(self):
            return {"profile_path": PROFILE_PATH,
                    "detector_roi": list(onn.detector_roi) if onn.detector_roi else None,
                    "detector_grid": list(onn.detector_grid),
                    "settle_s": onn.settle_s,
                    "frames_per_input": onn.frames_per_input,
                    "exposure_us": get_exposure_us(camera)}

        def do_GET(self):
            if not self._authed():
                return
            try:
                if self.path == "/status":
                    s = {d.name: d.status() for d in (laser, dmd, slm, camera)}
                    s["config"] = self._config_echo()
                    self._send(s)
                elif self.path == "/frame":
                    self._send({"frame": camera.grab().tolist()})
                else:
                    self._send({"error": "unknown path"}, 404)
            except Exception as e:
                self._send({"error": str(e)}, 500)

        def do_POST(self):
            if not self._authed():
                return
            try:
                n_bytes = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n_bytes) or b"{}")

                if self.path == "/forward":
                    X = np.asarray(req["X"])
                    r = onn.forward(X, keep_frames=bool(req.get("keep_frames", False)),
                                    progress=False)
                    out = {"Y": r.Y.tolist(), "t_wall": r.t_wall.tolist()}
                    if r.frames is not None:
                        out["frames"] = r.frames.tolist()
                    self._send(out)

                elif self.path == "/slm":
                    if req.get("blank"):
                        slm.blank()
                        self._send({"ok": True, "blank": True})
                    else:
                        w = np.asarray(req["phase"], dtype=np.float64)
                        kw = {k: req[k] for k in ("vmin", "vmax", "wrap", "macropixel")
                              if req.get(k) is not None}
                        H, W = slm.resolution[1], slm.resolution[0]
                        img = to_slm_phase(w, (H, W), **kw)
                        slm.write(img)
                        self._send({"ok": True, "input_shape": list(w.shape),
                                    "phase_min": int(img.min()), "phase_max": int(img.max())})

                elif self.path == "/laser":
                    if "power_mw" in req:
                        laser.set_power_mw(float(req["power_mw"]))
                    if req.get("emission") is True:
                        laser.emission_on()
                    if req.get("emission") is False:
                        laser.emission_off()
                    self._send(laser.status())

                elif self.path == "/metrics":
                    self._send(contrast_metric(onn, n=int(req.get("n", 20)),
                                               warmup=int(req.get("warmup", 6))))

                elif self.path == "/calibrate":
                    self._send(calibrate(onn, dmd, camera, profile,
                                         target_peak=float(req.get("target_peak", 200.0)),
                                         n=int(req.get("n", 20)),
                                         warmup=int(req.get("warmup", 6)),
                                         write_yaml=bool(req.get("write_yaml", True))))
                else:
                    self._send({"error": "unknown path"}, 404)
            except Exception as e:
                self._send({"error": str(e)}, 500)

        def log_message(self, fmt, *args):
            print(f"  [{self.address_string()}] {fmt % args}")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\nserving bench on http://0.0.0.0:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_bench(laser, dmd, slm, camera, use_laser)


if __name__ == "__main__":
    main()
