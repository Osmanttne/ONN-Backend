"""Bench hardware server — run this ON THE BENCH PC (the one with the ports).

    py -3.10 onn_server.py            # connects all devices, serves on port 8765
    py -3.10 onn_server.py --no-laser # devices without emission

It is THE single process that owns the laser/DMD/SLM/camera. Everything else
(A100 notebooks, scripts, GUIs) talks to it over HTTP:

    POST /forward   body: {"X": [[...]] or [[[...]]]}   -> {"Y": [...], "t_wall": [...]}
    GET  /status                                        -> per-device status
    POST /laser     body: {"power_mw": 12.0} or {"emission": true/false}
    POST /slm       body: {"phase": [[...]] , "vmin":?, "vmax":?, "wrap":?,
                           "macropixel":?}  or {"blank": true}
                    numerical data -> 8-bit phase via to_slm_phase -> SLM
    GET  /frame                                         -> one camera frame (list)

Stop with Ctrl+C (shuts the bench down safely).
"""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from hardware.base import load_profile
from onn.forward import ONNForward
from onn.tensor_bitmap import to_slm_phase

PORT = 8765


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


def main():
    use_laser = "--no-laser" not in sys.argv
    profile = load_profile("config/onn_nico.yaml")
    print("connecting bench...")
    laser, dmd, slm, camera = build_bench(profile, use_laser)
    onn = ONNForward.from_profile(dmd, camera, profile)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/status":
                    self._send({d.name: d.status() for d in (laser, dmd, slm, camera)})
                elif self.path == "/frame":
                    self._send({"frame": camera.grab().tolist()})
                else:
                    self._send({"error": "unknown path"}, 404)
            except Exception as e:
                self._send({"error": str(e)}, 500)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
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
                        kw = {}
                        for k in ("vmin", "vmax", "wrap", "macropixel"):
                            if req.get(k) is not None:
                                kw[k] = req[k]
                        H, W = slm.resolution[1], slm.resolution[0]
                        img = to_slm_phase(w, (H, W), **kw)
                        slm.write(img)
                        self._send({"ok": True, "input_shape": list(w.shape),
                                    "phase_min": int(img.min()),
                                    "phase_max": int(img.max())})
                elif self.path == "/laser":
                    if "power_mw" in req:
                        laser.set_power_mw(float(req["power_mw"]))
                    if req.get("emission") is True:
                        laser.emission_on()
                    if req.get("emission") is False:
                        laser.emission_off()
                    self._send(laser.status())
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
        print("\nshutting bench down safely...")
        dmd.free()
        slm.blank()
        if use_laser:
            laser.emission_off()
        for dev in (camera, slm, dmd, laser):
            dev.disconnect()
        print("done")


if __name__ == "__main__":
    main()
