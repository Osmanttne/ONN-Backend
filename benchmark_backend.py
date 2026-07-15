"""Backend benchmark — run AFTER check_bench.py shows 7/7.

    py -3.10 benchmark_backend.py            # full benchmark (laser stays at default power)
    py -3.10 benchmark_backend.py --no-laser # skip laser (emission stays off)

Measures, against real hardware:
  1. per-device command latency (laser query, DMD upload+project, SLM write, camera grab)
  2. end-to-end forward-pass step time and throughput (inputs/second)
  3. measurement noise floor: repeat the same input N times, spread of Y
  4. response contrast: |Y(x) - Y(~x)| vs noise — is the optical signal real?
  5. settle-time sweep: how small can settle_s go before Y degrades

Results print as a table and save to data/benchmark.json.
CAUTION: this enables laser emission at the profile's default power (10 mW)
unless --no-laser is passed. Make sure the bench is safe before running.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np

from hardware.base import load_profile
from onn.forward import ONNForward

RESULTS: dict = {}


def timeit(fn, n=10, warmup=2):
    """Return (mean_ms, stdev_ms, min_ms) over n calls after warmup."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(ts), (statistics.stdev(ts) if n > 1 else 0.0), min(ts)


def row(name, mean, sd, mn, unit="ms"):
    print(f"  {name:34} {mean:9.2f} ± {sd:6.2f} {unit}   (best {mn:.2f})")
    RESULTS[name] = {"mean": mean, "stdev": sd, "min": mn, "unit": unit}


def main():
    use_laser = "--no-laser" not in sys.argv
    profile = load_profile("config/onn_nico.yaml")

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

    print("connecting bench...")
    for dev in (laser, dmd, slm, camera):
        dev.connect()
        print(f"  {dev.name}: ready")
    if use_laser:
        laser.emission_on()
        print(f"  laser emitting at {laser.get_power_mw():.1f} mW")
        time.sleep(2.0)  # let power stabilize

    h, w = profile["onn"]["input_shape"]
    rng = np.random.default_rng(0)
    x = (rng.random((h, w)) > 0.5).astype(np.uint8)

    print("\n--- 1. per-device latency ---")
    m, s, mn = timeit(lambda: laser.get_power_mw(), n=10)
    row("laser: power query (serial)", m, s, mn)
    m, s, mn = timeit(lambda: dmd.project_input(x), n=10)
    row("dmd: encode+upload+project", m, s, mn)
    phase = np.zeros((profile["slm"]["resolution"][1],
                      profile["slm"]["resolution"][0]), dtype=np.uint8)
    m, s, mn = timeit(lambda: slm.write(phase), n=10)
    row("slm: full-frame phase write", m, s, mn)
    m, s, mn = timeit(lambda: camera.grab(), n=20)
    row("camera: single frame grab", m, s, mn)
    grab_ms = m

    print("\n--- 2. forward pass ---")
    onn = ONNForward.from_profile(dmd, camera, profile)
    m, s, mn = timeit(lambda: onn.step(x), n=15)
    row("forward step (project+settle+grab)", m, s, mn)
    step_ms = m
    print(f"  -> throughput: {1000.0 / step_ms:.2f} inputs/second "
          f"(settle_s = {onn.settle_s * 1000:.0f} ms of that)")
    RESULTS["throughput_inputs_per_s"] = 1000.0 / step_ms

    print("\n--- 3. repeatability (noise floor) ---")
    ys = np.stack([onn.step(x)[0] for _ in range(20)])
    noise = ys.std(axis=0).mean()
    drift = np.abs(ys[-5:].mean(axis=0) - ys[:5].mean(axis=0)).mean()
    print(f"  same input x, 20 reps: per-element std = {noise:.5f}, "
          f"first5-vs-last5 drift = {drift:.5f}")
    RESULTS["noise_floor"] = noise
    RESULTS["drift"] = float(drift)

    print("\n--- 4. response contrast ---")
    ya = ys.mean(axis=0)
    yb = np.stack([onn.step(1 - x)[0] for _ in range(10)]).mean(axis=0)
    contrast = np.abs(ya - yb).mean()
    snr = contrast / max(noise, 1e-12)
    print(f"  |Y(x) - Y(~x)| = {contrast:.5f}  ->  SNR = {snr:.1f}x noise floor")
    RESULTS["contrast"] = float(contrast)
    RESULTS["snr"] = float(snr)
    verdict = "GOOD - optical signal well above noise" if snr > 10 else \
              "MARGINAL - increase laser power, exposure, or frames_per_input" if snr > 3 else \
              "BAD - Y barely responds to X; check alignment/ROI/exposure"
    print(f"  verdict: {verdict}")

    print("\n--- 5. settle-time sweep ---")
    y_ref = ya
    original = onn.settle_s
    for settle in (0.2, 0.1, 0.05, 0.02, 0.01, 0.0):
        onn.settle_s = settle
        y = np.stack([onn.step(x)[0] for _ in range(5)]).mean(axis=0)
        err = np.abs(y - y_ref).mean()
        ok = "ok" if err < 3 * noise else "DEGRADED"
        print(f"  settle {settle * 1000:5.0f} ms: deviation {err:.5f}  [{ok}]")
        RESULTS[f"settle_{int(settle * 1000)}ms_err"] = float(err)
    onn.settle_s = original

    print("\nshutting down...")
    dmd.free()
    slm.blank()
    if use_laser:
        laser.emission_off()
    for dev in (camera, slm, dmd, laser):
        dev.disconnect()

    import pathlib
    pathlib.Path("data").mkdir(exist_ok=True)
    with open("data/benchmark.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("saved -> data/benchmark.json")

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"  camera frame:      {grab_ms:.1f} ms")
    print(f"  forward step:      {step_ms:.1f} ms  "
          f"({RESULTS['throughput_inputs_per_s']:.2f} inputs/s)")
    print(f"  noise floor:       {RESULTS['noise_floor']:.5f}")
    print(f"  contrast SNR:      {RESULTS['snr']:.1f}x")


if __name__ == "__main__":
    main()
