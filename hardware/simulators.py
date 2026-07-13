"""Simulator twins — identical interfaces to the real drivers, zero hardware.

SimCamera is wired to SimDmd and SimSlm through a SimBench object: the frame
it returns is a physically-flavored fake — the DMD pattern pooled to a grid,
passed through a fixed random linear "optical" transform (seeded, so runs are
reproducible), modulated by mean SLM phase, rendered with blur + shot noise.
That makes the notebook's forward pass produce a meaningful Y that genuinely
depends on X, which is exactly what we need to validate the pipeline shape
before touching the bench.
"""
from __future__ import annotations

import numpy as np

from .base import Device, DeviceState, SafetyLockError
from .dmd import encode_macropixels


class SimBench:
    """Shared optical state linking the simulated devices."""

    def __init__(self, mix_grid=(8, 8), seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.mix_grid = mix_grid
        n = mix_grid[0] * mix_grid[1]
        self.mixing = np.abs(self.rng.normal(0.4, 0.35, (n, n)))  # fixed "optics"
        self.laser_power_mw = 0.0
        self.emitting = False
        self.dmd_frame: np.ndarray | None = None
        self.slm_phase: np.ndarray | None = None


class SimLaser(Device):
    name = "laser"

    def __init__(self, bench: SimBench, default_power_mw=10.0, max_power_mw=30.0, **_):
        super().__init__()
        self.bench = bench
        self.default_power_mw, self.max_power_mw = default_power_mw, max_power_mw

    def connect(self):
        self.set_power_mw(self.default_power_mw)
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        self.emission_off()
        self.state = DeviceState.DISCONNECTED

    def set_power_mw(self, mw):
        if mw > self.max_power_mw:
            raise ValueError(f"{mw} mW exceeds software cap {self.max_power_mw} mW")
        self.bench.laser_power_mw = float(mw)

    def get_power_mw(self):
        return self.bench.laser_power_mw

    def emission_on(self):
        self.bench.emitting = True
        self.state = DeviceState.RUNNING

    def emission_off(self):
        self.bench.emitting = False
        if self.state is DeviceState.RUNNING:
            self.state = DeviceState.READY

    def status(self):
        return {"state": self.state.value, "wavelength_nm": 660.0,
                "power_mw": self.bench.laser_power_mw,
                "emitting": self.bench.emitting,
                "diode_temp_c": 25.0, "baseplate_temp_c": 20.9, "faults": "0"}


class SimDmd(Device):
    name = "dmd"

    def __init__(self, bench: SimBench, shape=(768, 1024),
                 macropixel="auto", invert=False, **_):
        super().__init__()
        self.bench, self.shape = bench, shape
        self.macropixel, self.invert = macropixel, invert

    def connect(self):
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        self.free()
        self.state = DeviceState.DISCONNECTED

    def project(self, frame, loop=True):
        self._require_ready()
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != self.shape:
            raise ValueError(f"frame {frame.shape} != DMD {self.shape}")
        self.bench.dmd_frame = frame
        self.state = DeviceState.RUNNING

    def project_input(self, x):
        frame = encode_macropixels(x, self.shape, self.macropixel, self.invert)
        self.project(frame)
        return frame

    def project_sequence(self, frames, picture_time_us=None, loop=False):
        self.project(np.asarray(frames)[0])  # sim: show first frame

    def stop(self):
        self.state = DeviceState.READY

    def free(self):
        self.bench.dmd_frame = None
        if self.state is DeviceState.RUNNING:
            self.state = DeviceState.READY

    def status(self):
        return {"state": self.state.value, "shape": self.shape,
                "sequence_loaded": self.bench.dmd_frame is not None}


class SimSlm(Device):
    name = "slm"

    def __init__(self, bench: SimBench, lut_file="sim.lut", wfc_file="sim.bmp",
                 resolution=(1920, 1152), coverglass_threshold_v=6.171,
                 temp_warn_c=40.0, **_):
        super().__init__()
        self.bench = bench
        self.lut_file, self.wfc_file = lut_file, wfc_file
        self.resolution = tuple(resolution)
        self.coverglass_threshold_v = coverglass_threshold_v
        self.coverglass_locked = False
        self._coverglass_v = 5.5   # starts below threshold, auto-adjusts upward
        self.temp_warn_c = temp_warn_c

    def connect(self):
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        self.state = DeviceState.DISCONNECTED

    def write(self, phase):
        self._require_ready()
        phase = np.asarray(phase)
        if phase.shape != (self.resolution[1], self.resolution[0]):
            raise ValueError(f"phase {phase.shape} != SLM {self.resolution[::-1]}")
        self.bench.slm_phase = phase.astype(np.uint8)

    def blank(self):
        self.write(np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8))

    def get_temperature_c(self):
        return 22.8 + 0.05 * np.random.randn()

    def get_coverglass_v(self):
        # emulate the hardware auto-adjusting toward the threshold
        if not self.coverglass_locked:
            self._coverglass_v = min(self._coverglass_v + 0.25,
                                     self.coverglass_threshold_v)
        return self._coverglass_v

    def set_coverglass_v(self, volts):
        if self.coverglass_locked:
            raise SafetyLockError("coverglass voltage is locked at threshold; "
                                  "changing it can damage the SLM")
        if volts >= self.coverglass_threshold_v:
            raise SafetyLockError(f"refusing {volts} V >= threshold "
                                  f"{self.coverglass_threshold_v} V")
        self._coverglass_v = float(volts)

    def poll_coverglass(self):
        v = self.get_coverglass_v()
        if not self.coverglass_locked and v >= self.coverglass_threshold_v:
            self.coverglass_locked = True
        return {"voltage_v": v, "locked": self.coverglass_locked}

    def status(self):
        return {"state": self.state.value, "lut": str(self.lut_file),
                "wfc": str(self.wfc_file),
                "temperature_c": self.get_temperature_c(),
                **self.poll_coverglass()}


class SimCamera(Device):
    name = "camera"

    def __init__(self, bench: SimBench, preset=None, shape=(1080, 1440), **_):
        super().__init__()
        self.bench, self.shape = bench, shape
        self.preset = preset or {}
        self.rng = np.random.default_rng(1)

    def connect(self):
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        self.state = DeviceState.DISCONNECTED

    def apply_preset(self, preset):
        self.preset = preset

    def start(self):
        self.state = DeviceState.RUNNING

    def stop(self):
        if self.state is DeviceState.RUNNING:
            self.state = DeviceState.READY

    def grab(self, timeout_ms: int = 2000) -> np.ndarray:
        self._require_ready()
        b = self.bench
        gh, gw = b.mix_grid
        frame = np.zeros(self.shape)
        if b.emitting and b.laser_power_mw > 0 and b.dmd_frame is not None:
            # pool DMD to the mixing grid
            d = b.dmd_frame.astype(np.float64) / 255.0
            H, W = d.shape
            pooled = d[:H - H % gh, :W - W % gw]
            pooled = pooled.reshape(gh, H // gh, gw, W // gw).mean(axis=(1, 3))
            # SLM phase scales the mixing (crude stand-in for modulation)
            slm_factor = 1.0
            if b.slm_phase is not None:
                slm_factor = 0.5 + b.slm_phase.mean() / 255.0
            y = (b.mixing * slm_factor) @ pooled.ravel()
            y = y.reshape(gh, gw) * (b.laser_power_mw / 10.0)
            # render grid as blocks on the sensor
            frame = np.kron(y, np.ones((self.shape[0] // gh, self.shape[1] // gw)))
            frame = frame[:self.shape[0], :self.shape[1]]
        # normalize to 8-bit-ish scale, add shot + read noise
        if frame.max() > 0:
            frame = frame / frame.max() * 200.0
        frame = frame + self.rng.poisson(2.0, self.shape) + self.rng.normal(0, 1.0, self.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def grab_mean(self, n: int = 1, timeout_ms: int = 2000) -> np.ndarray:
        return np.mean([self.grab().astype(np.float64) for _ in range(max(1, n))], axis=0)

    def status(self):
        return {"state": self.state.value, "model": "Sim Grasshopper3",
                "shape": self.shape}


def make_sim_bench(profile: dict, seed: int = 0):
    """Build the full simulated bench from a profile dict."""
    grid = tuple(profile.get("onn", {}).get("detector_grid", [8, 8]))
    bench = SimBench(mix_grid=grid, seed=seed)
    laser = SimLaser(bench,
                     default_power_mw=profile["laser"]["default_power_mw"],
                     max_power_mw=profile["laser"]["max_power_mw"])
    dmd = SimDmd(bench, macropixel=profile["dmd"]["macropixel"],
                 invert=profile["dmd"]["invert"])
    slm = SimSlm(bench, lut_file=profile["slm"]["lut_file"],
                 wfc_file=profile["slm"]["wfc_file"],
                 resolution=profile["slm"]["resolution"],
                 coverglass_threshold_v=profile["slm"]["coverglass"]["threshold_v"])
    camera = SimCamera(bench, preset=profile["camera"])
    return bench, laser, dmd, slm, camera
