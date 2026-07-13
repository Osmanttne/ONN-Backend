"""Meadowlark Blink HDMI SLM (1920x1152, 8-bit) driver.

Prefers the maintained slmsuite wrapper; the coverglass-voltage safety
lockout is enforced here in the driver so no code path can bypass it.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np

from .base import Device, DeviceState, SafetyLockError

log = logging.getLogger("onn.slm")

try:
    from slmsuite.hardware.slms.meadowlark import Meadowlark
    _HAS_SLMSUITE = True
except ImportError:  # pragma: no cover
    _HAS_SLMSUITE = False


class SlmMeadowlark(Device):
    name = "slm"

    def __init__(self, lut_file: str, wfc_file: str,
                 resolution=(1920, 1152), bit_depth: int = 8,
                 coverglass_threshold_v: float = 6.171,
                 temp_warn_c: float = 40.0, sdk_path: str | None = None):
        super().__init__()
        if not _HAS_SLMSUITE:
            raise ImportError("slmsuite not installed / Blink SDK DLL not found")
        self.lut_file = pathlib.Path(lut_file)
        self.wfc_file = pathlib.Path(wfc_file)   # .bmp (confirmed format)
        self.resolution = tuple(resolution)
        self.bit_depth = bit_depth
        self.coverglass_threshold_v = coverglass_threshold_v
        self.coverglass_locked = False
        self.temp_warn_c = temp_warn_c
        self._sdk_path = sdk_path
        self._slm = None
        self._wfc = None

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        if not self.lut_file.exists():
            raise FileNotFoundError(f"LUT file missing: {self.lut_file} — "
                                    "the correct LUT is mandatory for valid results")
        kwargs = {"lut_path": str(self.lut_file)}
        if self._sdk_path:
            kwargs["sdk_path"] = self._sdk_path
        self._slm = Meadowlark(**kwargs)
        self._load_wfc()
        log.info("SLM connected, LUT=%s, WFC=%s", self.lut_file.name, self.wfc_file.name)
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        if self._slm:
            try:
                self.blank()
            finally:
                self._slm.close()
                self._slm = None
        self.state = DeviceState.DISCONNECTED

    def _load_wfc(self):
        """Load the .bmp wavefront-correction image; added to every write."""
        if self.wfc_file.exists():
            from PIL import Image
            self._wfc = np.asarray(Image.open(self.wfc_file).convert("L"), dtype=np.uint16)
        else:
            log.warning("WFC file missing (%s) — writing patterns uncorrected", self.wfc_file)
            self._wfc = None

    # -- pattern output ----------------------------------------------------
    def write(self, phase: np.ndarray):
        """Write an 8-bit phase image (H, W) to the SLM, WFC applied."""
        self._require_ready()
        phase = np.asarray(phase)
        if phase.shape != (self.resolution[1], self.resolution[0]):
            raise ValueError(f"phase {phase.shape} != SLM {self.resolution[::-1]}")
        levels = 2 ** self.bit_depth
        if self._wfc is not None and self._wfc.shape == phase.shape:
            phase = (phase.astype(np.uint16) + self._wfc) % levels
        self._slm.write(phase.astype(np.uint8))

    def blank(self):
        self.write(np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8))

    # -- telemetry & safety --------------------------------------------------
    def get_temperature_c(self) -> float:
        t = float(self._slm.get_temperature())
        if t > self.temp_warn_c:
            log.warning("SLM temperature %.2f C exceeds warn limit %.1f C", t, self.temp_warn_c)
        return t

    def get_coverglass_v(self) -> float:
        return float(self._slm.slm_lib.Get_cover_voltage())  # Blink SDK call

    def set_coverglass_v(self, volts: float):
        """Blocked permanently once the threshold has been reached."""
        self._check_coverglass_lock()
        if volts >= self.coverglass_threshold_v:
            raise SafetyLockError(
                f"refusing to set coverglass to {volts} V >= threshold "
                f"{self.coverglass_threshold_v} V")
        self._slm.slm_lib.Set_cover_voltage(float(volts))

    def poll_coverglass(self) -> dict:
        """Call periodically: latches the lock the moment threshold is reached."""
        v = self.get_coverglass_v()
        if not self.coverglass_locked and v >= self.coverglass_threshold_v:
            self.coverglass_locked = True
            log.info("coverglass reached %.3f V — LOCKED (threshold %.3f V)",
                     v, self.coverglass_threshold_v)
        return {"voltage_v": v, "locked": self.coverglass_locked}

    def _check_coverglass_lock(self):
        if self.coverglass_locked:
            raise SafetyLockError(
                "coverglass voltage is locked at threshold; changing it can "
                "damage the SLM and is not permitted")

    def status(self) -> dict:
        s = {"state": self.state.value, "lut": self.lut_file.name,
             "wfc": self.wfc_file.name, "coverglass_locked": self.coverglass_locked}
        if self._slm and self.state is not DeviceState.DISCONNECTED:
            s["temperature_c"] = self.get_temperature_c()
            s.update(self.poll_coverglass())
        return s
