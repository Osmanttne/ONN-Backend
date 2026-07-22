"""FLIR Grasshopper3 GS3-U3-89S6M driver via the Spinnaker SDK (PySpin).

Design point from the requirements: the user never hand-tunes nodes —
apply_preset() pushes the whole known-good parameter set in one call.
"""
from __future__ import annotations

import logging

import numpy as np

from .base import Device, DeviceState

log = logging.getLogger("onn.camera")

try:
    import PySpin
    _HAS_PYSPIN = True
except ImportError:  # pragma: no cover
    _HAS_PYSPIN = False


class CameraFLIR(Device):
    name = "camera"

    def __init__(self, preset: dict | None = None, serial: str | None = None):
        super().__init__()
        if not _HAS_PYSPIN:
            raise ImportError("PySpin (Spinnaker SDK) not installed")
        self.preset = preset or {}
        self.serial = serial
        self._system = None
        self._cam = None
        self._acquiring = False

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        self._system = PySpin.System.GetInstance()
        cams = self._system.GetCameras()
        if cams.GetSize() == 0:
            cams.Clear()
            raise RuntimeError("no FLIR cameras found")
        self._cam = (cams.GetBySerial(self.serial) if self.serial else cams.GetByIndex(0))
        cams.Clear()
        self._cam.Init()
        model = self._cam.TLDevice.DeviceModelName.GetValue()
        log.info("connected to %s", model)
        if self.preset:
            self.apply_preset(self.preset)
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        if self._cam:
            try:
                self.stop()
            finally:
                self._cam.DeInit()
                del self._cam
                self._cam = None
        if self._system:
            self._system.ReleaseInstance()
            self._system = None
        self.state = DeviceState.DISCONNECTED

    # -- the efficiency feature -------------------------------------------
    def apply_preset(self, preset: dict):
        """Push the full parameter set from the profile in one shot.

        Every node is set best-effort: GenICam trees differ between camera
        models/firmware, so an absent or read-only node is skipped and
        reported rather than aborting the whole preset.
        """
        c = self._cam
        applied, skipped = [], []

        def attempt(label, fn):
            try:
                fn()
                applied.append(label)
            except Exception as e:
                skipped.append(f"{label} ({type(e).__name__})")

        def enum_set(node, value):
            getattr(c, node).SetValue(getattr(PySpin, f"{node}_{value}"))

        attempt("AcquisitionMode", lambda: enum_set(
            "AcquisitionMode", preset.get("acquisition_mode", "Continuous")))
        attempt("ExposureAuto", lambda: enum_set(
            "ExposureAuto", preset.get("exposure_auto", "Off")))
        if preset.get("exposure_auto", "Off") == "Off" and "exposure_time_us" in preset:
            attempt("ExposureTime", lambda: c.ExposureTime.SetValue(
                float(preset["exposure_time_us"])))
        attempt("GainAuto", lambda: enum_set(
            "GainAuto", preset.get("gain_auto", "Off")))
        if preset.get("gain_auto", "Off") == "Off" and "gain_db" in preset:
            attempt("Gain", lambda: c.Gain.SetValue(float(preset["gain_db"])))
        if "gamma" in preset:
            attempt("GammaEnable", lambda: c.GammaEnable.SetValue(True))
            attempt("Gamma", lambda: c.Gamma.SetValue(float(preset["gamma"])))
        if "black_level_pct" in preset:
            attempt("BlackLevel", lambda: c.BlackLevel.SetValue(
                float(preset["black_level_pct"])))
        if "acquisition_frame_rate_hz" in preset:
            attempt("AcqFrameRateEnable",
                    lambda: c.AcquisitionFrameRateEnable.SetValue(True))
            attempt("AcquisitionFrameRate", lambda: c.AcquisitionFrameRate.SetValue(
                float(preset["acquisition_frame_rate_hz"])))
        if "device_link_throughput_limit" in preset:
            attempt("DeviceLinkThroughputLimit",
                    lambda: c.DeviceLinkThroughputLimit.SetValue(
                        int(preset["device_link_throughput_limit"])))

        log.info("camera preset: %d applied, %d skipped%s",
                 len(applied), len(skipped),
                 f" [{', '.join(skipped)}]" if skipped else "")
        if skipped:
            print(f"        camera preset note — skipped: {', '.join(skipped)}")

    # -- acquisition ---------------------------------------------------------
    def start(self):
        if not self._acquiring:
            self._cam.BeginAcquisition()
            self._acquiring = True
            self.state = DeviceState.RUNNING

    def stop(self):
        if self._acquiring:
            self._cam.EndAcquisition()
            self._acquiring = False
            self.state = DeviceState.READY

    def grab(self, timeout_ms: int = 2000) -> np.ndarray:
        """Return one frame as a numpy array (starts acquisition if needed)."""
        self._require_ready()
        self.start()
        img = self._cam.GetNextImage(timeout_ms)
        try:
            if img.IsIncomplete():
                raise RuntimeError(f"incomplete image: {img.GetImageStatus()}")
            arr = img.GetNDArray().copy()
        finally:
            img.Release()
        return arr

    def grab_mean(self, n: int = 1, timeout_ms: int = 2000) -> np.ndarray:
        """Average n frames — the noise-reduction knob for the forward pass."""
        frames = [self.grab(timeout_ms).astype(np.float64) for _ in range(max(1, n))]
        return np.mean(frames, axis=0)

    def status(self) -> dict:
        """Best-effort: GenICam nodes can become unreadable mid-acquisition;
        a status probe must never take the server down over that."""
        s = {"state": self.state.value, "acquiring": self._acquiring}
        if self._cam:
            for key, read in (("model", lambda: self._cam.TLDevice.DeviceModelName.GetValue()),
                              ("fps", lambda: float(self._cam.AcquisitionFrameRate.GetValue())),
                              ("exposure_us", lambda: float(self._cam.ExposureTime.GetValue()))):
                try:
                    s[key] = read()
                except Exception:
                    pass
        return s
