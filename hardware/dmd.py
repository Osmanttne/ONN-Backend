"""ViALUX V-Module / Dev-Kit DMD driver via the ALP-4 API (ALP4lib).

The key backend job: take an input matrix (small, binary, possibly one
timestep of a temporal stack) and put it on the mirrors as macropixels.
"""
from __future__ import annotations

import logging

import numpy as np

from .base import Device, DeviceState

log = logging.getLogger("onn.dmd")

try:
    from ALP4 import ALP4, ALP_DEFAULT  # pip install ALP4lib (needs ViALUX DLL)
    _HAS_ALP = True
except ImportError:  # pragma: no cover
    _HAS_ALP = False


def encode_macropixels(x: np.ndarray, dmd_shape: tuple[int, int],
                       macropixel: int | str = "auto",
                       invert: bool = False) -> np.ndarray:
    """Upscale a small input matrix to a full binary DMD frame.

    x           : (h, w) array; nonzero = ON
    dmd_shape   : (H, W) mirror array size
    macropixel  : block size in mirrors per input element, or "auto"
    returns     : (H, W) uint8 frame with values {0, 255}, input centered
    """
    x = (np.asarray(x) > 0).astype(np.uint8)
    h, w = x.shape
    H, W = dmd_shape
    if macropixel == "auto":
        macropixel = max(1, min(H // h, W // w))
    m = int(macropixel)
    if h * m > H or w * m > W:
        raise ValueError(f"input {x.shape} at macropixel={m} exceeds DMD {dmd_shape}")
    big = np.kron(x, np.ones((m, m), dtype=np.uint8))
    frame = np.zeros((H, W), dtype=np.uint8)
    oy, ox = (H - h * m) // 2, (W - w * m) // 2
    frame[oy:oy + h * m, ox:ox + w * m] = big
    if invert:
        frame = 1 - frame
    return frame * 255


class DmdViALUX(Device):
    name = "dmd"

    def __init__(self, bit_depth: int = 1, picture_time_us: int = 100_000,
                 macropixel: int | str = "auto", invert: bool = False,
                 alp_version: str = "4.3"):
        super().__init__()
        if not _HAS_ALP:
            raise ImportError("ALP4lib not installed / ViALUX ALP DLL not found")
        self.bit_depth = bit_depth
        self.picture_time_us = picture_time_us
        self.macropixel, self.invert = macropixel, invert
        self._alp_version = alp_version
        self._dev = None
        self._seq = None
        self.shape: tuple[int, int] | None = None

    # -- lifecycle -----------------------------------------------------
    def connect(self):
        self._dev = ALP4(version=self._alp_version)
        self._dev.Initialize()
        self.shape = (self._dev.nSizeY, self._dev.nSizeX)
        log.info("DMD %s connected, %sx%s mirrors",
                 self._dev.DevInquire(2001) if hasattr(self._dev, "DevInquire") else "",
                 self.shape[1], self.shape[0])
        self.state = DeviceState.READY
        return self

    def disconnect(self):
        if self._dev:
            try:
                self.free()
            finally:
                self._dev.Free()
                self._dev = None
        self.state = DeviceState.DISCONNECTED

    # -- EasyProj equivalents -------------------------------------------
    def project(self, frame: np.ndarray, loop: bool = True):
        """Upload one full-resolution frame and start projecting it."""
        self._require_ready()
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != self.shape:
            raise ValueError(f"frame {frame.shape} != DMD {self.shape}; "
                             "use project_input() for small matrices")
        self.free()
        self._seq = self._dev.SeqAlloc(nbImg=1, bitDepth=self.bit_depth)
        self._dev.SeqPut(imgData=frame.ravel())
        self._dev.SetTiming(pictureTime=self.picture_time_us)
        self._dev.Run(loop=loop)
        self.state = DeviceState.RUNNING

    def project_input(self, x: np.ndarray):
        """The ONN entry point: small input matrix -> macropixel frame -> mirrors."""
        frame = encode_macropixels(x, self.shape, self.macropixel, self.invert)
        self.project(frame)
        return frame

    def project_sequence(self, frames: np.ndarray, picture_time_us: int | None = None,
                         loop: bool = False):
        """Upload a temporal stack (T, H, W) as one onboard ALP sequence.

        The ALP hardware then paces the frames itself at picture_time_us —
        this is the precise-timing alternative to the software-timed loop.
        """
        self._require_ready()
        frames = np.asarray(frames, dtype=np.uint8)
        self.free()
        self._seq = self._dev.SeqAlloc(nbImg=len(frames), bitDepth=self.bit_depth)
        self._dev.SeqPut(imgData=frames.reshape(len(frames), -1))
        self._dev.SetTiming(pictureTime=picture_time_us or self.picture_time_us)
        self._dev.Run(loop=loop)
        self.state = DeviceState.RUNNING

    def stop(self):
        self._require_ready()
        self._dev.Halt()
        self.state = DeviceState.READY

    def free(self):
        """EasyProj 'Free': halt and release any sequence from the DMD."""
        if self._dev and self._seq is not None:
            self._dev.Halt()
            self._dev.FreeSeq()
            self._seq = None
        if self.state is DeviceState.RUNNING:
            self.state = DeviceState.READY

    def status(self) -> dict:
        return {"state": self.state.value, "shape": self.shape,
                "sequence_loaded": self._seq is not None}
