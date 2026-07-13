"""The core backend contract:

    X (T, h, w) temporal binary input  ->  optics  ->  Y (T, gh, gw) predictions

For each timestep t: encode X[t] onto the DMD as macropixels, wait for the
optics to settle, capture at the detector, and pool the frame over the
detector ROI into the (gh, gw) prediction matrix Y[t].

v1 timing is software-paced (project -> sleep -> grab). For fast/precise
temporal inputs, upgrade path is DmdViALUX.project_sequence() (onboard ALP
timing) + camera hardware trigger — the interfaces here don't change.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np


@dataclasses.dataclass
class ForwardResult:
    Y: np.ndarray                    # (T, gh, gw) prediction matrix over time
    frames: np.ndarray | None        # (T, H, W) raw detector frames, if kept
    dmd_frames: np.ndarray | None    # (T, Hd, Wd) what was shown, if kept
    t_wall: np.ndarray               # (T,) wall-clock timestamp per step


def pool_to_grid(frame: np.ndarray, grid: tuple[int, int],
                 roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Mean-pool a detector frame (over an optional ROI) into (gh, gw)."""
    f = np.asarray(frame, dtype=np.float64)
    if roi is not None:
        x, y, w, h = roi
        f = f[y:y + h, x:x + w]
    gh, gw = grid
    H, W = f.shape
    f = f[:H - H % gh, :W - W % gw]
    return f.reshape(gh, H // gh, gw, W // gw).mean(axis=(1, 3))


class ONNForward:
    """Binds DMD + camera (and optionally laser/SLM context) into forward()."""

    def __init__(self, dmd, camera, detector_grid=(4, 4), detector_roi=None,
                 settle_s: float = 0.05, frames_per_input: int = 1,
                 normalize_y: bool = True):
        self.dmd, self.camera = dmd, camera
        self.detector_grid = tuple(detector_grid)
        self.detector_roi = tuple(detector_roi) if detector_roi else None
        self.settle_s = settle_s
        self.frames_per_input = frames_per_input
        self.normalize_y = normalize_y

    @classmethod
    def from_profile(cls, dmd, camera, profile: dict):
        o = profile["onn"]
        return cls(dmd, camera,
                   detector_grid=o["detector_grid"],
                   detector_roi=o.get("detector_roi"),
                   settle_s=o["settle_s"],
                   frames_per_input=o.get("frames_per_input", 1),
                   normalize_y=o.get("normalize_y", True))

    # -- single step -------------------------------------------------
    def step(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One timestep: input matrix -> (y, raw_frame, dmd_frame)."""
        dmd_frame = self.dmd.project_input(x)
        time.sleep(self.settle_s)
        frame = self.camera.grab_mean(self.frames_per_input)
        y = pool_to_grid(frame, self.detector_grid, self.detector_roi)
        if self.normalize_y:
            y = y / 255.0
        return y, frame, dmd_frame

    # -- temporal forward pass ------------------------------------------
    def forward(self, X: np.ndarray, keep_frames: bool = False,
                progress: bool = True) -> ForwardResult:
        """Run the temporal input stack X (T, h, w) through the optics.

        Returns ForwardResult with Y of shape (T, gh, gw).
        Accepts a single (h, w) matrix too (treated as T=1).
        """
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[None, ...]
        if X.ndim != 3:
            raise ValueError(f"X must be (T, h, w) or (h, w); got {X.shape}")

        T = len(X)
        Y = np.empty((T, *self.detector_grid))
        t_wall = np.empty(T)
        frames, dmd_frames = ([] if keep_frames else None), ([] if keep_frames else None)

        for t in range(T):
            y, frame, dmd_frame = self.step(X[t])
            Y[t], t_wall[t] = y, time.time()
            if keep_frames:
                frames.append(frame)
                dmd_frames.append(dmd_frame)
            if progress and (t % max(1, T // 10) == 0 or t == T - 1):
                print(f"  step {t + 1}/{T}  y[max]={y.max():.3f}")

        self.dmd.stop()
        return ForwardResult(
            Y=Y,
            frames=np.asarray(frames) if keep_frames else None,
            dmd_frames=np.asarray(dmd_frames) if keep_frames else None,
            t_wall=t_wall,
        )


def save_result(result: ForwardResult, X: np.ndarray, path: str, meta: dict | None = None):
    """Persist a run: inputs, predictions, timestamps (+ frames if kept)."""
    payload = {"X": np.asarray(X), "Y": result.Y, "t_wall": result.t_wall}
    if result.frames is not None:
        payload["frames"] = result.frames
    if meta:
        import json
        payload["meta_json"] = np.frombuffer(
            json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **payload)
    return path
