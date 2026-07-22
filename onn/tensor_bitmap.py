"""Tensor -> device-bitmap conversion layer.

Ray's task: GPU-side data lives as torch tensors; the DMD wants a binary
mirror bitmap (H, W in {0, 255}) and the SLM wants an 8-bit phase image
(H, W in 0..255). This module is the map between the two worlds.

Accepts torch tensors (CPU or GPU, any float/int dtype, with or without
batch dims) or plain numpy arrays. torch is optional — everything works
with numpy alone.

    from onn.tensor_bitmap import to_dmd_bitmap, to_slm_phase, batch_to_dmd

    frame  = to_dmd_bitmap(t, dmd_shape=(1080, 1920))       # one input
    frames = batch_to_dmd(T, dmd_shape=(1080, 1920))        # (T, h, w) stack
    phase  = to_slm_phase(w, slm_shape=(1152, 1920))        # weights -> phase
"""
from __future__ import annotations

import numpy as np

from hardware.dmd import encode_macropixels


def _to_numpy(t) -> np.ndarray:
    """torch tensor (any device, grad or not) or array-like -> numpy array."""
    if hasattr(t, "detach"):          # torch.Tensor without importing torch
        t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        t = t.numpy()
    return np.asarray(t)


def _squeeze_2d(a: np.ndarray) -> np.ndarray:
    """Drop leading singleton batch/channel dims: (1,1,h,w)->(h,w). Reject >2D."""
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected a single 2D input, got shape {a.shape}; "
                         "use batch_to_dmd()/batch_to_slm() for stacks")
    return a


# ---------------- DMD: anything -> binary mirror bitmap ----------------

def to_dmd_bitmap(t, dmd_shape: tuple[int, int], threshold: float = 0.5,
                  macropixel: int | str = "auto", invert: bool = False,
                  normalize: bool = True) -> np.ndarray:
    """Map one tensor/array (h, w) to a full binary DMD frame (H, W) uint8 {0,255}.

    Float data is min-max normalized to [0,1] first (normalize=True), then
    binarized at `threshold`. Integer/bool data is treated as already binary
    (nonzero = ON) unless it has more than two distinct values, in which case
    it is normalized+thresholded like floats.
    """
    a = _squeeze_2d(_to_numpy(t)).astype(np.float64)
    vals = np.unique(a)
    if len(vals) > 2:
        if normalize:
            lo, hi = a.min(), a.max()
            a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        a = (a >= threshold)
    else:
        a = (a > vals.min()) if len(vals) == 2 else (a > 0)
    return encode_macropixels(a.astype(np.uint8), dmd_shape,
                              macropixel=macropixel, invert=invert)


def batch_to_dmd(T, dmd_shape: tuple[int, int], **kw) -> np.ndarray:
    """(T, h, w) tensor/array -> (T, H, W) stack of DMD frames.

    Feed the result to DmdViALUX.project_sequence() for onboard-timed
    temporal playback, or index it frame-by-frame.
    """
    a = _to_numpy(T)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected (T, h, w), got {a.shape}")
    return np.stack([to_dmd_bitmap(x, dmd_shape, **kw) for x in a])


# ---------------- SLM: numerical data -> 8-bit phase map ----------------

def to_slm_phase(t, slm_shape: tuple[int, int], vmin: float | None = None,
                 vmax: float | None = None, levels: int = 256,
                 macropixel: int | str = "auto", wrap: bool = False) -> np.ndarray:
    """Map one tensor/array (h, w) of real values to an SLM phase image.

    Linear map [vmin, vmax] -> [0, levels-1] (defaults: data min/max), clipped.
    wrap=True instead applies modulo (for data already in phase units where
    2*pi wrapping is meaningful). Result is upscaled to slm_shape as centered
    macropixels, dtype uint8.
    """
    a = _squeeze_2d(_to_numpy(t)).astype(np.float64)
    if wrap:
        g = np.mod(a, levels).astype(np.uint8)
    else:
        lo = a.min() if vmin is None else float(vmin)
        hi = a.max() if vmax is None else float(vmax)
        span = hi - lo
        norm = (a - lo) / span if span > 0 else np.zeros_like(a)
        g = np.clip(np.round(norm * (levels - 1)), 0, levels - 1).astype(np.uint8)

    H, W = slm_shape
    h, w = g.shape
    m = max(1, min(H // h, W // w)) if macropixel == "auto" else int(macropixel)
    if h * m > H or w * m > W:
        raise ValueError(f"input {g.shape} at macropixel={m} exceeds SLM {slm_shape}")
    big = np.kron(g, np.ones((m, m), dtype=np.uint8))
    frame = np.zeros((H, W), dtype=np.uint8)
    oy, ox = (H - h * m) // 2, (W - w * m) // 2
    frame[oy:oy + h * m, ox:ox + w * m] = big
    return frame


def batch_to_slm(T, slm_shape: tuple[int, int], **kw) -> np.ndarray:
    """(T, h, w) tensor/array -> (T, H, W) stack of SLM phase images.

    Note: vmin/vmax default per-frame; pass explicit vmin/vmax for a
    consistent scale across the whole batch.
    """
    a = _to_numpy(T)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected (T, h, w), got {a.shape}")
    return np.stack([to_slm_phase(x, slm_shape, **kw) for x in a])
