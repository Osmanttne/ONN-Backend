"""Named pattern generators — no file browsing, patterns are made on demand.

DMD patterns are binary amplitude (uint8 {0,255}); SLM patterns are 8-bit
phase (uint8 0..255). Both are plain numpy arrays sized to the target device.
"""
from __future__ import annotations

import numpy as np


# ---------------- DMD (binary amplitude) ----------------

def solid(shape, on=True):
    return np.full(shape, 255 if on else 0, dtype=np.uint8)


def circle(shape, radius_frac=0.25, center=None, invert=False):
    """'White circle on black background' — the canonical example."""
    H, W = shape
    cy, cx = center or (H / 2, W / 2)
    yy, xx = np.mgrid[0:H, 0:W]
    r = radius_frac * min(H, W)
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    out = np.where(mask, 255, 0).astype(np.uint8)
    return 255 - out if invert else out


def stripes(shape, period_px=32, vertical=True, duty=0.5):
    H, W = shape
    axis = np.arange(W if vertical else H)
    line = ((axis % period_px) < duty * period_px).astype(np.uint8) * 255
    return np.tile(line, (H, 1)) if vertical else np.tile(line[:, None], (1, W))


def checkerboard(shape, block_px=64):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    return (((yy // block_px + xx // block_px) % 2) * 255).astype(np.uint8)


DMD_PATTERNS = {
    "white circle on black background": circle,
    "black circle on white background": lambda s, **k: circle(s, invert=True, **k),
    "vertical stripes": stripes,
    "horizontal stripes": lambda s, **k: stripes(s, vertical=False, **k),
    "checkerboard": checkerboard,
    "all on": solid,
    "all off": lambda s: solid(s, on=False),
}


def dmd_pattern(name: str, shape, **kwargs) -> np.ndarray:
    try:
        return DMD_PATTERNS[name.lower()](shape, **kwargs)
    except KeyError:
        raise KeyError(f"unknown pattern {name!r}; options: {sorted(DMD_PATTERNS)}")


# ---------------- SLM (8-bit phase) ----------------

def blank_phase(shape):
    return np.zeros(shape, dtype=np.uint8)


def blazed_grating(shape, period_px=16, vertical=True):
    H, W = shape
    axis = np.arange(W if vertical else H)
    ramp = ((axis % period_px) / period_px * 256).astype(np.uint8)
    return np.tile(ramp, (H, 1)) if vertical else np.tile(ramp[:, None], (1, W))


def lens_phase(shape, defocus=1.0, center=None):
    """Quadratic (defocus) phase, matching the Lens_Radius..._Defocus... files."""
    H, W = shape
    cy, cx = center or (H / 2, W / 2)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((yy - cy) / H) ** 2 + ((xx - cx) / W) ** 2
    return ((defocus * r2 * 4096) % 256).astype(np.uint8)


SLM_PATTERNS = {
    "blank": blank_phase,
    "blazed grating": blazed_grating,
    "lens": lens_phase,
}


def slm_pattern(name: str, shape, **kwargs) -> np.ndarray:
    try:
        return SLM_PATTERNS[name.lower()](shape, **kwargs)
    except KeyError:
        raise KeyError(f"unknown pattern {name!r}; options: {sorted(SLM_PATTERNS)}")
