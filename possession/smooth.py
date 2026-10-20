"""Window statistics that ignore NaN, via cumulative sums so the window width costs nothing."""

import numpy as np


def window_mean(x, half):
    """Centred mean over +/- half frames, ignoring NaN. NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def window_frac(mask, half):
    """Fraction of frames in +/- half where mask is True."""
    return window_mean(mask.astype(np.float64), half)


def frames_since(mask):
    """For each frame, frames since mask was last True (0 on a True frame)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    last = np.maximum.accumulate(idx)
    out = np.arange(len(mask)) - last
    out[last < 0] = len(mask)
    return out


def frames_until(mask):
    """For each frame, frames until mask is next True."""
    return frames_since(mask[::-1])[::-1]
