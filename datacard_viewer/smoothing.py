"""Bin-content smoothing ported from ROOT's TH1::SmoothArray (353QH twice)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _smooth_once(values):
    """Apply the median, quadratic interpolation, and running mean stages."""
    smoothed = values.copy()
    for stage, width in enumerate((3, 5, 3)):
        previous = smoothed.copy()
        radius = width // 2
        if len(values) >= width:
            smoothed[radius:-radius] = np.median(
                sliding_window_view(previous, width), axis=1
            )
        if stage == 0:
            # Update the left endpoint first, as ROOT does for three-bin inputs.
            smoothed[0] = np.median(
                [smoothed[1], smoothed[0], 3 * smoothed[1] - 2 * smoothed[2]]
            )
            smoothed[-1] = np.median(
                [smoothed[-2], smoothed[-1], 3 * smoothed[-2] - 2 * smoothed[-3]]
            )
        elif stage == 1:
            smoothed[1] = np.median(previous[:3])
            smoothed[-2] = np.median(previous[-3:])

    interpolated = smoothed.copy()
    for i in range(2, len(values) - 2):
        center = smoothed[i]
        if smoothed[i - 1] != center or smoothed[i + 1] != center:
            continue
        left_delta = smoothed[i - 2] - center
        right_delta = smoothed[i + 2] - center
        if left_delta * right_delta <= 0:
            continue
        step = -1 if abs(right_delta) > abs(left_delta) else 1
        left = smoothed[i - 2 * step]
        right = smoothed[i + 2 * step]
        interpolated[i] = -0.5 * left + center / 0.75 + right / 6.0
        interpolated[i + step] = 0.5 * (right - left) + center

    smoothed[1:-1] = (
        0.25 * interpolated[:-2]
        + 0.5 * interpolated[1:-1]
        + 0.25 * interpolated[2:]
    )
    return smoothed


def smooth_array(values, ntimes):
    """Return smoothed bin contents without modifying the input array."""
    result = np.array(values, dtype=np.float64, copy=True)
    if result.size < 3:
        return result
    for _ in range(ntimes):
        smooth = _smooth_once(result)
        corrected = smooth + _smooth_once(result - smooth)
        # ROOT clamps each pass only if its input contains no negative bins.
        result = corrected if result.min() < 0 else np.maximum(corrected, 0.0)
    return result
