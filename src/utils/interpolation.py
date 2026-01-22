"""Shared interpolation utilities for gesture recognition"""

import numpy as np
from typing import List
from scipy import interpolate


def interpolate_missing_frames(
    sequence: List[np.ndarray], skip_indices: List[int]
) -> List[np.ndarray]:
    """Interpolate missing frames using cubic spline interpolation"""
    if not skip_indices or len(sequence) < 3:
        return sequence

    sequence_array = np.array(sequence)
    interpolated = sequence_array.copy()
    available = [i for i in range(len(sequence)) if i not in skip_indices]

    if len(available) < 2:
        return sequence

    for dim in range(sequence_array.shape[1]):
        values = sequence_array[available, dim]
        if np.all(values == values[0]):
            interpolated[skip_indices, dim] = values[0]
            continue

        try:
            kind = "cubic" if len(available) >= 4 else "linear"
            interp_func = interpolate.interp1d(
                available,
                values,
                kind=kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated[skip_indices, dim] = interp_func(skip_indices)
        except Exception:
            for skip_idx in skip_indices:
                nearest = min(available, key=lambda x: abs(x - skip_idx))
                interpolated[skip_idx, dim] = sequence_array[nearest, dim]

    return list(interpolated)


def apply_frame_skipping(
    sequence: List[np.ndarray], skip_pattern: int
) -> List[np.ndarray]:
    """Apply frame skipping pattern and interpolate missing frames"""
    if skip_pattern == 0 or len(sequence) < 3:
        return sequence

    skip_indices = (
        list(range(1, len(sequence), 2))
        if skip_pattern == 1
        else [i for i in range(len(sequence)) if i % 3 != 0]
    )

    modified = [
        np.zeros_like(frame) if i in skip_indices else frame
        for i, frame in enumerate(sequence)
    ]
    return interpolate_missing_frames(modified, skip_indices)
