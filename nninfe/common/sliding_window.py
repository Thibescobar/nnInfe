"""Sliding-window utilities."""

import math
from typing import List, Tuple

import numpy as np


def compute_patch_positions(
    image_shape_zyx: Tuple[int, ...],
    patch_size_zyx: Tuple[int, ...],
    overlap: float,
) -> Tuple[List[Tuple[int, int, int]], Tuple[float, ...]]:
    """
    Compute top-left corner positions (z, y, x) for sliding window patches.
    *overlap* is a proportion in [0, 1) (e.g. 0.5 = 50% overlap).

    The number of patches per axis is computed so that the overlap is at
    least the requested value. Positions are then distributed evenly so
    that the first patch starts at 0 and the last patch ends exactly at
    the image boundary.

    Returns (positions, step_sizes) where step_sizes is the effective
    floating-point step per axis.
    """
    overlap_vox = tuple(p * overlap for p in patch_size_zyx)

    n_per_axis: List[int] = []
    for img, p, ov in zip(image_shape_zyx, patch_size_zyx, overlap_vox):
        n = math.ceil((img - p) / (p - ov) + 1) if img > p else 1
        n_per_axis.append(n)

    max_step = tuple(img - p for img, p in zip(image_shape_zyx, patch_size_zyx))
    step_sizes = tuple(ms / (n - 1) if n > 1 else 0.0 for ms, n in zip(max_step, n_per_axis))

    steps_per_axis: List[List[int]] = []
    for axis in range(3):
        steps_per_axis.append([int(i * step_sizes[axis]) for i in range(n_per_axis[axis])])

    positions: List[Tuple[int, int, int]] = [
        (z, y, x)
        for z in steps_per_axis[0]
        for y in steps_per_axis[1]
        for x in steps_per_axis[2]
    ]
    return positions, step_sizes


def extract_patch(
    volume: np.ndarray,
    position_zyx: Tuple[int, int, int],
    patch_size_zyx: Tuple[int, ...],
) -> np.ndarray:
    """Extract a patch from a 3-D volume. Returns shape (D, H, W)."""
    z, y, x = position_zyx
    pz, py, px = patch_size_zyx
    return volume[z : z + pz, y : y + py, x : x + px]
