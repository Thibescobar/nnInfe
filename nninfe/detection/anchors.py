"""Anchor generation for nnDetection ONNX outputs."""

from itertools import product
from typing import List, Sequence, Tuple, Union

import numpy as np


def _generate_cell_anchors(
    width: Tuple[int, ...],
    height: Tuple[int, ...],
    depth: Tuple[int, ...],
) -> np.ndarray:
    """Generate base anchors for one feature level."""
    all_sizes = np.array(list(product(width, height, depth)), dtype=np.float32) / 2.0
    return np.stack(
        [
            -all_sizes[:, 0],
            -all_sizes[:, 1],
            all_sizes[:, 0],
            all_sizes[:, 1],
            -all_sizes[:, 2],
            all_sizes[:, 2],
        ],
        axis=1,
    )


def generate_anchors(
    widths: Sequence[Union[int, Sequence[int]]],
    heights: Sequence[Union[int, Sequence[int]]],
    depths: Sequence[Union[int, Sequence[int]]],
    patch_size: Tuple[int, ...],
    feature_map_sizes: List[List[int]],
) -> np.ndarray:
    """Generate all anchors for all feature levels. Returns (num_anchors, 6)."""
    if not isinstance(widths[0], (list, tuple)):
        widths = [(w,) for w in widths]
    if not isinstance(heights[0], (list, tuple)):
        heights = [(h,) for h in heights]
    if not isinstance(depths[0], (list, tuple)):
        depths = [(d,) for d in depths]

    all_anchors: List[np.ndarray] = []
    for (w, h, d), fm in zip(zip(widths, heights, depths), feature_map_sizes):
        base = _generate_cell_anchors(w, h, d)
        stride = [int(p / s) for p, s in zip(patch_size, fm)]
        s0, s1, s2 = fm
        st0, st1, st2 = stride

        shifts_d0 = np.arange(0, s0, dtype=np.float32) * st0
        shifts_d1 = np.arange(0, s1, dtype=np.float32) * st1
        shifts_d2 = np.arange(0, s2, dtype=np.float32) * st2

        shift_d0, shift_d1, shift_d2 = np.meshgrid(shifts_d0, shifts_d1, shifts_d2, indexing="ij")

        shifts = np.stack(
            [
                shift_d0.ravel(),
                shift_d1.ravel(),
                shift_d0.ravel(),
                shift_d1.ravel(),
                shift_d2.ravel(),
                shift_d2.ravel(),
            ],
            axis=1,
        )

        level_anchors = (shifts[:, None, :] + base[None, :, :]).reshape(-1, 6)
        all_anchors.append(level_anchors)

    return np.concatenate(all_anchors, axis=0)


def compute_feature_map_sizes(
    patch_size: Tuple[int, ...],
    strides: List[List[int]],
    decoder_levels: Sequence[int],
) -> List[List[int]]:
    """Compute feature map sizes from patch_size, encoder strides and decoder levels."""
    cum = [1, 1, 1]
    all_fm: List[List[int]] = []
    for s in strides:
        cum = [c * si for c, si in zip(cum, s)]
        fm = [p // c for p, c in zip(patch_size, cum)]
        all_fm.append(fm)
    return [all_fm[dl - 1] for dl in decoder_levels]


def compute_anchors(
    plan_inference: dict,
    patch_size: Tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    """Return anchors array of shape (batch_size, num_anchors, 6)."""
    anchors_cfg = plan_inference["anchors"]
    feature_map_sizes = compute_feature_map_sizes(
        patch_size,
        plan_inference["architecture"]["strides"],
        plan_inference["architecture"]["decoder_levels"],
    )
    anchors = generate_anchors(
        widths=anchors_cfg["width"],
        heights=anchors_cfg["height"],
        depths=anchors_cfg["depth"],
        patch_size=patch_size,
        feature_map_sizes=feature_map_sizes,
    )
    return np.repeat(anchors[None, :, :], batch_size, axis=0).astype(np.float32)
