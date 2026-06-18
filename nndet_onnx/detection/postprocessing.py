"""Detection post-processing and geometric helpers."""

import math
from typing import Dict, List, Tuple

import numpy as np

from nndet_onnx.common.constants import D0_MAX, D0_MIN, D1_MAX, D1_MIN, D2_MAX, D2_MIN


def _build_gaussian_sigma(patch_size_zyx: Tuple[int, ...], edge_value: float = 0.001):
    sigmas = []
    for p in patch_size_zyx:
        half = p / 2.0
        sigma2 = -(half**2) / (2.0 * math.log(edge_value))
        sigmas.append(math.sqrt(sigma2))
    return tuple(sigmas)


def gaussian_weight_for_boxes(
    boxes: np.ndarray,
    patch_size_zyx: Tuple[int, ...],
    edge_value: float = 0.001,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    sigmas = _build_gaussian_sigma(patch_size_zyx, edge_value)

    c_d0 = (boxes[:, D0_MIN] + boxes[:, D0_MAX]) / 2.0
    c_d1 = (boxes[:, D1_MIN] + boxes[:, D1_MAX]) / 2.0
    c_d2 = (boxes[:, D2_MIN] + boxes[:, D2_MAX]) / 2.0

    patch_c_d0 = patch_size_zyx[0] / 2.0
    patch_c_d1 = patch_size_zyx[1] / 2.0
    patch_c_d2 = patch_size_zyx[2] / 2.0

    sigma_d0, sigma_d1, sigma_d2 = sigmas[0], sigmas[1], sigmas[2]

    exponent = (
        -((c_d0 - patch_c_d0) ** 2) / (2.0 * sigma_d0**2)
        - ((c_d1 - patch_c_d1) ** 2) / (2.0 * sigma_d1**2)
        - ((c_d2 - patch_c_d2) ** 2) / (2.0 * sigma_d2**2)
    )
    return np.exp(exponent).astype(np.float32)


def translate_boxes(
    detection: Dict[str, np.ndarray],
    offset_zyx: Tuple[int, int, int],
) -> Dict[str, np.ndarray]:
    if len(detection["boxes"]) == 0:
        return detection
    oz, oy, ox = offset_zyx
    shift = np.array([oz, oy, oz, oy, ox, ox], dtype=np.float32)
    return {**detection, "boxes": detection["boxes"] + shift}


def filter_by_score(
    detection: Dict[str, np.ndarray],
    score_thresh: float,
) -> Dict[str, np.ndarray]:
    keep = detection["scores"] > score_thresh
    return {k: v[keep] for k, v in detection.items()}


def filter_small_boxes(
    detection: Dict[str, np.ndarray],
    spacing_xyz: Tuple[float, ...],
    min_size_mm: float,
) -> Dict[str, np.ndarray]:
    boxes = detection["boxes"]
    if len(boxes) == 0:
        return detection

    d_dim0 = (boxes[:, D0_MAX] - boxes[:, D0_MIN]) * spacing_xyz[2]
    d_dim1 = (boxes[:, D1_MAX] - boxes[:, D1_MIN]) * spacing_xyz[1]
    d_dim2 = (boxes[:, D2_MAX] - boxes[:, D2_MIN]) * spacing_xyz[0]
    keep = (d_dim0 >= min_size_mm) & (d_dim1 >= min_size_mm) & (d_dim2 >= min_size_mm)
    return {k: v[keep] for k, v in detection.items()}


def nms_nndet(
    detection: Dict[str, np.ndarray],
    iou_threshold: float,
) -> Dict[str, np.ndarray]:
    try:
        import torch
        from nndet.core.boxes import nms
    except ImportError:
        raise RuntimeError(
            "nms_backend='nndet' requires torch and nndet packages. "
            "Install them or use --nms-backend numpy instead."
        )

    if len(detection["boxes"]) == 0:
        return detection
    boxes_t = torch.from_numpy(detection["boxes"])
    scores_t = torch.from_numpy(detection["scores"])
    keep = nms(boxes_t, scores_t, iou_threshold=iou_threshold)
    keep = keep.numpy() if isinstance(keep, torch.Tensor) else np.asarray(keep)
    return {k: v[keep] for k, v in detection.items()}


def _iou_3d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    i_d0_lo = np.maximum(box[D0_MIN], boxes[:, D0_MIN])
    i_d1_lo = np.maximum(box[D1_MIN], boxes[:, D1_MIN])
    i_d0_hi = np.minimum(box[D0_MAX], boxes[:, D0_MAX])
    i_d1_hi = np.minimum(box[D1_MAX], boxes[:, D1_MAX])
    i_d2_lo = np.maximum(box[D2_MIN], boxes[:, D2_MIN])
    i_d2_hi = np.minimum(box[D2_MAX], boxes[:, D2_MAX])

    inter = np.maximum(0, i_d0_hi - i_d0_lo) * np.maximum(0, i_d1_hi - i_d1_lo) * np.maximum(0, i_d2_hi - i_d2_lo)

    def volume(b: np.ndarray) -> np.ndarray:
        return (
            np.maximum(0, b[..., D0_MAX] - b[..., D0_MIN])
            * np.maximum(0, b[..., D1_MAX] - b[..., D1_MIN])
            * np.maximum(0, b[..., D2_MAX] - b[..., D2_MIN])
        )

    vol_box = volume(box)
    vol_boxes = volume(boxes)
    union = vol_box + vol_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def nms_numpy(
    detection: Dict[str, np.ndarray],
    iou_threshold: float,
) -> Dict[str, np.ndarray]:
    boxes = detection["boxes"]
    scores = detection["scores"]
    if len(boxes) == 0:
        return detection

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = _iou_3d(boxes[i], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    keep_arr = np.array(keep)
    return {k: v[keep_arr] for k, v in detection.items()}


def apply_nms(
    detection: Dict[str, np.ndarray],
    iou_threshold: float,
    nms_backend: str = "numpy",
) -> Dict[str, np.ndarray]:
    if nms_backend == "nndet":
        return nms_nndet(detection, iou_threshold)
    return nms_numpy(detection, iou_threshold)


def postprocess(
    detection: Dict[str, np.ndarray],
    spacing_xyz: Tuple[float, ...],
    score_thresh: float = 0.5,
    min_size_mm: float = 2.0,
    iou_threshold: float = 0.3,
    nms_backend: str = "numpy",
) -> Dict[str, np.ndarray]:
    detection = filter_by_score(detection, score_thresh)
    detection = filter_small_boxes(detection, spacing_xyz, min_size_mm)
    detection = apply_nms(detection, iou_threshold, nms_backend)
    return detection


def clip_boxes_to_image_shape(
    detection: Dict[str, np.ndarray],
    image_shape_zyx: Tuple[int, ...],
) -> Dict[str, np.ndarray]:
    if len(detection["boxes"]) == 0:
        return detection
        
    boxes = detection["boxes"].copy()
    z_max, y_max, x_max = image_shape_zyx
    
    boxes[:, D0_MIN] = np.clip(boxes[:, D0_MIN], 0, z_max)
    boxes[:, D0_MAX] = np.clip(boxes[:, D0_MAX], 0, z_max)
    boxes[:, D1_MIN] = np.clip(boxes[:, D1_MIN], 0, y_max)
    boxes[:, D1_MAX] = np.clip(boxes[:, D1_MAX], 0, y_max)
    boxes[:, D2_MIN] = np.clip(boxes[:, D2_MIN], 0, x_max)
    boxes[:, D2_MAX] = np.clip(boxes[:, D2_MAX], 0, x_max)
    
    valid = (boxes[:, D0_MAX] > boxes[:, D0_MIN]) & \
            (boxes[:, D1_MAX] > boxes[:, D1_MIN]) & \
            (boxes[:, D2_MAX] > boxes[:, D2_MIN])
            
    clipped_det = {k: v[valid].copy() if k == "boxes" else v[valid] for k, v in detection.items()}
    clipped_det["boxes"] = boxes[valid]
    return clipped_det


def merge_detections(
    all_detections: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Concatenate detections from multiple patches into one dict."""
    if not all_detections:
        return {
            "boxes": np.empty((0, 6), np.float32),
            "scores": np.empty((0,), np.float32),
            "labels": np.empty((0,), np.int64),
        }
    keys = all_detections[0].keys()
    return {k: np.concatenate([d[k] for d in all_detections], axis=0) for k in keys}
