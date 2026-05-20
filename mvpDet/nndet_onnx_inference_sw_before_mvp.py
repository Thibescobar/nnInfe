"""
nnDet ONNX Inference Pipeline — Sliding Window
Standalone inference pipeline for nnDet models exported to ONNX.
Preprocesses a NIfTI image, generates anchors, runs sliding-window inference
with Gaussian score weighting, and post-processes detections.
"""

import argparse
import math
import os
import pickle as pkl
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union


import numpy as np
import onnxruntime as ort
import SimpleITK as sitk


# nnDet box format indices: (dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
# dim0 = Z,  dim1 = Y,  dim2 = X
D0_MIN, D1_MIN, D0_MAX, D1_MAX, D2_MIN, D2_MAX = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def _generate_cell_anchors(
    width: Tuple[int, ...],
    height: Tuple[int, ...],
    depth: Tuple[int, ...],
) -> np.ndarray:
    """Generate base anchors for one feature level."""
    all_sizes = (
        np.array(list(product(width, height, depth)), dtype=np.float32) / 2.0
    )
    return np.stack(
        [
            -all_sizes[:, 0], -all_sizes[:, 1],
             all_sizes[:, 0],  all_sizes[:, 1],
            -all_sizes[:, 2],  all_sizes[:, 2],
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

        shift_d0, shift_d1, shift_d2 = np.meshgrid(
            shifts_d0, shifts_d1, shifts_d2, indexing="ij",
        )

        shifts = np.stack(
            [shift_d0.ravel(), shift_d1.ravel(), shift_d0.ravel(),
             shift_d1.ravel(), shift_d2.ravel(), shift_d2.ravel()],
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
    """Compute feature map sizes from patch_size, encoder strides and decoder_levels.

    decoder_levels are 1-indexed (nnDet convention): level k corresponds to
    the cumulative product of strides[0..k-1].
    """
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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def resample_image(image: sitk.Image, target_spacing_zyx: List[float]) -> sitk.Image:
    """Resample *image* to *target_spacing_zyx* (given in Z-Y-X order)."""
    new_spacing = target_spacing_zyx[::-1]  # convert to X-Y-Z
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(np.round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)


def clip_image(image: sitk.Image, lower: float, upper: float) -> sitk.Image:
    """Clamp intensities to [lower, upper]."""
    clamper = sitk.ClampImageFilter()
    clamper.SetLowerBound(lower)
    clamper.SetUpperBound(upper)
    return clamper.Execute(image)


def normalize_image(image: sitk.Image, mean: float, std: float) -> sitk.Image:
    """Z-score normalization: (image - mean) / std."""
    zscorer = sitk.ShiftScaleImageFilter()
    zscorer.SetShift(-mean)
    zscorer.SetScale(1.0 / std)
    return zscorer.Execute(image)


def preprocess_image(
    image_path: str,
    plan_inference: dict,
    verbose: bool = True,
) -> sitk.Image:
    """Preprocessing chain (no crop): cast → resample → clip → normalize."""
    image = sitk.ReadImage(image_path)
    image = sitk.Cast(image, sitk.sitkFloat32)
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        print(f"      original  size (XYZ): {image.GetSize()}  spacing: {image.GetSpacing()}", flush=True)
        print(f"                intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  mean: {_sf.GetMean():.1f}  std: {_sf.GetSigma():.1f}", flush=True)

    image = resample_image(image, plan_inference["target_spacing"])
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        print(f"      resampled size (XYZ): {image.GetSize()}  spacing: {tuple(round(s, 4) for s in image.GetSpacing())}", flush=True)
        print(f"                intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  mean: {_sf.GetMean():.1f}  std: {_sf.GetSigma():.1f}", flush=True)

    intensity = plan_inference["dataset_properties"]["intensity_properties"][0]
    image = clip_image(
        image,
        lower=intensity["percentile_00_5"],
        upper=intensity["percentile_99_5"],
    )
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        print(f"      clipped   intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  (percentiles [{intensity['percentile_00_5']:.1f}, {intensity['percentile_99_5']:.1f}])", flush=True)

    image = normalize_image(image, mean=intensity["mean"], std=intensity["std"])
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        print(f"      normalized intensity range: [{_sf.GetMinimum():.2f}, {_sf.GetMaximum():.2f}]  mean: {_sf.GetMean():.2f}  std: {_sf.GetSigma():.2f}", flush=True)

    return image


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def compute_patch_positions(
    image_shape_zyx: Tuple[int, ...],
    patch_size_zyx: Tuple[int, ...],
    overlap: float,
) -> Tuple[List[Tuple[int, int, int]], Tuple[float, ...]]:
    """
    Compute top-left corner positions (z, y, x) for sliding window patches.
    *overlap* is a proportion in [0, 1) (e.g. 0.5 = 50% overlap).

    The number of patches per axis is computed so that the overlap is at
    least the requested value.  Positions are then distributed evenly so
    that the first patch starts at 0 and the last patch ends exactly at
    the image boundary.

    Returns (positions, step_sizes) where step_sizes is the effective
    floating-point step per axis.
    """
    overlap_vox = tuple(p * overlap for p in patch_size_zyx)

    # Number of patches per axis (guarantee overlap >= requested)
    n_per_axis: List[int] = []
    for img, p, ov in zip(image_shape_zyx, patch_size_zyx, overlap_vox):
        n = math.ceil((img - p) / (p - ov) + 1) if img > p else 1
        n_per_axis.append(n)

    # Maximum starting index per axis
    max_step = tuple(img - p for img, p in zip(image_shape_zyx, patch_size_zyx))

    # Evenly distributed step size
    step_sizes = tuple(
        ms / (n - 1) if n > 1 else 0.0
        for ms, n in zip(max_step, n_per_axis)
    )

    # Generate per-axis starting indices
    steps_per_axis: List[List[int]] = []
    for axis in range(3):
        steps_per_axis.append(
            [int(i * step_sizes[axis]) for i in range(n_per_axis[axis])]
        )

    # Combine into positions
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


# ---------------------------------------------------------------------------
# Gaussian score weighting
# ---------------------------------------------------------------------------

def _build_gaussian_sigma(patch_size_zyx: Tuple[int, ...], edge_value: float = 0.001):
    """
    Compute per-axis sigma so that the Gaussian equals *edge_value* at patch
    borders and 1.0 at the centre.

    G(r) = exp(-r^2 / (2*sigma^2))
    At edge: r = patch_size/2, G = edge_value
    => sigma^2 = -(patch_size/2)^2 / (2 * ln(edge_value))
    """
    sigmas = []
    for p in patch_size_zyx:
        half = p / 2.0
        sigma2 = -(half ** 2) / (2.0 * math.log(edge_value))
        sigmas.append(math.sqrt(sigma2))
    return tuple(sigmas)


def gaussian_weight_for_boxes(
    boxes: np.ndarray,
    patch_size_zyx: Tuple[int, ...],
    edge_value: float = 0.001,
) -> np.ndarray:
    """
    Compute a Gaussian weight for each box based on the distance of its centre
    to the patch centre.  Boxes use the nnDet 6-index format; patch coordinates
    are in (Z, Y, X) order.

    Returns an array of weights in [edge_value, 1.0], one per box.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    sigmas = _build_gaussian_sigma(patch_size_zyx, edge_value)

    c_d0 = (boxes[:, D0_MIN] + boxes[:, D0_MAX]) / 2.0  # Z centre
    c_d1 = (boxes[:, D1_MIN] + boxes[:, D1_MAX]) / 2.0  # Y centre
    c_d2 = (boxes[:, D2_MIN] + boxes[:, D2_MAX]) / 2.0  # X centre

    patch_c_d0 = patch_size_zyx[0] / 2.0  # Z centre
    patch_c_d1 = patch_size_zyx[1] / 2.0  # Y centre
    patch_c_d2 = patch_size_zyx[2] / 2.0  # X centre

    sigma_d0, sigma_d1, sigma_d2 = sigmas[0], sigmas[1], sigmas[2]

    exponent = (
        -((c_d0 - patch_c_d0) ** 2) / (2.0 * sigma_d0 ** 2)
        - ((c_d1 - patch_c_d1) ** 2) / (2.0 * sigma_d1 ** 2)
        - ((c_d2 - patch_c_d2) ** 2) / (2.0 * sigma_d2 ** 2)
    )
    weights = np.exp(exponent).astype(np.float32)
    return weights


# ---------------------------------------------------------------------------
# Inference — session creation & execution
# ---------------------------------------------------------------------------

BACKENDS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def create_session(
    model_path: str,
    backend: str = "cpu",
    trt_fp16: bool = False,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with the appropriate providers.

    Supported backends: cpu, cuda, openvino, trt.
    For TensorRT, an engine cache is stored next to the model to speed up
    subsequent runs.  Set *trt_fp16* to enable FP16 inference on GPU.
    """
    providers = BACKENDS[backend]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # errors only

    # Suppress C++ level warnings (Memcpy, ScatterND, etc.)
    ort.set_default_logger_severity(3)

    provider_options: list = []
    for prov in providers:
        if prov == "TensorrtExecutionProvider":
            cache_dir = str(Path(model_path).parent / "trt_engine_cache")
            os.makedirs(cache_dir, exist_ok=True)
            provider_options.append({
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": cache_dir,
                "trt_fp16_enable": "True" if trt_fp16 else "False",
            })
        elif prov == "OpenVINOExecutionProvider":
            provider_options.append({"device_type": "CPU"})
        else:
            provider_options.append({})

    session = ort.InferenceSession(
        model_path, sess_options=opts,
        providers=providers, provider_options=provider_options,
    )
    actual = session.get_providers()
    print(f"      ONNX Runtime providers: {actual}\n", flush=True)
    return session


def run_inference(
    session: ort.InferenceSession,
    input_array: np.ndarray,
    anchors_batch: np.ndarray,
) -> list:
    """Run ONNX inference and return raw outputs."""
    return session.run(None, {"images": input_array, "anchors": anchors_batch})


def parse_outputs(
    outputs: list,
    batch_size: int,
) -> List[Dict[str, np.ndarray]]:
    """Parse raw ONNX outputs into per-batch detections."""
    detections = []
    for b in range(batch_size):
        detections.append({
            "boxes": outputs[b],
            "scores": outputs[b + batch_size],
            "labels": outputs[b + 2 * batch_size],
        })
    return detections


def translate_boxes(
    detection: Dict[str, np.ndarray],
    offset_zyx: Tuple[int, int, int],
) -> Dict[str, np.ndarray]:
    """Translate box coordinates from patch-local to image-global space.

    Box format is (dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
    where dim0=Z, dim1=Y, dim2=X (nnDet convention).
    """
    if len(detection["boxes"]) == 0:
        return detection
    oz, oy, ox = offset_zyx
    # dim0=Z → oz, dim1=Y → oy, dim2=X → ox
    shift = np.array([oz, oy, oz, oy, ox, ox], dtype=np.float32)
    return {
        "boxes": detection["boxes"] + shift,
        "scores": detection["scores"],
        "labels": detection["labels"],
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

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
    
    # dx = (boxes[:, 2] - boxes[:, 0]) * spacing_xyz[0]
    # dy = (boxes[:, 3] - boxes[:, 1]) * spacing_xyz[1]
    # dz = (boxes[:, 5] - boxes[:, 4]) * spacing_xyz[2]
    # keep = (dx >= min_size_mm) & (dy >= min_size_mm) & (dz >= min_size_mm)

    # Box format: (dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
    # dim0=Z, dim1=Y, dim2=X; spacing_xyz = (X, Y, Z)
    d_dim0 = (boxes[:, D0_MAX] - boxes[:, D0_MIN]) * spacing_xyz[2]  # Z extent × Z spacing
    d_dim1 = (boxes[:, D1_MAX] - boxes[:, D1_MIN]) * spacing_xyz[1]  # Y extent × Y spacing
    d_dim2 = (boxes[:, D2_MAX] - boxes[:, D2_MIN]) * spacing_xyz[0]  # X extent × X spacing
    keep = (d_dim0 >= min_size_mm) & (d_dim1 >= min_size_mm) & (d_dim2 >= min_size_mm)

    return {k: v[keep] for k, v in detection.items()}


# ---- NMS alternative A: nndet + torch ----

def nms_nndet(
    detection: Dict[str, np.ndarray],
    iou_threshold: float,
) -> Dict[str, np.ndarray]:
    import torch
    from nndet.core.boxes import nms

    if len(detection["boxes"]) == 0:
        return detection
    boxes_t = torch.from_numpy(detection["boxes"])
    scores_t = torch.from_numpy(detection["scores"])
    keep = nms(boxes_t, scores_t, iou_threshold=iou_threshold)
    keep = keep.numpy() if isinstance(keep, torch.Tensor) else np.asarray(keep)
    return {k: v[keep] for k, v in detection.items()}


# ---- NMS alternative B: pure numpy ----

def _iou_3d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    # Intersection bounds per dimension
    i_d0_lo = np.maximum(box[D0_MIN], boxes[:, D0_MIN])
    i_d1_lo = np.maximum(box[D1_MIN], boxes[:, D1_MIN])
    i_d0_hi = np.minimum(box[D0_MAX], boxes[:, D0_MAX])
    i_d1_hi = np.minimum(box[D1_MAX], boxes[:, D1_MAX])
    i_d2_lo = np.maximum(box[D2_MIN], boxes[:, D2_MIN])
    i_d2_hi = np.minimum(box[D2_MAX], boxes[:, D2_MAX])

    inter = (
        np.maximum(0, i_d0_hi - i_d0_lo)
        * np.maximum(0, i_d1_hi - i_d1_lo)
        * np.maximum(0, i_d2_hi - i_d2_lo)
    )

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


def merge_detections(
    all_detections: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Concatenate detections from multiple patches into one dict."""
    if not all_detections:
        return {"boxes": np.empty((0, 6), np.float32),
                "scores": np.empty((0,), np.float32),
                "labels": np.empty((0,), np.int64)}
    return {
        "boxes": np.concatenate([d["boxes"] for d in all_detections], axis=0),
        "scores": np.concatenate([d["scores"] for d in all_detections], axis=0),
        "labels": np.concatenate([d["labels"] for d in all_detections], axis=0),
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def detections_to_mask(
    detection: Dict[str, np.ndarray],
    image_shape_zyx: Tuple[int, ...],
    reference_image: sitk.Image,
) -> sitk.Image:
    """Build a connected-component label map from detected boxes."""
    mask = np.zeros(image_shape_zyx, dtype=np.uint8)
    for box in detection["boxes"]:
        # Box format: (dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
        # where dim0=Z, dim1=Y, dim2=X (nnDet convention)
        d0a, d1a, d0b, d1b, d2a, d2b = box.astype(int)
        # Clip to image bounds (image_shape_zyx = Z, Y, X)
        d0a, d0b = max(0, d0a), min(image_shape_zyx[0], d0b)
        d1a, d1b = max(0, d1a), min(image_shape_zyx[1], d1b)
        d2a, d2b = max(0, d2a), min(image_shape_zyx[2], d2b)
        mask[d0a:d0b, d1a:d1b, d2a:d2b] = 1

    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(reference_image)
    return sitk.ConnectedComponent(mask_sitk)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="nnDet ONNX inference pipeline (sliding window)",
    )

    # Paths
    parser.add_argument("--model-path", required=True, help="Path to model_onnx.onnx")
    parser.add_argument("--plan-path", required=True, help="Path to plan_inference.pkl")
    parser.add_argument("--image-path", help="Path to input NIfTI image")
    parser.add_argument("--output", help="Output detection mask (.nii.gz)")

    # Sliding window
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Overlap between patches as proportion in [0,1) (default: 0.5)",
    )

    # Post-processing
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold (default: 0.5)")
    parser.add_argument("--min-size-mm", type=float, default=2.0, help="Min box size in mm (default: 2.0)")
    parser.add_argument("--iou-threshold", type=float, default=None, help="NMS IoU threshold (default: from plan)")
    parser.add_argument(
        "--nms-backend", choices=["numpy", "nndet"], default="numpy",
        help="NMS implementation (default: numpy)",
    )
    parser.add_argument(
        "--no-global-nms", action="store_true",
        help="Disable the final global NMS after merging all patches",
    )

    # Runtime
    parser.add_argument(
        "--backend", choices=list(BACKENDS.keys()), default="cpu",
        help="Inference backend: cpu, openvino, tensorrt (default: cpu)",
    )
    parser.add_argument(
        "--trt-fp16", action="store_true",
        help="Enable FP16 inference for TensorRT backend",
    )
    parser.add_argument(
        "--build-engine-only", action="store_true",
        help="Build TRT engine cache and exit (no inference)",
    )

    args = parser.parse_args()

    # ---- Print all parameters (including defaults) ----
    defaults = {a.dest: a.default for a in parser._actions if a.default is not argparse.SUPPRESS}
    print("Parameters:", flush=True)
    for name, value in vars(args).items():
        tag = ""
        if name in defaults and value == defaults[name]:
            if name == "iou_threshold":
                tag = "  (auto: from plan)"
            else:
                tag = "  (default)"
        print(f"  --{name.replace('_', '-')} : {value}{tag}", flush=True)
    print(flush=True)

    # ---- Load configs ----
    with open(args.plan_path, "rb") as f:
        plan_inference = pkl.load(f)

    patch_size = tuple(plan_inference["patch_size"])  # Z, Y, X

    # Read batch_size from ONNX model input shape (dim 0 of 'images')
    if args.backend == "trt":
        cache_dir = Path(args.model_path).parent / "trt_engine_cache"
        has_cache = cache_dir.exists() and any(cache_dir.glob("*.engine"))
        if has_cache:
            print(f"      Loading TensorRT session (using cached engines from {cache_dir}) …", flush=True)
        else:
            print("      Creating TensorRT session (no cache found, compiling engines — this may take several minutes) …", flush=True)
    t_session = time.time()
    session = create_session(args.model_path, args.backend, args.trt_fp16)
    batch_size: int = session.get_inputs()[0].shape[0]
    print(f"      Session ready  ({time.time() - t_session:.2f}s)", flush=True)

    if args.build_engine_only:
        print("\nEngine built and cached. Exiting.", flush=True)
        return

    if not args.image_path or not args.output:
        parser.error("--image-path and --output are required for inference")

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = plan_inference["inference_plan"]["model_iou"]
        print(f"      iou-threshold not specified, using plan value: {iou_threshold}\n", flush=True)

    t_total = time.time()

    # ---- Anchors ----
    print("[1/5] Computing anchors …", flush=True)
    t0 = time.time()
    anchors_batch = compute_anchors(plan_inference, patch_size, batch_size)
    print(f"      anchors shape: {anchors_batch.shape}  ({time.time() - t0:.2f}s)", flush=True)

    # ---- Preprocessing ----
    print("[2/5] Preprocessing image …", flush=True)
    t0 = time.time()
    preprocessed = preprocess_image(args.image_path, plan_inference)
    volume = sitk.GetArrayFromImage(preprocessed)  # (Z, Y, X)
    image_shape = volume.shape
    spacing_xyz = preprocessed.GetSpacing()
    print(f"      preprocessing done  ({time.time() - t0:.2f}s)", flush=True)

    # ---- Sliding window positions ----
    print("[3/5] Building sliding window positions …", flush=True)
    t0 = time.time()
    positions, step_sizes = compute_patch_positions(image_shape, patch_size, args.overlap)
    actual_overlap = tuple(
        round(1.0 - s / p, 4) if p > 0 else 0.0
        for s, p in zip(step_sizes, patch_size)
    )
    n_patches = len(positions)
    n_batches = math.ceil(n_patches / batch_size)
    print(f"      step sizes (ZYX): ({step_sizes[0]:.1f}, {step_sizes[1]:.1f}, {step_sizes[2]:.1f})  actual overlap: {actual_overlap}  (requested: {args.overlap})", flush=True)
    print(f"      {n_patches} patches, {n_batches} batches (batch_size={batch_size})  ({time.time() - t0:.2f}s)", flush=True)

    # ---- Inference ----
    print("[4/5] Running inference …", flush=True)
    t0 = time.time()
    # session already created above (for batch_size extraction)

    all_detections: List[Dict[str, np.ndarray]] = []
    bar_width = 40

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_patches)
        batch_positions = positions[start:end]
        actual_count = len(batch_positions)

        # Extract patches
        patches = [
            extract_patch(volume, pos, patch_size) for pos in batch_positions
        ]

        # Pad incomplete last batch by repeating the last patch
        while len(patches) < batch_size:
            patches.append(patches[-1])

        # Build input tensor: (B, 1, Z, Y, X)
        input_array = np.stack(
            [p[np.newaxis, ...] for p in patches], axis=0
        ).astype(np.float32)

        raw_outputs = run_inference(session, input_array, anchors_batch)
        detections = parse_outputs(raw_outputs, batch_size)

        # Process only real patches (skip padding duplicates)
        for i in range(actual_count):
            det = detections[i]

            # Per-patch post-processing (score + size + NMS)
            det = postprocess(
                det,
                spacing_xyz=spacing_xyz,
                score_thresh=args.score_thresh,
                min_size_mm=args.min_size_mm,
                iou_threshold=iou_threshold,
                nms_backend=args.nms_backend,
            )

            # Gaussian score weighting
            if len(det["boxes"]) > 0:
                weights = gaussian_weight_for_boxes(det["boxes"], patch_size)
                det = {**det, "scores": det["scores"] * weights}

            # Translate to global coordinates
            det = translate_boxes(det, batch_positions[i])

            if len(det["boxes"]) > 0:
                all_detections.append(det)

        # Progress bar
        progress = (batch_idx + 1) / n_batches
        filled = int(bar_width * progress)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        elapsed = time.time() - t0
        time_per_iter = elapsed / (batch_idx + 1)
        remaining = time_per_iter * (n_batches - batch_idx - 1)
        sys.stdout.write(
            f"\r      [{bar}] {progress:6.1%}  "
            f"batch {batch_idx + 1}/{n_batches}  "
            f"patch {end}/{n_patches}  "
            f"{time_per_iter:.2f}s/batch  "
            f"elapsed {elapsed:.0f}s  remaining {remaining:.0f}s"
        )
        sys.stdout.flush()

    print(f"\n      inference done ({time.time() - t0:.2f}s)", flush=True)

    # ---- Merge + optional global NMS ----
    print("[5/5] Merging detections …", flush=True)
    t0 = time.time()
    merged = merge_detections(all_detections)
    print(f"      total detections before global NMS: {len(merged['boxes'])}", flush=True)

    if not args.no_global_nms and len(merged["boxes"]) > 0:
        merged = apply_nms(merged, iou_threshold, args.nms_backend)
        print(f"      detections after global NMS:        {len(merged['boxes'])}  ({time.time() - t0:.2f}s)", flush=True)
    elif args.no_global_nms:
        print(f"      global NMS disabled  ({time.time() - t0:.2f}s)", flush=True)
    else:
        print(f"      no detections  ({time.time() - t0:.2f}s)", flush=True)

    # ---- Export ----
    t0 = time.time()
    cc = detections_to_mask(merged, image_shape, preprocessed)
    sitk.WriteImage(cc, args.output)
    print(f"      saved → {args.output}  ({time.time() - t0:.2f}s)", flush=True)

    print(f"\nDone. Total time: {time.time() - t_total:.2f}s", flush=True)


if __name__ == "__main__":
    print("\nPipeline of detection inference based on a onnx model created from nnDetection…\n", flush=True)
    main()
