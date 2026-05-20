"""
nnDet ONNX Inference Pipeline — Sliding Window
Standalone inference pipeline for nnDet models exported to ONNX.
Preprocesses a NIfTI image, generates anchors, runs sliding-window inference
with Gaussian score weighting, and post-processes detections.
"""

import argparse
import math
import pickle as pkl
import sys
import time
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Union


import numpy as np
import onnxruntime as ort
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

class AnchorGenerator3DSONNX:
    def __init__(
        self,
        width: Sequence[Union[int, Sequence[int]]],
        height: Sequence[Union[int, Sequence[int]]],
        depth: Sequence[Union[int, Sequence[int]]],
    ):
        if not isinstance(width[0], Sequence):
            width = [(w,) for w in width]
        if not isinstance(height[0], Sequence):
            height = [(h,) for h in height]
        if not isinstance(depth[0], Sequence):
            depth = [(d,) for d in depth]
        self.width = width
        self.height = height
        self.depth = depth
        assert len(self.width) == len(self.height) == len(self.depth)
        self.cell_anchors: Optional[List[np.ndarray]] = None

    def set_cell_anchors(self) -> None:
        if self.cell_anchors is not None:
            return
        self.cell_anchors = [
            self._generate_anchors(w, h, d)
            for w, h, d in zip(self.width, self.height, self.depth)
        ]

    @staticmethod
    def _generate_anchors(
        width: Tuple[int, ...],
        height: Tuple[int, ...],
        depth: Tuple[int, ...],
    ) -> np.ndarray:
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

    def grid_anchors(
        self,
        grid_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
    ) -> Tuple[List[np.ndarray], List[int]]:
        assert self.cell_anchors is not None
        assert len(grid_sizes) == len(strides) == len(self.cell_anchors)

        anchors: List[np.ndarray] = []
        anchors_per_level: List[int] = []

        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            s0, s1, s2 = size
            st0, st1, st2 = stride

            shifts_x = np.arange(0, s0, dtype=np.float32) * st0
            shifts_y = np.arange(0, s1, dtype=np.float32) * st1
            shifts_z = np.arange(0, s2, dtype=np.float32) * st2

            shift_x, shift_y, shift_z = np.meshgrid(
                shifts_x, shifts_y, shifts_z, indexing="ij"
            )
            shift_x = shift_x.ravel()
            shift_y = shift_y.ravel()
            shift_z = shift_z.ravel()

            shifts = np.stack(
                [shift_x, shift_y, shift_x, shift_y, shift_z, shift_z], axis=1
            )

            all_anchors = shifts[:, None, :] + base_anchors[None, :, :]
            all_anchors = all_anchors.reshape(-1, 6)

            anchors.append(all_anchors)
            anchors_per_level.append(all_anchors.shape[0])

        return anchors, anchors_per_level

    def forward(
        self,
        patch_size: Tuple[int, ...],
        feature_map_size: List[List[int]],
    ) -> np.ndarray:
        grid_sizes = feature_map_size
        strides = [
            [int(i / s) for i, s in zip(patch_size, fm)]
            for fm in grid_sizes
        ]
        self.set_cell_anchors()
        anchors_per_fm, _ = self.grid_anchors(grid_sizes, strides)
        return np.concatenate(anchors_per_fm, axis=0)


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
    generator = AnchorGenerator3DSONNX(
        width=anchors_cfg["width"],
        height=anchors_cfg["height"],
        depth=anchors_cfg["depth"],
    )
    feature_map_size = compute_feature_map_sizes(
        patch_size,
        plan_inference["architecture"]["strides"],
        plan_inference["architecture"]["decoder_levels"],
    )
    anchors = generator.forward(patch_size, feature_map_size)
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
        arr = sitk.GetArrayFromImage(image)
        print(f"      original  size (XYZ): {image.GetSize()}  spacing: {image.GetSpacing()}", flush=True)
        print(f"                intensity range: [{arr.min():.1f}, {arr.max():.1f}]  mean: {arr.mean():.1f}  std: {arr.std():.1f}", flush=True)

    image = resample_image(image, plan_inference["target_spacing"])
    if verbose:
        arr = sitk.GetArrayFromImage(image)
        print(f"      resampled size (XYZ): {image.GetSize()}  spacing: {tuple(round(s, 4) for s in image.GetSpacing())}", flush=True)
        print(f"                intensity range: [{arr.min():.1f}, {arr.max():.1f}]  mean: {arr.mean():.1f}  std: {arr.std():.1f}", flush=True)

    intensity = plan_inference["dataset_properties"]["intensity_properties"][0]
    image = clip_image(
        image,
        lower=intensity["percentile_00_5"],
        upper=intensity["percentile_99_5"],
    )
    if verbose:
        arr = sitk.GetArrayFromImage(image)
        print(f"      clipped   intensity range: [{arr.min():.1f}, {arr.max():.1f}]  (percentiles [{intensity['percentile_00_5']:.1f}, {intensity['percentile_99_5']:.1f}])", flush=True)

    image = normalize_image(image, mean=intensity["mean"], std=intensity["std"])
    if verbose:
        arr = sitk.GetArrayFromImage(image)
        print(f"      normalized intensity range: [{arr.min():.2f}, {arr.max():.2f}]  mean: {arr.mean():.2f}  std: {arr.std():.2f}", flush=True)

    return image


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def compute_patch_positions(
    image_shape_zyx: Tuple[int, ...],
    patch_size_zyx: Tuple[int, ...],
    overlap: float,
) -> List[Tuple[int, int, int]]:
    """
    Compute top-left corner positions (z, y, x) for sliding window patches.
    *overlap* is a proportion in [0, 1) (e.g. 0.5 = 50% overlap).
    Patches at the border are shifted inward so they don't exceed image bounds.
    """
    stride = tuple(
        max(1, int(round(p * (1.0 - overlap))))
        for p in patch_size_zyx
    )
    positions: List[Tuple[int, int, int]] = []
    for z in range(0, image_shape_zyx[0], stride[0]):
        for y in range(0, image_shape_zyx[1], stride[1]):
            for x in range(0, image_shape_zyx[2], stride[2]):
                # Clamp to stay within image bounds
                z_c = min(z, image_shape_zyx[0] - patch_size_zyx[0])
                y_c = min(y, image_shape_zyx[1] - patch_size_zyx[1])
                x_c = min(x, image_shape_zyx[2] - patch_size_zyx[2])
                pos = (max(0, z_c), max(0, y_c), max(0, x_c))
                if pos not in positions:
                    positions.append(pos)
    return positions


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
    to the patch centre. Boxes are in (x1, y1, x2, y2, z1, z2) format;
    patch coordinates are in (z, y, x) order.

    Returns an array of weights in [edge_value, 1.0], one per box.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)

    # sigmas[i] corresponds to patch_size_zyx[i]: [0]=Z, [1]=Y, [2]=X
    sigmas = _build_gaussian_sigma(patch_size_zyx, edge_value)

    # Box format: (dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
    # where dim0=Z, dim1=Y, dim2=X
    c_d0 = (boxes[:, 0] + boxes[:, 2]) / 2.0  # Z centre
    c_d1 = (boxes[:, 1] + boxes[:, 3]) / 2.0  # Y centre
    c_d2 = (boxes[:, 4] + boxes[:, 5]) / 2.0  # X centre

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
# Inference
# ---------------------------------------------------------------------------

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
    d_dim0 = (boxes[:, 2] - boxes[:, 0]) * spacing_xyz[2]  # Z extent × Z spacing
    d_dim1 = (boxes[:, 3] - boxes[:, 1]) * spacing_xyz[1]  # Y extent × Y spacing
    d_dim2 = (boxes[:, 5] - boxes[:, 4]) * spacing_xyz[0]  # X extent × X spacing
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
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    z1 = np.maximum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])

    inter = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
        * np.maximum(0, z2 - z1)
    )

    def volume(b: np.ndarray) -> np.ndarray:
        return (
            np.maximum(0, b[..., 2] - b[..., 0])
            * np.maximum(0, b[..., 3] - b[..., 1])
            * np.maximum(0, b[..., 5] - b[..., 4])
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
    connected_components: bool = True,
) -> Tuple[sitk.Image, Optional[sitk.Image]]:
    mask = np.zeros(image_shape_zyx, dtype=np.float32)
    for box in detection["boxes"]:
        # Box format: (dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
        # where dim0=Z, dim1=Y, dim2=X (nnDet convention)
        d0a, d1a, d0b, d1b, d2a, d2b = box.astype(int)
        # Clip to image bounds (image_shape_zyx = Z, Y, X)
        d0a, d0b = max(0, d0a), min(image_shape_zyx[0], d0b)
        d1a, d1b = max(0, d1a), min(image_shape_zyx[1], d1b)
        d2a, d2b = max(0, d2a), min(image_shape_zyx[2], d2b)
        mask[d0a:d0b, d1a:d1b, d2a:d2b] = 1.0

    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.CopyInformation(reference_image)
    mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

    cc_sitk = None
    if connected_components:
        cc_sitk = sitk.ConnectedComponent(mask_sitk)

    return mask_sitk, cc_sitk


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
    parser.add_argument("--image-path", required=True, help="Path to input NIfTI image")
    parser.add_argument("--output", required=True, help="Output detection mask (.nii.gz)")

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
        "--providers", nargs="+", default=["CPUExecutionProvider"],
        help="ONNX Runtime execution providers",
    )

    args = parser.parse_args()

    # ---- Load configs ----
    with open(args.plan_path, "rb") as f:
        plan_inference = pkl.load(f)

    patch_size = tuple(plan_inference["patch_size"])  # Z, Y, X

    # Read batch_size from ONNX model input shape (dim 0 of 'images')
    session = ort.InferenceSession(args.model_path, providers=args.providers)
    batch_size: int = session.get_inputs()[0].shape[0]

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = plan_inference["inference_plan"]["model_iou"]

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
    positions = compute_patch_positions(image_shape, patch_size, args.overlap)
    n_patches = len(positions)
    n_batches = math.ceil(n_patches / batch_size)
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
    _, cc = detections_to_mask(
        merged,
        image_shape,
        preprocessed,
        connected_components=True,
    )
    sitk.WriteImage(cc, args.output)
    print(f"      saved → {args.output}  ({time.time() - t0:.2f}s)", flush=True)

    print(f"Done. Total time: {time.time() - t_total:.2f}s", flush=True)


if __name__ == "__main__":
    main()
