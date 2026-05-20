"""
nnDet ONNX Inference Pipeline
Standalone inference pipeline for nnDet models exported to ONNX.
Preprocesses a NIfTI image, generates anchors, runs inference, and post-processes detections.
"""

import argparse
import json
import pickle as pkl
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


def compute_anchors(
    model_info: dict,
    patch_size: Tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    """Return anchors array of shape (batch_size, num_anchors, 6)."""
    generator = AnchorGenerator3DSONNX(
        width=model_info["plan_anchors"]["width"],
        height=model_info["plan_anchors"]["height"],
        depth=model_info["plan_anchors"]["depth"],
    )
    anchors = generator.forward(patch_size, model_info["feature_map_size"])
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


def crop_to_patch(
    image: sitk.Image,
    patch_size_zyx: Tuple[int, ...],
    crop_lower_xyz: List[int],
) -> sitk.Image:
    """Crop *image* so the result matches *patch_size_zyx*."""
    size = image.GetSize()  # X, Y, Z
    upper = [
        size[0] - patch_size_zyx[2] - crop_lower_xyz[0],
        size[1] - patch_size_zyx[1] - crop_lower_xyz[1],
        size[2] - patch_size_zyx[0] - crop_lower_xyz[2],
    ]
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize(crop_lower_xyz)
    cropper.SetUpperBoundaryCropSize(upper)
    return cropper.Execute(image)


def preprocess(
    image_path: str,
    plan_inference: dict,
    patch_size_zyx: Tuple[int, ...],
    crop_lower_xyz: List[int],
) -> sitk.Image:
    """Full preprocessing chain: cast → resample → clip → normalize → crop."""
    image = sitk.ReadImage(image_path)
    image = sitk.Cast(image, sitk.sitkFloat32)

    # Resample
    image = resample_image(image, plan_inference["target_spacing"])

    # Clip
    intensity = plan_inference["dataset_properties"]["intensity_properties"][0]
    image = clip_image(
        image,
        lower=intensity["percentile_00_5"],
        upper=intensity["percentile_99_5"],
    )

    # Normalize
    image = normalize_image(image, mean=intensity["mean"], std=intensity["std"])

    # Crop
    image = crop_to_patch(image, patch_size_zyx, crop_lower_xyz)
    return image


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model_path: str,
    input_array: np.ndarray,
    anchors_batch: np.ndarray,
    providers: Optional[List[str]] = None,
) -> list:
    """Run ONNX inference and return raw outputs."""
    if providers is None:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
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


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def filter_by_score(
    detection: Dict[str, np.ndarray],
    score_thresh: float,
) -> Dict[str, np.ndarray]:
    """Keep only detections with score > *score_thresh*."""
    keep = detection["scores"] > score_thresh
    return {k: v[keep] for k, v in detection.items()}


def filter_small_boxes(
    detection: Dict[str, np.ndarray],
    spacing_xyz: Tuple[float, ...],
    min_size_mm: float,
) -> Dict[str, np.ndarray]:
    """Remove boxes where any axis is smaller than *min_size_mm*."""
    boxes = detection["boxes"]
    dx = (boxes[:, 2] - boxes[:, 0]) * spacing_xyz[0]
    dy = (boxes[:, 3] - boxes[:, 1]) * spacing_xyz[1]
    dz = (boxes[:, 5] - boxes[:, 4]) * spacing_xyz[2]
    keep = (dx >= min_size_mm) & (dy >= min_size_mm) & (dz >= min_size_mm)
    return {k: v[keep] for k, v in detection.items()}


# ---- NMS alternative A: nndet + torch ----

def nms_nndet(
    detection: Dict[str, np.ndarray],
    iou_threshold: float,
) -> Dict[str, np.ndarray]:
    """NMS using nndet (requires torch + nndet installed)."""
    import torch
    from nndet.core.boxes import nms

    boxes_t = torch.from_numpy(detection["boxes"])
    scores_t = torch.from_numpy(detection["scores"])
    keep = nms(boxes_t, scores_t, iou_threshold=iou_threshold)
    keep = keep.numpy() if isinstance(keep, torch.Tensor) else np.asarray(keep)
    return {k: v[keep] for k, v in detection.items()}


# ---- NMS alternative B: pure numpy ----

def _iou_3d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes (x1,y1,x2,y2,z1,z2)."""
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
    """Pure-numpy 3D NMS (no torch / nndet dependency)."""
    boxes = detection["boxes"]
    scores = detection["scores"]

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


def postprocess(
    detection: Dict[str, np.ndarray],
    spacing_xyz: Tuple[float, ...],
    score_thresh: float = 0.5,
    min_size_mm: float = 2.0,
    iou_threshold: float = 0.3,
    nms_backend: str = "numpy",
) -> Dict[str, np.ndarray]:
    """Full post-processing chain: score filter → size filter → NMS."""
    detection = filter_by_score(detection, score_thresh)
    detection = filter_small_boxes(detection, spacing_xyz, min_size_mm)

    if nms_backend == "nndet":
        detection = nms_nndet(detection, iou_threshold)
    else:
        detection = nms_numpy(detection, iou_threshold)

    return detection


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def detections_to_mask(
    detection: Dict[str, np.ndarray],
    patch_size_zyx: Tuple[int, ...],
    reference_image: sitk.Image,
    connected_components: bool = True,
) -> Tuple[sitk.Image, Optional[sitk.Image]]:
    """Create a binary mask (and optionally a connected-component label map)."""
    mask = np.zeros(patch_size_zyx, dtype=np.float32)
    for box in detection["boxes"]:
        x1, y1, x2, y2, z1, z2 = box.astype(int)
        mask[x1:x2, y1:y2, z1:z2] = 1.0

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
        description="nnDet ONNX inference pipeline",
    )
    # Paths
    parser.add_argument("--model-path", required=True, help="Path to model_onnx.onnx")
    parser.add_argument("--json-path", required=True, help="Path to model_onnx.json")
    parser.add_argument("--plan-path", required=True, help="Path to plan_inference.pkl")
    parser.add_argument("--image-path", required=True, help="Path to input NIfTI image")
    parser.add_argument("--output-mask", required=True, help="Path for output binary mask (.nii.gz)")
    parser.add_argument("--output-mask-cc", default=None, help="Path for connected-component mask (.nii.gz)")

    # Preprocessing
    parser.add_argument(
        "--crop-lower", type=int, nargs=3, default=[40, 80, 40],
        help="Lower crop boundary in X Y Z (default: 40 80 40)",
    )

    # Post-processing
    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold (default: 0.5)")
    parser.add_argument("--min-size-mm", type=float, default=2.0, help="Min box size in mm (default: 2.0)")
    parser.add_argument("--iou-threshold", type=float, default=None, help="NMS IoU threshold (default: from plan)")
    parser.add_argument(
        "--nms-backend", choices=["numpy", "nndet"], default="numpy",
        help="NMS implementation to use (default: numpy)",
    )

    # Runtime
    parser.add_argument(
        "--providers", nargs="+", default=["CPUExecutionProvider"],
        help="ONNX Runtime execution providers",
    )

    args = parser.parse_args()

    # ---- Load configs ----
    with open(args.json_path, "r") as f:
        model_info = json.load(f)
    with open(args.plan_path, "rb") as f:
        plan_inference = pkl.load(f)

    patch_size = tuple(model_info["patch_size"])  # Z, Y, X
    batch_size = model_info["batch_size"]

    # ---- Anchors ----
    print("[1/4] Computing anchors …")
    anchors_batch = compute_anchors(model_info, patch_size, batch_size)
    print(f"      anchors shape: {anchors_batch.shape}")

    # ---- Preprocessing ----
    print("[2/4] Preprocessing image …")
    cropped_image = preprocess(
        args.image_path,
        plan_inference,
        patch_size,
        args.crop_lower,
    )
    array = sitk.GetArrayFromImage(cropped_image)  # (D, H, W)
    input_array = np.expand_dims(array, axis=(0, 1))  # (1, 1, D, H, W)
    input_array = np.repeat(input_array, batch_size, axis=0).astype(np.float32)
    print(f"      input shape:   {input_array.shape}")

    # ---- Inference ----
    print("[3/4] Running ONNX inference …")
    raw_outputs = run_inference(args.model_path, input_array, anchors_batch, args.providers)
    detections = parse_outputs(raw_outputs, batch_size)
    print(f"      raw detections (batch 0): {len(detections[0]['boxes'])}")

    # ---- Post-processing (batch 0) ----
    print("[4/4] Post-processing …")
    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = plan_inference["inference_plan"]["model_iou"]

    result = postprocess(
        detections[0],
        spacing_xyz=cropped_image.GetSpacing(),
        score_thresh=args.score_thresh,
        min_size_mm=args.min_size_mm,
        iou_threshold=iou_threshold,
        nms_backend=args.nms_backend,
    )
    print(f"      kept detections: {len(result['boxes'])}")

    # ---- Export ----
    mask, cc = detections_to_mask(
        result,
        patch_size,
        cropped_image,
        connected_components=args.output_mask_cc is not None,
    )
    sitk.WriteImage(mask, args.output_mask)
    print(f"      saved mask → {args.output_mask}")

    if cc is not None and args.output_mask_cc:
        sitk.WriteImage(cc, args.output_mask_cc)
        print(f"      saved CC   → {args.output_mask_cc}")

    print("Done.")


if __name__ == "__main__":
    main()
