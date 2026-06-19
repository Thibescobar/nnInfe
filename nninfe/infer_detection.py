"""nnDet ONNX inference pipeline (detection) built on shared modules."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

from nninfe.common.cli import collect_nifti_inputs
from nninfe.common.io import read_image_metadata, resample_mask_to_reference
from nninfe.common.preprocessing import preprocess_image
from nninfe.common.session import BACKENDS, create_session, parse_outputs, run_inference
from nninfe.common.sliding_window import compute_patch_positions, extract_patch
from nninfe.detection.anchors import compute_anchors
from nninfe.detection.export import (
    detections_to_mask,
    export_detections_csv,
    export_detections_json,
    export_detections_pkl,
)
from nninfe.detection.postprocessing import (
    apply_nms,
    clip_boxes_to_image_shape,
    gaussian_weight_for_boxes,
    merge_detections,
    postprocess,
    translate_boxes,
)
from nninfe.segmentation.pipeline import pad_volume_to_patch_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="nnDet ONNX inference pipeline (sliding window)",
    )

    parser.add_argument("--model-path", required=True, help="Path to model_onnx.onnx")
    parser.add_argument("--plan-path", required=True, help="Path to plan_inference.json")
    parser.add_argument("--image-path", help="Path to a single input NIfTI image")
    parser.add_argument("--image-dir", help="Path to a directory of NIfTI images (batch mode)")
    parser.add_argument("--output-dir", help="Output directory for results (mask, JSON, CSV)")

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap between patches as proportion in [0,1) (default: 0.5)",
    )

    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold (default: 0.5)")
    parser.add_argument("--min-size-mm", type=float, default=2.0, help="Min box size in mm (default: 2.0)")
    parser.add_argument("--iou-threshold", type=float, default=None, help="NMS IoU threshold (default: from plan)")
    parser.add_argument(
        "--nms-backend",
        choices=["numpy", "nndet"],
        default="numpy",
        help="NMS implementation (default: numpy)",
    )
    parser.add_argument(
        "--no-global-nms",
        action="store_true",
        help="Disable the final global NMS after merging all patches",
    )
    parser.add_argument(
        "--pad-value",
        default="0.0",
        help="Padding value to use. Can be a number or 'min' to use the minimum value of the image minus 1 (default: 0.0)",
    )

    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="cpu",
        help="Inference backend: cpu, openvino, tensorrt (default: cpu)",
    )
    parser.add_argument(
        "--trt-fp16",
        action="store_true",
        help="Enable FP16 inference for TensorRT backend",
    )
    parser.add_argument(
        "--build-engine-only",
        action="store_true",
        help="Build TRT engine cache and exit (no inference)",
    )
    parser.add_argument(
        "--export-pkl",
        action="store_true",
        help="Also export detections as .pkl (nnDetection-compatible, for validation)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    plan_path = Path(args.plan_path)

    if not model_path.is_file():
        sys.exit(f"Error: model not found: {args.model_path}")
    if model_path.suffix != ".onnx":
        sys.exit(f"Error: model must be .onnx, got: {model_path.suffix}")
    if not plan_path.is_file():
        sys.exit(f"Error: plan not found: {args.plan_path}")
    if plan_path.suffix != ".json":
        sys.exit(f"Error: plan must be .json, got: {plan_path.suffix}")

    if not args.build_engine_only:
        if not args.output_dir:
            sys.exit("Error: --output-dir is required for inference")
        image_paths = collect_nifti_inputs(args.image_path, args.image_dir)
    else:
        image_paths = []

    if not 0.0 <= args.overlap < 1.0:
        sys.exit(f"Error: --overlap must be in [0, 1), got: {args.overlap}")
    if not 0.0 <= args.score_thresh <= 1.0:
        sys.exit(f"Error: --score-thresh must be in [0, 1], got: {args.score_thresh}")

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

    with open(args.plan_path, "r") as f:
        plan_inference = json.load(f)
    print(f"      config loaded from: {args.plan_path}", flush=True)

    patch_size = tuple(plan_inference["patch_size"])

    if args.backend == "trt":
        precision = "fp16" if args.trt_fp16 else "fp32"
        cache_dir = Path(args.model_path).parent / f"trt_engine_cache_{precision}"
        has_cache = cache_dir.exists() and any(cache_dir.glob("*.engine"))
        if has_cache:
            print(f"      Loading TensorRT session ({precision}, cached engines from {cache_dir}) ...", flush=True)
        else:
            print(
                "      Creating TensorRT session "
                f"({precision}, no cache found, compiling engines -- this may take several minutes) ...",
                flush=True,
            )

    t_session = time.time()
    session = create_session(args.model_path, args.backend, args.trt_fp16)
    batch_size: int = session.get_inputs()[0].shape[0]
    print(f"      Session ready  ({time.time() - t_session:.2f}s)", flush=True)

    if args.build_engine_only:
        print("\nEngine built and cached. Exiting.", flush=True)
        return

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = plan_inference["inference_plan"]["model_iou"]
        print(f"      iou-threshold not specified, using plan value: {iou_threshold}\n", flush=True)

    print("[1/5] Computing anchors ...", flush=True)
    t0 = time.time()
    anchors_batch = compute_anchors(plan_inference, patch_size, batch_size)
    print(f"      anchors shape: {anchors_batch.shape}  ({time.time() - t0:.2f}s)", flush=True)

    if args.image_dir:
        print(f"\n      Found {len(image_paths)} images in {args.image_dir}\n", flush=True)

    t_total = time.time()
    summary = []

    for img_idx, img_path in enumerate(image_paths):
        image_name = img_path.name
        if image_name.endswith(".nii.gz"):
            image_name = image_name[:-7]
        elif image_name.endswith(".nii"):
            image_name = image_name[:-4]
            
        if len(image_paths) > 1:
            print(f"\n{'='*60}", flush=True)
            print(f"  Image {img_idx + 1}/{len(image_paths)}: {img_path.name}", flush=True)
            print(f"{'='*60}", flush=True)

        result = process_single_image(
            session=session,
            image_path=str(img_path),
            output_dir=Path(args.output_dir),
            plan_inference=plan_inference,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=iou_threshold,
            overlap=args.overlap,
            score_thresh=args.score_thresh,
            min_size_mm=args.min_size_mm,
            nms_backend=args.nms_backend,
            no_global_nms=args.no_global_nms,
            export_pkl=args.export_pkl,
            pad_value=args.pad_value,
        )
        summary.append((image_name, result))

    if len(image_paths) > 1:
        print(f"\n{'='*60}", flush=True)
        print(f"  Summary: {len(image_paths)} images processed", flush=True)
        for name, n_det in summary:
            print(f"    {name}: {n_det} detections", flush=True)

    print(f"\nDone. Total time: {time.time() - t_total:.2f}s", flush=True)


def process_single_image(
    session,
    image_path: str,
    output_dir: Path,
    plan_inference: dict,
    patch_size: tuple,
    batch_size: int,
    anchors_batch: np.ndarray,
    iou_threshold: float,
    overlap: float,
    score_thresh: float,
    min_size_mm: float,
    nms_backend: str,
    no_global_nms: bool,
    export_pkl: bool,
    pad_value: str,
) -> int:
    """Process a single image through the full pipeline. Returns detection count."""
    image_name = Path(image_path).name
    if image_name.endswith(".nii.gz"):
        image_name = image_name[:-7]
    elif image_name.endswith(".nii"):
        image_name = image_name[:-4]

    print("[2/5] Preprocessing image ...", flush=True)
    t0 = time.time()
    preprocessed = preprocess_image(image_path, plan_inference)
    volume = sitk.GetArrayFromImage(preprocessed)
    spacing_xyz = preprocessed.GetSpacing()
    print(f"      preprocessing done  ({time.time() - t0:.2f}s)", flush=True)

    volume_padded, original_shape = pad_volume_to_patch_size(volume, patch_size, pad_value=pad_value)
    if volume_padded.shape != original_shape:
        print(
            f"      padded volume for inference: {original_shape} -> {volume_padded.shape}",
            flush=True,
        )
    padded_shape = volume_padded.shape

    print("[3/5] Building sliding window positions ...", flush=True)
    t0 = time.time()
    positions, step_sizes = compute_patch_positions(padded_shape, patch_size, overlap)
    actual_overlap = tuple(round(1.0 - s / p, 4) if p > 0 else 0.0 for s, p in zip(step_sizes, patch_size))
    n_patches = len(positions)
    n_batches = math.ceil(n_patches / batch_size)
    print(
        f"      step sizes (ZYX): ({step_sizes[0]:.1f}, {step_sizes[1]:.1f}, {step_sizes[2]:.1f})  "
        f"actual overlap: {actual_overlap}  (requested: {overlap})",
        flush=True,
    )
    print(f"      {n_patches} patches, {n_batches} batches (batch_size={batch_size})  ({time.time() - t0:.2f}s)", flush=True)

    print("[4/5] Running inference ...", flush=True)
    t0 = time.time()

    all_detections: List[Dict[str, np.ndarray]] = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_patches)
        batch_positions = positions[start:end]
        actual_count = len(batch_positions)

        patches = [extract_patch(volume_padded, pos, patch_size) for pos in batch_positions]
        while len(patches) < batch_size:
            patches.append(patches[-1])

        input_array = np.stack([p[np.newaxis, ...] for p in patches], axis=0).astype(np.float32)

        raw_outputs = run_inference(session, input_array, anchors_batch)
        detections = parse_outputs(raw_outputs, batch_size)

        for i in range(actual_count):
            det = postprocess(
                detections[i],
                spacing_xyz=spacing_xyz,
                score_thresh=score_thresh,
                min_size_mm=min_size_mm,
                iou_threshold=iou_threshold,
                nms_backend=nms_backend,
            )

            if len(det["boxes"]) > 0:
                weights = gaussian_weight_for_boxes(det["boxes"], patch_size)
                det = {
                    **det,
                    "scores": det["scores"] * weights,
                    "scores_original": det["scores"].copy(),
                }

            det = translate_boxes(det, batch_positions[i])
            if len(det["boxes"]) > 0:
                all_detections.append(det)

        progress = (batch_idx + 1) / n_batches
        elapsed = time.time() - t0
        time_per_iter = elapsed / (batch_idx + 1)
        remaining = time_per_iter * (n_batches - batch_idx - 1)

        n_printed_batch = 3
        if batch_idx % n_printed_batch == 0 or batch_idx == n_batches - 1:
            print(
                f"            progress {progress:6.1%}  "
                f"batch {batch_idx + 1}/{n_batches}  "
                f"patch {end}/{n_patches}  "
                f"{time_per_iter:.2f}s/batch  "
                f"elapsed {elapsed:.0f}s  remaining {remaining:.0f}s",
                flush=True,
            )

    print(f"      inference done ({time.time() - t0:.2f}s)", flush=True)

    print("[5/5] Merging detections ...", flush=True)
    t0 = time.time()
    merged = merge_detections(all_detections)
    print(f"      total detections before global NMS: {len(merged['boxes'])}", flush=True)

    if not no_global_nms and len(merged["boxes"]) > 0:
        merged = apply_nms(merged, iou_threshold, nms_backend)
        print(f"      detections after global NMS:        {len(merged['boxes'])}  ({time.time() - t0:.2f}s)", flush=True)
    elif no_global_nms:
        print(f"      global NMS disabled  ({time.time() - t0:.2f}s)", flush=True)
    else:
        print(f"      no detections  ({time.time() - t0:.2f}s)", flush=True)

    if "scores_original" in merged:
        merged["scores"] = merged.pop("scores_original")
        
    merged = clip_boxes_to_image_shape(merged, original_shape)

    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    mask_path = str(output_dir / f"{image_name}_mask.nii.gz")
    cc = detections_to_mask(merged, original_shape, preprocessed)
    cc = resample_mask_to_reference(cc, image_path)
    sitk.WriteImage(cc, mask_path)
    print(f"      mask  -> {mask_path}", flush=True)

    json_path = str(output_dir / f"{image_name}_boxes.json")

    orig_meta = read_image_metadata(image_path)
    resampled_meta = {
        "size_xyz": preprocessed.GetSize(),
        "spacing_xyz": preprocessed.GetSpacing(),
        "origin": preprocessed.GetOrigin(),
        "direction": preprocessed.GetDirection(),
    }
    export_detections_json(merged, json_path, ref_meta=orig_meta, current_meta=resampled_meta)
    print(f"      boxes -> {json_path}", flush=True)

    csv_path = str(output_dir / f"{image_name}_boxes.csv")
    export_detections_csv(
        merged,
        csv_path,
        image_name=image_name,
        ref_meta=orig_meta,
        current_meta=resampled_meta,
    )
    print(f"      csv   -> {csv_path}", flush=True)

    if export_pkl:
        pkl_path = str(output_dir / f"{image_name}_boxes.pkl")
        export_detections_pkl(merged, pkl_path, ref_meta=orig_meta, current_meta=resampled_meta)
        print(f"      pkl   -> {pkl_path}", flush=True)

    print(f"      exports done  ({time.time() - t0:.2f}s)", flush=True)

    n_detections = len(merged["boxes"])

    done_path = str(output_dir / ".done")
    with open(done_path, "w") as f:
        f.write("done")
    print(f"      .done file -> {done_path}", flush=True)

    return n_detections


if __name__ == "__main__":
    print("\nStandalone nnDetection ONNX inference pipeline...\n", flush=True)
    main()
