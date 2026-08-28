"""nnDet ONNX inference pipeline (detection) built on shared modules."""

import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

from nninfe.common.cli import collect_image_inputs, log_parameters, make_parser
from nninfe.common.errors import (
    EXIT_PARTIAL,
    EXIT_RUNTIME,
    ExportError,
    ImageIOError,
    NnInfeError,
    SessionError,
    fail_usage,
    translate_errors,
    write_image_status,
)
from nninfe.common.io import write_detection_dicom_sr
from nninfe.common.logging_setup import configure_logging
from nninfe.common.manifest import build_run_context, classify_error, write_run_manifest
from nninfe.common.preprocessing import pad_volume_to_patch_size, preprocess_image, resample_mask_to_reference
from nninfe.common.session import (
    BACKENDS,
    build_only_completion_message,
    create_session,
    parse_outputs,
    run_inference,
    session_creation_message,
)
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

# Expected operational failures per stage (everything else is a bug -> propagates to EXIT_RUNTIME).
# Reading: SimpleITK raises RuntimeError, missing files OSError, our own single-series check
# ValueError. Export: adds pydicom's InvalidDicomError (source series re-read by the DICOM writers).
_READ_ERRORS = (OSError, RuntimeError, ValueError)
_EXPORT_ERRORS = (OSError, RuntimeError, ValueError, InvalidDicomError)

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    parser = make_parser("nnDet ONNX inference pipeline (sliding window)")

    parser.add_argument("--model-path", required=True, help="Path to model_onnx.onnx")
    parser.add_argument("--plan-path", required=True, help="Path to plan_inference.json")
    parser.add_argument("--image-path", help="Single input image: a NIfTI file (.nii/.nii.gz) or a DICOM series directory")
    parser.add_argument("--image-dir", help="Batch mode: a directory where each entry is one image (a NIfTI file or a DICOM series subdirectory)")
    parser.add_argument("--output-dir", help="Output directory for results (mask, JSON, CSV)")

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap between patches as proportion in [0,1)",
    )

    parser.add_argument("--score-thresh", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--min-size-mm", type=float, default=2.0, help="Min box size in mm")
    parser.add_argument("--iou-threshold", type=float, default=None, help="NMS IoU threshold (read from plan when omitted)")
    parser.add_argument(
        "--nms-backend",
        choices=["numpy", "nndet"],
        default="numpy",
        help="NMS implementation",
    )
    parser.add_argument(
        "--no-global-nms",
        action="store_true",
        help="Disable the final global NMS after merging all patches",
    )
    parser.add_argument(
        "--pad-value",
        default="0.0",
        help="Padding value to use. Can be a number or 'min' to use the minimum value of the image minus 1",
    )

    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="cpu",
        help="Inference backend",
    )
    parser.add_argument(
        "--trt-fp16",
        action="store_true",
        help="Enable FP16 inference for TensorRT backend",
    )
    parser.add_argument(
        "--build-engine-only",
        action="store_true",
        help="Initialize the backend session and exit; with TRT, build or load the engine cache",
    )
    parser.add_argument(
        "--export-pkl",
        action="store_true",
        help="Also export detections as .pkl (nnDetection-compatible, for validation)",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "nifti", "dicom-sr", "both"],
        default="nifti",
        help="Result format; the boxes JSON is always written (and PKL if --export-pkl). "
        "json = JSON only; nifti adds the mask NIfTI + CSV; dicom-sr adds a DICOM "
        "Structured Report (requires DICOM input); both = nifti + dicom-sr",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    plan_path = Path(args.plan_path)

    if not model_path.is_file():
        fail_usage(f"model not found: {args.model_path}")
    if model_path.suffix != ".onnx":
        fail_usage(f"model must be .onnx, got: {model_path.suffix}")
    if not plan_path.is_file():
        fail_usage(f"plan not found: {args.plan_path}")
    if plan_path.suffix != ".json":
        fail_usage(f"plan must be .json, got: {plan_path.suffix}")

    if not args.build_engine_only:
        if not args.output_dir:
            fail_usage("--output-dir is required for inference")
        image_paths = collect_image_inputs(args.image_path, args.image_dir)
    else:
        image_paths = []

    if not 0.0 <= args.overlap < 1.0:
        fail_usage(f"--overlap must be in [0, 1), got: {args.overlap}")
    if not 0.0 <= args.score_thresh <= 1.0:
        fail_usage(f"--score-thresh must be in [0, 1], got: {args.score_thresh}")
    if args.output_format in ("dicom-sr", "both") and args.image_path and not Path(args.image_path).is_dir():
        fail_usage(
            "--output-format dicom-sr/both requires DICOM input (a series directory); "
            f"--image-path is not a directory: {args.image_path}"
        )

    log_parameters(logger, parser, args, dynamic_defaults={"iou_threshold": "from plan"})

    logger.info("Loading plan …")
    with open(args.plan_path, "r") as f:
        plan_inference = json.load(f)
    logger.info(f"      config loaded from: {args.plan_path}")

    patch_size = tuple(plan_inference["patch_size"])

    logger.info(session_creation_message(args.model_path, args.backend, args.trt_fp16))
    t_session = time.time()
    try:
        session = create_session(args.model_path, args.backend, args.trt_fp16)
    except SessionError as exc:
        # Fatal: without a session no image can be processed (bad model, driver/engine mismatch).
        logger.error(f"FATAL: {exc}")
        sys.exit(exc.exit_code)
    batch_size: int = session.get_inputs()[0].shape[0]
    logger.info(f"Session ready ({time.time() - t_session:.2f}s)")

    if args.build_engine_only:
        logger.info(build_only_completion_message(args.backend))
        return

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = plan_inference["inference_plan"]["model_iou"]
        logger.info(f"      iou-threshold not specified, using plan value: {iou_threshold}")

    logger.info("[1/5] Computing anchors …")
    t0 = time.time()
    anchors_batch = compute_anchors(plan_inference, patch_size, batch_size)
    logger.info(f"      anchors shape: {anchors_batch.shape}  ({time.time() - t0:.2f}s)")

    if args.image_dir:
        logger.info(f"      Found {len(image_paths)} images in {args.image_dir}")

    # Run-level manifest context, built once (model/plan hashed a single time for the whole batch).
    run_context = build_run_context(
        pipeline="detection",
        model_path=args.model_path,
        plan_path=args.plan_path,
        backend=args.backend,
        providers=session.get_providers(),
        params={
            "overlap": args.overlap, "score_thresh": args.score_thresh, "min_size_mm": args.min_size_mm,
            "iou_threshold": iou_threshold, "nms_backend": args.nms_backend,
            "no_global_nms": args.no_global_nms, "pad_value": args.pad_value,
            "output_format": args.output_format, "export_pkl": args.export_pkl,
        },
    )

    t_total = time.time()
    summary = []
    operational_failures = []  # expected per-image failures (typed) -> partial batch
    bug_failures = []          # unexpected errors (programming bugs) -> EXIT_RUNTIME

    for img_idx, img_path in enumerate(image_paths):
        image_name = img_path.name
        if image_name.endswith(".nii.gz"):
            image_name = image_name[:-7]
        elif image_name.endswith(".nii"):
            image_name = image_name[:-4]

        if len(image_paths) > 1:
            logger.info(f"{'='*60}")
            logger.info(f"  Image {img_idx + 1}/{len(image_paths)}: {img_path.name}")
            logger.info(f"{'='*60}")

        # Per-image fault isolation: a failing image never discards the others' results. Expected
        # (typed) failures are marked and the batch stays "partial"; an unexpected error is a bug —
        # it is logged loudly with a full traceback and forces EXIT_RUNTIME so it is never mistaken
        # for a benign per-image failure (policy (a)).
        t_img = time.time()
        try:
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
                output_format=args.output_format,
            )
            summary.append((image_name, result))
            write_run_manifest(args.output_dir, image_name, run_context, "ok", duration_s=time.time() - t_img)
        except NnInfeError as exc:
            kind = type(exc).__name__
            # Sober, non-PHI detail (typed name + stable message; the cause is chained for logs).
            logger.error(f"  ERROR [{image_name}]: {kind}: {exc}")
            write_image_status(args.output_dir, image_name, ok=False, detail=f"{kind}: {exc}")
            write_run_manifest(args.output_dir, image_name, run_context, "failed",
                               duration_s=time.time() - t_img, error=classify_error(exc))
            summary.append((image_name, None))
            operational_failures.append((image_name, exc))
        except Exception as exc:
            logger.error(f"  INTERNAL ERROR [{image_name}] — this is a bug, not an operational failure:")
            traceback.print_exc()
            write_image_status(args.output_dir, image_name, ok=False, detail=f"InternalError: {image_name}")
            write_run_manifest(args.output_dir, image_name, run_context, "failed",
                               duration_s=time.time() - t_img, error=classify_error(exc))
            summary.append((image_name, None))
            bug_failures.append(image_name)

    n_failed = len(operational_failures) + len(bug_failures)
    if len(image_paths) > 1:
        logger.info(f"{'='*60}")
        logger.info(f"  Summary: {len(image_paths)} images, {n_failed} failed")
        for name, n_det in summary:
            status = "FAILED" if n_det is None else f"{n_det} detections"
            logger.info(f"    {name}: {status}")

    # Batch-complete marker (kept for backward-compatible orchestration: now means "run
    # finished", while {name}.done / {name}.failed carry each image's actual outcome).
    Path(args.output_dir, ".done").write_text("done")

    logger.info(f"Done. Total time: {time.time() - t_total:.2f}s")

    if bug_failures:
        # A programming bug occurred: never mask it behind the benign "partial" code.
        sys.exit(EXIT_RUNTIME)
    if operational_failures:
        # All images failed the same way -> surface that typed exit code; otherwise partial.
        if len(operational_failures) == len(image_paths):
            sys.exit(operational_failures[0][1].exit_code)
        sys.exit(EXIT_PARTIAL)


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
    output_format: str = "nifti",
) -> int:
    """Process a single image through the full pipeline. Returns detection count.

    The ``_boxes.json`` detection record is always written (and ``_boxes.pkl`` whenever
    ``export_pkl``). ``output_format`` then selects the rest: ``json`` (nothing further),
    ``nifti`` (default: adds the NIfTI mask + CSV), ``dicom-sr`` (adds a DICOM SR referencing the
    source series — DICOM input only), or ``both`` (nifti + dicom-sr).
    """
    image_name = Path(image_path).name
    if image_name.endswith(".nii.gz"):
        image_name = image_name[:-7]
    elif image_name.endswith(".nii"):
        image_name = image_name[:-4]

    logger.info("[2/5] Preprocessing image …")
    t0 = time.time()
    with translate_errors(ImageIOError, _READ_ERRORS, "failed to read or preprocess input image"):
        preprocessed, orig_meta = preprocess_image(image_path, plan_inference)
        volume = sitk.GetArrayFromImage(preprocessed)
    spacing_xyz = preprocessed.GetSpacing()
    logger.info(f"      preprocessing done  ({time.time() - t0:.2f}s)")

    volume_padded, original_shape = pad_volume_to_patch_size(volume, patch_size, pad_value=pad_value)
    if volume_padded.shape != original_shape:
        logger.info(
            f"      padded volume for inference: {original_shape} -> {volume_padded.shape}"
        )
    padded_shape = volume_padded.shape

    logger.info("[3/5] Building sliding window positions …")
    t0 = time.time()
    positions, step_sizes = compute_patch_positions(padded_shape, patch_size, overlap)
    actual_overlap = tuple(round(1.0 - s / p, 4) if p > 0 else 0.0 for s, p in zip(step_sizes, patch_size))
    n_patches = len(positions)
    n_batches = math.ceil(n_patches / batch_size)
    logger.info(
        f"      step sizes (ZYX): ({step_sizes[0]:.1f}, {step_sizes[1]:.1f}, {step_sizes[2]:.1f})  "
        f"actual overlap: {actual_overlap}  (requested: {overlap})"
    )
    logger.info(f"      {n_patches} patches, {n_batches} batches (batch_size={batch_size})  ({time.time() - t0:.2f}s)")

    logger.info("[4/5] Running inference …")
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

        input_array = np.stack([p[np.newaxis, ...] for p in patches], axis=0).astype(np.float32, copy=False)

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
            logger.info(
                f"            progress {progress:6.1%}  "
                f"batch {batch_idx + 1}/{n_batches}  "
                f"patch {end}/{n_patches}  "
                f"{time_per_iter:.2f}s/batch  "
                f"elapsed {elapsed:.0f}s  remaining {remaining:.0f}s"
            )

    logger.info(f"      inference done ({time.time() - t0:.2f}s)")

    logger.info("[5/5] Merging detections …")
    t0 = time.time()
    merged = merge_detections(all_detections)
    logger.info(f"      total detections before global NMS: {len(merged['boxes'])}")

    if not no_global_nms and len(merged["boxes"]) > 0:
        merged = apply_nms(merged, iou_threshold, nms_backend)
        logger.info(f"      detections after global NMS:        {len(merged['boxes'])}  ({time.time() - t0:.2f}s)")
    elif no_global_nms:
        logger.info(f"      global NMS disabled  ({time.time() - t0:.2f}s)")
    else:
        logger.info(f"      no detections  ({time.time() - t0:.2f}s)")

    if "scores_original" in merged:
        merged["scores"] = merged.pop("scores_original")

    merged = clip_boxes_to_image_shape(merged, original_shape)

    t0 = time.time()
    with translate_errors(ExportError, _EXPORT_ERRORS, "failed to export results"):
        os.makedirs(output_dir, exist_ok=True)

        resampled_meta = {
            "size_xyz": preprocessed.GetSize(),
            "spacing_xyz": preprocessed.GetSpacing(),
            "origin": preprocessed.GetOrigin(),
            "direction": preprocessed.GetDirection(),
        }

        # The JSON detection record is always written, regardless of --output-format; the PKL is
        # written whenever explicitly requested (both are box records, independent of the rendering).
        json_path = str(output_dir / f"{image_name}_boxes.json")
        export_detections_json(merged, json_path, ref_meta=orig_meta, current_meta=resampled_meta)
        logger.info(f"      boxes -> {json_path}")

        if export_pkl:
            pkl_path = str(output_dir / f"{image_name}_boxes.pkl")
            export_detections_pkl(merged, pkl_path, ref_meta=orig_meta, current_meta=resampled_meta)
            logger.info(f"      pkl   -> {pkl_path}")

        if output_format in ("nifti", "both"):
            mask_path = str(output_dir / f"{image_name}_mask.nii.gz")
            cc = detections_to_mask(merged, original_shape, preprocessed)
            cc = resample_mask_to_reference(cc, orig_meta)
            sitk.WriteImage(cc, mask_path)
            logger.info(f"      mask  -> {mask_path}")

            csv_path = str(output_dir / f"{image_name}_boxes.csv")
            export_detections_csv(
                merged,
                csv_path,
                image_name=image_name,
                ref_meta=orig_meta,
                current_meta=resampled_meta,
            )
            logger.info(f"      csv   -> {csv_path}")

        if output_format in ("dicom-sr", "both"):
            if Path(image_path).is_dir():
                sr_path = str(output_dir / f"{image_name}_sr.dcm")
                written = write_detection_dicom_sr(
                    merged["boxes"],
                    merged["scores"],
                    merged["labels"],
                    image_path,
                    orig_meta,
                    sr_path,
                    current_meta=resampled_meta,
                    source_files=orig_meta.get("source_files"),
                )
                if written:
                    logger.info(f"      SR    -> {written}")
                else:
                    logger.info("      SR    -> skipped (no detections)")
            else:
                logger.info(
                    f"      SR    -> skipped ('{image_name}' is not a DICOM series; "
                    "DICOM SR output requires DICOM input)"
                )

    logger.info(f"      exports done  ({time.time() - t0:.2f}s)")

    n_detections = len(merged["boxes"])

    # Per-image success sentinel (external orchestration reads {name}.done / {name}.failed).
    status_path = write_image_status(output_dir, image_name, ok=True)
    logger.info(f"      status -> {status_path}")

    return n_detections


if __name__ == "__main__":
    configure_logging()
    logger.info("Standalone nnDetection ONNX inference pipeline…")
    main()
