"""nnUNet ONNX segmentation inference pipeline using shared common modules."""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

from nninfe.common.cli import collect_image_inputs
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
from nninfe.common.io import write_segmentation_dicom_seg
from nninfe.common.logging_setup import configure_logging
from nninfe.common.manifest import build_run_context, classify_error, write_run_manifest
from nninfe.common.preprocessing import pad_volume_to_patch_size, preprocess_image, resample_mask_to_reference
from nninfe.common.session import BACKENDS, create_session
from nninfe.segmentation.pipeline import (
    build_reference_mask,
    crop_volume_to_shape,
    extract_plan_inference,
    flip_image_axes,
    run_sliding_window_segmentation,
)

# Expected operational failures per stage (everything else is a bug -> propagates to EXIT_RUNTIME).
# Reading: SimpleITK raises RuntimeError, missing files OSError, our own single-series check
# ValueError. Export: adds pydicom's InvalidDicomError (source series re-read by the DICOM writer).
_READ_ERRORS = (OSError, RuntimeError, ValueError)
_EXPORT_ERRORS = (OSError, RuntimeError, ValueError, InvalidDicomError)

logger = logging.getLogger(__name__)


def _resolve_patch_size_zyx(session, plan_patch_size: tuple) -> tuple:
    """Resolve ZYX patch size from plan and model input shape (NCDHW)."""
    shape = session.get_inputs()[0].shape
    spatial = shape[2:5]

    concrete = []
    for d in spatial:
        if isinstance(d, int) and d > 0:
            concrete.append(d)
        else:
            concrete = []
            break

    if len(concrete) == 3:
        model_patch = tuple(concrete)
        if model_patch != tuple(plan_patch_size):
            logger.info(
                f"      patch-size from model input (NCDHW) overrides plan: {tuple(plan_patch_size)} -> {model_patch}"
            )
        return model_patch

    return tuple(plan_patch_size)


def process_single_image(
    session,
    image_path: str,
    output_dir: Path,
    plan_inference: dict,
    patch_size: tuple,
    overlap: float,
    pad_value: str = "0.0",
    output_format: str = "nifti",
    seg_encoding: str = "binary",
) -> str:
    """Process one image and return the primary output path.

    ``output_format`` is ``nifti`` (default), ``dicom-seg`` (a DICOM SEG referencing the source
    series — requires DICOM input), or ``both``. ``seg_encoding`` (``binary`` | ``labelmap``)
    selects the DICOM SEG representation (see ``write_segmentation_dicom_seg``).
    """
    image_name = Path(image_path).name
    if image_name.endswith(".nii.gz"):
        image_name = image_name[:-7]
    elif image_name.endswith(".nii"):
        image_name = image_name[:-4]

    logger.info("[1/3] Preprocessing image …")
    t0 = time.time()
    with translate_errors(ImageIOError, _READ_ERRORS, "failed to read or preprocess input image"):
        preprocessed, orig_meta = preprocess_image(image_path, plan_inference)
        preprocessed_for_infer = flip_image_axes(preprocessed, True, True, False)
        volume = sitk.GetArrayFromImage(preprocessed_for_infer)
    logger.info(f"      preprocessing done  ({time.time() - t0:.2f}s)")

    volume_for_infer, original_shape = pad_volume_to_patch_size(volume, patch_size, pad_value=pad_value)
    if volume_for_infer.shape != volume.shape:
        logger.info(
            f"      padded volume for inference: {volume.shape} -> {volume_for_infer.shape}"
        )

    logger.info("[2/3] Running segmentation inference …")
    batch_size = session.get_inputs()[0].shape[0]
    if not isinstance(batch_size, int) or batch_size <= 0:
        batch_size = 1

    labels = run_sliding_window_segmentation(
        session=session,
        volume_zyx=volume_for_infer,
        patch_size_zyx=patch_size,
        batch_size=batch_size,
        overlap=overlap,
        verbose=True,
        progress_every=3,
    )

    labels = crop_volume_to_shape(labels, original_shape)
    labels_img = sitk.GetImageFromArray(labels)
    labels_img.CopyInformation(preprocessed_for_infer)
    labels_img = flip_image_axes(labels_img, True, True, False)
    labels = sitk.GetArrayFromImage(labels_img)

    logger.info("[3/3] Exporting mask …")
    t0 = time.time()
    outputs = []
    with translate_errors(ExportError, _EXPORT_ERRORS, "failed to export results"):
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_ref = build_reference_mask(
            labels_zyx=labels,
            preprocessed_image=preprocessed,
            reference=orig_meta,
            resample_mask_to_reference=resample_mask_to_reference,
        )

        if output_format in ("nifti", "both"):
            out_nii = str(output_dir / f"{image_name}_seg.nii.gz")
            sitk.WriteImage(mask_ref, out_nii)
            outputs.append(out_nii)
            logger.info(f"      mask  -> {out_nii}")

        if output_format in ("dicom-seg", "both"):
            if Path(image_path).is_dir():
                out_dcm = str(output_dir / f"{image_name}_seg.dcm")
                written = write_segmentation_dicom_seg(
                    mask_ref, image_path, out_dcm, seg_encoding=seg_encoding,
                    source_files=orig_meta.get("source_files"),
                )
                if written:
                    outputs.append(written)
                    logger.info(f"      SEG   -> {written}")
                else:
                    logger.info("      SEG   -> skipped (mask has no foreground to segment)")
            else:
                logger.info(
                    f"      SEG   -> skipped ('{image_name}' is not a DICOM series; "
                    "DICOM SEG output requires DICOM input)"
                )

    # Per-image success sentinel (external orchestration reads {name}.done / {name}.failed).
    status_path = write_image_status(output_dir, image_name, ok=True)
    logger.info(f"      status -> {status_path}")
    logger.info(f"      exports done  ({time.time() - t0:.2f}s)")
    return outputs[0] if outputs else status_path


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="nnUNet ONNX segmentation inference pipeline (sliding window)",
    )
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument("--plan-path", required=True, help="Path to nnUNet plans.json")
    parser.add_argument("--configuration", default="3d_fullres", help="nnUNet plan configuration name")
    parser.add_argument("--image-path", help="Single input image: a NIfTI file (.nii/.nii.gz) or a DICOM series directory")
    parser.add_argument("--image-dir", help="Batch mode: a directory where each entry is one image (a NIfTI file or a DICOM series subdirectory)")
    parser.add_argument("--output-dir", help="Output directory for segmentation masks")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap in [0,1)")
    parser.add_argument(
        "--pad-value",
        default="0.0",
        help="Padding value to use. Can be a number or 'min' to use the minimum value of the image minus 1 (default: 0.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["nifti", "dicom-seg", "both"],
        default="nifti",
        help="Result format: nifti (default), dicom-seg (DICOM SEG referencing the source series, "
        "requires DICOM input), or both",
    )
    parser.add_argument(
        "--seg-encoding",
        choices=["binary", "labelmap"],
        default="binary",
        help="DICOM SEG representation: binary (default, widest viewer support) or labelmap "
        "(compact, size independent of class count — better for many-class masks, needs a newer viewer)",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="cpu",
        help="Inference backend: cpu, openvino, cuda, trt",
    )
    parser.add_argument("--trt-fp16", action="store_true", help="Enable FP16 for TensorRT backend")
    parser.add_argument(
        "--build-engine-only",
        action="store_true",
        help="Build backend session (and TRT cache) then exit",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    plan_path = Path(args.plan_path)
    if not model_path.is_file():
        fail_usage(f"model not found: {args.model_path}")
    if not plan_path.is_file():
        fail_usage(f"plan not found: {args.plan_path}")
    if args.output_dir is None and not args.build_engine_only:
        fail_usage("--output-dir is required for inference")
    if not 0.0 <= args.overlap < 1.0:
        fail_usage(f"--overlap must be in [0, 1), got: {args.overlap}")
    if args.output_format in ("dicom-seg", "both") and args.image_path and not Path(args.image_path).is_dir():
        fail_usage(
            "--output-format dicom-seg/both requires DICOM input (a series directory); "
            f"--image-path is not a directory: {args.image_path}"
        )

    defaults = {a.dest: a.default for a in parser._actions if a.default is not argparse.SUPPRESS}
    logger.info("Parameters:")
    for name, value in vars(args).items():
        tag = ""
        if name in defaults and value == defaults[name]:
            tag = "  (default)"
        logger.info(f"  --{name.replace('_', '-')} : {value}{tag}")
    logger.info("")

    logger.info("Loading plan …")
    with open(plan_path, "r") as f:
        plans = json.load(f)
    plan_inference = extract_plan_inference(plans, configuration=args.configuration)
    plan_patch_size = tuple(plan_inference["patch_size"])
    logger.info(f"      config loaded from: {args.plan_path}")

    if args.backend == "trt":
        precision = "fp16" if args.trt_fp16 else "fp32"
        cache_dir = Path(args.model_path).parent / f"trt_engine_cache_{precision}"
        has_cache = cache_dir.exists() and any(cache_dir.glob("*.engine"))
        if has_cache:
            logger.info(f"      Loading TensorRT session ({precision}, cached engines from {cache_dir}) …")
        else:
            logger.info(
                "      Creating TensorRT session "
                f"({precision}, no cache found, compiling engines — this may take several minutes) …"
            )

    logger.info("Creating ONNX Runtime session …")
    t0 = time.time()
    try:
        session = create_session(str(model_path), backend=args.backend, trt_fp16=args.trt_fp16)
    except SessionError as exc:
        # Fatal: without a session no image can be processed (bad model, driver/engine mismatch).
        logger.error(f"FATAL: {exc}")
        sys.exit(exc.exit_code)
    patch_size = _resolve_patch_size_zyx(session, plan_patch_size)
    logger.info(f"Session ready ({time.time() - t0:.2f}s)")

    if args.build_engine_only:
        logger.info("Engine/session initialized. Exiting.")
        return

    image_paths = collect_image_inputs(args.image_path, args.image_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.image_dir:
        logger.info(f"\n      Found {len(image_paths)} images in {args.image_dir}\n")

    # Run-level manifest context, built once (model/plan hashed a single time for the whole batch).
    run_context = build_run_context(
        pipeline="segmentation",
        model_path=args.model_path,
        plan_path=args.plan_path,
        backend=args.backend,
        providers=session.get_providers(),
        params={
            "configuration": args.configuration, "overlap": args.overlap, "pad_value": args.pad_value,
            "output_format": args.output_format, "seg_encoding": args.seg_encoding,
        },
    )

    t_total = time.time()
    summary = []
    operational_failures = []  # expected per-image failures (typed) -> partial batch
    bug_failures = []          # unexpected errors (programming bugs) -> EXIT_RUNTIME

    for idx, image_path in enumerate(image_paths):
        image_name = image_path.name
        if image_name.endswith(".nii.gz"):
            image_name = image_name[:-7]
        elif image_name.endswith(".nii"):
            image_name = image_name[:-4]

        if len(image_paths) > 1:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Image {idx + 1}/{len(image_paths)}: {image_path.name}")
            logger.info(f"{'='*60}")

        # Per-image fault isolation: a failing image never discards the others' results. Expected
        # (typed) failures are marked and the batch stays "partial"; an unexpected error is a bug —
        # it is logged loudly with a full traceback and forces EXIT_RUNTIME so it is never mistaken
        # for a benign per-image failure (policy (a)).
        t_img = time.time()
        try:
            out_mask = process_single_image(
                session=session,
                image_path=str(image_path),
                output_dir=output_dir,
                plan_inference=plan_inference,
                patch_size=patch_size,
                overlap=args.overlap,
                pad_value=args.pad_value,
                output_format=args.output_format,
                seg_encoding=args.seg_encoding,
            )
            summary.append((image_name, out_mask))
            write_run_manifest(output_dir, image_name, run_context, "ok", duration_s=time.time() - t_img)
        except NnInfeError as exc:
            kind = type(exc).__name__
            # Sober, non-PHI detail (typed name + stable message; the cause is chained for logs).
            logger.error(f"  ERROR [{image_name}]: {kind}: {exc}")
            write_image_status(output_dir, image_name, ok=False, detail=f"{kind}: {exc}")
            write_run_manifest(output_dir, image_name, run_context, "failed",
                               duration_s=time.time() - t_img, error=classify_error(exc))
            summary.append((image_name, None))
            operational_failures.append((image_name, exc))
        except Exception as exc:
            logger.error(f"  INTERNAL ERROR [{image_name}] — this is a bug, not an operational failure:")
            traceback.print_exc()
            write_image_status(output_dir, image_name, ok=False, detail=f"InternalError: {image_name}")
            write_run_manifest(output_dir, image_name, run_context, "failed",
                               duration_s=time.time() - t_img, error=classify_error(exc))
            summary.append((image_name, None))
            bug_failures.append(image_name)

    n_failed = len(operational_failures) + len(bug_failures)
    if len(image_paths) > 1:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Summary: {len(image_paths)} images, {n_failed} failed")
        for name, out_mask in summary:
            logger.info(f"    {name}: {'FAILED' if out_mask is None else out_mask}")

    # Batch-complete marker (kept for backward-compatible orchestration: now means "run
    # finished", while {name}.done / {name}.failed carry each image's actual outcome).
    Path(output_dir, ".done").write_text("done")

    logger.info(f"\nDone. Total time: {time.time() - t_total:.2f}s")

    if bug_failures:
        # A programming bug occurred: never mask it behind the benign "partial" code.
        sys.exit(EXIT_RUNTIME)
    if operational_failures:
        # All images failed the same way -> surface that typed exit code; otherwise partial.
        if len(operational_failures) == len(image_paths):
            sys.exit(operational_failures[0][1].exit_code)
        sys.exit(EXIT_PARTIAL)


if __name__ == "__main__":
    configure_logging()
    logger.info("\nStandalone nnUNet ONNX segmentation pipeline…\n")
    main()
