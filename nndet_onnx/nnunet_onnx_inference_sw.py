"""nnUNet ONNX segmentation inference pipeline using shared common modules."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import SimpleITK as sitk

from nndet_onnx.common.cli import collect_nifti_inputs
from nndet_onnx.common.io import resample_mask_to_reference
from nndet_onnx.common.preprocessing import preprocess_image
from nndet_onnx.common.session import BACKENDS, create_session
from nndet_onnx.segmentation.pipeline import (
    crop_volume_to_shape,
    export_segmentation_mask,
    extract_plan_inference,
    flip_image_axes,
    pad_volume_to_patch_size,
    run_sliding_window_segmentation,
)


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
            print(
                f"      patch-size from model input (NCDHW) overrides plan: {tuple(plan_patch_size)} -> {model_patch}",
                flush=True,
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
) -> str:
    """Process one image and return output mask path."""
    image_name = Path(image_path).name
    if image_name.endswith(".nii.gz"):
        image_name = image_name[:-7]
    elif image_name.endswith(".nii"):
        image_name = image_name[:-4]

    print("[1/3] Preprocessing image …", flush=True)
    t0 = time.time()
    preprocessed = preprocess_image(image_path, plan_inference)
    preprocessed_for_infer = flip_image_axes(preprocessed, True, True, False)
    volume = sitk.GetArrayFromImage(preprocessed_for_infer)
    print(f"      preprocessing done  ({time.time() - t0:.2f}s)", flush=True)

    volume_for_infer, original_shape = pad_volume_to_patch_size(volume, patch_size)
    if volume_for_infer.shape != volume.shape:
        print(
            f"      padded volume for inference: {volume.shape} -> {volume_for_infer.shape}",
            flush=True,
        )

    print("[2/3] Running segmentation inference …", flush=True)
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

    print("[3/3] Exporting mask …", flush=True)
    t0 = time.time()
    out_mask = str(output_dir / f"{image_name}_seg.nii.gz")
    export_segmentation_mask(
        labels_zyx=labels,
        preprocessed_image=preprocessed,
        reference_image_path=image_path,
        output_path=out_mask,
        resample_mask_to_reference=resample_mask_to_reference,
    )
    done_path = str(output_dir / ".done")
    with open(done_path, "w") as f:
        f.write("done")
    print(f"      mask  -> {out_mask}", flush=True)
    print(f"      .done -> {done_path}", flush=True)
    print(f"      exports done  ({time.time() - t0:.2f}s)", flush=True)
    return out_mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="nnUNet ONNX segmentation inference pipeline (sliding window)",
    )
    parser.add_argument("--model-path", required=True, help="Path to ONNX model")
    parser.add_argument("--plan-path", required=True, help="Path to nnUNet plans.json")
    parser.add_argument("--configuration", default="3d_fullres", help="nnUNet plan configuration name")
    parser.add_argument("--image-path", help="Path to a single input NIfTI image")
    parser.add_argument("--image-dir", help="Path to a directory of NIfTI images")
    parser.add_argument("--output-dir", help="Output directory for segmentation masks")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap in [0,1)")
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
        sys.exit(f"Error: model not found: {args.model_path}")
    if not plan_path.is_file():
        sys.exit(f"Error: plan not found: {args.plan_path}")
    if args.output_dir is None and not args.build_engine_only:
        sys.exit("Error: --output-dir is required for inference")
    if not 0.0 <= args.overlap < 1.0:
        sys.exit(f"Error: --overlap must be in [0, 1), got: {args.overlap}")

    defaults = {a.dest: a.default for a in parser._actions if a.default is not argparse.SUPPRESS}
    print("Parameters:", flush=True)
    for name, value in vars(args).items():
        tag = ""
        if name in defaults and value == defaults[name]:
            tag = "  (default)"
        print(f"  --{name.replace('_', '-')} : {value}{tag}", flush=True)
    print(flush=True)

    print("Loading plan …", flush=True)
    with open(plan_path, "r") as f:
        plans = json.load(f)
    plan_inference = extract_plan_inference(plans, configuration=args.configuration)
    plan_patch_size = tuple(plan_inference["patch_size"])
    print(f"      config loaded from: {args.plan_path}", flush=True)

    if args.backend == "trt":
        precision = "fp16" if args.trt_fp16 else "fp32"
        cache_dir = Path(args.model_path).parent / f"trt_engine_cache_{precision}"
        has_cache = cache_dir.exists() and any(cache_dir.glob("*.engine"))
        if has_cache:
            print(f"      Loading TensorRT session ({precision}, cached engines from {cache_dir}) …", flush=True)
        else:
            print(
                "      Creating TensorRT session "
                f"({precision}, no cache found, compiling engines — this may take several minutes) …",
                flush=True,
            )

    print("Creating ONNX Runtime session …", flush=True)
    t0 = time.time()
    session = create_session(str(model_path), backend=args.backend, trt_fp16=args.trt_fp16)
    patch_size = _resolve_patch_size_zyx(session, plan_patch_size)
    print(f"Session ready ({time.time() - t0:.2f}s)", flush=True)

    if args.build_engine_only:
        print("Engine/session initialized. Exiting.", flush=True)
        return

    image_paths = collect_nifti_inputs(args.image_path, args.image_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.image_dir:
        print(f"\n      Found {len(image_paths)} images in {args.image_dir}\n", flush=True)

    t_total = time.time()
    summary = []

    for idx, image_path in enumerate(image_paths):
        image_name = image_path.name
        if image_name.endswith(".nii.gz"):
            image_name = image_name[:-7]
        elif image_name.endswith(".nii"):
            image_name = image_name[:-4]
            
        if len(image_paths) > 1:
            print(f"\n{'='*60}", flush=True)
            print(f"  Image {idx + 1}/{len(image_paths)}: {image_path.name}", flush=True)
            print(f"{'='*60}", flush=True)
        out_mask = process_single_image(
            session=session,
            image_path=str(image_path),
            output_dir=output_dir,
            plan_inference=plan_inference,
            patch_size=patch_size,
            overlap=args.overlap,
        )
        summary.append((image_name, out_mask))

    if len(image_paths) > 1:
        print(f"\n{'='*60}", flush=True)
        print(f"  Summary: {len(image_paths)} images processed", flush=True)
        for name, out_mask in summary:
            print(f"    {name}: {out_mask}", flush=True)

    print(f"\nDone. Total time: {time.time() - t_total:.2f}s", flush=True)


if __name__ == "__main__":
    print("\nStandalone nnUNet ONNX segmentation pipeline…\n", flush=True)
    main()
