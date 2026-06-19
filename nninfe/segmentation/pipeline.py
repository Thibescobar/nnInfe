"""Segmentation inference helpers built on shared pipeline primitives."""

import math
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from nninfe.common.sliding_window import compute_patch_positions, extract_patch


def flip_image_axes(image: sitk.Image, flip_x: bool, flip_y: bool, flip_z: bool) -> sitk.Image:
    """Flip image axes in physical space (SimpleITK order: X, Y, Z)."""
    flipper = sitk.FlipImageFilter()
    flipper.SetFlipAxes([flip_x, flip_y, flip_z])
    return flipper.Execute(image)


def extract_plan_inference(plans: dict, configuration: str = "3d_fullres") -> dict:
    """Extract a normalized inference plan from nnUNet plans.json-like structure."""
    cfg = plans.get("configurations", {}).get(configuration)
    if cfg is None:
        available = sorted(plans.get("configurations", {}).keys())
        raise ValueError(
            f"Configuration '{configuration}' not found in plans. Available: {available}"
        )

    intensity_by_channel = plans.get("foreground_intensity_properties_per_channel", {})
    channel0 = intensity_by_channel.get("0")
    if channel0 is None:
        raise ValueError(
            "Missing foreground_intensity_properties_per_channel['0'] in plans."
        )

    patch_size = cfg.get("patch_size")
    spacing = cfg.get("spacing")
    if patch_size is None or spacing is None:
        raise ValueError("Configuration must define patch_size and spacing.")

    normalized_plan = {
        "patch_size": [int(v) for v in patch_size],
        "target_spacing": [float(v) for v in spacing],
        "intensity_properties": {
            "percentile_00_5": float(channel0["percentile_00_5"]),
            "percentile_99_5": float(channel0["percentile_99_5"]),
            "mean": float(channel0["mean"]),
            "std": float(channel0["std"]),
        },
    }

    if "transpose_forward" in plans:
        normalized_plan["transpose_forward"] = [int(v) for v in plans["transpose_forward"]]
    if "transpose_backward" in plans:
        normalized_plan["transpose_backward"] = [int(v) for v in plans["transpose_backward"]]

    return normalized_plan


def pad_volume_to_patch_size(
    volume_zyx: np.ndarray,
    patch_size_zyx: Tuple[int, int, int],
    pad_value: Union[float, str] = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Pad at the end of each axis so dimensions are at least patch size."""
    original_shape = tuple(int(v) for v in volume_zyx.shape)
    pad_width = []
    for dim, patch in zip(original_shape, patch_size_zyx):
        tail = max(0, patch - dim)
        pad_width.append((0, tail))

    if any(tail > 0 for _, tail in pad_width):
        if isinstance(pad_value, str) and pad_value.lower() == "min":
            constant_value = float(np.min(volume_zyx)) - 1.0
        else:
            constant_value = float(pad_value)
        padded = np.pad(volume_zyx, pad_width=pad_width, mode="constant", constant_values=constant_value)
        return padded, original_shape

    return volume_zyx, original_shape


def crop_volume_to_shape(
    volume_zyx: np.ndarray,
    shape_zyx: Tuple[int, int, int],
) -> np.ndarray:
    """Crop back to target shape (inverse of end-only padding)."""
    z, y, x = shape_zyx
    return volume_zyx[:z, :y, :x]


def _gaussian_importance_map(
    patch_size_zyx: Tuple[int, int, int],
    edge_value: float = 0.001,
) -> np.ndarray:
    """Create a 3D gaussian importance map in ZYX order with values in [edge_value, 1]."""
    grids = [np.arange(p, dtype=np.float32) for p in patch_size_zyx]
    zz, yy, xx = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")

    centers = [(p - 1) / 2.0 for p in patch_size_zyx]
    sigmas = [max(p / 8.0, 1e-6) for p in patch_size_zyx]

    exponent = (
        -((zz - centers[0]) ** 2) / (2.0 * sigmas[0] ** 2)
        - ((yy - centers[1]) ** 2) / (2.0 * sigmas[1] ** 2)
        - ((xx - centers[2]) ** 2) / (2.0 * sigmas[2] ** 2)
    )
    gauss = np.exp(exponent).astype(np.float32)
    gauss = gauss / max(float(gauss.max()), 1e-6)
    gauss = np.maximum(gauss, edge_value)
    return gauss


def _normalize_logits_shape(raw_output: np.ndarray) -> np.ndarray:
    """Normalize model output to shape (B, C, Z, Y, X)."""
    if raw_output.ndim == 5:
        return raw_output
    if raw_output.ndim == 4:
        return raw_output[:, None, ...]
    raise ValueError(
        "Unsupported segmentation output rank "
        f"{raw_output.ndim}; expected 4D or 5D tensor."
    )


def _run_segmentation_batch(
    session,
    input_name: str,
    patches: List[np.ndarray],
) -> np.ndarray:
    """Run one segmentation batch and return logits as (B, C, Z, Y, X)."""
    input_array = np.stack([p[np.newaxis, ...] for p in patches], axis=0).astype(np.float32)
    outputs = session.run(None, {input_name: input_array})
    if not outputs:
        raise RuntimeError("Segmentation model returned no outputs.")
    return _normalize_logits_shape(np.asarray(outputs[0]))


def run_sliding_window_segmentation(
    session,
    volume_zyx: np.ndarray,
    patch_size_zyx: Tuple[int, int, int],
    batch_size: int,
    overlap: float,
    verbose: bool = False,
    progress_every: int = 3,
) -> np.ndarray:
    """Run segmentation over volume and return label map in preprocessed space (ZYX)."""
    image_shape = tuple(int(v) for v in volume_zyx.shape)
    positions, step_sizes = compute_patch_positions(image_shape, patch_size_zyx, overlap)
    if not positions:
        raise RuntimeError("No sliding-window positions were generated.")

    n_patches = len(positions)
    n_batches = math.ceil(n_patches / batch_size)

    if verbose:
        actual_overlap = tuple(
            round(1.0 - s / p, 4) if p > 0 else 0.0
            for s, p in zip(step_sizes, patch_size_zyx)
        )
        print(
            f"      step sizes (ZYX): ({step_sizes[0]:.1f}, {step_sizes[1]:.1f}, {step_sizes[2]:.1f})  "
            f"actual overlap: {actual_overlap}  (requested: {overlap})",
            flush=True,
        )
        print(
            f"      {n_patches} patches, {n_batches} batches (batch_size={batch_size})",
            flush=True,
        )

    input_name = session.get_inputs()[0].name
    gaussian = _gaussian_importance_map(patch_size_zyx)

    logits_acc = None
    weight_acc = np.zeros(image_shape, dtype=np.float32)

    t0 = time.time()
    progress_every = max(int(progress_every), 1)

    for batch_idx, start in enumerate(range(0, len(positions), batch_size)):
        batch_positions = positions[start : start + batch_size]
        patches = [extract_patch(volume_zyx, pos, patch_size_zyx) for pos in batch_positions]
        logits_batch = _run_segmentation_batch(session, input_name, patches)

        if logits_acc is None:
            num_classes = int(logits_batch.shape[1])
            logits_acc = np.zeros((num_classes,) + image_shape, dtype=np.float32)

        for i, (z, y, x) in enumerate(batch_positions):
            logits_acc[:, z : z + patch_size_zyx[0], y : y + patch_size_zyx[1], x : x + patch_size_zyx[2]] += (
                logits_batch[i] * gaussian
            )
            weight_acc[z : z + patch_size_zyx[0], y : y + patch_size_zyx[1], x : x + patch_size_zyx[2]] += gaussian

        if verbose and (batch_idx % progress_every == 0 or batch_idx == n_batches - 1):
            progress = (batch_idx + 1) / n_batches
            elapsed = time.time() - t0
            time_per_iter = elapsed / (batch_idx + 1)
            remaining = time_per_iter * (n_batches - batch_idx - 1)
            end_patch = min(start + batch_size, n_patches)
            print(
                f"            progress {progress:6.1%}  "
                f"batch {batch_idx + 1}/{n_batches}  "
                f"patch {end_patch}/{n_patches}  "
                f"{time_per_iter:.2f}s/batch  "
                f"elapsed {elapsed:.0f}s  remaining {remaining:.0f}s",
                flush=True,
            )

    if logits_acc is None:
        raise RuntimeError("Segmentation reconstruction failed: no logits accumulated.")

    logits_acc = logits_acc / np.maximum(weight_acc[None, ...], 1e-6)
    label_map = np.argmax(logits_acc, axis=0).astype(np.uint16)

    if verbose:
        print(f"      inference done ({time.time() - t0:.2f}s)", flush=True)

    return label_map


def export_segmentation_mask(
    labels_zyx: np.ndarray,
    preprocessed_image: sitk.Image,
    reference_image_path: str,
    output_path: str,
    resample_mask_to_reference,
) -> None:
    """Export segmentation mask as NIfTI in original image geometry."""
    mask = sitk.GetImageFromArray(labels_zyx)
    mask.CopyInformation(preprocessed_image)
    mask_ref = resample_mask_to_reference(mask, reference_image_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask_ref, output_path)
