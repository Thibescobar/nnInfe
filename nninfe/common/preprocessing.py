"""Image preprocessing helpers."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from nninfe.common.io import read_dicom_series, read_image, read_image_metadata
from nninfe.common.validation import validate_input_image

logger = logging.getLogger(__name__)


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


def resample_image(
    image: sitk.Image,
    target_spacing_zyx: List[float],
    transpose_forward_zyx: Optional[List[int]] = None,
) -> sitk.Image:
    """Resample *image* to *target_spacing_zyx* (Z-Y-X), accounting for transpose if provided."""
    if transpose_forward_zyx is None:
        new_spacing = target_spacing_zyx[::-1]  # convert to X-Y-Z
    else:
        # Match Aurore mapping logic: map[transpose_idx] = spacing
        my_map = {
            transpose_forward_zyx[0]: target_spacing_zyx[0],
            transpose_forward_zyx[1]: target_spacing_zyx[1],
            transpose_forward_zyx[2]: target_spacing_zyx[2],
        }
        # In aurore: vector1 = { myMap[1], myMap[2], myMap[0] }
        new_spacing = [my_map[1], my_map[2], my_map[0]]

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
    # Use a low out-of-image value consistent with padding strategy.
    min_value = float(sitk.GetArrayViewFromImage(image).min())
    resampler.SetDefaultPixelValue(min_value - 1.0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(image)


def resample_mask_to_reference(mask: sitk.Image, reference) -> sitk.Image:
    """Resample a label mask back to the geometry of a reference image (NIfTI or DICOM).

    Inverse of :func:`resample_image`: maps a model-space result into the original input's
    voxel frame so it overlays the source. Nearest-neighbor (label-preserving).

    ``reference`` is either a path (read via :func:`nninfe.common.io.read_image_metadata`)
    or an already-read geometry dict (same keys). Passing the dict avoids re-reading the
    input — notably a multi-slice DICOM series, which has no header-only geometry read.
    """
    meta = reference if isinstance(reference, dict) else read_image_metadata(reference)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(meta["spacing_xyz"])
    resampler.SetSize(meta["size_xyz"])
    resampler.SetOutputDirection(meta["direction"])
    resampler.SetOutputOrigin(meta["origin"])
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(mask)


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
) -> Tuple[sitk.Image, dict]:
    """Preprocessing chain (no crop): cast -> resample -> clip -> normalize.

    ``image_path`` may be a NIfTI file or a DICOM series directory (see ``read_image``).
    Returns ``(preprocessed_image, original_metadata)`` — the original geometry
    (size/spacing/origin/direction) is captured from the single input read so results can
    be mapped back to the input frame without re-reading it (key for DICOM series, which
    have no cheap header-only geometry read).
    """
    # For a DICOM series, keep the sorted source file list from this single read so the DICOM
    # result writers can reference the instances without a second (costly) directory scan.
    source_files = None
    if Path(image_path).is_dir():
        image, source_files = read_dicom_series(image_path)
    else:
        image = read_image(image_path)
    validate_input_image(image)  # cheap O(1) geometry guard before any processing
    original_metadata = {
        "size_xyz": image.GetSize(),
        "spacing_xyz": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "source_files": source_files,
    }
    image = sitk.Cast(image, sitk.sitkFloat32)
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        logger.info(f"      original  size (XYZ): {image.GetSize()}  spacing: {image.GetSpacing()}")
        logger.info(f"                intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  mean: {_sf.GetMean():.1f}  std: {_sf.GetSigma():.1f}")

    image = resample_image(
        image,
        plan_inference["target_spacing"],
        plan_inference.get("transpose_forward"),
    )
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        logger.info(f"      resampled size (XYZ): {image.GetSize()}  spacing: {tuple(round(s, 4) for s in image.GetSpacing())}")
        logger.info(f"                intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  mean: {_sf.GetMean():.1f}  std: {_sf.GetSigma():.1f}")

    intensity = plan_inference["intensity_properties"]
    image = clip_image(
        image,
        lower=intensity["percentile_00_5"],
        upper=intensity["percentile_99_5"],
    )
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        logger.info(f"      clipped   intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  (percentiles [{intensity['percentile_00_5']:.1f}, {intensity['percentile_99_5']:.1f}])")

    image = normalize_image(image, mean=intensity["mean"], std=intensity["std"])
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        logger.info(f"      normalized intensity range: [{_sf.GetMinimum():.2f}, {_sf.GetMaximum():.2f}]  mean: {_sf.GetMean():.2f}  std: {_sf.GetSigma():.2f}")

    return image, original_metadata
