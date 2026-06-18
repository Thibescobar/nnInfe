"""Image preprocessing helpers."""

from typing import List, Optional

import numpy as np
import SimpleITK as sitk


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
    """Preprocessing chain (no crop): cast -> resample -> clip -> normalize."""
    image = sitk.ReadImage(image_path)
    image = sitk.Cast(image, sitk.sitkFloat32)
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        print(f"      original  size (XYZ): {image.GetSize()}  spacing: {image.GetSpacing()}", flush=True)
        print(f"                intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  mean: {_sf.GetMean():.1f}  std: {_sf.GetSigma():.1f}", flush=True)

    image = resample_image(
        image,
        plan_inference["target_spacing"],
        plan_inference.get("transpose_forward"),
    )
    if verbose:
        _sf = sitk.StatisticsImageFilter()
        _sf.Execute(image)
        print(f"      resampled size (XYZ): {image.GetSize()}  spacing: {tuple(round(s, 4) for s in image.GetSpacing())}", flush=True)
        print(f"                intensity range: [{_sf.GetMinimum():.1f}, {_sf.GetMaximum():.1f}]  mean: {_sf.GetMean():.1f}  std: {_sf.GetSigma():.1f}", flush=True)

    intensity = plan_inference["intensity_properties"]
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
