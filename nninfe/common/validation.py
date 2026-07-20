"""Input sanity guards.

A deliberately **light, O(1)** check run once on each image right after it is read: it inspects
only the geometry/header (dimension, channel count, spacing, size) — never the voxel data — so it
adds no measurable time and catches the inputs that would otherwise silently corrupt the whole
run (a 2-D image, an RGB image, zero/negative spacing). Deeper data-quality checks (NaN scan,
all-constant volume, intensity-vs-plan / wrong-modality) are intentionally left out to keep this
free and simple; they would each cost a full-volume pass and can be added later if a real bad-data
incident justifies it.

On failure it raises :class:`InputValidationError`, which the CLIs isolate per image (exit code 8).
"""

import math

import SimpleITK as sitk

from nninfe.common.errors import InputValidationError


def validate_input_image(image: sitk.Image) -> None:
    """Raise :class:`InputValidationError` if *image* cannot be processed by the pipeline. Only
    header fields are read, so this is constant-time regardless of image size."""
    dim = image.GetDimension()
    if dim != 3:
        raise InputValidationError(f"expected a 3D image, got {dim}D")

    channels = image.GetNumberOfComponentsPerPixel()
    if channels != 1:
        raise InputValidationError(f"expected a single-channel image, got {channels} channels")

    spacing = image.GetSpacing()
    if not all(math.isfinite(s) and s > 0 for s in spacing):
        raise InputValidationError(f"invalid voxel spacing {tuple(spacing)}")

    size = image.GetSize()
    if any(s <= 0 for s in size):
        raise InputValidationError(f"empty image with size {tuple(size)}")
