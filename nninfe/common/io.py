"""Image metadata and mask geometry helpers."""

import SimpleITK as sitk


def resample_mask_to_reference(mask: sitk.Image, reference_path: str) -> sitk.Image:
    """Resample a label mask to match the geometry of a reference image."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(reference_path)
    reader.ReadImageInformation()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reader.GetSpacing())
    resampler.SetSize(reader.GetSize())
    resampler.SetOutputDirection(reader.GetDirection())
    resampler.SetOutputOrigin(reader.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(mask)


def read_image_metadata(image_path: str) -> dict:
    """Read image metadata without loading pixel data (header only)."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    reader.ReadImageInformation()
    return {
        "size_xyz": reader.GetSize(),
        "spacing_xyz": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection(),
    }
