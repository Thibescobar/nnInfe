"""Image I/O: read a NIfTI file or a DICOM series into a ``sitk.Image``, plus header/geometry
and DICOM traceability metadata reads.

A DICOM series is read into the *same* ``sitk.Image`` the rest of the pipeline consumes, so
the core detection/segmentation code is format-agnostic — DICOM vs NIfTI is just an input
detail handled here. This module only *reads* (spatial resampling of results lives in
``preprocessing``). Portable: SimpleITK + pydicom (both cross-platform).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import SimpleITK as sitk

# DICOM tag -> friendly key, captured for traceability and for referencing the source
# study/series when writing DICOM results.
_SERIES_TAGS = {
    "0010|0020": "patient_id",
    "0020|000d": "study_instance_uid",
    "0020|000e": "series_instance_uid",
    "0020|0052": "frame_of_reference_uid",
    "0008|0060": "modality",
}


def read_dicom_series(series_dir: str) -> Tuple[sitk.Image, List[str]]:
    """Read a single DICOM series from a directory into a ``sitk.Image``.

    Returns the volume (geometry, spacing and modality rescale handled by SimpleITK) and
    the sorted list of source DICOM file paths. Raises ``ValueError`` on a missing folder,
    no series, or more than one series (callers should point at a single-series folder).
    """
    directory = Path(series_dir)
    if not directory.is_dir():
        raise ValueError(f"DICOM series directory not found: {series_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(directory))
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {series_dir}")
    if len(series_ids) > 1:
        raise ValueError(
            f"Multiple DICOM series ({len(series_ids)}) found in {series_dir}; "
            "point the input at a folder containing a single series."
        )

    files = reader.GetGDCMSeriesFileNames(str(directory), series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    return image, list(files)


def read_series_metadata(dicom_file: str) -> Dict[str, str]:
    """Read the traceability/reference metadata (UIDs, patient id, modality) from one slice
    of a series. Missing tags are omitted rather than raising.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    meta: Dict[str, str] = {}
    for tag, key in _SERIES_TAGS.items():
        if reader.HasMetaDataKey(tag):
            meta[key] = reader.GetMetaData(tag).strip()
    return meta


def looks_like_dicom(path: str) -> bool:
    """Heuristic input routing: a directory (single-series folder) or a ``.dcm`` file is
    treated as DICOM; ``.nii``/``.nii.gz`` stays on the NIfTI path.
    """
    p = Path(path)
    if p.is_dir():
        return True
    return p.suffix.lower() == ".dcm"


def read_image(path: str) -> sitk.Image:
    """Read one image into a ``sitk.Image``, dispatching on the input kind: a directory is
    read as a DICOM series, anything else (``.nii``/``.nii.gz``, or a single ``.dcm``) via
    SimpleITK's file reader. Single entry point so DICOM and NIfTI inputs are interchangeable.
    """
    p = Path(path)
    if p.is_dir():
        image, _ = read_dicom_series(str(p))
        return image
    return sitk.ReadImage(str(p))


def read_image_metadata(image_path: str) -> dict:
    """Return size/spacing/origin/direction (XYZ) for a NIfTI file or a DICOM series.

    NIfTI is read header-only (cheap); a DICOM series directory is read via ``read_image``
    (SimpleITK has no header-only multi-slice geometry read).
    """
    if Path(image_path).is_dir():
        img = read_image(image_path)
        return {
            "size_xyz": img.GetSize(),
            "spacing_xyz": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
        }
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.ReadImageInformation()
    return {
        "size_xyz": reader.GetSize(),
        "spacing_xyz": reader.GetSpacing(),
        "origin": reader.GetOrigin(),
        "direction": reader.GetDirection(),
    }
