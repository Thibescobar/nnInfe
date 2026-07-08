"""Image I/O for the pipeline: read NIfTI files and DICOM series into a ``sitk.Image`` (plus
header/geometry and DICOM traceability metadata), and write results back out as DICOM objects.

Reads: a DICOM series is read into the *same* ``sitk.Image`` the rest of the pipeline consumes,
so the core detection/segmentation code is format-agnostic — DICOM vs NIfTI is just an input
detail handled here.

Writes: DICOM result export lives here too. A segmentation mask can be written as a DICOM
Segmentation (SEG) referencing the source series (:func:`write_segmentation_dicom_seg`); further
result objects (Structured Report, Secondary Capture, …) are meant to be added alongside.
Spatial resampling of results into the source frame still lives in ``preprocessing``.

Portable: SimpleITK + pydicom + highdicom (all cross-platform).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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


def list_dicom_series_files(series_dir: str) -> List[str]:
    """Return the sorted file paths of the single DICOM series in *series_dir* (no pixel read).

    Same single-series validation as :func:`read_dicom_series`; use this when the pixel volume
    is read separately — e.g. by pydicom when writing a DICOM SEG that references the source
    instances. Raises ``ValueError`` on a missing folder, no series, or more than one series.
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
    return list(reader.GetGDCMSeriesFileNames(str(directory), series_ids[0]))


def read_dicom_series(series_dir: str) -> Tuple[sitk.Image, List[str]]:
    """Read a single DICOM series from a directory into a ``sitk.Image``.

    Returns the volume (geometry, spacing and modality rescale handled by SimpleITK) and
    the sorted list of source DICOM file paths. Raises ``ValueError`` on a missing folder,
    no series, or more than one series (callers should point at a single-series folder).
    """
    files = list_dicom_series_files(series_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    return image, files


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


def _seg_pixel_array_from_mask(mask_ref: sitk.Image, source_datasets: list) -> Tuple[np.ndarray, list]:
    """Build a ``(frames, rows, cols)`` label array whose frame ``i`` is pixel-aligned to
    ``source_datasets[i]``.

    Each source instance is mapped to its plane in *mask_ref* by physical position
    (``ImagePositionPatient`` -> voxel index), so the correspondence is correct regardless of
    the series' slice sort direction. Instances that fall outside the mask volume are dropped —
    and dropped from the returned dataset list too, preserving the 1:1 frame↔instance
    correspondence highdicom requires (``pixel_array[i]`` <-> ``source_images[i]``).
    """
    mask_arr = sitk.GetArrayFromImage(mask_ref)  # (Z, Y, X)
    n_z = mask_arr.shape[0]
    frames: List[np.ndarray] = []
    kept: list = []
    for ds in source_datasets:
        ipp = [float(v) for v in ds.ImagePositionPatient]
        _, _, k = mask_ref.TransformPhysicalPointToIndex(ipp)
        if 0 <= k < n_z:
            frames.append(mask_arr[k])
            kept.append(ds)
    if not frames:
        raise ValueError(
            "No source DICOM instance maps into the mask volume; "
            "source series and mask geometry do not match."
        )
    return np.stack(frames, axis=0), kept


# Patient/Study attributes highdicom reads *directly* off the source image when building a SEG
# (no default -> a missing tag raises AttributeError). These are all Type 2 (required, but a
# zero-length value is legal), and real-world / anonymized series frequently drop some
# (AccessionNumber, StudyID, dates, patient demographics). Backfilling "" keeps the SEG
# standards-conformant instead of crashing on such inputs.
_SEG_SOURCE_TYPE2_ATTRS = (
    "PatientID",
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
)


def _backfill_seg_source_attributes(datasets: list) -> None:
    """Ensure every source dataset carries the Type-2 attributes highdicom needs (see above),
    setting any that are missing to an empty value. Mutates in place (these are our own
    freshly-read copies, never written back to disk)."""
    for ds in datasets:
        for attr in _SEG_SOURCE_TYPE2_ATTRS:
            if attr not in ds:
                setattr(ds, attr, "")


def write_segmentation_dicom_seg(
    mask_ref: sitk.Image,
    source_series_dir: str,
    output_path: str,
    segment_labels: Optional[Dict[int, str]] = None,
    seg_encoding: str = "binary",
    series_number: int = 100,
    manufacturer: str = "nninfe",
    device_serial_number: str = "0",
) -> Optional[str]:
    """Write *mask_ref* (an integer label map already resampled to the source geometry) as a
    DICOM Segmentation (SEG) object referencing the DICOM series in *source_series_dir*.

    Requires DICOM *input*: a SEG is spatially bound to its source instances through a shared
    Frame of Reference, so a source series must exist to reference. Each non-zero label becomes
    one segment; sparse labels are remapped to contiguous segment numbers ``1..N`` (the original
    label id is kept in the segment label unless overridden via *segment_labels*, a
    ``{original_label: name}`` mapping). The ``nninfe`` version is recorded as the algorithm and
    software version for traceability. Returns the output path, or ``None`` if the mask has no
    foreground (an empty SEG is not valid DICOM).

    ``seg_encoding`` picks the DICOM segmentation type:

    - ``binary`` (default): one binary plane per segment — widest viewer/PACS support, but the
      frame count (and file size) grows with the number of segments.
    - ``labelmap``: a single compact label map — size is independent of the class count and it
      matches an argmax (mutually-exclusive) mask, but it is a newer representation that older
      viewers may not read.
    """
    import highdicom as hd
    import pydicom
    from pydicom.sr.codedict import codes

    from nninfe import __version__

    seg_type = {
        "binary": hd.seg.SegmentationTypeValues.BINARY,
        "labelmap": hd.seg.SegmentationTypeValues.LABELMAP,
    }.get(seg_encoding)
    if seg_type is None:
        raise ValueError(f"seg_encoding must be 'binary' or 'labelmap', got: {seg_encoding!r}")

    files = list_dicom_series_files(source_series_dir)
    source_datasets = [pydicom.dcmread(f) for f in files]
    _backfill_seg_source_attributes(source_datasets)
    pixel_array, source_datasets = _seg_pixel_array_from_mask(mask_ref, source_datasets)

    present = sorted(int(v) for v in np.unique(pixel_array) if v != 0)
    if not present:
        return None

    remap = {orig: i + 1 for i, orig in enumerate(present)}
    seg_pixels = np.zeros(pixel_array.shape, dtype=np.uint8)
    for orig, num in remap.items():
        seg_pixels[pixel_array == orig] = num

    labels = segment_labels or {}
    algorithm = hd.AlgorithmIdentificationSequence(
        name="nninfe",
        family=codes.DCM.ArtificialIntelligence,
        version=__version__,
    )
    segment_descriptions = [
        hd.seg.SegmentDescription(
            segment_number=num,
            segment_label=labels.get(orig, f"Label {orig}"),
            segmented_property_category=codes.SCT.AnatomicalStructure,
            segmented_property_type=codes.SCT.Tissue,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm,
        )
        for orig, num in remap.items()
    ]

    seg = hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=seg_pixels,
        segmentation_type=seg_type,
        segment_descriptions=segment_descriptions,
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=manufacturer,
        manufacturer_model_name="nninfe",
        software_versions=__version__,
        device_serial_number=device_serial_number,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seg.save_as(output_path)
    return output_path
