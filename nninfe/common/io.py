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

import colorsys
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
# or SR (no default -> a missing tag raises AttributeError). These are all Type 2 (required, but
# a zero-length value is legal), and real-world / anonymized series frequently drop some
# (AccessionNumber, StudyID, dates, patient demographics). Backfilling "" keeps the output
# standards-conformant instead of crashing on such inputs.
_SOURCE_TYPE2_ATTRS = (
    "PatientID",
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "AccessionNumber",
    "StudyID",
    "StudyDate",
    "StudyTime",
)


def _backfill_source_attributes(datasets: list) -> None:
    """Ensure every source dataset carries the Type-2 attributes highdicom needs (see above),
    setting any that are missing to an empty value. Mutates in place (these are our own
    freshly-read copies, never written back to disk)."""
    for ds in datasets:
        for attr in _SOURCE_TYPE2_ATTRS:
            if attr not in ds:
                setattr(ds, attr, "")


def _segment_display_rgb(index: int) -> Tuple[int, int, int]:
    """A distinct, deterministic display RGB (0-255) for the *index*-th segment (0-based).

    Golden-ratio hue spacing keeps colors well separated even for many segments (a whole-body
    model can have 80+), at fixed saturation/value so they stay vivid and legible. Without a
    per-segment recommended color, viewers render every segment in one default color.
    """
    hue = (index * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return round(r * 255), round(g * 255), round(b * 255)


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
    _backfill_source_attributes(source_datasets)
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
            display_color=hd.color.CIELabColor.from_rgb(*_segment_display_rgb(num - 1)),
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
        series_description="nnInfe segmentation SEG",
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seg.save_as(output_path)
    return output_path


def _nest_regions_into_measurements_for_cornerstone(dataset) -> None:
    """Relocate each planar ROI's SCOORD so Cornerstone3D/OHIF can *hydrate* the box.

    highdicom emits a standards-compliant TID 1410 group: the ROI SCOORD is a direct child of the
    measurement-group CONTAINER, a *sibling* of the measurement NUMs. Cornerstone3D's SR adapter,
    however, only looks for the SCOORD *inside the first NUM* (``RelationshipType`` INFERRED FROM) —
    the layout dcmjs/cornerstone writes. When the SCOORD is a sibling instead, hydration throws
    "No spatial coordinates group found" and the box is drawn read-only but never listed.

    This walks the SR content tree and, for every container holding both a top-level SCOORD and a
    NUM, moves the SCOORD(s) into the first NUM's ContentSequence as INFERRED FROM (their nested
    SELECTED FROM image reference is preserved). OHIF's read-only parser also accepts this nested
    layout, so display is unaffected. Trade-off: the output becomes cornerstone-flavored (mildly
    non-standard) rather than strict TID 1410.
    """
    from pydicom.sequence import Sequence

    seq = getattr(dataset, "ContentSequence", None)
    if not seq:
        return
    items = list(seq)
    num_items = [it for it in items if getattr(it, "ValueType", None) == "NUM"]
    scoord_items = [it for it in items if getattr(it, "ValueType", None) in ("SCOORD", "SCOORD3D")]
    if num_items and scoord_items:
        target_num = num_items[0]
        existing = list(getattr(target_num, "ContentSequence", []) or [])
        for sc in scoord_items:
            sc.RelationshipType = "INFERRED FROM"
        target_num.ContentSequence = Sequence(existing + scoord_items)
        moved_ids = {id(sc) for sc in scoord_items}
        dataset.ContentSequence = Sequence([it for it in items if id(it) not in moved_ids])
    for it in dataset.ContentSequence:
        _nest_regions_into_measurements_for_cornerstone(it)


def write_detection_dicom_sr(
    boxes_zyx: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    source_series_dir: str,
    ref_meta: dict,
    output_path: str,
    current_meta: Optional[dict] = None,
    algorithm_name: str = "nninfe",
    algorithm_version: Optional[str] = None,
    manufacturer: str = "nninfe",
    finding_category=None,
    finding_type=None,
    series_number: int = 200,
    cornerstone_compatible: bool = True,
) -> Optional[str]:
    """Write detections as a DICOM Structured Report (TID 1500 Measurement Report) in a
    Comprehensive 3D SR referencing the source series in *source_series_dir*.

    One TID 1410 planar measurement group per detection: the location is a rectangle (closed 2D
    POLYLINE, the box's X/Y extent) on the source slice nearest the centre, plus size measurements
    (Length/Width/Depth, Volume)
    and the detection Score — all with standard SCT codes. Requires DICOM *input* (the SR
    references source SOP instances). ``boxes_zyx`` is ``(N, 6)`` in nnDetection box order
    (see :mod:`nninfe.common.constants`); if ``current_meta`` is given, boxes are rescaled from
    that (resampled) voxel space to *ref_meta*'s (original) voxel space, matching the other
    exporters. ``finding_category`` / ``finding_type`` default to generic SCT codes (overridable).
    Returns the output path, or ``None`` if there are no detections.

    ``cornerstone_compatible`` picks the flavor of the emitted SR:

    - ``True`` (default): tailor the SR so OHIF/Cornerstone3D can *hydrate* each box into its
      editable measurement panel as a RectangleROI. Two viewer-specific quirks are applied — the
      Tracking Identifier is namespaced to the Cornerstone RectangleROI tool, and each ROI's SCOORD
      is nested inside its first NUM (the layout Cornerstone's SR adapter expects, see
      :func:`_nest_regions_into_measurements_for_cornerstone`). The result is mildly non-standard.
    - ``False``: emit a strict, standards-conformant TID 1410 report (neutral tracking identifiers,
      ROI SCOORD kept at the measurement-group level). Read cleanly by any conformant viewer/PACS,
      but OHIF will only show it read-only (no hydration into the measurement list).
    """
    import highdicom as hd
    import pydicom
    from highdicom.sr.templates import Measurement
    from pydicom.sr.codedict import codes

    from nninfe import __version__
    from nninfe.common.constants import D0_MAX, D0_MIN, D1_MAX, D1_MIN, D2_MAX, D2_MIN

    boxes = np.asarray(boxes_zyx, dtype=np.float32)
    if len(boxes) == 0:
        return None

    algorithm_version = __version__ if algorithm_version is None else algorithm_version
    finding_category = finding_category or codes.SCT.MorphologicallyAbnormalStructure
    finding_type = finding_type or codes.SCT.Lesion

    # Rescale resampled-space boxes back to the original voxel grid (== source series grid), so
    # the 2D point lands on the right source pixel — same scaling the JSON/CSV exporters apply.
    ref_spacing = tuple(float(s) for s in ref_meta["spacing_xyz"])
    if current_meta is not None and tuple(current_meta["spacing_xyz"]) != ref_spacing:
        cur = current_meta["spacing_xyz"]
        s0, s1, s2 = cur[2] / ref_spacing[2], cur[1] / ref_spacing[1], cur[0] / ref_spacing[0]
        boxes = boxes * np.array([s0, s1, s0, s1, s2, s2], dtype=np.float32)

    files = list_dicom_series_files(source_series_dir)
    source_datasets = [pydicom.dcmread(f) for f in files]
    _backfill_source_attributes(source_datasets)
    slice_z = np.array([float(ds.ImagePositionPatient[2]) for ds in source_datasets])

    # Geometry-only reference image for voxel -> world (handles spacing/origin/direction).
    ref_img = sitk.Image(
        int(ref_meta["size_xyz"][0]), int(ref_meta["size_xyz"][1]), int(ref_meta["size_xyz"][2]), sitk.sitkUInt8
    )
    ref_img.SetSpacing(list(ref_spacing))
    ref_img.SetOrigin([float(o) for o in ref_meta["origin"]])
    ref_img.SetDirection([float(d) for d in ref_meta["direction"]])

    observation_context = hd.sr.ObservationContext(
        observer_device_context=hd.sr.ObserverContext(
            observer_type=codes.DCM.Device,
            observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
                manufacturer_name=manufacturer, model_name=algorithm_name, uid=hd.UID()
            ),
        )
    )
    algorithm_id = hd.sr.AlgorithmIdentification(name=algorithm_name, version=algorithm_version)

    groups = []
    for i, box in enumerate(boxes):
        cx = (box[D2_MIN] + box[D2_MAX]) / 2.0
        cy = (box[D1_MIN] + box[D1_MAX]) / 2.0
        cz = (box[D0_MIN] + box[D0_MAX]) / 2.0
        world_z = ref_img.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(cz)))[2]
        size_x = float((box[D2_MAX] - box[D2_MIN]) * ref_spacing[0])
        size_y = float((box[D1_MAX] - box[D1_MIN]) * ref_spacing[1])
        size_z = float((box[D0_MAX] - box[D0_MIN]) * ref_spacing[2])

        nearest = int(np.argmin(np.abs(slice_z - world_z)))
        # Rectangle (closed POLYLINE through the 4 box corners) on the centre slice. To fall back
        # to a single centre POINT, comment this block and uncomment the POINT one below.
        region = hd.sr.ImageRegion(
            graphic_type=hd.sr.GraphicTypeValues.POLYLINE,
            graphic_data=np.array(
                [
                    [box[D2_MIN], box[D1_MIN]],
                    [box[D2_MAX], box[D1_MIN]],
                    [box[D2_MAX], box[D1_MAX]],
                    [box[D2_MIN], box[D1_MAX]],
                    [box[D2_MIN], box[D1_MIN]],
                ],
                dtype=np.float32,
            ),
            source_image=hd.sr.SourceImageForRegion.from_source_image(source_datasets[nearest]),
        )
        # region = hd.sr.ImageRegion(
        #     graphic_type=hd.sr.GraphicTypeValues.POINT,
        #     graphic_data=np.array([[cx, cy]], dtype=np.float32),
        #     source_image=hd.sr.SourceImageForRegion.from_source_image(source_datasets[nearest]),
        # )
        measurements = [
            Measurement(name=codes.SCT.Length, value=size_x, unit=codes.UCUM.Millimeter),
            Measurement(name=codes.SCT.Width, value=size_y, unit=codes.UCUM.Millimeter),
            Measurement(name=codes.SCT.Depth, value=size_z, unit=codes.UCUM.Millimeter),
            Measurement(name=codes.SCT.Volume, value=size_x * size_y * size_z, unit=codes.UCUM.CubicMillimeter),
            Measurement(name=codes.SCT.Score, value=float(scores[i]), unit=codes.UCUM.NoUnits),
        ]
        groups.append(
            hd.sr.PlanarROIMeasurementsAndQualitativeEvaluations(
                referenced_region=region,
                algorithm_id=algorithm_id,
                measurements=measurements,
                # In cornerstone-compatible mode the Tracking Identifier is namespaced to a known
                # Cornerstone tool ("<CORNERSTONE_3D_TAG>:<ToolName>[:suffix]") so OHIF hydrates a
                # closed 4/5-point POLYLINE into a RectangleROI. Otherwise a neutral identifier is
                # used (strict TID 1410). Per-detection uniqueness comes from `uid` either way.
                tracking_identifier=hd.sr.TrackingIdentifier(
                    identifier=(
                        f"Cornerstone3DTools@^0.1.0:RectangleROI:detection_{i + 1:03d}"
                        if cornerstone_compatible
                        else f"detection_{i + 1:03d}"
                    ),
                    uid=hd.UID(),
                ),
                finding_category=finding_category,
                finding_type=finding_type,
            )
        )

    report = hd.sr.MeasurementReport(
        observation_context=observation_context,
        imaging_measurements=groups,
        procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
        title=codes.DCM.ImagingMeasurementReport,
    )
    sr = hd.sr.Comprehensive3DSR(
        evidence=source_datasets,
        content=report,
        series_instance_uid=hd.UID(),
        series_number=series_number,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer=manufacturer,
        manufacturer_model_name=algorithm_name,
        software_versions=algorithm_version,
        series_description="nnInfe detection SR",
    )
    # Restructure into the SCOORD-inside-NUM layout Cornerstone3D/OHIF hydration expects
    # (see the helper's docstring). Skipped for strict TID 1410 output. Must run before save_as.
    if cornerstone_compatible:
        _nest_regions_into_measurements_for_cornerstone(sr)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sr.save_as(output_path)
    return output_path
