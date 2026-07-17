"""Tests for DICOM Segmentation (SEG) output (nninfe.common.io.write_segmentation_dicom_seg)
and its wiring into the segmentation CLI.

A tiny synthetic CT series is generated with pydicom (no external data / PHI). It carries the
Patient/Study Type-2 attributes highdicom copies onto the SEG, so it stands in for a real
series. Round-trips go through highdicom to check the SEG is spatially faithful.
"""

from pathlib import Path
from unittest.mock import MagicMock

import highdicom as hd
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from nninfe import __version__
from nninfe.common.io import read_dicom_series, write_segmentation_dicom_seg
from nninfe.infer_segmentation import process_single_image

NX, NY, NZ = 16, 16, 8
DX, DY, DZ = 0.7617, 0.7617, 1.25
OX, OY, OZ = 10.0, 20.0, 30.0


def _write_synthetic_series(out_dir, omit_type2_attrs=False):
    """Write an NZ-slice axial CT series with the attributes highdicom needs to build a SEG.

    With ``omit_type2_attrs=True`` the optional Type-2 patient/study tags (AccessionNumber,
    StudyID, dates, demographics) are left out — mimicking an anonymized / partial real-world
    series, which highdicom reads directly and would otherwise crash on.
    """
    study_uid, series_uid, for_uid = generate_uid(), generate_uid(), generate_uid()
    vol = np.random.default_rng(0).integers(-1000, 2000, size=(NZ, NY, NX)).astype(np.int16)
    for k in range(NZ):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.PatientID = "TEST-0001"
        ds.PatientName = "Synthetic^CT"
        if not omit_type2_attrs:
            ds.PatientBirthDate = "19700101"
            ds.PatientSex = "O"
            ds.AccessionNumber = "ACC0001"
            ds.StudyID = "1"
            ds.StudyDate = "20260101"
            ds.StudyTime = "120000"
        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = NY, NX
        ds.PixelSpacing = [DY, DX]
        ds.SliceThickness = DZ
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [OX, OY, OZ + k * DZ]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PixelData = vol[k].tobytes()
        ds.save_as(str(out_dir / f"slice_{k:03d}.dcm"), enforce_file_format=True)


def _mask_in_source_geometry(series_dir, labels_zyx):
    """Wrap a ZYX label array as a sitk.Image aligned to the series' geometry."""
    image, _ = read_dicom_series(str(series_dir))
    mask = sitk.GetImageFromArray(labels_zyx.astype(np.uint8))
    mask.CopyInformation(image)
    return mask


class TestWriteSegmentationDicomSeg:
    def test_roundtrip_segments_and_positions(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)

        labels = np.zeros((NZ, NY, NX), dtype=np.uint8)
        labels[2:4, 4:8, 4:8] = 1  # label 1 on slices 2-3
        labels[5, 2:6, 2:6] = 2  # label 2 on slice 5
        mask = _mask_in_source_geometry(series_dir, labels)

        out = tmp_path / "seg.dcm"
        result = write_segmentation_dicom_seg(mask, str(series_dir), str(out))
        assert result == str(out)
        assert out.exists()

        seg = pydicom.dcmread(str(out))
        assert seg.Modality == "SEG"
        assert seg.SoftwareVersions == __version__
        assert len(seg.SegmentSequence) == 2
        assert [s.SegmentLabel for s in seg.SegmentSequence] == ["Label 1", "Label 2"]

        # Each segment gets a distinct recommended display color (else viewers show one color
        # for every segment).
        colors = [tuple(s.RecommendedDisplayCIELabValue) for s in seg.SegmentSequence]
        assert all(len(c) == 3 for c in colors)
        assert len(set(colors)) == len(colors)

        # Reconstruct per-segment through highdicom and check spatial fidelity.
        _, files = read_dicom_series(str(series_dir))
        src_uids = [pydicom.dcmread(f).SOPInstanceUID for f in files]
        hseg = hd.seg.segread(str(out))
        rec1 = hseg.get_pixels_by_source_instance(source_sop_instance_uids=src_uids, segment_numbers=[1])[..., 0]
        rec2 = hseg.get_pixels_by_source_instance(source_sop_instance_uids=src_uids, segment_numbers=[2])[..., 0]
        assert int(rec1.sum()) == int((labels == 1).sum())
        assert int(rec2.sum()) == int((labels == 2).sum())
        np.testing.assert_array_equal(np.where(rec1.any(axis=(1, 2)))[0], [2, 3])
        np.testing.assert_array_equal(np.where(rec2.any(axis=(1, 2)))[0], [5])

    def test_empty_mask_returns_none(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        mask = _mask_in_source_geometry(series_dir, np.zeros((NZ, NY, NX), dtype=np.uint8))

        out = tmp_path / "seg.dcm"
        assert write_segmentation_dicom_seg(mask, str(series_dir), str(out)) is None
        assert not out.exists()

    def test_sparse_labels_remapped_to_contiguous_segments(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)

        labels = np.zeros((NZ, NY, NX), dtype=np.uint8)
        labels[1, 2:6, 2:6] = 3  # sparse original ids: 3 and 7
        labels[4, 8:12, 8:12] = 7
        mask = _mask_in_source_geometry(series_dir, labels)

        out = tmp_path / "seg.dcm"
        write_segmentation_dicom_seg(mask, str(series_dir), str(out))

        seg = pydicom.dcmread(str(out))
        # Two contiguous segment numbers, labels keep the original ids.
        assert [s.SegmentNumber for s in seg.SegmentSequence] == [1, 2]
        assert [s.SegmentLabel for s in seg.SegmentSequence] == ["Label 3", "Label 7"]

    def test_segment_labels_override(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)

        labels = np.zeros((NZ, NY, NX), dtype=np.uint8)
        labels[3, 4:8, 4:8] = 1
        mask = _mask_in_source_geometry(series_dir, labels)

        out = tmp_path / "seg.dcm"
        write_segmentation_dicom_seg(mask, str(series_dir), str(out), segment_labels={1: "Liver"})
        seg = pydicom.dcmread(str(out))
        assert seg.SegmentSequence[0].SegmentLabel == "Liver"

    def test_labelmap_encoding_roundtrip(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)

        labels = np.zeros((NZ, NY, NX), dtype=np.uint8)
        labels[2:4, 4:8, 4:8] = 1
        labels[5, 2:6, 2:6] = 2
        mask = _mask_in_source_geometry(series_dir, labels)

        out = tmp_path / "seg.dcm"
        write_segmentation_dicom_seg(mask, str(series_dir), str(out), seg_encoding="labelmap")

        seg = pydicom.dcmread(str(out))
        assert seg.SegmentationType == "LABELMAP"
        # LABELMAP carries an auto-added background segment (#0); our labels are #1 and #2.
        foreground = [s for s in seg.SegmentSequence if int(s.SegmentNumber) >= 1]
        assert [s.SegmentLabel for s in foreground] == ["Label 1", "Label 2"]

        _, files = read_dicom_series(str(series_dir))
        src_uids = [pydicom.dcmread(f).SOPInstanceUID for f in files]
        hseg = hd.seg.segread(str(out))
        rec = hseg.get_pixels_by_source_instance(
            source_sop_instance_uids=src_uids, combine_segments=True, skip_overlap_checks=True
        )
        assert int((rec == 1).sum()) == int((labels == 1).sum())
        assert int((rec == 2).sum()) == int((labels == 2).sum())

    def test_invalid_encoding_raises(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        labels = np.zeros((NZ, NY, NX), dtype=np.uint8)
        labels[3, 4:8, 4:8] = 1
        mask = _mask_in_source_geometry(series_dir, labels)
        with pytest.raises(ValueError, match="seg_encoding"):
            write_segmentation_dicom_seg(mask, str(series_dir), str(tmp_path / "x.dcm"), seg_encoding="bogus")

    def test_source_missing_type2_attrs(self, tmp_path):
        """Anonymized/real series often lack Type-2 tags (e.g. AccessionNumber) that highdicom
        reads directly off the source image; they must be backfilled, not crash the SEG write."""
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir, omit_type2_attrs=True)

        labels = np.zeros((NZ, NY, NX), dtype=np.uint8)
        labels[3, 4:8, 4:8] = 1
        mask = _mask_in_source_geometry(series_dir, labels)

        out = tmp_path / "seg.dcm"
        assert write_segmentation_dicom_seg(mask, str(series_dir), str(out)) == str(out)
        seg = pydicom.dcmread(str(out))
        assert seg.Modality == "SEG"
        assert len(seg.SegmentSequence) == 1
        assert seg.AccessionNumber == ""  # backfilled to empty (valid Type-2), did not crash


def _seg_plan():
    """A plan whose target spacing equals the synthetic series (ZYX), so no resampling scaling."""
    return {
        "patch_size": [NZ, NY, NX],
        "target_spacing": [DZ, DY, DX],
        "intensity_properties": {
            "percentile_00_5": -1000.0,
            "percentile_99_5": 2000.0,
            "mean": 0.0,
            "std": 500.0,
        },
    }


def _foreground_session():
    """Mock ONNX session: one patch, 2 classes, class 1 wins inside a central box."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 1, NZ, NY, NX])]
    logits = np.zeros((1, 2, NZ, NY, NX), dtype=np.float32)
    logits[:, 1, 2:6, 4:12, 4:12] = 5.0
    mock_session.run.return_value = [logits]
    return mock_session


class TestSegmentationCliDicomOutput:
    def test_dicom_seg_output(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        output_dir = tmp_path / "out"

        out = process_single_image(
            session=_foreground_session(),
            image_path=str(series_dir),
            output_dir=output_dir,
            plan_inference=_seg_plan(),
            patch_size=(NZ, NY, NX),
            overlap=0.5,
            output_format="dicom-seg",
        )
        assert out.endswith("_seg.dcm")
        assert Path(out).exists()
        assert not (output_dir / "series_seg.nii.gz").exists()

        seg = pydicom.dcmread(out)
        assert seg.Modality == "SEG"
        assert seg.SoftwareVersions == __version__
        assert len(seg.SegmentSequence) >= 1

    def test_dicom_seg_labelmap_encoding(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        output_dir = tmp_path / "out"

        out = process_single_image(
            session=_foreground_session(),
            image_path=str(series_dir),
            output_dir=output_dir,
            plan_inference=_seg_plan(),
            patch_size=(NZ, NY, NX),
            overlap=0.5,
            output_format="dicom-seg",
            seg_encoding="labelmap",
        )
        assert pydicom.dcmread(out).SegmentationType == "LABELMAP"

    def test_both_formats(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        output_dir = tmp_path / "out"

        process_single_image(
            session=_foreground_session(),
            image_path=str(series_dir),
            output_dir=output_dir,
            plan_inference=_seg_plan(),
            patch_size=(NZ, NY, NX),
            overlap=0.5,
            output_format="both",
        )
        assert (output_dir / "series_seg.nii.gz").exists()
        assert (output_dir / "series_seg.dcm").exists()

    def test_dicom_seg_skipped_for_nifti_input(self, tmp_path):
        arr = np.random.uniform(-200, 200, (NZ, NY, NX)).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((DX, DY, DZ))
        nifti = tmp_path / "vol.nii.gz"
        sitk.WriteImage(img, str(nifti))
        output_dir = tmp_path / "out"

        out = process_single_image(
            session=_foreground_session(),
            image_path=str(nifti),
            output_dir=output_dir,
            plan_inference=_seg_plan(),
            patch_size=(NZ, NY, NX),
            overlap=0.5,
            output_format="dicom-seg",
        )
        # SEG requires DICOM input: none written, but the run still completes (per-image
        # status sentinel present, and it is what process_single_image returns).
        assert not list(output_dir.glob("*.dcm"))
        assert (output_dir / "vol.done").exists()
        assert out == str(output_dir / "vol.done")

    def test_main_dicom_seg_requires_dicom_input(self, tmp_path):
        import json
        from unittest.mock import patch

        from nninfe.infer_segmentation import main

        model_path = tmp_path / "model.onnx"
        model_path.write_text("")
        plan_path = tmp_path / "plans.json"
        plan_path.write_text(json.dumps({
            "foreground_intensity_properties_per_channel": {
                "0": {"percentile_00_5": 0.0, "percentile_99_5": 100.0, "mean": 50.0, "std": 10.0}
            },
            "configurations": {"3d_fullres": {"patch_size": [8, 8, 8], "spacing": [1.0, 1.0, 1.0]}},
        }))
        nifti = tmp_path / "vol.nii.gz"
        nifti.write_text("")

        argv = [
            "nninfe-seg", "--model-path", str(model_path), "--plan-path", str(plan_path),
            "--image-path", str(nifti), "--output-dir", str(tmp_path / "out"),
            "--output-format", "dicom-seg",
        ]
        with patch("sys.argv", argv), pytest.raises(SystemExit) as exc:
            main()
        # Usage error -> EXIT_USAGE (2), message on stderr.
        assert exc.value.code == 2
