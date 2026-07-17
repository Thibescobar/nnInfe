"""Tests for detection DICOM SR output (nninfe.common.io.write_detection_dicom_sr) and its
wiring into the detection CLI.

Uses a tiny synthetic CT series (pydicom, no PHI) carrying the Patient/Study attributes highdicom
copies onto the SR. Round-trips go through highdicom / pydicom to check the report is valid and
holds one measurement group per detection.
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

from nninfe.common.io import read_image_metadata, write_detection_dicom_sr
from nninfe.detection.anchors import compute_anchors
from nninfe.infer_detection import process_single_image

COMPREHENSIVE_3D_SR = "1.2.840.10008.5.1.4.1.1.88.34"
SCORE_CODE = "246262008"  # SCT "Score (attribute)"

NX, NY, NZ = 16, 16, 8
DX, DY, DZ = 0.7617, 0.7617, 1.25
OX, OY, OZ = 10.0, 20.0, 30.0


def _write_synthetic_series(out_dir, omit_type2_attrs=False):
    study_uid, series_uid, for_uid = generate_uid(), generate_uid(), generate_uid()
    for k in range(NZ):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.PatientID, ds.PatientName = "TEST-0001", "Synthetic^CT"
        if not omit_type2_attrs:
            ds.PatientBirthDate, ds.PatientSex = "19700101", "O"
            ds.AccessionNumber, ds.StudyID = "ACC0001", "1"
            ds.StudyDate, ds.StudyTime = "20260101", "120000"
        ds.Modality = "CT"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, for_uid
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
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PixelData = np.zeros((NY, NX), np.int16).tobytes()
        ds.save_as(str(out_dir / f"slice_{k:03d}.dcm"), enforce_file_format=True)


def _count_measurements(ds, code_value):
    """Recursively count NUM content items whose concept code == code_value."""
    n = 0
    for item in ds.get("ContentSequence", []):
        if item.get("ValueType") == "NUM":
            cn = item.get("ConceptNameCodeSequence")
            if cn and cn[0].CodeValue == code_value:
                n += 1
        n += _count_measurements(item, code_value)
    return n


def _collect_scoords(ds, out=None):
    """Recursively collect SCOORD content items (the per-detection image regions)."""
    if out is None:
        out = []
    for item in ds.get("ContentSequence", []):
        if item.get("ValueType") == "SCOORD":
            out.append(item)
        _collect_scoords(item, out)
    return out


class TestWriteDetectionDicomSr:
    def test_roundtrip(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        ref_meta = read_image_metadata(str(series_dir))

        # boxes in original voxel space: (z0, y0, z1, y1, x0, x1)
        boxes = np.array([[2, 4, 4, 8, 4, 8], [5, 2, 6, 6, 2, 6]], np.float32)
        scores = np.array([0.91, 0.77], np.float32)
        labels = np.array([0, 0], np.int64)

        out = tmp_path / "sr.dcm"
        result = write_detection_dicom_sr(boxes, scores, labels, str(series_dir), ref_meta, str(out))
        assert result == str(out)

        sr = pydicom.dcmread(str(out))
        assert sr.SOPClassUID == COMPREHENSIVE_3D_SR
        assert sr.Modality == "SR"
        # One Score measurement per detection => two measurement groups.
        assert _count_measurements(sr, SCORE_CODE) == 2
        # Each detection is drawn as a rectangle: closed POLYLINE through the 4 box corners.
        scoords = _collect_scoords(sr)
        assert len(scoords) == 2
        for item, box in zip(scoords, boxes):
            assert item.GraphicType == "POLYLINE"
            pts = np.asarray(item.GraphicData, np.float32).reshape(-1, 2)
            assert len(pts) == 5
            assert np.allclose(pts[0], pts[-1])  # closed
            x0, y0, x1, y1 = box[4], box[1], box[5], box[3]  # (z0, y0, z1, y1, x0, x1)
            assert np.allclose(pts.min(axis=0), [x0, y0]) and np.allclose(pts.max(axis=0), [x1, y1])
        # highdicom re-reads it (validates on parse).
        assert type(hd.sr.srread(str(out))).__name__ == "Comprehensive3DSR"

    def test_empty_returns_none(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        ref_meta = read_image_metadata(str(series_dir))
        out = tmp_path / "sr.dcm"
        result = write_detection_dicom_sr(
            np.empty((0, 6), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int64),
            str(series_dir), ref_meta, str(out),
        )
        assert result is None
        assert not out.exists()

    def test_source_missing_type2_attrs(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir, omit_type2_attrs=True)
        ref_meta = read_image_metadata(str(series_dir))
        boxes = np.array([[3, 4, 5, 8, 4, 8]], np.float32)
        out = tmp_path / "sr.dcm"
        result = write_detection_dicom_sr(
            boxes, np.array([0.8], np.float32), np.array([0], np.int64),
            str(series_dir), ref_meta, str(out),
        )
        assert result == str(out)
        assert pydicom.dcmread(str(out)).AccessionNumber == ""


def _det_plan():
    return {
        "patch_size": [8, 8, 8],
        "target_spacing": [DZ, DY, DX],  # == series spacing (ZYX) -> no resample scaling
        "intensity_properties": {
            "percentile_00_5": -1000.0, "percentile_99_5": 2000.0, "mean": 0.0, "std": 500.0,
        },
        "anchors": {"width": [4, 8], "height": [4, 8], "depth": [4, 8]},
        "architecture": {"strides": [[1, 1, 1], [2, 2, 2]], "decoder_levels": [1, 2]},
        "inference_plan": {"model_iou": 0.1, "model_score_thresh": 0.5, "model_topk": 100,
                           "model_detections_per_image": 100, "remove_small_boxes": 2.0},
    }


def _det_session(batch_size=2):
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(shape=[batch_size, 1, 8, 8, 8])]

    def fake_run(_, inputs):
        return [
            np.array([[1, 1, 6, 6, 1, 6]], np.float32),
            np.array([[2, 2, 7, 7, 2, 7]], np.float32),
            np.array([0.9], np.float32),
            np.array([0.85], np.float32),
            np.array([0], np.int64),
            np.array([0], np.int64),
        ]

    session.run.side_effect = fake_run
    return session


def _run_det(image_path, output_dir, output_format):
    plan = _det_plan()
    return process_single_image(
        session=_det_session(),
        image_path=str(image_path),
        output_dir=Path(output_dir),
        plan_inference=plan,
        patch_size=(8, 8, 8),
        batch_size=2,
        anchors_batch=compute_anchors(plan, (8, 8, 8), 2),
        iou_threshold=0.1,
        overlap=0.5,
        score_thresh=0.3,
        min_size_mm=0.5,
        nms_backend="numpy",
        no_global_nms=False,
        export_pkl=False,
        pad_value="0.0",
        output_format=output_format,
    )


class TestDetectionCliSrOutput:
    def test_dicom_sr_output(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        out_dir = tmp_path / "out"

        n = _run_det(series_dir, out_dir, "dicom-sr")
        assert isinstance(n, int) and n > 0
        assert (out_dir / "series_sr.dcm").exists()
        # JSON is always written; mask/CSV stay gated to the nifti format.
        assert (out_dir / "series_boxes.json").exists()
        assert not (out_dir / "series_mask.nii.gz").exists()
        assert not (out_dir / "series_boxes.csv").exists()
        sr = pydicom.dcmread(str(out_dir / "series_sr.dcm"))
        assert sr.SOPClassUID == COMPREHENSIVE_3D_SR and sr.Modality == "SR"

    def test_json_only_format(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        out_dir = tmp_path / "out"

        _run_det(series_dir, out_dir, "json")
        # json-only: the record, nothing else.
        assert (out_dir / "series_boxes.json").exists()
        assert not (out_dir / "series_boxes.csv").exists()
        assert not (out_dir / "series_mask.nii.gz").exists()
        assert not list(out_dir.glob("*.dcm"))

    def test_both_formats(self, tmp_path):
        series_dir = tmp_path / "series"
        series_dir.mkdir()
        _write_synthetic_series(series_dir)
        out_dir = tmp_path / "out"

        _run_det(series_dir, out_dir, "both")
        assert (out_dir / "series_sr.dcm").exists()
        assert (out_dir / "series_mask.nii.gz").exists()
        assert (out_dir / "series_boxes.json").exists()
        assert (out_dir / "series_boxes.csv").exists()

    def test_sr_skipped_for_nifti_input(self, tmp_path):
        arr = np.random.uniform(-200, 200, (NZ, NY, NX)).astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((DX, DY, DZ))
        nifti = tmp_path / "vol.nii.gz"
        sitk.WriteImage(img, str(nifti))
        out_dir = tmp_path / "out"

        _run_det(nifti, out_dir, "dicom-sr")
        assert not list(out_dir.glob("*.dcm"))
        assert (out_dir / "vol.done").exists()

    def test_main_dicom_sr_requires_dicom_input(self, tmp_path):
        import json
        from unittest.mock import patch

        from nninfe.infer_detection import main

        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(_det_plan()))
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"fake")
        nifti = tmp_path / "img.nii.gz"
        nifti.write_bytes(b"")

        argv = [
            "nninfe-det", "--model-path", str(model_path), "--plan-path", str(plan_path),
            "--image-path", str(nifti), "--output-dir", str(tmp_path / "out"),
            "--output-format", "dicom-sr",
        ]
        with patch("sys.argv", argv), pytest.raises(SystemExit) as exc:
            main()
        # Usage error -> EXIT_USAGE (2), message on stderr.
        assert exc.value.code == 2
