"""Tests for export functions (mask, JSON, CSV, PKL, metadata)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from nninfe.common.io import read_image_metadata
from nninfe.common.preprocessing import resample_mask_to_reference
from nninfe.detection.export import (
    detections_to_mask,
    export_detections_csv,
    export_detections_json,
    export_detections_pkl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def reference_image():
    """A small synthetic SimpleITK image (Z=10, Y=20, X=30)."""
    arr = np.zeros((10, 20, 30), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


@pytest.fixture()
def sample_detection():
    return {
        "boxes": np.array([[2, 5, 8, 15, 10, 25]], dtype=np.float32),
        "scores": np.array([0.85], dtype=np.float32),
        "labels": np.array([0], dtype=np.int64),
    }


@pytest.fixture()
def empty_detection():
    return {
        "boxes": np.empty((0, 6), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
    }


@pytest.fixture()
def ref_meta():
    return {
        "size_xyz": (30, 20, 10),
        "spacing_xyz": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }


@pytest.fixture()
def nifti_path(tmp_path):
    """Write a synthetic NIfTI to a temp file and return the path."""
    arr = np.zeros((10, 20, 30), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.75, 1.25))
    img.SetOrigin((10.0, 20.0, 30.0))
    path = str(tmp_path / "test.nii.gz")
    sitk.WriteImage(img, path)
    return path


# ---------------------------------------------------------------------------
# Mask creation
# ---------------------------------------------------------------------------


class TestDetectionsToMask:
    def test_single_box(self, sample_detection, reference_image):
        mask = detections_to_mask(sample_detection, (10, 20, 30), reference_image)
        arr = sitk.GetArrayFromImage(mask)
        # Region inside box should be labeled (> 0)
        assert arr[5, 10, 17] > 0
        # Region outside box should be 0
        assert arr[0, 0, 0] == 0

    def test_empty_detections(self, empty_detection, reference_image):
        mask = detections_to_mask(empty_detection, (10, 20, 30), reference_image)
        arr = sitk.GetArrayFromImage(mask)
        assert arr.sum() == 0

    def test_mask_inherits_metadata(self, sample_detection, reference_image):
        mask = detections_to_mask(sample_detection, (10, 20, 30), reference_image)
        assert mask.GetSpacing() == reference_image.GetSpacing()
        assert mask.GetOrigin() == reference_image.GetOrigin()

    def test_two_boxes_different_labels(self, reference_image):
        """Two non-overlapping boxes produce two connected components."""
        detection = {
            "boxes": np.array([
                [0, 0, 3, 3, 0, 3],
                [7, 15, 10, 20, 25, 30],
            ], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "labels": np.array([0, 0], dtype=np.int64),
        }
        mask = detections_to_mask(detection, (10, 20, 30), reference_image)
        arr = sitk.GetArrayFromImage(mask)
        unique = set(np.unique(arr)) - {0}
        assert len(unique) == 2  # two connected components


# ---------------------------------------------------------------------------
# Resample mask to reference
# ---------------------------------------------------------------------------


class TestResampleMaskToReference:
    def test_resamples_to_original_geometry(self, nifti_path):
        """Mask is resampled to match the reference image geometry."""
        # Create a small mask in a different geometry
        mask_arr = np.ones((5, 10, 15), dtype=np.uint8)
        mask = sitk.GetImageFromArray(mask_arr)
        mask.SetSpacing((1.0, 1.5, 2.5))
        mask.SetOrigin((10.0, 20.0, 30.0))

        resampled = resample_mask_to_reference(mask, nifti_path)
        assert resampled.GetSpacing() == pytest.approx((0.5, 0.75, 1.25))
        assert resampled.GetSize() == (30, 20, 10)


# ---------------------------------------------------------------------------
# Read image metadata
# ---------------------------------------------------------------------------


class TestReadImageMetadata:
    def test_reads_spacing_and_size(self, nifti_path):
        meta = read_image_metadata(nifti_path)
        assert meta["size_xyz"] == (30, 20, 10)
        assert meta["spacing_xyz"] == pytest.approx((0.5, 0.75, 1.25))

    def test_reads_origin(self, nifti_path):
        meta = read_image_metadata(nifti_path)
        assert meta["origin"] == pytest.approx((10.0, 20.0, 30.0))


# ---------------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_writes_valid_json(self, sample_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.json")
        export_detections_json(sample_detection, out, ref_meta=ref_meta)
        with open(out) as f:
            data = json.load(f)
        assert "pred_boxes" in data
        assert "pred_scores" in data
        assert "pred_labels" in data
        assert len(data["pred_boxes"]) == 1

    def test_empty_detections(self, empty_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.json")
        export_detections_json(empty_detection, out, ref_meta=ref_meta)
        with open(out) as f:
            data = json.load(f)
        assert len(data["pred_boxes"]) == 0

    def test_contains_metadata(self, sample_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.json")
        export_detections_json(sample_detection, out, ref_meta=ref_meta)
        with open(out) as f:
            data = json.load(f)
        assert "itk_spacing" in data
        assert "itk_origin" in data
        assert "original_size_of_raw_data" in data


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------


class TestExportCsv:
    def test_writes_header_and_rows(self, sample_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.csv")
        export_detections_csv(sample_detection, out, image_name="test", ref_meta=ref_meta)
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 detection
        assert lines[0].startswith("image_name,detection_id")

    def test_empty_detections(self, empty_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.csv")
        export_detections_csv(empty_detection, out, image_name="test", ref_meta=ref_meta)
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 1  # header only


# ---------------------------------------------------------------------------
# Export PKL
# ---------------------------------------------------------------------------


class TestExportPkl:
    def test_writes_valid_pickle(self, sample_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.pkl")
        export_detections_pkl(sample_detection, out, ref_meta=ref_meta)
        with open(out, "rb") as f:
            data = pickle.load(f)
        assert "pred_boxes" in data
        assert isinstance(data["pred_boxes"], np.ndarray)
        assert data["pred_boxes"].dtype == np.float32

    def test_labels_dtype(self, sample_detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.pkl")
        export_detections_pkl(sample_detection, out, ref_meta=ref_meta)
        with open(out, "rb") as f:
            data = pickle.load(f)
        assert data["pred_labels"].dtype == np.int64
