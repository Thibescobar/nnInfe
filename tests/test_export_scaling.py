"""Tests for coordinate scaling in export functions and CSV content."""

import csv
import json
import pickle

import numpy as np
import pytest

from nninfe.detection.export import (
    export_detections_csv,
    export_detections_json,
    export_detections_pkl,
)


@pytest.fixture()
def detection():
    return {
        "boxes": np.array([[10, 20, 30, 40, 50, 60]], dtype=np.float32),
        "scores": np.array([0.95], dtype=np.float32),
        "labels": np.array([0], dtype=np.int64),
    }


@pytest.fixture()
def ref_meta():
    return {
        "size_xyz": (100, 200, 50),
        "spacing_xyz": (1.0, 1.0, 2.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }


@pytest.fixture()
def current_meta():
    """Resampled space with different spacing."""
    return {
        "size_xyz": (200, 200, 25),
        "spacing_xyz": (0.5, 1.0, 4.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }


class TestExportJsonScaling:
    def test_no_scaling_without_current_meta(self, detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.json")
        export_detections_json(detection, out, ref_meta=ref_meta, current_meta=None)
        with open(out) as f:
            data = json.load(f)
        np.testing.assert_array_almost_equal(data["pred_boxes"][0], [10, 20, 30, 40, 50, 60])

    def test_scaling_with_current_meta(self, detection, ref_meta, current_meta, tmp_path):
        out = str(tmp_path / "boxes.json")
        export_detections_json(detection, out, ref_meta=ref_meta, current_meta=current_meta)
        with open(out) as f:
            data = json.load(f)
        # Boxes should be scaled: different spacing ratio per axis
        boxes = np.array(data["pred_boxes"][0])
        # Original boxes should NOT be equal (scaling happened)
        assert not np.allclose(boxes, [10, 20, 30, 40, 50, 60])

    def test_no_scaling_when_spacings_match(self, detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.json")
        same_meta = dict(ref_meta)
        export_detections_json(detection, out, ref_meta=ref_meta, current_meta=same_meta)
        with open(out) as f:
            data = json.load(f)
        np.testing.assert_array_almost_equal(data["pred_boxes"][0], [10, 20, 30, 40, 50, 60])


class TestExportPklScaling:
    def test_scaling_with_current_meta(self, detection, ref_meta, current_meta, tmp_path):
        out = str(tmp_path / "boxes.pkl")
        export_detections_pkl(detection, out, ref_meta=ref_meta, current_meta=current_meta)
        with open(out, "rb") as f:
            data = pickle.load(f)
        boxes = data["pred_boxes"]
        assert not np.allclose(boxes, [[10, 20, 30, 40, 50, 60]])
        assert data["pred_boxes"].dtype == np.float32

    def test_original_size_is_reversed(self, detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.pkl")
        export_detections_pkl(detection, out, ref_meta=ref_meta)
        with open(out, "rb") as f:
            data = pickle.load(f)
        # original_size_of_raw_data should be reversed from XYZ to ZYX
        np.testing.assert_array_equal(data["original_size_of_raw_data"], [50, 200, 100])


class TestExportCsvContent:
    def test_csv_world_coordinates(self, detection, ref_meta, tmp_path):
        out = str(tmp_path / "boxes.csv")
        export_detections_csv(detection, out, image_name="img1", ref_meta=ref_meta)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["image_name"] == "img1"
        assert rows[0]["detection_id"] == "1"
        assert float(rows[0]["score"]) == pytest.approx(0.95, abs=0.01)

    def test_csv_with_scaling(self, detection, ref_meta, current_meta, tmp_path):
        out = str(tmp_path / "boxes.csv")
        export_detections_csv(detection, out, image_name="test", ref_meta=ref_meta, current_meta=current_meta)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        # Volume should be present and positive
        assert float(rows[0]["volume_mm3"]) > 0

    def test_csv_physical_sizes(self, ref_meta, tmp_path):
        det = {
            "boxes": np.array([[0, 0, 10, 20, 0, 30]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        out = str(tmp_path / "boxes.csv")
        export_detections_csv(det, out, image_name="test", ref_meta=ref_meta)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # spacing_xyz = (1.0, 1.0, 2.0), box dim2 extent = 30, dim0 = 10, dim1 = 20
        # size_z = 10 * 2.0 = 20.0 (dim0=Z, spacing_xyz[2]=Z)
        assert float(rows[0]["size_z_mm"]) == pytest.approx(20.0)
        # size_y = 20 * 1.0 = 20.0
        assert float(rows[0]["size_y_mm"]) == pytest.approx(20.0)
        # size_x = 30 * 1.0 = 30.0
        assert float(rows[0]["size_x_mm"]) == pytest.approx(30.0)
