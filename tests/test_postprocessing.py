"""Tests for post-processing: filtering, NMS, merging, Gaussian weighting."""

import numpy as np
import pytest

from nndet_onnx.nndet_onnx_inference_sw import (
    _iou_3d,
    filter_by_score,
    filter_small_boxes,
    gaussian_weight_for_boxes,
    merge_detections,
    nms_numpy,
    translate_boxes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_detection():
    """Detection dict with 4 boxes of varying scores."""
    return {
        "boxes": np.array([
            [10, 20, 30, 40, 50, 60],
            [0, 0, 5, 5, 0, 5],
            [10, 10, 50, 50, 10, 50],
            [15, 25, 35, 45, 55, 65],
        ], dtype=np.float32),
        "scores": np.array([0.9, 0.3, 0.7, 0.1], dtype=np.float32),
        "labels": np.array([0, 0, 0, 0], dtype=np.int64),
    }


@pytest.fixture()
def empty_detection():
    return {
        "boxes": np.empty((0, 6), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Filter by score
# ---------------------------------------------------------------------------


class TestFilterByScore:
    def test_keeps_above_threshold(self, sample_detection):
        result = filter_by_score(sample_detection, score_thresh=0.5)
        assert len(result["boxes"]) == 2
        assert all(result["scores"] > 0.5)

    def test_removes_all(self, sample_detection):
        result = filter_by_score(sample_detection, score_thresh=0.95)
        assert len(result["boxes"]) == 0

    def test_keeps_all(self, sample_detection):
        result = filter_by_score(sample_detection, score_thresh=0.0)
        assert len(result["boxes"]) == 4

    def test_empty_input(self, empty_detection):
        result = filter_by_score(empty_detection, score_thresh=0.5)
        assert len(result["boxes"]) == 0

    def test_preserves_labels(self, sample_detection):
        result = filter_by_score(sample_detection, score_thresh=0.5)
        assert len(result["labels"]) == len(result["boxes"])


# ---------------------------------------------------------------------------
# Filter small boxes (axis convention correctness)
# ---------------------------------------------------------------------------


class TestFilterSmallBoxes:
    def test_axis_convention_z_large_xy_small(self):
        """Box large in Z but small in Y/X → removed.

        Box format: (d0_min, d1_min, d0_max, d1_max, d2_min, d2_max)
        dim0=Z, dim1=Y, dim2=X;  spacing_xyz = (X, Y, Z)
        """
        detection = {
            "boxes": np.array([[0, 0, 10, 1, 0, 1]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = filter_small_boxes(detection, spacing_xyz=(1.0, 1.0, 1.0), min_size_mm=5.0)
        assert len(result["boxes"]) == 0

    def test_keeps_large_box(self):
        detection = {
            "boxes": np.array([[0, 0, 20, 20, 0, 20]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = filter_small_boxes(detection, spacing_xyz=(1.0, 1.0, 1.0), min_size_mm=5.0)
        assert len(result["boxes"]) == 1

    def test_spacing_scaling(self):
        """Physical size depends on spacing, not just voxel extent."""
        # 4 voxels × 2.0 mm/voxel = 8 mm → passes min_size_mm=5
        detection = {
            "boxes": np.array([[0, 0, 4, 4, 0, 4]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = filter_small_boxes(detection, spacing_xyz=(2.0, 2.0, 2.0), min_size_mm=5.0)
        assert len(result["boxes"]) == 1

    def test_anisotropic_spacing(self):
        """Anisotropic spacing: box passes in Z (thick slices) but fails in X."""
        # Z extent: 2 voxels × 5.0 mm = 10 mm ✓
        # Y extent: 2 voxels × 1.0 mm = 2 mm ✗
        # X extent: 2 voxels × 1.0 mm = 2 mm ✗
        detection = {
            "boxes": np.array([[0, 0, 2, 2, 0, 2]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        # spacing_xyz = (X=1.0, Y=1.0, Z=5.0)
        result = filter_small_boxes(detection, spacing_xyz=(1.0, 1.0, 5.0), min_size_mm=5.0)
        assert len(result["boxes"]) == 0  # Y and X too small

    def test_empty_input(self, empty_detection):
        result = filter_small_boxes(empty_detection, spacing_xyz=(1.0, 1.0, 1.0), min_size_mm=2.0)
        assert len(result["boxes"]) == 0


# ---------------------------------------------------------------------------
# IoU 3D
# ---------------------------------------------------------------------------


class TestIoU3D:
    def test_identical_boxes(self):
        box = np.array([0, 0, 10, 10, 0, 10], dtype=np.float32)
        boxes = np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32)
        iou = _iou_3d(box, boxes)
        np.testing.assert_almost_equal(iou[0], 1.0)

    def test_no_overlap(self):
        box = np.array([0, 0, 10, 10, 0, 10], dtype=np.float32)
        boxes = np.array([[20, 20, 30, 30, 20, 30]], dtype=np.float32)
        iou = _iou_3d(box, boxes)
        assert iou[0] == 0.0

    def test_partial_overlap(self):
        """Half overlap in each dim: intersection = 5×5×5 = 125."""
        box = np.array([0, 0, 10, 10, 0, 10], dtype=np.float32)
        boxes = np.array([[5, 5, 15, 15, 5, 15]], dtype=np.float32)
        iou = _iou_3d(box, boxes)
        # Each volume = 1000, intersection = 125, union = 1875
        np.testing.assert_almost_equal(iou[0], 125.0 / 1875.0, decimal=5)

    def test_contained_box(self):
        """Small box fully inside larger box."""
        big = np.array([0, 0, 20, 20, 0, 20], dtype=np.float32)
        small = np.array([[5, 5, 10, 10, 5, 10]], dtype=np.float32)
        iou = _iou_3d(big, small)
        # intersection = 5×5×5 = 125, union = 8000 + 125 - 125 = 8000
        np.testing.assert_almost_equal(iou[0], 125.0 / 8000.0, decimal=5)

    def test_multiple_boxes(self):
        box = np.array([0, 0, 10, 10, 0, 10], dtype=np.float32)
        boxes = np.array([
            [0, 0, 10, 10, 0, 10],    # identical → 1.0
            [20, 20, 30, 30, 20, 30],  # disjoint → 0.0
        ], dtype=np.float32)
        iou = _iou_3d(box, boxes)
        np.testing.assert_almost_equal(iou[0], 1.0)
        assert iou[1] == 0.0


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


class TestNmsNumpy:
    def test_suppresses_overlapping(self):
        """Two heavily overlapping boxes → keep only the higher-scoring one."""
        detection = {
            "boxes": np.array([
                [0, 0, 10, 10, 0, 10],
                [1, 1, 11, 11, 1, 11],
            ], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "labels": np.array([0, 0], dtype=np.int64),
        }
        result = nms_numpy(detection, iou_threshold=0.3)
        assert len(result["boxes"]) == 1
        assert result["scores"][0] == pytest.approx(0.9)

    def test_keeps_non_overlapping(self):
        detection = {
            "boxes": np.array([
                [0, 0, 10, 10, 0, 10],
                [50, 50, 60, 60, 50, 60],
            ], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "labels": np.array([0, 0], dtype=np.int64),
        }
        result = nms_numpy(detection, iou_threshold=0.3)
        assert len(result["boxes"]) == 2

    def test_empty_input(self, empty_detection):
        result = nms_numpy(empty_detection, iou_threshold=0.3)
        assert len(result["boxes"]) == 0

    def test_single_box(self):
        detection = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.5], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = nms_numpy(detection, iou_threshold=0.3)
        assert len(result["boxes"]) == 1


# ---------------------------------------------------------------------------
# Translate boxes
# ---------------------------------------------------------------------------


class TestTranslateBoxes:
    def test_offset_applied(self):
        """Offset (z, y, x) correctly maps to box format."""
        detection = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = translate_boxes(detection, offset_zyx=(100, 200, 300))
        expected = np.array([[100, 200, 110, 210, 300, 310]], dtype=np.float32)
        np.testing.assert_array_equal(result["boxes"], expected)

    def test_scores_unchanged(self):
        detection = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = translate_boxes(detection, offset_zyx=(5, 5, 5))
        assert result["scores"][0] == pytest.approx(0.9)

    def test_empty_input(self, empty_detection):
        result = translate_boxes(empty_detection, offset_zyx=(10, 20, 30))
        assert len(result["boxes"]) == 0


# ---------------------------------------------------------------------------
# Merge detections
# ---------------------------------------------------------------------------


class TestMergeDetections:
    def test_concatenation(self):
        d1 = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        d2 = {
            "boxes": np.array([[20, 20, 30, 30, 20, 30]], dtype=np.float32),
            "scores": np.array([0.8], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        merged = merge_detections([d1, d2])
        assert len(merged["boxes"]) == 2
        assert len(merged["scores"]) == 2

    def test_empty_list(self):
        merged = merge_detections([])
        assert merged["boxes"].shape == (0, 6)
        assert merged["scores"].shape == (0,)

    def test_preserves_extra_keys(self):
        """Extra keys (e.g. scores_original) are also merged."""
        d1 = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
            "scores_original": np.array([0.95], dtype=np.float32),
        }
        merged = merge_detections([d1])
        assert "scores_original" in merged


# ---------------------------------------------------------------------------
# Gaussian weight
# ---------------------------------------------------------------------------


class TestGaussianWeightForBoxes:
    def test_center_box_highest(self):
        """A box centered in the patch has weight ≈ 1.0."""
        patch_size = (64, 96, 96)
        center_box = np.array([[28, 44, 36, 52, 44, 52]], dtype=np.float32)
        weights = gaussian_weight_for_boxes(center_box, patch_size)
        assert weights[0] > 0.99

    def test_edge_box_lower(self):
        """Edge box has lower weight than center box."""
        patch_size = (64, 96, 96)
        center_box = np.array([[28, 44, 36, 52, 44, 52]], dtype=np.float32)
        edge_box = np.array([[0, 0, 8, 8, 0, 8]], dtype=np.float32)
        w_center = gaussian_weight_for_boxes(center_box, patch_size)
        w_edge = gaussian_weight_for_boxes(edge_box, patch_size)
        assert w_center[0] > w_edge[0]

    def test_symmetry(self):
        """Boxes equidistant from center have equal weights."""
        patch_size = (64, 64, 64)
        box_a = np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32)
        box_b = np.array([[54, 54, 64, 64, 54, 64]], dtype=np.float32)
        w_a = gaussian_weight_for_boxes(box_a, patch_size)
        w_b = gaussian_weight_for_boxes(box_b, patch_size)
        np.testing.assert_almost_equal(w_a[0], w_b[0], decimal=5)

    def test_empty_input(self):
        weights = gaussian_weight_for_boxes(np.empty((0, 6), dtype=np.float32), (64, 96, 96))
        assert len(weights) == 0

    def test_weights_in_range(self):
        """All weights are in (0, 1]."""
        patch_size = (64, 96, 96)
        boxes = np.array([
            [0, 0, 10, 10, 0, 10],
            [28, 44, 36, 52, 44, 52],
            [54, 86, 64, 96, 86, 96],
        ], dtype=np.float32)
        weights = gaussian_weight_for_boxes(boxes, patch_size)
        assert all(0 < w <= 1.0 for w in weights)
