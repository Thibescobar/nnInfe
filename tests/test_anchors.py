"""Tests for anchor generation functions."""

import numpy as np
import pytest

from nninfe.detection.anchors import (
    _generate_cell_anchors,
    compute_anchors,
    compute_feature_map_sizes,
    generate_anchors,
)


class TestGenerateCellAnchors:
    def test_single_size_shape(self):
        """Single anchor size produces (1, 6) array."""
        anchors = _generate_cell_anchors(width=(10,), height=(10,), depth=(10,))
        assert anchors.shape == (1, 6)

    def test_single_size_symmetric(self):
        """A cube anchor is symmetric around the origin."""
        anchors = _generate_cell_anchors(width=(10,), height=(10,), depth=(10,))
        a = anchors[0]
        assert a[0] == -a[2]  # d0_min == -d0_max
        assert a[1] == -a[3]  # d1_min == -d1_max
        assert a[4] == -a[5]  # d2_min == -d2_max

    def test_multiple_sizes(self):
        """Multiple sizes produce one anchor per combination."""
        anchors = _generate_cell_anchors(width=(8, 16), height=(8, 16), depth=(8, 16))
        assert anchors.shape == (8, 6)  # 2×2×2 combinations

    def test_dtype_is_float32(self):
        anchors = _generate_cell_anchors(width=(10,), height=(10,), depth=(10,))
        assert anchors.dtype == np.float32

    def test_half_extents(self):
        """Extents are half the specified size (centered at origin)."""
        anchors = _generate_cell_anchors(width=(20,), height=(30,), depth=(40,))
        a = anchors[0]
        assert a[0] == pytest.approx(-10.0)  # d0_min = -width/2
        assert a[2] == pytest.approx(10.0)   # d0_max = +width/2
        assert a[1] == pytest.approx(-15.0)  # d1_min = -height/2
        assert a[3] == pytest.approx(15.0)   # d1_max = +height/2
        assert a[4] == pytest.approx(-20.0)  # d2_min = -depth/2
        assert a[5] == pytest.approx(20.0)   # d2_max = +depth/2


class TestComputeFeatureMapSizes:
    def test_single_stride(self):
        """Single stride [2,2,2] on patch [64,96,96] at level 1."""
        fm = compute_feature_map_sizes(
            patch_size=(64, 96, 96),
            strides=[[2, 2, 2]],
            decoder_levels=[1],
        )
        assert fm == [[32, 48, 48]]

    def test_multiple_levels(self):
        """Multiple decoder levels select correct feature maps."""
        fm = compute_feature_map_sizes(
            patch_size=(64, 96, 96),
            strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            decoder_levels=[1, 2, 3],
        )
        assert fm[0] == [32, 48, 48]
        assert fm[1] == [16, 24, 24]
        assert fm[2] == [8, 12, 12]

    def test_non_uniform_strides(self):
        """Non-uniform strides (Z stride differs from Y/X)."""
        fm = compute_feature_map_sizes(
            patch_size=(64, 96, 96),
            strides=[[1, 2, 2], [2, 2, 2]],
            decoder_levels=[1, 2],
        )
        assert fm[0] == [64, 48, 48]  # Z not downsampled
        assert fm[1] == [32, 24, 24]


class TestGenerateAnchors:
    def test_total_count(self):
        """Total anchors = fm_cells × anchors_per_cell."""
        anchors = generate_anchors(
            widths=[(10,)], heights=[(10,)], depths=[(10,)],
            patch_size=(64, 64, 64),
            feature_map_sizes=[[4, 4, 4]],
        )
        assert anchors.shape == (64, 6)  # 4×4×4 cells × 1 anchor

    def test_multi_level(self):
        """Multiple feature levels produce cumulative anchor count."""
        anchors = generate_anchors(
            widths=[(8,), (16,)],
            heights=[(8,), (16,)],
            depths=[(8,), (16,)],
            patch_size=(32, 32, 32),
            feature_map_sizes=[[4, 4, 4], [2, 2, 2]],
        )
        expected = 4 * 4 * 4 + 2 * 2 * 2  # 64 + 8 = 72
        assert anchors.shape == (expected, 6)

    def test_output_columns(self):
        anchors = generate_anchors(
            widths=[(8,)], heights=[(8,)], depths=[(8,)],
            patch_size=(32, 32, 32),
            feature_map_sizes=[[4, 4, 4]],
        )
        assert anchors.shape[1] == 6


class TestComputeAnchors:
    @pytest.fixture()
    def plan(self):
        return {
            "anchors": {"width": [10], "height": [10], "depth": [10]},
            "architecture": {
                "strides": [[2, 2, 2]],
                "decoder_levels": [1],
            },
        }

    def test_batch_expansion(self, plan):
        """compute_anchors returns (batch_size, num_anchors, 6)."""
        anchors = compute_anchors(plan, patch_size=(16, 16, 16), batch_size=4)
        assert anchors.shape[0] == 4
        assert anchors.shape[2] == 6
        assert anchors.dtype == np.float32

    def test_batches_identical(self, plan):
        """All batch elements are identical copies."""
        anchors = compute_anchors(plan, patch_size=(16, 16, 16), batch_size=3)
        np.testing.assert_array_equal(anchors[0], anchors[1])
        np.testing.assert_array_equal(anchors[0], anchors[2])
