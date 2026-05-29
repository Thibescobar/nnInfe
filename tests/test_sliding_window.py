"""Tests for sliding window patch generation."""

import numpy as np

from nndet_onnx.nndet_onnx_inference_sw import (
    compute_patch_positions,
    extract_patch,
)


class TestComputePatchPositions:
    def test_image_equals_patch(self):
        """Image exactly equal to patch → single position at origin."""
        positions, steps = compute_patch_positions(
            image_shape_zyx=(64, 96, 96),
            patch_size_zyx=(64, 96, 96),
            overlap=0.5,
        )
        assert len(positions) == 1
        assert positions[0] == (0, 0, 0)

    def test_double_size_no_overlap(self):
        """Image = 2× patch with overlap=0 → 8 patches (2×2×2)."""
        positions, steps = compute_patch_positions(
            image_shape_zyx=(20, 20, 20),
            patch_size_zyx=(10, 10, 10),
            overlap=0.0,
        )
        assert len(positions) == 8

    def test_overlap_guarantee(self):
        """Actual overlap is at least the requested overlap."""
        positions, steps = compute_patch_positions(
            image_shape_zyx=(100, 150, 150),
            patch_size_zyx=(64, 96, 96),
            overlap=0.5,
        )
        for step, patch_dim in zip(steps, (64, 96, 96)):
            if step > 0:
                actual_overlap = 1.0 - step / patch_dim
                assert actual_overlap >= 0.5 - 1e-6

    def test_last_patch_covers_boundary(self):
        """Last patch in each axis ends at the image boundary."""
        image_shape = (100, 150, 200)
        patch_size = (64, 96, 96)
        positions, _ = compute_patch_positions(image_shape, patch_size, overlap=0.5)

        max_z = max(p[0] for p in positions)
        max_y = max(p[1] for p in positions)
        max_x = max(p[2] for p in positions)

        assert max_z + patch_size[0] == image_shape[0]
        assert max_y + patch_size[1] == image_shape[1]
        assert max_x + patch_size[2] == image_shape[2]

    def test_positions_non_negative(self):
        """All positions are >= 0."""
        positions, _ = compute_patch_positions(
            image_shape_zyx=(128, 128, 128),
            patch_size_zyx=(64, 96, 96),
            overlap=0.5,
        )
        for z, y, x in positions:
            assert z >= 0
            assert y >= 0
            assert x >= 0

    def test_step_sizes_returned(self):
        """Step sizes tuple has 3 elements."""
        _, steps = compute_patch_positions(
            image_shape_zyx=(128, 128, 128),
            patch_size_zyx=(64, 64, 64),
            overlap=0.5,
        )
        assert len(steps) == 3


class TestExtractPatch:
    def test_correct_region(self):
        """Extract a known region from a volume."""
        volume = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
        patch = extract_patch(volume, (2, 3, 1), (4, 4, 4))
        assert patch.shape == (4, 4, 4)
        np.testing.assert_array_equal(patch, volume[2:6, 3:7, 1:5])

    def test_full_volume(self):
        """Extracting full volume returns the same data."""
        volume = np.ones((10, 10, 10), dtype=np.float32)
        patch = extract_patch(volume, (0, 0, 0), (10, 10, 10))
        assert patch.shape == (10, 10, 10)
        np.testing.assert_array_equal(patch, volume)
