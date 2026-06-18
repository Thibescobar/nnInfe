"""Tests for preprocessing functions (resample, clip, normalize, preprocess_image)."""

import numpy as np
import pytest
import SimpleITK as sitk

from nndet_onnx.common.preprocessing import (
    clip_image,
    normalize_image,
    preprocess_image,
    resample_image,
)


def _make_sitk_image(array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Helper: create a SimpleITK image from a numpy array (ZYX)."""
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing(spacing)  # XYZ
    img.SetOrigin(origin)
    return img


class TestResampleImage:
    def test_doubles_spacing(self):
        arr = np.ones((10, 20, 30), dtype=np.float32)
        img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        # target_spacing is ZYX: double all spacings
        result = resample_image(img, target_spacing_zyx=[2.0, 2.0, 2.0])
        # Size should halve (XYZ order from GetSize)
        assert result.GetSize() == (15, 10, 5)
        assert result.GetSpacing() == pytest.approx((2.0, 2.0, 2.0))

    def test_preserves_origin(self):
        arr = np.zeros((8, 8, 8), dtype=np.float32)
        img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0), origin=(10.0, 20.0, 30.0))
        result = resample_image(img, target_spacing_zyx=[1.0, 1.0, 1.0])
        assert result.GetOrigin() == pytest.approx((10.0, 20.0, 30.0))

    def test_anisotropic_spacing(self):
        arr = np.ones((10, 20, 40), dtype=np.float32)
        img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        # target ZYX = [2.0, 1.0, 0.5] → XYZ spacing = [0.5, 1.0, 2.0]
        result = resample_image(img, target_spacing_zyx=[2.0, 1.0, 0.5])
        assert result.GetSpacing() == pytest.approx((0.5, 1.0, 2.0))
        # X: 40 * 1.0/0.5 = 80, Y: 20 * 1.0/1.0 = 20, Z: 10 * 1.0/2.0 = 5
        assert result.GetSize() == (80, 20, 5)

    def test_applies_transpose_forward_to_target_spacing(self):
        arr = np.ones((10, 20, 40), dtype=np.float32)
        img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))

        result = resample_image(
            img,
            target_spacing_zyx=[2.0, 1.0, 0.5],
            transpose_forward_zyx=[2, 0, 1],
        )

        # Uses Aurore's explicit map: my_map = {2: 2.0, 0: 1.0, 1: 0.5}
        # mapped to vector1 = [my_map[1], my_map[2], my_map[0]] = [0.5, 2.0, 1.0]
        assert result.GetSpacing() == pytest.approx((0.5, 2.0, 1.0))
        # X: 40 * 1.0/0.5 = 80, Y: 20 * 1.0/2.0 = 10, Z: 10 * 1.0/1.0 = 10
        assert result.GetSize() == (80, 10, 10)


class TestClipImage:
    def test_clamps_values(self):
        arr = np.array([[[0, 50, 100, 200, 255]]], dtype=np.float32)
        img = _make_sitk_image(arr)
        result = clip_image(img, lower=50.0, upper=200.0)
        result_arr = sitk.GetArrayFromImage(result)
        assert result_arr.min() == pytest.approx(50.0)
        assert result_arr.max() == pytest.approx(200.0)

    def test_no_change_within_bounds(self):
        arr = np.array([[[60, 100, 150]]], dtype=np.float32)
        img = _make_sitk_image(arr)
        result = clip_image(img, lower=0.0, upper=200.0)
        result_arr = sitk.GetArrayFromImage(result)
        np.testing.assert_array_almost_equal(result_arr.flatten(), [60, 100, 150])


class TestNormalizeImage:
    def test_zscore(self):
        arr = np.array([[[10, 20, 30]]], dtype=np.float32)
        img = _make_sitk_image(arr)
        result = normalize_image(img, mean=20.0, std=10.0)
        result_arr = sitk.GetArrayFromImage(result)
        np.testing.assert_array_almost_equal(result_arr.flatten(), [-1.0, 0.0, 1.0])

    def test_zero_mean_unit_std(self):
        arr = np.array([[[100, 100, 100]]], dtype=np.float32)
        img = _make_sitk_image(arr)
        result = normalize_image(img, mean=100.0, std=1.0)
        result_arr = sitk.GetArrayFromImage(result)
        np.testing.assert_array_almost_equal(result_arr.flatten(), [0.0, 0.0, 0.0])


class TestPreprocessImage:
    def test_full_chain(self, tmp_path):
        """Full preprocessing: cast → resample → clip → normalize."""
        arr = np.random.uniform(0, 1000, (16, 16, 16)).astype(np.float32)
        img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        nifti_path = str(tmp_path / "test.nii.gz")
        sitk.WriteImage(img, nifti_path)

        plan = {
            "target_spacing": [1.0, 1.0, 1.0],  # ZYX, same as input
            "intensity_properties": {
                "percentile_00_5": 50.0,
                "percentile_99_5": 950.0,
                "mean": 500.0,
                "std": 200.0,
            },
        }
        result = preprocess_image(nifti_path, plan, verbose=False)
        result_arr = sitk.GetArrayFromImage(result)

        # After clipping to [50, 950] and normalizing with mean=500, std=200
        assert result_arr.shape == (16, 16, 16)
        # Clipped range normalized: (50-500)/200 = -2.25, (950-500)/200 = 2.25
        assert result_arr.min() >= -2.3
        assert result_arr.max() <= 2.3

    def test_verbose_output(self, tmp_path, capsys):
        arr = np.ones((8, 8, 8), dtype=np.float32) * 100
        img = _make_sitk_image(arr)
        nifti_path = str(tmp_path / "test.nii.gz")
        sitk.WriteImage(img, nifti_path)

        plan = {
            "target_spacing": [1.0, 1.0, 1.0],
            "intensity_properties": {
                "percentile_00_5": 0.0,
                "percentile_99_5": 200.0,
                "mean": 100.0,
                "std": 50.0,
            },
        }
        preprocess_image(nifti_path, plan, verbose=True)
        captured = capsys.readouterr()
        assert "original" in captured.out
        assert "resampled" in captured.out
        assert "clipped" in captured.out
        assert "normalized" in captured.out
