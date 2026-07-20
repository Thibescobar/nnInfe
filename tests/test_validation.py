"""Tests for the Phase-3 input sanity guards (nninfe.common.validation)."""

import numpy as np
import pytest
import SimpleITK as sitk

from nninfe.common.errors import EXIT_INPUT, InputValidationError
from nninfe.common.manifest import classify_error
from nninfe.common.validation import validate_input_image


class _GeomStub:
    """Minimal duck-typed stand-in exposing just the header accessors validate_input_image uses,
    so each guard can be exercised without SimpleITK's construction constraints."""

    def __init__(self, dim=3, channels=1, spacing=(1.0, 1.0, 1.0), size=(8, 8, 8)):
        self._dim, self._channels, self._spacing, self._size = dim, channels, spacing, size

    def GetDimension(self):
        return self._dim

    def GetNumberOfComponentsPerPixel(self):
        return self._channels

    def GetSpacing(self):
        return self._spacing

    def GetSize(self):
        return self._size


class TestValidateInputImage:
    def test_valid_image_passes(self):
        validate_input_image(_GeomStub())  # must not raise

    def test_real_3d_scalar_image_passes(self):
        img = sitk.GetImageFromArray(np.zeros((8, 8, 8), np.float32))
        img.SetSpacing((0.7, 0.7, 1.25))
        validate_input_image(img)

    def test_rejects_non_3d(self):
        with pytest.raises(InputValidationError, match="3D"):
            validate_input_image(_GeomStub(dim=2))

    def test_rejects_multichannel(self):
        with pytest.raises(InputValidationError, match="channel"):
            validate_input_image(_GeomStub(channels=3))

    def test_rejects_non_positive_spacing(self):
        with pytest.raises(InputValidationError, match="spacing"):
            validate_input_image(_GeomStub(spacing=(1.0, 0.0, 1.0)))

    def test_rejects_non_finite_spacing(self):
        with pytest.raises(InputValidationError, match="spacing"):
            validate_input_image(_GeomStub(spacing=(1.0, float("nan"), 1.0)))

    def test_rejects_empty_size(self):
        with pytest.raises(InputValidationError, match="empty"):
            validate_input_image(_GeomStub(size=(8, 0, 8)))

    def test_error_exit_code_and_classification(self):
        assert InputValidationError.exit_code == EXIT_INPUT
        assert classify_error(InputValidationError("x")) == {
            "type": "InputValidationError", "scope": "image", "retryable": False,
        }


class TestPreprocessRejectsBadInput:
    def test_preprocess_image_raises_on_2d_nifti(self, tmp_path):
        from nninfe.common.preprocessing import preprocess_image

        img = sitk.GetImageFromArray(np.zeros((8, 8), np.float32))  # 2D
        path = str(tmp_path / "flat.nii.gz")
        sitk.WriteImage(img, path)
        plan = {
            "target_spacing": [1.0, 1.0, 1.0],
            "intensity_properties": {"percentile_00_5": 0.0, "percentile_99_5": 1.0, "mean": 0.0, "std": 1.0},
        }
        with pytest.raises(InputValidationError):
            preprocess_image(path, plan, verbose=False)
