"""Tests for nnUNet segmentation helpers and CLI-level processing."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import SimpleITK as sitk

from nndet_onnx.nnunet_onnx_inference_sw import process_single_image
from nndet_onnx.segmentation.pipeline import (
    extract_plan_inference,
    flip_image_axes,
    pad_volume_to_patch_size,
    run_sliding_window_segmentation,
)


def _make_nifti(tmp_path, shape=(8, 8, 8), name="seg.nii.gz"):
    arr = np.random.uniform(-200, 200, shape).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    path = str(tmp_path / name)
    sitk.WriteImage(img, path)
    return path


def test_extract_plan_inference():
    plans = {
        "configurations": {
            "3d_fullres": {
                "patch_size": [4, 5, 6],
                "spacing": [1.2, 1.3, 1.4],
            }
        },
        "foreground_intensity_properties_per_channel": {
            "0": {
                "percentile_00_5": -1000.0,
                "percentile_99_5": 1200.0,
                "mean": -100.0,
                "std": 250.0,
            }
        },
    }
    plan = extract_plan_inference(plans)
    assert plan["patch_size"] == [4, 5, 6]
    assert plan["target_spacing"] == [1.2, 1.3, 1.4]
    assert plan["intensity_properties"]["std"] == 250.0


def test_run_sliding_window_segmentation_binary_shape():
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    patch_size = (4, 4, 4)

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[2, 1, 4, 4, 4])]

    # Output logits: class 0 low, class 1 high -> expected label 1 everywhere covered.
    logits = np.zeros((2, 2, 4, 4, 4), dtype=np.float32)
    logits[:, 1, ...] = 2.0
    mock_session.run.return_value = [logits]

    labels = run_sliding_window_segmentation(
        session=mock_session,
        volume_zyx=volume,
        patch_size_zyx=patch_size,
        batch_size=2,
        overlap=0.5,
    )
    assert labels.shape == volume.shape
    assert labels.dtype == np.uint16
    assert labels.max() == 1


def test_process_single_image_exports_mask(tmp_path):
    nifti_path = _make_nifti(tmp_path, shape=(8, 8, 8))
    output_dir = tmp_path / "out"

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 1, 8, 8, 8])]
    logits = np.zeros((1, 2, 8, 8, 8), dtype=np.float32)
    logits[:, 1, ...] = 3.0
    mock_session.run.return_value = [logits]

    plan = {
        "patch_size": [8, 8, 8],
        "target_spacing": [1.0, 1.0, 1.0],
        "intensity_properties": {
            "percentile_00_5": -300.0,
            "percentile_99_5": 300.0,
            "mean": 0.0,
            "std": 100.0,
        },
    }

    out = process_single_image(
        session=mock_session,
        image_path=nifti_path,
        output_dir=Path(output_dir),
        plan_inference=plan,
        patch_size=(8, 8, 8),
        overlap=0.5,
    )
    assert Path(out).exists()
    assert (output_dir / ".done").exists()


def test_process_single_image_pads_small_volume(tmp_path):
    nifti_path = _make_nifti(tmp_path, shape=(6, 6, 6), name="small.nii.gz")
    output_dir = tmp_path / "out_small"

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 1, 8, 8, 8])]
    logits = np.zeros((1, 2, 8, 8, 8), dtype=np.float32)
    logits[:, 1, ...] = 1.0
    mock_session.run.return_value = [logits]

    plan = {
        "patch_size": [8, 8, 8],
        "target_spacing": [1.0, 1.0, 1.0],
        "intensity_properties": {
            "percentile_00_5": -300.0,
            "percentile_99_5": 300.0,
            "mean": 0.0,
            "std": 100.0,
        },
    }

    out = process_single_image(
        session=mock_session,
        image_path=nifti_path,
        output_dir=Path(output_dir),
        plan_inference=plan,
        patch_size=(8, 8, 8),
        overlap=0.5,
    )

    assert Path(out).exists()
    out_img = sitk.ReadImage(out)
    assert out_img.GetSize() == (6, 6, 6)


def test_resolve_patch_size_zyx_prefers_model_shape():
    from nndet_onnx.nnunet_onnx_inference_sw import _resolve_patch_size_zyx

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(shape=[1, 1, 128, 112, 112])]

    resolved = _resolve_patch_size_zyx(mock_session, (112, 112, 128))
    assert resolved == (128, 112, 112)


def test_flip_image_axes_roundtrip():
    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.75, 1.25))
    img.SetOrigin((10.0, 20.0, 30.0))

    flipped = flip_image_axes(img, True, True, False)
    restored = flip_image_axes(flipped, True, True, False)

    restored_arr = sitk.GetArrayFromImage(restored)
    np.testing.assert_array_equal(restored_arr, arr)
    assert restored.GetSpacing() == img.GetSpacing()


def test_pad_volume_to_patch_size_uses_min_minus_one():
    volume = np.array(
        [
            [[3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=np.float32,
    )

    padded, original_shape = pad_volume_to_patch_size(volume, patch_size_zyx=(3, 3, 3))

    assert original_shape == (2, 2, 2)
    assert padded.shape == (3, 3, 3)
    assert padded[2, 2, 2] == np.float32(2.0)
