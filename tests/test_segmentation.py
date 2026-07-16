"""Tests for nnUNet segmentation helpers and CLI-level processing."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from nninfe.infer_segmentation import main, process_single_image
from nninfe.segmentation.pipeline import (
    extract_plan_inference,
    flip_image_axes,
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


def test_run_sliding_window_segmentation_average_logits_invariant():
    """The label map is identical with and without the logit averaging: the accumulated
    weights are shared by all classes at each voxel, so dividing never changes the argmax."""
    volume = np.zeros((6, 6, 6), dtype=np.float32)
    patch_size = (4, 4, 4)

    rng = np.random.default_rng(42)
    logits = rng.normal(size=(2, 3, 4, 4, 4)).astype(np.float32)

    def make_session():
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="images", shape=[2, 1, 4, 4, 4])]
        session.run.return_value = [logits]
        return session

    kwargs = dict(volume_zyx=volume, patch_size_zyx=patch_size, batch_size=2, overlap=0.5)
    labels_raw = run_sliding_window_segmentation(session=make_session(), average_logits=False, **kwargs)
    labels_avg = run_sliding_window_segmentation(session=make_session(), average_logits=True, **kwargs)
    np.testing.assert_array_equal(labels_raw, labels_avg)


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
    from nninfe.infer_segmentation import _resolve_patch_size_zyx

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

class TestSegmentationMain:
    @patch("nninfe.infer_segmentation.create_session")
    @patch("nninfe.infer_segmentation.process_single_image")
    @patch("sys.argv", new_callable=list)
    def test_main_single_image(self, mock_argv, mock_process, mock_create, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.write_text("")

        plan_path = tmp_path / "plans.json"
        plans = {
            "foreground_intensity_properties_per_channel": {
                "0": {
                    "percentile_00_5": 0.0,
                    "percentile_99_5": 100.0,
                    "mean": 50.0,
                    "std": 10.0
                }
            },
            "configurations": {
                "3d_fullres": {
                    "patch_size": [128, 128, 128],
                    "spacing": [1.0, 1.0, 1.0],
                    "normalization_schemes": ["ZScoreNormalization"]
                }
            }
        }
        plan_path.write_text(json.dumps(plans))

        image_path = tmp_path / "image.nii.gz"
        image_path.write_text("")

        output_dir = tmp_path / "output"

        mock_argv[:] = [
            "nninfe-seg",
            "--model-path", str(model_path),
            "--plan-path", str(plan_path),
            "--image-path", str(image_path),
            "--output-dir", str(output_dir),
            "--overlap", "0.5",
            "--pad-value", "min"
        ]

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[1, 1, 128, 128, 128])]
        mock_create.return_value = mock_session
        mock_process.return_value = str(output_dir / "image_seg.nii.gz")

        main()

        mock_create.assert_called_once_with(str(model_path), backend="cpu", trt_fp16=False)
        mock_process.assert_called_once()
        kwargs = mock_process.call_args.kwargs
        assert kwargs["session"] == mock_session
        assert kwargs["image_path"] == str(image_path)
        assert kwargs["output_dir"] == output_dir
        assert kwargs["overlap"] == 0.5
        assert kwargs["pad_value"] == "min"

    @patch("nninfe.infer_segmentation.create_session")
    @patch("sys.argv", new_callable=list)
    def test_main_build_engine_only(self, mock_argv, mock_create, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.write_text("")

        plan_path = tmp_path / "plans.json"
        plans = {
            "foreground_intensity_properties_per_channel": {
                "0": {
                    "percentile_00_5": 0.0,
                    "percentile_99_5": 100.0,
                    "mean": 50.0,
                    "std": 10.0
                }
            },
            "configurations": {
                "3d_fullres": {
                    "patch_size": [64, 64, 64],
                    "spacing": [1.0, 1.0, 1.0],
                    "normalization_schemes": ["ZScoreNormalization"]
                }
            }
        }
        plan_path.write_text(json.dumps(plans))

        mock_argv[:] = [
            "nninfe-seg",
            "--model-path", str(model_path),
            "--plan-path", str(plan_path),
            "--backend", "trt",
            "--trt-fp16",
            "--build-engine-only"
        ]

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[1, 1, "batch", 64, 64])]
        mock_create.return_value = mock_session

        main()

        mock_create.assert_called_once_with(str(model_path), backend="trt", trt_fp16=True)

    @patch("sys.argv", new_callable=list)
    def test_main_missing_model_exits(self, mock_argv, tmp_path):
        mock_argv[:] = [
            "nninfe-seg",
            "--model-path", str(tmp_path / "missing.onnx"),
            "--plan-path", "dummy.json"
        ]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "not found" in str(exc.value)

