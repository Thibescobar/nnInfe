"""Tests for main() and process_single_image() with mocked I/O."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from nninfe.infer_detection import process_single_image


def _make_plan():
    return {
        "patch_size": [8, 8, 8],
        "target_spacing": [1.0, 1.0, 1.0],
        "intensity_properties": {
            "percentile_00_5": 0.0,
            "percentile_99_5": 1000.0,
            "mean": 500.0,
            "std": 200.0,
        },
        "anchors": {
            "width": [4, 8],
            "height": [4, 8],
            "depth": [4, 8],
        },
        "architecture": {
            "strides": [[1, 1, 1], [2, 2, 2]],
            "decoder_levels": [1, 2],
        },
        "inference_plan": {
            "model_iou": 0.1,
            "model_score_thresh": 0.5,
            "model_topk": 100,
            "model_detections_per_image": 100,
            "remove_small_boxes": 2.0,
        },
    }


def _make_nifti(tmp_path, shape=(16, 16, 16), name="test.nii.gz"):
    arr = np.random.uniform(0, 1000, shape).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    path = str(tmp_path / name)
    sitk.WriteImage(img, path)
    return path


class TestProcessSingleImage:
    def test_returns_detection_count(self, tmp_path):
        """process_single_image runs end-to-end and returns int count."""
        nifti_path = _make_nifti(tmp_path)
        plan = _make_plan()
        batch_size = 2
        patch_size = tuple(plan["patch_size"])

        # Mock session
        mock_session = MagicMock()
        # Each run returns: batch_size boxes + batch_size scores + batch_size labels
        mock_session.run.return_value = [
            # boxes per batch element
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            # scores
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            # labels
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]

        from nninfe.detection.anchors import compute_anchors
        anchors_batch = compute_anchors(plan, patch_size, batch_size)

        output_dir = tmp_path / "results"

        result = process_single_image(
            session=mock_session,
            image_path=nifti_path,
            output_dir=output_dir,
            plan_inference=plan,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=0.1,
            overlap=0.5,
            score_thresh=0.5,
            min_size_mm=2.0,
            nms_backend="numpy",
            no_global_nms=False,
            export_pkl=False,
            pad_value="0.0",
        )

        assert isinstance(result, int)
        assert result >= 0
        # Output files should exist
        assert (output_dir / "test_mask.nii.gz").exists()
        assert (output_dir / "test_boxes.json").exists()
        assert (output_dir / "test_boxes.csv").exists()

    def test_with_detections(self, tmp_path):
        """process_single_image with actual detections from mock session."""
        nifti_path = _make_nifti(tmp_path)
        plan = _make_plan()
        batch_size = 2
        patch_size = tuple(plan["patch_size"])

        mock_session = MagicMock()

        def fake_run(_, inputs):
            return [
                # boxes: one detection per batch element
                np.array([[1, 1, 5, 5, 1, 5]], dtype=np.float32),
                np.array([[2, 2, 6, 6, 2, 6]], dtype=np.float32),
                # scores
                np.array([0.9], dtype=np.float32),
                np.array([0.85], dtype=np.float32),
                # labels
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
            ]

        mock_session.run.side_effect = fake_run

        from nninfe.detection.anchors import compute_anchors
        anchors_batch = compute_anchors(plan, patch_size, batch_size)
        output_dir = tmp_path / "results"

        result = process_single_image(
            session=mock_session,
            image_path=nifti_path,
            output_dir=output_dir,
            plan_inference=plan,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=0.1,
            overlap=0.5,
            score_thresh=0.3,
            min_size_mm=0.5,
            nms_backend="numpy",
            no_global_nms=False,
            export_pkl=True,
            pad_value="0.0",
        )

        assert result > 0
        assert (output_dir / "test_boxes.pkl").exists()

    def test_no_global_nms(self, tmp_path):
        """process_single_image with no_global_nms=True."""
        nifti_path = _make_nifti(tmp_path)
        plan = _make_plan()
        batch_size = 2
        patch_size = tuple(plan["patch_size"])

        mock_session = MagicMock()
        mock_session.run.return_value = [
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]

        from nninfe.detection.anchors import compute_anchors
        anchors_batch = compute_anchors(plan, patch_size, batch_size)
        output_dir = tmp_path / "results"

        result = process_single_image(
            session=mock_session,
            image_path=nifti_path,
            output_dir=output_dir,
            plan_inference=plan,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=0.1,
            overlap=0.5,
            score_thresh=0.5,
            min_size_mm=2.0,
            nms_backend="numpy",
            no_global_nms=True,
            export_pkl=False,
            pad_value="0.0",
        )
        assert isinstance(result, int)


class TestMain:
    @patch("nninfe.infer_detection.create_session")
    def test_main_single_image(self, mock_create_session, tmp_path):
        """main() with a single image in minimal mode."""
        import json

        from nninfe.infer_detection import main

        # Write plan
        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        # Write fake ONNX (main only checks extension + existence)
        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        # Write NIfTI
        nifti_path = _make_nifti(tmp_path)

        output_dir = str(tmp_path / "out")

        # Mock session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2, 1, 8, 8, 8])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.run.return_value = [
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]
        mock_create_session.return_value = mock_session

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--image-path", nifti_path,
                "--output-dir", output_dir,
                "--backend", "cpu",
            ],
        ):
            main()

        assert Path(output_dir).exists()

    @patch("nninfe.infer_detection.create_session")
    def test_main_build_engine_only(self, mock_create_session, tmp_path):
        """main() with --build-engine-only exits early."""
        import json

        from nninfe.infer_detection import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2, 1, 8, 8, 8])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_create_session.return_value = mock_session

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--backend", "cpu",
                "--build-engine-only",
            ],
        ):
            main()

        # No inference should have been run
        mock_session.run.assert_not_called()

    @patch("nninfe.infer_detection.create_session")
    def test_main_batch_mode(self, mock_create_session, tmp_path):
        """main() with --image-dir processes multiple images."""
        import json

        from nninfe.infer_detection import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        # Write 2 NIfTI images in a directory
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _make_nifti(img_dir, name="img1.nii.gz")
        _make_nifti(img_dir, name="img2.nii.gz")

        output_dir = str(tmp_path / "out")

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2, 1, 8, 8, 8])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.run.return_value = [
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]
        mock_create_session.return_value = mock_session

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--image-dir", str(img_dir),
                "--output-dir", output_dir,
                "--backend", "cpu",
            ],
        ):
            main()

        # Both images should have outputs
        assert (Path(output_dir) / "img1_boxes.json").exists()
        assert (Path(output_dir) / "img2_boxes.json").exists()

    def test_main_missing_model_exits(self, tmp_path):
        """main() exits if model file doesn't exist."""
        import json

        from nninfe.infer_detection import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", "/nonexistent/model.onnx",
                "--plan-path", plan_path,
                "--image-path", "/fake/image.nii.gz",
                "--output-dir", str(tmp_path / "out"),
            ],
        ):
            with pytest.raises(SystemExit):
                main()

    def test_main_mutual_exclusion(self, tmp_path):
        """main() exits if both --image-path and --image-dir given."""
        import json

        from nninfe.infer_detection import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        nifti_path = _make_nifti(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--image-path", nifti_path,
                "--image-dir", str(img_dir),
                "--output-dir", str(tmp_path / "out"),
            ],
        ):
            with pytest.raises(SystemExit):
                main()

    @patch("sys.argv", new_callable=list)
    def test_main_exits(self, mock_argv, tmp_path):
        from nninfe.infer_detection import main

        model_path = tmp_path / "model.onnx"
        plan_path = tmp_path / "plan.json"

        # Missing model
        mock_argv[:] = ["nninfe-det", "--model-path", str(model_path), "--plan-path", str(plan_path)]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "model not found" in str(exc.value)

        # Invalid model ext
        model_txt = tmp_path / "model.txt"
        model_txt.write_text("")
        mock_argv[:] = ["nninfe-det", "--model-path", str(model_txt), "--plan-path", str(plan_path)]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "must be .onnx" in str(exc.value)

        # Valid model, missing plan
        model_onnx = tmp_path / "actual_model.onnx"
        model_onnx.write_text("")
        mock_argv[:] = ["nninfe-det", "--model-path", str(model_onnx), "--plan-path", str(plan_path)]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "plan not found" in str(exc.value)

        # Invalid plan ext
        plan_txt = tmp_path / "plan.txt"
        plan_txt.write_text("")
        mock_argv[:] = ["nninfe-det", "--model-path", str(model_onnx), "--plan-path", str(plan_txt)]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "must be .json" in str(exc.value)

        # Missing output dir
        actual_plan = tmp_path / "actual_plan.json"
        actual_plan.write_text("{}")
        nifti = tmp_path / "img.nii.gz"
        nifti.write_text("")
        mock_argv[:] = ["nninfe-det", "--model-path", str(model_onnx), "--plan-path", str(actual_plan), "--image-path", str(nifti)]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "output-dir is required" in str(exc.value)

        # Invalid overlap
        mock_argv[:] = [
            "nninfe-det", "--model-path", str(model_onnx), "--plan-path", str(actual_plan),
            "--image-path", str(nifti), "--output-dir", str(tmp_path), "--overlap", "1.5"
        ]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "overlap must be in" in str(exc.value)

        # Invalid score thresh
        mock_argv[:] = [
            "nninfe-det", "--model-path", str(model_onnx), "--plan-path", str(actual_plan),
            "--image-path", str(nifti), "--output-dir", str(tmp_path), "--score-thresh", "2.0"
        ]
        with pytest.raises(SystemExit) as exc:
            main()
        assert "score-thresh must be in" in str(exc.value)

    @patch("nninfe.infer_detection.create_session")
    @patch("nninfe.infer_detection.process_single_image")
    @patch("sys.argv", new_callable=list)
    def test_main_trt_batch(self, mock_argv, mock_process, mock_create, tmp_path):
        import json

        from nninfe.infer_detection import main

        model = tmp_path / "model.onnx"
        model.write_text("")
        plan = tmp_path / "plan.json"

        plan_dict = _make_plan()
        plan.write_text(json.dumps(plan_dict))

        # Make batch images
        d = tmp_path / "imgs"
        d.mkdir()
        (d / "1.nii.gz").write_text("")
        (d / "2.nii.gz").write_text("")

        # Fake cache
        cache = tmp_path / "trt_engine_cache_fp16"
        cache.mkdir()
        (cache / "model.engine").write_text("")

        mock_argv[:] = [
            "nninfe-det", "--model-path", str(model), "--plan-path", str(plan),
            "--image-dir", str(d), "--output-dir", str(tmp_path),
            "--backend", "trt", "--trt-fp16"
        ]

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2])]
        mock_create.return_value = mock_session
        mock_process.return_value = 1

        main()
        assert mock_process.call_count == 2
