import argparse

import pytest

from nninfe.common.cli import DefaultsHelpFormatter, collect_image_inputs
from nninfe.common.errors import EXIT_USAGE


def _assert_usage_exit(exc, capsys, expected_msg):
    """Usage errors exit with EXIT_USAGE (2) and print a clean message to stderr (no traceback)."""
    assert exc.value.code == EXIT_USAGE
    assert expected_msg in capsys.readouterr().err


def test_collect_both_args_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs("img.nii", "dir/")
    _assert_usage_exit(exc, capsys, "mutually exclusive")


def test_collect_neither_args_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs()
    _assert_usage_exit(exc, capsys, "required")


def test_collect_missing_image_exits(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_path=str(tmp_path / "missing.nii"))
    _assert_usage_exit(exc, capsys, "not found")


def test_collect_invalid_ext_exits(tmp_path, capsys):
    invalid = tmp_path / "image.txt"
    invalid.write_text("")
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_path=str(invalid))
    _assert_usage_exit(exc, capsys, "must be a NIfTI file")


def test_collect_missing_dir_exits(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_dir=str(tmp_path / "missing_dir"))
    _assert_usage_exit(exc, capsys, "directory not found")


def test_collect_empty_dir_exits(tmp_path, capsys):
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    (empty / "not_an_image.txt").write_text("")
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_dir=str(empty))
    _assert_usage_exit(exc, capsys, "no NIfTI files or DICOM series directories")


def test_help_formatter_uses_actual_concrete_defaults():
    parser = argparse.ArgumentParser(formatter_class=DefaultsHelpFormatter)
    parser.add_argument("--overlap", type=float, default=0.5, help="Patch overlap")
    parser.add_argument("--optional", default=None, help="Optional value")
    parser.add_argument("--feature", action="store_true", help="Enable feature")

    help_text = parser.format_help()

    assert "Patch overlap (default: 0.5)" in help_text
    assert "Optional value (default: None)" not in help_text
    assert "Enable feature (default: False)" not in help_text


def test_collect_valid_dir_nifti(tmp_path):
    d = tmp_path / "valid_dir"
    d.mkdir()
    f1 = d / "a.nii.gz"
    f2 = d / "b.nii"
    f1.write_text("")
    f2.write_text("")
    res = collect_image_inputs(image_dir=str(d))
    assert res == [f1, f2]


def test_collect_dir_with_dicom_subdirs(tmp_path):
    d = tmp_path / "studies"
    d.mkdir()
    s1 = d / "series1"
    s2 = d / "series2"
    s1.mkdir()
    s2.mkdir()
    res = collect_image_inputs(image_dir=str(d))
    assert res == [s1, s2]


def test_collect_dir_mixed_nifti_and_dicom(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    nii = d / "a.nii.gz"
    nii.write_text("")
    series = d / "series1"
    series.mkdir()
    (d / "notes.txt").write_text("")  # ignored
    res = collect_image_inputs(image_dir=str(d))
    assert set(res) == {nii, series}


def test_collect_image_path_directory_is_dicom(tmp_path):
    series = tmp_path / "series1"
    series.mkdir()
    res = collect_image_inputs(image_path=str(series))
    assert res == [series]
