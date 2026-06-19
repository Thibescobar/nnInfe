import pytest
from pathlib import Path
from nninfe.common.cli import collect_nifti_inputs

def test_collect_both_args_exits():
    with pytest.raises(SystemExit) as exc:
        collect_nifti_inputs("img.nii", "dir/")
    assert "mutually exclusive" in str(exc.value)

def test_collect_neither_args_exits():
    with pytest.raises(SystemExit) as exc:
        collect_nifti_inputs()
    assert "required" in str(exc.value)

def test_collect_missing_image_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        collect_nifti_inputs(image_path=str(tmp_path / "missing.nii"))
    assert "not found" in str(exc.value)

def test_collect_invalid_ext_exits(tmp_path):
    invalid = tmp_path / "image.txt"
    invalid.write_text("")
    with pytest.raises(SystemExit) as exc:
        collect_nifti_inputs(image_path=str(invalid))
    assert "must be .nii or .nii.gz" in str(exc.value)

def test_collect_missing_dir_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        collect_nifti_inputs(image_dir=str(tmp_path / "missing_dir"))
    assert "directory not found" in str(exc.value)

def test_collect_empty_dir_exits(tmp_path):
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    (empty / "not_nifti.txt").write_text("")
    with pytest.raises(SystemExit) as exc:
        collect_nifti_inputs(image_dir=str(empty))
    assert "no .nii or .nii.gz" in str(exc.value)

def test_collect_valid_dir(tmp_path):
    d = tmp_path / "valid_dir"
    d.mkdir()
    f1 = d / "a.nii.gz"
    f2 = d / "b.nii"
    f1.write_text("")
    f2.write_text("")
    res = collect_nifti_inputs(image_dir=str(d))
    assert len(res) == 2
    assert f1 in res
    assert f2 in res
