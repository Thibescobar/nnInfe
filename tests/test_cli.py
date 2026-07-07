import pytest

from nninfe.common.cli import collect_image_inputs


def test_collect_both_args_exits():
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs("img.nii", "dir/")
    assert "mutually exclusive" in str(exc.value)


def test_collect_neither_args_exits():
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs()
    assert "required" in str(exc.value)


def test_collect_missing_image_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_path=str(tmp_path / "missing.nii"))
    assert "not found" in str(exc.value)


def test_collect_invalid_ext_exits(tmp_path):
    invalid = tmp_path / "image.txt"
    invalid.write_text("")
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_path=str(invalid))
    assert "must be a NIfTI file" in str(exc.value)


def test_collect_missing_dir_exits(tmp_path):
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_dir=str(tmp_path / "missing_dir"))
    assert "directory not found" in str(exc.value)


def test_collect_empty_dir_exits(tmp_path):
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    (empty / "not_an_image.txt").write_text("")
    with pytest.raises(SystemExit) as exc:
        collect_image_inputs(image_dir=str(empty))
    assert "no NIfTI files or DICOM series directories" in str(exc.value)


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
