"""Tests for image I/O (nninfe.common.io): DICOM series read, NIfTI/DICOM dispatch, metadata.

Generates a tiny synthetic CT series with pydicom (no external data / PHI) and checks the
geometry and metadata survive the read into a sitk.Image.
"""

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from nninfe.common.io import (
    looks_like_dicom,
    read_dicom_series,
    read_image,
    read_image_metadata,
    read_series_metadata,
)

NX, NY, NZ = 16, 16, 8
DX, DY, DZ = 0.7617, 0.7617, 1.25
OX, OY, OZ = 10.0, 20.0, 30.0


def _write_synthetic_series(out_dir):
    study_uid, series_uid, for_uid = generate_uid(), generate_uid(), generate_uid()
    vol = np.random.default_rng(0).integers(-1000, 2000, size=(NZ, NY, NX)).astype(np.int16)
    for k in range(NZ):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        # Encoding is taken from file_meta.TransferSyntaxUID (Explicit VR LE) at save time.
        ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
        ds.PatientID = "TEST-0001"
        ds.PatientName = "Synthetic^CT"
        ds.Modality = "CT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = NY, NX
        ds.PixelSpacing = [DY, DX]
        ds.SliceThickness = DZ
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [OX, OY, OZ + k * DZ]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PixelData = vol[k].tobytes()
        ds.save_as(str(out_dir / f"slice_{k:03d}.dcm"), enforce_file_format=True)


def test_read_dicom_series_geometry(tmp_path):
    _write_synthetic_series(tmp_path)
    assert looks_like_dicom(str(tmp_path))

    image, files = read_dicom_series(str(tmp_path))
    assert image.GetSize() == (NX, NY, NZ)
    assert image.GetSpacing() == pytest.approx((DX, DY, DZ), abs=1e-3)
    assert image.GetOrigin() == pytest.approx((OX, OY, OZ), abs=1e-3)
    assert len(files) == NZ


def test_read_series_metadata(tmp_path):
    _write_synthetic_series(tmp_path)
    _, files = read_dicom_series(str(tmp_path))
    meta = read_series_metadata(files[0])
    for key in ("patient_id", "study_instance_uid", "series_instance_uid", "frame_of_reference_uid", "modality"):
        assert key in meta
    assert meta["modality"] == "CT"
    assert meta["patient_id"] == "TEST-0001"


def test_read_dicom_series_missing_dir():
    with pytest.raises(ValueError, match="not found"):
        read_dicom_series("/no/such/dicom/dir")


def test_read_image_dispatches_dicom_dir(tmp_path):
    _write_synthetic_series(tmp_path)
    img = read_image(str(tmp_path))
    assert img.GetSize() == (NX, NY, NZ)
    assert img.GetSpacing() == pytest.approx((DX, DY, DZ), abs=1e-3)


def test_read_image_dispatches_nifti_file(tmp_path):
    ref = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.int16))  # ZYX -> size (6, 5, 4)
    nii = tmp_path / "vol.nii.gz"
    sitk.WriteImage(ref, str(nii))
    img = read_image(str(nii))
    assert img.GetSize() == (6, 5, 4)


def test_read_image_metadata_dicom_dir(tmp_path):
    _write_synthetic_series(tmp_path)
    meta = read_image_metadata(str(tmp_path))
    assert meta["size_xyz"] == (NX, NY, NZ)
    assert meta["spacing_xyz"] == pytest.approx((DX, DY, DZ), abs=1e-3)


def test_read_image_metadata_nifti_file(tmp_path):
    ref = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.int16))
    nii = tmp_path / "vol.nii.gz"
    sitk.WriteImage(ref, str(nii))
    meta = read_image_metadata(str(nii))
    assert meta["size_xyz"] == (6, 5, 4)
