"""CLI input helpers shared by detection and segmentation entrypoints."""

import sys
from pathlib import Path
from typing import List


def _is_nifti(p: Path) -> bool:
    return p.name.endswith(".nii") or p.name.endswith(".nii.gz")


def collect_image_inputs(image_path: str = None, image_dir: str = None) -> List[Path]:
    """Validate the image arguments and return the list of input images.

    Each input is one image: either a NIfTI file (``.nii``/``.nii.gz``) or a DICOM series
    directory. With ``--image-dir``, **every entry** is treated as one image — a NIfTI file
    or a subdirectory holding a DICOM series (so a folder of series-folders is a DICOM batch,
    and NIfTI and DICOM inputs can be mixed in the same batch).
    """
    if image_path and image_dir:
        sys.exit("Error: --image-path and --image-dir are mutually exclusive")
    if not image_path and not image_dir:
        sys.exit("Error: --image-path or --image-dir is required for inference")

    if image_path:
        path = Path(image_path)
        if path.is_dir():
            return [path]  # DICOM series directory
        if not path.is_file():
            sys.exit(f"Error: image not found: {image_path}")
        if not _is_nifti(path):
            sys.exit(
                "Error: --image-path must be a NIfTI file (.nii/.nii.gz) or a DICOM series "
                f"directory, got: {path.name}"
            )
        return [path]

    directory = Path(image_dir)
    if not directory.is_dir():
        sys.exit(f"Error: image directory not found: {image_dir}")

    # Each entry is one image: a NIfTI file, or a subdirectory (DICOM series).
    inputs = sorted(p for p in directory.iterdir() if p.is_dir() or _is_nifti(p))
    if not inputs:
        sys.exit(f"Error: no NIfTI files or DICOM series directories found in {image_dir}")
    return inputs
