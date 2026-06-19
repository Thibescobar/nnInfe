"""CLI input helpers shared by detection and segmentation entrypoints."""

import sys
from pathlib import Path
from typing import List


def collect_nifti_inputs(image_path: str = None, image_dir: str = None) -> List[Path]:
    """Validate image args and return sorted input NIfTI paths."""
    if image_path and image_dir:
        sys.exit("Error: --image-path and --image-dir are mutually exclusive")
    if not image_path and not image_dir:
        sys.exit("Error: --image-path or --image-dir is required for inference")

    if image_path:
        path = Path(image_path)
        if not path.is_file():
            sys.exit(f"Error: image not found: {image_path}")
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            sys.exit(f"Error: image must be .nii or .nii.gz, got: {path.name}")
        return [path]

    directory = Path(image_dir)
    if not directory.is_dir():
        sys.exit(f"Error: image directory not found: {image_dir}")

    images = sorted([p for p in directory.iterdir() if p.name.endswith(".nii") or p.name.endswith(".nii.gz")])
    if not images:
        sys.exit(f"Error: no .nii or .nii.gz files found in {image_dir}")
    return images
