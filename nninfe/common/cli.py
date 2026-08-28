"""CLI helpers shared by detection and segmentation entrypoints."""

import argparse
import logging
from pathlib import Path
from typing import List

from nninfe.common.errors import fail_usage


class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show meaningful defaults without cluttering help with ``None``/``False``.

    Boolean flags already communicate their disabled default through their wording, and ``None``
    generally means either "not supplied" or a value resolved dynamically by the pipeline.
    Concrete defaults such as overlap, backend, output format, and configuration are appended by
    :class:`argparse.ArgumentDefaultsHelpFormatter` from the actual parser value, preventing the
    help text from drifting away from the implementation.
    """

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def make_parser(description: str) -> argparse.ArgumentParser:
    """Create a consistently formatted nninfe argument parser."""
    return argparse.ArgumentParser(description=description, formatter_class=DefaultsHelpFormatter)


def log_parameters(
    logger: logging.Logger,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    dynamic_defaults: dict[str, str] | None = None,
) -> None:
    """Log parsed parameters with consistent default annotations."""
    defaults = {a.dest: a.default for a in parser._actions if a.default is not argparse.SUPPRESS}
    dynamic_defaults = dynamic_defaults or {}

    logger.info("Parameters:")
    for name, value in vars(args).items():
        tag = ""
        if name in defaults and value == defaults[name]:
            if name in dynamic_defaults:
                tag = f"  (auto: {dynamic_defaults[name]})"
            else:
                tag = "  (default)"
        logger.info(f"  --{name.replace('_', '-')} : {value}{tag}")


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
        fail_usage("--image-path and --image-dir are mutually exclusive")
    if not image_path and not image_dir:
        fail_usage("--image-path or --image-dir is required for inference")

    if image_path:
        path = Path(image_path)
        if path.is_dir():
            return [path]  # DICOM series directory
        if not path.is_file():
            fail_usage(f"image not found: {image_path}")
        if not _is_nifti(path):
            fail_usage(
                "--image-path must be a NIfTI file (.nii/.nii.gz) or a DICOM series "
                f"directory, got: {path.name}"
            )
        return [path]

    directory = Path(image_dir)
    if not directory.is_dir():
        fail_usage(f"image directory not found: {image_dir}")

    # Each entry is one image: a NIfTI file, or a subdirectory (DICOM series).
    inputs = sorted(p for p in directory.iterdir() if p.is_dir() or _is_nifti(p))
    if not inputs:
        fail_usage(f"no NIfTI files or DICOM series directories found in {image_dir}")
    return inputs
