# Project Guidelines

## Overview

Standalone ONNX inference pipeline for **nnDetection** 3D medical object detection (RetinaUNet).
Runs sliding-window inference on NIfTI CT volumes without PyTorch or nnDetection dependencies.
This Python codebase is the **functional spec for a future C++ port** (ITK + ONNX Runtime / TensorRT).

## Code Style

- **Line length:** 120 chars (enforced by ruff)
- **Linting:** ruff with rules E, F, W, I enabled; E501 ignored. `nndet_onnx/tools/` is excluded from linting
- **Type hints:** Full `typing` annotations on all function signatures (`Tuple[int, ...]`, `List[Dict[str, np.ndarray]]`, etc.)
- **Docstrings:** Concise one-liner for simple functions; multi-line with parameter descriptions, units, and axis ordering for complex logic
- **Naming:** `snake_case` for functions/variables; `UPPER_CASE` for constants; `_` prefix for private helpers

## Architecture

Single-file pipeline ([nndet_onnx/nndet_onnx_inference_sw.py](../nndet_onnx/nndet_onnx_inference_sw.py), ~1195 lines) with 5 sequential stages:

1. **Preprocessing** — Load NIfTI → Float32 → Resample → Clip → Z-score normalize
2. **Sliding Window** — Compute patch positions with configurable overlap; batched extraction
3. **Inference** — ONNX Runtime session (4 backends: `cpu`, `cuda`, `openvino`, `trt`)
4. **Post-processing** — Score filter → Size filter → Per-patch NMS (Gaussian-weighted) → Global NMS → Restore original scores
5. **Export** — Mask (NIfTI), JSON, CSV, optional PKL

Helper tools in `nndet_onnx/tools/` (excluded from linting, one-shot scripts).

## Axes Convention — CRITICAL

This is the single most important convention and historical source of bugs.

- **nnDet box format:** 6 indices — `(dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)` where `dim0=Z, dim1=Y, dim2=X`
- **Named index constants** (`D0_MIN=0, D1_MIN=1, D2_MIN=4`, etc.) — always use these, never magic numbers
- **Suffix convention:** variables use `_zyx` or `_xyz` suffix to declare axis order explicitly
- **SimpleITK returns XYZ** — flip when interfacing with nnDet's ZYX format
- `target_spacing` in plan_inference.json is **ZYX**

When writing or reviewing code that touches coordinates, **always verify axis ordering** and use the `_zyx`/`_xyz` suffix.

## Build and Test

```bash
# Install (editable, CPU + dev tools)
pip install -e ".[cpu,dev]"

# Lint
ruff check nndet_onnx/ tests/

# Test with coverage
pytest tests/ --cov=nndet_onnx --cov-report=term-missing

# Test (verbose)
pytest tests/ -v
```

CI runs lint + tests on Python 3.9 and 3.11 (Ubuntu). See [ci.yml](workflows/ci.yml).

## Conventions

- **Detection dict:** `{"boxes": ndarray(N,6), "scores": ndarray(N), "labels": ndarray(N)}` — this is the universal data structure between pipeline stages
- **Backend mutual exclusion:** `onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-openvino` are mutually exclusive pip packages — use separate conda envs
- **Pure functions preferred:** Anchor generation, filtering, NMS are all stateless pure functions
- **No silent failures:** Early validation in `main()` with `sys.exit()` + descriptive messages; `try/except ImportError` for optional backends
- **TensorRT requires shape inference:** Models must be preprocessed with `nndet_onnx_shape_inference.py` before using TRT backend
- **TRT engine cache is GPU-specific:** Tied to GPU architecture (e.g. sm86) + TRT version; auto-regenerated if missing

## Testing

- Tests use synthetic data — no GPU or real model files required
- Fixtures: `_make_plan()` for config, `_make_nifti()` for images, `sample_detection()` for boxes
- ORT sessions are mocked in integration tests
- `@pytest.mark.parametrize` for axis/threshold/spacing variations
- When adding new functionality, add corresponding tests that verify axis ordering

## Key References

- [README.md](../README.md) — Full CLI reference, installation, benchmarks
- [NOTES_REVIEW_SW.md](../nndet_onnx/notes/NOTES_REVIEW_SW.md) — Development history, design decisions, known issues
