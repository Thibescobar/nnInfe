# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`nninfe` is a **standalone ONNX inference pipeline** for **nnDetection** (3D medical object detection) and **nnUNet** (3D segmentation). It runs RetinaUNet-3D and U-Net models via sliding window on full CT/medical volumes **without any dependency on nnDetection, nnUNet, or PyTorch** — inference is ONNX Runtime, image ops are SimpleITK, math is numpy.

A stated design goal shapes the whole codebase: it doubles as the **functional specification for a future C++ port** (ITK + ONNX Runtime + TensorRT). This is *why* the code avoids heavyweight deps, keeps logic explicit/portable, and obsesses over axis-order correctness. Keep that constraint in mind before reaching for a Python-only convenience.

## Commands

```bash
# Install (backends are mutually exclusive — see gotcha below). Add ,dev for tooling.
pip install -e ".[cpu,dev]"        # CPU, all platforms
pip install -e ".[openvino,dev]"   # OpenVINO, Linux only
pip install -e ".[gpu,dev]"        # cuda + trt, Linux & Windows

# Lint (nninfe/tools/ is excluded from ruff; line-length 120, E501 ignored)
ruff check nninfe/ tests/

# Tests (pytest auto-runs coverage via addopts in pyproject.toml)
python -m pytest tests/ -v
python -m pytest tests/test_postprocessing.py -v               # one file
python -m pytest tests/test_postprocessing.py::TestNmsNumpy -v # one class/test
```

**After running the suite, update the README badges to match** — the `tests-<N>%20passed` and `coverage-<P>%25` shields at the top of `README.md` are maintained by hand and drift silently. Take `<N>` from the pytest summary line and `<P>` from the `TOTAL … <P>%` coverage line (coverage runs automatically via `--cov` in `pyproject.toml`). Do this by default whenever tests are run, not only when asked.

Two console entry points (defined in `pyproject.toml [project.scripts]`):
- `nninfe-det` → `nninfe/infer_detection.py:main` (nnDetection)
- `nninfe-seg` → `nninfe/infer_segmentation.py:main` (nnUNet, needs `--configuration`, default `3d_fullres`)

One-shot prep tools (not part of the runtime pipeline, excluded from lint):
- `python nninfe/tools/pkl_to_json.py --pkl plan_inference.pkl` — converts nnDetection's `plan_inference.pkl` to the JSON plan the detection CLI consumes.
- `python nninfe/tools/onnx_shape_inference.py --input model.onnx` — annotates intermediate shapes → `model_onnx_shaped.onnx`, typically required for the `trt` backend.

The full CLI reference, backend matrix, benchmarks, and I/O JSON schemas live in [README.md](README.md) — consult it rather than re-deriving flags.

## Project structure

*Keep this map current when a module is added or its role changes — it's the fast orientation for a fresh session. Deliberately module-level: individual test files are not enumerated (that list churns and drifts), so it stays cheap to maintain.*

```
nninfe/
├── infer_detection.py       # `nninfe-det` entry point — detection pipeline (anchors → preprocess → sliding window → per-patch NMS → merge → export)
├── infer_segmentation.py    # `nninfe-seg` entry point — segmentation pipeline (preprocess → flip → sliding-window logit accumulation → argmax → export)
├── common/                  # shared by both pipelines
│   ├── cli.py               # --image-path / --image-dir resolution (NIfTI file or DICOM series dir)
│   ├── constants.py         # box axis-index constants D0_MIN…D2_MAX
│   ├── errors.py            # typed error taxonomy + exit codes + translate_errors / fail_usage / write_image_status
│   ├── logging_setup.py     # configure_logging(): 'nninfe' logger, INFO→stdout / WARNING+→stderr, bare format (terminal-identical to old print)
│   ├── manifest.py          # per-image {name}_manifest.json audit record (version, model/plan sha256, params, providers, duration, outcome, scope/retryable)
│   ├── io.py                # image I/O hub: read NIfTI + DICOM series; write DICOM SEG (binary|labelmap) + detection SR (TID 1500)
│   ├── preprocessing.py     # resample / clip / z-score / pad / resample-back-to-reference
│   ├── session.py           # ORT session, BACKENDS map, in-process GPU lib preload, run/parse
│   └── sliding_window.py    # patch positions (overlap ≥ requested) + patch extraction
├── detection/
│   ├── anchors.py           # anchors generated purely from the JSON plan
│   ├── postprocessing.py    # score/size filters, 3D NMS (numpy|nndet), gaussian weighting, merge
│   └── export.py            # bbox mask (connected components) / JSON / CSV / PKL, rescaled to original voxel space
├── segmentation/
│   └── pipeline.py          # nnUNet plan extraction, gaussian-importance accumulation, build_reference_mask, exports
└── tools/                   # one-shot prep utilities — NOT runtime, excluded from ruff
    ├── pkl_to_json.py            # nnDetection plan_inference.pkl → JSON
    └── onnx_shape_inference.py   # annotate intermediate shapes (needed for TRT)

tests/                       # pytest; the ONNX session is mocked (no real model). Roughly one module
                             # per area (test_postprocessing, test_segmentation, test_dicom_seg, test_io, …)
```

## Architecture

### Shared core + two thin pipelines

`nninfe/common/` holds everything both pipelines share; `detection/` and `segmentation/` add only the model-family-specific logic. Both CLIs follow the same skeleton: validate args → `create_session` once → loop images through `process_single_image` (session, anchors, and plan are created once and **reused across a batch**).

- `common/session.py` — backend→provider mapping (`BACKENDS` dict), session creation, and `run_inference`/`parse_outputs`. **Batch size is read from the ONNX input shape** (`session.get_inputs()[0].shape[0]`), not a CLI flag. Also owns `_ORT_EXCEPTIONS` (ONNX Runtime's exception classes, collected dynamically — see the error-handling gotcha below).
- `common/logging_setup.py` — `configure_logging()` sets up the `nninfe` logger. **Progress uses `logging`, not `print`, but the terminal output is byte-identical to before** (INFO→stdout / WARNING+→stderr, bare `%(message)s`, flush per record). Modules log via `logging.getLogger(__name__)`; both CLIs call `configure_logging()` at startup. A regression harness diffs captured terminal output to guarantee the "identical" property.
- `common/manifest.py` — writes a per-image `{name}_manifest.json` audit record. Run-level context (nninfe version, model & plan SHA-256, backend, providers, params) is built **once per run** (`build_run_context`, so the model is hashed a single time); per image it adds outcome/duration and, on failure, `classify_error()`'s `{type, scope, retryable}` for the orchestrator.
- `common/preprocessing.py` — `preprocess_image` chain: read → cast → `resample_image` (to plan `target_spacing`) → clip (percentiles) → z-score normalize. `resample_mask_to_reference` is the inverse used to map results back to the *original* input geometry (nearest-neighbor).
- `common/sliding_window.py` — `compute_patch_positions` distributes patches evenly so overlap is **≥** requested and the last patch ends exactly at the boundary; `extract_patch` slices.
- `common/io.py` — the pipeline's image I/O hub. Single `read_image` entry point makes **DICOM series directories and NIfTI files interchangeable** on input; the rest of the pipeline never knows which it got. Also the home for **DICOM result export**: `write_segmentation_dicom_seg` writes a mask as a DICOM SEG (`binary` default, or `labelmap` for many-class masks); `write_detection_dicom_sr` writes detections as a DICOM Structured Report (TID 1500 Measurement Report / TID 1410 planar groups, Comprehensive 3D SR). Both reference the source series, require DICOM input, and backfill missing Type-2 patient/study tags via `_backfill_source_attributes` (real/anonymized series often drop `AccessionNumber` etc., which highdicom reads without a default). Secondary Capture could join them here next. `list_dicom_series_files` returns the series file list without a pixel read.
- `common/cli.py` — `collect_image_inputs` resolves `--image-path` (single file or one DICOM dir) vs `--image-dir` (batch; each entry is one image — NIfTI file or DICOM subdir).

### Axis conventions — the #1 source of bugs

This is the single trickiest thing in the codebase and the most important to get right for the C++ port. Three coexisting orderings:

| Source | Order |
|--------|-------|
| Plan JSON (`patch_size`, `target_spacing`), numpy arrays from `sitk.GetArrayFromImage()`, feature maps, anchors | **ZYX** (dim0=Z axial, dim1=Y, dim2=X) |
| SimpleITK `.GetSpacing()`/`.GetOrigin()`/`.GetSize()` | **XYZ** |
| Detection box (6 values) | `(Z_min, Y_min, Z_max, Y_max, X_min, X_max)` |

Box indices are named constants in `common/constants.py`: `D0_MIN=0, D1_MIN=1, D0_MAX=2, D1_MAX=3, D2_MIN=4, D2_MAX=5`. **Always use these**, never bare literals — and note the non-obvious interleaving (D2/X pair is last, not `min,max,min,max,min,max` by axis). When indexing `spacing_xyz`, remember `[0]→X`, `[2]→Z`, so ZYX code reads spacing in reverse (see `filter_small_boxes` and `_rescale_boxes_to_ref`).

### Detection pipeline (`nninfe/infer_detection.py` + `detection/`)

Per-image flow: anchors → preprocess → pad → sliding window → **per-patch** post-process → translate to global coords → merge → global NMS → export.

Key design decisions (don't "fix" these — they intentionally mirror nnDetection):
- **Anchors** (`detection/anchors.py`) are generated *purely from the JSON plan* (strides, decoder levels, anchor width/height/depth), with no nnDetection code. `compute_anchors` returns them pre-tiled to batch size.
- **Gaussian weighting** (`gaussian_weight_for_boxes`) multiplies scores *only to prioritize center detections during NMS merge*. Original scores are stashed in `scores_original` and **restored before export** (`merged["scores"] = merged.pop("scores_original")`). Exported scores are always the raw model scores.
- **NMS backend** is pluggable: `numpy` (default, dependency-free, `nms_numpy`/`_iou_3d`) or `nndet` (requires torch+nndet, for validation parity).
- **Batch padding**: the last incomplete batch is padded by repeating the final patch; only the real patches (`actual_count`) go through post-processing.
- **Exports** (`detection/export.py`): `_boxes.json` (nnDetection-compatible) is **always** written — the canonical detection record — and `_boxes.pkl` whenever `--export-pkl`. `--output-format {json,nifti,dicom-sr,both}` (default `nifti`) then selects the renderings: `nifti` adds `_mask.nii.gz` (connected-component label map of **filled bounding boxes**, not contours) + `_boxes.csv`; `dicom-sr`/`both` (DICOM input only) adds `_sr.dcm` via `io.write_detection_dicom_sr`. Boxes are rescaled from resampled to original voxel space via `_rescale_boxes_to_ref` (the SR writer re-applies the same rescale internally, keyed on `current_meta`).

### Segmentation pipeline (`nninfe/infer_segmentation.py` + `segmentation/pipeline.py`)

Per-image flow: preprocess → **flip axes** → pad → sliding-window logit accumulation → argmax → crop → flip back → resample to reference → export. `build_reference_mask` produces the label map in the original input geometry once; the CLI's `--output-format` then writes it as `_seg.nii.gz` (NIfTI) and/or `_seg.dcm` (DICOM SEG, DICOM input only) — so a new output format is a new writer over that shared reference mask.

- `extract_plan_inference` normalizes nnUNet's `plans.json` (selecting the `--configuration` sub-config) into the same internal plan shape detection uses.
- `_resolve_patch_size_zyx` prefers the **model input shape** (NCDHW) over the plan's `patch_size` when the model has concrete spatial dims.
- Unlike detection's per-patch box NMS, segmentation **accumulates weighted logits** into a full-volume buffer using a `_gaussian_importance_map`, divides by accumulated weights, then `argmax` over classes → label map.
- `flip_image_axes(preprocessed, True, True, False)` (flip X and Y) is applied before inference and reversed after — required to match the model's expected orientation.

### Backends & in-process GPU library loading

`BACKENDS` maps `cpu`/`cuda`/`openvino`/`trt` to ONNX Runtime provider chains (each falls back toward CPU). The non-obvious machinery is `_preload_gpu_libraries` in `session.py`: it makes the pip-installed CUDA/cuDNN/TensorRT wheels discoverable **at runtime, in-process**, so no `LD_LIBRARY_PATH` or conda activation scripts are needed on either Linux or Windows. It calls ORT's cross-platform `preload_dlls()` for CUDA+cuDNN and puts `tensorrt_libs` on the native loader path (prepend `PATH` on Windows; `ctypes.CDLL` preload with RUNPATH resolution on Linux). It's idempotent and a no-op for cpu/openvino.

For `trt`, engines are compiled and cached in `trt_engine_cache_{fp16,fp32}/` next to the model (2–5 min first run). Use `--build-engine-only` to pre-compile without running inference.

## Conventions & gotchas

- **`onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-openvino` are mutually exclusive** — they install to the same `onnxruntime` namespace and silently overwrite each other. One backend per environment (the README documents separate conda envs: `nnInfe`, `nnInfe-ov`, `nnInfe-trt`).
- **`tensorrt-cu12` is pinned `<11`** in `pyproject.toml` because onnxruntime-gpu links nvinfer 10; a TRT 11 wheel would silently fall back to CPU. Only bump it alongside a matching onnxruntime-gpu build.
- **Python ≥ 3.10** is required (GPU CUDA-12 wheels need onnxruntime-gpu ≥ 1.20, which dropped 3.9).
- **Tests never load a real ONNX model** — the session is a `MagicMock` whose `.run` returns hand-crafted output arrays (see `tests/test_integration.py`). This mirrors `parse_outputs`' expected layout: `batch_size` box arrays, then `batch_size` score arrays, then `batch_size` label arrays. Follow this pattern for new inference tests.
- **All progress output is `print(..., flush=True)`**, not the `logging` module (a known limitation flagged for production). Match the existing `[n/N] step …` style if adding pipeline steps.
- **`nninfe/data/` is not tracked in git** — all paths are passed via CLI, so models/images can live anywhere; `data/` is just a dev convenience location.
- **Completion sentinels** (used by external orchestration): each image writes a per-image `{name}.done` on success or `{name}.failed` (carrying the error) on failure; the batch also writes a single shared `.done` when the whole run finishes (kept for backward compatibility — it now means "run complete", not per-image status). Written via `write_image_status` in `common/errors.py`.
- **Error handling / exit codes** (Phases 1–2 of the production-robustness plan done; Phase 3 = reproducibility hashes in result files + input guards, Phase 4 = verification harness, still open): `common/errors.py` defines the typed-error taxonomy (`SessionError`/`InferenceError`/`ImageIOError`/`ExportError`, all `NnInfeError`) and the stable exit-code contract (0 ok, 1 unexpected/bug, 2 usage, 3 session, 4 inference, 5 image I/O, 6 export, 7 partial batch). Three error families are kept distinct:
  - **Usage errors** (bad args/config) → `fail_usage()` prints a clean message and exits `2` (no traceback).
  - **Operational errors** (expected dependency failures) → translated to typed errors *only for the exceptions a dependency actually raises*, via `translate_errors(cls, caught, message)` (session/inference at the ORT boundary in `session.py`/`pipeline.py`; preprocess→`ImageIOError`, export→`ExportError` in the CLIs). Note: ONNX Runtime exceptions subclass `Exception` **directly** (not `RuntimeError`) with no shared base, so `session.py` collects them dynamically into `_ORT_EXCEPTIONS`.
  - **Programming bugs** (anything not in `caught`) are **never** relabeled — they propagate to Python's default exit `1` + traceback. In the batch loop a bug on one image is isolated (logged with traceback, marked `{name}.failed`) but forces `EXIT_RUNTIME`, never the benign partial code (policy (a)). Operational per-image failures → `EXIT_PARTIAL`.
  - Persisted error strings (the `{name}.failed` sentinel, the typed messages) are **PHI-sober**: a stable message only, never the raw dependency `str(exc)` (which can carry a patient-bearing path); the original cause is chained via `from exc` for the logs.
