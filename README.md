# ONNX Inference Pipeline for nnDetection & nnUNet

![Python](https://img.shields.io/badge/python-≥3.10-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)
![Tests](https://img.shields.io/badge/tests-120%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)
![Linting](https://img.shields.io/badge/linting-ruff-purple)

Standalone ONNX inference pipeline for **nnDetection** (3D medical object detection) and **nnUNet** (3D medical image segmentation).
Runs RetinaUNet 3D and U-Net-based models using sliding window on full volumes, without any dependency on nnDetection, nnUNet, or PyTorch.

> **Goal**: This Python project initially serves as the functional specification for a **C++ port** using ITK + ONNX Runtime, and TensorRT. It could be used as is as well because it is self-contained.

---

## Citation

If you use this package in your product, research, or publications, please cite or refer to it as:

> Escobar, Thibault (2026). **nnInfe: A standalone ONNX inference pipeline optimized for nnDetection and nnUNet**. GitHub.

Suggested BibTeX entry:

```bibtex
@software{escobar2026nninfe,
  title = {nninfe: A standalone ONNX inference pipeline for nnDetection and nnUNet},
  author = {Escobar, Thibault},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Thibescobar/nnInfe},
  license = {Apache-2.0}
}
```

Please also cite the original nnDetection, nnUNet, ONNX Runtime, and other upstream tools when applicable.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Pipeline Architecture](#pipeline-architecture)
- [Axis Conventions](#axis-conventions)
- [Input / Output Formats](#input--output-formats)
- [Inference Backends](#inference-backends)
- [TensorRT Engine Cache](#tensorrt-engine-cache)
- [Preparation Tools](#preparation-tools)
- [Benchmarks](#benchmarks)
- [Limitations & Known Issues](#limitations--known-issues)

---

## Project Structure

```
nninfe/
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Package config, dependencies, ruff & pytest settings
├── LICENSE                            # Apache 2.0
├── README.md                          # This file
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI (lint + test, Python 3.10 & 3.11)
├── nninfe/
│   ├── __init__.py
│   ├── infer_detection.py     # Detection CLI
│   ├── infer_segmentation.py    # Segmentation CLI
│   ├── common/                        # Shared preprocessing, sliding-window, I/O, session
│   ├── detection/                     # Detection-specific anchors, post-processing, export
│   ├── segmentation/                  # Segmentation-specific plan, reconstruction
│   ├── tools/
│   │   ├── pkl_to_json.py       # Convert plan_inference.pkl → JSON (one-shot)
│   │   └── onnx_shape_inference.py  # ONNX shape inference for TRT (one-shot)
│   └── data/                          # ⚠ NOT TRACKED IN GIT — see below
└── tests/
    ├── test_anchors.py                # Detection anchor generation
    ├── test_cli.py                    # Common CLI validation helpers
    ├── test_export.py                 # Detection export and mask helpers
    ├── test_export_scaling.py         # Detection coordinate scaling
    ├── test_integration.py            # Detection end-to-end with mocked session
    ├── test_postprocessing.py         # Detection post-processing
    ├── test_preprocessing.py          # Shared preprocessing helpers
    ├── test_segmentation.py           # Segmentation plan + reconstruction + export
    ├── test_session.py                # Session creation and inference helpers
    └── test_sliding_window.py         # Shared sliding-window logic
```

### `data/` folder (external, not versioned)

All file paths are passed via CLI arguments (`--model-path`, `--plan-path`, `--image-path`, etc.), so **you can store your model and images anywhere on your system**. The `data/` folder inside `nninfe/` is simply a convenience location used during development and is excluded from the git repository.

Required files to run inference:
- An ONNX model file (`.onnx`) — passed via `--model-path`.
- An inference config file (`.json`) — passed via `--plan-path` (e.g.`plan_inference.json` for detection, `plans.json` for segmentation).

Optional / auto-generated:
- `trt_engine_cache_fp16/` — TensorRT compiled engines, created automatically next to the model on first TRT run. Specific to GPU architecture (e.g. sm86 for RTX 3070), regenerated if missing.

---

## Requirements

- Python ≥ 3.10
- numpy
- SimpleITK
- onnxruntime (variant depends on backend, see [Installation](#installation))

Optional (for `--nms-backend nndet`):
- nnDetection framework (nndet)
- torch, torchvision

---

## Installation

### Platform support

| Backend | Windows | Linux |
|---------|---------|-------|
| `cpu` | ✅ | ✅ |
| `openvino` |  | ✅ |
| `cuda` | ✅ | ✅ |
| `trt` | ✅ | ✅ |

> **Important:** `onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-openvino` are **mutually exclusive** pip packages and they all install to the same `onnxruntime` namespace. Installing one silently overwrites the other.

> **Support on Windows:** Only `openvino` remains untested on Windows.

### Base setup

Each backend lives in its own conda environment, and they all follow the same three steps — create the env, activate it, install the package. Only the pip extra differs:

```bash
conda create -n <env-name> python=3.10
conda activate <env-name>
pip install -e "<extra>"
```

| Environment | Backends | `<extra>` |
|-------------|----------|-----------|
| `nnInfe` (all platforms) | `cpu` | `.[cpu]` |
| `nnInfe-ov` (Linux) | `cpu`, `openvino` | `.[openvino]` |
| `nnInfe-trt` (Linux & Windows) | `cpu`, `cuda`, `trt` | `.[gpu]` |


> **GPU automatic linkage:** The `.[gpu]` extra (installed by the base setup above) is the **unified CUDA 12 stack** used identically on Linux and Windows: `onnxruntime-gpu`, the CUDA 12 / cuDNN 9 runtime wheels, and a compatible `tensorrt-cu12` (pinned `<11` — rationale in [pyproject.toml](pyproject.toml)). No `LD_LIBRARY_PATH` and no conda activation script are needed: `create_session()` makes these libraries discoverable in-process at runtime (see `_preload_gpu_libraries` in `nninfe/common/session.py`) by calling ONNX Runtime's cross-platform `preload_dlls()` for CUDA + cuDNN and placing `tensorrt_libs` on the native loader search path (prepended to `PATH` on Windows, preloaded with RUNPATH resolution on Linux). Both the `cuda` and `trt` backends have been verified on Linux and Windows. Check the `ONNX Runtime providers` line printed at session creation.

> **GPU Linux fallback (rarely needed):** on a hardened or non-standard loader configuration where the wheel's RUNPATH is ignored, the arch-specific `libnvinfer_builder_resource_*.so` may not be found and the TensorRT execution provider falls back to CPU. If that happens, add the wheel's `tensorrt_libs` to `LD_LIBRARY_PATH`:
> ```bash
> export LD_LIBRARY_PATH="$(python -c 'import os,sysconfig;print(os.path.join(sysconfig.get_paths()["purelib"],"tensorrt_libs"))'):$LD_LIBRARY_PATH"
> ```

> **Why Python ≥ 3.10?** The GPU backends use the CUDA 12 wheels, which require `onnxruntime-gpu` ≥ 1.20 — and ONNX Runtime **dropped Python 3.9 at v1.20** (3.9 caps at ORT 1.19.2). The project therefore requires Python ≥ 3.10 (`requires-python = ">=3.10"`). `--nms-backend nndet` could need some extra work to be installed with Python > 3.9.

> **Tip:** If you want a single unified environment with all backends, you can [build ONNX Runtime from source](https://onnxruntime.ai/docs/build/) with multiple execution providers enabled (e.g. `--use_cuda --use_tensorrt --use_openvino`).

### Docker images

Three `python:3.10-slim`-based Dockerfiles build backend-specific images:

| File | Backends | Install |
|------|----------|---------|
| `Dockerfile` | `cpu` | `.[cpu]` |
| `Dockerfile.ov` | `cpu`, `openvino` | `.[openvino]` |
| `Dockerfile.trt` | `cpu`, `cuda`, `trt` | `.[gpu]`, multi-stage + slimmed |

Example:
```bash
docker build -f Dockerfile.trt -t nninfe-trt .
docker run --rm --gpus all nninfe-trt nninfe-det --backend trt --trt-fp16 \
  --model-path /data/model_onnx_shaped.onnx --plan-path /data/plan_inference.json \
  --image-path /data/img.nii.gz --output-dir /data/out
```

**GPU image size.** `Dockerfile.trt` keeps the current versions (onnxruntime-gpu 1.23.x, TensorRT 10.16) but is slimmed in a multi-stage build: it keeps only TensorRT's **PTX/JIT builder resource** — dropping the per-arch `sm*` and Windows `win_*` resources (TensorRT JIT-compiles the engine for whatever GPU it runs on) — then strips debug symbols and drops headers, static libs and bytecode. Result: **~8 GB** (from ~17 GB unslimmed), still portable across NVIDIA architectures, with both `cuda` and `trt` working.

> **Opt-in `--build-arg SLIM_CUDNN=1` (→ ~6.5 GB).** Additionally drops cuDNN's precompiled + advanced kernels. In the `trt` path convolutions run *inside* TensorRT, so cuDNN is unused — **but** the `cuda` backend routes convolutions through cuDNN and would break on a conv model. Use it **only** for TensorRT-only deployments, and validate with your own model first.

---

## Quick Start

### Prepare model files (if needed)

Using tools, convert the inference plan from pickle format to JSON and add intermediate shapes to the ONNX model (often required for TRT backend).


### Run inference

**Detection (nnDetection):**

Single image with TRT FP16 (fastest):

```bash
conda activate nnInfe-trt
nninfe-det \
  --model-path /path/to/model_onnx(_shaped).onnx \
  --plan-path /path/to/plan_inference.json \
  --image-path /path/to/image.nii.gz \
  --output-dir /path/to/output \
  --backend trt --trt-fp16
```

Batch mode (all NIfTI in a directory):

```bash
nninfe-det \
  --model-path /path/to/model_onnx(_shaped).onnx \
  --plan-path /path/to/plan_inference.json \
  --image-dir /path/to/images/ \
  --output-dir /path/to/output \
  --backend trt --trt-fp16
```

CPU only (no GPU required):

```bash
conda activate nnInfe
nninfe-det \
  --model-path /path/to/model_onnx.onnx \
  --plan-path /path/to/plan_inference.json \
  --image-path /path/to/image.nii.gz \
  --output-dir /path/to/output \
  --backend cpu
```

**Segmentation (nnUNet):**

Single image with TRT FP16:

> Only this example, other execution capabilities analogous to detection.

```bash
conda activate nnInfe-trt
nninfe-seg \
  --model-path /path/to/model_onnx_shaped.onnx \
  --plan-path /path/to/plans.json \
  --configuration <config_value> \
  --image-path /path/to/image.nii.gz \
  --output-dir /path/to/output \
  --backend trt --trt-fp16
```

---

## CLI Reference

### Required

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to the ONNX model (`.onnx`). Prefer using `model_onnx_shaped.onnx` for TRT. |
| `--plan-path` | Path to `plan_inference.json` (detection) or `plans.json` (configuration file). |

### Image input (one required, mutually exclusive)

| Argument | Description |
|----------|-------------|
| `--image-path` | Path to a single NIfTI image (`.nii` or `.nii.gz`). |
| `--image-dir` | Path to a directory of NIfTI images (batch mode). |

### Output

| Argument | Description |
|----------|-------------|
| `--output-dir` | Output directory. |
| `--export-pkl` | Also export `{name}_boxes.pkl` for detection (nnDetection-compatible, for validation with `nndet_boxes2nii`). |

### Sliding Window

| Argument | Default | Description |
|----------|---------|-------------|
| `--overlap` | `0.5` | Minimum overlap ratio between adjacent patches (0 = no overlap, 0.5 = 50%, <1.0). Actual overlap may be slightly higher to fit the image boundaries. |
| `--pad-value` | `0.0` | Padding value to use when original image is smaller than the patch size. Can be a number or `min` to use the minimum value of the image minus 1. |

### Post-processing for detection

| Argument | Default | Description |
|----------|---------|-------------|
| `--score-thresh` | `0.5` | Minimum detection score. |
| `--min-size-mm` | `2.0` | Minimum box dimension in mm (filters tiny detections). |
| `--iou-threshold` | from plan | NMS IoU threshold. Default read from `plan_inference.json` (`model_iou`). |
| `--nms-backend` | `numpy` | NMS implementation: `numpy` (no extra deps) or `nndet` (requires nnDetection). |
| `--no-global-nms` | off | Disable the final global NMS after merging all patches. |

### Runtime

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `cpu` | Inference backend: `cpu`, `openvino`, `cuda`, `trt`. |
| `--trt-fp16` | off | Enable FP16 inference for TensorRT. |
| `--build-engine-only` | off | Build TRT engine cache and exit (no image/output needed). |

### `nninfe-seg configuration` 

| Argument | Description |
|----------|-------------|
| `--configuration` | Plan configuration name (default: `3d_fullres`). |


## Pipeline Architecture

### Detection Pipeline Architecture (nnDetection)

The pipeline processes each detection image through 5 steps:

```
[1/5] Anchor generation
  └── Compute the anchors per patch from plan config (strides, decoder levels, anchor sizes)

[2/5] Preprocessing
  ├── Resample to target spacing (ZYX from plan), mapping axes through transpose_forward if provided
  ├── Clip intensity (percentile 0.5 – 99.5 from plan)
  └── Normalize (z-score with mean/std from plan)

[3/5] Sliding window
  ├── Automatic padding (with minimum pixel value - 1) if the resampled image is smaller than patch size
  ├── Compute uniform patch positions with overlap ≥ requested
  └── Extract patches (size read from plan_inference.json, e.g. [64, 96, 96] ZYX)

[4/5] Batched inference
  ├── Group patches into batches (batch size read from ONNX model input shape)
  ├── Pad last batch by repeating final patch
  ├── Run ONNX inference
  ├── Gaussian weighting (for NMS priority only, original scores preserved)
  ├── Per-patch post-processing: score filter → size filter → NMS
  └── Translate boxes to global image coordinates

[5/5] Merge + export
  ├── Merge all patch detections
  ├── Global NMS (optional, enabled by default)
  ├── Restore original (unweighted) scores
  └── Export: mask NIfTI + JSON + CSV (+ optional PKL)
```

### Segmentation Pipeline Architecture (nnUNet)

The pipeline processes each segmentation image through 3 main steps:

```
[1/3] Preprocessing
  ├── Resample to target spacing (ZYX from plan), mapping axes through transpose_forward if provided
  ├── Flip axes to model expected orientation (if applicable)
  └── Pad volume (with minimum pixel value - 1) if smaller than patch size

[2/3] Sliding Window Inference
  ├── Predict patches in batches
  └── Accumulate output logits into a global volume using a 3D gaussian importance map

[3/3] Reconstruction & Export
  ├── Crop padded boundaries to restore original shape
  ├── Compute argmax across class probabilities to generate the final mask
  └── Resample mask back to original reference geometry and export
```

### Key design points

- **Gaussian weighting**: Applied to scores only for NMS merge priority. Original model scores are restored before export. This matches nnDetection's behavior.
- **Anchors**: Generated purely from `plan_inference.json` (strides, decoder levels, anchor sizes). No dependency on nnDetection code.
- **Batch padding**: Incomplete last batch padded by repeating the final patch. Only real patches are processed in post-processing.
- **Session reuse**: In batch mode (`--image-dir`), the ONNX session and anchors are created once and shared across all images.

---

## Axis Conventions

> This is the single tricky section (for avoiding bugs during the C++ port for instance).

### nnDetection internal convention
- **dim0 = Z** (axial slices), **dim1 = Y** (anterior-posterior), **dim2 = X** (left-right)
- This applies to: patch_size, target_spacing, box coordinates, anchors, feature maps

### Box format (6 values)
```
(dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)
   Z_min     Y_min     Z_max     Y_max     X_min     X_max
```
Named constants in code: `D0_MIN=0, D1_MIN=1, D0_MAX=2, D1_MAX=3, D2_MIN=4, D2_MAX=5`

### SimpleITK convention
- Spacing, origin, size are returned in **XYZ** order
- `GetArrayFromImage()` returns numpy array in **ZYX** order

### Mapping rules
| Context | Order |
|---------|-------|
| `plan_inference.json` → patch_size, target_spacing | ZYX |
| SimpleITK `.GetSpacing()`, `.GetOrigin()`, `.GetSize()` | XYZ |
| numpy volume from `sitk.GetArrayFromImage()` | ZYX |
| Box coordinates (model output) | Z, Y, Z, Y, X, X |
| `spacing_xyz[0]` → X, `spacing_xyz[2]` → Z | XYZ |

---

## Input / Output Formats

### Input: `plan_inference.json` (Detection)

Converted from nnDetection's `plan_inference.pkl` using `tools/pkl_to_json.py`.

```json
{
  "patch_size": [64, 96, 96],
  "target_spacing": [1.25, 0.7617, 0.7617],
  "anchors": {
    "width": [[4,6,8], [8,12,16], [16,24,32], [16,24,32]],
    "height": [[7,9,11], [14,18,22], [28,36,44], [56,72,88]],
    "depth": [[7,9,12], [14,18,24], [28,36,48], [56,72,96]]
  },
  "architecture": {
    "strides": [[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
    "decoder_levels": [0, 1, 2, 3]
  },
  "intensity_properties": {
    "percentile_00_5": -1024.0,
    "percentile_99_5": 3071.0,
    "mean": -158.58,
    "std": 440.20
  },
  "inference_plan": {
    "model_iou": 0.1,
    "model_score_thresh": 0.5,
    "model_topk": 300,
    "model_detections_per_image": 100,
    "remove_small_boxes": 2.0
  }
}
```

All ZYX-ordered. `anchors.width` ↔ dim0 (Z), `height` ↔ dim1 (Y), `depth` ↔ dim2 (X).

### Input: `plans.json` (Segmentation)

For segmentation tasks, supply the raw `plans.json` converted from the `plans.pkl` outputted by nnUNet natively at the training step. Selecting the correct inner config is done by supplying `--configuration` (defaults to `3d_fullres`) during inference. It handles spacing, patch sizes, and required axes flips naturally.

### Output: `{name}_boxes.json` (Detection)

nnDetection-compatible format. Boxes are scaled to the **original image** voxel space.

```json
{
  "pred_boxes": [[z0, y0, z1, y1, x0, x1], ...],
  "pred_scores": [0.95, 0.87, ...],
  "pred_labels": [0, 0, ...],
  "restore": true,
  "original_size_of_raw_data": [Z, Y, X],
  "itk_origin": [ox, oy, oz],
  "itk_spacing": [sx, sy, sz],
  "itk_direction": [1,0,0, 0,1,0, 0,0,1]
}
```

### Output: `{name}_boxes.csv`

One row per detection, with both voxel and world coordinates:

```
image_name, detection_id, label, score,
z_min, y_min, x_min, z_max, y_max, x_max,
center_x_mm, center_y_mm, center_z_mm,
size_x_mm, size_y_mm, size_z_mm, volume_mm3
```

### Output: `{name}_boxes.pkl` (optional, `--export-pkl`)

Pickle dict with numpy arrays, compatible with nnDetection CLI tools (`nndet_boxes2nii`).
Same structure as the JSON but with `np.float32` / `np.int64` arrays instead of lists.

### Output: `{name}_mask.nii.gz` (Detection)

NIfTI label map where each detection is a connected component with a unique integer label.
Resampled to the **original image geometry** (spacing, origin, direction, size) using nearest-neighbor interpolation, so it can be directly overlaid on the source image in any compatible viewer.

>This mask corresponds to boxes, not to segmentation contours.

### Output: `{name}_seg.nii.gz` (Segmentation)

A voxel-level label map where integer values represent semantic classes as defined in standard nnUNet exports. Re-sampled natively back to the input reference image's spacing and geometry.

---

## Inference Backends

| Backend | Provider chain | Model file | Conda env |
|---------|---------------|------------|-----------|
| `cpu` | CPU | `model_onnx.onnx` | either |
| `openvino` | OpenVINO → CPU | `model_onnx.onnx` | nnInfe-ov |
| `cuda` | CUDA → CPU | `model_onnx.onnx` | nnInfe-trt |
| `trt` | TensorRT → CUDA → CPU | `model_onnx_shaped.onnx` | nnInfe-trt |

- Use `--trt-fp16` for FP16 precision (recommended, fonctionnaly same results, ×2+ speedup vs TRT FP32).
- Use `--build-engine-only` to pre-compile engines without running inference.

---

## TensorRT Engine Cache

ONNX Runtime compiles the model into a TRT engine or multiple TRT sub-engines. This takes ~2-5 minutes. The engines are cached in `trt_engine_cache_{precision}/` next to the model.

```
data/model/trt_engine_cache_fp16/
├── *.engine              # Compiled TRT sub-graphs (~33 files, 129 MB total)
└── *.timing              # Autotuning results
```

**Important**: The cache is tied to:
- GPU architecture (e.g., sm86 for RTX 3070)
- TensorRT version (e.g., 10.3)
- ONNX model, hash-based

Changing any of these requires rebuilding the cache (lazy or `--build-engine-only`).

> **Tip:** The `.engine` files are large and machine-specific, but the `.timing` file is lightweight and reusable. Keeping only the `.timing` file allows TensorRT to skip the autotuning phase during rebuild, significantly reducing compilation time.

>**Why multiple engines?** ONNX Runtime splits the graph into TRT-supported sub-graphs + CUDA fallback for unsupported ops (ScatterND, NonZero). A single unified engine is preferred if possible (speed, size, maintainability).

---

## Benchmarks

Done for detection. Measured on a single CT scan (294 patches, 74 batches, overlap 0.5, score-thresh 0.5) with an RTX 3070.

| Backend | Inference | Total | s/batch | Speedup vs CPU | Detections |
|---------|-----------|-------|---------|----------------|------------|
| `cpu` | 218s | 220s | 2.95 | ×1 | 26 |
| `openvino` | 113s | 115s | 1.52 | ×1.9 | 26 |
| `cuda` | 14s | 16s | 0.19 | ×15.6 | 26 |
| `trt --trt-fp16` | 6s | 8s | 0.08 | **×36.9** | 26 |

All backends produce **26 detections** — results are consistent across backends (negligible numerical variations).

---

## Limitations & Known Issues

- **Multiple classes for detection**: Detection currently exposes class output natively mapped (label 0, etc). Multi-class might require specific per-class NMS tracking in nnDetection pipelines if custom configuration differs.
- **No DICOM handling**: Input must be NIfTI (`.nii` or `.nii.gz`). DICOM→NIfTI conversion should be done upstream. Patched soon.
- **Mask is bounding-box based for detection**: While the native nnDetection framework give the possibility to output segmentation contours for some detected objects (not all), the output mask for the present detection pipeline (`_mask.nii.gz`) fills bounding boxes. Pixel-level contours is reserved for the segmentation pipeline for the moment. Could be patched if there are needs.
- **Single fold**: Uses one fold only. Multi-fold ensemble was intentionally deferred for industrialization simplicity and speed across both detection and segmentation.
- **Structured logging**: Currently all output is `print()`. Production should use Python `logging` with levels (DEBUG/INFO/WARNING).
- **Pipeline versioning**: Exports (JSON/CSV/PKL) should include a pipeline version number for traceability (medical device regulation).
- **ONNX Runtime error handling**: No try/except around session creation or inference. GPU OOM, driver mismatch, or engine incompatibility will produce raw Python exceptions.

---

## Development

```bash
# CPU (all platforms)
pip install -e ".[cpu,dev]"

# OpenVINO (Linux only)
pip install -e ".[openvino,dev]"

# GPU / cuda + trt (Linux & Windows)
pip install -e ".[gpu,dev]"
```

```bash
# Lint
ruff check nninfe/ tests/

# Run tests
python -m pytest tests/ -v
```
