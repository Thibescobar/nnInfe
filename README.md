# nnDet ONNX Inference Pipeline

![Python](https://img.shields.io/badge/python-≥3.9-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)
![Tests](https://img.shields.io/badge/tests-99%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Linting](https://img.shields.io/badge/linting-ruff-purple)

Standalone inference pipeline for **nnDetection** (3D medical object detection) exported to ONNX.
Runs a RetinaUNet 3D model with sliding window on volumes, without any dependency on nnDetection or PyTorch.

> **Goal**: This Python project serves as the functional specification for a future **C++ port** using ITK + ONNX Runtime / TensorRT. It could be used as is as well because self-contained.

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
mvpDet/
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Package config, dependencies, ruff & pytest settings
├── LICENSE                            # Apache 2.0
├── README.md                          # This file
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI (lint + test, Python 3.9 & 3.11)
├── nndet_onnx/
│   ├── __init__.py
│   ├── nndet_onnx_inference_sw.py     # Main inference script (~1195 lines)
│   ├── tools/
│   │   ├── nndet_pkl_to_json.py       # Convert plan_inference.pkl → JSON (one-shot)
│   │   └── nndet_onnx_shape_inference.py  # ONNX shape inference for TRT (one-shot)
│   ├── notes/
│   │   └── NOTES_REVIEW_SW.md         # Development notes & history
│   └── data/                          # ⚠ NOT IN GIT — see below
└── tests/
    ├── test_anchors.py                # Anchor generation (13 tests)
    ├── test_sliding_window.py         # Sliding window & patch extraction (8 tests)
    ├── test_postprocessing.py         # NMS, filtering, merging (30 tests)
    ├── test_export.py                 # Mask, resampling, export formats (14 tests)
    ├── test_export_scaling.py         # Coordinate scaling in exports (8 tests)
    ├── test_preprocessing.py          # Resample, clip, normalize (9 tests)
    ├── test_session.py                # Session creation, inference, NMS backends (9 tests)
    └── test_integration.py            # End-to-end with mocked session (8 tests)
```

### `data/` folder (external, not versioned)

All file paths are passed via CLI arguments (`--model-path`, `--plan-path`, `--image-path`, etc.), so **you can store your model and images anywhere on your system**. The `data/` folder inside `nndet_onnx/` is simply a convenience location used during development and is excluded from the git repository.

Required files to run inference:
- An ONNX model file (`.onnx`) — passed via `--model-path`
- An inference config (`plan_inference.json`) — passed via `--plan-path`

Optional / auto-generated:
- `trt_engine_cache_fp16/` — TensorRT compiled engines, created automatically next to the model on first TRT run. Specific to GPU architecture (e.g. sm86 for RTX 3070), regenerated if missing.

---

## Requirements

- Python ≥ 3.9
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
| `openvino` | ❌ | ✅ |
| `cuda` | ❌ | ✅ |
| `trt` | ❌ | ✅ |

> **Important:** `onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-openvino` are **mutually exclusive** pip packages — they all install to the same `onnxruntime` namespace. Installing one silently overwrites the other.

### CPU environment (all platforms)

```bash
conda create -n nnDetPy39 python=3.9
conda activate nnDetPy39
pip install -e ".[cpu]"
```

### OpenVINO environment (Linux only)

```bash
conda create -n nnDetPy39-ov python=3.9
conda activate nnDetPy39-ov
pip install -e ".[openvino]"
```

Supports backends: `cpu`, `openvino`.

### GPU environment (Linux only)

```bash
conda create -n nnDetPy39-trt python=3.9
conda activate nnDetPy39-trt
pip install -e .
pip install onnxruntime-gpu \
  --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
conda install cudnn=8
pip install tensorrt==10.3.0
pip install nvidia-cuda-runtime-cu12==12.2.2 nvidia-cublas-cu12==12.2.5.6
```

> `onnxruntime-gpu` is installed manually because the CUDA version depends on your driver. Always install it **after** `pip install -e .`.

Then configure `LD_LIBRARY_PATH` (create once):

```bash
CONDA_PREFIX=$CONDA_PREFIX
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SITE/tensorrt_libs:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
```

Supports backends: `cpu`, `cuda`, `trt`.

### Validated GPU Configuration

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3070 (8 GB, sm86) |
| Driver | 535.309 |
| CUDA toolkit (system) | 11.7 |
| TensorRT | 10.3 |
| Python | 3.9 |

> **Tip:** If you want a single unified environment with all backends, you can [build ONNX Runtime from source](https://onnxruntime.ai/docs/build/) with multiple execution providers enabled (e.g. `--use_cuda --use_tensorrt --use_openvino`).

---

## Quick Start

### 1. Prepare model files (if needed)

Convert the inference plan from nnDetection's pickle format to JSON:

```bash
python nndet_onnx/tools/nndet_pkl_to_json.py \
  --pkl /path/to/plan_inference.pkl \
  --output /path/to/plan_inference.json
```

Add intermediate shapes to the ONNX model (required for TRT backend):

```bash
python nndet_onnx/tools/nndet_onnx_shape_inference.py \
  --input /path/to/model_onnx.onnx \
  --output /path/to/model_onnx_shaped.onnx
```

### 2. Run inference

Single image with TRT FP16 (fastest):

```bash
conda activate nnDetPy39-trt
nndet-infer \
  --model-path /path/to/model_onnx_shaped.onnx \
  --plan-path /path/to/plan_inference.json \
  --image-path /path/to/image.nii.gz \
  --output-dir /path/to/output \
  --backend trt --trt-fp16
```

Batch mode (all NIfTI in a directory):

```bash
nndet-infer \
  --model-path /path/to/model_onnx_shaped.onnx \
  --plan-path /path/to/plan_inference.json \
  --image-dir /path/to/images/ \
  --output-dir /path/to/output \
  --backend trt --trt-fp16
```

CPU only (no GPU required):

```bash
conda activate nnDetPy39
nndet-infer \
  --model-path /path/to/model_onnx.onnx \
  --plan-path /path/to/plan_inference.json \
  --image-path /path/to/image.nii.gz \
  --output-dir /path/to/output \
  --backend cpu
```

---

## CLI Reference

### Required

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to the ONNX model (`.onnx`). Use `model_onnx_shaped.onnx` for TRT. |
| `--plan-path` | Path to `plan_inference.json` (inference configuration). |

### Image input (one required, mutually exclusive)

| Argument | Description |
|----------|-------------|
| `--image-path` | Path to a single NIfTI image (`.nii` or `.nii.gz`). |
| `--image-dir` | Path to a directory of NIfTI images (batch mode). |

### Output

| Argument | Description |
|----------|-------------|
| `--output-dir` | Output directory. Per image: `{name}_mask.nii.gz`, `{name}_boxes.json`, `{name}_boxes.csv`. |
| `--export-pkl` | Also export `{name}_boxes.pkl` (nnDetection-compatible, for validation with `nndet_boxes2nii`). |

### Sliding Window

| Argument | Default | Description |
|----------|---------|-------------|
| `--overlap` | `0.5` | Minimum overlap ratio between adjacent patches (0 = no overlap, 0.5 = 50%). Actual overlap may be slightly higher due to rounding. |

### Post-processing

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

---

## Pipeline Architecture

The pipeline processes each image through 5 steps:

```
[1/5] Anchor generation
  └── Compute 284,796 anchors per patch from plan config (strides, decoder levels, anchor sizes)

[2/5] Preprocessing
  ├── Resample to target spacing (ZYX from plan)
  ├── Clip intensity (percentile 0.5 – 99.5 from plan)
  └── Normalize (z-score with mean/std from plan)

[3/5] Sliding window
  ├── Compute uniform patch positions with overlap ≥ requested
  └── Extract patches (size read from plan_inference.json, e.g. [64, 96, 96] ZYX)

[4/5] Batched inference
  ├── Group patches into batches (batch size read from ONNX model input shape)
  ├── Pad last batch by repeating final patch
  ├── Run ONNX inference → 12 output tensors (4×boxes, 4×scores, 4×labels)
  ├── Per-patch post-processing: score filter → size filter → NMS
  ├── Gaussian weighting (for NMS priority only, original scores preserved)
  └── Translate boxes to global image coordinates

[5/5] Merge + export
  ├── Merge all patch detections
  ├── Global NMS (optional, enabled by default)
  ├── Restore original (unweighted) scores
  └── Export: mask NIfTI + JSON + CSV (+ optional PKL)
```

### Key design points

- **Gaussian weighting**: Applied to scores only for NMS merge priority. Original model scores are restored before export. This matches nnDetection's behavior.
- **Anchors**: Generated purely from `plan_inference.json` (strides, decoder levels, anchor sizes). No dependency on nnDetection code.
- **Batch padding**: Incomplete last batch padded by repeating the final patch. Only real patches are processed in post-processing.
- **Session reuse**: In batch mode (`--image-dir`), the ONNX session and anchors are created once and shared across all images.

---

## Axis Conventions

> This is the single most important section for avoiding bugs during the C++ port.

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

### Input: `plan_inference.json`

Converted from nnDetection's `plan_inference.pkl` using `tools/nndet_pkl_to_json.py`.

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

### Output: `{name}_boxes.json`

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

### Output: `{name}_mask.nii.gz`

NIfTI label map where each detection is a connected component with a unique integer label.
Resampled to the **original image geometry** (spacing, origin, direction, size) using nearest-neighbor interpolation, so it can be directly overlaid on the source image in any compatible viewer.

---

## Inference Backends

| Backend | Provider chain | Model file | Conda env |
|---------|---------------|------------|-----------|
| `cpu` | CPU | `model_onnx.onnx` | either |
| `openvino` | OpenVINO → CPU | `model_onnx.onnx` | nnDetPy39-ov (Linux only) |
| `cuda` | CUDA → CPU | `model_onnx.onnx` | nnDetPy39-trt |
| `trt` | TensorRT → CUDA → CPU | `model_onnx_shaped.onnx` | nnDetPy39-trt |

- **TRT requires** the shaped model (`onnx.shape_inference` applied). See [Preparation Tools](#preparation-tools).
- Use `--trt-fp16` for FP16 precision (recommended, same results, ×2 speedup vs TRT FP32).
- Use `--build-engine-only` to pre-compile engines without running inference.

---

## TensorRT Engine Cache

On first run with `--backend trt`, ONNX Runtime compiles the model into multiple TRT sub-engines. This takes ~2 minutes. The engines are cached in `trt_engine_cache_{precision}/` next to the model.

```
data/model/trt_engine_cache_fp16/
├── *.engine              # Compiled TRT sub-graphs (~33 files, 129 MB total)
└── *.timing              # Autotuning results
```

**Important**: The cache is tied to:
- GPU architecture (e.g., sm86 for RTX 3070)
- TensorRT version (10.3)
- ONNX model (hash-based)

Changing any of these requires rebuilding the cache (`--build-engine-only`).

> **Tip:** The `.engine` files are large and machine-specific, but the `.timing` file is lightweight and reusable. Keeping only the `.timing` file allows TensorRT to skip the autotuning phase during rebuild, significantly reducing compilation time.

**Why multiple engines?** ONNX Runtime splits the graph into TRT-supported sub-graphs + CUDA fallback for unsupported ops (ScatterND, NonZero). For the C++ port, a single unified engine is preferred — see notes on hybrid architecture in `notes/NOTES_REVIEW_SW.md`.

---

## Preparation Tools

### Convert `plan_inference.pkl` → JSON

One-shot conversion of nnDetection's training plan to JSON (readable by C++ / nlohmann::json).

```bash
python tools/nndet_pkl_to_json.py \
  --pkl /path/to/plan_inference.pkl \
  --output data/model/plan_inference.json
```

### ONNX Shape Inference (required for TRT)

Annotates intermediate tensor shapes in the ONNX graph. Required by TensorRT.

```bash
python tools/nndet_onnx_shape_inference.py \
  --input data/model/model_onnx.onnx \
  --output data/model/model_onnx_shaped.onnx
```

Requires: `pip install onnx`

---

## Benchmarks

Measured on a single CT scan (294 patches, 74 batches, overlap 0.5, score-thresh 0.5) with an RTX 3070.

| Backend | Inference | Total | s/batch | Speedup vs CPU | Detections |
|---------|-----------|-------|---------|----------------|------------|
| `cpu` | 218s | 220s | 2.95 | ×1 | 26 |
| `openvino` | 113s | 115s | 1.52 | ×1.9 | 26 |
| `cuda` | 14s | 16s | 0.19 | ×15.6 | 26 |
| `trt --trt-fp16` | 6s | 8s | 0.08 | **×36.9** | 26 |

All backends produce **26 detections** — results are consistent across backends (negligible numerical variation).

---

## Limitations & Known Issues

- **Single class**: The current model detects one class only (label 0). Multi-class support would require per-class NMS. Patched soon.
- **Image must be ≥ patch_size after resampling**: If any resampled dimension is smaller than the patch size defined in `plan_inference.json`, the pipeline exits with an error. Patched soon.
- **No DICOM input**: Input must be NIfTI (`.nii` or `.nii.gz`). DICOM→NIfTI conversion should be done upstream. Patched soon.
- **Mask is bounding-box based**: The output mask fills detection bounding boxes, not a pixel-level segmentation. Enhanced soon.
- **Single fold**: Uses one fold only. Multi-fold ensemble was intentionally deferred for industrialization simplicity and speed.

### Deferred to post-MVP (before production)

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

# GPU (Linux only — see GPU environment above for onnxruntime-gpu setup)
pip install -e ".[dev]"
```

```bash
# Lint
ruff check nndet_onnx/ tests/

# Run tests
python -m pytest tests/ -v
```
