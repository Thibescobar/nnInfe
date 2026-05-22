# nnDet ONNX Inference Pipeline

Standalone inference pipeline for **nnDetection** (3D medical object detection) exported to ONNX.
Runs a RetinaUNet 3D model with sliding window on NIfTI CT volumes, without any dependency on nnDetection or PyTorch.

> **Goal**: This Python prototype serves as the functional specification for a future **C++ port** using ITK + ONNX Runtime / TensorRT native.

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
nndet_onnx/
├── nndet_onnx_inference_sw.py        # Main inference script (~1195 lines)
├── README.md                          # This file
├── .gitignore
├── tools/
│   ├── nndet_pkl_to_json.py           # Convert plan_inference.pkl → JSON (one-shot)
│   └── nndet_onnx_shape_inference.py  # ONNX shape inference for TRT (one-shot)
├── notes/
│   └── NOTES_REVIEW_SW.md            # Development notes & history
├── tests/                             # Unit tests (TODO)
└── data/                              # ⚠ NOT IN GIT — see below
```

### `data/` folder (external, not versioned)

The `data/` folder contains large binary files (models, images, TRT cache) and is **excluded from the git repository**. It must be provided separately when setting up a new environment.

Expected structure:

```
data/
├── model/
│   ├── model_onnx.onnx               # Original ONNX export (~343 MB)
│   ├── model_onnx_shaped.onnx        # With intermediate shapes, required for TRT backend (~343 MB)
│   ├── plan_inference.json            # Inference config (patch_size, spacing, anchors, NMS threshold)
│   └── trt_engine_cache_fp16/         # TensorRT compiled engines (machine-specific, auto-generated)
│       ├── *.engine                   # One engine per subgraph (sm86, fp16)
│       ├── *.profile                  # Optimization profiles
│       └── *.timing                   # Timing cache
└── test_images/
    ├── <image_name>.nii.gz            # Single NIfTI image for --image-path
    └── <image_dir>/                   # Directory of NIfTI images for --image-dir (batch mode)
        ├── image1.nii.gz
        ├── image2.nii.gz
        └── ...
```

**Required files** to run inference:
- `data/model/model_onnx.onnx` (or `model_onnx_shaped.onnx` for TRT backend)
- `data/model/plan_inference.json`

**Auto-generated** (created at first TRT run):
- `data/model/trt_engine_cache_fp16/` — compiled engines, specific to GPU architecture (e.g. sm86 for RTX 3070). Regenerated automatically if missing, first run takes several minutes.

---

## Requirements

- Python 3.9
- numpy
- SimpleITK
- onnxruntime (variant depends on backend, see [Installation](#installation))

Optional (for `--nms-backend nndet`):
- nnDetection framework (nndet)
- torch, torchvision

---

## Installation

Two conda environments are used because `onnxruntime-gpu` and `onnxruntime-openvino` are mutually exclusive pip packages.

### Environment 1: CPU + OpenVINO

```bash
conda create -n nnDetPy39 python=3.9
conda activate nnDetPy39
pip install numpy SimpleITK onnxruntime-openvino==1.19.0
```

Supports backends: `cpu`, `openvino`.

### Environment 2: CUDA + TensorRT

```bash
conda create -n nnDetPy39-trt python=3.9
conda activate nnDetPy39-trt
pip install numpy SimpleITK
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
conda install cudnn=8
pip install tensorrt==10.3.0
pip install nvidia-cuda-runtime-cu12==12.2.2 nvidia-cublas-cu12==12.2.5.6
```

Then configure `LD_LIBRARY_PATH` (create once):

```bash
CONDA_PREFIX=$CONDA_PREFIX
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << 'EOF'
#!/bin/bash
SITE="$CONDA_PREFIX/lib/python3.9/site-packages"
export LD_LIBRARY_PATH="$SITE/tensorrt_libs:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF
```

Supports backends: `cpu`, `cuda`, `trt`.

### Validated Configuration

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3070 (8 GB, sm86) |
| Driver | 535.309 |
| CUDA toolkit (system) | 11.7 |
| TensorRT | 10.3 |
| Python | 3.9 |

---

## Quick Start

```bash
cd /home/eqip/nndet_onnx

# Single image, TRT FP16 (fastest)
conda activate nnDetPy39-trt
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx_shaped.onnx \
  --plan-path data/model/plan_inference.json \
  --image-path data/test_images/1_AV_LA.nii.gz \
  --output-dir data/test_images/1_AV_LA_results \
  --backend trt --trt-fp16

# Batch mode (all NIfTI in a directory)
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx_shaped.onnx \
  --plan-path data/model/plan_inference.json \
  --image-dir data/test_images/1_AV_LA \
  --output-dir data/test_images/1_AV_LA_results_batch \
  --backend trt --trt-fp16

# CPU only (no GPU required)
conda activate nnDetPy39
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx.onnx \
  --plan-path data/model/plan_inference.json \
  --image-path data/test_images/1_AV_LA.nii.gz \
  --output-dir data/test_images/1_AV_LA_results \
  --backend cpu
```

---

## CLI Reference

```
python nndet_onnx_inference_sw.py [OPTIONS]
```

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
| `--overlap` | `0.5` | Overlap between patches, proportion in [0, 1). Actual overlap ≥ requested (guaranteed). |

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
  └── Extract patches of size [64, 96, 96] (Z, Y, X)

[4/5] Batched inference
  ├── Group patches into batches of 4 (model's fixed batch size)
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
Resampled to the **original image geometry** (spacing, origin, direction, size) using nearest-neighbor interpolation, so it can be directly overlaid on the source image in any viewer.

---

## Inference Backends

| Backend | Provider chain | Model file | Conda env |
|---------|---------------|------------|-----------|
| `cpu` | CPU | `model_onnx.onnx` | either |
| `openvino` | OpenVINO → CPU | `model_onnx.onnx` | nnDetPy39 |
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

Image: 1_AV_LA (CT scan), 294 patches, 74 batches, overlap 0.5, score-thresh 0.5.

| Backend | Inference | Total | s/batch | Speedup vs CPU | Detections |
|---------|-----------|-------|---------|----------------|------------|
| `cpu` | 218s | 220s | 2.95 | ×1 | 26 |
| `openvino` | 113s | 115s | 1.52 | ×1.9 | 26 |
| `cuda` | 14s | 16s | 0.19 | ×15.6 | 26 |
| `trt --trt-fp16` | 6s | 8s | 0.08 | **×36.9** | 26 |

All backends produce **26 detections** — results are consistent across backends (minor numerical variation before NMS, identical after).

---

## Limitations & Known Issues

- **Single class**: The current model detects one class only (label 0). Multi-class support would require per-class NMS.
- **Single fold**: Uses fold 0 only. Multi-fold ensemble was intentionally deferred for industrialization simplicity.
- **Image must be ≥ patch_size after resampling**: If any resampled dimension is smaller than the patch (64×96×96), the pipeline exits with an error.
- **No DICOM input**: Input must be NIfTI (`.nii` or `.nii.gz`). DICOM→NIfTI conversion should be done upstream.
- **Legacy commented code**: `filter_small_boxes` contains commented-out legacy code (pre-axis-fix version) kept for team reference. Remove before external delivery.
- **Mask is bounding-box based**: The output mask fills detection bounding boxes, not a pixel-level segmentation.

### Deferred to post-MVP (before production)

- **Structured logging**: Currently all output is `print()`. Production should use Python `logging` with levels (DEBUG/INFO/WARNING).
- **Pipeline versioning**: Exports (JSON/CSV/PKL) should include a pipeline version number for traceability (medical device regulation).
- **ONNX Runtime error handling**: No try/except around session creation or inference. GPU OOM, driver mismatch, or engine incompatibility will produce raw Python exceptions.

---

## Development Notes

Detailed development history, technical decisions, and context for resuming work are in [`notes/NOTES_REVIEW_SW.md`](notes/NOTES_REVIEW_SW.md).
