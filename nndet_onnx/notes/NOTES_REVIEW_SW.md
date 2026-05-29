# Notes de projet — nnDet ONNX Inference Pipeline
> Dernière mise à jour : 2026-05-29

Pipeline d'inférence standalone pour **nnDet** (détection 3D médicale) exporté en ONNX.
But final : **port C++** avec ITK + ONNX Runtime / TensorRT natif.
Le script Python sert de **prototype de référence** et de **spécification fonctionnelle**.

---

## 1. Historique du projet

### Phase 1 — Prototype (2026-03-30)
- Analyse du notebook `nndet_lucia_try.ipynb` (pipeline nnDet ONNX pas-à-pas)
- Création de `nndet_onnx_inference.py` — script single-patch, 2 backends NMS
- Validation : 2 détections sur 1_AV_LA avec un seul patch

### Phase 2 — Sliding window (2026-03-30)
- Création de `nndet_onnx_inference_sw.py` — sliding window avec overlap, Gaussian weighting, global NMS
- Batched inference (batch_size=4, padding du dernier batch)
- Validation : 294 patches, 110→65 détections

### Phase 3 — Bug fixes axes (2026-03-30)
- **4 bugs d'axes corrigés** (tous causés par la convention nnDet : dim0=Z, dim1=Y, dim2=X)
  - `translate_boxes` : shift [ox, oy, ox, oy, oz, oz] → [oz, oy, oz, oy, ox, ox]
  - `detections_to_mask` : mapping z/y/x corrigé
  - `gaussian_weight_for_boxes` : centres cx/cy/cz mappés correctement
  - `filter_small_boxes` : spacing_xyz[0]=X appliqué à dim2=X (pas à dim0)

### Phase 4 — Simplification config (2026-03-30)
- Suppression de la dépendance `model_onnx.json`
- Migration vers `plan_inference.pkl` + ONNX model inspection
- `feature_map_sizes` calculé depuis strides + decoder_levels

### Phase 5 — UX (2026-03-30)
- Timers, barre de progression avec ETA, verbose preprocessing
- Overlap en proportion [0,1) au lieu de pourcentage
- `flush=True` sur tous les prints (buffering conda)

### Phase 6 — Code review + refactoring (2026-03-31)
- `StatisticsImageFilter` au lieu de 4× `GetArrayFromImage` (zero-copy)
- Constantes d'axes `D0_MIN..D2_MAX` utilisées partout
- `_iou_3d` : variables renommées (i_d0_lo, i_d1_lo, etc.)
- Anchor class → fonctions pures (`_generate_cell_anchors`, `generate_anchors`)
- `detections_to_mask` simplifié (CC, uint8, retour unique)
- `compute_patch_positions` réécrit (logique C++, overlap ≥ demandé garanti)

### Phase 7 — Backends d'inférence (2026-04-02)
- `create_session()` avec 4 backends : `cpu`, `cuda`, `openvino`, `trt`
- `--trt-fp16`, `--build-engine-only`, cache TRT automatique
- Shape inference requise pour TRT (fournir `model_onnx_shaped.onnx`)
- 2 envs conda : `nnDetPy39` (CPU+OpenVINO), `nnDetPy39-trt` (CUDA+TRT)

### Phase 8 — MVP (2026-04-02)
- **pkl → JSON converter** (`nndet_pkl_to_json.py`) — script principal migré vers JSON uniquement
- **Shape inference script** (`nndet_onnx_shape_inference.py`) — standalone, one-shot
- Validé : 26 détections avec `plan_inference.json` + TRT FP16 en 5.57s

### Phase 9 — Consolidation (2026-05-22)
- **Exports** : JSON (format nnDetection), CSV (coordonnées voxel + monde), PKL (`--export-pkl`, compatible `nndet_boxes2nii`)
- **Fix Gaussian scores** : pondération utilisée uniquement pour priorité NMS, `scores_original` restaurés avant export
- **Fix translate/merge** : `{**detection, "boxes": ...}` + concat dynamique des clés (plus de hardcoding 3 clés)
- **Validation outputs** : JSON→pkl→`nndet_boxes2nii` → NIfTI identique à la référence nnDetection
- **Validation inputs** : fichiers, extensions, paramètres, image < patch_size
- **Batch multi-images** : `--image-dir`, session + anchors partagés, résumé final
- **`process_single_image()`** extraite de `main()` pour le batch
- **`nms_nndet`** : `try/except ImportError` avec message clair
- **Arborescence projet** : tools, data, notes, tests
- Script principal : **~1162 lignes** (contre ~838 à la Phase 8)

---

## 2. Référence technique

### Modèle nnDet
- **Architecture** : RetinaUNet 3D, batch_size fixe = 4
- **Input ONNX** : `images` [4, 1, Z, Y, X], `anchors` [:, :, :]
- **Output ONNX** : 12 tenseurs (4×boxes + 4×scores + 4×labels)
- **patch_size** (ZYX) : [Z, Y, X]
- **NMS IoU threshold** : `plan_inference["inference_plan"]["model_iou"]`

### Convention d'axes — source de tous les bugs historiques
- **nnDet** : `dim0=Z, dim1=Y, dim2=X`
- **Box format** : `(dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)` → constantes `D0_MIN..D2_MAX`
- **SimpleITK** : spacing/origin/size en **XYZ**
- **plan_inference** : target_spacing en **ZYX**

### Benchmarks (1_AV_LA, 294 patches, 74 batches, overlap 0.5)

| Backend | Env conda | Inférence | Total | s/batch | Speedup | Détections |
|---------|-----------|-----------|-------|---------|---------|------------|
| `cpu` | nnDetPy39 | 218s | 220s | 2.95 | ×1 | 26 |
| `openvino` | nnDetPy39 | 113s | 115s | 1.52 | ×1.9 | 26 |
| `cuda` | nnDetPy39-trt | 14s | 16s | 0.19 | ×15.6 | 26 |
| `trt --trt-fp16` | nnDetPy39-trt | 6s | 8s | 0.08 | **×36.9** | 26 |

### Environnements conda

**nnDetPy39** (CPU + OpenVINO) :
- `onnxruntime-openvino` 1.19.0, SimpleITK, numpy
- Providers : OpenVINO, CPU

**nnDetPy39-trt** (CUDA + TensorRT) :
- `onnxruntime-gpu` (CUDA 11 index) + `cudnn` 8.x (conda) + `tensorrt` 10.3 (pip)
- `nvidia-cuda-runtime-cu12==12.2.2`, `nvidia-cublas-cu12==12.2.5.6` (pip)
- `LD_LIBRARY_PATH` via `activate.d/env_vars.sh` (automatique)
- Providers : TensorRT, CUDA, CPU

> Les packages `onnxruntime-gpu` et `onnxruntime-openvino` sont mutuellement exclusifs → 2 envs conda.

### Notes TensorRT
- ONNX doit passer par `onnx.shape_inference` → `model_onnx_shaped.onnx`
- Première run : compilation engine (~2 min), cache dans `trt_engine_cache_{precision}/`
- Runs suivantes : session ready en ~4s
- Le cache contient ~33 engines (129 Mo) : ORT découpe le graphe en sous-graphes TRT + fallback CUDA
- FP16 : résultats cohérents (26 détections), variation numérique minime avant NMS
- Cache lié à l'architecture GPU (sm86) et la version TRT — rebuild si changement

### Port C++ — TensorRT
Pour le C++, on veut un **engine TRT unique** (pas le cache multi-engine d'ORT). Meilleure approche :
**Architecture hybride** : TRT pour le backbone (1 engine) + C++ natif pour le post-processing (anchor decoding, NMS). Options alternatives : `trtexec` (plugins custom pour ops non supportées), TensorRT API C++ (`nvinfer1::IBuilder`), ou `onnx-simplifier`/`polygraphy surgeon` pour éliminer les ops dynamiques.

### Décisions de design conservées

| Item | Raison | À revisiter |
|------|--------|-------------|
| Parallélisme ORT | Par défaut utilise tous les cœurs, correct pour mono-session | Multi-GPU / pipeline |
| Global NMS O(n²) | ~110 détections = négligeable | Si n >> 1000 |
| Dict pour détections | Préférence utilisateur | `struct` en C++ |
| Code commenté `filter_small_boxes` | Legacy reference (pre-axis-fix version) | Clean up in future refactor |
| 3 dépendances | numpy, onnxruntime, SimpleITK — toutes indispensables | — |

---

## 3. Guide de reprise

> Lire cette section en entier avant toute action dans une nouvelle conversation.

### Ce qu'il reste
- [x] **Tests unitaires** : pytest, fixtures synthétiques, coverage sur anchors, IoU, NMS, axes, sliding window (65 tests, 42% coverage)

### Reporté post-MVP (avant production)
- [ ] **Logging structuré** : remplacer `print()` par `logging` avec niveaux (DEBUG/INFO/WARNING)
- [ ] **Versioning pipeline** : numéro de version dans les exports JSON/CSV/PKL (traçabilité réglementaire)
- [ ] **Error handling ORT** : try/except autour de la création de session et de l'inférence (GPU OOM, driver mismatch)
- [ ] **Lecture DICOM** : `ImageSeriesReader` SimpleITK, support `--image-path` (dossier DICOM) et `--image-dir` (batch de dossiers DICOM)

### Fichiers et chemins

**Project structure**

| Fichier | Chemin | Rôle |
|---------|--------|------|
| Script principal | `nndet_onnx/nndet_onnx_inference_sw.py` (~1195 lignes) | Pipeline complet |
| Converter pkl→JSON | `nndet_onnx/tools/nndet_pkl_to_json.py` (113 lignes) | One-shot |
| Shape inference | `nndet_onnx/tools/nndet_onnx_shape_inference.py` (66 lignes) | One-shot, requis pour TRT |
| Notes | `nndet_onnx/notes/NOTES_REVIEW_SW.md` | Ce fichier |
| Tests | `tests/` (4 fichiers, 65 tests) | pytest + pytest-cov |

**Data files (not versioned)**

Model files (`.onnx`, `plan_inference.json`) and test images are passed via CLI arguments. See the README for details.

### Commandes de test

```bash
# TRT FP16 — single image (env nnDetPy39-trt)
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx_shaped.onnx \
  --plan-path data/model/plan_inference.json \
  --image-path data/test_images/1_AV_LA.nii.gz \
  --output-dir data/test_images/1_AV_LA_results \
  --backend trt --trt-fp16

# TRT FP16 — batch mode
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx_shaped.onnx \
  --plan-path data/model/plan_inference.json \
  --image-dir data/test_images/1_AV_LA \
  --output-dir data/test_images/1_AV_LA_results_batch \
  --backend trt --trt-fp16

# CPU (env nnDetPy39)
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx.onnx \
  --plan-path data/model/plan_inference.json \
  --image-path data/test_images/1_AV_LA.nii.gz \
  --output-dir data/test_images/1_AV_LA_results \
  --backend cpu

# Build engine TRT seulement
python nndet_onnx_inference_sw.py \
  --model-path data/model/model_onnx_shaped.onnx \
  --plan-path data/model/plan_inference.json \
  --backend trt --trt-fp16 --build-engine-only

# Tools one-shot
python tools/nndet_onnx_shape_inference.py --input data/model/model_onnx.onnx --output data/model/model_onnx_shaped.onnx
python tools/nndet_pkl_to_json.py --pkl /path/to/plan_inference.pkl --output data/model/plan_inference.json
```

> Toutes les commandes supposent d'être à la racine du projet. Depuis le package installé, utiliser `nndet-infer` directement.

### Système
- **Machine** : Dell XPS 8950, Ubuntu 22.04
- **GPU** : NVIDIA RTX 3070 (8 Go, sm86), driver 535.309
- **CUDA toolkit** : 11.7 (`nvcc`)
