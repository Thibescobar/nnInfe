# Notes de projet — nnDet ONNX Inference Pipeline
> Dernière mise à jour : 2026-04-02 (fin de journée)

---

## Historique du projet

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
- Vérification que `feature_map_sizes` se calcule depuis strides + decoder_levels

### Phase 5 — UX (2026-03-30)
- Timers sur chaque étape
- Barre de progression avec ETA
- Verbose preprocessing (stats à chaque étape)
- Overlap en proportion [0,1) au lieu de pourcentage
- `flush=True` sur tous les prints (buffering conda)

### Phase 6 — Code review + refactoring (2026-03-31)
- `StatisticsImageFilter` au lieu de 4× `GetArrayFromImage` (zero-copy)
- Constantes d'axes `D0_MIN, D1_MIN, D0_MAX, D1_MAX, D2_MIN, D2_MAX`
- `_iou_3d` : variables renommées (i_d0_lo, i_d1_lo, etc.)
- `AnchorGenerator3DSONNX` class → `_generate_cell_anchors()` + `generate_anchors()` (fonctions pures)
- `detections_to_mask` simplifié (toujours CC, uint8 direct, retour unique)
- `compute_patch_positions` réécrit avec logique C++ (positions uniformes, overlap ≥ demandé garanti)
- Stride arrondi par `int()` (floor) → overlap jamais inférieur à la demande
- Print des params harmonisé : `(default)` / `(auto: from plan)`

### Phase 7 — Backends d'inférence (2026-04-02)
- Ajout de `create_session()` avec 4 backends : `cpu`, `cuda`, `openvino`, `trt`
- `--backend` remplace `--providers` (noms lisibles)
- `--trt-fp16` pour FP16 TensorRT
- `--build-engine-only` pour pré-compiler les engines TRT
- Cache engine TRT automatique dans `trt_engine_cache/`
- Shape inference requise pour TRT : fournir manuellement `model_onnx_shaped.onnx` via `--model-path`
- `ort.set_default_logger_severity(3)` pour supprimer les warnings C++
- Print adaptatif : "Loading" (cache trouvé) vs "Creating" (première compilation)
- 2 envs conda configurés (`nnDetPy39` OpenVINO, `nnDetPy39-trt` CUDA+TRT)
- `activate.d/env_vars.sh` pour LD_LIBRARY_PATH automatique

### Phase 8 — MVP (2026-04-02)
- **pkl → JSON converter** : script `nndet_pkl_to_json.py` créé, extrait les 9 champs utiles du pkl en JSON
- Script principal migré de pkl à **JSON uniquement** (`import pickle` supprimé)
- **Shape inference auto supprimée** (décision utilisateur) : le `model_onnx_shaped.onnx` est fourni manuellement
- Fonction `_ensure_shape_inference` supprimée du code
- Copie de sauvegarde créée par l'utilisateur : `nndet_onnx_inference_sw_beforeMVP.py`
- Validé : 26 détections avec `plan_inference.json` + TRT FP16 en 5.57s
- **Shape inference script** : script standalone `nndet_onnx_shape_inference.py` créé — prend un ONNX en entrée, vérifie si les shapes sont déjà inférées, produit `_shaped.onnx` si nécessaire. Utilitaire one-shot pour préparer le modèle TRT.

### Fichiers actuels
- `nndet_onnx_inference_sw.py` — script principal (~838 lignes)
- `nndet_onnx_inference_sw_beforeMVP.py` — copie de sauvegarde avant MVP
- `nndet_onnx_inference.py` — ancien script single-patch (référence)
- `nndet_pkl_to_json.py` — converter pkl → JSON (one-shot)
- `nndet_onnx_shape_inference.py` — shape inference ONNX (one-shot, requis pour TRT)
- `NOTES_REVIEW_SW.md` — ce fichier
- `_dump_pkl.py`, `_check_fm.py` — utilitaires debug

---

## Review initiale (2026-03-31) — Bilan des décisions

### Corrections appliquées

| Item | Quoi | Détail |
|------|------|--------|
| Performance verbose | `preprocess_image` | `GetArrayFromImage` (×4 copies) → `StatisticsImageFilter` (zero-copy) |
| Constantes d'axes | Box indices | Ajout de `D0_MIN, D1_MIN, D0_MAX, D1_MAX, D2_MIN, D2_MAX`, utilisées dans `gaussian_weight_for_boxes`, `filter_small_boxes`, `_iou_3d` |
| Nommage `_iou_3d` | Variables | `x1,y1,z1` → `i_d0_lo, i_d1_lo, i_d0_hi, ...` + constantes dans `volume()` |
| Anchor class → fonctions | Architecture | `AnchorGenerator3DSONNX` → `_generate_cell_anchors()` + `generate_anchors()` (fonctions pures) |
| `detections_to_mask` | Simplification | Toujours CC, masque `uint8` direct, retour unique `sitk.Image` |
| `compute_patch_positions` | Réécriture | Logique C++ : numPatches par ceil, stepSize uniforme, premier patch à 0, dernier au bord |
| Overlap garanti | Stride | Arrondi par `int()` (floor), affichage overlap réel vs demandé |
| Print params | UX | `(default)` pour valeurs par défaut, `(auto: from plan)` pour iou-threshold |

### Conservé volontairement

| Item | Raison | À revisiter |
|------|--------|-------------|
| Parallélisme ONNX Runtime | Par défaut ORT utilise tous les cœurs, correct pour mono-session | Si multi-GPU ou pipeline parallèle |
| Global NMS en O(n²) | ~110 détections = négligeable. Solutions si besoin : NMS par classe, grille spatiale, soft-NMS, backend GPU, tri+early-stop (déjà en place) | Si n détections >> 1000 |
| Dict pour détections | Préférence utilisateur, pas de NamedTuple | Sera un `struct` en C++ |
| Code commenté `filter_small_boxes` | Rémanent pour l'équipe (ancienne version buggée) | Supprimer avant livraison externe |
| Pas de garde image < patch_size | Connu | Traité lors du port C++ |

### Légèreté du livrable
Déjà optimal : 3 dépendances (numpy, onnxruntime, SimpleITK), toutes indispensables. Backend nndet/torch correctement optionnel.

## Backends d'inférence

### Benchmarks (image 1_AV_LA, 294 patches, overlap 0.5)

| Backend | Env conda | Inférence | Total | s/batch | Speedup vs CPU | Détections |
|---------|-----------|-----------|-------|---------|----------------|------------|
| `cpu` | nnDetPy39 | 218s | 220s | 2.95 | ×1 | 26 |
| `openvino` | nnDetPy39 | 113s | 115s | 1.52 | ×1.9 | 26 |
| `cuda` | nnDetPy39-trt | 14s | 16s | 0.19 | ×15.6 | 26 |
| `trt --trt-fp16` | nnDetPy39-trt | 6s | 8s | 0.08 | **×36.9** | 26 |

### Configuration des envs

**nnDetPy39** (CPU + OpenVINO) :
- `onnxruntime-openvino` 1.19.0
- Providers : OpenVINO, CPU

**nnDetPy39-trt** (CUDA + TensorRT) :
- `onnxruntime-gpu` (CUDA 11 index) + `cudnn` 8.x (conda) + `tensorrt` 10.3 (pip)
- `nvidia-cuda-runtime-cu12==12.2.2`, `nvidia-cublas-cu12==12.2.5.6` (pip)
- `LD_LIBRARY_PATH` configuré via `activate.d/env_vars.sh` (automatique)
- Providers : TensorRT, CUDA, CPU

### Notes TensorRT
- Le modèle ONNX doit être passé par `onnx.shape_inference` → `model_onnx_shaped.onnx`
- Première run compile l'engine TRT (~2 min), cache dans `trt_engine_cache/` à côté du modèle
- Runs suivantes : session ready en ~4s
- `--build-engine-only` : compile l'engine sans lancer d'inférence (pas besoin de --image-path/--output)
- FP16 : résultats cohérents (26 détections identiques), variation numérique minime (94-95 avant NMS vs 110 CPU)
- Le cache TRT contient ~33 engines (129 Mo) : onnxruntime découpe le graphe en sous-graphes TRT (ops supportées) + fallback CUDA (ops non supportées comme ScatterND, NonZero)

### Port C++ — TensorRT
Pour le port C++, on veut un **engine TRT unique** (pas le cache multi-engine d'onnxruntime).
Options :
1. **`trtexec`** : convertit l'ONNX complet en un seul `.engine`, mais les ops non supportées par TRT nécessitent des plugins custom ou un refactoring du graphe ONNX (supprimer/externaliser les ops non-TRT)
2. **TensorRT API C++ native** : `nvinfer1::IBuilder` + `INetworkDefinition` — contrôle total, mais il faut gérer les ops non supportées manuellement
3. **Graphe ONNX simplifié** : utiliser `onnx-simplifier` ou `polygraphy surgeon` pour remplacer les ops dynamiques (NonZero, ScatterND) par des ops statiques compatibles TRT, puis un seul engine
4. **Architecture hybride C++** : TRT pour le backbone (1 engine) + code C++ natif pour le post-processing (anchor decoding, NMS) — probablement la meilleure approche car le post-processing est léger et ne bénéficie pas de TRT

### Backends disponibles
- `cpu` : CPUExecutionProvider
- `cuda` : CUDAExecutionProvider → fallback CPU
- `openvino` : OpenVINOExecutionProvider → fallback CPU
- `trt` : TensorrtExecutionProvider → CUDA → CPU
- `--trt-fp16` : active FP16 pour TensorRT

### Cohabitation OV + TRT
Les packages pip `onnxruntime-gpu` et `onnxruntime-openvino` sont mutuellement exclusifs (même module Python).
Solutions : deux envs conda (actuel), ou build from source avec les deux flags (futur si nécessaire).

---

## Récapitulatif — Du notebook au MVP

### Point de départ
Un notebook Jupyter (`nndet_lucia_try.ipynb`) exploratoire qui testait l'inférence ONNX d'un modèle nnDet (RetinaUNet 3D) sur un seul patch. L'objectif était de valider que le modèle exporté en ONNX fonctionnait, en dehors du framework nnDet/PyTorch.

### Ce qu'on a construit
Un script standalone Python (`nndet_onnx_inference_sw.py`, ~838 lignes) capable de :
- Prétraiter une image NIfTI (resampling, clipping, normalisation) à partir de la config du plan d'entraînement
- Générer les anchors 3D (284 796 par patch) sans dépendance à nnDet
- Découper le volume en patches avec sliding window uniforme et overlap garanti
- Inférer par batch de 4 sur **4 backends** (CPU, OpenVINO, CUDA, TensorRT FP16)
- Post-traiter (score filter, size filter, NMS) avec pondération Gaussienne
- Exporter un masque connected-components en NIfTI

### Pourquoi c'était difficile
- La convention d'axes nnDet (`dim0=Z, dim1=Y, dim2=X`) est trompeuse — 4 bugs d'axes trouvés et corrigés, tous dans le mapping entre coordonnées box et coordonnées image/spacing
- La config nnDet est éclatée entre `plan_inference.pkl`, les poids du modèle ONNX, et un JSON redondant (supprimé)
- TensorRT demande shape inference sur l'ONNX, un runtime CUDA 12 alors que le système a CUDA 11.7, et TensorRT 10.3 (pas la dernière) pour la compatibilité driver
- Les packages pip `onnxruntime-gpu` et `onnxruntime-openvino` ne coexistent pas → 2 envs conda

### Où on en est
Le script fonctionne end-to-end sur les 4 backends. Les résultats sont cohérents (26 détections sur 1_AV_LA quel que soit le backend). Le speedup TRT FP16 est de ×37 vs CPU (6s d'inférence pour 294 patches). La copie `_beforeMVP` sauvegarde cet état.

### Ce qu'il reste (MVP)
Avant de passer au C++ : convertir le pkl en JSON (lisible par le C++), automatiser la shape inference TRT, ajouter l'export des détections en JSON/CSV, valider les inputs, et écrire des tests unitaires pour ne plus jamais avoir de régressions d'axes.

---

## MVP Python — Roadmap (avant port C++)

### Haute priorité
- [x] **pkl → JSON converter** : Script `nndet_pkl_to_json.py` créé. Convertit le pkl en JSON lisible par C++ (nlohmann::json). Le script d'inférence lit maintenant le JSON uniquement (`import pickle` supprimé). Validé : 26 détections identiques avec `plan_inference.json`.
- [x] ~~**Shape inference auto pour TRT**~~ **ANNULÉ puis remplacé** : l'utilisateur préfère un script standalone (`nndet_onnx_shape_inference.py`) plutôt qu'une fonction intégrée au script principal. Script créé — prend `--input` et `--output`, vérifie les shapes existantes. Fonction `_ensure_shape_inference` supprimée du script principal.
- [ ] **Export détections JSON/CSV** : boxes/scores/labels en plus du masque CC, pour évaluation et debug.
- [ ] **Tests unitaires** : pytest, fixtures synthétiques, coverage sur anchors, IoU, NMS, axes, sliding window.

### Moyenne priorité
- [ ] **Validation des inputs** : image < patch_size, fichiers inexistants, pkl clés manquantes, overlap hors [0,1).
- [ ] **Supprimer nms_nndet** : impose torch + nndet, nms_numpy suffit. Simplifier le livrable.
- [ ] **Batch multi-images** : `--image-dir` ou fichier liste pour évaluer un dataset entier.

### Basse priorité
- [ ] **Multi-fold ensemble** : nnDet 5 folds. Actuellement fold0 seul.
- [ ] **Nettoyage code commenté** : `filter_small_boxes` rémanent — à supprimer avant livraison externe.

### Livraison (après MVP, avant port C++)
- [ ] **README.md complet** : documentation exhaustive faisant aussi office de doc technique. Couvre : installation (envs conda, dépendances), usage CLI (tous les args, exemples par backend), format des fichiers (JSON config, ONNX, JSON détections), architecture du pipeline, conventions d'axes, limitations connues.
- [ ] **Arborescence projet propre** : réorganiser tous les fichiers (scripts, configs, utilitaires, notes) dans une structure de dossiers claire pour faciliter l'exportation vers le collègue ingénieur industrialisation qui s'occupera du packaging.

---

## Contexte pour reprise de conversation

> Cette section contient tout ce qu'il faut pour reprendre le travail après un clear ou dans une nouvelle conversation.
> **Lire ce fichier en entier avant toute action.**

### Projet
Pipeline d'inférence standalone pour **nnDet** (détection 3D médicale) exporté en ONNX. Le but final est un **port C++** avec ITK + ONNX Runtime / TensorRT natif. Le script Python sert de **prototype de référence** et de **spécification fonctionnelle** pour le C++.

### Fichiers et chemins

| Fichier | Chemin | Rôle |
|---------|--------|------|
| Script principal | `/home/eqip/Downloads/nndet_onnx_inference_sw.py` (~838 lignes) | Pipeline complet |
| Copie pré-MVP | `/home/eqip/Downloads/nndet_onnx_inference_sw_beforeMVP.py` | Sauvegarde état stable |
| Converter pkl→JSON | `/home/eqip/Downloads/nndet_pkl_to_json.py` | One-shot, utilitaire |
| Shape inference ONNX | `/home/eqip/Downloads/nndet_onnx_shape_inference.py` | One-shot, requis pour TRT |
| Copie pré-refacto | `/home/eqip/Downloads/nndet_onnx_inference.py` | Ancien script single-patch (référence) |
| Notes | `/home/eqip/Downloads/NOTES_REVIEW_SW.md` | **Ce fichier** — lire en entier avant toute action |
| Modèle ONNX | `/media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx.onnx` | Modèle original |
| Modèle ONNX shaped | `/media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx_shaped.onnx` | Modèle avec shape inference (auto-généré pour TRT) |
| Plan inference JSON | `/media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.json` | Config nnDet (**format actuel**) |
| Plan inference pkl | `/media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.pkl` | Config nnDet (ancien, source pour conversion) |
| Cache TRT | `/media/eqip/T9/le/repos_Git/repos_GB/fold0/trt_engine_cache/` | 33 engines, 129 Mo |
| Image test | `/media/eqip/T9/datasets__storage/SATT/BDD/1_AV_LA/1_AV_LA.nii.gz` | CT scan NIfTI |

### Environnements conda

| Env | Packages clés | Backends | Python |
|-----|---------------|----------|--------|
| `nnDetPy39` | `onnxruntime-openvino` 1.19.0, SimpleITK, numpy | cpu, openvino | 3.9 |
| `nnDetPy39-trt` | `onnxruntime-gpu` (CUDA 11), cuDNN 8, TensorRT 10.3, CUDA runtime 12.2 | cpu, cuda, trt | 3.9 |

`nnDetPy39-trt` a un script `activate.d/env_vars.sh` qui configure `LD_LIBRARY_PATH` automatiquement.

### Modèle nnDet — Specs techniques
- **Architecture** : RetinaUNet 3D, batch_size fixe = 4
- **Input ONNX** : `images` shape `[4, 1, 64, 96, 96]`, `anchors` shape `[4, 284796, 6]`
- **Output ONNX** : 12 tenseurs (4×boxes + 4×scores + 4×labels)
- **patch_size** (ZYX) : `[64, 96, 96]`
- **Convention axes** : `dim0=Z, dim1=Y, dim2=X` — **source de tous les bugs historiques**
- **Box format** : `(dim0_min, dim1_min, dim0_max, dim1_max, dim2_min, dim2_max)` → constantes `D0_MIN..D2_MAX`
- **Spacing** : SimpleITK renvoie XYZ, le pkl stocke ZYX
- **target_spacing** : dans le pkl, en ZYX
- **NMS IoU threshold** : 0.1 (depuis `plan_inference["inference_plan"]["model_iou"]`)

### Résultats de référence (1_AV_LA, overlap 0.5, score-thresh 0.5)
- 294 patches, 74 batches
- 94-110 détections avant global NMS (varie selon backend, variations numériques GPU/CPU)
- **26 détections** après global NMS — **résultat attendu quel que soit le backend**
- Temps : CPU 220s, OpenVINO 115s, CUDA 16s, TRT FP16 8s

### Commandes de test

```bash
# CPU (env nnDetPy39)
python /home/eqip/Downloads/nndet_onnx_inference_sw.py \
  --model-path /media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx.onnx \
  --plan-path /media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.json \
  --image-path /media/eqip/T9/datasets__storage/SATT/BDD/1_AV_LA/1_AV_LA.nii.gz \
  --output /media/eqip/T9/le/repos_Git/repos_GB/fold0/test_cpu.nii.gz \
  --overlap 0.5 --score-thresh 0.5 --backend cpu

# TRT FP16 (env nnDetPy39-trt) — utiliser le modèle shaped
python /home/eqip/Downloads/nndet_onnx_inference_sw.py \
  --model-path /media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx_shaped.onnx \
  --plan-path /media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.json \
  --image-path /media/eqip/T9/datasets__storage/SATT/BDD/1_AV_LA/1_AV_LA.nii.gz \
  --output /media/eqip/T9/le/repos_Git/repos_GB/fold0/test_trt.nii.gz \
  --overlap 0.5 --score-thresh 0.5 --backend trt --trt-fp16

# Build engine TRT seulement
python /home/eqip/Downloads/nndet_onnx_inference_sw.py \
  --model-path /media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx_shaped.onnx \
  --plan-path /media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.json \
  --backend trt --trt-fp16 --build-engine-only

# Shape inference ONNX (env nnDetPy39, one-shot, requis pour TRT)
python /home/eqip/Downloads/nndet_onnx_shape_inference.py \
  --input /media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx.onnx \
  --output /media/eqip/T9/le/repos_Git/repos_GB/fold0/model_onnx_shaped.onnx

# Conversion pkl → JSON (env nnDetPy39, one-shot)
python /home/eqip/Downloads/nndet_pkl_to_json.py \
  --pkl /media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.pkl \
  --output /media/eqip/T9/le/repos_Git/repos_GB/fold0/plan_inference.json
```

### Système
- **Machine** : Dell XPS 8950, Ubuntu
- **GPU** : NVIDIA RTX 3070 (8 Go), driver 535 (supporte CUDA ≤ 12.2)
- **CUDA toolkit système** : 11.7 (`nvcc`)
- **Disque** : `/media/eqip/T9/` = disque externe (modèles + données)

### Prochaine action
~~pkl→JSON~~ ✓ ~~shape inference auto~~ → script standalone ✓ → **Prochain : export détections JSON/CSV** (MVP 3), puis validation inputs, supprimer nms_nndet, tests unitaires. Puis port C++.