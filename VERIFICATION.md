# Verification & release checklist

This is the **QA procedure a tester runs before accepting a release** — the acceptance criteria,
not the "how to run" (for run commands and flags, see [README.md](README.md)).

By design there is **no automated verification harness in this repo**: it would require bundling
models and patient images, or wiring the code to remote storage. Neither is wanted. Instead, the
tester runs the steps below against **their own** models, images, and known-good reference outputs,
kept **outside** the repository. The pytest suite (`python -m pytest tests/`) covers logic on a
*mocked* ONNX session; it does **not** replace this real-model check.

> Notation: `{name}` is the input basename (e.g. an image `1_AV_LA` → `1_AV_LA_boxes.json`,
> `1_AV_LA.done`, `1_AV_LA_manifest.json`). Exit codes are the stable contract from
> `nninfe/common/errors.py`.

## 1. Sanity run (detection and segmentation)

Run each pipeline on a known-good case (commands in the README). Accept only if **all** hold:

- process exits **0**;
- the results for the chosen `--output-format` are present:
  - **detection**: `_boxes.json` is **always** written (plus `_boxes.pkl` with `--export-pkl`);
    `nifti` (default) adds `_mask.nii.gz` + `_boxes.csv`; `dicom-sr` adds the DICOM SR `_sr.dcm`;
    `both` adds all of them;
  - **segmentation**: `nifti` (default) writes `_seg.nii.gz`; `dicom-seg` writes the DICOM SEG
    `_seg.dcm`; `both` writes both;
  - the DICOM outputs (`_sr.dcm`, `_seg.dcm`) require **DICOM input** and, as part of the sanity
    check, should **open correctly in a viewer** (e.g. Weasis / OHIF) referencing the source series;
- `{name}.done` exists and `{name}.failed` does **not**;
- `{name}_manifest.json` has `"outcome": "ok"`, and its `model.sha256` / `plan.sha256` match the
  exact model and plan under test (the traceability link — verify these are the intended binaries);
- terminal progress looks sane (patch counts, no unexpected warnings on stderr).

## 2. Golden-output regression

Compare the run against a **known-good reference the tester keeps outside the repo** for the same
fixed input + model + params. Use the canonical machine-readable records as the source of truth:

- detection: `_boxes.json` — boxes/scores/labels match the reference within a small tolerance
  (the `_sr.dcm` / `_mask.nii.gz` are renderings of these same boxes; spot-check them visually but
  regress against the JSON);
- segmentation: `_seg.nii.gz` — label map identical, or Dice ≈ 1.0 within tolerance (the `_seg.dcm`
  is a rendering of the same mask).

Any drift beyond tolerance is a regression to investigate before release. Refresh the reference
only on an *intended* change, and record which model/params produced it.

## 3. Cross-backend consistency

Run the same case on `cpu` and on `trt`/`cuda`. Results must agree within a small numerical
tolerance (fp16 TRT drifts slightly more than fp32). This catches backend/engine issues that a
single-backend run hides. (Historically detection produced the same detection count across
backends — see the README benchmarks.)

## 4. Error-path spot checks (fail-safe behaviour)

Confirm the pipeline fails **predictably** — run once and check the exit code:

| Scenario | Expected |
|----------|----------|
| Malformed / 2-D / multi-channel input image | exit **8** (`InputValidationError`), `{name}.failed` written |
| Missing or non-`.onnx` model, bad args | exit **2** (usage), clean message on stderr, no traceback |
| Unreadable model / driver-engine mismatch | exit **3** (session), fatal |
| Batch where one image is bad, the rest valid | run continues, good images produced, exit **7** (partial); the bad one has `{name}.failed` + a manifest with `outcome: "failed"` and an `error.scope`/`error.retryable` classification |

A programming bug (should not happen in a release) would exit **1** with a full traceback — never
masked as a partial batch.

## 5. Model pre-flight (optional, TRT)

`nninfe-det --build-engine-only …` (and the `-seg` equivalent) builds the session / compiles and
caches the TensorRT engine **without running inference** — a fast check that the model loads and
the engine compiles on the target GPU before a full run.

---

**Sign-off:** a release is accepted when 1–4 pass on the tester's reference cases for both
pipelines and both the CPU and GPU backends in scope. Record the nninfe version and the model/plan
SHA-256 (from the manifest) alongside the sign-off for traceability.
