#!/usr/bin/env python3
"""Convert plan_inference.pkl to a JSON file readable by C++ (nlohmann::json).

Only extracts the fields used by nndet_onnx_inference_sw.py:
  - patch_size
  - target_spacing (ZYX order)
    - transpose_forward / transpose_backward (if present)
  - anchors (width, height, depth per decoder level)
  - architecture (strides, decoder_levels)
  - intensity_properties (global: percentile_00_5, percentile_99_5, mean, std)
  - inference_plan (model_iou, model_score_thresh, model_topk,
                    model_detections_per_image, remove_small_boxes)

Usage:
    python nndet_pkl_to_json.py --pkl /path/to/plan_inference.pkl [--output /path/to/plan_inference.json]
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path


def convert(pkl_path: str, output_path: str = None) -> None:
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"Error: {pkl_path} does not exist", file=sys.stderr)
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        plan = pickle.load(f)

    # --- Extract fields used by the inference script ---

    patch_size = plan["patch_size"]  # list[int], ZYX

    target_spacing = plan["target_spacing"]  # list[float], ZYX

    # Anchors: per decoder level, each is a tuple of floats → convert to list
    anchors_cfg = plan["anchors"]
    anchors = {
        "width": [list(t) for t in anchors_cfg["width"]],
        "height": [list(t) for t in anchors_cfg["height"]],
        "depth": [list(t) for t in anchors_cfg["depth"]],
    }

    # Architecture: strides (list of lists) + decoder_levels (tuple → list)
    arch = plan["architecture"]
    architecture = {
        "strides": arch["strides"],
        "decoder_levels": list(arch["decoder_levels"]),
    }

    # Intensity properties: global stats from modality 0
    ip0 = plan["dataset_properties"]["intensity_properties"][0]
    global_keys = ["percentile_00_5", "percentile_99_5", "mean", "std"]
    intensity_properties = {}
    for k in global_keys:
        val = ip0[k]
        if math.isnan(val) or math.isinf(val):
            print(f"Warning: intensity_properties[{k}] = {val}", file=sys.stderr)
        intensity_properties[k] = float(val)

    # Inference plan: useful thresholds
    ip = plan["inference_plan"]
    inference_plan = {
        "model_iou": float(ip["model_iou"]),
        "model_score_thresh": float(ip["model_score_thresh"]),
        "model_topk": int(ip["model_topk"]),
        "model_detections_per_image": int(ip["model_detections_per_image"]),
        "remove_small_boxes": float(ip["remove_small_boxes"]),
    }

    # --- Assemble ---

    out = {
        "_comment": f"Converted from {pkl_path.name} by nndet_pkl_to_json.py",
        "patch_size": patch_size,
        "target_spacing": target_spacing,
        "anchors": anchors,
        "architecture": architecture,
        "intensity_properties": intensity_properties,
        "inference_plan": inference_plan,
    }

    if "transpose_forward" in plan:
        out["transpose_forward"] = list(plan["transpose_forward"])
    if "transpose_backward" in plan:
        out["transpose_backward"] = list(plan["transpose_backward"])

    # --- Write ---

    if output_path is None:
        output_path = pkl_path.with_suffix(".json")
    else:
        output_path = Path(output_path)

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Written: {output_path}")
    print(f"  patch_size:          {patch_size}")
    print(f"  target_spacing:      {target_spacing}")
    print(f"  anchors levels:      {len(anchors['width'])} (sizes per level: {[len(t) for t in anchors['width']]})")
    print(f"  architecture strides: {architecture['strides']}")
    print(f"  decoder_levels:      {architecture['decoder_levels']}")
    print(f"  clip range:          [{intensity_properties['percentile_00_5']}, {intensity_properties['percentile_99_5']}]")
    print(f"  normalize:           mean={intensity_properties['mean']:.4f}  std={intensity_properties['std']:.4f}")
    print(f"  model_iou:           {inference_plan['model_iou']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert plan_inference.pkl → JSON")
    parser.add_argument("--pkl", required=True, help="Path to plan_inference.pkl")
    parser.add_argument("--output", default=None, help="Output JSON path (default: same dir, .json extension)")
    args = parser.parse_args()
    convert(args.pkl, args.output)
