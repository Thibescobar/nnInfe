"""
Run ONNX shape inference on a model and save the result.
Required for TensorRT backend which needs intermediate shapes annotated.

Usage:
    python nndet_onnx_shape_inference.py \
        --input  model_onnx.onnx \
        --output model_onnx_shaped.onnx
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ONNX shape inference and save the result.",
    )
    parser.add_argument("--input", required=True, help="Path to the input ONNX model")
    parser.add_argument(
        "--output", default=None,
        help="Path to the output model (default: <input>_shaped.onnx)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        output = args.input.replace(".onnx", "_shaped.onnx")

    import onnx
    from onnx import shape_inference

    print(f"Loading model: {args.input}", flush=True)
    model = onnx.load(args.input)

    # Check if shapes are already present
    has_shapes = any(
        vi.type.tensor_type.HasField("shape")
        for vi in model.graph.value_info
    )
    if has_shapes:
        print("Model already has intermediate shape info.", flush=True)
        if args.output is None:
            print("Nothing to do. Use --output to force a save.", flush=True)
            return

    print("Running shape inference …", flush=True)
    model_shaped = shape_inference.infer_shapes(model)

    print(f"Saving: {output}", flush=True)
    onnx.save(model_shaped, output)

    n_shapes = sum(
        1 for vi in model_shaped.graph.value_info
        if vi.type.tensor_type.HasField("shape")
    )
    print(f"Done. {n_shapes} intermediate tensors now have shape info.", flush=True)


if __name__ == "__main__":
    main()
