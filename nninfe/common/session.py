"""ONNX Runtime session helpers."""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

BACKENDS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}


def create_session(
    model_path: str,
    backend: str = "cpu",
    trt_fp16: bool = False,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with the appropriate providers."""
    providers = BACKENDS[backend]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    ort.set_default_logger_severity(3)

    provider_options: list = []
    for prov in providers:
        if prov == "TensorrtExecutionProvider":
            precision = "fp16" if trt_fp16 else "fp32"
            cache_dir = str(Path(model_path).parent / f"trt_engine_cache_{precision}")
            os.makedirs(cache_dir, exist_ok=True)
            provider_options.append({
                "trt_engine_cache_enable": "True",
                "trt_engine_cache_path": cache_dir,
                "trt_fp16_enable": "True" if trt_fp16 else "False",
                "trt_timing_cache_enable": "True",
            })
        elif prov == "OpenVINOExecutionProvider":
            provider_options.append({"device_type": "CPU"})
        else:
            provider_options.append({})

    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers,
        provider_options=provider_options,
    )
    actual = session.get_providers()
    print(f"      ONNX Runtime providers: {actual}\\n", flush=True)
    return session


def run_inference(
    session: ort.InferenceSession,
    input_array: np.ndarray,
    anchors_batch: np.ndarray,
) -> list:
    """Run ONNX inference and return raw outputs."""
    return session.run(None, {"images": input_array, "anchors": anchors_batch})


def parse_outputs(outputs: list, batch_size: int) -> List[Dict[str, np.ndarray]]:
    """Parse raw ONNX outputs into per-batch detections."""
    detections = []
    for b in range(batch_size):
        detections.append(
            {
                "boxes": outputs[b],
                "scores": outputs[b + batch_size],
                "labels": outputs[b + 2 * batch_size],
            }
        )
    return detections
