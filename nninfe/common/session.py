"""ONNX Runtime session helpers."""

import ctypes
import importlib
import logging
import os
import sys
import sysconfig
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from nninfe.common.errors import InferenceError, SessionError

logger = logging.getLogger(__name__)

# ONNX Runtime performs native GPU discovery while its Python module is imported. Keep that import
# lazy so lightweight CLI operations such as ``nninfe-*- --help`` do not initialize hardware (and
# do not emit platform discovery warnings).
ort: Any = None
_ORT_EXCEPTIONS: tuple = ()


def _ort_exception_types() -> tuple:
    """The exception classes ONNX Runtime raises (``Fail``, ``InvalidArgument``,
    ``InvalidProtobuf``, ``NoSuchFile``, …). They are collected dynamically from ORT's pybind
    module because — verified empirically — they subclass ``Exception`` *directly* (not
    ``RuntimeError``) and share no common base, so they cannot be caught by a stdlib base class.
    Collecting the whole module also stays correct if ORT adds a class in a future version."""
    try:
        _state = importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")
    except Exception:
        return ()
    return tuple(o for o in vars(_state).values() if isinstance(o, type) and issubclass(o, Exception))


def _load_ort() -> Any:
    """Import ONNX Runtime on first actual backend use, never merely for CLI parsing."""
    global ort, _ORT_EXCEPTIONS
    if ort is None:
        ort = importlib.import_module("onnxruntime")
    if not _ORT_EXCEPTIONS:
        _ORT_EXCEPTIONS = _ort_exception_types()
    return ort


def ort_exception_types() -> tuple:
    """Return the concrete exception classes raised by the installed ONNX Runtime."""
    global _ORT_EXCEPTIONS
    # Do not import ORT as a side effect of exception handling. A normal session creation has
    # already loaded it; direct library callers may also have imported it themselves.
    if not _ORT_EXCEPTIONS and "onnxruntime" in sys.modules:
        _ORT_EXCEPTIONS = _ort_exception_types()
    return _ORT_EXCEPTIONS

BACKENDS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}

_gpu_libraries_preloaded = set()


def session_creation_message(model_path: str, backend: str, trt_fp16: bool = False) -> str:
    """Describe the session that is about to be created, including TRT cache state."""
    if backend != "trt":
        return f"Creating ONNX Runtime session (backend={backend}) …"

    precision = "fp16" if trt_fp16 else "fp32"
    cache_dir = Path(model_path).parent / f"trt_engine_cache_{precision}"
    has_cache = cache_dir.exists() and any(cache_dir.glob("*.engine"))
    if has_cache:
        cache_state = f"TensorRT engine cache found: {cache_dir}"
    else:
        cache_state = "no TensorRT engine cache found; compilation may take several minutes"
    return f"Creating ONNX Runtime session (backend=trt, precision={precision}, {cache_state}) …"


def build_only_completion_message(backend: str) -> str:
    """Return an accurate completion message for ``--build-engine-only``."""
    if backend == "trt":
        return "TensorRT engine cache ready. Exiting."
    return f"ONNX Runtime session ready (backend={backend}; no engine cache required). Exiting."


def _preload_gpu_libraries(backend: str) -> None:
    """Make the pip-installed CUDA/cuDNN/TensorRT libraries discoverable in-process,
    on both Windows and Linux — no LD_LIBRARY_PATH or conda activation script needed.

    CUDA 12 + cuDNN 9 are loaded through ONNX Runtime's own cross-platform
    ``preload_dlls()`` (ORT >= 1.21). TensorRT is not covered by it, so ``tensorrt_libs``
    is placed on the native loader search path: prepended to PATH on Windows; on Linux
    the core TensorRT libraries are preloaded and locate their arch-specific
    builder-resource siblings via the wheel's RUNPATH (``$ORIGIN``). No-op for the
    ``cpu``/``openvino`` backends and when the GPU wheels are absent (e.g. a CPU-only
    install). Idempotent per stage.
    """
    if backend not in ("cuda", "trt"):
        return

    ort_module = _load_ort()

    # CUDA 12 + cuDNN 9 — ONNX Runtime's official cross-platform preloader.
    if "cuda_cudnn" not in _gpu_libraries_preloaded:
        _gpu_libraries_preloaded.add("cuda_cudnn")
        if hasattr(ort_module, "preload_dlls"):
            try:
                ort_module.preload_dlls()
            except Exception:
                pass

    if backend != "trt" or "tensorrt" in _gpu_libraries_preloaded:
        return
    _gpu_libraries_preloaded.add("tensorrt")

    trt_dir = Path(sysconfig.get_paths()["purelib"]) / "tensorrt_libs"
    if not trt_dir.is_dir():
        return
    if sys.platform == "win32":
        # ORT resolves its TensorRT provider's dependencies (nvinfer_*.dll) via PATH, and
        # TensorRT loads the arch-specific builder-resource DLL lazily by name, so the dir
        # must be on the search path here — preloading the libraries alone is not enough.
        os.add_dll_directory(str(trt_dir))
        os.environ["PATH"] = str(trt_dir) + os.pathsep + os.environ.get("PATH", "")
    else:
        # Preload the core libraries (nvinfer core first: plugin/parser depend on it). Their
        # arch-specific builder-resource siblings are then resolved via the wheel's RUNPATH.
        for stem in ("libnvinfer.so", "libnvinfer_plugin.so", "libnvonnxparser.so"):
            for lib in sorted(trt_dir.glob(stem + ".*")):
                try:
                    ctypes.CDLL(str(lib))
                except OSError:
                    pass


def create_session(
    model_path: str,
    backend: str = "cpu",
    trt_fp16: bool = False,
) -> Any:
    """Create an ONNX Runtime session with the appropriate providers."""
    try:
        ort_module = _load_ort()
    except (ImportError, OSError) as exc:
        raise SessionError("failed to import ONNX Runtime") from exc
    _preload_gpu_libraries(backend)
    providers = BACKENDS[backend]
    opts = ort_module.SessionOptions()
    opts.log_severity_level = 3
    ort_module.set_default_logger_severity(3)

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

    try:
        session = ort_module.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
            provider_options=provider_options,
        )
    except (*ort_exception_types(), OSError) as exc:
        # Expected operational failures: malformed/missing model, provider unavailable,
        # driver/engine mismatch. Programming bugs are left to propagate (-> EXIT_RUNTIME). The
        # message stays sober (backend only, no raw cause) — the cause is chained for the logs.
        raise SessionError(f"failed to create ONNX Runtime session (backend={backend})") from exc
    actual = session.get_providers()
    logger.info(f"      ONNX Runtime providers: {actual}")
    requested = providers[0]
    if requested not in actual:
        raise SessionError(
            f"requested backend provider is unavailable (backend={backend}, provider={requested})"
        )
    return session


def run_inference(
    session: Any,
    input_array: np.ndarray,
    anchors_batch: np.ndarray,
) -> list:
    """Run ONNX inference and return raw outputs."""
    try:
        return session.run(None, {"images": input_array, "anchors": anchors_batch})
    except ort_exception_types() as exc:
        # Expected ORT execution failures (GPU OOM, engine incompatibility, invalid binding).
        # Bugs on our side are left to propagate as EXIT_RUNTIME rather than masked here.
        raise InferenceError("ONNX inference failed") from exc


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
