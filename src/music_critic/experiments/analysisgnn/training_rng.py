"""Shared, failure-atomic RNG state for reproducible training.

Historical schema and error identifiers are retained for checkpoint compatibility.
No experimental fusion model or trainer is imported.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import random
from typing import Iterator
import numpy as np
import torch
from music_critic.experiments.pop909_cl_frozen.checkpoint import (
    FrozenCheckpointError as FusionResumeError, _preserve_caller_rng,
)

SAFE_RNG_SCHEMA = "HarmonicFusionSafeRNG@1.0.0"

def _fail(reason: str, detail: str = "") -> None:
    raise FusionResumeError(f"fusion_resume.{reason}", detail)


def capture_safe_rng() -> dict[str, object]:
    """Snapshot Python, NumPy and initialized Torch generators as safe values."""
    version, values, gaussian = random.getstate()
    kind, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "schema_version": SAFE_RNG_SCHEMA,
        "python": {"version": version, "state": list(values), "gaussian": gaussian},
        "numpy": {"kind": kind, "keys": torch.tensor(keys.astype(np.int64), dtype=torch.int64, device="cpu"),
                  "position": position, "has_gauss": has_gauss, "cached_gaussian": cached},
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": ([value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_initialized() else []),
    }


def prepare_training_resume_runtime(device: str, *, expected_cuda_device_count: int | None = None) -> dict[str, object]:
    """Explicit caller-owned setup for loading in a new process.

    The strict checkpoint loader never initializes CUDA. A CUDA dispatcher
    calls this function after authorizing its queue slot, before loading any
    checkpoint, and must state the expected visible device count. This setup
    uses only the current runtime's generators; no saved RNG is applied.
    Python/NumPy/CPU RNG and any pre-existing CUDA RNG remain unchanged.
    Initializing previously absent CUDA contexts is an explicit side effect.
    """
    if device not in ("cpu", "cuda"):
        _fail("runtime_device_invalid")
    if device == "cpu":
        if expected_cuda_device_count is not None:
            _fail("runtime_cuda_count_not_applicable")
        return {"schema_version": "HarmonicFusionResumeRuntime@1.0.0", "device_type": "cpu",
                "cuda_initialized_by_call": False, "cuda_device_count": 0,
                "verified_cuda_ordinals": [], "cuda_runtime_accessed": False,
                "saved_rng_applied": False}
    if type(expected_cuda_device_count) is not int or expected_cuda_device_count <= 0:
        _fail("runtime_expected_cuda_count_required")
    if not torch.cuda.is_available():
        _fail("runtime_cuda_unavailable")
    count = torch.cuda.device_count()
    if type(count) is not int or count != expected_cuda_device_count:
        _fail("runtime_cuda_inventory_mismatch", "visible devices differ from the explicit reservation")
    initialized_before = torch.cuda.is_initialized()
    with preserve_caller_rng():
        if not initialized_before:
            torch.cuda.init()
        if not torch.cuda.is_initialized():
            _fail("runtime_cuda_initialization_failed")
        if torch.cuda.device_count() != expected_cuda_device_count:
            _fail("runtime_cuda_inventory_mismatch", "visible device count changed during initialization")
        # This checks every visible ordinal on a separate scratch generator;
        # malformed or incomplete runtime state cannot be declared ready.
        state = capture_safe_rng()
        _validate_safe_rng(state, device_type="cuda")
    return {"schema_version": "HarmonicFusionResumeRuntime@1.0.0", "device_type": "cuda",
            "cuda_initialized_by_call": not initialized_before, "cuda_device_count": count,
            "verified_cuda_ordinals": list(range(count)), "cuda_runtime_accessed": True,
            "saved_rng_applied": False}


def _validate_safe_rng(state: object, *, device_type: str | None = None) -> None:
    """Validate every saved generator before changing any caller RNG.

    The ordered ``torch_cuda`` list is the complete generator inventory:
    entry i belongs to CUDA ordinal i. A CUDA trainer must provide all of it;
    an empty list is only a valid snapshot for a context not requiring CUDA.
    No device is initialized here merely to make an incomplete snapshot pass.
    """
    if device_type not in (None, "cpu", "cuda"):
        _fail("rng_device_type_invalid")
    if (type(state) is not dict or set(state) != {
            "schema_version", "python", "numpy", "torch_cpu", "torch_cuda"}
            or state["schema_version"] != SAFE_RNG_SCHEMA):
        _fail("rng_invalid", "exact safe RNG envelope required")
    py = state["python"]
    if (type(py) is not dict or set(py) != {"version", "state", "gaussian"}
            or type(py["version"]) is not int or py["version"] != 3
            or type(py["state"]) is not list or len(py["state"]) != 625
            or any(type(value) is not int or not 0 <= value < 2 ** 32 for value in py["state"][:-1])
            or type(py["state"][-1]) is not int or not 0 <= py["state"][-1] <= 624
            or (py["gaussian"] is not None and
                (type(py["gaussian"]) is not float or not math.isfinite(py["gaussian"])) )):
        _fail("rng_python_invalid")
    np_state = state["numpy"]
    if (type(np_state) is not dict or set(np_state) != {
            "kind", "keys", "position", "has_gauss", "cached_gaussian"}
            or np_state["kind"] != "MT19937"):
        _fail("rng_numpy_invalid")
    keys = np_state["keys"]
    if (type(keys) is not torch.Tensor or keys.device.type != "cpu"
            or keys.layout != torch.strided or keys.dtype != torch.int64
            or keys.shape != (624,) or bool(((keys < 0) | (keys >= 2 ** 32)).any())
            or type(np_state["position"]) is not int or not 0 <= np_state["position"] <= 624
            or type(np_state["has_gauss"]) is not int or np_state["has_gauss"] not in (0, 1)
            or type(np_state["cached_gaussian"]) is not float
            or not math.isfinite(np_state["cached_gaussian"])):
        _fail("rng_numpy_invalid")
    cpu = state["torch_cpu"]
    cuda = state["torch_cuda"]
    if type(cuda) is not list:
        _fail("rng_cuda_invalid")
    if device_type == "cuda" and not cuda:
        _fail("rng_cuda_inventory_missing", "CUDA trainer requires a nonempty complete RNG inventory")
    if cuda:
        if not torch.cuda.is_initialized():
            _fail("rng_cuda_runtime_mismatch", "matching CUDA generators must already be initialized")
        count = torch.cuda.device_count()
        if type(count) is not int or count <= 0 or len(cuda) != count:
            _fail("rng_cuda_runtime_mismatch", "one saved RNG state per CUDA ordinal required")
    for value in [cpu, *cuda]:
        if (type(value) is not torch.Tensor or value.dtype != torch.uint8
                or value.layout != torch.strided or value.device.type != "cpu"
                or value.ndim != 1 or not value.numel()):
            _fail("rng_torch_invalid")
    try:
        random.Random().setstate((py["version"], tuple(py["state"]), py["gaussian"]))
        np.random.RandomState().set_state(_numpy_state(np_state))
        torch.Generator(device="cpu").set_state(cpu)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise FusionResumeError("fusion_resume.rng_invalid", "generator state rejected") from exc
    if cuda:
        try:
            for index, value in enumerate(cuda):
                torch.Generator(device=f"cuda:{index}").set_state(value)
        except (RuntimeError, ValueError, TypeError) as exc:
            raise FusionResumeError("fusion_resume.rng_cuda_invalid", "CUDA state rejected") from exc


def _numpy_state(state):
    return (state["kind"], state["keys"].numpy().astype(np.uint32), state["position"],
            state["has_gauss"], state["cached_gaussian"])


def _apply_rng(state) -> None:
    py = state["python"]
    random.setstate((py["version"], tuple(py["state"]), py["gaussian"]))
    np.random.set_state(_numpy_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def restore_safe_rng(state: object, *, device_type: str | None = None) -> None:
    """Validate all generators first, then restore them with rollback on failure."""
    _validate_safe_rng(state, device_type=device_type)
    before = capture_safe_rng()
    try:
        _apply_rng(state)
    except BaseException:
        _apply_rng(before)
        raise


@contextmanager
def preserve_caller_rng() -> Iterator[None]:
    with _preserve_caller_rng():
        yield

