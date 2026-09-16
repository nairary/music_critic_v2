"""Restricted, failure-atomic full checkpoints for bounded fusion continuation.

This format is deliberately separate from historical C2 and model-only files.
It restores a new session, never mutates a caller's live trainer, and never
applies saved RNG to the caller. Device changes are not exact resume.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch

from music_critic.experiments.pop909_cl_frozen.checkpoint import (
    FrozenCheckpointError as FusionResumeError,
    _checked_artifact,
    _preserve_caller_rng,
    _validate_model_contract,
    _validate_model_state,
)

from .contracts import canonical_json, fingerprint
from .corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from .full_orbit_training import FullOrbitSampler
from .harmonic_logit_fusion import fusion_model_contract

if TYPE_CHECKING:
    from .harmonic_fusion_training import FusionTrainingSession


FUSION_TRAINING_CHECKPOINT_SCHEMA = "HarmonicFusionTrainingCheckpoint@1.1.0"
SAFE_RNG_SCHEMA = "HarmonicFusionSafeRNG@1.0.0"
_FIELDS = {
    "schema_version", "model_contract", "model_contract_fingerprint",
    "model_state", "model_state_fingerprint", "optimizer_contract",
    "optimizer_state", "scheduler_state", "scaler_state", "config",
    "config_fingerprint", "bindings", "bindings_fingerprint", "applied_updates",
    "sampler_state", "rng_state", "history", "validations",
    "source_model_state", "device_type", "optimizer_steps", "pending_validation",
}


def _fail(reason: str, detail: str = "") -> None:
    raise FusionResumeError(f"fusion_resume.{reason}", detail)


def _same_json(left: object, right: object, reason: str) -> None:
    try:
        same = canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError, RecursionError) as exc:
        raise FusionResumeError(f"fusion_resume.{reason}", "canonical JSON required") from exc
    if not same:
        _fail(reason)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return deepcopy(value)


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


def _optimizer_contract(session) -> dict[str, object]:
    return {
        "type": "torch.optim.AdamW", "scheduler": "torch.optim.lr_scheduler.LambdaLR",
        "precision": "fp32", "scaler_enabled": False,
        "ordered_parameter_names": [name for name, _ in session.model.named_parameters()],
    }


def _lr(config, step: int) -> float:
    from .harmonic_fusion_training import schedule_factor

    return config.peak_lr * schedule_factor(config, step)


def _validate_progress(payload, session) -> None:
    updates = payload["applied_updates"]
    config = session.config
    if type(updates) is not int or not 0 <= updates <= config.total_updates:
        _fail("updates_invalid")
    expected_sampler = FullOrbitSampler(session.sampler.table, seed=config.seed,
                                       position=updates * config.batch_size).state_dict()
    _same_json(payload["sampler_state"], expected_sampler, "sampler_mismatch")
    history = payload["history"]
    if type(history) is not list or len(history) != updates:
        _fail("history_invalid")
    sampler = FullOrbitSampler(session.sampler.table, seed=config.seed)
    for index, row in enumerate(history):
        if type(row) is not dict or set(row) != {"update", "draws", "loss", "lr_used", "lr_next"}:
            _fail("history_invalid")
        expected = {"update": index + 1,
                    "draws": [asdict(sampler.peek(offset)) for offset in range(config.batch_size)],
                    "loss": row["loss"], "lr_used": _lr(config, index), "lr_next": _lr(config, index + 1)}
        if type(row["loss"]) is not float or not math.isfinite(row["loss"]):
            _fail("history_invalid", "finite scalar loss required")
        _same_json(row, expected, "history_mismatch")
        sampler.advance_after_applied_update(config.batch_size)
    validations = payload["validations"]
    if type(validations) is not list:
        _fail("validations_invalid")
    previous = -1
    for row in validations:
        if (type(row) is not dict or set(row) != {"update", "metrics"}
                or type(row.get("update")) is not int or type(row["metrics"]) is not dict):
            _fail("validations_invalid")
        update = row["update"]
        if update <= previous or update > updates or update < 0:
            _fail("validations_invalid")
        if update not in (0, config.total_updates) and update % config.validation_interval:
            _fail("validations_invalid")
        _same_json(row, row, "validations_invalid")
        score = row["metrics"].get("corrected_primary_macro_score", "missing")
        if score is not None and (type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1):
            _fail("validations_invalid")
        previous = update
    from .harmonic_fusion_training import pending_validation_update
    try:
        pending = pending_validation_update(config, updates, validations)
    except ValueError as exc:
        raise FusionResumeError("fusion_resume.validation_schedule_incomplete", str(exc)) from exc
    _same_json(payload["pending_validation"], pending, "pending_validation_mismatch")


def _validate_optimizer(payload, session) -> None:
    _same_json(payload["optimizer_contract"], _optimizer_contract(session), "optimizer_contract_mismatch")
    supplied = payload["optimizer_state"]
    expected = session.optimizer.state_dict()
    if type(supplied) is not dict or set(supplied) != {"state", "param_groups"}:
        _fail("optimizer_state_invalid")
    groups = deepcopy(expected["param_groups"])
    groups[0]["lr"] = _lr(session.config, payload["applied_updates"])
    _same_json(supplied["param_groups"], groups, "optimizer_groups_mismatch")
    states = supplied["state"]
    parameters = list(session.model.named_parameters())
    counters = payload["optimizer_steps"]
    if (type(counters) is not dict or list(counters) != [name for name, _ in parameters]
            or any(type(value) is not int or not 0 <= value <= payload["applied_updates"]
                   for value in counters.values())):
        _fail("optimizer_steps_invalid")
    if (type(states) is not dict or any(type(key) is not int or not 0 <= key < len(parameters) for key in states)
            or (payload["applied_updates"] == 0 and states)):
        _fail("optimizer_state_invalid")
    if set(states) != {index for index, (name, _) in enumerate(parameters) if counters[name] > 0}:
        _fail("optimizer_state_inventory_mismatch")
    for key, state in states.items():
        name, parameter = parameters[key]
        if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            _fail("optimizer_slots_invalid", name)
        step = state["step"]
        if (type(step) is not torch.Tensor or step.dtype != torch.float32
                or step.layout != torch.strided or step.device.type != "cpu" or step.shape != ()
                or not math.isfinite(step.item()) or not step.item().is_integer()
                or not 1 <= step.item() <= payload["applied_updates"]
                or step.item() != counters[name]):
            _fail("optimizer_step_invalid", name)
        for slot in ("exp_avg", "exp_avg_sq"):
            tensor = state[slot]
            if (type(tensor) is not torch.Tensor or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype or tensor.layout != torch.strided
                    or tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all())
                    or (slot == "exp_avg_sq" and bool((tensor < 0).any()))):
                _fail("optimizer_tensor_invalid", f"{name}:{slot}")
    scheduler = deepcopy(session.scheduler.state_dict())
    scheduler.update(last_epoch=payload["applied_updates"], _step_count=payload["applied_updates"] + 1,
                     _last_lr=[_lr(session.config, payload["applied_updates"])])
    _same_json(payload["scheduler_state"], scheduler, "scheduler_mismatch")
    _same_json(payload["scaler_state"], {}, "scaler_mismatch")


def _validate_payload(payload, session) -> None:
    if type(payload) is not dict or payload.get("schema_version") != FUSION_TRAINING_CHECKPOINT_SCHEMA:
        _fail("schema_invalid", "full fusion training checkpoint required")
    if set(payload) != _FIELDS:
        _fail("envelope_keys_mismatch")
    _same_json(payload["config"], session.config.to_dict(), "config_mismatch")
    if payload["config_fingerprint"] != fingerprint(session.config.to_dict()):
        _fail("config_mismatch")
    _same_json(payload["bindings"], session.bindings, "bindings_mismatch")
    if payload["bindings_fingerprint"] != fingerprint(session.bindings):
        _fail("bindings_mismatch")
    if payload["device_type"] != torch.device(session.device).type:
        _fail("device_mismatch", "cross-device continuation is not exact resume")
    source = payload["source_model_state"]
    if type(source) is not str or re.fullmatch(r"[0-9a-f]{64}", source) is None:
        _fail("source_model_state_invalid")
    expected_source = session.bindings.get("source_model_state_fingerprint")
    if expected_source is not None and source != expected_source:
        _fail("source_model_state_mismatch")
    contract = fusion_model_contract(session.model)
    if payload["model_contract_fingerprint"] != contract["fingerprint"]:
        _fail("model_contract_mismatch")
    _validate_model_contract(payload["model_contract"], contract)
    _validate_model_state(payload["model_state"], session.model)
    if payload["model_state_fingerprint"] != model_state_fingerprint(payload["model_state"]):
        _fail("model_state_fingerprint_mismatch")
    _validate_progress(payload, session)
    if payload["applied_updates"] == 0:
        base_state = {name.removeprefix("base."): value for name, value in payload["model_state"].items()
                      if name.startswith("base.")}
        if model_state_fingerprint(base_state) != source:
            _fail("initial_source_state_mismatch")
        if any(bool(value.ne(0).any()) for name, value in payload["model_state"].items()
               if name.startswith("fusion.")):
            _fail("initial_fusion_nonzero")
    _validate_optimizer(payload, session)
    _validate_safe_rng(payload["rng_state"], device_type=payload["device_type"])


def create_training_payload(session: FusionTrainingSession) -> dict[str, object]:
    """Detached complete snapshot; only successful applied-update boundaries."""
    contract = fusion_model_contract(session.model)
    state = _snapshot(session.model.state_dict())
    payload = {
        "schema_version": FUSION_TRAINING_CHECKPOINT_SCHEMA,
        "model_contract": contract, "model_contract_fingerprint": contract["fingerprint"],
        "model_state": state, "model_state_fingerprint": model_state_fingerprint(state),
        "optimizer_contract": _optimizer_contract(session),
        "optimizer_state": _snapshot(session.optimizer.state_dict()),
        "optimizer_steps": deepcopy(session.optimizer_steps),
        "scheduler_state": _snapshot(session.scheduler.state_dict()),
        "scaler_state": _snapshot(session.scaler.state_dict()),
        "config": session.config.to_dict(), "config_fingerprint": fingerprint(session.config.to_dict()),
        "bindings": deepcopy(session.bindings), "bindings_fingerprint": fingerprint(session.bindings),
        "applied_updates": session.applied_updates, "sampler_state": session.sampler.state_dict(),
        "rng_state": _snapshot(session.rng_state), "history": deepcopy(session.history),
        "validations": deepcopy(session.validations), "source_model_state": session.source_model_state,
        "pending_validation": session.pending_validation,
        "device_type": torch.device(session.device).type,
    }
    # Validation uses a fresh optimizer/scheduler reference, not progressed live
    # hyperparameters; no serialization is allowed to quietly normalize state.
    with preserve_caller_rng():
        reference = _fresh_session(payload, session.config, session.bindings, session.sampler.table, session.device)
        _validate_payload(payload, reference)
    return payload


def save_training_checkpoint(session: FusionTrainingSession, path) -> dict[str, object]:
    """Atomically publish one immutable file; existing paths are never replaced."""
    payload = create_training_payload(session)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}


def _fresh_session(payload, config, bindings, table, device):
    from .harmonic_fusion_training import FusionTrainingSession

    contract = payload.get("model_contract")
    if type(contract) is not dict or type(contract.get("enabled")) is not bool:
        _fail("model_contract_mismatch")
    with torch.device("cpu"):
        base = CorrectedAnalysisGNNModel().to(dtype=torch.float32)
    return FusionTrainingSession(base=base, enabled=contract["enabled"], config=config,
                                bindings=bindings, table=table, device=device)


def load_training_checkpoint(path, *, expected_sha256: str, expected_size_bytes: int,
                             config, bindings, table, device="cpu", expected_enabled=None):
    """Check exact bytes and every contract before restoring a new trainer."""
    with preserve_caller_rng():
        content = _checked_artifact(path, expected_sha256, expected_size_bytes)
        try:
            payload = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise FusionResumeError("fusion_resume.restricted_load_failed", "no unsafe fallback") from exc
        if type(payload) is not dict or payload.get("schema_version") != FUSION_TRAINING_CHECKPOINT_SCHEMA:
            _fail("schema_invalid", "full fusion training checkpoint required")
        if set(payload) != _FIELDS:
            _fail("envelope_keys_mismatch")
        # Fail before even constructing a fresh CUDA model when its saved
        # generator inventory is missing or cannot be verified on this runtime.
        _validate_safe_rng(payload["rng_state"], device_type=payload["device_type"])
        session = _fresh_session(payload, config, bindings, table, device)
        if expected_enabled is not None:
            if type(expected_enabled) is not bool or session.model.enabled is not expected_enabled:
                _fail("arm_mismatch")
        _validate_payload(payload, session)
        session.model.load_state_dict(payload["model_state"], strict=True)
        if model_state_fingerprint(session.model) != payload["model_state_fingerprint"]:
            _fail("loaded_state_mismatch")
        session.optimizer.load_state_dict(payload["optimizer_state"])
        session.scheduler.load_state_dict(payload["scheduler_state"])
        session.scaler.load_state_dict(payload["scaler_state"])
        session.sampler.load_state_dict(payload["sampler_state"])
        session.applied_updates = payload["applied_updates"]
        session.optimizer_steps = deepcopy(payload["optimizer_steps"])
        session.history = deepcopy(payload["history"])
        session.validations = deepcopy(payload["validations"])
        session.rng_state = _snapshot(payload["rng_state"])
        session.source_model_state = payload["source_model_state"]
        return session


__all__ = ["FUSION_TRAINING_CHECKPOINT_SCHEMA", "SAFE_RNG_SCHEMA", "FusionResumeError",
           "capture_safe_rng", "restore_safe_rng", "preserve_caller_rng", "prepare_training_resume_runtime",
           "create_training_payload", "save_training_checkpoint", "load_training_checkpoint"]
