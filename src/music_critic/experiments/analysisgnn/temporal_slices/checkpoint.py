"""Restricted, immutable full checkpoints for the temporal C2 experiment.

Only committed applied-update boundaries are saved. Worker queues and caches
are deliberately excluded: restart them from the restored sampler cursor.
CUDA must already be initialized by the authorized runner before CUDA resume.
The caller provides fresh components; validation precedes mutation and any
restoration failure rolls their state and all RNG domains back.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import torch

from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.corrected_model import corrected_model_contract, model_state_fingerprint
from music_critic.experiments.analysisgnn.training_rng import (
    capture_safe_rng, restore_safe_rng, _validate_safe_rng,
)
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
from music_critic.experiments.pop909_cl_frozen.checkpoint import _checked_artifact, _validate_model_state
from .graph import SCHEMA as GRAPH_SCHEMA, FEATURES, EDGES
from .model import MODEL_SCHEMA, TemporalSliceModel
from .alignment import ALIGNMENT_SCHEMA


CHECKPOINT_SCHEMA = "TemporalSliceTrainingCheckpoint@1.0.0"
_FIELDS = {
    "schema_version", "model_contract", "model_state", "model_state_fingerprint",
    "module_training", "optimizer_contract", "optimizer_state", "scheduler_state",
    "scaler_state", "config", "bindings", "runtime", "progress", "sampler_state",
    "rng_state", "state_fingerprint",
}
_PROGRESS_FIELDS = {
    "applied_updates", "draw_cursor", "validation_history", "best_score",
    "best_update", "pending_validation", "elapsed_seconds",
}


class TemporalCheckpointError(ValueError):
    """A checkpoint cannot establish an exact temporal training continuation."""


def _fail(reason):
    raise TemporalCheckpointError("temporal_checkpoint." + reason)


def _same(a, b, reason):
    try:
        equal = canonical_json(a) == canonical_json(b)
    except (TypeError, ValueError, RecursionError) as exc:
        raise TemporalCheckpointError("temporal_checkpoint.non_json_metadata") from exc
    if not equal:
        _fail(reason)


def _snapshot(value):
    if type(value) is torch.Tensor:
        if value.layout != torch.strided or not bool(torch.isfinite(value).all()):
            _fail("nonfinite_or_sparse_state")
        return value.detach().cpu().clone()
    if type(value) is dict:
        if any(type(k) not in (str, int) for k in value):
            _fail("unsupported_state_key")
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, dict):  # state_dict is an OrderedDict.
        return _snapshot(dict(value))
    if type(value) in (list, tuple):
        return type(value)(_snapshot(v) for v in value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    _fail("unsupported_or_nonfinite_state")


def temporal_model_contract(model):
    if type(model) is not TemporalSliceModel:
        _fail("temporal_model_required")
    contract = {
        "schema_version": MODEL_SCHEMA,
        "graph_schema": GRAPH_SCHEMA,
        "target_alignment_schema": ALIGNMENT_SCHEMA,
        "base": corrected_model_contract(model.base),
        "features": list(FEATURES), "edges": [list(e) for e in EDGES],
        "layer_destinations": [list(layer.destinations) for layer in model.temporal_layers],
        "task_routes": {spec.task_id: list(spec.node_types) for spec in model.base.task_specs},
        "parameter_layout": [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype),
                              "requires_grad": p.requires_grad}
                             for n, p in model.named_parameters()],
        "dropout_probabilities": {n: m.p for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)},
        "normalization_epsilons": {n: m.eps for n, m in model.named_modules() if isinstance(m, torch.nn.LayerNorm)},
        "heads": "same18;harmonic=slice_then_beat;cadence=slice;note=note",
    }
    contract["fingerprint"] = fingerprint(contract)
    return contract


def _device(model):
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1:
        _fail("mixed_model_devices")
    device = next(iter(devices))
    if device.type not in ("cpu", "cuda"):
        _fail("unsupported_device")
    return device


def _runtime(model):
    device = _device(model)
    cuda = device.type == "cuda"
    if cuda and not torch.cuda.is_initialized():
        _fail("cuda_runtime_not_initialized")
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "device": str(device), "cuda_build": torch.version.cuda,
        "cuda_devices": torch.cuda.device_count() if cuda else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def _optimizer_contract(model, optimizer, scheduler, scaler):
    if type(optimizer) is not torch.optim.AdamW or type(scheduler) is not torch.optim.lr_scheduler.LambdaLR:
        _fail("optimizer_scheduler_class_mismatch")
    if scheduler.optimizer is not optimizer:
        _fail("scheduler_optimizer_mismatch")
    names = {id(p): n for n, p in model.named_parameters()}
    groups, seen = [], []
    for group in optimizer.param_groups:
        try:
            ordered = [names[id(p)] for p in group["params"]]
        except KeyError as exc:
            raise TemporalCheckpointError("temporal_checkpoint.foreign_optimizer_parameter") from exc
        seen.extend(ordered)
        groups.append({"parameter_names": ordered, "options": {
            k: _snapshot(v) for k, v in group.items() if k not in ("params", "lr")}})
    if len(seen) != len(set(seen)) or set(seen) != set(names.values()):
        _fail("optimizer_parameter_inventory")
    return {"optimizer": "torch.optim.AdamW", "scheduler": "torch.optim.lr_scheduler.LambdaLR",
            "groups": groups, "scaler_enabled": bool(scaler.is_enabled())}


def _validate_progress(progress, config, sampler_state):
    if type(config) is not dict or type(progress) is not dict or set(progress) != _PROGRESS_FIELDS:
        _fail("progress_or_config_schema")
    for key in ("batch_size", "total_updates", "scheduler_total_updates", "seed"):
        if type(config.get(key)) is not int or config[key] < (0 if key == "seed" else 1):
            _fail("config_" + key)
    if config["scheduler_total_updates"] < config["total_updates"]:
        _fail("scheduler_horizon")
    update = progress["applied_updates"]
    if type(update) is not int or not 0 <= update <= config["total_updates"]:
        _fail("applied_updates")
    cursor = progress["draw_cursor"]
    if type(cursor) is not int or cursor != update * config["batch_size"]:
        _fail("committed_draw_cursor")
    if type(sampler_state) is not dict or sampler_state.get("position") != cursor:
        _fail("sampler_cursor")
    if type(sampler_state.get("position")) is not int or sampler_state.get("seed") != config["seed"]:
        _fail("sampler_seed_or_position")
    body = {k: v for k, v in sampler_state.items() if k != "fingerprint"}
    if sampler_state.get("fingerprint") != fingerprint(body):
        _fail("sampler_fingerprint")
    history = progress["validation_history"]
    if type(history) is not list:
        _fail("validation_history")
    last = -1
    for row in history:
        if type(row) is not dict or type(row.get("update")) is not int or not last < row["update"] <= update:
            _fail("validation_history_order")
        last = row["update"]
    _same(history, history, "validation_history_json")
    best, score = progress["best_update"], progress["best_score"]
    if (best is None) != (score is None):
        _fail("best_state")
    if best is not None and (type(best) is not int or not 0 <= best <= update
                            or type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1):
        _fail("best_state")
    pending = progress["pending_validation"]
    if pending is not None and (type(pending) is not int or pending != update or pending <= last):
        _fail("pending_validation")
    elapsed = progress["elapsed_seconds"]
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
        _fail("elapsed_seconds")


def _validate_optimizer_state(payload, model, optimizer, scheduler, scaler):
    _same(payload["optimizer_contract"], _optimizer_contract(model, optimizer, scheduler, scaler), "optimizer_contract")
    state = payload["optimizer_state"]
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        _fail("optimizer_state_schema")
    groups = state["param_groups"]
    reference = optimizer.state_dict()["param_groups"]
    if type(groups) is not list or len(groups) != len(reference):
        _fail("optimizer_groups")
    update = payload["progress"]["applied_updates"]
    rates = [base * function(update) for base, function in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    parameter_by_id = {}
    for group, expected, live, lr in zip(groups, reference, optimizer.param_groups, rates):
        expected = dict(expected, lr=lr)
        _same(group, expected, "optimizer_groups")
        parameter_by_id.update(zip(group["params"], live["params"]))
    slots = state["state"]
    if (type(slots) is not dict or not set(slots).issubset(parameter_by_id)
            or (update == 0 and slots) or (update > 0 and not slots)):
        _fail("optimizer_slot_inventory")
    for key, entry in slots.items():
        if type(key) is not int or type(entry) is not dict or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            _fail("optimizer_slot_schema")
        step = entry["step"]
        if type(step) is not torch.Tensor or step.shape != () or step.dtype != torch.float32 or step.device.type != "cpu":
            _fail("optimizer_step")
        if not math.isfinite(step.item()) or not step.item().is_integer() or not 1 <= step.item() <= update:
            _fail("optimizer_step")
        parameter = parameter_by_id[key]
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = entry[name]
            if (type(tensor) is not torch.Tensor or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype
                    or tensor.device.type != "cpu" or tensor.layout != torch.strided or not bool(torch.isfinite(tensor).all())):
                _fail("optimizer_tensor")
        if bool((entry["exp_avg_sq"] < 0).any()):
            _fail("optimizer_negative_second_moment")
    expected = deepcopy(scheduler.state_dict())
    expected.update(last_epoch=update, _step_count=update + 1, _last_lr=rates)
    _same(payload["scheduler_state"], expected, "scheduler_state")
    saved_scaler = payload["scaler_state"]
    if not scaler.is_enabled():
        _same(saved_scaler, {}, "disabled_scaler_state")
    else:
        expected_keys = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
        if type(saved_scaler) is not dict or set(saved_scaler) != expected_keys:
            _fail("scaler_state")
        current = scaler.state_dict()
        for name in ("growth_factor", "backoff_factor", "growth_interval"):
            if saved_scaler[name] != current[name]:
                _fail("scaler_config")
        if type(saved_scaler["scale"]) is not float or not math.isfinite(saved_scaler["scale"]) or saved_scaler["scale"] <= 0:
            _fail("scaler_scale")
        count = saved_scaler["_growth_tracker"]
        if type(count) is not int or not 0 <= count < saved_scaler["growth_interval"]:
            _fail("scaler_growth_tracker")


def _validate(payload, *, model, optimizer, scheduler, scaler, config, bindings, sampler=None):
    if type(payload) is not dict or payload.get("schema_version") != CHECKPOINT_SCHEMA or set(payload) != _FIELDS:
        _fail("temporal_checkpoint_schema_required")
    expected_hash = recursive_state_fingerprint({k: v for k, v in payload.items() if k != "state_fingerprint"})
    if payload["state_fingerprint"] != expected_hash:
        _fail("state_fingerprint")
    _same(payload["model_contract"], temporal_model_contract(model), "model_contract")
    _same(payload["config"], config, "config_binding")
    if type(bindings) is not dict or not bindings:
        _fail("bindings_required")
    _same(payload["bindings"], bindings, "data_source_environment_binding")
    _same(payload["runtime"], _runtime(model), "runtime_binding")
    _validate_model_state(payload["model_state"], model)
    if payload["model_state_fingerprint"] != model_state_fingerprint(payload["model_state"]):
        _fail("model_state_fingerprint")
    modes = payload["module_training"]
    if type(modes) is not dict or set(modes) != dict(model.named_modules()).keys() or any(type(v) is not bool for v in modes.values()):
        _fail("module_training_modes")
    _validate_progress(payload["progress"], config, payload["sampler_state"])
    _validate_optimizer_state(payload, model, optimizer, scheduler, scaler)
    _validate_safe_rng(payload["rng_state"], device_type=_device(model).type)
    if sampler is not None:
        # Validate on a clone before touching the live sampler or trainer.
        probe = deepcopy(sampler)
        probe.load_state_dict(payload["sampler_state"])
        _same(probe.state_dict(), payload["sampler_state"], "sampler_contract")


def create_temporal_payload(*, model, optimizer, scheduler, scaler, config, bindings, progress, sampler_state):
    """Capture a detached safe snapshot without advancing any generator."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA, "model_contract": temporal_model_contract(model),
        "model_state": _snapshot(model.state_dict()),
        "module_training": {n: m.training for n, m in model.named_modules()},
        "optimizer_contract": _optimizer_contract(model, optimizer, scheduler, scaler),
        "optimizer_state": _snapshot(optimizer.state_dict()), "scheduler_state": _snapshot(scheduler.state_dict()),
        "scaler_state": _snapshot(scaler.state_dict()), "config": _snapshot(config), "bindings": _snapshot(bindings),
        "runtime": _runtime(model), "progress": _snapshot(progress), "sampler_state": _snapshot(sampler_state),
        "rng_state": capture_safe_rng(),
    }
    payload["model_state_fingerprint"] = model_state_fingerprint(payload["model_state"])
    payload["state_fingerprint"] = recursive_state_fingerprint(payload)
    _validate(payload, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, config=config, bindings=bindings)
    return payload


def save_temporal_checkpoint(path, **components):
    """Publish one immutable file atomically; return its external byte binding.

    The runner stores this result in its manifest and updates a latest pointer
    separately. An existing checkpoint path is never replaced, including when
    its contents happen to match.
    """
    payload = create_temporal_payload(**components)
    buffer = BytesIO()
    torch.save(payload, buffer)
    content = buffer.getvalue()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix="." + target.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}


def load_temporal_checkpoint(path, *, expected_sha256, expected_size_bytes, model, optimizer,
                             scheduler, scaler, config, bindings, sampler=None, restore_rng=True):
    """Validate bytes and all bindings, then restore a complete training state.

    A successful call applies saved RNG by default. For inspection only,
    ``restore_rng=False`` returns it in the result without changing generators.
    Worker queues must be discarded and recreated after this call; the saved
    sampler is the last acknowledged boundary, never a prefetch reservation.
    """
    content = _checked_artifact(path, expected_sha256, expected_size_bytes)
    try:
        payload = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TemporalCheckpointError("temporal_checkpoint.restricted_load_failed") from exc
    _validate(payload, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
              config=config, bindings=bindings, sampler=sampler)
    before = {"model": _snapshot(model.state_dict()), "optimizer": _snapshot(optimizer.state_dict()),
              "scheduler": _snapshot(scheduler.state_dict()), "scaler": _snapshot(scaler.state_dict()),
              "rng": capture_safe_rng(), "modes": {n: m.training for n, m in model.named_modules()},
              "sampler": deepcopy(sampler.state_dict()) if sampler is not None else None}

    def apply(states, modes, sampler_state, rng):
        model.load_state_dict(states["model"], strict=True)
        optimizer.load_state_dict(states["optimizer"])
        scheduler.load_state_dict(states["scheduler"])
        scaler.load_state_dict(states["scaler"])
        for name, module in model.named_modules():
            module.training = modes[name]
        if sampler is not None:
            sampler.load_state_dict(sampler_state)
        if rng is not None:
            restore_safe_rng(rng, device_type=_device(model).type)

    try:
        apply({n: payload[n + "_state"] for n in ("model", "optimizer", "scheduler", "scaler")},
              payload["module_training"], payload["sampler_state"], payload["rng_state"] if restore_rng else None)
        if model_state_fingerprint(model) != payload["model_state_fingerprint"]:
            _fail("restored_model_fingerprint")
    except BaseException:
        apply(before, before["modes"], before["sampler"], before["rng"])
        raise
    return {"progress": deepcopy(payload["progress"]), "sampler_state": deepcopy(payload["sampler_state"]),
            "rng_state": _snapshot(payload["rng_state"]), "model_contract": deepcopy(payload["model_contract"]),
            "state_fingerprint": payload["state_fingerprint"]}


__all__ = ["CHECKPOINT_SCHEMA", "TemporalCheckpointError", "temporal_model_contract",
           "create_temporal_payload", "save_temporal_checkpoint", "load_temporal_checkpoint"]
