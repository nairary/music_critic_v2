#!/usr/bin/env python3
"""Emit non-portable bounded CUDA+AMP mechanics evidence for Phase 8A."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import torch
from torch import Tensor

from scripts.accept_phase8a_hierarchical_masking import (
    PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION,
    _graph_matches_snapshot,
    _model,
    _snapshot_graph,
)
from music_critic.device import (
    DEVICE_TRANSFER_CONTRACT_VERSION,
    RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION,
    resolve_runtime_device,
)
from music_critic.data import QualityFlag, TargetArray
from music_critic.graph import (
    RAW_FEATURE_REGISTRY,
    build_raw_graph,
    graph_fingerprint,
)
from music_critic.ssl.bounded_fixture import mutate_piece_pitch_group
from music_critic.ssl.contracts import (
    MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
    MASK_PLAN_CONTRACT_VERSION,
    MASK_POLICY_VERSION,
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
    SSL_CONTRACT_VERSION,
    SSLContractError,
    canonical_sha256,
)
from music_critic.ssl.data import (
    SSLBatch,
    SSLRawSample,
    collate_ssl_samples,
)
from music_critic.ssl.hierarchical_masking import (
    HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION,
    HIERARCHY_MASK_POLICIES,
    HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT,
    HIERARCHY_MASK_POLICY_VERSION,
    HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION,
    HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION,
    HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
    HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION,
    HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION,
    INDEPENDENT_NOTE_PITCH,
    CONTIGUOUS_BAR_PITCH_SPAN,
    SPAN_FINAL_CHOICE_RANK_METHOD,
    SPAN_POOL_MEMBERSHIP_RANK_METHOD,
    SPAN_SELECTION_METHOD,
    HierarchyMaskPolicyConfig,
    HierarchyMaskUnavailableError,
    build_batched_hierarchy_mask_resolutions,
)
from music_critic.ssl.hierarchy_fixture import (
    PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION,
    build_phase8a_hierarchy_fixture,
)
from music_critic.ssl.engine import (
    NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION,
    PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION,
    SSL_TRAINING_REPORT_VERSION,
    _build_no_leakage_mutation_evidence,
    _build_pitch_sensitive_reconstruction_evidence,
    _fp32_pitch_mutation_diagnostics,
    _pitch_reconstruction_loss_changed,
)
from music_critic.ssl.masking import (
    PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
    prepare_mask_binding,
    validate_prepared_mask_binding,
)
from music_critic.ssl.model import (
    PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION,
    SSL_MODEL_CONTRACT_VERSION,
    SSL_MODEL_OUTPUT_CONTRACT_VERSION,
)
from music_critic.ssl.objective import (
    ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION,
    MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION,
    REPRESENTATION_LOSS_CONTRACT_VERSION,
    SSL_OBJECTIVE_CONTRACT_VERSION,
)


PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION = "1.1.0"
_EXACT_RTX_3090_DEVICE_NAME = "NVIDIA GeForce RTX 3090"
_GLOBAL_SEED = 42
_EPOCH = 0
_MASK_RATE = 0.30
_EXPECTED_MASKED_PITCH_SLOTS = {
    ("note", "categorical", "pitch"),
    ("note", "categorical", "pitch_class"),
    ("note", "categorical", "octave"),
    ("note", "continuous", "track_relative_pitch"),
    ("track", "continuous", "mean_pitch"),
    ("track", "continuous", "pitch_std"),
    ("track", "continuous", "min_pitch"),
    ("track", "continuous", "max_pitch"),
}


def _single_policy_config(policy: str) -> HierarchyMaskPolicyConfig:
    return HierarchyMaskPolicyConfig.create(
        weights={policy: 1.0},
        min_span_bars=1,
        max_span_bars=2,
    )


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return None
    values = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    return values[0] if values else None


def _configure_cublas_determinism() -> None:
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif configured not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "phase8a.cuda.cublas_workspace_config_invalid"
        )


@contextmanager
def _preserved_deterministic_cuda_runtime() -> Iterator[None]:
    """Apply deterministic CUDA settings without polluting a pytest process."""

    previous_workspace_present = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cpu_rng = torch.get_rng_state().clone()
    previous_cuda_rng: list[Tensor] | None = None
    _configure_cublas_determinism()
    try:
        if torch.cuda.is_available():
            previous_cuda_rng = [
                value.clone() for value in torch.cuda.get_rng_state_all()
            ]
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.set_rng_state(previous_cpu_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = (
            previous_cudnn_deterministic
        )
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        if previous_workspace_present:
            assert previous_workspace is not None
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _assert_batch_device(batch: SSLBatch, device: torch.device) -> None:
    for store in batch.raw_graph_batch.stores:
        for value in store.values():
            if isinstance(value, Tensor) and value.device != device:
                raise RuntimeError(
                    "Phase 8A model-facing tensor is on the wrong device"
                )


def _tensor_equal(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and bool(torch.equal(left, right))
    )


def _values_bit_exact(left: object, right: object) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and _tensor_equal(left, right)
        )
    if is_dataclass(left) or is_dataclass(right):
        if (
            not is_dataclass(left)
            or not is_dataclass(right)
            or type(left) is not type(right)
        ):
            return False
        left_fields = fields(left)
        right_fields = fields(right)
        return (
            tuple(field.name for field in left_fields)
            == tuple(field.name for field in right_fields)
            and all(
                _values_bit_exact(
                    getattr(left, field.name),
                    getattr(right, field.name),
                )
                for field in left_fields
            )
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or tuple(left) != tuple(right)
        ):
            return False
        return all(
            _values_bit_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) or isinstance(
        right, (tuple, list)
    ):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _values_bit_exact(first, second)
                for first, second in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _outputs_bit_exact(
    left: Any,
    right: Any,
    *,
    allow_envelope_contract_difference: bool = False,
) -> bool:
    if not is_dataclass(left) or not is_dataclass(right):
        return False
    if (
        type(left) is not type(right)
        and not allow_envelope_contract_difference
    ):
        return False
    left_fields = fields(left)
    right_fields = fields(right)
    if tuple(field.name for field in left_fields) != tuple(
        field.name for field in right_fields
    ):
        return False
    for field in left_fields:
        if (
            field.name == "contract_version"
            and allow_envelope_contract_difference
        ):
            continue
        if not _values_bit_exact(
            getattr(left, field.name),
            getattr(right, field.name),
        ):
            return False
    return True


def _iter_tensors(
    value: object,
    *,
    location: str,
) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield location, value
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _iter_tensors(
                getattr(value, field.name),
                location=f"{location}.{field.name}",
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_tensors(
                child,
                location=f"{location}[{key!r}]",
            )
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _iter_tensors(
                child,
                location=f"{location}[{index}]",
            )


def _output_tensor_evidence(
    output: object,
    *,
    device: torch.device,
) -> dict[str, object]:
    tensors = tuple(_iter_tensors(output, location="output"))
    if not tensors:
        raise RuntimeError("Phase 8A forward emitted no tensors")
    wrong_device = [
        location
        for location, value in tensors
        if value.device != device
    ]
    nonfinite = [
        location
        for location, value in tensors
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
    ]
    if wrong_device:
        raise RuntimeError(
            "Phase 8A forward tensor is on the wrong device:"
            + ",".join(wrong_device)
        )
    if nonfinite:
        raise RuntimeError(
            "Phase 8A forward emitted non-finite tensors:"
            + ",".join(nonfinite)
        )
    return {
        "tensor_count": len(tensors),
        "all_tensors_on_cuda_0": True,
        "all_floating_tensors_finite": True,
    }


def _loss_evidence(
    output: Any,
    *,
    device: torch.device,
) -> dict[str, object]:
    tensors: dict[str, Tensor | None] = {
        "note_numerator": output.note_loss.numerator,
        "note_mean": output.note_loss.mean,
        "bar_numerator": output.bar_latent.loss.numerator,
        "bar_mean": output.bar_latent.loss.mean,
        "song_numerator": output.song_latent.loss.numerator,
        "song_mean": output.song_latent.loss.mean,
        "total": output.objective.total_loss,
    }
    for view in output.note_loss.view_losses:
        tensors[
            f"note_view_{view.decoder_view_index}_numerator"
        ] = view.loss.numerator
        tensors[
            f"note_view_{view.decoder_view_index}_mean"
        ] = view.loss.mean
    if any(value is None for value in tensors.values()):
        raise RuntimeError("Phase 8A required objective is unavailable")
    resolved = {
        name: value
        for name, value in tensors.items()
        if value is not None
    }
    if not all(
        value.dtype == torch.float32
        and value.ndim == 0
        and value.device == device
        and bool(torch.isfinite(value))
        for value in resolved.values()
    ):
        raise RuntimeError(
            "Phase 8A CUDA AMP objective did not remain finite FP32"
        )
    return {
        "all_required_objectives_float32": True,
        "all_required_objectives_finite": True,
        "all_required_objectives_on_cuda_0": True,
        "dtypes": {
            name: str(value.dtype) for name, value in resolved.items()
        },
        "values": {
            name: float(value.detach().cpu())
            for name, value in resolved.items()
        },
    }


def _gradient_evidence(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> dict[str, object]:
    required_prefixes = (
        "encoder.local_baseline.encoder.feature_encoder.",
        "encoder.local_baseline.encoder.layers.",
        "encoder.context_encoder.pooling.",
        "encoder.context_encoder.transformer.",
        "encoder.context_encoder.fusion.",
        "decoder.",
        "bar_projector_predictor.",
        "song_projector_predictor.",
    )
    named = tuple(model.named_parameters())
    present = tuple(
        (name, parameter.grad)
        for name, parameter in named
        if parameter.grad is not None
    )
    if not present:
        raise RuntimeError("Phase 8A CUDA AMP gradients are absent")
    wrong_device = tuple(
        name
        for name, gradient in present
        if gradient is not None and gradient.device != device
    )
    nonfinite = tuple(
        name
        for name, gradient in present
        if gradient is not None
        and not bool(torch.isfinite(gradient).all())
    )
    if wrong_device:
        raise RuntimeError(
            "Phase 8A CUDA AMP gradient is on the wrong device:"
            + ",".join(wrong_device)
        )
    if nonfinite:
        raise RuntimeError(
            "Phase 8A CUDA AMP gradient is non-finite:"
            + ",".join(nonfinite)
        )
    groups: dict[str, bool] = {}
    for prefix in required_prefixes:
        groups[prefix] = any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad))
            for name, parameter in named
            if name.startswith(prefix)
        )
    token = dict(named)["feature_mask_token"]
    token_ok = (
        token.grad is not None
        and bool(torch.count_nonzero(token.grad))
    )
    if not all(groups.values()) or not token_ok:
        raise RuntimeError(
            "Phase 8A CUDA AMP required trainable path lacks gradient"
        )
    return {
        "present_gradient_tensor_count": len(present),
        "all_present_gradients_on_cuda_0": True,
        "all_present_gradients_finite": True,
        "required_module_groups": groups,
        "feature_mask_token_finite_nonzero": token_ok,
        "all_expected_paths_finite_nonzero": True,
    }


def _overlay_closes_pitch_dependencies(binding: Any) -> bool:
    present = {
        (slot.node_type, slot.kind, slot.feature_name)
        for slot in binding.feature_overlay.slot_masks
        if slot.global_node_indices
    }
    return _EXPECTED_MASKED_PITCH_SLOTS <= present


def _raw_sample(graph: Any, sample: SSLRawSample) -> SSLRawSample:
    return SSLRawSample(
        raw_graph=graph,
        raw_graph_fingerprint=graph_fingerprint(graph),
        dataset_id=sample.dataset_id,
        piece_id=sample.piece_id,
    )


def _target_blind_pair(
    piece: Any,
    sample: SSLRawSample,
) -> tuple[SSLBatch, SSLBatch]:
    changed_piece = replace(
        piece,
        source_path="ignored/phase8a-cuda-sidecar.mid",
        targets=(
            TargetArray(
                target_id="target:phase8a-cuda-inert",
                task="quality.overall",
                annotation_view_id=None,
                alignment_type="piece",
                entity_ids=(piece.piece_id,),
                value_type="scalar",
                class_labels=None,
                values=(0.75,),
                mask=(True,),
                confidence=(1.0,),
                source=("synthetic",),
                provenance=(piece.provenance[0].provenance_id,),
            ),
        ),
        provenance=(
            replace(
                piece.provenance[0],
                source="phase8a_cuda_sidecar_mutation",
                details=(("diagnostic", "changed"),),
            ),
        ),
        quality_flags=(
            QualityFlag(
                code="phase8a.cuda.diagnostic",
                severity="info",
                message="target-blind CUDA sidecar mutation",
                entity_ids=(piece.piece_id,),
                provenance_id=piece.provenance[0].provenance_id,
            ),
        ),
    )
    changed_graph = build_raw_graph(changed_piece)
    if graph_fingerprint(changed_graph) != graph_fingerprint(
        sample.raw_graph
    ):
        raise RuntimeError(
            "target/provenance sidecars entered the raw graph boundary"
        )
    return (
        collate_ssl_samples((sample,)),
        collate_ssl_samples((_raw_sample(changed_graph, sample),)),
    )


def _target_provenance_blindness(
    source_batch: SSLBatch,
    changed_batch: SSLBatch,
    *,
    policy: str,
    device: torch.device,
) -> bool:
    config = _single_policy_config(policy)
    kwargs = {
        "policy_config": config,
        "global_seed": _GLOBAL_SEED,
        "epoch": _EPOCH,
        "requested_mask_rate": _MASK_RATE,
        "stage": "train",
    }
    source_binding = prepare_hierarchy_mask_binding(
        source_batch,
        **kwargs,
    )
    changed_binding = prepare_hierarchy_mask_binding(
        changed_batch,
        **kwargs,
    )
    if source_binding.to_dict() != changed_binding.to_dict():
        return False
    source, moved_source = move_ssl_batch_with_prepared_binding(
        source_batch,
        source_binding,
        device,
    )
    changed, moved_changed = move_ssl_batch_with_prepared_binding(
        changed_batch,
        changed_binding,
        device,
    )
    model = _model().to(device).eval()
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        source_output = model.forward_hierarchy(
            source,
            prepared_mask_binding=moved_source,
        )
        changed_output = model.forward_hierarchy(
            changed,
            prepared_mask_binding=moved_changed,
        )
    _output_tensor_evidence(source_output, device=device)
    _output_tensor_evidence(changed_output, device=device)
    return _outputs_bit_exact(source_output, changed_output)


def _pitch_mutation_evidence(
    piece: Any,
    sample: SSLRawSample,
    *,
    policy: str,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    source_batch = collate_ssl_samples((sample,))
    source_cpu_snapshot = _snapshot_graph(source_batch.raw_graph_batch)
    config = _single_policy_config(policy)
    kwargs = {
        "policy_config": config,
        "global_seed": _GLOBAL_SEED,
        "epoch": _EPOCH,
        "requested_mask_rate": _MASK_RATE,
        "stage": "train",
    }
    source_binding = prepare_hierarchy_mask_binding(
        source_batch,
        **kwargs,
    )
    selected = source_binding.mask_plans[
        0
    ].selected_local_node_indices
    mutation = mutate_piece_pitch_group(piece, selected)
    changed_batch = collate_ssl_samples(
        (_raw_sample(mutation.mutated_raw_graph, sample),)
    )
    changed_cpu_snapshot = _snapshot_graph(changed_batch.raw_graph_batch)
    changed_binding = prepare_hierarchy_mask_binding(
        changed_batch,
        **kwargs,
    )
    if source_binding.to_dict() != changed_binding.to_dict():
        raise RuntimeError(
            "coherent pitch mutation changed the portable mask binding"
        )
    source, moved_source = move_ssl_batch_with_prepared_binding(
        source_batch,
        source_binding,
        device,
    )
    changed, moved_changed = move_ssl_batch_with_prepared_binding(
        changed_batch,
        changed_binding,
        device,
    )
    source_cuda_snapshot = _snapshot_graph(source.raw_graph_batch)
    changed_cuda_snapshot = _snapshot_graph(changed.raw_graph_batch)
    model = _model().to(device).eval()
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        source_output = model.forward_hierarchy(
            source,
            prepared_mask_binding=moved_source,
        )
        changed_output = model.forward_hierarchy(
            changed,
            prepared_mask_binding=moved_changed,
        )
    _output_tensor_evidence(source_output, device=device)
    _output_tensor_evidence(changed_output, device=device)
    online_equal = all(
        torch.equal(
            source_output.online_encoder.fused.embeddings[node_type],
            changed_output.online_encoder.fused.embeddings[node_type],
        )
        for node_type in source_output.online_encoder.fused.embeddings
    )
    predictions_equal = all(
        torch.equal(left, right)
        for left, right in zip(
            source_output.decoder_predictions,
            changed_output.decoder_predictions,
            strict=True,
        )
    )
    selected_indices = source_output.selected_global_note_indices
    source_selected_target = source_output.targets.note.index_select(
        0,
        selected_indices,
    )
    changed_selected_target = changed_output.targets.note.index_select(
        0,
        selected_indices,
    )
    target_changed = not torch.equal(
        source_selected_target,
        changed_selected_target,
    )
    source_loss = source_output.note_loss.mean
    changed_loss = changed_output.note_loss.mean
    if source_loss is None or changed_loss is None:
        raise RuntimeError(
            "Phase 8A pitch mutation loss is unavailable"
        )
    loss_changed = _pitch_reconstruction_loss_changed(
        source_loss,
        changed_loss,
    )
    diagnostics = _fp32_pitch_mutation_diagnostics(
        source_output.decoder_predictions,
        source_selected_target,
        changed_selected_target,
    )
    raw_graphs_unchanged = all(
        (
            _graph_matches_snapshot(
                source_batch.raw_graph_batch,
                source_cpu_snapshot,
            ),
            _graph_matches_snapshot(
                changed_batch.raw_graph_batch,
                changed_cpu_snapshot,
            ),
            _graph_matches_snapshot(
                source.raw_graph_batch,
                source_cuda_snapshot,
            ),
            _graph_matches_snapshot(
                changed.raw_graph_batch,
                changed_cuda_snapshot,
            ),
        )
    )
    runtime_source_binding = {
        "scope": "phase8a_cuda_runtime_to_rebuilt_canonical_source",
        "runtime_source_raw_graph_fingerprint": (
            sample.raw_graph_fingerprint
        ),
        "rebuilt_source_raw_graph_fingerprint": (
            mutation.source_raw_graph_fingerprint
        ),
        "identity_exact": (
            mutation.source_piece.dataset_name == sample.dataset_id
            and mutation.source_piece.piece_id == sample.piece_id
        ),
        "fingerprint_exact": (
            mutation.source_raw_graph_fingerprint
            == sample.raw_graph_fingerprint
        ),
    }
    runtime_source_binding["passed"] = (
        runtime_source_binding["identity_exact"]
        and runtime_source_binding["fingerprint_exact"]
    )
    if not (
        online_equal
        and predictions_equal
        and target_changed
        and loss_changed
        and raw_graphs_unchanged
        and runtime_source_binding["passed"]
        and diagnostics["metrics_finite"]
    ):
        raise RuntimeError(
            "Phase 8A coherent pitch mutation evidence failed"
        )
    no_leakage = _build_no_leakage_mutation_evidence(
        {
            "applicable": True,
            "mutation_applicable": True,
            "unavailable_reason": None,
            "raw_graph_stores_bit_exact_after_view": (
                raw_graphs_unchanged
            ),
            "runtime_source_binding": runtime_source_binding,
            "fixed_mask_plan": (
                source_binding.mask_plans == changed_binding.mask_plans
            ),
            "fixed_prepared_binding_fingerprint": (
                source_binding.fingerprint == changed_binding.fingerprint
            ),
            "prepared_mask_binding_fingerprint": (
                source_binding.fingerprint
            ),
            "online_embeddings_bit_exact_after_masked_mutation": (
                online_equal
            ),
            "online_predictions_bit_exact_after_masked_mutation": (
                predictions_equal
            ),
            "full_view_target_changed": target_changed,
            "metrics_finite": diagnostics["metrics_finite"],
            "mutation_policy_fingerprint": mutation.policy_fingerprint,
            "mutation_instance_fingerprint": (
                mutation.mutation_instance_fingerprint
            ),
        }
    )
    pitch_sensitive = _build_pitch_sensitive_reconstruction_evidence(
        {
            "applicable": True,
            "mutation_applicable": True,
            "unavailable_reason": None,
            "full_view_target_changed": target_changed,
            "reconstruction_loss_changed": loss_changed,
            "source_note_loss": float(source_loss.detach().cpu()),
            "mutated_note_loss": float(changed_loss.detach().cpu()),
            **diagnostics,
            "mutation_policy_fingerprint": mutation.policy_fingerprint,
            "mutation_instance_fingerprint": (
                mutation.mutation_instance_fingerprint
            ),
        }
    )
    if no_leakage["passed"] is not True:
        raise RuntimeError("Phase 8A no-leakage evidence did not pass")
    if pitch_sensitive["passed"] is not True:
        raise RuntimeError(
            "Phase 8A pitch-sensitive evidence did not pass"
        )
    if no_leakage is pitch_sensitive:
        raise AssertionError("mutation evidence domains alias")
    return no_leakage, pitch_sensitive


def _mutate_unmasked_velocity(batch: SSLBatch) -> SSLBatch:
    graph = deepcopy(batch.raw_graph_batch)
    start = int(graph["note"].ptr[0].item())
    end = int(graph["note"].ptr[1].item())
    if start >= end:
        raise RuntimeError("Phase 8A source sample has no notes")
    column = RAW_FEATURE_REGISTRY.names(
        "note",
        "continuous",
    ).index("velocity")
    available = graph["note"].x_cont_available[start:end, column]
    available_rows = torch.nonzero(
        available,
        as_tuple=False,
    ).flatten()
    if int(available_rows.numel()) == 0:
        raise RuntimeError(
            "Phase 8A source sample has no available velocity"
        )
    row = start + int(available_rows[0].item())
    availability_before = graph["note"].x_cont_available.detach().clone()
    graph["note"].x_cont[row, column] += 5.0
    if not torch.equal(
        graph["note"].x_cont_available,
        availability_before,
    ):
        raise RuntimeError(
            "Phase 8A source isolation changed feature availability"
        )
    return SSLBatch(
        raw_graph_batch=graph,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        sample_count=batch.sample_count,
        node_count=batch.node_count,
        edge_count=batch.edge_count,
    )


def _source_sample_isolation(
    source_batch: SSLBatch,
    original_output: Any,
    *,
    policy: str,
    model: torch.nn.Module,
    device: torch.device,
) -> bool:
    changed_cpu = _mutate_unmasked_velocity(source_batch)
    binding = prepare_hierarchy_mask_binding(
        changed_cpu,
        policy_config=_single_policy_config(policy),
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    changed_batch, moved = move_ssl_batch_with_prepared_binding(
        changed_cpu,
        binding,
        device,
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        changed = model.forward_hierarchy(
            changed_batch,
            prepared_mask_binding=moved,
        )
    _output_tensor_evidence(changed, device=device)
    sample_zero_changed = False
    for node_type, before in (
        original_output.online_encoder.fused.embeddings.items()
    ):
        membership = (
            original_output.online_encoder.fused.batch_membership[
                node_type
            ]
        )
        source_rows = torch.nonzero(
            membership == 0,
            as_tuple=False,
        ).flatten()
        other_rows = torch.nonzero(
            membership != 0,
            as_tuple=False,
        ).flatten()
        if source_rows.numel() and not torch.equal(
            before.index_select(0, source_rows),
            changed.online_encoder.fused.embeddings[
                node_type
            ].index_select(0, source_rows),
        ):
            sample_zero_changed = True
        if not torch.equal(
            before.index_select(0, other_rows),
            changed.online_encoder.fused.embeddings[
                node_type
            ].index_select(0, other_rows),
        ):
            return False
    selected_membership = (
        original_output.online_encoder.fused.batch_membership[
            "note"
        ].index_select(
            0,
            original_output.selected_global_note_indices,
        )
    )
    other_selected = torch.nonzero(
        selected_membership != 0,
        as_tuple=False,
    ).flatten()
    if any(
        not torch.equal(
            before.index_select(0, other_selected),
            after.index_select(0, other_selected),
        )
        for before, after in zip(
            original_output.decoder_predictions,
            changed.decoder_predictions,
            strict=True,
        )
    ):
        return False
    return sample_zero_changed


@contextmanager
def _guard_host_materialization(graph: Any) -> Iterator[None]:
    graph_tensor_ids: set[int] = set()
    graph_storage_tokens: set[tuple[str, int | None, int, int]] = set()
    for store in graph.stores:
        for value in store.values():
            if not isinstance(value, Tensor):
                continue
            graph_tensor_ids.add(id(value))
            storage = value.untyped_storage()
            if storage.nbytes() > 0:
                graph_storage_tokens.add(
                    (
                        value.device.type,
                        value.device.index,
                        storage.data_ptr(),
                        storage.nbytes(),
                    )
                )

    def belongs_to_graph(value: Tensor) -> bool:
        if id(value) in graph_tensor_ids:
            return True
        storage = value.untyped_storage()
        if storage.nbytes() == 0:
            return False
        return (
            value.device.type,
            value.device.index,
            storage.data_ptr(),
            storage.nbytes(),
        ) in graph_storage_tokens

    original_cpu = Tensor.cpu
    original_tolist = Tensor.tolist
    original_item = Tensor.item
    original_to = Tensor.to

    def guarded_cpu(value: Tensor, *args: object, **kwargs: object):
        if belongs_to_graph(value) or value.numel() > 1:
            raise AssertionError(
                "bulk tensor cpu() in prepared forward"
            )
        return original_cpu(value, *args, **kwargs)

    def guarded_tolist(value: Tensor, *args: object, **kwargs: object):
        if belongs_to_graph(value) or value.numel() > 1:
            raise AssertionError(
                "bulk tensor tolist() in prepared forward"
            )
        return original_tolist(value, *args, **kwargs)

    def guarded_item(value: Tensor, *args: object, **kwargs: object):
        if belongs_to_graph(value):
            raise AssertionError("graph tensor item() in prepared forward")
        return original_item(value, *args, **kwargs)

    def guarded_to(value: Tensor, *args: object, **kwargs: object):
        requested = kwargs.get("device")
        if requested is None and args and isinstance(
            args[0], (str, torch.device)
        ):
            requested = args[0]
        if (
            requested is not None
            and torch.device(requested).type == "cpu"
            and (
                belongs_to_graph(value)
                or (
                    value.device.type != "cpu"
                    and value.numel() > 1
                )
            )
        ):
            raise AssertionError(
                "graph-sized accelerator tensor to(cpu) in prepared forward"
            )
        return original_to(value, *args, **kwargs)

    Tensor.cpu = guarded_cpu  # type: ignore[method-assign]
    Tensor.tolist = guarded_tolist  # type: ignore[method-assign]
    Tensor.item = guarded_item  # type: ignore[method-assign]
    Tensor.to = guarded_to  # type: ignore[method-assign]
    try:
        yield
    finally:
        Tensor.cpu = original_cpu  # type: ignore[method-assign]
        Tensor.tolist = original_tolist  # type: ignore[method-assign]
        Tensor.item = original_item  # type: ignore[method-assign]
        Tensor.to = original_to  # type: ignore[method-assign]


def _prepared_mutation_rejected(
    source_batch: SSLBatch,
    *,
    policy: str,
    device: torch.device,
) -> bool:
    binding = prepare_hierarchy_mask_binding(
        source_batch,
        policy_config=_single_policy_config(policy),
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    batch, moved_binding = move_ssl_batch_with_prepared_binding(
        source_batch,
        binding,
        device,
    )
    batch.raw_graph_batch["note"].x_cat.add_(0)
    model = _model().to(device).eval()
    encoder_calls = 0

    def count_call(_module: object, _inputs: object) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    encoder = (
        model.encoder.local_baseline.encoder.feature_encoder.node_encoders[
            "song"
        ]
    )
    handle = encoder.register_forward_pre_hook(count_call)
    try:
        try:
            model.forward_hierarchy(
                batch,
                prepared_mask_binding=moved_binding,
            )
        except SSLContractError as exc:
            rejected = (
                "ssl.prepared_binding.runtime_input_changed" in str(exc)
            )
        else:
            rejected = False
    finally:
        handle.remove()
    return rejected and encoder_calls == 0


def _policy_acceptance(
    source_batch: SSLBatch,
    *,
    policy: str,
    device: torch.device,
    source_piece: Any,
    source_sample: SSLRawSample,
    target_blind_batches: tuple[SSLBatch, SSLBatch],
) -> tuple[dict[str, object], dict[str, object]]:
    config = _single_policy_config(policy)
    resolutions = build_batched_hierarchy_mask_resolutions(
        source_batch.raw_graph_batch,
        dataset_ids=source_batch.dataset_ids,
        piece_ids=source_batch.piece_ids,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
        policy_config=config,
    )
    binding = prepare_hierarchy_mask_binding(
        source_batch,
        policy_config=config,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    source_snapshot = _snapshot_graph(source_batch.raw_graph_batch)
    batch, moved_binding = move_ssl_batch_with_prepared_binding(
        source_batch,
        binding,
        device,
    )
    _assert_batch_device(batch, device)
    if moved_binding.selected_global_note_indices_tensor.device != device:
        raise RuntimeError("prepared selected-index sidecar is on wrong device")
    validate_prepared_mask_binding(
        batch,
        moved_binding,
        expected_mask_rate=_MASK_RATE,
    )
    graph_snapshot = _snapshot_graph(batch.raw_graph_batch)
    binding_snapshot = deepcopy(moved_binding.to_dict())
    model = _model().to(device).eval()
    parameter_snapshot = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ), _guard_host_materialization(batch.raw_graph_batch):
        first = model.forward_hierarchy(
            batch,
            prepared_mask_binding=moved_binding,
        )
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        repeated = model.forward_hierarchy(
            batch,
            prepared_mask_binding=moved_binding,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not _outputs_bit_exact(first, repeated):
        raise RuntimeError("Phase 8A CUDA repeat is not bit-exact")
    forward_tensor_evidence = _output_tensor_evidence(
        first,
        device=device,
    )
    _output_tensor_evidence(repeated, device=device)
    loss_evidence = _loss_evidence(first, device=device)
    source_isolation = _source_sample_isolation(
        source_batch,
        first,
        policy=policy,
        model=model,
        device=device,
    )
    if not source_isolation:
        raise RuntimeError(
            "Phase 8A CUDA source-sample isolation failed"
        )
    target_blindness = _target_provenance_blindness(
        target_blind_batches[0],
        target_blind_batches[1],
        policy=policy,
        device=device,
    )
    if not target_blindness:
        raise RuntimeError(
            "Phase 8A target/provenance blindness failed on CUDA"
        )
    no_leakage, pitch_sensitive = _pitch_mutation_evidence(
        source_piece,
        source_sample,
        policy=policy,
        device=device,
    )

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        train_output = model.forward_hierarchy(
            batch,
            prepared_mask_binding=moved_binding,
        )
    total_loss = train_output.objective.total_loss
    if total_loss is None:
        raise RuntimeError("Phase 8A CUDA objective is unavailable")
    total_loss.backward()
    _output_tensor_evidence(train_output, device=device)
    gradient_evidence = _gradient_evidence(model, device=device)
    if not _graph_matches_snapshot(batch.raw_graph_batch, graph_snapshot):
        raise RuntimeError("Phase 8A CUDA forward mutated its graph")
    if not _graph_matches_snapshot(
        source_batch.raw_graph_batch,
        source_snapshot,
    ):
        raise RuntimeError("Phase 8A CUDA transfer mutated its CPU source")
    if moved_binding.to_dict() != binding_snapshot:
        raise RuntimeError("Phase 8A CUDA forward mutated its binding")
    if not all(
        torch.equal(value, parameter_snapshot[name])
        for name, value in model.state_dict().items()
    ):
        raise RuntimeError(
            "Phase 8A CUDA backward mutated parameters without an optimizer"
        )
    if not _overlay_closes_pitch_dependencies(moved_binding):
        raise RuntimeError(
            "Phase 8A overlay does not close pitch dependencies"
        )
    mutation_rejected = _prepared_mutation_rejected(
        source_batch,
        policy=policy,
        device=device,
    )
    if not mutation_rejected:
        raise RuntimeError(
            "Phase 8A prepared mutation reached encoder computation"
        )

    policy_report = {
        "policy": policy,
        "requested_device": "cuda:0",
        "resolved_device": str(device),
        "amp_enabled": True,
        "amp_dtype": "torch.float16",
        "policy_configuration": config.to_dict(),
        "resolution_fingerprints": [
            resolution.fingerprint for resolution in resolutions
        ],
        "resolved_policies": [
            resolution.resolved_policy for resolution in resolutions
        ],
        "plan_fingerprints": list(
            moved_binding.ordered_plan_fingerprints
        ),
        "overlay_fingerprint": (
            moved_binding.feature_overlay.fingerprint
        ),
        "prepared_binding_contract_version": (
            moved_binding.contract_version
        ),
        "prepared_binding_fingerprint": moved_binding.fingerprint,
        "prepared_binding_validated_on_cuda": True,
        "all_model_facing_tensors_on_cuda_0": True,
        "deterministic_repeat_bit_exact": True,
        "forward_tensors": forward_tensor_evidence,
        "losses": loss_evidence,
        "gradients": gradient_evidence,
        "raw_cpu_source_unchanged": True,
        "raw_cuda_graph_unchanged": True,
        "prepared_binding_unchanged": True,
        "model_parameters_unchanged_without_optimizer": True,
        "no_graph_sized_accelerator_to_host_materialization": True,
        "prepared_mutation_rejected_before_encoder": True,
        "masked_pitch_and_collateral_track_slots_closed": True,
        "source_sample_isolation": True,
        "target_provenance_diagnostic_blindness": True,
        "no_leakage_mutation_evidence": no_leakage,
        "pitch_sensitive_reconstruction_evidence": pitch_sensitive,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "quality_claim": None,
        "loss_decrease_is_acceptance_criterion": False,
        "correct_target_preference_is_acceptance_criterion": False,
    }
    portable = {
        "policy": policy,
        "config_fingerprint": config.fingerprint,
        "resolution_fingerprints": policy_report[
            "resolution_fingerprints"
        ],
        "plan_fingerprints": policy_report["plan_fingerprints"],
        "overlay_fingerprint": policy_report["overlay_fingerprint"],
        "prepared_binding_fingerprint": policy_report[
            "prepared_binding_fingerprint"
        ],
    }
    return policy_report, portable


def _independent_control_acceptance(
    source_batch: SSLBatch,
    *,
    device: torch.device,
) -> dict[str, object]:
    direct = prepare_mask_binding(
        source_batch,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    dispatched = prepare_hierarchy_mask_binding(
        source_batch,
        policy_config=_single_policy_config(
            INDEPENDENT_NOTE_PITCH
        ),
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    if direct.to_dict() != dispatched.to_dict():
        raise RuntimeError(
            "independent hierarchy dispatch differs from Phase 7A binding"
        )
    direct_batch, moved_direct = move_ssl_batch_with_prepared_binding(
        source_batch,
        direct,
        device,
    )
    dispatched_batch, moved_dispatched = (
        move_ssl_batch_with_prepared_binding(
            source_batch,
            dispatched,
            device,
        )
    )
    model = _model().to(device).eval()
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        phase7a = model(
            direct_batch,
            prepared_mask_binding=moved_direct,
        )
        phase8a = model.forward_hierarchy(
            dispatched_batch,
            prepared_mask_binding=moved_dispatched,
        )
    phase7a_tensors = _output_tensor_evidence(
        phase7a,
        device=device,
    )
    phase8a_tensors = _output_tensor_evidence(
        phase8a,
        device=device,
    )
    if not _outputs_bit_exact(
        phase7a,
        phase8a,
        allow_envelope_contract_difference=True,
    ):
        raise RuntimeError(
            "independent hierarchy dispatch differs from Phase 7A output"
        )
    return {
        "portable_binding_bit_exact": True,
        "cuda_amp_model_facing_output_bit_exact": True,
        "phase7a_forward_tensors": phase7a_tensors,
        "phase8a_forward_tensors": phase8a_tensors,
        "prepared_binding_contract_version": (
            PREPARED_MASK_BINDING_CONTRACT_VERSION
        ),
        "binding_fingerprint": direct.fingerprint,
    }


def _mixture_acceptance(
    source_batch: SSLBatch,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    config = HierarchyMaskPolicyConfig()
    first = build_batched_hierarchy_mask_resolutions(
        source_batch.raw_graph_batch,
        dataset_ids=source_batch.dataset_ids,
        piece_ids=source_batch.piece_ids,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
        policy_config=config,
    )
    repeated = build_batched_hierarchy_mask_resolutions(
        source_batch.raw_graph_batch,
        dataset_ids=source_batch.dataset_ids,
        piece_ids=source_batch.piece_ids,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
        policy_config=config,
    )
    if first != repeated or not all(
        resolution.resolved_policy in resolution.eligible_policies
        and resolution.policy_configuration_fingerprint
        == config.fingerprint
        for resolution in first
    ):
        raise RuntimeError("Phase 8A mixture resolution is inconsistent")
    binding = prepare_hierarchy_mask_binding(
        source_batch,
        policy_config=config,
        global_seed=_GLOBAL_SEED,
        epoch=_EPOCH,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    batch, moved = move_ssl_batch_with_prepared_binding(
        source_batch,
        binding,
        device,
    )
    model = _model().to(device).eval()
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=torch.float16,
    ):
        output = model.forward_hierarchy(
            batch,
            prepared_mask_binding=moved,
        )
    mixture_forward_tensors = _output_tensor_evidence(
        output,
        device=device,
    )
    _loss_evidence(output, device=device)
    unavailable_config = HierarchyMaskPolicyConfig.create(
        weights={
            INDEPENDENT_NOTE_PITCH: 1.0,
            CONTIGUOUS_BAR_PITCH_SPAN: 1.0,
        },
        min_span_bars=8,
        max_span_bars=8,
    )
    unavailable_resolutions = (
        build_batched_hierarchy_mask_resolutions(
            source_batch.raw_graph_batch,
            dataset_ids=source_batch.dataset_ids,
            piece_ids=source_batch.piece_ids,
            global_seed=_GLOBAL_SEED,
            epoch=_EPOCH,
            requested_mask_rate=_MASK_RATE,
            stage="train",
            policy_config=unavailable_config,
        )
    )
    if not all(
        resolution.resolved_policy == INDEPENDENT_NOTE_PITCH
        and CONTIGUOUS_BAR_PITCH_SPAN
        not in resolution.eligible_policies
        for resolution in unavailable_resolutions
    ):
        raise RuntimeError(
            "unavailable hierarchy policy entered mixture resolution"
        )
    impossible_config = HierarchyMaskPolicyConfig.create(
        weights={CONTIGUOUS_BAR_PITCH_SPAN: 1.0},
        min_span_bars=8,
        max_span_bars=8,
    )
    try:
        prepare_hierarchy_mask_binding(
            source_batch,
            policy_config=impossible_config,
            global_seed=_GLOBAL_SEED,
            epoch=_EPOCH,
            requested_mask_rate=_MASK_RATE,
            stage="train",
        )
    except HierarchyMaskUnavailableError:
        impossible_rejected = True
    else:
        impossible_rejected = False
    if not impossible_rejected:
        raise RuntimeError(
            "unavailable-only mixture silently fell back"
        )
    report = {
        "config_fingerprint": config.fingerprint,
        "resolution_fingerprints": [
            resolution.fingerprint for resolution in first
        ],
        "resolved_policies": [
            resolution.resolved_policy for resolution in first
        ],
        "eligible_policies": [
            list(resolution.eligible_policies)
            for resolution in first
        ],
        "repeat_bit_exact": True,
        "resolved_only_from_eligible_set": True,
        "resolution_bound_to_config_fingerprint": True,
        "prepared_binding_fingerprint": moved.fingerprint,
        "finite_cuda_amp_forward": True,
        "forward_tensors": mixture_forward_tensors,
        "unavailable_policy_excluded_before_encoder": True,
        "unavailable_only_configuration_rejected": True,
        "unavailable_policy_silent_fallback": False,
    }
    portable = {
        key: report[key]
        for key in (
            "config_fingerprint",
            "resolution_fingerprints",
            "resolved_policies",
            "eligible_policies",
            "prepared_binding_fingerprint",
        )
    }
    return report, portable


def _portable_contract_bindings() -> dict[str, str]:
    return {
        "runtime_device_resolution": (
            RUNTIME_DEVICE_RESOLUTION_CONTRACT_VERSION
        ),
        "device_transfer": DEVICE_TRANSFER_CONTRACT_VERSION,
        "ssl_umbrella": SSL_CONTRACT_VERSION,
        "ssl_training_report": SSL_TRAINING_REPORT_VERSION,
        "ssl_model": SSL_MODEL_CONTRACT_VERSION,
        "ssl_model_output": SSL_MODEL_OUTPUT_CONTRACT_VERSION,
        "representation_loss": REPRESENTATION_LOSS_CONTRACT_VERSION,
        "multi_view_representation_loss": (
            MULTI_VIEW_REPRESENTATION_LOSS_CONTRACT_VERSION
        ),
        "ssl_objective": SSL_OBJECTIVE_CONTRACT_VERSION,
        "anti_collapse_diagnostics": (
            ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION
        ),
        "no_leakage_mutation_evidence": (
            NO_LEAKAGE_MUTATION_EVIDENCE_CONTRACT_VERSION
        ),
        "pitch_sensitive_reconstruction_evidence": (
            PITCH_SENSITIVE_RECONSTRUCTION_EVIDENCE_CONTRACT_VERSION
        ),
        "phase7a_mask_plan": MASK_PLAN_CONTRACT_VERSION,
        "phase7a_mask_policy": MASK_POLICY_VERSION,
        "feature_overlay": MASKED_FEATURE_OVERLAY_CONTRACT_VERSION,
        "prepared_binding": PREPARED_MASK_BINDING_CONTRACT_VERSION,
        "hierarchy_prepared_binding": (
            PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
        ),
        "hierarchical_mask_plan": (
            HIERARCHICAL_MASK_PLAN_CONTRACT_VERSION
        ),
        "hierarchy_mask_policy": HIERARCHY_MASK_POLICY_VERSION,
        "hierarchy_policy_config": (
            HIERARCHY_POLICY_CONFIG_CONTRACT_VERSION
        ),
        "hierarchy_policy_mixture": (
            HIERARCHY_POLICY_MIXTURE_CONTRACT_VERSION
        ),
        "hierarchy_selection_evidence": (
            HIERARCHY_SELECTION_EVIDENCE_CONTRACT_VERSION
        ),
        "hierarchy_unavailable_reason": (
            HIERARCHY_UNAVAILABLE_REASON_CONTRACT_VERSION
        ),
        "hierarchy_prepared_binding_profile": (
            HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
        ),
        "bounded_fixture": (
            PHASE8A_HIERARCHY_FIXTURE_CONTRACT_VERSION
        ),
        "hierarchy_ssl_output": (
            PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION
        ),
    }


def _portable_policy_projection(
    report: Mapping[str, object],
) -> dict[str, object]:
    try:
        policies = report["policies"]
        if not isinstance(policies, Mapping) or set(policies) != set(
            HIERARCHY_MASK_POLICIES
        ):
            raise TypeError
        projection: dict[str, object] = {}
        for policy in HIERARCHY_MASK_POLICIES:
            payload = policies[policy]
            if not isinstance(payload, Mapping):
                raise TypeError
            policy_config = payload["policy_config"]
            resolutions = payload["eligibility_and_resolution"]
            plans = payload["plans"]
            if (
                not isinstance(policy_config, Mapping)
                or not isinstance(resolutions, list)
                or not isinstance(plans, list)
            ):
                raise TypeError
            projection[policy] = {
                "policy": policy,
                "config_fingerprint": policy_config["fingerprint"],
                "resolution_fingerprints": [
                    row["resolution_fingerprint"]
                    for row in resolutions
                    if isinstance(row, Mapping)
                ],
                "plan_fingerprints": [
                    row["plan_fingerprint"]
                    for row in plans
                    if isinstance(row, Mapping)
                ],
                "overlay_fingerprint": payload[
                    "overlay_fingerprint"
                ],
                "prepared_binding_fingerprint": payload[
                    "prepared_binding_fingerprint"
                ],
            }
            if (
                len(projection[policy]["resolution_fingerprints"])
                != len(resolutions)
                or len(projection[policy]["plan_fingerprints"])
                != len(plans)
            ):
                raise TypeError
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "phase8a.cuda.portable_cpu_report_shape_invalid"
        ) from exc
    return projection


def _validate_portable_cpu_report(
    portable_report: Mapping[str, object],
    *,
    portable_report_sha256: str,
    contracts: Mapping[str, str],
    policy_contract_fingerprint: str,
    fixture_fingerprints: Mapping[str, object],
    model_metadata_fingerprint: str,
    portable_policies: Mapping[str, object],
) -> dict[str, object]:
    if (
        len(portable_report_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in portable_report_sha256
        )
    ):
        raise RuntimeError(
            "phase8a.cuda.portable_cpu_report_sha256_invalid"
        )
    try:
        fixture = portable_report["fixture"]
        configuration = portable_report["configuration"]
        if (
            not isinstance(fixture, Mapping)
            or not isinstance(configuration, Mapping)
        ):
            raise TypeError
        checks = {
            "acceptance_contract_exact": (
                portable_report["acceptance_contract_version"]
                == PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION
            ),
            "contracts_exact": portable_report["contracts"] == contracts,
            "policy_contract_fingerprint_exact": (
                portable_report[
                    "hierarchy_mask_policy_contract_fingerprint"
                ]
                == policy_contract_fingerprint
            ),
            "fixture_fingerprints_exact": (
                fixture["fingerprints"] == fixture_fingerprints
            ),
            "model_metadata_fingerprint_exact": (
                portable_report[
                    "model_contract_metadata_fingerprint"
                ]
                == model_metadata_fingerprint
            ),
            "policy_fingerprints_exact": (
                _portable_policy_projection(portable_report)
                == portable_policies
            ),
            "cpu_device_exact": configuration["device"] == "cpu",
            "all_policies_exercised": (
                portable_report[
                    "all_policies_independently_exercised"
                ]
                is True
            ),
            "cuda_measurement_absent": (
                portable_report["cuda_measurement"] is None
            ),
            "quality_claim_absent": (
                portable_report["quality_claim"] is None
            ),
        }
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "phase8a.cuda.portable_cpu_report_shape_invalid"
        ) from exc
    if not all(checks.values()):
        failed = ",".join(
            name for name, passed in checks.items() if not passed
        )
        raise RuntimeError(
            "phase8a.cuda.portable_cpu_report_mismatch:" + failed
        )
    return {
        "provided": True,
        "validated": True,
        "sha256": portable_report_sha256,
        "acceptance_contract_version": (
            PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION
        ),
        **checks,
    }


def _build_phase8a_cuda_amp_hardware_report(
    *,
    device: str = "cuda:0",
    expected_head: str | None = None,
    expected_device_name: str | None = None,
    portable_report_sha256: str | None = None,
    portable_cpu_report: Mapping[str, object] | None = None,
    require_clean: bool = False,
) -> dict[str, object]:
    """Run bounded Phase 8A mechanics on one concrete CUDA device."""

    resolved_device = resolve_runtime_device(device)
    if resolved_device != torch.device("cuda", 0):
        raise RuntimeError("Phase 8A acceptance did not resolve cuda:0")
    head = _git("rev-parse", "HEAD")
    _git(
        "merge-base",
        "--is-ancestor",
        "a20393293a9ba4fad5721a9f7b90edb82bb67752",
        "HEAD",
    )
    if expected_head is not None and head != expected_head:
        raise RuntimeError(
            f"expected exact HEAD {expected_head}, found {head}"
        )
    clean = not bool(_git("status", "--porcelain=v1"))
    if require_clean and not clean:
        raise RuntimeError("exact-final CUDA acceptance requires a clean tree")

    torch.manual_seed(811)
    torch.cuda.manual_seed_all(811)
    properties = torch.cuda.get_device_properties(resolved_device)
    if (
        expected_device_name is not None
        and properties.name != expected_device_name
    ):
        raise RuntimeError(
            f"expected {expected_device_name!r}, found {properties.name!r}"
        )
    driver_version = _driver_version()
    if require_clean and driver_version is None:
        raise RuntimeError(
            "exact-final CUDA acceptance requires NVIDIA driver evidence"
        )
    fixture = build_phase8a_hierarchy_fixture()
    train_samples = fixture.raw_samples("train")
    source_batch = collate_ssl_samples(train_samples)
    target_blind_batches = _target_blind_pair(
        fixture.train_pieces[0],
        train_samples[0],
    )
    model_metadata = _model().ssl_contract_metadata()
    model_metadata_fingerprint = canonical_sha256(model_metadata)
    policies: dict[str, object] = {}
    portable_policies: dict[str, object] = {}
    for policy in HIERARCHY_MASK_POLICIES:
        report, portable = _policy_acceptance(
            source_batch,
            policy=policy,
            device=resolved_device,
            source_piece=fixture.train_pieces[0],
            source_sample=train_samples[0],
            target_blind_batches=target_blind_batches,
        )
        policies[policy] = report
        portable_policies[policy] = portable
    control = _independent_control_acceptance(
        source_batch,
        device=resolved_device,
    )
    mixture, portable_mixture = _mixture_acceptance(
        source_batch,
        device=resolved_device,
    )
    global_peak_allocated = max(
        int(report["peak_allocated_bytes"])
        for report in policies.values()
        if isinstance(report, dict)
    )
    global_peak_reserved = max(
        int(report["peak_reserved_bytes"])
        for report in policies.values()
        if isinstance(report, dict)
    )
    portable_contracts = _portable_contract_bindings()
    contracts = {
        **portable_contracts,
        "phase8a_bounded_acceptance": (
            PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION
        ),
        "phase8a_cuda_amp_hardware_evidence": (
            PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION
        ),
    }
    portable_semantics = {
        "contracts": contracts,
        "hierarchy_policy_contract_fingerprint": (
            HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT
        ),
        "span_selection_rank_contract": {
            "selection_method": SPAN_SELECTION_METHOD,
            "pool_membership_rank_method": (
                SPAN_POOL_MEMBERSHIP_RANK_METHOD
            ),
            "final_choice_rank_method": (
                SPAN_FINAL_CHOICE_RANK_METHOD
            ),
            "collision_fallback": (
                "track_start_end_descendants_v1"
            ),
        },
        "fixture_fingerprints": fixture.fingerprint_bundle(),
        "model_metadata_fingerprint": model_metadata_fingerprint,
        "policies": portable_policies,
        "mixture": portable_mixture,
        "independent_control_binding_fingerprint": control[
            "binding_fingerprint"
        ],
    }
    if portable_cpu_report is None:
        portable_report_validation = {
            "provided": False,
            "validated": False,
            "sha256": portable_report_sha256,
            "status": "optional_pytest_run_without_cpu_report",
        }
    else:
        if portable_report_sha256 is None:
            raise RuntimeError(
                "phase8a.cuda.portable_cpu_report_sha256_missing"
            )
        portable_report_validation = _validate_portable_cpu_report(
            portable_cpu_report,
            portable_report_sha256=portable_report_sha256,
            contracts=portable_contracts,
            policy_contract_fingerprint=(
                HIERARCHY_MASK_POLICY_CONTRACT_FINGERPRINT
            ),
            fixture_fingerprints=fixture.fingerprint_bundle(),
            model_metadata_fingerprint=model_metadata_fingerprint,
            portable_policies=portable_policies,
        )
    all_five_policies_exercised = (
        len(HIERARCHY_MASK_POLICIES) == 5
        and len(policies) == 5
        and tuple(policies) == HIERARCHY_MASK_POLICIES
    )
    if not all_five_policies_exercised:
        raise RuntimeError(
            "Phase 8A acceptance did not exercise exactly five policies"
        )
    report: dict[str, object] = {
        "evidence_kind": "phase8a_cuda_amp_hardware_evidence",
        "hardware_evidence_contract_version": (
            PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION
        ),
        "portable": False,
        "source": {
            "git_head_sha": head,
            "expected_head": expected_head,
            "expected_head_match": (
                expected_head is None or expected_head == head
            ),
            "source_tree_clean": clean,
            "hotfix_ancestor_required": (
                "a20393293a9ba4fad5721a9f7b90edb82bb67752"
            ),
        },
        "portable_binding": {
            "portable_cpu_report_sha256": portable_report_sha256,
            "portable_cpu_report_validation": (
                portable_report_validation
            ),
            "portable_semantics_fingerprint": canonical_sha256(
                portable_semantics
            ),
            "semantics": portable_semantics,
        },
        "runtime": {
            "requested_device": device,
            "resolved_device": str(resolved_device),
            "amp_enabled": True,
            "amp_dtype": "torch.float16",
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "driver_version": driver_version,
            "gpu_name": properties.name,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": (
                torch.backends.cudnn.deterministic
            ),
        },
        "contracts": contracts,
        "model_metadata_fingerprint": model_metadata_fingerprint,
        "policies": policies,
        "all_five_policies_exercised": all_five_policies_exercised,
        "mixture": mixture,
        "independent_control": control,
        "global_peak_allocated_bytes": global_peak_allocated,
        "global_peak_reserved_bytes": global_peak_reserved,
        "quality_claim": None,
        "performance_thresholds": None,
        "loss_decrease_is_acceptance_criterion": False,
        "correct_target_preference_is_acceptance_criterion": False,
        "gpu_values_are_portable_fingerprint_inputs": False,
    }
    report["hardware_evidence_fingerprint"] = canonical_sha256(report)
    return report


def build_phase8a_cuda_amp_hardware_report(
    *,
    device: str = "cuda:0",
    expected_head: str | None = None,
    expected_device_name: str | None = None,
    portable_report_path: Path | None = None,
    require_clean: bool = False,
) -> dict[str, object]:
    """Run bounded Phase 8A mechanics on one concrete CUDA device."""

    if device != "cuda:0":
        raise ValueError("Phase 8A hardware acceptance requires cuda:0")
    if portable_report_path is not None and not isinstance(
        portable_report_path,
        Path,
    ):
        raise ValueError("portable report path must be a pathlib.Path")
    if require_clean:
        if expected_head is None:
            raise ValueError(
                "exact-final CUDA acceptance requires expected HEAD"
            )
        if expected_device_name != _EXACT_RTX_3090_DEVICE_NAME:
            raise ValueError(
                "exact-final CUDA acceptance requires "
                + _EXACT_RTX_3090_DEVICE_NAME
            )
        if portable_report_path is None:
            raise ValueError(
                "exact-final CUDA acceptance requires portable CPU report"
            )
    portable_cpu_report = (
        None
        if portable_report_path is None
        else _load_portable_cpu_report(portable_report_path)
    )
    portable_report_sha256 = (
        None
        if portable_report_path is None
        else _sha256_file(portable_report_path)
    )
    with _preserved_deterministic_cuda_runtime():
        return _build_phase8a_cuda_amp_hardware_report(
            device=device,
            expected_head=expected_head,
            expected_device_name=expected_device_name,
            portable_report_sha256=portable_report_sha256,
            portable_cpu_report=portable_cpu_report,
            require_clean=require_clean,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_portable_cpu_report(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "phase8a.cuda.portable_cpu_report_unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            "phase8a.cuda.portable_cpu_report_root_invalid"
        )
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=("float16",),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--expected-device-name",
        required=True,
        choices=(_EXACT_RTX_3090_DEVICE_NAME,),
    )
    parser.add_argument("--portable-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.amp:
        parser.error("--amp is required for Phase 8A CUDA acceptance")
    if (
        len(arguments.expected_head) != 40
        or any(
            character not in "0123456789abcdef"
            for character in arguments.expected_head
        )
    ):
        parser.error("--expected-head must be a lowercase 40-character SHA")
    report = build_phase8a_cuda_amp_hardware_report(
        device=arguments.device,
        expected_head=arguments.expected_head,
        expected_device_name=arguments.expected_device_name,
        portable_report_path=arguments.portable_report,
        require_clean=True,
    )
    text = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    _atomic_write(arguments.output, text)
    print(text)


if __name__ == "__main__":
    main()
