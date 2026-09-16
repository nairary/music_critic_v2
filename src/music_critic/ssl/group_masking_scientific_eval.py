"""Sequential, fixed-membership VALIDATION for the bounded WS2 pilot.

This is an evaluator, never a reader/discovery policy or checkpoint selector.
The caller binds every raw batch and the complete ordered membership beforehand.
The reference corrected accumulator includes valid VALIDATION-only classes in
metrics; the historical supervised loss separately masks TRAIN-unsupported
classes. Neither population is silently removed from this report.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
import math
import random

import numpy as np
import torch

from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS, CorrectedValidationAccumulator, align_target_sidecars_after_prediction,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID
from music_critic.ssl.data import move_ssl_batch, validate_ssl_batch
from music_critic.ssl.group_masking_pilot import (
    POLICIES, GroupPilotConfig, PilotBatch, _runtime_binding,
    mask_accounting, tree_fingerprint,
)
from music_critic.ssl.hierarchical_masking import HierarchyMaskPolicyConfig, HierarchyMaskUnavailableError
from music_critic.ssl.masking import (
    move_ssl_batch_with_prepared_binding, prepare_hierarchy_mask_binding,
)
from music_critic.ssl.objective import StreamingAntiCollapseDiagnostics
from music_critic.ssl.theory_retention_harness import (
    TheoryRetentionHarness, _loss_with_empty_candidate_heads,
)

EVALUATION_SCHEMA = "WS2GroupMaskingScientificValidation@1.0.0"
RECEIPT_SCHEMA = "WS2GroupMaskingValidationReceipt@1.0.0"


@dataclass(frozen=True)
class ValidationResourceLimits:
    """Explicit singleton full-record bounds, independent of TRAIN eligibility.

    A caller must bind feasible limits before execution. Exceeding one aborts
    the evaluation; it never silently removes that member or changes a graph.
    External timeout/memory supervision is still required.
    """
    max_notes_per_piece: int
    max_nodes_per_piece: int
    max_edges_per_piece: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("group_validation.explicit_positive_resource_limits_required")


def _require_finite(value, *, reason="nonfinite_output"):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"group_validation.{reason}")
    elif isinstance(value, Mapping):
        for child in value.values():
            _require_finite(child, reason=reason)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _require_finite(child, reason=reason)
    elif is_dataclass(value):
        for field in fields(value):
            _require_finite(getattr(value, field.name), reason=reason)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"group_validation.{reason}")


def _identities(raw):
    return tuple(zip(raw.dataset_ids, raw.piece_ids, strict=True))


def validation_batch_binding(batch: PilotBatch, config: GroupPilotConfig, *,
                             resource_limits: ValidationResourceLimits | None = None) -> str:
    """Validate CPU budgets and bind raw bytes, opaque sidecars and TRAIN weights.

    Hashing a preloaded sidecar does not align or interpret its labels. The
    prediction call below receives only raw graph tensors.
    """
    if not isinstance(batch, PilotBatch) or batch.ssl is None or batch.theory is None:
        raise ValueError("group_validation.raw_ssl_and_theory_required")
    if resource_limits is None:
        # Backward-compatible tiny generated fixtures only. Production callers
        # must provide independently bound full-record validation resources.
        if any(dataset not in {"synthetic", "phase7a-bounded"} for dataset in batch.theory.dataset_ids):
            raise ValueError("group_validation.explicit_resource_limits_required")
        resource_limits = ValidationResourceLimits(config.max_notes_per_batch,
            config.max_nodes_per_batch, config.max_edges_per_batch)
    if not isinstance(resource_limits, ValidationResourceLimits):
        raise ValueError("group_validation.explicit_resource_limits_required")
    def raw_evidence(raw):
        validate_ssl_batch(raw)
        if raw.sample_count != 1:
            raise ValueError("group_validation.singleton_full_record_required")
        if (raw.raw_graph_batch["note"].num_nodes > resource_limits.max_notes_per_piece
                or raw.node_count > resource_limits.max_nodes_per_piece
                or raw.edge_count > resource_limits.max_edges_per_piece):
            raise ValueError("group_validation.resource_bound_exceeded_no_member_exclusion")
        if any(value.device.type != "cpu" for store in raw.raw_graph_batch.stores
               for value in store.values() if isinstance(value, torch.Tensor)):
            raise ValueError("group_validation.cpu_preparation_required")
        return {"identities": [list(pair) for pair in _identities(raw)],
                "raw_batch_fingerprint": tree_fingerprint(raw.raw_graph_batch.to_dict())}
    ssl, theory = raw_evidence(batch.ssl), raw_evidence(batch.theory)
    if ssl != theory:
        raise ValueError("group_validation.same_heldout_raw_required")
    if len(batch.sidecars) != batch.theory.sample_count or batch.class_weights is None:
        raise ValueError("group_validation.sidecars_and_weights_required")
    weights = batch.class_weights
    if set(weights.values) != set(ACTIVE_HEADS) or set(weights.supported) != set(ACTIVE_HEADS):
        raise ValueError("group_validation.class_weight_inventory_invalid")
    for task in ACTIVE_HEADS:
        values, supported = weights.values[task], weights.supported[task]
        size = TASK_BY_ID[task].class_count
        if (not isinstance(values, torch.Tensor) or not isinstance(supported, torch.Tensor)
                or values.shape != (size,) or supported.shape != (size,)
                or values.dtype != torch.float32 or supported.dtype != torch.bool
                or values.device.type != "cpu" or supported.device.type != "cpu"
                or not bool(torch.isfinite(values).all()) or not bool((values > 0).all())):
            raise ValueError("group_validation.class_weight_tensor_invalid")
    return tree_fingerprint({
        "schema": EVALUATION_SCHEMA, "split": "validation", "raw": ssl,
        "resource_limits": asdict(resource_limits),
        "sidecars": batch.sidecars,
        "weights": {"declared_fingerprint": weights.payload_fingerprint,
                    "values": dict(weights.values), "supported": dict(weights.supported)},
    })


def validation_policy_bindings(raw, config: GroupPilotConfig):
    """Record each policy's structural availability without excluding theory.

    Only the declared hierarchy-unavailable exception is converted to coverage.
    Malformed inputs, numerical errors and all other exceptions remain fatal.
    Singleton processing ensures one unavailable piece cannot hide its peers.
    """
    if raw.sample_count != 1:
        raise ValueError("group_validation.singleton_full_record_required")
    identity = list(_identities(raw)[0])
    bindings, availability = {}, {}
    for policy in POLICIES:
        try:
            binding = prepare_hierarchy_mask_binding(
                raw, policy_config=HierarchyMaskPolicyConfig.create(weights={policy: 1.0}),
                global_seed=config.seed, epoch=0, requested_mask_rate=config.mask_rate, stage="validation")
        except HierarchyMaskUnavailableError as exc:
            if len(exc.resolutions) != 1 or list(exc.resolutions[0].sample_identity) != identity:
                raise ValueError("group_validation.unavailable_resolution_identity_mismatch") from exc
            resolution = exc.resolutions[0]
            eligibility = next(item for item in resolution.eligibility if item.policy == policy)
            if eligibility.eligible or eligibility.unavailable_reason is None or resolution.plan is not None:
                raise ValueError("group_validation.invalid_unavailable_resolution") from exc
            bindings[policy] = None
            availability[policy] = {"piece": identity, "available": False,
                "reason": eligibility.unavailable_reason.to_dict(), "plan_fingerprint": None,
                "resolution_fingerprint": resolution.fingerprint}
            continue
        plan = binding.mask_plans[0]
        available = plan.selected_count > 0
        bindings[policy] = binding if available else None
        availability[policy] = {"piece": identity, "available": available,
            "reason": None if available else {"code": "no_eligible_notes", "source": "phase7a_mask_plan"},
            "plan_fingerprint": plan.fingerprint, "resolution_fingerprint": None}
    return bindings, availability


def _metric_alignment(output, alignment):
    """Avoid gathering placeholder indices from genuinely empty candidates.

    Original unavailable/alignment-failed rows remain in coverage accounting.
    This temporary reference-accumulator input removes only unaddressable,
    invalid rows; it never manufactures predictions or labels.
    """
    heads = dict(alignment.heads)
    for task, rows in heads.items():
        if output.logits[task].shape[0] or not rows.valid_mask.numel():
            continue
        if bool(rows.valid_mask.any()):
            raise ValueError("group_validation.valid_target_without_raw_candidates")
        heads[task] = replace(
            rows, candidate_indices=rows.candidate_indices[:0], values=rows.values[:0],
            valid_mask=rows.valid_mask[:0], sample_indices=rows.sample_indices[:0],
            entity_ids=(), record_ids=(), component_ids=(), masked_row_count=0,
            alignment_failure_count=0,
        )
    return replace(alignment, heads=heads)


class _SSLAccumulator:
    """Sum/count losses and existing O(D) exact stage-wide diagnostics."""

    def __init__(self, weights):
        self.losses = {level: {"numerator": 0.0, "denominator": 0, "zero_norm_count": 0}
                       for level in ("note", "bar", "song")}
        self.collapse = {level: StreamingAntiCollapseDiagnostics() for level in self.losses}
        self.mask_rows = []
        self.availability = []
        self.samples = 0
        self.weights = {level: float(getattr(weights, level + "_weight")) for level in self.losses}

    def observe_availability(self, row):
        self.availability.append(deepcopy(row))

    def update(self, output, masks):
        _require_finite(output)
        weights = {level: float(getattr(output.objective.weights, level + "_weight"))
                   for level in self.losses}
        if self.weights != weights:
            raise ValueError("group_validation.ssl_objective_drift")
        for level, loss in (("note", output.note_loss), ("bar", output.bar_latent.loss),
                            ("song", output.song_latent.loss)):
            row = self.losses[level]
            row["numerator"] += float(loss.numerator.detach())
            row["denominator"] += loss.denominator
            row["zero_norm_count"] += loss.zero_norm_count
        selected = output.targets.note.index_select(0, output.selected_global_note_indices)
        # Match the unchanged per-batch diagnostic definition: mean decoder
        # view prediction per selected note, not repeated targets across views.
        prediction = torch.stack(output.decoder_predictions).mean(0)
        self.collapse["note"].update(selected, prediction)
        for level in ("bar", "song"):
            latent = getattr(output, level + "_latent")
            self.collapse[level].update(latent.target, latent.prediction)
        self.mask_rows.extend(deepcopy(masks))
        self.samples += len(masks)

    def finalize(self):
        components = {level: {**row,
            "mean": row["numerator"] / row["denominator"] if row["denominator"] else None,
            "unavailable_reason": None if row["denominator"] else "no_eligible_rows"}
            for level, row in self.losses.items()}
        unavailable = [level for level in components
                       if self.weights[level] > 0 and components[level]["mean"] is None]
        total = None if unavailable else sum(self.weights[level] * components[level]["mean"]
            for level in components if self.weights[level] > 0)
        unavailable_rows = [row for row in self.availability if not row["available"]]
        reason_counts = {}
        for row in unavailable_rows:
            code = row["reason"]["code"]
            reason_counts[code] = reason_counts.get(code, 0) + 1
        return {"samples": self.samples, "components": components, "weights": self.weights,
                "weighted_total": total, "unavailable_components": unavailable,
                "aggregation": "component_sum_count_then_fixed_weight_sum",
                "collapse": {level: acc.to_dict() if acc.embedding_dim is not None else None
                             for level, acc in self.collapse.items()},
                "unavailable_reason": "no_available_policy_pieces" if not self.samples else None,
                "membership_pieces": len(self.availability), "available_pieces": self.samples,
                "unavailable_pieces": len(unavailable_rows), "unavailable_reason_counts": reason_counts,
                "availability": self.availability,
                "masks": self.mask_rows}


def evaluate_scientific_validation(
    harness: TheoryRetentionHarness, *, batch_factory: Callable[[], Iterable[PilotBatch]],
    config: GroupPilotConfig, resource_limits: ValidationResourceLimits | None = None,
    expected_membership: Sequence[tuple[str, str]],
    expected_batch_bindings: Sequence[str], split: str = "validation",
    on_receipt: Callable[[dict], None] | None = None,
    runtime_validator: Callable[[], None] | None = None,
    train_seen_tuples: set[tuple[str, ...]] | None = None,
) -> dict:
    """Evaluate every bound member once, with no update or selection.

    Receipts precede factory/forward work and count completed shared Transformer
    calls even on failure. The caller must fsync them and publish a result only
    after this function returns. External timeout/resource supervision remains
    necessary. Fixed masks use validation epoch zero under both policies.
    An absent TRAIN tuple inventory makes only seen/unseen metrics unavailable;
    an explicitly empty inventory is available and classifies all joint events
    as unseen. The supplied inventory is snapshotted before reader callbacks.
    """
    if split != "validation":
        raise ValueError("group_validation.validation_only")
    membership = tuple(expected_membership)
    bindings = tuple(expected_batch_bindings)
    if (not membership or len(set(membership)) != len(membership)
            or any(not isinstance(pair, tuple) or len(pair) != 2
                   or any(not isinstance(part, str) or not part for part in pair) for pair in membership)):
        raise ValueError("group_validation.fixed_membership_invalid")
    if (not bindings or any(not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value) for value in bindings)):
        raise ValueError("group_validation.batch_bindings_invalid")
    train_seen_tuples = None if train_seen_tuples is None else set(train_seen_tuples)
    train_seen_tuple_binding = (None if train_seen_tuples is None
                                else tree_fingerprint(sorted(train_seen_tuples)))
    device = next(harness.parameters()).device
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and device.index is None):
        raise ValueError("group_validation.explicit_cpu_or_indexed_cuda_required")
    baseline_runtime = tree_fingerprint(_runtime_binding(device))
    initial_state = model_state_fingerprint(harness)
    modes = [(module, module.training) for module in harness.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [] if device.type == "cpu" else [device.index]
    accumulator, class_binding = None, None
    requested_resource_limits = resource_limits
    if resource_limits is None:
        resource_limits = ValidationResourceLimits(config.max_notes_per_batch,
            config.max_nodes_per_batch, config.max_edges_per_batch)
    if not isinstance(resource_limits, ValidationResourceLimits):
        raise ValueError("group_validation.explicit_resource_limits_required")
    aggregates = {policy: _SSLAccumulator(config.ssl_config().weights()) for policy in POLICIES}
    coverage = {task: {"source_rows": 0, "valid_metric_rows": 0, "supported_loss_rows": 0,
                      "masked_rows": 0, "unsupported_train_rows": 0, "alignment_failure_rows": 0}
                for task in ACTIVE_HEADS}
    observed, confirmed_batches, seen, batch_rows = 0, 0, [], []
    receipt = {"schema": RECEIPT_SCHEMA, "status": "started", "stage": "started",
               "batch_index": 0, "policy": None, "batch_binding": None,
               "policy_availability": None,
               "observed_encoder_forwards": 0, "confirmed_batches": 0,
               "optimizer_outcome": "not_applied"}

    def emit(stage, *, status="in_progress", **values):
        receipt.update(stage=stage, status=status, observed_encoder_forwards=observed,
                       confirmed_batches=confirmed_batches, **values)
        if on_receipt is not None:
            on_receipt(deepcopy(receipt))

    def runtime():
        if tree_fingerprint(_runtime_binding(device)) != baseline_runtime:
            raise ValueError("group_validation.runtime_drift")
        if runtime_validator is not None:
            runtime_validator()

    def count_forward(*_):
        nonlocal observed
        observed += 1
        emit("forward")

    handle = None
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            emit("started", status="started")
            runtime()
            _require_finite(harness.state_dict(), reason="nonfinite_model_state")
            harness.eval()
            handle = harness.theory_model.context_encoder.transformer.register_forward_hook(count_forward)
            emit("factory")
            iterator = iter(batch_factory())
            for index, expected in enumerate(bindings):
                emit("batch_started", status="started", batch_index=index, policy=None,
                     policy_availability=None, batch_binding=expected)
                runtime()
                emit("factory")
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError("group_validation.incomplete_membership") from exc
                emit("batch_binding")
                if validation_batch_binding(batch, config, resource_limits=requested_resource_limits) != expected:
                    raise ValueError("group_validation.batch_binding_mismatch")
                identities = _identities(batch.theory)
                if tuple(membership[len(seen):len(seen) + len(identities)]) != identities:
                    raise ValueError("group_validation.membership_mismatch")
                weights_binding = tree_fingerprint({"values": dict(batch.class_weights.values),
                    "supported": dict(batch.class_weights.supported),
                    "declared_fingerprint": batch.class_weights.payload_fingerprint})
                if accumulator is None:
                    accumulator = CorrectedValidationAccumulator(batch.class_weights, train_seen_tuples=train_seen_tuples)
                    class_binding = weights_binding
                elif class_binding != weights_binding:
                    raise ValueError("group_validation.class_weights_drift")
                emit("masking")
                masks, availability = validation_policy_bindings(batch.ssl, config)
                start_forwards = observed
                emit("theory_transfer")
                raw = batch.theory.raw_graph_batch if device.type == "cpu" else move_ssl_batch(batch.theory, device).raw_graph_batch
                runtime()
                emit("theory_forward")
                output = harness.theory_model(raw)
                emit("theory_output_validation")
                _require_finite(output)
                runtime()
                # Labels, masks and confidence are interpreted only after raw logits exist.
                emit("target_alignment")
                alignment = align_target_sidecars_after_prediction(output, raw, batch.sidecars)
                loss = _loss_with_empty_candidate_heads(output, alignment, batch.class_weights)
                _require_finite(loss)
                if not loss.fp32_boundary:
                    raise ValueError("group_validation.theory_fp32_boundary")
                accumulator.update(output, _metric_alignment(output, alignment), sidecars=batch.sidecars)
                for task, head in loss.heads.items():
                    rows, counts = alignment.heads[task], coverage[task]
                    counts["source_rows"] += rows.valid_mask.numel()
                    counts["valid_metric_rows"] += int(rows.valid_mask.sum())
                    counts["supported_loss_rows"] += head.valid_row_count
                    counts["masked_rows"] += head.masked_row_count
                    counts["unsupported_train_rows"] += head.unsupported_row_count
                    counts["alignment_failure_rows"] += rows.alignment_failure_count
                del output, loss, alignment, raw
                paired = {}
                for policy in POLICIES:
                    emit("ssl_policy", policy=policy, policy_availability=availability[policy])
                    aggregates[policy].observe_availability(availability[policy])
                    if masks[policy] is None:
                        paired[policy] = []
                        emit("ssl_unavailable", policy=policy, policy_availability=availability[policy])
                        continue
                    emit("ssl_transfer", policy=policy)
                    ssl, binding = batch.ssl, masks[policy]
                    if device.type != "cpu":
                        ssl, binding = move_ssl_batch_with_prepared_binding(ssl, binding, device)
                    runtime()
                    emit("ssl_forward")
                    forward = harness.ssl_model.forward_hierarchy if binding.hierarchy_profile_version is not None else harness.ssl_model.forward
                    ssl_output = forward(ssl, prepared_mask_binding=binding)
                    emit("ssl_output_validation")
                    _require_finite(ssl_output)
                    runtime()
                    paired[policy] = mask_accounting(masks[policy])
                    aggregates[policy].update(ssl_output, paired[policy])
                    del ssl_output, ssl, binding
                expected_forwards = 1 + 2 * sum(row["available"] for row in availability.values())
                if observed - start_forwards != expected_forwards:
                    raise ValueError("group_validation.encoder_forward_accounting_mismatch")
                runtime()
                seen.extend(identities)
                confirmed_batches += 1
                batch_rows.append({"batch_index": index, "batch_binding": expected,
                    "identities": [list(pair) for pair in identities], "encoder_forwards": observed - start_forwards,
                    "paired_masks": paired, "policy_availability": availability})
                emit("batch_completed", status="completed", policy=None, policy_availability=None)
                del batch, masks
            emit("membership_completion", policy=None)
            exhausted = object()
            if next(iterator, exhausted) is not exhausted:
                raise ValueError("group_validation.extra_membership")
            if tuple(seen) != membership:
                raise ValueError("group_validation.incomplete_membership")
            theory = accumulator.finalize()
            # The historical accumulator treats None as an empty set. Preserve
            # its total-joint/per-head metrics while withholding an unsupported
            # seen/unseen partition when no TRAIN inventory was supplied.
            theory["train_seen_tuple_inventory"] = {
                "available": train_seen_tuples is not None,
                "tuple_count": None if train_seen_tuples is None else len(train_seen_tuples),
                "unavailable_reason": "train_seen_tuple_inventory_unavailable" if train_seen_tuples is None else None,
            }
            if train_seen_tuples is None:
                for field in ("seen_tuple_joint_accuracy", "seen_tuple_support",
                              "unseen_tuple_joint_accuracy", "unseen_tuple_support", "unseen_tuple_count"):
                    theory[field] = None
            for task, counts in coverage.items():
                if (counts["source_rows"] != counts["supported_loss_rows"] + counts["masked_rows"] + counts["unsupported_train_rows"]
                        or counts["valid_metric_rows"] != theory["per_head"][task]["support"]):
                    raise ValueError("group_validation.coverage_accounting_mismatch")
                counts["coverage"] = counts["valid_metric_rows"] / counts["source_rows"] if counts["source_rows"] else None
                theory["per_head"][task]["coverage"] = deepcopy(counts)
            result = {"schema": EVALUATION_SCHEMA, "split": "validation", "test_access": "closed",
                "model_state_fingerprint": initial_state,
                "membership": [list(pair) for pair in membership],
                "membership_fingerprint": tree_fingerprint(membership),
                "batch_bindings": list(bindings), "class_weights_fingerprint": class_binding,
                "mask_seed": config.seed, "mask_rate": config.mask_rate, "mask_epoch": 0,
                "mask_stage": "validation", "runtime": _runtime_binding(device),
                "resource_limits": asdict(resource_limits),
                "observed_samples": len(seen), "confirmed_batches": confirmed_batches,
                "observed_encoder_forwards": observed, "optimizer_updates": 0,
                "theory": theory, "ssl": {policy: aggregate.finalize() for policy, aggregate in aggregates.items()},
                "batches": batch_rows, "train_seen_tuple_binding": train_seen_tuple_binding,
                "metric_population": "all_valid_validation_targets_including_train_unsupported_classes",
                "head_metric_units": {task: TASK_BY_ID[task].prediction_level for task in ACTIVE_HEADS},
                "selection": "fixed_endpoint_descriptive_no_checkpoint_selection"}
            result["ssl_coverage_fingerprint"] = tree_fingerprint({policy: report["availability"]
                for policy, report in result["ssl"].items()})
            _require_finite(result)
            runtime()
            if model_state_fingerprint(harness) != initial_state:
                raise ValueError("group_validation.model_state_changed")
            emit("completed", status="completed")
            return result
    except BaseException as exc:
        try:
            emit(receipt["stage"], status="failed", error_type=type(exc).__name__, reason=str(exc))
        except BaseException:
            pass
        raise
    finally:
        if handle is not None:
            handle.remove()
        # Preserve heterogeneous preexisting modes despite shared submodules.
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def compare_theory_endpoints(baseline: Mapping, endpoint: Mapping, recovered: Mapping) -> dict:
    """Absolute per-head forgetting/recovery deltas on identical bound targets.

    Positive forgetting denotes deterioration (reversed for cross entropy).
    No relative ratio, winner, best checkpoint or significance claim is formed.
    """
    binding_keys = ("schema", "split", "membership_fingerprint", "batch_bindings", "class_weights_fingerprint",
                    "mask_seed", "mask_rate", "mask_epoch", "mask_stage", "runtime", "train_seen_tuple_binding",
                    "resource_limits", "ssl_coverage_fingerprint")
    if any(any(report.get(key) != baseline.get(key) for key in binding_keys) for report in (endpoint, recovered)):
        raise ValueError("group_validation.comparison_binding_mismatch")
    if baseline.get("schema") != EVALUATION_SCHEMA or baseline.get("split") != "validation":
        raise ValueError("group_validation.comparison_schema_invalid")
    metrics = ("accuracy", "balanced_accuracy", "macro_f1_observed_validation_classes", "masked_cross_entropy")
    heads = {}
    for task in ACTIVE_HEADS:
        rows = [report["theory"]["per_head"][task] for report in (baseline, endpoint, recovered)]
        if any(row["support"] != rows[0]["support"] or row.get("coverage") != rows[0].get("coverage") for row in rows[1:]):
            raise ValueError("group_validation.comparison_support_mismatch")
        deltas = {}
        for metric in metrics:
            values = [row[metric] for row in rows]
            if rows[0]["support"] == 0 or any(value is None for value in values):
                deltas[metric] = {"endpoint_minus_baseline": None, "recovered_minus_endpoint": None,
                                  "recovered_minus_baseline": None, "forgetting": None,
                                  "recovery_gain": None, "remaining_gap": None}
                continue
            _require_finite(values)
            start, end, final = map(float, values)
            orientation = -1.0 if metric == "masked_cross_entropy" else 1.0
            deltas[metric] = {"endpoint_minus_baseline": end - start,
                "recovered_minus_endpoint": final - end, "recovered_minus_baseline": final - start,
                "forgetting": orientation * (start - end),
                "recovery_gain": orientation * (final - end), "remaining_gap": orientation * (start - final)}
        heads[task] = {"support": rows[0]["support"], "metrics": deltas}
    return {"schema": "WS2GroupMaskingForgettingRecovery@1.0.0",
            "membership_fingerprint": baseline["membership_fingerprint"], "per_head": heads,
            "interpretation": "absolute_deltas_only_positive_forgetting_is_deterioration",
            "relative_recovery_fraction": None, "scientific_superiority_claim": False}
