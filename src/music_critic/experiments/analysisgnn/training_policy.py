"""Frozen Phase 9E-B5B AnalysisGNN training-policy contracts.

The module is deliberately a planning boundary.  It serializes the official
reproduction profile and the paired corrected profiles, and provides small
pure helpers with which the loss and sampler semantics can be tested.  It does
not construct a model, optimizer, trainer, data loader, checkpoint, or metric
evaluator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Literal

import torch
from torch.nn import functional as F

from music_critic.experiments.analysisgnn.contracts import (
    ANALYSISGNN_COMMIT,
    ANALYSISGNN_REPOSITORY,
    fingerprint,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    COMPATIBILITY_QUALITY_VOCABULARY_ID,
    CORRECTED_QUALITY_VOCABULARY_ID,
    CORRECTED_V2_METRIC_ID,
    PAPER_CANDIDATE_UNIVERSE_ID,
    PAPER_DEFINED_JOINT_COMPONENTS,
    PAPER_TEXT_COMPATIBILITY_METRIC_ID,
    PRODUCTION_TASKS,
    TASK_BY_ID,
    get_vocabulary,
)
from music_critic.experiments.analysisgnn.transposition import (
    CORRECTED_PROFILE_ID as CORRECTED_TRANSPOSITION_PROFILE_ID,
    OFFICIAL_PROFILE_ID as OFFICIAL_TRANSPOSITION_PROFILE_ID,
    corrected_transposition_profile,
    official_transposition_evidence,
    select_record_shift,
)


TRAINING_POLICY_SCHEMA = "AnalysisGNNTrainingPolicy@1.0.0"
TRAINING_PROFILE_SCHEMA = "AnalysisGNNTrainingProfile@1.0.0"
HEAD_ROLE_CONTRACT_VERSION = "analysisgnn-corrected-head-roles-v1"
LOSS_CONTRACT_VERSION = "analysisgnn-corrected-masked-multitask-loss-v1"
CLASS_WEIGHT_CONTRACT_VERSION = "analysisgnn-corrected-train-class-weights-v1"
SAMPLER_CONTRACT_VERSION = "analysisgnn-corrected-component-sampler-v1"
METRIC_CONTRACT_VERSION = "analysisgnn-corrected-validation-metrics-v1"

OFFICIAL_TRAINING_PROFILE_ID = "analysisgnn-official-reproduction-e115182-v1"
CORRECTED_NO_TRANSPOSITION_PROFILE_ID = (
    "music-critic-v2-corrected-no-transposition-v1"
)
CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID = (
    "music-critic-v2-corrected-safe-transposition-v1"
)

B3_SEMANTIC_FINGERPRINT = (
    "94a19ed6bbecbbd0497310233c8a8ff4e34311b414124593a7326c759ff07954"
)
B4_SEMANTIC_FINGERPRINT = (
    "4b1edf9f47815bafa5e197be87b9331a19789142c0625ef4aceda1f87649df4d"
)
B5A_SEMANTIC_FINGERPRINT = (
    "b8aba86430fe2c87b250a5d1d1adc7557eed41ac54f24ae6cff32fd8bc815644"
)
CORRECTED_SPLIT_FINGERPRINT = (
    "67a3082efd1a6bfd8c4eeb098ad833c7778dd141c0cca40f2d55f9e0e9546614"
)
CORRECTED_SPLIT_ASSIGNMENTS_SHA256 = (
    "056de0942918976981401db5017f6e567bc8949bae91ae395f0bd669adbfd51a"
)
CORRECTED_TASK_REGISTRY_FINGERPRINT = (
    "232e1b45542334755018f1b710741532924eee3facb77a14656fa3c1c8ce2b42"
)
CORRECTED_VOCABULARY_FINGERPRINT = (
    "3895c52d46fc980250484850afabc953d74108d9a766a409aa4e158926523deb"
)
CORRECTED_TEST_LOCK_FINGERPRINT = (
    "9e2e182cf00924fb502a9f85ee9104f82321918a26b2e8793f15dd8dee13ecb2"
)

PRIMARY_HEADS = (
    "local_key",
    "tonicized_key",
    "root",
    "bass",
    "primary_degree",
    "secondary_degree",
    "quality",
    "inversion",
)
AUXILIARY_HEADS = (
    "roman_numeral",
    "pitch_class_set",
    "harmonic_rhythm",
    "cadence",
    "pedal",
    "metrical_strength",
    "note_degree",
    "chord_tone",
    "is_root",
    "is_bass",
)
DEFERRED_HEADS = ("phrase", "section")

GROUP_WEIGHTS = {"primary": 1.0, "auxiliary": 0.25, "deferred": 0.0}
RAW_B4_HEAD_STATUS = {
    "local_key": "descriptive_only",
    "tonicized_key": "descriptive_only",
    "root": "insufficient_support",
    "bass": "insufficient_support",
    "primary_degree": "trainable_with_reweighting",
    "secondary_degree": "insufficient_support",
    "quality": "insufficient_support",
    "inversion": "trainable",
    "roman_numeral": "descriptive_only",
    "pitch_class_set": "insufficient_support",
    "harmonic_rhythm": "trainable_with_reweighting",
    "cadence": "trainable_with_reweighting",
    "phrase": "descriptive_only",
    "section": "descriptive_only",
    "pedal": "trainable_with_reweighting",
    "metrical_strength": "descriptive_only",
    "note_degree": "descriptive_only",
    "chord_tone": "trainable_with_reweighting",
    "is_root": "trainable_with_reweighting",
    "is_bass": "trainable_with_reweighting",
}

OFFICIAL_SOURCE_EVIDENCE = (
    {
        "path": "analysisgnn/train/train_analysisgnn.py",
        "sha256": "aac75dad00c3f637e96ae85b2bf8fd1b37f015f171d7f49afabdfb77c2d85a45",
        "symbols": ["TASK_DICT", "get_parser", "main"],
    },
    {
        "path": "analysisgnn/models/analysis.py",
        "sha256": "c5efc9086ce101a0732fb883ef508d09040a7109abc6b73bdee4bc9ba396dc5a",
        "symbols": [
            "ContinualAnalysisGNN.common_step",
            "ContinualAnalysisGNN.validation_step",
            "ContinualAnalysisGNN.test_step",
            "ContinualAnalysisGNN.configure_optimizers",
        ],
    },
    {
        "path": "analysisgnn/models/chord.py",
        "sha256": "74a2849d80f228419a2f18a754a7b11afdd0e977dd0407d3619ea027e22693ec",
        "symbols": ["MultiTaskLoss"],
    },
    {
        "path": "analysisgnn/data/datamodules/analysis.py",
        "sha256": "97546aad356e7632eb33714dddc930a758d6a8a6e5c04e1592b2b1d0f7797de4",
        "symbols": [
            "AnalysisDataModule.setup",
            "AnalysisDataModule.train_dataloader",
            "AnalysisDataModule.val_dataloader",
            "AnalysisDataModule.test_dataloader",
        ],
    },
    {
        "path": "analysisgnn/data/datasets/cadence.py",
        "sha256": "fa057f897b5cf01efd93472569655293fbaab35806c4298c2c6a82ca12ae65e7",
        "symbols": ["CompleteGraphCadenceDataset"],
    },
    {
        "path": "analysisgnn/data/datasets/chord.py",
        "sha256": "ac3b3dd3a0c8f5f2c536a43eb71d9d9690c973ba6461a4f7a6ae59e8f30718f1",
        "symbols": ["RNAGraphDataset", "RNAplusGraphDataset"],
    },
    {
        "path": "analysisgnn/data/datasets/dlc.py",
        "sha256": "3144af37692c708916f4d90924bc8b3dd63beceb2859013c6ef3b9db62853e36",
        "symbols": ["DLCGraphDataset", "DLCplusGraphDataset"],
    },
)


class AnalysisGNNTrainingPolicyError(ValueError):
    """Raised when a payload violates the frozen B5B contract."""


@dataclass(frozen=True, slots=True)
class HeadRoleSpec:
    task_id: str
    role: Literal["primary", "auxiliary", "deferred"]
    loss_active: bool
    metric_reportable: bool
    group_weight: float
    vocabulary_id: str
    class_count: int
    b4_raw_status: str
    deferred_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisGNNTrainingProfile:
    """Serializable contract for one future AnalysisGNN experiment."""

    schema_version: str
    profile_id: str
    scientific_role: str
    dataset_contract: Mapping[str, object]
    split_fingerprint: str | None
    task_registry_fingerprint: str
    vocabulary_fingerprint: str
    head_roles: Mapping[str, object]
    transposition_policy: Mapping[str, object]
    loss_aggregation: Mapping[str, object]
    class_weight_policy: Mapping[str, object]
    sampler_policy: Mapping[str, object]
    validation_metrics: Mapping[str, object]
    model_selection_metric: Mapping[str, object]
    optimizer_training_budget: Mapping[str, object]
    seed_policy: Mapping[str, object]
    test_lock_state: Mapping[str, object]
    reproducibility_limitations: tuple[str, ...]
    unresolved_decisions: tuple[str, ...]
    runnable: bool
    reproduction_status: str

    def semantic_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def semantic_fingerprint(self) -> str:
        return fingerprint(self.semantic_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.semantic_payload()
        payload["semantic_fingerprint"] = self.semantic_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class CorrectedLossResult:
    total: torch.Tensor | None
    primary: torch.Tensor | None
    auxiliary: torch.Tensor | None
    per_head: Mapping[str, torch.Tensor]
    zero_valid_heads: tuple[str, ...]


def _with_fingerprint(payload: Mapping[str, object]) -> dict[str, object]:
    value = dict(payload)
    value["fingerprint"] = fingerprint(payload)
    return value


def corrected_head_roles() -> tuple[HeadRoleSpec, ...]:
    """Return the exact 8/10/2 partition of the corrected 20-head registry."""

    roles: list[HeadRoleSpec] = []
    for task in PRODUCTION_TASKS:
        if task.task_id in PRIMARY_HEADS:
            role = "primary"
        elif task.task_id in AUXILIARY_HEADS:
            role = "auxiliary"
        elif task.task_id in DEFERRED_HEADS:
            role = "deferred"
        else:  # pragma: no cover - guards future registry drift
            raise AnalysisGNNTrainingPolicyError(
                f"head {task.task_id!r} has no B5B role"
            )
        deferred = role == "deferred"
        roles.append(
            HeadRoleSpec(
                task_id=task.task_id,
                role=role,
                loss_active=not deferred,
                metric_reportable=not deferred,
                group_weight=GROUP_WEIGHTS[role],
                vocabulary_id=task.vocabulary_id,
                class_count=task.class_count,
                b4_raw_status=RAW_B4_HEAD_STATUS[task.task_id],
                deferred_reason=(
                    "missing_negative_supervision" if deferred else None
                ),
            )
        )
    task_ids = [row.task_id for row in roles]
    if len(roles) != 20 or len(task_ids) != len(set(task_ids)):
        raise AnalysisGNNTrainingPolicyError("head roles must cover 20 unique heads")
    counts = {role: sum(row.role == role for row in roles) for role in GROUP_WEIGHTS}
    if counts != {"primary": 8, "auxiliary": 10, "deferred": 2}:
        raise AnalysisGNNTrainingPolicyError("head-role counts must be 8/10/2")
    return tuple(roles)


def head_role_contract() -> dict[str, object]:
    roles = corrected_head_roles()
    payload: dict[str, object] = {
        "version": HEAD_ROLE_CONTRACT_VERSION,
        "registry_id": "analysisgnn-corrected-production-heads-v1",
        "head_count": len(roles),
        "roles": [asdict(row) for row in roles],
        "role_counts": {
            role: sum(row.role == role for row in roles) for role in GROUP_WEIGHTS
        },
        "staff_included": False,
        "quality_vocabulary_id": CORRECTED_QUALITY_VOCABULARY_ID,
        "quality_class_count": TASK_BY_ID["quality"].class_count,
        "roman_numeral_class_count": TASK_BY_ID["roman_numeral"].class_count,
        "b4_semantic_fingerprint": B4_SEMANTIC_FINGERPRINT,
    }
    return _with_fingerprint(payload)


def corrected_loss_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "version": LOSS_CONTRACT_VERSION,
        "per_head": "masked_weighted_cross_entropy_mean_over_valid_rows",
        "missing_target_policy": "excluded_not_class_zero",
        "zero_valid_head_policy": "exclude_from_group_denominator_and_log",
        "within_group_head_weight": "equal_after_per_head_mean",
        "primary_heads": list(PRIMARY_HEADS),
        "auxiliary_heads": list(AUXILIARY_HEADS),
        "deferred_heads": list(DEFERRED_HEADS),
        "group_weights": dict(GROUP_WEIGHTS),
        "formula": "L_total=mean(available_primary_head_losses)+0.25*mean(available_auxiliary_head_losses)",
        "forbidden": [
            "learned_uncertainty_weights",
            "GradNorm",
            "dynamic_task_weights",
            "joint_accuracy_as_loss",
            "unregistered_single_head_boost",
        ],
        "source_entry_policy": "canonical_target_rows_before_entity_broadcast",
    }
    return _with_fingerprint(payload)


def masked_weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute one head mean over valid rows, or ``None`` for zero support."""

    if logits.ndim != 2 or targets.ndim != 1 or valid_mask.ndim != 1:
        raise AnalysisGNNTrainingPolicyError("loss tensors have invalid rank")
    if logits.shape[0] != targets.shape[0] or targets.shape != valid_mask.shape:
        raise AnalysisGNNTrainingPolicyError("loss tensors have incompatible rows")
    mask = valid_mask.to(dtype=torch.bool, device=targets.device)
    if not bool(mask.any()):
        return None
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    if bool((selected_targets < 0).any()) or bool(
        (selected_targets >= logits.shape[1]).any()
    ):
        raise AnalysisGNNTrainingPolicyError("valid target is outside vocabulary")
    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.numel() != logits.shape[1]:
            raise AnalysisGNNTrainingPolicyError("class-weight width mismatch")
        class_weights = class_weights.to(
            device=logits.device, dtype=logits.dtype
        )
    return F.cross_entropy(
        selected_logits,
        selected_targets,
        weight=class_weights,
        reduction="mean",
    )


def aggregate_corrected_losses(
    per_head: Mapping[str, torch.Tensor | None],
) -> CorrectedLossResult:
    """Apply the frozen group means without fabricating zero-head losses."""

    unknown = set(per_head) - set(TASK_BY_ID)
    if unknown:
        raise AnalysisGNNTrainingPolicyError(f"unknown loss heads: {sorted(unknown)!r}")
    deferred_present = set(per_head) & set(DEFERRED_HEADS)
    if any(per_head[task] is not None for task in deferred_present):
        raise AnalysisGNNTrainingPolicyError("deferred heads cannot enter loss")
    zero_valid = tuple(
        task
        for task in (*PRIMARY_HEADS, *AUXILIARY_HEADS)
        if task in per_head and per_head[task] is None
    )
    active = {
        task: value
        for task, value in per_head.items()
        if task not in DEFERRED_HEADS and value is not None
    }

    def group_mean(heads: Sequence[str]) -> torch.Tensor | None:
        terms = [active[task] for task in heads if task in active]
        return torch.stack(terms).mean() if terms else None

    primary = group_mean(PRIMARY_HEADS)
    auxiliary = group_mean(AUXILIARY_HEADS)
    terms: list[torch.Tensor] = []
    if primary is not None:
        terms.append(primary)
    if auxiliary is not None:
        terms.append(auxiliary * GROUP_WEIGHTS["auxiliary"])
    total = torch.stack(terms).sum() if terms else None
    return CorrectedLossResult(
        total=total,
        primary=primary,
        auxiliary=auxiliary,
        per_head=active,
        zero_valid_heads=zero_valid,
    )


def corrected_masked_multitask_loss(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    class_weights: Mapping[str, torch.Tensor] | None = None,
) -> CorrectedLossResult:
    """Testable loss primitive; it is not connected to a trainer."""

    if set(logits) != set(targets) or set(logits) != set(masks):
        raise AnalysisGNNTrainingPolicyError("logit/target/mask head sets differ")
    weights = class_weights or {}
    losses: dict[str, torch.Tensor | None] = {}
    for task in logits:
        if task in DEFERRED_HEADS:
            continue
        losses[task] = masked_weighted_cross_entropy(
            logits[task],
            targets[task],
            masks[task],
            class_weights=weights.get(task),
        )
    return aggregate_corrected_losses(losses)


def _normalized_class_weights(counts: Sequence[int]) -> tuple[float | None, ...]:
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise AnalysisGNNTrainingPolicyError("class counts must be nonnegative integers")
    raw = [None if count == 0 else 1.0 / math.sqrt(count) for count in counts]
    supported = [value for value in raw if value is not None]
    if not supported:
        return tuple(None for _ in counts)
    first_mean = sum(supported) / len(supported)
    first_normalized = [
        None if value is None else value / first_mean for value in raw
    ]
    clipped = [
        None if value is None else min(4.0, max(0.25, value))
        for value in first_normalized
    ]
    second_values = [value for value in clipped if value is not None]
    second_mean = sum(second_values) / len(second_values)
    second_normalized = [
        None if value is None else value / second_mean for value in clipped
    ]
    # A plain second division can move a clipped boundary outside [0.25, 4].
    # Solve the equivalent bounded mean-one projection deterministically so
    # both explicit policy invariants remain true in the final payload.
    lower, upper = 0.0, 16.0
    for _ in range(96):
        scale = (lower + upper) / 2.0
        projected = [
            min(4.0, max(0.25, value * scale))
            for value in second_normalized
            if value is not None
        ]
        if sum(projected) / len(projected) < 1.0:
            lower = scale
        else:
            upper = scale
    scale = (lower + upper) / 2.0
    return tuple(
        None
        if value is None
        else round(min(4.0, max(0.25, value * scale)), 12)
        for value in second_normalized
    )


def class_weight_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "version": CLASS_WEIGHT_CONTRACT_VERSION,
        "count_unit": "canonical_target_row_before_entity_broadcast",
        "source_splits": ["train"],
        "validation_used": False,
        "test_used": False,
        "augmented_view_multiplier_used": False,
        "raw_formula": "1/sqrt(train_count[c])",
        "first_normalization": "mean_supported_equals_1",
        "clip_interval": [0.25, 4.0],
        "second_normalization": "mean_supported_equals_1",
        "bounded_second_normalization": (
            "deterministic_mean_one_projection_preserving_final_clip_interval"
        ),
        "zero_count": {
            "weight": None,
            "train_supported": False,
            "positive_targets_synthesized": False,
        },
        "vocabulary_policy": "retain_all_semantic_logits",
        "quality_augmented_sixth_synthesis": False,
        "roman_numeral_class_count": 184,
    }
    return _with_fingerprint(payload)


def build_class_weight_payload(
    train_rows_by_task: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Build the full TRAIN-only payload from B4 canonical class rows."""

    if set(train_rows_by_task) != set(TASK_BY_ID):
        raise AnalysisGNNTrainingPolicyError("class rows must cover all 20 heads")
    heads: list[dict[str, object]] = []
    for task in PRODUCTION_TASKS:
        rows = sorted(
            train_rows_by_task[task.task_id], key=lambda row: int(row["class_id"])
        )
        if len(rows) != task.class_count:
            raise AnalysisGNNTrainingPolicyError(
                f"{task.task_id} class-row count differs from vocabulary"
            )
        counts = [int(row["canonical_target_row_count"]) for row in rows]
        weights = _normalized_class_weights(counts)
        class_rows: list[dict[str, object]] = []
        for expected_id, (row, weight) in enumerate(zip(rows, weights, strict=True)):
            class_id = int(row["class_id"])
            if class_id != expected_id:
                raise AnalysisGNNTrainingPolicyError(
                    f"{task.task_id} class IDs are not contiguous"
                )
            value = str(row["class_value"])
            if value != get_vocabulary(task.vocabulary_id).labels[class_id]:
                raise AnalysisGNNTrainingPolicyError(
                    f"{task.task_id} class value differs from frozen vocabulary"
                )
            count = counts[class_id]
            support_tier = str(row["support_tier"])
            if count == 0:
                support_status = "unsupported"
            elif support_tier in {"insufficient", "fragile"}:
                support_status = "weakly_supported"
            else:
                support_status = "supported"
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_value": value,
                    "train_canonical_target_row_count": count,
                    "train_component_count": int(row["component_count"]),
                    "train_support_tier": support_tier,
                    "support_status": support_status,
                    "train_supported": count > 0,
                    "weight": weight,
                }
            )
        supported_weights = [
            float(row["weight"])
            for row in class_rows
            if row["weight"] is not None
        ]
        heads.append(
            {
                "task_id": task.task_id,
                "vocabulary_id": task.vocabulary_id,
                "class_count": task.class_count,
                "loss_active": task.task_id not in DEFERRED_HEADS,
                "classes": class_rows,
                "unsupported_classes": [
                    row["class_value"]
                    for row in class_rows
                    if row["support_status"] == "unsupported"
                ],
                "weakly_supported_classes": [
                    row["class_value"]
                    for row in class_rows
                    if row["support_status"] == "weakly_supported"
                ],
                "supported_weight_mean": (
                    round(sum(supported_weights) / len(supported_weights), 12)
                    if supported_weights
                    else None
                ),
            }
        )
    payload: dict[str, object] = {
        "contract": class_weight_contract(),
        "b4_semantic_fingerprint": B4_SEMANTIC_FINGERPRINT,
        "head_count": len(heads),
        "heads": heads,
    }
    return _with_fingerprint(payload)


def validate_class_weight_payload(payload: Mapping[str, object]) -> None:
    observed = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if observed != fingerprint(body):
        raise AnalysisGNNTrainingPolicyError("class-weight fingerprint mismatch")
    heads = payload.get("heads")
    if not isinstance(heads, list) or len(heads) != 20:
        raise AnalysisGNNTrainingPolicyError("class-weight payload must have 20 heads")
    rebuilt_input: dict[str, list[dict[str, object]]] = {}
    for head in heads:
        if not isinstance(head, dict) or not isinstance(head.get("classes"), list):
            raise AnalysisGNNTrainingPolicyError("invalid class-weight head")
        rebuilt_input[str(head["task_id"])] = [
            {
                "class_id": row["class_id"],
                "class_value": row["class_value"],
                "canonical_target_row_count": row[
                    "train_canonical_target_row_count"
                ],
                "component_count": row["train_component_count"],
                "support_tier": row["train_support_tier"],
            }
            for row in head["classes"]
        ]
    rebuilt = build_class_weight_payload(rebuilt_input)
    if rebuilt != payload:
        raise AnalysisGNNTrainingPolicyError("class-weight payload is not reproducible")


def component_sampler_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "version": SAMPLER_CONTRACT_VERSION,
        "draws_per_epoch": 1295,
        "train_component_count": 1209,
        "steps": [
            "uniform_train_source_component",
            "uniform_record_within_component",
            "choose_valid_graph_or_window_view",
            "resolve_transposition_from_profile",
            "apply_target_masks",
        ],
        "component_probability": "1/train_component_count",
        "record_probability": "1/(train_component_count*records_in_component)",
        "rng": "sha256(contract_version,seed,epoch,draw_index,stage,component_if_applicable)",
        "with_replacement_across_draws": True,
        "broadcast_note_rows_are_samples": False,
        "augmented_variants_are_records": False,
        "split_mutation": False,
        "validation_sampling": "identity_complete_fixed_view_without_oversampling",
        "test_sampling": "disabled_loader_not_created",
    }
    return _with_fingerprint(payload)


def _draw_index(parts: Sequence[object], size: int) -> int:
    if size <= 0:
        raise AnalysisGNNTrainingPolicyError("cannot draw from an empty population")
    payload = "\x00".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % size


def component_balanced_record_draw(
    component_records: Mapping[str, Sequence[str]],
    *,
    seed: int,
    epoch: int,
    draw_index: int,
) -> tuple[str, str]:
    """Return a deterministic pseudo-uniform component then record draw."""

    if epoch < 0 or draw_index < 0:
        raise AnalysisGNNTrainingPolicyError("epoch and draw index must be nonnegative")
    components = sorted(component_records)
    if not components or any(not component_records[key] for key in components):
        raise AnalysisGNNTrainingPolicyError("components must be nonempty")
    contract = SAMPLER_CONTRACT_VERSION
    component = components[
        _draw_index((contract, seed, epoch, draw_index, "component"), len(components))
    ]
    records = sorted(component_records[component])
    if len(records) != len(set(records)):
        raise AnalysisGNNTrainingPolicyError("component records must be unique")
    record = records[
        _draw_index(
            (contract, seed, epoch, draw_index, "record", component), len(records)
        )
    ]
    return component, record


def corrected_sampler_draw(
    component_records: Mapping[str, Sequence[str]],
    *,
    profile_id: str,
    valid_shifts_by_record: Mapping[str, Sequence[int]],
    seed: int,
    epoch: int,
    draw_index: int,
) -> tuple[str, str, int]:
    component, record = component_balanced_record_draw(
        component_records, seed=seed, epoch=epoch, draw_index=draw_index
    )
    if profile_id == CORRECTED_NO_TRANSPOSITION_PROFILE_ID:
        shift = 0
    elif profile_id == CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID:
        shifts = tuple(valid_shifts_by_record.get(record, ()))
        if not shifts:
            raise AnalysisGNNTrainingPolicyError("C1 record has no valid shift")
        shift = select_record_shift(record, shifts, seed=seed, epoch=epoch)
    else:
        raise AnalysisGNNTrainingPolicyError("sampler accepts only C0 or C1")
    return component, record, shift


def corrected_metric_contract() -> dict[str, object]:
    reportable = tuple(task for task in TASK_BY_ID if task not in DEFERRED_HEADS)
    payload: dict[str, object] = {
        "version": METRIC_CONTRACT_VERSION,
        "reportable_heads": list(reportable),
        "deferred_heads": list(DEFERRED_HEADS),
        "per_head_metrics": [
            "masked_cross_entropy",
            "accuracy",
            "balanced_accuracy",
            "macro_f1_observed_validation_classes",
            "per_class_precision_recall_f1_support",
            "train_supported_class_count",
            "train_absent_class_count",
            "validation_only_class_count",
            "record_support",
            "component_support",
            "full_vocabulary_coverage",
        ],
        "macro_f1_scope_disclosure": (
            "observed_validation_classes_and_full_vocabulary_coverage_are_separate"
        ),
        "primary_model_selection": {
            "metric_id": "corrected_primary_macro_score",
            "heads": list(PRIMARY_HEADS),
            "formula": "mean_macro_f1_observed_over_primary_heads_with_valid_validation_targets",
            "zero_valid_head_policy": "exclude_and_log",
            "auxiliary_included": False,
            "deferred_included": False,
            "direction": "max",
        },
        "joint_metrics": [
            {
                "metric_id": CORRECTED_V2_METRIC_ID,
                "unit": "harmonic_event",
                "components": list(PAPER_DEFINED_JOINT_COMPONENTS),
                "quality_vocabulary_id": CORRECTED_QUALITY_VOCABULARY_ID,
                "paper_compatible": False,
            },
            {
                "metric_id": PAPER_TEXT_COMPATIBILITY_METRIC_ID,
                "unit": "note",
                "components": list(PAPER_DEFINED_JOINT_COMPONENTS),
                "quality_vocabulary_id": COMPATIBILITY_QUALITY_VOCABULARY_ID,
                "requires_unambiguous_note_to_harmonic_event": True,
                "inference_implemented": False,
            },
        ],
        "joint_slices": [
            "train",
            "validation",
            "seen_tuple",
            "unseen_tuple",
            "validation_tuple_absent_from_train_count",
        ],
        "roman_numeral": {
            "direct_auxiliary_metrics": ["accuracy", "macro_f1_observed"],
            "direct_class_count": 184,
            "derived_harmonic_correctness_separate": True,
        },
        "test_metrics": False,
    }
    return _with_fingerprint(payload)


def corrected_optimizer_envelope() -> dict[str, object]:
    """Freeze known B1-compatible values and expose model-owned unknowns."""

    payload: dict[str, object] = {
        "source_contract": "Phase9EB1Config@1.1.1",
        "model_identity": "analysisgnn-hybridgnn-3x256-bigru2-bidirectional-out128-v2-adaptation",
        "model_implementation_status": "not_implemented",
        "model_contract": {
            "encoder": "HybridGNN",
            "hidden_channels": 256,
            "output_channels": 128,
            "num_layers": 3,
            "dropout": 0.3,
            "use_jk": True,
            "use_beat_hierarchy": True,
            "use_measure_hierarchy": True,
            "bigru_layers": 2,
            "bigru_bidirectional": True,
            "logit_fusion": True,
            "head_registry": "analysisgnn-corrected-production-heads-v1",
        },
        "parameter_budget": None,
        "optimizer": "AdamW",
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "warmup_applied_updates": 500,
        "scheduler": "linear_warmup_then_cosine",
        "gradient_clip_norm": 1.0,
        "batch_window_policy": None,
        "mixed_precision": "fp32_baseline",
        "budgets": {
            "smoke_optimizer_updates": 2,
            "pilot_optimizer_updates": 500,
            "main_optimizer_updates": 10000,
            "pilot_validation_interval": 100,
            "main_validation_interval": 500,
            "checkpoint_interval": 500,
        },
        "checkpoint_selection_metric": "corrected_primary_macro_score",
        "early_stopping": False,
        "fixed_update_budget": True,
        "deterministic_flags": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "epoch_boundary_resume": True,
        },
        "unresolved": ["parameter_budget", "batch_window_policy"],
    }
    return _with_fingerprint(payload)


def official_contracts() -> dict[str, object]:
    """Serialize pinned-code behavior without correcting its defects."""

    official_heads = [
        ("cadence", 4),
        ("localkey", 50),
        ("tonkey", 50),
        ("quality", 15),
        ("inversion", 4),
        ("root", 38),
        ("bass", 38),
        ("degree1", 22),
        ("degree2", 22),
        ("hrythm", 2),
        ("pcset", 94),
        ("romanNumeral", 185),
        ("section", 2),
        ("phrase", 2),
        ("organ_point", 2),
        ("tpc_in_label", 2),
        ("tpc_is_root", 2),
        ("tpc_is_bass", 2),
        ("downbeat", 45),
        ("note_degree", 49),
        ("staff", 4),
    ]
    heads = _with_fingerprint(
        {
            "registry": "pinned_TASK_DICT_unique_keys",
            "head_count": 21,
            "heads": [
                {"code_name": name, "class_count": count, "loss_active_if_present": True}
                for name, count in official_heads
            ],
            "staff_included": True,
            "quality_vocabulary_id": COMPATIBILITY_QUALITY_VOCABULARY_ID,
            "quality_class_count": 15,
            "roman_numeral_head_class_count": 185,
        }
    )
    loss = _with_fingerprint(
        {
            "per_head": "CrossEntropyLoss(ignore_index=-1,label_smoothing=0.1)",
            "multi_task_strategy": "learned_half_inverse_square_plus_log1p_square",
            "normalization": "divide_current_loader_total_by_present_label_dict_size",
            "combined_loader_normalization": "mean_over_all_combined_loader_keys",
            "feature_l2_regularizer_weight": 0.1,
            "edge_loss_default": False,
            "class_weights": None,
            "known_defects_preserved": [
                "out_of_range_labels_are_replaced_by_class_zero",
                "learned_weight_parameter_index_follows_present_dict_iteration_not_global_task_id",
            ],
        }
    )
    sampler = _with_fingerprint(
        {
            "loader": "MuseNeighborLoader",
            "combined_loader": "CombinedLoader(min_size)",
            "main_tasks": ["all", "cadence", "rna"],
            "batch_size_public_run": 240,
            "per_loader_batch_size": 80,
            "subgraph_size": 500,
            "num_neighbors": [5, 5],
            "subgraph_sample_ratio": 0.5,
            "component_balanced": False,
            "loader_shuffle": "not_set_in_pinned_call_framework_default",
            "augmentation": "materialized_views_before_train_validation_split",
        }
    )
    metrics = _with_fingerprint(
        {
            "validation": ["total_loss", "per_head_loss", "accuracy", "macro_f1"],
            "validation_note_nct_joint_components": [
                "localkey",
                "degree1",
                "degree2",
                "quality",
                "inversion",
            ],
            "test_onset_joint_components": [
                "degree1",
                "degree2",
                "quality",
                "inversion",
            ],
            "test_nct_joint_components": [
                "localkey",
                "degree1",
                "degree2",
                "quality",
                "inversion",
            ],
            "model_selection": "min_val/total_loss",
            "paper_code_disagreement_preserved": True,
        }
    )
    optimizer = _with_fingerprint(
        {
            "source": "historical public run rhsjiz03 plus pinned source",
            "model": "HybridGNN_3x256_out128_jk_logit_fusion_beats_measures",
            "parameter_budget": None,
            "optimizer": "AdamW",
            "learning_rate": 0.005,
            "weight_decay": 0.005,
            "scheduler": "custom_step_interval_linear_warmup_cosine",
            "warmup": "min(500,total_steps//20)",
            "eta_min": 0.00005,
            "epochs_config": 100,
            "trainer_max_epochs": 101,
            "gradient_clip_norm": 1.0,
            "swa_public_run": {"enabled": True, "lr": 0.00005, "epoch_start": 50},
            "mixed_precision": "not_enabled_in_pinned_entrypoint",
        }
    )
    payload = {
        "head_roles": heads,
        "loss": loss,
        "class_weights": _with_fingerprint(
            {"policy": "none", "source": "pinned ContinualAnalysisGNN"}
        ),
        "sampler": sampler,
        "metrics": metrics,
        "optimizer": optimizer,
        "source_evidence": list(OFFICIAL_SOURCE_EVIDENCE),
        "paper_text_evidence": {
            "task_count": 20,
            "prediction_unit": "note",
            "joint_components": list(PAPER_DEFINED_JOINT_COMPONENTS),
            "quality_class_count": 15,
        },
        "pinned_code_vs_paper": {
            "unique_head_count": 21,
            "code_only_head": "staff",
            "validation_joint_includes_local_key": True,
            "test_onset_joint_includes_local_key": False,
        },
        "available_corpus_evidence": {
            "dilemmadata_present": True,
            "separate_cadence_corpus_present": False,
            "corrected_v2_substitution_allowed": False,
        },
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _corrected_dataset_contract() -> dict[str, object]:
    return _with_fingerprint(
        {
            "universe_id": PAPER_CANDIDATE_UNIVERSE_ID,
            "record_count": 1619,
            "split_record_counts": {"train": 1295, "validation": 162, "test": 162},
            "split_component_counts": {"train": 1209, "validation": 147, "test": 151},
            "split_assignments_sha256": CORRECTED_SPLIT_ASSIGNMENTS_SHA256,
            "b3_semantic_fingerprint": B3_SEMANTIC_FINGERPRINT,
            "component_identity_frozen": True,
            "test_assignment_frozen": True,
            "test_targets_available_to_training_policy": False,
        }
    )


def _corrected_test_lock() -> dict[str, object]:
    return {
        "lock_fingerprint": CORRECTED_TEST_LOCK_FINGERPRINT,
        "assignment_frozen": True,
        "loader_created": False,
        "targets_read": False,
        "metrics_computed": False,
        "checkpoint_selection_allowed": False,
        "explicit_future_unlock_required": True,
    }


def _corrected_seed_policy() -> dict[str, object]:
    return {
        "smoke_seed": 17,
        "pilot_seed": 17,
        "main_single_seed": 17,
        "confirmation_seeds": [17, 23, 42],
        "multi_seed_before_single_seed_analysis": False,
        "paired_c0_c1_initialization_and_order": True,
    }


def _corrected_transposition_policy(enabled: bool) -> dict[str, object]:
    if not enabled:
        return _with_fingerprint(
            {
                "enabled": False,
                "profile_id": "identity-only",
                "train": "identity_only",
                "validation": "identity_only",
                "test": "identity_only_without_target_access",
            }
        )
    policy = corrected_transposition_profile()
    return _with_fingerprint(
        {
            "enabled": True,
            "profile_id": CORRECTED_TRANSPOSITION_PROFILE_ID,
            "profile_fingerprint": policy["fingerprint"],
            "train": "on_the_fly_uniform_over_record_valid_shifts_including_identity",
            "validation": "identity_only",
            "test": "identity_only_without_target_access",
            "whole_shift_fail_closed": True,
            "octave_folding": False,
            "component_identity_changes": False,
        }
    )


def build_training_profiles(
    class_weight_payload: Mapping[str, object],
) -> dict[str, AnalysisGNNTrainingProfile]:
    """Build O/C0/C1 from one calculated class-weight payload."""

    validate_class_weight_payload(class_weight_payload)
    official = official_contracts()
    official_transposition = official_transposition_evidence()
    official_profile = AnalysisGNNTrainingProfile(
        schema_version=TRAINING_PROFILE_SCHEMA,
        profile_id=OFFICIAL_TRAINING_PROFILE_ID,
        scientific_role="official_pinned_code_reproduction",
        dataset_contract={
            "identity": "official_AnalysisGNN_AN_DLC_plus_external_cadence_corpora",
            "official_training_contract_fingerprint": official["fingerprint"],
            "corrected_v2_dataset_substitution_allowed": False,
            "dilemmadata_available": True,
            "separate_cadence_corpus_available": False,
            "exact_official_split_available": False,
        },
        split_fingerprint=None,
        task_registry_fingerprint=str(official["head_roles"]["fingerprint"]),  # type: ignore[index]
        vocabulary_fingerprint="09e7b4b6a07c43d0b4c0eb0bcc6e4e7d2101c4cd89e2ec81f8824cad50da9da5",
        head_roles=official["head_roles"],  # type: ignore[arg-type]
        transposition_policy={
            "enabled": True,
            "profile_id": OFFICIAL_TRANSPOSITION_PROFILE_ID,
            "profile_fingerprint": official_transposition["fingerprint"],
            "semantics": official_transposition,
        },
        loss_aggregation=official["loss"],  # type: ignore[arg-type]
        class_weight_policy=official["class_weights"],  # type: ignore[arg-type]
        sampler_policy=official["sampler"],  # type: ignore[arg-type]
        validation_metrics=official["metrics"],  # type: ignore[arg-type]
        model_selection_metric={"metric_id": "val/total_loss", "direction": "min"},
        optimizer_training_budget=official["optimizer"],  # type: ignore[arg-type]
        seed_policy={
            "pinned_entrypoint_seed": 0,
            "historical_public_run_seed_evidence": "not_attested_as_complete_rng_state",
        },
        test_lock_state={
            "official_code_has_test_loader_and_do_eval_path": True,
            "b5b_execution": False,
            "v2_test_targets_used": False,
        },
        reproducibility_limitations=(
            "historical W&B run source commit 7738a282 differs from pinned public e115182",
            "exact historical GraphMuse revision is unpublished",
            "separate official cadence corpus is unavailable",
            "official split and materialized-view bytes are not frozen locally",
            "paper text and pinned validation/test joint metrics disagree",
            "pinned source is under construction and preserves known label/transposition defects",
        ),
        unresolved_decisions=(
            "exact official corpus and split identity",
            "exact historical dependency revision set",
            "complete historical RNG and loader-order state",
            "parameter budget for the exact historical environment",
        ),
        runnable=False,
        reproduction_status="partial_contract_only",
    )

    roles = head_role_contract()
    loss = corrected_loss_contract()
    sampler = component_sampler_contract()
    metrics = corrected_metric_contract()
    optimizer = corrected_optimizer_envelope()
    dataset = _corrected_dataset_contract()
    class_weight_ref = {
        "contract_fingerprint": class_weight_payload["contract"]["fingerprint"],  # type: ignore[index]
        "payload_fingerprint": class_weight_payload["fingerprint"],
        "full_payload_location": "phase9eb5b_training_policy_fixture.class_weight_payload",
        "train_only": True,
    }
    common = {
        "schema_version": TRAINING_PROFILE_SCHEMA,
        "scientific_role": "corrected_v2_transposition_ablation",
        "dataset_contract": dataset,
        "split_fingerprint": CORRECTED_SPLIT_FINGERPRINT,
        "task_registry_fingerprint": CORRECTED_TASK_REGISTRY_FINGERPRINT,
        "vocabulary_fingerprint": CORRECTED_VOCABULARY_FINGERPRINT,
        "head_roles": roles,
        "loss_aggregation": loss,
        "class_weight_policy": class_weight_ref,
        "sampler_policy": sampler,
        "validation_metrics": metrics,
        "model_selection_metric": metrics["primary_model_selection"],
        "optimizer_training_budget": optimizer,
        "seed_policy": _corrected_seed_policy(),
        "test_lock_state": _corrected_test_lock(),
        "reproducibility_limitations": (
            "model and trainer are not implemented in Phase 9E-B5B",
            "no accuracy or augmentation-benefit claim exists before paired execution",
        ),
        "unresolved_decisions": (
            "exact parameter budget after 20-head implementation",
            "exact graph/window batching implementation",
        ),
        "runnable": False,
        "reproduction_status": "ready_for_model_implementation",
    }
    c0 = AnalysisGNNTrainingProfile(
        profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
        transposition_policy=_corrected_transposition_policy(False),
        **common,
    )
    c1 = AnalysisGNNTrainingProfile(
        profile_id=CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
        transposition_policy=_corrected_transposition_policy(True),
        **common,
    )
    return {"O": official_profile, "C0": c0, "C1": c1}


def _nested_diff(left: object, right: object, prefix: str = "") -> tuple[str, ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_nested_diff(left[key], right[key], path))
        return tuple(paths)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return (prefix,)
        paths: list[str] = []
        for index, (lvalue, rvalue) in enumerate(zip(left, right, strict=True)):
            paths.extend(_nested_diff(lvalue, rvalue, f"{prefix}[{index}]"))
        return tuple(paths)
    return () if left == right else (prefix,)


def corrected_profile_comparison(
    c0: AnalysisGNNTrainingProfile,
    c1: AnalysisGNNTrainingProfile,
) -> dict[str, object]:
    """Prove that the only substantive C0/C1 delta is transposition."""

    left = c0.semantic_payload()
    right = c1.semantic_payload()
    identity_differences = []
    if left.pop("profile_id") != right.pop("profile_id"):
        identity_differences.append("profile_id")
    left_transposition = left.pop("transposition_policy")
    right_transposition = right.pop("transposition_policy")
    other = _nested_diff(left, right)
    transposition_differences = _nested_diff(
        left_transposition, right_transposition, "transposition_policy"
    )
    payload: dict[str, object] = {
        "profile_ids": [c0.profile_id, c1.profile_id],
        "identity_differences": identity_differences,
        "substantive_difference_domains": (
            ["transposition_policy"] if transposition_differences else []
        ),
        "transposition_difference_paths": list(transposition_differences),
        "other_difference_paths": list(other),
        "only_transposition_differs": bool(transposition_differences) and not other,
    }
    return _with_fingerprint(payload)


def stop_gate_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "finite_total_loss": True,
        "finite_per_head_losses": True,
        "finite_logits": True,
        "active_head_nonzero_gradient_control_window": True,
        "masked_targets_excluded": True,
        "deferred_heads_excluded_from_optimizer": True,
        "test_loader_created": False,
        "test_targets_read": False,
        "frozen_split_component_fingerprints_required": True,
        "c0_c1_only_transposition_difference": True,
        "validation_after_optimizer_step": True,
        "checkpoint_selection_uses_test": False,
        "unsupported_train_classes_reported": True,
        "entity_row_volume_cannot_set_task_weight": True,
    }
    return _with_fingerprint(payload)


def experiment_matrix_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "stage_1_smoke": {
            "profiles": "each_runnable_profile",
            "optimizer_updates": 2,
            "seed": 17,
            "test": "disabled",
        },
        "stage_2_bounded_pilot": {
            "profiles": ["O", "C0", "C1"],
            "optimizer_updates": 500,
            "seed": 17,
            "validation_every_updates": 100,
            "test": "disabled",
            "skip_nonrunnable_without_substitution": True,
        },
        "stage_3_main_single_seed": {
            "profiles": ["C0", "C1"],
            "seed": 17,
            "paired_everything_except_transposition": True,
            "sole_augmentation_benefit_comparison": True,
        },
        "stage_4_multi_seed_confirmation": {
            "profiles": ["C0", "C1"],
            "seeds": [17, 23, 42],
            "requires_single_seed_analysis_first": True,
        },
    }
    return _with_fingerprint(payload)


def combined_training_policy_contract(
    class_weight_payload: Mapping[str, object],
) -> dict[str, object]:
    profiles = build_training_profiles(class_weight_payload)
    comparison = corrected_profile_comparison(profiles["C0"], profiles["C1"])
    payload: dict[str, object] = {
        "schema": TRAINING_POLICY_SCHEMA,
        "head_role_fingerprint": head_role_contract()["fingerprint"],
        "loss_fingerprint": corrected_loss_contract()["fingerprint"],
        "class_weight_fingerprint": class_weight_payload["fingerprint"],
        "sampler_fingerprint": component_sampler_contract()["fingerprint"],
        "metric_fingerprint": corrected_metric_contract()["fingerprint"],
        "profile_fingerprints": {
            key: profile.semantic_fingerprint for key, profile in profiles.items()
        },
        "corrected_profile_comparison_fingerprint": comparison["fingerprint"],
        "stop_gate_fingerprint": stop_gate_contract()["fingerprint"],
        "experiment_matrix_fingerprint": experiment_matrix_contract()["fingerprint"],
        "training_run": False,
        "validation_inference_run": False,
        "test_evaluated": False,
        "test_targets_used_for_evaluation": False,
    }
    return _with_fingerprint(payload)


__all__ = [
    "AUXILIARY_HEADS",
    "AnalysisGNNTrainingPolicyError",
    "AnalysisGNNTrainingProfile",
    "CLASS_WEIGHT_CONTRACT_VERSION",
    "CORRECTED_NO_TRANSPOSITION_PROFILE_ID",
    "CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID",
    "CorrectedLossResult",
    "DEFERRED_HEADS",
    "GROUP_WEIGHTS",
    "HEAD_ROLE_CONTRACT_VERSION",
    "LOSS_CONTRACT_VERSION",
    "METRIC_CONTRACT_VERSION",
    "OFFICIAL_TRAINING_PROFILE_ID",
    "PRIMARY_HEADS",
    "SAMPLER_CONTRACT_VERSION",
    "TRAINING_POLICY_SCHEMA",
    "TRAINING_PROFILE_SCHEMA",
    "aggregate_corrected_losses",
    "build_class_weight_payload",
    "build_training_profiles",
    "class_weight_contract",
    "combined_training_policy_contract",
    "component_balanced_record_draw",
    "component_sampler_contract",
    "corrected_head_roles",
    "corrected_loss_contract",
    "corrected_masked_multitask_loss",
    "corrected_metric_contract",
    "corrected_optimizer_envelope",
    "corrected_profile_comparison",
    "corrected_sampler_draw",
    "experiment_matrix_contract",
    "head_role_contract",
    "masked_weighted_cross_entropy",
    "official_contracts",
    "stop_gate_contract",
    "validate_class_weight_payload",
]
