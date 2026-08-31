"""Phase 9E-B4 deterministic AnalysisGNN class-balance audit contract.

The contract consumes target observations only after a caller has applied the
frozen split assignment.  It is deliberately independent of model, loss, and
sampling runtime code.  Entity rows and canonical source-target rows are
different units throughout: repeated/broadcast entities may share one stable
source-row identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Literal, TypeVar

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    COMPATIBILITY_QUALITY_VOCABULARY_ID,
    CORRECTED_QUALITY_VOCABULARY_ID,
    PAPER_DEFINED_JOINT_COMPONENTS,
    PRODUCTION_TASKS,
    TaskSpec,
    get_task,
    get_vocabulary,
    project_quality_for_analysisgnn,
)


CLASS_BALANCE_SCHEMA = "DilemmadataAnalysisGNNClassBalanceAudit@1.0.0"
CLASS_BALANCE_CONTRACT_VERSION = "analysisgnn-class-balance-contract-v1"
SPLIT_ORDER = ("train", "validation")
TRAIN_SUPPORT_ORDER = ("absent", "insufficient", "fragile", "usable", "broad")
VALIDATION_SUPPORT_ORDER = ("unobservable", "fragile_validation", "observable")
TRAINABILITY_ORDER = (
    "trainable",
    "trainable_with_reweighting",
    "insufficient_support",
    "descriptive_only",
)
TRAIN_THRESHOLDS = {
    "insufficient_min_canonical_rows": 20,
    "insufficient_min_components": 3,
    "usable_min_canonical_rows": 100,
    "usable_min_components": 10,
    "broad_min_canonical_rows": 1000,
    "broad_min_components": 50,
}
VALIDATION_THRESHOLDS = {
    "observable_min_canonical_rows": 10,
    "observable_min_components": 2,
}
HEAD_THRESHOLDS = {
    "descriptive_problem_class_fraction": 0.25,
    "descriptive_min_available_train_components": 20,
    "majority_share": 0.50,
    "max_to_min_nonzero_ratio": 20.0,
    "normalized_entropy": 0.70,
}
SAMPLING_DIAGNOSTIC_THRESHOLDS = {
    "largest_component_share": 0.50,
    "effective_component_fraction": 0.50,
    "broadcast_factor": 2.0,
}
EFFECTIVE_NUMBER_BETA = 0.9999
WEIGHT_METHODS = (
    "inverse_frequency",
    "inverse_sqrt_frequency",
    "effective_number",
)
FUTURE_MULTICLASS_METRICS = (
    "micro_accuracy",
    "balanced_accuracy",
    "macro_f1",
    "per_class_precision_recall_f1",
    "support_by_class",
    "record_level_support",
    "component_level_support",
)
FUTURE_JOINT_METRICS = (
    "exact_joint_accuracy",
    "component_wise_accuracy",
    "support_count",
    "seen_tuple_accuracy",
    "unseen_tuple_accuracy",
)


class AnalysisGNNClassBalanceError(ValueError):
    """A B4 observation or artifact violates the frozen audit contract."""


@dataclass(frozen=True, slots=True)
class AuditTaskSpec:
    task_id: str
    vocabulary_id: str
    entity_type: str
    labels: tuple[str, ...]

    @property
    def vocabulary_size(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, slots=True)
class EntityTargetObservation:
    task_id: str
    entity_id: str
    source_row_id: str | None
    class_value: str | None
    available: bool
    masked: bool


@dataclass(frozen=True, slots=True)
class RecordTargetObservations:
    record_id: str
    component_id: str
    dialect: Literal["an_joint", "dlc"]
    split: Literal["train", "validation"]
    targets: tuple[EntityTargetObservation, ...]


@dataclass(frozen=True, slots=True)
class JointTupleObservation:
    mode: Literal["corrected_harmonic_event", "compatibility_note"]
    split: Literal["train", "validation"]
    record_id: str
    component_id: str
    dialect: Literal["an_joint", "dlc"]
    row_id: str
    canonical_harmonic_row_id: str
    values: tuple[str, str, str, str, str]


@dataclass(slots=True)
class _ClassStats:
    entity_count: int = 0
    canonical_target_row_count: int = 0
    records: set[str] = field(default_factory=set)
    components: set[str] = field(default_factory=set)
    record_rows: Counter[str] = field(default_factory=Counter)
    component_rows: Counter[str] = field(default_factory=Counter)
    dialect_entity_count: Counter[str] = field(default_factory=Counter)
    dialect_row_count: Counter[str] = field(default_factory=Counter)
    dialect_records: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    dialect_components: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )


def _task_spec(task: TaskSpec) -> AuditTaskSpec:
    vocabulary = get_vocabulary(task.vocabulary_id)
    return AuditTaskSpec(
        task_id=task.task_id,
        vocabulary_id=task.vocabulary_id,
        entity_type=task.entity_type,
        labels=vocabulary.labels,
    )


PRODUCTION_AUDIT_TASKS = tuple(_task_spec(task) for task in PRODUCTION_TASKS)
COMPATIBILITY_QUALITY_TASK = AuditTaskSpec(
    task_id="quality_compatibility",
    vocabulary_id=COMPATIBILITY_QUALITY_VOCABULARY_ID,
    entity_type="harmonic_event",
    labels=get_vocabulary(COMPATIBILITY_QUALITY_VOCABULARY_ID).labels,
)


def class_balance_contract() -> dict[str, object]:
    """Return the versioned formulas/constants included in every fingerprint."""

    payload: dict[str, object] = {
        "effective_component_count_formula": "(sum_i n_i)^2/sum_i(n_i^2)",
        "effective_number_beta": EFFECTIVE_NUMBER_BETA,
        "effective_number_weight_formula": "(1-beta)/(1-beta**n)",
        "entropy_formula": "-sum(p_i*log(p_i))/log(vocabulary_size)",
        "future_joint_metrics": list(FUTURE_JOINT_METRICS),
        "future_multiclass_metrics": list(FUTURE_MULTICLASS_METRICS),
        "head_thresholds": HEAD_THRESHOLDS,
        "missing_policy": "missing_or_masked_is_never_a_class",
        "sampling_diagnostic_thresholds": SAMPLING_DIAGNOSTIC_THRESHOLDS,
        "schema": CLASS_BALANCE_SCHEMA,
        "split_order": list(SPLIT_ORDER),
        "train_thresholds": TRAIN_THRESHOLDS,
        "validation_thresholds": VALIDATION_THRESHOLDS,
        "version": CLASS_BALANCE_CONTRACT_VERSION,
        "weight_methods": list(WEIGHT_METHODS),
        "weight_normalization": "mean_one_over_nonzero_train_classes",
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def train_support_tier(canonical_rows: int, components: int) -> str:
    if canonical_rows < 0 or components < 0:
        raise AnalysisGNNClassBalanceError("support counts cannot be negative")
    if canonical_rows == 0:
        return "absent"
    if (
        canonical_rows < TRAIN_THRESHOLDS["insufficient_min_canonical_rows"]
        or components < TRAIN_THRESHOLDS["insufficient_min_components"]
    ):
        return "insufficient"
    if (
        canonical_rows >= TRAIN_THRESHOLDS["broad_min_canonical_rows"]
        and components >= TRAIN_THRESHOLDS["broad_min_components"]
    ):
        return "broad"
    if (
        canonical_rows >= TRAIN_THRESHOLDS["usable_min_canonical_rows"]
        and components >= TRAIN_THRESHOLDS["usable_min_components"]
    ):
        return "usable"
    return "fragile"


def validation_support_tier(canonical_rows: int, components: int) -> str:
    if canonical_rows < 0 or components < 0:
        raise AnalysisGNNClassBalanceError("support counts cannot be negative")
    if canonical_rows == 0:
        return "unobservable"
    if (
        canonical_rows < VALIDATION_THRESHOLDS["observable_min_canonical_rows"]
        or components < VALIDATION_THRESHOLDS["observable_min_components"]
    ):
        return "fragile_validation"
    return "observable"


def _shares(counts: Mapping[str, int]) -> tuple[float, float, float]:
    values = sorted((int(value) for value in counts.values() if value > 0), reverse=True)
    total = sum(values)
    if total == 0:
        return 0.0, 0.0, 0.0
    largest = values[0] / total
    top_five = sum(values[:5]) / total
    effective = total * total / sum(value * value for value in values)
    return largest, top_five, effective


def _round(value: float) -> float:
    return round(value, 12)


def _dialect_name(value: str) -> str:
    return "an" if value == "an_joint" else value


class ClassBalanceAccumulator:
    """Incrementally aggregate per-class support without retaining corpus rows."""

    def __init__(self, tasks: Sequence[AuditTaskSpec] = PRODUCTION_AUDIT_TASKS):
        self.tasks = tuple(tasks)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        if len(self.task_by_id) != len(self.tasks):
            raise AnalysisGNNClassBalanceError("duplicate audit task ID")
        self.class_ids = {
            task.task_id: {value: index for index, value in enumerate(task.labels)}
            for task in self.tasks
        }
        self.stats: dict[tuple[str, str, int], _ClassStats] = defaultdict(_ClassStats)
        self.state_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.available_components: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.records_seen: Counter[str] = Counter()

    def add_record(self, record: RecordTargetObservations) -> None:
        if record.split not in SPLIT_ORDER:
            raise AnalysisGNNClassBalanceError(
                "target-aware class balance accepts only TRAIN/VALIDATION"
            )
        if record.dialect not in ("an_joint", "dlc"):
            raise AnalysisGNNClassBalanceError("unknown Dilemmadata dialect")
        self.records_seen[record.split] += 1
        seen_entities: set[tuple[str, str]] = set()
        local_rows: set[tuple[str, int, str]] = set()
        for observation in record.targets:
            task = self.task_by_id.get(observation.task_id)
            if task is None:
                raise AnalysisGNNClassBalanceError(
                    f"unknown audit task {observation.task_id!r}"
                )
            entity_key = (observation.task_id, observation.entity_id)
            if entity_key in seen_entities:
                raise AnalysisGNNClassBalanceError("duplicate task/entity observation")
            seen_entities.add(entity_key)
            state_key = (observation.task_id, record.split)
            if observation.available and not observation.masked:
                if observation.class_value is None:
                    raise AnalysisGNNClassBalanceError(
                        "available observation requires a class value"
                    )
                class_id = self.class_ids[task.task_id].get(observation.class_value)
                if class_id is None:
                    raise AnalysisGNNClassBalanceError(
                        f"value outside frozen vocabulary: {task.task_id}="
                        f"{observation.class_value!r}"
                    )
                if observation.source_row_id is None:
                    raise AnalysisGNNClassBalanceError(
                        "available observation requires canonical source-row identity"
                    )
                self.state_counts[state_key]["available_count"] += 1
                self.available_components[state_key].add(record.component_id)
                stats = self.stats[(task.task_id, record.split, class_id)]
                stats.entity_count += 1
                dialect = _dialect_name(record.dialect)
                stats.dialect_entity_count[dialect] += 1
                stats.records.add(record.record_id)
                stats.components.add(record.component_id)
                stats.dialect_records[dialect].add(record.record_id)
                stats.dialect_components[dialect].add(record.component_id)
                row_key = (task.task_id, class_id, observation.source_row_id)
                local_rows.add(row_key)
            elif observation.masked:
                self.state_counts[state_key]["masked_count"] += 1
            else:
                self.state_counts[state_key]["absent_count"] += 1
        for task_id, class_id, _source_row_id in sorted(local_rows):
            stats = self.stats[(task_id, record.split, class_id)]
            stats.canonical_target_row_count += 1
            stats.record_rows[record.record_id] += 1
            stats.component_rows[record.component_id] += 1
            dialect = _dialect_name(record.dialect)
            stats.dialect_row_count[dialect] += 1

    def class_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for task in self.tasks:
            for split in SPLIT_ORDER:
                for class_id, class_value in enumerate(task.labels):
                    stats = self.stats[(task.task_id, split, class_id)]
                    largest_record, _record_top_five, _record_effective = _shares(
                        stats.record_rows
                    )
                    largest_component, top_five, effective = _shares(
                        stats.component_rows
                    )
                    an_rows = stats.dialect_row_count["an"]
                    dlc_rows = stats.dialect_row_count["dlc"]
                    if an_rows and dlc_rows:
                        dialect_support = "shared_an_dlc"
                    elif an_rows:
                        dialect_support = "an_only"
                    elif dlc_rows:
                        dialect_support = "dlc_only"
                    else:
                        dialect_support = "absent"
                    row_count = stats.canonical_target_row_count
                    entity_count = stats.entity_count
                    rows.append(
                        {
                            "an_component_count": len(stats.dialect_components["an"]),
                            "an_entity_count": stats.dialect_entity_count["an"],
                            "an_record_count": len(stats.dialect_records["an"]),
                            "an_target_row_count": an_rows,
                            "broadcast_factor": (
                                None if row_count == 0 else _round(entity_count / row_count)
                            ),
                            "canonical_target_row_count": row_count,
                            "class_id": class_id,
                            "class_value": class_value,
                            "component_count": len(stats.components),
                            "dialect_support": dialect_support,
                            "dlc_component_count": len(stats.dialect_components["dlc"]),
                            "dlc_entity_count": stats.dialect_entity_count["dlc"],
                            "dlc_record_count": len(stats.dialect_records["dlc"]),
                            "dlc_target_row_count": dlc_rows,
                            "effective_component_count": _round(effective),
                            "entity_count": entity_count,
                            "largest_component_share": _round(largest_component),
                            "largest_record_share": _round(largest_record),
                            "record_count": len(stats.records),
                            "split": split,
                            "support_tier": (
                                train_support_tier(row_count, len(stats.components))
                                if split == "train"
                                else None
                            ),
                            "task_id": task.task_id,
                            "top_5_components_share": _round(top_five),
                            "validation_tier": (
                                validation_support_tier(row_count, len(stats.components))
                                if split == "validation"
                                else None
                            ),
                            "vocabulary_id": task.vocabulary_id,
                        }
                    )
        return rows

    def head_summaries(self, class_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        by_task_split: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for row in class_rows:
            by_task_split[(str(row["task_id"]), str(row["split"]))].append(row)
        summaries: list[dict[str, object]] = []
        for task in self.tasks:
            train = by_task_split[(task.task_id, "train")]
            validation = by_task_split[(task.task_id, "validation")]
            counts = [int(row["canonical_target_row_count"]) for row in train]
            total = sum(counts)
            observed = [(index, count) for index, count in enumerate(counts) if count > 0]
            if observed:
                majority_id, majority_count = max(observed, key=lambda item: (item[1], -item[0]))
                minority_id, minority_count = min(observed, key=lambda item: (item[1], item[0]))
                majority_share = majority_count / total
                ratio = majority_count / minority_count
            else:
                majority_id = minority_id = None
                majority_count = minority_count = 0
                majority_share = 0.0
                ratio = None
            entropy = -sum(
                (count / total) * math.log(count / total)
                for count in counts
                if count > 0 and total > 0
            )
            normalized_entropy = (
                entropy / math.log(task.vocabulary_size)
                if total > 0 and task.vocabulary_size > 1
                else 0.0
            )
            train_tiers = Counter(str(row["support_tier"]) for row in train)
            validation_tiers = Counter(str(row["validation_tier"]) for row in validation)
            train_problem = [
                str(row["class_value"])
                for row in train
                if row["support_tier"] in {"absent", "insufficient"}
            ]
            recommendation, reasons = recommend_head_trainability(
                vocabulary_size=task.vocabulary_size,
                train_tiers=[str(row["support_tier"]) for row in train],
                validation_tiers=[str(row["validation_tier"]) for row in validation],
                available_train_components=len(
                    self.available_components[(task.task_id, "train")]
                ),
                majority_share=majority_share,
                max_to_min_nonzero_ratio=ratio,
                normalized_entropy=normalized_entropy,
            )
            dialect_counter = Counter(str(row["dialect_support"]) for row in train)
            summaries.append(
                {
                    "absent_classes": [
                        str(row["class_value"]) for row in train if row["support_tier"] == "absent"
                    ],
                    "available_train_component_count": len(
                        self.available_components[(task.task_id, "train")]
                    ),
                    "broad_classes": [
                        str(row["class_value"]) for row in train if row["support_tier"] == "broad"
                    ],
                    "dialect_coverage": dict(sorted(dialect_counter.items())),
                    "effective_class_count": _round(math.exp(entropy)) if total else 0.0,
                    "fragile_classes": [
                        str(row["class_value"]) for row in train if row["support_tier"] == "fragile"
                    ],
                    "insufficient_classes": [
                        str(row["class_value"]) for row in train if row["support_tier"] == "insufficient"
                    ],
                    "majority_class": (
                        None
                        if majority_id is None
                        else {
                            "class_id": majority_id,
                            "class_value": task.labels[majority_id],
                            "canonical_target_row_count": majority_count,
                        }
                    ),
                    "majority_share": _round(majority_share),
                    "masked_fraction": _round(self._masked_fraction(task.task_id, "train")),
                    "max_to_min_nonzero_ratio": None if ratio is None else _round(ratio),
                    "minority_nonzero_class": (
                        None
                        if minority_id is None
                        else {
                            "class_id": minority_id,
                            "class_value": task.labels[minority_id],
                            "canonical_target_row_count": minority_count,
                        }
                    ),
                    "normalized_entropy": _round(normalized_entropy),
                    "recommendation": recommendation,
                    "recommendation_reasons": reasons,
                    "state_counts": {
                        split: {
                            key: self.state_counts[(task.task_id, split)][key]
                            for key in ("available_count", "masked_count", "absent_count")
                        }
                        for split in SPLIT_ORDER
                    },
                    "support_tier_counts": {
                        tier: train_tiers[tier] for tier in TRAIN_SUPPORT_ORDER
                    },
                    "task_id": task.task_id,
                    "train_missing_class_count": len(train_problem),
                    "train_observed_class_count": sum(count > 0 for count in counts),
                    "usable_classes": [
                        str(row["class_value"]) for row in train if row["support_tier"] == "usable"
                    ],
                    "validation_fragile_classes": [
                        str(row["class_value"])
                        for row in validation
                        if row["validation_tier"] == "fragile_validation"
                    ],
                    "validation_missing_class_count": validation_tiers["unobservable"],
                    "validation_observed_class_count": sum(
                        int(row["canonical_target_row_count"]) > 0 for row in validation
                    ),
                    "validation_tier_counts": {
                        tier: validation_tiers[tier] for tier in VALIDATION_SUPPORT_ORDER
                    },
                    "validation_unobservable_classes": [
                        str(row["class_value"])
                        for row in validation
                        if row["validation_tier"] == "unobservable"
                    ],
                    "vocabulary_id": task.vocabulary_id,
                    "vocabulary_size": task.vocabulary_size,
                }
            )
        return summaries

    def _masked_fraction(self, task_id: str, split: str) -> float:
        states = self.state_counts[(task_id, split)]
        total = sum(states.values())
        return states["masked_count"] / total if total else 0.0


def recommend_head_trainability(
    *,
    vocabulary_size: int,
    train_tiers: Sequence[str],
    validation_tiers: Sequence[str],
    available_train_components: int,
    majority_share: float,
    max_to_min_nonzero_ratio: float | None,
    normalized_entropy: float,
) -> tuple[str, list[str]]:
    """Apply the B4 decision rules in their frozen priority order."""

    if len(train_tiers) != vocabulary_size or len(validation_tiers) != vocabulary_size:
        raise AnalysisGNNClassBalanceError("tier vectors must cover the vocabulary")
    problem_count = sum(tier in {"absent", "insufficient"} for tier in train_tiers)
    descriptive_reasons: list[str] = []
    if vocabulary_size and problem_count / vocabulary_size > HEAD_THRESHOLDS["descriptive_problem_class_fraction"]:
        descriptive_reasons.append("more_than_25_percent_train_classes_absent_or_insufficient")
    if available_train_components < HEAD_THRESHOLDS["descriptive_min_available_train_components"]:
        descriptive_reasons.append("fewer_than_20_available_train_components")
    if descriptive_reasons:
        return "descriptive_only", descriptive_reasons
    insufficient_reasons: list[str] = []
    if problem_count:
        insufficient_reasons.append("train_class_absent_or_insufficient")
    if any(tier == "unobservable" for tier in validation_tiers):
        insufficient_reasons.append("train_supported_class_unobservable_in_validation")
    if insufficient_reasons:
        return "insufficient_support", insufficient_reasons
    if any(tier != "observable" for tier in validation_tiers):
        return "insufficient_support", ["validation_support_below_observable_threshold"]
    imbalance: list[str] = []
    if majority_share > HEAD_THRESHOLDS["majority_share"]:
        imbalance.append("majority_share_above_0_50")
    if max_to_min_nonzero_ratio is not None and max_to_min_nonzero_ratio > HEAD_THRESHOLDS["max_to_min_nonzero_ratio"]:
        imbalance.append("max_to_min_nonzero_ratio_above_20")
    if normalized_entropy < HEAD_THRESHOLDS["normalized_entropy"]:
        imbalance.append("normalized_entropy_below_0_70")
    if any(tier == "fragile" for tier in train_tiers):
        imbalance.append("train_class_fragile")
    if imbalance:
        return "trainable_with_reweighting", imbalance
    if all(tier in {"usable", "broad"} for tier in train_tiers):
        return "trainable", ["all_train_classes_usable_and_validation_classes_observable"]
    return "trainable_with_reweighting", ["train_support_below_usable"]


def observations_from_sidecar(
    sidecar: Mapping[str, object], *, split: Literal["train", "validation"]
) -> RecordTargetObservations:
    """Project a B3 expanded sidecar into B4 production-head observations."""

    record_id = str(sidecar.get("record_id", ""))
    component_id = str(sidecar.get("source_component_id", ""))
    dialect = sidecar.get("dialect")
    entities = sidecar.get("entities")
    if not record_id or not component_id or dialect not in ("an_joint", "dlc"):
        raise AnalysisGNNClassBalanceError("sidecar record identity is invalid")
    if not isinstance(entities, list):
        raise AnalysisGNNClassBalanceError("expanded sidecar entities are required")
    tasks_by_entity: dict[str, list[AuditTaskSpec]] = defaultdict(list)
    for task in PRODUCTION_AUDIT_TASKS:
        tasks_by_entity[task.entity_type].append(task)
    targets: list[EntityTargetObservation] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise AnalysisGNNClassBalanceError("sidecar entity row is invalid")
        entity_type = str(entity.get("entity_type", ""))
        entity_id = str(entity.get("canonical_entity_id", ""))
        states = entity.get("targets")
        if not isinstance(states, dict):
            raise AnalysisGNNClassBalanceError("sidecar target states are required")
        for task in tasks_by_entity.get(entity_type, ()):
            state = states.get(task.task_id)
            if state is None:
                targets.append(
                    EntityTargetObservation(task.task_id, entity_id, None, None, False, False)
                )
                continue
            if not isinstance(state, dict):
                raise AnalysisGNNClassBalanceError("target state is invalid")
            provenance = state.get("provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            ordinal = provenance.get("source_row_ordinal")
            source_field = provenance.get("source_field", task.task_id)
            source_row_id = (
                None
                if ordinal is None
                else f"{source_field}:row:{ordinal}"
            )
            targets.append(
                EntityTargetObservation(
                    task_id=task.task_id,
                    entity_id=entity_id,
                    source_row_id=source_row_id,
                    class_value=(
                        str(state["canonical_value"])
                        if state.get("canonical_value") is not None
                        else None
                    ),
                    available=state.get("available") is True,
                    masked=state.get("masked") is True,
                )
            )
    return RecordTargetObservations(
        record_id=record_id,
        component_id=component_id,
        dialect=dialect,
        split=split,
        targets=tuple(targets),
    )


def project_quality_record(record: RecordTargetObservations) -> RecordTargetObservations:
    projected: list[EntityTargetObservation] = []
    for observation in record.targets:
        if observation.task_id != "quality":
            continue
        value = (
            project_quality_for_analysisgnn(observation.class_value)
            if observation.class_value is not None
            else None
        )
        projected.append(
            EntityTargetObservation(
                task_id=COMPATIBILITY_QUALITY_TASK.task_id,
                entity_id=observation.entity_id,
                source_row_id=observation.source_row_id,
                class_value=value,
                available=observation.available,
                masked=observation.masked,
            )
        )
    return RecordTargetObservations(
        record.record_id,
        record.component_id,
        record.dialect,
        record.split,
        tuple(projected),
    )


def candidate_class_weights(
    class_rows: Sequence[Mapping[str, object]],
    head_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compute TRAIN-only diagnostic vectors; unsupported classes stay null."""

    train_by_task: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in class_rows:
        if row["split"] == "train":
            train_by_task[str(row["task_id"])].append(row)
    recommendation_by_task = {
        str(row["task_id"]): str(row["recommendation"]) for row in head_summaries
    }
    heads: list[dict[str, object]] = []
    for task_id, rows in train_by_task.items():
        ordered = sorted(rows, key=lambda row: int(row["class_id"]))
        counts = [int(row["canonical_target_row_count"]) for row in ordered]
        raw_vectors = {
            "inverse_frequency": [None if count == 0 else 1.0 / count for count in counts],
            "inverse_sqrt_frequency": [None if count == 0 else 1.0 / math.sqrt(count) for count in counts],
            "effective_number": [
                None
                if count == 0
                else (1.0 - EFFECTIVE_NUMBER_BETA) / (1.0 - EFFECTIVE_NUMBER_BETA**count)
                for count in counts
            ],
        }
        vectors: dict[str, list[dict[str, object]]] = {}
        for method, raw in raw_vectors.items():
            nonzero = [value for value in raw if value is not None]
            mean = sum(nonzero) / len(nonzero) if nonzero else 1.0
            vectors[method] = [
                {
                    "class_id": int(row["class_id"]),
                    "class_value": str(row["class_value"]),
                    "status": "unsupported" if value is None else "candidate",
                    "train_canonical_target_row_count": count,
                    "weight": None if value is None else _round(value / mean),
                }
                for row, count, value in zip(ordered, counts, raw, strict=True)
            ]
        head_status = recommendation_by_task[task_id]
        concentrated = any(
            int(row["canonical_target_row_count"]) > 0
            and (
                float(row["largest_component_share"]) > SAMPLING_DIAGNOSTIC_THRESHOLDS["largest_component_share"]
                or (
                    int(row["component_count"]) > 0
                    and float(row["effective_component_count"]) / int(row["component_count"])
                    < SAMPLING_DIAGNOSTIC_THRESHOLDS["effective_component_fraction"]
                )
                or (
                    row["broadcast_factor"] is not None
                    and float(row["broadcast_factor"]) > SAMPLING_DIAGNOSTIC_THRESHOLDS["broadcast_factor"]
                )
            )
            for row in ordered
        )
        if head_status in {"descriptive_only", "insufficient_support"}:
            policy = "head_not_ready"
        elif concentrated:
            policy = "component_balanced_sampling_candidate"
        elif head_status == "trainable_with_reweighting":
            policy = "class_weighting_candidate"
        else:
            policy = "unweighted_baseline"
        heads.append(
            {
                "diagnostic_policy_recommendation": policy,
                "task_id": task_id,
                "trainability": head_status,
                "vectors": vectors,
            }
        )
    payload: dict[str, object] = {
        "beta": EFFECTIVE_NUMBER_BETA,
        "heads": heads,
        "train_only": True,
        "validation_counts_used": False,
        "weighting_policy_frozen": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def roman_numeral_summary(class_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [row for row in class_rows if row["task_id"] == "roman_numeral"]
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    validation_by_value = {str(row["class_value"]): row for row in validation}
    nonzero = [row for row in train if int(row["canonical_target_row_count"]) > 0]
    descending = sorted(
        nonzero,
        key=lambda row: (-int(row["canonical_target_row_count"]), int(row["class_id"])),
    )
    ascending = sorted(
        nonzero,
        key=lambda row: (int(row["canonical_target_row_count"]), int(row["class_id"])),
    )
    total = sum(int(row["canonical_target_row_count"]) for row in train)
    def compact(row: Mapping[str, object]) -> dict[str, object]:
        return {
            "canonical_target_row_count": int(row["canonical_target_row_count"]),
            "class_id": int(row["class_id"]),
            "class_value": str(row["class_value"]),
            "component_count": int(row["component_count"]),
        }
    return {
        "bottom_20_nonzero": [compact(row) for row in ascending[:20]],
        "class_count": len(train),
        "component_threshold_counts": {
            "lt_3": sum(int(row["component_count"]) < 3 for row in train),
            "lt_10": sum(int(row["component_count"]) < 10 for row in train),
            "lt_20": sum(int(row["component_count"]) < 20 for row in train),
        },
        "dialect_only_classes": [
            str(row["class_value"])
            for row in train
            if row["dialect_support"] in {"an_only", "dlc_only"}
        ],
        "required_vocabulary_evidence": {
            "#VII": any(row["class_value"] == "#VII" for row in train),
            "#VIIbvio7": any(row["class_value"] == "#VIIbvio7" for row in train),
            "bvio7": any(row["class_value"] == "bvio7" for row in train),
            "none": any(str(row["class_value"]).lower() == "none" for row in train),
        },
        "target_row_threshold_counts": {
            "lt_20": sum(int(row["canonical_target_row_count"]) < 20 for row in train),
            "lt_100": sum(int(row["canonical_target_row_count"]) < 100 for row in train),
            "lt_1000": sum(int(row["canonical_target_row_count"]) < 1000 for row in train),
        },
        "top_20": [compact(row) for row in descending[:20]],
        "top_coverage": {
            f"top_{limit}": _round(
                sum(int(row["canonical_target_row_count"]) for row in descending[:limit]) / total
            ) if total else 0.0
            for limit in (10, 20, 50)
        },
        "train_absent_class_count": sum(int(row["canonical_target_row_count"]) == 0 for row in train),
        "validation_absent_class_count": sum(
            int(row["canonical_target_row_count"]) == 0 for row in validation
        ),
        "validation_classes_absent_in_train": [
            value
            for value, row in validation_by_value.items()
            if int(row["canonical_target_row_count"]) > 0
            and next(
                int(train_row["canonical_target_row_count"])
                for train_row in train
                if train_row["class_value"] == value
            ) == 0
        ],
    }


class JointTupleAccumulator:
    def __init__(self) -> None:
        self.row_counts: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
        self.canonical_counts: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
        self.records: dict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)
        self.components: dict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)
        self.component_rows: dict[tuple[str, str, tuple[str, ...]], Counter[str]] = defaultdict(Counter)
        self.total_rows: Counter[tuple[str, str]] = Counter()
        self.total_canonical_rows: Counter[tuple[str, str]] = Counter()

    def add_record(self, observations: Iterable[JointTupleObservation]) -> None:
        identity: dict[tuple[str, str, tuple[str, ...], str], JointTupleObservation] = {}
        for observation in observations:
            if observation.split not in SPLIT_ORDER:
                raise AnalysisGNNClassBalanceError("joint targets accept only TRAIN/VALIDATION")
            values = tuple(observation.values)
            key = (observation.mode, observation.split, values)
            self.row_counts[key] += 1
            self.total_rows[(observation.mode, observation.split)] += 1
            self.records[key].add(observation.record_id)
            self.components[key].add(observation.component_id)
            canonical_key = (*key, observation.canonical_harmonic_row_id)
            identity[canonical_key] = observation
        for canonical_key, observation in sorted(identity.items()):
            mode, split, values, _canonical_id = canonical_key
            key = (mode, split, values)
            self.canonical_counts[key] += 1
            self.total_canonical_rows[(mode, split)] += 1
            self.component_rows[key][observation.component_id] += 1

    def rows(self) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for mode in ("corrected_harmonic_event", "compatibility_note"):
            for split in SPLIT_ORDER:
                keys = sorted(
                    key for key in self.row_counts if key[0] == mode and key[1] == split
                )
                for _mode, _split, values in keys:
                    key = (mode, split, values)
                    largest, top_five, effective = _shares(self.component_rows[key])
                    output.append(
                        {
                            "canonical_harmonic_target_row_count": self.canonical_counts[key],
                            "component_count": len(self.components[key]),
                            "effective_component_count": _round(effective),
                            "largest_component_share": _round(largest),
                            "mode": mode,
                            "record_count": len(self.records[key]),
                            "row_count": self.row_counts[key],
                            "split": split,
                            "top_5_components_share": _round(top_five),
                            "tuple": {
                                name: value for name, value in zip(PAPER_DEFINED_JOINT_COMPONENTS, values, strict=True)
                            },
                        }
                    )
        return output

    def summary(self, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for mode in ("corrected_harmonic_event", "compatibility_note"):
            mode_rows = [row for row in rows if row["mode"] == mode]
            train = [row for row in mode_rows if row["split"] == "train"]
            validation = [row for row in mode_rows if row["split"] == "validation"]
            train_tuples = {_tuple_key(row["tuple"]) for row in train}
            validation_tuples = {_tuple_key(row["tuple"]) for row in validation}
            result[mode] = {
                "train": self._split_summary(mode, "train", train),
                "validation": self._split_summary(mode, "validation", validation),
                "validation_tuples_unseen_in_train": [
                    _tuple_payload(values) for values in sorted(validation_tuples - train_tuples)
                ],
            }
        return result

    def _split_summary(
        self, mode: str, split: str, rows: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        descending = sorted(
            rows,
            key=lambda row: (-int(row["canonical_harmonic_target_row_count"]), _tuple_key(row["tuple"])),
        )
        canonical_total = self.total_canonical_rows[(mode, split)]
        frequencies = Counter(int(row["canonical_harmonic_target_row_count"]) for row in rows)
        records = set()
        components = set()
        for key, values in self.records.items():
            if key[0] == mode and key[1] == split:
                records.update(values)
                components.update(self.components[key])
        return {
            "canonical_harmonic_target_rows": canonical_total,
            "component_count": len(components),
            "frequency_distribution": {
                str(frequency): count for frequency, count in sorted(frequencies.items())
            },
            "record_count": len(records),
            "row_count": self.total_rows[(mode, split)],
            "singleton_tuple_count": frequencies[1],
            "top_20": [
                {
                    "canonical_harmonic_target_row_count": int(row["canonical_harmonic_target_row_count"]),
                    "row_count": int(row["row_count"]),
                    "tuple": row["tuple"],
                }
                for row in descending[:20]
            ],
            "top_coverage": {
                f"top_{limit}": _round(
                    sum(int(row["canonical_harmonic_target_row_count"]) for row in descending[:limit])
                    / canonical_total
                ) if canonical_total else 0.0
                for limit in (10, 20, 50)
            },
            "unique_tuple_count": len(rows),
        }


def _tuple_key(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise AnalysisGNNClassBalanceError("joint tuple payload is invalid")
    return tuple(str(value[name]) for name in PAPER_DEFINED_JOINT_COMPONENTS)


def _tuple_payload(values: Sequence[str]) -> dict[str, str]:
    return dict(zip(PAPER_DEFINED_JOINT_COMPONENTS, values, strict=True))


def joint_observations_from_sidecar(
    sidecar: Mapping[str, object], *, split: Literal["train", "validation"]
) -> tuple[JointTupleObservation, ...]:
    entities = sidecar.get("entities")
    relations = sidecar.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise AnalysisGNNClassBalanceError("expanded sidecar relations are required")
    record_id = str(sidecar["record_id"])
    component_id = str(sidecar["source_component_id"])
    dialect = sidecar["dialect"]
    if dialect not in ("an_joint", "dlc"):
        raise AnalysisGNNClassBalanceError("unknown sidecar dialect")
    harmonic: dict[str, tuple[str, str, str, str, str]] = {}
    note_ids: set[str] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("canonical_entity_id", ""))
        if entity.get("entity_type") == "note":
            note_ids.add(entity_id)
        if entity.get("entity_type") != "harmonic_event":
            continue
        targets = entity.get("targets")
        if not isinstance(targets, dict):
            continue
        states = [targets.get(name) for name in PAPER_DEFINED_JOINT_COMPONENTS]
        if not all(
            isinstance(state, dict)
            and state.get("available") is True
            and state.get("masked") is False
            and state.get("canonical_value") is not None
            for state in states
        ):
            continue
        values = tuple(str(state["canonical_value"]) for state in states if isinstance(state, dict))
        harmonic[entity_id] = values  # type: ignore[assignment]
    output: list[JointTupleObservation] = []
    for harmonic_id, values in sorted(harmonic.items()):
        output.append(
            JointTupleObservation(
                "corrected_harmonic_event",
                split,
                record_id,
                component_id,
                dialect,
                harmonic_id,
                harmonic_id,
                values,
            )
        )
    for relation in relations:
        if not isinstance(relation, dict) or relation.get("relation") != "note_to_harmonic_event":
            continue
        note_id = str(relation.get("source_entity_id", ""))
        harmonic_id = str(relation.get("target_entity_id", ""))
        if note_id not in note_ids or harmonic_id not in harmonic:
            continue
        values = list(harmonic[harmonic_id])
        values[3] = project_quality_for_analysisgnn(values[3])
        output.append(
            JointTupleObservation(
                "compatibility_note",
                split,
                record_id,
                component_id,
                dialect,
                note_id,
                harmonic_id,
                tuple(values),  # type: ignore[arg-type]
            )
        )
    return tuple(output)


T = TypeVar("T")


def load_train_validation_only(
    assignments: Sequence[Mapping[str, object]],
    loader: Callable[[str, Literal["train", "validation"]], T],
) -> tuple[list[T], dict[str, object]]:
    """Filter frozen assignments before any target-bearing loader invocation."""

    loaded: list[T] = []
    counts = Counter(str(row.get("split")) for row in assignments)
    for row in assignments:
        split = row.get("split")
        if split == "test":
            continue
        if split not in SPLIT_ORDER:
            raise AnalysisGNNClassBalanceError("unknown frozen split")
        loaded.append(loader(str(row["record_id"]), split))
    evidence = {
        "test_assignments_seen": counts["test"] > 0,
        "test_assignment_record_count": counts["test"],
        "test_target_records_opened": 0,
        "test_target_rows_loaded": 0,
        "test_targets_counted": False,
        "test_targets_used_for_decisions": False,
        "test_evaluated": False,
    }
    return loaded, evidence


def semantic_fingerprint(
    *,
    class_rows: Sequence[Mapping[str, object]],
    head_summaries: Sequence[Mapping[str, object]],
    joint_rows: Sequence[Mapping[str, object]],
    recommendations: Mapping[str, object],
    weights: Mapping[str, object],
) -> str:
    return fingerprint(
        {
            "class_balance_contract": class_balance_contract()["fingerprint"],
            "class_rows": class_rows,
            "head_summaries": head_summaries,
            "joint_rows": joint_rows,
            "recommendations": recommendations,
            "weights": weights,
        }
    )


def compact_problem_classes(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        key: list(summary[key])
        for key in (
            "absent_classes",
            "insufficient_classes",
            "fragile_classes",
            "validation_unobservable_classes",
            "validation_fragile_classes",
        )
    }


def recommendation_payload(head_summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    heads = [
        {
            "future_metrics": list(FUTURE_MULTICLASS_METRICS),
            "recommendation": row["recommendation"],
            "reasons": row["recommendation_reasons"],
            "task_id": row["task_id"],
        }
        for row in head_summaries
    ]
    payload: dict[str, object] = {
        "baseline_sufficiency_not_model_quality_guarantee": True,
        "heads": heads,
        "joint_future_metrics": list(FUTURE_JOINT_METRICS),
        "model_implemented": False,
        "training_run": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def task_row_lookup(
    class_rows: Sequence[Mapping[str, object]], task_id: str, class_value: str
) -> dict[str, Mapping[str, object]]:
    return {
        str(row["split"]): row
        for row in class_rows
        if row["task_id"] == task_id and row["class_value"] == class_value
    }


def quality_focus_summary(
    corrected_rows: Sequence[Mapping[str, object]],
    compatibility_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    focus = (
        "augmented seventh chord",
        "augmented major tetrachord",
        "augmented triad",
    )
    return {
        "compatibility_quality_15": {
            "vocabulary_id": COMPATIBILITY_QUALITY_VOCABULARY_ID,
            "augmented_triad": task_row_lookup(
                compatibility_rows, "quality_compatibility", "augmented triad"
            ),
        },
        "corrected_quality_17": {
            "vocabulary_id": CORRECTED_QUALITY_VOCABULARY_ID,
            "focus_classes": {
                value: task_row_lookup(corrected_rows, "quality", value) for value in focus
            },
        },
        "projection": {
            "augmented seventh chord": "augmented triad",
            "augmented major tetrachord": "augmented triad",
            "other_classes": "identity",
            "missing": "mask",
        },
    }


def dataclass_payload(value: object) -> dict[str, object]:
    """Small public helper used by fixture-oriented tests."""

    if not hasattr(value, "__dataclass_fields__"):
        raise AnalysisGNNClassBalanceError("value is not a dataclass")
    return asdict(value)  # type: ignore[arg-type]
