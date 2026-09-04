"""Phase 9E-B5H full eligible-orbit transposition expansion contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedValidationAccumulator,
    CorrectedTrainingError,
    ProductionArtifactPaths,
    align_target_sidecars_after_prediction,
    load_frozen_class_weights,
    load_production_record,
    move_raw_graph_batch,
    production_component_records,
    production_valid_shifts,
    train_seen_joint_tuples,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    CORRECTED_V2_METRIC_ID,
)
from music_critic.experiments.analysisgnn.training_policy import (
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
    build_training_profiles,
    corrected_optimizer_envelope,
)
from music_critic.experiments.analysisgnn.transposition import (
    ABSOLUTE_TASKS,
    BOOLEAN_TASKS,
    PITCH_CLASS_SET_TASKS,
    RELATIVE_TASKS,
    SHIFT_PCS,
    STRUCTURAL_TASKS,
    canonical_directed_transposition,
    directed_transposition_contract,
)


FULL_ORBIT_PROFILE_ID = "music-critic-v2-corrected-full-orbit-transposition-v1"
FULL_ORBIT_SCHEMA = "CorrectedAnalysisGNNFullOrbitProfile@1.0.0"
FULL_ORBIT_TABLE_SCHEMA = "CorrectedAnalysisGNNFullOrbitTable@1.0.0"
FULL_ORBIT_RUNTIME_SCHEMA = "CorrectedAnalysisGNNFullOrbitRuntime@1.0.0"
FULL_ORBIT_PREFLIGHT_SCHEMA = "Phase9EB5HFullOrbitPreflight@1.0.0"
FULL_ORBIT_FIXTURE_SCHEMA = "Phase9EB5HFullOrbitFixture@1.0.0"
FULL_ORBIT_RNG_DOMAIN = "sha256_B5H_full_orbit_epoch_permutation_v1"
B5F_SOURCE_HEAD = "e9de6ba5e63a9c0443bb78dce975956ae997640b"
FULL_ORBIT_SEED = 17
FULL_ORBIT_BATCH_SIZE = 2
FULL_ORBIT_UPDATE_BUDGET = 120_000
FULL_ORBIT_DRAW_BUDGET = 240_000
FULL_ORBIT_WARMUP_UPDATES = 6_000
FULL_ORBIT_VALIDATION_INTERVAL = 5_000
FULL_ORBIT_CHECKPOINT_INTERVAL = 500
FULL_ORBIT_PROGRESS_INTERVAL = 25
FULL_ORBIT_PEAK_LEARNING_RATE = 0.005
EXPECTED_BASE_RECORDS = 1_295
EXPECTED_NOMINAL_PAIRS = 15_540
EXPECTED_ELIGIBLE_PAIRS = 15_389
EXPECTED_EXCLUDED_PAIRS = 151
EXPECTED_VALID_SHIFT_DISTRIBUTION = {
    12: 1231,
    11: 31,
    10: 8,
    9: 12,
    8: 3,
    7: 8,
    6: 1,
    2: 1,
}


@dataclass(frozen=True, slots=True, order=True)
class FullOrbitPair:
    record_id: str
    shift_pc: int
    component_id: str

    def __post_init__(self) -> None:
        if not self.record_id or not self.component_id or self.shift_pc not in SHIFT_PCS:
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.pair_invalid", repr(asdict(self))
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FullOrbitDraw:
    orbit_epoch: int
    orbit_index: int
    component_id: str
    record_id: str
    shift_pc: int


@dataclass(frozen=True, slots=True)
class FullOrbitRuntimeConfig:
    schema_version: str = FULL_ORBIT_RUNTIME_SCHEMA
    profile_id: str = FULL_ORBIT_PROFILE_ID
    seed: int = FULL_ORBIT_SEED
    device: str = "cuda"
    batch_size: int = FULL_ORBIT_BATCH_SIZE
    applied_update_budget: int = FULL_ORBIT_UPDATE_BUDGET
    train_draw_budget: int = FULL_ORBIT_DRAW_BUDGET
    warmup_applied_updates: int = FULL_ORBIT_WARMUP_UPDATES
    validation_interval: int = FULL_ORBIT_VALIDATION_INTERVAL
    mixed_precision: str = "fp32_baseline"
    early_stopping: bool = False
    test_enabled: bool = False

    def __post_init__(self) -> None:
        if (
            self.profile_id != FULL_ORBIT_PROFILE_ID
            or self.seed != FULL_ORBIT_SEED
            or self.device != "cuda"
            or self.batch_size != FULL_ORBIT_BATCH_SIZE
            or self.applied_update_budget != FULL_ORBIT_UPDATE_BUDGET
            or self.train_draw_budget != FULL_ORBIT_DRAW_BUDGET
            or self.train_draw_budget != self.applied_update_budget * self.batch_size
            or self.warmup_applied_updates != FULL_ORBIT_WARMUP_UPDATES
            or self.mixed_precision != "fp32_baseline"
            or self.early_stopping
            or self.test_enabled
        ):
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.runtime_contract_changed", repr(asdict(self))
            )

    def to_dict(self) -> dict[str, object]:
        body = asdict(self)
        body.update(
            {
                "scheduler": "linear_warmup_then_cosine_decay",
                "peak_learning_rate": FULL_ORBIT_PEAK_LEARNING_RATE,
                "checkpoint_selection": "identity_validation_corrected_primary_macro_score",
                "primary_validation_view": "identity_only",
                "diagnostic_validation_view": "all_eligible_shifts_per_shift_and_macro",
                "from_scratch": True,
                "loader_workers": 0,
                "rng_domains": {
                    "model_initialization_torch_seed": self.seed,
                    "dropout_torch_seed": self.seed * 1000 + 1,
                    "orbit_permutation": FULL_ORBIT_RNG_DOMAIN,
                },
                "test_loader_created": False,
                "test_targets_read": False,
                "test_metrics_computed": False,
            }
        )
        body["fingerprint"] = fingerprint(body)
        return body


def _record_components(
    component_records: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for component_id, records in sorted(component_records.items()):
        for record_id in records:
            if record_id in result:
                raise CorrectedTrainingError(
                    "analysisgnn.full_orbit.record_component_ambiguous", record_id
                )
            result[str(record_id)] = str(component_id)
    return result


def build_full_orbit_table(
    component_records: Mapping[str, Sequence[str]],
    valid_shifts_by_record: Mapping[str, Sequence[int]],
) -> tuple[FullOrbitPair, ...]:
    """Build the stable TRAIN-only table sorted by ``(record_id, shift_pc)``."""

    components = _record_components(component_records)
    if set(components) != set(valid_shifts_by_record):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.record_inventory_mismatch",
            f"components={len(components)} eligibility={len(valid_shifts_by_record)}",
        )
    rows: list[FullOrbitPair] = []
    for record_id in sorted(valid_shifts_by_record):
        shifts = tuple(sorted(set(int(value) for value in valid_shifts_by_record[record_id])))
        if not shifts or 0 not in shifts or any(value not in SHIFT_PCS for value in shifts):
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.shift_set_invalid", record_id
            )
        rows.extend(FullOrbitPair(record_id, shift, components[record_id]) for shift in shifts)
    result = tuple(rows)
    if len(result) != len(set(result)):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.duplicate_pair", str(len(result))
        )
    return result


def full_orbit_table_contract(table: Sequence[FullOrbitPair]) -> dict[str, object]:
    records = Counter(row.record_id for row in table)
    shift_distribution = Counter(records.values())
    identities = sum(row.shift_pc == 0 for row in table)
    body: dict[str, object] = {
        "schema": FULL_ORBIT_TABLE_SCHEMA,
        "canonical_sort": ["record_id", "shift_pc"],
        "base_train_records": len(records),
        "nominal_record_shift_pairs": len(records) * len(SHIFT_PCS),
        "eligible_train_pairs": len(table),
        "excluded_train_pairs": len(records) * len(SHIFT_PCS) - len(table),
        "valid_shift_count_distribution": {
            str(key): value for key, value in sorted(shift_distribution.items(), reverse=True)
        },
        "identity_pairs": identities,
        "identity_fraction": identities / len(table),
        "table_fingerprint": fingerprint([row.to_dict() for row in table]),
        "variants_are_independent_musical_works": False,
        "raw_graphs_materialized": False,
        "source_split_changed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _pair_permutation_key(pair: FullOrbitPair, *, seed: int, epoch: int) -> str:
    return fingerprint(
        {
            "domain": FULL_ORBIT_RNG_DOMAIN,
            "epoch": epoch,
            "profile_id": FULL_ORBIT_PROFILE_ID,
            "record_id": pair.record_id,
            "seed": seed,
            "shift_pc": pair.shift_pc,
        }
    )


def full_orbit_epoch_permutation(
    table: Sequence[FullOrbitPair], *, seed: int, epoch: int
) -> tuple[FullOrbitPair, ...]:
    if seed != FULL_ORBIT_SEED or epoch < 0:
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.permutation_input_invalid", f"seed={seed} epoch={epoch}"
        )
    return tuple(
        sorted(table, key=lambda row: (_pair_permutation_key(row, seed=seed, epoch=epoch), row))
    )


class FullOrbitSampler:
    """Stateful no-replacement sampler over deterministic orbit epochs."""

    def __init__(
        self,
        table: Sequence[FullOrbitPair],
        *,
        seed: int = FULL_ORBIT_SEED,
        position: int = 0,
    ) -> None:
        if not table or len(table) != len(set(table)) or position < 0:
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.sampler_input_invalid", f"rows={len(table)} position={position}"
            )
        self.table = tuple(table)
        self.seed = seed
        self.position = position
        self._epoch_cache: dict[int, tuple[FullOrbitPair, ...]] = {}

    @property
    def draws_per_epoch(self) -> int:
        return len(self.table)

    def _epoch(self, epoch: int) -> tuple[FullOrbitPair, ...]:
        if epoch not in self._epoch_cache:
            self._epoch_cache = {
                epoch: full_orbit_epoch_permutation(self.table, seed=self.seed, epoch=epoch)
            }
        return self._epoch_cache[epoch]

    def peek(self, offset: int = 0) -> FullOrbitDraw:
        if offset < 0:
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.sampler_offset_invalid", str(offset)
            )
        epoch, index = divmod(self.position + offset, self.draws_per_epoch)
        row = self._epoch(epoch)[index]
        return FullOrbitDraw(epoch, index, row.component_id, row.record_id, row.shift_pc)

    def advance_after_applied_update(self, draws: int = FULL_ORBIT_BATCH_SIZE) -> None:
        if draws <= 0:
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.sampler_advance_invalid", str(draws)
            )
        self.position += draws

    def state_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "position": self.position,
            "seed": self.seed,
            "profile_id": FULL_ORBIT_PROFILE_ID,
            "table_fingerprint": fingerprint([row.to_dict() for row in self.table]),
            "rng_domain": FULL_ORBIT_RNG_DOMAIN,
        }
        body["fingerprint"] = fingerprint(body)
        return body

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = self.state_dict()
        for key in ("seed", "profile_id", "table_fingerprint", "rng_domain"):
            if state.get(key) != expected[key]:
                raise CorrectedTrainingError(
                    "analysisgnn.full_orbit.sampler_resume_mismatch", key
                )
        self.position = int(state["position"])


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_pair_evidence(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> tuple[dict[str, object], ...]:
    """Return all TRAIN exclusions with source and eligibility seals."""

    train = set(production_valid_shifts(paths))
    source_sha = _sha256_file(paths.b5a_shift_eligibility)
    rows: list[dict[str, object]] = []
    with paths.b5a_shift_eligibility.open("r", encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            record_id = str(source["record_id"])
            if record_id not in train or source.get("corrected_valid") is True:
                continue
            body: dict[str, object] = {
                "record_id": record_id,
                "shift_pc": int(source["shift_pc"]),
                "structured_reasons": list(source.get("corrected_invalid_reasons", ())),
                "source_identity_fingerprint": fingerprint(
                    {
                        "record_id": record_id,
                        "source_component_id": source["source_component_id"],
                        "dialect": source["dialect"],
                    }
                ),
                "eligibility_source_sha256": source_sha,
                "eligibility_fingerprint": fingerprint(source),
            }
            rows.append(body)
    return tuple(sorted(rows, key=lambda row: (str(row["record_id"]), int(row["shift_pc"]))))


def full_orbit_profile_contract() -> dict[str, object]:
    config = FullOrbitRuntimeConfig()
    body: dict[str, object] = {
        "schema": FULL_ORBIT_SCHEMA,
        "phase": "9E-B5H",
        "profile": "C2",
        "profile_id": FULL_ORBIT_PROFILE_ID,
        "scope": "TRAIN_only_after_piece_disjoint_split",
        "dataset_semantics": "all_eligible_record_shift_pairs_once_per_orbit_epoch",
        "pair_order": "stable_record_id_shift_pc_then_epoch_permutation",
        "rng_domain": FULL_ORBIT_RNG_DOMAIN,
        "identity_weighting": "exactly_once_per_record_no_extra_weight",
        "canonical_raw_transform": "directed_forward_signed_semitones",
        "semantic_target_transform": "shift_pc",
        "runtime": config.to_dict(),
        "orbit_epochs_at_budget": FULL_ORBIT_DRAW_BUDGET / EXPECTED_ELIGIBLE_PAIRS,
        "complete_orbit_epochs": FULL_ORBIT_DRAW_BUDGET // EXPECTED_ELIGIBLE_PAIRS,
        "partial_final_orbit_draws": FULL_ORBIT_DRAW_BUDGET % EXPECTED_ELIGIBLE_PAIRS,
        "primary_validation": "identity_only",
        "diagnostic_validation": {
            "view": "all_eligible_validation_shifts",
            "per_shift": ["score", "loss", "corrected_joint"],
            "aggregates": [
                "macro_over_shifts",
                "worst_shift_score",
                "identity_score",
                "worst_shift_minus_identity",
            ],
            "groups": ["absolute", "pitch_class_set", "relative_invariant", "structural", "boolean"],
            "replaces_primary_metric": False,
        },
        "from_scratch": True,
        "resume_from_c1_checkpoint": False,
        "directed_contract_fingerprint": directed_transposition_contract()["fingerprint"],
        "model_architecture_changed": False,
        "split_changed": False,
        "raw_cache_changed": False,
        "test_enabled": False,
        "full_orbit_training_run": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def build_full_orbit_optimizer_scheduler(
    model: CorrectedAnalysisGNNModel,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    envelope = corrected_optimizer_envelope()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FULL_ORBIT_PEAK_LEARNING_RATE,
        weight_decay=float(envelope["weight_decay"]),
    )

    def multiplier(step: int) -> float:
        if step < FULL_ORBIT_WARMUP_UPDATES:
            return float(step + 1) / FULL_ORBIT_WARMUP_UPDATES
        progress = (step - FULL_ORBIT_WARMUP_UPDATES) / max(
            1, FULL_ORBIT_UPDATE_BUDGET - FULL_ORBIT_WARMUP_UPDATES
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _validation_valid_shifts(
    paths: ProductionArtifactPaths,
) -> dict[str, tuple[int, ...]]:
    assignments_path = paths.b3_root / "split_assignments.jsonl"
    with assignments_path.open("r", encoding="utf-8") as handle:
        validation = {
            str(row["record_id"])
            for line in handle
            if line.strip()
            for row in (json.loads(line),)
            if row["split"] == "validation"
        }
    grouped: dict[str, list[int]] = {record_id: [] for record_id in validation}
    with paths.b5a_shift_eligibility.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in validation and row.get("corrected_valid") is True:
                grouped[record_id].append(int(row["shift_pc"]))
    result = {key: tuple(sorted(value)) for key, value in sorted(grouped.items())}
    if set(result) != validation or any(0 not in shifts for shifts in result.values()):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.validation_eligibility_incomplete", str(len(result))
        )
    return result


def _metric_group(
    metrics: Mapping[str, object], tasks: Sequence[str]
) -> dict[str, float | None]:
    heads = metrics["per_head"]
    assert isinstance(heads, Mapping)
    selected = [heads[task] for task in tasks if task in heads]
    scores = [
        float(row["macro_f1_observed_validation_classes"])
        for row in selected
        if isinstance(row, Mapping)
        and row.get("macro_f1_observed_validation_classes") is not None
    ]
    losses = [
        float(row["masked_cross_entropy"])
        for row in selected
        if isinstance(row, Mapping) and row.get("masked_cross_entropy") is not None
    ]
    return {
        "macro_f1": None if not scores else sum(scores) / len(scores),
        "masked_cross_entropy": None if not losses else sum(losses) / len(losses),
    }


def run_full_orbit_diagnostic_validation(
    model: CorrectedAnalysisGNNModel,
    *,
    device: str,
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, object]:
    """Evaluate VALIDATION per eligible shift; never replaces identity selection."""

    eligibility = _validation_valid_shifts(paths)
    class_weights = load_frozen_class_weights()
    seen = train_seen_joint_tuples(paths)
    accumulators = {
        shift: CorrectedValidationAccumulator(class_weights, train_seen_tuples=seen)
        for shift in SHIFT_PCS
    }
    record_counts = Counter()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for record_id, shifts in eligibility.items():
                batch, sidecar = load_production_record(
                    record_id, split="validation", paths=paths
                )
                for shift in shifts:
                    raw = move_raw_graph_batch(
                        transpose_raw_graph_batch(batch.raw_graph_batch, (shift,)),
                        device,
                    )
                    output = model(raw)
                    alignment = align_target_sidecars_after_prediction(
                        output, raw, (sidecar,), shifts=(shift,)
                    )
                    accumulators[shift].update(output, alignment, sidecars=(sidecar,))
                    record_counts[shift] += 1
    finally:
        model.train(was_training)
    per_shift: dict[str, object] = {}
    group_tasks = {
        "absolute": ABSOLUTE_TASKS,
        "pitch_class_set": PITCH_CLASS_SET_TASKS,
        "relative_invariant": RELATIVE_TASKS,
        "structural": STRUCTURAL_TASKS,
        "boolean": BOOLEAN_TASKS,
    }
    scores: dict[int, float] = {}
    for shift, accumulator in accumulators.items():
        metrics = accumulator.finalize()
        score = metrics["corrected_primary_macro_score"]
        if score is not None:
            scores[shift] = float(score)
        head_rows = metrics["per_head"]
        assert isinstance(head_rows, Mapping)
        head_losses = [
            float(row["masked_cross_entropy"])
            for row in head_rows.values()
            if isinstance(row, Mapping) and row.get("masked_cross_entropy") is not None
        ]
        per_shift[str(shift)] = {
            "record_count": record_counts[shift],
            "corrected_primary_macro_score": score,
            "masked_cross_entropy_macro": (
                None if not head_losses else sum(head_losses) / len(head_losses)
            ),
            "corrected_joint_accuracy": metrics.get(
                CORRECTED_V2_METRIC_ID
            ),
            "groups": {
                name: _metric_group(metrics, tasks) for name, tasks in group_tasks.items()
            },
            "per_head": metrics["per_head"],
        }
    worst_shift = min(scores, key=scores.get) if scores else None
    identity_score = scores.get(0)
    worst_score = None if worst_shift is None else scores[worst_shift]
    body: dict[str, object] = {
        "schema": "Phase9EB5HAllShiftValidation@1.0.0",
        "per_shift": per_shift,
        "macro_over_shifts": None if not scores else sum(scores.values()) / len(scores),
        "worst_shift": worst_shift,
        "worst_shift_score": worst_score,
        "identity_score": identity_score,
        "worst_shift_minus_identity": (
            None if worst_score is None or identity_score is None else worst_score - identity_score
        ),
        "primary_checkpoint_selection_replaced": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def full_orbit_preflight(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, object]:
    components = production_component_records(paths)
    valid_shifts = production_valid_shifts(paths)
    table = build_full_orbit_table(components, valid_shifts)
    table_contract = full_orbit_table_contract(table)
    distribution = Counter(len(values) for values in valid_shifts.values())
    exclusions = excluded_pair_evidence(paths)
    if (
        len(valid_shifts) != EXPECTED_BASE_RECORDS
        or len(table) != EXPECTED_ELIGIBLE_PAIRS
        or len(exclusions) != EXPECTED_EXCLUDED_PAIRS
        or dict(distribution) != EXPECTED_VALID_SHIFT_DISTRIBUTION
    ):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.pinned_counts_mismatch",
            f"records={len(valid_shifts)} pairs={len(table)} excluded={len(exclusions)} distribution={dict(distribution)}",
        )
    sampler = FullOrbitSampler(table)
    first_epoch = tuple(sampler.peek(offset) for offset in range(len(table)))
    partial_start = (FULL_ORBIT_DRAW_BUDGET // len(table)) * len(table)
    partial = tuple(
        sampler.peek(partial_start + offset)
        for offset in range(FULL_ORBIT_DRAW_BUDGET % len(table))
    )
    torch.manual_seed(FULL_ORBIT_SEED)
    model = CorrectedAnalysisGNNModel()
    repo_root = Path(__file__).resolve().parents[4]
    b5b_fixture = json.loads(
        (repo_root / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json").read_text(
            encoding="utf-8"
        )
    )
    training_profiles = build_training_profiles(b5b_fixture["class_weight_payload"])
    body: dict[str, object] = {
        "schema": FULL_ORBIT_PREFLIGHT_SCHEMA,
        "phase": "9E-B5H",
        "source_head": B5F_SOURCE_HEAD,
        "valid": True,
        "profile": full_orbit_profile_contract(),
        "orbit_table": table_contract,
        "excluded_pair_count": len(exclusions),
        "excluded_pairs_fingerprint": fingerprint(exclusions),
        "first_epoch_draw_count": len(first_epoch),
        "first_epoch_unique_pair_count": len({(row.record_id, row.shift_pc) for row in first_epoch}),
        "first_epoch_permutation_fingerprint": fingerprint([asdict(row) for row in first_epoch]),
        "partial_final_epoch_draw_count": len(partial),
        "partial_final_epoch_fingerprint": fingerprint([asdict(row) for row in partial]),
        "initial_model_state_fingerprint": model_state_fingerprint(model),
        "model_contract_fingerprint": corrected_model_contract(model)["fingerprint"],
        "c0_profile_id": CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
        "c1_profile_id": CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
        "c0_c1_profile_fingerprints": {
            row.profile_id: row.semantic_fingerprint for row in training_profiles.values() if row.profile_id in {
                CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
                CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
            }
        },
        "inverse_contract_valid": True,
        "full_orbit_profile_valid": True,
        "ready_for_full_orbit_training": True,
        "full_orbit_training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def compact_full_orbit_fixture(preflight: Mapping[str, object]) -> dict[str, object]:
    """Seal compact B5H evidence without embedding the 15,389-row table."""

    orbit = preflight.get("orbit_table")
    if not isinstance(orbit, Mapping):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.preflight_orbit_missing", repr(orbit)
        )
    body: dict[str, object] = {
        "schema": FULL_ORBIT_FIXTURE_SCHEMA,
        "phase": "9E-B5H",
        "source_head": B5F_SOURCE_HEAD,
        "profile": full_orbit_profile_contract(),
        "orbit_table": dict(orbit),
        "excluded_pair_count": preflight["excluded_pair_count"],
        "excluded_pairs_fingerprint": preflight["excluded_pairs_fingerprint"],
        "first_epoch_draw_count": preflight["first_epoch_draw_count"],
        "first_epoch_unique_pair_count": preflight["first_epoch_unique_pair_count"],
        "first_epoch_permutation_fingerprint": preflight[
            "first_epoch_permutation_fingerprint"
        ],
        "partial_final_epoch_draw_count": preflight[
            "partial_final_epoch_draw_count"
        ],
        "partial_final_epoch_fingerprint": preflight[
            "partial_final_epoch_fingerprint"
        ],
        "initial_model_state_fingerprint": preflight[
            "initial_model_state_fingerprint"
        ],
        "model_contract_fingerprint": preflight["model_contract_fingerprint"],
        "c0_profile_id": preflight["c0_profile_id"],
        "c1_profile_id": preflight["c1_profile_id"],
        "c0_c1_profile_fingerprints": preflight[
            "c0_c1_profile_fingerprints"
        ],
        "inverse_contract_valid": True,
        "full_orbit_profile_valid": True,
        "ready_for_full_orbit_training": True,
        "full_orbit_training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["evidence_fingerprint"] = fingerprint(body)
    body["fixture_fingerprint"] = fingerprint(body)
    return body


def check_full_orbit_fixture(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.pop("fixture_fingerprint", None)
    if observed != fingerprint(payload):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.fixture_fingerprint_mismatch", str(path)
        )
    payload["fixture_fingerprint"] = observed
    required = {
        "schema": FULL_ORBIT_FIXTURE_SCHEMA,
        "inverse_contract_valid": True,
        "full_orbit_profile_valid": True,
        "ready_for_full_orbit_training": True,
        "full_orbit_training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
        "first_epoch_draw_count": EXPECTED_ELIGIBLE_PAIRS,
        "first_epoch_unique_pair_count": EXPECTED_ELIGIBLE_PAIRS,
        "partial_final_epoch_draw_count": FULL_ORBIT_DRAW_BUDGET
        % EXPECTED_ELIGIBLE_PAIRS,
        "excluded_pair_count": EXPECTED_EXCLUDED_PAIRS,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.fixture_status_invalid", repr(required)
        )
    if payload.get("profile") != full_orbit_profile_contract():
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.profile_contract_changed", str(path)
        )
    orbit = payload.get("orbit_table")
    if not isinstance(orbit, Mapping) or not (
        orbit.get("base_train_records") == EXPECTED_BASE_RECORDS
        and orbit.get("nominal_record_shift_pairs") == EXPECTED_NOMINAL_PAIRS
        and orbit.get("eligible_train_pairs") == EXPECTED_ELIGIBLE_PAIRS
        and orbit.get("excluded_train_pairs") == EXPECTED_EXCLUDED_PAIRS
        and orbit.get("identity_pairs") == EXPECTED_BASE_RECORDS
        and orbit.get("valid_shift_count_distribution")
        == {str(key): value for key, value in sorted(EXPECTED_VALID_SHIFT_DISTRIBUTION.items(), reverse=True)}
    ):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.fixture_orbit_invalid", repr(orbit)
        )
    return payload


__all__ = [
    "EXPECTED_BASE_RECORDS",
    "EXPECTED_ELIGIBLE_PAIRS",
    "EXPECTED_EXCLUDED_PAIRS",
    "EXPECTED_NOMINAL_PAIRS",
    "EXPECTED_VALID_SHIFT_DISTRIBUTION",
    "FULL_ORBIT_BATCH_SIZE",
    "FULL_ORBIT_CHECKPOINT_INTERVAL",
    "FULL_ORBIT_DRAW_BUDGET",
    "FULL_ORBIT_FIXTURE_SCHEMA",
    "FULL_ORBIT_PEAK_LEARNING_RATE",
    "FULL_ORBIT_PROFILE_ID",
    "FULL_ORBIT_PROGRESS_INTERVAL",
    "FULL_ORBIT_RNG_DOMAIN",
    "FULL_ORBIT_SEED",
    "FULL_ORBIT_UPDATE_BUDGET",
    "FULL_ORBIT_VALIDATION_INTERVAL",
    "FULL_ORBIT_WARMUP_UPDATES",
    "FullOrbitDraw",
    "FullOrbitPair",
    "FullOrbitRuntimeConfig",
    "FullOrbitSampler",
    "build_full_orbit_optimizer_scheduler",
    "build_full_orbit_table",
    "check_full_orbit_fixture",
    "compact_full_orbit_fixture",
    "excluded_pair_evidence",
    "full_orbit_epoch_permutation",
    "full_orbit_preflight",
    "full_orbit_profile_contract",
    "full_orbit_table_contract",
    "run_full_orbit_diagnostic_validation",
]
