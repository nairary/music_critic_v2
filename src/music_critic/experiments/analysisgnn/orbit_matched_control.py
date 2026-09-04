"""Phase 9E-B5I schedule-matched C0 control for the 120k C2 experiment.

The control consumes the exact deterministic C2 full-orbit draw schedule, but
uses every scheduled shift only as a sampling stratum.  Both the raw graph and
the target sidecar are presented with identity shift zero.  This keeps record
order, multiplicity, optimizer budget, scheduler, validation cadence, and model
initialization matched to C2 while isolating TRAIN transposition.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
    ProductionArtifactPaths,
    production_component_records,
    production_valid_shifts,
)
from music_critic.experiments.analysisgnn.full_orbit_training import (
    FULL_ORBIT_BATCH_SIZE,
    FULL_ORBIT_CHECKPOINT_INTERVAL,
    FULL_ORBIT_DRAW_BUDGET,
    FULL_ORBIT_PEAK_LEARNING_RATE,
    FULL_ORBIT_PROFILE_ID,
    FULL_ORBIT_PROGRESS_INTERVAL,
    FULL_ORBIT_RNG_DOMAIN,
    FULL_ORBIT_SEED,
    FULL_ORBIT_UPDATE_BUDGET,
    FULL_ORBIT_VALIDATION_INTERVAL,
    FULL_ORBIT_WARMUP_UPDATES,
    FullOrbitDraw,
    FullOrbitPair,
    FullOrbitRuntimeConfig,
    FullOrbitSampler,
    build_full_orbit_optimizer_scheduler,
    build_full_orbit_table,
    full_orbit_preflight,
    full_orbit_profile_contract,
)
from music_critic.experiments.analysisgnn.training_policy import (
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
)


ORBIT_MATCHED_CONTROL_PROFILE_ID = (
    "music-critic-v2-corrected-no-transposition-orbit-matched-v1"
)
ORBIT_MATCHED_CONTROL_SCHEMA = (
    "CorrectedAnalysisGNNOrbitMatchedControlProfile@1.0.0"
)
ORBIT_MATCHED_CONTROL_RUNTIME_SCHEMA = (
    "CorrectedAnalysisGNNOrbitMatchedControlRuntime@1.0.0"
)
ORBIT_MATCHED_CONTROL_PREFLIGHT_SCHEMA = (
    "Phase9EB5IOrbitMatchedControlPreflight@1.0.0"
)
ORBIT_MATCHED_CONTROL_HISTORY_SCHEMA = (
    "Phase9EB5IOrbitMatchedControlHistory@1.0.0"
)
ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC = 0
ORBIT_MATCHED_CONTROL_LABEL = "C0-120K-MATCHED"


@dataclass(frozen=True, slots=True, order=True)
class OrbitMatchedControlDraw:
    """One C2 schedule row projected to an identity-only C0 training view."""

    orbit_epoch: int
    orbit_index: int
    component_id: str
    record_id: str
    schedule_shift_pc: int
    applied_shift_pc: int = ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC

    def __post_init__(self) -> None:
        if (
            self.orbit_epoch < 0
            or self.orbit_index < 0
            or not self.component_id
            or not self.record_id
            or self.schedule_shift_pc not in range(12)
            or self.applied_shift_pc != ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC
        ):
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.draw_invalid",
                repr(asdict(self)),
            )


def project_full_orbit_draw(draw: FullOrbitDraw) -> OrbitMatchedControlDraw:
    """Retain C2 sampling identity while forcing the applied shift to zero."""

    return OrbitMatchedControlDraw(
        orbit_epoch=draw.orbit_epoch,
        orbit_index=draw.orbit_index,
        component_id=draw.component_id,
        record_id=draw.record_id,
        schedule_shift_pc=draw.shift_pc,
    )


class OrbitMatchedControlSampler:
    """Checkpointable identity projection over the exact C2 orbit sampler."""

    def __init__(
        self,
        table: Sequence[FullOrbitPair],
        *,
        seed: int = FULL_ORBIT_SEED,
        position: int = 0,
    ) -> None:
        self._source = FullOrbitSampler(table, seed=seed, position=position)

    @property
    def position(self) -> int:
        return self._source.position

    @property
    def draws_per_epoch(self) -> int:
        return self._source.draws_per_epoch

    @property
    def table(self) -> tuple[FullOrbitPair, ...]:
        return self._source.table

    @property
    def seed(self) -> int:
        return self._source.seed

    def peek(self, offset: int = 0) -> OrbitMatchedControlDraw:
        return project_full_orbit_draw(self._source.peek(offset))

    def advance_after_applied_update(
        self, draws: int = FULL_ORBIT_BATCH_SIZE
    ) -> None:
        self._source.advance_after_applied_update(draws)

    def state_dict(self) -> dict[str, object]:
        source = self._source.state_dict()
        body: dict[str, object] = {
            "position": source["position"],
            "seed": source["seed"],
            "profile_id": ORBIT_MATCHED_CONTROL_PROFILE_ID,
            "matched_schedule_profile_id": FULL_ORBIT_PROFILE_ID,
            "table_fingerprint": source["table_fingerprint"],
            "rng_domain": source["rng_domain"],
            "applied_shift_pc": ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC,
        }
        body["fingerprint"] = fingerprint(body)
        return body

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        body = dict(state)
        observed_fingerprint = body.pop("fingerprint", None)
        if observed_fingerprint != fingerprint(body):
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.sampler_fingerprint_mismatch",
                repr(observed_fingerprint),
            )
        expected = self.state_dict()
        for key in (
            "seed",
            "profile_id",
            "matched_schedule_profile_id",
            "table_fingerprint",
            "rng_domain",
            "applied_shift_pc",
        ):
            if body.get(key) != expected[key]:
                raise CorrectedTrainingError(
                    "analysisgnn.orbit_matched_control.sampler_resume_mismatch",
                    key,
                )
        position = int(body["position"])
        if position < 0:
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.sampler_position_invalid",
                str(position),
            )
        source_state = self._source.state_dict()
        source_state["position"] = position
        source_body = {
            key: value for key, value in source_state.items() if key != "fingerprint"
        }
        source_state["fingerprint"] = fingerprint(source_body)
        self._source.load_state_dict(source_state)


@dataclass(frozen=True, slots=True)
class OrbitMatchedControlRuntimeConfig:
    """The C2 runtime envelope with identity-only TRAIN transforms."""

    schema_version: str = ORBIT_MATCHED_CONTROL_RUNTIME_SCHEMA
    profile_id: str = ORBIT_MATCHED_CONTROL_PROFILE_ID
    semantic_base_profile_id: str = CORRECTED_NO_TRANSPOSITION_PROFILE_ID
    matched_schedule_profile_id: str = FULL_ORBIT_PROFILE_ID
    seed: int = FULL_ORBIT_SEED
    device: str = "cuda"
    batch_size: int = FULL_ORBIT_BATCH_SIZE
    applied_update_budget: int = FULL_ORBIT_UPDATE_BUDGET
    train_draw_budget: int = FULL_ORBIT_DRAW_BUDGET
    warmup_applied_updates: int = FULL_ORBIT_WARMUP_UPDATES
    validation_interval: int = FULL_ORBIT_VALIDATION_INTERVAL
    checkpoint_interval: int = FULL_ORBIT_CHECKPOINT_INTERVAL
    progress_interval: int = FULL_ORBIT_PROGRESS_INTERVAL
    applied_shift_pc: int = ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC
    mixed_precision: str = "fp32_baseline"
    early_stopping: bool = False
    test_enabled: bool = False

    def __post_init__(self) -> None:
        c2 = FullOrbitRuntimeConfig()
        if (
            self.schema_version != ORBIT_MATCHED_CONTROL_RUNTIME_SCHEMA
            or self.profile_id != ORBIT_MATCHED_CONTROL_PROFILE_ID
            or self.semantic_base_profile_id
            != CORRECTED_NO_TRANSPOSITION_PROFILE_ID
            or self.matched_schedule_profile_id != FULL_ORBIT_PROFILE_ID
            or self.seed != c2.seed
            or self.device != c2.device
            or self.batch_size != c2.batch_size
            or self.applied_update_budget != c2.applied_update_budget
            or self.train_draw_budget != c2.train_draw_budget
            or self.warmup_applied_updates != c2.warmup_applied_updates
            or self.validation_interval != c2.validation_interval
            or self.checkpoint_interval != FULL_ORBIT_CHECKPOINT_INTERVAL
            or self.progress_interval != FULL_ORBIT_PROGRESS_INTERVAL
            or self.applied_shift_pc != ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC
            or self.mixed_precision != c2.mixed_precision
            or self.early_stopping
            or self.test_enabled
            or self.train_draw_budget
            != self.applied_update_budget * self.batch_size
        ):
            raise CorrectedTrainingError(
                "analysisgnn.orbit_matched_control.runtime_contract_changed",
                repr(asdict(self)),
            )

    def to_dict(self) -> dict[str, object]:
        body = asdict(self)
        body.update(
            {
                "scheduler": "linear_warmup_then_cosine_decay",
                "peak_learning_rate": FULL_ORBIT_PEAK_LEARNING_RATE,
                "checkpoint_selection": (
                    "identity_validation_corrected_primary_macro_score"
                ),
                "primary_validation_view": "identity_only",
                "diagnostic_validation_view": (
                    "all_eligible_shifts_per_shift_and_macro"
                ),
                "train_schedule": (
                    "exact_C2_full_orbit_record_shift_rows_and_permutations"
                ),
                "schedule_shift_role": "sampling_stratum_only_not_applied",
                "raw_graph_transform": "identity_shift_zero",
                "semantic_target_transform": "identity_shift_zero",
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


def orbit_matched_control_profile_contract() -> dict[str, object]:
    """Describe the causal C0-vs-C2 comparison boundary."""

    c2_profile = full_orbit_profile_contract()
    config = OrbitMatchedControlRuntimeConfig()
    body: dict[str, object] = {
        "schema": ORBIT_MATCHED_CONTROL_SCHEMA,
        "phase": "9E-B5I",
        "profile": ORBIT_MATCHED_CONTROL_LABEL,
        "profile_id": ORBIT_MATCHED_CONTROL_PROFILE_ID,
        "semantic_base_profile_id": CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
        "comparison_profile": "C2",
        "comparison_profile_id": FULL_ORBIT_PROFILE_ID,
        "comparison_profile_fingerprint": c2_profile["fingerprint"],
        "purpose": "isolate_train_transposition_at_matched_120k_compute",
        "scope": "TRAIN_only_after_piece_disjoint_split",
        "dataset_semantics": (
            "reuse_each_C2_eligible_record_shift_row_as_a_sampling_stratum_"
            "while_applying_identity_to_raw_and_targets"
        ),
        "pair_order": "exact_C2_full_orbit_epoch_permutation",
        "rng_domain": FULL_ORBIT_RNG_DOMAIN,
        "record_multiplicity": "exactly_equal_to_C2",
        "scheduled_shift_applied": False,
        "applied_shift_pc": ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC,
        "runtime": config.to_dict(),
        "primary_validation": "identity_only",
        "diagnostic_validation": "same_all_shift_diagnostic_as_C2",
        "from_scratch": True,
        "resume_from_c0_10k_checkpoint": False,
        "resume_from_c2_checkpoint": False,
        "model_architecture_changed": False,
        "loss_or_class_weights_changed": False,
        "split_changed": False,
        "raw_cache_changed": False,
        "c2_implementation_changed": False,
        "test_enabled": False,
        "control_training_run": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def build_orbit_matched_control_optimizer_scheduler(
    model: CorrectedAnalysisGNNModel,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Delegate to the exact C2 optimizer/scheduler implementation."""

    return build_full_orbit_optimizer_scheduler(model)


def _source_draw_payload(draw: OrbitMatchedControlDraw) -> dict[str, object]:
    return {
        "orbit_epoch": draw.orbit_epoch,
        "orbit_index": draw.orbit_index,
        "component_id": draw.component_id,
        "record_id": draw.record_id,
        "shift_pc": draw.schedule_shift_pc,
    }


def orbit_matched_control_preflight(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, object]:
    """Bind the control to the exact production C2 table and permutations."""

    c2_preflight = full_orbit_preflight(paths)
    if (
        c2_preflight.get("valid") is not True
        or c2_preflight.get("ready_for_full_orbit_training") is not True
        or c2_preflight.get("test_loader_created") is not False
        or c2_preflight.get("test_targets_read") is not False
        or c2_preflight.get("test_metrics_computed") is not False
    ):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.c2_preflight_invalid",
            repr(c2_preflight.get("fingerprint")),
        )

    table = build_full_orbit_table(
        production_component_records(paths), production_valid_shifts(paths)
    )
    sampler = OrbitMatchedControlSampler(table)
    first_epoch = tuple(sampler.peek(offset) for offset in range(len(table)))
    partial_start = (FULL_ORBIT_DRAW_BUDGET // len(table)) * len(table)
    partial = tuple(
        sampler.peek(partial_start + offset)
        for offset in range(FULL_ORBIT_DRAW_BUDGET % len(table))
    )
    first_source_fingerprint = fingerprint(
        [_source_draw_payload(draw) for draw in first_epoch]
    )
    partial_source_fingerprint = fingerprint(
        [_source_draw_payload(draw) for draw in partial]
    )
    first_matches = (
        first_source_fingerprint
        == c2_preflight.get("first_epoch_permutation_fingerprint")
    )
    partial_matches = (
        partial_source_fingerprint
        == c2_preflight.get("partial_final_epoch_fingerprint")
    )
    if not first_matches or not partial_matches:
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.schedule_mismatch",
            f"first={first_matches} partial={partial_matches}",
        )

    body: dict[str, object] = {
        "schema": ORBIT_MATCHED_CONTROL_PREFLIGHT_SCHEMA,
        "phase": "9E-B5I",
        "valid": True,
        "profile": orbit_matched_control_profile_contract(),
        "matched_c2_preflight_fingerprint": c2_preflight["fingerprint"],
        "matched_c2_profile_fingerprint": c2_preflight["profile"]["fingerprint"],
        "orbit_table": c2_preflight["orbit_table"],
        "first_epoch_draw_count": len(first_epoch),
        "first_epoch_source_permutation_fingerprint": first_source_fingerprint,
        "first_epoch_matches_c2": first_matches,
        "partial_final_epoch_draw_count": len(partial),
        "partial_final_epoch_source_fingerprint": partial_source_fingerprint,
        "partial_final_epoch_matches_c2": partial_matches,
        "scheduled_shift_values": sorted(
            {draw.schedule_shift_pc for draw in first_epoch}
        ),
        "applied_shift_values": [ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC],
        "initial_model_state_fingerprint": c2_preflight[
            "initial_model_state_fingerprint"
        ],
        "model_contract_fingerprint": c2_preflight[
            "model_contract_fingerprint"
        ],
        "same_model_initialization_as_c2": True,
        "same_optimizer_scheduler_as_c2": True,
        "same_record_multiplicity_and_order_as_c2": True,
        "only_train_view_difference": "C2_schedule_shift_vs_identity_zero",
        "ready_for_orbit_matched_control_training": True,
        "control_training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def completed_control_history_contract(
    record_history: Sequence[str],
    schedule_shift_history: Sequence[int],
    applied_shift_history: Sequence[int],
    *,
    expected_draws: int = FULL_ORBIT_DRAW_BUDGET,
) -> dict[str, object]:
    """Validate and fingerprint the completed causal schedule projection."""

    if (
        expected_draws <= 0
        or len(record_history) != expected_draws
        or len(schedule_shift_history) != expected_draws
        or len(applied_shift_history) != expected_draws
    ):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.history_length_mismatch",
            (
                f"expected={expected_draws} records={len(record_history)} "
                f"scheduled={len(schedule_shift_history)} "
                f"applied={len(applied_shift_history)}"
            ),
        )
    if any(not record_id for record_id in record_history):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.history_record_invalid",
            "empty record id",
        )
    if any(shift not in range(12) for shift in schedule_shift_history):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.history_schedule_shift_invalid",
            repr(sorted(set(schedule_shift_history))),
        )
    if any(
        shift != ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC
        for shift in applied_shift_history
    ):
        raise CorrectedTrainingError(
            "analysisgnn.orbit_matched_control.non_identity_shift_applied",
            repr(sorted(set(applied_shift_history))),
        )

    scheduled_distribution = Counter(schedule_shift_history)
    applied_distribution = Counter(applied_shift_history)
    body: dict[str, Any] = {
        "schema": ORBIT_MATCHED_CONTROL_HISTORY_SCHEMA,
        "draw_count": expected_draws,
        "record_history_fingerprint": fingerprint(list(record_history)),
        "schedule_shift_history_fingerprint": fingerprint(
            list(schedule_shift_history)
        ),
        "record_schedule_pair_fingerprint": fingerprint(
            [
                [record_id, shift]
                for record_id, shift in zip(
                    record_history, schedule_shift_history, strict=True
                )
            ]
        ),
        "applied_shift_history_fingerprint": fingerprint(
            list(applied_shift_history)
        ),
        "scheduled_shift_distribution": {
            str(key): value for key, value in sorted(scheduled_distribution.items())
        },
        "applied_shift_distribution": {
            str(key): value for key, value in sorted(applied_distribution.items())
        },
        "scheduled_shift_is_sampling_stratum_only": True,
        "all_applied_shifts_are_identity": True,
    }
    body["fingerprint"] = fingerprint(body)
    return body


__all__ = [
    "ORBIT_MATCHED_CONTROL_APPLIED_SHIFT_PC",
    "ORBIT_MATCHED_CONTROL_HISTORY_SCHEMA",
    "ORBIT_MATCHED_CONTROL_LABEL",
    "ORBIT_MATCHED_CONTROL_PREFLIGHT_SCHEMA",
    "ORBIT_MATCHED_CONTROL_PROFILE_ID",
    "ORBIT_MATCHED_CONTROL_RUNTIME_SCHEMA",
    "ORBIT_MATCHED_CONTROL_SCHEMA",
    "OrbitMatchedControlDraw",
    "OrbitMatchedControlRuntimeConfig",
    "OrbitMatchedControlSampler",
    "build_orbit_matched_control_optimizer_scheduler",
    "completed_control_history_contract",
    "orbit_matched_control_preflight",
    "orbit_matched_control_profile_contract",
    "project_full_orbit_draw",
]
