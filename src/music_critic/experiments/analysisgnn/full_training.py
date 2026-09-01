"""Frozen Phase 9E-B5D paired full-training screen contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedRuntimeConfig,
    CorrectedTrainingError,
    resolved_optimizer_contract,
)
from music_critic.experiments.analysisgnn.training_policy import (
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
)


FULL_TRAINING_SCHEMA = "CorrectedAnalysisGNNFullTraining@1.0.0"
FULL_RUN_SUMMARY_SCHEMA = "Phase9EB5DFullRunSummary@1.0.0"
FULL_COMPARISON_SCHEMA = "Phase9EB5DC0C1Comparison@1.0.0"
FULL_AUDIT_SCHEMA = "Phase9EB5DFullTrainingAudit@1.0.0"
FULL_SEED = 17
FULL_BATCH_SIZE = 2
FULL_UPDATE_BUDGET = 10_000
FULL_DRAWS_PER_EPOCH = 1_295
FULL_VALIDATION_INTERVAL = 500
FULL_CHECKPOINT_INTERVAL = 100
FULL_PROGRESS_INTERVAL = 25
FULL_WARMUP_UPDATES = 500
FULL_PROFILE_IDS = {
    "C0": CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    "C1": CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
}


@dataclass(frozen=True, slots=True)
class FullTrainingContract:
    schema: str = FULL_TRAINING_SCHEMA
    phase: str = "9E-B5D"
    seed: int = FULL_SEED
    batch_size: int = FULL_BATCH_SIZE
    applied_update_budget: int = FULL_UPDATE_BUDGET
    draws_per_update: int = FULL_BATCH_SIZE
    draws_per_epoch: int = FULL_DRAWS_PER_EPOCH
    validation_interval: int = FULL_VALIDATION_INTERVAL
    checkpoint_interval: int = FULL_CHECKPOINT_INTERVAL
    progress_interval: int = FULL_PROGRESS_INTERVAL
    warmup_applied_updates: int = FULL_WARMUP_UPDATES
    scheduler: str = "linear_warmup_then_cosine_decay"
    mixed_precision: str = "fp32_baseline"
    early_stopping: bool = False
    checkpoint_selection: str = "corrected_primary_macro_score"
    validation_view: str = "identity_only"
    test_enabled: bool = False
    profile_o_enabled: bool = False
    multi_seed_claim: bool = False

    @property
    def train_draw_budget(self) -> int:
        return self.applied_update_budget * self.draws_per_update

    @property
    def sampler_epochs(self) -> float:
        return self.train_draw_budget / self.draws_per_epoch

    def to_dict(self) -> dict[str, object]:
        body = asdict(self)
        body["train_draw_budget"] = self.train_draw_budget
        body["sampler_epochs"] = self.sampler_epochs
        body["profiles"] = dict(FULL_PROFILE_IDS)
        body["validation_updates"] = list(full_validation_updates())
        body["optimizer"] = resolved_optimizer_contract(batch_size=self.batch_size)
        body["fingerprint"] = fingerprint(body)
        return body


def full_training_contract() -> dict[str, object]:
    return FullTrainingContract().to_dict()


def full_validation_updates() -> tuple[int, ...]:
    return tuple(range(0, FULL_UPDATE_BUDGET + 1, FULL_VALIDATION_INTERVAL))


def full_runtime_config(profile: str) -> CorrectedRuntimeConfig:
    if profile not in FULL_PROFILE_IDS:
        raise CorrectedTrainingError(
            "analysisgnn.full.profile_invalid", profile
        )
    return CorrectedRuntimeConfig(
        profile_id=FULL_PROFILE_IDS[profile],
        seed=FULL_SEED,
        device="cuda",
        batch_size=FULL_BATCH_SIZE,
        applied_update_budget=FULL_UPDATE_BUDGET,
        validation_interval=FULL_VALIDATION_INTERVAL,
    )


def full_run_root_name(profile: str) -> str:
    if profile not in FULL_PROFILE_IDS:
        raise CorrectedTrainingError(
            "analysisgnn.full.profile_invalid", profile
        )
    return f"{profile.casefold()}-seed{FULL_SEED}-full-u{FULL_UPDATE_BUDGET}"


def _validate_validation_rows(
    rows: Sequence[Mapping[str, object]], *, profile: str
) -> None:
    updates = tuple(int(row["applied_update"]) for row in rows)
    if updates != full_validation_updates():
        raise CorrectedTrainingError(
            "analysisgnn.full.validation_schedule_mismatch",
            f"{profile}:{updates}",
        )
    scores = [row.get("corrected_primary_macro_score") for row in rows]
    if any(
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        for score in scores
    ):
        raise CorrectedTrainingError(
            "analysisgnn.full.primary_score_missing", profile
        )


def build_full_comparison(
    *,
    summaries: Mapping[str, Mapping[str, object]],
    validation: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    if set(summaries) != {"C0", "C1"} or set(validation) != {"C0", "C1"}:
        raise CorrectedTrainingError(
            "analysisgnn.full.comparison_profiles_incomplete", "C0/C1 required"
        )
    for profile in ("C0", "C1"):
        summary = summaries[profile]
        if (
            summary.get("valid") is not True
            or summary.get("schema") != FULL_RUN_SUMMARY_SCHEMA
            or summary.get("profile") != profile
            or summary.get("profile_id") != FULL_PROFILE_IDS[profile]
            or summary.get("phase") != "9E-B5D"
            or summary.get("seed") != FULL_SEED
            or summary.get("batch_size") != FULL_BATCH_SIZE
            or summary.get("applied_updates") != FULL_UPDATE_BUDGET
            or summary.get("train_draws")
            != FULL_UPDATE_BUDGET * FULL_BATCH_SIZE
            or summary.get("validation_updates")
            != list(full_validation_updates())
            or summary.get("full_training_run") is not True
            or summary.get("test_evaluated") is not False
            or summary.get("test_targets_used_for_evaluation") is not False
            or summary.get("multi_seed_run") is not False
            or summary.get("nan_count") != 0
            or summary.get("overflow_count") != 0
            or summary.get("skipped_update_count") != 0
        ):
            raise CorrectedTrainingError(
                "analysisgnn.full.summary_invalid", profile
            )
        if (
            not isinstance(summary.get("best_primary_score"), (int, float))
            or isinstance(summary.get("best_primary_score"), bool)
            or not math.isfinite(float(summary["best_primary_score"]))
            or not isinstance(summary.get("best_update"), int)
            or isinstance(summary.get("best_update"), bool)
            or summary.get("best_update") not in full_validation_updates()
            or not isinstance(summary.get("final_primary_score"), (int, float))
            or isinstance(summary.get("final_primary_score"), bool)
            or not math.isfinite(float(summary["final_primary_score"]))
        ):
            raise CorrectedTrainingError(
                "analysisgnn.full.best_checkpoint_invalid", profile
            )
        _validate_validation_rows(validation[profile], profile=profile)

    initial_equal = (
        summaries["C0"]["initial_model_state_fingerprint"]
        == summaries["C1"]["initial_model_state_fingerprint"]
    )
    records_equal = (
        summaries["C0"]["record_schedule_fingerprint"]
        == summaries["C1"]["record_schedule_fingerprint"]
    )
    shifts_differ = (
        summaries["C0"]["transposition_schedule_fingerprint"]
        != summaries["C1"]["transposition_schedule_fingerprint"]
    )
    if not initial_equal or not records_equal or not shifts_differ:
        raise CorrectedTrainingError(
            "analysisgnn.full.causal_pairing_failed",
            f"initial={initial_equal} records={records_equal} shifts={shifts_differ}",
        )

    for profile in ("C0", "C1"):
        summary = summaries[profile]
        if float(summary["final_primary_score"]) != float(
            validation[profile][-1]["corrected_primary_macro_score"]
        ):
            raise CorrectedTrainingError(
                "analysisgnn.full.final_score_mismatch", profile
            )
        sealed = dict(summary)
        observed_fingerprint = sealed.pop("fingerprint", None)
        if observed_fingerprint != fingerprint(sealed):
            raise CorrectedTrainingError(
                "analysisgnn.full.summary_fingerprint_mismatch", profile
            )

    final_scores = {
        profile: float(validation[profile][-1]["corrected_primary_macro_score"])
        for profile in ("C0", "C1")
    }
    best_scores = {
        profile: float(summaries[profile]["best_primary_score"])
        for profile in ("C0", "C1")
    }
    final_delta = final_scores["C1"] - final_scores["C0"]
    best_delta = best_scores["C1"] - best_scores["C0"]
    direction = (
        "positive" if final_delta > 0 else "negative" if final_delta < 0 else "no"
    )
    payload: dict[str, Any] = {
        "schema": FULL_COMPARISON_SCHEMA,
        "phase": "9E-B5D",
        "seed": FULL_SEED,
        "applied_updates": FULL_UPDATE_BUDGET,
        "train_draws": FULL_UPDATE_BUDGET * FULL_BATCH_SIZE,
        "sampler_epochs": FullTrainingContract().sampler_epochs,
        "initial_state_fingerprints_equal": initial_equal,
        "record_schedule_fingerprints_equal": records_equal,
        "transposition_schedule_fingerprints_differ": shifts_differ,
        "final_primary_scores": final_scores,
        "final_primary_score_delta_C1_minus_C0": final_delta,
        "best_primary_scores": best_scores,
        "best_primary_score_delta_C1_minus_C0": best_delta,
        "best_updates": {
            profile: int(summaries[profile]["best_update"])
            for profile in ("C0", "C1")
        },
        "validation_curves": {
            profile: [
                {
                    "applied_update": int(row["applied_update"]),
                    "corrected_primary_macro_score": float(
                        row["corrected_primary_macro_score"]
                    ),
                }
                for row in validation[profile]
            ]
            for profile in ("C0", "C1")
        },
        "directional_conclusion": (
            f"seed-17 10000-update paired full screen shows {direction} "
            "directional evidence"
        ),
        "statistical_improvement_claim": False,
        "test_evaluated": False,
        "multi_seed_run": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


__all__ = [
    "FULL_AUDIT_SCHEMA",
    "FULL_BATCH_SIZE",
    "FULL_CHECKPOINT_INTERVAL",
    "FULL_COMPARISON_SCHEMA",
    "FULL_DRAWS_PER_EPOCH",
    "FULL_PROFILE_IDS",
    "FULL_PROGRESS_INTERVAL",
    "FULL_RUN_SUMMARY_SCHEMA",
    "FULL_SEED",
    "FULL_TRAINING_SCHEMA",
    "FULL_UPDATE_BUDGET",
    "FULL_VALIDATION_INTERVAL",
    "FULL_WARMUP_UPDATES",
    "FullTrainingContract",
    "build_full_comparison",
    "full_run_root_name",
    "full_runtime_config",
    "full_training_contract",
    "full_validation_updates",
]
