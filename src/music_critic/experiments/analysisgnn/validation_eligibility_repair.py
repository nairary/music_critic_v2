"""Repair B5H/B5I all-shift VALIDATION eligibility.

B5A eligibility is TRAIN-only. The old finalizer queried it for VALIDATION,
leaving 162 empty shift lists after C2 had completed 120k updates. This module
derives VALIDATION eligibility through the already-audited executable B5G raw
and target checks. TEST remains unopened and training is unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from typing import Callable

from music_critic.experiments.analysisgnn import full_orbit_training
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedTrainingError,
    ProductionArtifactPaths,
    frozen_split_assignments,
    load_production_record,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.transposition import (
    SHIFT_PCS,
    AnalysisGNNTranspositionError,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    audit_sidecar_targets,
    prepare_sidecar_diagnostic_context,
)


VALIDATION_ELIGIBILITY_REPAIR_SCHEMA = (
    "Phase9EB5JValidationEligibilityRepair@1.0.0"
)
REPAIRED_FINALIZER_ID = (
    "music-critic-v2-validation-derived-all-shift-finalizer-v1"
)
ProgressCallback = Callable[[int, int, str], None]


def valid_shifts_for_validation_record(
    raw_graph_batch: object,
    sidecar: Mapping[str, object],
) -> tuple[int, ...]:
    """Derive executable shifts without reading TRAIN-only B5A rows."""

    prepared = prepare_sidecar_diagnostic_context(sidecar)
    valid: list[int] = []
    for shift_pc in SHIFT_PCS:
        try:
            transpose_raw_graph_batch(raw_graph_batch, (shift_pc,))
        except AnalysisGNNTranspositionError:
            continue
        target = audit_sidecar_targets(
            sidecar,
            shift_pc=shift_pc,
            prepared=prepared,
        )
        if (
            target["target_vocabulary_closed"] is True
            and target["round_trip_valid"] is True
            and target["masks_preserved"] is True
            and target["entity_ids_preserved"] is True
        ):
            valid.append(shift_pc)
    result = tuple(valid)
    if 0 not in result:
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.validation_identity_ineligible",
            str(sidecar.get("record_id", "<unknown>")),
        )
    return result


def validation_valid_shifts(
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, tuple[int, ...]]:
    """Load only VALIDATION records and derive each admissible shift set."""

    assignments = frozen_split_assignments(paths)
    record_ids = tuple(
        sorted(
            record_id
            for record_id, row in assignments.items()
            if row["split"] == "validation"
        )
    )
    result: dict[str, tuple[int, ...]] = {}
    for index, record_id in enumerate(record_ids, 1):
        batch, sidecar = load_production_record(
            record_id,
            split="validation",
            paths=paths,
        )
        if sidecar.get("record_id") != record_id:
            raise CorrectedTrainingError(
                "analysisgnn.full_orbit.validation_sidecar_binding_mismatch",
                record_id,
            )
        result[record_id] = valid_shifts_for_validation_record(
            batch.raw_graph_batch,
            sidecar,
        )
        if progress is not None:
            progress(index, len(record_ids), record_id)
    if len(result) != 162 or any(0 not in shifts for shifts in result.values()):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.validation_eligibility_incomplete",
            f"records={len(result)} identity={sum(0 in x for x in result.values())}",
        )
    return result


def validation_eligibility_repair_contract(
    eligibility: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    normalized = {
        str(record_id): tuple(sorted(set(int(value) for value in shifts)))
        for record_id, shifts in sorted(eligibility.items())
    }
    if len(normalized) != 162 or any(
        not shifts or 0 not in shifts for shifts in normalized.values()
    ):
        raise CorrectedTrainingError(
            "analysisgnn.full_orbit.validation_eligibility_invalid",
            f"records={len(normalized)}",
        )
    per_shift = Counter(
        shift for shifts in normalized.values() for shift in shifts
    )
    body: dict[str, object] = {
        "schema": VALIDATION_ELIGIBILITY_REPAIR_SCHEMA,
        "finalizer_id": REPAIRED_FINALIZER_ID,
        "record_count": len(normalized),
        "identity_record_count": per_shift[0],
        "eligible_record_shift_pairs": sum(map(len, normalized.values())),
        "per_shift_record_count": {
            str(shift): per_shift[shift] for shift in SHIFT_PCS
        },
        "eligibility_fingerprint": fingerprint(
            [[record_id, list(shifts)] for record_id, shifts in normalized.items()]
        ),
        "source_split": "validation",
        "eligibility_rule": "B5G_executable_raw_transform_and_target_closure",
        "train_b5a_eligibility_used": False,
        "training_or_optimizer_step_executed": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def json_progress(event: str) -> ProgressCallback:
    def progress(index: int, total: int, record_id: str) -> None:
        if index % 20 == 0 or index == total:
            print(
                json.dumps(
                    {
                        "event": event,
                        "record_index": index,
                        "record_count": total,
                        "record_id": record_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    return progress


def run_repaired_full_orbit_diagnostic_validation(
    model: CorrectedAnalysisGNNModel,
    *,
    device: str,
    paths: ProductionArtifactPaths = ProductionArtifactPaths(),
) -> dict[str, object]:
    """Run unchanged B5H metrics after repairing only shift eligibility."""

    eligibility = validation_valid_shifts(
        paths,
        progress=json_progress("validation_eligibility_progress"),
    )
    repair = validation_eligibility_repair_contract(eligibility)
    original = full_orbit_training._validation_valid_shifts
    full_orbit_training._validation_valid_shifts = lambda _paths: eligibility
    try:
        result = full_orbit_training.run_full_orbit_diagnostic_validation(
            model,
            device=device,
            paths=paths,
        )
    finally:
        full_orbit_training._validation_valid_shifts = original
    body = dict(result)
    body.pop("fingerprint", None)
    body["eligibility_repair"] = repair
    body["fingerprint"] = fingerprint(body)
    return body


__all__ = [
    "REPAIRED_FINALIZER_ID",
    "VALIDATION_ELIGIBILITY_REPAIR_SCHEMA",
    "json_progress",
    "run_repaired_full_orbit_diagnostic_validation",
    "valid_shifts_for_validation_record",
    "validation_eligibility_repair_contract",
    "validation_valid_shifts",
]
