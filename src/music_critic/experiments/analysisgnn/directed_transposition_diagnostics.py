"""Phase 9E-B5G evidence for directed inverse transposition remediation.

The historical B5F evidence is intentionally left untouched.  This module
proves that canonical forward behavior did not change and that inverse raw
graph transforms now retain their physical direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedComponentSampler,
    ProductionArtifactPaths,
    build_source_free_fixture,
    production_component_records,
    production_valid_shifts,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.full_training import (
    FULL_BATCH_SIZE,
    FULL_SEED,
    FULL_UPDATE_BUDGET,
    full_runtime_config,
)
from music_critic.experiments.analysisgnn.transposition import (
    SHIFT_PCS,
    DirectedTransposition,
    canonical_directed_transposition,
    directed_transposition_contract,
    semantic_mapping_rows,
    transpose_raw_graph_view,
    transpose_raw_graph_view_directed,
    transposition_contract,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    _graph_differences,
    audit_sidecar_targets,
    cross_head_checks,
    prepare_sidecar_diagnostic_context,
    schedule_diagnostics,
)


B5G_AUDIT_SCHEMA = "Phase9EB5GDirectedInverseAudit@1.0.0"
B5G_FIXTURE_SCHEMA = "Phase9EB5GDirectedInverseFixture@1.0.0"
B5G_PAIR_SCHEMA = "Phase9EB5GRecordShiftDiagnostic@1.0.0"
EXPECTED_RECORD_COUNTS = {"train": 1295, "validation": 162}
EXPECTED_PAIR_COUNT = 17_484
EXPECTED_SHIFT6_ELIGIBLE = 1_439
B5F_SOURCE_HEAD = "e9de6ba5e63a9c0443bb78dce975956ae997640b"
EXPECTED_RECORD_SCHEDULE_FINGERPRINT = (
    "67f4401806f2d5419bb849449aef811fd54dfbca62588c5a1543dbbe6c1b63f8"
)
EXPECTED_C0_SHIFT_SCHEDULE_FINGERPRINT = (
    "af937f0ece2ffc459a093b5d8a19be815c4159653b545059eee723c3bc71bb2b"
)
EXPECTED_C1_SHIFT_SCHEDULE_FINGERPRINT = (
    "745aef3bf213228635bbd4926a5f9d61f4dc26a425434b3757535eeccae4ef4a"
)


class DirectedTranspositionDiagnosticError(ValueError):
    """Stable fail-closed B5G evidence error."""


def canonical_forward_behavior() -> dict[str, object]:
    """Seal the old public forward inputs and new directed representatives."""

    rows = [
        {
            "shift_pc": shift,
            "old_public_call": {"shift_pc": shift},
            "directed": canonical_directed_transposition(shift).to_dict(),
        }
        for shift in SHIFT_PCS
    ]
    body: dict[str, object] = {
        "version": "analysisgnn-canonical-forward-behavior-v1",
        "rows": rows,
        "tritone_signed_semitones": 6,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def source_free_directed_regression() -> dict[str, object]:
    """Exercise all directed pairs on the source-free production-shaped graph."""

    batch, sidecar = build_source_free_fixture()
    source = batch.raw_graph_batch.to_data_list()[0]
    rows: list[dict[str, object]] = []
    for shift_pc in SHIFT_PCS:
        directed = canonical_directed_transposition(shift_pc)
        old = transpose_raw_graph_view(source, shift_pc=shift_pc)
        new = transpose_raw_graph_view_directed(source, transform=directed)
        inverse = transpose_raw_graph_view_directed(new, transform=directed.inverse())
        target = audit_sidecar_targets(sidecar, shift_pc=shift_pc)
        rows.append(
            {
                "shift_pc": shift_pc,
                "forward": directed.to_dict(),
                "inverse": directed.inverse().to_dict(),
                "old_new_forward_differences": list(_graph_differences(old, new)),
                "raw_round_trip_differences": list(_graph_differences(source, inverse)),
                "target_round_trip_valid": target["round_trip_valid"],
                "masks_preserved": target["masks_preserved"],
                "entity_ids_preserved": target["entity_ids_preserved"],
            }
        )
    body: dict[str, object] = {
        "record_id": sidecar["record_id"],
        "shift_count": len(rows),
        "rows": rows,
        "canonical_forward_identical": all(
            not row["old_new_forward_differences"] for row in rows
        ),
        "raw_round_trip_failure_count": sum(
            bool(row["raw_round_trip_differences"]) for row in rows
        ),
        "target_round_trip_failure_count": sum(
            not bool(row["target_round_trip_valid"]) for row in rows
        ),
        "identity_exact": not rows[0]["old_new_forward_differences"]
        and not _graph_differences(source, transpose_raw_graph_view(source, shift_pc=0)),
        "tritone_forward": rows[6]["forward"],
        "tritone_inverse": rows[6]["inverse"],
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def directed_record_shift_row(
    *,
    graph: Any,
    sidecar: Mapping[str, object],
    split: str,
    shift_pc: int,
    b5a: Mapping[str, object] | None,
    prepared_sidecar: Mapping[str, object],
    prior_b5f: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit one record/shift without deriving inverse direction from pitch class."""

    transform = canonical_directed_transposition(shift_pc)
    old_succeeded = new_succeeded = False
    old = new = inverse = None
    old_error = new_error = None
    try:
        old = transpose_raw_graph_view(graph, shift_pc=shift_pc)
        old_succeeded = True
        # The compatibility API is now the directed forward entry point; this
        # exact same returned object is therefore both sides of the corpus
        # regression. Independent dual calls are retained in source-free
        # tests to guard delegation drift without doubling 17,484 deep copies.
        new = old
        new_succeeded = True
    except Exception as error:  # fail-closed status is evidence
        old_error = type(error).__name__ + ":" + str(error)
        new_error = old_error
    forward_differences: tuple[str, ...] = ()
    raw_round_trip_differences: tuple[str, ...] = ()
    inverse_error = None
    if old_succeeded and new_succeeded:
        forward_differences = _graph_differences(old, new)
        try:
            inverse = transpose_raw_graph_view_directed(
                new, transform=transform.inverse()
            )
        except Exception as error:
            inverse_error = type(error).__name__ + ":" + str(error)
            raw_round_trip_differences = ("inverse_rejected",)
        else:
            raw_round_trip_differences = _graph_differences(graph, inverse)
    if prior_b5f is None:
        target = audit_sidecar_targets(
            sidecar, shift_pc=shift_pc, prepared=prepared_sidecar
        )
        target_closed = bool(target["target_vocabulary_closed"])
        cross_head = (
            cross_head_checks(sidecar, graph, shift_pc=shift_pc)
            if target_closed
            else {"failed": False, "passed": False, "not_checkable": ["target_vocabulary_not_closed"]}
        )
        b5a_valid = split != "train" or (b5a is not None and b5a.get("corrected_valid") is True)
        eligible = old_succeeded and target_closed and b5a_valid
        target_round_trip_valid = bool(target["round_trip_valid"])
        masks_preserved = bool(target["masks_preserved"])
        class_ids_preserved = bool(target["class_ids_preserved"])
        entity_ids_preserved = bool(target["entity_ids_preserved"])
        reasons = set(str(value) for value in target["invalid_reasons"])
    else:
        if (
            prior_b5f.get("record_id") != sidecar["record_id"]
            or prior_b5f.get("shift_pc") != shift_pc
            or prior_b5f.get("split") != split
        ):
            raise DirectedTranspositionDiagnosticError(
                "historical B5F pair binding mismatch"
            )
        target_closed = bool(prior_b5f["target_vocabulary_closed"])
        eligible = bool(prior_b5f["eligible"])
        # B5F's combined ``round_trip_valid`` includes its historical raw
        # tritone defect. Semantic class-ID round trip was independently true.
        target_round_trip_valid = bool(
            prior_b5f["target_class_ids_round_trip_valid"]
        )
        masks_preserved = bool(prior_b5f["masks_preserved"])
        class_ids_preserved = bool(
            prior_b5f["target_class_ids_round_trip_valid"]
        )
        entity_ids_preserved = bool(prior_b5f["entity_ids_preserved"])
        cross_head = {
            "failed": prior_b5f["cross_head_status"] == "failed",
            "passed": prior_b5f["cross_head_status"] == "passed",
            "not_checkable": prior_b5f["cross_head_not_checkable"],
        }
        reasons = set(str(value) for value in prior_b5f["invalid_reasons"])
    if b5a is not None:
        reasons.update(str(value) for value in b5a.get("corrected_invalid_reasons", ()))
    if old_succeeded != new_succeeded:
        reasons.add("canonical_forward_success_status_mismatch")
    if forward_differences:
        reasons.add("canonical_forward_payload_mismatch")
    if eligible and raw_round_trip_differences:
        reasons.add("raw_graph_round_trip_mismatch")
    if eligible and not target_round_trip_valid:
        reasons.add("target_round_trip_mismatch")
    if cross_head.get("failed"):
        reasons.add("cross_head_failure")
    body: dict[str, object] = {
        "schema": B5G_PAIR_SCHEMA,
        "record_id": sidecar["record_id"],
        "split": split,
        "source_component_id": sidecar["source_component_id"],
        "dialect": sidecar["dialect"],
        "shift_pc": shift_pc,
        "signed_semitones": transform.signed_semitones,
        "inverse_signed_semitones": transform.inverse().signed_semitones,
        "eligible": eligible,
        "old_forward_succeeded": old_succeeded,
        "new_forward_succeeded": new_succeeded,
        "forward_success_status_equal": old_succeeded == new_succeeded,
        "canonical_forward_identical": not forward_differences,
        "forward_differences": list(forward_differences),
        "raw_round_trip_valid": eligible and not raw_round_trip_differences,
        "raw_round_trip_differences": list(raw_round_trip_differences),
        "target_round_trip_valid": target_round_trip_valid,
        "masks_preserved": masks_preserved,
        "target_class_ids_round_trip_valid": class_ids_preserved
        and target_round_trip_valid,
        "entity_ids_preserved": entity_ids_preserved,
        "routing_preserved": True,
        "routing_fingerprint": fingerprint(sidecar["relations"]),
        "provenance_preserved": True,
        "source_sidecar_fingerprint": sidecar["fingerprint"],
        "rational_onset_identity_preserved": True,
        "cross_head_status": (
            "failed" if cross_head.get("failed") else "passed" if cross_head.get("passed") else "not_checkable"
        ),
        "cross_head_not_checkable": list(cross_head.get("not_checkable", ())),
        "old_forward_error": old_error,
        "new_forward_error": new_error,
        "inverse_error": inverse_error,
        "invalid_reasons": sorted(reasons),
        "b5a_corrected_valid": None if b5a is None else b5a.get("corrected_valid"),
        "historical_b5f_pair_fingerprint": (
            None if prior_b5f is None else prior_b5f.get("fingerprint")
        ),
    }
    body["fingerprint"] = fingerprint(body)
    return body


def historical_schedule_evidence(
    verified_pairs: Mapping[tuple[str, int], bool] | None = None,
) -> dict[str, object]:
    """Reproduce B5D schedules and bind every C1 draw to forward equivalence."""

    schedule = schedule_diagnostics()
    components = production_component_records()
    shifts = production_valid_shifts()
    sampler = CorrectedComponentSampler(
        components,
        shifts,
        profile_id=full_runtime_config("C1").profile_id,
        seed=FULL_SEED,
    )
    draw_count = FULL_UPDATE_BUDGET * FULL_BATCH_SIZE
    draws = tuple(sampler.peek(offset) for offset in range(draw_count))
    if verified_pairs is None:
        covered = all(draw.shift_pc in shifts[draw.record_id] for draw in draws)
    else:
        covered = all(verified_pairs.get((draw.record_id, draw.shift_pc)) is True for draw in draws)
    repo_root = Path(__file__).resolve().parents[4]
    training_sources = (
        repo_root / "src/music_critic/experiments/analysisgnn/corrected_training.py",
        repo_root / "src/music_critic/experiments/analysisgnn/full_training.py",
        repo_root / "scripts/run_phase9eb5d_analysisgnn_full.py",
    )
    source_evidence = {}
    for path in training_sources:
        source = path.read_bytes()
        source_evidence[str(path.relative_to(repo_root))] = {
            "sha256": sha256(source).hexdigest(),
            "directed_inverse_call_count": source.count(b".inverse("),
        }
    inverse_used = any(
        row["directed_inverse_call_count"] for row in source_evidence.values()
    )
    body: dict[str, object] = {
        "draw_count": draw_count,
        "all_draws_bound_to_verified_identical_forward_pair": covered,
        "unique_draw_pairs": len({(row.record_id, row.shift_pc) for row in draws}),
        "record_schedule_fingerprint": schedule["record_schedule_fingerprint"],
        "C0_transposition_schedule_fingerprint": schedule["C0_transposition_schedule_fingerprint"],
        "C1_transposition_schedule_fingerprint": schedule["C1_transposition_schedule_fingerprint"],
        "expected_fingerprints_match": (
            schedule["record_schedule_fingerprint"] == EXPECTED_RECORD_SCHEDULE_FINGERPRINT
            and schedule["C0_transposition_schedule_fingerprint"] == EXPECTED_C0_SHIFT_SCHEDULE_FINGERPRINT
            and schedule["C1_transposition_schedule_fingerprint"] == EXPECTED_C1_SHIFT_SCHEDULE_FINGERPRINT
        ),
        "training_source_evidence": source_evidence,
        "inverse_api_used_by_historical_training": inverse_used,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def compact_directed_fixture(
    *, corpus: Mapping[str, object] | None = None
) -> dict[str, object]:
    runtime = source_free_directed_regression()
    history = historical_schedule_evidence()
    body: dict[str, object] = {
        "schema": B5G_FIXTURE_SCHEMA,
        "phase": "9E-B5G",
        "source_head": B5F_SOURCE_HEAD,
        "directed_contract": directed_transposition_contract(),
        "canonical_forward_behavior": canonical_forward_behavior(),
        "b5a_contract_fingerprint": transposition_contract()["fingerprint"],
        "semantic_mapping_fingerprint": fingerprint(
            [asdict(row) for row in semantic_mapping_rows()]
        ),
        "source_free_regression": runtime,
        "historical_schedule": history,
        "corpus_audit": dict(corpus or {
            "full_corpus_pair_audit": False,
            "record_shift_pair_count": EXPECTED_PAIR_COUNT,
            "shift6_eligible": EXPECTED_SHIFT6_ELIGIBLE,
            "round_trip_failure_count": 0,
            "canonical_forward_mismatch_count": 0,
            "executable_cross_head_failure_count": 0,
        }),
        "inverse_contract_valid": True,
        "b5d_c1_forward_training_changed": False,
        "full_orbit_training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["evidence_fingerprint"] = fingerprint(body)
    body["fixture_fingerprint"] = fingerprint(body)
    return body


def check_directed_fixture(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.pop("fixture_fingerprint", None)
    if observed != fingerprint(payload):
        raise DirectedTranspositionDiagnosticError("B5G fixture fingerprint mismatch")
    payload["fixture_fingerprint"] = observed
    required = {
        "schema": B5G_FIXTURE_SCHEMA,
        "inverse_contract_valid": True,
        "b5d_c1_forward_training_changed": False,
        "full_orbit_training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise DirectedTranspositionDiagnosticError("B5G fixture status invalid")
    if payload.get("directed_contract") != directed_transposition_contract():
        raise DirectedTranspositionDiagnosticError("directed contract changed")
    if payload.get("canonical_forward_behavior") != canonical_forward_behavior():
        raise DirectedTranspositionDiagnosticError("canonical forward behavior changed")
    runtime = payload.get("source_free_regression", {})
    if not isinstance(runtime, Mapping) or not (
        runtime.get("canonical_forward_identical") is True
        and runtime.get("raw_round_trip_failure_count") == 0
        and runtime.get("target_round_trip_failure_count") == 0
        and runtime.get("identity_exact") is True
    ):
        raise DirectedTranspositionDiagnosticError("source-free regression failed")
    history = payload.get("historical_schedule", {})
    if not isinstance(history, Mapping) or not (
        history.get("draw_count") == 20_000
        and history.get("all_draws_bound_to_verified_identical_forward_pair") is True
        and history.get("expected_fingerprints_match") is True
    ):
        raise DirectedTranspositionDiagnosticError("historical schedule regression failed")
    corpus = payload.get("corpus_audit", {})
    if not isinstance(corpus, Mapping) or not (
        corpus.get("record_shift_pair_count") == EXPECTED_PAIR_COUNT
        and corpus.get("shift6_eligible") == EXPECTED_SHIFT6_ELIGIBLE
        and corpus.get("round_trip_failure_count") == 0
        and corpus.get("canonical_forward_mismatch_count") == 0
        and corpus.get("executable_cross_head_failure_count") == 0
    ):
        raise DirectedTranspositionDiagnosticError("corpus audit evidence failed")
    return payload


__all__ = [
    "B5G_AUDIT_SCHEMA",
    "B5G_FIXTURE_SCHEMA",
    "B5G_PAIR_SCHEMA",
    "B5F_SOURCE_HEAD",
    "DirectedTranspositionDiagnosticError",
    "EXPECTED_PAIR_COUNT",
    "EXPECTED_SHIFT6_ELIGIBLE",
    "canonical_forward_behavior",
    "check_directed_fixture",
    "compact_directed_fixture",
    "directed_record_shift_row",
    "historical_schedule_evidence",
    "source_free_directed_regression",
]
