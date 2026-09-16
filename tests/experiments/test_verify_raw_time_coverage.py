from copy import deepcopy
from fractions import Fraction
import gzip
import json

import pytest

from scripts import verify_raw_time_coverage as verify


def counters():
    head = {"target_rows": 4, "source_available_rows": 4, "unknown_available_rows": 0,
            "eligible_rows": 4, "historical_valid_rows": 3, "eligible_times": 3,
            "ideal_exact_errors": 1}
    variant = {"candidate_positions": 2, "covered_rows": 3, "uncovered_rows": 1,
               "covered_times": 2, "uncovered_times": 1,
               "historical_kept_rows": 2, "historical_lost_rows": 1,
               "newly_covered_rows": 1, "covered_exact_errors": 1, "conflict_times": 1,
               "historical_kept_exact_errors": 0, "historical_kept_old_slot_errors": 1,
               "temporal_errors_removed_on_kept_rows": 1}
    return head, variant


def selection_input():
    variants = {p: {h: {"historical_lost_rows": 0, "uncovered_rows": 0}
                    for h in verify.TIME_HEADS} for p in verify.POLICIES}
    inventory = dict(zip(verify.POLICIES, (10, 13, 12, 15)))
    return variants, inventory


def test_reconstructs_independent_exact_unions_and_terminal_boundaries():
    result = verify.reconstruct_candidates({"note_onsets": ["-1/2", "0", "1/3"],
        "note_offsets": ["0", "1/3", "2/3"], "beat_boundaries": ["-1/2", "1/2", "3/2"]})
    assert result[verify.POLICIES[0]] == {Fraction(-1, 2), Fraction(0), Fraction(1, 3)}
    assert Fraction(3, 2) in result[verify.POLICIES[-1]]
    assert len(result[verify.POLICIES[-1]]) == 6
    assert result[verify.POLICIES[1]] | result[verify.POLICIES[2]] == result[verify.POLICIES[-1]]


@pytest.mark.parametrize("values", [["1/3", "0"], ["0", "0"], ["2/4"], [0.5]])
def test_rejects_noncanonical_ambiguous_time_sources(values):
    with pytest.raises(ValueError):
        verify.reconstruct_candidates({"note_onsets": values, "note_offsets": [], "beat_boundaries": []})


def test_counter_comparison_detects_disappearing_nonzero_key():
    with pytest.raises(ValueError, match="conflict_slots"):
        verify.same_counts({"target_rows": 4}, {"target_rows": 4, "conflict_slots": 1}, "baseline")
    verify.same_counts({"target_rows": 4}, {"target_rows": 4, "conflict_slots": 0}, "baseline")


def test_coverage_equations_accept_unmapped_recovery_without_hiding_historical_loss():
    h, v = counters()
    verify.check_coverage(h, v, "fixture")


@pytest.mark.parametrize("field", ["covered_rows", "historical_lost_rows", "newly_covered_rows",
                                  "historical_kept_exact_errors", "temporal_errors_removed_on_kept_rows"])
def test_coverage_equations_reject_inconsistent_counts(field):
    h, v = counters()
    v[field] += 1
    with pytest.raises(ValueError):
        verify.check_coverage(h, v, "fixture")


def test_negative_or_boolean_counters_rejected():
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="non-integer/negative"):
            verify.counts({"rows": bad}, "bad")


def test_selection_requires_preservation_and_all_cadence_then_minimum():
    variants, inventory = selection_input()
    variants[verify.POLICIES[0]]["root"]["historical_lost_rows"] = 1
    variants[verify.POLICIES[2]]["cadence"]["uncovered_rows"] = 1
    qualifying, selected = verify.expected_selection(variants, inventory)
    assert qualifying == [verify.POLICIES[1], verify.POLICIES[3]]
    assert selected == verify.POLICIES[1]


def test_selection_does_not_require_all_old_unmapped_harmony_and_has_stable_tie():
    variants, inventory = selection_input()
    variants[verify.POLICIES[0]]["root"]["uncovered_rows"] = 2
    inventory[verify.POLICIES[2]] = inventory[verify.POLICIES[0]]
    assert verify.expected_selection(variants, inventory)[1] == verify.POLICIES[0]
    for p in verify.POLICIES:
        variants[p]["cadence"]["uncovered_rows"] = 1
    assert verify.expected_selection(variants, inventory) == ([], None)


def test_duplicate_record_ids_fail():
    with pytest.raises(ValueError, match="duplicate record"):
        verify.unique_index([{"record_id": "same"}, {"record_id": "same"}], "fixture")


def test_incomplete_run_rejected_before_access_to_other_inputs(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"schema": "RawTimeCoverage@1.0.0", "status": "running"}))
    with pytest.raises(ValueError, match="only a completed"):
        verify.verify(tmp_path, tmp_path / "missing-bindings", tmp_path / "missing-parent")


def test_existing_output_refused_before_run_access(tmp_path):
    output = tmp_path / "verification.json"
    output.write_text("immutable\n")
    with pytest.raises(ValueError, match="output already exists"):
        verify.main(["--run", str(tmp_path / "missing"), "--bindings", str(tmp_path / "missing"),
                     "--output", str(output)])
    assert output.read_text() == "immutable\n"


def test_complete_source_free_artifact_bundle_is_verified(tmp_path, monkeypatch):
    """Exercise artifact plumbing; corpus payloads do not exist in this fixture.

    The production fixed inventory is represented by empty synthetic pieces.
    Registry schema validation is outside this fixture; parent artifact hashes
    and all membership/accounting checks still execute normally.
    """
    from music_critic.experiments import registry
    monkeypatch.setattr(registry, "validate_record", lambda _: None)
    run = tmp_path / "run"
    run.mkdir()
    bindings = tmp_path / "bindings"
    (bindings / "metadata").mkdir(parents=True)
    parent_root = tmp_path / "outputs/target_slot_census_002/run"
    parent_root.mkdir(parents=True)
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    assignments = [{"record_id": f"fixture:{i}", "source_component_id": f"component:{i % 1209}",
                    "dialect": "an_joint" if i < 284 else "dlc", "split": "train"} for i in range(1295)]
    split_path = bindings / "metadata/split_assignments.jsonl"
    write_rows(split_path, assignments)
    split_sha = verify.digest(split_path)["sha256"]
    monkeypatch.setattr(verify, "SPLIT_SHA", split_sha)
    h = {k: 0 for k in ("target_rows", "source_available_rows", "unknown_available_rows",
                        "eligible_rows", "historical_valid_rows", "eligible_times", "ideal_exact_errors")}
    v = {"candidate_positions": 0, "pieces_with_uncovered_rows": 0, "pieces_with_historical_loss": 0}
    heads = {head: deepcopy(h) for head in verify.TIME_HEADS}
    variants = {p: {head: deepcopy(v) for head in verify.TIME_HEADS} for p in verify.POLICIES}
    notes = {head: {"target_rows": 0, "valid_rows": 0} for head in verify.NOTE_HEADS}
    inventory = {k: 0 for k in (*verify.POLICIES, "raw_note_count", "historical_beat_count")}
    sources, pieces, baseline, raw = [], [], [], []
    for a in assignments:
        rid = a["record_id"]
        sources.append({"record_id": rid, "split": "train", "component_id": a["source_component_id"],
                        "dialect": a["dialect"], "piece_sha256": "a" * 64, "sidecar_sha256": "b" * 64})
        pieces.append({"record_id": rid, "dialect": a["dialect"], "inventory": inventory, "heads": heads,
                       "variants": variants, "unchanged_note_heads": notes, "distinct_eligible_times": 0,
                       "covered_distinct_eligible_times": {p: 0 for p in verify.POLICIES}})
        baseline.append({"record_id": rid, "heads": {
            **{head: {"target_rows": 0, "valid_rows": 0, "source_available_rows": 0,
                      "unknown_available_rows": 0, "pieces": 1, "pieces_with_conflicts": 0}
               for head in verify.TIME_HEADS}, **notes}})
        raw.append({"record_id": rid, "piece_sha256": "a" * 64, "schema": "RawTimeSources@1.0.0",
                    "sources": {"note_onsets": [], "note_offsets": [], "beat_boundaries": []}})
    for name, rows in (("sources.jsonl", sources), ("pieces.jsonl", pieces)):
        write_rows(run / name, rows)
    write_rows(parent_root / "sources.jsonl", sources)
    write_rows(parent_root / "pieces.jsonl", baseline)
    (parent_root / "summary.json").write_text("{}\n")
    with gzip.open(run / "raw_candidates.jsonl.gz", "wt") as stream:
        stream.write("".join(json.dumps(r) + "\n" for r in raw))
    for name in ("uncovered.jsonl", "eligible_old_unmapped.jsonl", "simultaneous_conflicts.jsonl"):
        (run / name).write_text("")
    parent = {"experiment_id": "EXP-TARGET-SLOT-CENSUS-002", "status": "completed", "artifacts": [
        {"locator": f"outputs/target_slot_census_002/run/{name}", **verify.digest(parent_root / name)}
        for name in ("sources.jsonl", "pieces.jsonl", "summary.json")]}
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    summary = {"schema": "RawTimeCoverage@1.0.0", "status": "completed", "completed_records": 1295,
        "expected_records": 1295, "failures": [], "split_sha256": split_sha,
        "validation_payload_accessed": False, "test_payload_accessed": False,
        "inventory": inventory, "heads": heads, "class_counts": {head: {} for head in verify.TIME_HEADS},
        "variants": variants, "by_dialect": {d: variants for d in ("an_joint", "dlc")},
        "unchanged_note_heads": notes, "qualifying_policies": list(verify.POLICIES),
        "selected_policy": verify.POLICIES[0], "selection_status": "candidate_only_not_a_trained_readout",
        "historical_hash_matches": {"piece_equal": 1295, "sidecar_equal": 1295}}
    (run / "summary.json").write_text(json.dumps(summary))
    result = verify.verify(run, bindings, parent_path, tmp_path)
    assert result["status"] == "passed" and result["verified_records"] == 1295
    assert result["raw_tsv_read"] is False
