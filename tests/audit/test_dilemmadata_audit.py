from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from scripts.audit_dilemmadata import (
    DilemmadataAuditError,
    build_report,
    dumps_report,
    ensure_output_outside_root,
    main,
    manifest_projection,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "dilemmadata"
CORPUS = FIXTURES / "corpus"


def _record(report: dict[str, object], record_id: str) -> dict[str, object]:
    records = report["per_record"]
    assert isinstance(records, list)
    return next(row for row in records if row["record_id"] == record_id)


def test_bounded_fixture_covers_both_dialects_and_raw_contract() -> None:
    report = build_report(CORPUS)

    assert report["record_inventory"]["discovered_primary_record_count"] == 3
    assert report["record_inventory"]["note_rows_by_dialect"] == {
        "an_joint": 4,
        "dlc": 2,
    }
    assert report["formats"]["primary_dialects"] == ["an_joint", "dlc"]
    raw = report["raw_musical_representation"]
    assert raw["classification"] == "score-derived symbolic note arrays, not raw MIDI"
    assert raw["target_independent_note_projection"] is True
    assert raw["acceptance_counts"]["raw_compatible_note_projection_records"] == 3
    assert report["quarantine"]["record_count"] == 0


def test_report_is_deterministic_canonical_and_location_independent(tmp_path: Path) -> None:
    first = build_report(CORPUS)
    second = build_report(CORPUS)
    assert dumps_report(first) == dumps_report(second)
    assert first["semantic_fingerprint"] == second["semantic_fingerprint"]
    assert str(CORPUS.resolve()) not in dumps_report(first)

    relocated = tmp_path / "different-installation-name"
    shutil.copytree(CORPUS, relocated)
    third = build_report(relocated)
    assert first["semantic_fingerprint"] == third["semantic_fingerprint"]


def test_target_mutation_does_not_change_raw_or_midi_projection(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    before = _record(build_report(copied), "an:training:same")
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    payload = source.read_text(encoding="utf-8")
    source.write_text(payload.replace("\tI\tC\tM\t", "\tvi\tC\tM\t", 1), encoding="utf-8")
    after = _record(build_report(copied), "an:training:same")

    assert before["raw_observation_fingerprint"] == after["raw_observation_fingerprint"]
    assert (
        before["midi_compatible_note_projection_fingerprint"]
        == after["midi_compatible_note_projection_fingerprint"]
    )
    assert before["target_sidecar_fingerprint"] != after["target_sidecar_fingerprint"]


def test_grouping_closes_exact_inputs_scores_and_explicit_overlaps() -> None:
    grouping = build_report(CORPUS)["grouping"]
    assert grouping["component_count"] == 1
    assert grouping["exact_equivalent_input_cluster_count"] == 1
    assert grouping["alternative_analysis_cluster_count"] == 1
    assert grouping["explicit_cross_source_overlap_count"] == 1
    assert grouping["suggested_split_conflict_count"] == 1
    assert set(grouping["multi_record_components"][0]["record_ids"]) == {
        "an:training:same",
        "an:validation:same-alt",
        "dlc:demo:same",
    }


def test_malformed_records_are_structurally_quarantined(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    shutil.copy2(
        FIXTURES / "malformed" / "an_missing_field_joint.tsv",
        copied / "pitch_arrays" / "AN" / "training" / "broken_joint.tsv",
    )
    errors = copied / "pitch_arrays" / "DLC" / "errors"
    errors.mkdir()
    shutil.copy2(FIXTURES / "malformed" / "dlc_bad_width.tsv", errors / "bad.tsv")

    report = build_report(copied)
    quarantine = report["quarantine"]
    assert quarantine["record_count"] == 2
    assert quarantine["category_counts"] == {
        "missing_required_field": 1,
        "row_width_mismatch": 1,
    }
    records = {row["record_id"]: row for row in quarantine["records"]}
    assert records["an:training:broken"]["error_counts"]["missing_required_field"] == 1
    assert records["dlc:errors:bad"]["error_counts"]["row_width_mismatch"] == 1
    assert all(not Path(row["relative_path"]).is_absolute() for row in quarantine["records"])


def test_cli_supports_env_output_limit_and_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MUSIC_CRITIC_DILEMMADATA_ROOT", os.fspath(CORPUS))
    reference = build_report(CORPUS)
    manifest = tmp_path / "fixture-manifest.json"
    manifest.write_text(
        json.dumps(manifest_projection(reference), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    assert main(["--output", os.fspath(output), "--check", os.fspath(manifest)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == reference

    bounded = tmp_path / "bounded.json"
    assert main(["--output", os.fspath(bounded), "--limit", "1"]) == 0
    bounded_report = json.loads(bounded.read_text(encoding="utf-8"))
    assert bounded_report["record_inventory"]["selected_primary_record_count"] == 1
    assert bounded_report["readiness"]["evidence_violations"] == [
        "bounded_limit_is_not_complete_evidence"
    ]


def test_cli_rejects_writes_below_dataset_root() -> None:
    inside = CORPUS / "audit-output.json"
    assert main(["--root", os.fspath(CORPUS), "--output", os.fspath(inside)]) == 2
    assert not inside.exists()
    try:
        ensure_output_outside_root(CORPUS, inside)
    except DilemmadataAuditError:
        pass
    else:
        raise AssertionError("output inside corpus root was accepted")


def test_target_inventory_uses_masks_and_keeps_source_specific_semantics() -> None:
    targets = build_report(CORPUS)["target_inventory"]
    assert targets["roman_numeral"]["cross_source_mapping"] == "source_specific"
    assert targets["borrowed_harmony"]["by_dialect"]["an_joint"]["available"] == 0
    assert targets["borrowed_harmony"]["by_dialect"]["an_joint"]["masked"] == 4
    assert targets["voice_role"]["cross_source_mapping"] == "incompatible"
    assert targets["cadence"]["by_dialect"]["dlc"]["available"] == 1
    assert targets["global_key"]["by_dialect"]["dlc"][
        "source_entries_after_note_row_deduplication"
    ] == 1
    assert targets["chord_boundary"]["by_dialect"]["dlc"][
        "source_entries_after_note_row_deduplication"
    ] == 2
    assert targets["note_degree"]["by_dialect"]["dlc"][
        "source_entries_after_note_row_deduplication"
    ] == 2
