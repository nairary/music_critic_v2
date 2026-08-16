from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import scripts.audit_dilemmadata as audit_module

from scripts.audit_dilemmadata import (
    BoundedMemoryMultisetFingerprint,
    DilemmadataAuditError,
    NOTE_MULTISET_FINGERPRINT_DOMAIN,
    _compare_upstream,
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
    assert report["readiness"]["acceptance_backed_release_ready"] is False


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
        before["midi_note_event_multiset_grouping_fingerprint"]
        == after["midi_note_event_multiset_grouping_fingerprint"]
    )
    assert before["target_sidecar_fingerprint"] != after["target_sidecar_fingerprint"]


def test_grouping_closes_note_multisets_scores_and_explicit_overlaps() -> None:
    grouping = build_report(CORPUS)["grouping"]
    assert grouping["component_count"] == 1
    assert grouping["midi_note_event_multiset_equivalent_cluster_count"] == 1
    assert grouping["candidate_multiple_analysis_group_count"] == 1
    assert grouping["explicit_cross_source_overlap_count"] == 1
    assert grouping["suggested_split_conflict_count"] == 1
    assert set(grouping["multi_record_components"][0]["record_ids"]) == {
        "an:training:same",
        "an:validation:same-alt",
        "dlc:demo:same",
    }
    contract = grouping["midi_note_event_multiset_fingerprint_contract"]
    assert contract["bounded_memory"] is True
    assert contract["full_input_identity"] is False


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("\tP1\t1\t", "\tP1\t9\t"),
        ("\t0\t4\t4\tP1\t", "\t0\t3\t4\tP1\t"),
        ("\t60\tTrue\tC\t", "\t60\tFalse\tC\t"),
    ],
    ids=["voice", "meter", "tie"],
)
def test_narrow_note_multiset_fingerprint_is_not_full_input_identity(
    tmp_path: Path, old: str, new: str
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    before = _record(build_report(copied), "an:training:same")
    payload = source.read_text(encoding="utf-8")
    assert old in payload
    source.write_text(payload.replace(old, new, 1), encoding="utf-8")
    after = _record(build_report(copied), "an:training:same")

    assert (
        before["midi_note_event_multiset_grouping_fingerprint"]
        == after["midi_note_event_multiset_grouping_fingerprint"]
    )
    assert before["raw_observation_fingerprint"] != after["raw_observation_fingerprint"]


def test_bounded_multiset_fingerprint_has_domain_and_collision_defenses() -> None:
    def digest(domain: bytes, values: list[int]) -> str:
        fingerprint = BoundedMemoryMultisetFingerprint(domain)
        for value in values:
            fingerprint.add(value.to_bytes(32, "big"))
        return fingerprint.hexdigest()

    assert digest(NOTE_MULTISET_FINGERPRINT_DOMAIN, [1, 6]) == digest(
        NOTE_MULTISET_FINGERPRINT_DOMAIN, [6, 1]
    )
    # Both pairs have count=2, sum=7, and xor=7. The squared-sum term keeps
    # this deliberately adversarial low-order construction distinct.
    assert digest(NOTE_MULTISET_FINGERPRINT_DOMAIN, [1, 6]) != digest(
        NOTE_MULTISET_FINGERPRINT_DOMAIN, [2, 5]
    )
    assert digest(NOTE_MULTISET_FINGERPRINT_DOMAIN, [1, 6]) != digest(
        b"dilemmadata.midi-note-event-multiset-grouping.2\0", [1, 6]
    )
    assert digest(NOTE_MULTISET_FINGERPRINT_DOMAIN, [1, 1]) != digest(
        NOTE_MULTISET_FINGERPRINT_DOMAIN, [1]
    )


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

    for family in targets.values():
        for dialect in family["by_dialect"].values():
            assert (
                dialect["available"]
                + dialect["masked"]
                + dialect["missing"]
                + dialect["unsupported"]
                == dialect["rows_examined"]
            )
            assert dialect["primary_state_partition_valid"] is True
            assert dialect["ambiguous"] <= dialect["available"]
            assert dialect["ambiguous_subset_of_available"] is True


def test_gate_states_are_mutually_exclusive_for_true_false_missing_and_malformed(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    lines = source.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    gate_index = header.index("valid_chord_label")
    template = lines[1].split("\t")
    rows = []
    for gate in ("True", "False", "", "malformed"):
        row = template.copy()
        row[gate_index] = gate
        rows.append("\t".join(row))
    source.write_text("\n".join([lines[0], *rows]) + "\n", encoding="utf-8")

    report = build_report(copied)
    states = report["target_inventory"]["roman_numeral"]["by_dialect"]["an_joint"]
    # The untouched validation record contributes two additional available rows.
    assert states["rows_examined"] == 6
    assert states["available"] == 3
    assert states["masked"] == 2
    assert states["missing"] == 0
    assert states["unsupported"] == 1
    assert states["primary_state_total"] == states["rows_examined"]
    assert report["target_diagnostics"]["primary_state_policy"][
        "missing_is_negative"
    ] is False


def test_row_level_alt_label_is_not_family_wide_ambiguity(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "DLC" / "demo" / "same.tsv"
    rows = source.read_text(encoding="utf-8").splitlines()
    rows[0] += "\talt_label"
    rows[1] += "\talternative-row-evidence"
    rows[2] += "\t"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = build_report(copied)
    assert report["target_diagnostics"]["alt_label"]["by_dialect"]["dlc"] == 1
    assert all(
        family["by_dialect"]["dlc"]["ambiguous"] == 0
        for family in report["target_inventory"].values()
    )


def _make_clean_checkout(path: Path) -> str:
    shutil.copytree(CORPUS, path)
    subprocess.run(["git", "init", "-q", os.fspath(path)], check=True)
    subprocess.run(
        ["git", "-C", os.fspath(path), "config", "user.name", "Audit Test"], check=True
    )
    subprocess.run(
        ["git", "-C", os.fspath(path), "config", "user.email", "audit@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", os.fspath(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", os.fspath(path), "commit", "-q", "-m", "fixture"], check=True
    )
    return subprocess.run(
        ["git", "-C", os.fspath(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_upstream_comparison_is_acceptance_backed_with_stable_failure_categories(
    tmp_path: Path, monkeypatch
) -> None:
    clean = tmp_path / "clean"
    commit = _make_clean_checkout(clean)
    monkeypatch.setattr(audit_module, "RELEASE_COMMIT", commit)
    matched = _compare_upstream(CORPUS, clean)
    assert matched["performed"] is True
    assert matched["exact_match"] is True
    assert matched["checkout_clean"] is True
    assert matched["matching_file_count"] > 0
    assert matched["failure_categories"] == []

    monkeypatch.setattr(audit_module, "RELEASE_COMMIT", "0" * 40)
    assert _compare_upstream(CORPUS, clean)["failure_categories"] == [
        "upstream_commit_mismatch"
    ]
    monkeypatch.setattr(audit_module, "RELEASE_COMMIT", commit)

    local_only = tmp_path / "local-only"
    shutil.copytree(CORPUS, local_only)
    (local_only / "local-only.txt").write_text("local\n", encoding="utf-8")
    assert "upstream_local_only_files" in _compare_upstream(local_only, clean)[
        "failure_categories"
    ]

    (clean / "upstream-only.txt").write_text("upstream\n", encoding="utf-8")
    assert "upstream_only_files" in _compare_upstream(CORPUS, clean)[
        "failure_categories"
    ]
    (clean / "upstream-only.txt").unlink()

    readme = clean / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "mismatch\n", encoding="utf-8")
    categories = _compare_upstream(CORPUS, clean)["failure_categories"]
    assert "upstream_content_mismatch" in categories
    assert "upstream_checkout_dirty" in categories
