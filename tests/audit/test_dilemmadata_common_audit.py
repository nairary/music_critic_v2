from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.audit_dilemmadata_common_projection import (
    DEFAULT_MANIFEST,
    build_common_audit_report,
    check_committed_manifest,
    manifest_projection,
    validate_manifest_self_fingerprint,
)
from music_critic.tasks.dilemmadata_common import (
    ANALYSISGNN_REFERENCE,
    DILEMMADATA_COMMON_HARMONIC_REGISTRY,
    dumps_dilemmadata_common_audit_manifest,
    dumps_dilemmadata_common_audit_report,
    loads_dilemmadata_common_audit_manifest,
    make_dilemmadata_common_audit_manifest,
)
from tests.adapters.test_dilemmadata import CORPUS, _fixture_identity


BASE_SHA = "6490e231716cb191d4e476c0f4854adc03c57eb4"


def _report():
    return build_common_audit_report(
        CORPUS,
        identity=_fixture_identity(CORPUS),
        base_git_sha=BASE_SHA,
    )


def test_bounded_common_audit_is_deterministic_path_free_and_complete() -> None:
    first = _report()
    second = _report()
    assert dumps_dilemmadata_common_audit_report(first) == (
        dumps_dilemmadata_common_audit_report(second)
    )
    payload = dumps_dilemmadata_common_audit_report(first)
    assert str(CORPUS.resolve()) not in payload
    assert first.source_record_count == 3
    assert first.source_component_count == 1
    assert first.annotation_view_count == first.projection_count == 3
    assert len(first.collapse_table) > 0
    assert len(first.analysisgnn_parity) == 89
    assert {row.name for row in first.invariance_evidence} == {
        "canonical_piece",
        "grouping",
        "model_input",
        "raw_graph",
        "source_target_bundle",
    }
    assert all(row.unchanged for row in first.invariance_evidence)
    assert first.test_target_access_policy == (
        "representation_audit_only_no_model_inference_metrics_selection_or_unlock"
    )


def test_audit_separates_full_source_runtime_split_and_mapping_states() -> None:
    report = _report()
    names = {fact.name for fact in report.facts}
    assert {
        "full_source_mapping_state_count",
        "accepted_common_mapping_state_count",
        "source_native_class_count",
        "common_class_count",
        "train_support_min",
        "train_support_max",
        "train_imbalance_ratio_microunits",
    } <= names
    full_dimensions = [
        dict(fact.dimensions)
        for fact in report.facts
        if fact.name == "full_source_mapping_state_count"
    ]
    assert {row["raw_status"] for row in full_dimensions} == {"accepted"}
    assert {row["dialect"] for row in full_dimensions} == {"an_joint", "dlc"}
    assert {row["state"] for row in full_dimensions} >= {
        "exact",
        "unsupported",
    }
    assert any(row.comparison == "conflict" for row in report.overlap_evidence)


def test_analysisgnn_parity_and_lossy_collapses_are_explicit() -> None:
    report = _report()
    assert ANALYSISGNN_REFERENCE.license_spdx == "MIT"
    assert ANALYSISGNN_REFERENCE.fingerprint in {
        value for name, value in report.fingerprints if name == "analysisgnn_reference"
    }
    parity_by_source = {
        (task, source): (reference, common, agreement)
        for task, source, reference, common, agreement in report.analysisgnn_parity
    }
    assert parity_by_source[("dilemmadata.an.chord.inversion", "2")] == (
        "second",
        "second",
        "agree",
    )
    assert parity_by_source[("dilemmadata.dlc.chord.inversion", "2")] == (
        "third",
        "third",
        "agree",
    )
    inversion_parity = [
        row for row in report.analysisgnn_parity if row[0].endswith("chord.inversion")
    ]
    assert Counter(row[4] for row in inversion_parity) == Counter({"agree": 10})
    assert Counter(row[4] for row in report.analysisgnn_parity) == Counter(
        {"agree": 36, "diverge": 2, "not_applicable": 51}
    )
    divergence = {
        (task, source, reference, common)
        for task, source, reference, common, agreement in report.analysisgnn_parity
        if agreement == "diverge"
    }
    assert divergence == {
        (
            "dilemmadata.dlc.chord.quality",
            "+7",
            "augmented triad",
            "augmented seventh chord",
        ),
        (
            "dilemmadata.dlc.chord.quality",
            "+M7",
            "augmented triad",
            "augmented major tetrachord",
        ),
    }
    assert any(
        any(state == "coarsened" for _dialect, _source, state in row.source_rows)
        for row in report.collapse_table
    )


def test_manifest_is_canonical_self_fingerprinted_and_source_free(tmp_path: Path) -> None:
    report = _report()
    bounded = manifest_projection(report)
    # The three-record fixture has no validation/test members, so it is
    # intentionally not production-ready.  Source-free mechanics are tested
    # with the same report bindings and an explicit ready gate.
    assert bounded.ready is False
    manifest = make_dilemmadata_common_audit_manifest(
        report,
        summary_facts=bounded.summary_facts,
        ready=True,
    )
    payload = dumps_dilemmadata_common_audit_manifest(manifest)
    assert loads_dilemmadata_common_audit_manifest(payload) == manifest
    assert validate_manifest_self_fingerprint(json.loads(payload))
    path = tmp_path / "manifest.json"
    path.write_text(payload, encoding="utf-8")
    checked = check_committed_manifest(path)
    # A valid self-fingerprint cannot promote a bounded report that lacks the
    # production split and invariant evidence.
    assert checked["valid"] is False
    assert checked["ready"] is False

    mutated = json.loads(payload)
    mutated["summary_facts"][0]["value"] += 1
    assert not validate_manifest_self_fingerprint(mutated)


def test_committed_common_manifest_contract_when_present() -> None:
    # The implementation task commits this file from the opt-in full audit.
    assert DEFAULT_MANIFEST.is_file()
    checked = check_committed_manifest(DEFAULT_MANIFEST)
    assert checked["valid"] is True
    assert checked["ready"] is True
    manifest = loads_dilemmadata_common_audit_manifest(
        DEFAULT_MANIFEST.read_text(encoding="utf-8")
    )
    assert manifest.registry_fingerprint == (
        DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint
    )
    assert manifest.analysisgnn_reference_fingerprint == (
        ANALYSISGNN_REFERENCE.fingerprint
    )
    summary = {
        (fact.name, fact.dimensions): fact.value for fact in manifest.summary_facts
    }
    assert summary[
        ("overlap_comparison_total", (("comparison", "unavailable"),))
    ] == 702
    assert summary[
        ("accepted_common_mapping_state_total", (("state", "invalid"),))
    ] == 0
    assert summary[
        ("candidate_same_input_alternative_group_count", ())
    ] == 30
    assert summary[("analysisgnn_parity_row_count", ())] == 89
    assert summary[
        ("analysisgnn_parity_total", (("agreement", "agree"),))
    ] == 36
    assert summary[
        ("analysisgnn_parity_total", (("agreement", "diverge"),))
    ] == 2
    assert summary[
        ("analysisgnn_parity_total", (("agreement", "not_applicable"),))
    ] == 51
