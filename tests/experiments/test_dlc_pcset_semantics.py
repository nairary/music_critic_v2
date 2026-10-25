"""Synthetic source rows: no real corpus or checkpoint access."""

from __future__ import annotations

import csv
from dataclasses import replace
import shutil

import pytest

from music_critic.experiments.analysisgnn.multitask_contract import (
    DLC_PCSET_DECODER_ID,
    LEGACY_TARGET_SIDECAR_VERSION,
    TARGET_SIDECAR_VERSION,
    AnalysisGNNMultitaskContractError,
    get_entity_mappings,
    materialize_target_sidecar,
    materialize_target_sidecar_descriptor,
    sidecar_contract_counts,
)
from music_critic.experiments.analysisgnn.transposition import transform_semantic_value
from music_critic.graph import build_raw_graph, model_input_fingerprint
from tests.adapters.test_dilemmadata import CORPUS, _accepted


def _source(tmp_path, *, dialect="dlc", **fields):
    record_id = "dlc:demo:same" if dialect == "dlc" else "an:training:same"
    accepted = _accepted(CORPUS, record_id)
    with accepted.record.path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = list(reader.fieldnames)
        rows = list(reader)
    for name in fields:
        if name not in header:
            header.append(name)
    for row in rows:
        row.update(fields)
    path = tmp_path / "synthetic.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    # Only target columns change; retain the real adapter's raw/source bindings.
    return replace(accepted, record=replace(accepted.record, path=path))


def _pcsets(sidecar):
    return [
        entity["targets"]["pitch_class_set"]
        for entity in sidecar["entities"]
        if entity["entity_type"] == "harmonic_event"
    ]


def test_dlc_fifths_are_projected_to_absolute_pitch_classes(tmp_path):
    # Independent example: C/E/A# relative to F -> F/A/D# -> PCs 3,5,9.
    accepted = _source(tmp_path, chord_tones="(0, 4, 10)", localkey_tpc="-1")
    states = _pcsets(materialize_target_sidecar(accepted))
    assert states
    assert all(state["canonical_value"] == "3,5,9" for state in states)
    assert all(state["available"] and not state["masked"] for state in states)
    assert all(state["provenance"]["decoder_id"] == DLC_PCSET_DECODER_ID for state in states)
    assert all(state["source_value"] == "(0, 4, 10)" for state in states)


def test_dlc_normalizes_order_duplicates_and_fifths_outside_pc_range(tmp_path):
    accepted = _source(
        tmp_path, chord_tones="(10, 0, 4, 0, -12, 12)", localkey_tpc="-1",
        added_tones="(1, 2)", a_pcset="0,4,10",
    )
    states = _pcsets(materialize_target_sidecar(accepted))
    assert all(state["canonical_value"] == "3,5,9" for state in states)
    assert all(state["provenance"]["source_field"] == "chord_tones" for state in states)
    assert all(state["provenance"]["includes_added_tones"] is False for state in states)


@pytest.mark.parametrize("localkey,reason", [
    ("", "dlc_localkey_tpc_missing"), ("None", "dlc_localkey_tpc_missing"),
    ("NaN", "dlc_localkey_tpc_missing"), ("True", "dlc_localkey_tpc_invalid"),
    ("1.5", "dlc_localkey_tpc_invalid"), ("inf", "dlc_localkey_tpc_invalid"),
])
def test_missing_or_invalid_localkey_never_becomes_a_target(tmp_path, localkey, reason):
    accepted = _source(tmp_path, chord_tones="(0, 4, 10)", localkey_tpc=localkey)
    for state in _pcsets(materialize_target_sidecar(accepted)):
        assert state["masked"] and not state["available"]
        assert state["canonical_value"] is None
        assert state["missing_reason"] == reason


@pytest.mark.parametrize("tones", ["()", "(True, 4, 1)", "(0, 4.5, 1)", "{0, 4, 1}", "bad", "[(0, 4)]"])
def test_invalid_fifths_never_become_a_target(tmp_path, tones):
    accepted = _source(tmp_path, chord_tones=tones, localkey_tpc="0")
    for state in _pcsets(materialize_target_sidecar(accepted)):
        assert state["masked"] and not state["available"]
        assert state["missing_reason"] == "dlc_chord_tones_invalid"


def test_valid_decoding_outside_pinned_vocabulary_is_masked(tmp_path):
    accepted = _source(tmp_path, chord_tones="(0,)", localkey_tpc="0")
    for state in _pcsets(materialize_target_sidecar(accepted)):
        assert state["masked"] and not state["available"]
        assert state["canonical_value"] is None
        assert state["missing_reason"] == "unknown_value_not_in_pinned_production_vocabulary"
        assert state["provenance"]["decoded_absolute_pcset"] == "0"


@pytest.mark.parametrize("repair", [False, True])
def test_existing_validity_and_repair_masks_take_precedence(tmp_path, repair):
    accepted = _source(tmp_path, chord_tones="(0, 4, 1)", localkey_tpc="0",
                       valid_chord_label="True" if repair else "False")
    if repair:
        accepted = replace(accepted, alignment_evidence=replace(
            accepted.alignment_evidence,
            rows=tuple(replace(row, repair_mask_scope="all") for row in accepted.alignment_evidence.rows),
        ))
    expected = "raw_repair_mask_scope_all" if repair else "valid_chord_label_false_or_missing"
    for state in _pcsets(materialize_target_sidecar(accepted)):
        assert state["masked"] and not state["available"]
        assert state["missing_reason"] == expected


@pytest.mark.parametrize("tonic,expected", [
    (0, "0,4,7"), (7, "1,5,8"), (2, "2,6,9"), (9, "3,7,10"),
    (4, "4,8,11"), (11, "0,5,9"), (6, "1,6,10"), (1, "2,7,11"),
    (8, "0,3,8"), (3, "1,4,9"), (10, "2,5,10"), (5, "3,6,11"),
])
def test_decoding_commutes_with_all_twelve_pc_transpositions(tmp_path, tonic, expected):
    accepted = _source(tmp_path, chord_tones="(0, 4, 1)", localkey_tpc=str(tonic))
    assert all(state["canonical_value"] == expected for state in _pcsets(materialize_target_sidecar(accepted)))
    shift = (7 * tonic) % 12
    assert transform_semantic_value("pitch_class_set", "0,4,7", shift_pc=shift, dialect="dlc") == expected
    assert transform_semantic_value("pitch_class_set", expected, shift_pc=(-shift) % 12, dialect="dlc") == "0,4,7"


def test_an_uses_absolute_a_pcset_without_dlc_conversion(tmp_path):
    accepted = _source(tmp_path, dialect="an_joint", a_pcset="(0, 4, 7)",
                       chord_tones="(0, 4, 10)", localkey_tpc="invalid")
    assert all(state["canonical_value"] == "0,4,7" for state in _pcsets(materialize_target_sidecar(accepted)))


def test_targets_do_not_change_raw_model_inputs(tmp_path):
    accepted = _source(tmp_path, chord_tones="(0, 4, 10)", localkey_tpc="-1")
    # Rediscovery authenticates the edited synthetic source before conversion.
    corpus = tmp_path / "corpus"
    shutil.copytree(CORPUS, corpus)
    shutil.copyfile(accepted.record.path, corpus / accepted.record.relative_path)
    converted = _accepted(corpus, accepted.record.record_id)
    assert model_input_fingerprint(build_raw_graph(accepted.piece)) == model_input_fingerprint(build_raw_graph(converted.piece))


@pytest.mark.parametrize("version", [LEGACY_TARGET_SIDECAR_VERSION, TARGET_SIDECAR_VERSION])
def test_descriptor_and_expanded_targets_agree(tmp_path, version):
    accepted = _source(tmp_path, chord_tones="(0, 4, 10)", localkey_tpc="-1")
    sidecar = materialize_target_sidecar(accepted, target_sidecar_version=version)
    descriptor = materialize_target_sidecar_descriptor(accepted, target_sidecar_version=version)
    assert descriptor["schema_version"] == sidecar["schema_version"] == version
    assert descriptor["task_states"] == sidecar_contract_counts(sidecar)["task_states"]
    assert descriptor["entity_counts"] == sidecar["entity_counts"]
    assert descriptor["relation_counts"] == sidecar["relation_counts"]
    assert get_entity_mappings(sidecar)


@pytest.mark.parametrize("record_id,expanded,descriptor", [
    ("dlc:demo:same", "96eea7f7676e7faeba22d79f74286f0649f4000761d7d9a31d9026a4910e29ec", "15740b8d7264eea20f8eb9b6b3c45fb87d0e95f30d35a4a3e5a656ef5fc32940"),
    ("an:training:same", "0bec8da224a6f72b3ce69c5163bc17d476d5a843858302cdc21c49e003a5e256", "efcfb34820748bb19be048fa1010da0762ee5a660e75e7bec7a0ce31eba21d7a"),
])
def test_legacy_fixture_fingerprints_match_pre_repair_snapshot(record_id, expanded, descriptor):
    accepted = _accepted(CORPUS, record_id)
    kwargs = {"target_sidecar_version": LEGACY_TARGET_SIDECAR_VERSION}
    assert materialize_target_sidecar(accepted, **kwargs)["fingerprint"] == expanded
    assert materialize_target_sidecar_descriptor(accepted, **kwargs)["fingerprint"] == descriptor


def test_legacy_wrong_semantics_are_only_explicit_and_fingerprints_differ(tmp_path):
    accepted = _source(tmp_path, chord_tones="(0, 4, 10)", localkey_tpc="-1")
    legacy = materialize_target_sidecar(accepted, target_sidecar_version=LEGACY_TARGET_SIDECAR_VERSION)
    current = materialize_target_sidecar(accepted)
    assert all(state["canonical_value"] == "0,4,10" for state in _pcsets(legacy))
    assert legacy["fingerprint"] == "92505dc3e039a1454b3f54ae7cfd2d591f94866809ab0d9e6255cf9e0d64b57f"
    assert current["schema_version"] == TARGET_SIDECAR_VERSION != LEGACY_TARGET_SIDECAR_VERSION
    assert current["fingerprint"] != legacy["fingerprint"]


@pytest.mark.parametrize("materializer", [materialize_target_sidecar, materialize_target_sidecar_descriptor])
def test_unknown_version_fails_closed(materializer):
    with pytest.raises(AnalysisGNNMultitaskContractError, match="unknown target sidecar version"):
        materializer(_accepted(CORPUS, "dlc:demo:same"), target_sidecar_version="typo")
