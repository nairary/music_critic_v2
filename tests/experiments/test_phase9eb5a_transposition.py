from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import CanonicalPiece, load_piece
from music_critic.experiments.analysisgnn.class_balance import (
    EntityTargetObservation,
    RecordTargetObservations,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID
from music_critic.experiments.analysisgnn.transposition import (
    CORRECTED_PROFILE_ID,
    OFFICIAL_PROFILE_ID,
    SEMANTIC_MAPPING_ROWS,
    AnalysisGNNTranspositionError,
    AugmentedGraphIdentity,
    PostTranspositionAccumulator,
    corrected_transposition_profile,
    graph_changed_fields,
    mapping_composition_summary,
    model_input_collision_fingerprint,
    official_transposition_evidence,
    select_record_shift,
    semantic_mapping_index,
    transformation_registry,
    transform_semantic_value,
    transpose_raw_graph_view,
    transpose_record_observations,
    valid_shift_for_midi,
)
from music_critic.graph import build_raw_graph, graph_fingerprint


FIXTURE = Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"


def _piece() -> CanonicalPiece:
    return load_piece(FIXTURE)


def _target(
    task_id: str,
    value: str | None,
    *,
    available: bool = True,
    masked: bool = False,
) -> EntityTargetObservation:
    return EntityTargetObservation(
        task_id,
        f"{task_id}:entity",
        f"{task_id}:source-row" if value is not None else None,
        value,
        available,
        masked,
    )


def _record(dialect: str = "an_joint") -> RecordTargetObservations:
    return RecordTargetObservations(
        "record:fixture",
        "component:fixture",
        dialect,  # type: ignore[arg-type]
        "train",
        (
            _target("local_key", "C"),
            _target("root", "C"),
            _target("quality", "augmented seventh chord"),
            _target("inversion", "2"),
            _target("roman_numeral", "V7"),
            _target("note_degree", "#4"),
            _target("phrase", None, available=False, masked=True),
        ),
    )


def test_profiles_are_separate_pinned_contracts() -> None:
    official = official_transposition_evidence()
    corrected = corrected_transposition_profile()
    assert official["profile_id"] == OFFICIAL_PROFILE_ID
    assert corrected["profile_id"] == CORRECTED_PROFILE_ID
    assert official["fingerprint"] != corrected["fingerprint"]
    assert corrected["signed_semitones"] == [0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1]
    assert corrected["validation_behavior"] == "identity_only"
    assert corrected["test_behavior"] == "identity_only_without_target_access"


def test_registry_classifies_every_frozen_head_and_note_degree_is_relative() -> None:
    registry = transformation_registry()
    assert len(registry) == len(TASK_BY_ID) == 20
    assert {row.task_id for row in registry} == set(TASK_BY_ID)
    by_task = {row.task_id: row for row in registry}
    assert by_task["local_key"].transformation_kind == "absolute_pitch_transpose"
    assert by_task["pitch_class_set"].transformation_kind == "pitch_class_set_transpose"
    assert by_task["note_degree"].transformation_kind == "relative_label_invariant"
    assert "NoteDegree49" in by_task["note_degree"].official_behavior


def test_identity_preserves_graph_targets_and_raw_fingerprint() -> None:
    graph = build_raw_graph(_piece(), raw_only=True)
    before = graph_fingerprint(graph)
    view = transpose_raw_graph_view(graph, shift_pc=0)
    assert graph_changed_fields(graph, view) == ()
    assert graph_fingerprint(view) == before
    record = _record()
    assert transpose_record_observations(record, shift_pc=0) == record
    identity = AugmentedGraphIdentity(
        record.record_id,
        before,
        CORRECTED_PROFILE_ID,
        0,
        0,
    )
    assert len(identity.fingerprint) == 64


def test_graph_shift_changes_only_allowlisted_pitch_features() -> None:
    graph = build_raw_graph(_piece(), raw_only=True)
    view = transpose_raw_graph_view(graph, shift_pc=2)
    changed = set(graph_changed_fields(graph, view))
    assert {"note.pitch", "note.pitch_class", "note.octave"} >= (
        changed - {"note.track_relative_pitch"}
    )
    assert changed <= {
        "note.pitch",
        "note.pitch_class",
        "note.octave",
        "note.track_relative_pitch",
    }
    for edge_type in graph.edge_types:
        assert torch.equal(graph[edge_type].edge_index, view[edge_type].edge_index)
    for node_type in graph.node_types:
        assert graph[node_type].num_nodes == view[node_type].num_nodes
        assert tuple(graph[node_type].entity_id) == tuple(view[node_type].entity_id)


@pytest.mark.parametrize(("pitches", "shift", "valid"), [([0], 11, False), ([127], 1, False), ([0, 127], 0, True), ([60, 72], 6, True)])
def test_midi_range_is_record_level_fail_closed(
    pitches: list[int], shift: int, valid: bool
) -> None:
    assert valid_shift_for_midi(pitches, shift) is valid


def test_graph_range_failure_does_not_mutate_source() -> None:
    graph = build_raw_graph(_piece(), raw_only=True)
    pitch_index = tuple(graph["note"].cat_feature_names).index("pitch")
    percussion_index = tuple(graph["note"].cat_feature_names).index("is_percussion")
    first_non_drum = int(graph["note"].x_cat[:, percussion_index].eq(0).nonzero()[0])
    graph["note"].x_cat[first_non_drum, pitch_index] = 127
    before = graph["note"].x_cat.clone()
    with pytest.raises(AnalysisGNNTranspositionError, match="midi_range_violation"):
        transpose_raw_graph_view(graph, shift_pc=1)
    assert torch.equal(graph["note"].x_cat, before)


@pytest.mark.parametrize("dialect", ["an_joint", "dlc"])
def test_absolute_and_relative_target_rules_are_source_aware(dialect: str) -> None:
    shifted = transpose_record_observations(_record(dialect), shift_pc=1)
    values = {row.task_id: row.class_value for row in shifted.targets}
    assert values["local_key"] == "Db"
    assert values["root"] == "Db"
    assert values["quality"] == "augmented seventh chord"
    assert values["inversion"] == "2"
    assert values["roman_numeral"] == "V7"
    assert values["note_degree"] == "#4"
    phrase = next(row for row in shifted.targets if row.task_id == "phrase")
    assert phrase.available is False and phrase.masked is True and phrase.class_value is None
    assert shifted.component_id == "component:fixture"


def test_pitch_class_set_is_cyclically_transposed_and_sorted_after_wrap() -> None:
    assert transform_semantic_value(
        "pitch_class_set", "0,4,7", shift_pc=8, dialect="dlc"
    ) == "0,3,8"


def test_enharmonic_mapping_is_deterministic_and_oov_is_rejected() -> None:
    index = semantic_mapping_index()
    first = index[("local_key", "an_joint", "C", 6)]
    second = index[("local_key", "an_joint", "C", 6)]
    assert first == second and first.spelling_strategy.endswith("involutive_pc_rank_tritone")
    invalid = next(
        row
        for row in SEMANTIC_MAPPING_ROWS
        if row.invalid_reason == "target_oov" and row.source_task_id == "local_key"
    )
    with pytest.raises(AnalysisGNNTranspositionError, match="not closed"):
        transform_semantic_value(
            invalid.source_task_id,
            invalid.source_semantic_value,
            shift_pc=invalid.shift_pc,
            dialect=invalid.dialect,
        )


@pytest.mark.parametrize(
    ("value", "shift_pc", "expected"),
    [("C#", 2, "D#"), ("Bb", 2, "C"), ("C##", 0, "C##")],
)
def test_sharp_flat_and_double_accidental_regressions(
    value: str, shift_pc: int, expected: str
) -> None:
    assert transform_semantic_value(
        "local_key",
        value,
        shift_pc=shift_pc,
        dialect="an_joint",
    ) == expected


@pytest.mark.parametrize(
    "quality", ["augmented seventh chord", "augmented major tetrachord"]
)
def test_corrected_plus_seven_and_plus_major_seven_remain_distinct(
    quality: str,
) -> None:
    assert transform_semantic_value(
        "quality", quality, shift_pc=9, dialect="dlc"
    ) == quality


def test_all_valid_mapping_rows_round_trip_and_promised_composition_holds() -> None:
    assert all(row.round_trip_valid for row in SEMANTIC_MAPPING_ROWS if row.valid)
    composition = mapping_composition_summary()
    assert composition["promised_checked"] > 0
    assert composition["promised_failure_count"] == 0
    assert composition["spelling_diagnostic_failure_count"] > 0


def test_seeded_one_draw_is_reproducible_and_uses_only_valid_shifts() -> None:
    valid = (0, 2, 6, 11)
    first = select_record_shift("r", valid, seed=7, epoch=3)
    assert first in valid
    assert first == select_record_shift("r", valid, seed=7, epoch=3)
    assert any(
        select_record_shift("r", valid, seed=7, epoch=epoch) != first
        for epoch in range(4, 40)
    )


def test_full_orbit_expectation_does_not_inflate_independent_components() -> None:
    audit = PostTranspositionAccumulator("corrected_v2")
    audit.add_record(_record(), (0, 1))
    assert audit.variant_counts["record_variants"] == 2
    assert audit.source_components["quality"] == {"component:fixture"}
    assert sum(
        count for (task, _value), count in audit.expected_rows.items() if task == "quality"
    ) == pytest.approx(1.0)
    assert audit.expected_rows[("quality", "augmented seventh chord")] == pytest.approx(1.0)
    assert audit.full_rows[("quality", "augmented seventh chord")] == 2
    assert audit.expected_rows[("roman_numeral", "V7")] == pytest.approx(1.0)


def test_extreme_piece_can_be_constructed_without_octave_folding() -> None:
    piece = _piece()
    low = replace(piece.notes[0], pitch=0)
    high = replace(piece.notes[1], pitch=127)
    extreme = replace(piece, notes=(low, high, *piece.notes[2:]))
    assert not valid_shift_for_midi(
        (note.pitch for note in extreme.notes if not note.is_percussion), 1
    )
    assert not valid_shift_for_midi(
        (note.pitch for note in extreme.notes if not note.is_percussion), 11
    )


def test_raw_only_transposition_equivalence_collision_is_detectable() -> None:
    piece = _piece()
    shifted_piece = replace(
        piece,
        notes=tuple(
            note if note.is_percussion else replace(note, pitch=note.pitch + 2)
            for note in piece.notes
        ),
    )
    assert model_input_collision_fingerprint(piece, shift_pc=2) == (
        model_input_collision_fingerprint(shifted_piece, shift_pc=0)
    )
