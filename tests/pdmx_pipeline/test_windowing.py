from __future__ import annotations

from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

from music_critic.data import (
    CanonicalBar,
    CanonicalBeat,
    KeySignatureEvent,
    MeterEvent,
    RationalTime as Q,
    TempoEvent,
    dumps_piece,
    load_piece,
    validate_or_raise,
)
from music_critic.graph import build_raw_graph, graph_fingerprint, model_input_fingerprint
from music_critic.pdmx_pipeline import windowing
from music_critic.pdmx_pipeline.windowing import WindowConfig, WindowingError, iter_bar_windows


FIXTURE = Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json"


def raw_piece():
    source = load_piece(FIXTURE)
    return replace(
        source, split="train", annotations=(), targets=(),
        provenance=tuple(record for record in source.provenance if record.provenance_id != "prov:theory"),
    )


def meter_piece():
    """Pickup + complete 4/4 + complete 3/4 + incomplete final 3/4."""
    source = raw_piece()
    added_bars = (
        CanonicalBar("bar:002", 2, Q(5), Q(3), "meter:three", Q(0), False, False, "2", "prov:source"),
        CanonicalBar("bar:003", 3, Q(8), Q(1), "meter:three", Q(0), False, True, "3", "prov:source"),
    )
    added_beats = tuple(
        CanonicalBeat(f"beat:added-{index}", "bar:002" if index < 3 else "bar:003",
                      "meter:three", index if index < 3 else 0, Q(5 + index), Q(1),
                      Q(index if index < 3 else 0), index in (0, 3), None, "prov:source")
        for index in range(4)
    )
    template = source.notes[0]
    added_notes = (
        replace(template, note_id="note:triplet", onset_qn=Q(17, 3), duration_qn=Q(1, 3)),
        replace(template, note_id="note:last-cross", onset_qn=Q(23, 3), duration_qn=Q(1)),
        replace(template, note_id="note:terminal", onset_qn=Q(9), duration_qn=Q(0), is_grace=True),
    )
    source = replace(
        source, duration_qn=Q(9), bars=source.bars + added_bars,
        beats=source.beats + added_beats, notes=source.notes + added_notes,
        meter_events=source.meter_events + (MeterEvent("meter:three", Q(5), 3, 4, "prov:source"),),
        tempo_events=(TempoEvent("tempo:000", Q(0), 500_000, "prov:source"),
                      TempoEvent("tempo:triplet", Q(2, 3), 600_000, "prov:source"),
                      TempoEvent("tempo:boundary", Q(5), 400_000, "prov:source")),
        key_signature_events=(KeySignatureEvent("keysig:zero", Q(0), 0, "major", None, "prov:source"),),
    )
    validate_or_raise(source)
    return source


def test_clipping_exact_offset_identity_and_provenance():
    source = raw_piece()
    source = replace(source, notes=(replace(source.notes[0], onset_qn=Q(1, 3), duration_qn=Q(5, 3)), *source.notes[1:]))
    windows = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1)))
    assert len(windows) == 2
    assert [item.global_offset_qn for item in windows] == [Q(0), Q(1)]
    assert [item.source_bar_ids for item in windows] == [("bar:000",), ("bar:001",)]
    assert [(item.source_bar_start, item.source_bar_stop) for item in windows] == [(0, 1), (1, 2)]
    assert all(item.source_piece_id == source.piece_id for item in windows)
    assert all(item.piece.source_group_id == source.source_group_id for item in windows)
    assert all(item.piece.split == "train" for item in windows)
    first = windows[0].piece.notes[0]
    assert (first.onset_qn, first.duration_qn) == (Q(1, 3), Q(2, 3))
    assert first.note_id != source.notes[0].note_id
    second = next(note for note, original in zip(windows[1].piece.notes, windows[1].original_note_ids)
                  if original == source.notes[0].note_id)
    assert (second.onset_qn, second.duration_qn) == (Q(0), Q(1))
    assert second.note_id not in (source.notes[0].note_id, first.note_id)
    assert second.source_onset_ticks == source.notes[0].source_onset_ticks
    assert [item.clipped_note_count for item in windows] == [1, 1]
    provenance = next(record for record in windows[1].piece.provenance if record.provenance_id == second.provenance_id)
    assert provenance.parents == (source.notes[0].provenance_id,)
    assert dict(provenance.details)["original_note_id"] == source.notes[0].note_id
    assert dict(provenance.details)["clipped_left"] is True
    assert dict(provenance.details)["clipped_right"] is False
    assert source.notes[1].note_id in {note.note_id for note in windows[1].piece.notes}
    assert [dumps_piece(item.piece) for item in windows] == [
        dumps_piece(item.piece) for item in iter_bar_windows(source, WindowConfig(bar_count=1, stride=1))
    ]


def test_tempo_meter_context_pickup_and_incomplete_bar():
    source = meter_piece()
    results = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1)))
    assert len(results) == 4
    first, second, third, fourth = (result.piece for result in results)
    assert first.bars[0].is_pickup and first.bars[0].metric_offset_qn == Q(3)
    assert first.tempo_events[1].onset_qn == Q(2, 3)
    assert [(event.onset_qn, event.microseconds_per_quarter) for event in second.tempo_events] == [(Q(0), 600_000)]
    assert second.tempo_events[0].tempo_event_id != "tempo:triplet"
    assert second.meter_events[0].meter_event_id != source.meter_events[0].meter_event_id
    assert second.bars[0].meter_event_id == second.meter_events[0].meter_event_id
    assert all(beat.meter_event_id == second.bars[0].meter_event_id for beat in second.beats)
    assert third.meter_events[0].meter_event_id == "meter:three"
    assert third.meter_events[0].numerator == 3
    assert third.notes[0].onset_qn == Q(2, 3)
    assert third.notes[0].duration_qn == Q(1, 3)
    assert fourth.bars[0].is_incomplete and not fourth.bars[0].is_pickup
    assert fourth.bars[0].index == 0 and fourth.duration_qn == Q(1)
    assert fourth.notes[-1].note_id == "note:terminal"
    assert fourth.notes[-1].onset_qn == fourth.duration_qn
    assert second.key_signature_events[0].onset_qn == Q(0)
    for piece in (first, second, third, fourth):
        validate_or_raise(piece)


def test_next_window_boundary_events_do_not_change_previous_raw_graph():
    source = meter_piece()
    source = replace(source, key_signature_events=source.key_signature_events + (
        KeySignatureEvent("keysig:boundary", Q(5), 2, "major", None, "prov:source"),
    ))
    changed = replace(source, tempo_events=source.tempo_events[:-1] + (
        replace(source.tempo_events[-1], microseconds_per_quarter=300_000),
    ))
    original_windows = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1)))
    changed_windows = list(iter_bar_windows(changed, WindowConfig(bar_count=1, stride=1)))
    previous = original_windows[1].piece
    following = original_windows[2].piece
    assert all(event.onset_qn < previous.duration_qn for events in (
        previous.tempo_events, previous.meter_events, previous.key_signature_events,
    ) for event in events)
    assert graph_fingerprint(build_raw_graph(previous)) == graph_fingerprint(build_raw_graph(changed_windows[1].piece))
    assert following.tempo_events[0].microseconds_per_quarter == 400_000
    assert changed_windows[2].piece.tempo_events[0].microseconds_per_quarter == 300_000
    assert graph_fingerprint(build_raw_graph(following)) != graph_fingerprint(build_raw_graph(changed_windows[2].piece))
    assert following.meter_events[0].meter_event_id == "meter:three"
    assert following.key_signature_events[0].key_signature_event_id == "keysig:boundary"
    projection = next(record for record in previous.provenance
                      if dict(record.details).get("operation") == "bar_window")
    assert dict(projection.details)["event_end_boundary"] == "open_window_boundary"


def test_true_source_terminal_events_preserve_terminal_grace_and_full_piece_graph():
    source = meter_piece()
    source = replace(source,
        tempo_events=source.tempo_events + (TempoEvent("tempo:terminal", Q(9), 700_000, "prov:source"),),
        meter_events=source.meter_events + (MeterEvent("meter:terminal", Q(9), 6, 8, "prov:source"),),
        key_signature_events=source.key_signature_events + (
            KeySignatureEvent("keysig:terminal", Q(9), 1, "major", None, "prov:source"),
        ),
    )
    validate_or_raise(source)
    [result] = iter_bar_windows(source, WindowConfig(bar_count=8, stride=8))
    assert result.piece.tempo_events == source.tempo_events
    assert result.piece.meter_events == source.meter_events
    assert result.piece.key_signature_events == source.key_signature_events
    assert result.piece.notes[-1].is_grace and result.piece.notes[-1].onset_qn == Q(9)
    assert model_input_fingerprint(build_raw_graph(result.piece)) == model_input_fingerprint(build_raw_graph(source))
    last = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1)))[-1].piece
    assert last.tempo_events[-1].onset_qn == last.duration_qn == Q(1)
    assert last.meter_events[-1].onset_qn == last.duration_qn
    assert last.key_signature_events[-1].onset_qn == last.duration_qn
    projection = next(record for record in last.provenance
                      if dict(record.details).get("operation") == "bar_window")
    assert dict(projection.details)["event_end_boundary"] == "closed_source_terminal"


def test_boundary_grace_note_belongs_only_to_following_nonoverlap_window():
    source = raw_piece()
    grace = replace(source.notes[0], note_id="note:boundary-grace", onset_qn=Q(1), duration_qn=Q(0), is_grace=True)
    source = replace(source, notes=(source.notes[0], grace, *source.notes[1:]))
    # Equal-onset canonical ordering puts this pitch before the existing 72.
    validate_or_raise(source)
    results = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1)))
    assert "note:boundary-grace" not in results[0].original_note_ids
    assert "note:boundary-grace" in results[1].original_note_ids


def test_raw_graph_without_optional_metadata_or_theory():
    source = meter_piece()
    source = replace(source, notes=tuple(replace(note, velocity=None, channel=None,
        program=None, staff=None, voice=None, spelling_step=None, spelling_alter=None,
        articulations=None, dynamic=None) for note in source.notes))
    for result in iter_bar_windows(source, WindowConfig(bar_count=2, stride=2)):
        assert not result.piece.annotations and not result.piece.targets
        assert all(note.voice is None for note in result.piece.notes)
        graph = build_raw_graph(result.piece)
        assert ("note", "next_in_track", "note") in graph.edge_types
        assert not any("voice" in edge for relation in graph.edge_types for edge in relation)
        altered = replace(result.piece, source_group_id="different-group", split="validation")
        assert model_input_fingerprint(graph) == model_input_fingerprint(build_raw_graph(altered))


@pytest.mark.parametrize("split", ["test", "TEST", None, "training"])
def test_split_guard_precedes_validation(monkeypatch, split):
    source = replace(raw_piece(), split=split)
    def forbidden(_piece):
        pytest.fail("canonical validation was reached before split guard")
    monkeypatch.setattr(windowing, "validate_or_raise", forbidden)
    with pytest.raises(WindowingError, match="pdmx.split_not_allowed"):
        next(iter_bar_windows(source))


def test_validation_overlap_requires_explicit_piece_evaluation_acknowledgement():
    source = replace(meter_piece(), split="validation")
    with pytest.raises(WindowingError, match="validation_overlap_forbidden"):
        list(iter_bar_windows(source, WindowConfig(bar_count=2, stride=1)))
    results = list(iter_bar_windows(source, WindowConfig(bar_count=2, stride=1, allow_validation_overlap=True)))
    assert len(results) == 4
    assert {item.evaluation_unit for item in results} == {"source_piece"}
    assert len({item.source_piece_id for item in results}) == 1


def test_invalid_empty_resource_and_annotation_rejections():
    source = raw_piece()
    with pytest.raises(WindowingError) as error:
        next(iter_bar_windows(replace(source, duration_qn=Q(-1))))
    assert error.value.category == "pdmx.canonical_invalid"
    with pytest.raises(WindowingError, match="pdmx.window_not_raw"):
        next(iter_bar_windows(replace(load_piece(FIXTURE), split="train")))
    empty = list(iter_bar_windows(replace(source, notes=()), WindowConfig(bar_count=1, stride=1)))
    assert len(empty) == 2 and all(item.reason == "pdmx.window_empty" and item.piece is None for item in empty)
    with pytest.raises(WindowingError, match="pdmx.resource_windows"):
        next(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1, max_windows_per_piece=1)))
    limited = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=1, max_notes_per_window=1)))
    assert limited[0].piece is not None
    assert limited[1].piece is None and limited[1].reason == "pdmx.resource_window_notes"


def test_projection_programming_exception_is_not_source_quarantine(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("implementation defect")
    monkeypatch.setattr(windowing, "_make_window", broken)
    with pytest.raises(RuntimeError, match="implementation defect"):
        next(iter_bar_windows(raw_piece()))


def test_zero_duration_score_without_bars_has_explicit_result():
    source = replace(raw_piece(), duration_qn=Q(0), notes=(), bars=(), beats=())
    [result] = iter_bar_windows(source)
    assert result.piece is None and result.reason == "pdmx.window_no_bars"
    assert result.source_bar_ids == () and result.global_offset_qn == Q(0)


def test_clipped_note_does_not_continue_into_nonoverlapping_gap():
    source = meter_piece()
    results = list(iter_bar_windows(source, WindowConfig(bar_count=1, stride=2)))
    assert [result.source_bar_start for result in results] == [0, 2]
    assert "note:melody-000" not in results[1].original_note_ids
    assert "note:triplet" in results[1].original_note_ids


@pytest.mark.parametrize("kwargs", [{"bar_count": 0}, {"stride": True},
    {"max_notes_per_window": -1}, {"max_windows_per_piece": 0}, {"allow_validation_overlap": 1}])
def test_config_rejects_invalid_budget(kwargs):
    with pytest.raises(ValueError):
        WindowConfig(**kwargs)


def test_generated_sequence_only_projects_requested_windows(monkeypatch):
    source = raw_piece()
    bar_template = source.bars[1]
    beat_template = source.beats[1]
    note_template = source.notes[0]
    count = 64
    source = replace(source, duration_qn=Q(4 * count),
        bars=tuple(replace(bar_template, bar_id=f"bar:generated-{index}", index=index,
                           start_qn=Q(4 * index)) for index in range(count)),
        beats=tuple(replace(beat_template, beat_id=f"beat:generated-{index}",
                            bar_id=f"bar:generated-{index // 4}", start_qn=Q(index),
                            index_in_bar=index % 4, position_in_bar_qn=Q(index % 4),
                            is_downbeat=index % 4 == 0) for index in range(4 * count)),
        notes=tuple(replace(note_template, note_id=f"note:generated-{index}",
                            onset_qn=Q(4 * index), duration_qn=Q(1)) for index in range(count)))
    original = windowing._make_window
    calls = []
    def counted(*args, **kwargs):
        calls.append((args[1], len(args[3])))
        return original(*args, **kwargs)
    monkeypatch.setattr(windowing, "_make_window", counted)
    stream = iter_bar_windows(source, WindowConfig(bar_count=2, stride=2))
    assert calls == []
    assert len(list(islice(stream, 3))) == 3
    assert calls == [(0, 2), (2, 2), (4, 2)]
