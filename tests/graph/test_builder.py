from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import mido
import torch
from torch_geometric.data import Batch, HeteroData

from music_critic.adapters import (
    HookTheoryAdapterConfig,
    MidiAdapterConfig,
    convert_hooktheory_record,
    load_midi_piece,
)
from music_critic.data import (
    CanonicalPiece,
    CanonicalTrack,
    MeterEvent,
    RationalTime,
    TempoEvent,
)
from music_critic.graph import (
    FEATURE_REGISTRY_VERSION,
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    REVERSE_EDGE_TYPES,
    build_raw_graph,
    dumps_graph,
    graph_fingerprint,
    validate_raw_graph,
)


def _column(graph: HeteroData, node_type: str, name: str) -> torch.Tensor:
    names = tuple(graph[node_type].cont_feature_names)
    return graph[node_type].x_cont[:, names.index(name)]


def _write_multitrack_midi(path: Path) -> Path:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=400_000, time=480))
    midi.tracks.append(conductor)
    melody = mido.MidiTrack()
    melody.extend(
        [
            mido.Message("program_change", channel=0, program=0, time=0),
            mido.Message("note_on", channel=0, note=60, velocity=90, time=0),
            mido.Message("note_off", channel=0, note=60, velocity=0, time=960),
        ]
    )
    midi.tracks.append(melody)
    drums = mido.MidiTrack()
    drums.extend(
        [
            mido.Message("note_on", channel=9, note=36, velocity=100, time=0),
            mido.Message("note_off", channel=9, note=36, velocity=0, time=120),
        ]
    )
    midi.tracks.append(drums)
    midi.tracks.append(mido.MidiTrack())
    midi.save(path)
    return path


def _hooktheory_piece(*, include_targets: bool = True) -> CanonicalPiece:
    record = {
        "hash": "graph-hook",
        "split": "test",
        "json": {
            "endBeat": 9,
            "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
            "tempos": [
                {"beat": 1, "bpm": 120},
                {"beat": 5, "bpm": 90},
            ],
            "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
            "notes": [
                {"beat": 1, "duration": 5, "sd": "1", "octave": 0, "isRest": False},
                {"beat": 1, "duration": 1, "sd": "3", "octave": 0, "isRest": False},
            ],
            "chords": [
                {
                    "beat": 1,
                    "duration": 4,
                    "root": 1,
                    "type": 5,
                    "inversion": 0,
                    "adds": [],
                    "omits": [],
                    "alterations": [],
                    "suspensions": [],
                    "borrowed": None,
                    "isRest": False,
                    "applied": 0,
                    "alternate": "",
                    "pedal": None,
                }
            ],
        },
    }
    return convert_hooktheory_record(
        "graph-hook",
        record,
        config=HookTheoryAdapterConfig("hooktheory-graph", include_targets),
        source_path="4_merged.json",
    )


def test_graph_contract_versions_nodes_edges_and_candidates(
    canonical_piece: CanonicalPiece,
) -> None:
    graph = build_raw_graph(canonical_piece, raw_only=True)
    assert isinstance(graph, HeteroData)
    assert tuple(graph.node_types) == MANDATORY_NODE_TYPES
    assert tuple(graph.edge_types) == MANDATORY_EDGE_TYPES
    assert graph.schema_version == canonical_piece.schema_version
    assert graph.graph_schema_version == GRAPH_SCHEMA_VERSION
    assert graph.feature_registry_version == FEATURE_REGISTRY_VERSION
    assert graph.graph_builder_version == GRAPH_BUILDER_VERSION
    assert graph.raw_only is True
    assert graph["beat"].candidate_slot.all()
    assert graph["onset"].candidate_slot.all()
    validate_raw_graph(graph)


def test_counts_pickup_sustain_temporal_order_and_no_cliques(
    canonical_piece: CanonicalPiece,
) -> None:
    graph = build_raw_graph(canonical_piece)
    assert {node: graph[node].num_nodes for node in graph.node_types} == {
        "song": 1,
        "track": 2,
        "bar": 2,
        "beat": 5,
        "onset": 4,
        "note": 6,
    }
    assert _column(graph, "bar", "metric_offset_qn").tolist() == [3.0, 0.0]
    assert graph[("note", "active_at", "beat")].edge_index.tolist() == [
        [0, 0, 1, 2, 4, 4, 5],
        [0, 1, 1, 1, 2, 3, 3],
    ]
    for node_type, relation in (
        ("bar", "next_bar"),
        ("beat", "next_beat"),
        ("onset", "next_onset"),
    ):
        edge_index = graph[(node_type, relation, node_type)].edge_index
        assert torch.all(edge_index[1] == edge_index[0] + 1)
    note_relations = {
        relation for source, relation, destination in graph.edge_types
        if source == destination == "note"
    }
    assert note_relations == {"next_in_track", "previous_in_track"}


def test_reverse_relations_are_exact_transposes(canonical_piece: CanonicalPiece) -> None:
    graph = build_raw_graph(canonical_piece)
    for forward, reverse in REVERSE_EDGE_TYPES.items():
        assert torch.equal(
            graph[reverse].edge_index,
            graph[forward].edge_index.flip(0),
        )


def test_pyg_batch_offsets_never_create_cross_graph_edges(
    canonical_piece: CanonicalPiece,
) -> None:
    graph = build_raw_graph(canonical_piece)
    batch = Batch.from_data_list([graph, graph])
    for source_type, relation, destination_type in batch.edge_types:
        edge_index = batch[(source_type, relation, destination_type)].edge_index
        assert torch.equal(
            batch[source_type].batch[edge_index[0]],
            batch[destination_type].batch[edge_index[1]],
        )


def test_meter_and_tempo_changes_are_raw_features(
    canonical_piece: CanonicalPiece,
) -> None:
    piece = _hooktheory_piece()
    graph = build_raw_graph(piece)
    assert len(piece.tempo_events) == 2
    assert _column(graph, "beat", "tempo_us_per_qn").tolist()[:5] == [
        500000.0,
        500000.0,
        500000.0,
        500000.0,
        666667.0,
    ]

    meter_change = MeterEvent(
        meter_event_id="meter:change",
        onset_qn=RationalTime(1),
        numerator=3,
        denominator=4,
        provenance_id=None,
    )
    first_bar, second_bar = canonical_piece.bars
    changed_bar = replace(
        second_bar,
        duration_qn=RationalTime(3),
        meter_event_id=meter_change.meter_event_id,
    )
    first_beat, *following_beats = canonical_piece.beats
    changed_beats = (
        first_beat,
        *(
            replace(beat, meter_event_id=meter_change.meter_event_id)
            for beat in following_beats[:3]
        ),
    )
    changed_piece = replace(
        canonical_piece,
        duration_qn=RationalTime(4),
        bars=(first_bar, changed_bar),
        beats=changed_beats,
        meter_events=(*canonical_piece.meter_events, meter_change),
        tempo_events=(
            canonical_piece.tempo_events[0],
            TempoEvent(
                tempo_event_id="tempo:change",
                onset_qn=RationalTime(2),
                microseconds_per_quarter=400_000,
                provenance_id=None,
            ),
        ),
    )
    changed_graph = build_raw_graph(changed_piece)
    numerator_column = tuple(changed_graph["bar"].cat_feature_names).index(
        "meter_numerator"
    )
    assert changed_graph["bar"].x_cat[:, numerator_column].tolist() == [4, 3]
    assert _column(changed_graph, "beat", "tempo_us_per_qn").tolist() == [
        500000.0,
        500000.0,
        400000.0,
        400000.0,
    ]
    assert changed_graph[("beat", "next_beat", "beat")].edge_index.shape[1] == 3


def test_multiple_tracks_drums_and_empty_tracks(canonical_piece: CanonicalPiece) -> None:
    empty = CanonicalTrack(
        track_id="track:empty",
        source_track_index=2,
        name="Empty",
        instrument_name=None,
        program=None,
        channel=None,
        is_percussion=False,
        provenance_id=None,
    )
    graph = build_raw_graph(replace(canonical_piece, tracks=(*canonical_piece.tracks, empty)))
    assert graph["track"].num_nodes == 3
    assert graph[("song", "contains_track", "track")].edge_index.shape[1] == 3
    assert graph[("track", "contains_note", "note")].edge_index[0].max().item() == 1
    percussion_column = tuple(graph["track"].cat_feature_names).index("is_percussion")
    assert graph["track"].x_cat[:, percussion_column].tolist() == [0, 1, 0]
    assert not graph["track"].x_cont_available[2].all()


def test_deterministic_ordering_and_json_serialization(
    canonical_piece: CanonicalPiece,
    tmp_path: Path,
) -> None:
    first = build_raw_graph(canonical_piece)
    second = build_raw_graph(canonical_piece)
    assert dumps_graph(first) == dumps_graph(second)
    assert graph_fingerprint(first) == graph_fingerprint(second)
    payload = json.loads(dumps_graph(first))
    assert [node["node_type"] for node in payload["nodes"]] == list(
        MANDATORY_NODE_TYPES
    )
    assert [(e["source_type"], e["relation"], e["destination_type"]) for e in payload["edges"]] == list(MANDATORY_EDGE_TYPES)


def test_generic_midi_and_hooktheory_share_model_facing_schema(tmp_path: Path) -> None:
    midi_piece = load_midi_piece(
        _write_multitrack_midi(tmp_path / "multi.mid"),
        config=MidiAdapterConfig("generic-midi-graph"),
    )
    hook_piece = _hooktheory_piece()
    midi_graph = build_raw_graph(midi_piece, raw_only=True)
    hook_graph = build_raw_graph(hook_piece, raw_only=True)
    assert tuple(midi_graph.node_types) == tuple(hook_graph.node_types)
    assert tuple(midi_graph.edge_types) == tuple(hook_graph.edge_types)
    for node_type in MANDATORY_NODE_TYPES:
        assert midi_graph[node_type].cat_feature_names == hook_graph[node_type].cat_feature_names
        assert midi_graph[node_type].cont_feature_names == hook_graph[node_type].cont_feature_names
        assert midi_graph[node_type].x_cat.shape[1:] == hook_graph[node_type].x_cat.shape[1:]
        assert midi_graph[node_type].x_cont.shape[1:] == hook_graph[node_type].x_cont.shape[1:]
    assert any(track.is_percussion for track in midi_piece.tracks)
    assert any(
        not any(note.track_id == track.track_id for note in midi_piece.notes)
        for track in midi_piece.tracks
    )


def test_hooktheory_adapter_target_hiding_is_graph_identical() -> None:
    visible = _hooktheory_piece(include_targets=True)
    hidden = _hooktheory_piece(include_targets=False)
    assert visible.targets
    assert not hidden.targets
    assert dumps_graph(build_raw_graph(visible)) == dumps_graph(build_raw_graph(hidden))


def _dense_piece(canonical_piece: CanonicalPiece, count: int) -> CanonicalPiece:
    source = canonical_piece.notes[0]
    notes = tuple(
        replace(
            source,
            note_id=f"note:dense-{index:04d}",
            pitch=48 + index % 36,
            onset_qn=RationalTime(0),
            duration_qn=RationalTime(1),
        )
        for index in range(count)
    )
    return replace(
        canonical_piece,
        tracks=(canonical_piece.tracks[0],),
        notes=notes,
        targets=(),
        annotations=(),
    )


def _total_edges(graph: HeteroData) -> int:
    return sum(graph[edge_type].edge_index.shape[1] for edge_type in graph.edge_types)


def test_graph_growth_is_linear_without_simultaneous_note_cliques(
    canonical_piece: CanonicalPiece,
) -> None:
    small = build_raw_graph(_dense_piece(canonical_piece, 32))
    large = build_raw_graph(_dense_piece(canonical_piece, 256))
    assert small["onset"].num_nodes == large["onset"].num_nodes == 1
    small_edges = _total_edges(small)
    large_edges = _total_edges(large)
    assert large_edges < small_edges * 9
    assert large_edges < 12 * 256 + 100
