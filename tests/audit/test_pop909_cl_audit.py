from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import mido

from music_critic.adapters import MidiAdapterConfig, load_midi_piece
from music_critic.graph import build_raw_graph, graph_fingerprint
from scripts.audit_pop909_cl import (
    Pop909ClAuditError,
    analyze_meter_boundaries,
    build_report,
    chord_span_diagnostics,
    chord_normalization,
    discover_pop909_cl,
    dumps_report,
    ensure_output_outside_root,
    extract_chord_blocks,
    inspect_instrument_contract,
    main,
    project_score_midi_bytes,
    propose_lineage_group_id,
    propose_source_group_id,
)


def _conductor(*, song_172_meters: bool = False) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.extend([
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("key_signature", key="C", time=0),
    ])
    if song_172_meters:
        track.extend([
            mido.MetaMessage("time_signature", numerator=6, denominator=8, time=85_080),
            mido.MetaMessage("time_signature", numerator=4, denominator=4, time=16_320),
            mido.MetaMessage("end_of_track", time=0),
        ])
    else:
        track.append(mido.MetaMessage("end_of_track", time=1_920))
    return track


def _score_track(*, name: str = "piano") -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.extend([
        mido.MetaMessage("track_name", name=name, time=0),
        mido.Message("program_change", channel=0, program=0, time=0),
        mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
        mido.Message("note_off", channel=0, note=60, velocity=0, time=480),
        mido.Message("note_on", channel=0, note=64, velocity=80, time=0),
        mido.Message("note_off", channel=0, note=64, velocity=0, time=480),
        mido.MetaMessage("end_of_track", time=960),
    ])
    return track


def _chord_track(variant: str = "normal") -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.extend([
        mido.MetaMessage("track_name", name="chords", time=0),
        mido.Message("program_change", channel=1, program=0, time=0),
    ])
    if variant == "deleted":
        track.append(mido.MetaMessage("end_of_track", time=1_920))
        return track
    first = (60, 64, 67) if variant == "normal" else (62, 65, 69)
    for index, pitch in enumerate(first):
        track.append(mido.Message("note_on", channel=1, note=pitch, velocity=70, time=0))
    for index, pitch in enumerate(first):
        track.append(mido.Message("note_off", channel=1, note=pitch, velocity=0, time=480 if index == 0 else 0))
    # A two-pitch shape is deliberately unsupported, with an implicit N gap.
    track.extend([
        mido.Message("note_on", channel=1, note=60, velocity=70, time=480),
        mido.Message("note_on", channel=1, note=61, velocity=70, time=0),
        mido.Message("note_off", channel=1, note=60, velocity=0, time=480),
        mido.Message("note_off", channel=1, note=61, velocity=0, time=0),
        mido.MetaMessage("end_of_track", time=480),
    ])
    return track


def _write_cl_midi(
    path: Path,
    *,
    chord_variant: str = "normal",
    chord_track_count: int = 1,
    include_score: bool = True,
    score_name: str = "piano",
) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(_conductor())
    if include_score:
        midi.tracks.append(_score_track(name=score_name))
    for _ in range(chord_track_count):
        midi.tracks.append(_chord_track(chord_variant))
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def _midi_from_bytes(payload: bytes) -> mido.MidiFile:
    return mido.MidiFile(file=BytesIO(payload))


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_nested_discovery_normalizes_whitespace_and_excludes_appledouble(tmp_path: Path) -> None:
    root = tmp_path / "install"
    corpus = root / "POP909_processed" / "POP909_processed"
    _write_cl_midi(corpus / "043 .mid")
    _write_cl_midi(corpus / "001.mid")
    _write_cl_midi(corpus / "001 .mid")
    noise = root / "POP909_processed" / "__MACOSX" / "._001.mid"
    noise.parent.mkdir(parents=True)
    noise.write_bytes(b"filesystem metadata")

    first = discover_pop909_cl(root)
    second = discover_pop909_cl(root)
    assert first == second
    assert first.corpus_root == corpus.resolve()
    assert [asset.song_id for asset in first.assets] == ["001", "043"]
    assert first.assets[1].relative_to_corpus == "043 .mid"
    assert first.duplicate_song_ids == ((
        "001",
        (
            "POP909_processed/POP909_processed/001 .mid",
            "POP909_processed/POP909_processed/001.mid",
        ),
    ),)
    assert first.noise_files == (noise.resolve(),)
    assert propose_source_group_id("043") == "pop909-cl:043"
    assert propose_lineage_group_id("043") == "pop909-lineage:043"


def test_channel_contract_rejects_missing_and_ambiguous_instruments(tmp_path: Path) -> None:
    missing_chord = tmp_path / "missing-chord.mid"
    _write_cl_midi(missing_chord, chord_track_count=0)
    contract = inspect_instrument_contract(mido.MidiFile(missing_chord))
    assert [row["category"] for row in contract.failures] == ["missing_chord_instrument"]
    missing_evidence = extract_chord_blocks(
        mido.MidiFile(missing_chord),
        contract,
        source_path="367.mid",
        source_sha256="c" * 64,
        score_duration_tick=1_920,
    )
    assert missing_evidence["status"] == "unavailable"
    assert all(value is False for value in missing_evidence["task_availability"].values())

    ambiguous_chord = tmp_path / "ambiguous-chord.mid"
    _write_cl_midi(ambiguous_chord, chord_track_count=2)
    contract = inspect_instrument_contract(mido.MidiFile(ambiguous_chord))
    assert contract.chord_track_indices == (2, 3)
    assert [row["category"] for row in contract.failures] == ["ambiguous_chord_instrument"]

    missing_score = tmp_path / "missing-score.mid"
    _write_cl_midi(missing_score, include_score=False)
    contract = inspect_instrument_contract(mido.MidiFile(missing_score))
    assert [row["category"] for row in contract.failures] == ["missing_score_instrument"]

    ambiguous_score = tmp_path / "ambiguous-score.mid"
    _write_cl_midi(ambiguous_score)
    midi = mido.MidiFile(ambiguous_score)
    midi.tracks.append(_score_track(name="second score"))
    midi.save(ambiguous_score)
    contract = inspect_instrument_contract(mido.MidiFile(ambiguous_score))
    assert contract.score_track_indices == (1, 3)
    assert [row["category"] for row in contract.failures] == ["ambiguous_score_instrument"]

    excluded_meta = tmp_path / "excluded-meta.mid"
    _write_cl_midi(excluded_meta)
    midi = mido.MidiFile(excluded_meta)
    midi.tracks[2].insert(2, mido.MetaMessage("key_signature", key="G", time=0))
    midi.save(excluded_meta)
    contract = inspect_instrument_contract(mido.MidiFile(excluded_meta))
    assert [row["category"] for row in contract.failures] == [
        "required_global_meta_on_excluded_instrument"
    ]
    try:
        project_score_midi_bytes(mido.MidiFile(excluded_meta), contract)
    except Pop909ClAuditError as exc:
        assert "global meta-events" in str(exc)
    else:
        raise AssertionError("excluded-instrument global metadata was silently dropped")


def test_chord_blocks_preserve_ticks_multisets_gaps_and_unsupported_shapes(tmp_path: Path) -> None:
    path = tmp_path / "001.mid"
    _write_cl_midi(path)
    midi = mido.MidiFile(path)
    contract = inspect_instrument_contract(midi)
    evidence = extract_chord_blocks(
        midi,
        contract,
        source_path="001.mid",
        source_sha256="a" * 64,
        score_duration_tick=1_920,
    )
    assert evidence["status"] == "available"
    assert evidence["block_count"] == 2
    assert evidence["blocks"][0]["onset_tick"] == 0
    assert evidence["blocks"][0]["end_tick"] == 480
    assert evidence["blocks"][0]["midi_pitch_multiset"] == [60, 64, 67]
    assert evidence["blocks"][0]["pitch_class_set"] == [0, 4, 7]
    assert evidence["blocks"][0]["source_track_index"] == 2
    assert evidence["blocks"][0]["source_sha256"] == "a" * 64
    assert evidence["blocks"][0]["provenance"] == {
        "provenance_id": "pop909_cl.raw_chord_block",
        "source": "human",
        "details": ["human_corrected", "expert_reviewed"],
        "confidence": None,
    }
    assert evidence["blocks"][0]["normalization"]["provenance"]["source"] == "derived"
    assert evidence["blocks"][0]["target_fields"]["root"]["provenance"] == (
        "pop909_cl.upstream_normalized_target"
    )
    assert evidence["blocks"][0]["target_fields"]["root"]["available"] is True
    assert evidence["blocks"][0]["target_fields"]["quality"]["available"] is True
    assert evidence["blocks"][0]["target_fields"]["inversion"]["available"] is True
    assert evidence["blocks"][1]["normalization"]["status"] == "unsupported"
    assert evidence["blocks"][1]["target_fields"]["boundary"]["available"] is True
    assert evidence["blocks"][1]["target_fields"]["bass"]["available"] is True
    for task in ("root", "quality", "inversion"):
        assert evidence["blocks"][1]["target_fields"][task]["available"] is False
    assert [
        (gap["kind"], gap["start_tick"], gap["end_tick"])
        for gap in evidence["implicit_n_gaps"]
    ] == [("internal_no_chord", 480, 960)]
    gap = evidence["implicit_n_gaps"][0]
    assert gap["target_field"]["available"] is True
    assert gap["target_field"]["value"] == "N"
    assert gap["target_field"]["source"] == "derived"
    assert gap["target_field"]["provenance"] == "pop909_cl.upstream_implicit_n"
    assert evidence["trailing_unannotated_span"]["start_tick"] == 1_440
    assert evidence["trailing_unannotated_span"]["end_tick"] == 1_920
    assert evidence["trailing_unannotated_span"]["target_field"] == {
        "available": False,
        "value": None,
        "source": None,
        "provenance": None,
    }
    ambiguous = chord_normalization({0, 3, 6, 9}, bass_pc=0)
    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["target_fields"]["root"]["available"] is False
    assert ambiguous["target_fields"]["inversion"]["available"] is False
    assert ambiguous["target_fields"]["quality"]["available"] is True
    assert ambiguous["target_fields"]["quality"]["value"] == "o7"
    assert chord_normalization({0, 1}, bass_pc=0)["status"] == "unsupported"
    gaps, trailing, overlaps = chord_span_diagnostics(
        [
            {"onset_tick": 0, "end_tick": 1_000},
            {"onset_tick": 500, "end_tick": 600},
            {"onset_tick": 700, "end_tick": 800},
            {"onset_tick": 1_200, "end_tick": 1_400},
        ],
        1_600,
    )
    assert overlaps == 2
    assert [(row["start_tick"], row["end_tick"]) for row in gaps] == [
        (1_000, 1_200),
    ]
    assert trailing is not None
    assert (trailing["start_tick"], trailing["end_tick"]) == (1_400, 1_600)


def test_pairing_anomalies_preserve_exact_source_evidence_and_mark_coverage() -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.extend([_conductor(), _score_track()])
    track = mido.MidiTrack([
        mido.MetaMessage("track_name", name="chords", time=0),
        mido.Message("program_change", channel=1, program=0, time=0),
        mido.Message("note_off", channel=1, note=55, velocity=9, time=5),
        mido.Message("note_on", channel=1, note=60, velocity=77, time=5),
        mido.Message("note_on", channel=1, note=64, velocity=66, time=0),
        mido.Message("note_off", channel=1, note=64, velocity=4, time=100),
        mido.MetaMessage("end_of_track", time=90),
    ])
    midi.tracks.append(track)
    evidence = extract_chord_blocks(
        midi,
        inspect_instrument_contract(midi),
        source_path="POP909_processed/999.mid",
        source_sha256="b" * 64,
        score_duration_tick=200,
    )
    events = evidence["pairing_diagnostics"]["events"]
    assert events[0] == {
        "anomaly_id": "unmatched_note_off:0",
        "category": "unmatched_note_off",
        "tick": 5,
        "pitch": 55,
        "velocity": 9,
        "channel": 1,
        "message_type": "note_off",
        "ordinal": 0,
        "source_track_index": 2,
        "source_path": "POP909_processed/999.mid",
        "source_sha256": "b" * 64,
        "affected_block_onsets": [],
        "affected_span_ids": ["implicit_n:0"],
        "affected_interval": {
            "start_tick": 5,
            "end_tick": 5,
            "basis": "unmatched_point_event",
        },
    }
    assert events[1]["category"] == "dangling_note_on"
    assert events[1]["tick"] == 10
    assert events[1]["pitch"] == 60
    assert events[1]["velocity"] == 77
    assert events[1]["ordinal"] == 1
    assert events[1]["affected_block_onsets"] == [10]
    assert events[1]["affected_interval"] == {
        "start_tick": 10,
        "end_tick": 200,
        "basis": "open_note_to_score_end",
    }
    assert evidence["blocks"][0]["pairing_anomaly_ids"] == ["dangling_note_on:1"]
    assert evidence["implicit_n_gaps"][0]["pairing_anomaly_ids"] == [
        "unmatched_note_off:0"
    ]


def test_chord_mutations_do_not_change_score_projection_or_raw_graph(tmp_path: Path) -> None:
    projections: list[bytes] = []
    pieces = []
    block_signatures = []
    unsafe_note_counts = []
    projected_path = tmp_path / "projected.mid"
    for variant in ("normal", "replacement", "deleted"):
        source = tmp_path / f"source-{variant}.mid"
        _write_cl_midi(source, chord_variant=variant)
        midi = mido.MidiFile(source)
        contract = inspect_instrument_contract(midi)
        projection = project_score_midi_bytes(midi, contract)
        projections.append(projection)
        projected_path.write_bytes(projection)
        pieces.append(load_midi_piece(
            projected_path,
            config=MidiAdapterConfig(dataset_name="pop909_cl", source_group_id="pop909-cl:001"),
        ))
        chord = extract_chord_blocks(
            midi,
            contract,
            source_path=source.name,
            source_sha256=sha256(source.read_bytes()).hexdigest(),
            score_duration_tick=1_920,
        )
        block_signatures.append([
            (block["onset_tick"], block["midi_pitch_multiset"])
            for block in chord["blocks"]
        ])
        unsafe = load_midi_piece(
            source,
            config=MidiAdapterConfig(dataset_name="unsafe", source_group_id="pop909-cl:001"),
        )
        unsafe_note_counts.append(len(unsafe.notes))

    assert projections[0] == projections[1] == projections[2]
    projected_midi = _midi_from_bytes(projections[0])
    projected_meta = {message.type for track in projected_midi.tracks for message in track}
    assert {"set_tempo", "time_signature", "key_signature"} <= projected_meta
    assert pieces[0].tracks == pieces[1].tracks == pieces[2].tracks
    assert pieces[0].notes == pieces[1].notes == pieces[2].notes
    assert len(pieces[0].notes) == 2
    assert len(set(graph_fingerprint(build_raw_graph(piece)) for piece in pieces)) == 1
    assert block_signatures[0] != block_signatures[1]
    assert block_signatures[2] == []
    assert unsafe_note_counts[:2] == [7, 7]
    assert unsafe_note_counts[2] == 2


def test_song_172_meter_evidence_is_exact_and_explains_rejection() -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.extend([_conductor(song_172_meters=True), _score_track(), _chord_track()])
    rows = analyze_meter_boundaries(midi)
    assert rows[1] == {
        "tick": 85_080,
        "source_track_index": 0,
        "message_index": 3,
        "previous_meter": "4/4",
        "new_meter": "6/8",
        "active_bar_length_ticks": 1_920,
        "previous_bar_boundary_tick": 84_480,
        "next_bar_boundary_tick": 86_400,
        "offset_inside_bar_ticks": 600,
        "on_expected_boundary": False,
    }
    assert rows[2]["tick"] == 101_400
    assert rows[2]["previous_meter"] == "6/8"
    assert rows[2]["offset_inside_bar_ticks"] == 480
    assert rows[2]["next_bar_boundary_tick"] == 102_360


def test_report_is_deterministic_and_never_writes_under_source_root(tmp_path: Path) -> None:
    root = tmp_path / "install"
    corpus = root / "POP909_processed" / "POP909_processed"
    _write_cl_midi(corpus / "001.mid")
    _write_cl_midi(corpus / "043 .mid", score_name="MIDI 01")
    before = _tree_hashes(root)
    first = build_report(root)
    second = build_report(root)
    assert dumps_report(first) == dumps_report(second)
    assert first["corpus_identity"]["corpus_midi_file_count"] == 2
    assert first["score_only_crosswalk"]["converted"] == 2
    assert first["score_only_crosswalk"]["phase_4b_mvp_policy"] == {
        "policy": "retain_documented_quarantine",
        "accepted_song_count": 2,
        "quarantined_song_ids": [],
        "reason": "midi_adapter.meter_change_inside_bar",
    }
    assert first["chord_annotation_inventory"]["total_blocks"] == 4
    inventory = first["chord_annotation_inventory"]
    assert inventory["raw_block_provenance"]["source"] == "human"
    assert inventory["normalized_target_provenance"]["source"] == "derived"
    assert inventory["implicit_n_provenance"]["source"] == "derived"
    assert inventory["normalized_target_provenance"]["derivation_chain"][1][
        "method"
    ] == "POP909-CL process_pop909.py:get_chord_quality"
    assert "gap-event construction" in inventory["implicit_n_provenance"][
        "derivation_chain"
    ][1]["method"]
    assert inventory["implicit_n_gap_count"] == 2
    assert inventory["trailing_unannotated_span_count"] == 2
    assert inventory["task_mask_counts"]["root"] == {
        "available": 2,
        "unavailable": 2,
    }
    assert inventory["task_mask_counts"]["quality"] == {
        "available": 2,
        "unavailable": 2,
    }
    assert first["strict"]["production_blockers"] == [
        "phase_4b_production_adapter_not_implemented"
    ]
    assert _tree_hashes(root) == before

    output = tmp_path / "report.json"
    assert main(["--root", str(root), "--output", str(output)]) == 0
    assert output.is_file()
    assert _tree_hashes(root) == before
    inside = root / "report.json"
    assert main(["--root", str(root), "--output", str(inside)]) == 2
    assert not inside.exists()
    try:
        ensure_output_outside_root(root, inside)
    except Pop909ClAuditError:
        pass
    else:
        raise AssertionError("output inside the source root was accepted")
