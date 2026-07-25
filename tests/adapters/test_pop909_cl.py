from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import mido
import pytest

from music_critic.adapters import (
    Pop909ClAdapterConfig,
    Pop909ClConversionError,
    Pop909ClCorpusIdentity,
    Pop909ClCorpusIdentityError,
    Pop909ClCorpusRecord,
    convert_pop909_cl_file,
    discover_pop909_cl_corpus,
)
from music_critic.adapters.pop909_cl import (
    POP909_CL_TASKS,
    inspect_pop909_cl_instruments,
    project_pop909_cl_score_bytes,
)
from music_critic.data import dumps_piece, loads_piece
from music_critic.graph import build_raw_graph, graph_fingerprint
from scripts.accept_pop909_cl_adapter import _load_expectations


def _conductor(*, quarantine_meter: bool = False) -> mido.MidiTrack:
    track = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.MetaMessage(
                "time_signature", numerator=4, denominator=4, time=0
            ),
            mido.MetaMessage("key_signature", key="C", time=0),
        ]
    )
    if quarantine_meter:
        track.extend(
            [
                mido.MetaMessage(
                    "time_signature",
                    numerator=6,
                    denominator=8,
                    time=85_080,
                ),
                mido.MetaMessage(
                    "time_signature",
                    numerator=4,
                    denominator=4,
                    time=16_320,
                ),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    else:
        track.append(mido.MetaMessage("end_of_track", time=1_920))
    return track


def _score_track(*, name: str = "piano", channel: int = 0) -> mido.MidiTrack:
    return mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name=name, time=0),
            mido.Message("program_change", channel=channel, program=0, time=0),
            mido.Message(
                "note_on", channel=channel, note=60, velocity=80, time=0
            ),
            mido.Message(
                "note_off", channel=channel, note=60, velocity=0, time=480
            ),
            mido.Message(
                "note_on", channel=channel, note=64, velocity=80, time=0
            ),
            mido.Message(
                "note_off", channel=channel, note=64, velocity=0, time=480
            ),
            mido.MetaMessage("end_of_track", time=960),
        ]
    )


def _chord_track(
    *,
    first: tuple[int, ...] = (60, 64, 67),
    second: tuple[int, ...] = (60, 61),
    repeated: bool = False,
    mixed_end: bool = False,
    anomalies: bool = False,
) -> mido.MidiTrack:
    track = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="chords", time=0),
            mido.Message("program_change", channel=1, program=0, time=0),
        ]
    )
    if anomalies:
        track.append(
            mido.Message(
                "note_off", channel=1, note=55, velocity=9, time=5
            )
        )
    pitches = (*first, first[0]) if repeated else first
    for pitch in pitches:
        track.append(
            mido.Message(
                "note_on", channel=1, note=pitch, velocity=70, time=0
            )
        )
    for index, pitch in enumerate(pitches):
        track.append(
            mido.Message(
                "note_off",
                channel=1,
                note=pitch,
                velocity=0,
                time=480 if index == 0 else (1 if mixed_end else 0),
            )
        )
    for pitch in second:
        track.append(
            mido.Message(
                "note_on",
                channel=1,
                note=pitch,
                velocity=70,
                time=480 if pitch == second[0] else 0,
            )
        )
    for index, pitch in enumerate(second):
        if anomalies and index == 0:
            continue
        track.append(
            mido.Message(
                "note_off",
                channel=1,
                note=pitch,
                velocity=0,
                time=480 if index == 0 else 0,
            )
        )
    track.append(mido.MetaMessage("end_of_track", time=480))
    return track


def _write_midi(
    path: Path,
    *,
    song_id: str = "001",
    include_chords: bool = True,
    chord_first: tuple[int, ...] = (60, 64, 67),
    score_name: str = "piano",
    quarantine_meter: bool = False,
    repeated: bool = False,
    mixed_end: bool = False,
    anomalies: bool = False,
) -> Pop909ClCorpusRecord:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.extend(
        [
            _conductor(quarantine_meter=quarantine_meter),
            _score_track(name=score_name),
        ]
    )
    if include_chords:
        midi.tracks.append(
            _chord_track(
                first=chord_first,
                repeated=repeated,
                mixed_end=mixed_end,
                anomalies=anomalies,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)
    checksum = sha256(path.read_bytes()).hexdigest()
    return Pop909ClCorpusRecord(
        song_id=song_id,
        path=path,
        relative_path=f"POP909_processed/{path.name}",
        corpus_relative_path=path.name,
        sha256=checksum,
        source_group_id=f"pop909-cl:{song_id}",
        lineage_group_id=f"pop909-lineage:{song_id}",
    )


def _fingerprint(paths: list[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _raw_projection(piece):
    raw_provenance = tuple(
        record
        for record in piece.provenance
        if not record.provenance_id.startswith("prov:pop909-cl")
    )
    raw_flags = tuple(
        flag
        for flag in piece.quality_flags
        if not flag.code.startswith("pop909_cl.")
    )
    return replace(
        piece,
        annotations=(),
        targets=(),
        provenance=raw_provenance,
        quality_flags=raw_flags,
    )


@pytest.mark.parametrize("nested", [False, True])
def test_direct_and_nested_discovery_preserve_043_and_exclude_noise(
    tmp_path: Path, nested: bool
) -> None:
    root = tmp_path / "install"
    corpus = root / "POP909_processed"
    if nested:
        corpus = corpus / "POP909_processed"
    paths = [corpus / "001.mid", corpus / "043 .mid"]
    _write_midi(paths[0], song_id="001")
    _write_midi(paths[1], song_id="043")
    noise = root / "__MACOSX" / "._001.mid"
    noise.parent.mkdir(parents=True)
    noise.write_bytes(b"AppleDouble")
    identity = Pop909ClCorpusIdentity(
        expected_song_ids=("001", "043"),
        expected_content_fingerprint=_fingerprint(paths),
    )
    discovery = discover_pop909_cl_corpus(root, identity=identity)
    assert discovery.is_valid
    assert [record.song_id for record in discovery.records] == ["001", "043"]
    assert discovery.records[1].corpus_relative_path == "043 .mid"
    assert discovery.records[1].relative_path.endswith("043 .mid")
    assert discovery.noise_paths == ("__MACOSX/._001.mid",)
    assert discovery.records[0].source_group_id == "pop909-cl:001"
    assert discovery.records[0].lineage_group_id == "pop909-lineage:001"


def test_discovery_diagnoses_missing_duplicate_malformed_and_fingerprint(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "POP909_processed"
    first = corpus / "001.mid"
    duplicate = corpus / "001 .mid"
    malformed = corpus / "bad.mid"
    _write_midi(first)
    _write_midi(duplicate)
    _write_midi(malformed)
    discovery = discover_pop909_cl_corpus(
        tmp_path,
        identity=Pop909ClCorpusIdentity(
            expected_song_ids=("001", "002"),
            expected_content_fingerprint="0" * 64,
        ),
        require_valid=False,
    )
    assert {issue.category for issue in discovery.issues} == {
        "content_fingerprint_mismatch",
        "duplicate_song_id",
        "malformed_midi_id",
        "missing_song_id",
    }
    with pytest.raises(Pop909ClCorpusIdentityError):
        discover_pop909_cl_corpus(
            tmp_path,
            identity=Pop909ClCorpusIdentity(
                expected_song_ids=("001", "002"),
                expected_content_fingerprint="0" * 64,
            ),
        )


def test_file_mutation_after_discovery_always_fails_fingerprint_check(
    tmp_path: Path,
) -> None:
    path = tmp_path / "POP909_processed" / "001.mid"
    _write_midi(path)
    discovery = discover_pop909_cl_corpus(
        tmp_path,
        identity=Pop909ClCorpusIdentity(
            expected_song_ids=("001",),
            expected_content_fingerprint=_fingerprint([path]),
        ),
    )

    _write_midi(path, chord_first=(62, 65, 69))

    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(discovery.records[0])
    assert caught.value.category == "pop909_cl.file_fingerprint_mismatch"


def test_instrument_routing_uses_channels_not_misleading_names(tmp_path: Path) -> None:
    record = _write_midi(
        tmp_path / "658.mid",
        song_id="658",
        include_chords=False,
        score_name="chords",
    )
    result = convert_pop909_cl_file(record)
    assert result.status == "accepted_missing_targets"
    assert result.piece.tracks[1].name == "chords"
    assert {track.channel for track in result.piece.tracks} == {None, 0}
    assert all(note.channel == 0 for note in result.piece.notes)

    midi = mido.MidiFile(record.path)
    midi.tracks.append(_score_track(name="MIDI 01"))
    midi.save(record.path)
    changed = replace(record, sha256=sha256(record.path.read_bytes()).hexdigest())
    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(changed)
    assert caught.value.category == "pop909_cl.instrument.invalid"

    multiple_chords = mido.MidiFile(record.path)
    multiple_chords.tracks[1] = _score_track(name="piano")
    multiple_chords.tracks.append(_chord_track())
    multiple_chords.tracks.append(_chord_track())
    multiple_chords.save(record.path)
    changed = replace(record, sha256=sha256(record.path.read_bytes()).hexdigest())
    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(changed)
    assert caught.value.category == "pop909_cl.instrument.invalid"

    missing_score = mido.MidiFile(type=1, ticks_per_beat=480)
    missing_score.tracks.extend([_conductor(), _chord_track()])
    missing_score.save(record.path)
    changed = replace(record, sha256=sha256(record.path.read_bytes()).hexdigest())
    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(changed)
    assert caught.value.category == "pop909_cl.instrument.invalid"

    mixed = mido.MidiFile(type=1, ticks_per_beat=480)
    mixed.tracks.extend([_conductor(), _score_track()])
    mixed.tracks[1].append(
        mido.Message("program_change", channel=2, program=0, time=0)
    )
    mixed.save(record.path)
    changed = replace(record, sha256=sha256(record.path.read_bytes()).hexdigest())
    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(changed)
    assert caught.value.category == "pop909_cl.instrument.invalid"


def test_exact_evidence_targets_masks_provenance_and_round_trip(
    tmp_path: Path,
) -> None:
    record = _write_midi(
        tmp_path / "001.mid",
        repeated=True,
        mixed_end=True,
        anomalies=True,
    )
    result = convert_pop909_cl_file(record)
    assert result.status == "accepted"
    evidence = result.chord_evidence
    assert evidence.ppqn == 480
    assert evidence.blocks[0].midi_pitch_multiset == (60, 60, 64, 67)
    assert evidence.blocks[0].note_end_ticks == (485, 486, 487, 488)
    assert evidence.blocks[0].repeated_pitch
    assert evidence.blocks[0].mixed_end_ticks
    assert evidence.blocks[0].candidates
    assert evidence.blocks[1].normalization_status == "unsupported"
    assert not evidence.blocks[1].root_available
    assert not evidence.blocks[1].quality_available
    assert not evidence.blocks[1].inversion_available
    assert evidence.pairing_anomalies[0].category == "unmatched_note_off"
    assert evidence.pairing_anomalies[0].tick == 5
    assert evidence.pairing_anomalies[0].source_path.endswith("001.mid")
    assert evidence.pairing_anomalies[1].category == "dangling_note_on"
    assert evidence.pairing_anomalies[1].affected_interval_end_tick == 1_920

    targets = {target.task: target for target in result.piece.targets}
    assert tuple(sorted(targets)) == POP909_CL_TASKS
    assert targets["pop909_cl.chord.boundary"].mask == (True, True)
    assert targets["pop909_cl.chord.bass"].mask == (True, True)
    assert targets["pop909_cl.chord.root"].mask == (True, False)
    assert targets["pop909_cl.chord.quality"].mask == (True, False)
    assert targets["pop909_cl.chord.inversion"].mask == (True, False)
    assert targets["pop909_cl.chord.no_chord"].values[-1] is None
    assert targets["pop909_cl.chord.no_chord"].mask[-1] is False
    assert targets["pop909_cl.chord.bass"].source == ("human", "human")
    assert targets["pop909_cl.chord.root"].source == ("derived", None)
    assert all(value is None for value in targets["pop909_cl.chord.root"].confidence)
    assert not result.validation_report.errors
    payload = dumps_piece(result.piece)
    assert dumps_piece(loads_piece(payload)) == payload
    assert record.sha256 in payload
    assert "human_corrected" in payload
    assert "expert_reviewed" in payload
    assert "candidates" in payload


def test_ambiguous_candidates_keep_quality_and_bass_but_mask_root_inversion(
    tmp_path: Path,
) -> None:
    record = _write_midi(
        tmp_path / "262.mid",
        song_id="262",
        chord_first=(60, 63, 66, 69),
    )
    result = convert_pop909_cl_file(record)
    block = result.chord_evidence.blocks[0]
    assert block.normalization_status == "ambiguous"
    assert len(block.candidates) == 4
    assert {candidate.quality for candidate in block.candidates} == {"o7"}
    assert not block.root_available
    assert block.quality_available
    assert not block.inversion_available
    targets = {target.task: target for target in result.piece.targets}
    assert targets["pop909_cl.chord.bass"].mask[0] is True
    assert targets["pop909_cl.chord.quality"].mask[0] is True
    assert targets["pop909_cl.chord.root"].mask[0] is False
    assert targets["pop909_cl.chord.inversion"].mask[0] is False


@pytest.mark.parametrize("song_id", ["367", "658"])
def test_expected_missing_targets_are_explicitly_masked(
    tmp_path: Path, song_id: str
) -> None:
    record = _write_midi(
        tmp_path / f"{song_id}.mid",
        song_id=song_id,
        include_chords=False,
        score_name="chords" if song_id == "658" else "piano",
    )
    result = convert_pop909_cl_file(record)
    assert result.status == "accepted_missing_targets"
    assert len(result.piece.annotations) == 1
    assert len(result.piece.targets) == 6
    for target in result.piece.targets:
        assert target.mask == (False,)
        assert target.values == (None,)
        assert target.source == (None,)
        assert target.provenance == (None,)
    assert not result.validation_report.errors


def test_target_hidden_and_chord_mutations_leave_raw_and_graph_invariant(
    tmp_path: Path,
) -> None:
    records = [
        _write_midi(tmp_path / "normal.mid", chord_first=(60, 64, 67)),
        _write_midi(tmp_path / "replacement.mid", chord_first=(62, 65, 69)),
    ]
    deleted = _write_midi(tmp_path / "deleted.mid")
    deleted_midi = mido.MidiFile(deleted.path)
    deleted_midi.tracks[2] = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="chords", time=0),
            mido.Message("program_change", channel=1, program=0, time=0),
            mido.MetaMessage("end_of_track", time=1_920),
        ]
    )
    deleted_midi.save(deleted.path)
    records.append(
        replace(deleted, sha256=sha256(deleted.path.read_bytes()).hexdigest())
    )
    visible = convert_pop909_cl_file(records[0])
    hidden = convert_pop909_cl_file(
        records[0], config=Pop909ClAdapterConfig(include_targets=False)
    )
    assert _raw_projection(visible.piece) == hidden.piece
    assert graph_fingerprint(build_raw_graph(visible.piece)) == graph_fingerprint(
        build_raw_graph(hidden.piece)
    )

    converted = [convert_pop909_cl_file(record) for record in records]
    midi_files = [mido.MidiFile(record.path) for record in records]
    resolutions = [inspect_pop909_cl_instruments(midi) for midi in midi_files]
    projections = [
        project_pop909_cl_score_bytes(midi, resolution)
        for midi, resolution in zip(midi_files, resolutions)
    ]
    assert projections[0] == projections[1] == projections[2]
    assert len({result.score_projection_sha256 for result in converted}) == 1
    assert converted[0].piece.tracks == converted[1].piece.tracks == converted[2].piece.tracks
    assert converted[0].piece.notes == converted[1].piece.notes == converted[2].piece.notes
    assert converted[0].chord_evidence.blocks[0].midi_pitch_multiset != (
        converted[1].chord_evidence.blocks[0].midi_pitch_multiset
    )
    assert converted[2].chord_evidence.blocks == ()
    assert converted[2].chord_evidence.trailing_spans[0].available is False
    assert len(
        {
            graph_fingerprint(build_raw_graph(result.piece))
            for result in converted
        }
    ) == 1


def test_song_172_is_quarantined_only_for_actual_generic_meter_error(
    tmp_path: Path,
) -> None:
    record = _write_midi(
        tmp_path / "172.mid",
        song_id="172",
        quarantine_meter=True,
    )
    result = convert_pop909_cl_file(record)
    assert result.status == "quarantined"
    assert result.category == "midi_adapter.meter_change_inside_bar"
    assert result.record.source_group_id == "pop909-cl:172"
    assert result.record.lineage_group_id == "pop909-lineage:172"
    assert "172.mid" in result.source_error
    assert "music-critic-pop909-cl-" not in result.source_error

    ordinary = _write_midi(tmp_path / "172-fixed.mid", song_id="172")
    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(ordinary)
    assert caught.value.category == "pop909_cl.quarantine_expected_failure_missing"


def test_conversion_never_writes_under_corpus_root(tmp_path: Path) -> None:
    record = _write_midi(tmp_path / "POP909_processed" / "001.mid")
    before = {
        path.relative_to(tmp_path).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    convert_pop909_cl_file(record)
    after = {
        path.relative_to(tmp_path).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_versioned_production_manifest_and_unreadable_source(
    tmp_path: Path,
) -> None:
    manifest = _load_expectations(
        Path("tests/fixtures/pop909_cl/production_manifest.json")
    )
    assert manifest["adapter_version"] == "1.0.0"
    assert manifest["expected"]["accepted"] == 908
    missing = Pop909ClCorpusRecord(
        song_id="001",
        path=tmp_path / "missing.mid",
        relative_path="POP909_processed/missing.mid",
        corpus_relative_path="missing.mid",
        sha256="0" * 64,
        source_group_id="pop909-cl:001",
        lineage_group_id="pop909-lineage:001",
    )
    with pytest.raises(Pop909ClConversionError) as caught:
        convert_pop909_cl_file(missing)
    assert caught.value.category == "pop909_cl.midi_unreadable"
