"""Source-free synthetic coverage of the real-reader adapter boundary."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import mido
import pytest

from music_critic.adapters import pop909_cl
from music_critic.adapters.pop909_cl import (
    Pop909ClAdapterConfig,
    Pop909ClConversionError,
    Pop909ClCorpusRecord,
    convert_pop909_cl_file,
    inspect_pop909_cl_instruments,
    pop909_cl_raw_input_group_id,
    pop909_lineage_group_id,
    project_pop909_cl_score_bytes,
)
from music_critic.adapters.pop909_cl_frozen_source import read_raw_source
from music_critic.adapters.pop909_cl_synthetic import build_pop909_cl_smoke_midi
from music_critic.data import validate_piece
from music_critic.graph import build_raw_graph, graph_fingerprint, model_input_fingerprint


def _record(tmp_path, *, midi=None, song_id="001", name="source.mid"):
    midi = build_pop909_cl_smoke_midi() if midi is None else midi
    stream = BytesIO()
    midi.save(file=stream)
    payload = stream.getvalue()
    path = tmp_path / name
    path.write_bytes(payload)
    resolution = inspect_pop909_cl_instruments(midi)
    projection = project_pop909_cl_score_bytes(midi, resolution)
    digest = sha256(projection).hexdigest()
    return Pop909ClCorpusRecord(
        song_id=song_id, path=path, relative_path=f"POP909_processed/{song_id}.mid",
        corpus_relative_path=f"{song_id}.mid", sha256=sha256(payload).hexdigest(),
        source_group_id=pop909_cl_raw_input_group_id(digest),
        lineage_group_id=pop909_lineage_group_id(song_id),
    ), digest


def _deny(*args, **kwargs):
    raise AssertionError("forbidden read or target extraction")


@pytest.mark.parametrize("split", ["train", "validation"])
def test_raw_reader_uses_score_only_and_preserves_identity(tmp_path, monkeypatch, split):
    record, digest = _record(tmp_path)
    monkeypatch.setattr(pop909_cl, "_extract_chord_evidence", _deny)
    piece = read_raw_source(record, split=split, expected_projection_sha256=digest)
    assert piece.piece_id == "piece:pop909-cl-001"
    assert piece.source_group_id == record.source_group_id
    assert piece.dataset_name == "pop909_cl" and piece.split == split
    assert piece.source_path == record.relative_path
    assert len(piece.notes) == 5
    assert {note.channel for note in piece.notes} == {0}
    assert piece.targets == () and piece.annotations == ()
    assert not validate_piece(piece).errors


def test_channel_one_pitch_timing_extent_mutation_preserves_raw_graph(tmp_path):
    first, digest = _record(tmp_path)
    midi = build_pop909_cl_smoke_midi()
    for message in midi.tracks[2]:
        if message.type in {"note_on", "note_off"}:
            message.note += 1
            if message.time:
                message.time *= 8
    second, mutated_digest = _record(tmp_path, midi=midi, name="mutated.mid")
    assert first.sha256 != second.sha256 and digest == mutated_digest
    pieces = [read_raw_source(record, split="train", expected_projection_sha256=digest)
              for record in (first, second)]
    assert pieces[0].duration_qn == pieces[1].duration_qn
    assert pieces[0].notes == pieces[1].notes
    graphs = [build_raw_graph(piece) for piece in pieces]
    assert graph_fingerprint(graphs[0]) == graph_fingerprint(graphs[1])
    assert model_input_fingerprint(graphs[0]) == model_input_fingerprint(graphs[1])


def test_reader_matches_existing_raw_conversion_and_reads_source_once(tmp_path, monkeypatch):
    record, digest = _record(tmp_path)
    accepted = convert_pop909_cl_file(record, config=Pop909ClAdapterConfig(include_targets=False))
    original_read = Path.read_bytes
    source_reads = []

    def read(path):
        if path == record.path:
            source_reads.append(path)
            assert len(source_reads) == 1
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    actual = read_raw_source(record, split="validation", expected_projection_sha256=digest)
    assert actual == replace(accepted.piece, split="validation")
    assert source_reads == [record.path]


@pytest.mark.parametrize("split", ["test", "TEST", "synthetic", "", None])
def test_forbidden_split_fails_before_source_read(tmp_path, monkeypatch, split):
    record, digest = _record(tmp_path)
    monkeypatch.setattr(Path, "read_bytes", _deny)
    category = "test_access_forbidden" if isinstance(split, str) and split.lower() == "test" else "split_invalid"
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split=split, expected_projection_sha256=digest)
    assert error.value.category == f"pop909_frozen.{category}"


@pytest.mark.parametrize("kind", ["source", "projection", "group", "lineage"])
def test_changed_bindings_fail_closed(tmp_path, kind):
    record, digest = _record(tmp_path)
    if kind == "source":
        record.path.write_bytes(record.path.read_bytes() + b"changed")
    elif kind == "projection":
        digest = "0" * 64
        record = replace(record, source_group_id=pop909_cl_raw_input_group_id(digest))
    elif kind == "group":
        record = replace(record, source_group_id=pop909_cl_raw_input_group_id("0" * 64))
    else:
        record = replace(record, lineage_group_id="pop909-lineage:002")
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split="train", expected_projection_sha256=digest)
    assert error.value.category == {
        "source": "pop909_frozen.source_fingerprint_mismatch",
        "projection": "pop909_frozen.projection_fingerprint_mismatch",
        "group": "pop909_frozen.source_group_mismatch",
        "lineage": "pop909_frozen.record_identity_mismatch",
    }[kind]


@pytest.mark.parametrize("song_id", ["367", "658"])
def test_expected_missing_targets_are_valid_raw_inputs(tmp_path, song_id):
    midi = build_pop909_cl_smoke_midi()
    midi.tracks.pop()
    record, digest = _record(tmp_path, midi=midi, song_id=song_id)
    piece = read_raw_source(record, split="validation", expected_projection_sha256=digest)
    assert len(piece.notes) == 5 and piece.targets == ()


def test_unexpected_missing_targets_fail_production_instrument_contract(tmp_path):
    midi = build_pop909_cl_smoke_midi()
    midi.tracks.pop()
    record, digest = _record(tmp_path, midi=midi)
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split="train", expected_projection_sha256=digest)
    assert error.value.category == "pop909_frozen.missing_chord_unexpected"


def test_pinned_quarantine_is_denied_before_read(tmp_path, monkeypatch):
    record, digest = _record(tmp_path, song_id="172")
    monkeypatch.setattr(Path, "read_bytes", _deny)
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split="train", expected_projection_sha256=digest)
    assert error.value.category == "pop909_frozen.quarantined_record"


def test_generic_inside_bar_meter_failure_is_not_repaired(tmp_path):
    midi = build_pop909_cl_smoke_midi()
    midi.tracks[0].insert(2, mido.MetaMessage("time_signature", numerator=6, denominator=8, time=600))
    record, digest = _record(tmp_path, midi=midi)
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split="train", expected_projection_sha256=digest)
    assert error.value.category == "pop909_frozen.score_conversion_failed"


def test_mixed_channels_fail_without_chord_extraction(tmp_path, monkeypatch):
    record, digest = _record(tmp_path)
    midi = build_pop909_cl_smoke_midi()
    midi.tracks[1].append(mido.Message("note_on", note=60, velocity=1, channel=2))
    stream = BytesIO()
    midi.save(file=stream)
    record.path.write_bytes(stream.getvalue())
    record = replace(record, sha256=sha256(stream.getvalue()).hexdigest())
    monkeypatch.setattr(pop909_cl, "_extract_chord_evidence", _deny)
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split="train", expected_projection_sha256=digest)
    assert error.value.category == "pop909_frozen.instrument_invalid"


def test_hash_bound_malformed_midi_is_rejected(tmp_path):
    record, digest = _record(tmp_path)
    payload = b"not a MIDI file"
    record.path.write_bytes(payload)
    record = replace(record, sha256=sha256(payload).hexdigest())
    with pytest.raises(Pop909ClConversionError) as error:
        read_raw_source(record, split="train", expected_projection_sha256=digest)
    assert error.value.category == "pop909_frozen.midi_unreadable"
