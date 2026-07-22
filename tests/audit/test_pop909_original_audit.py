from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import mido

from scripts.audit_pop909_original import (
    Pop909AuditError,
    build_report,
    compare_timings,
    discover_dataset,
    dumps_report,
    ensure_output_outside_root,
    main,
    parse_annotation_line,
    parse_chord_label,
    parse_key_label,
    propose_source_group_id,
)


def _write_role_midi(path: Path, *, names: tuple[str, ...] = ("MELODY", "BRIDGE", "PIANO")) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    conductor.extend([
        mido.MetaMessage("set_tempo", tempo=500_000, time=0),
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0),
        mido.MetaMessage("end_of_track", time=1920),
    ])
    midi.tracks.append(conductor)
    for index, name in enumerate(names):
        track = mido.MidiTrack()
        track.extend([
            mido.MetaMessage("track_name", name=name, time=0),
            mido.Message("program_change", channel=index, program=0, time=0),
            mido.Message("note_on", channel=index, note=60 + index, velocity=80, time=0),
            mido.Message("note_off", channel=index, note=60 + index, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=1440),
        ])
        midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(path)


def _write_annotations(song_dir: Path, *, malformed: bool = False) -> None:
    values = {
        "beat_audio.txt": "0.00\t1.0\n0.50\t2.0\n",
        "beat_midi.txt": "0.00 1.0 1.0\n0.50 0.0 0.0\n",
        "chord_audio.txt": "0.00\t0.50\tN\n0.50\t2.00\tC:maj(9)/3\n",
        "chord_midi.txt": "0.00 0.50 N\n0.50 2.00 C:maj(9)/3\n",
        "key_audio.txt": "0.00\t2.00\tC:maj\n",
    }
    if malformed:
        values["beat_audio.txt"] += "not-a-time\t3.0\n"
    for name, payload in values.items():
        (song_dir / name).write_text(payload, encoding="utf-8")


def _official_fixture(root: Path, *, songs: tuple[str, ...] = ("001", "002")) -> Path:
    corpus = root / "POP909"
    for song_id in reversed(songs):
        song_dir = corpus / song_id
        _write_role_midi(song_dir / f"{song_id}.mid")
        _write_annotations(song_dir)
    return root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_discovery_is_deterministic_and_groups_versions(tmp_path: Path) -> None:
    root = _official_fixture(tmp_path / "dataset")
    version = root / "POP909" / "001" / "versions" / "001-v2.mid"
    _write_role_midi(version)

    first = discover_dataset(root)
    second = discover_dataset(root)
    assert first == second
    assert [song.song_id for song in first.songs] == ["001", "002"]
    assert first.layout == "official_song_directories"
    assert first.songs[0].alternatives == (version.resolve(),)
    assert propose_source_group_id("001") == "pop909-original:001"


def test_missing_malformed_and_duplicate_assets_are_retained(tmp_path: Path) -> None:
    root = _official_fixture(tmp_path / "official", songs=("001",))
    missing = root / "POP909" / "001" / "key_audio.txt"
    missing.unlink()
    _write_annotations(root / "POP909" / "001", malformed=True)
    missing.unlink()
    report = build_report(root)
    assert {row["kind"] for row in report["discovery"]["missing_assets"]} == {
        "annotation.key_audio"
    }
    assert report["annotations"]["parser_failure_count"] == 1
    assert report["annotations"]["parser_failures"][0]["line"] == 3

    flat = tmp_path / "flat"
    flat.mkdir()
    _write_role_midi(flat / "001.mid")
    _write_role_midi(flat / "001 .mid")
    duplicate = discover_dataset(flat)
    assert duplicate.duplicate_song_ids[0][0] == "001"

    official = _official_fixture(tmp_path / "versions")
    _write_role_midi(official / "POP909" / "001" / "versions" / "001-v1.mid")
    _write_role_midi(official / "POP909" / "002" / "versions" / "001-v1.mid")
    version_duplicate = discover_dataset(official)
    assert version_duplicate.duplicate_version_ids[0][0] == "001-v1"


def test_annotation_line_parsing_preserves_decimal_and_columns() -> None:
    beat = parse_annotation_line("beat_midi", "0.055333 1.0 0.0", 1)
    assert str(beat["time_seconds"]) == "0.055333"
    assert str(beat["downbeat_simple"]) == "1.0"
    chord = parse_annotation_line("chord_audio", "0.0\t1.5\tBb:min7/b3", 2)
    assert str(chord["end_seconds"]) == "1.5"
    assert chord["label"] == "Bb:min7/b3"

    try:
        parse_annotation_line("key_audio", "2.0 1.0 C:maj", 3)
    except Pop909AuditError as exc:
        assert "end precedes start" in str(exc)
    else:
        raise AssertionError("reversed interval was accepted")


def test_chord_and_key_grammar_is_lossless_without_five_class_compression() -> None:
    parsed = parse_chord_label("C#:sus4(b7,9)/b7")
    assert parsed == {
        "raw": "C#:sus4(b7,9)/b7",
        "status": "parsed",
        "root": "C#",
        "root_pc": 1,
        "quality": "sus4",
        "extensions": ["9"],
        "alterations": ["b7"],
        "suspensions": ["sus4"],
        "bass": "b7",
        "bass_pc": 11,
        "lossless": True,
    }
    assert parse_chord_label("N")["status"] == "no_chord"
    assert parse_chord_label("not-a-chord")["lossless"] is False
    assert parse_key_label("Gb:min")["tonic_pc"] == 6
    assert parse_key_label("H:maj")["lossless"] is False


def test_timing_comparison_uses_signed_and_absolute_errors() -> None:
    report = compare_timings(
        [parse_annotation_line("beat_audio", "0.05 1", 1)["time_seconds"],
         parse_annotation_line("beat_audio", "1.12 2", 2)["time_seconds"]],
        [0, 1],
        duration_seconds=2,
    )
    assert report["count"] == 2
    assert report["absolute_seconds"]["median"] == 0.05
    assert report["absolute_seconds"]["maximum"] == 0.12
    assert report["signed_seconds"]["minimum"] == 0.05
    assert report["unmatched_over_100ms"] == 1


def test_report_serialization_is_deterministic_and_audits_vocabulary(tmp_path: Path) -> None:
    root = _official_fixture(tmp_path / "dataset", songs=("001",))
    first = build_report(root)
    second = build_report(root)
    assert dumps_report(first) == dumps_report(second)
    assert first["identity"]["corpus_id"] == "pop909_original"
    assert first["generic_midi_crosswalk"]["converted"] == 1
    assert first["track_role_evidence"]["primary_resolved"] == 1
    assert first["vocabularies"]["chords"]["lossless_coverage"] == 1.0
    assert first["vocabularies"]["keys"]["lossless_coverage"] == 1.0


def test_audit_never_writes_under_dataset_root(tmp_path: Path) -> None:
    root = _official_fixture(tmp_path / "dataset", songs=("001",))
    before = _tree_hashes(root)
    output = tmp_path / "report.json"
    assert main(["--root", str(root), "--output", str(output)]) == 0
    assert output.is_file()
    assert _tree_hashes(root) == before

    inside = root / "audit.json"
    assert main(["--root", str(root), "--output", str(inside)]) == 2
    assert not inside.exists()
    try:
        ensure_output_outside_root(root, inside)
    except Pop909AuditError:
        pass
    else:
        raise AssertionError("output inside root was accepted")
