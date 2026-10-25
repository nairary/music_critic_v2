"""Bounded synthetic suite integration; no corpus reads or quality labels."""
import base64
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

from music_critic.data import dump_piece, load_piece, RationalTime as Q


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


suite = module("genre_suite_for_test", SCRIPTS / "build_genre_intervention_suite.py")
operators = module("genre_operator_fixtures_for_suite", Path(__file__).with_name("test_genre_interventions.py"))
midi = module("genre_midi_for_suite_test", SCRIPTS / "export_genre_midi.py")
renderer = module("genre_renderer_for_suite_test", SCRIPTS / "render_genre_preview.py")


def make_descriptor(directory, identifier="fixture1", genre="rock", *, background=False):
    piece, roles = operators.ensemble.__wrapped__()
    # Preserve the exact symbolic fixture and use a declared faster tempo to keep
    # the integration test short. This is not a source performance assertion.
    piece = replace(piece, piece_id=f"piece:{identifier}", tempo_events=(replace(piece.tempo_events[0], microseconds_per_quarter=250000),))
    path = directory / f"{identifier}.json"
    dump_piece(piece, path)
    descriptor = dict(id=identifier, genre=genre, title=f"Synthetic {identifier}", piece_path=str(path), piece_sha256=sha256(path.read_bytes()).hexdigest(), seconds_per_quarter=.25, duration_seconds=4, roles={key: sorted(value) for key, value in roles.items()}, source=dict(source_id=f"synthetic:{identifier}", listening_note="Synthetic fixture, no human quality assessment"))
    if background:
        pcm = (((np.arange(4 * suite.SR) * 41) % 8001) - 4000).astype("<i2").tobytes()
        bg_path = directory / f"{identifier}.pcm"
        bg_path.write_bytes(pcm)
        descriptor.update(background_pcm_path=str(bg_path), background_pcm_sha256=sha256(pcm).hexdigest())
    return piece, roles, descriptor


def test_external_development_scope_must_survive_in_canonical_provenance(tmp_path):
    _, _, descriptor = make_descriptor(tmp_path)
    descriptor['source']['split_scope'] = 'local_listening_development_only_not_frozen_Dilemmadata_TRAIN'
    with pytest.raises(ValueError, match='split provenance'):
        suite.build_work(descriptor, tmp_path, tmp_path/'unused')


@pytest.fixture(scope="module")
def built_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("genre-suite-synthetic")
    first = make_descriptor(root)
    second = make_descriptor(root, "fixture2", "hiphop", background=True)
    descriptors = [first[2], second[2]]
    source_bytes = {d["piece_path"]: Path(d["piece_path"]).read_bytes() for d in descriptors}
    background_path = Path(second[2]["background_pcm_path"])
    background_bytes = background_path.read_bytes()
    output = root / "suite"
    summary = suite.build(descriptors, output, root / "unused-prior-pilot", require_twenty=False)
    return dict(root=root, output=output, descriptors=descriptors, sources=[first[0], second[0]], source_bytes=source_bytes, background_path=background_path, background_bytes=background_bytes, summary=summary, payload=json.loads((output / "pilot.json").read_text()), reports=json.loads((output / "verification.json").read_text()))


def test_two_work_suite_contains_all_aspects_and_explicit_quality_boundary(built_suite):
    result = built_suite
    assert result["summary"]["works"] == 2
    assert result["summary"]["genres"] == {"rock": 1, "hiphop": 1}
    assert result["summary"]["original_plus_alternatives"] == 17
    assert result["summary"]["quality_preference"] == "unreviewed"
    assert result["summary"]["clipping_sample_count"] == 0
    assert (result["output"] / "index.html").is_file()
    for work in result["payload"]["works"]:
        assert {"harmony", "rhythm", "melody", "arrangement"} <= {c["aspect"] for c in work["interventions"]}
        expected = {"H1", "H2", "R1", "R2", "M1", "M2", "A1"}
        if "background_pcm16" not in work:
            expected.add("S1")
        else:
            assert any(c["id"] == "S1" and c["reason"] for c in work["omitted"])
        assert {c["id"] for c in work["interventions"]} == expected
        assert all(c["quality_preference"] == "unreviewed" for c in work["interventions"])
        assert any(p["status"] == "conflict" for p in work["pairwise"])


def test_all_exported_midi_independently_matches_canonical_notes(built_suite):
    root = built_suite["output"]
    for report in built_suite["reports"]:
        stem = root / report["work"] / report["variant"]
        piece = load_piece(stem.with_suffix(".json"))
        tracks = {t.track_id: t for t in piece.tracks}
        expected = Counter()
        for note in piece.notes:
            if note.duration_qn == Q(0) and note.is_grace:
                continue
            program = note.program if note.program is not None else tracks[note.track_id].program
            expected[(note.pitch, note.onset_qn.to_fraction(), note.duration_qn.to_fraction(), 0 if program is None else program, note.is_percussion, 80 if note.velocity is None else note.velocity)] += 1
        raw = stem.with_suffix(".mid").read_bytes()
        actual = midi.inspect_preview_midi(raw)
        assert actual["notes"] == expected
        assert actual["tempos"] == Counter({(Fraction(0), 250000): 1})
        assert report["midi"]["midi_sha256"] == sha256(raw).hexdigest()
        assert report["midi"]["note_roundtrip"] and report["midi"]["tempo_roundtrip"] and report["midi"]["meter_roundtrip"]
        assert report["midi"]["global_port_channel_pitch_lifetimes_valid"]
        assert report["midi"]["click_notes"] == report["midi"]["target_markers"] == 0
        assert all(check["passed"] for check in report["checks"])


def test_wavs_match_shared_gain_and_unchanged_signed_pcm_background(built_suite):
    result = built_suite
    works = {w["id"]: w for w in result["payload"]["works"]}
    seen = {key: set() for key in works}
    for report in result["reports"]:
        work = works[report["work"]]
        audio = report["audio"]
        seen[work["id"]].add((audio["fixed_note_gain"], audio["fixed_background_gain"]))
        assert audio["fixed_note_gain"] == work["preview_gain"]
        assert audio["per_file_normalization"] is False
        assert audio["clipping_sample_count"] == 0
        stem = result["output"] / work["id"] / report["variant"]
        piece = load_piece(stem.with_suffix(".json"))
        events = suite.event_rows(piece)
        expected, _ = renderer.synthesize_preview(events, duration_seconds=4, gain=work["preview_gain"])
        if "background_pcm16" in work:
            raw = base64.b64decode(work["background_pcm16"])
            assert raw == result["background_bytes"]
            assert work["background_sha256"] == sha256(raw).hexdigest()
            background = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768 * work["background_gain"]
            background[-441:] *= np.linspace(1, 0, 441)
            expected += background
            assert audio["background_unchanged"] is True
        with wave.open(str(stem.with_suffix(".wav")), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 22050, 88200)
            actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(actual, np.rint(expected * 32767).astype("<i2"))
        assert audio["wav_sha256"] == sha256(stem.with_suffix(".wav").read_bytes()).hexdigest()
    assert all(len(gains) == 1 for gains in seen.values())
    assert result["background_path"].read_bytes() == result["background_bytes"]


def test_sources_unchanged_and_manifest_hashes_all_artifacts(built_suite):
    result = built_suite
    for path, original in result["source_bytes"].items():
        assert Path(path).read_bytes() == original
    for original, descriptor in zip(result["sources"], result["descriptors"]):
        assert load_piece(result["output"] / descriptor["id"] / "source_clip.json") == original
    manifest = json.loads((result["output"] / "artifact-manifest.json").read_text())
    assert len({row["path"] for row in manifest}) == len(manifest)
    for row in manifest:
        path = result["output"] / row["path"]
        assert sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert path.stat().st_size == row["size_bytes"]


def test_existing_suite_and_work_artifacts_cannot_be_overwritten(built_suite):
    result = built_suite
    before = {p.relative_to(result["output"]): p.read_bytes() for p in result["output"].rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        suite.build(result["descriptors"], result["output"], result["root"] / "unused", require_twenty=False)
    with pytest.raises(FileExistsError):
        suite.build_work(result["descriptors"][0], result["output"], result["root"] / "unused")
    after = {p.relative_to(result["output"]): p.read_bytes() for p in result["output"].rglob("*") if p.is_file()}
    assert after == before


def test_event_seconds_use_actual_constant_tempo(tmp_path):
    piece, _, _ = make_descriptor(tmp_path)
    tempo = replace(piece.tempo_events[0], microseconds_per_quarter=333333)
    notes = (replace(piece.notes[0], onset_qn=Q(1, 7), duration_qn=Q(2, 7)),)
    events = suite.event_rows(replace(piece, notes=notes, tempo_events=(tempo,)))
    assert events[0]["onset_qn"] == [1, 7]
    assert events[0]["duration_qn"] == [2, 7]
    assert events[0]["onset_seconds"] == pytest.approx(Fraction(333333, 7000000), abs=1e-15)
    assert events[0]["duration_seconds"] == pytest.approx(Fraction(666666, 7000000), abs=1e-15)


def test_varied_tempo_rejected_before_work_artifacts(tmp_path):
    piece, _, descriptor = make_descriptor(tmp_path)
    other = replace(piece.tempo_events[0], tempo_event_id="tempo:changed", onset_qn=Q(1), microseconds_per_quarter=500000)
    changed = replace(piece, tempo_events=(*piece.tempo_events, other))
    path = Path(descriptor["piece_path"])
    dump_piece(changed, path)
    descriptor["piece_sha256"] = sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="constant tempo"):
        suite.build_work(descriptor, output, tmp_path / "unused")
    assert not (output / descriptor["id"]).exists()


def test_prepared_hash_mismatch_rejected_before_work_artifacts(tmp_path):
    _, _, descriptor = make_descriptor(tmp_path)
    descriptor["piece_sha256"] = "0" * 64
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="changed after admission"):
        suite.build_work(descriptor, output, tmp_path / "unused")
    assert not (output / descriptor["id"]).exists()


def test_missing_prepared_hash_rejected_and_source_hash_fallback_accepted(tmp_path):
    _, _, descriptor = make_descriptor(tmp_path)
    digest = descriptor.pop("piece_sha256")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="hash|admission|digest"):
        suite.build_work(descriptor, output, tmp_path / "unused")
    assert not (output / descriptor["id"]).exists()
    descriptor["source"]["sha256"] = digest
    work, reports = suite.build_work(descriptor, output, tmp_path / "unused")
    assert work["id"] == descriptor["id"]
    assert len(reports) == 9


def test_missing_required_aspects_rejected_without_inventing_roles(tmp_path):
    _, _, descriptor = make_descriptor(tmp_path)
    descriptor["roles"] = {}
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="missing requested aspects"):
        suite.build_work(descriptor, output, tmp_path / "unused")
    assert not (output / descriptor["id"]).exists()


@pytest.mark.parametrize("field,value", [("seconds_per_quarter", .5), ("duration_seconds", 5)])
def test_descriptor_timing_mismatch_rejected(tmp_path, field, value):
    _, _, descriptor = make_descriptor(tmp_path)
    descriptor[field] = value
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="timing disagrees"):
        suite.build_work(descriptor, output, tmp_path / "unused")


def test_full_suite_requires_exact_genre_counts_and_unique_ids(tmp_path):
    _, _, descriptor = make_descriptor(tmp_path)
    with pytest.raises(ValueError, match="five sources"):
        suite.build([descriptor], tmp_path / "wrong-count", tmp_path / "unused")
    with pytest.raises(ValueError, match="Duplicate"):
        suite.build([descriptor, descriptor], tmp_path / "duplicate", tmp_path / "unused", require_twenty=False)
    assert not (tmp_path / "wrong-count").exists()
    assert not (tmp_path / "duplicate").exists()


def test_background_length_rejected_and_mixing_does_not_mutate_source(tmp_path):
    path = tmp_path / "short.pcm"
    path.write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="exactly match"):
        suite.load_background({"background_pcm_path": str(path), "background_pcm_sha256": sha256(path.read_bytes()).hexdigest()}, 3)
    background = np.linspace(-.2, .2, 2205)
    before = background.copy()
    samples, report = suite.mix_audio([], .1, .04, background, .3)
    np.testing.assert_array_equal(background, before)
    assert report["fixed_background_gain"] == .3
    assert samples[-1] == 0


@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_background_hash_required_and_checked(tmp_path, digest):
    path = tmp_path / "background.pcm"
    path.write_bytes(b"\x00\x01" * 3)
    descriptor = dict(background_pcm_path=str(path))
    if digest is not None:
        descriptor["background_pcm_sha256"] = digest
    with pytest.raises(ValueError, match="hash.*missing|mismatched"):
        suite.load_background(descriptor, 3)
