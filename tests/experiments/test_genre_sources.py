from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import importlib.util
from pathlib import Path
import struct
import wave

import mido
import pytest

from music_critic.data import RationalTime as Q, load_piece

spec = importlib.util.spec_from_file_location('genre_source_preparation_test',
    Path(__file__).parents[2] / 'scripts/prepare_genre_sources.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def midi(path, pitch=60, tempo=500000, ppq=480, extra_tempo=False):
    m = mido.MidiFile(type=1, ticks_per_beat=ppq)
    transport = mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=tempo),
                               mido.MetaMessage('time_signature', numerator=4, denominator=4)])
    m.tracks.append(transport)
    notes = mido.MidiTrack([mido.MetaMessage('track_name', name=f'mel{pitch}'),
        mido.MetaMessage('midi_port', port=pitch),
        mido.Message('note_on', channel=0, note=pitch, velocity=70, time=ppq),
        mido.Message('note_off', channel=0, note=pitch, time=ppq)])
    if extra_tempo:
        notes.insert(0, mido.MetaMessage('set_tempo', tempo=tempo))
    m.tracks.append(notes)
    m.save(path)


def test_time_normalization_is_exact_and_other_bytes_unchanged():
    raw = b'{"time":29640.0,"duration":1.0,"qpm":119.999,"name":"time 1.0"}'
    result, changes = prep.normalize_integral_json_times(raw)
    assert result == b'{"time":29640,"duration":1,"qpm":119.999,"name":"time 1.0"}'
    assert len(changes) == 2 and changes[0]['before'] == '29640.0'
    assert raw.startswith(b'{"time":29640.0')


def test_fractional_ticks_are_rejected_not_rounded():
    with pytest.raises(ValueError, match='nonintegral'):
        prep.normalize_integral_json_times(b'{"time":1.0001}')


def test_checksum_mismatch_rejected(tmp_path):
    p = tmp_path / 'x'; p.write_bytes(b'original')
    assert prep.verified_bytes(p, sha256(b'original').hexdigest()) == b'original'
    with pytest.raises(ValueError, match='checksum'):
        prep.verified_bytes(p, '0' * 64)


def test_merge_preserves_distinct_same_channel_stems_and_timing(tmp_path):
    a, b, out = [tmp_path / n for n in ('a.mid', 'b.mid', 'merged.mid')]
    midi(a, 60); midi(b, 64)
    originals = a.read_bytes(), b.read_bytes()
    assert prep.merge_midi_stems([a, b], out) == {0: [1], 1: [2]}
    m = mido.MidiFile(out)
    assert len(m.tracks) == 3
    assert [(t[2].note, t[2].time, t[3].time) for t in m.tracks[1:]] == [(60, 480, 480), (64, 480, 480)]
    assert (a.read_bytes(), b.read_bytes()) == originals
    assert sum(x.type == 'set_tempo' for t in m.tracks for x in t) == 1


@pytest.mark.parametrize('kwargs,reason', [({'tempo': 400000}, 'transport'), ({'ppq': 960}, 'PPQ')])
def test_merge_rejects_incompatible_timeline(tmp_path, kwargs, reason):
    a, b, out = [tmp_path / n for n in ('a.mid', 'b.mid', 'merged.mid')]
    midi(a); midi(b, **kwargs)
    with pytest.raises(ValueError, match=reason):
        prep.merge_midi_stems([a, b], out)
    assert not out.exists()


def test_output_must_be_new(tmp_path):
    with pytest.raises(FileExistsError):
        prep.prepare_external_sources(tmp_path / 'does-not-exist.json', tmp_path)


def test_audio_trim_keeps_source_samples_and_exact_declared_extent(tmp_path):
    path = tmp_path / 'drum.wav'; out = tmp_path / 'drum.pcm'
    values = list(range(100))
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
        w.writeframes(struct.pack('<100h', *values))
    before = path.read_bytes()
    r = prep.trim_background_pcm(path, out, Fraction(10, 22050), Fraction(20, 22050))
    assert struct.unpack('<20h', out.read_bytes()) == tuple(values[10:30])
    assert r['start_frame'] == 10 and r['frames'] == 20
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        prep.trim_background_pcm(path, out, 0, 1)


def test_audio_cannot_silently_pad_outside_source(tmp_path):
    path = tmp_path / 'drum.wav'
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(b'\0\0' * 100)
    with pytest.raises(ValueError, match='outside'):
        prep.trim_background_pcm(path, tmp_path / 'out.pcm', 1, 1)


def test_seconds_mapping_integrates_prior_tempo_segments():
    p = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    first = replace(p.tempo_events[0], onset_qn=Q(0), microseconds_per_quarter=500000)
    second = replace(first, tempo_event_id='tempo:new', onset_qn=Q(4), microseconds_per_quarter=1000000)
    p = replace(p, tempo_events=(first, second))
    assert prep.source_seconds_at(p, Q(4)) == 2
    assert prep.source_seconds_at(p, Q(5, 1)) == 3


def test_isolated_low_source_outliers_not_genuine_low_instrument():
    p = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    n = next(n for n in p.notes if not n.is_percussion)
    ns = tuple(replace(n, note_id=f'n:{i}', pitch=0 if i == 0 else 50) for i in range(100))
    p = replace(p, notes=ns)
    assert prep.isolated_low_outliers(p) == {'n:0'}
    assert prep.isolated_low_outliers(replace(p, notes=tuple(replace(n, pitch=12) for n in ns))) == set()


def test_final_canonical_retains_explicit_development_split_provenance():
    p = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    marked = prep.mark_development_scope(p, '0' * 64, '/local/manifest.json', 'external1')
    assert marked.notes == p.notes and marked.targets == p.targets
    record = marked.provenance[-1]
    assert dict(record.details)['split_scope'] == prep.SPLIT_SCOPE
    assert record.checksum_sha256 == '0' * 64
