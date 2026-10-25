#!/usr/bin/env python3
"""Prepare verified external listening clips without creating quality labels.

The canonical ``train`` value is a local development assignment required by the
existing window helper, explicitly separate from frozen Dilemmadata TRAIN.
Original payloads, source note identities, exact timing and target masks survive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys
import wave

import mido

from music_critic.adapters.midi import MidiAdapterConfig, load_midi_piece
from music_critic.adapters.pdmx import load_pdmx_piece
from music_critic.data import RationalTime as Q, ProvenanceRecord, dump_piece, validate_or_raise

_spec = importlib.util.spec_from_file_location(
    '_genre_source_windows', Path(__file__).resolve().parents[1]
    / 'src/music_critic/pdmx_pipeline/windowing.py')
_windows = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _windows
_spec.loader.exec_module(_windows)

VERSION = 'GenreSourcePreparation@1.0.0'
SPLIT_SCOPE = 'local_listening_development_only_not_frozen_Dilemmadata_TRAIN'
ROLES = ('lead', 'accompaniment', 'bass', 'drums')


def verified_bytes(path, expected):
    payload = Path(path).read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError(f'source checksum mismatch: {path}')
    return payload


def normalize_integral_json_times(payload):
    """Losslessly replace integral decimal time literals; reject fractional ticks.

    Only time/duration fields change lexical representation. All other bytes,
    including nonintegral tempo numbers, retain their original representation.
    """
    text = payload.decode('utf8')
    changes = []
    pattern = re.compile(r'("(?:time|duration)"\s*:\s*)(-?\d+\.\d+(?:[eE][+-]?\d+)?)')

    def fix(match):
        value = Fraction(match[2])
        if value.denominator != 1:
            raise ValueError('nonintegral source ticks require a separately specified projection')
        changes.append({'byte_character_offset': match.start(2), 'before': match[2],
                        'after': str(value.numerator)})
        return match[1] + str(value.numerator)

    return pattern.sub(fix, text).encode('utf8'), changes


def merge_midi_stems(paths, output_path):
    """Keep each source stem in its own track, with one verified transport map."""
    midis = [mido.MidiFile(p) for p in paths]
    if not midis or any(m.type != 1 for m in midis):
        raise ValueError('stem merge requires nonempty type-1 MIDI files')
    ppq = midis[0].ticks_per_beat
    if ppq <= 0 or any(m.ticks_per_beat != ppq for m in midis):
        raise ValueError('stem PPQ mismatch')
    maps = []
    for m in midis:
        events = []
        for track in m.tracks:
            tick = 0
            for message in track:
                tick += message.time
                if message.type in ('set_tempo', 'time_signature', 'key_signature'):
                    events.append((tick, str(message.copy(time=0))))
        maps.append(sorted(events))
    if any(events != maps[0] for events in maps[1:]):
        raise ValueError('stem transport map mismatch')
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    merged = mido.MidiFile(type=1, ticks_per_beat=ppq)
    merged.tracks.append(mido.MidiTrack(message.copy() for message in midis[0].tracks[0]))
    source_map = {}
    for index, m in enumerate(midis):
        if any(x.type in ('note_on', 'note_off') for x in m.tracks[0]):
            raise ValueError('transport track unexpectedly contains notes')
        source_map[index] = []
        for track in m.tracks[1:]:
            if any(x.type in ('set_tempo', 'time_signature', 'key_signature') for x in track):
                raise ValueError('stem notes track unexpectedly contains transport metadata')
            source_map[index].append(len(merged.tracks))
            merged.tracks.append(mido.MidiTrack(x.copy() for x in track))
    merged.save(output_path)
    return source_map


def role_tracks(piece, hints, stem_map=None):
    result = {}
    for role in ROLES:
        singular = 'drum' if role == 'drums' else role
        if stem_map is None:
            indices = set(hints.get(singular + '_track_indices', []))
        else:
            indices = {source for stem in hints.get(singular + '_stem_indices', [])
                       for source in stem_map[stem]}
        result[role] = sorted(t.track_id for t in piece.tracks
                              if t.source_track_index in indices)
    if set(result['drums']) & set(result['lead'] + result['accompaniment'] + result['bass']):
        raise ValueError('drum and pitched roles overlap')
    for role in ('lead', 'accompaniment', 'bass'):
        if not result[role]:
            raise ValueError(f'missing source role: {role}')
    return result


def source_seconds_at(piece, onset):
    total = Fraction(0)
    cursor = Q(0)
    tempo = piece.tempo_events[0].microseconds_per_quarter
    for event in piece.tempo_events[1:]:
        if event.onset_qn >= onset:
            break
        total += (event.onset_qn - cursor).to_fraction() * tempo / 1_000_000
        cursor, tempo = event.onset_qn, event.microseconds_per_quarter
    return total + (onset - cursor).to_fraction() * tempo / 1_000_000


def isolated_low_outliers(piece):
    """Flag tiny anomalous tails in an otherwise normal register, never repair."""
    result = set()
    for track in piece.tracks:
        notes = [n for n in piece.notes if n.track_id == track.track_id and not n.is_percussion]
        low = [n for n in notes if n.pitch < 18]
        if 0 < len(low) <= 3 and len(notes) >= 20 and len(low) / len(notes) < .025:
            result.update(n.note_id for n in low)
    return result


def select_window(piece, tracks, preferred_seconds=None, exclude_note_ids=(), genre=None):
    """Pick a bar-aligned ~30s constant-tempo crop with real role coverage."""
    counts = set()
    for event in piece.tempo_events:
        meter = next((m for m in reversed(piece.meter_events) if m.onset_qn <= event.onset_qn),
                     piece.meter_events[0])
        seconds_per_bar = event.microseconds_per_quarter / 1e6 * 4 * meter.numerator / meter.denominator
        counts.add(max(1, round(30 / seconds_per_bar)))
    excluded = set(exclude_note_ids)
    choices = []
    for count in sorted(counts):
        config = _windows.WindowConfig(bar_count=count, stride=1)
        for window in _windows.iter_bar_windows(piece, config):
            p = window.piece
            if p is None or len(window.source_bar_ids) != count:
                continue
            tempos = {t.microseconds_per_quarter for t in p.tempo_events}
            if len(tempos) != 1 or len({(m.numerator, m.denominator) for m in p.meter_events}) != 1:
                continue
            duration = float(p.duration_qn.to_fraction() * next(iter(tempos)) / 1_000_000)
            if not 26 <= duration <= 34 or excluded.intersection(window.original_note_ids):
                continue
            role_counts = {k: sum(n.track_id in v and n.duration_qn > Q(0) for n in p.notes)
                           for k, v in tracks.items()}
            if min(role_counts[k] for k in ('lead', 'bass', 'accompaniment')) < 8:
                continue
            if tracks['drums'] and role_counts['drums'] < 20:
                continue
            start = float(source_seconds_at(piece, window.global_offset_qn))
            # A supplied crop is source-audited for concurrent drum-bed activity.
            proximity = -abs(start - preferred_seconds) if preferred_seconds is not None else 0
            balance = min(role_counts[k] for k in ('lead', 'bass', 'accompaniment'))
            score = (proximity, balance, -abs(duration - 30), -start)
            choices.append((score, window, duration, role_counts))
    if not choices:
        raise ValueError(f'no eligible balanced constant-tempo source window: {piece.piece_id}')
    for choice in sorted(choices, key=lambda c: c[0], reverse=True):
        if genre is not None:
            from music_critic.experiments.genre_interventions import build_genre_interventions
            candidate_piece = choice[1].piece
            roles = {('melody' if k == 'lead' else k):
                     {n.note_id for n in candidate_piece.notes if n.track_id in v}
                     for k, v in tracks.items()}
            candidates, _ = build_genre_interventions(candidate_piece, roles, genre)
            if not {'harmony', 'rhythm', 'melody', 'arrangement'} <= {c['aspect'] for c in candidates}:
                continue
        return choice[1:]
    raise ValueError(f'no crop admits H/R/M/A interventions: {piece.piece_id}')


def trim_background_pcm(wav_path, output_path, start_seconds, duration_seconds):
    """Crop existing mono PCM16 audio. One explicit nearest-frame operation."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    with wave.open(str(wav_path)) as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (22050, 1, 2):
            raise ValueError('background must be mono PCM16 at22050Hz')
        start = round(float(start_seconds) * 22050)
        count = round(float(duration_seconds) * 22050)
        if start < 0 or start + count > reader.getnframes():
            raise ValueError('background crop lies outside source audio')
        reader.setpos(start)
        payload = reader.readframes(count)
    if len(payload) != count * 2:
        raise ValueError('background crop truncated')
    output_path.write_bytes(payload)
    return {'start_frame': start, 'frames': count, 'sample_rate': 22050,
            'sample_format': 'signed16_little_endian_mono',
            'sha256': sha256(payload).hexdigest(), 'rounding': 'nearest_audio_sample_only'}


def _preview_programs(piece, tracks):
    programs = {}
    for track in piece.tracks:
        if track.program is not None or track.is_percussion:
            continue
        name = (track.name or '').lower()
        program = 38 if track.track_id in tracks['bass'] else 81
        if 'piano' in name:
            program = 4
        elif 'string' in name or 'choir' in name or 'bed' in name:
            program = 89
        elif 'organ' in name:
            program = 16
        programs[track.track_id] = program
    return programs


def mark_development_scope(piece, manifest_hash, manifest_path, identifier):
    """Retain split provenance inside the final clip, beyond window closure."""
    record = ProvenanceRecord(
        'prov:zz-genre-listening-development', 'derivation', VERSION, identifier,
        None, VERSION, manifest_hash, None,
        tuple(sorted(p.provenance_id for p in piece.provenance)),
        (('source_manifest', str(manifest_path)), ('split_scope', SPLIT_SCOPE)))
    result = replace(piece, provenance=(*piece.provenance, record))
    validate_or_raise(result)
    return result


def prepare_external_sources(manifest_path, output_root):
    manifest_path, output_root = Path(manifest_path), Path(output_root)
    if output_root.exists():
        raise FileExistsError('source preparation requires a new output root')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = manifest['selected']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate source ids')
    hints_path = manifest_path.with_name('window-candidates.json')
    window_hints = json.loads(hints_path.read_text()) if hints_path.exists() else []
    hints = {r['id']: r['best30s_bar_aligned_candidate'][2] for r in window_hints
             if r['best30s_bar_aligned_candidate'] is not None}
    output_root.mkdir(parents=True)
    result = []
    for record in records:
        identifier = record['id']
        if re.fullmatch(r'[A-Za-z0-9_-]+', identifier) is None:
            raise ValueError('unsafe source identifier')
        directory = output_root / identifier
        directory.mkdir()
        changes, stem_map = [], None
        source_bindings = []
        if record['format'] == 'midi_stems_plus_drum_audio':
            for facts in record['track_facts']:
                payload = verified_bytes(facts['file'], facts['sha256'])
                source_bindings.append({'path': facts['file'], 'sha256': sha256(payload).hexdigest()})
            if record['midi_paths'] != [f['file'] for f in record['track_facts']]:
                raise ValueError('stem order differs from verified role metadata')
            merged = directory / 'source-merged.mid'
            stem_map = merge_midi_stems(record['midi_paths'], merged)
            piece = load_midi_piece(merged, config=MidiAdapterConfig(
                'mshoxxdb-listening-development', identifier, 'train'))
        elif record['format'] == 'musicrender_json':
            source_path = Path(record['source_path'])
            payload = verified_bytes(source_path, record['sha256'])
            source_bindings.append({'path': str(source_path), 'sha256': record['sha256']})
            normalized, changes = normalize_integral_json_times(payload)
            projection_path = source_path
            if changes:
                projection_path = directory / 'integral-time-projection.json'
                projection_path.write_bytes(normalized)
            piece = load_pdmx_piece(projection_path, source_id=identifier,
                                    source_group_id=identifier, split='train')
        elif record['format'] == 'midi':
            payload = verified_bytes(record['source_path'], record['sha256'])
            source_bindings.append({'path': record['source_path'], 'sha256': sha256(payload).hexdigest()})
            piece = load_midi_piece(record['source_path'], config=MidiAdapterConfig(
                'pdmx-midi-listening-development', identifier, 'train'))
        else:
            raise ValueError('unsupported source format')
        validate_or_raise(piece)
        tracks = role_tracks(piece, record['role_hints'], stem_map)
        outliers = isolated_low_outliers(piece) if record['genre'] != 'electronic' else set()
        window, duration, coverage = select_window(piece, tracks, hints.get(identifier), outliers,
                                                   record['genre'])
        clip = mark_development_scope(window.piece, sha256(manifest_bytes).hexdigest(),
                                      manifest_path, identifier)
        offset = source_seconds_at(piece, window.global_offset_qn)
        piece_path = directory / 'piece.json'
        dump_piece(clip, piece_path)
        role_ids = {('melody' if role == 'lead' else role):
                    [n.note_id for n in clip.notes if n.track_id in ids]
                    for role, ids in tracks.items()}
        source = {'source_record': record, 'verified_payloads': source_bindings,
                  'source_manifest_sha256': sha256(manifest_bytes).hexdigest(),
                  'split_scope': SPLIT_SCOPE, 'targets_created': False,
                  'integral_time_literal_normalizations': changes,
                  'excluded_source_low_outlier_ids': sorted(outliers),
                  'source_bar_start': window.source_bar_start,
                  'source_bar_stop': window.source_bar_stop,
                  'offset_qn': [window.global_offset_qn.num, window.global_offset_qn.den],
                  'source_offset_seconds_exact': [offset.numerator, offset.denominator],
                  'clipped_source_notes': window.clipped_note_count,
                  'role_note_counts': coverage,
                  'piece_sha256': sha256(piece_path.read_bytes()).hexdigest()}
        item = {'id': identifier, 'genre': record['genre'], 'title': record['title'],
                'piece_path': str(piece_path.resolve()), 'roles': role_ids,
                'piece_sha256': source['piece_sha256'],
                'role_track_ids': {('melody' if k == 'lead' else k): v for k, v in tracks.items()},
                'source': source,
                'seconds_per_quarter': clip.tempo_events[0].microseconds_per_quarter / 1e6,
                'duration_seconds': duration, 'source_offset_seconds': float(offset),
                'render_program_overrides': _preview_programs(clip, tracks)}
        from music_critic.experiments.genre_interventions import build_genre_interventions, verify_genre_variant
        from music_critic.experiments.music_interventions import Intervention, apply_interventions
        role_sets = {k: set(v) for k, v in role_ids.items()}
        candidates, omitted = build_genre_interventions(clip, role_sets, record['genre'])
        source['operator_admission'] = []
        for candidate in candidates:
            changed = apply_interventions(clip, (Intervention(
                candidate['id'], candidate['patches'], candidate['context']),)).piece
            verify_genre_variant(clip, changed, candidate, role_sets)
            source['operator_admission'].append({'id': candidate['id'], 'aspect': candidate['aspect'],
                                                'passed': True})
        source['omitted_operators'] = omitted
        if record.get('drum_wav_path'):
            verified_bytes(record['drum_wav_path'], record['drum_wav_sha256'])
            pcm_path = directory / 'drums.pcm'
            source['background_crop'] = trim_background_pcm(
                record['drum_wav_path'], pcm_path, offset, duration)
            item['background_pcm_path'] = str(pcm_path.resolve())
            item['background_pcm_sha256'] = source['background_crop']['sha256']
        (directory / 'receipt.json').write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
        result.append(item)
    (output_root / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepared = prepare_external_sources(args.manifest, args.output)
    print(json.dumps([{'id': r['id'], 'duration_seconds': r['duration_seconds'],
                       'source_offset_seconds': r['source_offset_seconds'],
                       'role_counts': {k: len(v) for k, v in r['roles'].items()}}
                      for r in prepared], ensure_ascii=False, indent=2))
