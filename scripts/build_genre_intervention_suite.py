#!/usr/bin/env python3
"""Build immutable listening artifacts from explicitly prepared source clips.

No source discovery, training or preference labelling. All audio variants of a
work share one gain, and an optional real drum bed stays unchanged. Symbolic
exports contain only observed MIDI notes, never inferred audio transcriptions.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
import wave

import numpy as np

from music_critic.data import load_piece, dump_piece, RationalTime as Q
from music_critic.data.validation import validate_or_raise
from music_critic.experiments.genre_interventions import build_genre_interventions, verify_genre_variant
from music_critic.experiments.music_interventions import Intervention, apply_interventions, check_compatibility
from build_intervention_player import build_player
from export_genre_midi import export_preview_midi
from render_genre_preview import synthesize_preview, SYNTH_VERSION
from build_harmonic_replacement_pilot import make_candidates, SPECS, verify_variant, checked_json
from build_music_intervention_pilot import harmonic_context

SR = 22050
ZERO = Q(0)


def qfloat(value):
    return value.num / value.den


def rat(value):
    return [value.num, value.den]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def constant_tempo(piece):
    values = {t.microseconds_per_quarter for t in piece.tempo_events}
    if len(values) != 1:
        raise ValueError('Listening clip must have one observed constant tempo')
    return next(iter(values)) / 1_000_000


def event_rows(piece, overrides=None):
    spq = constant_tempo(piece)
    tracks = {t.track_id: t for t in piece.tracks}
    overrides = overrides or {}
    if set(overrides) - tracks.keys():
        raise ValueError('Unknown rendering override track')
    result = []
    for n in piece.notes:
        program = n.program if n.program is not None else tracks[n.track_id].program
        result.append(dict(id=n.note_id, pitch=n.pitch, onset_qn=rat(n.onset_qn),
                           duration_qn=rat(n.duration_qn), onset_seconds=qfloat(n.onset_qn)*spq,
                           duration_seconds=qfloat(n.duration_qn)*spq, staff=n.staff, voice=n.voice,
                           track_id=n.track_id, velocity=80 if n.velocity is None else n.velocity,
                           program=overrides.get(n.track_id, 0 if program is None else program),
                           is_percussion=n.is_percussion))
    return result


def candidate_json(candidate, spq):
    patches = []
    for p in candidate['patches']:
        item = {'note_id': p.note_id}
        if p.pitch is not None: item['pitch'] = p.pitch
        if p.onset_qn is not None: item['onset_qn'] = rat(p.onset_qn)
        if p.duration_qn is not None: item['duration_qn'] = rat(p.duration_qn)
        if p.delete: item['delete'] = True
        patches.append(item)
    return {**{k: candidate[k] for k in ('id', 'aspect', 'title', 'description', 'check')},
            'patches': patches, 'context_note_ids': list(candidate['context']),
            'location_seconds': qfloat(candidate['onset'])*spq,
            'quality_preference': 'unreviewed', 'source_harmony': candidate.get('source_harmony'),
            'primary_aspect_is_hypothesis': True}


def contextual_candidates(descriptor, piece, pilot_root):
    """Retain the two already checked, user-liked harmonic replacements."""
    wid = descriptor.get('original_work_id')
    if wid not in SPECS:
        return []
    receipt = checked_json(pilot_root/'source-receipt.json',
        'fec2097dc63f9b3df1f867e3fed87732df96edfb0bad0218dd57657e6102f04e')
    artifact_rows = checked_json(pilot_root/'artifact-manifest.json',
        'e77d07fe262ad741dfb76bda01918e7e4ac0ac327fb65cae152e6c859c2465a8')
    bindings = {r['path']: r for r in artifact_rows}
    binding = bindings['pilot.json']
    payload = checked_json(pilot_root/'attempt003/pilot.json', binding['sha256'], binding['size_bytes'])
    work = next(w for w in payload['works'] if w['id'] == wid)
    row = next(r for r in receipt['files'] if r['record_id'] == work['source_id'] and r['kind'] == 'sidecar')
    if row['split'] != 'train':
        raise ValueError('Contextual replacement requires authenticated TRAIN annotation')
    sidecar = checked_json(Path(row['path']), row['sha256'], row['size_bytes'])
    if sidecar['record_id'] != work['source_id']:
        raise ValueError('Contextual annotation identity mismatch')
    binding = bindings[wid+'/source_clip.json']
    original_path = pilot_root/'attempt003'/wid/'source_clip.json'
    checked_json(original_path, binding['sha256'], binding['size_bytes'])
    original = load_piece(original_path)
    if original != piece:
        raise ValueError('Contextual replacements require the identical original clip')
    offset = Q(*work['global_offset_qn'])
    return make_candidates(piece, {t-offset: h for t, h in harmonic_context(sidecar).items()}, SPECS[wid])


def load_background(descriptor, frames):
    path = descriptor.get('background_pcm_path')
    if not path:
        return np.zeros(frames, dtype=np.float64), None
    raw = Path(path).read_bytes()
    expected = descriptor.get('background_pcm_sha256')
    if not expected or sha256(raw).hexdigest() != expected:
        raise ValueError('Drum audio bed hash is missing or mismatched')
    if len(raw) != frames*2:
        raise ValueError('Drum audio bed must exactly match the clip, mono PCM16 22050 Hz')
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float64)/32768
    return samples, raw


def mix_audio(events, duration, note_gain, background, background_gain):
    samples, report = synthesize_preview(events, duration_seconds=duration, sample_rate=SR, gain=note_gain)
    if len(background) != len(samples):
        raise ValueError('Audio bed frame mismatch')
    bed = background * background_gain
    # The note renderer has already applied the same one final 20ms fade.
    fade = min(len(bed), round(.02*SR))
    if fade:
        bed[-fade:] *= np.linspace(1., 0., fade) if fade > 1 else 1.
    samples += bed
    report.update(peak_absolute=float(np.max(np.abs(samples), initial=0)),
                  clipping_sample_count=int(np.count_nonzero(np.abs(samples) > 1)),
                  fixed_note_gain=note_gain, fixed_background_gain=background_gain,
                  background_unchanged=bool(background_gain))
    return samples, report


def build_work(descriptor, output, pilot_root):
    ident = descriptor['id']
    if not ident.replace('_', '').isalnum():
        raise ValueError('Unsafe work identifier')
    source_path = Path(descriptor['piece_path'])
    raw = source_path.read_bytes()
    expected = descriptor.get('piece_sha256') or descriptor.get('source', {}).get('sha256')
    if not expected or sha256(raw).hexdigest() != expected:
        raise ValueError('Prepared clip changed after admission')
    piece = load_piece(source_path)
    validate_or_raise(piece)
    if piece.split != 'train':
        raise ValueError('Only TRAIN or explicitly assigned development clips are admitted')
    split_scope = descriptor.get('source', {}).get('split_scope', '')
    if split_scope.startswith('local_listening_development') and not any(
            ('split_scope', split_scope) in p.details for p in piece.provenance):
        raise ValueError('Canonical clip lost its external development split provenance')
    roles = {k: set(v) for k, v in descriptor['roles'].items()}
    spq = constant_tempo(piece)
    duration = qfloat(piece.duration_qn)*spq
    if abs(descriptor['seconds_per_quarter']-spq) > 1e-9 or abs(descriptor['duration_seconds']-duration) > 1e-6:
        raise ValueError('Descriptor timing disagrees with canonical source')
    candidates, omitted = build_genre_interventions(piece, roles, descriptor['genre'])
    contextual = contextual_candidates(descriptor, piece, pilot_root)
    if contextual:
        candidates = contextual + [c for c in candidates if c['aspect'] != 'harmony']
        omitted = [c for c in omitted if c['aspect'] != 'harmony']
    if descriptor.get('background_pcm_path'):
        if any(c['id'] == 'S1' for c in candidates):
            omitted.append(dict(id='S1', aspect='structure', reason='Ударные доступны только как неизменное аудио; перестановка всех партий не представлена в MIDI.'))
        candidates = [c for c in candidates if c['id'] != 'S1']
    required = {'harmony', 'rhythm', 'melody', 'arrangement'}
    if required - {c['aspect'] for c in candidates}:
        raise ValueError(f'{ident}: missing requested aspects: {required - {c["aspect"] for c in candidates}}; {omitted}')
    directory = output/ident
    directory.mkdir()
    dump_piece(piece, directory/'source_clip.json')
    write_json(directory/'source.json', descriptor)
    interventions = {c['id']: Intervention(c['id'], c['patches'], c['context']) for c in candidates}
    versions = [('original', 'Исходник', [], piece, [])]
    for c in candidates:
        changed = apply_interventions(piece, (interventions[c['id']],)).piece
        check = verify_variant(piece, changed, c) if c in contextual else verify_genre_variant(piece, changed, c, roles)
        if not check['passed']:
            raise ValueError('Intervention postcondition failed')
        versions.append((c['id'], c['title'], [c['id']], changed, [check]))
    background, background_raw = load_background(descriptor, int(duration*SR+.5))
    overrides = descriptor.get('render_program_overrides', {})
    prepared = [(stem, name, ids, changed, checks, event_rows(changed, overrides)) for stem, name, ids, changed, checks in versions]
    # Two-pass peak measurement sets ONE gain for the complete comparison set.
    # No track/variant is individually normalized. The 0.8 ceiling leaves room
    # for browser Float32 summation differences.
    base_note_gain, base_bed_gain = .04, (.25 if background_raw else 0.)
    peak = max(mix_audio(events, duration, base_note_gain, background, base_bed_gain)[1]['peak_absolute']
               for _, _, _, _, _, events in prepared)
    if peak <= 0:
        raise ValueError('Silent source comparison set')
    factor = min(8., .8/peak)
    gain, bed_gain = base_note_gain*factor, base_bed_gain*factor
    reports, presets = [], []
    for stem, name, ids, changed, checks, events in prepared:
        dump_piece(changed, directory/(stem+'.json'))
        midi, midi_report = export_preview_midi(changed)
        (directory/(stem+'.mid')).write_bytes(midi)
        samples, audio_report = mix_audio(events, duration, gain, background, bed_gain)
        if audio_report['clipping_sample_count']:
            raise ValueError('Audio clipping')
        wav_path = directory/(stem+'.wav')
        with wave.open(str(wav_path), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(SR)
            stream.writeframes(np.rint(samples*32767).astype('<i2').tobytes())
        audio_report.update(wav_sha256=sha256(wav_path.read_bytes()).hexdigest(), wav_size_bytes=wav_path.stat().st_size)
        reports.append(dict(work=ident, variant=stem, midi=midi_report, audio=audio_report, checks=checks))
        presets.append(dict(name=name, ids=ids, file_stem=ident+'/'+stem,
                            compatibility='compatible' if ids else 'original', observable_checks=checks,
                            all_observable_conditions_hold=True))
    pairwise = []
    for i, left in enumerate(interventions):
        for right in list(interventions)[i+1:]:
            result = check_compatibility(interventions[left], interventions[right])
            pairwise.append(dict(left=left, right=right, status=result.status, reasons=list(result.reasons)))
    source_note = descriptor.get('source', {}).get('listening_note', '')
    if descriptor['genre'] == 'classical':
        tempo_note = 'Условный темп 120 BPM из подготовки нот; это не темп исходного исполнения. '
    else:
        tempo_note = f'Темп из MIDI/нот: {60/spq:.2f} BPM. '
    tempo_note += source_note
    if background_raw:
        tempo_note += ' Ударные — неизменная авторская аудиодорожка; в MIDI их нет. Ритмическая правка меняет нотные партии относительно неё.'
    if descriptor['genre'] == 'hiphop':
        tempo_note += ' Нотная аранжировка без текста и рэп-подачи.'
    result = dict(id=ident, genre=descriptor['genre'], title=descriptor['title'],
                  source_id=descriptor.get('source', {}).get('source_id', ident),
                  source=descriptor.get('source'), tempo_note=tempo_note,
                  duration_seconds=duration, seconds_per_quarter=spq, preview_gain=gain,
                  render_program_overrides=overrides,
                  notes=event_rows(piece, overrides), interventions=[candidate_json(c, spq) for c in candidates],
                  original_wav=ident+'/original.wav', original_midi=ident+'/original.mid',
                  presets=presets, pairwise=pairwise, omitted=omitted,
                  retained_counts=dict(Counter(c['aspect'] for c in candidates)))
    if background_raw:
        result.update(background_pcm16=base64.b64encode(background_raw).decode('ascii'), background_gain=bed_gain,
                      background_sha256=sha256(background_raw).hexdigest())
    write_json(directory/'verification.json', reports)
    return result, reports


def build(descriptors, output, pilot_root, *, require_twenty=True):
    if require_twenty and Counter(d['genre'] for d in descriptors) != Counter(dict(classical=5, rock=5, electronic=5, hiphop=5)):
        raise ValueError('Exactly five sources per requested genre are required')
    if len({d['id'] for d in descriptors}) != len(descriptors):
        raise ValueError('Duplicate source IDs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    works, reports = [], []
    write_json(output/'source-descriptors.json', descriptors)
    for descriptor in descriptors:
        work, checks = build_work(descriptor, output, pilot_root)
        works.append(work); reports.extend(checks)
        print(json.dumps(dict(work=work['id'], title=work['title'], counts=work['retained_counts'],
                              omitted=work['omitted'], elapsed_seconds=time.monotonic()-started), ensure_ascii=False), flush=True)
    payload = dict(version='GenreInterventionSuite@1.0.0', title='20 произведений · сравнение музыкальных правок',
                   renderer=SYNTH_VERSION, works=works, alternatives_only=True,
                   source_scope='Локальный исследовательский пилот · 5 TRAIN-произведений Dilemmadata + 15 внешних корпусных композиций для разработки · без TEST, обучения и автоматических оценок качества.')
    write_json(output/'pilot.json', payload)
    write_json(output/'verification.json', reports)
    build_player(payload, output/'index.html')
    summary = dict(works=len(works), genres=dict(Counter(w['genre'] for w in works)),
                   original_plus_alternatives=len(reports),
                   aspects=dict(sum((Counter(w['retained_counts']) for w in works), Counter())),
                   clipping_sample_count=sum(r['audio']['clipping_sample_count'] for r in reports),
                   elapsed_seconds=time.monotonic()-started, quality_preference='unreviewed',
                   omitted=[dict(work=w['id'], **item) for w in works for item in w['omitted']])
    write_json(output/'summary.json', summary)
    manifest = [dict(path=p.relative_to(output).as_posix(), sha256=sha256(p.read_bytes()).hexdigest(), size_bytes=p.stat().st_size)
                for p in sorted(output.rglob('*')) if p.is_file()]
    write_json(output/'artifact-manifest.json', manifest)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, default=Path('outputs/music_intervention_pilot_001'))
    args = parser.parse_args()
    entries = []
    for path in args.sources:
        value = json.loads(path.read_text())
        entries.extend(value if isinstance(value, list) else value['sources'])
    print(json.dumps(build(entries, args.output, args.pilot_root), ensure_ascii=False))
