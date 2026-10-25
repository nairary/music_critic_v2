"""Materialize explicitly admitted complete-file candidates without cropping.

This stage exports exact MIDI and full inspection packets. It does not infer
musical functions or claim a complete work/annotation from complete file reads.
Every source is admitted before its payload is opened; old bytes stay immutable.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from music_critic.data import dumps_piece, loads_piece
from music_critic.data.serialization import piece_to_dict
from music_critic.adapters.midi import MidiAdapterError
from music_critic.experiments.registry import load_record
from music_critic.tasks.event_theory_enrichment import canonical, validate_enrichment
try:
    from scripts.export_genre_midi import export_preview_midi
except ModuleNotFoundError:
    from export_genre_midi import export_preview_midi  # Direct script invocation.


GENRES = {'classical', 'jazz', 'rock', 'pop'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, raw):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': str(path), 'sha256': digest(raw), 'size_bytes': len(raw)}


def save_json(path, value, compressed=False):
    raw = (canonical(value) + '\n').encode()
    return save(path, gzip.compress(raw, mtime=0) if compressed else raw)


def read_bound(binding):
    raw = Path(binding['path']).read_bytes()
    if len(raw) != binding['size_bytes'] or digest(raw) != binding['sha256']:
        raise ValueError('source hash/size mismatch')
    return raw


def validate_admission(item):
    if item.get('primary_genre') not in GENRES or not item.get('work_family_id'):
        raise ValueError('genre and work family required')
    if not item.get('candidate_id') or not item.get('genre_evidence'):
        raise ValueError('candidate identity and genre evidence required')
    admission = item.get('admission', {})
    if (admission.get('status') != 'admitted'
            or admission.get('current_split') != 'train'
            or admission.get('known_heldout_family_conflict') is not False
            or not admission.get('evidence')):
        raise ValueError('explicit admitted TRAIN family receipt required')
    if item.get('source', {}).get('kind') not in {'canonical', 'pdmx', 'midi'}:
        raise ValueError('unsupported source kind')
    if not item['source'].get('binding', {}).get('path'):
        raise ValueError('explicit source path required')
    if item['source']['kind'] == 'canonical':
        b = item['source']['binding']
        if not isinstance(b.get('size_bytes'), int) or not b.get('sha256'):
            raise ValueError('canonical binding required before read')


def inspection_packet(piece, item, canonical_binding, bundle_binding):
    """All notes and bars, with compact columns but no sampling/truncation."""
    raw = piece_to_dict(piece)
    fields = ['note_id', 'track_id', 'onset_qn', 'duration_qn', 'pitch',
              'spelling_step', 'spelling_alter', 'staff', 'voice', 'is_percussion', 'is_grace']
    return {
        'schema': 'Corpus400FullInspectionPacket@1',
        'candidate_id': item['candidate_id'], 'work_family_id': item['work_family_id'],
        'primary_genre': item['primary_genre'], 'identity': item.get('identity', {}),
        'canonical_piece': canonical_binding, 'native_theory': bundle_binding,
        'source_extent': item.get('source_extent', {'status': 'unverified'}),
        'inspection_scope': 'entire_materialized_file', 'cropped': False,
        'duration_qn': raw['duration_qn'], 'tracks': raw['tracks'],
        'bars': raw['bars'], 'meter_events': raw['meter_events'],
        'key_signature_events': raw['key_signature_events'],
        'note_columns': fields,
        'note_rows': [[n.get(k) for k in fields] for n in raw['notes']],
        'note_count': len(piece.notes), 'all_notes_included': True,
        'analytical_sources': item.get('analytical_sources', []),
        'claim_boundary': 'Packet completeness is not work completeness or semantic analysis.',
    }


def materialize_one(item, output):
    validate_admission(item)  # Must happen before mkdir or any musical file read.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    admission_binding = save_json(output / 'admission.json', item)
    source = item['source']
    source_extras = None
    if source['kind'] == 'canonical':
        payload = read_bound(source['binding'])
        piece = loads_piece(payload)
        if piece.split not in (None, 'train'):
            raise ValueError('canonical source is not TRAIN')
        expected = item.get('piece_id')
        if expected and piece.piece_id != expected:
            raise ValueError('canonical piece identity mismatch')
        canonical_binding = save(output / 'piece.json', payload)
    else:
        path = Path(source['binding']['path'])
        raw = path.read_bytes()
        actual = {'path': str(path), 'sha256': digest(raw), 'size_bytes': len(raw)}
        for key in ('sha256', 'size_bytes'):
            if key in source['binding'] and source['binding'][key] != actual[key]:
                raise ValueError('raw source binding mismatch')
        suffix = '.json' if source['kind'] == 'pdmx' else '.mid'
        copied = output / ('source' + suffix)
        save(copied, raw)
        save_json(output / 'source-binding.json', actual)
        if source['kind'] == 'pdmx':
            from music_critic.adapters.pdmx import load_pdmx_piece
            piece = load_pdmx_piece(copied, source_id=item.get('source_id', item['candidate_id']),
                                   source_group_id=item['work_family_id'], split='train')
            source_data = json.loads(raw)
            # Source annotations are evidence for analysis, never extra raw notes.
            extras = {'schema': 'PdmxAnalyticalExtras@1', 'source_binding': actual,
                      'resolution': source_data['resolution'],
                      'annotations': source_data.get('annotations', []),
                      'lyrics': source_data.get('lyrics', []),
                      'tracks': [{'source_track_index': i, 'track_id': piece.tracks[i].track_id,
                                  'annotations': t.get('annotations', []),
                                  'chords': t.get('chords', []), 'lyrics': t.get('lyrics', [])}
                                 for i, t in enumerate(source_data.get('tracks', []))],
                      'claim_boundary': 'Uninterpreted source fields; no transfer/alignment or accuracy claim.'}
            source_extras = save_json(output / 'source-analytical-extras.json', extras)
        else:
            from music_critic.adapters.midi import MidiAdapterConfig, load_midi_piece
            config = MidiAdapterConfig(dataset_name=item.get('dataset_id', 'external_midi'),
                                       source_group_id=item['work_family_id'], split='train',
                                       piece_id=item.get('piece_id', 'piece:corpus400-' + digest(item['candidate_id'].encode())[:24]))
            piece = load_midi_piece(copied, config=config)
        canonical_binding = save(output / 'piece.json', (dumps_piece(piece) + '\n').encode())
    if not any(not n.is_percussion and n.duration_qn.to_fraction() > 0 for n in piece.notes):
        raise ValueError('no usable positive-duration pitched events')
    midi, midi_receipt = export_preview_midi(piece)
    midi_binding = save(output / 'score.mid', midi)
    midi_receipt['canonical_piece'] = canonical_binding
    midi_receipt['provenance'] = 'exact_export_of_canonical_events'
    map_binding = save_json(output / 'midi-event-map.json', midi_receipt)
    bundle_binding = None
    if item.get('native_theory'):
        raw = read_bound(item['native_theory'])
        bundle = json.loads(gzip.decompress(raw) if str(item['native_theory']['path']).endswith('.gz') else raw)
        validate_enrichment(piece, bundle)
        if any(bundle.get('canonical_piece', {}).get(k) != canonical_binding[k]
               for k in ('sha256', 'size_bytes')):
            raise ValueError('native theory canonical binding mismatch')
        # Preserve native bytes and original binding. Do not rewrite provenance.
        bundle_binding = save(output / ('native-theory.json.gz' if str(item['native_theory']['path']).endswith('.gz')
                                       else 'native-theory.json'), raw)
    packet = inspection_packet(piece, item, canonical_binding, bundle_binding)
    packet['source_analytical_extras'] = source_extras
    packet_binding = save_json(output / 'inspection-packet.json', packet)
    result = {
        'schema': 'Corpus400MaterializedWork@1', 'status': 'raw_materialized',
        'candidate_id': item['candidate_id'], 'work_family_id': item['work_family_id'],
        'primary_genre': item['primary_genre'], 'piece_id': piece.piece_id,
        'dataset_id': piece.dataset_name, 'notes': len(piece.notes), 'bars': len(piece.bars),
        'tracks': len(piece.tracks), 'duration_qn': piece_to_dict(piece)['duration_qn'],
        'canonical_piece': canonical_binding, 'midi': midi_binding, 'midi_event_map': map_binding,
        'native_theory': bundle_binding, 'inspection_packet': packet_binding,
        'source_analytical_extras': source_extras,
        'admission': admission_binding, 'source_extent': packet['source_extent'],
        'midi_event_coverage_complete': not midi_receipt['omitted_zero_duration_grace_ids'],
        'annotation_complete': False, 'all_required_layers_reviewed': False,
    }
    save_json(output / 'result.json', result)
    return result


def run(manifest, output, record_path):
    record = load_record(record_path)
    if record.get('status') not in ('planned', 'running') or record.get('phase') != 'tonal-relations-T08':
        raise ValueError('active T08 experiment record required')
    if record.get('kind') != 'corpus400_materialization':
        raise ValueError('materialization experiment scope required')
    manifest_binding = record.get('protocol', {}).get('admission_manifest', {})
    if not manifest_binding or Path(manifest_binding['path']).resolve() != Path(manifest).resolve():
        raise ValueError('experiment-bound admission manifest required')
    manifest_raw = read_bound(manifest_binding)
    items = [json.loads(line) for line in manifest_raw.decode().splitlines() if line.strip()]
    for item in items:
        validate_admission(item)
    if len({r['work_family_id'] for r in items}) != len(items):
        raise ValueError('duplicate family in manifest')
    if len({r['candidate_id'] for r in items}) != len(items):
        raise ValueError('duplicate candidate in manifest')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save(output / 'input-manifest.jsonl', manifest_raw)
    save(output / 'active-experiment-record.json', Path(record_path).read_bytes())
    started = time.perf_counter(); counts = {'success': 0, 'failed': 0}
    with (output / 'index.jsonl').open('x') as ledger:
        for ordinal, item in enumerate(items):
            token = digest(item['candidate_id'].encode())[:24]
            try:
                result = materialize_one(item, output / 'works' / token)
                counts['success'] += 1
            except (ValueError, OSError, MidiAdapterError) as error:
                result = {'candidate_id': item['candidate_id'], 'work_family_id': item['work_family_id'],
                          'primary_genre': item['primary_genre'], 'status': 'failed',
                          'error': type(error).__name__ + ': ' + str(error)}
                counts['failed'] += 1
            ledger.write(canonical(result) + '\n'); ledger.flush()
            if (ordinal + 1) % 20 == 0:
                print(canonical({'processed': ordinal + 1, **counts}), flush=True)
    summary = {'schema': 'Corpus400MaterializationSummary@1', **counts,
               'elapsed_seconds': time.perf_counter() - started, 'annotation_complete': 0}
    save_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--experiment-record', required=True)
    args = parser.parse_args()
    print(canonical(run(args.manifest, args.output, args.experiment_record)))


if __name__ == '__main__':
    main()
