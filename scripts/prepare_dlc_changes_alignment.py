"""Stream the existing changes alignment over every explicitly bound TRAIN piece.

No corpus discovery, graph augmentation, model execution or historical cache
mutation. The relative-target transposition gate is retained as prior evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
from itertools import groupby
import csv
import io
import json
from pathlib import Path
import time

from scripts.audit_dlc_changes_alignment import bind_selected_record, digest, save


def event_groups(path, expected_ids):
    """Require one sorted contiguous group for every manifest record."""
    with path.open() as stream:
        rows = (json.loads(line) for line in stream)
        expected = iter(sorted(expected_ids))
        for rid, group in groupby(rows, key=lambda e: e['record_id']):
            if rid != next(expected, None):
                raise ValueError('missing, duplicate or unexpected source-event record')
            yield rid, list(group)
        if next(expected, None) is not None:
            raise ValueError('missing source-event record at end')


def summarize_alignment(alignment, events):
    """Count annotations once; verify exact loss weights independently."""
    usable = [t for t in alignment['targets'] if t['target_mask']]
    supported = {eid for t in usable for eid in t['source_event_ids']}
    by_id = {e['source_annotation_id']: e for e in events}
    weights = Counter()
    for t in usable:
        weights.update(t['source_event_ids'])
    atom_weight = Counter()
    for t in alignment['targets']:
        weight = Fraction(t['annotation_weight'])
        expected = sum((Fraction(1, weights[eid]) for eid in t['source_event_ids']), Fraction()) if t['target_mask'] else Fraction()
        assert weight == expected
        if not t['target_mask']:
            assert t['mask_reasons'] and not any(t['values'])
        else:
            assert not t['mask_reasons']
            for eid in t['source_event_ids']:
                assert set(t['positive_tokens']) == set(by_id[eid]['positive_tokens'])
            for atom in t['positive_tokens']:
                atom_weight[atom] += weight
    assert sum((Fraction(t['annotation_weight']) for t in usable), Fraction()) == len(supported)
    before = Counter(atom for e in events if e['target_mask'] for atom in e['positive_tokens'])
    admitted = [e for e in events if e['target_mask'] and alignment['notation_provenance'][e['source_annotation_id']]['available']]
    notation = Counter(atom for e in admitted for atom in e['positive_tokens'])
    after = Counter(atom for eid in supported for atom in by_id[eid]['positive_tokens'])
    assert dict(atom_weight) == dict(after)
    assert all(after[k] <= notation[k] <= before[k] for k in set(before) | set(after))
    return {
        'counts': {**alignment['coverage'],
                   'source_rows': sum(len(e['source_row_numbers']) for e in events),
                   'source_unmasked_events': sum(e['target_mask'] for e in events),
                   'notation_admitted_events': len(admitted),
                   'notation_rejected_previously_unmasked_events': sum(e['target_mask'] for e in events) - len(admitted),
                   'usable_nonempty_events': sum(bool(by_id[eid]['positive_tokens']) for eid in supported),
                   'usable_empty_events': sum(not by_id[eid]['positive_tokens'] for eid in supported),
                   'usable_nonempty_onsets': sum(bool(t['positive_tokens']) for t in usable),
                   'usable_empty_onsets': sum(not t['positive_tokens'] for t in usable),
                   'beat_conflicts_without_other_reason': sum(t['mask_reasons'] == ['conflicting_modifier_sets'] for t in alignment['beat_diagnostic'])},
        'before_event_support': dict(before), 'notation_event_support': dict(notation),
        'usable_event_support': dict(after),
        'usable_onset_support': dict(Counter(a for t in usable for a in t['positive_tokens'])),
        'annotation_weight_sum': str(len(supported)),
        'notation_rejection_reasons': dict(Counter(alignment['notation_provenance'][e['source_annotation_id']]['reason']
            for e in events if e['target_mask'] and not alignment['notation_provenance'][e['source_annotation_id']]['available'])),
    }


def validate_inputs(target_root, parent_record):
    parent = json.loads(parent_record.read_text())
    assert parent['status'] == 'completed'
    artifacts = {Path(a['locator']).name: a for a in parent['artifacts']}
    def check(name):
        p = target_root/name
        assert digest(p) == artifacts[name]['sha256'] and p.stat().st_size == artifacts[name]['size_bytes']
    check('manifest.json')
    manifest = json.loads((target_root/'manifest.json').read_text())
    records = manifest['records']
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('TRAIN required before source/target payload access')
    assert len({r['record_id'] for r in records}) == len(records)
    root = Path(manifest['root']).resolve()
    paths = []
    for r in records:
        p = Path(r['relative_path'])
        assert not p.is_absolute() and '..' not in p.parts and p.parts[:2] == ('pitch_arrays', 'DLC')
        assert r['record_id'].startswith('dlc:') and (root/p).resolve().is_relative_to(root)
        paths.append((root/p).resolve())
    assert len(set(paths)) == len(paths)
    membership = manifest['membership_evidence']
    assert digest(Path(membership['path'])) == membership['sha256']
    snapshot = json.loads(Path(membership['path']).read_text())
    train = {}
    for step in snapshot['training_prefix15000']:
        for draw in step['draws']:
            rid, component = draw['record_id'], draw['component_id']
            assert rid not in train or train[rid] == component
            train[rid] = component
    assert all(train.get(r['record_id']) == r['component_id'] for r in records)
    for name in ('events.jsonl', 'vocabulary.json'):
        check(name)
    vocab = json.loads((target_root/'vocabulary.json').read_text())
    assert vocab['fit_split'] == 'train'
    return manifest, vocab, artifacts


def prepare(target_root, parent_record, output):
    from music_critic.adapters.dilemmadata import DilemmadataAccepted, convert_dilemmadata_record
    from music_critic.data import dumps_piece
    from music_critic.tasks.dlc_changes import source_events
    from music_critic.tasks.dlc_changes_alignment import align_changes_to_raw_onsets
    started = time.perf_counter()
    manifest, vocab, artifacts = validate_inputs(target_root, parent_record)
    records = {r['record_id']: r for r in manifest['records']}
    root = Path(manifest['root']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'bindings.json', {'parent_record_sha256': digest(parent_record), 'source_target_artifacts':
         {k: artifacts[k] for k in ('manifest.json', 'events.jsonl', 'vocabulary.json')},
         'membership_evidence': manifest['membership_evidence'], 'record_ids': sorted(records)})
    code = {}
    for name in ('scripts/prepare_dlc_changes_alignment.py', 'scripts/audit_dlc_changes_alignment.py',
                 'src/music_critic/tasks/dlc_changes.py', 'src/music_critic/tasks/dlc_changes_alignment.py',
                 'src/music_critic/adapters/dilemmadata.py', 'tests/tasks/test_dlc_changes_full_alignment.py'):
        p = Path(name); code[name] = digest(p)
        q = output/'source'/p; q.parent.mkdir(parents=True, exist_ok=True); q.write_bytes(p.read_bytes())
    save(output/'code.json', code)
    totals = Counter(); reasons = Counter(); beat_reasons = Counter(); notation_reasons = Counter()
    before = Counter(); notation = Counter(); support = Counter(); onset_support = Counter(); pieces = Counter()
    summaries = []; failures = []
    try:
        for rid, events in event_groups(target_root/'events.jsonl', records):
            directory = output/sha256(rid.encode()).hexdigest()[:24]
            try:
                row = records[rid]; record = bind_selected_record(row, root)
                payload = record.path.read_bytes()
                assert sha256(payload).hexdigest() == row['sha256'] and len(payload) == row['size_bytes']
                rows = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig')), delimiter='\t'))
                assert source_events(rows, rid) == [{k:v for k,v in e.items() if k != 'source'} for e in events]
                expected_source = {k: row[k] for k in ('relative_path', 'sha256', 'component_id', 'split')}
                assert all(e['source'] == expected_source for e in events)
                accepted = convert_dilemmadata_record(record)
                assert isinstance(accepted, DilemmadataAccepted), str(accepted)
                alignment = align_changes_to_raw_onsets(accepted.piece, accepted.alignment_evidence,
                    events, source_sha256=row['sha256'], vocabulary=vocab['classes'])
                stats = summarize_alignment(alignment, events)
                assert digest(record.path) == row['sha256']
                directory.mkdir()
                save(directory/'alignment.json', alignment)
                save(directory/'raw_alignment.json', asdict(accepted.alignment_evidence))
                (directory/'piece.json').write_text(dumps_piece(accepted.piece), encoding='utf-8')
                save(directory/'statistics.json', stats)
                counts = dict(stats['counts'])
                reasons.update(counts.pop('masked_onset_reasons')); beat_reasons.update(counts.pop('masked_beat_reasons'))
                counts['canonical_notes'] = len(accepted.piece.notes)
                totals.update(counts); before.update(stats['before_event_support']); notation.update(stats['notation_event_support'])
                support.update(stats['usable_event_support']); onset_support.update(stats['usable_onset_support'])
                pieces.update(stats['usable_event_support'].keys()); notation_reasons.update(stats['notation_rejection_reasons'])
                summaries.append({'record_id': rid, 'directory': directory.name, 'counts': counts,
                                  'alignment_fingerprint': alignment['fingerprint'],
                                  'source_sha256': row['sha256'], 'canonical_piece_sha256': alignment['canonical_piece_sha256']})
            except Exception as exc:
                failure = {'record_id': rid, 'exception': type(exc).__name__, 'message': str(exc)}
                failures.append(failure)
                with (output/'failures.jsonl').open('a') as f: f.write(json.dumps(failure)+'\n')
            with (output/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'processed': len(summaries)+len(failures), 'completed': len(summaries),
                                   'failures': len(failures), 'record_id': rid, 'elapsed_seconds': time.perf_counter()-started})+'\n')
            if (len(summaries)+len(failures)) % 50 == 0:
                print(json.dumps({'completed': len(summaries), 'failures': len(failures)}), flush=True)
    except Exception as exc:
        failures.append({'record_id': None, 'exception': type(exc).__name__, 'message': str(exc)})
    if not failures:
        assert len(summaries) == len(records)
        assert dict(before) == vocab['event_support']
    save(output/'class_statistics.json', {'fit_split': 'train', 'vocabulary': vocab['classes'],
        'support_unit': 'distinct_source_annotation_with_at_least_one_usable_raw_onset',
        'before_event_support': {k: before[k] for k in vocab['classes']},
        'notation_event_support': {k: notation[k] for k in vocab['classes']},
        'usable_event_support': {k: support[k] for k in vocab['classes']},
        'usable_onset_support': {k: onset_support[k] for k in vocab['classes']},
        'usable_piece_support': {k: pieces[k] for k in vocab['classes']}})
    report = {'status': 'failed' if failures else 'completed', 'requested_records': len(records),
        'completed_records': len(summaries), 'failures': failures, 'counts': dict(totals),
        'masked_onset_reasons': dict(reasons), 'masked_beat_reasons': dict(beat_reasons),
        'notation_rejection_reasons': dict(notation_reasons), 'records': summaries,
        'elapsed_seconds': time.perf_counter()-started, 'usable_classes': sum(support[k]>0 for k in vocab['classes']),
        'vocabulary_size': len(vocab['classes']), 'annotation_weight_sum': totals['source_events_with_usable_onset'],
        'no_training': True, 'no_model': True, 'no_TEST_payload': True, 'no_VALIDATION_payload': True,
        'historical_canonical_cache_identity_checked': False,
        'transposition_scope': 'Existing 144-view bounded gate retained; no new full-corpus graph augmentation'}
    save(output/'report.json', report)
    save(output/'artifact_manifest.json', {str(p.relative_to(output)): {'sha256': digest(p), 'size_bytes': p.stat().st_size}
         for p in sorted(output.rglob('*')) if p.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--parent-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    r = prepare(args.target_root, args.parent_record, args.output)
    print(json.dumps({k:v for k,v in r.items() if k != 'records'}, ensure_ascii=False, indent=2))
    if r['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__':
    main()
