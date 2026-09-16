"""Bounded, hash-bound TRAIN changes alignment audit; no model or TEST reads."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import time


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        f.write('\n')


def patterns(event):
    raw = event['source_changes_values']
    found = []
    if event['target_mask']:
        for value in ('64', '4', '^4^2', '+9', '-3'):
            if raw == [value]: found.append(value)
        if raw == ['b3'] and event['source_context']['chord'] == ['Ger6']: found.append('Ger6')
        if raw == ['^#8'] and event['source_context']['chord'] == ['IM7']: found.append('IM7')
        if any(a['degree'] == 0 for a in event['atoms']): found.append('zero_degree')
    if 'conflicting_changes' in event['mask_reasons']: found.append('conflict')
    return found


def select_pieces(events_path, records, maximum=12):
    """Source pattern selection first, then greedy TRAIN token coverage."""
    best = {}
    tokens = defaultdict(set)
    note_rows = Counter()
    with events_path.open() as stream:
        for line in stream:
            event = json.loads(line); rid = event['record_id']
            assert rid in records
            note_rows[rid] += len(event['source_row_numbers'])
            if event['target_mask']: tokens[rid].update(event['positive_tokens'])
            rank = (records[rid]['size_bytes'], rid, event['source_annotation_id'])
            for pattern in patterns(event):
                if pattern not in best or rank < best[pattern][0]: best[pattern] = (rank, event)
    selected = sorted({r[0][1] for r in best.values()})
    assert len(selected) <= maximum
    while len(selected) < maximum:
        covered = set().union(*(tokens[rid] for rid in selected))
        candidates = [rid for rid in records if rid not in selected and note_rows[rid] <= 20000]
        if not candidates: break
        rid = min(candidates, key=lambda r: (-len(tokens[r] - covered), records[r]['size_bytes'], r))
        if not tokens[rid] - covered: break
        selected.append(rid)
    return {'records': sorted(selected), 'examples': {p: e for p, (_, e) in best.items()},
            'covered_tokens': sorted(set().union(*(tokens[r] for r in selected))),
            'source_note_rows': {rid: note_rows[rid] for rid in selected}}


def bind_selected_record(row, root):
    """Bind one explicit source using raw-only parser stats; never discover corpus.

    A fresh local adapter record, not a claim of matching historical B2/cache
    byte identity. Group membership comes from the separately verified manifest.
    """
    from music_critic.adapters import dilemmadata as a
    path = (root / row['relative_path']).resolve()
    assert path.is_relative_to(root) and path.is_file()
    assert digest(path) == row['sha256'] and path.stat().st_size == row['size_bytes']
    parsed = a._parse_raw_file(path, 'dlc')
    _, collection, name = row['record_id'].split(':', 2)
    record = a.DilemmadataCorpusRecord(
        record_id=row['record_id'], piece_id=a._piece_id(row['record_id'], 'dlc'),
        dialect='dlc', path=path, relative_path=row['relative_path'],
        collection=collection, piece_name=name, suggested_split=None,
        physical_source_sha256=row['sha256'], raw_projection_sha256=parsed.raw_projection_sha256,
        raw_equivalence_id='dilemmadata-raw:' + parsed.raw_projection_sha256,
        grouping_fingerprint=parsed.grouping_fingerprint, source_group_id=row['component_id'],
        lineage_group_id=row['component_id'].replace('dilemmadata-component:', 'dilemmadata-lineage:'),
        source_resolution=parsed.source_resolution, score_path=None,
        score_relative_path=None, score_sha256=None, raw_issue_categories=parsed.categories,
        note_row_count=parsed.note_row_count, tie_continuation_row_count=parsed.tie_continuation_row_count,
        zero_duration_row_count=parsed.zero_duration_row_count,
        corpus_identity=a.DilemmadataCorpusIdentity(),
        record_binding_version=a.DILEMMADATA_RECORD_BINDING_VERSION, record_binding_sha256='',
    )
    return a._bind_record(record)


def chord_card(pattern, event, rows):
    first = rows[event['source_row_numbers'][0] - 2]
    local = int(first['localkey_tpc'])
    chord_tones = ast.literal_eval(first['chord_tones'])
    added = ast.literal_eval(first['added_tones'])
    assert isinstance(chord_tones, tuple) and isinstance(added, tuple)
    assert all(type(t) is int for t in (*chord_tones, *added))
    selected_rows = [rows[line-2] for line in event['source_row_numbers']]
    root = int(first['root_tpc']) * 7 % 12
    return {'pattern': pattern, 'record_id': event['record_id'],
            'source_annotation_id': event['source_annotation_id'],
            'source_rows': event['source_row_numbers'], 'source_context': event['source_context'],
            'changes': event['source_changes_values'], 'root_name': first['a_root'], 'root_pc': root,
            'source_notated_chord_pcs': sorted({(t + local)*7 % 12 for t in chord_tones}),
            'source_added_pcs': sorted({(t + local)*7 % 12 for t in added}),
            'observed_note_pcs_over_annotation': sorted({int(r['pitch']) % 12 for r in selected_rows}),
            'first_rows': [{k: r.get(k) for k in ('quarterbeats_playthrough', 'pitch', 'step', 'alter', 'staff', 'voice', 'is_note_onset')}
                           for r in selected_rows[:12]],
            'claim_boundary': 'Observed notes over an annotation include passing tones; not an exhaustive chord-set equality or a voice-resolution label.'}


def audit(target_root, parent_record, output):
    from music_critic.adapters.dilemmadata import DilemmadataAccepted, convert_dilemmadata_record
    from music_critic.data import dumps_piece
    from music_critic.graph import build_raw_graph, graph_fingerprint
    from music_critic.experiments.analysisgnn.transposition import (
        canonical_directed_transposition, transpose_raw_graph_view_directed,
    )
    from music_critic.tasks.dlc_changes import source_events
    from music_critic.tasks.dlc_changes_alignment import align_changes_to_raw_onsets, changes_targets_for_transposed_view
    started = time.perf_counter()
    parent = json.loads(parent_record.read_text())
    assert parent['status'] == 'completed'
    artifacts = {Path(a['locator']).name: a for a in parent['artifacts']}
    for name in ('manifest.json',):
        assert digest(target_root/name) == artifacts[name]['sha256']
        assert (target_root/name).stat().st_size == artifacts[name]['size_bytes']
    manifest = json.loads((target_root/'manifest.json').read_text())
    assert all(r['split'] == 'train' for r in manifest['records'])
    records = {r['record_id']: r for r in manifest['records']}
    assert len(records) == len(manifest['records'])
    for name in ('events.jsonl', 'vocabulary.json'):
        assert digest(target_root/name) == artifacts[name]['sha256']
        assert (target_root/name).stat().st_size == artifacts[name]['size_bytes']
    # Reconfirm every source manifest ID/component against actual retained draws.
    membership = manifest['membership_evidence']; mpath = Path(membership['path'])
    assert digest(mpath) == membership['sha256']
    snapshot = json.loads(mpath.read_text())
    train = {d['record_id']: d['component_id'] for s in snapshot['training_prefix15000'] for d in s['draws']}
    assert all(train.get(rid) == r['component_id'] for rid, r in records.items())
    selection = select_pieces(target_root/'events.jsonl', records)
    assert sum(records[r]['size_bytes'] for r in selection['records']) <= 52428800
    assert all(n <= 20000 for n in selection['source_note_rows'].values())
    output.mkdir(parents=True, exist_ok=False)
    save(output/'selection.json', selection)
    code = {}
    for name in ('src/music_critic/tasks/dlc_changes.py', 'src/music_critic/tasks/dlc_changes_alignment.py',
                 'src/music_critic/adapters/dilemmadata.py', 'src/music_critic/graph/builder.py',
                 'src/music_critic/experiments/analysisgnn/transposition.py',
                 'scripts/audit_dlc_changes_alignment.py', 'tests/tasks/test_dlc_changes_alignment.py'):
        p = Path(name); code[name] = digest(p)
        folder = output/'source'/p.parent; folder.mkdir(parents=True, exist_ok=True)
        (folder/p.name).write_bytes(p.read_bytes())
    save(output/'bindings.json', {'parent_record_sha256': digest(parent_record),
                                'source_target_artifacts': {n: artifacts[n] for n in ('manifest.json', 'events.jsonl', 'vocabulary.json')},
                                'membership_evidence': membership, 'code': code})
    chosen_events = defaultdict(list)
    with (target_root/'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            if e['record_id'] in selection['records']: chosen_events[e['record_id']].append(e)
    vocabulary = json.loads((target_root/'vocabulary.json').read_text())['classes']
    root = Path(manifest['root']).resolve()
    summaries = []; cards = []; failures = []
    for rid in selection['records']:
        try:
            row = records[rid]; record = bind_selected_record(row, root)
            payload = record.path.read_bytes()
            rows = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig')), delimiter='\t'))
            # Independently reconstruct the complete retained source sidecar.
            rebuilt = source_events(rows, rid)
            retained = [{k: v for k, v in e.items() if k != 'source'} for e in chosen_events[rid]]
            assert rebuilt == retained
            accepted = convert_dilemmadata_record(record)
            assert isinstance(accepted, DilemmadataAccepted), str(accepted)
            assert len(accepted.piece.notes) <= 20000
            alignment = align_changes_to_raw_onsets(accepted.piece, accepted.alignment_evidence,
                chosen_events[rid], source_sha256=row['sha256'], vocabulary=vocabulary)
            directory = output/sha256(rid.encode()).hexdigest()[:24]; directory.mkdir()
            save(directory/'alignment.json', alignment)
            save(directory/'raw_alignment.json', asdict(accepted.alignment_evidence))
            (directory/'piece.json').write_text(dumps_piece(accepted.piece), encoding='utf-8')
            graph = build_raw_graph(accepted.piece); graph_before = graph_fingerprint(graph)
            pitch_i = graph['note'].cat_feature_names.index('pitch')
            ids = list(graph['note'].entity_id); by_note = {identity: i for i, identity in enumerate(ids)}
            pitch_values = graph['note'].x_cat[:, pitch_i].tolist()
            shifts = []
            for shift in range(12):
                token = canonical_directed_transposition(shift)
                if not all(0 <= p + token.signed_semitones <= 127 for p in pitch_values):
                    shifts.append({'shift_pc': shift, 'status': 'ineligible_midi_range'}); continue
                view = transpose_raw_graph_view_directed(graph, transform=token)
                view_targets = changes_targets_for_transposed_view(alignment, view, transform=token)
                assert [t['values'] for t in view_targets['targets']] == [t['values'] for t in alignment['targets']]
                assert [t['target_mask'] for t in view_targets['targets']] == [t['target_mask'] for t in alignment['targets']]
                relative_checks = 0
                shifted = view['note'].x_cat[:, pitch_i].tolist()
                assert shifted == [p + token.signed_semitones for p in pitch_values]
                for binding in accepted.alignment_evidence.rows:
                    if binding.canonical_note_id is None: continue
                    src = rows[binding.line - 2]
                    if src['root_tpc'] in ('', '<NA>'): continue
                    root_pc = int(src['root_tpc']) * 7 % 12
                    idx = by_note[binding.canonical_note_id]
                    assert pitch_values[idx] == int(src['pitch'])
                    assert (shifted[idx] - (root_pc + shift)) % 12 == (int(src['pitch']) - root_pc) % 12
                    relative_checks += 1
                inverse = transpose_raw_graph_view_directed(view, transform=token.inverse())
                shifts.append({'shift_pc': shift, 'signed_semitones': token.signed_semitones,
                               'status': 'passed', 'relative_interval_checks': relative_checks,
                               'inverse_full_graph_exact': graph_fingerprint(inverse) == graph_before,
                               'bound_target_fingerprint': view_targets['fingerprint']})
            assert graph_fingerprint(graph) == graph_before
            assert digest(record.path) == row['sha256']
            save(directory/'shifts.json', shifts)
            summary = {'record_id': rid, 'artifact_directory': str(directory), 'canonical_notes': len(accepted.piece.notes),
                       'raw_graph_fingerprint': graph_before, 'source_sha256': row['sha256'],
                       'coverage': alignment['coverage'], 'shifts': shifts,
                       'tie_rows': sum(b.tie_continuation for b in accepted.alignment_evidence.rows),
                       'repair_rows': sum(b.repair_mask_scope != 'none' for b in accepted.alignment_evidence.rows),
                       'time_transform': None if accepted.repair_evidence is None else asdict(accepted.repair_evidence.time_transform)}
            summaries.append(summary)
            for pattern, event in selection['examples'].items():
                if event['record_id'] == rid: cards.append(chord_card(pattern, event, rows))
            print(json.dumps({'record_id': rid, 'onsets': alignment['coverage']['onset_candidates'],
                              'unmasked': alignment['coverage']['unmasked_onsets']}), flush=True)
        except Exception as exc:
            failures.append({'record_id': rid, 'exception': type(exc).__name__, 'message': str(exc)})
            print(json.dumps(failures[-1]), flush=True)
    save(output/'musical_cards.json', cards)
    report = {'status': 'completed' if not failures else 'failed', 'selection': selection['records'],
              'selected_vocabulary_coverage': len(selection['covered_tokens']), 'vocabulary_size': len(vocabulary),
              'records': summaries, 'failures': failures, 'elapsed_seconds': time.perf_counter()-started,
              'no_training': True, 'no_model': True, 'no_TEST_payload': True, 'no_VALIDATION_payload': True,
              'historical_canonical_cache_identity_checked': False,
              'musical_review_status': 'source_cards_prepared_for_notation_review_not_human_expert_validation'}
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
    report = audit(args.target_root, args.parent_record, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    if report['failures']: raise SystemExit(1)


if __name__ == '__main__':
    main()
