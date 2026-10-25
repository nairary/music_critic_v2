"""Frozen TRAIN coverage of raw-only exact-time readout candidates.

No model/graph execution, no training or historical cache writes. Candidates
are constructed before target projection, from raw note and beat timing only.
Theory labels are inspected exclusively on the diagnostic/alignment side.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

from scripts import audit_harmonic_target_slots as previous
from music_critic.data.raw_time_candidates import candidate_time_sets, raw_time_sources

POLICIES = ("note_onsets", "note_onsets_offsets", "note_onsets_beats",
            "note_onsets_offsets_beats")
TIME_HEADS = tuple(h for h in previous.HEADS
                  if previous.TASK_BY_ID[h].prediction_level != "note") + ("joint",)
NOTE_HEADS = tuple(h for h in previous.HEADS if h not in TIME_HEADS)
PARENT_ID = "EXP-TARGET-SLOT-CENSUS-002"


def label_key(label):
    return label if isinstance(label, str) else json.dumps(label, separators=(",", ":"))


def minimum_errors(groups):
    return sum(sum(c.values()) - max(c.values()) for c in groups.values() if c)


def verify_baseline_counts(actual, historical):
    ignored = {'pieces', 'pieces_with_conflicts'}  # added by the parent run wrapper
    for key in (set(actual) | set(historical)) - ignored:
        if actual.get(key, 0) != historical.get(key, 0):
            raise ValueError('historical projection changed: ' + key)


def select_policy(variants, inventories):
    qualifying = [p for p in POLICIES
                  if all(variants[p][h].get('historical_lost_rows', 0) == 0 for h in TIME_HEADS)
                  and variants[p]['cadence'].get('uncovered_rows', 0) == 0]
    selected = min(qualifying, key=lambda p: (inventories[p], POLICIES.index(p))) if qualifying else None
    return qualifying, selected


def unique_record_index(path):
    rows = [json.loads(s) for s in path.read_text().splitlines() if s]
    result = {r['record_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate parent record')
    return result


def eligible_joint(projected):
    """Joint support is source eligibility, not the old mapped intersection."""
    heads = [{r['entity']: r for r in projected[h]
              if r['source_available'] and r['known']} for h in previous.JOINT]
    common = set.intersection(*(set(h) for h in heads))
    result = []
    for entity in sorted(common):
        rows = [h[entity] for h in heads]
        if len({r['onset'] for r in rows}) != 1 or len({r['slot'] for r in rows}) != 1:
            raise ValueError('joint identity/time route mismatch')
        result.append({**rows[0], 'label': tuple(r['label'] for r in rows),
                       'valid': all(r['valid'] for r in rows),
                       'mapped': all(r['mapped'] for r in rows)})
    return result


def coverage_rows(rows, candidates):
    """Exact membership; eligibility includes known available old-unmapped rows.

    All source masks remain authoritative. A candidate without an available
    label does not create a training example or a negative class. Same-time
    disagreements remain separate rows, measured rather than resolved here.
    """
    times = {p: frozenset(str(t) for t in candidates[p]) for p in POLICIES}
    counts = Counter(target_rows=len(rows))
    groups = defaultdict(list)
    for row in rows:
        counts['historical_valid_rows'] += int(row['valid'])
        counts['source_available_rows'] += int(row['source_available'])
        counts['unknown_available_rows'] += int(row['source_available'] and not row['known'])
        if row['source_available'] and row['known']:
            if row['masked']:
                raise ValueError('available row contradicts source mask')
            onset = row['onset']
            if not isinstance(onset, str) or str(Fraction(onset)) != onset:
                raise ValueError('noncanonical target time')
            groups[onset].append(row)
        elif row['valid']:
            raise ValueError('historical valid row is not eligible')
    counts['eligible_rows'] = sum(map(len, groups.values()))
    counts['eligible_times'] = len(groups)
    variants = {p: Counter(candidate_positions=len(times[p])) for p in POLICIES}
    classes = defaultdict(Counter)
    kept_beat = {p: defaultdict(Counter) for p in POLICIES}
    conflicts = []
    for onset, group in groups.items():
        labels = Counter(r['label'] for r in group)
        hist = [r for r in group if r['valid']]
        hist_labels = Counter(r['label'] for r in hist)
        exact_errors = len(group) - max(labels.values())
        counts['ideal_exact_errors'] += exact_errors
        if exact_errors:
            conflicts.append({'onset': onset, 'rows': group,
                              'unavoidable_errors': exact_errors})
        for label, n in labels.items():
            classes[label_key(label)]['eligible_rows'] += n
        for p in POLICIES:
            c = variants[p]
            if onset in times[p]:
                c['covered_rows'] += len(group)
                c['covered_times'] += 1
                c['historical_kept_rows'] += len(hist)
                c['newly_covered_rows'] += len(group) - len(hist)
                c['covered_exact_errors'] += exact_errors
                c['conflict_times'] += int(exact_errors > 0)
                c['historical_kept_exact_errors'] += len(hist) - max(hist_labels.values(), default=0)
                for r in hist:
                    kept_beat[p][r['slot']][r['label']] += 1
                for label, n in labels.items():
                    classes[label_key(label)][p + '_covered_rows'] += n
            else:
                c['uncovered_rows'] += len(group)
                c['uncovered_times'] += 1
                c['historical_lost_rows'] += len(hist)
    for p, c in variants.items():
        c['historical_kept_old_slot_errors'] = minimum_errors(kept_beat[p])
        c['temporal_errors_removed_on_kept_rows'] = (c['historical_kept_old_slot_errors']
                                                     - c['historical_kept_exact_errors'])
        c['pieces_with_uncovered_rows'] = int(c['uncovered_rows'] > 0)
        c['pieces_with_historical_loss'] = int(c['historical_lost_rows'] > 0)
    misses = [r for t, rs in groups.items() if t not in times[POLICIES[-1]] for r in rs]
    recovered = [dict(r, covered_by=[p for p in POLICIES if t in times[p]])
                 for t, rs in groups.items() for r in rs if not r['valid']]
    return dict(counts), {p: dict(c) for p, c in variants.items()}, {
        k: dict(c) for k, c in classes.items()}, misses, recovered, conflicts


def parent_inputs(path):
    """Verify only the selected parent artifacts before reading them."""
    record = json.loads(path.read_text())
    if record['experiment_id'] != PARENT_ID or record['status'] != 'completed':
        raise ValueError('unexpected parent census')
    from music_critic.experiments.registry import validate_record
    validate_record(record)
    refs = {a['locator']: a for a in record['artifacts']}
    paths = {name: Path('outputs/target_slot_census_002/run') / name
             for name in ('sources.jsonl', 'pieces.jsonl', 'summary.json')}
    for p in paths.values():
        ref = refs[p.as_posix()]
        if previous.digest(p) != ref['sha256'] or p.stat().st_size != ref['size_bytes']:
            raise ValueError('parent artifact mismatch: ' + p.name)
    return paths


def run(args):
    started = time.perf_counter()
    assignments = previous.train_assignments(args.bindings / 'metadata/split_assignments.jsonl')
    parent = parent_inputs(args.parent)
    sources = unique_record_index(parent['sources.jsonl'])
    baseline = {rid: r['heads'] for rid, r in unique_record_index(parent['pieces.jsonl']).items()}
    if set(sources) != {a['record_id'] for a in assignments} or set(baseline) != set(sources):
        raise ValueError('parent TRAIN coverage mismatch')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    previous.save(output / 'plan.json', {
        'schema': 'RawTimeCoveragePlan@1.0.0', 'policies': POLICIES,
        'beat_boundaries': 'starts and ends, including terminal end',
        'target_eligibility': 'source_available AND vocabulary_known; includes historical unmapped rows',
        'selection': 'fewest corpus-wide candidate positions among four policies preserving all historical-valid rows and covering all known available cadence rows; if none qualify, no default is selected',
        'train_records': len(assignments), 'split_sha256': previous.SPLIT_SHA,
        'candidate_generation_before_target_projection': True,
        'model_execution': False, 'validation_payload_accessed': False, 'test_payload_accessed': False,
    })
    totals = {h: Counter() for h in TIME_HEADS}
    variants = {p: {h: Counter() for h in TIME_HEADS} for p in POLICIES}
    dialects = {d: {p: {h: Counter() for h in TIME_HEADS} for p in POLICIES}
                for d in ('an_joint', 'dlc')}
    classes = {h: defaultdict(Counter) for h in TIME_HEADS}
    inventories = Counter()
    unchanged = {h: Counter() for h in NOTE_HEADS}
    matches = Counter()
    finished = 0
    failures = []
    with ((output / 'pieces.jsonl').open('x') as pieces,
          (output / 'sources.jsonl').open('x') as bindings,
          (output / 'uncovered.jsonl').open('x') as missing,
          (output / 'eligible_old_unmapped.jsonl').open('x') as recovered_file,
          (output / 'simultaneous_conflicts.jsonl').open('x') as conflict_file,
          gzip.open(output / 'raw_candidates.jsonl.gz', 'xt', encoding='utf-8') as raw_file):
        for a in assignments:
            rid = a['record_id']
            try:
                historical = sources[rid]
                if historical['split'] != 'train' or historical['component_id'] != a['source_component_id']:
                    raise ValueError('parent membership mismatch')
                source, relative, *_ = previous.selected_source(args.corpus, a)
                if relative != historical['source_path'] or previous.digest(source) != historical['source_sha256']:
                    raise ValueError('parent physical source mismatch')
                record = previous.bind_source(args.corpus, a)
                accepted = previous.adapter.convert_dilemmadata_record(record)
                if not isinstance(accepted, previous.adapter.DilemmadataAccepted):
                    raise ValueError('conversion rejected')
                piece = accepted.piece
                raw_sources = raw_time_sources(piece)
                candidates = candidate_time_sets(piece)
                if tuple(candidates) != POLICIES:
                    raise ValueError('unexpected candidate policy contract')
                sidecar = previous.materialize_target_sidecar(accepted, target_sidecar_version=previous.LEGACY_TARGET_SIDECAR_VERSION)
                piece_sha = sha256(previous.dumps_piece(piece).encode()).hexdigest()
                sidecar_sha = sha256(json.dumps(sidecar, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if piece_sha != historical['piece_sha256'] or sidecar_sha != historical['sidecar_sha256']:
                    raise ValueError('historical canonical or target bytes changed')
                matches.update(piece_equal=1, sidecar_equal=1)
                projected = previous.project_targets(piece, sidecar)
                inventory = {p: len(candidates[p]) for p in POLICIES}
                inventory.update(raw_note_count=len(piece.notes), historical_beat_count=len(piece.beats))
                inventories.update(inventory)
                result = {'record_id': rid, 'dialect': a['dialect'], 'inventory': inventory,
                          'heads': {}, 'variants': {p: {} for p in POLICIES}, 'unchanged_note_heads': {}}
                eligible_times = set()
                for h, rows in projected.items():
                    old_counts, _, _ = previous.summarize_rows(rows)
                    # Compare every old source/mask/error counter, not only totals.
                    verify_baseline_counts(old_counts, baseline[rid][h])
                    if h in NOTE_HEADS:
                        unchanged[h].update(old_counts)
                        result['unchanged_note_heads'][h] = old_counts
                        continue
                    if h == 'joint':
                        rows = eligible_joint(projected)
                    eligible_times.update(r['onset'] for r in rows if r['source_available'] and r['known'])
                    c, pv, pc, misses, gained, conflicts = coverage_rows(rows, candidates)
                    totals[h].update(c)
                    result['heads'][h] = c
                    for p in POLICIES:
                        variants[p][h].update(pv[p]); dialects[a['dialect']][p][h].update(pv[p])
                        result['variants'][p][h] = pv[p]
                    for label, cc in pc.items():
                        classes[h][label].update(cc)
                    for stream, events in ((missing, misses), (recovered_file, gained), (conflict_file, conflicts)):
                        for event in events:
                            stream.write(json.dumps({'record_id': rid, 'head': h, **event}, sort_keys=True) + '\n')
                result['distinct_eligible_times'] = len(eligible_times)
                result['covered_distinct_eligible_times'] = {
                    p: len(eligible_times.intersection(map(str, candidates[p]))) for p in POLICIES}
                if previous.digest(source) != historical['source_sha256']:
                    raise ValueError('source changed during run')
                raw_file.write(json.dumps({'record_id': rid, 'piece_sha256': piece_sha,
                    'schema': 'RawTimeSources@1.0.0',
                    'sources': {k: list(map(str, sorted(v))) for k, v in raw_sources.items()}}, sort_keys=True) + '\n')
                bindings.write(json.dumps(historical, sort_keys=True) + '\n')
                pieces.write(json.dumps(result, sort_keys=True) + '\n'); pieces.flush()
                finished += 1
                del accepted, piece, sidecar, projected
            except Exception as exc:
                failure = {'record_id': rid, 'error': type(exc).__name__, 'message': str(exc)}
                failures.append(failure)
                previous.save(output / 'failures.json', failures)
                print(json.dumps({'event': 'failed', **failure}), flush=True)
                break
            if finished % 25 == 0 or finished == 1:
                print(json.dumps({'event': 'progress', 'completed': finished, 'expected': len(assignments),
                    'elapsed_seconds': time.perf_counter() - started,
                    'union_root_uncovered': variants[POLICIES[-1]]['root']['uncovered_rows'],
                    'union_cadence_uncovered': variants[POLICIES[-1]]['cadence']['uncovered_rows']}), flush=True)
    qualifying, selected = select_policy(variants, inventories) if not failures else ([], None)
    summary = {'schema': 'RawTimeCoverage@1.0.0', 'status': 'failed' if failures else 'completed',
        'expected_records': len(assignments), 'completed_records': finished, 'failures': failures,
        'elapsed_seconds': time.perf_counter() - started, 'historical_hash_matches': dict(matches),
        'split_sha256': previous.SPLIT_SHA, 'inventory': dict(inventories),
        'heads': {h: dict(c) for h, c in totals.items()},
        'variants': {p: {h: dict(c) for h, c in hs.items()} for p, hs in variants.items()},
        'by_dialect': {d: {p: {h: dict(c) for h, c in hs.items()} for p, hs in ps.items()} for d, ps in dialects.items()},
        'class_counts': {h: {k: dict(c) for k, c in cs.items()} for h, cs in classes.items()},
        'unchanged_note_heads': {h: dict(c) for h, c in unchanged.items()},
        'qualifying_policies': qualifying, 'selected_policy': selected,
        'selection_status': 'candidate_only_not_a_trained_readout' if selected else 'no_lossless_policy_selected',
        'validation_payload_accessed': False, 'test_payload_accessed': False}
    previous.save(output / 'summary.json', summary)
    print(json.dumps({'event': 'finished', 'status': summary['status'], 'records': finished,
                     'selected_policy': selected, 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=Path('docs/experiments/records/EXP-TARGET-SLOT-CENSUS-002.json'))
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
