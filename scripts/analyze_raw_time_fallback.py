"""Offline extension of a completed raw-time census; no corpus payload reads.

Compare fixed beat-boundary subdivisions as supersets of the full raw union,
and preserve old harmonic support with separate legacy beat fallback slots.
This is structural target coverage/error accounting, never a model metric.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import time

from scripts.audit_harmonic_target_slots import digest, save

FULL_UNION = 'note_onsets_offsets_beats'
SUBDIVISIONS = (2, 3, 4, 6, 12)


def with_subdivisions(sources, divisions):
    """Fixed divisions of adjacent raw beat-boundary intervals, including gaps.

    These intervals are observable from raw timing; they are not inferred
    chord spans. Preserve raw on/off positions outside the beat boundary span.
    """
    if type(divisions) is not int or divisions not in SUBDIVISIONS:
        raise ValueError('undeclared subdivision count')
    boundaries = sorted(sources['beat_boundaries'])
    result = set().union(*sources.values())
    for left, right in zip(boundaries, boundaries[1:]):
        result.update(left + (right - left) * Fraction(j, divisions)
                      for j in range(1, divisions))
    return frozenset(result)


def route_uncovered(rows, covered_times, *, allow_fallback):
    """Rows are only the original union misses; old valid certifies old slot."""
    groups = defaultdict(Counter)
    counts = Counter()
    for row in rows:
        if not row['source_available'] or not row['known'] or row['masked']:
            raise ValueError('uncovered artifact contains an ineligible row')
        label = tuple(row['label']) if isinstance(row['label'], list) else row['label']
        if Fraction(row['onset']) in covered_times:
            slot = ('time', row['onset'])
            counts['additional_fine_rows'] += 1
        elif allow_fallback and row['valid'] and row['mapped'] and row['slot'] is not None:
            slot = ('legacy_beat', row['slot'])
            counts['fallback_rows'] += 1
        else:
            counts['unmatched_rows'] += 1
            continue
        groups[slot][label] += 1
    for (route, _), labels in groups.items():
        errors = sum(labels.values()) - max(labels.values())
        counts[route + '_unavoidable_errors'] += errors
        counts[route + '_conflict_slots'] += int(errors > 0)
    return dict(counts)


def verified_parent(record_path):
    from music_critic.experiments.registry import validate_record
    record = json.loads(record_path.read_text())
    validate_record(record)
    if record['experiment_id'] != 'EXP-RAW-TIME-COVERAGE-001' or record['status'] != 'completed':
        raise ValueError('requires completed first census')
    artifacts = {a['locator']: a for a in record['artifacts']}
    root = Path('outputs/raw_time_coverage_001/run')
    for name in ('summary.json', 'raw_candidates.jsonl.gz', 'uncovered.jsonl'):
        p = root / name; ref = artifacts[p.as_posix()]
        if digest(p) != ref['sha256'] or p.stat().st_size != ref['size_bytes']:
            raise ValueError('parent artifact changed')
    verification_path = root.parent / 'verification.json'
    ref = artifacts[verification_path.as_posix()]
    if digest(verification_path) != ref['sha256']:
        raise ValueError('parent verification changed')
    verification = json.loads(verification_path.read_text())
    if verification['status'] != 'passed' or verification['verified_records'] != 1295:
        raise ValueError('parent not independently verified')
    return root


def run(args):
    started = time.perf_counter()
    root = verified_parent(args.parent)
    baseline = json.loads((root / 'summary.json').read_text())
    if (baseline['schema'] != 'RawTimeCoverage@1.0.0' or baseline['status'] != 'completed'
            or baseline['expected_records'] != baseline['completed_records'] or baseline['completed_records'] != 1295
            or baseline['failures']):
        raise ValueError('incomplete parent')
    missing = defaultdict(lambda: defaultdict(list))
    for line in (root / 'uncovered.jsonl').read_text().splitlines():
        r = json.loads(line); missing[r['record_id']][r['head']].append(r)
    methods = ('raw_union', *(f'raw_union_subdivide_{n}' for n in SUBDIVISIONS))
    counts = {m: {h: Counter() for h in baseline['heads']} for m in methods}
    inventory = Counter()
    seen = set()
    args.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(root / 'raw_candidates.jsonl.gz', 'rt', encoding='utf-8') as stream, (args.output / 'pieces.jsonl').open('x') as out:
        for line in stream:
            item = json.loads(line); rid = item['record_id']
            if rid in seen:
                raise ValueError('duplicate candidate record')
            seen.add(rid)
            sources = {k: frozenset(map(Fraction, v)) for k, v in item['sources'].items()}
            policies = {'raw_union': frozenset().union(*sources.values()),
                        **{f'raw_union_subdivide_{n}': with_subdivisions(sources, n) for n in SUBDIVISIONS}}
            result = {'record_id': rid, 'policies': {}}
            for m, candidates in policies.items():
                inventory[m] += len(candidates)
                by_head = {}
                for h in baseline['heads']:
                    c = route_uncovered(missing[rid][h], candidates, allow_fallback=h != 'cadence')
                    counts[m][h].update(c)
                    by_head[h] = c
                result['policies'][m] = {'fine_positions': len(candidates), 'additional_routes': by_head}
            out.write(json.dumps(result, sort_keys=True) + '\n')
    if len(seen) != 1295 or not set(missing) <= seen:
        raise ValueError('candidate coverage mismatch')
    verified_parent(args.parent)  # Input hashes must remain fixed throughout the analysis.
    result = {}
    for m in methods:
        result[m] = {}
        for h, base in baseline['variants'][FULL_UNION].items():
            c = counts[m][h]
            result[m][h] = {
                'eligible_rows': baseline['heads'][h]['eligible_rows'],
                'historical_valid_rows': baseline['heads'][h]['historical_valid_rows'],
                'fine_rows': base.get('covered_rows', 0) + c['additional_fine_rows'],
                'fallback_rows': c['fallback_rows'], 'unmatched_rows': c['unmatched_rows'],
                'fine_exact_errors': base.get('covered_exact_errors', 0) + c['time_unavoidable_errors'],
                'fallback_errors': c['legacy_beat_unavoidable_errors'],
                'matched_slot_error_floor': base.get('covered_exact_errors', 0) + c['time_unavoidable_errors'] + c['legacy_beat_unavoidable_errors'],
            }
            r = result[m][h]
            if r['eligible_rows'] != r['fine_rows'] + r['fallback_rows'] + r['unmatched_rows']:
                raise ValueError('route coverage conservation failure')
            r['full_support'] = r['unmatched_rows'] == 0
            r['all_eligible_error_floor'] = r['matched_slot_error_floor'] if r['full_support'] else None
    summary = {'schema': 'RawTimeFallbackAnalysis@1.0.0', 'status': 'completed',
        'records': len(seen), 'elapsed_seconds': time.perf_counter() - started,
        'methods': result, 'fine_position_counts': dict(inventory),
        'existing_beat_positions': baseline['inventory']['historical_beat_count'],
        'fallback_candidate_requirement': 'all raw beat IDs must have outputs before reading any target; no gold-driven selection of fallback candidates',
        'candidate_choice': 'raw_union plus preserved legacy harmonic beat fallback; subdivisions are diagnostic alternatives',
        'all_counts': 'native TRAIN target rows; source masks and classes unchanged; not model predictions',
        'subdivision_scope': 'intervals between adjacent raw beat boundary timestamps, including gaps',
        'validation_payload_accessed': False, 'test_payload_accessed': False,
        'corpus_payload_read': False, 'model_execution': False}
    save(args.output / 'summary.json', summary)
    print(json.dumps({'status': 'completed', 'records': len(seen), 'elapsed_seconds': summary['elapsed_seconds'],
                      'fine_positions': dict(inventory), 'root_routes': result['raw_union']['root']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('docs/experiments/records/EXP-RAW-TIME-COVERAGE-001.json'))
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
