"""Project existing identity catalogs and split metadata; never open music files."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time


def identity_values(value):
    if value is None:
        return []
    if isinstance(value, str) and value.startswith('['):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        if isinstance(parsed, list):
            return [v for item in parsed for v in identity_values(item)]
    text = str(value).strip()
    return [] if text.casefold() in {'', 'na', 'n/a', 'none', 'null', 'nan', '<na>'} else [text]


def file_state(root, relative):
    if not relative:
        return {'path': None, 'status': 'unspecified'}
    path = Path(relative)
    root = Path(root).resolve()
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe_catalog_file_path')
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError('catalog_file_symlink_escape')
    return {'path': str(path), 'status': 'present' if resolved.is_file() else 'missing',
            'method': 'stat_only_no_content_read'}


def membership(source_id, native, index, splits):
    record = index.get(source_id, {})
    piece_id = record.get('piece_id')
    assigned = splits.get(piece_id, {})
    current = assigned.get('split')
    normalize = lambda s: {'val': 'validation', 'valid': 'validation'}.get(str(s).lower(), str(s).lower())
    return {'source_id': source_id, 'piece_id': piece_id, 'native_split': native,
            'current_split': current, 'component_fingerprint': assigned.get('component_fingerprint'),
            'native_current_conflict': normalize(native) != normalize(current) if native and current else None}


def unique_index(rows, key):
    result = {}
    for row in rows:
        identifier = row[key]
        if identifier in result:
            raise ValueError('duplicate_metadata_identity:' + str(identifier))
        result[identifier] = row
    return result


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(path, repo):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.relative_to(repo)), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def run(repo, output):
    repo = Path(repo).resolve()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = {
        'hooktheory': repo / 'outputs/corpus_overlap_20261005_001/hooktheory-catalog.jsonl',
        'pop909': repo / 'outputs/corpus_overlap_20261005_001/pop909-catalog.jsonl',
        'pdmx': repo / 'data/pdmx/PDMX.csv',
        'global': repo / 'data/cache/global.split.json',
        'hook_index': repo / 'data/cache/hooktheory.index.json',
        'pop_index': repo / 'data/cache/pop909_cl.index.json',
        'dilemma_components': repo / 'outputs/research_resume_20260913/EXP-RESEARCH-DILEMMA-BINDINGS-001/metadata/source_components.jsonl',
        'dilemma_split': repo / 'outputs/research_resume_20260913/EXP-RESEARCH-DILEMMA-BINDINGS-001/metadata/split_assignments.jsonl',
        'dilemma_raw': repo / 'outputs/research_resume_20260913/EXP-RESEARCH-DILEMMA-BINDINGS-001/metadata/full_raw_manifest.json',
    }
    inputs = [fingerprint(path, repo) for path in paths.values()]
    # Authenticate reused metadata projections against the prior sealed audit.
    prior = json.loads((repo / 'docs/experiments/records/EXP-HT-POP-PDMX-OVERLAP-001.json').read_text())
    for key in ('hooktheory', 'pop909'):
        observed = fingerprint(paths[key], repo)
        expected = next(a for a in prior['artifacts'] if a['locator'] == observed['path'])
        if (observed['sha256'], observed['size_bytes']) != (expected['sha256'], expected['size_bytes']):
            raise ValueError('retained_catalog_fingerprint_mismatch:' + key)
    splits = unique_index(json.loads(paths['global'].read_text())['assignments'], 'piece_id')
    indices = {name: unique_index(json.loads(paths[key].read_text())['records'], 'source_identity')
               for name, key in [('hooktheory', 'hook_index'), ('pop909', 'pop_index')]}
    counts = {}

    def emit(name, rows):
        coverage = Counter()
        seen = set()
        with (output / (name + '.jsonl')).open('x') as stream:
            for row in rows:
                if row['id'] in seen:
                    raise ValueError('duplicate_catalog_id:' + row['id'])
                seen.add(row['id'])
                coverage['records'] += 1
                coverage['with_title'] += bool(row['titles'])
                coverage['with_creator'] += bool(row['creators'])
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        counts[name] = dict(coverage)

    def hook():
        for row in read_jsonl(paths['hooktheory']):
            members = []
            for clip in row.get('source_clips', []):
                member = membership(clip['clip_id'], clip.get('metadata', {}).get('split'), indices['hooktheory'], splits)
                cached = indices['hooktheory'].get(clip['clip_id'], {}).get('canonical_relative_path')
                member['canonical_availability'] = file_state(repo / 'data/cache/hooktheory/multisource-cache-v1.0.0', cached)
                members.append(member)
            row['membership'] = {'scope': 'individual_clips_not_whole_composition', 'records': members}
            yield row
    emit('hooktheory', hook())

    def pop():
        for row in read_jsonl(paths['pop909']):
            source_id = str(row['id']).split(':')[-1].zfill(3)
            row['membership'] = membership(source_id, None, indices['pop909'], splits)
            row['membership']['scope'] = 'original_POP909_identity_to_CL_source_identity; no_note_alignment'
            cached = indices['pop909'].get(source_id, {}).get('canonical_relative_path')
            row['availability'] = file_state(repo / 'data/cache/pop909_cl/multisource-cache-v1.0.0', cached)
            yield row
    emit('pop909', pop())

    def pdmx():
        with paths['pdmx'].open(newline='') as stream:
            for row in csv.DictReader(stream):
                yield {'dataset': 'pdmx', 'id': row['path'],
                       'titles': [{'field': k, 'value': v} for k in ('song_name', 'title') for v in identity_values(row.get(k))],
                       'creators': [{'field': k, 'value': v} for k in ('artist_name', 'composer_name') for v in identity_values(row.get(k))],
                       'source_metadata': {k: row.get(k) for k in ('path', 'song_name', 'title', 'artist_name', 'composer_name', 'best_path', 'best_arrangement', 'best_unique_arrangement', 'mid', 'mxl', 'license', 'license_url', 'license_conflict')},
                       'membership': {'current_split': None, 'reason': 'pdmx_not_in_current_global_split_manifest'},
                       'availability': {'status': 'not_checked', 'reason': 'stat_only_check_reserved_for_matched_candidates'}}
    emit('pdmx', pdmx())

    components = read_jsonl(paths['dilemma_components'])
    evidence = unique_index([e for c in components for e in c['source_evidence']], 'record_id')
    assignments = unique_index(read_jsonl(paths['dilemma_split']), 'record_id')
    ids = json.loads(paths['dilemma_raw'].read_text())['record_ids']
    def dilemma():
        for record_id in ids:
            e = evidence.get(record_id, {})
            split = assignments.get(record_id)
            fields = record_id.split(':')
            relative = ('pitch_arrays/AN/' + fields[1] + '/' + fields[2] + '_joint.tsv'
                        if fields[0] == 'an' else 'pitch_arrays/DLC/' + fields[1] + '/' + fields[2] + '.tsv')
            yield {'dataset': 'dilemmadata', 'id': record_id,
                   'titles': [{'field': k, 'value': v} for k in ('title', 'movement') for v in identity_values(e.get(k))],
                   'creators': [{'field': 'composer', 'value': v} for v in identity_values(e.get('composer'))],
                   'source_metadata': dict(e, metadata_snapshot=str(paths['dilemma_components'].relative_to(repo))),
                   'membership': {'saved_profile_split': split['split'] if split else None,
                                  'split_evidence': str(paths['dilemma_split'].relative_to(repo)),
                                  'native_split': fields[1] if fields[0] == 'an' else None,
                                  'current_global_split': None, 'in_saved_paper_candidate_profile': split is not None},
                   'availability': file_state(repo / 'data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75', relative)}
    emit('dilemmadata', dilemma())
    receipt = {'inputs': inputs, 'catalogs': counts, 'elapsed_seconds': time.monotonic() - started,
               'access': 'identity metadata, index metadata/counters, split manifests and stat only; no music file content',
               'limitations': ['Dilemmadata saved profile from 2026-09-13; current global manifest only covers HookTheory/POP909.',
                               'Raw inventory includes excluded duplicate records; unknown title is unsearchable, not no overlap.',
                               'PDMX candidate musical-file existence is checked separately.']}
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo, args.output), ensure_ascii=False, indent=2))
