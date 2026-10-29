"""Authenticated accepted-overlay admission for immutable HIERGRAPH V3 exports.

Only metadata is composed here. Retained executable artifacts are never imported.
The exporter authenticates each declared source payload before decoding it.
"""
from copy import deepcopy
from pathlib import Path
import re

from music_critic.experiments.hiergraph_export import (
    checked_read, _json_bytes, _require, write_json, file_binding, fingerprint,
)


def apply_accepted_overlay(admission, split, rows, review_records):
    """Replace exact baseline bundles after fit, review and geometry admission."""
    result = deepcopy(admission)
    works = {w['id']: w for w in result['works']}
    assignments = {a['work_id']: a for a in split['assignments']}
    seen = set()
    for row in rows:
        identity = row['work_id']
        _require(identity in works and identity not in seen, 'unknown/duplicate accepted work')
        seen.add(identity)
        work, assignment = works[identity], assignments[identity]
        _require(assignment['split'] == 'fit' and row['assignment'] == assignment,
                 'accepted overlay requires exact fit assignment')
        _require(row['piece_id'] == work['piece_id'] and row['family'] == work['family'],
                 'overlay identity mismatch')
        assets = row['assets']
        for asset, role in [('canonical_piece', 'canonical_piece'), ('baseline_bundle', 'bundle')]:
            baseline = work['bindings'][role]
            _require(all(assets[asset][k] == baseline[k] for k in ('sha256', 'size_bytes')),
                     'overlay canonical/baseline binding mismatch')
        review = review_records.get(row['review_id'], {})
        _require(row.get('independently_reviewed') is True and review.get('status') == 'completed'
                 and review.get('record_fingerprint') == row['review_record_fingerprint'],
                 'accepted review record mismatch')
        basis = row.get('basis3', {})
        _require(all(isinstance(basis.get(k), str) and re.fullmatch('[0-9a-f]{64}', basis[k])
                     for k in ('bundle_fingerprint', 'profile_fingerprint', 'assessment_fingerprint')),
                 'three-part musical review basis required')
        for scope in ('harmonies', 'note_contexts'):
            geometry = row['delta']['structure'][scope]
            _require(geometry['shared_geometry_changes'] == 0 and geometry['removed'] == 0,
                     'accepted overlay changed existing H/h geometry')
        new_binding = dict(assets['bundle'], role='bundle', work_id=identity)
        work['bindings']['bundle'] = new_binding
        work['accepted_overlay'] = dict(review_id=row['review_id'],
            review_record_fingerprint=row['review_record_fingerprint'], basis3=deepcopy(basis),
            baseline_bundle=deepcopy(assets['baseline_bundle']))
    for index, entry in enumerate(result['entries']):
        if entry.get('work_id') in seen and entry.get('role') == 'bundle':
            result['entries'][index] = deepcopy(works[entry['work_id']]['bindings']['bundle'])
    result['overlay_accepted_works'] = sorted(seen)
    result['total_bound_bytes'] = sum(w['bindings'][r]['size_bytes']
        for w in result['works'] for r in ('canonical_piece', 'bundle'))
    return result


def prepare_overlay_export(repository, base_record, release_record, output, *, source_fallback=None):
    """Freeze exact source metadata before the existing CPU exporter is invoked.

    Caller must have a registered planned/running experiment and must enforce
    payload-access authorization. Returns exporter keyword arguments.
    """
    from music_critic.experiments.registry import validate_record
    repository, output = Path(repository).resolve(), Path(output).resolve()
    def artifact(record, role):
        matches = [a for a in record['artifacts'] if a['role'] == role]
        _require(len(matches) == 1, 'unique bound metadata role required: ' + role)
        a = matches[0]
        return dict(path=a['locator'], sha256=a['sha256'], size_bytes=a['size_bytes'])
    def read(record, role):
        return _json_bytes(checked_read(repository, artifact(record, role), 1024 * 1024))
    _require(base_record['status'] == release_record['status'] == 'completed', 'terminal parents required')
    admission, split, config = [read(base_record, r) for r in ('admission.json', 'split.json', 'config.json')]
    manifest_binding = artifact(release_record, 'manifest.jsonl')
    rows = [_json_bytes(line) for line in checked_read(repository, manifest_binding, 1024 * 1024).splitlines()]
    _require(len(rows) == release_record['results']['accepted_works'] == 15, 'expected15 accepted replacements')
    reviews = {}
    for row in rows:
        rid = row['review_id']
        _require(re.fullmatch('[A-Za-z0-9._-]+', rid), 'unsafe review identity')
        review = _json_bytes((repository/'docs/experiments/records'/f'{rid}.json').read_bytes())
        validate_record(review)
        reviews[rid] = review
    merged = apply_accepted_overlay(admission, split, rows, reviews)
    source_root = repository
    if source_fallback is not None:
        import os
        from music_critic.experiments.hiergraph_export import safe_relative
        fallback = Path(source_fallback).resolve()
        source_root = Path(os.path.commonpath([repository, fallback]))
        replacements = {}
        for work in merged['works']:
            for role in ('canonical_piece', 'bundle'):
                binding = work['bindings'][role]
                preferred = safe_relative(repository, binding['path'])
                selected = preferred if preferred.is_file() else safe_relative(fallback, binding['path'])
                _require(selected.is_file(), 'admitted payload missing in both explicit source roots')
                old = binding['path']
                binding['path'] = str(selected.relative_to(source_root))
                replacements[(work['id'], role, old)] = binding
        for index, entry in enumerate(merged['entries']):
            key = (entry.get('work_id'), entry.get('role'), entry.get('path'))
            if key in replacements:
                merged['entries'][index] = deepcopy(replacements[key])
    output.mkdir(parents=True, exist_ok=False)
    ab = write_json(output/'admission.json', merged)
    source_directory = (repository/artifact(base_record, 'split.json')['path']).parent
    for binding in split['parent_manifests']:
        raw = checked_read(source_directory, binding, 1024*1024)
        destination = output/binding['path']; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream: stream.write(raw)
    raw_split = checked_read(repository, artifact(base_record, 'split.json'), 1024*1024)
    with (output/'split.json').open('xb') as stream: stream.write(raw_split)
    sb = file_binding(output/'split.json')
    config.update(admission_sha256=ab['sha256'], split_sha256=sb['sha256'])
    config['caps'].update(total_wall_seconds=1800, worker_rss_bytes=6*1024**3,
                          max_input_bytes=merged['total_bound_bytes'])
    cb = write_json(output/'config.json', config)
    provenance = dict(schema='HiergraphOverlayExportPreparation@1',
        parent_export=base_record['experiment_id'], accepted_release=release_record['experiment_id'],
        accepted_manifest=manifest_binding, accepted_works=sorted(merged['overlay_accepted_works']),
        unchanged_split_fingerprint=split['fingerprint'], metadata=dict(admission=ab, split=sb, config=cb),
        source_bindings={w['id']: w['bindings'] for w in merged['works']},
        accepted_review_basis={w['id']: w['accepted_overlay'] for w in merged['works'] if 'accepted_overlay' in w})
    provenance['fingerprint'] = fingerprint(provenance)
    write_json(output/'provenance.json', provenance)
    return dict(admission_path=output/'admission.json', admission_sha256=ab['sha256'],
        split_path=output/'split.json', split_sha256=sb['sha256'],
        config_path=output/'config.json', config_sha256=cb['sha256'], source_root=source_root)
