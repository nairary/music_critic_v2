"""Compile individually authored source roles into canonical MIDI targets.

No prose interpretation occurs here. Source annotations, canonical pieces and
alignment evidence are authenticated before decoding, and outputs are new files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re


def _fraction(value):
    if not isinstance(value, str):
        raise ValueError('time must be an exact fraction string')
    return Fraction(value)


def _text(value):
    value = Fraction(value)
    return f'{value.numerator}/{value.denominator}'


def read_bound_bytes(binding):
    data = Path(binding['path']).read_bytes()
    if sha256(data).hexdigest() != binding['sha256']:
        raise ValueError(f"SHA-256 mismatch: {binding['path']}")
    if len(data) != binding['size_bytes']:
        raise ValueError(f"size mismatch: {binding['path']}")
    return data


def read_bound_json(binding):
    return json.loads(read_bound_bytes(binding))


def resolve_role_assignment(assignment, bindings):
    """Map one role to one note; tied source segments may jointly cover it."""
    index = {row['source_event_id']: row for row in bindings}
    if len(index) != len(bindings):
        raise ValueError('duplicate source binding IDs')
    ids = assignment['source_event_ids']
    related = assignment['related_source_event_ids']
    if not ids or len(ids) != len(set(ids)) or len(related) != len(set(related)):
        raise ValueError('assignment requires unique source IDs')
    for event_id in (*ids, *related):
        if event_id not in index or not index[event_id]['binding_available']:
            raise ValueError(f'unavailable source binding: {event_id}')
    rows = [index[event_id] for event_id in ids]
    notes = {row['canonical_note_id'] for row in rows}
    if len(notes) != 1 or None in notes:
        raise ValueError('one role assignment must identify exactly one canonical note')
    interval = assignment['interval']
    if interval['time_unit'] != 'quarter_note':
        raise ValueError('canonical targets require exact quarter-note time')
    start, end = _fraction(interval['start']), _fraction(interval['end'])
    if start >= end:
        raise ValueError('role interval must be positive')
    offsets = set()
    spans = []
    for row in rows:
        a, b = _fraction(row['source_start_qn']), _fraction(row['source_end_qn'])
        ca, cb = _fraction(row['canonical_start_qn']), _fraction(row['canonical_end_qn'])
        if b <= a or cb - ca != b - a:
            raise ValueError('source/canonical segment duration mismatch')
        offsets.add(ca - a)
        spans.append((a, b))
    if len(offsets) != 1:
        raise ValueError('inconsistent source/canonical time offsets')
    covered_to = start
    for a, b in sorted(spans):
        if a <= covered_to:
            covered_to = max(covered_to, b)
    if covered_to < end or not any(a <= start < b for a, b in spans):
        raise ValueError('source event coverage does not contain assignment interval')
    offset = next(iter(offsets))
    return dict(note_id=next(iter(notes)), start_qn=_text(start + offset),
                end_qn=_text(end + offset),
                related_note_ids=sorted({index[i]['canonical_note_id'] for i in related}))


def _dump_new(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def file_binding(path):
    data = path.read_bytes()
    return dict(path=str(path), sha256=sha256(data).hexdigest(), size_bytes=len(data))


def materialize_release(manifest, output_dir):
    """Build all bundles before publishing a new release directory."""
    from music_critic.analysis.target_binding import bind_source_events
    from music_critic.data.serialization import loads_piece
    from music_critic.data import RationalTime
    from music_critic.tasks.relational_targets import (
        RelationalTargetAssignment, build_relational_target_bundle,
        dumps_relational_assignments,
    )
    from music_critic.tasks.multisource import dumps_target_bundle, loads_target_bundle

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'release directory already exists: {output_dir}')
    if manifest['schema'] != 'RelationalMIDITargetInputs@1':
        raise ValueError('unsupported input manifest schema')
    source = read_bound_json(manifest['source_assignments'])
    if source['schema'] != 'RelationalSourceRoleTargets@1':
        raise ValueError('unsupported source role schema')
    source_assignments = source['assignments']
    if len({a['assignment_id'] for a in source_assignments}) != len(source_assignments):
        raise ValueError('duplicate assignment IDs')
    cases = [w['case_id'] for w in manifest['works']]
    if len(cases) != len(set(cases)) or not set(a['case_id'] for a in source_assignments) <= set(cases):
        raise ValueError('duplicate or unknown work in source assignments')
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*', case) for case in cases):
        raise ValueError('unsafe case ID')
    prepared = []
    for work in manifest['works']:
        admission = read_bound_json(work['admission'])
        if (admission['current_split'] != 'train'
                or admission['known_family_heldout_conflict'] is not False
                or admission['record_id'] != work['record_id']):
            raise ValueError('source admission does not authorize TRAIN target binding')
        # Authentication of the metadata observed at admission precedes music.
        for metadata_binding in admission['metadata_bindings']:
            read_bound_bytes(metadata_binding)
        packet = read_bound_json(work['source_events'])
        read_bound_bytes(packet['source_table'])
        piece = loads_piece(read_bound_bytes(work['canonical_piece']))
        alignment = read_bound_json(work['raw_alignment'])
        bindings = bind_source_events(
            piece, alignment, packet, record_id=work['record_id'],
            source_sha256=work['source_sha256'], current_split='train',
            source_to_canonical_offset_qn=work.get('source_to_canonical_offset_qn', '0/1'),
        )
        rows = [asdict(b) for b in bindings]
        assignments = []
        for authored in source_assignments:
            if authored['case_id'] != work['case_id']:
                continue
            if authored['source_version_id'] != packet['source_version_id']:
                raise ValueError('source role belongs to a different source version')
            if (authored['available'] is not True or authored['origin_status'] != 'admitted_weak'
                    or authored['target_value'] != 'present' or authored['source'] != 'pseudo_label'
                    or authored['confidence'] is not None or authored['human_gold'] is not False):
                raise ValueError('release only accepts explicitly admitted weak positives')
            resolved = resolve_role_assignment(authored, rows)
            provenance = authored['annotation_provenance']
            assignments.append(RelationalTargetAssignment(
                assignment_id=authored['assignment_id'], dataset_id=piece.dataset_name,
                piece_id=piece.piece_id, note_id=resolved['note_id'], flag=authored['flag'],
                start_qn=RationalTime.from_fraction(_fraction(resolved['start_qn'])),
                end_qn=RationalTime.from_fraction(_fraction(resolved['end_qn'])),
                relation_id=authored['relation_id'], source_event_ids=tuple(authored['source_event_ids']),
                related_note_ids=tuple(resolved['related_note_ids']),
                source_version_id=authored['source_version_id'], annotation_origin='astra_inference',
                annotation_status='admitted_weak', admitted_weak=True,
                model_requested=provenance['model_requested'],
                reasoning_effort=provenance['reasoning_effort_requested'],
            ))
        if piece.split not in (None, 'train'):
            raise ValueError('embedded canonical split conflicts with TRAIN admission')
        # Older raw archives omit split. Use authenticated current membership
        # only in this target-building view; never rewrite source piece bytes.
        target_piece = replace(piece, split=admission['current_split'])
        bundle = build_relational_target_bundle(target_piece, assignments, analysis_view_id='astra_ultra_relations_v1')
        serialized = dumps_target_bundle(bundle, indent=2)
        if loads_target_bundle(serialized) != bundle:
            raise ValueError('target bundle roundtrip mismatch')
        positives = {t.task_id: sum(t.availability_mask) for t in bundle.targets}
        prepared.append((work, piece, rows, assignments, serialized, positives))
    if not sum(sum(x[5].values()) for x in prepared):
        raise ValueError('release has no usable targets')
    output_dir.mkdir(parents=True)
    released = []
    for work, piece, rows, assignments, serialized, positives in prepared:
        directory = output_dir / work['case_id']; directory.mkdir()
        (directory / 'target-bundle.json').write_text(serialized + '\n')
        (directory / 'event-assignments.json').write_text(dumps_relational_assignments(assignments, indent=2) + '\n')
        _dump_new(directory / 'source-bindings.json', rows)
        released.append(dict(case_id=work['case_id'], piece_id=piece.piece_id,
                             canonical_piece=work['canonical_piece'], split='train',
                             archived_piece_split=piece.split, admission=work['admission'],
                             target_bundle=file_binding(directory / 'target-bundle.json'),
                             event_assignments=file_binding(directory / 'event-assignments.json'),
                             source_bindings=file_binding(directory / 'source-bindings.json'),
                             canonical_note_count=len(piece.notes), source_event_count=len(rows),
                             assignment_count=len(assignments), available_by_task=positives))
    result = dict(schema='RelationalMIDITargetRelease@1', experiment_id=manifest['experiment_id'],
                  works=released, source_assignments=manifest['source_assignments'],
                  exclusions=manifest.get('exclusions', []),
                  total_assignments=sum(w['assignment_count'] for w in released),
                  total_available_note_flags=sum(sum(w['available_by_task'].values()) for w in released),
                  supervision='Astra weak positive-unlabeled; not independent human gold',
                  source='pseudo_label', confidence=None,
                  projection='Flag occurs in a subinterval of note; exact times/relations retained in event-assignments')
    _dump_new(output_dir / 'index.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    result = materialize_release(json.loads(args.manifest.read_text()), args.output_dir)
    print(json.dumps({k: result[k] for k in ('total_assignments', 'total_available_note_flags')}))


if __name__ == '__main__':
    main()
