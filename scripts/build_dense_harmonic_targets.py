"""Publish exact dense TRAIN targets and independently attributed context roles.

All musical inputs are hash checked before decoding. A release is write-once;
the canonical piece and the older sparse role release remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re


def bound_bytes(binding):
    raw = Path(binding['path']).read_bytes()
    if sha256(raw).hexdigest() != binding['sha256'] or len(raw) != binding['size_bytes']:
        raise ValueError(f"input hash/size mismatch: {binding['path']}")
    return raw


def bound_json(binding):
    return json.loads(bound_bytes(binding))


def binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


def fraction(value):
    if (not isinstance(value, dict) or type(value.get('num')) is not int
            or type(value.get('den')) is not int or value['den'] <= 0):
        raise ValueError('expected exact rational time')
    result = Fraction(value['num'], value['den'])
    if (result.numerator, result.denominator) != (value['num'], value['den']):
        raise ValueError('rational time must be reduced')
    return result


def validate_context_assignments(piece, packet, assignments):
    """Preserve partial intervals; never broadcast positives over a whole note."""
    notes = {note.note_id: (i, note) for i, note in enumerate(piece.notes)}
    harmonies = {h['id']: h for h in packet['harmonies']}
    source_segments = {s['source_event_id']: s for s in packet['note_segments']}
    output, seen = [], set()
    for assignment in assignments:
        aid = assignment['assignment_id']
        if aid in seen:
            raise ValueError('duplicate context assignment')
        seen.add(aid)
        if assignment['note_id'] not in notes or not set(assignment['related_note_ids']) <= notes.keys():
            raise ValueError('unknown canonical participant')
        if assignment['source_version_id'] != packet['source_version_id']:
            raise ValueError('context source version mismatch')
        if assignment['canonical_piece_id'] != piece.piece_id:
            raise ValueError('context canonical piece identity mismatch')
        if assignment['human_gold'] is not False or assignment['confidence'] is not None:
            raise ValueError('unverified context cannot claim gold/confidence')
        status, origin = assignment['status'], assignment['origin']
        if status not in ('admitted_weak', 'candidate', 'unknown'):
            raise ValueError('unsupported context status')
        if origin not in ('astra_individual_context', 'algorithmic_candidate'):
            raise ValueError('unsupported context origin')
        if status == 'admitted_weak' and origin != 'astra_individual_context':
            raise ValueError('algorithmic candidates cannot be admitted Astra labels')
        admitted = status == 'admitted_weak'
        if (assignment['target_available'] is not admitted
                or assignment['target_value'] != ('present' if admitted else None)):
            raise ValueError('context target availability/status mismatch')
        if not isinstance(assignment['role'], str) or not assignment['role'] or not assignment['evidence']:
            raise ValueError('context role and evidence required')
        start, end = fraction(assignment['start_qn']), fraction(assignment['end_qn'])
        index, note = notes[assignment['note_id']]
        onset, offset = note.onset_qn.to_fraction(), (note.onset_qn + note.duration_qn).to_fraction()
        if not onset <= start < end <= offset:
            raise ValueError('context interval outside canonical note')
        source_ids = assignment['source_event_ids']
        if not source_ids or len(set(source_ids)) != len(source_ids) or not set(source_ids) <= source_segments.keys():
            raise ValueError('missing or unknown context source event')
        selected_segments = [source_segments[sid] for sid in source_ids]
        if any(s['note_id'] != assignment['note_id'] for s in selected_segments):
            raise ValueError('context source event belongs to a different canonical note')
        if admitted and any(s['binding_available'] is not True for s in selected_segments):
            raise ValueError('context source binding unavailable')
        source_spans = sorted((fraction(s['start_qn']), fraction(s['end_qn'])) for s in selected_segments)
        source_covered = start
        for left, right in source_spans:
            if left <= source_covered:
                source_covered = max(source_covered, right)
        if source_covered < end or not any(left <= start < right for left, right in source_spans):
            raise ValueError('context source events do not cover assignment')
        ids = assignment['source_harmony_ids']
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= harmonies.keys():
            raise ValueError('missing or unknown context harmony')
        spans = sorted((fraction(harmonies[h]['start_qn']), fraction(harmonies[h]['end_qn'])) for h in ids)
        covered = start
        for left, right in spans:
            if left <= covered:
                covered = max(covered, right)
        if covered < end or not any(left <= start < right for left, right in spans):
            raise ValueError('context harmony does not cover assignment')
        output.append({**assignment, 'canonical_note_index': index,
                       'available': status == 'admitted_weak'})
    return output


def numeric_context(assignments, roles):
    """One positive-unlabeled categorical row per authored exact interval."""
    return dict(
        role_vocabulary=list(roles),
        assignment_ids=[a['assignment_id'] for a in assignments],
        canonical_note_indices=[a['canonical_note_index'] for a in assignments],
        intervals=[[a['start_qn'], a['end_qn']] for a in assignments],
        values=[roles.index(a['role']) if a['available'] else -1 for a in assignments],
        availability_mask=[a['available'] for a in assignments],
        origins=[a['origin'] for a in assignments],
        unassigned_semantics='unknown; no inferred negative examples',
    )


def dump_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
        handle.write('\n')


def materialize_release(manifest, destination):
    from music_critic.data import loads_piece
    from music_critic.tasks.dense_harmonic_targets import (
        build_dense_harmonic_targets, collate_dense_harmonic_targets,
        dumps_dense_harmonic_targets, loads_dense_harmonic_targets,
    )

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('release already exists')
    if manifest['schema'] != 'DenseHarmonicTargetInputs@1':
        raise ValueError('unsupported dense input schema')
    source = bound_json(manifest['context_assignments'])
    if (source['schema'] != 'DenseContextRoleAssignments@1'
            or source['model_requested'] != 'gpt-6-astra'
            or source['reasoning_effort_requested'] != 'ultra'):
        raise ValueError('unexpected context annotation provenance')
    cases = [work['case_id'] for work in manifest['works']]
    if len(set(cases)) != len(cases) or any(not re.fullmatch('[a-z0-9][a-z0-9-]*', case) for case in cases):
        raise ValueError('duplicate or unsafe case IDs')
    assignments = source['assignments']
    if (not {a['case_id'] for a in assignments} <= set(cases)
            or len({a['assignment_id'] for a in assignments}) != len(assignments)):
        raise ValueError('unknown case or duplicate assignment')
    roles = source.get('role_vocabulary', sorted({a['role'] for a in assignments}))
    if len(set(roles)) != len(roles) or not {a['role'] for a in assignments} <= set(roles):
        raise ValueError('invalid context role vocabulary')
    prepared = []
    for work in manifest['works']:
        admission = bound_json(work['admission'])
        if (admission['record_id'] != work['record_id'] or admission['current_split'] != 'train'
                or admission['known_family_heldout_conflict'] is not False):
            raise ValueError('work lacks TRAIN admission')
        for metadata in admission['metadata_bindings']:
            bound_bytes(metadata)
        packet = bound_json(work['source_packet'])
        if packet['canonical_piece'] != work['canonical_piece']:
            raise ValueError('source packet canonical binding mismatch')
        expected = admission['source_expected']
        if (packet['record_id'] != work['record_id'] or packet['case_id'] != work['case_id']
                or expected['record_id'] != work['record_id']
                or packet['source_sha256'] != expected['sha256']
                or packet['source_version_id'] != f"{work['record_id']}@sha256:{expected['sha256']}"):
            raise ValueError('source packet admission identity mismatch')
        manifests = [entry for entry in admission['metadata_bindings']
                     if Path(entry['path']).name == 'dilemma-data-manifest.json']
        if len(manifests) != 1:
            raise ValueError('current split manifest is required')
        matches = [row for row in bound_json(manifests[0])['records'] if row['record_id'] == work['record_id']]
        if (len(matches) != 1 or matches[0]['split'] != 'train'
                or matches[0]['source_component_id'] != expected['component_id']
                or matches[0]['piece']['sha256'] != work['canonical_piece']['sha256']
                or matches[0]['piece']['size_bytes'] != work['canonical_piece']['size_bytes']):
            raise ValueError('current TRAIN source/piece membership mismatch')
        piece = loads_piece(bound_bytes(work['canonical_piece']))
        if piece.split not in (None, 'train'):
            raise ValueError('canonical split conflicts with TRAIN admission')
        bundle = build_dense_harmonic_targets(piece, packet)
        payload = dumps_dense_harmonic_targets(bundle)
        if loads_dense_harmonic_targets(payload, piece=piece) != bundle:
            raise ValueError('dense serialization roundtrip failed')
        authored = validate_context_assignments(piece, packet, [
            a for a in assignments if a['case_id'] == work['case_id']])
        # Reopening the old bundle authenticates preservation, not new inference.
        previous = bound_json(work['previous_event_assignments'])
        prepared.append((work, piece, bundle, authored, previous))
    batch = collate_dense_harmonic_targets([(piece, bundle) for _, piece, bundle, _, _ in prepared])
    if not any(bool(mask.any()) for mask in batch.masks.values()):
        raise ValueError('dense release has no available numerical targets')
    destination.mkdir(parents=True)
    released = []
    for work, piece, bundle, authored, previous in prepared:
        folder = destination / work['case_id']
        folder.mkdir()
        dump_new(folder / 'dense-targets.json', bundle)
        numeric = collate_dense_harmonic_targets([(piece, bundle)])
        dump_new(folder / 'dense-numeric.json', dict(
            schema='DenseHarmonicNumeric@1', piece_id=piece.piece_id,
            dense_targets=binding(folder / 'dense-targets.json'),
            field_vocabularies=bundle['field_vocabularies'],
            row_ids=list(numeric.row_ids),
            canonical_note_indices=numeric.local_note_indices.tolist(),
            values={key: value.tolist() for key, value in numeric.values.items()},
            availability_masks={key: value.tolist() for key, value in numeric.masks.items()},
        ))
        dump_new(folder / 'context-assignments.json', {'assignments': authored})
        dump_new(folder / 'context-numeric.json', numeric_context(authored, roles))
        dump_new(folder / 'previous-event-assignments.json', previous)
        available = Counter()
        unknown = Counter()
        for row in bundle['rows']:
            for field, value in row['fields'].items():
                if value['available']:
                    available[field] += 1
                else:
                    unknown[f"{field}:{value['reason']}"] += 1
        released.append({**work, 'piece_id': piece.piece_id, 'canonical_note_count': len(piece.notes),
                         'row_count': len(bundle['rows']), 'available_by_field': dict(available),
                         'unknown_by_reason': dict(unknown),
                         'context_admitted_count': sum(a['available'] for a in authored),
                         'context_unavailable_count': sum(not a['available'] for a in authored),
                         'dense_targets': binding(folder / 'dense-targets.json'),
                         'dense_numeric': binding(folder / 'dense-numeric.json'),
                         'context_assignments': binding(folder / 'context-assignments.json'),
                         'context_numeric': binding(folder / 'context-numeric.json'),
                         'previous_event_assignments_copy': binding(folder / 'previous-event-assignments.json')})
    result = dict(schema='DenseHarmonicTargetRelease@1', experiment_id=manifest['experiment_id'],
                  works=released, context_role_vocabulary=roles,
                  context_source=manifest['context_assignments'],
                  numeric_batch_rows=len(batch.row_ids),
                  numeric_available_by_field={k: int(v.sum().item()) for k, v in batch.masks.items()},
                  canonical_notes=sum(w['canonical_note_count'] for w in released),
                  semantics='auxiliary interval targets; source-derived facts and separate Astra weak roles; no quality score')
    dump_new(destination / 'index.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = materialize_release(json.loads(args.manifest.read_text()), args.destination)
    print(json.dumps({key: result[key] for key in ('canonical_notes', 'numeric_batch_rows', 'numeric_available_by_field')}))


if __name__ == '__main__':
    main()
