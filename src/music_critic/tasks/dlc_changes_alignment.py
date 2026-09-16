"""Target-only changes alignment to every raw note-attack onset.

No gold segmentation, source timestamp inference, or raw graph mutation. Beat
aggregation is diagnostic: different modifier sets never become their union.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from hashlib import sha256
import re

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence, _alignment_evidence_fingerprint,
)
from music_critic.data import CanonicalPiece, dumps_piece
from music_critic.data.serialization import canonical_json_sha256
from music_critic.tasks.dlc_changes import SCHEMA as SOURCE_SCHEMA, encode_presence, parse_changes

SCHEMA = "DLCChangesAlignment@1.1.0"


def notation_provenance(event: Mapping) -> dict:
    """Require an explained source-field-to-chord-symbol relation.

    This is a conservative admission check, not a new full DCML parser. Some
    symbols have documented implicit changes; unexplained inherited/expanded
    values remain unknown instead of becoming either positives or negatives.
    """
    chords = event['source_context']['chord']
    if len(chords) != 1:
        return {'available': False, 'reason': 'ambiguous_chord_symbol'}
    chord = chords[0]
    groups = re.findall(r'\(([^()]*)\)', chord)
    if len(groups) > 1 or chord.count('(') != len(groups) or chord.count(')') != len(groups):
        return {'available': False, 'reason': 'unsupported_chord_symbol_syntax'}
    try:
        literal = [a['token'] for a in parse_changes(groups[0] if groups else '')]
    except ValueError:
        return {'available': False, 'reason': 'unparseable_literal_changes'}
    special = re.fullmatch(r'(It|Ger|Fr)[0-9]*(?:\([^()]*\))?(?:/[^()]+)?', chord)
    expected = list(literal)
    if special:
        expected.append('b5' if special[1] == 'Fr' else 'b3')
    values = event['source_changes_values']
    if len(values) != 1:
        return {'available': False, 'reason': 'source_changes_conflict'}
    try:
        source = [a['token'] for a in parse_changes(values[0])]
    except ValueError:
        return {'available': False, 'reason': 'source_changes_unavailable'}
    available = set(source) == set(expected)
    return {'available': available, 'reason': ('recognized_special_expansion' if special else 'literal_changes_match')
            if available else 'unexplained_changes_for_chord_symbol',
            'expected_tokens': sorted(set(expected)), 'source_tokens': sorted(set(source))}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError('changes_alignment.' + reason)


def _time(value) -> Fraction:
    return Fraction(value.num, value.den)


def _seal(payload: dict) -> dict:
    return {**payload, 'fingerprint': canonical_json_sha256(payload)}


def align_changes_to_raw_onsets(
    piece: CanonicalPiece,
    evidence: DilemmadataRawTargetAlignmentEvidence,
    events: Sequence[Mapping],
    *,
    source_sha256: str,
    vocabulary: Sequence[str],
) -> dict:
    """Join complete source groups to raw adapter lineage, then raw candidates.

    Caller authenticates source file/events and derives evidence using the raw
    adapter. This function validates its self-fingerprint and exact piece/row
    bindings without reopening source files. It is not a corpus/split locator.
    """
    _require(evidence.record_id.startswith('dlc:'), 'dlc_required')
    _require(evidence.piece_id == piece.piece_id, 'piece_identity')
    _require(evidence.canonical_piece_sha256 == sha256(dumps_piece(piece).encode()).hexdigest(), 'piece_hash')
    _require(evidence.fingerprint == _alignment_evidence_fingerprint(evidence), 'evidence_hash')
    _require(evidence.version in ('1.1.0', '1.2.0'), 'evidence_version')
    vocabulary = list(vocabulary)
    _require(len(vocabulary) == len(set(vocabulary)) and bool(vocabulary), 'vocabulary')
    for atom in vocabulary:
        _require([a['token'] for a in parse_changes(atom)] == [atom], 'atomic_vocabulary')
    notes = {note.note_id: note for note in piece.notes}
    raw_onsets = sorted({_time(note.onset_qn) for note in piece.notes})
    bindings = {row.line: row for row in evidence.rows}
    _require(len(bindings) == len(evidence.rows), 'duplicate_binding')
    by_event = {}
    notation = {}
    event_by_line = {}
    for event in events:
        _require(event.get('schema') == SOURCE_SCHEMA, 'source_schema')
        _require(event.get('record_id') == evidence.record_id, 'record_identity')
        _require(event['source']['sha256'] == source_sha256, 'source_hash')
        _require(event['source']['split'] == 'train', 'train_only')
        identity = event['source_annotation_id']
        _require(identity not in by_event, 'duplicate_event')
        _require(bool(event['source_row_numbers']), 'empty_event')
        _require(type(event['target_mask']) is bool and event['target_mask'] == (not event['mask_reasons']), 'mask_consistency')
        if event['target_mask']:
            values = event['source_changes_values']
            _require(len(values) == 1, 'changes_conflict')
            parsed = parse_changes(values[0])
            _require(parsed == event['atoms'], 'atoms_mismatch')
            _require(sorted({a['token'] for a in parsed}) == event['positive_tokens'], 'tokens_mismatch')
        by_event[identity] = event
        notation[identity] = notation_provenance(event)
        for line in event['source_row_numbers']:
            _require(line in bindings and line not in event_by_line, 'source_row_identity')
            event_by_line[line] = identity
    _require(set(event_by_line) == set(bindings), 'incomplete_source_row_coverage')
    by_time = defaultdict(list)
    note_bindings = []
    for line, row in sorted(bindings.items()):
        _require(row.canonical_note_id is None or row.canonical_note_id in notes, 'canonical_note_missing')
        if row.canonical_note_id is not None:
            note = notes[row.canonical_note_id]
            onset, end, anchor = _time(note.onset_qn), _time(note.onset_qn + note.duration_qn), _time(row.onset_qn)
            _require(onset <= anchor <= end, 'row_outside_note')
        by_time[_time(row.onset_qn)].append(line)
        note_bindings.append({'source_row': line, 'source_event_id': event_by_line[line],
                              'canonical_note_id': row.canonical_note_id,
                              'anchor_qn': str(_time(row.onset_qn)),
                              'tie_continuation': row.tie_continuation,
                              'repair_mask_scope': row.repair_mask_scope})

    def candidate(index: int, entity_id: str, time: Fraction, lines: list[int]) -> dict:
        event_ids = sorted({event_by_line[line] for line in lines})
        reasons = set()
        if not event_ids:
            reasons.add('no_source_rows_at_raw_candidate')
        if any(bindings[line].repair_mask_scope != 'none' or bindings[line].canonical_note_id is None for line in lines):
            reasons.add('raw_repair_binding_mask')
        encoded = [encode_presence(by_event[eid], vocabulary) for eid in event_ids]
        if any(not all(mask) for _, mask in encoded):
            reasons.add('source_mask_or_unknown_atom')
        if any(not notation[eid]['available'] for eid in event_ids):
            reasons.add('unexplained_source_notation')
        sets = {tuple(values) for values, mask in encoded if all(mask)}
        if len(sets) > 1:
            reasons.add('conflicting_modifier_sets')
        usable = not reasons
        return {'candidate_index': index, 'entity_id': entity_id, 'onset_qn': str(time),
                'source_event_ids': event_ids, 'source_rows': sorted(lines),
                'positive_tokens': ([vocabulary[i] for i, v in enumerate(next(iter(sets))) if v] if usable else []),
                'values': list(next(iter(sets))) if usable else [0] * len(vocabulary),
                'target_mask': usable, 'mask_reasons': sorted(reasons)}

    onset_targets = [candidate(i, f'onset:{time.numerator}_{time.denominator}', time, by_time.get(time, []))
                     for i, time in enumerate(raw_onsets)]
    support = Counter(eid for target in onset_targets if target['target_mask'] for eid in target['source_event_ids'])
    for target in onset_targets:
        target['annotation_weight'] = str(sum((Fraction(1, support[eid]) for eid in target['source_event_ids']), Fraction(0))) if target['target_mask'] else '0'
    # Diagnostic baseline: one prediction for a full raw beat cannot represent
    # different modifier sets on several onsets inside it.
    beat_starts = [_time(beat.start_qn) for beat in piece.beats]
    beat_lines = defaultdict(list)
    for time, lines in by_time.items():
        index = bisect_right(beat_starts, time) - 1
        if index >= 0 and time < _time(piece.beats[index].start_qn + piece.beats[index].duration_qn):
            beat_lines[index].extend(lines)
    beat_targets = [candidate(i, beat.beat_id, _time(beat.start_qn), beat_lines[i]) for i, beat in enumerate(piece.beats)]
    return _seal({'schema': SCHEMA, 'record_id': evidence.record_id, 'piece_id': piece.piece_id,
                  'canonical_piece_sha256': evidence.canonical_piece_sha256,
                  'source_sha256': source_sha256, 'raw_alignment_fingerprint': evidence.fingerprint,
                  'vocabulary': vocabulary, 'vocabulary_fingerprint': canonical_json_sha256(vocabulary),
                  'prediction_level': 'onset', 'candidate_rule': 'all_raw_note_attack_onsets',
                  'semantic_target': 'annotated_harmonic_modifiers_at_observed_onset_not_note_role',
                  'targets': onset_targets, 'source_note_bindings': note_bindings,
                  'notation_provenance': notation,
                  'beat_diagnostic': beat_targets,
                  'coverage': {'source_events': len(events), 'onset_candidates': len(onset_targets),
                               'unmasked_onsets': sum(t['target_mask'] for t in onset_targets),
                               'source_events_with_usable_onset': len(support),
                               'source_events_with_unexplained_notation': sum(not n['available'] for n in notation.values()),
                               'source_anchor_times_without_raw_attack': len(set(by_time) - set(raw_onsets)),
                               'masked_onset_reasons': dict(Counter(r for t in onset_targets for r in t['mask_reasons'])),
                               'masked_beat_reasons': dict(Counter(r for t in beat_targets for r in t['mask_reasons']))}})


def changes_targets_for_transposed_view(alignment: Mapping, graph, *, transform) -> dict:
    """Bind invariant relative targets to an already transposed raw graph.

    The graph transform (including range validation) is the caller's job.
    Only whole-piece, mode-preserving transposition is supported. No source
    chord strings/absolute keys are republished as if they were transposed.
    """
    from music_critic.experiments.analysisgnn.transposition import DirectedTransposition
    from music_critic.graph.validation import validate_raw_graph
    _require(isinstance(transform, DirectedTransposition), 'directed_transform_required')
    _require(alignment['schema'] == SCHEMA, 'alignment_schema')
    _require(alignment['fingerprint'] == canonical_json_sha256({k: v for k, v in alignment.items() if k != 'fingerprint'}), 'alignment_hash')
    validate_raw_graph(graph)
    _require(tuple(graph['song'].entity_id) == (alignment['piece_id'],), 'graph_piece_identity')
    _require(tuple(graph['onset'].entity_id) == tuple(t['entity_id'] for t in alignment['targets']), 'graph_candidate_identity')
    targets = [{k: target[k] for k in ('candidate_index', 'entity_id', 'values', 'target_mask', 'annotation_weight')}
               for target in alignment['targets']]
    # Canonical JSON roundtrip gives independent mutable containers to callers.
    import json
    targets = json.loads(json.dumps(targets))
    return _seal({'schema': SCHEMA, 'parent_alignment_fingerprint': alignment['fingerprint'],
                  'transform': transform.to_dict(), 'target_policy': 'whole_piece_relative_notation_invariant',
                  'targets': targets})
