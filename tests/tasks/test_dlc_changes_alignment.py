from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence, DilemmadataSourceRowBinding,
    _alignment_evidence_fingerprint,
)
from music_critic.data import RationalTime, dumps_piece, load_piece
from music_critic.experiments.analysisgnn.transposition import (
    canonical_directed_transposition, transpose_raw_graph_view_directed,
)
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.tasks.dlc_changes import source_events
from music_critic.tasks.dlc_changes_alignment import (
    align_changes_to_raw_onsets, changes_targets_for_transposed_view, notation_provenance,
)

RID = 'dlc:fixture:changes'
SHA = 'a' * 64
VOCAB = ['4', '6', '+9']


def fixture(times=(0, Fraction(1, 2), 1), changes=('64', '4', '')):
    base = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    prototype = next(n for n in base.notes if not n.is_percussion)
    notes = tuple(replace(prototype, note_id=f'note:changes-{i}', pitch=60 + i,
                          onset_qn=RationalTime(t.numerator, t.denominator),
                          duration_qn=RationalTime(2)) for i, t in enumerate(map(Fraction, times)))
    piece = replace(base, notes=notes, targets=(), annotations=())
    rows = [{'unfolded_harmony_index': str(i), 'changes': change, 'chord': f'V({change})',
             'valid_chord_label': 'True', 'quarterbeats_playthrough': str(t)}
            for i, (t, change) in enumerate(zip(times, changes))]
    bindings = tuple(DilemmadataSourceRowBinding(i, i+2, n.onset_qn, n.note_id, False)
                     for i, n in enumerate(notes))
    evidence = make_evidence(piece, bindings)
    return piece, evidence, make_events(rows)


def make_evidence(piece, bindings):
    evidence = DilemmadataRawTargetAlignmentEvidence(
        version='1.2.0', record_id=RID, piece_id=piece.piece_id,
        record_binding_sha256='b'*64, raw_projection_sha256='c'*64,
        canonical_piece_sha256=sha256(dumps_piece(piece).encode()).hexdigest(),
        raw_source_fields=('pitch',), rows=bindings, fingerprint='',
        raw_repair_evidence_fingerprint='d'*64,
    )
    return replace(evidence, fingerprint=_alignment_evidence_fingerprint(evidence))


def make_events(rows):
    events = source_events(rows, RID)
    for e in events:
        e['source'] = {'sha256': SHA, 'split': 'train'}
    return events


def align(piece, evidence, events, vocabulary=VOCAB):
    return align_changes_to_raw_onsets(piece, evidence, events, source_sha256=SHA, vocabulary=vocabulary)


def test_subbeat_changes_remain_distinct_and_beat_union_is_forbidden():
    piece, evidence, events = fixture()
    result = align(piece, evidence, events)
    assert [t['positive_tokens'] for t in result['targets']] == [['4', '6'], ['4'], []]
    assert all(t['target_mask'] for t in result['targets'])
    assert any('conflicting_modifier_sets' in t['mask_reasons'] for t in result['beat_diagnostic'])
    assert sum(Fraction(t['annotation_weight']) for t in result['targets']) == 3


def test_source_time_is_not_used_instead_of_canonical_pickup_offset():
    piece, evidence, events = fixture(times=(Fraction(1, 3),), changes=('4',))
    events[0]['first_observed_note_onset_qn'] = '0'
    result = align(piece, evidence, events)
    assert result['targets'][0]['onset_qn'] == '1/3'
    assert result['targets'][0]['entity_id'] == 'onset:1_3'


def test_tie_context_change_does_not_overwrite_note_attack_target():
    piece, evidence, events = fixture(times=(0, 1), changes=('64', '4'))
    extra = replace(evidence.rows[0], ordinal=2, line=4, onset_qn=RationalTime(1), tie_continuation=True)
    evidence = make_evidence(piece, (*evidence.rows, extra))
    events[1]['source_row_numbers'].append(4)
    result = align(piece, evidence, events)
    assert result['targets'][0]['positive_tokens'] == ['4', '6']
    assert result['targets'][1]['positive_tokens'] == ['4']
    assert result['source_note_bindings'][-1]['canonical_note_id'] == piece.notes[0].note_id


def test_no_invented_candidate_for_a_tie_only_annotation_time():
    piece, evidence, events = fixture(times=(0,), changes=('4',))
    extra = replace(evidence.rows[0], ordinal=1, line=3, onset_qn=RationalTime(1), tie_continuation=True)
    evidence = make_evidence(piece, (*evidence.rows, extra))
    events[0]['source_row_numbers'].append(3)
    result = align(piece, evidence, events)
    assert len(result['targets']) == 1
    assert result['coverage']['source_anchor_times_without_raw_attack'] == 1


def test_repetition_does_not_increase_annotation_weight():
    piece, evidence, events = fixture(changes=('4', '4', '4'))
    events[0]['source_row_numbers'] = [2, 3, 4]
    result = align(piece, evidence, events[:1])
    assert [t['annotation_weight'] for t in result['targets']] == ['1/3'] * 3


@pytest.mark.parametrize('reason', ['invalid_source', 'unknown_atom', 'repair', 'conflicting_sets'])
def test_ambiguity_or_missing_is_masked_not_negative(reason):
    piece, evidence, events = fixture(times=(0, 0), changes=('4', '4'))
    vocabulary = VOCAB
    if reason == 'invalid_source':
        events[0].update(target_mask=False, mask_reasons=['invalid_chord'])
    elif reason == 'unknown_atom':
        vocabulary = ['6']
    elif reason == 'repair':
        evidence = make_evidence(piece, (replace(evidence.rows[0], repair_mask_scope='note'), evidence.rows[1]))
    else:
        from music_critic.tasks.dlc_changes import parse_changes
        events[1].update(source_changes_values=['6'], atoms=parse_changes('6'), positive_tokens=['6'])
    result = align(piece, evidence, events, vocabulary)
    assert not result['targets'][0]['target_mask']
    assert not any(result['targets'][0]['values'])


@pytest.mark.parametrize('corruption', ['piece', 'hash', 'missing_row', 'duplicate_row', 'event_record', 'test', 'tokens'])
def test_incompatible_provenance_fails_closed(corruption):
    piece, evidence, events = fixture()
    if corruption == 'piece': piece = replace(piece, piece_id='piece:other')
    if corruption == 'hash': evidence = replace(evidence, fingerprint='0'*64)
    if corruption == 'missing_row': events.pop()
    if corruption == 'duplicate_row': events[1]['source_row_numbers'] = [2]
    if corruption == 'event_record': events[0]['record_id'] = 'dlc:other:record'
    if corruption == 'test': events[0]['source']['split'] = 'test'
    if corruption == 'tokens': events[0]['positive_tokens'] = ['+9']
    with pytest.raises(ValueError): align(piece, evidence, events)


@pytest.mark.parametrize('shift', range(12))
def test_all_directed_shifts_and_inverse_preserve_targets_and_candidates(shift):
    piece, evidence, events = fixture()
    graph = build_raw_graph(piece)
    original = graph_fingerprint(graph)
    result = align(piece, evidence, events)
    token = canonical_directed_transposition(shift)
    view = transpose_raw_graph_view_directed(graph, transform=token)
    bound = changes_targets_for_transposed_view(result, view, transform=token)
    assert [r['values'] for r in bound['targets']] == [r['values'] for r in result['targets']]
    bound['targets'][0]['values'][0] = 19
    assert result['targets'][0]['values'][0] == 1
    restored = transpose_raw_graph_view_directed(view, transform=token.inverse())
    assert graph_fingerprint(restored) == original == graph_fingerprint(graph)


def test_target_changes_do_not_change_raw_candidates_or_graph():
    piece, evidence, events = fixture()
    graph = build_raw_graph(piece); original = graph_fingerprint(graph)
    first = align(piece, evidence, events)
    modified = deepcopy(events)
    for e in modified: e.update(target_mask=False, mask_reasons=['unavailable'])
    second = align(piece, evidence, modified)
    assert [t['entity_id'] for t in first['targets']] == [t['entity_id'] for t in second['targets']]
    assert graph_fingerprint(graph) == original


@pytest.mark.parametrize(('chord', 'changes', 'available'), [
    ('V(64)', '64', True), ('V64', '64', False), ('Ger6', 'b3', True),
    ('Fr43', 'b5', True), ('It6', 'b3', True), ('IM7', '^#8', False),
    ('IM7', '', True), ('i6(#0)', '#0', True), ('V7(+9)', '+9', True),
])
def test_literal_or_explained_special_provenance(chord, changes, available):
    e = make_events([{'unfolded_harmony_index': '0', 'chord': chord, 'changes': changes,
                     'valid_chord_label': 'True'}])[0]
    assert notation_provenance(e)['available'] is available


def test_unexplained_im7_is_masked_not_rewritten_as_no_changes():
    piece, evidence, events = fixture(times=(0,), changes=('^#8',))
    events[0]['source_context']['chord'] = ['IM7']
    result = align(piece, evidence, events, ['^#8'])
    assert not result['targets'][0]['target_mask']
    assert result['targets'][0]['mask_reasons'] == ['unexplained_source_notation']
    assert events[0]['positive_tokens'] == ['^#8']
