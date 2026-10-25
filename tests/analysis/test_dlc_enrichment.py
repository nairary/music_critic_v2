"""Source-only witnesses for native DLC annotations, not musical accuracy."""
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import pytest

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence, DilemmadataSourceRowBinding,
    _alignment_evidence_fingerprint,
)
from music_critic.data import RationalTime, dumps_piece, load_piece
from music_critic.analysis.dlc_enrichment import build_dlc_enrichment, numerical_targets


def fixture():
    p = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    n = replace(next(n for n in p.notes if not n.is_percussion), note_id='note:test',
                onset_qn=RationalTime(0), duration_qn=RationalTime(2), pitch=60, staff=1, voice=1)
    p = replace(p, notes=(n,), targets=(), annotations=(), split='train')
    e = DilemmadataRawTargetAlignmentEvidence('1.1.0', 'dlc:test:one', p.piece_id,
        'b'*64, 'c'*64, sha256(dumps_piece(p).encode()).hexdigest(), ('pitch',),
        (DilemmadataSourceRowBinding(0,2,n.onset_qn,n.note_id,False),), '')
    e = replace(e, fingerprint=_alignment_evidence_fingerprint(e))
    rows = [dict(quarterbeats_playthrough='0', duration='1/2', pitch='60', staff='1', voice='1',
        is_note_onset='True', valid_chord_label='True', label='V7/V', a_localKey='C',
        a_tonicizedKey='G', changes='', note_degree='1', tpc_is_in_label='False',
        unfolded_harmony_index='0', cadence_type='', valid_cadence_label='False')]
    meta = dict(record_id='dlc:test:one', source_sha256='a'*64, current_split='train',
                canonical_piece=dict(path='fixture.json',sha256=e.canonical_piece_sha256,size_bytes=1),
                source_bindings=[])
    return p, asdict(e), rows, meta


def test_native_annotations_are_not_invented_harmony_spans():
    args=fixture();bundle = build_dlc_enrichment(*args)
    from music_critic.tasks.event_theory_enrichment import validate_enrichment
    validate_enrichment(args[0],bundle)
    assert bundle['harmonies'] == []
    c, = bundle['note_contexts']
    assert c['applicability'] == 'source_note_annotation' and c['harmony_id'] is None
    assert c['fields']['dlc.label']['value'] == 'V7/V'
    assert c['fields']['dlc.tpc_is_in_label']['value'] is False
    assert c['fields']['dlc.tpc_is_in_label']['available'] is True
    assert c['fields']['dlc.changes']['value'] == ''
    assert not c['fields']['dlc.cadence_type']['available']


def test_invalid_harmony_masks_theory_and_retains_source_values():
    args = fixture(); args[2][0]['valid_chord_label'] = 'False'
    b = build_dlc_enrichment(*args); f = b['note_contexts'][0]['fields']['dlc.label']
    assert f['available'] is False and f['value'] is None
    assert b['source_annotation_groups'][0]['raw_fields']['label'] == 'V7/V'


@pytest.mark.parametrize('split',['test','validation',None])
def test_split_guard(split):
    args=fixture(); args[3]['current_split']=split
    with pytest.raises(ValueError): build_dlc_enrichment(*args)


def test_numeric_false_is_available_and_unknown_is_minus_one():
    b=build_dlc_enrichment(*fixture()); n=numerical_targets(b)
    assert n['masks']['dlc.tpc_is_in_label'] == [True]
    assert n['values']['dlc.tpc_is_in_label'] == [0]
    assert n['field_vocabularies']['dlc.tpc_is_in_label'] == [False]
    assert n['values']['dlc.cadence_type'] == [-1]


def test_corrupt_note_geometry_rejected():
    args=fixture(); args[2][0]['pitch']='61'
    with pytest.raises(ValueError, match='pitch mismatch'): build_dlc_enrichment(*args)


def test_missing_native_tail_is_explicit_unknown_not_last_chord_extension():
    args=fixture();args[2][0]['duration']='1/4';b=build_dlc_enrichment(*args)
    from music_critic.tasks.event_theory_enrichment import validate_enrichment
    validate_enrichment(args[0],b)
    assert len(b['note_contexts'])==2
    tail=b['note_contexts'][1]
    assert tail['start_qn']=={'num':1,'den':1} and tail['end_qn']=={'num':2,'den':1}
    assert not any(f['available'] for f in tail['fields'].values())


def test_zero_duration_retains_point_but_masks_sounding_roles():
    p,e,rows,meta=fixture(); p=replace(p,notes=(replace(p.notes[0],duration_qn=RationalTime(0),is_grace=True),))
    e['canonical_piece_sha256']=sha256(dumps_piece(p).encode()).hexdigest()
    typed=DilemmadataRawTargetAlignmentEvidence(**{**e,'raw_source_fields':tuple(e['raw_source_fields']),
        'rows':tuple(DilemmadataSourceRowBinding(**{**r,'onset_qn':RationalTime(**r['onset_qn'])}) for r in e['rows'])})
    e=asdict(replace(typed,fingerprint=_alignment_evidence_fingerprint(typed)))
    rows[0]['duration']='0'; b=build_dlc_enrichment(p,e,rows,meta)
    c=b['note_contexts'][0]; assert c['start_qn']==c['end_qn']
    assert not c['fields']['dlc.note_degree']['available']


def test_repaired_cross_staff_tie_is_accounted_with_all_native_fields_masked():
    from tests.analysis.test_target_binding import reseal
    from music_critic.tasks.event_theory_enrichment import validate_enrichment
    p,e,rows,meta=fixture()
    provenance=replace(p.provenance[0],provenance_id='prov:remediated-fixture',kind='conversion',
        source='music_critic.dilemmadata_adapter',parents=('prov:source',),
        details=(('raw_repair_evidence_fingerprint','d'*64),))
    p=replace(p,provenance=(*p.provenance[:2],provenance,*p.provenance[2:]))
    e.update(version='1.2.0',raw_repair_evidence_fingerprint='d'*64)
    e['rows']=(*e['rows'],{**e['rows'][0],'ordinal':1,'line':3,
                           'onset_qn':{'num':1,'den':1},'tie_continuation':True})
    e=reseal(p,e);meta['canonical_piece']['sha256']=e['canonical_piece_sha256']
    rows[0]['duration']='1/4'
    rows.append({**rows[0],'quarterbeats_playthrough':'1','is_note_onset':'False','staff':'2'})
    b=build_dlc_enrichment(p,e,rows,meta);validate_enrichment(p,b)
    attack,continuation=b['note_contexts']
    assert attack['fields']['dlc.label']['available']
    assert continuation['note_id']==p.notes[0].note_id
    assert continuation['source_staff']==2 and p.notes[0].staff==1
    assert continuation['binding_unknown_reason']=='remediated_continuation_staff_change'
    assert continuation['start_qn']=={'num':1,'den':1}
    assert continuation['end_qn']=={'num':2,'den':1}
    assert all(not f['available'] and f['value'] is None for f in continuation['fields'].values())
    numeric=numerical_targets(b)
    assert numeric['masks']['dlc.label']==[True,False]
    assert numeric['values']['dlc.label'][1]==-1
