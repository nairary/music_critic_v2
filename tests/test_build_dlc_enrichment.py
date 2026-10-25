"""Immutable exporter/admission mechanics, with no real corpus access."""
from pathlib import Path
import pytest
from scripts.build_dlc_enrichment import admit_records, write_gzip_once, read_bound, attach_t06


def test_admission_excludes_heldout_before_payload():
    rows=[dict(record_id='dlc:a:x',split='train',component_id='g'),dict(record_id='dlc:a:y',split='train',component_id='h')]
    current=[dict(record_id='dlc:a:x',split='test',source_component_id='g'),dict(record_id='dlc:a:y',split='train',source_component_id='h')]
    good,skip=admit_records(rows,current)
    assert [r['record_id'] for r in good]==['dlc:a:y']
    assert skip[0]['reason']=='not_current_train'


def test_component_crossing_split_fails_admission():
    rows=[dict(record_id='dlc:a:x',split='train',component_id='g')]
    current=[dict(record_id='dlc:a:x',split='train',source_component_id='g'),dict(record_id='dlc:a:z',split='validation',source_component_id='g')]
    good,skip=admit_records(rows,current)
    assert not good and skip[0]['reason']=='component_has_heldout'


def test_immutable_gzip_and_hash(tmp_path):
    p=tmp_path/'x.json.gz'; b=write_gzip_once(p,dict(a=False,b=None))
    assert read_bound(tmp_path,b)
    with pytest.raises(FileExistsError):write_gzip_once(p,{})
    p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash'):read_bound(tmp_path,b)


def test_t06_merge_preserves_native_origin_and_counts():
    from music_critic.data import load_piece
    from dataclasses import replace
    piece=load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json')
    piece=replace(piece,notes=(piece.notes[0],),targets=(),annotations=(),split='train')
    note=piece.notes[0]
    q=lambda x:dict(num=x.numerator,den=x.denominator)
    a=note.onset_qn.to_fraction();z=a+note.duration_qn.to_fraction()
    dense=dict(canonical_piece_sha256='a'*64,harmonies=[],rows=[dict(row_id='r',note_id=note.note_id,
        note_index=0,start_qn=q(a),end_qn=q(z),harmony_id=None,fields={
            'source_chord_member':dict(value=False,available=True,origin='source_native',reason=None,source_fields=['tpc_is_in_label']),
            'chord_degree':dict(value=None,available=False,origin='unknown',reason='unknown_spelling',source_fields=[])})])
    bundle=dict(schema='EventTheoryEnrichment@1',dataset_id=piece.dataset_name,piece_id=piece.piece_id,
        current_split='train',canonical_piece=dict(sha256='a'*64),harmonies=[],note_contexts=[],relations=[],coverage={})
    attach_t06(bundle,piece,dense)
    f=bundle['note_contexts'][0]['fields']
    assert f['dense.source_chord_member']['origin']=='source_native'
    assert f['dense.source_chord_member']['value'] is False
    assert f['dense.chord_degree']['unknown_reason']=='unknown_spelling'
    assert bundle['coverage']['available_by_field']=={'dense.source_chord_member':1}


def t06_interval_fixture(start, end, *, context_harmony='h'):
    from dataclasses import replace
    from music_critic.data import RationalTime, load_piece
    piece=load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json')
    note=replace(piece.notes[0],onset_qn=RationalTime(0),duration_qn=RationalTime(1))
    piece=replace(piece,notes=(note,),duration_qn=RationalTime(1),targets=(),annotations=(),split='train')
    q=lambda n:dict(num=n,den=1)
    dense=dict(canonical_piece_sha256='a'*64,
        harmonies=[dict(id='h',start_qn=q(start),end_qn=q(end),
                        values={'label':'V7/V'},availability={'label':True})],
        rows=[dict(row_id='r',note_id=note.note_id,note_index=0,start_qn=q(0),end_qn=q(1),
            harmony_id=context_harmony,fields={
                'source_chord_member':dict(value=False,available=True,origin='source_native',
                                           reason=None,source_fields=['tpc_is_in_label'])})])
    bundle=dict(schema='EventTheoryEnrichment@1',dataset_id=piece.dataset_name,piece_id=piece.piece_id,
        current_split='train',canonical_piece=dict(sha256='a'*64),harmonies=[],note_contexts=[],relations=[],coverage={})
    return piece,dense,bundle


def test_t06_right_overhang_is_applicability_clip_with_original_interval_retained():
    from copy import deepcopy
    piece,dense,bundle=t06_interval_fixture(0,2)
    original_dense=deepcopy(dense)
    attach_t06(bundle,piece,dense)
    harmony,=bundle['harmonies']
    assert harmony['start_qn']=={'num':0,'den':1}
    assert harmony['end_qn']=={'num':1,'den':1}
    assert harmony['source_interval']=={'start_qn':{'num':0,'den':1},'end_qn':{'num':2,'den':1}}
    assert harmony['interval_projection']=='intersection_with_canonical_piece_extent'
    assert harmony['fields']['dlc.label']['value']=='V7/V'
    assert bundle['note_contexts'][0]['end_qn']=={'num':1,'den':1}
    assert bundle['note_contexts'][0]['fields']['dense.source_chord_member']['value'] is False
    assert dense==original_dense


def test_t06_left_overhang_preserves_original_negative_source_start():
    piece,dense,bundle=t06_interval_fixture(-1,1)
    attach_t06(bundle,piece,dense)
    harmony,=bundle['harmonies']
    assert harmony['start_qn']=={'num':0,'den':1}
    assert harmony['end_qn']=={'num':1,'den':1}
    assert harmony['source_interval']=={'start_qn':{'num':-1,'den':1},'end_qn':{'num':1,'den':1}}
    assert bundle['note_contexts'][0]['start_qn']=={'num':0,'den':1}
    assert bundle['note_contexts'][0]['harmony_id']=='t06:h'


@pytest.mark.parametrize('start,end',[(-2,-1),(1,2),(2,3)])
def test_t06_outside_harmony_is_metadata_not_positive_span(start,end):
    from copy import deepcopy
    piece,dense,bundle=t06_interval_fixture(start,end,context_harmony=None)
    original=deepcopy(dense['harmonies'][0])
    attach_t06(bundle,piece,dense)
    assert bundle['harmonies']==[]
    assert bundle['coverage']['exact_harmonic_spans']==0
    assert bundle['note_contexts'][0]['harmony_id'] is None
    omitted,=bundle['provenance']['t06_interval_projection']['omitted_harmonies']
    assert omitted['source_harmony']==original
    assert omitted['reason']=='no_positive_intersection_with_canonical_piece_extent'


def test_t06_outside_harmony_context_reference_rejected_not_reassigned():
    piece,dense,bundle=t06_interval_fixture(1,2)
    with pytest.raises(ValueError,match='context references harmony outside'):
        attach_t06(bundle,piece,dense)
    assert bundle['note_contexts']==[]


def test_t06_in_bounds_harmony_unchanged_and_context_preserved_exactly():
    piece,dense,bundle=t06_interval_fixture(0,1)
    attach_t06(bundle,piece,dense)
    harmony,=bundle['harmonies']
    assert harmony==dict(id='t06:h',start_qn={'num':0,'den':1},end_qn={'num':1,'den':1},fields={
        'dlc.label':dict(value='V7/V',available=True,origin='source_native',confidence=None,evidence=['h'],
                         unknown_reason=None,semantic_level='exact_harmonic_interval')})
    context,=bundle['note_contexts']
    assert (context['note_id'],context['canonical_note_index'],context['start_qn'],context['end_qn'])==(
        dense['rows'][0]['note_id'],0,{'num':0,'den':1},{'num':1,'den':1})
    projection=bundle['provenance']['t06_interval_projection']
    assert projection['clipped_harmonies']==projection['omitted_harmonies']==[]
