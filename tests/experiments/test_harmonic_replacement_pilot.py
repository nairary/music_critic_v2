from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest
from music_critic.data import load_piece, RationalTime as Q

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('harmonic_pilot_test',SCRIPTS/'build_harmonic_replacement_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def piece():
    p=load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json');n=p.notes[0]
    notes=tuple(replace(n,note_id=f'note:n{i}',pitch=pitch,onset_qn=Q(2),duration_qn=Q(1),staff=staff,voice=voice)
                for i,(pitch,staff,voice) in enumerate([(43,2,2),(59,2,1),(62,2,1),(71,1,1)]))
    return replace(p,notes=notes,targets=(),annotations=(),quality_flags=())


def labels():
    def h(root,roman,quality):return dict(root=root,roman_numeral=roman,quality=quality,local_key='G',source_entities=['source'])
    return {Q(1):h('D','V7','dominant seventh chord'),Q(2):h('G','I','major triad')}


def test_cadence_requires_root_relation_not_roman_alone():
    p=piece();s=pilot.SPECS['work3'];ctx=labels()
    assert pilot.cadence_evidence(p,ctx,Q(2),Q(3),s)['melody_pitch']==71
    ctx[Q(1)]['root']='B' # applied dominant must not masquerade as V of G
    with pytest.raises(ValueError,match='local dominant'):
        pilot.cadence_evidence(p,ctx,Q(2),Q(3),s)


@pytest.mark.parametrize('missing',['root','local_key','roman_numeral','quality'])
def test_cadence_unknown_annotation_is_barrier(missing):
    ctx=labels();ctx[Q(5,2)]={**ctx[Q(2)],missing:None}
    with pytest.raises(ValueError,match='ambiguous'):
        pilot.cadence_evidence(piece(),ctx,Q(2),Q(3),pilot.SPECS['work3'])


def test_whole_chord_refuses_boundary_spill():
    p=piece();ns=list(p.notes);ns[0]=replace(ns[0],onset_qn=Q(1),duration_qn=Q(2))
    with pytest.raises(ValueError,match='boundary spill'):
        pilot.inside_notes(replace(p,notes=tuple(ns)),Q(2),Q(3))


def test_triad_mapping_changes_all_chord_roles_and_preserves_melody():
    p=piece();ns=pilot.inside_notes(p,Q(2),Q(3));mapping=pilot.map_chord(ns,7,4,2)
    assert {pitch%12 for pitch in mapping.values()}=={2,6,9}
    c=pilot.candidate(p,'H','test','test',mapping,[(Q(2),Q(3))],[{2,6,9}])
    changed=pilot.apply_interventions(p,(pilot.Intervention('H',c['patches'],c['context']),)).piece
    assert pilot.verify_variant(p,changed,c)['preserved_staff1']
    assert len(c['patches'])==3
    bad=replace(changed,notes=tuple(replace(n,pitch=n.pitch+1) if n.staff==1 else n for n in changed.notes))
    with pytest.raises(ValueError,match='unexpected'):
        pilot.verify_variant(p,bad,c)


def test_frozen_chord_fills_missing_members_without_moving_bass_root():
    p=piece();notes=list(p.notes[:3])
    notes.extend(replace(notes[1],note_id=f'extra{i}',pitch=55) for i in range(3))
    mapping=pilot.frozen_chord(notes,2)
    assert set(p%12 for p in mapping.values())=={2,6,9,0}
    assert mapping['note:n0']%12==2


def test_expected_chord_and_rhythm_are_independently_rechecked():
    p=piece();mapping={n.note_id:n.pitch+1 for n in p.notes if n.staff==2}
    c=pilot.candidate(p,'H','test','test',mapping,[(Q(2),Q(3))],[{8,0,3}])
    changed=pilot.apply_interventions(p,(pilot.Intervention('H',c['patches'],c['context']),)).piece
    assert pilot.verify_variant(p,changed,c)['passed']
    bad=replace(changed,notes=tuple(replace(n,duration_qn=Q(1,2)) if n.staff==2 else n for n in changed.notes))
    with pytest.raises(ValueError,match='non-pitch'):
        pilot.verify_variant(p,bad,c)
    c['check']['spans'][0]['expected_pcs']=[0,3,7]
    with pytest.raises(ValueError,match='pitch classes'):
        pilot.verify_variant(p,changed,c)


def test_source_hash_guard_refuses_changed_payload(tmp_path):
    p=tmp_path/'data.json';p.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):
        pilot.checked_json(p,'0'*64)
