from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

from music_critic.data import load_piece, RationalTime as Q

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('pilot_builder_test',SCRIPTS/'build_music_intervention_pilot.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def piece():
    return load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json')


def test_half_open_overlap_and_exact_spacing():
    p=piece();n=p.notes[0]
    a=replace(n,onset_qn=Q(0),duration_qn=Q(1,3))
    b=replace(n,onset_qn=Q(1,3),duration_qn=Q(1,3))
    assert pilot.overlap(a,b)==Q(0)
    cs=[dict(onset=Q(x,3),key=str(x)) for x in range(4)]
    assert [c['onset'] for c in pilot.spaced(cs,minimum=Q(2,3))]==[Q(0),Q(2,3)]


def test_rhythm_edit_can_remove_the_harmonic_witness():
    p=piece();n=p.notes[0]
    a=replace(n,note_id='a',pitch=61,onset_qn=Q(0),duration_qn=Q(1,4))
    b=replace(n,note_id='b',pitch=60,onset_qn=Q(0),duration_qn=Q(1,4))
    c=dict(id='H1',check=dict(kind='new_semitone_clash',note='a',witness='b',after_pitch=61,minimum_overlap=[1,4]))
    assert pilot.verify_effect(replace(p,notes=(a,b)),c)['passed']
    assert not pilot.verify_effect(replace(p,notes=(replace(a,onset_qn=Q(1,4)),b)),c)['passed']
    assert not pilot.verify_effect(replace(p,notes=(a,)),c)['passed']


def test_melody_check_detects_context_cancellation():
    p=piece();n=p.notes[0]
    ns=tuple(replace(n,note_id=k,pitch=pi,onset_qn=Q(i),duration_qn=Q(1))
             for i,(k,pi) in enumerate([('a',60),('b',74),('c',64)]))
    c=dict(id='M1',check=dict(kind='octave_line_break',note='b',previous='a',following='c',before_pitch=62,after_pitch=74))
    assert pilot.verify_effect(replace(p,notes=ns),c)['passed']
    assert not pilot.verify_effect(replace(p,notes=(replace(ns[0],pitch=72),ns[1],ns[2])),c)['passed']


def test_unknown_harmony_never_becomes_default_class():
    s={'entities':[{'entity_type':'harmonic_event','onset_qn':{'num':0,'den':1},'source_entity_id':'x',
                   'targets':{k:dict(available=False,masked=True,canonical_value='C') for k in ['local_key','root','quality','roman_numeral']}}]}
    assert pilot.harmonic_context(s)[Q(0)]['root'] is None
    assert pilot.parse_root('unknown') is None
    assert pilot.parse_root('Bb')==10


def test_midi_preserves_same_pitch_independent_lifetimes():
    p=piece();n=next(n for n in p.notes if not n.is_percussion)
    ns=(replace(n,note_id='note:a',pitch=60,onset_qn=Q(0),duration_qn=Q(1),staff=1,voice=1),
        replace(n,note_id='note:b',pitch=60,onset_qn=Q(1,2),duration_qn=Q(1),staff=1,voice=1))
    clean=replace(p,notes=ns,targets=(),annotations=(),quality_flags=())
    payload,report=pilot.routed_midi(clean)
    assert payload.startswith(b'MThd') and report['note_roundtrip'] and report['notes']==2


def test_grace_omission_is_preview_only_and_reported():
    p=piece();n=next(n for n in p.notes if not n.is_percussion)
    ns=(replace(n,note_id='note:a',pitch=60,onset_qn=Q(0),duration_qn=Q(1)),
        replace(n,note_id='note:grace',pitch=62,onset_qn=Q(1),duration_qn=Q(0),is_grace=True))
    clean=replace(p,notes=ns,targets=(),annotations=(),quality_flags=())
    _,report=pilot.routed_midi(clean)
    assert report['omitted_zero_duration_grace_ids']==['note:grace']
    assert len(clean.notes)==2 and report['notes']==1


def test_control_detects_hidden_time_or_note_changes():
    p=piece();changed=replace(p,notes=tuple(replace(n,pitch=n.pitch+2) for n in p.notes))
    assert pilot.verify_transposition(p,changed,2)
    assert not pilot.verify_transposition(p,replace(changed,notes=changed.notes[1:]),2)
    wrong=replace(changed.notes[0],onset_qn=changed.notes[0].onset_qn+Q(1,4))
    assert not pilot.verify_transposition(p,replace(changed,notes=(wrong,*changed.notes[1:])),2)
