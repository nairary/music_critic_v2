"""Synthetic contracts for separate chord and observed interval scopes."""
from copy import deepcopy
import unittest

from music_critic.tasks.chord_interval_projection import (
    INTERVAL_FIELDS, build_note_context, interval_contract, parse_tones, project_intervals,
)


def known(value, **extra):
    return dict(value=value, available=True, annotation_state='known', origin='source_derived',
                confidence=None, evidence=['synthetic:source'], unknown_reason=None, **extra)


def root(pc=0, **extra):
    return dict(value=pc, available=True, state='exact', information_loss=[], **extra)


def q(n, d=1):
    return dict(num=n, den=d)


def row(value=None, attributes=None):
    return dict(id='H0', view_id='v0', start_qn=q(0), end_qn=q(1),
                fields={'analysis.sonority': known(value or {'root_pitch_class': 0, 'tones': ['C', 'E', 'G']},
                                                  attributes=attributes or {})})


def notes(pitches=(60, 64, 67), start=0, duration=2):
    return {f'n{i}': dict(note_id=f'n{i}', pitch=p, onset_qn=q(start), duration_qn=q(duration),
                          is_percussion=False) for i,p in enumerate(pitches)}


def observed(pitches=(60,64,67)):
    r = row({'label':'literal_key_collection', 'reading':'complete_collection',
             'root_pitch_class':0, 'tones':list(pitches), 'tones_format':'midi'},
            attributes={'key_active_note_ids':[f'n{i}' for i in range(len(pitches))],
                        'key_pitches':list(pitches), 'possible_pedal_held_note_ids':[]})
    r['source_event_ids'] = [f'n{i}' for i in range(len(pitches))]
    return r


def complete_template(r):
    r['note_template'] = dict(schema='AuthoredNoteTemplate@1', id='synthetic:t', evidence=['synthetic:t'],
        basis='midi_pitch_class', enharmonic_policy='author_declared_pc_mapping',
        members=['0','4','7'], members_complete=True)
    return r


def literal():
    r=observed();r['fields']['analysis.sonority']['value'].update(
        label='observed_trumpet_sonority', reading='exact_observed_aggregate_root_unresolved')
    attrs=dict(literal_root_pitch_class=0, literal_quality='major', structural_root_asserted=False,
               reference_ids=['n0','n1','n2'], reference_interval=dict(start_qn=q(0),end_qn=q(1)),
               target_scope='instantaneous_complete_pitch_class_sonority')
    r['fields']['analysis.extensions']=known([],attributes=deepcopy(attrs))
    r['fields']['analysis.inversion']=known('root_position',attributes=deepcopy(attrs))
    r['fields']['analysis.sonority']['attributes']['completion012_literal_reading']=dict(
        harmony_id='H0', start_qn=q(0), end_qn=q(1), note_ids=['n0','n1','n2'], pitches=[60,64,67],
        pitch_classes=[0,4,7], literal_root_pitch_class=0, literal_quality='major',
        structural_root_asserted=False, function_asserted=False)
    r['fields']['analysis.sonority']['value'].pop('root_pitch_class', None)
    return r


class ChordIntervalTests(unittest.TestCase):
    def test_known_root_with_partial_tones_gives_positive_intervals_only(self):
        result = project_intervals(row(), resolved_root=root(), notes={})
        for interval in (0, 4, 7):
            self.assertIs(result[f'H.chord.interval_pc.{interval}']['value'], True)
        self.assertFalse(result['H.chord.interval_pc.1']['available'])
        self.assertTrue(all(not m['available'] for name,m in result.items() if name.startswith('H.observed.')))


    def test_pitch_formats_aliases_enharmonic_and_octave_crossing(self):
        self.assertEqual(parse_tones({'tones':['B#3','C4']})['midi_pitches'],[60])
        self.assertEqual(parse_tones({'tones':['E-','G','B-flat']})['pcs'],[3,7,10])
        self.assertEqual(parse_tones({'tones':['D♭','C#']})['pcs'],[1])
        self.assertEqual(parse_tones({'tones':[60,64],'tones_format':'midi'})['pcs'],[0,4])
        self.assertEqual(parse_tones({'tones':[0,4],'tones_format':'pitch_class'})['state'],'exact')
        for values in ([60,64],[True],['H4'],['C##b4'],['C10']):
            self.assertFalse(parse_tones({'tones':values})['available'])
        self.assertFalse(parse_tones({'tones':[True],'tones_format':'pitch_class'})['available'])
        self.assertIn('octave_removed',parse_tones({'tones':['C4']})['information_loss'])
        self.assertIn('enharmonic_spelling_removed',parse_tones({'tones':['C']})['information_loss'])

    def test_complete_interpreted_template_permits_zeros(self):
        r=complete_template(row({'label':'C_major','root_pitch_class':0,'tones':['C','E','G']}))
        result=project_intervals(r,resolved_root=root(),notes={})
        self.assertIs(result['H.chord.interval_pc.1']['value'],False)
        self.assertTrue(all(m['available'] for n,m in result.items() if n.startswith('H.chord.')))
        self.assertFalse(any(m['available'] for n,m in result.items() if n.startswith('H.observed.')))

    def test_bare_completeness_flags_cannot_license_zeros(self):
        r=row({'root_pitch_class':0,'tones':['C','E','G'],'complete':True,'tones_complete':True})
        result=project_intervals(r,resolved_root=root(),notes={})
        self.assertFalse(result['H.chord.interval_pc.1']['available'])

    def test_complete_template_requires_exact_members_and_interpreted_route(self):
        r=complete_template(row({'label':'C_major','tones':['C','E','G']}))
        r['note_template']['members']=['0','4']
        result=project_intervals(r,resolved_root=root(),notes={})
        self.assertFalse(any(m['available'] for n,m in result.items() if n.startswith('H.chord.')))
        r=row({'tones':['C','E','G']});complete_template(r)
        result=project_intervals(r,resolved_root=root(),notes={})
        self.assertFalse(result['H.chord.interval_pc.1']['available'])

    def test_complete_spelled_and_midi_templates(self):
        r=row({'label':'C_major','tones':['C4','E4','G4']});complete_template(r)
        for basis,members in [('spelled_pitch_class',['C:0','E:0','G:0']),('midi_pitch',['60','64','67'])]:
            r['note_template'].update(basis=basis,members=members)
            self.assertIs(project_intervals(r,resolved_root=root(),notes={})['H.chord.interval_pc.1']['value'],False)

    def test_rootless_closed_set_does_not_invent_root_presence(self):
        r=row({'label':'Cmaj7','tones':['E','G','B']});complete_template(r)
        r['note_template']['members']=['4','7','11']
        result=project_intervals(r,resolved_root=root(),notes={})
        self.assertIs(result['H.chord.interval_pc.0']['value'],False)
        self.assertIs(result['H.chord.interval_pc.4']['value'],True)

    def test_transposition_preserves_relative_values_and_masks_all_twelve_shifts(self):
        expected=None
        for shift in range(12):
            r=row({'root_pitch_class':shift,'tones':[(p+shift)%12 for p in (0,4,7)],'tones_format':'pitch_class'})
            result=project_intervals(r,resolved_root=root(shift),notes={})
            current=[(result[f'H.chord.interval_pc.{i}']['value'],result[f'H.chord.interval_pc.{i}']['available']) for i in range(12)]
            expected=expected or current
            self.assertEqual(current,expected)

    def test_observed_complete_set_has_separate_closed_absolute_and_relative_fields(self):
        result=project_intervals(observed(),resolved_root=root(),notes=build_note_context(notes()))
        self.assertIs(result['H.observed.pitch_class.0']['value'],True)
        self.assertIs(result['H.observed.pitch_class.1']['value'],False)
        self.assertIs(result['H.observed.interval_pc.4']['value'],True)
        self.assertIs(result['H.observed.interval_pc.1']['value'],False)
        self.assertEqual(result['H.observed.root_pc']['value'],0)
        self.assertFalse(any(m['available'] for n,m in result.items() if n.startswith('H.chord.')))

    def test_observed_unknown_root_preserves_absolute_set(self):
        r=observed();r['fields']['analysis.sonority']['value'].pop('root_pitch_class')
        result=project_intervals(r,resolved_root=dict(value=None,available=False,state='ambiguous'),notes=notes())
        self.assertTrue(all(m['available'] for n,m in result.items() if n.startswith('H.observed.pitch_class.')))
        self.assertFalse(any(m['available'] for n,m in result.items() if n.startswith('H.observed.interval_pc.')))

    def test_observed_requires_all_canonical_participants_and_correct_tones(self):
        for mutate in (
            lambda r:r.update(source_event_ids=['n0','n1']),
            lambda r:r['fields']['analysis.sonority']['value'].update(tones=[60,64,68]),
            lambda r:r['fields']['analysis.sonority']['attributes'].update(key_pitches=[60,64]),
            lambda r:r['fields']['analysis.sonority']['attributes'].update(key_active_note_ids=['n0','n2']),
            lambda r:r['fields']['analysis.sonority']['attributes'].update(possible_pedal_held_note_ids=['nPedal']),
        ):
            r=observed();mutate(r)
            self.assertFalse(any(m['available'] for m in project_intervals(r,resolved_root=root(),notes=notes()).values()))

    def test_observed_register_optional_when_pcs_and_ids_authenticate(self):
        r=observed();r['fields']['analysis.sonority']['value'].update(tones=['C','E','G'],tones_format='spelled_pitch')
        result=project_intervals(r,resolved_root=root(),notes=notes())
        self.assertIs(result['H.observed.pitch_class.1']['value'],False)

    def test_observed_internal_boundaries_block_window_union_negatives(self):
        ns=notes();ns['n0']['duration_qn']=q(1,2)
        r=observed()
        result=project_intervals(r,resolved_root=root(),notes=ns)
        self.assertFalse(any(m['available'] for m in result.values()))

    def test_half_open_boundaries_sustains_percussion_and_points(self):
        ns=notes(start=0,duration=3)
        ns['ended']=dict(note_id='ended',pitch=61,onset_qn=q(0),duration_qn=q(1),is_percussion=False)
        ns['next']=dict(note_id='next',pitch=62,onset_qn=q(2),duration_qn=q(1),is_percussion=False)
        ns['drum']=dict(note_id='drum',pitch=38,onset_qn=q(0),duration_qn=q(3),is_percussion=True)
        ns['point']=dict(note_id='point',pitch=63,onset_qn=q(1),duration_qn=q(0),is_percussion=False)
        r=observed();r.update(start_qn=q(1),end_qn=q(2))
        result=project_intervals(r,resolved_root=root(),notes=ns)
        self.assertIs(result['H.observed.pitch_class.1']['value'],False)

    def test_exact_reduced_geometry_required(self):
        r=observed();r['start_qn']=dict(num=0,den=2)
        self.assertFalse(project_intervals(r,resolved_root=root(),notes=notes())['H.observed.pitch_class.0']['available'])

    def test_literal_root_is_independent_of_structural_root_unknown(self):
        result=project_intervals(literal(),resolved_root=dict(value=None,available=False,state='masked'),notes=notes())
        self.assertEqual(result['H.observed.root_pc']['value'],0)
        self.assertIs(result['H.observed.interval_pc.1']['value'],False)
        self.assertFalse(any(m['available'] for n,m in result.items() if n.startswith('H.chord.')))

    def test_literal_identity_span_pitches_ids_and_quality_fail_relative_only(self):
        for name,value in [('harmony_id','wrong'),('start_qn',q(1)),('note_ids',['n0']),
                           ('pitches',[60,64]),('pitch_classes',[0,4]),('literal_root_pitch_class',1),
                           ('literal_quality','minor'),('structural_root_asserted',True)]:
            r=literal();r['fields']['analysis.sonority']['attributes']['completion012_literal_reading'][name]=value
            result=project_intervals(r,resolved_root=dict(value=None,available=False,state='masked'),notes=notes())
            with self.subTest(name=name):
                self.assertTrue(result['H.observed.pitch_class.0']['available'])
                self.assertFalse(result['H.observed.root_pc']['available'])
                self.assertFalse(result['H.observed.interval_pc.0']['available'])

    def test_literal_and_direct_root_conflicts_do_not_erase_absolute_set(self):
        r=literal();r['fields']['analysis.sonority']['value']['root_pitch_class']=1
        result=project_intervals(r,resolved_root=root(),notes=notes())
        self.assertTrue(result['H.observed.pitch_class.0']['available'])
        self.assertFalse(result['H.observed.root_pc']['available'])

    def test_literal_linked_field_root_scope_or_interval_conflicts_mask_relative(self):
        for key,val in [('literal_root_pitch_class',1),('target_scope','structural_chord'),
                        ('reference_ids',['n0']),('reference_interval',dict(start_qn=q(0),end_qn=q(2)))]:
            r=literal();r['fields']['analysis.inversion']['attributes'][key]=val
            result=project_intervals(r,resolved_root=root(),notes=notes())
            self.assertTrue(result['H.observed.pitch_class.0']['available'])
            self.assertFalse(result['H.observed.root_pc']['available'])

    def test_unavailable_source_scale_and_unresolved_scope_never_become_chord_targets(self):
        r=row();r['fields']['analysis.sonority'].update(value=None,available=False,annotation_state='unresolved')
        self.assertFalse(any(m['available'] for m in project_intervals(r,resolved_root=root(),notes={}).values()))
        for scope in ['scale_collection','root_unresolved']:
            r=row();r['fields']['analysis.sonority']['value']['reading']=scope
            self.assertFalse(any(m['available'] for m in project_intervals(r,resolved_root=root(),notes={}).values()))

    def test_ambiguous_or_unknown_root_never_creates_relative_labels(self):
        for r in [dict(value=None,available=False,state='ambiguous'),dict(value=True,available=True,state='exact'),dict(value=12,available=True,state='exact')]:
            self.assertFalse(any(m['available'] for m in project_intervals(row(),resolved_root=r,notes={}).values()))

    def test_root_information_loss_is_retained_on_observed_relative_route(self):
        r=observed();r['fields']['analysis.sonority']['value'].pop('root_pitch_class')
        r['fields']['analysis.sonority']['value']['root']='C'
        result=project_intervals(r,resolved_root=root(1),notes=notes())
        self.assertIn('spelling_removed',result['H.observed.interval_pc.4']['information_loss'])
        self.assertIn('spelling_removed',result['H.observed.root_pc']['information_loss'])
        self.assertEqual(result['H.observed.root_pc']['value'],0)
        self.assertEqual(result['H.observed.root_pc']['source_paths'],['analysis.sonority'])

    def test_no_mutation_and_context_has_no_input_aliases(self):
        r=literal();ns=notes();before=deepcopy((r,ns));context=build_note_context(ns)
        project_intervals(r,resolved_root=root(),notes=context)
        self.assertEqual((r,ns),before)
        ns['n0']['pitch']=61
        self.assertEqual(context.notes['n0'][0],60)

    def test_contract_is_self_hashed_and_has_exact_separate_field_domains(self):
        import hashlib,json
        c=interval_contract();fingerprint=c.pop('fingerprint')
        self.assertEqual(hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),fingerprint)
        self.assertEqual(len(INTERVAL_FIELDS),37)
        self.assertEqual(INTERVAL_FIELDS['H.observed.root_pc'],tuple(range(12)))
        self.assertTrue(all(values==(False,True) for field,values in INTERVAL_FIELDS.items() if field!='H.observed.root_pc'))


    def test_outer_structural_root_does_not_override_literal_observed_root(self):
        r=literal();v=r['fields']['analysis.sonority']['value'];v.pop('root_pitch_class',None)
        v.update(tones=[57,60,64])
        a=r['fields']['analysis.sonority']['attributes']
        a.update(key_pitches=[57,60,64])
        a['completion012_literal_reading'].update(pitches=[57,60,64],pitch_classes=[0,4,9],
                                                 literal_root_pitch_class=9,literal_quality='minor')
        for field in ('analysis.extensions','analysis.inversion'):
            r['fields'][field]['attributes'].update(literal_root_pitch_class=9,literal_quality='minor')
        result=project_intervals(r,resolved_root=root(0),notes=notes((57,60,64)))
        self.assertEqual(result['H.observed.root_pc']['value'],9)
        for pc in (0,3,7):
            self.assertIs(result[f'H.observed.interval_pc.{pc}']['value'],True)
        self.assertFalse(any(m['available'] for n,m in result.items() if n.startswith('H.chord.')))

    def test_explicit_scale_attribute_and_unknown_scope_do_not_fall_through(self):
        for target_scope in ('scale_collection','unknown_source_scope'):
            r=row(attributes={'target_scope':target_scope})
            self.assertFalse(any(m['available'] for m in project_intervals(r,resolved_root=root(),notes={}).values()))

    def test_interpreted_relative_paths_retain_outer_root_provenance(self):
        r=root();r['source_paths']=['analysis.root_pitch_class','analysis.sonority']
        result=project_intervals(row(),resolved_root=r,notes={})
        self.assertEqual(result['H.chord.interval_pc.4']['source_paths'],['analysis.sonority','analysis.root_pitch_class'])


    def test_literal_root_survives_unavailable_and_unscoped_structural_fields(self):
        for separate in (
            dict(value=None,available=False,annotation_state='unreviewed'),
            known('first_inversion'),
        ):
            r=literal();r['fields']['analysis.inversion']=separate
            result=project_intervals(r,resolved_root=root(1),notes=notes())
            self.assertEqual(result['H.observed.root_pc']['value'],0)
            self.assertIs(result['H.observed.interval_pc.1']['value'],False)

    def test_pedal_scope_is_rejected_even_without_optional_key_attributes(self):
        r=observed();a=r['fields']['analysis.sonority']['attributes']
        a.pop('key_active_note_ids');a.pop('key_pitches');a['possible_pedal_held_note_ids']=['pedal_unknown']
        result=project_intervals(r,resolved_root=root(),notes=notes())
        self.assertFalse(any(m['available'] for m in result.values()))

    def test_malformed_source_attributes_fail_closed(self):
        r=observed();r['fields']['analysis.sonority']['attributes']=None
        self.assertFalse(any(m['available'] for m in project_intervals(r,resolved_root=root(),notes=notes()).values()))


    def test_explicit_interpreted_scope_admits_complete_template_without_chord_name(self):
        for location in ('value','attributes'):
            r=complete_template(row({'label':'unresolved_chord_name','tones':['C','E','G']}))
            r['fields']['analysis.sonority'][location]['target_scope']='interpreted_chord_components'
            result=project_intervals(r,resolved_root=root(),notes={})
            with self.subTest(location=location):
                self.assertIs(result['H.chord.interval_pc.1']['value'],False)
                self.assertIs(result['H.chord.interval_pc.4']['value'],True)


    def test_conflicting_observed_and_interpreted_scopes_preserve_only_absolute_set(self):
        for location,key in [('value','target_scope'),('attributes','target_scope'),('value','reading'),('attributes','reading')]:
            r=observed();r['fields']['analysis.sonority'][location][key]='interpreted_chord_components'
            result=project_intervals(r,resolved_root=root(),notes=notes())
            self.assertTrue(result['H.observed.pitch_class.0']['available'])
            self.assertFalse(result['H.observed.root_pc']['available'])
            self.assertFalse(result['H.observed.interval_pc.1']['available'])
            self.assertFalse(result['H.chord.interval_pc.0']['available'])


    def test_malformed_template_identity_or_evidence_cannot_license_negatives(self):
        for update in [dict(id=True),dict(id=1),dict(id='  '),dict(evidence=True),
                       dict(evidence='source'),dict(evidence=[]),dict(evidence=['']),
                       dict(evidence=['  ']),dict(evidence=[1])]:
            r=complete_template(row({'label':'C_major','tones':['C','E','G']}))
            r['note_template'].update(update)
            result=project_intervals(r,resolved_root=root(),notes={})
            with self.subTest(update=update):
                self.assertFalse(result['H.chord.interval_pc.1']['available'])


if __name__ == '__main__':
    unittest.main()
