"""Synthetic leaf-API contracts; no corpus reads or scientific quality claims.

The project tasks facade eagerly imports optional PyG. In a stdlib-only runtime
import the real leaf modules normally through the lazy tasks facade.
Full tensor admission remains a separate check.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

from music_critic.tasks.hiergraph_projection import project_field


def known(value, state='known'):
    return dict(value=value if state == 'known' else None,
                available=state == 'known', annotation_state=state,
                origin='model_individual_context' if state == 'known' else 'unknown',
                confidence=None, evidence=['synthetic:source'],
                unknown_reason=None if state == 'known' else state)


class FieldTests(unittest.TestCase):
    def test_key_aliases_preserve_raw_and_independent_mode_mask(self):
        left = known('E-flat major')
        before = deepcopy(left)
        a = project_field('analysis.key', left)
        b = project_field('analysis.key', known('Eb major'))
        self.assertEqual(a['tonic_pc']['value'], 3)
        self.assertEqual(a['tonic_pc']['value'], b['tonic_pc']['value'])
        self.assertEqual(a['mode']['value'], 'major')
        self.assertEqual(left, before)
        bare = project_field('analysis.key', known('A'))
        self.assertEqual(bare['tonic_pc']['value'], 9)
        self.assertFalse(bare['mode']['available'])

    def test_known_modes_only_coarsen_other_unknown_is_masked(self):
        for spelling, tonic in [('B dorian', 11), ('B-flat blues reference', 10)]:
            result = project_field('analysis.key', known(spelling))
            self.assertEqual(result['tonic_pc']['value'], tonic)
            self.assertEqual(result['mode']['value'], 'other')
            self.assertEqual(result['mode']['state'], 'coarsened')
        for mode in ['unknown', None]:
            result = project_field('analysis.key', known({'tonic_pc': 0, 'mode': mode}))
            self.assertFalse(result['mode']['available'])
            self.assertTrue(result['tonic_pc']['available'])
        self.assertFalse(project_field('analysis.key', known('E_centered_chromatic_tonality'))['tonic_pc']['available'])

    def test_source_states_and_projection_states_remain_distinct(self):
        self.assertEqual(project_field('analysis.function', None)['value']['state'], 'missing')
        for state in ('not_applicable', 'unreviewed', 'unresolved', 'conflicting'):
            field = known(None, state)
            if state == 'not_applicable':
                field['applicability_reason'] = 'synthetic explicit inapplicability'
            result = project_field('analysis.function', field)['value']
            self.assertEqual(result['state'], 'masked')
            self.assertFalse(result['available'])
            self.assertEqual(result['diagnostic'], state)
        self.assertEqual(project_field('analysis.function', known('rare_unknown'))['value']['state'], 'unsupported')
        self.assertEqual(project_field('analysis.function', known(['tonic']))['value']['state'], 'invalid')

    def test_false_is_known_but_zero_is_invalid_membership(self):
        a = project_field('analysis.chord_membership', known(False))['value']
        b = project_field('analysis.chord_membership', known(0))['value']
        self.assertIs(a['value'], False)
        self.assertTrue(a['available'])
        self.assertFalse(b['available'])
        self.assertEqual(b['state'], 'invalid')

    def test_sonority_components_ignore_free_labels_and_report_conflicts(self):
        result = project_field('analysis.sonority', known({
            'label': 'literal_key_collection', 'root': 'Eb', 'root_pitch_class': 3,
            'quality': 'major', 'bass': 'G', 'tones': ['Eb', 'G', 'Bb']}))
        self.assertEqual(result['root_pc']['value'], 3)
        self.assertEqual(result['quality']['value'], 'major triad')
        self.assertEqual(result['bass_pc']['value'], 7)
        conflicting = project_field('analysis.sonority', known({'root': 'C', 'root_pitch_class': 2, 'quality': 'minor'}))
        self.assertEqual(conflicting['root_pc']['state'], 'ambiguous')
        self.assertEqual(conflicting['quality']['value'], 'minor triad')
        only_label = project_field('analysis.sonority', known({'label': 'C_major', 'tones': ['C']}))
        self.assertFalse(only_label['root_pc']['available'])
        self.assertFalse(only_label['quality']['available'])

    def test_quality_aliases_are_scoped_and_descriptions_deferred(self):
        for label in ('major', 'major triad'):
            self.assertEqual(project_field('analysis.sonority', known({'quality': label}))['quality']['value'], 'major triad')
        for label in ('major_core', 'major triad; incomplete realized support (omits F)', 'single_pitch'):
            result = project_field('analysis.sonority', known({'quality': label}))['quality']
            self.assertFalse(result['available'])
            self.assertEqual(result['state'], 'unsupported')
        self.assertFalse(project_field('analysis.function', known({'label': 'tonic'}))['value']['available'])

    def test_partial_tone_lists_never_make_negative_pitch_classes(self):
        result = project_field('analysis.sonority', known({'tones': ['C', 'E', 'G']}))
        for pc in (0, 4, 7):
            self.assertIs(result[f'pitch_class.{pc}']['value'], True)
        for pc in (1, 2, 3, 5, 6, 8, 9, 10, 11):
            self.assertFalse(result[f'pitch_class.{pc}']['available'])
            self.assertIsNone(result[f'pitch_class.{pc}']['value'])
        self.assertFalse(project_field('analysis.sonority', known({'tones': [60, 64, 67]}))['pitch_class.0']['available'])
        declared = project_field('analysis.sonority', known({'tones': [60, 64, 67], 'tones_format': 'midi'}))
        self.assertIs(declared['pitch_class.0']['value'], True)
        invalid = project_field('analysis.sonority', known({'tones': [True], 'tones_format': 'pitch_class'}))
        self.assertEqual(invalid['pitch_class.1']['state'], 'invalid')

    def test_explicit_integer_pc_does_not_accept_bool_or_strings(self):
        for raw in (True, False, '0', -1, 12):
            self.assertEqual(project_field('analysis.sonority', known({'root_pitch_class': raw}))['root_pc']['state'], 'invalid')
        self.assertEqual(project_field('analysis.sonority', known({'root_pitch_class': 0}))['root_pc']['value'], 0)

    def test_basic_inversion_has_no_special_bass_classes(self):
        for raw, value in [('root_position', 'root'), ('first_inversion', 'first'), ('second_inversion', 'second'), ('third_inversion', 'third')]:
            self.assertEqual(project_field('analysis.inversion', known(raw))['value']['value'], value)
        for raw in ('ninth_in_bass', 'nonchord_Bb_bass', 'pedal_bass'):
            self.assertFalse(project_field('analysis.inversion', known(raw))['value']['available'])

    def test_degrees_require_admitted_reference_and_preserve_compounds(self):
        from music_critic.tasks.hiergraph_projection import DEGREE_FRAMES
        for path, frame in DEGREE_FRAMES.items():
            no_frame = project_field(path, known('#9'))
            self.assertFalse(no_frame['generic']['available'])
            r = project_field(path, known('#9'), degree_frame=frame)
            self.assertEqual(r['generic']['value'], '#2')
            self.assertEqual(r['compound']['value'], '9')
            self.assertEqual(r['generic']['reference_frame'], frame)
            for raw, generic, compound in [('11', '4', '11'), ('b13', 'b6', '13'), ('b3', 'b3', None)]:
                r = project_field(path, known(raw), degree_frame=frame)
                self.assertEqual(r['generic']['value'], generic)
                self.assertEqual(r['compound']['value'], compound)
                self.assertEqual(r['compound']['available'], compound is not None)
            for raw in (False, 0, '8', '12', '#0', 'b3/5'):
                self.assertFalse(project_field(path, known(raw), degree_frame=frame)['generic']['available'])

    def test_field_local_categories_and_deferred_voice_roman(self):
        self.assertFalse(project_field('analysis.function', known('mixed'))['value']['available'])
        for path in ('P/analysis.function', 'P/analysis.process', 'analysis.temporal_role', 'analysis.voice_role'):
            self.assertTrue(project_field(path, known('mixed'))['value']['available'])
        for path, value in [('analysis.voice', 'voice_001'), ('analysis.roman', 'V/V')]:
            self.assertEqual(project_field(path, known(value))['value']['rule'], 'deferred_open_value')


from music_critic.data import CanonicalNote, CanonicalPiece, CanonicalTrack, PieceMetadata, RationalTime, dumps_piece, MeterEvent, TempoEvent, ProvenanceRecord
from music_critic.tasks.event_theory_enrichment import EnrichmentError, canonical
from music_critic.tasks.hiergraph_projection import (
    project_hiergraph_targets, build_projection_vocabulary, encode_projection,
    projection_contract, DEGREE_FRAMES, ONTOLOGY,
)


def q(n, d=1):
    return {'num': n, 'den': d}


def fixture():
    note = CanonicalNote(note_id='note:n', track_id='track:t', pitch=60, onset_qn=RationalTime(0),
        duration_qn=RationalTime(2), velocity=64, channel=0, program=0, is_percussion=False,
        is_grace=False, spelling_step=None, spelling_alter=None, staff=None, voice=None,
        articulations=None, dynamic=None, source_onset_ticks=None, source_duration_ticks=None,
        source_onset_seconds=None, source_duration_seconds=None, provenance_id=None)
    piece = CanonicalPiece(schema_version='2.0.0', piece_id='piece:p', dataset_name='synthetic',
        source_group_id='f', split='train', source_path=None, source_resolution=None,
        duration_qn=RationalTime(2), metadata=PieceMetadata(source_format='midi', title=None,
        creators=None, collection=None, movement_title=None, movement_number=None,
        genres=None, copyright=None, language=None), tracks=(CanonicalTrack('track:t', 0, None, None, 0, 0, False, None),), notes=(note,), bars=(), beats=(),
        tempo_events=(TempoEvent('tempo:0', RationalTime(0), 500000, None),), meter_events=(MeterEvent('meter:0', RationalTime(0), 4, 4, None),), key_signature_events=(), annotations=(), targets=(),
        provenance=(ProvenanceRecord('prov:synthetic', 'synthetic', 'fixture', None, None, None, None, None, (), ()),), quality_flags=())
    def row(identity, a, b, fields, **kw):
        return dict(id=identity, start_qn=q(a), end_qn=q(b), view_id='v', fields=fields, **kw)
    harmonies = [row('H0', 0, 1, {'analysis.function': known('tonic'), 'analysis.key': known('Eb major'),
        'analysis.sonority': known({'root': 'Eb', 'quality': 'major', 'tones': ['Eb', 'G', 'Bb']}),
        'analysis.inversion': known('root_position'), 'analysis.roman': known('I')}),
        row('H1', 1, 2, {'analysis.function': known('dominant'), 'analysis.key': known('Bb major')})]
    contexts = [row('h0', 0, 1, {'analysis.chord_membership': known(True),
        'analysis.chord_degree': known('#9'), 'analysis.key_degree': known('b3'),
        'analysis.temporal_role': known('structural'), 'analysis.voice_role': known('melody'),
        'analysis.voice': known('voice_001')}, note_id='note:n', harmony_id='H0'),
        row('h1', 1, 2, {'analysis.chord_membership': known(False)}, note_id='note:n', harmony_id='H1')]
    process = row('P0', 0, 2, {'analysis.function': known('tonic'), 'analysis.process': known('prolongation')},
                  object_type='harmonic_process', member_ids=['H0', 'h0', 'h1'])
    bundle = dict(schema='EventTheoryEnrichment@1', dataset_id=piece.dataset_name,
        piece_id=piece.piece_id, current_split='train', harmonies=harmonies, note_contexts=contexts,
        objects=[process], relations=[], provenance={'annotator': 'synthetic', 'human_gold': False})
    return piece, bundle


def project(piece, bundle, family='fit'):
    return project_hiergraph_targets(piece, bundle, family_id=family, degree_frames=DEGREE_FRAMES)


def fit(projections, heads=('H.function', 'h.chord_membership', 'P.function')):
    return build_projection_vocabulary(projections, fit_families=['fit'],
        split_fingerprint='a' * 64, active_fields=heads)


class ConsumerTests(unittest.TestCase):
    def test_preserves_source_identity_raw_provenance_and_held_note_contexts(self):
        piece, bundle = fixture()
        before, raw = deepcopy(bundle), dumps_piece(piece)
        p = project(piece, bundle)
        self.assertEqual(bundle, before)
        self.assertEqual(p['source'], before)
        self.assertEqual(dumps_piece(piece), raw)
        rows = p['bundle']['note_contexts']
        self.assertEqual([r['note_id'] for r in rows], ['note:n', 'note:n'])
        self.assertEqual([r['harmony_id'] for r in rows], ['H0', 'H1'])
        self.assertEqual([r['fields']['h.chord_membership']['value'] for r in rows], [True, False])
        self.assertEqual(p['source']['note_contexts'][0]['fields']['analysis.chord_degree']['value'], '#9')
        self.assertEqual(p['mappings']['h0']['h.chord_degree']['reference_frame'], 'H.root.major_reference')
        self.assertIsNone(rows[0]['fields']['h.chord_degree']['confidence'])
        self.assertEqual(rows[0]['fields']['h.chord_degree']['origin'], 'source_derived')
        self.assertEqual(p['source']['provenance'], before['provenance'])

    def test_explicit_key_components_fill_absence_and_mask_disagreement(self):
        piece, bundle = fixture()
        f = bundle['harmonies'][0]['fields']
        f['analysis.key'] = known('A')
        f['analysis.mode'] = known('minor')
        f['analysis.tonic_pc'] = known(9)
        p = project(piece, bundle)
        fields = p['bundle']['harmonies'][0]['fields']
        self.assertEqual(fields['H.key.mode']['value'], 'minor')
        self.assertEqual(fields['H.key.tonic_pc']['value'], 9)
        f['analysis.tonic_pc'] = known(0)
        p = project(piece, bundle)
        self.assertEqual(p['mappings']['H0']['H.key.tonic_pc']['state'], 'ambiguous')
        self.assertTrue(p['bundle']['harmonies'][0]['fields']['H.key.mode']['available'])
        f['analysis.key'] = known('A major')
        p = project(piece, bundle)
        self.assertEqual(p['mappings']['H0']['H.key.mode']['state'], 'ambiguous')

    def test_conflicting_sonority_and_explicit_root_are_component_local(self):
        piece, bundle = fixture()
        bundle['harmonies'][0]['fields']['analysis.root_pitch_class'] = known(0)
        p = project(piece, bundle)
        self.assertEqual(p['mappings']['H0']['H.sonority.root_pc']['state'], 'ambiguous')
        self.assertTrue(p['bundle']['harmonies'][0]['fields']['H.sonority.quality']['available'])

    def test_frozen_vocab_is_fit_only_deterministic_self_bound_and_typed(self):
        piece, bundle = fixture()
        p = project(piece, bundle)
        a = fit([p]); b = fit([p])
        self.assertEqual(a, b)
        self.assertNotIn('H.function', a['objects']['fields']['harmonic_process'])
        self.assertNotIn('P.function', a['event']['fields']['harmonies'])
        self.assertEqual(a['event']['fields']['note_contexts']['h.chord_membership'], [False, True])
        self.assertNotIn('analysis.roman', canonical(a))
        selection = deepcopy(bundle)
        selection['harmonies'][0]['fields']['analysis.function'] = known('coloristic')
        selected = project(piece, selection, 'selection')
        snapshot = deepcopy(a)
        n = encode_projection(piece, selected, a, split_fingerprint='a' * 64)
        self.assertEqual(n['targets']['harmonies']['values']['H.function'][0], -1)
        self.assertFalse(n['targets']['harmonies']['masks']['H.function'][0])
        self.assertEqual(n['targets']['oov_counts']['harmonies:H.function'], 1)
        self.assertEqual(a, snapshot)
        with self.assertRaisesRegex(EnrichmentError, 'non-fit'):
            fit([selected])
        with self.assertRaisesRegex(EnrichmentError, 'split'):
            encode_projection(piece, selected, a, split_fingerprint='b' * 64)
        a['event']['fields']['harmonies']['H.function'].append('coloristic')
        with self.assertRaisesRegex(EnrichmentError, 'fingerprint'):
            encode_projection(piece, p, a, split_fingerprint='a' * 64)

    def test_vocab_bundle_order_determinism_and_multiple_family_admission(self):
        from dataclasses import replace
        piece, bundle = fixture(); p = project(piece, bundle)
        second_piece = replace(piece, piece_id='other')
        other = deepcopy(bundle); other['piece_id'] = 'other'
        other['harmonies'][0]['fields']['analysis.function'] = known('coloristic')
        second = project(second_piece, other)
        self.assertEqual(fit([p, second]), fit([second, p]))
        with self.assertRaisesRegex(EnrichmentError, 'duplicate'):
            fit([p, p])

    def test_selection_new_fields_are_explicitly_counted_without_vocab_growth(self):
        piece, bundle = fixture()
        fit_source = deepcopy(bundle); fit_source['objects'] = []
        a = fit([project(piece, fit_source)])
        n = encode_projection(piece, project(piece, bundle, 'selection'), a, split_fingerprint='a' * 64)
        self.assertFalse(n['objects']['groups']['harmonic_process']['masks']['P.function'][0])
        self.assertEqual(n['objects']['oov_counts']['harmonic_process:P.function'], 1)

    def test_projection_hash_detects_mutation_and_requires_explicit_heads(self):
        piece, bundle = fixture(); p = project(piece, bundle)
        with self.assertRaisesRegex(EnrichmentError, 'active'):
            fit([p], heads=('analysis.roman',))
        with self.assertRaisesRegex(EnrichmentError, 'split'):
            build_projection_vocabulary([p], fit_families=['fit'], split_fingerprint='global TRAIN', active_fields=['H.function'])
        p['bundle']['harmonies'][0]['fields']['H.function']['value'] = 'coloristic'
        with self.assertRaisesRegex(EnrichmentError, 'fingerprint'):
            fit([p])

    def test_geometry_identity_and_global_test_fail_before_projection(self):
        for mutate in [lambda b: b.update(piece_id='wrong'), lambda b: b.update(current_split='test'),
                       lambda b: b['harmonies'][0].update(end_qn=q(3)),
                       lambda b: b['note_contexts'][0].update(start_qn={'num': 0, 'den': 2})]:
            piece, bundle = fixture(); mutate(bundle)
            with self.assertRaises(EnrichmentError):
                project(piece, bundle)

    def test_active_contract_contains_all_finite_categories_for_every_constructor(self):
        c = projection_contract()
        self.assertEqual(len(c['ontology']['H.function']), 7)
        self.assertEqual(len(c['ontology']['h.temporal_role']), 13)
        self.assertEqual(len(c['ontology']['h.voice_role']), 9)
        self.assertEqual(len(c['ontology']['P.process']), 12)
        self.assertNotIn('constructor', c)
        self.assertNotIn('analysis.voice', ONTOLOGY)

    def test_many_to_many_addresses_and_incoming_outgoing_decisions_preserved(self):
        piece, bundle = fixture()
        relation = dict(known(True), id='r', type='resolves_to', source_ids=['note:n', 'h0'],
                        target_ids=['h1'], start_qn=q(0), end_qn=q(2), view_id='v')
        bundle['relations'] = [relation]
        bundle['note_contexts'][0]['fields']['analysis.resolution'] = known({'status': 'linked', 'relation_ids': ['r']})
        bundle['note_contexts'][0]['fields']['analysis.preparation'] = known({'status': 'none', 'relation_ids': [], 'reason': 'synthetic reviewed'})
        p = project(piece, bundle)
        self.assertEqual(len(p['bundle']['relations']), 1)
        self.assertEqual(p['bundle']['relations'][0]['source_ids'], ['note:n', 'h0'])
        self.assertEqual(p['bundle']['relations'][0]['target_ids'], ['h1'])
        self.assertTrue(p['bundle']['note_contexts'][0]['fields']['h.resolution.linked']['value'])
        self.assertIs(p['bundle']['note_contexts'][0]['fields']['h.preparation.linked']['value'], False)
        self.assertEqual(p['source']['note_contexts'][0]['fields']['analysis.resolution']['value']['relation_ids'], ['r'])
        wrong = deepcopy(bundle)
        wrong['note_contexts'][0]['fields']['analysis.preparation'] = known({'status': 'linked', 'relation_ids': ['r']})
        p = project(piece, wrong)
        self.assertFalse(p['bundle']['note_contexts'][0]['fields']['h.preparation.linked']['available'])
        self.assertEqual(p['mappings']['h0']['h.preparation.linked']['state'], 'invalid')

    def test_explicit_none_contradiction_does_not_become_negative(self):
        piece, bundle = fixture()
        bundle['relations'] = [dict(known(True), id='r', type='resolves_to', source_ids=['h0'],
                                   target_ids=['h1'], start_qn=q(0), end_qn=q(2), view_id='v')]
        bundle['note_contexts'][0]['fields']['analysis.resolution'] = known({'status': 'none', 'relation_ids': [], 'reason': 'reviewed'})
        p = project(piece, bundle)
        self.assertEqual(p['mappings']['h0']['h.resolution.linked']['state'], 'invalid')
        self.assertFalse(p['bundle']['note_contexts'][0]['fields']['h.resolution.linked']['available'])

    def test_target_changes_do_not_change_raw_piece_and_missing_does_not_create_no_object(self):
        piece, bundle = fixture(); raw = dumps_piece(piece)
        p = project(piece, bundle); vocab = fit([p])
        bundle['harmonies'][0]['fields']['analysis.function'] = known(None, 'unresolved')
        changed = project(piece, bundle)
        result = encode_projection(piece, changed, vocab, split_fingerprint='a' * 64)
        self.assertFalse(result['targets']['harmonies']['masks']['H.function'][0])
        self.assertEqual(dumps_piece(piece), raw)
        self.assertEqual(len(result['targets']['harmonies']['ids']), 2)
        self.assertNotIn('no_object', canonical(result))

    def test_explicit_masked_components_block_lower_priority_descriptions(self):
        for state in ('conflicting', 'unresolved', 'unreviewed', 'not_applicable'):
            for path, target in [('analysis.mode', 'H.key.mode'), ('analysis.tonic_pc', 'H.key.tonic_pc'),
                                 ('analysis.root', 'H.sonority.root_pc')]:
                piece, bundle = fixture(); f = known(None, state)
                if state == 'not_applicable':
                    f['applicability_reason'] = 'synthetic reviewed inapplicability'
                bundle['harmonies'][0]['fields'][path] = f
                p = project(piece, bundle)
                self.assertFalse(p['bundle']['harmonies'][0]['fields'][target]['available'], (state, path))
                self.assertEqual(p['mappings']['H0'][target]['state'], 'masked')
                self.assertEqual(p['mappings']['H0'][target]['diagnostic'], state)

    def test_tone_information_loss_is_route_specific(self):
        for raw, fmt, expected in [([0, 4, 7], 'pitch_class', []),
                                   ([60, 64, 67], 'midi', ['octave_removed']),
                                   (['C', 'E', 'G'], 'spelled_pitch', ['enharmonic_spelling_removed'])]:
            r = project_field('analysis.sonority', known({'tones': raw, 'tones_format': fmt}))['pitch_class.0']
            self.assertEqual(r['information_loss'], expected)
            self.assertEqual(r['state'], 'coarsened' if expected else 'exact')

    def test_coarsening_is_admitted_before_fitting_and_unseen_stays_visible(self):
        piece, bundle = fixture(); p = project(piece, bundle)
        heads = ['H.key.tonic_pc', 'H.function']
        blocked = fit([p], heads=heads)
        allowed = build_projection_vocabulary([p], fit_families=['fit'], split_fingerprint='a' * 64,
            active_fields=heads, allow_coarsened_fields=['H.key.tonic_pc'])
        a = encode_projection(piece, p, blocked, split_fingerprint='a' * 64)
        b = encode_projection(piece, p, allowed, split_fingerprint='a' * 64)
        self.assertEqual(a['targets']['harmonies']['masks']['H.key.tonic_pc'], [False, False])
        self.assertEqual(b['targets']['harmonies']['masks']['H.key.tonic_pc'], [True, True])
        self.assertEqual(a['coarsening_excluded_counts']['H.key.tonic_pc'], 2)
        support = {row['value']: row for row in blocked['fit_support']['H.function']}
        self.assertEqual(support['tonic']['count'], 1)
        self.assertTrue(support['coloristic']['unseen_in_fit'])
        self.assertNotIn('coloristic', blocked['event']['fields']['harmonies']['H.function'])
        self.assertNotEqual(blocked['fingerprint'], allowed['fingerprint'])

    def test_raw_deferred_and_addressed_evidence_survive_numeric_encoding(self):
        piece, bundle = fixture()
        bundle['relations'] = [dict(known(True), id='r', type='prepares', source_ids=['h0'],
            target_ids=['h1'], start_qn=q(0), end_qn=q(2), view_id='v')]
        p = project(piece, bundle); result = encode_projection(piece, p, fit([p]), split_fingerprint='a' * 64)
        self.assertEqual(result['source'], bundle)
        deferred = {(r['id'], r['source_path']) for r in result['deferred']}
        self.assertIn(('H0', 'analysis.roman'), deferred)
        self.assertIn(('h0', 'analysis.voice'), deferred)
        self.assertEqual(result['addressed_relations'][0]['source_ids'], ['h0'])
        self.assertEqual(result['addressed_relations'][0]['target_ids'], ['h1'])
        self.assertEqual(result['targets']['relations']['ids'], [])
        self.assertEqual(result['source']['note_contexts'][0]['fields']['analysis.voice']['value'], 'voice_001')

    def test_new_event_field_absent_in_fit_is_retained_and_oov_counted(self):
        piece, bundle = fixture(); source = deepcopy(bundle)
        source['harmonies'] = []; source['objects'] = []
        for row in source['note_contexts']: row['harmony_id'] = None
        p = project(piece, source); vocabulary = fit([p])
        target = project(piece, bundle, 'selection')
        result = encode_projection(piece, target, vocabulary, split_fingerprint='a' * 64)
        self.assertEqual(result['targets']['harmonies']['values']['H.function'], [-1, -1])
        self.assertEqual(result['targets']['oov_counts']['harmonies:H.function'], 2)

    def test_relation_view_and_order_conflicts_are_masked(self):
        for sources, targets, view in [(['h1'], ['h0'], 'v'), (['h0'], ['h1'], 'other'), (['h0'], ['h0'], 'v')]:
            piece, bundle = fixture()
            bundle['relations'] = [dict(known(True), id='r', type='resolves_to', source_ids=sources,
                target_ids=targets, start_qn=q(0), end_qn=q(2), view_id=view)]
            p = project(piece, bundle)
            self.assertFalse(p['bundle']['relations'][0]['available'])
            self.assertEqual(p['mappings']['r']['p.addressed']['state'], 'invalid')

    def test_arbitrary_object_type_cannot_enter_event_namespaces(self):
        for kind in ('harmonies', 'note_contexts', 'phrase', 'ensemble'):
            piece, bundle = fixture()
            bundle['objects'][0]['object_type'] = kind
            p = project(piece, bundle)
            self.assertEqual(p['bundle']['objects'][0]['fields'], {})
            self.assertEqual(p['mappings']['P0'], {})
            self.assertTrue(any(r['id'] == 'P0' for r in p['deferred']))

    def test_inversion_quality_cardinality_conflict_masks_only_inversion(self):
        piece, bundle = fixture()
        bundle['harmonies'][0]['fields']['analysis.inversion'] = known('third_inversion')
        p = project(piece, bundle)
        self.assertEqual(p['mappings']['H0']['H.inversion']['state'], 'ambiguous')
        self.assertFalse(p['bundle']['harmonies'][0]['fields']['H.inversion']['available'])
        self.assertTrue(p['bundle']['harmonies'][0]['fields']['H.sonority.quality']['available'])
        bundle['harmonies'][0]['fields']['analysis.sonority']['value']['quality'] = 'dominant_seventh'
        p = project(piece, bundle)
        self.assertEqual(p['bundle']['harmonies'][0]['fields']['H.inversion']['value'], 'third')

    def test_degree_without_bound_H_retains_raw_but_masks_reference(self):
        piece, bundle = fixture()
        bundle['note_contexts'][0]['harmony_id'] = None
        p = project(piece, bundle)
        self.assertFalse(p['bundle']['note_contexts'][0]['fields']['h.chord_degree']['available'])
        self.assertEqual(p['mappings']['h0']['h.chord_degree']['rule'], 'missing_H_reference')
        self.assertEqual(p['source']['note_contexts'][0]['fields']['analysis.chord_degree']['value'], '#9')

    def test_single_source_zero_confidence_is_preserved_and_not_missing(self):
        piece, bundle = fixture()
        bundle['note_contexts'][1]['fields']['analysis.chord_membership']['confidence'] = 0
        p = project(piece, bundle)
        field = p['bundle']['note_contexts'][1]['fields']['h.chord_membership']
        self.assertIs(field['value'], False)
        self.assertEqual(field['confidence'], 0)
        self.assertIsNotNone(field['confidence'])
        self.assertIsNone(p['bundle']['note_contexts'][0]['fields']['h.chord_membership']['confidence'])

    def test_aliases_encode_identically_without_erasing_source_spelling(self):
        piece, bundle = fixture(); p = project(piece, bundle)
        vocabulary = build_projection_vocabulary([p], fit_families=['fit'], split_fingerprint='a' * 64,
            active_fields=['H.key.tonic_pc'], allow_coarsened_fields=['H.key.tonic_pc'])
        other = deepcopy(bundle); other['harmonies'][0]['fields']['analysis.key']['value'] = 'E-flat major'
        selected = project(piece, other, 'selection')
        a = encode_projection(piece, p, vocabulary, split_fingerprint='a' * 64)
        b = encode_projection(piece, selected, vocabulary, split_fingerprint='a' * 64)
        self.assertEqual(a['targets']['harmonies']['values']['H.key.tonic_pc'], b['targets']['harmonies']['values']['H.key.tonic_pc'])
        self.assertNotEqual(a['source_fingerprint'], b['source_fingerprint'])
        self.assertEqual(b['source']['harmonies'][0]['fields']['analysis.key']['value'], 'E-flat major')

    def test_roman_does_not_infer_function_and_equal_functions_keep_instances(self):
        piece, bundle = fixture()
        bundle['harmonies'][0]['fields']['analysis.function'] = known(None, 'unresolved')
        for raw in ('I', 'iii', 'vi', 'V/V'):
            bundle['harmonies'][0]['fields']['analysis.roman'] = known(raw)
            self.assertFalse(project(piece, bundle)['bundle']['harmonies'][0]['fields']['H.function']['available'])
        bundle['harmonies'][0]['fields']['analysis.function'] = known('dominant')
        rows = project(piece, bundle)['bundle']['harmonies']
        self.assertEqual([r['fields']['H.function']['value'] for r in rows], ['dominant', 'dominant'])
        self.assertEqual([r['id'] for r in rows], ['H0', 'H1'])

    def test_every_field_manifest_binds_scope_type_frames_semantic_ids_and_masks(self):
        from music_critic.tasks.hiergraph_projection import projection_field_manifest, VERSION
        manifest = projection_field_manifest()
        self.assertEqual(set(manifest), set(ONTOLOGY))
        self.assertEqual(manifest['H.function']['task_id'], 'hiergraph.v1.H.function')
        self.assertEqual(manifest['h.chord_degree']['reference_frame'], 'H.root.major_reference')
        self.assertEqual(manifest['H.sonority.pitch_class.0']['supervision_regime'], 'positive_unlabeled')
        for name, row in manifest.items():
            self.assertTrue(row['source_paths'])
            self.assertEqual(row['semantic_ids'], list(ONTOLOGY[name]))
            self.assertEqual(row['projection_version'], VERSION)
            self.assertEqual(row['unavailable_sentinel'], -1)
            self.assertFalse(row['default_active'])

    def test_explicit_component_priority_does_not_invent_information_loss(self):
        result = project_field('analysis.sonority', known({'root_pitch_class': 3, 'root': 'Eb'}))['root_pc']
        self.assertEqual(result['value'], 3)
        self.assertEqual(result['state'], 'exact')
        self.assertEqual(result['information_loss'], [])
        self.assertTrue(any(r['state'] == 'coarsened' for r in result['component_routes']))
        piece, bundle = fixture()
        fields = bundle['harmonies'][0]['fields']
        fields['analysis.key'] = known('E_centered_chromatic_tonality')
        fields['analysis.tonic_pc'] = known(4)
        result = project(piece, bundle)['mappings']['H0']['H.key.tonic_pc']
        self.assertEqual(result['value'], 4)
        self.assertEqual(result['state'], 'exact')
        self.assertTrue(any(r['state'] == 'unsupported' for r in result['component_routes']))

    def test_nested_key_conflict_cannot_be_revived_by_explicit_component(self):
        piece, bundle = fixture(); fields = bundle['harmonies'][0]['fields']
        fields['analysis.key'] = known({'tonic_pc': 0, 'tonic': 'D', 'mode': 'major'})
        fields['analysis.tonic_pc'] = known(0)
        p = project(piece, bundle)
        self.assertFalse(p['bundle']['harmonies'][0]['fields']['H.key.tonic_pc']['available'])
        self.assertEqual(p['mappings']['H0']['H.key.tonic_pc']['state'], 'ambiguous')
        self.assertEqual(p['bundle']['harmonies'][0]['fields']['H.key.mode']['value'], 'major')

    def test_nested_sonority_conflict_cannot_be_revived_by_explicit_component(self):
        piece, bundle = fixture(); fields = bundle['harmonies'][0]['fields']
        fields['analysis.sonority'] = known({'root_pitch_class': 0, 'root': 'D', 'quality': 'major'})
        fields['analysis.root_pitch_class'] = known(0)
        p = project(piece, bundle)
        self.assertFalse(p['bundle']['harmonies'][0]['fields']['H.sonority.root_pc']['available'])
        self.assertEqual(p['mappings']['H0']['H.sonority.root_pc']['state'], 'ambiguous')
        self.assertEqual(p['bundle']['harmonies'][0]['fields']['H.sonority.quality']['value'], 'major triad')

    @unittest.skipUnless(importlib.util.find_spec('torch_geometric') is not None, 'existing full ontology dependencies unavailable')
    def test_regular_encoding_policy_binds_fit_ids_not_ontology_order(self):
        from music_critic.tasks.hiergraph_projection import projection_encoding_specs
        piece, bundle = fixture(); p = project(piece, bundle); vocabulary = fit([p])
        specs = {s.task_id: s for s in projection_encoding_specs(vocabulary)}
        self.assertEqual(specs['hiergraph.v1.H.function'].vocabulary,
                         tuple(canonical(v) for v in vocabulary['event']['fields']['harmonies']['H.function']))
        self.assertEqual(specs['hiergraph.v1.H.roman'].encoding_kind, 'open_string_cpu')


class FactorIntegrationTests(unittest.TestCase):
    def test_extended_chord_preserves_geometry_and_separate_factors(self):
        piece, bundle = fixture()
        bundle['harmonies'][0]['fields']['analysis.sonority'] = known({'label': 'Fmaj9/G'})
        before = deepcopy(bundle)
        p = project(piece, bundle)
        f = p['bundle']['harmonies'][0]['fields']
        self.assertEqual(f['H.sonority.core']['value'], 'major')
        self.assertEqual(f['H.sonority.seventh']['value'], '7')
        self.assertIs(f['H.sonority.extension.9']['value'], True)
        self.assertEqual(f['H.bass_degree']['value'], '9')
        self.assertFalse(f['H.inversion']['available'])
        self.assertEqual(bundle, before)
        self.assertEqual(p['bundle']['harmonies'][0]['start_qn'], before['harmonies'][0]['start_qn'])
        self.assertEqual(len(p['bundle']['note_contexts']), len(before['note_contexts']))

    def test_masked_explicit_root_blocks_label_fallback(self):
        piece, bundle = fixture(); f = bundle['harmonies'][0]['fields']
        f['analysis.sonority'] = known({'label': 'C7(b9)'})
        f['analysis.root_pitch_class'] = known(None, 'unresolved')
        p = project(piece, bundle)
        self.assertFalse(p['bundle']['harmonies'][0]['fields']['H.sonority.root_pc']['available'])
        self.assertTrue(p['bundle']['harmonies'][0]['fields']['H.sonority.extension.b9']['available'])

    def test_explicit_inversion_conflict_with_slash_bass_masks_inversion(self):
        piece, bundle = fixture(); f = bundle['harmonies'][0]['fields']
        f['analysis.sonority'] = known({'label': 'C7/E'})
        f['analysis.inversion'] = known('root_position')
        p = project(piece, bundle)
        self.assertFalse(p['bundle']['harmonies'][0]['fields']['H.inversion']['available'])
        self.assertEqual(p['bundle']['harmonies'][0]['fields']['H.bass_pc']['value'], 4)

    def test_outer_root_or_bass_conflict_masks_derived_degree_and_inversion(self):
        for path,value in [('analysis.root','D'),('analysis.bass','G')]:
            piece,bundle=fixture(); fields=bundle['harmonies'][0]['fields']
            fields['analysis.sonority']=known({'label':'C/E'})
            fields.pop('analysis.inversion',None)
            fields[path]=known(value)
            p=project(piece,bundle); out=p['bundle']['harmonies'][0]['fields']
            self.assertFalse(out['H.bass_degree']['available'],path)
            self.assertFalse(out['H.inversion']['available'],path)
            self.assertTrue(out['H.sonority.core']['available'])

    def test_explicit_bass_degree_does_not_require_absolute_root_or_bass(self):
        piece,bundle=fixture(); fields=bundle['harmonies'][0]['fields']
        fields['analysis.sonority']=known({'quality':'major','bass_degree':'3'})
        fields.pop('analysis.inversion',None)
        p=project(piece,bundle); out=p['bundle']['harmonies'][0]['fields']
        self.assertFalse(out['H.sonority.root_pc']['available'])
        self.assertFalse(out['H.bass_pc']['available'])
        self.assertEqual(out['H.bass_degree']['value'],'3')
        self.assertEqual(out['H.inversion']['value'],'first')

    def test_independently_authored_inversion_survives_unknown_root(self):
        piece,bundle=fixture(); fields=bundle['harmonies'][0]['fields']
        fields['analysis.sonority']=known({'quality':'major'})
        fields['analysis.root']=known(None,'unresolved')
        fields['analysis.inversion']=known('first_inversion')
        p=project(piece,bundle); out=p['bundle']['harmonies'][0]['fields']
        self.assertFalse(out['H.bass_degree']['available'])
        self.assertEqual(out['H.inversion']['value'],'first')

    def test_compact_projection_keeps_authenticated_refs_and_numeric_parity(self):
        from music_critic.tasks.hiergraph_projection import compact_projection
        piece, bundle = fixture(); p = project(piece, bundle)
        bindings = {role: dict(path=role+'.json', sha256=c*64, size_bytes=123)
                    for role,c in [('bundle','a'),('canonical_piece','b')]}
        c = compact_projection(p, bindings)
        self.assertEqual(c['source']['schema'], 'HiergraphSourceBinding@1')
        self.assertEqual(c['source']['bindings'], bindings)
        self.assertEqual(c['source_fingerprint'], p['source_fingerprint'])
        self.assertEqual(c['bundle'], p['bundle'])
        vc = fit([c]); vf = fit([p])
        self.assertEqual(vc['event'], vf['event'])
        ec = encode_projection(piece,c,vc,split_fingerprint='a'*64)
        ef = encode_projection(piece,p,vf,split_fingerprint='a'*64)
        self.assertEqual(ec['targets'], ef['targets'])
        changed = deepcopy(c); changed['source']['bindings']['bundle']['sha256']='c'*64
        with self.assertRaises(EnrichmentError): fit([changed])
        with self.assertRaises(EnrichmentError): compact_projection(p, {'bundle':bindings['bundle']})

    def test_direct_bound_projection_matches_post_compaction(self):
        from music_critic.tasks.hiergraph_projection import compact_projection
        piece, bundle = fixture()
        bindings = {role: dict(path=role+'.json', sha256=c*64, size_bytes=123)
                    for role,c in [('bundle','a'),('canonical_piece','b')]}
        full = project(piece,bundle)
        direct = project_hiergraph_targets(piece,bundle,family_id='fit',degree_frames=dict(DEGREE_FRAMES),source_binding=bindings)
        compact = compact_projection(full,bindings)
        self.assertEqual(direct['bundle'], compact['bundle'])
        self.assertEqual(direct['source'], compact['source'])
        self.assertEqual(direct['mappings'], compact['mappings'])


class SharedPrimitiveTests(unittest.TestCase):
    def test_exact_shared_vocabulary_and_literal_pitch_grammar(self):
        from music_critic.tasks.harmonic_projection_primitives import DEGREES, spelling_pitch_class, COMMON_QUALITY_INTERVALS
        self.assertEqual(len(DEGREES), 49)
        self.assertEqual(DEGREES[:7], ('bbb1', 'bb1', 'b1', '1', '#1', '##1', '###1'))
        for raw, pc in [('C', 0), ('Eb', 3), ('E-', 3), ('F##', 7), ('b#', 0), ('Cb', 11)]:
            self.assertEqual(spelling_pitch_class(raw), pc)
        for raw in ('Eb major', 'E-flat', 'C4', 'B♭', 'Cb#', 'H'):
            self.assertIsNone(spelling_pitch_class(raw))
        self.assertEqual(COMMON_QUALITY_INTERVALS['major triad'], (0, 4, 7))
        self.assertEqual(COMMON_QUALITY_INTERVALS['dominant seventh chord'], (0, 4, 7, 10))


class IntervalIntegrationTests(unittest.TestCase):
    def test_new_contract_keeps_scoped_intervals_separate(self):
        contract = projection_contract()
        self.assertEqual(contract['schema'], 'HiergraphProjection@3.0.0')
        for scope in ('H.chord.interval_pc', 'H.observed.pitch_class', 'H.observed.interval_pc'):
            for pc in range(12):
                self.assertEqual(contract['fields'][f'{scope}.{pc}']['semantic_ids'], [False, True])
        self.assertIn('H.observed.root_pc', contract['fields'])
        self.assertEqual(contract['fields']['H.sonority.pitch_class.0']['supervision_regime'], 'positive_unlabeled')

    def test_root_and_spelled_tones_yield_partial_intervals(self):
        piece, bundle = fixture()
        bundle['harmonies'][0]['fields']['analysis.sonority']['value']['tones'] = ['Eb3', 'G4', 'Bb4']
        p = project(piece, bundle)
        fields = p['bundle']['harmonies'][0]['fields']
        for pc in (0, 4, 7):
            self.assertIs(fields[f'H.chord.interval_pc.{pc}']['value'], True)
        self.assertFalse(fields['H.chord.interval_pc.2']['available'])
        self.assertTrue(fields['H.sonority.pitch_class.3']['available'])
        self.assertFalse(fields['H.observed.interval_pc.4']['available'])
        self.assertEqual(p['source']['harmonies'][0]['fields']['analysis.sonority']['value']['tones'], ['Eb3', 'G4', 'Bb4'])

    def test_final_outer_root_conflict_masks_relative_not_source_tones(self):
        piece, bundle = fixture()
        bundle['harmonies'][0]['fields']['analysis.root_pitch_class'] = known(0)
        p = project(piece, bundle)
        fields = p['bundle']['harmonies'][0]['fields']
        self.assertFalse(fields['H.sonority.root_pc']['available'])
        self.assertTrue(fields['H.sonority.pitch_class.3']['available'])
        self.assertTrue(all(not fields[f'H.chord.interval_pc.{pc}']['available'] for pc in range(12)))

    def test_v3_vocab_and_numeric_bind_scoped_field_ids(self):
        piece, bundle = fixture(); p = project(piece, bundle)
        name = 'H.chord.interval_pc.4'
        v = build_projection_vocabulary([p], fit_families=['fit'], split_fingerprint='a'*64,
            active_fields=[name], allow_coarsened_fields=[name])
        self.assertEqual(v['schema'], 'HiergraphVocabulary@3')
        n = encode_projection(piece,p,v,split_fingerprint='a'*64)
        self.assertEqual(n['schema'], 'HiergraphNumericTargets@3')
        self.assertEqual(n['targets']['harmonies']['masks'][name], [True, False])
        self.assertEqual(v['event']['fields']['harmonies'][name], [True])
        old = deepcopy(v); old['schema'] = 'HiergraphVocabulary@2'
        with self.assertRaises(EnrichmentError): encode_projection(piece,p,old,split_fingerprint='a'*64)
