"""Synthetic public chord-factorization contracts; never read source corpora."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

if importlib.util.find_spec('torch_geometric') is None:
    import music_critic
    if 'music_critic.tasks' not in sys.modules:
        package = types.ModuleType('music_critic.tasks')
        package.__path__ = [str(Path(music_critic.__file__).parent / 'tasks')]
        sys.modules['music_critic.tasks'] = package

from music_critic.tasks.chord_structure_projection import (
    FACTOR_FIELDS, VERSION, chord_structure_contract, factor_sonority,
)


class ChordStructureTests(unittest.TestCase):
    def test_major_add_ninth_does_not_create_seventh(self):
        factors = factor_sonority({'label': 'Cadd9'})
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertIs(factors['H.sonority.addition.9']['value'], True)
        self.assertFalse(factors['H.sonority.seventh']['available'])
        self.assertFalse(factors['H.sonority.extension.9']['available'])

    def test_extended_symbol_and_slash_bass_keep_compound(self):
        factors = factor_sonority({'label': 'Fmaj9/G'})
        for field, expected in [('H.sonority.root_pc', 5), ('H.sonority.core', 'major'),
                                ('H.sonority.seventh', '7'), ('H.sonority.extension.9', True),
                                ('H.bass_pc', 7), ('H.bass_degree', '9')]:
            self.assertEqual(factors[field]['value'], expected, field)
        self.assertFalse(factors['H.inversion']['available'])
        self.assertEqual(factors['H.sonority.root_pc']['state'], 'coarsened')

    def test_omitted_unlisted_ninth_is_explicit_false(self):
        factors = factor_sonority({'label': 'Gm11no9'})
        self.assertEqual(factors['H.sonority.core']['value'], 'minor')
        self.assertEqual(factors['H.sonority.seventh']['value'], 'b7')
        self.assertIs(factors['H.sonority.extension.11']['value'], True)
        self.assertIs(factors['H.sonority.extension.9']['value'], False)
        self.assertIs(factors['H.sonority.omission.9']['value'], True)

    def test_multiple_alterations_and_omissions(self):
        factors = factor_sonority({'label': 'E7(b9,#9,no3)'})
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertEqual(factors['H.sonority.seventh']['value'], 'b7')
        self.assertIs(factors['H.sonority.alteration.b9']['value'], True)
        self.assertIs(factors['H.sonority.alteration.#9']['value'], True)
        self.assertIs(factors['H.sonority.omission.3']['value'], True)

    def test_common_concatenations_have_exact_factors(self):
        examples = [
            ('B(add11,13)', 'major', None, ('addition.11', 'addition.13')),
            ('Ab69', 'major', None, ('addition.6', 'addition.9')),
            ('BbM713', 'major', '7', ('extension.13',)),
            ('Eb911', 'major', 'b7', ('extension.9', 'extension.11')),
            ('Bb13s11', 'major', 'b7', ('extension.13', 'alteration.#11')),
            ('B♭7 + #9,#11,#9', 'major', 'b7', ('alteration.#9', 'alteration.#11')),
        ]
        for label, core, seventh, positives in examples:
            with self.subTest(label=label):
                factors = factor_sonority({'label': label})
                self.assertEqual(factors['H.sonority.core']['value'], core)
                self.assertEqual(factors['H.sonority.seventh']['value'], seventh)
                for field in positives:
                    self.assertIs(factors['H.sonority.' + field]['value'], True, field)

    def test_contract_is_finite_and_self_hashed(self):
        import hashlib
        import json
        contract = chord_structure_contract()
        fingerprint = contract.pop('fingerprint')
        self.assertEqual(contract['schema'], VERSION)
        encoded = json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        self.assertEqual(fingerprint, hashlib.sha256(encoded.encode()).hexdigest())
        self.assertEqual(set(factor_sonority({})), set(FACTOR_FIELDS))
        self.assertNotIn('unknown', FACTOR_FIELDS['H.sonority.core'])

    def test_simple_quality_reconciles_without_negating_symbol_extensions(self):
        factors = factor_sonority({'quality': 'major triad', 'label': 'Fmaj9/G'})
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertEqual(factors['H.sonority.seventh']['value'], '7')
        self.assertIs(factors['H.sonority.extension.9']['value'], True)

    def test_core_conflict_masks_only_core(self):
        factors = factor_sonority({'quality': 'minor', 'label': 'C7b9'})
        self.assertEqual(factors['H.sonority.core']['state'], 'ambiguous')
        self.assertEqual(factors['H.sonority.seventh']['value'], 'b7')
        self.assertIs(factors['H.sonority.extension.b9']['value'], True)
        self.assertEqual(factors['H.sonority.root_pc']['value'], 0)

    def test_seventh_conflict_masks_only_seventh(self):
        factors = factor_sonority({'quality': 'major seventh chord', 'label': 'C7b9'})
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertEqual(factors['H.sonority.seventh']['state'], 'ambiguous')
        self.assertIs(factors['H.sonority.extension.b9']['value'], True)

    def test_root_conflict_does_not_erase_quality(self):
        factors = factor_sonority({'root': 'D', 'label': 'C7/E'})
        self.assertEqual(factors['H.sonority.root_pc']['state'], 'ambiguous')
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertEqual(factors['H.bass_pc']['value'], 4)
        self.assertFalse(factors['H.inversion']['available'])

    def test_unsupported_label_and_prose_never_guess_core(self):
        for label in ['Colorful dominant feel', 'C major-ish', 'C7 perhaps', 'literal_key_collection']:
            factors = factor_sonority({'label': label})
            self.assertFalse(factors['H.sonority.core']['available'], label)
            self.assertFalse(factors['H.sonority.root_pc']['available'], label)
        factors = factor_sonority({'quality': 'ominous minor seventh sort of chord', 'tones': ['C', 'Eb', 'G']})
        self.assertFalse(factors['H.sonority.core']['available'])

    def test_finite_extended_quality_aliases(self):
        for quality, core, seventh, extension in [
            ('dominant ninth', 'major', 'b7', '9'),
            ('dominant seventh with thirteenth', 'major', 'b7', '13'),
            ('diminished seventh with eleventh', 'diminished', 'bb7', '11'),
        ]:
            factors = factor_sonority({'quality': quality})
            self.assertEqual(factors['H.sonority.core']['value'], core)
            self.assertEqual(factors['H.sonority.seventh']['value'], seventh)
            self.assertIs(factors['H.sonority.extension.' + extension]['value'], True)
        self.assertFalse(factor_sonority({'quality': 'new dominant ninth prose'})['H.sonority.core']['available'])

    def test_legacy_label_is_explicitly_bounded(self):
        factors = factor_sonority({'label': 'Fsharp_major'})
        self.assertEqual(factors['H.sonority.root_pc']['value'], 6)
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertFalse(factor_sonority({'label': 'Fsharp_major_new_prose'})['H.sonority.core']['available'])

    def test_alt_keeps_core_and_seventh_without_guessing_alterations(self):
        factors = factor_sonority({'label': 'C7alt'})
        self.assertEqual(factors['H.sonority.core']['value'], 'major')
        self.assertEqual(factors['H.sonority.seventh']['value'], 'b7')
        for token in ('b5', '#5', 'b9', '#9', '#11', 'b13'):
            mapped = factors['H.sonority.alteration.' + token]
            self.assertEqual(mapped['state'], 'unsupported')
            self.assertIn('does not specify', mapped['diagnostic'])

    def test_bass_degree_without_extension_stays_simple(self):
        factors = factor_sonority({'label': 'C/D'})
        self.assertEqual(factors['H.bass_degree']['value'], '2')
        self.assertFalse(factors['H.inversion']['available'])

    def test_basic_spelled_inversions(self):
        for label, inversion, degree in [('C/E', 'first', '3'), ('Cm/Eb', 'first', 'b3'),
            ('C/G', 'second', '5'), ('Cdim/Gb', 'second', 'b5'),
            ('Caug/G#', 'second', '#5'), ('C7/Bb', 'third', 'b7'),
            ('Cmaj7/B', 'third', '7'), ('Cdim7/Bbb', 'third', 'bb7'), ('C/C', 'root', '1')]:
            with self.subTest(label=label):
                factors = factor_sonority({'label': label})
                self.assertEqual(factors['H.inversion']['value'], inversion)
                self.assertEqual(factors['H.bass_degree']['value'], degree)
        self.assertFalse(factor_sonority({'label': 'C7#9/D#'})['H.inversion']['available'])

    def test_omission_conflicts_are_local(self):
        for label, target in [('C9no9', 'extension.9'), ('Cadd9no9', 'addition.9')]:
            factors = factor_sonority({'label': label})
            self.assertEqual(factors['H.sonority.' + target]['state'], 'ambiguous')
            self.assertIs(factors['H.sonority.omission.9']['value'], True)
            self.assertEqual(factors['H.sonority.core']['value'], 'major')
        factors = factor_sonority({'label': 'C7no7'})
        self.assertEqual(factors['H.sonority.seventh']['state'], 'ambiguous')
        self.assertEqual(factors['H.sonority.core']['value'], 'major')

    def test_explicit_pc_priority_does_not_hide_disagreement(self):
        factors = factor_sonority({'root_pc': 1, 'root': 'C#', 'label': 'Dbmaj7'})
        self.assertEqual(factors['H.sonority.root_pc']['value'], 1)
        self.assertEqual(factors['H.sonority.root_pc']['state'], 'exact')
        self.assertEqual(factors['H.sonority.root_pc']['information_loss'], [])
        self.assertEqual(factor_sonority({'root_pc': True})['H.sonority.root_pc']['state'], 'invalid')
        self.assertEqual(factor_sonority({'root_pc': None, 'label': 'C7'})['H.sonority.root_pc']['state'], 'masked')

    def test_missing_and_wrong_types_do_not_become_negatives(self):
        self.assertTrue(all(not result['available'] for result in factor_sonority({}).values()))
        self.assertTrue(all(result['state'] == 'invalid' for result in factor_sonority('C7').values()))
        for value in [None, False, 7]:
            factors = factor_sonority({'label': value})
            self.assertFalse(factors['H.sonority.core']['available'])

    def test_input_and_contract_have_no_mutable_aliasing(self):
        from copy import deepcopy
        value = {'label': 'B♭7 + #9,#11,#9', 'quality': 'dominant seventh chord', 'tones': [60, 64]}
        before = deepcopy(value)
        first = factor_sonority(value)
        self.assertEqual(value, before)
        first['H.sonority.core']['value'] = 'bad'
        self.assertEqual(factor_sonority(value)['H.sonority.core']['value'], 'major')
        contract = chord_structure_contract()
        contract['fields']['H.sonority.core'].append('bad')
        self.assertNotIn('bad', FACTOR_FIELDS['H.sonority.core'])

    def test_third_unspecified_never_means_power(self):
        for quality in ['perfect_fifth_third_unspecified',
                        'root and fifth support; third unspecified',
                        'root and fifth; third unspecified']:
            self.assertFalse(factor_sonority({'quality': quality})['H.sonority.core']['available'])
        self.assertEqual(factor_sonority({'label': 'C5'})['H.sonority.core']['value'], 'power')

    def test_suspended_core_and_malformed_separators(self):
        for label in ['C7sus4', 'Csus4', 'Csus']:
            self.assertEqual(factor_sonority({'label': label})['H.sonority.core']['value'], 'sus4')
        for label in ['C7,,9', 'C7+', 'C7++9', 'C((7))', 'C(7']:
            self.assertFalse(factor_sonority({'label': label})['H.sonority.core']['available'], label)

    def test_minor_and_augmented_short_symbols_are_unambiguous(self):
        for label, core in [('C-', 'minor'), ('C+', 'augmented'), ('Cø7', 'diminished')]:
            factors = factor_sonority({'label': label})
            self.assertEqual(factors['H.sonority.root_pc']['value'], 0, label)
            self.assertEqual(factors['H.sonority.core']['value'], core, label)
        self.assertEqual(factor_sonority({'label': 'Cø7'})['H.sonority.seventh']['value'], 'b7')

    def test_altered_fifth_stays_a_fifth_for_basic_inversion(self):
        for label in ['C7#5/G#', 'Cm7b5/Gb']:
            factors = factor_sonority({'label': label})
            self.assertEqual(factors['H.inversion']['value'], 'second', label)

    def test_explicit_compound_bass_degree_reconciles_with_simple_slash(self):
        factors = factor_sonority({'label': 'C/D', 'bass_degree': '9'})
        self.assertEqual(factors['H.bass_degree']['value'], '9')
        self.assertFalse(factors['H.inversion']['available'])
        factors = factor_sonority({'label': 'C/D', 'bass_degree': 'b9'})
        self.assertEqual(factors['H.bass_degree']['state'], 'ambiguous')

    def test_explicit_omission_across_quality_and_symbol_is_reconciled(self):
        factors = factor_sonority({'quality': 'dominant ninth', 'label': 'C7no9'})
        self.assertEqual(factors['H.sonority.extension.9']['state'], 'ambiguous')
        self.assertEqual(factors['H.sonority.seventh']['value'], 'b7')
        self.assertIs(factors['H.sonority.omission.9']['value'], True)

    def test_nested_explicit_pitch_envelopes_keep_spelling_route(self):
        factors = factor_sonority({'root': {'value': 'C', 'available': True, 'annotation_state': 'known'},
                                   'quality': 'major', 'bass': 'E'})
        self.assertEqual(factors['H.sonority.root_pc']['value'], 0)
        self.assertEqual(factors['H.bass_degree']['value'], '3')
        self.assertEqual(factors['H.inversion']['value'], 'first')
        factors = factor_sonority({'root': {'value': None, 'available': False, 'annotation_state': 'unreviewed'},
                                   'label': 'C/E'})
        self.assertEqual(factors['H.sonority.root_pc']['state'], 'masked')
        self.assertFalse(factors['H.bass_degree']['available'])

    def test_omitted_bass_member_cannot_license_basic_inversion(self):
        for label in ['Cno1/C', 'Cno3/E', 'Cno5/G']:
            factors = factor_sonority({'label': label})
            self.assertEqual(factors['H.inversion']['state'], 'ambiguous', label)
            self.assertEqual(factors['H.sonority.core']['value'], 'major')

    def test_altered_or_unspecified_fifth_does_not_license_natural_fifth_bass(self):
        for label in ['C7#5/G', 'C7b5/G', 'C7alt/G']:
            with self.subTest(label=label):
                factors = factor_sonority({'label': label})
                self.assertFalse(factors['H.inversion']['available'])
                self.assertNotEqual(factors['H.inversion']['value'], 'second')
                self.assertEqual(factors['H.bass_degree']['value'], '5')
                self.assertEqual(factors['H.sonority.core']['value'], 'major')


if __name__ == '__main__':
    unittest.main()
