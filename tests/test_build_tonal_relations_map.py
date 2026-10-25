"""Synthetic converter checks; no theory truth is assumed."""
import copy
import importlib.util
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildTests(unittest.TestCase):
    def fixture(self):
        return {'schema': 'tonal-relations-review@1', 'title': 'Synthetic', 'cases': [{
            'id': 'x', 'title': 'Synthetic', 'genres': ['synthetic'], 'membership': 'train',
            'time_unit': 'quarter_note', 'time_origin': 'fixture', 'limitations': ['fixture only'],
            'assets': [{'locator': 'fixture.json', 'sha256': 'a' * 64}],
            'events': [
                {'id': 'bass', 'start': '0/1', 'duration': '2/1', 'pitch': 48, 'voice': 'track 0', 'source_ref': 'row 1'},
                {'id': 'upper', 'start': '1/1', 'duration': '1/1', 'pitch': 67, 'voice': 'track 1', 'source_ref': 'row 2'}],
            'claims': [{'id': 'claim', 'source_url': 'https://example.org/source', 'source_anchor': 'fixture',
                        'statement': 'Synthetic relation', 'relation': 'held_bass', 'status': 'partial',
                        'assessment': 'Not gold', 'participants': [{'role': 'bass', 'event_ids': ['bass']},
                                                                  {'role': 'upper', 'event_ids': ['upper']}],
                        'alternatives': ['Other interpretation'], 'missing': ['Expert review']}]}]}

    def test_sustained_bass_enters_later_sonority_without_relabeling_source(self):
        builder = load('build_tonal_relations_map')
        source = self.fixture()
        original = copy.deepcopy(source)
        result = builder.build_map(source, {})
        case = result['cases'][0]
        chord = next(o for o in case['objects'] if o['kind'] == 'sonority')
        self.assertEqual((chord['start'], chord['end'], chord['event_ids']), ('1/1', '2/1', ['bass', 'upper']))
        self.assertEqual(case['claims'], source['cases'][0]['claims'])
        self.assertEqual(source, original)
        self.assertFalse(case['relations'][0]['target_available'])
        self.assertEqual(case['coverage']['part']['status'], 'unavailable')

    def test_explicit_part_keeps_basis_and_does_not_create_other_levels(self):
        builder = load('build_tonal_relations_map')
        obj = {'id': 'part:1', 'kind': 'part', 'label': 'Documented source part',
               'start': '0/1', 'end': '2/1', 'event_ids': ['bass'], 'basis': 'Fixture source part ID, not voice inference'}
        plan = {'x': {'objects': [obj]}}
        result = builder.build_map(self.fixture(), plan)
        case = result['cases'][0]
        self.assertIn(obj, case['objects'])
        self.assertEqual(case['coverage']['part']['status'], 'partial')
        self.assertEqual(case['coverage']['phrase']['status'], 'unavailable')
        self.assertEqual(plan, {'x': {'objects': [obj]}})

    def test_excluded_case_cannot_gain_objects_from_plan(self):
        builder = load('build_tonal_relations_map')
        source = self.fixture()
        case = source['cases'][0]
        case.update(membership='excluded', events=[])
        case['claims'][0].update(status='rejected', participants=[], missing=['Not admitted'])
        result = builder.build_map(source, {})['cases'][0]
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['relations'], [])
        with self.assertRaises(ValueError):
            builder.build_map(source, {'x': {'objects': [{'kind': 'part'}]}})

    def test_unknown_plan_case_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            load('build_tonal_relations_map').build_map(self.fixture(), {'typo': {}})

    def test_explicit_source_window_and_context_are_both_retained(self):
        source = self.fixture()
        claim = source['cases'][0]['claims'][0]
        claim['context_event_ids'] = ['bass']
        claim['source_window_event_ids'] = ['upper', 'bass']
        case = load('build_tonal_relations_map').build_map(source, {})['cases'][0]
        self.assertEqual(case['relations'][0]['context_event_ids'], ['bass', 'upper'])
        self.assertEqual(case['claims'][0], claim)

    def test_hash_mismatch_fails_before_json_decode_or_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'source').write_text('INVALID JSON')
            (path / 'plan').write_text('{}')
            run = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'scripts/build_tonal_relations_map.py'),
                                  str(path / 'source'), str(path / 'plan'), str(path / 'result'),
                                  '--source-sha256', '0' * 64], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('SHA-256 mismatch', run.stderr)
            self.assertNotIn('JSONDecodeError', run.stderr)
            self.assertFalse((path / 'result').exists())


if __name__ == '__main__':
    unittest.main()
