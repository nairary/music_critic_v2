"""Synthetic identity metadata fixtures; no corpus payload access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).parents[1] / 'scripts/project_local_music_catalogs.py'
spec = importlib.util.spec_from_file_location('local_catalogs', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalCatalogTests(unittest.TestCase):
    def test_identity_list_preserves_alternatives_and_ignores_null(self):
        self.assertEqual(module.identity_values('["A", "B", null, ""]'), ['A', 'B'])
        self.assertEqual(module.identity_values('NA'), [])

    def test_file_check_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertEqual(module.file_state(root, 'missing.mid')['status'], 'missing')
            with self.assertRaises(ValueError):
                module.file_state(root, '../outside.mid')
            with self.assertRaises(ValueError):
                module.file_state(root, '/tmp/outside.mid')

    def test_split_disagreement_is_retained(self):
        result = module.membership('clip', 'TEST',
                                   {'clip': {'piece_id': 'piece:1'}},
                                   {'piece:1': {'split': 'train', 'component_fingerprint': 'c'}})
        self.assertEqual(result['native_split'], 'TEST')
        self.assertEqual(result['current_split'], 'train')
        self.assertTrue(result['native_current_conflict'])

    def test_unknown_current_is_not_train(self):
        result = module.membership('clip', 'TRAIN', {}, {})
        self.assertIsNone(result['current_split'])
        self.assertIsNone(result['native_current_conflict'])


if __name__ == '__main__':
    unittest.main()
