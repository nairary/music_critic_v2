"""Public tasks imports must not require an unrelated neural runtime."""

from pathlib import Path
import os
import importlib.util
import subprocess
import sys
import textwrap
import unittest


_REPOSITORY = Path(__file__).resolve().parents[1]


class TasksLazyFacadeTests(unittest.TestCase):
    def run_isolated(self, source):
        # -S excludes site packages: use real stdlib imports without fake modules.
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(_REPOSITORY / 'src')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run(
            [sys.executable, '-S', '-c', textwrap.dedent(source)],
            cwd=_REPOSITORY, env=environment, capture_output=True, text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_import_does_not_load_neural_dependencies(self):
        self.run_isolated('''
            import sys
            import music_critic.tasks
            assert not any(
                name.split('.')[0] in {'torch', 'torch_geometric', 'mido'}
                for name in sys.modules
            )
            assert 'music_critic.graph' not in sys.modules
        ''')


    def test_prior_public_names_order_and_origins_are_preserved(self):
        # Frozen from the eager facade's ImportFrom and __all__ AST before this
        # refactor; no dependency modules need to be imported to check parity.
        self.run_isolated('''
            import hashlib
            import json
            import music_critic.tasks as tasks
            assert len(tasks.__all__) == 184
            assert len(set(tasks.__all__)) == len(tasks.__all__)
            assert set(tasks._EXPORT_MODULES) == set(tasks.__all__)
            payload = [(name, tasks._EXPORT_MODULES[name]) for name in tasks.__all__]
            assert hashlib.sha256(json.dumps(payload).encode()).hexdigest() == (
                '8f12e88671b9a6845100ecd4fe3e9fd10dab65dfe37835353efb4d8d924f6a14'
            )
        ''')

    def test_dir_lists_deferred_exports_without_loading_them(self):
        self.run_isolated('''
            import sys
            import music_critic.tasks as tasks
            assert set(tasks.__all__).issubset(dir(tasks))
            assert dir(tasks) == sorted(set(dir(tasks)))
            assert not any(name.startswith('music_critic.tasks.') for name in sys.modules)
        ''')

    def test_pure_leaf_modules_import_through_normal_package(self):
        self.run_isolated('''
            import importlib
            import sys
            from music_critic.tasks import (
                event_theory_enrichment, whole_work_annotation, hiergraph_projection,
            )
            for module in (
                event_theory_enrichment, whole_work_annotation, hiergraph_projection,
            ):
                assert module is importlib.import_module(module.__name__)
            assert callable(hiergraph_projection.project_field)
            assert 'music_critic.graph' not in sys.modules
            assert 'music_critic.tasks.collator' not in sys.modules
            assert 'music_critic.tasks.alignment' not in sys.modules
            assert not any(
                name.split('.')[0] in {'torch', 'torch_geometric', 'mido'}
                for name in sys.modules
            )
        ''')

    def test_heavy_public_api_propagates_genuine_missing_dependency(self):
        self.run_isolated('''
            import importlib
            import sys
            import music_critic.tasks as tasks
            for load in (
                lambda: getattr(tasks, 'collate_multisource_samples'),
                lambda: importlib.import_module('music_critic.tasks.collator'),
            ):
                try:
                    load()
                except ModuleNotFoundError as error:
                    assert error.name == 'torch', error
                else:
                    raise AssertionError('stdlib-only process unexpectedly imported Torch')
            assert 'collate_multisource_samples' not in vars(tasks)
            from music_critic.tasks import hiergraph_projection
            assert callable(hiergraph_projection.project_field)
        ''')

    def test_target_types_keep_their_actual_midi_dependency_error(self):
        self.run_isolated('''
            import sys
            import music_critic.tasks as tasks
            try:
                from music_critic.tasks import TargetFamilySpec
            except ModuleNotFoundError as error:
                assert error.name == 'mido', error
            else:
                raise AssertionError('stdlib-only process unexpectedly imported mido')
            assert 'TargetFamilySpec' not in vars(tasks)
            assert 'music_critic.graph' not in sys.modules
        ''')

    def test_unknown_attribute_does_not_import_dependencies(self):
        self.run_isolated('''
            import sys
            import music_critic.tasks as tasks
            before = set(sys.modules)
            try:
                tasks.not_a_public_task
            except AttributeError as error:
                assert str(error) == (
                    "module 'music_critic.tasks' has no attribute 'not_a_public_task'"
                )
            else:
                raise AssertionError('unknown attribute should be rejected')
            assert set(sys.modules) == before
        ''')

    @unittest.skipUnless(importlib.util.find_spec('mido'), 'mido is not installed')
    def test_public_types_are_genuine_cached_definitions(self):
        # Isolate this too: sibling tests must not supply a substitute package.
        self.run_isolated('''
            import sys
            sys.path.extend(SITE_PACKAGE_PATHS)
            import music_critic.tasks as tasks
            from music_critic.tasks import TargetFamilySpec, TargetEncodingSpec
            from music_critic.tasks.ontology import TargetFamilySpec as Definition
            from music_critic.tasks.encoding import TargetEncodingSpec as Encoding
            assert TargetFamilySpec is Definition
            assert TargetEncodingSpec is Encoding
            assert tasks.TargetFamilySpec is Definition
            assert vars(tasks)['TargetFamilySpec'] is Definition
            assert 'music_critic.graph' not in sys.modules
            assert not any(
                name.split('.')[0] in {'torch', 'torch_geometric'}
                for name in sys.modules
            )
        '''.replace('SITE_PACKAGE_PATHS', repr([
            path for path in sys.path if 'site-packages' in path
        ])))


if __name__ == '__main__':
    unittest.main()
