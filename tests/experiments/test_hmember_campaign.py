"""Supervisor tests use only temporary metadata and short owned subprocesses."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('hmember_campaign', ROOT / 'scripts/run_hmember_campaign.py')
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def timing(update=0.1, forward=0.001):
    return {kind: dict(update_seconds=update, forward_seconds=forward) for kind in campaign.KINDS}


def launch_binding(root):
    dataset = root / 'dataset'
    dataset.mkdir()
    manifest = dataset / 'manifest.json'
    manifest.write_text('{}')
    archive = root / 'source.tar'
    archive.write_bytes(b'synthetic source archive')
    def bound(path):
        data = path.read_bytes()
        return dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    source = dict(dataset_manifest=bound(manifest), source_archive=dict(bound(archive), path='source.tar'))
    path = root / 'binding.json'
    path.write_text(json.dumps(source))
    return dataset, path


class CampaignTests(unittest.TestCase):
    def test_exclusive_campaign_lock_refuses_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'campaign'
            campaign._claim_campaign(path)
            with self.assertRaises(FileExistsError):
                campaign._claim_campaign(path)
            self.assertTrue((path / 'exclusive.lock').exists())

    def test_common_budget_uses_slowest_condition_and_includes_full_eval(self):
        times = timing()
        times['graph']['update_seconds'] = 1
        result = campaign._choose_steps(times, 100, 200)
        self.assertEqual(result['evaluation_queries'], 400)
        self.assertEqual(result['steps'], 332)
        self.assertTrue(result['accepted'])
        self.assertEqual(campaign._choose_steps(timing(update=0.001), 1, 1)['steps'], 2000)

    def test_expensive_full_eval_fails_without_reducing_query_count(self):
        result = campaign._choose_steps(timing(forward=1), 1000, 1000)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['steps'], 0)
        self.assertEqual(result['evaluation_queries'], 3000)
        times = timing()
        times['graph']['forward_seconds'] = float('nan')
        with self.assertRaises(ValueError):
            campaign._choose_steps(times, 1, 1)

    def test_checkpoint_overhead_is_included_in_common_budget(self):
        times = timing(update=1)
        baseline = campaign._choose_steps(times, 10, 10)
        for values in times.values():
            values['checkpoint_seconds'] = 3
        with_checkpoints = campaign._choose_steps(times, 10, 10)
        self.assertLess(with_checkpoints['steps'], baseline['steps'])
        self.assertEqual(with_checkpoints['estimated_checkpoint_seconds'], 27)

    def test_owned_success_and_timeout_reap_the_child(self):
        with tempfile.TemporaryDirectory() as directory:
            success = campaign._run_owned([sys.executable, '-c', "print('ok')"], Path(directory) / 'ok.log', 2)
            self.assertEqual(success['returncode'], 0)
            self.assertFalse(success['timed_out'])
            self.assertTrue(success['leader_reaped'])
            timeout = campaign._run_owned([sys.executable, '-c', 'import time; time.sleep(10)'], Path(directory) / 'timeout.log', 0.05)
            self.assertTrue(timeout['timed_out'])
            self.assertTrue(timeout['leader_reaped'])
            self.assertLess(timeout['wall_seconds'], 1)

    def test_post_candidate_support_requires_mixed_h_and_independent_families(self):
        support = {split: dict(mixed_h=20, mixed_works=['a', 'b'], mixed_families=['f1', 'f2'])
                   for split in ('train', 'selection', 'dev')}
        campaign._check_support(support)
        support['dev']['mixed_h'] = 19
        with self.assertRaisesRegex(ValueError, 'dev'):
            campaign._check_support(support)
        support['dev']['mixed_h'] = 20
        support['dev']['mixed_families'] = ['f1']
        with self.assertRaisesRegex(ValueError, 'independent families'):
            campaign._check_support(support)

    def test_remaining_only_moves_unfinished_records_forward(self):
        states = {'planned': 'planned', 'running': 'running', 'done': 'completed', 'failed': 'failed'}
        writes = []
        def read(path):
            return dict(status=states[Path(path).stem])
        def move(path, status, **kwargs):
            writes.append((Path(path).stem, status))
        with patch.object(campaign, 'load_record', read), patch.object(campaign, 'transition', move):
            campaign._abort_remaining('unused', list(states), 'synthetic stop')
        self.assertEqual(writes, [('planned', 'aborted'), ('running', 'aborted')])

    def test_progress_fallback_preserves_updates_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            checkpoint = dict(path='checkpoint-000007.pt', sha256='a' * 64, size_bytes=10)
            path.write_text(json.dumps(dict(kind='graph', seed=17, status='running',
                requested_updates=50, applied_updates=7, checkpoints=[checkpoint])))
            result = campaign._condition_result(directory, 'graph', 17, 50, True)
            self.assertEqual(result['status'], 'aborted')
            self.assertEqual(result['applied_updates'], 7)
            self.assertEqual(result['checkpoints'], [checkpoint])
            self.assertTrue(result['progress_fallback'])
            with self.assertRaises(ValueError):
                campaign._condition_result(directory, 'mlp', 17, 50, True)
        self.assertEqual(campaign.WORKER_SECONDS, 590)
        self.assertEqual(campaign.CONDITION_SECONDS, 600)

    def test_gate_record_uses_portable_content_bindings(self):
        from music_critic.experiments.registry import _reject_absolute_locators
        local = dict(status='completed', verified_source=dict(
            dataset_manifest=dict(path='/tmp/campaign/dataset/manifest.json', sha256='a' * 64, size_bytes=10),
            source_archive=dict(path='/tmp/source.tar', sha256='b' * 64, size_bytes=20)))
        portable = campaign._portable_gate_result(local)
        _reject_absolute_locators(portable)
        self.assertNotIn('path', portable['verified_source']['dataset_manifest'])
        self.assertEqual(portable['verified_source']['source_archive']['sha256'], 'b' * 64)
        self.assertIn('path', local['verified_source']['dataset_manifest'])

    def test_authoritative_binding_checked_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, path = launch_binding(root)
            source = campaign._load_source_binding(str(path))
            verified = campaign._verify_launch_bindings(dataset, source)
            self.assertIn('dataset_manifest', verified)
            (dataset / 'manifest.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                campaign._verify_launch_bindings(dataset, source)
            with self.assertRaises(ValueError):
                campaign._load_source_binding('{}')

    def test_gate_failure_aborts_remaining_without_training_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, source = launch_binding(root)
            args = SimpleNamespace(campaign=root / 'campaign', dataset=dataset,
                                   checkpoint=root / 'unused.pt', records=root / 'records',
                                   source_binding=str(source))
            states = {campaign.GATE_ID: 'planned'}
            states.update({campaign.condition_id(kind, seed): 'planned'
                           for seed in campaign.SEEDS for kind in campaign.KINDS})
            commands = []
            def read(path):
                return dict(status=states[Path(path).stem])
            def move(path, status, **kwargs):
                states[Path(path).stem] = status
            def run(command, log, timeout):
                commands.append(command)
                return dict(returncode=1, timed_out=False, leader_reaped=True)
            with patch.object(campaign, 'load_record', read), patch.object(campaign, 'transition', move), patch.object(campaign, '_run_owned', run):
                with self.assertRaisesRegex(RuntimeError, 'gate'):
                    campaign.supervise(args)
            self.assertEqual(len(commands), 1)
            self.assertIn('gate', commands[0])
            self.assertEqual(states[campaign.GATE_ID], 'failed')
            self.assertEqual(sum(state == 'aborted' for state in states.values()), 12)
            self.assertEqual(json.loads((root / 'campaign' / 'campaign.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
