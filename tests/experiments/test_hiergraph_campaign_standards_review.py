"""Independent synthetic launch admission review; no payload or GPU access."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from music_critic.experiments.hierarchical_graph_analyzer import campaign
from music_critic.experiments.hiergraph_export import file_binding
from music_critic.experiments.registry import canonical_json_bytes, load_record, seal_record
from test_registry import _record


def write_record(path, record):
    path.write_bytes(canonical_json_bytes(seal_record(record)) + b'\n')


class CampaignAdmissionStandardsTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'campaign'
        root.mkdir()
        source = Path(directory) / 'source-test'
        source.mkdir()
        (source / 'worker.py').write_text('# inert synthetic source\n')
        source_manifest = json.dumps({'files': [file_binding(source / 'worker.py', source)]}).encode()
        (source / 'source-manifest.json').write_bytes(source_manifest)
        config = dict(selection_unit='applied_updates', selection_steps=[2000, 5000, 10000, 20000, 40000, 60000, 80000],
                      max_attempted_steps=160000, condition_wall_seconds=21600,
                      source_manifest_sha256=sha256(source_manifest).hexdigest(),
                      target_manifest_sha256='a' * 64, cache_manifest_sha256='b' * 64)
        (root / 'config.json').write_text(json.dumps(config))
        binding = file_binding(root / 'config.json', root)
        plan = dict(schema='Hiergraph80KCampaign@1', condition_wall_seconds=21600,
                    campaign_wall_seconds=111600, config=binding, source_directory=source.name,
                    campaign_record='parent.json', conditions=[])
        for kind in ('E1', 'E2', 'E3', 'E4', 'E5'):
            row = dict(kind=kind, seed=17, experiment_id=kind.lower(), record=kind.lower() + '.json', output='runs/' + kind)
            plan['conditions'].append(row)
            record = _record(experiment_id=row['experiment_id'], status='planned')
            record['budget'] = dict(planned=dict(total_wall_seconds=21600))
            record['protocol'] = dict(kind=kind, launch=dict(mode='train', config_sha256=binding['sha256'],
                source_manifest_sha256=config['source_manifest_sha256'],
                target_manifest_sha256=config['target_manifest_sha256'],
                cache_manifest_sha256=config['cache_manifest_sha256'], output_name=kind,
                wall_seconds=21600, kind=kind, seed=17))
            write_record(root / row['record'], record)
        return root, plan

    def admit(self, root, plan, parent_budget=111600):
        raw = json.dumps(plan).encode()
        digest = sha256(raw).hexdigest()
        parent = _record(experiment_id='parent', status='planned')
        parent.update(protocol=dict(plan_sha256=digest), budget=dict(planned=dict(total_wall_seconds=parent_budget)))
        write_record(root / 'parent.json', parent)
        return campaign.validate_plan(root, raw, digest)

    def test_bound_synthetic_plan_admitted_and_source_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.fixture(directory)
            self.assertEqual(self.admit(root, plan), plan)
            (Path(directory) / 'source-test' / 'worker.py').write_text('# changed source\n')
            with self.assertRaisesRegex(ValueError, 'input (SHA/size|size) mismatch'):
                self.admit(root, plan)

    def test_exact_integer_caps_and_registered_parent_budget(self):
        for field, value in [('condition_wall_seconds', True), ('condition_wall_seconds', 21600.0),
                             ('condition_wall_seconds', 21601), ('campaign_wall_seconds', 111601)]:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                root, plan = self.fixture(directory)
                plan[field] = value
                with self.assertRaisesRegex(ValueError, 'authorized integer'):
                    self.admit(root, plan)
        with tempfile.TemporaryDirectory() as directory:
            root, plan = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'registered budget'):
                self.admit(root, plan, parent_budget=111599)

    def test_child_identity_and_registered_budget_cannot_contradict_plan(self):
        for field in ('experiment_id', 'kind', 'seed', 'budget'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root, plan = self.fixture(directory)
                path = root / 'e1.json'
                record = load_record(path)
                if field == 'experiment_id':
                    plan['conditions'][0]['experiment_id'] = 'other'
                elif field == 'kind':
                    record['protocol']['kind'] = 'E2'
                elif field == 'seed':
                    record['seeds'] = [19]
                else:
                    record['budget']['planned']['total_wall_seconds'] = 21599
                write_record(path, record)
                with self.assertRaisesRegex(ValueError, 'identity/seed/budget'):
                    self.admit(root, plan)

    def test_inherited_lock_survives_supervisor_descriptor_close(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.lock'
            worker = None
            try:
                with campaign.gpu_lock(path) as lock_fd:
                    worker = subprocess.Popen(
                        [sys.executable, '-c', "import time; print('ready', flush=True); time.sleep(30)"],
                        pass_fds=(lock_fd,), stdout=subprocess.PIPE, text=True)
                    self.assertEqual(worker.stdout.readline().strip(), 'ready')
                # Closing the supervisor's descriptor is equivalent to process
                # death for this flock; the worker still owns the open description.
                self.assertIsNone(worker.poll())
                with self.assertRaisesRegex(RuntimeError, 'GPU lock'):
                    with campaign.gpu_lock(path):
                        pass
            finally:
                if worker is not None:
                    worker.terminate()
                    worker.wait(timeout=5)
                    worker.stdout.close()
            with campaign.gpu_lock(path):
                pass
