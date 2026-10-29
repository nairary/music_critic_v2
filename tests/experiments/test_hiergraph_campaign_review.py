"""Independent synthetic process-group cleanup review; no accelerator/corpus."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from music_critic.experiments.hierarchical_graph_analyzer import campaign
import test_hiergraph_campaign as campaign_fixtures


WORKER = r'''from pathlib import Path
import fcntl, subprocess, sys, time
from music_critic.experiments.registry import load_record, seal_record, canonical_json_bytes
record, mode, previous = sys.argv[1:]
root = Path(record).parent
lock_path, ready_path = root / 'descendant.lock', root / 'descendant.ready'
if mode == 'orphan':
    code = """import fcntl, signal, sys, time
from pathlib import Path
lock = open(sys.argv[1], 'a')
fcntl.flock(lock, fcntl.LOCK_EX)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[2]).write_text('ready')
time.sleep(60)
"""
    subprocess.Popen([sys.executable, '-c', code, str(lock_path), str(ready_path)])
    deadline = time.monotonic() + 5
    while not ready_path.exists():
        if time.monotonic() > deadline: raise RuntimeError('synthetic child startup timeout')
        time.sleep(.01)
else:
    # A succeeding next condition is evidence that the previous condition's
    # SIGTERM-resistant descendant no longer owns its process resource.
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
r = load_record(record)
r['status'] = 'completed'
r['results'] = {'summary': 'synthetic completed',
                'actual_state': {'applied_updates': 80000, 'attempted_steps': 80000}}
Path(record).write_bytes(canonical_json_bytes(seal_record(r)) + b'\n')
'''


class CampaignCleanupReviewTests(unittest.TestCase):
    def test_exited_leader_descendant_is_reaped_before_next_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root, plan, command = campaign_fixtures.CampaignTests().fixture(parent, ['orphan', 'verify'])
            (parent / 'worker.py').write_text(WORKER)
            plan.update(condition_wall_seconds=15, campaign_wall_seconds=40, cleanup_seconds=.1)
            try:
                with patch.object(campaign, 'command_for', side_effect=command):
                    result = campaign.run_campaign(root, plan)
                self.assertEqual([row['status'] for row in result['conditions']],
                                 ['completed', 'completed'])
            finally:
                progress = root / 'progress.json'
                if progress.exists():
                    for row in json.loads(progress.read_text()).get('conditions', []):
                        if row.get('pid'):
                            try:
                                os.killpg(row['pid'], signal.SIGKILL)
                            except ProcessLookupError:
                                pass
