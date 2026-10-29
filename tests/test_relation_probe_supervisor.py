"""Synthetic process ownership: stopping a supervisor must stop its worker."""
import subprocess,sys,unittest
from scripts.run_relation_probe import stop_owned


class SupervisorTests(unittest.TestCase):
    def test_owned_detached_worker_is_terminated_and_reaped(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        try:
            stop_owned(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:child.kill();child.wait()
