import tempfile
import unittest
from pathlib import Path

from music_critic.experiments.harmonic_event_reasoner.evidence import plan, transition
from music_critic.experiments.registry import load_record


class EvidenceTests(unittest.TestCase):
    def test_canonical_forward_terminal_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'EXP-FIXTURE.json'
            plan(p,'EXP-FIXTURE','fixture',{},dict(commit='fixture',dirty=True))
            self.assertEqual(load_record(p)['status'],'planned')
            transition(p,'running')
            transition(p,'failed',reason='fixture intentional failure',actual=dict(applied_updates=0))
            payload=p.read_bytes()
            self.assertEqual(load_record(p)['status'],'failed')
            with self.assertRaises(ValueError):transition(p,'completed')
            self.assertEqual(p.read_bytes(),payload)
            with self.assertRaises(Exception):plan(p,'EXP-FIXTURE','fixture',{},dict(commit='fixture',dirty=True))

    def test_access_is_unknown_until_evidence_available(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'EXP-FIXTURE.json'
            plan(p,'EXP-FIXTURE','fixture',{},dict(commit='fixture',dirty=True))
            self.assertIsNone(load_record(p)['data_access']['test_inputs_accessed'])
            transition(p,'completed',observed_train_only=True,results=dict(fixture=True))
            self.assertIs(load_record(p)['data_access']['test_inputs_accessed'],False)


if __name__=='__main__':unittest.main()
