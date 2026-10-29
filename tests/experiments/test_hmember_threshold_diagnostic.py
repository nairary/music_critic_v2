import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_path = Path(__file__).resolve().parents[2] / 'scripts/diagnose_hmember_thresholds.py'
_spec = importlib.util.spec_from_file_location('threshold_diagnostic', _path)
diagnostic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostic)


class ThresholdDiagnosticTests(unittest.TestCase):
    def test_authority_requires_all_twelve_unique_conditions(self):
        conditions=[{'id':f'EXP-HMEMBER-80K-S{s}-{k}-001'} for s in (17,42,73)
                    for k in ('MLP','TRANSE','SELF','GRAPH')]
        diagnostic.validate_authority({'conditions':conditions})
        for bad in ([],conditions[:-1],conditions+[conditions[0]],conditions[:-1]+[conditions[0]]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                diagnostic.validate_authority({'conditions':bad})

    def test_verified_bytes_rejects_wrong_binding_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=b'{"fixture":true}\n'; (root/'input.json').write_bytes(data)
            binding={'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            self.assertEqual(diagnostic.verified_bytes(root,'input.json',binding),data)
            with self.assertRaises(ValueError):
                diagnostic.verified_bytes(root,'../input.json',binding)
            with self.assertRaises(ValueError):
                diagnostic.verified_bytes(root,'input.json',dict(binding,size_bytes=1))
            (root/'alias.json').symlink_to(root/'input.json')
            with self.assertRaises(ValueError):
                diagnostic.verified_bytes(root,'alias.json',binding)

    def test_duration_predictions_reject_selection_or_missing_split(self):
        self.assertEqual(diagnostic.load_dev(b'{"split":"dev"}\n'),[{'split':'dev'}])
        for payload in (b'',b'{}',b'{"split":"selection"}',
                        b'{"split":"dev"}\n{"split":"selection"}'):
            with self.subTest(payload=payload),self.assertRaises(ValueError):
                diagnostic.load_dev(payload)

    def test_policies_separate_grid_boundary_from_refinement(self):
        rows=[dict(work='w',h_id='h',scores=[.96,.94,.92,.8],y=[1,0,0,0],mask=[True]*4)]
        policies,error=diagnostic.threshold_policies(rows)
        self.assertIsNone(error)
        self.assertLessEqual(policies['coarse_replay'],.9)
        self.assertLessEqual(policies['exact_bounded'],.9)
        self.assertEqual(policies['exact_full'],.96)
        self.assertEqual(policies['fpr10'],.96)

    def test_infeasible_fpr_is_preserved_as_unavailable(self):
        rows=[dict(work='w',h_id='h',scores=[1.,1.],y=[1,0],mask=[True]*2)]
        policies,error=diagnostic.threshold_policies(rows)
        self.assertIsNone(policies['fpr10'])
        self.assertIn('no attainable threshold',error)


if __name__=='__main__':unittest.main()
