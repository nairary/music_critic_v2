"""Exact boundary supervision and raw-exclusion accounting integration."""
from fractions import Fraction
from types import SimpleNamespace
import unittest
from music_critic.experiments.hierarchical_graph_analyzer.pipeline import bind_boundaries,Samples


class PipelineTests(unittest.TestCase):
    def test_endpoint_lookup_is_exact_not_float_tolerance(self):
        target=dict(exact_spans=[(Fraction(0),Fraction(1,3)),(Fraction(1,3),Fraction(2,3)),
                                (Fraction(1,3)+Fraction(1,10**12),Fraction(1))])
        result=bind_boundaries(target,[[0,1],[1,3],[2,3],[1,1]])
        self.assertEqual(result['known_boundary_indices'],[1,2])
        self.assertEqual(result['boundary_endpoint_coverage'],dict(known_interior=3,on_grid=2,off_grid=1))
        self.assertNotIn('known_boundary_indices',target)
    def test_missing_raw_work_kept_as_unrecovered_known_h(self):
        target=dict(ids=['h'],spans=[[0,1]],fields={},exact_spans=[(Fraction(0),Fraction(1))],
            core_span=(Fraction(0),Fraction(1)),halo_span=(Fraction(0),Fraction(1)),
            family_id='f',work_id='w',view_id='v',scope_id='harmonies',coverage=dict(outside_halo=0))
        dataset=SimpleNamespace(targets_for_tile=lambda *a,**kw:[target])
        cache=SimpleNamespace(works={'w':dict(split='dev',tiles=[],exclusions=[dict(reason='capacity',core_start=[0,1],core_end=[1,1],halo_start=[0,1],halo_end=[1,1])])})
        samples=Samples(dataset,cache,'dev')
        self.assertEqual(len(samples),0)
        reports=samples.missing_reports();self.assertEqual(reports[0]['known_h'],1)
        self.assertEqual(reports[0]['recovered_h'],0)
