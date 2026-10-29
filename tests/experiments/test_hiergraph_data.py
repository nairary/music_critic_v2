"""Synthetic target-side geometry and authenticated-loader contracts."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from music_critic.experiments.hiergraph_export import write_json,write_gzip_json,fingerprint
from music_critic.experiments.hierarchical_graph_analyzer.data import load_target_dataset,targets_for_tile


def numeric():
    return dict(schema='HiergraphNumericTargets@3',vocabulary_fingerprint='v',targets=dict(piece_id='p',harmonies=dict(
        ids=['a','b','c','point','outside'],intervals=[[[0,1],[1,1]],[[1,1],[3,1]],[[1,1],[2,1]],[[2,1],[2,1]],[[3,1],[5,1]]],
        view_ids=['A','A','B','A','A'],values={'H.function':[0,-1,1,-1,0]},masks={'H.function':[True,False,True,False,True]})),evidence={},source={})

class TileTargetTests(unittest.TestCase):
    def test_exact_core_ownership_halo_and_views_preserve_unknown(self):
        result=targets_for_tile(numeric(),active_fields=['H.function'],core_start=Fraction(1),core_end=Fraction(3),halo_start=Fraction(0),halo_end=Fraction(4))
        self.assertEqual([r['view_id'] for r in result],['A','B'])
        a,b=result
        self.assertEqual(a['ids'],['b']);self.assertEqual(a['spans'],[[.25,.75]])
        self.assertEqual(a['exact_spans'],[(Fraction(1),Fraction(3))])
        self.assertEqual(a['fields']['H.function'],dict(values=[-1],mask=[False]))
        self.assertEqual(a['known_negative_spans'],[])
        self.assertEqual(a['coverage']['point_events'],1)
        self.assertEqual(a['coverage']['not_owned'],2)
        self.assertEqual(b['ids'],['c'])
    def test_owned_halo_crossings_are_counted_without_clipping(self):
        rows=targets_for_tile(numeric(),active_fields=['H.function'],core_start=Fraction(3),core_end=Fraction(4),halo_start=Fraction(2),halo_end=Fraction(4))
        a=next(r for r in rows if r['view_id']=='A')
        self.assertEqual(a['ids'],[]);self.assertEqual(a['coverage']['outside_halo'],1)
        self.assertEqual(a['coverage']['outside_halo_ids'],['outside'])
        self.assertEqual(a['coverage']['outside_halo_known_fields'],{'H.function':1})
        self.assertEqual(a['coverage']['point_events_owned'],0)
    def test_half_open_ownership_is_unique_and_float_geometry_rejected(self):
        n=numeric()
        left=targets_for_tile(n,active_fields=['H.function'],core_start=0,core_end=1,halo_start=0,halo_end=5)
        right=targets_for_tile(n,active_fields=['H.function'],core_start=1,core_end=4,halo_start=0,halo_end=5)
        self.assertEqual([i for r in left for i in r['ids']],['a'])
        self.assertEqual([i for r in right for i in r['ids']],['b','outside','c'])
        with self.assertRaises(ValueError):
            targets_for_tile(n,active_fields=['H.function'],core_start=0.0,core_end=1,halo_start=0,halo_end=5)

    def test_pointless_empty_labels_and_invalid_intervals(self):
        for key,val in [('core_end',0),('halo_end',0)]:
            kw=dict(core_start=0,core_end=2,halo_start=0,halo_end=3);kw[key]=val
            with self.assertRaises(ValueError):targets_for_tile(numeric(),active_fields=['H.function'],**kw)
        n=numeric();n['targets']['harmonies']['masks']['H.function'][0]=False
        n['targets']['harmonies']['values']['H.function'][0]=0
        with self.assertRaises(ValueError):targets_for_tile(n,active_fields=['H.function'],core_start=0,core_end=2,halo_start=0,halo_end=3)

class AuthenticatedDatasetTests(unittest.TestCase):
    def fixture(self,root):
        vocab=dict(schema='HiergraphVocabulary@3',contract_fingerprint='c',split_fingerprint='s',event=dict(fields=dict(harmonies={'H.function':['tonic','dominant'],'h.chord_membership':[False,True]})),objects=dict(fields={}),active_fields=['H.function','H.unsupported','P.function'])
        vocab['fingerprint']=fingerprint(vocab);vb=write_json(root/'vocabulary.json',vocab)
        heads=dict(schema='HiergraphTrainableHeads@1',vocabulary_fingerprint=vocab['fingerprint'],contract_fingerprint='c',fields={'H.function':dict(active=True),'h.chord_membership':dict(active=True)})
        heads['fingerprint']=fingerprint(heads);hb=write_json(root/'heads.json',heads)
        n=numeric();n['vocabulary_fingerprint']=vocab['fingerprint'];nb=write_gzip_json(root/'numeric.gz',n,100000)
        manifest=dict(schema='HiergraphDataset@1',status='completed',vocabulary=vb,heads=hb,contract_fingerprint='c',split_fingerprint='s',works=[dict(work_id='w',piece_id='p',family_id='f',split='fit',numeric=nb)],test_locked=True)
        manifest['fingerprint']=fingerprint(manifest);return write_json(root/'manifest.json',manifest)
    def test_loader_authenticates_complete_chain_and_h_only_head_maps(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);mb=self.fixture(root);ds=load_target_dataset(root,manifest_sha256=mb['sha256'])
            self.assertEqual(ds.head_sizes,{'H.function':2});self.assertEqual(ds.value_maps['h.chord_membership'],[False,True])
            self.assertEqual(ds.value_maps['H.unsupported'],[]);self.assertEqual(ds.value_maps['P.function'],[])
            samples=ds.targets_for_tile('w',core_start=0,core_end=3,halo_start=0,halo_end=4)
            self.assertEqual(samples[0]['work_id'],'w');self.assertEqual(samples[0]['family_id'],'f')
            (root/'numeric.gz').write_bytes(b'bad')
            ds=load_target_dataset(root,manifest_sha256=mb['sha256'])
            with self.assertRaises(ValueError):ds.load_work('w')
    def test_manifest_tampering_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);mb=self.fixture(root);(root/'manifest.json').write_bytes(b'not-json')
            with self.assertRaises(ValueError):load_target_dataset(root,manifest_sha256=mb['sha256'])
