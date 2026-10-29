"""Synthetic whole-piece raw routing and target-side supervision contracts."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import unittest

from music_critic.experiments.hierarchical_graph_analyzer.full_data import targets_for_work


def numeric():
    return dict(schema='HiergraphNumericTargets@3', vocabulary_fingerprint='v', targets=dict(
        piece_id='p', harmonies=dict(ids=['H0','H1','HB'], intervals=[[[0,1],[2,1]],[[2,1],[4,1]],[[0,1],[4,1]]],view_ids=['A','A','B'],values={'H.function':[0,1,-1]},masks={'H.function':[True,True,False]}),
        note_contexts=dict(ids=['h0','h1','orphan'], intervals=[[[1,1],[2,1]],[[2,1],[3,1]],[[3,1],[4,1]]],view_ids=['A','A','A'],note_indices=[0,0,1],harmony_ids=['H0','H1',None],applicability=['note_harmonic_context']*3,values={'h.chord_membership':[1,0,-1]},masks={'h.chord_membership':[True,True,False]}),
        participants=[dict(id='n0',kind='note',canonical_note_index=0),dict(id='n1',kind='note',canonical_note_index=1)]),
        objects=dict(groups={'harmonic_process':dict(ids=['P0'],view_ids=['A'],intervals=[[[0,1],[4,1]]],member_ids=[['h0','h1','H1']],values={'P.process':[0]},masks={'P.process':[True]},states={'P.process':['known']})}),
        addressed_relations=[dict(id='r0',type='resolves_to',source_ids=['h0','n1'],target_ids=['h1'],start_qn=[1,1],end_qn=[3,1],view_id='A',value=True,available=True,origin='source',confidence=0.0),dict(id='r1',type='prepares',source_ids=['n0'],target_ids=['h0'],start_qn=[0,1],end_qn=[2,1],view_id='A',value=False,available=True)],evidence={'h0':{'source':'synthetic'}})


class WholeTargetsTests(unittest.TestCase):
    def test_native_numeric_views_links_and_many_to_many_are_preserved(self):
        src=numeric();before=deepcopy(src)
        rows=targets_for_work(src,active_fields=['H.function','h.chord_membership','P.process'],origin=0,duration=4)
        self.assertEqual([r['view_id'] for r in rows],['A','B']);a,b=rows
        self.assertEqual(a['H']['spans'],[[0.,.5],[.5,1.]])
        self.assertEqual(a['h']['note_index'],[0,0,1]);self.assertEqual(a['h']['H_index'],[0,1,-1])
        self.assertEqual(a['h']['fields']['h.chord_membership'],{'values':[1,0,-1],'mask':[True,True,False]})
        self.assertEqual(a['P']['spans'],[[0.,1.]])
        self.assertEqual([(x['P_index'],x['endpoint']['kind'],x['endpoint']['index']) for x in a['participants']],[(0,'h',0),(0,'h',1),(0,'H',1)])
        self.assertEqual(len(a['relations']),2)
        self.assertEqual([(x['kind'],x['index']) for x in a['relations'][0]['source']],[('h',0),('n',1)])
        self.assertTrue(a['relations'][0]['mask']);self.assertEqual(a['relations'][0]['confidence'],0.)
        self.assertIs(a['relations'][1]['value'],False);self.assertTrue(a['relations'][1]['mask'])
        self.assertEqual(a['coverage']['h_without_H'],1);self.assertEqual(b['H']['fields']['H.function']['mask'],[False])
        self.assertEqual(src,before)

    def test_unknown_relations_stay_masked_and_invalid_numeric_fails(self):
        src=numeric();src['addressed_relations'][0].update(value=None,available=False)
        a=targets_for_work(src,active_fields=['H.function'],origin=0,duration=4)[0]
        self.assertFalse(a['relations'][0]['mask']);self.assertIsNone(a['relations'][0]['value'])
        src=numeric();src['targets']['note_contexts']['masks']['h.chord_membership'][0]=False
        with self.assertRaisesRegex(ValueError,'sentinel'):
            targets_for_work(src,active_fields=['h.chord_membership'],origin=0,duration=4)
        with self.assertRaises(ValueError):targets_for_work(numeric(),active_fields=[],origin=0.0,duration=4)


class TargetValidationTests(unittest.TestCase):
    def test_relation_endpoints_are_sets_and_context_metadata_aligns(self):
        src=numeric();src['addressed_relations'][0]['source_ids']=['h0','h0']
        with self.assertRaisesRegex(ValueError,'endpoint'):
            targets_for_work(src,active_fields=[],origin=0,duration=4)
        src=numeric();src['targets']['note_contexts']['applicability']=[]
        with self.assertRaisesRegex(ValueError,'applicability'):
            targets_for_work(src,active_fields=[],origin=0,duration=4)

    def test_geometry_eligibility_uses_exact_point_and_work_bounds(self):
        src=numeric()
        src['targets']['harmonies']['intervals'][0]=[[0,1],[0,1]]
        src['targets']['harmonies']['intervals'][1]=[[2,1],[5,1]]
        src['targets']['note_contexts']['intervals'][0]=[[1,3],[2,3]]
        row=targets_for_work(src,active_fields=[],origin=0,duration=4)[0]
        self.assertEqual(row['H']['geometry_eligible'],[False,False])
        self.assertEqual(row['h']['geometry_eligible'],[True,True,True])
        self.assertEqual(row['P']['geometry_eligible'],[True])
        self.assertEqual(row['H']['ids'],['H0','H1'])
        self.assertEqual(row['H']['exact_spans'][1],(Fraction(2),Fraction(5)))

    def test_unknown_object_addresses_remain_positive_coverage_misses(self):
        src=numeric();src['objects']['groups']['harmonic_process']['member_ids'][0].append('melodic-object')
        src['targets']['note_contexts']['intervals'][0]=[[1,3],[2,3]]
        a=targets_for_work(src,active_fields=[],origin=0,duration=4)[0]
        missing=a['participants'][-1]
        self.assertEqual(missing['endpoint'],dict(kind='unsupported',index=-1,id='melodic-object',reason='unsupported_address'))
        self.assertTrue(missing['mask']);self.assertTrue(missing['value'])
        self.assertEqual(a['h']['exact_spans'][0],(Fraction(1,3),Fraction(2,3)))
        self.assertEqual(a['coverage']['participants_without_H_h'],1)


class WholeRawTests(unittest.TestCase):
    def setUp(self):
        import torch
        from music_critic.data import load_piece
        from music_critic.relation_probe.data import sanitize
        from music_critic.pll.contracts import MaskQuery
        from music_critic.pll.observed_view import ObservedMusicView
        from music_critic.pll.masking import build_observed_tiles
        from music_critic.pll.graph import build_tile_graph
        from music_critic.experiments.hierarchical_graph_analyzer.cache import RawGraphProfile, BASE_NODES, assemble_raw_tile
        torch.set_num_threads(1)
        self.piece = sanitize(load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json'))
        start = self.piece.bars[0].start_qn
        end = self.piece.bars[-1].start_qn + self.piece.bars[-1].duration_qn
        view = ObservedMusicView(self.piece, MaskQuery(start, end, tuple(t.track_id for t in self.piece.tracks)), tuple(range(len(self.piece.bars))))
        profile = RawGraphProfile(core_bars=1, halo_bars=0)
        self.tiles, self.expected = [], []
        lookup = {n.note_id:i for i,n in enumerate(self.piece.notes)}
        for tile in build_observed_tiles(view, profile):
            graph = build_tile_graph(tile, profile)
            encoded = {k:torch.zeros((graph.slice_features.shape[0] if k=='slice' else graph.base[k].num_nodes,128)) for k in BASE_NODES}
            payload = assemble_raw_tile(self.piece, tile, graph, encoded)
            self.tiles.append((payload['raw'],payload['metadata']))
            self.expected.append([lookup[identity] for _,identity in graph.keys['note']])

    def test_native_cache_routing_matches_graph_and_keeps_grace_percussion(self):
        from music_critic.experiments.hierarchical_graph_analyzer.full_data import assemble_work_raw
        raw = assemble_work_raw(self.piece, self.tiles)
        self.assertEqual(raw['note_ids'], tuple(n.note_id for n in self.piece.notes))
        self.assertEqual([x.tolist() for x in raw['note_index_by_tile']], self.expected)
        self.assertEqual(raw['origin'], Fraction(0)); self.assertEqual(raw['duration'], Fraction(5))
        self.assertEqual(raw['coverage']['canonical_notes'], 6)
        self.assertGreater(raw['coverage']['overlapping_note_occurrences'],0)
        self.assertEqual(raw['notes']['is_grace'].sum().item(),1)
        self.assertEqual(raw['notes']['is_percussion'].sum().item(),2)
        self.assertEqual(raw['bar_spans'].shape,(2,2))
        self.assertNotIn('targets',raw)
        self.assertNotIn('work_id',raw)

    def test_corrupt_note_identity_and_missing_core_fail_closed(self):
        from music_critic.experiments.hierarchical_graph_analyzer.full_data import assemble_work_raw
        with self.assertRaisesRegex(ValueError,'core'):
            assemble_work_raw(self.piece,self.tiles[1:])
        bad=deepcopy(self.tiles)
        bad[0][1]['fragments'][0]['note_id']='wrong'
        with self.assertRaisesRegex(ValueError,'identity'):
            assemble_work_raw(self.piece,bad)
        with self.assertRaisesRegex(ValueError,'capacity'):
            assemble_work_raw(self.piece,self.tiles,max_notes=5)
        bad=deepcopy(self.tiles);bad[0][1]['halo_start']=0.0
        with self.assertRaisesRegex(ValueError,'exact rational'):
            assemble_work_raw(self.piece,bad)

    def test_cap_stops_lazy_loading_before_exhausting_work(self):
        from music_critic.experiments.hierarchical_graph_analyzer.full_data import assemble_work_raw
        reads=[]
        def lazy():
            for i in range(100):
                reads.append(i);yield self.tiles[0]
        with self.assertRaisesRegex(ValueError,'capacity'):
            assemble_work_raw(self.piece,lazy(),max_tiles=1)
        self.assertLessEqual(len(reads),2)
        reads.clear()
        with self.assertRaisesRegex(ValueError,'capacity'):
            assemble_work_raw(self.piece,lazy(),max_notes=1)
        self.assertEqual(reads,[])

    def test_exact_core_anchor_owner_avoids_float_boundary_roundoff(self):
        import torch
        from dataclasses import replace
        from music_critic.data import RationalTime
        from music_critic.experiments.hierarchical_graph_analyzer.full_data import assemble_work_raw
        piece=replace(self.piece,notes=(),bars=(
            replace(self.piece.bars[0],start_qn=RationalTime(0),duration_qn=RationalTime(2)),
            replace(self.piece.bars[1],start_qn=RationalTime(2),duration_qn=RationalTime(4))))
        tiles=[]
        for times,core in [([0,1,2],(0,2)),([1,2,3],(2,3)),([3,4,6],(3,6))]:
            a,b=times[0],times[-1]
            raw=dict(x=torch.zeros(3,266),node_type=torch.full((3,),6),edge_index=torch.empty(2,0,dtype=torch.long),
                edge_type=torch.empty(0,dtype=torch.long),node_time=torch.zeros(3,2),time_index=torch.arange(3),
                time_positions=torch.tensor([(t-a)/(b-a) for t in times]),bank_indices=torch.tensor([[0,1],[1,2]]),pool_mask=torch.ones(3,dtype=torch.bool))
            metadata=dict(core_start=[core[0],1],core_end=[core[1],1],halo_start=[a,1],halo_end=[b,1],
                raw_times=[[t,1] for t in times],fragments=[])
            tiles.append((raw,metadata))
        raw=assemble_work_raw(piece,tiles)
        self.assertEqual([x.tolist() for x in raw['owned_time_indices']],[[0,1],[1],[0,1]])
        a,b=raw['tile_spans'][1];reconstructed=a+(b-a)*raw['tiles'][1]['time_positions'][1]
        self.assertLess(reconstructed,raw['core_spans'][1,0])

    def test_raw_assembly_does_not_use_targets_or_mutate_cache(self):
        from dataclasses import replace
        import torch
        from music_critic.experiments.hierarchical_graph_analyzer.full_data import assemble_work_raw
        before=deepcopy(self.tiles)
        one=assemble_work_raw(self.piece,self.tiles)
        two=assemble_work_raw(replace(self.piece,targets=('never inspect',)),self.tiles)
        for key in ('tile_spans','core_spans','bar_spans'):
            torch.testing.assert_close(one[key],two[key],rtol=0,atol=0)
        for key in one['notes']:
            torch.testing.assert_close(one['notes'][key],two['notes'][key],rtol=0,atol=0)
        for (a,am),(b,bm) in zip(self.tiles,before):
            self.assertEqual(am,bm)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


class FullHeadTests(unittest.TestCase):
    def test_only_first_slice_P_gate_is_removed(self):
        from types import SimpleNamespace
        from music_critic.experiments.hierarchical_graph_analyzer.full_data import full_head_sizes
        dataset=SimpleNamespace(value_maps={'H.a':['a','b'],'h.b':[False,True]},
            object_value_maps={'harmonic_process':{'P.ready':['a','b'],'P.rare':['a'],'P.pu':['a','b']}},
            heads={'fields':{'H.a':{'active':True,'deferred_reasons':[]},
                'h.b':{'active':True,'deferred_reasons':[]},
                'P.ready':{'active':False,'deferred_reasons':['P_evidence_only_initial_slice']},
                'P.rare':{'active':False,'deferred_reasons':['P_evidence_only_initial_slice','fewer_than_two_fit_classes']},
                'P.pu':{'active':False,'deferred_reasons':['P_evidence_only_initial_slice','positive_unlabeled_negative_loss_protocol_not_admitted']}}})
        self.assertEqual(full_head_sizes(dataset),{'H':{'H.a':2},'h':{'h.b':2},'P':{'P.ready':2}})

class NativeProjectionTests(unittest.TestCase):
    def test_real_projection_encoder_feeds_whole_piece_adapter(self):
        import importlib.util
        path = Path(__file__).parent/'tasks/test_hiergraph_projection.py'
        spec = importlib.util.spec_from_file_location('full_data_synthetic_projection', path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        piece, bundle = fixture.fixture()
        projection = fixture.project(piece, bundle)
        vocabulary = fixture.fit([projection])
        native = fixture.encode_projection(piece, projection, vocabulary, split_fingerprint='a'*64)
        rows = targets_for_work(native, active_fields=['H.function','h.chord_membership','P.function'],
                                origin=0, duration=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['H']['ids'], ['H0','H1'])
        self.assertEqual(rows[0]['h']['note_index'], [0,0])
        self.assertEqual(rows[0]['h']['H_index'], [0,1])
        self.assertEqual(rows[0]['P']['ids'], ['P0'])
        self.assertEqual(len(rows[0]['participants']), 3)
        self.assertEqual(rows[0]['h']['fields']['h.chord_membership']['mask'], [True,True])
        self.assertNotEqual(*rows[0]['h']['fields']['h.chord_membership']['values'])
        self.assertTrue(rows[0]['P']['evidence']['P0'])


if __name__=='__main__':unittest.main()
