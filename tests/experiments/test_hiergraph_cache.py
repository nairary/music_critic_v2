"""Target-blind raw hierarchy mechanics on canonical synthetic fixture."""
from dataclasses import replace
from pathlib import Path
from fractions import Fraction
import unittest
import torch
from music_critic.data import load_piece
from music_critic.relation_probe.data import sanitize
from music_critic.pll.contracts import ContextProfile,MaskQuery
from music_critic.pll.observed_view import ObservedMusicView
from music_critic.pll.masking import build_observed_tiles
from music_critic.pll.graph import build_tile_graph
from music_critic.experiments.hierarchical_graph_analyzer.cache import (
    BASE_NODES,assemble_raw_tile,attach_global_bars,raw_bank,RAW_KEYS)
from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer


class RawCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.piece=sanitize(load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json'))
        start=self.piece.bars[0].start_qn;end=self.piece.bars[-1].start_qn+self.piece.bars[-1].duration_qn
        view=ObservedMusicView(self.piece,MaskQuery(start,end,tuple(t.track_id for t in self.piece.tracks)),tuple(range(len(self.piece.bars))))
        self.tile=build_observed_tiles(view,ContextProfile())[0]
        self.graph=build_tile_graph(self.tile)
        self.states={k:torch.ones((self.graph.slice_features.shape[0] if k=='slice' else self.graph.base[k].num_nodes,128))*i for i,k in enumerate(BASE_NODES)}
    def test_exact_fragments_and_target_independence(self):
        p=assemble_raw_tile(self.piece,self.tile,self.graph,self.states)
        self.assertEqual(set(p['raw']),set(RAW_KEYS))
        self.assertTrue(p['metadata']['fragments'])
        for f in p['metadata']['fragments']:
            note=next(n for n in self.piece.notes if n.note_id==f['note_id'])
            self.assertLess(Fraction(*f['start']),Fraction(*f['end']))
            self.assertGreaterEqual(Fraction(*f['start']),note.onset_qn.to_fraction())
        q=assemble_raw_tile(replace(self.piece,targets=('irrelevant sidecar',)),self.tile,self.graph,self.states)
        for k in RAW_KEYS:torch.testing.assert_close(p['raw'][k],q['raw'][k],rtol=0,atol=0)
    def test_fullwork_context_and_all_five_forward_backward(self):
        p=assemble_raw_tile(self.piece,self.tile,self.graph,self.states)
        context=dict(ids=[b.bar_id for b in self.piece.bars],states=torch.ones((len(self.piece.bars),128)),
                     spans=[[[b.start_qn.num,b.start_qn.den],[(b.start_qn+b.duration_qn).num,(b.start_qn+b.duration_qn).den]] for b in self.piece.bars])
        raw=attach_global_bars(p,context)
        self.assertEqual(int((raw['node_type']==8).sum()),len(self.piece.bars))
        for kind in ('E1','E2','E3','E4','E5'):
            m=HAnalyzer(kind,{'H.function':3},hidden=16,rounds=1,max_events=4)
            pred=m(raw);loss=pred['field_logits']['H.function'].sum();loss.backward()
            self.assertTrue(any(v.grad is not None and bool(v.grad.abs().sum()) for v in m.parameters()))
    def test_bank_exact_deterministic_raw_scales(self):
        times=[Fraction(0),Fraction(1,3),Fraction(1),Fraction(2)]
        self.assertEqual(raw_bank(times),raw_bank(times))
        self.assertIn((0,1),raw_bank(times));self.assertIn((0,2),raw_bank(times))
