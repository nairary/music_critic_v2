"""Synthetic contracts for complete H/h/P supervision, not musical evidence."""
import unittest
import torch
from music_critic.experiments.hierarchical_graph_analyzer.full_losses import full_loss


def events(spans, fields=None):
    return {'spans': torch.tensor(spans, dtype=torch.float32).reshape(-1, 2).requires_grad_(),
            'object_logits': torch.zeros(len(spans), requires_grad=True),
            'field_logits': fields or {}}


def prediction():
    return {'H': events([[0., .5], [.5, 1.]]),
            'h': {'spans': torch.tensor([[.1, .3], [.6, .8]], requires_grad=True),
                  'note_index': torch.tensor([0, 1]), 'H_index': torch.tensor([0, 1]),
                  'field_logits': {'h.chord_membership': torch.zeros(2, 2, requires_grad=True)}},
            'P': events([[0., 1.]]),
            'participants': {'P_index': torch.tensor([0, 0]), 'endpoint_level': ('H', 'h'),
                             'endpoint_index': torch.tensor([0, 1]),
                             'existence_logits': torch.zeros(2, requires_grad=True)},
            'relations': {'sources': (({'kind':'h','index':0},), ({'kind':'h','index':1},)),
                          'targets': (({'kind':'h','index':1},), ({'kind':'h','index':0},)),
                          'logits': {'resolves_to': torch.zeros(2, requires_grad=True)}}}


def target():
    return {'H': {'ids':['H0','H1'], 'spans':[[0., .5],[.5,1.]], 'fields':{}},
            'h': {'ids':['h0','h1'], 'spans':[[.1,.3],[.6,.8]], 'note_index':[0,1],
                  'H_index':[0,1], 'fields':{'h.chord_membership':{'values':[0,-1], 'mask':[True,False]}}},
            'P': {'ids':[], 'spans':[], 'fields':{}}, 'participants':[], 'relations':[]}


class FullLossTests(unittest.TestCase):
    def test_all_unknown_is_zero_with_zero_gradients(self):
        p=prediction();r=full_loss(p,{})
        self.assertFalse(r['supported']);self.assertEqual(r['loss'].item(),0)
        r['loss'].backward()
        for t in [p['H']['spans'],p['h']['field_logits']['h.chord_membership'],p['P']['spans'],
                  p['participants']['existence_logits'],p['relations']['logits']['resolves_to']]:
            self.assertIsNotNone(t.grad);self.assertEqual(t.grad.abs().sum().item(),0)

    def test_explicit_nonchord_label_is_not_geometric_membership(self):
        p=prediction();r=full_loss(p,target());r['loss'].backward()
        g=p['h']['field_logits']['h.chord_membership'].grad
        self.assertLess(g[0,0].item(),0);self.assertGreater(g[0,1].item(),0)
        self.assertEqual(g[1].abs().sum().item(),0)
        self.assertTrue(r['supported'])
        self.assertEqual(r['support']['h']['fields']['h.chord_membership'],1)


class FullLossEdgeCaseTests(unittest.TestCase):
    def test_missing_target_addresses_are_masked_and_covered(self):
        p=prediction();t=target()
        t['P']={'ids':['P0'],'spans':[[0.,1.]],'fields':{}}
        t['participants']=[dict(P_index=0,endpoint={'kind':'unsupported','index':-1,'id':'raw-object'},value=True,mask=True)]
        t['relations']=[dict(source=[{'kind':'h','index':0}],target=[{'kind':'H','index':-1,'id':'other-view'}],type='resolves_to',value=True,mask=True)]
        r=full_loss(p,t)
        self.assertEqual(r['support']['participants']['known'],1)
        self.assertEqual(r['support']['participants']['matched'],0)
        self.assertEqual(r['support']['relations']['matched'],0)

    def test_explicit_link_negatives_have_gradient_unknowns_do_not(self):
        p=prediction();t=target()
        t['P']={'ids':['P0'],'spans':[[0.,.9]],'fields':{}}
        t['participants']=[dict(P_index=0,endpoint={'kind':'H','index':0},value=False,mask=True),dict(P_index=0,endpoint={'kind':'h','index':1},mask=False)]
        t['relations']=[dict(source=[{'kind':'h','index':0}],target=[{'kind':'h','index':1}],type='resolves_to',value=False,mask=True)]
        r=full_loss(p,t);r['loss'].backward()
        self.assertGreater(p['P']['spans'].grad.abs().sum().item(),0)
        self.assertGreater(p['participants']['existence_logits'].grad[0].item(),0)
        self.assertEqual(p['participants']['existence_logits'].grad[1].item(),0)
        self.assertGreater(p['relations']['logits']['resolves_to'].grad[0].item(),0)
        self.assertEqual(p['relations']['logits']['resolves_to'].grad[1].item(),0)

    def test_note_identity_and_subcontexts_match_separately(self):
        p=prediction();t=target()
        p['h']['note_index']=torch.tensor([0,0]);p['h']['H_index']=torch.tensor([0,0])
        p['h']['spans']=torch.tensor([[.3,.4],[.1,.2]],requires_grad=True)
        t['h']['note_index']=[0,0];t['h']['H_index']=[0,0];t['h']['spans']=[[.1,.2],[.3,.4]]
        t['h']['fields']['h.chord_membership']={'values':[0,1],'mask':[True,True]}
        r=full_loss(p,t);self.assertEqual(r['matches']['h'],[(0,1),(1,0)])
        t['h']['note_index']=[9,9];r=full_loss(p,t)
        self.assertEqual(r['support']['h']['matched_contexts'],0)
        self.assertEqual(r['support']['h']['fields']['h.chord_membership'],0)

    def test_refined_boundaries_propagate_to_process_state(self):
        p=prediction();t=target();process_state=torch.tensor(.1,requires_grad=True)
        p['H']['spans']=p['H']['spans']+process_state*.1
        p['h']['spans']=p['h']['spans']+process_state*.1
        r=full_loss(p,t);r['loss'].backward()
        self.assertNotEqual(process_state.grad.item(),0)

    def test_all_disabled_scopes_skip_optimizer(self):
        r=full_loss(prediction(),target(),weights={k:0. for k in ('H','h','P','participants','relations')})
        self.assertFalse(r['supported']);self.assertEqual(r['loss'].item(),0)

    def test_bad_head_row_count_rejected_even_without_labels(self):
        p=prediction();p['h']['field_logits']['h.chord_membership']=torch.zeros(3,2,requires_grad=True)
        with self.assertRaisesRegex(ValueError,'align'):
            full_loss(p,{})

class FullGeometrySupportTests(unittest.TestCase):
    def test_point_H_P_are_unsupported_without_remapping_links_to_wrong_ids(self):
        p=prediction();t=target()
        t['H']['ids']=['point','H0','H1'];t['H']['spans']=[[0.,0.],[0.,.5],[.5,1.]]
        t['h']['H_index']=[1,2]
        t['P']={'ids':['point','P0'],'spans':[[.2,.2],[0.,1.]],'fields':{}}
        t['participants']=[dict(P_index=1,endpoint={'kind':'H','index':1},value=True,mask=True)]
        r=full_loss(p,t)
        self.assertEqual(r['matches']['H'],[(0,1),(1,2)])
        self.assertEqual(r['matches']['P'],[(0,1)])
        self.assertEqual(r['support']['H']['known_events'],3)
        self.assertEqual(r['support']['H']['geometry_excluded'],1)
        self.assertEqual(r['support']['participants']['matched'],1)

    def test_point_note_context_is_known_but_unmodeled(self):
        p=prediction();t=target();t['h']['spans'][0]=[.2,.2]
        r=full_loss(p,t)
        self.assertEqual(r['support']['h']['known_contexts'],2)
        self.assertEqual(r['matches']['h'],[(1,1)])

class FullHyperedgeTests(unittest.TestCase):
    def test_full_endpoint_set_is_matched_without_pairwise_expansion(self):
        p=prediction();t=target()
        sources=[{'kind':'h','index':0},{'kind':'h','index':1}]
        t['relations']=[dict(source=sources,target=[{'kind':'h','index':1}],type='resolves_to',value=True,mask=True)]
        r=full_loss(p,t);self.assertEqual(r['support']['relations']['known'],1)
        self.assertEqual(r['support']['relations']['matched'],0)
        p['relations']['sources']=(tuple(reversed(sources)),p['relations']['sources'][1])
        r=full_loss(p,t);self.assertEqual(r['support']['relations']['matched'],1)
        r['loss'].backward();self.assertLess(p['relations']['logits']['resolves_to'].grad[0].item(),0)

    def test_contradictory_known_relations_fail_closed(self):
        p=prediction();t=target()
        row=dict(source=[{'kind':'h','index':0}],target=[{'kind':'h','index':1}],type='resolves_to',value=True,mask=True)
        t['relations']=[row,{**row,'value':False}]
        with self.assertRaisesRegex(ValueError,'contradictory'):
            full_loss(p,t)


if __name__ == '__main__':
    unittest.main()
