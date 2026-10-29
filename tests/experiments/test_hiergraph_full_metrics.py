"""Synthetic full-graph metrics and support contracts."""
import unittest
import torch
from music_critic.experiments.hierarchical_graph_analyzer.full_metrics import evaluate_full_piece, aggregate_full_metrics


def ep(spans):
    return dict(spans=torch.tensor(spans).reshape(-1,2),object_logits=torch.zeros(len(spans)),field_logits={})


def fixture():
    p={'H':ep([[0.,.5]]),'P':ep([[0.,1.]]),
       'h':dict(spans=torch.tensor([[.1,.2]]),note_index=torch.tensor([0]),H_index=torch.tensor([0]),field_logits={'h.chord_membership':torch.tensor([[4.,-4.]])}),
       'participants':dict(P_index=torch.tensor([0]),endpoint_level=('H',),endpoint_index=torch.tensor([0]),existence_logits=torch.tensor([4.])),
       'relations':dict(sources=(({ 'kind':'h','index':0},),),targets=(({ 'kind':'H','index':0},),),logits={'prepares':torch.tensor([4.])})}
    t={'family_id':'f','work_id':'w','H':dict(ids=['H0','H1'],spans=[[0.,.5],[.5,1.]],fields={}),
       'h':dict(ids=['h0','h1'],spans=[[.1,.2],[.6,.7]],note_index=[0,1],H_index=[0,1],fields={'h.chord_membership':dict(values=[0,1],mask=[True,True])}),
       'P':dict(ids=['P0'],spans=[[0.,1.]],fields={}),
       'participants':[dict(P_index=0,endpoint={'kind':'H','index':0},value=True,mask=True),dict(P_index=0,endpoint={'kind':'H','index':1},value=True,mask=True)],
       'relations':[dict(source=[{'kind':'h','index':0}],target=[{'kind':'H','index':0}],type='prepares',value=True,mask=True),dict(source=[{'kind':'h','index':1}],target=[{'kind':'H','index':0}],type='prepares',value=True,mask=True)]}
    return p,t


class FullMetricsTests(unittest.TestCase):
    def test_missing_H_preserves_h_and_relation_denominators(self):
        p,t=fixture();r=evaluate_full_piece(p,t)
        self.assertEqual(r['H']['known_h_recovery'],.5)
        self.assertEqual(r['h']['fields']['h.chord_membership']['known_support'],2)
        self.assertEqual(r['h']['fields']['h.chord_membership']['joint_recovery'],.5)
        self.assertEqual(r['participants']['known_positive'],2)
        self.assertEqual(r['participants']['positive_recall'],.5)
        self.assertEqual(r['relations']['known_positive'],2)
        self.assertEqual(r['relations']['positive_recall'],.5)

    def test_family_macro_and_geometry_are_explicit(self):
        p,t=fixture();r=evaluate_full_piece(p,t)
        a=aggregate_full_metrics([r]);self.assertEqual(a['H']['known_h_recovery'],.5)
        with self.assertRaisesRegex(ValueError,'geometry'):
            aggregate_full_metrics([r,{**r,'geometry':'oracle'}])

    def test_no_labels_is_unavailable_not_zero(self):
        p,_=fixture();r=evaluate_full_piece(p,{'family_id':'f'})
        self.assertIsNone(r['H']['known_h_recovery']);self.assertIsNone(r['relations']['positive_recall'])
        self.assertEqual(r['relations']['known_positive'],0)


class FullMetricsEdgeCaseTests(unittest.TestCase):
    def test_unresolved_targets_and_hyperedges_are_recovery_misses(self):
        p,t=fixture()
        t['participants'].append(dict(P_index=0,endpoint={'kind':'unsupported','index':-1,'id':'other'},value=True,mask=True))
        t['relations']=[dict(source=[{'kind':'h','index':0}],target=[{'kind':'H','index':0},{'kind':'H','index':-1,'id':'other-view'}],type='prepares',value=True,mask=True)]
        r=evaluate_full_piece(p,t)
        self.assertEqual(r['participants']['known_positive'],3)
        self.assertEqual(r['participants']['positive_recall'],1/3)
        self.assertEqual(r['relations']['known_positive'],1)
        self.assertEqual(r['relations']['positive_recall'],0)
        self.assertEqual(r['relations']['by_type']['prepares']['recall_at5'],0)

    def test_roles_have_independent_support_and_masks(self):
        p,t=fixture();p['participants']['field_logits']={'P.member_role':torch.tensor([[3.,-3.]])}
        t['participants'][0]['fields']={'P.member_role':{'value':0,'mask':True}}
        t['participants'][1]['fields']={'P.member_role':{'value':1,'mask':True}}
        r=evaluate_full_piece(p,t);f=r['participants']['fields']['P.member_role']
        self.assertEqual(f['known_support'],2);self.assertEqual(f['joint_recovery'],.5)
        self.assertEqual(aggregate_full_metrics([r])['participants']['fields']['P.member_role']['joint_recovery'],.5)

    def test_oracle_requires_marked_model_output(self):
        p,t=fixture()
        with self.assertRaisesRegex(ValueError,'geometry'):evaluate_full_piece(p,t,geometry='oracle')
        p['geometry']='oracle';self.assertEqual(evaluate_full_piece(p,t,geometry='oracle')['geometry'],'oracle')

    def test_family_macro_excludes_unavailable_families(self):
        p,t=fixture();first=evaluate_full_piece(p,t);other=evaluate_full_piece(p,{'family_id':'unlabeled'})
        result=aggregate_full_metrics([first,other])
        self.assertEqual(result['H']['known_h_recovery'],.5)
        self.assertEqual(result['H']['supported_families_known_h_recovery'],1)

class FullGeometryMetricTests(unittest.TestCase):
    def test_point_events_remain_in_event_and_field_denominators(self):
        p,t=fixture();t['H']['spans'][1]=[.6,.6]
        t['H']['fields']={'H.function':{'values':[0,0],'mask':[True,True]}}
        p['H']['field_logits']={'H.function':torch.tensor([[4.,-4.]])}
        r=evaluate_full_piece(p,t)
        self.assertEqual(r['H']['known_h'],2)
        self.assertEqual(r['H']['known_h_recovery'],.5)
        self.assertEqual(r['H']['fields']['H.function']['known_support'],2)
        self.assertEqual(r['H']['joint_h_f_recovery'],.5)

class FullRetrievalTests(unittest.TestCase):
    def test_ranking_keeps_unknown_candidates_without_calling_them_negatives(self):
        p,t=fixture();p['relations']['sources']=(p['relations']['sources'][0],p['relations']['sources'][0])
        p['relations']['targets']=(p['relations']['targets'][0],({'kind':'P','index':0},))
        p['relations']['logits']['prepares']=torch.tensor([4.,8.])
        r=evaluate_full_piece(p,t)['relations']['by_type']['prepares']
        self.assertEqual(r['known'],2);self.assertEqual(r['scored'],1)
        self.assertEqual(r['known_negative'],0);self.assertEqual(r['conditional_precision'],1)
        self.assertEqual(r['recall_at1'],0)
        self.assertEqual(r['recall_at5'],.5)
        self.assertEqual(r['reciprocal_rank'],.25)


if __name__ == '__main__':
    unittest.main()
