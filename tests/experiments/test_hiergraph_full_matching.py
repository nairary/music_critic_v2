"""Large-graph matcher parity and invariant tests on synthetic intervals."""
import torch
from music_critic.experiments.hierarchical_graph_analyzer.full_matching import match_full_events
from music_critic.experiments.hierarchical_graph_analyzer.matching import match_events


def test_tensor_matcher_matches_exact_objective_on_nontied_intervals():
    generator=torch.Generator().manual_seed(71)
    start=torch.rand(28,generator=generator)*.8
    spans=torch.stack([start,start+.08+torch.rand(28,generator=generator)*.1],-1)
    p={'spans':spans,'object_logits':torch.zeros(28),'ids':[str(i) for i in range(28)]}
    t={'spans':(spans[:19]+.002).tolist(),'ids':['g'+str(i) for i in range(19)]}
    for mode in ('training','metric'):
        assert match_full_events(p,t,mode=mode,exact_pair_limit=0)==match_events(p,t,mode=mode)


def test_private_dummies_preserve_unmatched_and_domain_masks():
    p={'spans':torch.tensor([[0.,.1],[.6,.8]]),'object_logits':torch.zeros(2),'view_ids':['a','b']}
    t={'spans':[[.05,.15],[.6,.8],[.8,.9]],'ids':['x','y','z'],'view_ids':['b','a','b']}
    for mode in ('training','metric'):
        assert match_full_events(p,t,mode=mode,exact_pair_limit=0)==match_events(p,t,mode=mode)
    t['event_mask']=[False,False,False]
    assert match_full_events(p,t,exact_pair_limit=0)==[]


def test_exact_iou_threshold_boundary_is_retained():
    p={'spans':torch.tensor([[0.,.5],[.5,1.]]),'object_logits':torch.zeros(2)}
    t={'spans':[[0.,1.]],'ids':['g']}
    assert match_full_events(p,t,mode='metric',exact_pair_limit=0)==match_events(p,t,mode='metric')
