"""Synthetic full graph mechanics; no musical-quality claim."""
from fractions import Fraction
import pytest
import torch
from music_critic.experiments.hierarchical_graph_analyzer.full_model import FullGraphAnalyzer, FullGraphConfig


def raw_work():
    g=torch.Generator().manual_seed(23)
    tiles=[]
    for _ in range(2):
        tiles.append(dict(x=torch.randn(4,12,generator=g),node_type=torch.tensor([0,6,6,6]),
            node_time=torch.tensor([[0.,1.],[0.,0.],[.5,.5],[1.,1.]]),
            edge_index=torch.tensor([[0,1,2],[1,2,3]]),edge_type=torch.tensor([0,1,1]),
            time_index=torch.tensor([1,2,3]),time_positions=torch.tensor([0.,.5,1.]),
            bank_indices=torch.tensor([[0,1],[1,2]]),pool_mask=torch.tensor([False,True,True,True])))
    return dict(tiles=tuple(tiles),tile_spans=torch.tensor([[0.,.5],[.5,1.]]),
        core_spans=torch.tensor([[0.,.5],[.5,1.]]),owned_time_indices=(torch.tensor([0,1]),torch.tensor([0,1])),note_index_by_tile=(torch.tensor([0]),torch.tensor([1])),
        notes=dict(spans=torch.tensor([[0.,.5],[.5,1.]]),pitch=torch.tensor([60,64]),track_index=torch.tensor([0,1]),
                   is_percussion=torch.tensor([False,False]),is_grace=torch.tensor([False,False])),
        bar_spans=torch.tensor([[0.,.25],[.25,.5],[.5,.75],[.75,1.]]),note_ids=('n0','n1'),
        origin=Fraction(3),duration=Fraction(16),coverage={})


def model(**kw):
    torch.manual_seed(17)
    config=FullGraphConfig(input_width=12,hidden=16,attention_heads=2,constructor_rounds=1,
        rounds=2,process_scales=(1,2),max_h=32,max_p=32,max_contexts=128,**kw)
    return FullGraphAnalyzer({'H.function':3},{'h.chord_membership':2,'h.temporal_role':3},
        {'P.process':3},relation_types=('resolves_to',),config=config)


def test_complete_raw_forward_proposes_all_levels_and_refines_spans():
    m=model();p=m(raw_work())
    assert set(('H','h','P','participants','relations','notes','rounds'))<=set(p)
    for name in ['H','h','P']:
        assert len(p[name]['spans'])>0
        assert torch.isfinite(p[name]['states']).all()
        assert (p[name]['spans'][:,1]>p[name]['spans'][:,0]).all()
    assert not torch.allclose(p['rounds'][0]['H_spans'],p['H']['spans'])
    assert not torch.allclose(p['rounds'][0]['P_spans'],p['P']['spans'])
    assert len(p['notes']['states'])==2


def test_process_loss_reaches_E3_geometry_and_physical_note_representation():
    m=model();p=m(raw_work())
    loss=p['P']['field_logits']['P.process'].square().sum()+p['P']['spans'].square().sum()
    loss.backward()
    for parameter in [m.constructor.endpoint_head.weight,m.note_projection[0].weight,
                      m.round_layers[0].H_delta.weight,m.round_layers[0].P_to_H.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0


def test_distant_individual_nodes_change_local_H_and_not_only_one_mean():
    m=model();a=raw_work();b=raw_work();b['tiles'][1]['x'][0]+=3
    first=m(a);second=m(b)
    assert not torch.allclose(first['H']['states'][:2],second['H']['states'][:2])
    assert not torch.allclose(first['H']['spans'][:2],second['H']['spans'][:2])


def test_global_geometry_can_cross_initial_local_window():
    m=model()
    with torch.no_grad():
        for layer in m.round_layers:
            layer.H_delta.weight.zero_();layer.H_delta.bias.copy_(torch.tensor([0.,4.]))
    p=m(raw_work())
    assert (p['rounds'][0]['H_spans'][:2,1]<=.5).all()
    assert (p['H']['spans'][:2,1]>.5).all()
    n=p['h']['note_index'];j=p['h']['H_index'];bounds=raw_work()['notes']['spans'][n]
    assert (p['h']['spans'][:,0]>=bounds[:,0]).all()
    assert (p['h']['spans'][:,1]<=bounds[:,1]).all()
    assert (p['h']['spans'][:,0]>=p['H']['spans'][j,0]).all()
    assert (p['h']['spans'][:,1]<=p['H']['spans'][j,1]).all()


def test_target_like_input_is_rejected_and_capacity_is_explicit():
    a=raw_work();a['targets']={'H':{}}
    with pytest.raises(ValueError,match='raw_schema'):model()(a)
    from dataclasses import replace
    from music_critic.experiments.hierarchical_graph_analyzer.full_model import FullGraphCapacityError
    m=model();m.config=replace(m.config,max_h=1)
    with pytest.raises(FullGraphCapacityError,match='H_capacity'):m(raw_work())


def test_process_change_flows_to_H_h_and_notes():
    m=model();baseline=m(raw_work())
    with torch.no_grad():m.P_geometry.bias.add_(torch.arange(16)*.7)
    changed=m(raw_work())
    for key in ['H','h','notes']:
        assert not torch.allclose(baseline[key]['states'],changed[key]['states'])


def test_point_and_percussion_notes_retained_without_fabricated_membership():
    raw=raw_work();raw['notes']['spans'][1]=torch.tensor([.6,.6]);raw['notes']['is_percussion'][1]=True;raw['notes']['is_grace'][1]=True
    p=model()(raw)
    assert len(p['notes']['states'])==2
    assert p['coverage']['point_notes_without_h']==1
    assert not (p['h']['note_index']==1).any()


def test_oracle_is_explicit_and_preserves_supplied_geometry():
    m=model();H=torch.tensor([[.1,.7],[.7,.95]]);P=torch.tensor([[.05,.99]])
    p=m.oracle_forward(raw_work(),H,P)
    assert p['geometry']=='oracle' and p['oracle_levels']==('H','P')
    assert torch.equal(p['H']['spans'],H) and torch.equal(p['P']['spans'],P)
    assert not p['H']['states'].requires_grad


def test_native_full_supervision_routes_downstream_gradients_to_constructor():
    from music_critic.experiments.hierarchical_graph_analyzer.full_losses import full_loss
    m=model();p=m(raw_work())
    target={}
    for level,field in [('H','H.function'),('P','P.process')]:
        spans=p[level]['spans'].detach().clone()
        spans[:,1]=spans[:,0]+(spans[:,1]-spans[:,0])*.9
        target[level]=dict(ids=[str(i) for i in range(len(spans))],spans=spans.tolist(),
            fields={field:dict(values=[1]*len(spans),mask=[True]*len(spans))})
    n=len(p['h']['spans'])
    target['h']=dict(ids=[str(i) for i in range(n)],spans=p['h']['spans'].detach().tolist(),
        note_index=p['h']['note_index'].tolist(),H_index=p['h']['H_index'].tolist(),
        fields={'h.chord_membership':dict(values=[0]*n,mask=[True]*n)})
    part=p['participants']
    target['participants']=[dict(P_index=int(part['P_index'][0]),endpoint=dict(kind=part['endpoint_level'][0],index=int(part['endpoint_index'][0])),value=True,mask=True)]
    rel=p['relations']
    target['relations']=[dict(source=list(rel['sources'][0]),target=list(rel['targets'][0]),type='resolves_to',value=True,mask=True)]
    result=full_loss(p,target)
    assert result['supported']
    result['loss'].backward()
    for parameter in [m.constructor.endpoint_head.weight,m.note_projection[0].weight,
                      m.round_layers[0].H_delta.weight,m.round_layers[0].P_to_H.weight,
                      m.note_fields.heads[0].weight,m.process_fields.heads[0].weight,
                      m.participant_head[-1].weight,m.relation_heads[0][-1].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0


def test_oracle_empty_and_invalid_intervals_are_explicit():
    m=model();p=m.oracle_forward(raw_work(),torch.empty(0,2),torch.empty(0,2))
    assert not len(p['H']['spans']) and not len(p['P']['spans']) and not len(p['h']['spans'])
    with pytest.raises(ValueError,match='oracle_geometry'):
        m.oracle_forward(raw_work(),torch.tensor([[.2,.2]]),torch.empty(0,2))


def test_exact_owned_anchor_indices_are_authoritative_for_forward():
    raw=raw_work()
    # Deliberately make independently rounded core starts exclude anchor 0;
    # exact routing metadata remains the authority, not float comparisons.
    raw['core_spans'][:,0]+=torch.tensor([1e-6,1e-6])
    p=model()(raw)
    assert p['H']['ids']==('H:0:0','H:0:1','H:1:0','H:1:1')
    raw['owned_time_indices']=(torch.tensor([1,1]),torch.tensor([0,1]))
    with pytest.raises(ValueError,match='exact_owned_anchors'):model()(raw)


def test_bounded_overlap_search_matches_exhaustive_geometric_ranking():
    queries=torch.tensor([[0.,.4],[.2,.8],[.8,1.]])
    bank=torch.tensor([[0.,.3],[.2,.4],[.2,.4],[.5,.7],[.9,1.]])
    for penalize in [False,True]:
        rows=list(FullGraphAnalyzer._overlap_candidates(queries,bank,2,penalize_distance=penalize))
        for q,(chosen,dropped) in zip(queries,rows):
            overlap=torch.minimum(q[1],bank[:,1])-torch.maximum(q[0],bank[:,0])
            valid=torch.where(overlap>0)[0]
            score=overlap[valid]
            if penalize:score=score-(bank[valid].mean(-1)-q.mean()).abs()*.01
            expected=valid[torch.argsort(score,descending=True,stable=True)[:2]].sort().values.tolist()
            assert chosen==expected and dropped==len(valid)-len(expected)


def test_known_event_capacities_reject_before_local_encoder(monkeypatch):
    from dataclasses import replace
    from music_critic.experiments.hierarchical_graph_analyzer.full_model import FullGraphCapacityError
    for cap in ['max_h','max_p']:
        m=model();m.config=replace(m.config,**{cap:1})
        def unexpected_forward(raw):raise AssertionError('must reject before constructing any local graph')
        monkeypatch.setattr(m.constructor,'forward',unexpected_forward)
        with pytest.raises(FullGraphCapacityError):m(raw_work())


def test_oracle_capacity_uses_supplied_P_bank_not_unused_raw_proposals():
    from dataclasses import replace
    m=model();m.config=replace(m.config,max_p=1)
    p=m.oracle_forward(raw_work(),torch.tensor([[.1,.9]]),torch.tensor([[.1,.9]]))
    assert len(p['P']['spans'])==1
