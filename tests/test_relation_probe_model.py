"""Synthetic CPU geometry/gradient checks; not musical evidence."""
from pathlib import Path
import torch
import pytest
from music_critic.relation_probe.model import RelationHead,geometry_score,masked_pair_loss,Probe


def test_translation_is_directional_and_differentiable():
    a=torch.tensor([[1.,0.]],requires_grad=True);b=torch.tensor([[0.,1.]])
    r=torch.tensor([-1.,1.],requires_grad=True)
    assert geometry_score(a,b,r,0.,1.).item()==0
    assert geometry_score(b,a,r,0.,1.).item()==-8
    geometry_score(a,b,r,0.,1.).sum().backward()
    assert a.grad is not None and r.grad is not None


def test_unknown_loss_has_zero_gradient():
    logits=torch.tensor([0.,100.],requires_grad=True)
    loss=masked_pair_loss(logits,torch.tensor([1.,0.]),torch.tensor([True,False]))
    loss.backward();assert logits.grad[0]!=0 and logits.grad[1]==0
    assert masked_pair_loss(logits,torch.zeros(2),torch.zeros(2,dtype=torch.bool)).item()==0


@pytest.mark.parametrize('decoder',['mlp','transe'])
def test_heads_accept_multiple_positive_answers(decoder):
    h=RelationHead(10,decoder=decoder)
    features=torch.randn(3,10)
    score,presence=h(features,torch.tensor([0,0]),torch.tensor([1,2]),torch.tensor([0]))
    assert score.shape==(2,) and presence.shape==(1,)
    assert torch.isfinite(score).all()
    masked_pair_loss(score,torch.ones(2),torch.ones(2,dtype=torch.bool)).backward()
    assert h.projection.weight.grad.abs().sum()>0


def test_encoder_gradient_boundary_and_raw_input_shape():
    from music_critic.data import load_piece
    from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
    torch.set_num_threads(1)
    p=load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json')
    graph=build_temporal_batch((p,))
    model=Probe('random','transe',seed=17)
    event=dict(note_index=torch.tensor([0,1]),slice_index=torch.tensor([0,1]),raw=torch.zeros(2,10))
    h=model.features(graph,event)
    score,presence=model.head(h,torch.tensor([0]),torch.tensor([1]),torch.tensor([0]))
    (score.sum()+presence.sum()).backward()
    assert any(x.grad is not None for x in model.encoder.backbone.base.local_baseline.parameters())
    assert any(x.grad is not None for x in model.encoder.backbone.temporal_layers.parameters())
    assert all(x.grad is None for x in model.encoder.backbone.base.task_heads.parameters())
    model.zero_grad(set_to_none=True);model.freeze_encoder()
    h=model.features(graph,event)
    score,presence=model.head(h,torch.tensor([0]),torch.tensor([1]),torch.tensor([0]))
    (score.sum()+presence.sum()).backward()
    assert all(x.grad is None for x in model.encoder.parameters())
    assert model.head.projection.weight.grad is not None
