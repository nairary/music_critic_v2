"""Synthetic context/readout mechanics; no musical-quality claim."""
from dataclasses import replace
from fractions import Fraction as F
import torch
from music_critic.critic.contracts import (
    FeatureBinding, SampleGeometry, FeatureSample, CriticFeaturePyramid,
    RawNote, RawInterval,
)
from music_critic.critic.model import MultiscaleAspectCritic


def fixture_sample(ident='piece', width=8, empty=False):
    notes = () if empty else (
        RawNote('n0', 't0', F(0), F(3), 60),
        RawNote('n1', 't0', F(1), F(2), 64),
        RawNote('n2', 't1', F(0), F(1), 67),
        RawNote('n3', 't1', F(3), F(3), 72),
    )
    tracks = ('t0', 't1')
    intervals = {
        'beat': tuple(RawInterval(f'b{i}', F(i), F(i+1)) for i in range(4)),
        'bar': (RawInterval('bar0', F(0), F(2)), RawInterval('bar1', F(2), F(4))),
        'slice': tuple(RawInterval(f's{i}', F(i), F(i+1)) for i in range(4)),
        'onset': tuple(RawInterval(f'o{i}', F(i), F(i)) for i in (0,1,3)) if notes else (),
    }
    g = torch.Generator().manual_seed(3)
    sizes = {'note':len(notes), 'note_local':len(notes), 'track':2, 'song':1,
             **{k:len(v) for k,v in intervals.items()}}
    embeds={k:torch.randn(n,width,generator=g) for k,n in sizes.items()}
    return FeatureSample(SampleGeometry(ident,notes,tracks,intervals,F(4)),embeds,
                         'synthetic-content', {k:'synthetic' for k in embeds})


def pyramid(*samples):
    return CriticFeaturePyramid(samples, FeatureBinding('synthetic-source','synthetic-checkpoint','fixture'))


def test_four_aspects_and_exact_local_candidates():
    torch.manual_seed(10)
    model=MultiscaleAspectCritic(feature_dim=8,width=8)
    out=model(pyramid(fixture_sample())).samples[0]
    assert out.raw_scores.shape == (4,)
    assert out.availability.tolist() == [True]*4
    assert out.musical_error_location is None
    assert out.musical_error_reason == 'no_validated_targets'
    assert out.local['structure/boundary'].candidates[0].start == F(2)
    assert out.local['harmony/slice'].candidates[2].note_ids == ('n0',)
    assert torch.isfinite(out.raw_scores).all()


def test_batch_isolation_empty_and_frozen_inputs():
    torch.manual_seed(10)
    model=MultiscaleAspectCritic(feature_dim=8,width=8)
    sample=fixture_sample()
    sample=replace(sample, embeddings={k:v.requires_grad_() for k,v in sample.embeddings.items()})
    alone=model(pyramid(sample)).samples[0]
    mixed=model(pyramid(fixture_sample('empty',empty=True), sample, fixture_sample('other')))
    torch.testing.assert_close(alone.raw_scores,mixed.samples[1].raw_scores)
    assert not mixed.samples[0].availability.any()
    assert torch.isfinite(mixed.samples[0].raw_scores).all()
    (alone.raw_scores.sum()+sum(p.logits.sum() for p in alone.local.values())).backward()
    assert all(v.grad is None for v in sample.embeddings.values())
    for name in ('track_gru','slice_gru','bar_gru','fusion','readouts','local_heads'):
        grads=[p.grad for p in getattr(model,name).parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(g.abs().sum() for g in grads)>0, name


def test_simultaneous_note_and_track_row_order_do_not_change_readout():
    model=MultiscaleAspectCritic(feature_dim=8,width=8).eval()
    sample=fixture_sample()
    order=[2,0,3,1]
    geo=replace(sample.geometry,notes=tuple(sample.geometry.notes[i] for i in order),
                tracks=tuple(reversed(sample.geometry.tracks)))
    embeds=dict(sample.embeddings)
    for name in ('note','note_local'): embeds[name]=embeds[name][order]
    embeds['track']=embeds['track'].flip(0)
    changed=replace(sample,geometry=geo,embeddings=embeds)
    a=model(pyramid(sample)).samples[0]
    b=model(pyramid(changed)).samples[0]
    torch.testing.assert_close(a.raw_scores,b.raw_scores,atol=2e-6,rtol=2e-6)
    expected={c.candidate_id:v for c,v in zip(a.local['melody/note'].candidates,a.local['melody/note'].logits)}
    for c,v in zip(b.local['melody/note'].candidates,b.local['melody/note'].logits):
        torch.testing.assert_close(v,expected[c.candidate_id],atol=2e-6,rtol=2e-6)


def test_percussion_does_not_claim_pitched_aspects_and_geometry_matters():
    model=MultiscaleAspectCritic(feature_dim=8,width=8).eval()
    sample=fixture_sample()
    drums=replace(sample, geometry=replace(sample.geometry,
        notes=tuple(replace(n,is_drum=True) for n in sample.geometry.notes)))
    out=model(pyramid(drums)).samples[0]
    assert out.availability.tolist()==[False,True,False,True]
    assert not out.local['harmony/slice'].availability.any()
    shifted=replace(sample,geometry=replace(sample.geometry,
        notes=tuple(replace(n,start=n.start+F(1,3),end=n.end+F(1,3)) for n in sample.geometry.notes)))
    # The raw geometry is used by the new context paths, beyond existing embeddings.
    assert not torch.allclose(model(pyramid(sample)).samples[0].raw_scores,
                              model(pyramid(shifted)).samples[0].raw_scores)


def test_incomplete_pickup_does_not_create_a_complete_two_bar_span():
    model=MultiscaleAspectCritic(feature_dim=8,width=8)
    sample=fixture_sample()
    intervals=dict(sample.geometry.intervals)
    first,last=intervals['bar']
    intervals['bar']=(replace(first,complete=False),last)
    changed=replace(sample,geometry=replace(sample.geometry,intervals=intervals))
    out=model(pyramid(changed)).samples[0]
    assert len(out.contextual['span2'])==0
    assert len(out.local['structure/boundary'].candidates)==1


def test_empty_track_does_not_enter_new_context_pooling_at_fixed_features():
    model=MultiscaleAspectCritic(feature_dim=8,width=8).eval()
    sample=fixture_sample()
    geo=replace(sample.geometry,tracks=(*sample.geometry.tracks,'empty'))
    features=dict(sample.embeddings)
    features['track']=torch.cat((features['track'],torch.full((1,8),100.)))
    changed=replace(sample,geometry=geo,embeddings=features)
    torch.testing.assert_close(model(pyramid(sample)).samples[0].raw_scores,
                               model(pyramid(changed)).samples[0].raw_scores)


def test_all_span_scales_receive_gradients_on_a_long_synthetic_fragment():
    sample=fixture_sample()
    notes=tuple(replace(n,note_id=f'{n.note_id}_{j}',start=n.start+4*j,end=n.end+4*j)
                for j in range(6) for n in sample.geometry.notes)
    intervals={k:tuple(replace(i,entity_id=f'{i.entity_id}_{j}',start=i.start+4*j,end=i.end+4*j)
                      for j in range(6) for i in rows) for k,rows in sample.geometry.intervals.items()}
    g=torch.Generator().manual_seed(12)
    features={k:torch.randn((len(notes) if k in ('note','note_local') else
        2 if k=='track' else 1 if k=='song' else len(intervals[k])),8,generator=g)
        for k in sample.embeddings}
    long=replace(sample,geometry=replace(sample.geometry,notes=notes,intervals=intervals,duration=F(24)),
                 embeddings=features)
    model=MultiscaleAspectCritic(feature_dim=8,width=8)
    out=model(pyramid(long)).samples[0]
    assert [len(out.contextual[f'span{k}']) for k in (2,4,8)]==[11,9,5]
    out.raw_scores.sum().backward()
    for row in (2,3,4):
        assert model.fusion.relations.weight.grad[row].abs().sum()>0
    for row in (5,6,7):
        assert model.readouts['harmony'].queries.grad[row].abs().sum()>0
