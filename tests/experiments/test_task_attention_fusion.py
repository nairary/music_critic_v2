from copy import deepcopy
from hashlib import sha256
from dataclasses import replace
import pytest
import torch
from music_critic.experiments.analysisgnn import task_attention_fusion as attention
from music_critic.experiments.analysisgnn import fusion_accelerated_campaign as campaign
from music_critic.experiments.analysisgnn import fusion_accelerated_resume as resume
from music_critic.experiments.analysisgnn import corrected_training as ref
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.harmonic_fusion_resume import capture_safe_rng
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint as fp
from music_critic.experiments.analysisgnn.contracts import fingerprint
from tests.experiments.test_harmonic_logit_fusion import _predictions


def test_official_formula_and_no_cross_piece_attention():
    torch.manual_seed(7)
    m=attention.TaskAttentionFusion().eval();pred=_predictions();original={p.task_id:p.logits for p in pred}
    tokens=torch.stack([m.projections[h](original[h]) for h in m.task_ids],dim=1)
    mixed,_=m.attention(tokens,tokens,tokens)
    expected=m.norm(tokens+mixed);out=m(pred)
    for p in out:
        if p.task_id in m.task_ids:
            assert torch.equal(p.logits,m.outputs[p.task_id](expected[:,m.task_ids.index(p.task_id)]))
        else:
            assert p is next(v for v in pred if v.task_id==p.task_id)
    changed=[]
    for p in pred:
        z=p.logits.detach().clone()
        if p.task_id in m.task_ids:z[2:]*=torch.arange(1,z.shape[1]+1)
        changed.append(replace(p,logits=z))
    other=m(tuple(changed))
    assert all(torch.equal(p.logits[:2],q.logits[:2]) for p,q in zip(out,other))


def test_cross_task_gradient_and_non_identity():
    m=attention.TaskAttentionFusion().eval();pred=_predictions();out=m(pred)
    first=next(p for p in out if p.task_id==m.task_ids[0])
    assert not torch.equal(first.logits,next(p for p in pred if p.task_id==first.task_id).logits)
    first.logits.square().sum().backward()
    for p in pred:
        if p.task_id in m.task_ids:assert p.logits.grad is not None and p.logits.grad.abs().sum()>0
    assert m.attention.in_proj_weight.grad.abs().sum()>0


def test_misaligned_entities_rejected():
    pred=_predictions();task=attention.TaskAttentionFusion().task_ids[1]
    bad=tuple(replace(p,sample_indices=p.sample_indices.flip(0)) if p.task_id==task else p for p in pred)
    with pytest.raises(ValueError,match='alignment'):attention.TaskAttentionFusion()(bad)


def test_empty_beats_and_maskless_api():
    pred=_predictions(beat_samples=())
    assert attention.TaskAttentionFusion()(pred) is pred
    with pytest.raises(TypeError):attention.TaskAttentionFusion()(pred,targets={})


def test_control_and_attention_start_with_identical_base_and_training_rng():
    a=campaign.components('cpu','control');rng=capture_safe_rng();base=fp(a[0].base.state_dict())
    b=campaign.components('cpu','attention')
    assert base==fp(b[0].base.state_dict()) and fp(rng)==fp(capture_safe_rng())
    assert len(a[0].state_dict())<len(b[0].state_dict())
    for c in (a,b):
        ids=[id(p) for g in c[1].param_groups for p in g['params']]
        assert len(ids)==len(set(ids))==len(list(c[0].parameters()))
    raw,_=ref.build_source_free_fixture();a[0].eval()
    assert all(torch.equal(v,w) for v,w in zip(a[0](raw.raw_graph_batch).logits.values(),a[0].base(raw.raw_graph_batch).logits.values()))


@pytest.mark.parametrize('kind',['attention','control','mmoe'])
def test_model_checkpoint_and_exact_full_resume(tmp_path,kind):
    original=campaign.components('cpu',kind);raw,sidecar=ref.build_source_free_fixture()
    table=(FullOrbitPair(sidecar['record_id'],0,sidecar['source_component_id']),);sampler=FullOrbitSampler(table,seed=17)
    def step(c):
        return ref.attempt_applied_update(model=c[0],optimizer=c[1],scheduler=c[2],scaler=c[3],raw_graph_batch=raw.raw_graph_batch,
            sidecars=(sidecar,),shifts=(0,),class_weights=ref.load_frozen_class_weights())
    step(original);sampler.advance_after_applied_update(2)
    path=tmp_path/'model.pt';torch.save(campaign.create_model_payload(original[0]),path)
    pin={'expected_sha256':sha256(path.read_bytes()).hexdigest(),'expected_size_bytes':path.stat().st_size}
    rng=fp(capture_safe_rng())
    if kind == 'mmoe':
        from music_critic.experiments.analysisgnn.grouped_mmoe import load_model_checkpoint
        loaded=load_model_checkpoint(path,**pin)
    else:
        loaded=attention.load_model_checkpoint(path,enabled=kind=='attention',**pin)
    assert fp(loaded.state_dict())==fp(original[0].state_dict()) and fp(capture_safe_rng())==rng
    with pytest.raises(ValueError):attention.load_model_checkpoint(path,enabled=kind!='attention',**pin)
    queue={'committed_draw_cursor':2,'committed_batch_cursor':1,'start_draw_offset':0};queue['fingerprint']=fingerprint(queue)
    payload={'schema_version':'GroupedMMoEAttentionCampaignState@1.0.0' if kind=='mmoe' else 'TaskAttentionCampaignState@1.0.0','update':1,'config':{'model_kind':kind,'data_manifest_sha256':'a'*64},
        'model':campaign.create_model_payload(original[0]),'optimizer':deepcopy(original[1].state_dict()),'scheduler':original[2].state_dict(),
        'scaler':original[3].state_dict(),'sampler':sampler.state_dict(),'prefetch':queue,'rng':capture_safe_rng()}
    path=tmp_path/'full.pt';torch.save(payload,path)
    pin={'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}
    saved=resume.load_resume_payload(pin,expected_update=1,expected_data_sha256='a'*64,expected_model_kind=kind)
    with pytest.raises(ValueError):resume.load_resume_payload(pin,expected_update=1,expected_data_sha256='a'*64)
    loss=float(step(original).loss.total.detach());expected=fp((original[0].state_dict(),original[1].state_dict(),capture_safe_rng()))
    restored=campaign.components('cpu',kind)
    resume.restore_campaign_state(saved,*restored[:4],FullOrbitSampler(table,seed=17),device='cpu')
    assert float(step(restored).loss.total.detach())==loss
    assert fp((restored[0].state_dict(),restored[1].state_dict(),capture_safe_rng()))==expected
