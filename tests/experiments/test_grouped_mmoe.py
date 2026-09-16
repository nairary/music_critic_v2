from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import grouped_mmoe as mmoe
from music_critic.experiments.analysisgnn import corrected_training as ref
from music_critic.experiments.analysisgnn import fusion_accelerated_campaign as campaign
from music_critic.experiments.analysisgnn.harmonic_fusion_resume import capture_safe_rng
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint as fp


def test_bank_formula_task_specific_mixtures_and_row_isolation():
    bank=mmoe.ExpertBank(('a','b'),dim=3)
    x=torch.randn(7,3)
    e=bank.expert_values(x)
    for task in ('a','b'):
        y,w=bank.mix(task,x,e)
        assert torch.allclose(w.sum(-1),torch.ones(7))
        assert torch.allclose(y,torch.einsum('ne,ned->nd',w,e))
        z=x.clone();z[4:]+=100
        other,_=bank.mix(task,z,bank.expert_values(z))
        assert torch.equal(y[:4],other[:4])
    assert not torch.equal(bank.mix('a',x,e)[1],bank.mix('b',x,e)[1])


def test_all18_heads_use_two_banks_without_metadata_or_target_changes():
    model=campaign.components('cpu','mmoe')[0].eval()
    raw,_=ref.build_source_free_fixture()
    encoded=model.base.encode(raw.raw_graph_batch)
    original=model.base.task_heads(encoded.fused)
    identity=model.base.task_heads(encoded.fused,feature_transform=lambda task,values:values)
    assert all(torch.equal(a.logits,b.logits) for a,b in zip(original,identity))
    model.capture_routing=True
    result=model.predict_from_encoded(encoded)
    assert len(result.predictions)==18
    assert set(model.banks)=={'beat','note_onset'}
    assert len(model.banks['beat'].gates)==12 and len(model.banks['note_onset'].gates)==6
    for a,b in zip(original,result.predictions):
        assert a.task_id==b.task_id and a.logits.shape==b.logits.shape
        for name in ('sample_indices','global_entity_indices','candidate_node_type_codes',
                     'candidate_offsets_by_node_type','candidate_counts_by_node_type'):
            assert torch.equal(getattr(a,name),getattr(b,name))
    stats=model.routing_report();assert len(stats)==18
    for row in stats.values():
        assert sum(row['top1_counts'])==row['count']
        assert abs(sum(row['mean_weights'])-1)<1e-6
        assert 0<=row['mean_entropy']<=1.386295
    with pytest.raises(TypeError):model(raw.raw_graph_batch,targets={})
    with pytest.raises(ValueError,match='preserve candidate'):
        model.base.task_heads(encoded.fused,feature_transform=lambda task,values:values[:-1])


def test_note_and_onset_share_parameters_not_rows_and_empty_candidates_work():
    model=campaign.components('cpu','mmoe')[0].eval()
    raw,_=ref.build_source_free_fixture();encoded=model.base.encode(raw.raw_graph_batch)
    baseline=model.predict_from_encoded(encoded)
    emb=dict(encoded.fused.embeddings);emb['note']=emb['note']+3
    changed=model.predict_from_encoded(replace(encoded,fused=replace(encoded.fused,embeddings=emb)))
    for p,q in zip(baseline.predictions,changed.predictions):
        if model.routes[p.task_id] in ('onset','beat'):
            assert torch.equal(p.logits,q.logits)
        else:
            assert not torch.equal(p.logits,q.logits)
    emb={k:v[:0] for k,v in encoded.fused.embeddings.items()}
    memberships={k:v[:0] for k,v in encoded.fused.batch_membership.items()}
    # Isolated empty-candidate head fixture, not a modified hierarchy contract.
    empty=SimpleNamespace(fused=replace(encoded.fused,embeddings=emb,batch_membership=memberships))
    model.capture_routing=True
    assert all(p.logits.shape[0]==0 for p in model.predict_from_encoded(empty).predictions)
    assert all(v['count']==0 for v in model.routing_report().values())


def test_gradients_reach_every_expert_and_gate_and_groups_are_separate():
    model=campaign.components('cpu','mmoe')[0].eval()
    raw,_=ref.build_source_free_fixture();encoded=model.base.encode(raw.raw_graph_batch)
    model.predict_from_encoded(encoded).logits['note_degree'].square().mean().backward()
    assert all(p.grad is None for p in model.banks['beat'].parameters())
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.banks['note_onset'].experts.parameters())
    model.zero_grad(set_to_none=True)
    result=model(raw.raw_graph_batch)
    sum(v.square().mean() for v in result.logits.values()).backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.banks.parameters())


def test_initial_baseline_attention_and_rng_match_existing_attention():
    a=campaign.components('cpu','attention')
    base=fp(a[0].base.state_dict());fusion=fp(a[0].fusion.state_dict());rng=fp(capture_safe_rng())
    b=campaign.components('cpu','mmoe')
    assert base==fp(b[0].base.state_dict()) and fusion==fp(b[0].fusion.state_dict())
    assert rng==fp(capture_safe_rng())
    ids=[id(p) for g in b[1].param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(b[0].parameters()))
    assert mmoe.model_contract(b[0])['parameters']['mmoe']==536648


def test_model_loader_rejects_other_kinds_and_hash_tampering(tmp_path):
    from music_critic.experiments.analysisgnn.task_attention_fusion import load_model_checkpoint
    model=campaign.components('cpu','mmoe')[0]
    path=tmp_path/'model.pt';torch.save(campaign.create_model_payload(model),path)
    pin={'expected_sha256':sha256(path.read_bytes()).hexdigest(),'expected_size_bytes':path.stat().st_size}
    rng=fp(capture_safe_rng());loaded=mmoe.load_model_checkpoint(path,**pin)
    assert fp(model.state_dict())==fp(loaded.state_dict()) and fp(capture_safe_rng())==rng
    with pytest.raises(ValueError):load_model_checkpoint(path,enabled=True,**pin)
    with pytest.raises(ValueError):mmoe.load_model_checkpoint(path,**{**pin,'expected_sha256':'0'*64})
