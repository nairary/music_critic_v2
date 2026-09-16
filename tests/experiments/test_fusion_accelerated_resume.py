from dataclasses import asdict
from copy import deepcopy
from hashlib import sha256
import pytest
import torch
from music_critic.experiments.analysisgnn import fusion_accelerated_campaign as campaign
from music_critic.experiments.analysisgnn import fusion_accelerated_resume as resume
from music_critic.experiments.analysisgnn import corrected_training as ref
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.harmonic_fusion_checkpoint import create_model_payload
from music_critic.experiments.analysisgnn.harmonic_fusion_resume import capture_safe_rng,restore_safe_rng
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
from music_critic.experiments.analysisgnn.contracts import fingerprint


def test_suffix_exactly_matches_existing_stream():
    table=tuple(FullOrbitPair(r,s,r) for r in ('a','b') for s in range(12))
    jobs=resume.continuation_jobs(table,15000,15005)
    sampler=FullOrbitSampler(table,seed=17,position=30000)
    assert jobs[0].position==0 and jobs[0].draw_ids==(30000,30001)
    for job in jobs:
        assert job.record_ids==tuple(sampler.peek(k).record_id for k in range(2))
        assert job.applied_shifts==tuple(sampler.peek(k).shift_pc for k in range(2))
        sampler.advance_after_applied_update(2)


@pytest.mark.parametrize('bounds',[(15000,15000),(-1,80000),(15000,80001),(True,80000)])
def test_bad_suffix(bounds):
    with pytest.raises(ValueError,match='budget'):
        resume.continuation_jobs([], *bounds)


def test_runtime_mismatch_rejected():
    with pytest.raises(ValueError,match='runtime_torch'):
        resume.verify_resume_environment({'torch':'a'},{'torch':'b'})


def test_exact_next_update_after_restricted_roundtrip(tmp_path):
    raw, sidecar=ref.build_source_free_fixture()
    table=(FullOrbitPair(sidecar['record_id'],0,sidecar['source_component_id']),)
    weights=ref.load_frozen_class_weights()
    def step(c):
        return ref.attempt_applied_update(model=c[0],optimizer=c[1],scheduler=c[2],scaler=c[3],
            raw_graph_batch=raw.raw_graph_batch,sidecars=(sidecar,),shifts=(0,),class_weights=weights)
    original=campaign.components('cpu');sampler=FullOrbitSampler(table,seed=17)
    step(original);sampler.advance_after_applied_update(2)
    queue={'committed_draw_cursor':2,'committed_batch_cursor':1,'start_draw_offset':0}
    queue['fingerprint']=fingerprint(queue)
    payload={'schema_version':'FusionAcceleratedCampaignState@1.0.0','update':1,
        'config':{'data_manifest_sha256':'a'*64},'model':create_model_payload(original[0]),
        'optimizer':deepcopy(original[1].state_dict()),'scheduler':original[2].state_dict(),
        'scaler':original[3].state_dict(),'sampler':sampler.state_dict(),'prefetch':queue,'rng':capture_safe_rng()}
    path=tmp_path/'checkpoint.pt';torch.save(payload,path)
    pin={'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest(),'size_bytes':path.stat().st_size}
    loaded=resume.load_resume_payload(pin,expected_update=1,expected_data_sha256='a'*64)
    expected_loss=float(step(original).loss.total)
    expected=recursive_state_fingerprint((original[0].state_dict(),original[1].state_dict(),capture_safe_rng()))
    restored=campaign.components('cpu');new_sampler=FullOrbitSampler(table,seed=17)
    receipt=resume.restore_campaign_state(loaded,*restored[:4],new_sampler,device='cpu')
    assert receipt['next_update']==2 and new_sampler.position==2
    actual_loss=float(step(restored).loss.total)
    assert actual_loss==expected_loss
    assert recursive_state_fingerprint((restored[0].state_dict(),restored[1].state_dict(),capture_safe_rng()))==expected
    with pytest.raises(ValueError,match='checkpoint_hash'):
        resume.load_resume_payload({**pin,'sha256':'b'*64},expected_update=1,expected_data_sha256='a'*64)
    with pytest.raises(ValueError,match='checkpoint_update'):
        resume.load_resume_payload(pin,expected_update=2,expected_data_sha256='a'*64)
    broken=deepcopy(loaded);broken['prefetch']['committed_draw_cursor']=4
    with pytest.raises(ValueError,match='prefetch_fingerprint'):
        resume.restore_campaign_state(broken,*restored[:4],new_sampler,device='cpu')
