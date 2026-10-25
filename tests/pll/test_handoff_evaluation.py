from dataclasses import replace
import torch
import pytest
from scripts.evaluate_hcm1_handoff import bounded_generate,tree_equal
from tests.pll.test_model import small_model,example
from music_critic.pll.model import HCM1
from music_critic.pll.contracts import CodecProfile
from music_critic.pll.codec import NOTE,EOS

@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_bounded_generation_matches_public_sampler_and_preserves_incomplete():
    model=HCM1(replace(small_model().config,relations='none'),codec_profile=CodecProfile(max_events=1,max_denominator=1,max_numerator=1)).eval()
    with torch.no_grad():
        model.decoder.output.weight.zero_();model.decoder.output.bias.zero_()
        model.decoder.output.bias[NOTE]=20;model.decoder.output.bias[EOS]=-20
    context=example().context
    from dataclasses import asdict
    expected=model.generate(context,generator=torch.Generator().manual_seed(11))
    result=bounded_generate(model,context,seed=11,max_tokens=100,wall_seconds=10)
    assert result['status']=='completed' and result['events']==[asdict(e) for e in expected]
    partial=bounded_generate(model,context,seed=11,max_tokens=1,wall_seconds=10)
    assert partial['status']=='incomplete' and partial['tokens']==[NOTE] and 'events' not in partial
    with pytest.raises(InterruptedError):bounded_generate(model,context,seed=11,max_tokens=100,wall_seconds=10,stop=lambda:True)
    stopped=bounded_generate(model,context,seed=11,max_tokens=100,wall_seconds=.000001)
    assert stopped['status']=='incomplete' and stopped['reason']=='example_wall_budget'


def test_state_comparison_checks_nested_values_and_structure():
    a={'state':[torch.tensor([1.,2.])],'steps':(1,2)}
    assert tree_equal(a,a)
    assert not tree_equal(a,{**a,'steps':(1,3)})
    assert not tree_equal(a,{**a,'state':[torch.tensor([1.,3.])]})


@pytest.mark.parametrize('failure,observed,status',[
    (ValueError('synthetic failure'),[], 'failed'),
    (InterruptedError('synthetic stop'),[], 'aborted'),
    (ValueError('embedded TEST'),[dict(split='test',verified=False)], 'invalid'),
    (ValueError('bad hash'),[dict(split=None,verified=False)], 'failed'),
    (ValueError('bad second hash'),[dict(split='validation',verified=True),dict(split=None,verified=False)], 'failed'),
])
def test_registered_failure_finalizes_and_accounts_reads(tmp_path,monkeypatch,failure,observed,status):
    import json
    from types import SimpleNamespace
    import scripts.evaluate_hcm1_handoff as module
    from tests.pll.test_cli import prepare
    from music_critic.experiments.registry import load_record,seal_record,canonical_json_bytes
    _,_,rp,_=prepare(tmp_path)
    target=tmp_path/'docs/experiments/records'/rp.name;target.parent.mkdir(parents=True)
    record=load_record(rp);record['hcm_diagnostic']={'synthetic':True}
    if any(r['split']=='validation' for r in observed):record['data_access']['declared_policy']['validation_allowed']=True
    # Gate itself is synthetic; validation observation is exercised with evaluate policy.
    mode='evaluate' if record['data_access']['declared_policy']['validation_allowed'] else 'gate'
    target.write_bytes(canonical_json_bytes(seal_record(record))+b"\n")
    monkeypatch.setattr(module,'binding',lambda args:{'synthetic':True})
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'synthetic CPU fixture')
    def fail(args,out,stop,evidence):
        evidence['sampler']=SimpleNamespace(manifest=SimpleNamespace(unverified_reads=[]))
        evidence['reads']=observed
        raise failure
    monkeypatch.setattr(module,mode if mode=='gate' else 'evaluate',fail)
    args=SimpleNamespace(mode=mode,output=str(tmp_path/'result'),record=str(target),device='cpu',wall_seconds=10)
    with pytest.raises(RuntimeError,match=status):module.run(args)
    final=load_record(target)
    assert final['status']==status and isinstance(final['validity']['reason'],str)
    assert (tmp_path/'docs/EXPERIMENT_LEDGER.md').is_file()
    if status=='invalid':
        assert final['data_access']['test_inputs_accessed'] is True
        assert final['data_access']['test_targets_accessed'] is True
        assert final['data_access']['policy_violations']==['test_inputs_accessed','test_targets_accessed']
    elif any(r['split'] is None for r in observed):
        assert final['data_access']['test_inputs_accessed'] is None
        assert final['data_access']['access_evidence_status']==('partial_primary_evidence' if len(observed)>1 else 'pending_primary_evidence')


def test_compact_record_does_not_contain_generated_music():
    from scripts.evaluate_hcm1_handoff import compact_results
    r=compact_results(dict(parent_update=10000,requested=0,supported=0,scores=[],interpretation='fixture',
        generations=[dict(id='a',status='completed',events=[{'pitch':72}],tokens=[3,4,5])]))
    assert r['generations']==[dict(id='a',status='completed',reason=None,events=1,tokens=3)]
