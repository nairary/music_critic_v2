from dataclasses import replace
from hashlib import sha256
import random

import pytest
import torch

from music_critic.pll.runtime import PLLTrainer,TrainingConfig
from scripts.hcm1_stage2 import continue_full_state,SegmentTrainer
from tests.pll.test_runtime import trainer,example


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def parent():
    base=trainer()
    p=PLLTrainer(base.model,TrainingConfig(total_updates=3,warmup_updates=1),data_id='synthetic',protocol_id='parent')
    for _ in range(3):p.step((example(),))
    return p


def child(p):
    return continue_full_state(p,additional_updates=4,protocol_id='child',parent_sha256='a'*64,implementation_sha256='b'*64)


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_full_state_preserved_and_parent_not_changed():
    torch.manual_seed(42);random.seed(42);p=parent()
    oldbinding=dict(p.binding);rng=p._rng_state();c=child(p)
    equal(p.model.state_dict(),c.model.state_dict())
    equal(p.optimizer.state_dict(),c.optimizer.state_dict());equal(rng,c._rng_state())
    assert (c.updates,c.draws)==(3,3)
    assert p.binding==oldbinding and p.config.total_updates==3
    assert c.config.total_updates==7
    assert c._lr_multiplier(2)==p._lr_multiplier(2)
    assert c._lr_multiplier(3)==1.0 and c._lr_multiplier(4)==1.0
    assert c._lr_multiplier(7)==0.0


def test_child_resume_replays_next_step_exactly(tmp_path):
    p=parent();c=child(p);c.step((example(),))
    path=tmp_path/'child.pt';digest=c.save_checkpoint(path,sampler_state={'draws':4})
    expected=c.step((example(),));expectedweights=c.model.state_dict()
    d=child(p);assert d.load_checkpoint(path,expected_sha256=digest)=={'draws':4}
    actual=d.step((example(),));assert actual==expected
    equal(expectedweights,d.model.state_dict())
    assert sha256(path.read_bytes()).hexdigest()==digest


def test_strict_parent_resume_does_not_accept_child(tmp_path):
    p=parent();c=child(p);path=tmp_path/'child.pt';digest=c.save_checkpoint(path,sampler_state={})
    with pytest.raises(ValueError,match='binding'):
        p.load_checkpoint(path,expected_sha256=digest)


def test_unfinished_parent_rejected():
    p=trainer()
    with pytest.raises(ValueError,match='completed'):
        child(p)


def test_whole_completed_parent_keeps_full_sampler_and_exact_child_replay(tmp_path):
    import json
    from scripts.hcm1_stage2 import load_parent
    from music_critic.pll.cli import train
    from music_critic.pll.whole_data import WholeMaskSampler
    from tests.pll.test_whole_runtime import whole_run
    cfg, manifest, record, output = whole_run(tmp_path, updates=2, total=2)
    receipt = train(cfg, manifest, record, output)
    checkpoint = output/receipt['checkpoint']
    checksum = sha256(checkpoint.read_bytes()).hexdigest()
    p, sampler = load_parent(cfg, manifest, checkpoint, checksum,
        output/'training-binding.json', record, device='cpu')
    assert isinstance(sampler, WholeMaskSampler)
    assert p.updates == sampler.draws == 2
    initial_rng = p._rng_state()
    c = continue_full_state(p, additional_updates=4, protocol_id='whole-child',
        parent_sha256=checksum, implementation_sha256='b'*64)
    equal(p.model.state_dict(), c.model.state_dict())
    equal(p.optimizer.state_dict(), c.optimizer.state_dict())
    equal(initial_rng, c._rng_state())
    path = tmp_path/'whole-child.pt'
    child_sha = c.save_checkpoint(path, sampler_state=sampler.state_dict())
    first = sampler.draw()
    assert first.example.context.source_bar_count == 40
    expected = c.step((first.example,))
    weights = {k:v.clone() for k,v in c.model.state_dict().items()}
    sampler.load_state_dict(c.load_checkpoint(path, expected_sha256=child_sha))
    replay = sampler.draw()
    assert replay.receipt() == first.receipt()
    assert c.step((replay.example,)) == expected
    equal(weights, c.model.state_dict())
    assert sha256(checkpoint.read_bytes()).hexdigest() == checksum
