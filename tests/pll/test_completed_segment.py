import json
from hashlib import sha256
from pathlib import Path
import pytest
import torch
from scripts.hcm1_completed_segment import restore_segment
from scripts.hcm1_stage2 import continue_full_state
from tests.pll.test_stage2 import parent,equal
from tests.pll.test_runtime import example


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(tmp_path):
    p=parent();code=sha256(Path('scripts/hcm1_stage2.py').read_bytes()).hexdigest()
    c=continue_full_state(p,additional_updates=4,protocol_id='child',parent_sha256='a'*64,implementation_sha256=code)
    for _ in range(4):c.step((example(),))
    cp=tmp_path/'child.pt';digest=c.save_checkpoint(cp,sampler_state={'draws':7})
    bp=tmp_path/'binding.json';bp.write_text(json.dumps(c.binding))
    record=dict(status='completed',results={'global_updates':7},hcm_stage2=dict(parent_checkpoint_sha256='a'*64,
        additional_updates=4,protocol_id='child',implementation_sha256=code),
        artifacts=[dict(sha256=sha256(x.read_bytes()).hexdigest(),size_bytes=x.stat().st_size) for x in (cp,bp)])
    return p,c,cp,bp,record


def test_completed_segment_restores_exact_weights_and_sampler(tmp_path):
    p,c,cp,bp,r=fixture(tmp_path)
    actual,state=restore_segment(p,cp,bp,r,parent_sha256='a'*64)
    assert actual.updates==7 and state=={'draws':7}
    equal(c.model.state_dict(),actual.model.state_dict())
    equal(c.optimizer.state_dict(),actual.optimizer.state_dict())


@pytest.mark.parametrize('bad',['unfinished','hash','parent','code','progress','binding'])
def test_mismatched_segment_cannot_be_evaluated(tmp_path,bad):
    p,c,cp,bp,r=fixture(tmp_path)
    if bad=='unfinished':r['status']='running'
    elif bad=='hash':cp.write_bytes(cp.read_bytes()+b'x')
    elif bad=='parent':r['hcm_stage2']['parent_checkpoint_sha256']='c'*64
    elif bad=='code':r['hcm_stage2']['implementation_sha256']='d'*64
    elif bad=='progress':r['results']['global_updates']=6
    else:
        binding=json.loads(bp.read_text());binding['protocol_id']='wrong';bp.write_text(json.dumps(binding))
        r['artifacts'][1]=dict(sha256=sha256(bp.read_bytes()).hexdigest(),size_bytes=bp.stat().st_size)
    with pytest.raises(ValueError):restore_segment(p,cp,bp,r,parent_sha256='a'*64)


def test_checkpoint_replacement_after_record_verification_is_rejected(tmp_path,monkeypatch):
    p,c,cp,bp,r=fixture(tmp_path);original=Path.read_bytes;calls=0
    def changed(path):
        nonlocal calls
        raw=original(path)
        if path==cp:
            calls+=1
            if calls>1:return raw+b'replacement'
        return raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(ValueError,match='hash'):restore_segment(p,cp,bp,r,parent_sha256='a'*64)
