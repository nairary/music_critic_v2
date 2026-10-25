from dataclasses import replace
from pathlib import Path
import random
from hashlib import sha256

import pytest
import torch

from music_critic.data import RationalTime as R, load_piece
from music_critic.pll.contracts import CodecProfile, MaskQuery, ModelConfig
from music_critic.pll.masking import split_masked_piece
from music_critic.pll.model import HCM1
from music_critic.pll.runtime import PLLTrainer, TrainingConfig


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def trainer(data_id='synthetic'):
    config = ModelConfig(width=32, heads=4, event_layers=1, global_layers=1, ff_dim=64,
                         field_width=16, field_heads=4, field_ff_dim=32, relation_dim=8,
                         relation_components=2, dropout=.1)
    model = HCM1(config, codec_profile=CodecProfile(max_events=8,max_denominator=8,max_numerator=1024))
    return PLLTrainer(model, TrainingConfig(total_updates=3,warmup_updates=0,unfreeze_update=1),
                      data_id=data_id,protocol_id='synthetic-unit-protocol',source_id='synthetic-unit-source')


def example():
    p=load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json')
    return split_masked_piece(p,MaskQuery(R(1),R(5),(p.tracks[0].track_id,)))


def test_full_resume_matches_next_update_with_dropout_and_freeze_transition(tmp_path):
    torch.manual_seed(19); random.seed(19)
    a=trainer()
    a.step((example(),))
    path=tmp_path/'update-1.pt'
    sha=a.save_checkpoint(path,sampler_state={'cursor':1})
    expected=a.step((example(),))
    assert a.model.encoder.freeze_policy=='upper'
    b=trainer()
    state=b.load_checkpoint(path,expected_sha256=sha)
    assert state=={'cursor':1}
    actual=b.step((example(),))
    assert actual==expected
    for name,value in a.model.state_dict().items():
        torch.testing.assert_close(value,b.model.state_dict()[name],rtol=0,atol=0)
    assert b.updates==2 and b.draws==2
    with pytest.raises(FileExistsError):
        b.save_checkpoint(path,sampler_state={})


def test_resume_binding_and_hash_rejection_do_not_mutate_state(tmp_path):
    a=trainer(); path=tmp_path/'state.pt'; sha=a.save_checkpoint(path,sampler_state={})
    b=trainer(data_id='other-data')
    first=next(b.model.parameters()).detach().clone()
    with pytest.raises(ValueError,match='binding'):
        b.load_checkpoint(path,expected_sha256=sha)
    torch.testing.assert_close(first,next(b.model.parameters()),rtol=0,atol=0)
    with pytest.raises(ValueError,match='hash'):
        a.load_checkpoint(path,expected_sha256='0'*64)
    assert a.updates==0


def test_training_limits_and_microbatch_contract():
    a=trainer()
    with pytest.raises(ValueError,match='accumulation'):
        a.step(())
    with pytest.raises(ValueError):
        TrainingConfig(total_updates=0)


@pytest.mark.parametrize('damage', ['nan_moment', 'wrong_shape', 'negative_variance', 'lr', 'missing_state', 'empty_state', 'dropped_parameter'])
def test_corrupt_optimizer_is_rejected_before_application(tmp_path, damage):
    a=trainer(); a.step((example(),))
    source=tmp_path/'source.pt'; a.save_checkpoint(source,sampler_state={})
    state=torch.load(source,weights_only=True)
    first=next(iter(state['optimizer']['state'].values()))
    if damage=='nan_moment': first['exp_avg'].flatten()[0]=float('nan')
    elif damage=='wrong_shape': first['exp_avg']=torch.zeros(123)
    elif damage=='negative_variance': first['exp_avg_sq'].flatten()[0]=-1
    elif damage=='lr': state['optimizer']['param_groups'][0]['base_lr']=9.
    elif damage=='empty_state': state['optimizer']['state']={}
    elif damage=='dropped_parameter': state['optimizer']['state'].pop(next(iter(state['optimizer']['state'])))
    else: first.pop('exp_avg')
    bad=tmp_path/'bad.pt'; torch.save(state,bad)
    b=trainer(); prior=next(b.model.parameters()).detach().clone()
    with pytest.raises(ValueError,match='optimizer'):
        b.load_checkpoint(bad,expected_sha256=sha256(bad.read_bytes()).hexdigest())
    torch.testing.assert_close(prior,next(b.model.parameters()),rtol=0,atol=0)
    assert b.updates==0
