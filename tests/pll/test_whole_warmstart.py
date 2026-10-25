"""Weight-only migration uses synthetic checkpoints, never trained corpora."""
from hashlib import sha256
import random

import pytest
import torch

from music_critic.pll.model import HCM1
from music_critic.pll.runtime import PLLTrainer, TrainingConfig
from music_critic.pll.whole_model import WholePieceHCM
from tests.pll.whole_fixtures import small_codec, small_config, small_whole


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def models_and_checkpoint(tmp_path):
    old = HCM1(small_config(), codec_profile=small_codec())
    trainer = PLLTrainer(old, TrainingConfig(total_updates=2, warmup_updates=0),
                         data_id='synthetic', protocol_id='synthetic', source_id='fixture')
    path = tmp_path / 'source.pt'
    digest = trainer.save_checkpoint(path, sampler_state={'fixture': True})
    new = WholePieceHCM(small_config(), codec_profile=small_codec(), whole_config=small_whole())
    return old, new, path, digest


def test_warmstart_reuses_shared_weights_only_and_keeps_rng(tmp_path):
    from music_critic.pll.whole_warmstart import warm_start_hcm1
    old, new, path, digest = models_and_checkpoint(tmp_path)
    original = {k: v.clone() for k, v in new.state_dict().items()}
    rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    receipt = warm_start_hcm1(new, path, expected_sha256=digest,
        expected_size_bytes=path.stat().st_size, expected_update=0)
    shared = set(old.state_dict()) & set(original)
    shared = {k for k in shared if k.startswith(('encoder.', 'decoder.'))}
    for name, value in new.state_dict().items():
        expected = old.state_dict()[name] if name in shared else original[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert random.getstate() == py_rng
    assert receipt['copied_tensors'] == len(shared)
    assert receipt['optimizer_restored'] is False
    assert receipt['progress_restored'] is False
    assert new.warm_start_binding == receipt
    assert digest in new.encoder.weights_origin
    fresh = PLLTrainer(new, TrainingConfig(total_updates=2, warmup_updates=0),
                       data_id='whole-synthetic', protocol_id='whole-synthetic', source_id='fixture')
    assert fresh.updates == fresh.draws == 0
    assert fresh.optimizer.state_dict()['state'] == {}


@pytest.mark.parametrize('damage', ['hash', 'size', 'update', 'contract', 'codec',
                                   'missing', 'extra', 'shape', 'dtype', 'nan'])
def test_invalid_source_is_rejected_before_any_model_mutation(tmp_path, damage):
    from music_critic.pll.whole_warmstart import warm_start_hcm1
    _, new, path, digest = models_and_checkpoint(tmp_path)
    if damage not in ('hash', 'size'):
        payload = torch.load(path, weights_only=True)
        key = 'decoder.output.weight'
        if damage == 'update':
            payload['updates'] = 1
        elif damage == 'contract':
            payload['binding']['model']['model']['relations'] = 'none'
        elif damage == 'codec':
            payload['binding']['model']['codec']['max_events'] += 1
        elif damage == 'missing':
            payload['model'].pop(key)
        elif damage == 'extra':
            payload['model']['decoder.unknown.weight'] = torch.zeros(1)
        elif damage == 'shape':
            payload['model'][key] = payload['model'][key][:1]
        elif damage == 'dtype':
            payload['model'][key] = payload['model'][key].double()
        elif damage == 'nan':
            payload['model'][key][0, 0] = float('nan')
        torch.save(payload, path)
        digest = sha256(path.read_bytes()).hexdigest()
    before = {k: v.clone() for k, v in new.state_dict().items()}
    origin = new.encoder.weights_origin
    with pytest.raises(ValueError, match='warmstart'):
        warm_start_hcm1(new, path, expected_sha256='0' * 64 if damage == 'hash' else digest,
            expected_size_bytes=path.stat().st_size + (1 if damage == 'size' else 0), expected_update=0)
    for key, expected in before.items():
        torch.testing.assert_close(new.state_dict()[key], expected, rtol=0, atol=0)
    assert new.warm_start_binding is None and new.encoder.weights_origin == origin


def test_source_optimizer_rng_and_old_memory_are_explicitly_ignored(tmp_path):
    from music_critic.pll.whole_warmstart import warm_start_hcm1
    _, model, path, _ = models_and_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload['optimizer'] = {'unusable': 'discarded'}
    payload['rng'] = {'unusable': 'discarded'}
    payload['sampler_state'] = {'unusable': 'discarded'}
    payload['draws'] = 100000
    payload['model']['memory.discarded_new_format'] = 'deliberately not a target parameter'
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    receipt = warm_start_hcm1(model, path, expected_sha256=digest,
        expected_size_bytes=path.stat().st_size, expected_update=0)
    assert receipt['rng_restored'] is False
    with pytest.raises(ValueError, match='already_applied'):
        warm_start_hcm1(model, path, expected_sha256=digest,
            expected_size_bytes=path.stat().st_size, expected_update=0)
