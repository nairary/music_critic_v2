"""Synthetic frozen cache contracts; no real checkpoint or corpus."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import load_piece
from music_critic.critic.contracts import FeatureBinding
from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel


@pytest.fixture
def pyramid():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    piece = load_piece(Path(__file__).resolve().parents[1] / 'fixtures/data/canonical_piece_v2.json')
    binding = FeatureBinding('main-synthetic', 'random-init-synthetic', C2_ADAPTER_VERSION)
    result = BackboneFeatureAdapter(CorrectedAnalysisGNNModel(), binding).extract((piece,))
    yield result
    torch.set_num_threads(previous)


def test_cache_roundtrip_is_bound_and_detached(pyramid, tmp_path):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    keys = cache.put(pyramid)
    assert len(keys) == 1
    original = pyramid.samples[0]
    restored = cache.get(keys[0], pyramid.binding, original.raw_hash)
    assert restored.geometry == original.geometry
    assert restored.stages == original.stages
    for level, value in original.embeddings.items():
        torch.testing.assert_close(restored.embeddings[level], value, rtol=0, atol=0)
        assert not restored.embeddings[level].requires_grad
    with pytest.raises(ValueError, match='binding'):
        cache.get(keys[0], replace(pyramid.binding, checkpoint_id='other'), original.raw_hash)
    with pytest.raises(ValueError, match='raw'):
        cache.get(keys[0], pyramid.binding, '0' * 64)


@pytest.mark.parametrize('field,value', [('source_id', 'other-source'), ('adapter_version', 'critic-temporal-features-v1'),
                                        ('model_state_id', 'a' * 64)])
def test_cache_rejects_other_bindings(pyramid, tmp_path, field, value):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    key, = cache.put(pyramid)
    with pytest.raises(ValueError, match='binding'):
        cache.get(key, replace(pyramid.binding, **{field: value}), pyramid.samples[0].raw_hash)


def test_cache_requires_weight_identity_frozen_shapes_and_known_stages(pyramid, tmp_path):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    with pytest.raises(ValueError, match='model state binding'):
        cache.put(replace(pyramid, binding=replace(pyramid.binding, model_state_id='')))
    original = pyramid.samples[0]
    for value in (original.embeddings['note'][:1], original.embeddings['note'].clone().requires_grad_(),
                  torch.full_like(original.embeddings['note'], float('nan'))):
        sample = replace(original, embeddings={**original.embeddings, 'note': value})
        with pytest.raises(ValueError, match='detached aligned'):
            cache.put(replace(pyramid, samples=(sample,)))
    sample = replace(original, stages={**original.stages, 'note': 'unknown'})
    with pytest.raises(ValueError, match='stage'):
        cache.put(replace(pyramid, samples=(sample,)))


def test_cache_detects_corruption_and_rejects_path_keys(pyramid, tmp_path):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    key, = cache.put(pyramid)
    path = tmp_path / (key + '.pt')
    payload = torch.load(path, weights_only=True)
    payload['embeddings']['note'][0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='content integrity'):
        cache.get(key, pyramid.binding, pyramid.samples[0].raw_hash)
    with pytest.raises(ValueError, match='invalid cache key'):
        cache.get('../escape', pyramid.binding, pyramid.samples[0].raw_hash)


def test_cache_separates_raw_variants_and_is_idempotent(pyramid, tmp_path):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    variant = replace(pyramid.samples[0], raw_hash='a' * 64)
    pair = replace(pyramid, samples=(pyramid.samples[0], variant))
    keys = cache.put(pair)
    assert len(set(keys)) == 2
    assert cache.put(pair) == keys
    assert len(list(tmp_path.glob('*.pt'))) == 2


def test_cache_accepts_fp32_roundoff_but_rejects_conflicting_features(pyramid, tmp_path):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    keys = cache.put(pyramid)
    original = pyramid.samples[0]
    rounded = replace(original, embeddings={k: v + 1e-6 for k, v in original.embeddings.items()})
    assert cache.put(replace(pyramid, samples=(rounded,))) == keys
    different = replace(original, embeddings={k: v + 0.1 for k, v in original.embeddings.items()})
    with pytest.raises(ValueError, match='conflicting'):
        cache.put(replace(pyramid, samples=(different,)))


def test_cache_load_inside_inference_mode_remains_usable_for_training(pyramid, tmp_path):
    from music_critic.critic.cache import FrozenFeatureCache
    cache = FrozenFeatureCache(tmp_path)
    key, = cache.put(pyramid)
    with torch.inference_mode():
        sample = cache.get(key, pyramid.binding, pyramid.samples[0].raw_hash)
    head = torch.nn.Linear(128, 1)
    head(sample.embeddings['note']).sum().backward()
    assert head.weight.grad is not None
