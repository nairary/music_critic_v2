"""Synthetic checks for the explicitly admitted cloud C3 x3 relation-gate source."""
from pathlib import Path

import pytest
import torch

from music_critic.data import load_piece
from music_critic.critic.contracts import FeatureBinding
from music_critic.critic.features import BackboneFeatureAdapter
from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def piece():
    return load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')


def test_c3_source_is_explicit_and_preserves_same_pass_rich_feature_stages(piece, monkeypatch):
    from music_critic.critic.c3_backbone import (
        C3ThreePassRelationGateModel, C3_ADAPTER_VERSION, C3_SOURCE_ID,
        GATED_OUTPUT_SCHEMA, c3_relationgate_model_contract,
    )
    model = C3ThreePassRelationGateModel().eval()
    contract = c3_relationgate_model_contract(model)
    assert contract['rounds'] == 3 and contract['attention_heads'] == 4
    assert contract['relation_fusion'] == 'active_count_scaled_masked_softmax'
    assert [r['destinations'] for r in contract['layers']] == [
        ['note','onset','beat','bar','slice'], ['note','onset','beat','bar','slice'],
        ['note','beat','slice']]
    graph = build_temporal_batch((piece,))
    with torch.no_grad():
        historical = model(graph)
    assert historical.schema == GATED_OUTPUT_SCHEMA
    state_keys = tuple(model.state_dict())
    calls = []
    original = model.base.encode
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model.base, 'encode', counted)
    adapter = BackboneFeatureAdapter(model, FeatureBinding(C3_SOURCE_ID, 'synthetic-c3', C3_ADAPTER_VERSION))
    sample, = adapter.extract((piece,)).samples
    assert calls == [1] and tuple(model.state_dict()) == state_keys
    assert sample.stages['note'] == 'c3_relationgate_round_3'
    assert sample.stages['bar'] == sample.stages['onset'] == 'c3_relationgate_round_2'
    assert sample.stages['track'] == sample.stages['song'] == 'hierarchical_context'
    assert sample.stages['note_local'] == 'local_gnn'
    for level, values in historical.embeddings.items():
        torch.testing.assert_close(sample.embeddings[level], values, rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.parameters())


def test_loader_requires_artifact_source_architecture_and_weight_identity(tmp_path):
    from hashlib import sha256
    from music_critic.critic.c3_backbone import (
        C3ThreePassRelationGateModel, C3_SOURCE_ID, GATED_CHECKPOINT_SCHEMA,
        c3_relationgate_model_contract, load_c3_relationgate_adapter,
    )
    from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
    from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
    model = C3ThreePassRelationGateModel()
    payload = dict(schema_version=GATED_CHECKPOINT_SCHEMA,
        bindings={'source': C3_SOURCE_ID}, progress={'applied_updates':80000}, model_contract=c3_relationgate_model_contract(model),
        config={'model_kind':'c3_temporal_attention_3pass_relation_gate', 'temporal_width':128,
                'temporal_passes':3, 'relation_gate':True, 'attention_heads':4},
        model_state=model.state_dict(), model_state_fingerprint=model_state_fingerprint(model))
    payload['state_fingerprint'] = recursive_state_fingerprint(payload)
    path = tmp_path / 'synthetic.pt'
    torch.save(payload, path)
    pin = {'expected_sha256': sha256(path.read_bytes()).hexdigest(), 'expected_size_bytes':path.stat().st_size}
    rng = torch.get_rng_state().clone()
    adapter = load_c3_relationgate_adapter(path, **pin)
    assert torch.equal(rng, torch.get_rng_state())
    assert adapter.binding.model_state_id == model_state_fingerprint(model)
    assert adapter.binding.source_id == C3_SOURCE_ID
    assert adapter.binding.checkpoint_id == 'sha256:' + pin['expected_sha256']
    with pytest.raises(ValueError, match='artifact'):
        load_c3_relationgate_adapter(path, **{**pin,'expected_sha256':'0'*64})
    with pytest.raises(ValueError, match='source'):
        load_c3_relationgate_adapter(path, **pin, source_id='later-3090-changes')


def test_admitted_c3_contract_and_partition_fail_closed():
    from music_critic.critic.c3_backbone import (
        C3ThreePassRelationGateModel, C3_ADAPTER_VERSION, C3_SOURCE_ID, c3_relationgate_model_contract,
    )
    model = C3ThreePassRelationGateModel()
    # Independent golden from the manifest-bound archived cloud model contract.
    assert c3_relationgate_model_contract(model)['fingerprint'] == '5f391869cbd29161ad92c5b286e2f1a2b87b1aeaec7efbbf005eef620e7aa1a6'
    with pytest.raises(ValueError, match='source'):
        BackboneFeatureAdapter(model, FeatureBinding('other-source', 'synthetic', C3_ADAPTER_VERSION))
    model.temporal_layers[0].attention_heads = 8
    with pytest.raises(ValueError, match='architecture'):
        BackboneFeatureAdapter(model, FeatureBinding(C3_SOURCE_ID, 'synthetic', C3_ADAPTER_VERSION))


@pytest.mark.parametrize('change,match', [
    ('schema','schema'), ('source','source'), ('rounds','configuration'),
    ('contract','architecture'), ('update','update'), ('state','state integrity'),
    ('seal','envelope integrity'),
])
def test_loader_rejects_resealed_incompatible_checkpoint(tmp_path, change, match):
    from hashlib import sha256
    from music_critic.critic.c3_backbone import (
        C3ThreePassRelationGateModel, C3_SOURCE_ID, GATED_CHECKPOINT_SCHEMA,
        c3_relationgate_model_contract, load_c3_relationgate_adapter,
    )
    from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
    from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
    model = C3ThreePassRelationGateModel()
    payload = dict(schema_version=GATED_CHECKPOINT_SCHEMA,
        bindings={'source': C3_SOURCE_ID}, progress={'applied_updates':80000},
        model_contract=c3_relationgate_model_contract(model),
        config={'model_kind':'c3_temporal_attention_3pass_relation_gate', 'temporal_width':128,
                'temporal_passes':3, 'relation_gate':True, 'attention_heads':4},
        model_state=model.state_dict(), model_state_fingerprint=model_state_fingerprint(model))
    if change == 'schema': payload['schema_version'] = 'C3TemporalAttentionTrainingCheckpoint@1.0.0'
    elif change == 'source': payload['bindings']['source'] = '3090-changes'
    elif change == 'rounds': payload['config']['temporal_passes'] = 2
    elif change == 'contract': payload['model_contract']['relation_fusion'] = 'sum'
    elif change == 'update': payload['progress']['applied_updates'] = 60000
    elif change == 'state': next(iter(payload['model_state'].values())).add_(1)
    payload['state_fingerprint'] = recursive_state_fingerprint(payload)
    if change == 'seal': payload['state_fingerprint'] = '0' * 64
    path = tmp_path / 'synthetic-invalid.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        load_c3_relationgate_adapter(path, expected_sha256=sha256(path.read_bytes()).hexdigest(),
                                    expected_size_bytes=path.stat().st_size)


def test_c3_raw_batch_isolation_and_cache_roundtrip(piece, tmp_path):
    from dataclasses import replace
    from music_critic.critic.c3_backbone import C3ThreePassRelationGateModel, C3_SOURCE_ID, C3_ADAPTER_VERSION
    from music_critic.critic.cache import FrozenFeatureCache
    model = C3ThreePassRelationGateModel()
    adapter = BackboneFeatureAdapter(model, FeatureBinding(C3_SOURCE_ID, 'synthetic', C3_ADAPTER_VERSION))
    silent = replace(piece, notes=(), annotations=(), targets=())
    solo = adapter.extract((piece,))
    mixed = adapter.extract((silent, piece))
    for level, value in solo.samples[0].embeddings.items():
        torch.testing.assert_close(value, mixed.samples[1].embeddings[level], rtol=1e-5, atol=1e-5)
    cache = FrozenFeatureCache(tmp_path)
    key, = cache.put(solo)
    loaded = cache.get(key, solo.binding, solo.samples[0].raw_hash)
    assert loaded.stages == solo.samples[0].stages
    for level, value in solo.samples[0].embeddings.items():
        torch.testing.assert_close(value, loaded.embeddings[level], rtol=0, atol=0)
    model.temporal_layers[0].norms['note'].eps = 1e-3
    with pytest.raises(ValueError, match='architecture'):
        adapter.extract((piece,))
