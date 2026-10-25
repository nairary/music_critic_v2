"""Raw synthetic fixtures only; no corpus or checkpoint loading."""
from pathlib import Path
from fractions import Fraction

import pytest
import torch

from music_critic.data import load_piece
from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
from music_critic.experiments.analysisgnn.temporal_slices.model import TemporalSliceModel


@pytest.fixture
def piece():
    return load_piece(Path(__file__).resolve().parents[1] / "fixtures/data/canonical_piece_v2.json")


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_temporal_export_retains_base_from_one_pass_without_changing_old_output(piece, monkeypatch):
    model = TemporalSliceModel().eval()
    graph = build_temporal_batch((piece,))
    keys_before = tuple(model.state_dict())
    with torch.no_grad():
        expected = model(graph)
    calls = []
    encode = model.base.encode

    def counted(*args, **kwargs):
        calls.append(None)
        return encode(*args, **kwargs)

    monkeypatch.setattr(model.base, "encode", counted)
    with torch.no_grad():
        actual, base = model.forward_with_base(graph)
    assert len(calls) == 1
    assert tuple(model.state_dict()) == keys_before
    assert actual.candidate_keys == expected.candidate_keys
    for task in expected.logits:
        torch.testing.assert_close(actual.logits[task], expected.logits[task], rtol=0, atol=0)
    assert set(base.fused.embeddings) == {"note", "onset", "beat", "bar", "track", "song"}
    assert base.local_encoder.final_output.embeddings["note"].shape[0] == len(piece.notes)


def test_geometry_retains_raw_exact_times_and_sounding_interval_boundaries(piece):
    from music_critic.critic.features import geometry_from_piece
    from music_critic.critic.contracts import local_candidates
    geometry = geometry_from_piece(piece)
    assert geometry.sample_id == piece.piece_id
    assert geometry.duration == Fraction(5)
    assert tuple(n.note_id for n in geometry.notes) == tuple(n.note_id for n in piece.notes)
    assert all(isinstance(n.start, Fraction) for n in geometry.notes)
    slices = geometry.intervals["slice"]
    assert slices[-1].start == slices[-1].end == Fraction(5)
    assert all(a.end == b.start for a, b in zip(slices, slices[1:]))
    assert len(local_candidates(geometry)["harmony/slice"]) == len(slices)

@pytest.mark.parametrize('temporal', [False, True])
def test_adapter_exports_detached_aligned_latest_states_and_freezes_backbone(piece, temporal, monkeypatch):
    from music_critic.critic.features import (BackboneFeatureAdapter,
        C2_ADAPTER_VERSION, TEMPORAL_ADAPTER_VERSION)
    from music_critic.critic.contracts import FeatureBinding
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    model = TemporalSliceModel() if temporal else CorrectedAnalysisGNNModel()
    base = model.base if temporal else model
    calls = []
    encode = base.encode
    def counted(*args, **kwargs):
        calls.append(None)
        return encode(*args, **kwargs)
    monkeypatch.setattr(base, 'encode', counted)
    binding = FeatureBinding('synthetic-main', 'synthetic-random-init',
        TEMPORAL_ADAPTER_VERSION if temporal else C2_ADAPTER_VERSION)
    adapter = BackboneFeatureAdapter(model, binding)
    pyramid = adapter.extract((piece, piece))
    assert len(calls) == 1
    assert len(pyramid.samples) == 2
    assert len(pyramid.binding.model_state_id) == 64
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for sample in pyramid.samples:
        assert set(sample.embeddings) == {'note_local', 'note', 'onset', 'beat', 'bar', 'track', 'song', 'slice'}
        assert sample.embeddings['note_local'].shape == (6, 128)
        assert sample.embeddings['track'].shape == (2, 128)
        assert sample.embeddings['song'].shape == (1, 128)
        assert sample.embeddings['slice'].shape == (len(sample.geometry.intervals['slice']), 128)
        assert all(not v.requires_grad and v.grad_fn is None for v in sample.embeddings.values())
        assert set(sample.stages) == set(sample.embeddings)
    torch.testing.assert_close(pyramid.samples[0].embeddings['note'], pyramid.samples[1].embeddings['note'])
    if temporal:
        assert pyramid.samples[0].stages['bar'] == 'temporal_round_1'
        assert pyramid.samples[0].stages['note'] == 'temporal_round_2'
        assert pyramid.samples[0].stages['song'] == 'hierarchical_context'
    else:
        assert pyramid.samples[0].stages['slice'] == 'derived_sounding_note_mean'
        assert not pyramid.samples[0].embeddings['slice'][-1].any()


def test_geometry_marks_pickup_incomplete_and_rejects_noncanonical_row_order(piece):
    from dataclasses import replace
    from music_critic.critic.features import geometry_from_piece
    geometry = geometry_from_piece(piece)
    assert [b.complete for b in geometry.intervals['bar']] == [False, True]
    with pytest.raises(ValueError, match='invalid canonical'):
        geometry_from_piece(replace(piece, notes=tuple(reversed(piece.notes))))
    with pytest.raises(ValueError, match='invalid canonical'):
        geometry_from_piece(replace(piece, tracks=tuple(reversed(piece.tracks))))


def test_adapter_enforces_float32_even_under_callers_autocast(piece):
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
    from music_critic.critic.contracts import FeatureBinding
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    adapter = BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
        FeatureBinding('main-synthetic', 'random-init-synthetic', C2_ADAPTER_VERSION))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        pyramid = adapter.extract((piece,))
    assert all(v.dtype == torch.float32 for v in pyramid.samples[0].embeddings.values())


def test_adapter_rejects_wrong_source_type_and_changed_weights(piece):
    from dataclasses import replace
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION, TEMPORAL_ADAPTER_VERSION
    from music_critic.critic.contracts import FeatureBinding
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    model = CorrectedAnalysisGNNModel()
    binding = FeatureBinding('main-synthetic', 'random-init-synthetic', C2_ADAPTER_VERSION)
    with pytest.raises(ValueError, match='unsupported'):
        BackboneFeatureAdapter(model, replace(binding, adapter_version=TEMPORAL_ADAPTER_VERSION))
    with pytest.raises(ValueError, match='state binding'):
        BackboneFeatureAdapter(model, replace(binding, model_state_id='0' * 64))
    class IsolatedVariant(CorrectedAnalysisGNNModel):
        pass
    with pytest.raises(ValueError, match='unsupported'):
        BackboneFeatureAdapter(IsolatedVariant(), binding)
    adapter = BackboneFeatureAdapter(model, binding).to('cpu')
    with pytest.raises(TypeError, match='canonical raw'):
        adapter.extract([piece])
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(ValueError, match='weights changed'):
        adapter.extract((piece,))


def test_raw_targets_do_not_change_features_and_raw_variant_has_independent_hash(piece):
    from dataclasses import replace
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
    from music_critic.critic.contracts import FeatureBinding
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    adapter = BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
        FeatureBinding('main-synthetic', 'random-init-synthetic', C2_ADAPTER_VERSION))
    unlabeled = replace(piece, annotations=(), targets=())
    variant = replace(unlabeled, notes=(replace(piece.notes[0], pitch=68),) + piece.notes[1:])
    original, no_targets, edited = adapter.extract((piece, unlabeled, variant)).samples
    assert original.raw_hash == no_targets.raw_hash != edited.raw_hash
    for level in original.embeddings:
        torch.testing.assert_close(original.embeddings[level], no_targets.embeddings[level], rtol=0, atol=0)
    alone = adapter.extract((piece,)).samples[0]
    for level in original.embeddings:
        torch.testing.assert_close(original.embeddings[level], alone.embeddings[level], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('temporal', [False, True])
def test_empty_raw_piece_does_not_change_another_samples_features(piece, temporal):
    from dataclasses import replace
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION, TEMPORAL_ADAPTER_VERSION
    from music_critic.critic.contracts import FeatureBinding
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    adapter = BackboneFeatureAdapter(TemporalSliceModel() if temporal else CorrectedAnalysisGNNModel(),
        FeatureBinding('main-synthetic', 'random-init-synthetic',
                       TEMPORAL_ADAPTER_VERSION if temporal else C2_ADAPTER_VERSION))
    silent = replace(piece, notes=(), targets=(), annotations=())
    solo = adapter.extract((piece,)).samples[0]
    empty, mixed = adapter.extract((silent, piece)).samples
    assert empty.embeddings['note'].shape == (0, 128)
    if not temporal:
        assert not empty.embeddings['slice'].any()
    for level in solo.embeddings:
        torch.testing.assert_close(solo.embeddings[level], mixed.embeddings[level], rtol=1e-5, atol=1e-5)


def test_export_inside_inference_mode_remains_usable_for_training_new_heads(piece):
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
    from music_critic.critic.contracts import FeatureBinding
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    adapter = BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
        FeatureBinding('main-synthetic', 'random-init-synthetic', C2_ADAPTER_VERSION))
    with torch.inference_mode():
        sample = adapter.extract((piece,)).samples[0]
    head = torch.nn.Linear(128, 1)
    head(sample.embeddings['note']).sum().backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert not any(v.requires_grad for v in sample.embeddings.values())
