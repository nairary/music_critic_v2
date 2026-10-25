"""Synthetic CPU checks for the differentiable PLL export, not model quality."""
from pathlib import Path

import pytest
import torch

from music_critic.data import load_piece
from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
from music_critic.pll.encoder import PLLContextEncoder


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def graph():
    piece = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    return build_temporal_batch((piece,))


def test_export_has_latest_states_and_never_evaluates_heads(graph, monkeypatch):
    encoder = PLLContextEncoder(freeze_policy="full").eval()
    def forbidden(*args, **kwargs):
        raise AssertionError("theory heads must not be evaluated")
    monkeypatch.setattr(encoder.backbone.base.task_heads, "forward", forbidden)
    values = encoder._encode_graph(graph)
    assert set(values) == {"note_local", "note", "onset", "beat", "bar", "track", "song", "slice"}
    assert values["note"].shape == (6, 128)
    assert values["song"].shape == (1, 128)
    assert all(v.requires_grad for v in values.values())
    sum(v.square().mean() for v in values.values() if v.numel()).backward()
    assert encoder.backbone.slice_projection.weight.grad is not None
    assert any(p.grad is not None for p in encoder.backbone.base.local_baseline.parameters())
    assert all(p.grad is None and not p.requires_grad for p in encoder.backbone.base.task_heads.parameters())


@pytest.mark.parametrize("policy", ["frozen", "upper", "full"])
def test_freezing_keeps_frozen_dropout_off_and_clear_gradient_boundaries(graph, policy):
    encoder = PLLContextEncoder(freeze_policy=policy).train()
    if policy == "frozen":
        assert not encoder.backbone.training
    if policy in {"frozen", "upper"}:
        assert not encoder.backbone.base.local_baseline.training
    assert not encoder.backbone.base.task_heads.training
    flags = torch.ones((6, 2))
    values = encoder._encode_graph(graph, flags)
    values["note"][0, 0].backward()
    assert encoder.boundary_projection.weight.grad is not None
    assert bool(encoder.boundary_projection.weight.grad.abs().sum())
    local_grad = any(p.grad is not None for p in encoder.backbone.base.local_baseline.parameters())
    temporal_grad = any(p.grad is not None for p in encoder.backbone.temporal_layers.parameters())
    assert local_grad == (policy == "full")
    assert temporal_grad == (policy != "frozen")
    encoder.set_freeze_policy("frozen")
    assert all(p.grad is None for p in encoder.backbone.parameters())


def test_empty_note_and_onset_graph_is_finite(graph):
    from dataclasses import replace
    piece = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    piece = replace(piece, notes=(), annotations=())
    empty_graph = build_temporal_batch((piece,))
    encoder = PLLContextEncoder(freeze_policy="full")
    values = encoder._encode_graph(empty_graph)
    assert values["note"].shape == values["onset"].shape == (0, 128)
    assert all(torch.isfinite(v).all() for v in values.values())
    values["bar"][0, 0].backward()
    assert any(p.grad is not None for p in encoder.backbone.base.context_encoder.parameters())


def test_pretrained_requires_exact_artifact_binding_before_loading(tmp_path):
    path = tmp_path / "invalid.ckpt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        PLLContextEncoder.from_pretrained(path, expected_sha256="0" * 64,
                                          expected_size_bytes=path.stat().st_size)


def test_public_encoder_handles_true_carry_without_artificial_onsets():
    from dataclasses import replace
    from music_critic.data import RationalTime as Q
    from music_critic.pll.contracts import ContextProfile, MaskQuery
    from music_critic.pll.masking import split_masked_piece
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    p = replace(p, notes=(replace(p.notes[0], onset_qn=Q(0), duration_qn=Q(20)),))
    profile = ContextProfile(core_bars=1, halo_bars=0)
    view = split_masked_piece(p, MaskQuery(Q(1), Q(5), (p.tracks[1].track_id,)), profile).context
    model = PLLContextEncoder(profile, freeze_policy="full")
    encoded = model(view)
    assert len(encoded) == 2
    assert encoded[1].embeddings["onset"].shape == (0, 128)
    assert encoded[1].embeddings["note"].shape == (1, 128)
    encoded[1].embeddings["note"][0, 0].backward()
    assert model.boundary_projection.weight.grad is not None
    assert bool(model.boundary_projection.weight.grad.abs().sum())


def test_hidden_content_cannot_change_public_encoder_values():
    from dataclasses import replace
    from music_critic.data import RationalTime as Q
    from music_critic.pll.contracts import MaskQuery
    from music_critic.pll.masking import split_masked_piece
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    q = MaskQuery(Q(1), Q(5), tuple(t.track_id for t in p.tracks))
    other = replace(p, notes=tuple(n for n in p.notes if n.onset_qn < Q(1)), annotations=(), targets=())
    model = PLLContextEncoder(freeze_policy="full").eval()
    a, b = [model(split_masked_piece(x, q).context) for x in (p, other)]
    for first, second in zip(a, b):
        assert first.graph.fingerprint() == second.graph.fingerprint()
        for level in first.embeddings:
            torch.testing.assert_close(first.embeddings[level], second.embeddings[level], rtol=0, atol=0)
