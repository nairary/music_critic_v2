from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.models import (
    ACTIVE_TASK_IDS,
    CheckpointContractError,
    HIERARCHICAL_CHECKPOINT_CONTRACT_VERSION,
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    hierarchical_checkpoint_metadata,
    hierarchical_single_note_sensitivity,
    load_hierarchical_checkpoint,
    perturb_canonical_note_pitch,
    save_hierarchical_checkpoint,
)
from tests.tasks.test_multisource_contract import _hook_piece


def _config() -> HierarchicalBaselineConfig:
    return HierarchicalBaselineConfig(
        hidden_dim=16,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=4,
        ffn_multiplier=2,
        dropout=0.0,
    )


def _recursive_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and set(left) == set(right)
            and all(_recursive_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_recursive_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _trained_pair(mixed_batch, seed: int):
    torch.manual_seed(seed)
    model = HierarchicalHeterogeneousBaseline(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    output = model(mixed_batch)
    assert output.harmonic_loss.total_loss is not None
    assert output.reconstruction_loss is not None
    (
        output.harmonic_loss.total_loss + output.reconstruction_loss
    ).backward()
    optimizer.step()
    return model, optimizer


def test_hierarchical_checkpoint_round_trip_is_bit_exact(
    mixed_batch, tmp_path: Path
) -> None:
    model, optimizer = _trained_pair(mixed_batch, 811)
    model.eval()
    before = model(mixed_batch, include_reconstruction=False)
    path = tmp_path / "phase6b.pt"
    save_hierarchical_checkpoint(
        path, model, optimizer=optimizer
    )
    assert not tuple(tmp_path.glob(".phase6b.pt.*.tmp"))
    restored = HierarchicalHeterogeneousBaseline(_config()).eval()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    metadata = load_hierarchical_checkpoint(
        path, restored, optimizer=restored_optimizer
    )
    after = restored(mixed_batch, include_reconstruction=False)
    assert metadata == hierarchical_checkpoint_metadata(model)
    assert (
        metadata["hierarchical_checkpoint_contract_version"]
        == HIERARCHICAL_CHECKPOINT_CONTRACT_VERSION
    )
    assert tuple(item.task_id for item in after.predictions) == ACTIVE_TASK_IDS
    assert all(
        torch.equal(left.logits, right.logits)
        for left, right in zip(before.predictions, after.predictions)
    )
    assert torch.equal(
        before.encoder.coarse.song_embeddings,
        after.encoder.coarse.song_embeddings,
    )
    assert all(
        torch.equal(
            before.encoder.fused.embeddings[node_type],
            after.encoder.fused.embeddings[node_type],
        )
        for node_type in before.encoder.fused.embeddings
    )


def test_hierarchical_checkpoint_metadata_mismatch_is_atomic(
    mixed_batch, tmp_path: Path
) -> None:
    source, source_optimizer = _trained_pair(mixed_batch, 821)
    path = tmp_path / "metadata.pt"
    save_hierarchical_checkpoint(
        path, source, optimizer=source_optimizer
    )
    incompatible = HierarchicalHeterogeneousBaseline(
        replace(_config(), transformer_layers=2)
    )
    optimizer = torch.optim.Adam(incompatible.parameters(), lr=0.01)
    model_snapshot = {
        key: value.detach().clone()
        for key, value in incompatible.state_dict().items()
    }
    optimizer_snapshot = deepcopy(optimizer.state_dict())
    with pytest.raises(
        CheckpointContractError, match="metadata is incompatible"
    ):
        load_hierarchical_checkpoint(
            path, incompatible, optimizer=optimizer
        )
    assert _recursive_equal(incompatible.state_dict(), model_snapshot)
    assert _recursive_equal(optimizer.state_dict(), optimizer_snapshot)


def test_hierarchical_checkpoint_application_failure_rolls_back(
    mixed_batch, tmp_path: Path, monkeypatch
) -> None:
    source, source_optimizer = _trained_pair(mixed_batch, 827)
    path = tmp_path / "application.pt"
    save_hierarchical_checkpoint(
        path, source, optimizer=source_optimizer
    )
    model, optimizer = _trained_pair(mixed_batch, 829)
    model_snapshot = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    optimizer_snapshot = deepcopy(optimizer.state_dict())
    original_load = optimizer.load_state_dict
    call_count = 0

    def fail_after_mutation(state_dict):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            optimizer.param_groups[0]["lr"] = 321.0
            first_state = next(iter(optimizer.state.values()))
            first_state["exp_avg"].add_(19.0)
            raise RuntimeError("injected Phase 6B application failure")
        return original_load(state_dict)

    monkeypatch.setattr(
        optimizer, "load_state_dict", fail_after_mutation
    )
    with pytest.raises(
        CheckpointContractError,
        match="state application failed atomically",
    ):
        load_hierarchical_checkpoint(
            path, model, optimizer=optimizer
        )
    assert call_count == 2
    assert _recursive_equal(model.state_dict(), model_snapshot)
    assert _recursive_equal(optimizer.state_dict(), optimizer_snapshot)


def test_hierarchical_checkpoint_save_failure_is_atomic(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "phase6b.pt"
    path.write_bytes(b"previous")
    model = HierarchicalHeterogeneousBaseline(_config())

    def fail_save(*_args, **_kwargs):
        raise OSError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="injected save failure"):
        save_hierarchical_checkpoint(path, model)
    assert path.read_bytes() == b"previous"
    assert not tuple(tmp_path.glob(".phase6b.pt.*.tmp"))


def test_hierarchical_single_note_reports_local_coarse_and_fused_levels() -> None:
    torch.manual_seed(839)
    model = HierarchicalHeterogeneousBaseline(_config()).eval()
    original = replace(_hook_piece(), annotations=(), targets=())
    note_id = original.notes[0].note_id
    perturbed = perturb_canonical_note_pitch(original, note_id)
    report = hierarchical_single_note_sensitivity(
        model,
        original,
        perturbed,
        note_id=note_id,
    )
    assert report.topology_equal
    assert report.ownership_equal
    assert report.cardinality_equal
    assert report.local_note_retained
    assert report.unrelated_sample_unchanged
    stages = {(item.stage, item.node_type) for item in report.deltas}
    assert {
        ("phase6a_local", "note"),
        ("pooled", "bar"),
        ("pooled", "track"),
        ("contextual", "bar"),
        ("contextual", "track"),
        ("contextual", "song"),
        ("fused", "note"),
        ("fused", "onset"),
        ("fused", "beat"),
        ("fused", "bar"),
        ("fused", "track"),
    } <= stages
    assert any(item.l2 > 0 for item in report.deltas)
    assert report.reconstruction_logit_l2_delta > 0
