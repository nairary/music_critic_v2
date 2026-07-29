from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
)
from music_critic.ssl.transfer import (
    EncoderTransferError,
    export_pretrained_encoder_state,
    load_pretrained_encoder_state,
)


class _SSLWrapper(nn.Module):
    def __init__(self, encoder: HierarchicalHeterogeneousBaseline) -> None:
        super().__init__()
        self.encoder = encoder


def _model(*, task_weights=()) -> HierarchicalHeterogeneousBaseline:
    return HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
            task_weights=task_weights,
        )
    )


def test_encoder_transfer_is_strict_and_preserves_supervised_heads() -> None:
    torch.manual_seed(3)
    source = _SSLWrapper(_model())
    with torch.no_grad():
        for name, parameter in source.encoder.named_parameters():
            if name.startswith(
                (
                    "local_baseline.encoder.",
                    "context_encoder.pooling.",
                    "context_encoder.transformer.",
                    "context_encoder.fusion.",
                )
            ):
                parameter.add_(0.25)
    exported = export_pretrained_encoder_state(source)

    torch.manual_seed(11)
    supervised = _model(task_weights=(("theory.local_key.tonic_pc", 2.0),))
    before = deepcopy(supervised.state_dict())
    report = load_pretrained_encoder_state(supervised, exported)
    after = supervised.state_dict()

    assert report.loaded_parameters
    assert report.untouched_parameters
    assert all(
        name.startswith(
            (
                "local_baseline.encoder.",
                "context_encoder.pooling.",
                "context_encoder.transformer.",
                "context_encoder.fusion.",
            )
        )
        for name in report.loaded_parameters
    )
    exported_state = exported["encoder_state"]
    assert isinstance(exported_state, dict)
    for name in report.loaded_parameters:
        assert torch.equal(after[name].cpu(), exported_state[name])
    for name in report.untouched_parameters:
        assert torch.equal(after[name], before[name])


@pytest.mark.parametrize("mutation", ("missing", "unexpected"))
def test_encoder_transfer_rejects_missing_or_unexpected_keys_atomically(
    mutation: str,
) -> None:
    source = _SSLWrapper(_model())
    exported = export_pretrained_encoder_state(source)
    corrupted = deepcopy(exported)
    state = corrupted["encoder_state"]
    assert isinstance(state, dict)
    if mutation == "missing":
        state.pop(next(iter(state)))
    else:
        state["unexpected.parameter"] = torch.zeros(1)
    supervised = _model()
    before = deepcopy(supervised.state_dict())

    with pytest.raises(
        EncoderTransferError,
        match="state_keys_incompatible",
    ):
        load_pretrained_encoder_state(supervised, corrupted)

    for name, value in supervised.state_dict().items():
        assert torch.equal(value, before[name])
