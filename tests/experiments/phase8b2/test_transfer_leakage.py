from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from music_critic.experiments.phase8b2.contracts import Phase8B2ContractError
from music_critic.experiments.phase8b2.leakage import (
    target_mutation_evidence,
    validate_raw_only_ssl_inputs,
)
from music_critic.experiments.phase8b2.transfer import (
    prepare_downstream_model,
    verify_frozen_encoder,
)
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
)
from music_critic.ssl.transfer import export_pretrained_encoder_state


class _SSLWrapper(nn.Module):
    def __init__(self, encoder: HierarchicalHeterogeneousBaseline) -> None:
        super().__init__()
        self.encoder = encoder


def _model() -> HierarchicalHeterogeneousBaseline:
    return HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )


def test_frozen_probe_excludes_encoder_and_stays_bit_exact() -> None:
    torch.manual_seed(3)
    export = export_pretrained_encoder_state(_SSLWrapper(_model()))
    torch.manual_seed(7)
    model = _model()
    _, evidence = prepare_downstream_model(
        model,
        transfer_mode="frozen_probe",
        encoder_export=export,
    )
    loaded = set(evidence["loaded_trainable_parameter_names"])
    optimizer_names = set(evidence["optimizer_parameter_names"])
    assert loaded
    assert loaded.isdisjoint(optimizer_names)
    assert evidence["task_heads_fresh"]
    assert not evidence["ssl_decoder_transferred"]
    assert not evidence["ssl_optimizer_state_transferred"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    loss = sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward()
    optimizer.step()
    assert verify_frozen_encoder(model, evidence)["bit_exact"]


def test_full_finetune_encoder_is_trainable_and_heads_remain_fresh() -> None:
    export = export_pretrained_encoder_state(_SSLWrapper(_model()))
    model = _model()
    heads_before = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if "task_heads" in name
    }
    _, evidence = prepare_downstream_model(
        model,
        transfer_mode="full_finetune",
        encoder_export=export,
    )
    assert set(evidence["loaded_trainable_parameter_names"]) <= set(
        evidence["optimizer_parameter_names"]
    )
    for name, before in heads_before.items():
        assert torch.equal(model.state_dict()[name], before)


def test_scratch_rejects_ssl_export() -> None:
    export = export_pretrained_encoder_state(_SSLWrapper(_model()))
    with pytest.raises(
        Phase8B2ContractError, match="scratch_export_forbidden"
    ):
        prepare_downstream_model(
            _model(),
            transfer_mode="supervised_scratch",
            encoder_export=export,
        )


def test_raw_only_boundary_rejects_supervision() -> None:
    validate_raw_only_ssl_inputs({"raw_graph_batch": object()})
    with pytest.raises(
        Phase8B2ContractError, match="supervision_in_model_input"
    ):
        validate_raw_only_ssl_inputs(
            {"raw_graph_batch": object(), "target_provenance": object()}
        )


@pytest.mark.parametrize("mutation_kind", ("changed", "removed", "replaced"))
def test_target_sidecar_mutations_are_inert_through_transfer(
    mutation_kind: str,
) -> None:
    invariant = {
        "raw_input_fingerprint": "raw",
        "ssl_plan_fingerprint": "plan",
        "logits_fingerprint": "logits",
        "loss_fingerprint": "loss",
        "gradient_fingerprint": "gradient",
        "checkpoint_fingerprint": "checkpoint",
        "transferred_encoder_fingerprint": "encoder",
        "target_sidecar_fingerprint": "target-a",
    }
    mutated = deepcopy(invariant)
    mutated["target_sidecar_fingerprint"] = "target-b"
    evidence = target_mutation_evidence(
        invariant, mutated, mutation_kind=mutation_kind
    )
    assert evidence["passed"]
    assert evidence["transferred_encoders_equal"]


def test_target_mutation_mismatch_fails_closed() -> None:
    original = {name: name for name in (
        "raw_input_fingerprint",
        "ssl_plan_fingerprint",
        "logits_fingerprint",
        "loss_fingerprint",
        "gradient_fingerprint",
        "checkpoint_fingerprint",
        "transferred_encoder_fingerprint",
    )}
    original["target_sidecar_fingerprint"] = "a"
    mutated = deepcopy(original)
    mutated["target_sidecar_fingerprint"] = "b"
    mutated["gradient_fingerprint"] = "changed"
    with pytest.raises(
        Phase8B2ContractError, match="target_mutation_changed_ssl"
    ):
        target_mutation_evidence(original, mutated, mutation_kind="changed")
