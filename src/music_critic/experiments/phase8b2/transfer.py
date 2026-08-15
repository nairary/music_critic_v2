"""Downstream scratch/frozen/fine-tune transfer mechanics and evidence."""

from __future__ import annotations

from hashlib import sha256
from typing import Mapping

import torch
from torch import Tensor, nn

from music_critic.models import HierarchicalHeterogeneousBaseline
from music_critic.ssl.transfer import (
    ENCODER_EXPORT_CONTRACT_VERSION,
    EncoderTransferError,
    export_pretrained_encoder_state,
    load_pretrained_encoder_state,
)
from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_ARTIFACT_CONTRACT_VERSION,
    Phase8B2ContractError,
)


TRANSFER_PROTOCOL_VERSION = "1.0.0"


def tensor_state_fingerprint(state: Mapping[str, Tensor]) -> str:
    """Hash tensor values, names, shapes, and dtypes bit-exactly."""

    digest = sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encoder_export_fingerprint(export: object) -> str:
    if not isinstance(export, Mapping):
        raise Phase8B2ContractError(
            "phase8b2.transfer.export_invalid"
        )
    state = export.get("encoder_state")
    if not isinstance(state, Mapping) or any(
        not isinstance(value, Tensor) for value in state.values()
    ):
        raise Phase8B2ContractError(
            "phase8b2.transfer.export_state_invalid"
        )
    return tensor_state_fingerprint(state)


def transferred_encoder_state(
    model: HierarchicalHeterogeneousBaseline,
    parameter_names: tuple[str, ...],
) -> dict[str, Tensor]:
    state = model.state_dict()
    return {name: state[name] for name in parameter_names}


def prepare_downstream_model(
    model: nn.Module,
    *,
    transfer_mode: str,
    encoder_export: object | None,
) -> tuple[torch.optim.Optimizer | None, dict[str, object]]:
    """Apply only encoder state and establish optimizer membership evidence.

    The caller creates the supervised model (including fresh task heads) before
    this function.  The returned optimizer is deliberately ``None`` so the
    official training engine can construct it using its configured algorithm;
    evidence includes the exact names it must include.
    """

    if transfer_mode not in {
        "frozen_probe",
        "full_finetune",
        "supervised_scratch",
    }:
        raise Phase8B2ContractError(
            "phase8b2.transfer.mode_invalid"
        )
    if not isinstance(model, HierarchicalHeterogeneousBaseline):
        if transfer_mode != "supervised_scratch":
            raise Phase8B2ContractError(
                "phase8b2.transfer.pretrained_requires_hierarchical"
            )
        if encoder_export is not None:
            raise Phase8B2ContractError(
                "phase8b2.transfer.scratch_export_forbidden"
            )
        names = tuple(name for name, _ in model.named_parameters())
        return None, {
            "contract_version": TRANSFER_PROTOCOL_VERSION,
            "artifact_contract_version": (
                PHASE8B2_ARTIFACT_CONTRACT_VERSION
            ),
            "transfer_mode": transfer_mode,
            "encoder_export_contract_version": None,
            "loaded_parameter_names": [],
            "fresh_parameter_names": sorted(names),
            "optimizer_parameter_names": sorted(names),
            "encoder_frozen": False,
        }
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    if transfer_mode == "supervised_scratch":
        if encoder_export is not None:
            raise Phase8B2ContractError(
                "phase8b2.transfer.scratch_export_forbidden"
            )
        loaded_names: tuple[str, ...] = ()
        untouched_names = tuple(sorted(before))
        export_fingerprint = None
    else:
        if encoder_export is None:
            raise Phase8B2ContractError(
                "phase8b2.transfer.pretrained_export_required"
            )
        try:
            report = load_pretrained_encoder_state(model, encoder_export)
        except EncoderTransferError as exc:
            raise Phase8B2ContractError(str(exc)) from exc
        loaded_names = report.loaded_parameters
        untouched_names = report.untouched_parameters
        export_fingerprint = encoder_export_fingerprint(encoder_export)
    named_parameters = dict(model.named_parameters())
    loaded_parameter_names = tuple(
        name for name in loaded_names if name in named_parameters
    )
    if transfer_mode == "frozen_probe":
        for name in loaded_parameter_names:
            named_parameters[name].requires_grad_(False)
    optimizer_names = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if transfer_mode == "frozen_probe" and set(loaded_parameter_names) & set(
        optimizer_names
    ):
        raise Phase8B2ContractError(
            "phase8b2.transfer.frozen_encoder_in_optimizer"
        )
    if transfer_mode == "full_finetune" and not set(
        loaded_parameter_names
    ) <= set(optimizer_names):
        raise Phase8B2ContractError(
            "phase8b2.transfer.finetune_encoder_not_trainable"
        )
    after = model.state_dict()
    if any(
        not torch.equal(after[name], before[name])
        for name in untouched_names
    ):
        raise Phase8B2ContractError(
            "phase8b2.transfer.fresh_head_changed_during_load"
        )
    transferred_fingerprint = (
        None
        if not loaded_names
        else tensor_state_fingerprint(
            transferred_encoder_state(model, loaded_names)
        )
    )
    return None, {
        "contract_version": TRANSFER_PROTOCOL_VERSION,
        "artifact_contract_version": PHASE8B2_ARTIFACT_CONTRACT_VERSION,
        "transfer_mode": transfer_mode,
        "encoder_export_contract_version": (
            None
            if encoder_export is None
            else ENCODER_EXPORT_CONTRACT_VERSION
        ),
        "encoder_export_fingerprint": export_fingerprint,
        "transferred_encoder_fingerprint": transferred_fingerprint,
        "loaded_parameter_names": list(loaded_names),
        "loaded_trainable_parameter_names": list(loaded_parameter_names),
        "fresh_parameter_names": list(untouched_names),
        "optimizer_parameter_names": list(optimizer_names),
        "encoder_frozen": transfer_mode == "frozen_probe",
        "ssl_decoder_transferred": False,
        "ssl_optimizer_state_transferred": False,
        "task_heads_fresh": True,
    }


def verify_frozen_encoder(
    model: HierarchicalHeterogeneousBaseline,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    names_value = evidence.get("loaded_parameter_names")
    expected = evidence.get("transferred_encoder_fingerprint")
    if not isinstance(names_value, list) or not all(
        isinstance(name, str) for name in names_value
    ) or not isinstance(expected, str):
        raise Phase8B2ContractError(
            "phase8b2.transfer.frozen_evidence_invalid"
        )
    actual = tensor_state_fingerprint(
        transferred_encoder_state(model, tuple(names_value))
    )
    if actual != expected:
        raise Phase8B2ContractError(
            "phase8b2.transfer.frozen_encoder_changed"
        )
    return {
        "bit_exact": True,
        "before_fingerprint": expected,
        "after_fingerprint": actual,
    }


__all__ = [
    "TRANSFER_PROTOCOL_VERSION",
    "encoder_export_fingerprint",
    "export_pretrained_encoder_state",
    "prepare_downstream_model",
    "tensor_state_fingerprint",
    "transferred_encoder_state",
    "verify_frozen_encoder",
]
