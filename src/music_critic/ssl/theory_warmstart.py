"""Strict, in-memory corrected-theory to SSL bridge (not checkpoint export).

The four complete backbone groups map ``k -> encoder.k``. The full corrected
state is carried explicitly so that the separate BiGRU/18-head branch can be
retained. The Phase 7 encoder-only export contract is deliberately unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn

from music_critic.experiments.analysisgnn.contracts import canonical_json, fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNConfig,
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    model_state_fingerprint,
)
from music_critic.graph import GRAPH_BUILDER_VERSION, GRAPH_SCHEMA_VERSION
from music_critic.models.checkpoint import feature_registry_fingerprint
from music_critic.ssl.model import MaskedGraphSSLConfig, MaskedGraphSSLModel


THEORY_WARMSTART_VERSION = "theory-ssl-warmstart@1.0.0"
BACKBONE_GROUPS = (
    "local_baseline.encoder",
    "context_encoder.pooling",
    "context_encoder.transformer",
    "context_encoder.fusion",
)
RETAINED_GROUPS = ("sequence_decoder", "task_heads")
FRESH_GROUPS = (
    "feature_mask_token", "decoder", "bar_projector_predictor",
    "song_projector_predictor",
)
INACTIVE_GROUPS = (
    "encoder.local_baseline.task_heads",
    "encoder.local_baseline.reconstruction_heads",
)


class TheoryWarmstartError(ValueError):
    """A failure before application, or a rolled-back application failure."""


class TheoryWarmstartSSLModel(MaskedGraphSSLModel):
    """Existing SSL objectives at the fixed corrected architecture.

    Preserve heterogeneous child modes around the existing detached target pass.
    Restoration happens before online encoding, including on target failure.
    """

    def __init__(self, ssl_config: MaskedGraphSSLConfig = MaskedGraphSSLConfig()):
        super().__init__(CorrectedAnalysisGNNConfig().hierarchy_config(), ssl_config)

    def _full_view_targets(self, graph: object, *, prepared_input_token: object):
        modes = tuple((module, module.training) for module in self.encoder.modules())
        try:
            return super()._full_view_targets(
                graph, prepared_input_token=prepared_input_token
            )
        finally:
            for module, training in modes:
                module.training = training


@dataclass(frozen=True)
class TheoryWarmstartReport:
    contract_version: str
    provenance: str
    source_state_fingerprint: str
    contract_fingerprint: str
    transferred: Mapping[str, object]
    retained: Mapping[str, object]
    fresh: Mapping[str, object]
    inactive: Mapping[str, object]


@dataclass(frozen=True)
class TheoryWarmstartResult:
    ssl_model: TheoryWarmstartSSLModel
    theory_model: CorrectedAnalysisGNNModel
    report: TheoryWarmstartReport
    artifact_evidence: Mapping[str, object] | None = None


def _belongs(name: str, groups: tuple[str, ...]) -> bool:
    return any(name == group or name.startswith(group + ".") for group in groups)


def _canonical_equal(actual: object, expected: object) -> bool:
    # Python equality conflates integer, float, and boolean values. These are
    # distinct JSON contract values even when their numeric values compare equal.
    try:
        return canonical_json(actual) == canonical_json(expected)
    except (TypeError, ValueError, RecursionError):
        return False


def _valid_export_contract(actual: object, expected: Mapping[str, object]) -> bool:
    if not isinstance(actual, Mapping):
        return False
    try:
        unhashed = dict(actual)
        supplied_fingerprint = unhashed.pop("fingerprint")
        if (
            type(supplied_fingerprint) is not str
            or fingerprint(unhashed) != supplied_fingerprint
        ):
            return False
        return _canonical_equal(dict(actual), expected)
    except (KeyError, TypeError, ValueError, RecursionError):
        return False


def _architecture(model: nn.Module) -> dict[str, object]:
    # Tensor shapes alone cannot distinguish attention heads, dropout, residual
    # policy, activation, head routes, or per-node feature specs. Bind actual
    # public runtime configuration, excluding the independently managed modes.
    return {
        name: {
            "type": type(module).__module__ + "." + type(module).__qualname__,
            "repr": module.extra_repr(),
            "attributes": {
                key: _configuration_value(value)
                for key, value in vars(module).items()
                if not key.startswith("_") and key != "training"
            },
        }
        for name, module in model.named_modules()
    }


def _configuration_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {"dataclass": type(value).__module__ + "." + type(value).__qualname__,
                "value": _configuration_value(asdict(value))}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_configuration_value(row) for row in value]}
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _configuration_value(row) for key, row in value.items()}
    if callable(value) and hasattr(value, "__qualname__"):
        return {"callable": value.__module__ + "." + value.__qualname__}
    raise TheoryWarmstartError("ws2.runtime_configuration_unsupported")


def _validate_parameter_identity(model: nn.Module) -> None:
    parameters = tuple(model.named_parameters(remove_duplicate=False))
    if len(parameters) != len({id(parameter) for _, parameter in parameters}):
        raise TheoryWarmstartError("ws2.unexpected_parameter_alias")
    allocations: dict[torch.device, list[tuple[int, int]]] = {}
    for _, parameter in parameters:
        # Invalid tensor layouts/devices retain the structured tensor rejection
        # in _validate_state; they must not reach storage-specific operations.
        if parameter.layout != torch.strided or parameter.device.type == "meta":
            continue
        storage = parameter.untyped_storage()
        if not storage.nbytes():
            continue
        # Check whole storage ranges, since offset tensor views and separate
        # frombuffer storage wrappers can overlap despite distinct data_ptrs.
        # Both inputs require independent allocations before any mutation.
        start = storage.data_ptr()
        allocations.setdefault(parameter.device, []).append((start, start + storage.nbytes()))
    for ranges in allocations.values():
        previous_end = None
        for start, end in sorted(ranges):
            if previous_end is not None and start < previous_end:
                raise TheoryWarmstartError("ws2.unexpected_parameter_storage_alias")
            previous_end = end


def _contract(reference: CorrectedAnalysisGNNModel) -> dict[str, object]:
    payload = {
        "version": THEORY_WARMSTART_VERSION,
        "model": corrected_model_contract(reference),
        "feature_registry_fingerprint": feature_registry_fingerprint(),
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_builder_version": GRAPH_BUILDER_VERSION,
        "architecture": _architecture(reference),
        "backbone_mapping": {group: "encoder." + group for group in BACKBONE_GROUPS},
        "retained_groups": list(RETAINED_GROUPS),
        "fresh_groups": list(FRESH_GROUPS),
        "inactive_groups": list(INACTIVE_GROUPS),
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _reference() -> CorrectedAnalysisGNNModel:
    # Validation must not consume the caller's initialization/dropout RNG.
    with torch.random.fork_rng(devices=[]):
        return CorrectedAnalysisGNNModel()


def _validate_state(state: object, expected: Mapping[str, Tensor], label: str):
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise TheoryWarmstartError(f"ws2.{label}.state_keys_incompatible")
    for name, template in expected.items():
        value = state[name]
        if (
            not isinstance(value, Tensor) or value.layout != torch.strided
            or value.device.type == "meta" or value.shape != template.shape
            or value.dtype != template.dtype
        ):
            raise TheoryWarmstartError(f"ws2.{label}.tensor_incompatible:{name}")
        if not bool(torch.isfinite(value).all()):
            raise TheoryWarmstartError(f"ws2.{label}.nonfinite:{name}")
    return state


def export_theory_warmstart_state(source: CorrectedAnalysisGNNModel) -> dict[str, object]:
    """Clone a complete synthetic in-memory state; never label it encoder-only.

    This API makes no claims about checkpoint files or training provenance.
    A registered experiment needs its own checkpoint provenance contract.
    """
    return _export_state(source, provenance="synthetic_in_memory")


def _export_state(source: CorrectedAnalysisGNNModel, *, provenance: str):
    reference = _reference()
    if type(source) is not CorrectedAnalysisGNNModel or (
        not _canonical_equal(
            corrected_model_contract(source), corrected_model_contract(reference)
        )
        or not _canonical_equal(_architecture(source), _architecture(reference))
    ):
        raise TheoryWarmstartError("ws2.source.model_contract_incompatible")
    _validate_parameter_identity(source)
    state = _validate_state(source.state_dict(), reference.state_dict(), "source")
    cloned = {name: value.detach().cpu().clone() for name, value in state.items()}
    return {
        "contract": _contract(reference),
        "provenance": provenance,
        "corrected_state": cloned,
        "state_fingerprint": model_state_fingerprint(cloned),
    }


def _group_report(state: Mapping[str, Tensor], groups: tuple[str, ...]):
    report = {}
    for group in groups:
        selected = {key: value for key, value in state.items() if _belongs(key, (group,))}
        if not selected:
            raise TheoryWarmstartError(f"ws2.empty_group:{group}")
        report[group] = {
            "keys": tuple(sorted(selected)),
            "fingerprint": model_state_fingerprint(selected),
            "numel": sum(value.numel() for value in selected.values()),
        }
    return report


def load_theory_warmstart(
    destination: TheoryWarmstartSSLModel, export: object,
) -> TheoryWarmstartResult:
    """Validate everything, atomically load the backbone, retain and share it.

    The destination is an explicitly fresh WS2 SSL container. All destination
    state (including fresh and inactive groups) is attested before mutation.
    No keys are filtered out of validation and no partial load is permitted.
    """
    return _load_state(destination, export, provenance="synthetic_in_memory")


def load_model_only_theory_warmstart(
    destination: TheoryWarmstartSSLModel, path: object, *,
    expected_sha256: str, expected_size_bytes: int,
    expected_state_fingerprint: str,
) -> tuple[TheoryWarmstartResult, dict[str, object]]:
    """Separate artifact entrypoint; the old synthetic envelope stays strict.

    The existing restricted loader binds bytes, complete corrected contract and
    every tensor. An external experiment/source binding must establish that this
    artifact is the selected C2; a model-only payload cannot prove its history.
    No optimizer or historical RNG is loaded. The returned evidence describes
    actual artifact provenance, never synthetic provenance.
    """
    from music_critic.experiments.pop909_cl_frozen.checkpoint import (
        MODEL_ONLY_CHECKPOINT_SCHEMA, load_model_checkpoint,
    )

    loaded = load_model_checkpoint(
        path, expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    if loaded.evidence["checkpoint_schema"] != MODEL_ONLY_CHECKPOINT_SCHEMA:
        raise TheoryWarmstartError("ws2.model_only_schema_required")
    if loaded.evidence["model_state_fingerprint"] != expected_state_fingerprint:
        raise TheoryWarmstartError("ws2.model_only_state_binding_mismatch")
    provenance = "verified_model_only_artifact"
    result = _load_state(
        destination, _export_state(loaded.model, provenance=provenance),
        provenance=provenance,
    )
    result = replace(result, artifact_evidence=dict(loaded.evidence))
    return result, dict(loaded.evidence)


def _load_state(destination, export, *, provenance: str) -> TheoryWarmstartResult:
    reference = _reference()
    contract = _contract(reference)
    if not isinstance(export, Mapping) or set(export) != {
        "contract", "provenance", "corrected_state", "state_fingerprint"
    }:
        raise TheoryWarmstartError("ws2.export.fields_invalid")
    if (
        not _valid_export_contract(export["contract"], contract)
        or export["provenance"] != provenance
    ):
        raise TheoryWarmstartError("ws2.export.contract_incompatible")
    source = _validate_state(export["corrected_state"], reference.state_dict(), "export")
    if export["state_fingerprint"] != model_state_fingerprint(source):
        raise TheoryWarmstartError("ws2.export.state_fingerprint_incompatible")
    if type(destination) is not TheoryWarmstartSSLModel:
        raise TheoryWarmstartError("ws2.destination.model_type_incompatible")
    _validate_parameter_identity(destination)
    with torch.random.fork_rng(devices=[]):
        expected_ssl = TheoryWarmstartSSLModel(destination.ssl_config)
    if (
        destination.encoder_config != expected_ssl.encoder_config
        or not _canonical_equal(
            destination.ssl_contract_metadata(), expected_ssl.ssl_contract_metadata()
        )
        or not _canonical_equal(_architecture(destination), _architecture(expected_ssl))
    ):
        raise TheoryWarmstartError("ws2.destination.model_contract_incompatible")
    current = _validate_state(destination.state_dict(), expected_ssl.state_dict(), "destination")
    backbone = tuple(name for name in source if _belongs(name, BACKBONE_GROUPS))
    retained = tuple(name for name in source if _belongs(name, RETAINED_GROUPS))
    mapping = {name: "encoder." + name for name in backbone}
    expected_backbone = {name for name in current if _belongs(
        name, tuple("encoder." + group for group in BACKBONE_GROUPS))}
    accounted = set(mapping.values()) | {name for name in current
                                       if _belongs(name, FRESH_GROUPS + INACTIVE_GROUPS)}
    if set(backbone) | set(retained) != set(source) or (
        set(mapping.values()) != expected_backbone or accounted != set(current)
    ):
        raise TheoryWarmstartError("ws2.group_partition_incompatible")
    _validate_state(
        {mapped: source[name] for name, mapped in mapping.items()},
        {name: current[name] for name in expected_backbone}, "mapping",
    )
    # Stage independent copies on the destination's device before touching it.
    devices = {value.device for value in current.values()}
    if len(devices) != 1:
        raise TheoryWarmstartError("ws2.destination.single_device_required")
    theory = reference.to(next(iter(devices)))
    theory.load_state_dict(source, strict=True)
    original = {name: value.detach().clone() for name, value in current.items()}
    original_flags = {name: value.requires_grad for name, value in destination.named_parameters()}
    merged = dict(original)
    for name, mapped in mapping.items():
        merged[mapped] = source[name].detach().to(original[mapped].device).clone()
    transferred_report = _group_report(source, BACKBONE_GROUPS)
    for group, row in transferred_report.items():
        row["destination_keys"] = tuple("encoder." + key for key in row["keys"])
    report = TheoryWarmstartReport(
        contract_version=THEORY_WARMSTART_VERSION,
        provenance=provenance,
        source_state_fingerprint=export["state_fingerprint"],
        contract_fingerprint=contract["fingerprint"],
        transferred=transferred_report,
        retained=_group_report(source, RETAINED_GROUPS),
        fresh=_group_report(original, FRESH_GROUPS),
        inactive=_group_report(original, INACTIVE_GROUPS),
    )
    try:
        destination.load_state_dict(merged, strict=True)
        if model_state_fingerprint(destination) != model_state_fingerprint(merged):
            raise TheoryWarmstartError("ws2.application.fingerprint_mismatch")
        for group in BACKBONE_GROUPS:
            parent, _, child = group.rpartition(".")
            setattr(theory.get_submodule(parent), child,
                    destination.get_submodule("encoder." + group))
        for name, parameter in destination.named_parameters():
            if _belongs(name, INACTIVE_GROUPS):
                parameter.requires_grad_(False)
        if model_state_fingerprint(theory) != export["state_fingerprint"]:
            raise TheoryWarmstartError("ws2.retained.fingerprint_mismatch")
    except Exception as exc:
        destination.load_state_dict(original, strict=True)
        for name, parameter in destination.named_parameters():
            parameter.requires_grad_(original_flags[name])
        raise TheoryWarmstartError("ws2.application.failed_and_rolled_back") from exc
    return TheoryWarmstartResult(destination, theory, report)
