"""Fixture-only contract tests at the complete corrected architecture."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    model_state_fingerprint,
)
from music_critic.ssl.theory_warmstart import (
    BACKBONE_GROUPS,
    FRESH_GROUPS,
    INACTIVE_GROUPS,
    RETAINED_GROUPS,
    THEORY_WARMSTART_VERSION,
    TheoryWarmstartError,
    TheoryWarmstartSSLModel,
    export_theory_warmstart_state,
    load_theory_warmstart,
)
from music_critic.ssl.transfer import (
    EncoderTransferError,
    export_pretrained_encoder_state,
    validate_pretrained_encoder_export_structure,
)


def _belongs(name, groups):
    return any(name == group or name.startswith(group + ".") for group in groups)


def _snapshot(model):
    return (
        deepcopy(model.state_dict()),
        {name: parameter.requires_grad for name, parameter in model.named_parameters()},
        {name: module.training for name, module in model.named_modules()},
        {name: id(parameter) for name, parameter in model.named_parameters()},
    )


def _assert_snapshot(model, before):
    state, flags, modes, identities = before
    assert set(model.state_dict()) == set(state)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0, equal_nan=True)
    assert {name: p.requires_grad for name, p in model.named_parameters()} == flags
    assert {name: m.training for name, m in model.named_modules()} == modes
    assert {name: id(p) for name, p in model.named_parameters()} == identities


@pytest.fixture(scope="module")
def source():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(411)
        model = CorrectedAnalysisGNNModel()
    model.eval()
    model.context_encoder.train()
    return model


@pytest.fixture(scope="module")
def exported(source):
    return export_theory_warmstart_state(source)


@pytest.fixture
def destination():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(733)
        model = TheoryWarmstartSSLModel()
    # Failure checks include heterogeneous modes and non-default grad flags.
    model.eval()
    model.decoder.train()
    next(model.parameters()).requires_grad_(False)
    return model


def test_transfer_exact_complete_groups_retains_full_theory_and_fresh_ssl(
    source, exported, destination,
):
    source_before = _snapshot(source)
    destination_before = deepcopy(destination.state_dict())
    export_before = deepcopy(exported["corrected_state"])
    result = load_theory_warmstart(destination, exported)
    assert result.ssl_model is destination
    theory = result.theory_model
    assert corrected_model_contract(theory) == corrected_model_contract(source)
    config = theory.config
    assert (
        config.hidden_dim, config.local_gnn_layers, config.transformer_layers,
        config.attention_heads, config.ffn_multiplier, config.dropout,
    ) == (128, 3, 2, 4, 4, 0.1)
    assert len(theory.task_heads.heads) == 18
    assert model_state_fingerprint(theory) == model_state_fingerprint(source)
    assert not ({id(p) for p in source.parameters()} & {
        id(p) for p in (*destination.parameters(), *theory.parameters())
    })
    for group in BACKBONE_GROUPS:
        assert theory.get_submodule(group) is destination.get_submodule("encoder." + group)
    for name, tensor in source.state_dict().items():
        assert torch.equal(theory.state_dict()[name], tensor)
        if _belongs(name, BACKBONE_GROUPS):
            assert torch.equal(destination.state_dict()["encoder." + name], tensor)
        assert theory.state_dict()[name].data_ptr() != tensor.data_ptr()
        assert torch.equal(exported["corrected_state"][name], export_before[name])
    retained_ids = {
        id(parameter) for name, parameter in theory.named_parameters()
        if _belongs(name, RETAINED_GROUPS)
    }
    assert retained_ids.isdisjoint(id(p) for p in destination.parameters())
    for name, tensor in destination.state_dict().items():
        if _belongs(name, FRESH_GROUPS + INACTIVE_GROUPS):
            assert torch.equal(tensor, destination_before[name])
    for name, parameter in destination.named_parameters():
        if _belongs(name, INACTIVE_GROUPS):
            assert parameter.requires_grad is False
    _assert_snapshot(source, source_before)


def test_report_has_exact_group_partition_and_verified_in_memory_fingerprints(
    source, exported, destination,
):
    before = deepcopy(destination.state_dict())
    report = load_theory_warmstart(destination, exported).report
    assert report.contract_version == THEORY_WARMSTART_VERSION
    assert report.provenance == "synthetic_in_memory"
    assert report.source_state_fingerprint == model_state_fingerprint(source)
    assert report.contract_fingerprint == exported["contract"]["fingerprint"]
    contract = deepcopy(exported["contract"])
    assert contract.pop("fingerprint") == fingerprint(contract)
    expected_mapping = {group: "encoder." + group for group in BACKBONE_GROUPS}
    assert exported["contract"]["backbone_mapping"] == expected_mapping
    source_keys, destination_keys = set(), set()
    for rows, groups, state in (
        (report.transferred, BACKBONE_GROUPS, source.state_dict()),
        (report.retained, RETAINED_GROUPS, source.state_dict()),
        (report.fresh, FRESH_GROUPS, before),
        (report.inactive, INACTIVE_GROUPS, before),
    ):
        assert set(rows) == set(groups)
        for group, row in rows.items():
            selected = {k: v for k, v in state.items() if _belongs(k, (group,))}
            assert selected
            assert row["keys"] == tuple(sorted(selected))
            assert row["numel"] == sum(t.numel() for t in selected.values())
            assert row["fingerprint"] == model_state_fingerprint(selected)
            if rows is report.transferred:
                assert row["destination_keys"] == tuple("encoder." + k for k in row["keys"])
                destination_keys.update(row["destination_keys"])
            if rows is report.transferred or rows is report.retained:
                assert source_keys.isdisjoint(selected)
                source_keys.update(selected)
            else:
                assert destination_keys.isdisjoint(selected)
                destination_keys.update(selected)
    assert source_keys == set(source.state_dict())
    assert destination_keys == set(destination.state_dict())


def test_export_is_detached_independent_clone_and_consumes_no_rng(source):
    source_before = _snapshot(source)
    rng_before = torch.random.get_rng_state().clone()
    payload = export_theory_warmstart_state(source)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, tensor in payload["corrected_state"].items():
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False and tensor.grad_fn is None
        assert tensor.data_ptr() != source.state_dict()[name].data_ptr()
    first = next(iter(payload["corrected_state"].values()))
    first.add_(1)
    _assert_snapshot(source, source_before)


@pytest.mark.parametrize("mutation, error", (
    ("missing_key", "state_keys_incompatible"),
    ("extra_key", "state_keys_incompatible"),
    ("wrong_shape", "tensor_incompatible"),
    ("wrong_dtype", "tensor_incompatible"),
    ("not_tensor", "tensor_incompatible"),
    ("nan", "nonfinite"),
    ("inf", "nonfinite"),
    ("changed_state", "state_fingerprint_incompatible"),
    ("wrong_hash", "state_fingerprint_incompatible"),
    ("model_contract", "contract_incompatible"),
    ("feature_contract", "contract_incompatible"),
    ("graph_contract", "contract_incompatible"),
    ("mapping_contract", "contract_incompatible"),
    ("contract_hash", "contract_incompatible"),
    ("provenance", "contract_incompatible"),
    ("extra_field", "fields_invalid"),
))
def test_corrupt_export_rejected_before_any_destination_mutation(
    exported, destination, mutation, error, monkeypatch,
):
    payload = deepcopy(exported)
    state = payload["corrected_state"]
    key = next(iter(state))
    if mutation == "missing_key":
        state.pop(key)
    elif mutation == "extra_key":
        state["unexpected.weight"] = torch.zeros(1)
    elif mutation == "wrong_shape":
        state[key] = torch.zeros(state[key].numel() + 1)
    elif mutation == "wrong_dtype":
        state[key] = state[key].double()
    elif mutation == "not_tensor":
        state[key] = [0]
    elif mutation in ("nan", "inf"):
        state[key].reshape(-1)[0] = float(mutation)
    elif mutation == "changed_state":
        state[key].add_(0.125)
    elif mutation == "wrong_hash":
        payload["state_fingerprint"] = "0" * 64
    elif mutation == "model_contract":
        payload["contract"]["model"]["active_head_count"] = 20
    elif mutation == "feature_contract":
        payload["contract"]["feature_registry_fingerprint"] = "0" * 64
    elif mutation == "graph_contract":
        payload["contract"]["graph_schema_version"] = "unknown"
    elif mutation == "mapping_contract":
        payload["contract"]["backbone_mapping"][BACKBONE_GROUPS[0]] = "encoder.wrong"
    elif mutation == "contract_hash":
        payload["contract"]["fingerprint"] = "0" * 64
    elif mutation == "provenance":
        payload["provenance"] = "trained_checkpoint"
    elif mutation == "extra_field":
        payload["checkpoint_sha256"] = "0" * 64
    before = _snapshot(destination)
    rng_before = torch.random.get_rng_state().clone()

    def forbidden_load(*args, **kwargs):
        pytest.fail("Destination load must not run before complete validation")

    monkeypatch.setattr(destination, "load_state_dict", forbidden_load)
    with pytest.raises(TheoryWarmstartError, match=error):
        load_theory_warmstart(destination, payload)
    _assert_snapshot(destination, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("reseal", (False, True), ids=("stale_hash", "recomputed_hash"))
@pytest.mark.parametrize("field, value", (("active_head_count", 18.0), ("staff_included", 0)))
def test_equal_valued_contract_type_drift_is_rejected_atomically(
    exported, destination, field, value, reseal,
):
    payload = deepcopy(exported)
    contract = payload["contract"]
    contract["model"][field] = value
    # Python equality alone cannot distinguish either deliberate type drift.
    assert contract == exported["contract"]
    if reseal:
        contract["fingerprint"] = fingerprint({
            key: item for key, item in contract.items() if key != "fingerprint"
        })
    before = _snapshot(destination)
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(TheoryWarmstartError, match="ws2.export.contract_incompatible"):
        load_theory_warmstart(destination, payload)
    _assert_snapshot(destination, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("malformed", (None, [], "contract", float("nan"), float("inf"), object()))
def test_malformed_or_nonfinite_contract_is_rejected_atomically(exported, destination, malformed):
    payload = deepcopy(exported)
    if isinstance(malformed, float):
        payload["contract"]["model"]["active_head_count"] = malformed
    else:
        payload["contract"] = malformed
    before = _snapshot(destination)
    with pytest.raises(TheoryWarmstartError, match="ws2.export.contract_incompatible"):
        load_theory_warmstart(destination, payload)
    _assert_snapshot(destination, before)


@pytest.mark.parametrize("mutation", ("dropout", "attention_heads", "attention_heads_float", "config"))
def test_source_runtime_architecture_drift_rejected_without_mutation(source, mutation):
    changed = deepcopy(source)
    if mutation == "dropout":
        next(m for m in changed.modules() if isinstance(m, nn.Dropout)).p = 0.2
    elif mutation.startswith("attention_heads"):
        attention = next(m for m in changed.modules() if isinstance(m, nn.MultiheadAttention))
        attention.num_heads = 4.0 if mutation == "attention_heads_float" else 8
    else:
        object.__setattr__(changed.config, "dropout", 0.2)
    # Both changes preserve every state tensor shape.
    assert model_state_fingerprint(changed) == model_state_fingerprint(source)
    before = _snapshot(changed)
    with pytest.raises(TheoryWarmstartError, match="source.model_contract_incompatible"):
        export_theory_warmstart_state(changed)
    _assert_snapshot(changed, before)


@pytest.mark.parametrize("mutation, error", (
    ("dropout", "model_contract_incompatible"),
    ("attention_heads", "model_contract_incompatible"),
    ("attention_heads_float", "model_contract_incompatible"),
    ("encoder_config", "model_contract_incompatible"),
    ("wrong_dtype", "tensor_incompatible"),
    ("nan", "nonfinite"),
))
def test_destination_runtime_drift_rejected_atomically(exported, destination, mutation, error):
    if mutation == "dropout":
        next(m for m in destination.modules() if isinstance(m, nn.Dropout)).p = 0.2
    elif mutation.startswith("attention_heads"):
        attention = next(m for m in destination.modules() if isinstance(m, nn.MultiheadAttention))
        attention.num_heads = 4.0 if mutation == "attention_heads_float" else 8
    elif mutation == "encoder_config":
        destination.encoder_config = replace(destination.encoder_config, dropout=0.2)
    elif mutation == "wrong_dtype":
        parameter = next(destination.parameters())
        parameter.data = parameter.data.double()
    elif mutation == "nan":
        with torch.no_grad():
            next(destination.parameters()).reshape(-1)[0] = float("nan")
    before = _snapshot(destination)
    with pytest.raises(TheoryWarmstartError, match=error):
        load_theory_warmstart(destination, exported)
    _assert_snapshot(destination, before)


@pytest.mark.parametrize("branch", ("source", "destination"))
def test_unexpected_tied_parameters_rejected_without_mutation(source, exported, destination, branch):
    model = deepcopy(source) if branch == "source" else destination
    previous_by_shape = {}
    for name, parameter in model.named_parameters():
        shape = tuple(parameter.shape)
        if shape in previous_by_shape:
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, previous_by_shape[shape])
            break
        previous_by_shape[shape] = parameter
    else:
        pytest.fail("The full corrected architecture must contain compatible parameter pairs")
    before = _snapshot(model)
    with pytest.raises(TheoryWarmstartError, match="unexpected_parameter_alias"):
        if branch == "source":
            export_theory_warmstart_state(model)
        else:
            load_theory_warmstart(model, exported)
    _assert_snapshot(model, before)


@pytest.mark.parametrize("branch", ("source", "destination"))
@pytest.mark.parametrize("alias_kind", (
    "same_pointer", "overlapping_offset_view", "overlapping_buffer_storages",
))
def test_distinct_parameters_sharing_storage_rejected_atomically(
    source, exported, destination, branch, alias_kind,
):
    model = deepcopy(source) if branch == "source" else destination
    if branch == "destination":
        names = (
            "feature_mask_token",
            "encoder.local_baseline.task_heads.heads.task_00.0.bias",
        )
    else:
        names = tuple(
            name for name, parameter in model.named_parameters()
            if parameter.shape == (128,)
        )[:2]
    original = tuple(model.get_parameter(name) for name in names)
    assert len(original) == 2 and all(parameter.shape == (128,) for parameter in original)
    if alias_kind == "overlapping_buffer_storages":
        item_size = original[0].element_size()
        shared_buffer = bytearray(129 * item_size)
        replacements = tuple(
            nn.Parameter(
                torch.frombuffer(shared_buffer, dtype=parameter.dtype, count=128, offset=index * item_size),
                requires_grad=parameter.requires_grad,
            )
            for index, parameter in enumerate(original)
        )
    elif alias_kind == "overlapping_offset_view":
        storage = torch.zeros(129, dtype=original[0].dtype)
        replacements = (
            nn.Parameter(storage[:-1], requires_grad=original[0].requires_grad),
            nn.Parameter(storage[1:], requires_grad=original[1].requires_grad),
        )
    else:
        # Exact inactive-head/mask-token reproducer: independent Parameter
        # wrappers still share the complete allocation after detach().
        replacements = (
            original[0],
            nn.Parameter(original[0].detach(), requires_grad=original[1].requires_grad),
        )
    for name, replacement in zip(names, replacements, strict=True):
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
    first, second = (model.get_parameter(name) for name in names)
    assert first is not second
    if alias_kind == "overlapping_buffer_storages":
        assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
        assert first.data_ptr() < second.data_ptr() < first.data_ptr() + first.nbytes
    else:
        assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert (first.data_ptr() == second.data_ptr()) is (alias_kind == "same_pointer")
    before = _snapshot(model)
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(TheoryWarmstartError, match="unexpected_parameter_storage_alias"):
        if branch == "source":
            export_theory_warmstart_state(model)
        else:
            load_theory_warmstart(model, exported)
    _assert_snapshot(model, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("branch", ("source", "destination"))
@pytest.mark.parametrize("invalid_kind", ("meta", "sparse"))
def test_storage_preflight_preserves_structured_invalid_tensor_rejection(
    source, exported, destination, branch, invalid_kind,
):
    model = deepcopy(source) if branch == "source" else destination
    name, parameter = next(iter(model.named_parameters()))
    invalid = (
        torch.empty_like(parameter, device="meta")
        if invalid_kind == "meta" else parameter.detach().to_sparse()
    )
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent), leaf, nn.Parameter(invalid, requires_grad=parameter.requires_grad))
    before = _snapshot(model)
    with pytest.raises(TheoryWarmstartError, match=f"ws2.{branch}.tensor_incompatible"):
        if branch == "source":
            export_theory_warmstart_state(model)
        else:
            load_theory_warmstart(model, exported)
    _assert_snapshot(model, before)


@pytest.mark.parametrize("mutation, error", (("wrong_dtype", "tensor_incompatible"), ("nan", "nonfinite")))
def test_source_tensor_validation_does_not_mutate_source(source, mutation, error):
    changed = deepcopy(source)
    parameter = next(changed.parameters())
    with torch.no_grad():
        if mutation == "wrong_dtype":
            parameter.data = parameter.data.double()
        else:
            parameter.reshape(-1)[0] = float("nan")
    before = _snapshot(changed)
    with pytest.raises(TheoryWarmstartError, match=error):
        export_theory_warmstart_state(changed)
    _assert_snapshot(changed, before)


def test_loading_preserves_rng_and_existing_destination_modes(exported, destination):
    rng_before = torch.random.get_rng_state().clone()
    modes = {name: module.training for name, module in destination.named_modules()}
    load_theory_warmstart(destination, exported)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert {name: module.training for name, module in destination.named_modules()} == modes


def test_application_error_rolls_back_loaded_weights_and_grad_flags(
    exported, destination,
):
    before = _snapshot(destination)
    calls = []

    def fail_once(module, incompatible_keys):
        assert not incompatible_keys.missing_keys
        assert not incompatible_keys.unexpected_keys
        calls.append(model_state_fingerprint(module))
        if len(calls) == 1:
            next(destination.parameters()).requires_grad_(True)
            raise RuntimeError("synthetic failure after applying state")

    hook = destination.register_load_state_dict_post_hook(fail_once)
    try:
        with pytest.raises(TheoryWarmstartError, match="failed_and_rolled_back"):
            load_theory_warmstart(destination, exported)
    finally:
        hook.remove()
    assert len(calls) == 2
    assert calls[0] != model_state_fingerprint(before[0])
    assert calls[1] == model_state_fingerprint(before[0])
    _assert_snapshot(destination, before)


def test_bridge_rejects_full_checkpoint_and_old_export_boundary_stays_strict(
    source, exported, destination,
):
    before = _snapshot(destination)
    with pytest.raises(TheoryWarmstartError, match="fields_invalid"):
        load_theory_warmstart(destination, {"model_state": source.state_dict(), "optimizer": {}})
    _assert_snapshot(destination, before)
    with pytest.raises(EncoderTransferError, match="hierarchical_encoder_required"):
        export_pretrained_encoder_state(source)
    with pytest.raises(EncoderTransferError, match="export_fields_invalid"):
        validate_pretrained_encoder_export_structure(exported)
