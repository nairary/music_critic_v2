"""Source-free CPU mechanics; these checks do not establish musical quality."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from music_critic.experiments.analysisgnn.corrected_model import (
    CORRECTED_MODEL_SCHEMA,
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    corrected_task_head_specs,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    align_target_sidecars_after_prediction,
    corrected_supervised_loss,
)
from music_critic.experiments.analysisgnn.harmonic_logit_fusion import (
    FUSION_MODEL_SCHEMA,
    HarmonicFusionModel,
    HarmonicFusionOutput,
    HarmonicLogitFusion,
    HarmonicLogitFusionError,
    fusion_model_contract,
)
from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models.contracts import TASK_PREDICTION_CONTRACT_VERSION
from music_critic.models.heads import TaskPrediction
from music_critic.tasks import ENTITY_NODE_TYPE_TO_CODE
from tests.experiments.test_phase9eb5c_corrected_model import _batch
from tests.experiments.test_training_learning_audit import _fixture


HARMONIC = {
    "local_key", "tonicized_key", "root", "bass", "primary_degree",
    "secondary_degree", "quality", "inversion", "roman_numeral",
    "pitch_class_set", "harmonic_rhythm", "pedal",
}


@pytest.fixture(autouse=True)
def _isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield


def _predictions(*, beat_samples=(0, 0, 1, 1, 1)):
    """Canonical candidate rows for two artificial pieces of unequal lengths."""
    rows = []
    for index, spec in enumerate(corrected_task_head_specs()):
        node_type = spec.node_types[0]
        samples = beat_samples if node_type == "beat" else (0, 1)
        count = len(samples)
        code = ENTITY_NODE_TYPE_TO_CODE[node_type]
        offsets = torch.full((len(MANDATORY_NODE_TYPES),), -1, dtype=torch.long)
        counts = torch.zeros_like(offsets)
        offsets[code] = 0
        counts[code] = count
        values = torch.arange(count * spec.output_dim, dtype=torch.float32)
        logits = (values.reshape(count, spec.output_dim) / 128 + index + 1)
        rows.append(TaskPrediction(
            contract_version=TASK_PREDICTION_CONTRACT_VERSION,
            task_id=spec.task_id,
            source_adapter=spec.source_adapter,
            allowed_node_types=spec.node_types,
            candidate_node_type_codes=torch.full((count,), code, dtype=torch.long),
            global_entity_indices=torch.arange(count),
            sample_indices=torch.tensor(samples, dtype=torch.long),
            candidate_offsets_by_node_type=offsets,
            candidate_counts_by_node_type=counts,
            logits=logits.requires_grad_(),
        ))
    return tuple(rows)


def _by_task(predictions):
    return {row.task_id: row for row in predictions}


def _replace_task(predictions, task_id, **fields):
    return tuple(replace(row, **fields) if row.task_id == task_id else row
                 for row in predictions)


def _source_offset(fusion, destination, source):
    others = [task for task in fusion.harmonic_task_ids if task != destination]
    widths = {spec.task_id: spec.output_dim for spec in corrected_task_head_specs()}
    return sum(widths[task] for task in others[:others.index(source)])


def _connect(fusion, destination, source, value=0.25):
    with torch.no_grad():
        fusion.weights[destination][0, _source_offset(fusion, destination, source)] = value


def _loss(output, batch, sidecars, weights):
    alignment = align_target_sidecars_after_prediction(
        output, batch.raw_graph_batch, sidecars,
    )
    return corrected_supervised_loss(output, alignment, weights)


def test_fixed_registry_zero_parameters_no_bias_and_no_rng_consumption():
    before = torch.get_rng_state().clone()
    fusion = HarmonicLogitFusion()
    assert torch.equal(before, torch.get_rng_state())
    assert set(fusion.harmonic_task_ids) == HARMONIC
    assert len(fusion.harmonic_task_ids) == 12
    assert fusion.task_order == tuple(spec.task_id for spec in corrected_task_head_specs())
    dimensions = {spec.task_id: spec.output_dim for spec in corrected_task_head_specs()}
    assert sum(dimensions[task] for task in HARMONIC) == 518
    assert sum(parameter.numel() for parameter in fusion.parameters()) == 216_886
    assert set(dict(fusion.named_parameters())) == {f"weights.{task}" for task in HARMONIC}
    for task, weight in fusion.weights.items():
        assert weight.shape == (dimensions[task], 518 - dimensions[task])
        assert weight.dtype == torch.float32 and weight.requires_grad
        assert torch.count_nonzero(weight) == 0


@pytest.mark.parametrize("samples", [(0, 0, 1, 1, 1), ()])
def test_all_18_shapes_candidates_and_six_untouched_objects(samples):
    original = _predictions(beat_samples=samples)
    result = HarmonicLogitFusion()(original)
    assert tuple(row.task_id for row in result) == tuple(row.task_id for row in original)
    assert len(result) == 18
    for before, after in zip(original, result, strict=True):
        assert torch.equal(before.logits, after.logits)
        assert before.logits.shape == after.logits.shape
        for field in (
            "candidate_node_type_codes", "global_entity_indices", "sample_indices",
            "candidate_offsets_by_node_type", "candidate_counts_by_node_type",
        ):
            assert torch.equal(getattr(before, field), getattr(after, field))
        if before.task_id not in HARMONIC:
            assert after is before


@pytest.mark.parametrize("field", [
    "global_entity_indices", "sample_indices", "candidate_node_type_codes",
    "candidate_offsets_by_node_type", "candidate_counts_by_node_type",
    "allowed_node_types", "source_adapter",
])
def test_misaligned_candidates_rejected_even_with_equal_row_counts(field):
    predictions = _predictions()
    original = _by_task(predictions)["quality"]
    value = getattr(original, field)
    if field == "global_entity_indices":
        changed = value[torch.tensor([1, 0, 2, 3, 4])]
    elif field == "sample_indices":
        changed = value[torch.tensor([2, 1, 0, 3, 4])]
    elif field == "allowed_node_types":
        changed = ("note",)
    elif field == "source_adapter":
        changed = "synthetic:wrong-adapter"
    else:
        changed = value.clone()
        changed[ENTITY_NODE_TYPE_TO_CODE["beat"] if "by_node_type" in field else 0] += 1
    bad = _replace_task(predictions, "quality", **{field: changed})
    with pytest.raises(HarmonicLogitFusionError):
        HarmonicLogitFusion()(bad)


@pytest.mark.parametrize("kind", ["width", "float64", "float16", "nan", "inf", "missing", "duplicate", "order"])
def test_malformed_prediction_inventory_or_logits_rejected(kind):
    rows = _predictions()
    original = _by_task(rows)["quality"]
    if kind == "missing":
        rows = rows[:-1]
    elif kind == "duplicate":
        rows = rows[:-1] + (rows[0],)
    elif kind == "order":
        rows = (rows[1], rows[0], *rows[2:])
    else:
        logits = original.logits.detach().clone()
        if kind == "width":
            logits = logits[:, :-1]
        elif kind in {"float64", "float16"}:
            logits = logits.to(getattr(torch, kind))
        else:
            logits[0, 0] = float(kind)
        rows = _replace_task(rows, "quality", logits=logits)
    with pytest.raises(HarmonicLogitFusionError):
        HarmonicLogitFusion()(rows)


def test_corrections_use_original_logits_simultaneously_and_exclude_self():
    fusion = HarmonicLogitFusion()
    first, second, third = fusion.harmonic_task_ids[:3]
    _connect(fusion, first, second, 2.0)
    _connect(fusion, second, first, 3.0)
    _connect(fusion, third, second, 5.0)
    rows = _predictions()
    original = _by_task(rows)
    untouched = {row.task_id: row.logits.detach().clone() for row in rows}
    actual = _by_task(fusion(rows))
    for task in fusion.harmonic_task_ids:
        others = torch.cat([original[other].logits for other in fusion.harmonic_task_ids
                            if other != task], dim=-1)
        expected = original[task].logits + F.linear(others, fusion.weights[task])
        assert torch.equal(actual[task].logits, expected)
    sequential = original[second].logits[:, 0] + 3.0 * actual[first].logits[:, 0]
    assert not torch.equal(actual[second].logits[:, 0], sequential)
    assert all(torch.equal(row.logits, untouched[row.task_id]) for row in rows)


@pytest.mark.parametrize("kind", ["width", "float64", "nan", "inf", "overflow"])
def test_malformed_weights_and_finite_input_overflow_fail_closed(kind):
    fusion = HarmonicLogitFusion()
    if kind == "width":
        fusion.weights["quality"] = torch.nn.Parameter(fusion.weights["quality"][:, :-1].clone())
    elif kind == "float64":
        fusion.double()
    else:
        with torch.no_grad():
            fusion.weights["quality"].fill_(
                torch.finfo(torch.float32).max if kind == "overflow" else float(kind)
            )
    with pytest.raises(HarmonicLogitFusionError):
        fusion(_predictions())


@pytest.mark.parametrize("kind", ["negative_sample", "duplicate", "duplicate_across_samples", "out_of_range"])
def test_invalid_candidate_identities_rejected_when_shared_by_all_heads(kind):
    rows = _predictions()
    # Agreement alone must not legitimize negative sample IDs or duplicate rows.
    field = "sample_indices" if kind == "negative_sample" else "global_entity_indices"
    value = getattr(_by_task(rows)["quality"], field).clone()
    if kind == "negative_sample":
        value[0] = -1
    elif kind == "duplicate":
        value[1] = value[0]
    elif kind == "duplicate_across_samples":
        value[2] = value[0]
    else:
        value[-1] = len(value)
    changed = tuple(replace(row, **{field: value}) if row.task_id in HARMONIC else row
                    for row in rows)
    with pytest.raises(HarmonicLogitFusionError):
        HarmonicLogitFusion()(changed)


def test_batch_permutation_and_perturbation_do_not_mix_pieces_or_candidates():
    fusion = HarmonicLogitFusion()
    _connect(fusion, "quality", "root", 2.0)
    original = _predictions()
    initial = _by_task(fusion(original))
    # Swap the 2-row and 3-row pieces; global indices describe the new batch.
    permutation = torch.tensor([2, 3, 4, 0, 1])
    reordered = tuple(
        replace(row, logits=row.logits[permutation],
                sample_indices=torch.tensor([0, 0, 0, 1, 1]))
        if row.task_id in HARMONIC else row for row in original
    )
    swapped = _by_task(fusion(reordered))
    for task in HARMONIC:
        assert torch.equal(swapped[task].logits, initial[task].logits[permutation])
    root = _by_task(original)["root"].logits.detach().clone()
    root[3, 0] += 8
    changed = _by_task(fusion(_replace_task(original, "root", logits=root)))
    assert torch.equal(changed["quality"].logits[[0, 1, 2, 4]], initial["quality"].logits[[0, 1, 2, 4]])
    assert torch.equal(changed["quality"].logits[3, 0], initial["quality"].logits[3, 0] + 16)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_wrapper_construction_preserves_rng_base_state_and_mode(enabled, training):
    base = CorrectedAnalysisGNNModel().train(training)
    # Preserve a deliberate mixed mode too; construction must not call train().
    base.sequence_decoder.train(not training)
    modes = {name: child.training for name, child in base.named_modules()}
    state = {name: value.clone() for name, value in base.state_dict().items()}
    rng = torch.get_rng_state().clone()
    wrapper = HarmonicFusionModel(base, enabled=enabled)
    assert wrapper.base is base
    assert wrapper.training == training
    assert {name: child.training for name, child in base.named_modules()} == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(base.state_dict()[name], value) for name, value in state.items())
    if not enabled:
        assert wrapper.fusion is None
        assert tuple(wrapper.state_dict()) == tuple(f"base.{key}" for key in base.state_dict())
        assert {id(p) for p in wrapper.parameters()} == {id(p) for p in base.parameters()}


def test_disabled_wrapper_returns_exact_base_output_object(monkeypatch):
    base = CorrectedAnalysisGNNModel().eval()
    wrapper = HarmonicFusionModel(base, enabled=False)
    sentinel = object()
    calls = []

    def predict(raw, *, return_layers=False):
        calls.append((raw, return_layers))
        return sentinel

    monkeypatch.setattr(base, "predict", predict)
    raw = object()
    assert wrapper.predict(raw, return_layers=True) is sentinel
    assert wrapper(raw) is sentinel
    assert calls == [(raw, True), (raw, False)]


@pytest.mark.parametrize("enabled", [False, True])
def test_zero_fusion_exact_stochastic_logits_loss_base_gradients_state_and_rng(enabled):
    _, batch, sidecars, weights = _fixture()
    base = CorrectedAnalysisGNNModel().train()
    wrapper = HarmonicFusionModel(deepcopy(base), enabled=enabled).train()
    state = {name: value.clone() for name, value in base.state_dict().items()}
    torch.manual_seed(17001)
    before = torch.get_rng_state().clone()
    expected = base(batch.raw_graph_batch)
    expected_loss = _loss(expected, batch, sidecars, weights)
    expected_loss.total.backward()
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    actual = wrapper(batch.raw_graph_batch)
    actual_loss = _loss(actual, batch, sidecars, weights)
    actual_loss.total.backward()
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert torch.equal(expected_loss.total, actual_loss.total)
    assert all(torch.equal(expected.logits[task], actual.logits[task]) for task in expected.logits)
    actual_parameters = dict(wrapper.base.named_parameters())
    for name, parameter in base.named_parameters():
        observed = actual_parameters[name].grad
        assert (parameter.grad is None) == (observed is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, observed), name
    assert all(torch.equal(wrapper.base.state_dict()[name], value) for name, value in state.items())
    if enabled:
        assert isinstance(actual, HarmonicFusionOutput)
        assert actual.schema_version == FUSION_MODEL_SCHEMA != CORRECTED_MODEL_SCHEMA
        assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in wrapper.fusion.parameters())
    else:
        assert actual.schema_version == CORRECTED_MODEL_SCHEMA


@pytest.mark.parametrize("connected", [False, True])
def test_cross_task_derivative_and_initial_fusion_parameter_learning(connected):
    fusion = HarmonicLogitFusion()
    if connected:
        _connect(fusion, "quality", "root")
    rows = _predictions()
    source = _by_task(rows)
    result = _by_task(fusion(rows))
    F.cross_entropy(result["quality"].logits[:1], torch.tensor([0])).backward()
    root_gradient = source["root"].logits.grad
    assert root_gradient is not None
    assert bool(torch.count_nonzero(root_gradient)) is connected
    assert torch.count_nonzero(root_gradient[1:]) == 0
    assert fusion.weights["quality"].grad is not None
    assert torch.count_nonzero(fusion.weights["quality"].grad) > 0
    assert all(row.logits.grad is None for row in rows if row.task_id not in HARMONIC)


def test_nonzero_cross_task_loss_reaches_other_head_and_shared_backbone():
    base = CorrectedAnalysisGNNModel().eval()
    wrapper = HarmonicFusionModel(base)
    _connect(wrapper.fusion, "quality", "root")
    captured = []

    def isolate_cross_task_path(_module, _args, predictions):
        # Remove A's direct route only in this synthetic probe: any backbone
        # gradient must now travel through the selected B logits and fusion.
        isolated = tuple(replace(row, logits=row.logits.detach())
                         if row.task_id == "quality" else row for row in predictions)
        captured.append(isolated)
        return isolated

    handle = base.task_heads.register_forward_hook(isolate_cross_task_path)
    output = wrapper(_batch().raw_graph_batch)
    handle.remove()
    original = _by_task(captured[0])
    original["root"].logits.retain_grad()
    F.cross_entropy(output.logits["quality"][:1], torch.tensor([0])).backward()
    assert torch.count_nonzero(original["root"].logits.grad) > 0
    root_index = next(i for i, spec in enumerate(base.task_specs) if spec.task_id == "root")
    quality_index = next(i for i, spec in enumerate(base.task_specs) if spec.task_id == "quality")
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in base.task_heads.heads[f"task_{root_index:02d}"].parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad)
               for p in base.local_baseline.encoder.parameters())
    assert all(p.grad is None for p in base.task_heads.heads[f"task_{quality_index:02d}"].parameters())
    for task in set(output.logits) - HARMONIC:
        assert _by_task(output.predictions)[task] is original[task]
    parameters = list(wrapper.parameters())
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    optimizer = torch.optim.AdamW(parameters)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in base.parameters()} | {id(p) for p in wrapper.fusion.parameters()}


def test_targets_stay_post_prediction_and_missing_unsupported_masks_keep_old_loss():
    _, batch, sidecars, weights = _fixture()
    wrapper = HarmonicFusionModel(CorrectedAnalysisGNNModel().eval())
    _connect(wrapper.fusion, "quality", "root")
    missing = deepcopy(sidecars)
    for sidecar in missing:
        target = sidecar["entities"][0]["targets"]["quality"]
        target.update(available=False, masked=True, canonical_value=None,
                      source_value=None, missing_reason="synthetic_missing")
    unsupported_weights = deepcopy(weights)
    unsupported_weights.supported["root"][:2] = False
    with torch.no_grad():
        output = wrapper(batch.raw_graph_batch)
        full = _loss(output, batch, sidecars, weights)
        masked = _loss(output, batch, missing, unsupported_weights)
        repeated = wrapper(batch.raw_graph_batch)
    assert all(torch.equal(output.logits[task], repeated.logits[task]) for task in output.logits)
    assert full.heads["quality"].valid_row_count == 2
    assert masked.heads["quality"].weighted_ce is None
    assert masked.heads["quality"].masked_row_count == 2
    assert masked.heads["root"].unsupported_row_count == 2
    assert masked.heads["root"].weighted_ce is None
    assert torch.equal(masked.total, masked.primary + 0.25 * masked.auxiliary)
    for forbidden in ("targets", "masks", "sidecars", "confidence"):
        with pytest.raises(TypeError):
            wrapper.predict(batch.raw_graph_batch, **{forbidden: {}})


def test_old_contract_is_unchanged_and_enabled_contract_has_distinct_identity():
    base = CorrectedAnalysisGNNModel().eval()
    before = corrected_model_contract(base)
    disabled = HarmonicFusionModel(base, enabled=False)
    enabled = HarmonicFusionModel(base, enabled=True)
    assert corrected_model_contract(base) == before
    assert before["logit_fusion"] is False
    assert fusion_model_contract(disabled)["base_model_contract"] == before
    assert fusion_model_contract(enabled)["fingerprint"] != before["fingerprint"]
    assert all("fusion" not in key or key.startswith("context_encoder.fusion.") for key in base.state_dict())
