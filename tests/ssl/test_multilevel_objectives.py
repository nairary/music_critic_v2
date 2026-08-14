from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from music_critic.data.schema import QualityFlag, TargetArray
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.hierarchical_masking import (
    BEAT_PITCH_DESCENDANTS,
    CONTIGUOUS_BAR_PITCH_SPAN,
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
    HierarchyMaskPolicyConfig,
)
from music_critic.ssl.hierarchy_fixture import build_phase8a_hierarchy_fixture
from music_critic.ssl.masking import (
    PreparedHierarchyMaskBinding,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import MaskedGraphSSLConfig, MaskedGraphSSLModel
from music_critic.ssl.multilevel import (
    BEAT_LATENT,
    HIERARCHY_BAR_LATENT,
    ONSET_LATENT,
    PHASE7A_BAR_LATENT,
    PHASE7A_NOTE_RECONSTRUCTION,
    PHASE7A_SONG_LATENT,
    PHASE8B_NEW_OBJECTIVE_FAMILIES,
    PHASE8B_AMP_COMPUTE_CONTRACT,
    PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT,
    TRACK_LATENT,
    Phase8BMultilevelSSLModel,
    Phase8BObjectiveAccumulator,
    Phase8BObjectiveConfig,
    aggregate_phase8b_policy_pass_losses,
    build_phase8b_model,
    prepare_phase8b_objective_binding,
)


_POLICY_FOR_MODE = {
    "onset_only": ONSET_PITCH_DESCENDANTS,
    "beat_only": BEAT_PITCH_DESCENDANTS,
    "bar_only": CONTIGUOUS_BAR_PITCH_SPAN,
    "track_only": TRACK_BAR_PITCH_SPAN,
}
_FAMILY_FOR_MODE = {
    "onset_only": ONSET_LATENT,
    "beat_only": BEAT_LATENT,
    "bar_only": HIERARCHY_BAR_LATENT,
    "track_only": TRACK_LATENT,
}


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def train_batch():
    fixture = build_phase8a_hierarchy_fixture()
    return collate_ssl_samples(fixture.raw_samples("train"))


def _encoder_config() -> HierarchicalBaselineConfig:
    return HierarchicalBaselineConfig(
        hidden_dim=8,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    )


def _ssl_config() -> MaskedGraphSSLConfig:
    return MaskedGraphSSLConfig(
        mask_rate=0.30,
        decoder_views=1,
        decoder_remask_probability=0.0,
        decoder_hidden_dim=8,
        projector_hidden_dim=8,
        note_weight=1.0,
        bar_weight=1.0,
        song_weight=1.0,
    )


def _prepared(batch, policy: str):
    binding = prepare_hierarchy_mask_binding(
        batch,
        policy_config=HierarchyMaskPolicyConfig.create(
            weights={policy: 1.0},
            min_span_bars=1,
            max_span_bars=2,
        ),
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )
    return binding


def _model(mode: str, *, seed: int = 11) -> Phase8BMultilevelSSLModel:
    torch.manual_seed(seed)
    return Phase8BMultilevelSSLModel(
        _encoder_config(),
        _ssl_config(),
        Phase8BObjectiveConfig.for_mode(mode),
    )


def _forward(batch, mode: str, *, seed: int = 11):
    model = _model(mode, seed=seed)
    binding = _prepared(batch, _POLICY_FOR_MODE[mode])
    assert type(binding) is PreparedHierarchyMaskBinding
    objective_binding = prepare_phase8b_objective_binding(
        binding, model.phase8b_objective_config
    )
    output = model.forward_multilevel(
        batch,
        prepared_mask_binding=binding,
        prepared_objective_binding=objective_binding,
    )
    return model, binding, objective_binding, output


def test_registry_and_modes_are_versioned_and_independently_ablatable():
    assert len(PHASE8B_OBJECTIVE_REGISTRY_FINGERPRINT) == 64
    for mode, active_family in _FAMILY_FOR_MODE.items():
        config = Phase8BObjectiveConfig.for_mode(mode)
        assert tuple(
            family
            for family, weight in config.family_weights
            if weight > 0
        ) == (active_family,)
    equal = Phase8BObjectiveConfig.for_mode("multilevel_equal_weight")
    assert tuple(
        family for family, weight in equal.family_weights if weight > 0
    ) == PHASE8B_NEW_OBJECTIVE_FAMILIES
    control = Phase8BObjectiveConfig.for_mode("phase7a_control")
    assert tuple(
        family for family, weight in control.family_weights if weight > 0
    ) == (
        PHASE7A_NOTE_RECONSTRUCTION,
        PHASE7A_BAR_LATENT,
        PHASE7A_SONG_LATENT,
    )


@pytest.mark.parametrize(
    ("policy", "expected_available"),
    [
        (INDEPENDENT_NOTE_PITCH, ()),
        (ONSET_PITCH_DESCENDANTS, (ONSET_LATENT,)),
        (BEAT_PITCH_DESCENDANTS, (BEAT_LATENT,)),
        (CONTIGUOUS_BAR_PITCH_SPAN, (HIERARCHY_BAR_LATENT,)),
        (
            TRACK_BAR_PITCH_SPAN,
            (HIERARCHY_BAR_LATENT, TRACK_LATENT),
        ),
    ],
)
def test_exact_policy_eligibility_and_canonical_deduplication(
    train_batch, policy, expected_available
):
    binding = _prepared(train_batch, policy)
    objective_binding = prepare_phase8b_objective_binding(
        binding,
        Phase8BObjectiveConfig.for_mode("multilevel_equal_weight"),
    )
    available = tuple(
        row.family for row in objective_binding.eligible_entities if row.available
    )
    assert available == expected_available
    ptrs = dict(binding.node_ptrs)
    for row in objective_binding.eligible_entities:
        triples = tuple(
            zip(
                row.sample_indices,
                row.local_entity_indices,
                row.global_entity_indices,
                strict=True,
            )
        )
        assert triples == tuple(sorted(set(triples)))
        for sample, local, global_index in triples:
            assert global_index == ptrs[row.node_type][sample] + local
            assert global_index < ptrs[row.node_type][sample + 1]


@pytest.mark.parametrize("mode", tuple(_POLICY_FOR_MODE))
def test_level_formula_stop_gradient_and_active_gradient_paths(train_batch, mode):
    family = _FAMILY_FOR_MODE[mode]
    model = _model(mode)
    binding = _prepared(train_batch, _POLICY_FOR_MODE[mode])
    objective_binding = prepare_phase8b_objective_binding(
        binding, model.phase8b_objective_config
    )
    captured = []
    handle = model.phase8b_latent_heads[family].register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs)
    )
    try:
        output = model.forward_multilevel(
            train_batch,
            prepared_mask_binding=binding,
            prepared_objective_binding=objective_binding,
        )
    finally:
        handle.remove()
    prediction = next(row for row in output.latent_predictions if row.family == family)
    report = next(row for row in output.objective.family_losses if row.family == family)
    assert len(captured) == 1
    exact_indices = prediction.global_entity_indices
    assert torch.equal(
        captured[0][0],
        model._level_rows(output.base_output.online_encoder, family).index_select(
            0, exact_indices
        ),
    )
    full = model._detached_full_view_encoder(train_batch, binding)
    assert torch.equal(
        captured[0][1],
        model._level_rows(full, family).index_select(0, exact_indices),
    )
    assert prediction.target.requires_grad is False
    assert report.available
    assert report.active
    assert report.eligible_denominator == prediction.prediction.shape[0]
    assert report.mean_loss is not None
    assert torch.equal(
        report.mean_loss,
        report.numerator / report.eligible_denominator,
    )
    assert output.objective.total_loss is not None
    output.objective.total_loss.backward()
    head_parameters = tuple(model.phase8b_latent_heads[family].parameters())
    assert all(parameter.grad is not None for parameter in head_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in head_parameters)
    assert any(torch.count_nonzero(parameter.grad) for parameter in head_parameters)
    encoder_grads = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_grads
    assert all(torch.isfinite(gradient).all() for gradient in encoder_grads)
    assert any(torch.count_nonzero(gradient) for gradient in encoder_grads)


def test_zero_weight_heads_have_no_gradient_path(train_batch):
    model, _binding, _objective_binding, output = _forward(
        train_batch, "onset_only"
    )
    assert output.objective.total_loss is not None
    output.objective.total_loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.phase8b_latent_heads[ONSET_LATENT].parameters()
    )
    for family in (BEAT_LATENT, HIERARCHY_BAR_LATENT, TRACK_LATENT):
        assert all(
            parameter.grad is None
            for parameter in model.phase8b_latent_heads[family].parameters()
        )
    for inactive_module in (
        model.decoder,
        model.bar_projector_predictor,
        model.song_projector_predictor,
    ):
        assert all(
            parameter.grad is None for parameter in inactive_module.parameters()
        )


@pytest.mark.parametrize(
    ("mode", "policies"),
    (
        ("onset_only", (ONSET_PITCH_DESCENDANTS,)),
        (
            "multilevel_equal_weight",
            (
                ONSET_PITCH_DESCENDANTS,
                BEAT_PITCH_DESCENDANTS,
                CONTIGUOUS_BAR_PITCH_SPAN,
                TRACK_BAR_PITCH_SPAN,
            ),
        ),
    ),
)
def test_cpu_fp16_oracle_keeps_phase8b_head_and_scaled_backward_fp32_safe(
    train_batch, mode, policies
):
    model = _model(mode)
    outputs = []
    for policy in policies:
        binding = _prepared(train_batch, policy)
        objective_binding = prepare_phase8b_objective_binding(
            binding, model.phase8b_objective_config
        )
        with torch.autocast("cpu", dtype=torch.float16):
            output = model.forward_multilevel(
                train_batch,
                prepared_mask_binding=binding,
                prepared_objective_binding=objective_binding,
            )
        outputs.append((policy, output))
        for row in output.latent_predictions:
            assert row.prediction.dtype == torch.float32
            assert row.target.dtype == torch.float32
    batch_objective = aggregate_phase8b_policy_pass_losses(
        tuple(outputs), objective_config=model.phase8b_objective_config
    )
    assert PHASE8B_AMP_COMPUTE_CONTRACT.endswith("float32")
    assert batch_objective.total_loss is not None
    assert batch_objective.total_loss.dtype == torch.float32
    assert batch_objective.total_loss.grad_fn is not None
    (batch_objective.total_loss * 16384.0).backward()
    gradients = tuple(
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    for family in PHASE8B_NEW_OBJECTIVE_FAMILIES:
        head_gradients = tuple(
            parameter.grad
            for parameter in model.phase8b_latent_heads[family].parameters()
        )
        if model.phase8b_objective_config.weight(family) > 0.0:
            assert all(gradient is not None for gradient in head_gradients)
            assert all(
                torch.isfinite(gradient).all()
                for gradient in head_gradients
                if gradient is not None
            )
            assert any(
                torch.count_nonzero(gradient)
                for gradient in head_gradients
                if gradient is not None
            )
        else:
            assert all(gradient is None for gradient in head_gradients)


def test_unavailable_family_is_not_fabricated_as_zero(train_batch):
    model = _model("onset_only")
    binding = _prepared(train_batch, CONTIGUOUS_BAR_PITCH_SPAN)
    objective_binding = prepare_phase8b_objective_binding(
        binding, model.phase8b_objective_config
    )
    output = model.forward_multilevel(
        train_batch,
        prepared_mask_binding=binding,
        prepared_objective_binding=objective_binding,
    )
    onset = next(
        row for row in output.objective.family_losses if row.family == ONSET_LATENT
    )
    assert onset.eligible_denominator == 0
    assert onset.numerator is None
    assert onset.mean_loss is None
    assert not onset.available
    assert output.objective.total_loss is None


def test_forward_preserves_raw_graph_and_prepared_binding(train_batch):
    model = _model("bar_only")
    binding = _prepared(train_batch, CONTIGUOUS_BAR_PITCH_SPAN)
    objective_binding = prepare_phase8b_objective_binding(
        binding, model.phase8b_objective_config
    )
    graph_before = deepcopy(train_batch.raw_graph_batch)
    binding_before = binding.to_dict()
    objective_before = objective_binding.to_dict()
    model.forward_multilevel(
        train_batch,
        prepared_mask_binding=binding,
        prepared_objective_binding=objective_binding,
    )
    assert binding.to_dict() == binding_before
    assert objective_binding.to_dict() == objective_before
    for left, right in zip(
        train_batch.raw_graph_batch.to_data_list(),
        graph_before.to_data_list(),
        strict=True,
    ):
        for node_type in left.node_types:
            for name in left[node_type].keys():
                left_value = left[node_type][name]
                right_value = right[node_type][name]
                if isinstance(left_value, torch.Tensor):
                    assert torch.equal(left_value, right_value)
                else:
                    assert left_value == right_value
        for edge_type in left.edge_types:
            for name in left[edge_type].keys():
                left_value = left[edge_type][name]
                right_value = right[edge_type][name]
                if isinstance(left_value, torch.Tensor):
                    assert torch.equal(left_value, right_value)
                else:
                    assert left_value == right_value


def test_phase7a_control_constructs_literal_old_model_and_is_bit_exact(train_batch):
    torch.manual_seed(91)
    direct = MaskedGraphSSLModel(_encoder_config(), _ssl_config())
    torch.manual_seed(91)
    control = build_phase8b_model(
        _encoder_config(),
        _ssl_config(),
        Phase8BObjectiveConfig.for_mode("phase7a_control"),
    )
    assert type(control) is MaskedGraphSSLModel
    assert direct.state_dict().keys() == control.state_dict().keys()
    for name, value in direct.state_dict().items():
        assert torch.equal(value, control.state_dict()[name])
    binding = _prepared(train_batch, INDEPENDENT_NOTE_PITCH)
    direct_output = direct(
        train_batch,
        prepared_mask_binding=binding,
    )
    control_output = control(
        train_batch,
        prepared_mask_binding=binding,
    )
    assert torch.equal(
        direct_output.objective.total_loss,
        control_output.objective.total_loss,
    )
    assert direct_output.prepared_mask_binding_fingerprint == (
        control_output.prepared_mask_binding_fingerprint
    )


def test_fixed_weight_aggregation_and_metrics_retain_no_prediction_tensors(train_batch):
    _model_value, _binding, _objective_binding, output = _forward(
        train_batch, "track_only"
    )
    accumulator = Phase8BObjectiveAccumulator(
        Phase8BObjectiveConfig.for_mode("track_only")
    )
    accumulator.update(output)
    report = accumulator.finalize()
    assert report["retained_cuda_tensor_count"] == 0
    assert report["retained_prediction_tensor_count"] == 0
    assert report["total_loss"] == pytest.approx(
        float(output.objective.total_loss.detach())
    )
    assert not any(
        isinstance(value, torch.Tensor)
        for family in report["families"]
        for value in family.values()
    )


def test_new_head_parameter_count_is_exact():
    model = _model("multilevel_equal_weight")
    # One head has two MLPs.  Each MLP is 8x8 + bias, LayerNorm(8), 8x8 + bias.
    assert model.new_head_parameter_count() == 4 * 320


def _onset_report(model, samples):
    batch = collate_ssl_samples(samples)
    binding = _prepared(batch, ONSET_PITCH_DESCENDANTS)
    objective_binding = prepare_phase8b_objective_binding(
        binding, model.phase8b_objective_config
    )
    output = model.forward_multilevel(
        batch,
        prepared_mask_binding=binding,
        prepared_objective_binding=objective_binding,
    )
    accumulator = Phase8BObjectiveAccumulator(model.phase8b_objective_config)
    accumulator.update(output)
    return accumulator.finalize()


def _family(report, family):
    return next(row for row in report["families"] if row["family"] == family)


def test_level_loss_is_invariant_to_batch_partition_and_sample_order():
    samples = build_phase8a_hierarchy_fixture().raw_samples("train")
    model = _model("onset_only")
    model.eval()
    full = _onset_report(model, samples)
    reversed_report = _onset_report(model, tuple(reversed(samples)))
    partitioned = Phase8BObjectiveAccumulator(model.phase8b_objective_config)
    for sample in samples:
        batch = collate_ssl_samples((sample,))
        binding = _prepared(batch, ONSET_PITCH_DESCENDANTS)
        objective_binding = prepare_phase8b_objective_binding(
            binding, model.phase8b_objective_config
        )
        partitioned.update(
            model.forward_multilevel(
                batch,
                prepared_mask_binding=binding,
                prepared_objective_binding=objective_binding,
            )
        )
    partitioned_report = partitioned.finalize()
    reference = _family(full, ONSET_LATENT)
    for report in (reversed_report, partitioned_report):
        candidate = _family(report, ONSET_LATENT)
        assert candidate["eligible_denominator"] == reference[
            "eligible_denominator"
        ]
        assert candidate["numerator"] == pytest.approx(
            reference["numerator"], abs=1e-6
        )
        assert candidate["mean_loss"] == pytest.approx(
            reference["mean_loss"], abs=1e-7
        )


def test_target_provenance_mutation_does_not_change_multilevel_logits():
    piece = build_phase8a_hierarchy_fixture().supplemental_piece
    changed = replace(
        piece,
        source_path="ignored/phase8b-sidecar.mid",
        targets=(
            TargetArray(
                target_id="target:phase8b-inert",
                task="quality.overall",
                annotation_view_id=None,
                alignment_type="piece",
                entity_ids=(piece.piece_id,),
                value_type="scalar",
                class_labels=None,
                values=(0.75,),
                mask=(True,),
                confidence=(1.0,),
                source=("synthetic",),
                provenance=(piece.provenance[0].provenance_id,),
            ),
        ),
        provenance=(
            replace(
                piece.provenance[0],
                source="phase8b_sidecar_mutation",
                details=(("diagnostic", "changed"),),
            ),
        ),
        quality_flags=(
            QualityFlag(
                code="phase8b.test.diagnostic",
                severity="info",
                message="target-blind multilevel mutation",
                entity_ids=(piece.piece_id,),
                provenance_id=piece.provenance[0].provenance_id,
            ),
        ),
    )

    def sample(value):
        graph = build_raw_graph(value, assume_valid=True)
        return SSLRawSample(
            raw_graph=graph,
            raw_graph_fingerprint=graph_fingerprint(graph),
            dataset_id=value.dataset_name,
            piece_id=value.piece_id,
        )

    source_sample, changed_sample = sample(piece), sample(changed)
    assert source_sample.raw_graph_fingerprint == changed_sample.raw_graph_fingerprint
    model = _model("onset_only")
    model.eval()

    def output(value):
        batch = collate_ssl_samples((value,))
        binding = _prepared(batch, ONSET_PITCH_DESCENDANTS)
        objectives = prepare_phase8b_objective_binding(
            binding, model.phase8b_objective_config
        )
        return model.forward_multilevel(
            batch,
            prepared_mask_binding=binding,
            prepared_objective_binding=objectives,
        )

    original, mutated = output(source_sample), output(changed_sample)
    assert original.prepared_objective_binding_fingerprint == (
        mutated.prepared_objective_binding_fingerprint
    )
    assert torch.equal(
        original.objective.total_loss, mutated.objective.total_loss
    )
    assert len(original.latent_predictions) == len(mutated.latent_predictions)
    for left, right in zip(
        original.latent_predictions,
        mutated.latent_predictions,
        strict=True,
    ):
        assert torch.equal(left.global_entity_indices, right.global_entity_indices)
        assert torch.equal(left.prediction, right.prediction)
        assert torch.equal(left.target, right.target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optional_cuda_amp_uses_shared_deterministic_runtime(train_batch):
    from music_critic.ssl.deterministic_runtime import (
        deterministic_cuda_evidence_runtime,
    )

    model = _model("onset_only").to("cuda:0")
    binding = _prepared(train_batch, ONSET_PITCH_DESCENDANTS)
    moved_batch, moved_binding = move_ssl_batch_with_prepared_binding(
        train_batch, binding, "cuda:0"
    )
    objective_binding = prepare_phase8b_objective_binding(
        moved_binding, model.phase8b_objective_config
    )
    with deterministic_cuda_evidence_runtime():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.forward_multilevel(
                moved_batch,
                prepared_mask_binding=moved_binding,
                prepared_objective_binding=objective_binding,
            )
        assert output.objective.total_loss is not None
        assert torch.isfinite(output.objective.total_loss)
        batch_objective = aggregate_phase8b_policy_pass_losses(
            ((ONSET_PITCH_DESCENDANTS, output),),
            objective_config=model.phase8b_objective_config,
        )
        assert batch_objective.total_loss is not None
        accumulator = Phase8BObjectiveAccumulator(
            model.phase8b_objective_config
        )
        accumulator.update_batch(batch_objective)
        report = accumulator.finalize()
        assert report["packed_device_to_host_transfer_count"] == 1
        assert report["maximum_packed_d2h_transfers_per_cpu_batch"] == 1
        assert report["retained_cuda_tensor_count"] == 0
        assert report["retained_prediction_tensor_count"] == 0
        batch_objective.total_loss.backward()
