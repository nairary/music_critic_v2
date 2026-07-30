from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from scripts.benchmark_phase8a_hierarchical_masking import (
    PHASE8A_MASKING_BENCHMARK_CONTRACT_VERSION,
    benchmark_phase8a_policy,
)
from music_critic.ssl.phase8a_acceptance import (
    PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION,
    build_phase8a_bounded_acceptance_report,
)
from music_critic.ssl.phase8a_cuda_acceptance import (
    PHASE8A_CPU_CUDA_NUMERICAL_PARITY_ATOL,
    PHASE8A_CPU_CUDA_NUMERICAL_PARITY_RTOL,
    _cross_backend_numerical_parity,
    _graphs_cross_device_bit_exact,
    _outputs_bit_exact,
)
from music_critic.data import CanonicalPiece, QualityFlag, TargetArray
from music_critic.graph import (
    build_raw_graph,
    graph_fingerprint,
    model_input_fingerprint,
)
from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.contracts import SSLContractError
from music_critic.ssl.contracts import (
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
)
from music_critic.ssl.data import (
    SSLBatch,
    SSLRawSample,
    build_ssl_data_runtime,
    collate_ssl_samples,
)
from music_critic.ssl.hierarchical_masking import (
    CONTIGUOUS_BAR_PITCH_SPAN,
    HIERARCHY_MASK_POLICIES,
    INDEPENDENT_NOTE_PITCH,
    ONSET_PITCH_DESCENDANTS,
    TRACK_BAR_PITCH_SPAN,
    HierarchyMaskPolicyConfig,
)
from music_critic.ssl.hierarchy_fixture import (
    build_phase8a_hierarchy_fixture,
)
from music_critic.ssl.masking import (
    PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION,
    PreparedHierarchyMaskBinding,
    PreparedMaskBinding,
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
    prepare_mask_binding,
)
from music_critic.ssl.model import (
    PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION,
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
    Phase8AHierarchySSLForwardOutput,
)
from music_critic.training.config import DataConfig


_MASK_RATE = 0.30
_GLOBAL_SEED = 42


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def bounded_batch() -> SSLBatch:
    return build_ssl_data_runtime(
        DataConfig(),
        seed=_GLOBAL_SEED,
    ).first_train_batch


def _model(seed: int = 101) -> MaskedGraphSSLModel:
    torch.manual_seed(seed)
    return MaskedGraphSSLModel(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        ),
        MaskedGraphSSLConfig(
            mask_rate=_MASK_RATE,
            decoder_views=1,
            decoder_remask_probability=0.0,
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    )


def _policy_config(policy: str) -> HierarchyMaskPolicyConfig:
    return HierarchyMaskPolicyConfig.create(
        weights={policy: 1.0},
        min_span_bars=1,
        max_span_bars=2,
    )


def _prepared(
    batch: SSLBatch,
    policy: str,
    *,
    device: torch.device | str = "cpu",
):
    binding = prepare_hierarchy_mask_binding(
        batch,
        policy_config=_policy_config(policy),
        global_seed=_GLOBAL_SEED,
        epoch=0,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    return move_ssl_batch_with_prepared_binding(
        batch,
        binding,
        device,
    )


def _first_feature_encoder(model: MaskedGraphSSLModel) -> torch.nn.Module:
    return (
        model.encoder.local_baseline.encoder.feature_encoder.node_encoders[
            "song"
        ]
    )


def _assert_tensor_equal(left: Tensor, right: Tensor) -> None:
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert torch.equal(left, right)


def _raw_sample(piece: CanonicalPiece) -> SSLRawSample:
    graph = build_raw_graph(piece)
    return SSLRawSample(
        raw_graph=graph,
        raw_graph_fingerprint=graph_fingerprint(graph),
        dataset_id=piece.dataset_name,
        piece_id=piece.piece_id,
    )


def _batch_graph_fingerprints(
    batch: SSLBatch,
) -> tuple[tuple[str, str], ...]:
    graphs = batch.raw_graph_batch.to_data_list()
    for graph in graphs:
        if isinstance(graph.raw_only, Tensor):
            graph.raw_only = bool(graph.raw_only.item())
    return tuple(
        (
            graph_fingerprint(graph),
            model_input_fingerprint(graph),
        )
        for graph in graphs
    )


@pytest.mark.parametrize("policy", HIERARCHY_MASK_POLICIES)
def test_each_policy_prepares_and_has_finite_forward_backward_gradients(
    bounded_batch: SSLBatch,
    policy: str,
) -> None:
    model = _model(101 + HIERARCHY_MASK_POLICIES.index(policy)).train()
    batch, binding = _prepared(bounded_batch, policy)

    output = model.forward_hierarchy(
        batch,
        prepared_mask_binding=binding,
    )
    loss = output.objective.total_loss

    assert loss is not None
    if policy == INDEPENDENT_NOTE_PITCH:
        assert type(binding) is PreparedMaskBinding
        assert binding.contract_version == (
            PREPARED_MASK_BINDING_CONTRACT_VERSION
        ) == "1.1.0"
    else:
        assert type(binding) is PreparedHierarchyMaskBinding
        assert binding.contract_version == (
            PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
        ) == "1.2.0"
    assert type(output) is Phase8AHierarchySSLForwardOutput
    assert output.contract_version == (
        PHASE8A_HIERARCHY_SSL_OUTPUT_CONTRACT_VERSION
    ) == "1.0.0"
    assert bool(torch.isfinite(loss))
    assert output.mask_plans == binding.mask_plans
    assert output.feature_overlay == binding.feature_overlay
    assert output.prepared_mask_binding_fingerprint == binding.fingerprint
    assert int(output.selected_global_note_indices.numel()) > 0

    loss.backward()
    gradients = tuple(
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    assert model.feature_mask_token.grad is not None
    assert bool(torch.isfinite(model.feature_mask_token.grad).all())
    assert bool(torch.count_nonzero(model.feature_mask_token.grad))


@pytest.mark.parametrize("policy", HIERARCHY_MASK_POLICIES)
def test_each_policy_inherits_post_hotfix_float32_objective_under_autocast(
    bounded_batch: SSLBatch,
    policy: str,
) -> None:
    model = _model(151 + HIERARCHY_MASK_POLICIES.index(policy)).eval()
    batch, binding = _prepared(bounded_batch, policy)

    with torch.no_grad(), torch.autocast(
        "cpu",
        dtype=torch.bfloat16,
    ):
        output = model.forward_hierarchy(
            batch,
            prepared_mask_binding=binding,
        )

    required = [
        output.note_loss.numerator,
        output.note_loss.mean,
        output.bar_latent.loss.numerator,
        output.bar_latent.loss.mean,
        output.song_latent.loss.numerator,
        output.song_latent.loss.mean,
        output.objective.total_loss,
        *(
            value
            for view in output.note_loss.view_losses
            for value in (view.loss.numerator, view.loss.mean)
        ),
    ]
    assert all(value is not None for value in required)
    assert all(
        value.dtype == torch.float32
        and bool(torch.isfinite(value))
        for value in required
        if value is not None
    )


def test_phase7_public_forward_rejects_hierarchy_profile_before_encoder(
    bounded_batch: SSLBatch,
) -> None:
    model = _model().eval()
    batch, binding = _prepared(
        bounded_batch,
        "onset_pitch_descendants",
    )
    encoder_calls = 0

    def count_encoder_calls(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
    ) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = _first_feature_encoder(model).register_forward_pre_hook(
        count_encoder_calls
    )
    try:
        with pytest.raises(
            SSLContractError,
            match="use_forward_hierarchy",
        ):
            model(batch, prepared_mask_binding=binding)
    finally:
        handle.remove()

    assert encoder_calls == 0


def test_prepare_rejects_mutated_policy_config_before_control_delegation(
    bounded_batch: SSLBatch,
) -> None:
    config = _policy_config(INDEPENDENT_NOTE_PITCH)
    object.__setattr__(config, "fingerprint", "0" * 64)

    with pytest.raises(
        SSLContractError,
        match="policy_config_non_canonical",
    ):
        prepare_hierarchy_mask_binding(
            bounded_batch,
            policy_config=config,
            global_seed=_GLOBAL_SEED,
            epoch=0,
            requested_mask_rate=_MASK_RATE,
            stage="train",
        )


@pytest.mark.parametrize("policy", HIERARCHY_MASK_POLICIES)
def test_post_prepare_mutation_fails_before_encoder_and_is_atomic(
    bounded_batch: SSLBatch,
    policy: str,
) -> None:
    model = _model().eval()
    batch, binding = _prepared(bounded_batch, policy)
    feature_tensor = batch.raw_graph_batch["note"].x_cont
    feature_tensor.add_(0.0)

    tensor_after_mutation = feature_tensor.detach().clone()
    version_after_mutation = int(feature_tensor._version)
    binding_before_failure = binding.to_dict()
    model_before_failure = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    encoder_calls = 0

    def count_encoder_calls(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
    ) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    handle = _first_feature_encoder(model).register_forward_pre_hook(
        count_encoder_calls
    )
    try:
        with pytest.raises(
            SSLContractError,
            match=r"ssl\.prepared_binding\.runtime_input_changed",
        ):
            model.forward_hierarchy(
                batch,
                prepared_mask_binding=binding,
            )
    finally:
        handle.remove()

    assert encoder_calls == 0
    assert int(feature_tensor._version) == version_after_mutation
    assert torch.equal(feature_tensor, tensor_after_mutation)
    assert binding.to_dict() == binding_before_failure
    for name, value in model.state_dict().items():
        assert torch.equal(value, model_before_failure[name])


def test_independent_control_binding_and_output_remain_bit_exact(
    bounded_batch: SSLBatch,
) -> None:
    control_binding = prepare_mask_binding(
        bounded_batch,
        global_seed=_GLOBAL_SEED,
        epoch=0,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    dispatched_binding = prepare_hierarchy_mask_binding(
        bounded_batch,
        policy_config=_policy_config(INDEPENDENT_NOTE_PITCH),
        global_seed=_GLOBAL_SEED,
        epoch=0,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )

    assert dispatched_binding.to_dict() == control_binding.to_dict()
    assert dispatched_binding.fingerprint == control_binding.fingerprint
    assert dispatched_binding.mask_plans == control_binding.mask_plans
    assert (
        dispatched_binding.feature_overlay.to_dict()
        == control_binding.feature_overlay.to_dict()
    )
    assert (
        dispatched_binding.selected_global_note_indices
        == control_binding.selected_global_note_indices
    )
    _assert_tensor_equal(
        dispatched_binding.selected_global_note_indices_tensor,
        control_binding.selected_global_note_indices_tensor,
    )

    control_batch, moved_control = move_ssl_batch_with_prepared_binding(
        bounded_batch,
        control_binding,
        "cpu",
    )
    dispatched_batch, moved_dispatched = (
        move_ssl_batch_with_prepared_binding(
            bounded_batch,
            dispatched_binding,
            "cpu",
        )
    )
    model = _model().eval()
    with torch.no_grad():
        control = model(
            control_batch,
            prepared_mask_binding=moved_control,
        )
        dispatched = model(
            dispatched_batch,
            prepared_mask_binding=moved_dispatched,
        )

    assert control.mask_plans == dispatched.mask_plans
    assert control.decoder_remask_plans == dispatched.decoder_remask_plans
    assert control.feature_overlay.to_dict() == (
        dispatched.feature_overlay.to_dict()
    )
    _assert_tensor_equal(
        control.selected_global_note_indices,
        dispatched.selected_global_note_indices,
    )
    for node_type in control.online_encoder.fused.embeddings:
        _assert_tensor_equal(
            control.online_encoder.fused.embeddings[node_type],
            dispatched.online_encoder.fused.embeddings[node_type],
        )
        _assert_tensor_equal(
            control.online_encoder.fused.batch_membership[node_type],
            dispatched.online_encoder.fused.batch_membership[node_type],
        )
    for level in ("note", "bar", "song"):
        _assert_tensor_equal(
            getattr(control.targets, level),
            getattr(dispatched.targets, level),
        )
    for left, right in zip(
        control.decoder_predictions,
        dispatched.decoder_predictions,
        strict=True,
    ):
        _assert_tensor_equal(left, right)
    for left, right in (
        (control.note_loss.numerator, dispatched.note_loss.numerator),
        (control.note_loss.mean, dispatched.note_loss.mean),
        (
            control.bar_latent.loss.numerator,
            dispatched.bar_latent.loss.numerator,
        ),
        (control.bar_latent.loss.mean, dispatched.bar_latent.loss.mean),
        (
            control.song_latent.loss.numerator,
            dispatched.song_latent.loss.numerator,
        ),
        (control.song_latent.loss.mean, dispatched.song_latent.loss.mean),
        (
            control.objective.total_loss,
            dispatched.objective.total_loss,
        ),
    ):
        assert left is not None and right is not None
        _assert_tensor_equal(left, right)


def test_target_provenance_and_diagnostics_are_end_to_end_model_blind() -> None:
    fixture = build_phase8a_hierarchy_fixture()
    source_piece = fixture.supplemental_piece
    changed_piece = replace(
        source_piece,
        source_path="ignored/phase8a-prepared-sidecar.mid",
        targets=(
            TargetArray(
                target_id="target:phase8a-prepared-inert",
                task="quality.overall",
                annotation_view_id=None,
                alignment_type="piece",
                entity_ids=(source_piece.piece_id,),
                value_type="scalar",
                class_labels=None,
                values=(0.75,),
                mask=(True,),
                confidence=(1.0,),
                source=("synthetic",),
                provenance=(
                    source_piece.provenance[0].provenance_id,
                ),
            ),
        ),
        provenance=(
            replace(
                source_piece.provenance[0],
                source="phase8a_prepared_sidecar_mutation",
                details=(("diagnostic", "changed"),),
            ),
        ),
        quality_flags=(
            QualityFlag(
                code="phase8a.test.prepared_diagnostic",
                severity="info",
                message="target-blind prepared hierarchy mutation",
                entity_ids=(source_piece.piece_id,),
                provenance_id=(
                    source_piece.provenance[0].provenance_id
                ),
            ),
        ),
    )
    source_sample = _raw_sample(source_piece)
    changed_sample = _raw_sample(changed_piece)
    source_batch = collate_ssl_samples((source_sample,))
    changed_batch = collate_ssl_samples((changed_sample,))

    assert graph_fingerprint(source_sample.raw_graph) == (
        graph_fingerprint(changed_sample.raw_graph)
    )
    assert model_input_fingerprint(source_sample.raw_graph) == (
        model_input_fingerprint(changed_sample.raw_graph)
    )
    assert _batch_graph_fingerprints(source_batch) == (
        _batch_graph_fingerprints(changed_batch)
    )

    config = _policy_config(ONSET_PITCH_DESCENDANTS)
    source_binding = prepare_hierarchy_mask_binding(
        source_batch,
        policy_config=config,
        global_seed=_GLOBAL_SEED,
        epoch=0,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    changed_binding = prepare_hierarchy_mask_binding(
        changed_batch,
        policy_config=config,
        global_seed=_GLOBAL_SEED,
        epoch=0,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    assert source_binding.to_dict() == changed_binding.to_dict()
    assert source_binding.mask_plans == changed_binding.mask_plans
    assert source_binding.feature_overlay == changed_binding.feature_overlay
    _assert_tensor_equal(
        source_binding.selected_global_note_indices_tensor,
        changed_binding.selected_global_note_indices_tensor,
    )

    moved_source, moved_source_binding = (
        move_ssl_batch_with_prepared_binding(
            source_batch,
            source_binding,
            "cpu",
        )
    )
    moved_changed, moved_changed_binding = (
        move_ssl_batch_with_prepared_binding(
            changed_batch,
            changed_binding,
            "cpu",
        )
    )
    assert moved_source_binding.to_dict() == (
        moved_changed_binding.to_dict()
    )
    assert _batch_graph_fingerprints(moved_source) == (
        _batch_graph_fingerprints(moved_changed)
    )

    source_graph_before = _batch_graph_fingerprints(moved_source)
    changed_graph_before = _batch_graph_fingerprints(moved_changed)
    source_binding_before = deepcopy(moved_source_binding.to_dict())
    changed_binding_before = deepcopy(moved_changed_binding.to_dict())
    source_indices_before = (
        moved_source_binding.selected_global_note_indices_tensor
        .detach()
        .clone()
    )
    changed_indices_before = (
        moved_changed_binding.selected_global_note_indices_tensor
        .detach()
        .clone()
    )
    source_indices_version = int(
        moved_source_binding.selected_global_note_indices_tensor._version
    )
    changed_indices_version = int(
        moved_changed_binding.selected_global_note_indices_tensor._version
    )
    model = _model(401).eval()
    model_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    with torch.no_grad():
        source_output = model.forward_hierarchy(
            moved_source,
            prepared_mask_binding=moved_source_binding,
        )
        changed_output = model.forward_hierarchy(
            moved_changed,
            prepared_mask_binding=moved_changed_binding,
        )

    assert source_output.mask_plans == changed_output.mask_plans
    assert source_output.feature_overlay == changed_output.feature_overlay
    _assert_tensor_equal(
        source_output.selected_global_note_indices,
        changed_output.selected_global_note_indices,
    )
    for node_type in source_output.online_encoder.fused.embeddings:
        _assert_tensor_equal(
            source_output.online_encoder.fused.embeddings[node_type],
            changed_output.online_encoder.fused.embeddings[node_type],
        )
        _assert_tensor_equal(
            source_output.online_encoder.fused.batch_membership[node_type],
            changed_output.online_encoder.fused.batch_membership[node_type],
        )
    for level in ("note", "bar", "song"):
        _assert_tensor_equal(
            getattr(source_output.targets, level),
            getattr(changed_output.targets, level),
        )
    for source_prediction, changed_prediction in zip(
        source_output.decoder_predictions,
        changed_output.decoder_predictions,
        strict=True,
    ):
        _assert_tensor_equal(source_prediction, changed_prediction)
    for source_value, changed_value in (
        (
            source_output.note_loss.numerator,
            changed_output.note_loss.numerator,
        ),
        (source_output.note_loss.mean, changed_output.note_loss.mean),
        (
            source_output.bar_latent.loss.numerator,
            changed_output.bar_latent.loss.numerator,
        ),
        (
            source_output.bar_latent.loss.mean,
            changed_output.bar_latent.loss.mean,
        ),
        (
            source_output.song_latent.loss.numerator,
            changed_output.song_latent.loss.numerator,
        ),
        (
            source_output.song_latent.loss.mean,
            changed_output.song_latent.loss.mean,
        ),
        (
            source_output.objective.total_loss,
            changed_output.objective.total_loss,
        ),
    ):
        assert source_value is not None
        assert changed_value is not None
        _assert_tensor_equal(source_value, changed_value)

    assert _batch_graph_fingerprints(moved_source) == source_graph_before
    assert _batch_graph_fingerprints(moved_changed) == changed_graph_before
    assert moved_source_binding.to_dict() == source_binding_before
    assert moved_changed_binding.to_dict() == changed_binding_before
    assert int(
        moved_source_binding.selected_global_note_indices_tensor._version
    ) == source_indices_version
    assert int(
        moved_changed_binding.selected_global_note_indices_tensor._version
    ) == changed_indices_version
    _assert_tensor_equal(
        moved_source_binding.selected_global_note_indices_tensor,
        source_indices_before,
    )
    _assert_tensor_equal(
        moved_changed_binding.selected_global_note_indices_tensor,
        changed_indices_before,
    )
    for name, value in model.state_dict().items():
        _assert_tensor_equal(value, model_before[name])


def test_bounded_benchmark_reports_explicit_measurement_boundaries() -> None:
    report = benchmark_phase8a_policy(
        INDEPENDENT_NOTE_PITCH,
        repeats=1,
    )

    assert report["benchmark_contract_version"] == (
        PHASE8A_MASKING_BENCHMARK_CONTRACT_VERSION
    )
    assert report["policy"] == INDEPENDENT_NOTE_PITCH
    assert report["device"] == "cpu"
    assert report["gpu_measurement"] is None
    assert report["timing_acceptance_thresholds"] is None
    assert report["finite_existing_objective_forward"] is True
    assert "not Python heap" in report["measurement_boundary"][
        "retained_plan_metadata"
    ]
    assert "O(C*K)=O(C)" in report["measurement_boundary"][
        "span_selection_complexity"
    ]
    assert "O(C+S)" in report["measurement_boundary"][
        "candidate_generation_retained_state"
    ]
    assert report["counts"]["sample_count"] == 4
    assert report["counts"]["total_nodes"] > 0
    assert report["counts"]["total_edges"] > 0
    assert (
        report["counts"]["emitted_overlay_row_field_entries"]
        > report["counts"]["primary_descendant_note_entries"]
    )
    for operation in (
        "plan_construction",
        "relation_index_construction",
        "selected_descendant_resolution",
        "overlay_construction",
        "prepared_binding_construction",
        "prepared_forward",
    ):
        timing = report["timing_seconds"][operation]
        assert timing["min"] >= 0.0
        assert timing["min"] <= timing["mean"] <= timing["max"]
    assert report["retained_metadata"][
        "batch_plan_metadata_json_bytes"
    ] > 0
    assert report["retained_metadata"][
        "peak_retained_batch_plan_metadata_json_bytes"
    ] == report["retained_metadata"][
        "batch_plan_metadata_json_bytes"
    ]
    assert report["retained_metadata"][
        "prepared_binding_public_json_bytes"
    ] > 0


def test_bounded_acceptance_publishes_each_policy_mechanics() -> None:
    report = build_phase8a_bounded_acceptance_report()

    assert report["acceptance_contract_version"] == (
        PHASE8A_BOUNDED_ACCEPTANCE_CONTRACT_VERSION
    )
    assert report["all_policies_independently_exercised"] is True
    assert report["source_batch_unchanged"] is True
    assert report["mechanics_acceptance_gates"] == {
        "finite_existing_objectives": True,
        "loss_decrease_required": False,
        "positive_correct_target_preference_required": False,
        "no_leakage_and_pitch_sensitive_evidence_are_separate": True,
    }
    assert report["cuda_measurement"] is None
    assert report["optional_cuda_hardware_evidence"]["embedded"] is False
    assert report["optional_cuda_hardware_evidence"]["status"] == (
        "pending_independent_exact_final_rtx_run"
    )
    assert report["quality_claim"] is None
    diversity = report["span_diversity_audit"]
    assert diversity["total_valid_candidate_count"] == 36
    assert diversity["tolerance_qualified_candidate_count"] == 36
    assert diversity["unique_closest_candidate_count"] == 1
    assert diversity["actual_unique_selection_count"] == 36
    assert diversity["selected_budget_error_distribution"] == {
        "0": 7,
        "1": 249,
    }
    assert diversity["epoch_range"] == [0, 255]
    assert diversity["min_selected_start_bar"] == 0
    assert diversity["max_selected_start_bar"] == 11
    assert diversity["distinct_selected_track_count"] == 3
    assert diversity["canonical_prefix_escape_count"] == 224
    assert diversity[
        "all_tolerance_candidates_entered_a_retained_pool"
    ] is True
    assert diversity["distinct_retained_pool_member_count"] == 36
    assert diversity["fresh_replay_bit_exact"] is True
    assert diversity["reverse_enumeration_bit_exact"] is True
    assert diversity["permuted_enumeration_bit_exact"] is True
    assert diversity[
        "validation_epochs_0_and_999_bit_exact"
    ] is True
    assert diversity["slack_zero_exact_best_control"][
        "actual_unique_selection_count"
    ] == 1
    default_audit = diversity["default_bounded_fixture_audit"]
    assert default_audit["all_rows_have_actual_diversity"] is True
    assert default_audit["all_rows_replay_bit_exact"] is True
    assert default_audit[
        "all_rows_within_configured_budget_slack"
    ] is True
    assert [
        {
            "policy": row["policy"],
            "sample_identity": row["sample_identity"],
            "candidate_evidence": [
                row["total_valid_candidate_count"],
                row["best_budget_error"],
                row["tolerance_candidate_count"],
                row["admissible_pool_count"],
            ],
            "actual_unique_selection_count": (
                row["actual_unique_selection_count"]
            ),
            "selected_budget_error_distribution": (
                row["selected_budget_error_distribution"]
            ),
            "configured_pool_size_limit": (
                row["configured_pool_size_limit"]
            ),
            "configured_budget_error_slack": (
                row["configured_budget_error_slack"]
            ),
            "fresh_replay_bit_exact": row["fresh_replay_bit_exact"],
            "within_slack": (
                row[
                    "all_selected_errors_within_best_plus_slack"
                ]
            ),
        }
        for row in default_audit["rows"]
    ] == [
        {
            "policy": CONTIGUOUS_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase7a-bounded",
                "piece:phase7a-train-00",
            ],
            "candidate_evidence": [5, 1, 3, 3],
            "actual_unique_selection_count": 3,
            "selected_budget_error_distribution": {"1": 256},
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": CONTIGUOUS_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase7a-bounded",
                "piece:phase7a-train-01",
            ],
            "candidate_evidence": [2, 4, 2, 2],
            "actual_unique_selection_count": 2,
            "selected_budget_error_distribution": {"4": 256},
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": CONTIGUOUS_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase7a-bounded",
                "piece:phase7a-train-02",
            ],
            "candidate_evidence": [2, 3, 2, 2],
            "actual_unique_selection_count": 2,
            "selected_budget_error_distribution": {"3": 256},
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": CONTIGUOUS_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase8a-bounded",
                "piece:phase8a-hierarchy-oracle",
            ],
            "candidate_evidence": [5, 0, 2, 2],
            "actual_unique_selection_count": 2,
            "selected_budget_error_distribution": {
                "0": 130,
                "1": 126,
            },
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": TRACK_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase7a-bounded",
                "piece:phase7a-train-00",
            ],
            "candidate_evidence": [10, 1, 10, 4],
            "actual_unique_selection_count": 10,
            "selected_budget_error_distribution": {
                "1": 100,
                "2": 156,
            },
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": TRACK_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase7a-bounded",
                "piece:phase7a-train-01",
            ],
            "candidate_evidence": [9, 1, 9, 4],
            "actual_unique_selection_count": 9,
            "selected_budget_error_distribution": {
                "1": 88,
                "2": 168,
            },
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": TRACK_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase7a-bounded",
                "piece:phase7a-train-02",
            ],
            "candidate_evidence": [6, 0, 4, 4],
            "actual_unique_selection_count": 4,
            "selected_budget_error_distribution": {"0": 256},
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
        {
            "policy": TRACK_BAR_PITCH_SPAN,
            "sample_identity": [
                "phase8a-bounded",
                "piece:phase8a-hierarchy-oracle",
            ],
            "candidate_evidence": [10, 0, 9, 4],
            "actual_unique_selection_count": 9,
            "selected_budget_error_distribution": {
                "0": 63,
                "1": 193,
            },
            "configured_pool_size_limit": 4,
            "configured_budget_error_slack": 1,
            "fresh_replay_bit_exact": True,
            "within_slack": True,
        },
    ]
    assert report["fixture"]["counts"]["train"]["piece_count"] == 4
    assert tuple(report["policies"]) == HIERARCHY_MASK_POLICIES
    for policy, evidence in report["policies"].items():
        assert evidence["policy"] == policy
        assert evidence["finite_existing_phase7a_objective"] is True
        assert len(evidence["plans"]) == 4
        assert evidence["realized_policy_frequency"] == {policy: 4}
        assert evidence[
            "realized_policy_frequency_denominator"
        ] == 4
        assert evidence["realized_policy_fractions"] == {
            policy: 1.0
        }
        assert evidence["losses"]["total"] is not None
        assert evidence["gradient_coverage"][
            "all_present_gradients_finite"
        ] is True
        assert evidence["gradient_coverage"][
            "feature_mask_token_gradient_nonzero"
        ] is True
        assert all(evidence["mutation_checks"].values())
        assert all(
            plan["primary_masked_note_count"] > 0
            and plan["visible_pitched_note_count"] > 0
            and len(plan["visible_local_note_indices"])
            == plan["visible_pitched_note_count"]
            and len(plan["selected_local_unit_indices"]) > 0
            and len(plan["primary_masked_local_note_indices"]) > 0
            for plan in evidence["plans"]
        )


def _guard_graph_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
    graph: Any,
) -> None:
    graph_tensor_ids = {
        id(value)
        for store in graph.stores
        for value in store.values()
        if isinstance(value, Tensor)
    }
    original_cpu = Tensor.cpu
    original_tolist = Tensor.tolist
    original_item = Tensor.item
    original_to = Tensor.to

    def belongs_to_graph(value: Tensor) -> bool:
        return id(value) in graph_tensor_ids

    def guarded_cpu(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        if belongs_to_graph(value):
            raise AssertionError(
                "prepared forward materialized a graph tensor with cpu()"
            )
        return original_cpu(value, *args, **kwargs)

    def guarded_tolist(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        if belongs_to_graph(value):
            raise AssertionError(
                "prepared forward materialized a graph tensor with tolist()"
            )
        return original_tolist(value, *args, **kwargs)

    def guarded_item(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> object:
        if belongs_to_graph(value):
            raise AssertionError(
                "prepared forward materialized a graph tensor with item()"
            )
        return original_item(value, *args, **kwargs)

    def guarded_to(
        value: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        requested_device = kwargs.get("device")
        if (
            requested_device is None
            and args
            and isinstance(args[0], (str, torch.device))
        ):
            requested_device = args[0]
        if (
            requested_device is not None
            and torch.device(requested_device).type == "cpu"
            and (
                belongs_to_graph(value)
                or (
                    value.device.type != "cpu"
                    and value.numel() > 1
                )
            )
        ):
            raise AssertionError(
                "prepared forward materialized a graph-sized tensor "
                "with to(cpu)"
            )
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(Tensor, "cpu", guarded_cpu)
    monkeypatch.setattr(Tensor, "tolist", guarded_tolist)
    monkeypatch.setattr(Tensor, "item", guarded_item)
    monkeypatch.setattr(Tensor, "to", guarded_to)


@pytest.mark.parametrize("policy", HIERARCHY_MASK_POLICIES)
def test_prepared_cpu_forward_has_no_graph_sized_host_materialization(
    bounded_batch: SSLBatch,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    model = _model().eval()
    batch, binding = _prepared(bounded_batch, policy)
    _guard_graph_host_materialization(
        monkeypatch,
        batch.raw_graph_batch,
    )

    with torch.no_grad():
        output = model.forward_hierarchy(
            batch,
            prepared_mask_binding=binding,
        )

    assert output.objective.total_loss is not None
    assert bool(torch.isfinite(output.objective.total_loss))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 8A prepared parity requires actual CUDA",
)
@pytest.mark.parametrize("policy", HIERARCHY_MASK_POLICIES)
def test_optional_cuda_prepared_policy_parity(
    bounded_batch: SSLBatch,
    policy: str,
) -> None:
    policy_index = HIERARCHY_MASK_POLICIES.index(policy)
    cpu_model = _model(211 + policy_index).eval()
    cuda_model = deepcopy(cpu_model).cuda().eval()
    cpu_binding = prepare_hierarchy_mask_binding(
        bounded_batch,
        policy_config=_policy_config(policy),
        global_seed=_GLOBAL_SEED,
        epoch=0,
        requested_mask_rate=_MASK_RATE,
        stage="train",
    )
    cpu_batch, moved_cpu_binding = (
        move_ssl_batch_with_prepared_binding(
            bounded_batch,
            cpu_binding,
            "cpu",
        )
    )
    cuda_batch, moved_cuda_binding = (
        move_ssl_batch_with_prepared_binding(
            bounded_batch,
            cpu_binding,
            "cuda",
        )
    )

    with torch.no_grad():
        cpu_output = cpu_model.forward_hierarchy(
            cpu_batch,
            prepared_mask_binding=moved_cpu_binding,
        )
        cuda_output = cuda_model.forward_hierarchy(
            cuda_batch,
            prepared_mask_binding=moved_cuda_binding,
        )
        cuda_repeated = cuda_model.forward_hierarchy(
            cuda_batch,
            prepared_mask_binding=moved_cuda_binding,
        )

    assert moved_cpu_binding.to_dict() == moved_cuda_binding.to_dict()
    assert cpu_output.mask_plans == cuda_output.mask_plans
    assert cpu_output.feature_overlay.to_dict() == (
        cuda_output.feature_overlay.to_dict()
    )
    assert torch.equal(
        cpu_output.selected_global_note_indices,
        cuda_output.selected_global_note_indices.cpu(),
    )
    assert _graphs_cross_device_bit_exact(
        cpu_batch.raw_graph_batch,
        cuda_batch.raw_graph_batch,
    )
    assert _outputs_bit_exact(cuda_output, cuda_repeated)
    evidence = _cross_backend_numerical_parity(
        cpu_output,
        cuda_output,
    )
    assert evidence["configured_rtol"] == (
        PHASE8A_CPU_CUDA_NUMERICAL_PARITY_RTOL
    ) == 1.0e-3
    assert evidence["configured_atol"] == (
        PHASE8A_CPU_CUDA_NUMERICAL_PARITY_ATOL
    ) == 5.0e-5
    assert evidence["configured_min_cosine_similarity"] == 0.999
    assert evidence["cross_backend_parity_passed"] is True
    assert evidence["finite_status_exact"] is True
    assert evidence["shapes_exact"] is True
    assert evidence["dtypes_exact"] is True
    assert evidence["compared_tensor_count"] > 0
    assert evidence["minimum_cosine_similarity"] <= 1.0
    assert evidence["objective_difference"] >= 0.0
    assert tuple(evidence["per_node_type"]) == tuple(
        cpu_output.online_encoder.fused.embeddings
    )
    assert all(
        node_evidence["cross_backend_parity_passed"] is True
        and node_evidence["compared_tensor_count"] == 1
        for node_evidence in evidence["per_node_type"].values()
    )
