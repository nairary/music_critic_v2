from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import os

import pytest
import torch
from torch import Tensor

from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl import (
    DETERMINISTIC_CUDA_EVIDENCE_RUNTIME_CONTRACT_VERSION,
    PHASE8A_OUTPUT_DIFFERENCE_DIAGNOSTIC_CONTRACT_VERSION,
    PHASE8A_OUTPUT_DIFFERENCE_MAX_RETAINED,
    DeterministicCudaEvidenceRuntimeError,
    compare_phase8a_hierarchy_outputs,
    deterministic_cuda_evidence_runtime,
)
from music_critic.ssl.data import build_ssl_data_runtime
from music_critic.ssl.hierarchical_masking import (
    INDEPENDENT_NOTE_PITCH,
    HierarchyMaskPolicyConfig,
)
from music_critic.ssl.masking import (
    move_ssl_batch_with_prepared_binding,
    prepare_hierarchy_mask_binding,
)
from music_critic.ssl.model import (
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
    Phase8AHierarchySSLForwardOutput,
)
from music_critic.training.config import DataConfig


def _backend_state() -> tuple[bool, bool, bool, bool, bool, str | None]:
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        "CUBLAS_WORKSPACE_CONFIG" in os.environ,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


_INITIAL_TEST_PROCESS_BACKEND_STATE = _backend_state()


def test_deterministic_runtime_contract_is_public_and_versioned() -> None:
    assert DETERMINISTIC_CUDA_EVIDENCE_RUNTIME_CONTRACT_VERSION == "1.0.0"
    assert PHASE8A_OUTPUT_DIFFERENCE_DIAGNOSTIC_CONTRACT_VERSION == "1.0.0"
    assert PHASE8A_OUTPUT_DIFFERENCE_MAX_RETAINED == 64


def test_process_flag_isolation_polluter_is_confined_to_this_test() -> None:
    torch.use_deterministic_algorithms(
        not _INITIAL_TEST_PROCESS_BACKEND_STATE[0],
        warn_only=not _INITIAL_TEST_PROCESS_BACKEND_STATE[1],
    )
    torch.backends.cudnn.benchmark = not (
        _INITIAL_TEST_PROCESS_BACKEND_STATE[2]
    )
    torch.backends.cudnn.deterministic = not (
        _INITIAL_TEST_PROCESS_BACKEND_STATE[3]
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    assert _backend_state() != _INITIAL_TEST_PROCESS_BACKEND_STATE


def test_process_flags_are_restored_after_polluting_test() -> None:
    assert _backend_state() == _INITIAL_TEST_PROCESS_BACKEND_STATE


@pytest.mark.parametrize("raise_inside", [False, True])
def test_runtime_restores_cpu_rng_flags_workspace_and_exception_path(
    monkeypatch: pytest.MonkeyPatch,
    raise_inside: bool,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False, warn_only=False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(901)
    previous_backend = _backend_state()
    previous_rng = torch.get_rng_state().clone()

    def execute() -> None:
        with deterministic_cuda_evidence_runtime():
            assert torch.are_deterministic_algorithms_enabled() is True
            assert (
                torch.is_deterministic_algorithms_warn_only_enabled()
                is False
            )
            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cudnn.deterministic is True
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            torch.rand(5)
            if raise_inside:
                raise LookupError("intentional evidence failure")

    if raise_inside:
        with pytest.raises(LookupError, match="intentional evidence failure"):
            execute()
    else:
        execute()

    assert _backend_state() == previous_backend
    assert torch.equal(torch.get_rng_state(), previous_rng)


def test_runtime_restores_mocked_cuda_rng_and_supports_nested_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [value.clone() for value in cuda_state],
    )

    def set_rng_state_all(values: list[Tensor]) -> None:
        cuda_state[:] = [value.clone() for value in values]

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", set_rng_state_all)
    torch.manual_seed(907)
    outer_cpu_state = torch.get_rng_state().clone()
    outer_cuda_state = cuda_state[0].clone()

    with deterministic_cuda_evidence_runtime():
        torch.rand(3)
        cuda_state[0].add_(4)
        inner_cpu_state = torch.get_rng_state().clone()
        inner_cuda_state = cuda_state[0].clone()
        with deterministic_cuda_evidence_runtime():
            torch.rand(7)
            cuda_state[0].add_(9)
        assert torch.equal(torch.get_rng_state(), inner_cpu_state)
        assert torch.equal(cuda_state[0], inner_cuda_state)

    assert torch.equal(torch.get_rng_state(), outer_cpu_state)
    assert torch.equal(cuda_state[0], outer_cuda_state)

    with deterministic_cuda_evidence_runtime():
        first = torch.rand(4)
    with deterministic_cuda_evidence_runtime():
        second = torch.rand(4)
    assert torch.equal(first, second)


def test_runtime_rejects_invalid_workspace_before_state_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    before = _backend_state()
    rng = torch.get_rng_state().clone()
    with pytest.raises(
        DeterministicCudaEvidenceRuntimeError,
        match="cublas_workspace_config_invalid",
    ):
        with deterministic_cuda_evidence_runtime():
            raise AssertionError("unreachable")
    assert _backend_state() == before
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.fixture(scope="module")
def hierarchy_output() -> Phase8AHierarchySSLForwardOutput:
    batch = build_ssl_data_runtime(DataConfig(), seed=42).first_train_batch
    torch.manual_seed(919)
    model = MaskedGraphSSLModel(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        ),
        MaskedGraphSSLConfig(
            mask_rate=0.30,
            decoder_views=1,
            decoder_remask_probability=0.0,
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    ).eval()
    binding = prepare_hierarchy_mask_binding(
        batch,
        policy_config=HierarchyMaskPolicyConfig.create(
            weights={INDEPENDENT_NOTE_PITCH: 1.0},
            min_span_bars=1,
            max_span_bars=2,
        ),
        global_seed=42,
        epoch=0,
        requested_mask_rate=0.30,
        stage="train",
    )
    moved_batch, moved_binding = move_ssl_batch_with_prepared_binding(
        batch,
        binding,
        "cpu",
    )
    with torch.no_grad():
        return model.forward_hierarchy(
            moved_batch,
            prepared_mask_binding=moved_binding,
        )


def test_output_comparator_reports_equal_outputs_without_tensors(
    hierarchy_output: Phase8AHierarchySSLForwardOutput,
) -> None:
    diagnostic = compare_phase8a_hierarchy_outputs(
        hierarchy_output,
        deepcopy(hierarchy_output),
    )
    assert diagnostic.bit_exact is True
    assert diagnostic.first_difference_path is None
    assert diagnostic.total_difference_count == 0
    assert diagnostic.differences == ()


def test_output_comparator_reports_bounded_paths_groups_and_tensor_metrics(
    hierarchy_output: Phase8AHierarchySSLForwardOutput,
) -> None:
    changed = deepcopy(hierarchy_output)
    changed.online_encoder.fused.embeddings["note"][0, 0] = torch.nextafter(
        changed.online_encoder.fused.embeddings["note"][0, 0],
        torch.tensor(float("inf")),
    )
    changed.decoder_predictions[0][0, 0] += 0.25
    changed.targets.note[0, 0] += 0.5
    assert changed.objective.total_loss is not None
    changed.objective.total_loss.add_(0.75)

    diagnostic = compare_phase8a_hierarchy_outputs(
        hierarchy_output,
        changed,
    )
    report = diagnostic.to_dict()
    assert diagnostic.bit_exact is False
    assert diagnostic.first_difference_path is not None
    counts = dict(diagnostic.difference_count_by_group)
    assert counts["embeddings"] >= 1
    assert counts["predictions"] >= 1
    assert counts["targets"] >= 1
    assert counts["loss_tensors"] >= 1
    assert report["retained_path_by_group"]["embeddings"]
    assert report["retained_path_by_group"]["predictions"]
    assert report["retained_path_by_group"]["targets"]
    assert report["retained_path_by_group"]["loss_tensors"]

    tensor_differences = [
        value
        for value in diagnostic.differences
        if value.kind == "tensor"
    ]
    assert tensor_differences
    for difference in tensor_differences:
        assert difference.left_shape == difference.right_shape
        assert difference.left_dtype == difference.right_dtype
        assert difference.left_device == difference.right_device == "cpu"
        assert difference.different_element_count is not None
        assert difference.different_element_count > 0
        assert difference.max_absolute_difference is not None
        assert difference.max_relative_difference is not None
    embedding_difference = next(
        value for value in tensor_differences if value.group == "embeddings"
    )
    assert embedding_difference.max_ulp_difference == 1
    assert not any(
        isinstance(getattr(difference, field.name), Tensor)
        for difference in diagnostic.differences
        for field in fields(difference)
    )


def test_output_comparator_truncates_retained_evidence_but_counts_all(
    hierarchy_output: Phase8AHierarchySSLForwardOutput,
) -> None:
    changed = deepcopy(hierarchy_output)
    changed.decoder_predictions[0][0, 0] += 1.0
    changed.targets.note[0, 0] += 1.0
    diagnostic = compare_phase8a_hierarchy_outputs(
        hierarchy_output,
        changed,
        retained_limit=1,
    )
    assert diagnostic.total_difference_count >= 2
    assert diagnostic.retained_difference_count == 1
    assert diagnostic.truncated is True
    assert diagnostic.first_difference_path == diagnostic.differences[0].path


def test_output_comparator_reports_fp16_ulp_evidence(
    hierarchy_output: Phase8AHierarchySSLForwardOutput,
) -> None:
    first_prediction = hierarchy_output.decoder_predictions[0].to(
        torch.float16
    )
    second_prediction = first_prediction.clone()
    second_prediction[0, 0] = torch.nextafter(
        second_prediction[0, 0],
        torch.tensor(float("inf"), dtype=torch.float16),
    )
    left = replace(
        hierarchy_output,
        decoder_predictions=(first_prediction,),
    )
    right = replace(
        hierarchy_output,
        decoder_predictions=(second_prediction,),
    )
    diagnostic = compare_phase8a_hierarchy_outputs(left, right)
    difference = next(
        value
        for value in diagnostic.differences
        if value.path == "output.decoder_predictions[0]"
    )
    assert difference.left_dtype == "torch.float16"
    assert difference.max_ulp_difference == 1


def test_output_comparator_rejects_unbounded_retention(
    hierarchy_output: Phase8AHierarchySSLForwardOutput,
) -> None:
    with pytest.raises(ValueError, match="bounded contract"):
        compare_phase8a_hierarchy_outputs(
            hierarchy_output,
            hierarchy_output,
            retained_limit=65,
        )
