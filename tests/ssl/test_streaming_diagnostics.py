"""Exact mergeable streaming anti-collapse diagnostics tests."""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch.nn import functional as F

from music_critic.ssl.objective import (
    ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION,
    COSINE_EPSILON,
    StreamingAntiCollapseDiagnostics,
    anti_collapse_diagnostics,
)


def _dense_side(values: torch.Tensor) -> dict[str, object]:
    values64 = values.detach().to(dtype=torch.float64)
    row_count = int(values64.shape[0])
    if row_count == 0:
        return {
            "variance": None,
            "mean_norm": None,
            "zero_norm_count": 0,
            "mean_off_diagonal_cosine": None,
        }
    norms = torch.linalg.vector_norm(values64, dim=-1)
    normalized = F.normalize(
        values64,
        dim=-1,
        eps=COSINE_EPSILON,
    )
    pairwise = normalized @ normalized.T
    off_diagonal = ~torch.eye(row_count, dtype=torch.bool)
    return {
        "variance": values64.var(
            dim=0,
            unbiased=False,
        ).mean(),
        "mean_norm": norms.mean(),
        "zero_norm_count": int((norms == 0).count_nonzero()),
        "mean_off_diagonal_cosine": (
            None
            if row_count < 2
            else pairwise[off_diagonal].mean()
        ),
    }


def _assert_scalar_close(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
) -> None:
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def _assert_reports_close(
    left: dict[str, object],
    right: dict[str, object],
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, float):
            assert isinstance(right_value, float)
            assert left_value == pytest.approx(
                right_value,
                rel=1e-14,
                abs=1e-14,
            ), key
        else:
            assert left_value == right_value, key


def test_streaming_report_matches_dense_global_oracle() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    prediction = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.25, -0.5, 2.0],
        ],
        dtype=torch.float32,
    )
    expected_target = _dense_side(target)
    expected_prediction = _dense_side(prediction)

    report = (
        StreamingAntiCollapseDiagnostics()
        .update(target[:2], prediction[:2])
        .update(target[2:4], prediction[2:4])
        .update(target[4:], prediction[4:])
        .finalize()
    )

    assert report.contract_version == "1.1.1"
    assert (
        report.contract_version
        == ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION
    )
    assert report.aggregation_scope == "streaming_aggregate"
    assert report.source_dtype == "float32"
    assert report.accumulation_dtype == "float64"
    assert report.row_count == 5
    assert report.embedding_dim == 3
    _assert_scalar_close(
        report.target_embedding_variance,
        expected_target["variance"],
    )
    _assert_scalar_close(
        report.prediction_embedding_variance,
        expected_prediction["variance"],
    )
    _assert_scalar_close(
        report.target_mean_norm,
        expected_target["mean_norm"],
    )
    _assert_scalar_close(
        report.prediction_mean_norm,
        expected_prediction["mean_norm"],
    )
    assert (
        report.target_zero_norm_count
        == expected_target["zero_norm_count"]
    )
    assert (
        report.prediction_zero_norm_count
        == expected_prediction["zero_norm_count"]
    )
    _assert_scalar_close(
        report.target_mean_off_diagonal_cosine,
        expected_target["mean_off_diagonal_cosine"],
    )
    _assert_scalar_close(
        report.prediction_mean_off_diagonal_cosine,
        expected_prediction["mean_off_diagonal_cosine"],
    )
    assert report.unavailable_reason is None
    assert report.pairwise_unavailable_reason is None


def test_partition_order_and_merge_are_numerically_invariant() -> None:
    generator = torch.Generator().manual_seed(731)
    target = torch.randn(37, 7, generator=generator)
    prediction = torch.randn(37, 7, generator=generator)
    target[11].zero_()
    prediction[29].zero_()
    baseline = (
        StreamingAntiCollapseDiagnostics()
        .update(target, prediction)
        .to_dict()
    )

    partitions = ((0, 3), (3, 17), (17, 18), (18, 31), (31, 37))
    partitioned = StreamingAntiCollapseDiagnostics()
    for start, stop in partitions:
        partitioned.update(
            target[start:stop],
            prediction[start:stop],
        )

    order = torch.randperm(
        target.shape[0],
        generator=torch.Generator().manual_seed(997),
    )
    reordered = StreamingAntiCollapseDiagnostics()
    for indices in order.split((5, 2, 13, 1, 16)):
        reordered.update(
            target.index_select(0, indices),
            prediction.index_select(0, indices),
        )

    independent = []
    for start, stop in partitions:
        independent.append(
            StreamingAntiCollapseDiagnostics().update(
                target[start:stop],
                prediction[start:stop],
            )
        )
    merged = StreamingAntiCollapseDiagnostics(embedding_dim=7)
    for partial in reversed(independent):
        merged.merge(partial)

    _assert_reports_close(partitioned.to_dict(), baseline)
    _assert_reports_close(reordered.to_dict(), baseline)
    _assert_reports_close(merged.to_dict(), baseline)


def test_empty_and_singleton_have_structured_unavailability() -> None:
    empty = StreamingAntiCollapseDiagnostics(embedding_dim=4).finalize()
    assert empty.row_count == 0
    assert empty.unavailable_reason == "no_eligible_rows"
    assert empty.pairwise_unavailable_reason == "no_eligible_rows"
    assert empty.target_embedding_variance is None
    assert empty.prediction_mean_norm is None
    assert empty.target_mean_off_diagonal_cosine is None
    assert empty.prediction_mean_off_diagonal_cosine is None

    singleton = (
        StreamingAntiCollapseDiagnostics()
        .update(
            torch.zeros(1, 4),
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        )
        .finalize()
    )
    assert singleton.row_count == 1
    assert singleton.unavailable_reason is None
    assert (
        singleton.pairwise_unavailable_reason
        == "fewer_than_two_rows"
    )
    assert singleton.target_embedding_variance is not None
    assert singleton.target_embedding_variance.item() == 0.0
    assert singleton.target_zero_norm_count == 1
    assert singleton.prediction_zero_norm_count == 0
    assert singleton.target_mean_off_diagonal_cosine is None
    assert singleton.prediction_mean_off_diagonal_cosine is None


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ],
)
def test_dtype_aware_accumulation_is_finite(dtype: torch.dtype) -> None:
    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0e-4, -2.0e-4, 3.0e-4],
            [1.0, -1.0, 0.5],
        ],
        dtype=dtype,
    )
    prediction = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-3.0e-4, 1.0e-4, 2.0e-4],
            [0.75, -0.5, 1.25],
        ],
        dtype=dtype,
    )
    report = (
        StreamingAntiCollapseDiagnostics()
        .update(target, prediction)
        .to_dict()
    )
    expected_source_dtype = (
        "float64" if dtype == torch.float64 else "float32"
    )
    assert report["source_dtype"] == expected_source_dtype
    assert report["accumulation_dtype"] == "float64"
    for key, value in report.items():
        if isinstance(value, float):
            assert math.isfinite(value), key
    assert report["target_zero_norm_count"] == 1
    assert report["prediction_zero_norm_count"] == 1


def test_non_finite_update_is_rejected_without_partial_mutation() -> None:
    accumulator = StreamingAntiCollapseDiagnostics().update(
        torch.ones(2, 3),
        torch.full((2, 3), 2.0),
    )
    before = accumulator.to_dict()
    with pytest.raises(
        ValueError,
        match="finite embedding rows",
    ):
        accumulator.update(
            torch.ones(1, 3),
            torch.tensor([[float("nan"), 0.0, 1.0]]),
        )
    assert accumulator.to_dict() == before


def test_retained_state_is_linear_and_report_is_json_compatible() -> None:
    row_count = 257
    embedding_dim = 11
    generator = torch.Generator().manual_seed(811)
    target = torch.randn(
        row_count,
        embedding_dim,
        generator=generator,
    )
    prediction = torch.randn(
        row_count,
        embedding_dim,
        generator=generator,
    )
    accumulator = StreamingAntiCollapseDiagnostics().update(
        target,
        prediction,
    )

    # Each side retains mean, M2, normalized sum, and two scalar sums:
    # 2 * (3D + 2), independent of row_count and partition count.
    assert accumulator.retained_tensor_elements == 6 * embedding_dim + 4
    report = accumulator.to_dict()
    assert json.loads(json.dumps(report, sort_keys=True)) == report
    assert tuple(report) == (
        "contract_version",
        "aggregation_scope",
        "row_count",
        "embedding_dim",
        "source_dtype",
        "accumulation_dtype",
        "target_embedding_variance",
        "prediction_embedding_variance",
        "target_mean_norm",
        "prediction_mean_norm",
        "target_zero_norm_count",
        "prediction_zero_norm_count",
        "target_mean_off_diagonal_cosine",
        "prediction_mean_off_diagonal_cosine",
        "unavailable_reason",
        "pairwise_unavailable_reason",
        "pairwise_policy",
    )


def test_existing_single_batch_api_is_preserved_at_new_version() -> None:
    report = anti_collapse_diagnostics(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.5, 0.5], [-0.5, 0.5]]),
    )
    assert (
        report.contract_version
        == ANTI_COLLAPSE_DIAGNOSTICS_CONTRACT_VERSION
        == "1.1.1"
    )
    assert report.aggregation_scope == "single_batch"
    assert report.source_dtype == "float32"
    assert report.accumulation_dtype == "input_dtype"
    assert report.to_dict()["row_count"] == 2


def test_mixed_precision_batch_and_streaming_diagnostics_normalize() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    prediction = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float16,
        requires_grad=True,
    )
    target_before = target.detach().clone()
    prediction_before = prediction.detach().clone()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        immediate = anti_collapse_diagnostics(target, prediction)
        streaming = (
            StreamingAntiCollapseDiagnostics()
            .update(target[:2], prediction[:2])
            .update(target[2:], prediction[2:])
            .finalize()
        )

    assert immediate.contract_version == "1.1.1"
    assert immediate.source_dtype == "float32"
    assert immediate.accumulation_dtype == "input_dtype"
    for value in (
        immediate.target_embedding_variance,
        immediate.prediction_embedding_variance,
        immediate.target_mean_norm,
        immediate.prediction_mean_norm,
        immediate.target_mean_off_diagonal_cosine,
        immediate.prediction_mean_off_diagonal_cosine,
    ):
        assert value is not None
        assert value.dtype == torch.float32
        assert torch.isfinite(value)
    assert immediate.target_zero_norm_count == 1
    assert immediate.prediction_zero_norm_count == 1
    assert streaming.contract_version == "1.1.1"
    assert streaming.source_dtype == "float32"
    assert streaming.accumulation_dtype == "float64"
    assert streaming.target_zero_norm_count == 1
    assert streaming.prediction_zero_norm_count == 1
    assert target.grad is None
    assert prediction.grad is None
    assert target.dtype == torch.float32
    assert prediction.dtype == torch.float16
    assert torch.equal(target.detach(), target_before)
    assert torch.equal(prediction.detach(), prediction_before)


def test_merge_rejects_incompatible_width_or_dtype() -> None:
    width_three = StreamingAntiCollapseDiagnostics().update(
        torch.ones(2, 3),
        torch.ones(2, 3),
    )
    width_four = StreamingAntiCollapseDiagnostics().update(
        torch.ones(2, 4),
        torch.ones(2, 4),
    )
    with pytest.raises(ValueError, match="widths differ"):
        width_three.merge(width_four)

    float64 = StreamingAntiCollapseDiagnostics().update(
        torch.ones(2, 3, dtype=torch.float64),
        torch.ones(2, 3, dtype=torch.float64),
    )
    before = width_three.to_dict()
    with pytest.raises(ValueError, match="source dtypes differ"):
        width_three.merge(float64)
    assert width_three.to_dict() == before
