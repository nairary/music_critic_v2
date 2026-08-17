from __future__ import annotations

import copy

import pytest

from music_critic.evaluation import (
    DilemmadataEvaluationError,
    build_dilemmadata_train_priors,
    evaluate_dilemmadata_model,
    make_dilemmadata_test_unlock,
    paired_component_bootstrap,
)
from music_critic.evaluation import dilemmadata as evaluation_module
from music_critic.models import DILEMMADATA_ACTIVE_TASK_IDS
from music_critic.tasks import DILEMMADATA_TARGET_ENCODING_BY_TASK
from tests.models.test_dilemmadata_heads import _batch, _model


def _components(batch):
    return {
        (dataset_id, piece_id): f"component-{index}"
        for index, (dataset_id, piece_id) in enumerate(
            zip(batch.dataset_ids, batch.piece_ids, strict=True)
        )
    }


def test_validation_metrics_priors_records_components_and_access_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    batch = _batch()
    state = {"predicted": False}
    original_predict = model.predict
    original_join = evaluation_module.join_task_supervision

    def guarded_predict(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        state["predicted"] = True
        return result

    def guarded_join(*args, **kwargs):
        assert state["predicted"] is True
        return original_join(*args, **kwargs)

    monkeypatch.setattr(model, "predict", guarded_predict)
    monkeypatch.setattr(evaluation_module, "join_task_supervision", guarded_join)
    first = evaluate_dilemmadata_model(
        model,
        (batch,),
        component_by_identity=_components(batch),
        membership_fingerprint="a" * 64,
    )
    assert first["split"] == "validation"
    assert first["counts"]["source_entry_count"] > 0
    assert first["counts"]["expanded_row_count"] >= first["counts"]["source_entry_count"]
    assert first["counts"]["record_count"] == 2
    assert first["counts"]["component_count"] == 2
    assert first["counts"]["eligible_expanded_row_count"] > 0
    assert first["counts"]["masked_row_count"] >= 0
    assert first["counts"]["conflict_row_count"] >= 0
    assert first["counts"]["unaligned_available_row_count"] >= 0
    assert set(first["tasks"]) == set(DILEMMADATA_ACTIVE_TASK_IDS)
    for task_id, metrics in first["tasks"].items():
        if not metrics["available"]:
            assert metrics["undefined_reason"] == "zero_source_entries"
            continue
        assert metrics["nll"] >= 0
        assert metrics["macro_f1_rule"] == "supported_true_classes_v1"
        assert metrics["alignment_counts"]
        assert 0 <= metrics["top1_accuracy"] <= 1
        assert metrics["record_metrics"]
        assert metrics["component_metrics"]
        if task_id.endswith(".quality"):
            assert 0 <= metrics["top3_accuracy"] <= 1
        else:
            assert metrics["top3_accuracy"] is None
            assert metrics["top3_undefined_reason"] == (
                "not_applicable_non_quality_task"
            )

    prior_rows = list(first["entry_predictions"])
    present = {row["task_id"] for row in prior_rows}
    for task_id in set(DILEMMADATA_ACTIVE_TASK_IDS) - present:
        class_count = len(
            DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary
        )
        prior_rows.append(
            {
                "task_id": task_id,
                "label": 0,
                "log_probabilities": [0.0] * class_count,
            }
        )
    priors = build_dilemmadata_train_priors(
        prior_rows,
        train_membership_fingerprint="b" * 64,
    )
    state["predicted"] = False
    second = evaluate_dilemmadata_model(
        model,
        (batch,),
        component_by_identity=_components(batch),
        membership_fingerprint="a" * 64,
        train_priors=priors,
    )
    assert second["train_prior_fingerprint"] == priors["fingerprint"]
    assert all(
        metrics["train_only_baselines"] is None
        or metrics["train_only_baselines"]["source"] == "train_only"
        for metrics in second["tasks"].values()
    )


def test_test_split_is_locked_and_unlock_is_membership_bound() -> None:
    model = _model()
    batch = _batch()
    with pytest.raises(DilemmadataEvaluationError, match="test_locked"):
        evaluate_dilemmadata_model(
            model,
            (batch,),
            component_by_identity=_components(batch),
            split="test",
            membership_fingerprint="c" * 64,
        )
    unlock = make_dilemmadata_test_unlock("c" * 64)
    report = evaluate_dilemmadata_model(
        model,
        (batch,),
        component_by_identity=_components(batch),
        split="test",
        membership_fingerprint="c" * 64,
        test_unlock=unlock,
    )
    assert report["split"] == "test"
    with pytest.raises(
        DilemmadataEvaluationError, match="unlock_binding_mismatch"
    ):
        evaluate_dilemmadata_model(
            model,
            (batch,),
            component_by_identity=_components(batch),
            split="test",
            membership_fingerprint="d" * 64,
            test_unlock=unlock,
        )


def test_bootstrap_pairs_connected_components_not_rows() -> None:
    model = _model()
    batch = _batch()
    left = evaluate_dilemmadata_model(
        model,
        (batch,),
        component_by_identity=_components(batch),
        membership_fingerprint="a" * 64,
    )
    right = copy.deepcopy(left)
    first = right["entry_predictions"][0]
    label = first["label"]
    first["log_probabilities"] = [
        0.0 if index == label else -100.0
        for index in range(len(first["log_probabilities"]))
    ]
    result = paired_component_bootstrap(
        left, right, seed=17, replicates=100
    )
    assert result["unit"] == "connected_component"
    assert result["component_count"] == 2
    assert result["replicates"] == 100
