from __future__ import annotations

from dataclasses import replace
import inspect
import json
import math
from types import SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn

from music_critic.experiments.analysisgnn.contracts import (
    COMMON_BENCHMARK_CONFIG,
    EDGE_TYPES,
    HISTORICAL_CHECKPOINT_SHA256,
    NODE_TYPES,
    PHASE9EB1_CONTRACT_VERSION,
    Phase9EB1ContractError,
    TRANSPOSITIONS,
    canonical_json,
    graph_schema_fingerprint,
)
from music_critic.experiments.analysisgnn.metrics import (
    EntryPrediction,
    JOINT_ZERO_SUPPORT_REASON,
    aggregate_entry_predictions,
    benchmark_metrics,
    grouped_bootstrap,
    summarize_seeds,
)
from music_critic.experiments.analysisgnn.graph import _note_features, _transpose_spelling
from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.model import AnalysisGNNCommonModel
from music_critic.experiments.analysisgnn.optimization import (
    TwoTaskUncertaintyLoss,
    learning_rate_at_update,
)
from music_critic.experiments.analysisgnn.run import _nvidia_smi_report
from music_critic.experiments.analysisgnn.training import (
    _append,
    evaluate_seed_test,
    evaluate_validation,
    train_seed,
    validation_objective,
)


def test_frozen_protocol_surface() -> None:
    config = COMMON_BENCHMARK_CONFIG
    assert PHASE9EB1_CONTRACT_VERSION == "1.1.1"
    assert config.seeds == (17, 23, 42)
    assert config.train_transpositions == TRANSPOSITIONS
    assert config.validation_transpositions == ("P1",)
    assert config.test_transpositions == ("P1",)
    assert (config.quality_classes, config.inversion_classes) == (50, 4)
    assert config.class_weights is None
    assert config.macro_f1_rule == "mean_f1_over_supported_true_classes"
    assert (
        config.joint_quality_inversion_alignment
        == "same_record_shared_source_entity_id"
    )
    assert (
        config.undefined_metric_reporting
        == "null_plus_availability_support_and_reason"
    )
    assert config.applied_update_budget == 10_000
    assert config.graphs_per_candidate_update == 1
    assert config.training_schedule == "seeded_shuffled_source_transposition_views"
    assert len(HISTORICAL_CHECKPOINT_SHA256) == 64
    assert len(graph_schema_fingerprint()) == 64
    assert len(EDGE_TYPES) == 13
    assert NODE_TYPES == ("note", "measure", "beat")


def test_contract_rejects_scientific_drift() -> None:
    with pytest.raises(Phase9EB1ContractError):
        replace(COMMON_BENCHMARK_CONFIG, seeds=(17,))
    with pytest.raises(Phase9EB1ContractError):
        replace(COMMON_BENCHMARK_CONFIG, class_weights="balanced")  # type: ignore[arg-type]
    with pytest.raises(Phase9EB1ContractError):
        replace(COMMON_BENCHMARK_CONFIG, validation_transpositions=("P1", "M2"))


def test_warmup_and_cosine_are_indexed_by_applied_update() -> None:
    assert learning_rate_at_update(1) == pytest.approx(0.00001)
    assert learning_rate_at_update(500) == pytest.approx(0.005)
    assert learning_rate_at_update(10_000) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        learning_rate_at_update(0)


def test_uncertainty_loss_masks_missing_labels_without_class_weights() -> None:
    objective = TwoTaskUncertaintyLoss()
    logits = {
        "quality": torch.zeros((3, 50), requires_grad=True),
        "inversion": torch.zeros((3, 4), requires_grad=True),
    }
    labels = {
        "quality": torch.tensor([0, -1, 49]),
        "inversion": torch.tensor([-1, -1, -1]),
    }
    loss, per_task = objective(logits, labels)
    assert loss is not None and torch.isfinite(loss)
    assert tuple(per_task) == ("quality",)
    loss.backward()
    assert logits["quality"].grad is not None
    assert logits["quality"].grad[1].abs().sum() == 0
    assert logits["inversion"].grad is None


def test_model_projects_each_hierarchy_with_checkpoint_attested_two_linears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("graphmuse.nn.models.metrical_gnn")

    class FakeHybridGNN(nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

    module.HybridGNN = FakeHybridGNN  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "graphmuse.nn.models.metrical_gnn", module)
    model = AnalysisGNNCommonModel()
    assert model.encoder.kwargs["hidden_channels"] == 256
    assert model.encoder.kwargs["num_layers"] == 3
    assert model.encoder.kwargs["use_jk"] is True
    for node_type in ("measure", "beat"):
        linear_layers = [
            layer for layer in model.project[node_type] if isinstance(layer, nn.Linear)
        ]
        assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
            (25, 256),
            (256, 256),
        ]


def test_nvidia_smi_environment_capture_records_driver_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout="0, GPU-deadbeef, RTX 3090, 580.65, 24576, 8.6\n",
        stderr="",
    )
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: completed)
    report = _nvidia_smi_report()
    assert report["available"] is True
    assert report["gpus"] == [
        {
            "index": "0",
            "uuid": "GPU-deadbeef",
            "name": "RTX 3090",
            "driver_version": "580.65",
            "memory_mib": "24576",
            "compute_capability": "8.6",
        }
    ]


def test_entry_aggregation_uses_mean_note_log_probability() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    rows = aggregate_entry_predictions(
        record_id="record",
        piece_id="piece",
        split="validation",
        task="inversion",
        note_logits=logits,
        note_targets=torch.tensor([0, 0, -1]),
        note_membership_index=torch.tensor([[0, 1], [0, 0]]),
        entity_ids=("span:0", "span:1"),
        entry_masks=(True, False),
    )
    assert len(rows) == 2
    assert rows[0].mask and rows[0].target == 0
    assert math.isclose(
        sum(math.exp(value) for value in rows[0].logits), 1.0, abs_tol=1e-7
    )
    assert not rows[1].mask and rows[1].prediction == -1


def test_public_is_down_beat_feature_marks_each_exact_metric_beat() -> None:
    bar = SimpleNamespace(
        start_qn=RationalTime(0), duration_qn=RationalTime(4)
    )
    piece = SimpleNamespace(
        bars=(bar,),
        beats=tuple(
            SimpleNamespace(start_qn=RationalTime(index)) for index in range(4)
        ),
    )
    on_beat = SimpleNamespace(
        onset_qn=RationalTime(1), duration_qn=RationalTime(1)
    )
    off_beat = SimpleNamespace(
        onset_qn=RationalTime(3, 2), duration_qn=RationalTime(1)
    )
    assert _note_features(piece, on_beat, 60)[2] == 1.0
    assert _note_features(piece, off_beat, 60)[2] == 0.0


def test_public_midi_wrap_does_not_change_spelling_transposition() -> None:
    assert _transpose_spelling("G", 0, 127, "M7") == ("F", 1, 10)


def _prediction(task: str, target: int, prediction: int, entity: str) -> EntryPrediction:
    classes = 50 if task == "quality" else 4
    probability = 0.9
    other = (1.0 - probability) / (classes - 1)
    logits = tuple(
        math.log(probability if index == prediction else other)
        for index in range(classes)
    )
    return EntryPrediction(
        record_id="record",
        piece_id="piece",
        split="test",
        task=task,
        entity_id=entity,
        entry_index=0,
        target=target,
        mask=True,
        logits=logits,
        prediction=prediction,
    )


def test_metrics_include_task_and_joint_surfaces() -> None:
    rows = (
        _prediction("quality", 2, 2, "span"),
        _prediction("inversion", 1, 1, "span"),
    )
    metrics = benchmark_metrics(rows)
    assert metrics["quality"]["accuracy"] == 1.0
    assert metrics["inversion"]["balanced_accuracy"] == 1.0
    assert metrics["joint_quality_inversion"] == {
        "accuracy": 1.0,
        "available": True,
        "support": 1,
        "undefined_reason": None,
    }
    assert metrics["normalized_mean_nll"] > 0.0
    assert metrics["quality"]["majority_class_baseline_accuracy"] == 1.0
    assert len(metrics["quality"]["confusion_matrix"]) == 50
    assert len(metrics["inversion"]["per_class_support"]) == 4


def test_three_seed_summary_requires_exact_seed_set() -> None:
    task = {
        "quality": {"nll": 1.0, "macro_f1": 0.5, "balanced_accuracy": 0.6, "accuracy": 0.7},
        "inversion": {"nll": 0.8, "macro_f1": 0.4, "balanced_accuracy": 0.5, "accuracy": 0.6, "majority_class_baseline_accuracy": 0.4},
        "joint_quality_inversion": {
            "accuracy": 0.3,
            "available": True,
            "support": 1,
            "undefined_reason": None,
        },
        "normalized_mean_nll": 0.4,
    }
    task["quality"]["majority_class_baseline_accuracy"] = 0.2
    summary = summarize_seeds({17: task, 23: task, 42: task})
    assert summary["quality"]["accuracy"] == {"mean": 0.7, "std": 0.0}
    assert summary["joint_quality_inversion"]["available"] is True
    assert summary["normalized_mean_nll"] == {"mean": 0.4, "std": 0.0}
    with pytest.raises(ValueError):
        summarize_seeds({17: task})


def test_grouped_bootstrap_covers_all_primary_report_surfaces() -> None:
    rows = (
        _prediction("quality", 2, 2, "span"),
        _prediction("inversion", 1, 1, "span"),
    )
    intervals = grouped_bootstrap(rows, samples=4, seed=17)
    assert "majority_class_baseline_accuracy" in intervals["quality"]
    assert intervals["joint_quality_inversion"]["accuracy"] == (1.0, 1.0)
    assert intervals["joint_quality_inversion"]["available"] is True
    assert len(intervals["normalized_mean_nll"]) == 2


def test_real_validation_task_specific_identities_report_joint_zero_support() -> None:
    record_id = (
        "an:training:wir-openscore-liedercorpus-reichardt-zwolf-gesange-"
        "op-3-09-hier-liegt-ein-spielmann-begraben"
    )
    piece_id = "piece:dilemmadata-an-169e82e0dd5a1c1b17b80514"
    rows = (
        replace(
            _prediction(
                "quality",
                2,
                2,
                "span:dilemmadata-target-169e82e0dd5a1c1b17b80514-"
                "5b0f255e6072-00000000",
            ),
            record_id=record_id,
            piece_id=piece_id,
            split="validation",
        ),
        replace(
            _prediction(
                "inversion",
                1,
                1,
                "span:dilemmadata-target-169e82e0dd5a1c1b17b80514-"
                "5a5cd8125d46-00000000",
            ),
            record_id=record_id,
            piece_id=piece_id,
            split="validation",
        ),
    )
    metrics = benchmark_metrics(rows)
    assert metrics["joint_quality_inversion"] == {
        "accuracy": None,
        "available": False,
        "support": 0,
        "undefined_reason": JOINT_ZERO_SUPPORT_REASON,
    }
    assert validation_objective(metrics) == pytest.approx(
        metrics["normalized_mean_nll"]
    )
    payload = canonical_json(
        {"applied_update": 500, "metrics": metrics, "selection_score": 0.5}
    )
    decoded = json.loads(payload)
    assert decoded["metrics"]["joint_quality_inversion"]["accuracy"] is None

    intervals = grouped_bootstrap(rows, samples=4, seed=17)
    assert intervals["joint_quality_inversion"] == {
        "accuracy": None,
        "available": False,
        "defined_sample_count": 0,
        "requested_sample_count": 4,
        "undefined_reason": "zero_joint_quality_inversion_bootstrap_support",
    }
    summary = summarize_seeds({17: metrics, 23: metrics, 42: metrics})
    assert summary["joint_quality_inversion"] == {
        "accuracy": None,
        "available": False,
        "required_seed_count": 3,
        "supporting_seed_count": 0,
        "undefined_reason": "zero_joint_quality_inversion_seed_support",
    }


def test_canonical_json_reports_nested_non_finite_path() -> None:
    with pytest.raises(Phase9EB1ContractError) as error:
        canonical_json({"metrics": {"quality": {"nll": float("nan")}}})
    assert "$.metrics.quality.nll" in str(error.value)
    assert "nan" in str(error.value)


def test_append_rejects_non_finite_before_opening_artifact(tmp_path) -> None:
    output = tmp_path / "validation.jsonl"
    with pytest.raises(Phase9EB1ContractError, match=r"\$\.metrics\.accuracy"):
        _append(output, {"metrics": {"accuracy": float("inf")}})
    assert not output.exists()


def test_validation_objective_fails_closed_on_required_non_finite_nll() -> None:
    metrics: dict[str, object] = {
        "quality": {"nll": float("nan")},
        "inversion": {"nll": 0.5},
        "joint_quality_inversion": {
            "accuracy": None,
            "available": False,
            "support": 0,
            "undefined_reason": JOINT_ZERO_SUPPORT_REASON,
        },
    }
    with pytest.raises(ValueError, match=r"\$\.quality\.nll must be finite"):
        validation_objective(metrics)


def test_training_and_locked_test_are_separate_commands() -> None:
    assert 'split="test"' not in inspect.getsource(train_seed)
    assert "remote CUDA gate" in inspect.getsource(train_seed)
    assert 'split="test"' in inspect.getsource(evaluate_seed_test)
    assert "CUDA" in inspect.getsource(evaluate_seed_test)
    assert "split" not in inspect.signature(evaluate_validation).parameters
