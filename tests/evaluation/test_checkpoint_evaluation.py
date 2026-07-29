from __future__ import annotations

from dataclasses import replace
import json
import random
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from music_critic.evaluation.checkpoint import load_evaluation_checkpoint
from music_critic.evaluation.config import (
    EvaluationConfig,
    EvaluationDataConfig,
    EvaluationDeviceConfig,
)
from music_critic.evaluation.contracts import EvaluationContractError
from music_critic.evaluation.data import build_evaluation_data_runtime
from music_critic.evaluation.engine import _resolve_device, run_evaluation
from music_critic.models import LocalBaselineConfig, LocalHeterogeneousBaseline
from music_critic.training.checkpoint import save_training_checkpoint


def _checkpoint(tmp_path: Path) -> Path:
    runtime = build_evaluation_data_runtime(
        OmegaConf.structured(EvaluationDataConfig()),
        split="validation",
        seed=42,
    )
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(
            variant="feature_only",
            hidden_dim=8,
            gnn_layers=0,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        committed_metric_rows=0,
        resolved_config={},
        data_fingerprints={
            "kind": "bounded",
            "validation_membership_fingerprint": runtime.bindings[
                "evaluation_membership_fingerprint"
            ],
        },
    )
    return path


def _config(checkpoint: Path, output: Path) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint=str(checkpoint),
        output_dir=str(output),
        data=EvaluationDataConfig(),
        device=EvaluationDeviceConfig(),
    )


def test_evaluation_checkpoint_load_preserves_rng_and_ignores_training_state(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    random.seed(991)
    torch.manual_seed(991)
    python_before = random.getstate()
    torch_before = torch.get_rng_state().clone()

    _model, evidence = load_evaluation_checkpoint(checkpoint)

    assert random.getstate() == python_before
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert evidence["optimizer_state_loaded"] is False
    assert evidence["checkpoint_rng_state_loaded"] is False


def test_evaluation_runtime_resolves_abstract_cuda_to_current_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    resolved = _resolve_device(
        {"device": {"name": "cuda"}}
    )

    assert resolved == torch.device("cuda:1")


def test_direct_evaluation_checkpoint_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        EvaluationContractError,
        match=r"^evaluation\.device\.cuda_unavailable$",
    ):
        load_evaluation_checkpoint(checkpoint, device="cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bounded evaluation CUDA runtime requires CUDA",
)
def test_bounded_evaluation_uses_concrete_cuda_runtime(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    config = _config(checkpoint, tmp_path / "cuda-evaluation")
    config.device = EvaluationDeviceConfig(name="cuda")

    report = run_evaluation(config)

    assert report["status"] == "completed"
    assert report["sample_count"] > 0


def test_repeated_checkpoint_evaluation_is_bit_exact(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    run_evaluation(_config(checkpoint, first))
    run_evaluation(_config(checkpoint, second))

    for name in (
        "checkpoint_evidence.json",
        "train_priors.json",
        "metrics.json",
        "evaluation_report.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_dataset_and_task_results_are_source_native(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "isolated"
    run_evaluation(_config(checkpoint, output))
    metrics = json.loads(
        (output / "metrics.json").read_text(encoding="utf-8")
    )

    assert set(metrics["datasets"]) == {"hooktheory", "pop909_cl"}
    assert all(
        task_id.startswith("theory.")
        for task_id in metrics["datasets"]["hooktheory"]
    )
    assert all(
        task_id.startswith("pop909_cl.")
        for task_id in metrics["datasets"]["pop909_cl"]
    )
    summaries = metrics["macro_summaries"]
    assert summaries["cross_dataset_aggregation"] is False
    assert summaries["cross_encoding_aggregation"] is False
    assert {
        (group["dataset_id"], group["encoding_kind"])
        for group in summaries["groups"]
    } == {
        ("hooktheory", "closed_categorical_index"),
        ("hooktheory", "closed_multilabel"),
        ("pop909_cl", "closed_categorical_index"),
    }


def test_test_split_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    config = _config(tmp_path / "not-read.pt", tmp_path / "result")
    config.split = "test"
    with pytest.raises(
        EvaluationContractError,
        match="acknowledgement_required",
    ):
        run_evaluation(config)


def test_candidate_logits_do_not_depend_on_target_values(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    model, _evidence = load_evaluation_checkpoint(checkpoint)
    runtime = build_evaluation_data_runtime(
        OmegaConf.structured(EvaluationDataConfig()),
        split="validation",
        seed=42,
    )
    batch = next(iter(runtime.evaluation_loader()))
    target = batch.target_batches[0]
    mutated_values = (
        torch.zeros_like(target.values)
        if target.values.dtype == torch.bool
        else torch.remainder(target.values + 1, 2)
    )
    mutated = replace(
        batch,
        target_batches=(
            replace(target, values=mutated_values),
            *batch.target_batches[1:],
        ),
    )

    with torch.no_grad():
        first = model.predict(batch.raw_graph_batch)[1]
        second = model.predict(mutated.raw_graph_batch)[1]
    assert all(
        torch.equal(left.logits, right.logits)
        for left, right in zip(first, second, strict=True)
    )
