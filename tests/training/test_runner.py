from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize
import torch

from music_critic.tasks import collate_multisource_samples
from music_critic.training.config import register_training_configs
from music_critic.training.engine import run_training
from music_critic.training.models import build_baseline_model
from music_critic.training.device import move_multisource_batch


def _config(output: Path, experiment: str = "one_batch"):
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="training",
            overrides=[
                f"experiment={experiment}",
                "model=feature_only",
                "model.hidden_dim=8",
                "model.dropout=0",
                "data=bounded",
                "data.batch_size=3",
                "data.epoch_size=3",
                "data.validation_epoch_size=3",
                "optimizer.learning_rate=0.02",
                "device=cpu",
                f"output_dir={output}",
                *(
                    ["experiment.steps=6"]
                    if experiment == "one_batch"
                    else ["experiment.epochs=2"]
                ),
            ],
        )


def _assert_state_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


def test_one_batch_repetition_is_deterministic_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    first = run_training(_config(tmp_path / "first"))
    second = run_training(_config(tmp_path / "second"))

    assert first["initial"] == second["initial"]
    assert first["final"] == second["final"]
    assert first["candidate_counts"] == second["candidate_counts"]
    assert first["checkpoint_reload_bit_exact"] is True
    assert second["checkpoint_reload_bit_exact"] is True
    assert first["final"]["harmonic_loss"] < first["initial"][
        "harmonic_loss"
    ]
    assert first["final"]["reconstruction_loss"] < first["initial"][
        "reconstruction_loss"
    ]
    first_curve = [
        (
            row["harmonic_loss"],
            row["reconstruction_loss"],
            row["total_loss"],
        )
        for row in first["curve"]
    ]
    second_curve = [
        (
            row["harmonic_loss"],
            row["reconstruction_loss"],
            row["total_loss"],
        )
        for row in second["curve"]
    ]
    assert first_curve == second_curve
    for name in (
        "resolved_config.json",
        "fingerprints.json",
        "mixture_statistics.json",
        "one_batch_report.json",
        "one_batch.pt",
    ):
        assert (tmp_path / "first" / name).is_file()
    resolved = json.loads(
        (tmp_path / "first" / "resolved_config.json").read_text()
    )
    assert resolved["experiment"]["steps"] == 6
    assert resolved["model"]["name"] == "feature_only"


def test_epoch_boundary_resume_is_bit_exact_in_metrics(
    tmp_path: Path,
) -> None:
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted = run_training(
        _config(uninterrupted_dir, "smoke")
    )
    first_part_config = _config(resumed_dir, "smoke")
    first_part = run_training(
        first_part_config,
        stop_after_epoch=1,
    )
    assert first_part["completed_epochs"] == 1

    resume_config = _config(resumed_dir, "smoke")
    resume_config.experiment.resume_from = str(
        resumed_dir / "last.pt"
    )
    resumed = run_training(resume_config)

    assert resumed["start_epoch"] == 1
    assert resumed["completed_epochs"] == 2
    assert resumed["mid_epoch_resume_supported"] is False
    assert uninterrupted["best_validation_loss"] == resumed[
        "best_validation_loss"
    ]
    assert (
        uninterrupted_dir / "metrics.jsonl"
    ).read_bytes() == (resumed_dir / "metrics.jsonl").read_bytes()
    uninterrupted_state = torch.load(
        uninterrupted_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        resumed_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "next_epoch",
        "best_validation_loss",
        "rng_state",
    ):
        _assert_state_equal(
            uninterrupted_state[key],
            resumed_state[key],
        )
    for name in (
        "resolved_config.json",
        "fingerprints.json",
        "mixture_statistics.json",
        "metrics.jsonl",
        "training_report.json",
        "best.pt",
        "last.pt",
        "epoch-0001.pt",
        "epoch-0002.pt",
    ):
        assert (resumed_dir / name).is_file()


def test_reconstruction_trains_when_batch_has_no_harmonic_rows(
) -> None:
    # The third deterministic bounded fixture is raw-only. Its target sidecar
    # is valid but contains no available supervised harmonic rows.
    from music_critic.training.data import _bounded_samples

    train, _ = _bounded_samples()
    cpu_batch = collate_multisource_samples((train[2],))
    batch = move_multisource_batch(cpu_batch, "cpu")
    config = _config(Path("/tmp/unused-phase6c"))
    model = build_baseline_model(config.model)
    output = model(batch)
    assert output.harmonic_loss.total_loss is None
    assert output.reconstruction_loss is not None
    output.reconstruction_loss.backward()
    assert any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )
