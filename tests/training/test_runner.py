from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize
import pytest
import torch

from music_critic.tasks import collate_multisource_samples
from music_critic.training.checkpoint import capture_rng_state
from music_critic.training.config import register_training_configs
from music_critic.training.engine import (
    InjectedTrainingCrash,
    TrainingContractError,
    run_training,
)
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
                "data.validation_epoch_size=0",
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


def _artifact_snapshot(root: Path):
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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
        "run_manifest.json",
        "one_batch_report.json",
        "one_batch.pt",
    ):
        assert (tmp_path / "first" / name).is_file()
    resolved = json.loads(
        (tmp_path / "first" / "resolved_config.json").read_text()
    )
    assert resolved["experiment"]["steps"] == 6
    assert resolved["model"]["name"] == "feature_only"
    assert resolved["optimizer"]["learning_rate"] == 0.02
    assert resolved["objective"]["name"] == "one_batch_joint"
    assert resolved["objective"]["harmonic_weight"] == 1.0
    assert resolved["objective"]["reconstruction_weight"] == 1.0


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
        "committed_metric_rows",
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
    rows = [
        json.loads(line)
        for line in (resumed_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["epoch"] for row in rows] == [0, 1]
    for row in rows:
        train_transfers = row["train"]["runtime_transfer_evidence"]
        assert train_transfers["gradient_evidence_scans"] == 0
        assert (
            train_transfers[
                "metric_packed_device_to_host_transfers"
            ]
            <= row["train"]["batch_count"]
        )
        assert (
            train_transfers["metric_packed_host_materializations"]
            <= row["train"]["batch_count"]
        )
        assert train_transfers["retained_tensor_count"] == 0
        assert train_transfers["retained_device_tensor_count"] == 0
        assert train_transfers["retained_device_tensor_bytes"] == 0
        validation = row["validation"]
        assert validation["membership"]["selected_count"] == 3
        assert validation["membership"]["full_view_count"] == 3
        assert validation["membership"]["subset_limit"] == 0
        assert validation["dataset_counts"] == {
            "hooktheory": 2,
            "pop909_cl": 1,
        }
        validation_transfers = validation[
            "runtime_transfer_evidence"
        ]
        assert validation_transfers["gradient_evidence_scans"] == 0
        assert (
            validation_transfers[
                "metric_packed_device_to_host_transfers"
            ]
            <= validation["batch_count"]
        )
        assert (
            validation_transfers[
                "metric_packed_host_materializations"
            ]
            <= validation["batch_count"]
        )
        assert validation_transfers["retained_tensor_count"] == 0
        assert (
            validation_transfers["retained_device_tensor_count"] == 0
        )
        assert (
            validation_transfers["retained_device_tensor_bytes"] == 0
        )
        for task in validation["tasks"].values():
            assert task["eligible_row_count"] > 0
            assert task["loss_numerator"] >= 0


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


@pytest.mark.parametrize(
    "crash_after", ("metric_write", "checkpoint_write")
)
def test_epoch_commit_recovers_without_duplicate_or_lost_metrics(
    tmp_path: Path,
    crash_after: str,
) -> None:
    reference_dir = tmp_path / f"reference-{crash_after}"
    recovered_dir = tmp_path / f"recovered-{crash_after}"
    run_training(_config(reference_dir, "smoke"))
    run_training(
        _config(recovered_dir, "smoke"),
        stop_after_epoch=1,
    )

    crash_config = _config(recovered_dir, "smoke")
    crash_config.experiment.resume_from = str(
        recovered_dir / "last.pt"
    )
    with pytest.raises(InjectedTrainingCrash):
        run_training(crash_config, crash_after=crash_after)

    before_recovery = torch.load(
        recovered_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected_rows = 1 if crash_after == "metric_write" else 2
    assert before_recovery["committed_metric_rows"] == expected_rows
    assert len(
        (recovered_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1
    assert (recovered_dir / "epoch_metrics" / "pending.json").is_file()

    resume_config = _config(recovered_dir, "smoke")
    resume_config.experiment.resume_from = str(
        recovered_dir / "last.pt"
    )
    report = run_training(resume_config)
    assert report["completed_epochs"] == 2
    assert not (
        recovered_dir / "epoch_metrics" / "pending.json"
    ).exists()
    assert [
        path.name
        for path in sorted(
            (recovered_dir / "epoch_metrics").glob("epoch-*.json")
        )
    ] == ["epoch-0001.json", "epoch-0002.json"]
    assert (
        reference_dir / "metrics.jsonl"
    ).read_bytes() == (
        recovered_dir / "metrics.jsonl"
    ).read_bytes()
    rows = [
        json.loads(line)
        for line in (recovered_dir / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["epoch"] for row in rows] == [0, 1]


def test_fresh_run_rejects_managed_artifact_collision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collision"
    run_training(_config(output))
    before = _artifact_snapshot(output)

    with pytest.raises(
        TrainingContractError,
        match="training.output.managed_artifact_collision",
    ):
        run_training(_config(output))

    assert _artifact_snapshot(output) == before


def test_explicit_overwrite_cleans_only_managed_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "overwrite"
    run_training(_config(output, "smoke"))
    unknown_root = output / "user-notes.txt"
    unknown_root.write_text("preserve me", encoding="utf-8")
    unknown_journal = output / "epoch_metrics" / "user-data.txt"
    unknown_journal.write_text("preserve me too", encoding="utf-8")

    config = _config(output)
    config.experiment.overwrite_output = True
    report = run_training(config)

    assert report["checkpoint_reload_bit_exact"] is True
    assert unknown_root.read_text(encoding="utf-8") == "preserve me"
    assert (
        unknown_journal.read_text(encoding="utf-8")
        == "preserve me too"
    )
    for name in (
        "metrics.jsonl",
        "training_report.json",
        "last.pt",
        "best.pt",
        "epoch-0001.pt",
        "epoch-0002.pt",
    ):
        assert not (output / name).exists()
    assert (output / "one_batch.pt").is_file()
    assert (output / "run_manifest.json").is_file()


def test_incompatible_resume_preserves_rng_and_all_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "incompatible-resume"
    run_training(
        _config(output, "smoke"),
        stop_after_epoch=1,
    )
    before_artifacts = _artifact_snapshot(output)
    before_rng = capture_rng_state()
    config = _config(output, "smoke")
    config.experiment.resume_from = str(output / "last.pt")
    config.optimizer.learning_rate = 0.123

    with pytest.raises(
        TrainingContractError,
        match="training.output.run_manifest_binding_mismatch",
    ):
        run_training(config)

    _assert_state_equal(capture_rng_state(), before_rng)
    assert _artifact_snapshot(output) == before_artifacts


def test_cosine_epoch_rows_distinguish_used_and_next_lr(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cosine"
    config = _config(output, "smoke")
    config.experiment.epochs = 3
    config.scheduler.name = "cosine"
    run_training(config)
    rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert all("learning_rate" not in row for row in rows)
    assert [row["learning_rate_used"] for row in rows] == pytest.approx(
        [3e-4, 2.25e-4, 7.5e-5]
    )
    assert [row["next_learning_rate"] for row in rows] == pytest.approx(
        [2.25e-4, 7.5e-5, 0.0],
        abs=1e-12,
    )
