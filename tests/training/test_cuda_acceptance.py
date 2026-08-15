from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize
import pytest
import torch

from music_critic.device import resolve_cuda_device_index
from music_critic.graph import (
    RAW_FEATURE_REGISTRY,
    validate_raw_graph_batch,
)
from music_critic.training.config import register_training_configs
from music_critic.training.device import (
    move_multisource_batch,
    validate_device_batch,
)
from music_critic.training.engine import run_training
from music_critic.training.metrics import EpochMetricAccumulator
from music_critic.training.models import build_baseline_model


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 6C CUDA acceptance requires a CUDA runner",
)


def _small_hierarchical_config():
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="training",
            overrides=[
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=4",
                "model.dropout=0",
                "device=cuda",
            ],
        )


def _assert_batch_is_on_exact_device(batch, device: torch.device) -> None:
    for store in batch.raw_graph_batch.stores:
        for value in store.values():
            if isinstance(value, torch.Tensor):
                assert value.device == device
    for target in batch.target_batches:
        for value in (
            target.values,
            target.availability_mask,
            target.entity_indices,
            target.entity_index_mask,
            target.entity_node_type_codes,
            target.sample_indices,
            target.confidence,
            target.confidence_mask,
        ):
            if isinstance(value, torch.Tensor):
                assert value.device == device


def test_real_cuda_cli_runner_uses_amp_scaler_and_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cuda-cli"
    environment = dict(os.environ)
    source_root = str(Path.cwd() / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (source_root, environment.get("PYTHONPATH", "")),
        )
    )
    command = [
        sys.executable,
        "-m",
        "music_critic.training.run",
        "experiment=one_batch",
        "model=hierarchical",
        "data=bounded",
        "device=cuda",
        "device.amp=true",
        "experiment.steps=6",
        "model.hidden_dim=8",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=4",
        "model.dropout=0",
        f"output_dir={output}",
    ]
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    report = json.loads(
        (output / "one_batch_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["amp_enabled"] is True
    assert report["scaler_enabled"] is True
    assert report["optimizer_step_count"] == 6
    assert report["checkpoint_reload_bit_exact"] is True
    assert Path(report["checkpoint"]).is_file()
    assert all(
        math.isfinite(row[name])
        for row in report["curve"]
        for name in (
            "harmonic_loss",
            "reconstruction_loss",
            "total_loss",
        )
    )
    assert report["final"]["harmonic_loss"] < report["initial"][
        "harmonic_loss"
    ]
    assert report["final"]["reconstruction_loss"] < report["initial"][
        "reconstruction_loss"
    ]
    assert report["device"]["resolved_device"] == (
        f"cuda:{torch.cuda.current_device()}"
    )
    assert report["device"]["peak_allocated_bytes"] > 0
    assert report["device"]["peak_reserved_bytes"] > 0
    assert report["device"]["torch_version"] == torch.__version__
    assert report["device"]["cuda_runtime_version"] == torch.version.cuda
    assert (
        report["device"]["deterministic_algorithms_enabled"] is True
    )
    assert report["duration_seconds"] > 0


def test_cuda_feature_perturbation_changes_only_its_sample(
    bounded_runtime,
) -> None:
    velocity_column = RAW_FEATURE_REGISTRY.names(
        "note", "continuous"
    ).index("velocity")
    source_batch = next(
        (
            candidate
            for candidate in bounded_runtime.train_loader(0)
            if bool(
                (
                    (candidate.raw_graph_batch["note"].batch == 0)
                    & candidate.raw_graph_batch[
                        "note"
                    ].x_cont_available[:, velocity_column]
                ).any()
            )
        ),
        None,
    )
    assert source_batch is not None

    config = _small_hierarchical_config()
    batch = move_multisource_batch(
        source_batch, "cuda", non_blocking=True
    )
    _assert_batch_is_on_exact_device(
        batch,
        torch.device("cuda", torch.cuda.current_device()),
    )
    model = build_baseline_model(config.model).cuda().eval()
    with torch.no_grad():
        before = model(batch)

    note = batch.raw_graph_batch["note"]
    availability_before = note.x_cont_available[
        :, velocity_column
    ].clone()
    values_before = note.x_cont[:, velocity_column].clone()
    changed_mask = (
        (note.batch == 0)
        & note.x_cont_available[:, velocity_column]
    )
    changed_rows = torch.nonzero(
        changed_mask, as_tuple=False
    ).flatten()
    assert changed_rows.numel() > 0
    values = note.x_cont[changed_rows, velocity_column]
    note.x_cont[changed_rows, velocity_column] = torch.where(
        values < 127,
        values + 1,
        values - 1,
    )
    assert bool(
        note.x_cont[changed_rows, velocity_column].ge(0).all()
    )
    assert bool(
        note.x_cont[changed_rows, velocity_column].le(127).all()
    )
    assert torch.equal(
        note.x_cont_available[:, velocity_column],
        availability_before,
    )
    assert torch.equal(
        note.x_cont[~changed_mask, velocity_column],
        values_before[~changed_mask],
    )
    unavailable_rows = ~availability_before
    assert torch.equal(
        note.x_cont[unavailable_rows, velocity_column],
        values_before[unavailable_rows],
    )
    validate_raw_graph_batch(
        batch.raw_graph_batch,
        sample_count=len(batch.dataset_ids),
    )
    with torch.no_grad():
        after = model(batch)

    before_note = before.encoder.fused.embeddings["note"]
    after_note = after.encoder.fused.embeddings["note"]
    assert not torch.equal(
        before_note.index_select(0, changed_rows),
        after_note.index_select(0, changed_rows),
    )
    changed_own_logits = False
    for left, right in zip(
        before.predictions, after.predictions, strict=True
    ):
        sample_zero = left.sample_indices == 0
        if sample_zero.any() and not torch.equal(
            left.logits[sample_zero],
            right.logits[sample_zero],
        ):
            changed_own_logits = True
        other_samples = left.sample_indices != 0
        assert torch.equal(
            left.logits[other_samples],
            right.logits[other_samples],
        )
    assert changed_own_logits

    for node_type, embeddings in (
        before.encoder.fused.embeddings.items()
    ):
        membership = before.encoder.fused.batch_membership[node_type]
        other_samples = membership != 0
        assert torch.equal(
            embeddings[other_samples],
            after.encoder.fused.embeddings[node_type][other_samples],
        )


def _supervised_cuda_config(output: Path):
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="training",
            overrides=[
                "experiment=smoke",
                "experiment.epochs=2",
                "experiment.collect_gradient_evidence=true",
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=4",
                "model.dropout=0",
                "data=bounded",
                "data.batch_size=1",
                "data.epoch_size=6",
                "data.validation_epoch_size=0",
                "scheduler=cosine",
                "device=cuda",
                "device.amp=true",
                f"output_dir={output}",
            ],
        )


def _metric_rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _json_normalized(value):
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def test_hierarchical_cuda_supervised_resume_and_fixed_validation(
    tmp_path: Path,
) -> None:
    uninterrupted_dir = tmp_path / "cuda-uninterrupted"
    resumed_dir = tmp_path / "cuda-resumed"
    uninterrupted = run_training(
        _supervised_cuda_config(uninterrupted_dir)
    )
    first = run_training(
        _supervised_cuda_config(resumed_dir),
        stop_after_epoch=1,
    )
    assert first["completed_epochs"] == 1
    resume_config = _supervised_cuda_config(resumed_dir)
    resume_config.experiment.resume_from = str(
        resumed_dir / "last.pt"
    )
    resumed = run_training(resume_config)

    assert resumed["start_epoch"] == 1
    assert resumed["completed_epochs"] == 2
    assert resumed["amp_enabled"] is True
    assert resumed["scaler_enabled"] is True
    assert resumed["validation_membership"] == uninterrupted[
        "validation_membership"
    ]
    assert resumed["validation_membership"]["selected_count"] == 3
    assert (
        uninterrupted_dir / "metrics.jsonl"
    ).read_bytes() == (
        resumed_dir / "metrics.jsonl"
    ).read_bytes()
    checkpoint = torch.load(
        resumed_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["next_epoch"] == 2
    assert checkpoint["committed_metric_rows"] == 2
    assert checkpoint["scaler_state"]

    for row in _metric_rows(resumed_dir / "metrics.jsonl"):
        assert math.isfinite(row["train"]["objective_loss"])
        assert math.isfinite(row["validation"]["objective_loss"])
        train_evidence = row["train"]["runtime_transfer_evidence"]
        assert train_evidence["gradient_evidence_scans"] > 0
        assert train_evidence["retained_device_tensor_count"] == 0
        assert train_evidence["retained_device_tensor_bytes"] == 0
        assert (
            train_evidence[
                "metric_packed_device_to_host_transfers"
            ]
            <= row["train"]["batch_count"]
        )
        validation_evidence = row["validation"][
            "runtime_transfer_evidence"
        ]
        assert validation_evidence["retained_device_tensor_count"] == 0
        assert validation_evidence["retained_device_tensor_bytes"] == 0
        membership = row["validation"]["membership"]
        expected_membership = resumed["validation_membership"]
        assert membership["membership_fingerprint"] == (
            expected_membership["membership_fingerprint"]
        )
        assert membership["selected_count"] == expected_membership[
            "selected_count"
        ]
        assert membership["dataset_counts"] == expected_membership[
            "dataset_counts"
        ]
        assert membership["subset_limit"] == expected_membership[
            "subset_limit"
        ]
        assert membership["full_view_count"] == expected_membership[
            "full_view_count"
        ]
        assert _json_normalized(
            membership["identities"]
        ) == _json_normalized(expected_membership["identities"])
        assert _json_normalized(membership) == _json_normalized(
            expected_membership
        )
        assert row["validation"]["dataset_counts"] == (
            expected_membership["dataset_counts"]
        )


def test_cuda_metric_retention_is_constant_across_many_batches(
    bounded_batch,
) -> None:
    config = _small_hierarchical_config()
    batch = move_multisource_batch(bounded_batch, "cuda")
    model = build_baseline_model(config.model).cuda().eval()
    with torch.no_grad():
        output = model(batch)
    accumulator = EpochMetricAccumulator(
        harmonic_weight=1.0,
        reconstruction_weight=1.0,
        task_weights={},
    )
    accumulator.add(output, batch)
    cuda_device_index = resolve_cuda_device_index("cuda")
    torch.cuda.synchronize(cuda_device_index)
    steady_allocated = torch.cuda.memory_allocated(cuda_device_index)
    for _ in range(200):
        accumulator.add(output, batch)
        evidence = accumulator.storage_evidence()
        assert evidence["retained_device_tensor_count"] == 0
        assert evidence["retained_device_tensor_bytes"] == 0
    torch.cuda.synchronize(cuda_device_index)
    final_allocated = torch.cuda.memory_allocated(cuda_device_index)

    assert final_allocated <= steady_allocated
    assert accumulator.storage_evidence()["aggregate_bucket_count"] > 0


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="wrong-device validation requires at least two CUDA devices",
)
def test_wrong_cuda_index_is_rejected(
    bounded_batch,
) -> None:
    expected = torch.device("cuda:1")
    moved = move_multisource_batch(bounded_batch, expected)
    moved.raw_graph_batch["note"].x_cont = moved.raw_graph_batch[
        "note"
    ].x_cont.to("cuda:0")

    with pytest.raises(
        ValueError,
        match=r"training\.device\.graph_tensor_mismatch",
    ):
        validate_device_batch(
            moved,
            expected,
            source=bounded_batch,
        )
