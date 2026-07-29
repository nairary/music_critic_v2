from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf
import pytest
import torch

from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.data import build_ssl_data_runtime
from music_critic.ssl.engine import _evaluate, run_ssl_training
from music_critic.ssl.model import build_ssl_model


# Performance rows and the final report intentionally contain wall-clock
# measurements. These are the byte-stable semantic artifacts instead.
_DETERMINISTIC_ARTIFACTS = (
    "resolved_config.json",
    "fingerprints.json",
    "run_manifest.json",
    "initial_validation.json",
    "metrics.jsonl",
)


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _config(output: Path):
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="ssl_training",
            overrides=[
                "experiment=pretrain",
                "experiment.epochs=3",
                "experiment.validation_interval=1",
                "experiment.collect_gradient_evidence=false",
                "experiment.overwrite_output=true",
                "model=hierarchical",
                "data=bounded",
                "data.batch_size=3",
                "data.epoch_size=3",
                "data.validation_epoch_size=2",
                "data.workers=0",
                "device=cpu",
                "device.amp=false",
                "scheduler=none",
                "optimizer.learning_rate=0.0003",
                "ssl.mask_rate=0.30",
                "ssl.decoder_views=3",
                "ssl.decoder_remask_prob=0.20",
                "ssl.projector_hidden_dim=128",
                "ssl.decoder_hidden_dim=128",
                "seed=42",
                f"output_dir={output}",
            ],
        )


def _metric_rows(output: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _artifact_bytes(output: Path) -> dict[str, bytes]:
    return {
        name: (output / name).read_bytes()
        for name in _DETERMINISTIC_ARTIFACTS
    }


def _checkpoint_payloads(output: Path) -> dict[str, object]:
    return {
        name: torch.load(
            output / name,
            map_location="cpu",
            weights_only=True,
        )
        for name in ("last.pt", "best.pt")
    }


def _assert_state_equal(left: object, right: object) -> None:
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


def _assert_noncollapsed(metric: dict[str, object]) -> None:
    acceptance = metric["non_collapse_acceptance"]
    assert acceptance["passed"] is True
    assert set(acceptance["levels"]) == {"note", "bar", "song"}
    assert all(
        level["passed"] is True
        for level in acceptance["levels"].values()
    )


def _assert_diagnostic_aggregate_close(
    left: dict[str, object],
    right: dict[str, object],
) -> None:
    assert left.keys() == right.keys()
    for level in left:
        left_level = left[level]
        right_level = right[level]
        assert left_level.keys() == right_level.keys()
        for key, left_value in left_level.items():
            right_value = right_level[key]
            if isinstance(left_value, float):
                assert right_value == pytest.approx(
                    left_value,
                    rel=1e-12,
                    abs=1e-12,
                ), (level, key)
            else:
                assert right_value == left_value, (level, key)


def _assert_best_checkpoint_is_validation_minimum(
    output: Path,
    report: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    validation_losses = [
        (
            int(row["epoch"]),
            float(row["validation"]["total_ssl_loss"]),
        )
        for row in rows
    ]
    expected_epoch, expected_loss = min(
        validation_losses,
        key=lambda item: (item[1], item[0]),
    )
    assert report["best_checkpoint_selection"] == (
        "minimum_fixed_validation_total_ssl_loss"
    )
    assert report["best_validation_loss"] == expected_loss
    assert report["best_validation_epoch"] == expected_epoch
    assert Path(report["best_checkpoint"]) == output / "best.pt"

    best = torch.load(
        output / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    last = torch.load(
        output / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert best["best_validation_loss"] == expected_loss
    assert best["next_epoch"] == expected_epoch + 1
    assert best["epoch_journal"][-1]["epoch"] == expected_epoch
    assert best["epoch_journal"][-1]["validation"][
        "total_ssl_loss"
    ] == expected_loss
    assert last["next_epoch"] == report["configured_epochs"]
    assert last["best_validation_loss"] == expected_loss


def test_fresh_overwrite_rerun_has_deterministic_held_out_acceptance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "held-out-fresh-rerun"
    config = _config(output)
    assert (
        config.model.hidden_dim,
        config.model.local_gnn_layers,
        config.model.transformer_layers,
        config.model.attention_heads,
        config.model.ffn_multiplier,
        config.model.dropout,
    ) == (128, 3, 2, 4, 4, 0.1)
    assert config.optimizer.name == "adamw"
    assert config.optimizer.learning_rate == 3e-4
    assert config.experiment.epochs == 3

    first = run_ssl_training(config)
    first_rows = _metric_rows(output)
    first_artifacts = _artifact_bytes(output)
    first_checkpoints = _checkpoint_payloads(output)

    assert first["evidence_kind"] == (
        "bounded_phase7a_ssl_held_out_noncollapse"
    )
    assert first["completed_epochs"] == first["configured_epochs"] == 3
    assert first["device"]["resolved_device"] == "cpu"
    assert first["amp_enabled"] is False
    assert first["scaler_enabled"] is False
    assert len(first_rows) == 3
    assert all(row["validation"] is not None for row in first_rows)

    held_out = first["held_out_acceptance"]
    assert held_out["scope"] == (
        "bounded_held_out_mechanics_non_collapse_only"
    )
    assert held_out["passed"] is True
    assert held_out["initial_measurement_before_optimizer"] is True
    assert held_out["initial_optimizer_step_count"] == 0
    assert held_out["validation_mask_epoch"] == 0
    assert held_out["validation_checkpoint_selection_only"] is True
    assert held_out["checkpoint_selection_metric"] == (
        "minimum_fixed_validation_total_ssl_loss"
    )
    assert held_out["configured_epochs"] == 3
    assert held_out["completed_epochs"] == 3
    assert held_out["multiple_epochs"] is True
    assert held_out["trajectory_complete"] is True
    assert held_out["every_epoch_has_validation"] is True
    assert held_out["finite_losses"] is True
    assert held_out["finite_aggregate_diagnostics"] is True
    assert held_out["finite_losses_and_diagnostics"] is True
    assert held_out["non_collapse_checks_passed"] is True
    assert held_out["fixed_validation_plans_across_trajectory"] is True
    assert held_out[
        "fixed_validation_bindings_across_trajectory"
    ] is True
    assert held_out["fixed_validation_plan_fingerprints"]
    assert held_out[
        "fixed_validation_prepared_binding_fingerprints"
    ]
    assert len(held_out["epoch_trajectory"]) == 3
    assert held_out["effectiveness_claim"] is False

    _assert_noncollapsed(first["initial_validation"])
    assert (
        first["initial_validation"][
            "optimizer_step_count_at_measurement"
        ]
        == 0
    )
    assert [
        row["validation"]["optimizer_step_count_at_measurement"]
        for row in first_rows
    ] == [1, 2, 3]
    _assert_noncollapsed(first_rows[-1]["validation"])
    initial_plans = first["initial_validation"]["masking"][
        "plan_fingerprints"
    ]
    initial_bindings = first["initial_validation"]["masking"][
        "prepared_mask_binding_fingerprints"
    ]
    assert all(
        row["validation"]["masking"]["plan_fingerprints"]
        == initial_plans
        for row in first_rows
    )
    assert all(
        row["validation"]["masking"][
            "prepared_mask_binding_fingerprints"
        ]
        == initial_bindings
        for row in first_rows
    )
    _assert_best_checkpoint_is_validation_minimum(
        output,
        first,
        first_rows,
    )

    second = run_ssl_training(_config(output))
    second_rows = _metric_rows(output)
    second_artifacts = _artifact_bytes(output)
    second_checkpoints = _checkpoint_payloads(output)

    assert second["held_out_acceptance"] == held_out
    assert second["initial_validation"] == first["initial_validation"]
    assert second_rows == first_rows
    assert second_artifacts == first_artifacts
    _assert_state_equal(second_checkpoints, first_checkpoints)
    _assert_best_checkpoint_is_validation_minimum(
        output,
        second,
        second_rows,
    )


def test_emitted_aggregate_is_batch_partition_and_order_invariant(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "partition-invariance")
    config.data.batch_size = 2
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    runtime_two = build_ssl_data_runtime(config.data, seed=config.seed)

    config.data.batch_size = 1
    runtime_one = build_ssl_data_runtime(config.data, seed=config.seed)
    torch.manual_seed(config.seed)
    model = build_ssl_model(config.model, config.ssl).eval()
    device = torch.device("cpu")

    combined = _evaluate(
        model,
        runtime_two.validation_loader(),
        config=resolved,
        device=device,
        epoch=0,
    )
    individual_batches = tuple(runtime_one.validation_loader())
    partitioned = _evaluate(
        model,
        individual_batches,
        config=resolved,
        device=device,
        epoch=0,
    )
    reversed_partitioned = _evaluate(
        model,
        tuple(reversed(individual_batches)),
        config=resolved,
        device=device,
        epoch=0,
    )

    assert combined["sample_count"] == partitioned["sample_count"] == 2
    assert (
        combined["masking"]["plan_fingerprints"]
        == partitioned["masking"]["plan_fingerprints"]
        == reversed_partitioned["masking"]["plan_fingerprints"]
    )
    _assert_diagnostic_aggregate_close(
        combined["anti_collapse_aggregate"],
        partitioned["anti_collapse_aggregate"],
    )
    _assert_diagnostic_aggregate_close(
        combined["anti_collapse_aggregate"],
        reversed_partitioned["anti_collapse_aggregate"],
    )
