from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize
import pytest
import torch

from music_critic.models import DILEMMADATA_ACTIVE_TASK_IDS, class_weight_artifact
from music_critic.tasks import DILEMMADATA_TARGET_ENCODING_BY_TASK
from music_critic.training.config import register_training_configs
from music_critic.training.data import DataRuntime, ValidationMembership
from music_critic.training.engine import run_training
from music_critic.training import engine as training_engine
from tests.models.test_dilemmadata_heads import _batch


def _compose(*overrides: str):
    register_training_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="training",
            overrides=list(overrides),
        )


def _runtime() -> DataRuntime:
    batch = _batch()
    membership = ValidationMembership(
        identities=tuple(zip(batch.dataset_ids, batch.piece_ids, strict=True)),
        membership_fingerprint="v" * 64,
        dataset_counts={"dilemmadata": 2},
        full_view_count=2,
        selected_count=2,
        subset_limit=0,
    )
    return DataRuntime(
        first_train_batch=batch,
        train_loader=lambda epoch: (batch,),
        validation_loader=lambda: (batch,),
        validation_membership=membership,
        fingerprints={
            "raw_index_fingerprint": "r" * 64,
            "target_cache_index_fingerprint": "t" * 64,
        },
        mixture_statistics={"requested_weights": {"dilemmadata": 1.0}},
    )


def test_required_cli_preset_composes_exact_dilemmadata_pipeline() -> None:
    config = _compose(
        "experiment=dilemmadata_pilot",
        "model=hierarchical",
        "data=dilemmadata",
        "device=cuda",
    )
    assert config.experiment.name == "dilemmadata_pilot"
    assert config.model.name == "hierarchical"
    assert config.data.name == "dilemmadata"
    assert config.data.require_target_sidecars is True
    assert config.device.name == "cuda"
    assert config.experiment.default_reconstruction_weight == 0


def test_dilemmadata_one_batch_updates_heads_encoder_and_reloads_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        training_engine, "build_data_runtime", lambda config, seed: runtime
    )

    config = _compose(
        "experiment=dilemmadata_one_batch",
        "model=hierarchical",
        "data=dilemmadata",
        "device=cpu",
        "model.hidden_dim=16",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=4",
        "model.ffn_multiplier=2",
        "model.dropout=0",
        "experiment.steps=20",
        "optimizer.learning_rate=0.02",
        f"output_dir={tmp_path / 'output'}",
    )
    report = run_training(config)
    assert report["checkpoint_reload_bit_exact"] is True
    assert report["initial"]["reconstruction_loss"] is None
    assert report["final"]["reconstruction_loss"] is None
    assert report["final"]["harmonic_loss"] < report["initial"]["harmonic_loss"]
    assert report["phase8b2_transfer"]["supervised_heads_transferred"] is False
    assert report["phase8b2_transfer"]["ssl_heads_transferred"] is False
    assert set(report["candidate_counts"]) == set(DILEMMADATA_ACTIVE_TASK_IDS)
    assert report["fingerprints"]["target_cache_index_fingerprint"] == "t" * 64


def test_dilemmadata_one_batch_accepts_train_only_class_weight_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        training_engine, "build_data_runtime", lambda config, seed: runtime
    )
    artifact = class_weight_artifact(
        {
            task_id: tuple(
                0 if index == 0 else index
                for index, _ in enumerate(
                    DILEMMADATA_TARGET_ENCODING_BY_TASK[task_id].vocabulary
                )
            )
            for task_id in DILEMMADATA_ACTIVE_TASK_IDS
        },
        policy="inverse_sqrt_frequency_supported",
        train_membership_fingerprint="a" * 64,
    )
    artifact_path = tmp_path / "class_weights.json"
    import json

    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    config = _compose(
        "experiment=dilemmadata_one_batch",
        "model=hierarchical",
        "data=dilemmadata",
        "device=cpu",
        "model.hidden_dim=16",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=4",
        "model.ffn_multiplier=2",
        "model.dropout=0",
        "experiment.steps=2",
        "optimizer.learning_rate=0.02",
        f"objective.class_weight_artifact_path={artifact_path}",
        f"output_dir={tmp_path / 'output'}",
    )
    report = run_training(config)
    assert report["checkpoint_reload_bit_exact"] is True
    resolved = json.loads((tmp_path / "output" / "resolved_config.json").read_text())
    assert resolved["objective"]["class_weight_evidence"]["source_split"] == "train_only"
    assert resolved["objective"]["class_weight_evidence"]["amp_loss_scaling"] == {
        "initial_scale": 1.0,
        "growth_interval": 2**31 - 1,
        "reason": "rare_class_gradient_overflow_prevention",
    }


def test_dilemmadata_epoch_checkpoint_resume_is_bit_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime()
    monkeypatch.setattr(
        training_engine, "build_data_runtime", lambda config, seed: runtime
    )

    def config(output: Path):
        return _compose(
            "experiment=dilemmadata_smoke",
            "model=hierarchical",
            "data=dilemmadata",
            "device=cpu",
            "model.hidden_dim=16",
            "model.local_gnn_layers=1",
            "model.transformer_layers=1",
            "model.attention_heads=4",
            "model.ffn_multiplier=2",
            "model.dropout=0",
            "experiment.epochs=2",
            "experiment.steps=1",
            "optimizer.learning_rate=0.001",
            f"output_dir={output}",
        )

    uninterrupted_path = tmp_path / "uninterrupted"
    resumed_path = tmp_path / "resumed"
    run_training(config(uninterrupted_path))
    run_training(config(resumed_path), stop_after_epoch=1)
    resume = config(resumed_path)
    resume.experiment.resume_from = str(resumed_path / "last.pt")
    report = run_training(resume)
    assert report["start_epoch"] == 1
    assert report["completed_epochs"] == 2
    assert (uninterrupted_path / "metrics.jsonl").read_bytes() == (
        resumed_path / "metrics.jsonl"
    ).read_bytes()
    left = torch.load(
        uninterrupted_path / "last.pt", map_location="cpu", weights_only=True
    )["model_state"]
    right = torch.load(
        resumed_path / "last.pt", map_location="cpu", weights_only=True
    )["model_state"]
    assert set(left) == set(right)
    assert all(torch.equal(left[name], right[name]) for name in left)
