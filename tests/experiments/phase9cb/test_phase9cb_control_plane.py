from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize
import pytest
import torch

from music_critic.experiments.phase8b2.schedule import (
    RawDownstreamSampleSchedule,
    SeedDomains,
    build_raw_downstream_sample_schedule,
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cb import contracts
from music_critic.experiments.phase9cb.contracts import (
    PHASE9CB_CELLS,
    build_plan,
    file_sha256,
)
from music_critic.experiments.phase9cb.runner import (
    _training_command,
    _write,
    aggregate,
    verify_bundle,
)
from music_critic.tasks import (
    create_split_manifest,
    dumps_corpus_index,
    dumps_dilemmadata_target_cache_index,
    dumps_split_manifest,
)
from music_critic.training import engine as training_engine
from music_critic.training.config import DataConfig, register_training_configs
from music_critic.training.data import build_corpus_data_views
from music_critic.training.engine import run_training
from tests.tasks.test_dilemmadata_target_cache import _build


class _View:
    def record_identity(self, index: int) -> tuple[str, str]:
        return "dilemmadata", f"piece-{index}"


def _plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = {}
    for name in (
        "ssl_checkpoint",
        "raw_index",
        "target_index",
        "split_manifest",
        "class_weight_artifact",
        "train_priors",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        files[name] = path
    raw_cache = tmp_path / "raw"
    target_cache = tmp_path / "target"
    raw_cache.mkdir()
    target_cache.mkdir()
    monkeypatch.setattr(
        contracts,
        "build_corpus_data_views",
        lambda config: SimpleNamespace(train=_View()),
    )

    def schedule(
        dataset,
        *,
        weights,
        seed,
        first_epoch,
        epochs,
        steps_per_epoch,
        batch_size,
    ):
        del weights
        identities = tuple(
            dataset.record_identity(index)
            for _epoch in range(first_epoch, first_epoch + epochs)
            for index in range(steps_per_epoch * batch_size)
        )
        return RawDownstreamSampleSchedule(
            seed=seed,
            first_epoch=first_epoch,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            identities=identities,
            fingerprint=raw_downstream_sample_schedule_fingerprint(
                identities
            ),
        )

    monkeypatch.setattr(
        contracts, "build_raw_downstream_sample_schedule", schedule
    )
    monkeypatch.setattr(
        contracts, "_validate_weight_artifacts", lambda *args: None
    )
    return build_plan(
        {
            **{name: str(path) for name, path in files.items()},
            "ssl_checkpoint_sha256": file_sha256(files["ssl_checkpoint"]),
            "ssl_source_kind": "phase8b_multilevel_ssl",
            "raw_cache_root": str(raw_cache),
            "target_cache_root": str(target_cache),
            "epochs": 1,
            "steps_per_epoch": 2,
            "batch_size": 2,
            "git_head": "a" * 40,
        }
    )


def test_plan_fixes_matrix_schedule_checkpoint_and_explicit_ssl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, monkeypatch)
    assert tuple(row["cell_id"] for row in plan["cells"]) == PHASE9CB_CELLS
    assert {row["decoder_kind"] for row in plan["cells"]} == {"mlp", "onset_bigru"}
    assert plan["protocol"]["seed"] == 17
    assert plan["protocol"]["schedule"]["logical_updates"] == 2
    assert plan["protocol"]["schedule"]["targets_read_for_schedule"] is False
    assert all(row["comparison_checkpoint"] == "last.pt" for row in plan["cells"])
    ssl = next(row for row in plan["cells"] if row["cell_id"] == "ssl_onset_bigru")
    command = _training_command(plan, ssl, tmp_path / "engine", profile=False, resume=False)
    assert "+model.decoder.kind=onset_bigru" in command
    assert any(row.startswith("transfer.encoder_export_path=") for row in command)
    assert any(row.startswith("transfer.source_ssl_checkpoint_sha256=") for row in command)
    assert "device.name=cuda:0" in command
    assert "seed=17" in command


def test_profile_uses_one_production_schedule_builder_for_three_real_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _corpus,
        raw_cache,
        raw_index,
        target_cache,
        target_index,
        _report,
    ) = _build(tmp_path / "fixture")
    split = create_split_manifest(
        (raw_index,),
        {
            (record.dataset_id, record.piece_id): "train"
            for record in raw_index.records
        },
        seed=17,
    )
    raw_index_path = tmp_path / "dilemmadata.index.json"
    target_index_path = tmp_path / "dilemmadata-target.index.json"
    split_path = tmp_path / "dilemmadata.split.json"
    raw_index_path.write_text(dumps_corpus_index(raw_index), encoding="utf-8")
    target_index_path.write_text(
        dumps_dilemmadata_target_cache_index(target_index), encoding="utf-8"
    )
    split_path.write_text(dumps_split_manifest(split), encoding="utf-8")
    batch_size = 1
    production_steps = 5
    profile_steps = 3
    data_order_seed = SeedDomains.create(17).downstream_data_order
    data_config = DataConfig(
        name="dilemmadata",
        index_paths=[str(raw_index_path)],
        cache_roots=[str(raw_cache.root)],
        split_manifest=str(split_path),
        target_cache_index=str(target_index_path),
        target_cache_root=str(target_cache.root),
        require_target_sidecars=True,
        batch_size=batch_size,
        workers=0,
        epoch_size=batch_size * profile_steps,
        validation_epoch_size=0,
        mixture_weights={"dilemmadata": 1.0},
    )
    views = build_corpus_data_views(data_config)
    production = build_raw_downstream_sample_schedule(
        views.train,
        weights={"dilemmadata": 1.0},
        seed=data_order_seed,
        first_epoch=0,
        epochs=1,
        steps_per_epoch=production_steps,
        batch_size=batch_size,
    )
    profile = build_raw_downstream_sample_schedule(
        views.train,
        weights={"dilemmadata": 1.0},
        seed=data_order_seed,
        first_epoch=0,
        epochs=1,
        steps_per_epoch=profile_steps,
        batch_size=batch_size,
    )
    assert production.steps_per_epoch > profile.steps_per_epoch == 3
    assert len(profile.identities) == profile_steps * batch_size
    old_bug_fingerprint = contracts.fingerprint(
        {
            "contract_version": "1.2.0",
            "kind": "raw_downstream_sample_schedule",
            "identities": [list(identity) for identity in profile.identities],
        }
    )
    assert old_bug_fingerprint != profile.fingerprint

    monkeypatch.setattr(
        training_engine,
        "_validation_epoch",
        lambda *args, **kwargs: {
            "objective_loss": 1.0,
            "dataset_counts": {},
            "batch_count": 0,
        },
    )
    register_training_configs()
    task_ids = (
        "dilemmadata.an.chord.inversion,"
        "dilemmadata.an.chord.quality,"
        "dilemmadata.dlc.chord.inversion,"
        "dilemmadata.dlc.chord.quality"
    )
    observed_by_decoder = {}
    for decoder_kind in ("mlp", "onset_bigru"):
        output = tmp_path / f"engine-{decoder_kind}"
        with initialize(version_base="1.3", config_path=None):
            config = compose(
                config_name="training",
                overrides=[
                    "experiment=dilemmadata_scratch_vs_ssl",
                    "objective=supervised_harmonic",
                    "model=hierarchical",
                    f"+model.decoder.kind={decoder_kind}",
                    "model.hidden_dim=16",
                    "model.local_gnn_layers=1",
                    "model.transformer_layers=1",
                    "model.attention_heads=4",
                    "model.ffn_multiplier=2",
                    "model.dropout=0",
                    "data=dilemmadata",
                    f"data.index_paths=[{raw_index_path}]",
                    f"data.cache_roots=[{raw_cache.root}]",
                    f"data.target_cache_index={target_index_path}",
                    f"data.target_cache_root={target_cache.root}",
                    f"data.split_manifest={split_path}",
                    f"data.batch_size={batch_size}",
                    f"data.epoch_size={batch_size * profile_steps}",
                    "data.validation_epoch_size=0",
                    "data.workers=0",
                    "experiment.steps=3",
                    "experiment.epochs=1",
                    "experiment.optimizer_steps_per_epoch=3",
                    "optimizer.learning_rate=0.0003",
                    "scheduler=none",
                    "device=cpu",
                    f"output_dir={output}",
                    "transfer.contract_version=1.2.0",
                    "transfer.mode=supervised_scratch",
                    f"transfer.comparison_protocol_fingerprint={'c' * 64}",
                    "transfer.downstream_initialization_seed=17",
                    f"transfer.downstream_data_order_seed={data_order_seed}",
                    f"transfer.sample_schedule_fingerprint={profile.fingerprint}",
                    "transfer.logical_updates=3",
                    f"downstream_task_ids=[{task_ids}]",
                ],
            )
        report = run_training(config)
        metrics = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        observed = tuple(
            tuple(identity)
            for row in metrics
            for identity in row["phase8b2_downstream_sample_identities"]
        )
        observed_by_decoder[decoder_kind] = observed
        assert report["optimizer_step_attempt_count"] == 3
        assert report["optimizer_step_applied_count"] == 3
        assert report["optimizer_step_skipped_count"] == 0
        assert observed == profile.identities
        assert report["observed_downstream_schedule_fingerprint"] == (
            profile.fingerprint
        )
        assert report["actual_sample_schedule_verified"] is True
    assert observed_by_decoder["mlp"] == observed_by_decoder["onset_bigru"]


def test_verifier_checks_four_cells_pairing_metrics_test_lock_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, monkeypatch)
    root = tmp_path / "bundle"
    _write(root / "experiment_plan.json", plan)
    _write(root / "protocol.json", plan["protocol"])
    required_metrics = {
        "normalized_nll": 1.0,
        "macro_f1": 0.1,
        "balanced_accuracy": 0.2,
        "accuracy": 0.3,
        "per_class": [],
        "confusion_matrix": [],
        "true_class_support": [],
        "predicted_class_distribution": [],
        "prediction_entropy": 0.4,
        "alignment_counts": {},
    }
    for cell in plan["cells"]:
        directory = root / "cells" / cell["cell_id"]
        directory.mkdir(parents=True)
        contract = (
            {"decoder": {"kind": "onset_bigru"}}
            if cell["decoder_kind"] == "onset_bigru"
            else {}
        )
        (directory / "engine").mkdir()
        torch.save(
            {"metadata": {"model_contract": contract}, "model_state": {}},
            directory / "engine" / "last.pt",
        )
        fresh = "bigru" if cell["decoder_kind"] == "onset_bigru" else "mlp"
        report = {
            "cell_id": cell["cell_id"],
            "decoder_kind": cell["decoder_kind"],
            "encoder_initialization": cell["encoder_initialization"],
            "schedule_fingerprint": plan["protocol"]["schedule"]["sample_schedule_fingerprint"],
            "actual_sample_schedule_fingerprint": "same",
            "checkpoint": {
                "sha256": file_sha256(directory / "engine" / "last.pt")
            },
            "validation_checkpoint_sha256": file_sha256(
                directory / "engine" / "last.pt"
            ),
            "attempted_updates": 2,
            "applied_updates": 2,
            "skipped_updates": 0,
            "fresh_supervised_initialization_fingerprint": fresh,
            "transfer": {
                "fresh_supervised_preserved_after_transfer": True,
                "source_kind": (
                    "phase8b_multilevel_ssl"
                    if cell["encoder_initialization"] == "ssl"
                    else "supervised_scratch"
                ),
                "loaded_tensors": (
                    ["local_baseline.encoder.weight"]
                    if cell["encoder_initialization"] == "ssl"
                    else []
                ),
            },
        }
        _write(directory / "cell_report.json", report)
        _write(directory / "engine" / "training_report.json", {"fingerprints": {"same": True}})
        _write(
            directory / "validation_report.json",
            {
                "tasks": {
                    f"task-{index}": dict(required_metrics)
                    for index in range(4)
                },
                "aggregate": {
                    "task_count": 4,
                    "mean_normalized_nll": 1.0,
                    "mean_macro_f1": 0.1,
                    "mean_balanced_accuracy": 0.2,
                    "mean_accuracy": 0.3,
                    "mean_prediction_entropy": 0.4,
                },
            },
        )
    aggregate_report = aggregate(root, plan)
    assert set(aggregate_report["deltas"]) == {
        "decoder_effect_under_scratch",
        "decoder_effect_under_ssl",
        "ssl_effect_with_mlp",
        "ssl_effect_with_onset_bigru",
    }
    assert verify_bundle(root, expected_sha="a" * 40)["status"] == "verified"
    (root / "cells" / "scratch_mlp" / "validation_report.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metrics_incomplete|bundle_hash_invalid"):
        verify_bundle(root, expected_sha="a" * 40)
