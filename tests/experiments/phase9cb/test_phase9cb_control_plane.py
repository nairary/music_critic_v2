from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from music_critic.experiments.phase9cb import contracts
from music_critic.experiments.phase9cb.contracts import PHASE9CB_CELLS, build_plan, file_sha256
from music_critic.experiments.phase9cb.runner import (
    _training_command,
    _write,
    aggregate,
    verify_bundle,
)


class _View:
    def record_identity(self, index: int) -> tuple[str, str]:
        return "dilemmadata", f"piece-{index}"


class _Sampler:
    def __init__(self, dataset, *, weights, seed, epoch_size):
        del dataset, weights, seed
        self.epoch_size = epoch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return iter(range(self.epoch_size))


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
    monkeypatch.setattr(contracts, "load_corpus_index", lambda path: object())
    monkeypatch.setattr(contracts, "IndexedMultiSourceDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(contracts, "load_split_manifest", lambda path: object())
    monkeypatch.setattr(contracts, "MultiCorpusDataset", lambda *args, **kwargs: _View())
    monkeypatch.setattr(contracts, "DeterministicQuotaSampler", _Sampler)
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
