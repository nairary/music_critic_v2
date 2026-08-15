"""Isolated Phase 8B.2A subprocess operations used by the matrix DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize
from omegaconf import OmegaConf
import torch

from music_critic.experiments.phase8b2.contracts import fingerprint
from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.engine import _plain_config
from music_critic.ssl.multilevel import build_phase8b_model_from_config
from music_critic.ssl.phase8b_engine import (
    ResolvedPhase8B2Schedule,
    _prepare,
    _stage,
)
from music_critic.ssl.transfer import save_pretrained_encoder_export


def _ssl_config(overrides_path: Path) -> Any:
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if not isinstance(overrides, list) or not all(
        isinstance(value, str) for value in overrides
    ):
        raise ValueError("phase8b2.worker.overrides_invalid")
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(config_name="ssl_training", overrides=overrides)


def _preflight(overrides_path: Path) -> dict[str, object]:
    config = _plain_config(_ssl_config(overrides_path))
    (
        _output,
        device,
        runtime,
        model,
        _optimizer,
        _scheduler,
        _scaler,
        objective,
        masking,
        execution_mode,
        resolved,
    ) = _prepare(config)
    comparison = ResolvedPhase8B2Schedule.from_config(
        OmegaConf.create(resolved["phase8b2_schedule"]),
        masking=masking,
        objective_mode=objective.mode,
    )
    assert comparison is not None
    steps_per_epoch = int(
        resolved["experiment"].get("optimizer_steps_per_epoch", 0)
    ) or comparison.logical_updates
    remaining = comparison.logical_updates
    epoch = 0
    identities: list[list[str]] = []
    accounting = []
    while remaining:
        maximum = min(steps_per_epoch, remaining)
        metric, _ = _stage(
            model,
            runtime.train_loader(epoch),
            objective=objective,
            masking=masking,
            execution_mode=execution_mode,
            config=resolved,
            device=device,
            epoch=epoch,
            stage="validation",
            comparison=comparison,
            maximum_batches=maximum,
        )
        if (
            metric["batch_count"] != maximum
            or metric["available_batch_count"] != maximum
            or metric["skipped_or_unavailable_batch_count"] != 0
        ):
            raise ValueError(
                "phase8b2.worker.preflight_objective_unavailable"
            )
        identities.extend(metric["input_sample_identities"])
        accounting.append(metric["accounting"])
        remaining -= maximum
        epoch += 1
    observed = fingerprint(
        {
            "contract_version": "1.1.0",
            "kind": "raw_ssl_sample_schedule",
            "identities": identities,
        }
    )
    if observed != comparison.sample_schedule_fingerprint:
        raise ValueError("phase8b2.worker.preflight_schedule_mismatch")
    return {
        "preflight_contract_version": "1.1.0",
        "status": "passed",
        "variant_id": comparison.variant_id,
        "protocol_fingerprint": comparison.protocol_fingerprint,
        "sample_schedule_fingerprint": observed,
        "logical_updates": comparison.logical_updates,
        "objective_available_for_every_planned_batch": True,
        "accounting": accounting,
    }


def _export(ssl_output: Path, destination: Path) -> dict[str, object]:
    resolved = json.loads(
        (ssl_output / "resolved_config.json").read_text(encoding="utf-8")
    )
    checkpoint_path = ssl_output / "last.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model = build_phase8b_model_from_config(
        OmegaConf.create(resolved["model"]),
        OmegaConf.create(resolved["ssl"]),
        OmegaConf.create(resolved["phase8b_objective"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    save_pretrained_encoder_export(destination, model)
    report = json.loads(
        (ssl_output / "training_report.json").read_text(encoding="utf-8")
    )
    return {
        "encoder_export_worker_contract_version": "1.1.0",
        "status": "completed",
        "source_ssl_checkpoint": str(checkpoint_path.resolve()),
        "source_ssl_encoder_fingerprint": report[
            "encoder_state_fingerprints"
        ]["final"],
        "protocol_fingerprint": report["phase8b2_schedule"][
            "protocol_fingerprint"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    preflight = subparsers.add_parser("preflight-ssl")
    preflight.add_argument("overrides_json", type=Path)
    export = subparsers.add_parser("export-encoder")
    export.add_argument("ssl_output", type=Path)
    export.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.operation == "preflight-ssl":
        result = _preflight(args.overrides_json)
    else:
        result = _export(args.ssl_output, args.destination)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
