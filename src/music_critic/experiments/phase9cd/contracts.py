"""Fail-closed bindings for the Phase 9C-D BiGRU continuation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch

from music_critic.experiments.phase8b2.schedule import (
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cb.runner import verify_bundle as verify_phase9cb
from music_critic.experiments.phase9cc.runner import _read
from music_critic.experiments.phase9cc.training import model_state_fingerprint
from music_critic.experiments.phase9cc_continuation.contracts import (
    Phase9CCContinuationError,
    _git,
    _git_branch,
    _git_head,
    _production_schedule,
    _validate_config_against_parent,
    file_sha256,
    fingerprint,
)
from music_critic.experiments.phase9cc_continuation.runner import (
    verify_bundle as verify_mlp_reference,
)


PROTOCOL_VERSION = "1.0.0"
PLAN_VERSION = "1.0.0"
BRANCH = "phase/9cd-bigru-convergence-15000"
BASE_BRANCH = "phase/9cc-continuation-15000"
BASE_SHA = "a045c44c62dd881c2fbd667e70820aad7ca6282d"
PARENT_SHA = "786d0dd9320545f2eee50b6d59e609e72d96da49"
PARENT_MANIFEST_FINGERPRINT = (
    "82f9af67eaa06b69f5107a6a4729518eeff1dba561f81521f87ffd08dd1ddf0e"
)
PARENT_CHECKPOINTS = {
    "scratch_onset_bigru": (
        "03ceed8d7481923a047d200f45078ab0d7d75f2c310b0fac77df7892eba634cb"
    ),
    "ssl_onset_bigru": (
        "d143fb7ce71e7cd6bbb7a5efdb1a48b8f3a8e8e8c90ef95c7b1b04082d12dd6b"
    ),
}
PARENT_METRICS = {
    "scratch_onset_bigru": (0.6328962772991069, 0.1392916723323795, 0.15556244935475066),
    "ssl_onset_bigru": (0.6518300416038338, 0.12182077982195845, 0.14713863789887213),
}
CELLS = ("scratch_onset_bigru", "ssl_onset_bigru")
START_UPDATE = 3000
TARGET_UPDATE = 15000
MILESTONES = (3000, 6000, 9000, 12000, 15000)
MLP_SHA = BASE_SHA
MLP_MANIFEST_FINGERPRINT = (
    "da7b663da3b39d7ebe2426278610ba54d3b59af7e8d471f8d18584e287340088"
)
MLP_REPORT_FINGERPRINT = (
    "c8abc49b4fc90bfded668daadb387613dceb7d7b6f3f620f152ed636abb3b6d0"
)


def _checkpoint(path: Path) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Phase9CCContinuationError(
            f"phase9cd.parent_checkpoint_unreadable:{path}"
        ) from exc
    if not isinstance(value, dict):
        raise Phase9CCContinuationError("phase9cd.parent_checkpoint_invalid")
    return value


def _relative_checkpoint(parent_root: Path, cell_id: str) -> Path:
    directory = parent_root / "cells" / cell_id
    report = _read(directory / "cell_report.json")
    relative = Path(str(report.get("checkpoint", {}).get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise Phase9CCContinuationError(
            f"phase9cd.parent_checkpoint_path_invalid:{cell_id}"
        )
    path = directory / relative
    manifest = _read(parent_root / "bundle_manifest.json")
    manifest_name = str(path.relative_to(parent_root))
    if manifest.get("files", {}).get(manifest_name) != file_sha256(path):
        raise Phase9CCContinuationError(
            f"phase9cd.parent_checkpoint_manifest_mismatch:{cell_id}"
        )
    return path


def _metric_preflight(cell_id: str, report: Mapping[str, object]) -> None:
    expected = PARENT_METRICS[cell_id]
    aggregate = report.get("aggregate", {})
    observed = (
        aggregate.get("mean_normalized_nll"),
        aggregate.get("mean_macro_f1"),
        aggregate.get("mean_balanced_accuracy"),
    )
    if any(
        value is None
        or not math.isclose(float(value), target, rel_tol=1e-12, abs_tol=1e-12)
        for value, target in zip(observed, expected, strict=True)
    ):
        raise Phase9CCContinuationError(
            f"phase9cd.parent_metric_binding_mismatch:{cell_id}"
        )


def build_plan(
    parent_root: Path,
    config_path: Path,
    mlp_reference_root: Path,
    *,
    start_update: int = START_UPDATE,
    target_update: int = TARGET_UPDATE,
    validation_milestones: tuple[int, ...] = MILESTONES,
    _bounded: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the exact two-cell decoder-neutral continuation plan."""

    bounded = dict(_bounded or {})
    parent_root = parent_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    mlp_reference_root = mlp_reference_root.expanduser().resolve()
    if not bounded and (
        start_update != START_UPDATE
        or target_update != TARGET_UPDATE
        or validation_milestones != MILESTONES
    ):
        raise Phase9CCContinuationError("phase9cd.production_protocol_invalid")
    telemetry_interval = int(bounded.get("telemetry_interval", 100))
    checkpoint_interval = int(bounded.get("checkpoint_interval", 1000))
    if (
        start_update <= 0
        or target_update <= start_update
        or validation_milestones[0] != start_update
        or validation_milestones[-1] != target_update
        or tuple(sorted(set(validation_milestones))) != validation_milestones
        or telemetry_interval <= 0
        or checkpoint_interval <= 0
        or start_update % telemetry_interval
        or target_update % telemetry_interval
        or start_update % checkpoint_interval
        or target_update % checkpoint_interval
    ):
        raise Phase9CCContinuationError("phase9cd.budget_invalid")

    parent_plan = _read(parent_root / "experiment_plan.json")
    parent_manifest = _read(parent_root / "bundle_manifest.json")
    parent_verified = verify_phase9cb(parent_root, expected_sha=PARENT_SHA)
    expected_parent_manifest = str(
        bounded.get("parent_manifest_fingerprint", PARENT_MANIFEST_FINGERPRINT)
    )
    if (
        parent_manifest.get("fingerprint") != expected_parent_manifest
        or parent_verified.get("manifest_fingerprint") != expected_parent_manifest
        or parent_plan["protocol"].get("git_head") != str(
            bounded.get("parent_sha", PARENT_SHA)
        )
        or any(parent_plan["protocol"]["test_lock"].values())
    ):
        raise Phase9CCContinuationError("phase9cd.parent_binding_invalid")
    current_sha = _git_head()
    current_branch = _git_branch()
    _validate_config_against_parent(
        config_path, parent_plan, continuation_git_head=current_sha
    )

    full_identities = (
        tuple(tuple(row) for row in bounded["schedule_identities"])
        if "schedule_identities" in bounded
        else _production_schedule(parent_plan, target_update)
    )
    parent_identities = _production_schedule(parent_plan, start_update)
    batch_size = int(parent_plan["protocol"]["schedule"]["batch_size"])
    prefix_count = start_update * batch_size
    if (
        len(full_identities) != target_update * batch_size
        or full_identities[:prefix_count] != parent_identities
        or raw_downstream_sample_schedule_fingerprint(parent_identities)
        != parent_plan["protocol"]["schedule"]["sample_schedule_fingerprint"]
    ):
        raise Phase9CCContinuationError("phase9cd.schedule_prefix_mismatch")

    mlp_plan = _read(mlp_reference_root / "continuation_plan.json")
    mlp_report = _read(mlp_reference_root / "convergence_report.json")
    mlp_manifest = _read(mlp_reference_root / "manifest.json")
    expected_mlp_manifest = str(
        bounded.get("mlp_manifest_fingerprint", MLP_MANIFEST_FINGERPRINT)
    )
    expected_mlp_report = str(
        bounded.get("mlp_report_fingerprint", MLP_REPORT_FINGERPRINT)
    )
    if not bounded:
        verify_mlp_reference(mlp_reference_root, expected_sha=MLP_SHA)
    if (
        mlp_manifest.get("fingerprint") != expected_mlp_manifest
        or mlp_report.get("fingerprint") != expected_mlp_report
        or mlp_plan["protocol"]["schedule"].get("full_schedule_fingerprint")
        != raw_downstream_sample_schedule_fingerprint(full_identities)
    ):
        raise Phase9CCContinuationError("phase9cd.mlp_reference_invalid")

    parent_checkpoints = {}
    membership = None
    for cell_id in CELLS:
        path = _relative_checkpoint(parent_root, cell_id)
        payload = _checkpoint(path)
        report = _read(parent_root / "cells" / cell_id / "cell_report.json")
        validation = _read(parent_root / "cells" / cell_id / "validation_report.json")
        _metric_preflight(cell_id, validation)
        decoder = (
            payload.get("metadata", {})
            .get("model_contract", {})
            .get("decoder", {})
            .get("kind")
        )
        observed_membership = validation.get("membership_fingerprint")
        membership = membership or observed_membership
        if (
            file_sha256(path)
            != str(bounded.get("checkpoint_hashes", PARENT_CHECKPOINTS)[cell_id])
            or report.get("applied_updates") != start_update
            or report.get("checkpoint", {}).get("sha256") != file_sha256(path)
            or decoder != "onset_bigru"
            or observed_membership != membership
        ):
            raise Phase9CCContinuationError(
                f"phase9cd.parent_checkpoint_invalid:{cell_id}"
            )
        parent_checkpoints[cell_id] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "model_state_fingerprint": model_state_fingerprint(payload["model_state"]),
            "attempted_updates": report["attempted_updates"],
            "skipped_updates": report["skipped_updates"],
        }

    if not bounded and (
        current_branch != BRANCH or _git("rev-parse", f"origin/{BASE_BRANCH}") != BASE_SHA
    ):
        raise Phase9CCContinuationError("phase9cd.git_binding_invalid")
    full_fingerprint = raw_downstream_sample_schedule_fingerprint(full_identities)
    schedule = {
        "contract_version": "1.0.0",
        "epoch": 0,
        "start_applied_update": start_update,
        "target_applied_update": target_update,
        "additional_applied_updates": target_update - start_update,
        "batch_size": batch_size,
        "epoch_size": target_update * batch_size,
        "optimizer_steps_per_epoch": target_update,
        "telemetry_interval_applied": telemetry_interval,
        "checkpoint_interval_applied": checkpoint_interval,
        "validation_milestones": list(validation_milestones),
        "maximum_consecutive_skips": int(bounded.get("maximum_consecutive_skips", 8)),
        "downstream_initialization_seed": parent_plan["protocol"]["schedule"]["downstream_initialization_seed"],
        "downstream_data_order_seed": parent_plan["protocol"]["schedule"]["downstream_data_order_seed"],
        "parent_schedule_fingerprint": raw_downstream_sample_schedule_fingerprint(parent_identities),
        "full_schedule_fingerprint": full_fingerprint,
        "continuation_schedule_fingerprint": raw_downstream_sample_schedule_fingerprint(full_identities[prefix_count:]),
        "sample_count": len(full_identities),
        "identity_contract_version": "1.2.0",
        "resume_boundary": "applied_update_mid_epoch",
        "optimizer": "adamw",
        "learning_rate": float(parent_plan["protocol"]["schedule"]["learning_rate"]),
        "scheduler": "none",
        "amp": "float16",
    }
    schedule = {**schedule, "fingerprint": fingerprint(schedule)}
    parent_binding = {
        "contract_version": "1.0.0",
        "kind": "phase9cb",
        "root": str(parent_root),
        "git_sha": PARENT_SHA,
        "git_branch": "phase/9cb-onset-bigru-decoder",
        "plan_fingerprint": parent_plan["fingerprint"],
        "protocol_fingerprint": parent_plan["protocol"]["fingerprint"],
        "manifest_path": str(parent_root / "bundle_manifest.json"),
        "manifest_sha256": file_sha256(parent_root / "bundle_manifest.json"),
        "manifest_fingerprint": parent_manifest["fingerprint"],
        "payload_sha256": None,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoints": parent_checkpoints,
        "schedule_fingerprint": schedule["parent_schedule_fingerprint"],
        "validation_membership_fingerprint": membership,
        "test_lock": parent_plan["protocol"]["test_lock"],
    }
    parent_binding = {**parent_binding, "fingerprint": fingerprint(parent_binding)}
    mlp_binding = {
        "contract_version": "1.0.0",
        "root": str(mlp_reference_root),
        "git_sha": MLP_SHA,
        "manifest_sha256": file_sha256(mlp_reference_root / "manifest.json"),
        "manifest_fingerprint": mlp_manifest["fingerprint"],
        "report_sha256": file_sha256(mlp_reference_root / "convergence_report.json"),
        "report_fingerprint": mlp_report["fingerprint"],
        "schedule_fingerprint": full_fingerprint,
    }
    mlp_binding = {**mlp_binding, "fingerprint": fingerprint(mlp_binding)}
    protocol = {
        "contract_version": PROTOCOL_VERSION,
        "phase": "9C-D",
        "git_head": current_sha,
        "git_branch": current_branch,
        "seed": 17,
        "cells": list(CELLS),
        "tasks": list(parent_plan["protocol"].get("tasks", ())) or [
            "dilemmadata.an.chord.inversion",
            "dilemmadata.an.chord.quality",
            "dilemmadata.dlc.chord.inversion",
            "dilemmadata.dlc.chord.quality",
        ],
        "model": None,
        "bindings": parent_plan["protocol"]["bindings"],
        "validation_membership": {
            "split": "validation",
            "evaluation_membership_fingerprint": membership,
        },
        "schedule": schedule,
        "parent_binding": parent_binding,
        "mlp_reference": mlp_binding,
        "test_lock": parent_plan["protocol"]["test_lock"],
        "claim_boundary": "one_seed_descriptive_decoder_convergence_no_plateau_verdict",
        "bounded_test_protocol": bool(bounded),
    }
    if bounded:
        protocol["bounded_schedule_identities"] = [list(row) for row in full_identities]
    protocol = {**protocol, "fingerprint": fingerprint(protocol)}
    cells = [
        {
            "cell_id": cell_id,
            "decoder_kind": "onset_bigru",
            "parent_checkpoint": parent_checkpoints[cell_id],
            "restore_mode": "model_optimizer_scaler_scheduler_rng_sampler",
            "encoder_reload": False,
            "validation_milestones": list(validation_milestones),
        }
        for cell_id in CELLS
    ]
    plan = {
        "contract_version": PLAN_VERSION,
        "protocol": protocol,
        "cells": cells,
        "production_started": False,
    }
    return {**plan, "fingerprint": fingerprint(plan)}


__all__ = ["BRANCH", "CELLS", "MILESTONES", "START_UPDATE", "TARGET_UPDATE", "build_plan"]
