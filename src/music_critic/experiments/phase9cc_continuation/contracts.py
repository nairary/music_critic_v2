"""Immutable contract for continuing the verified Phase 9C-C trajectory."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Mapping

import torch

from music_critic.experiments.phase8b2.schedule import (
    build_raw_downstream_sample_schedule,
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cc.contracts import (
    PHASE9CC_CELLS,
    PHASE9CC_TASKS,
    canonical_bytes,
    file_sha256,
    fingerprint,
)
from music_critic.experiments.phase9cc.runner import (
    _read,
    verify_bundle as verify_parent_bundle,
)
from music_critic.experiments.phase9cc.training import (
    _schedule as parent_schedule,
    model_state_fingerprint,
)
from music_critic.training.config import DataConfig
from music_critic.training.data import build_corpus_data_views


CONTINUATION_PROTOCOL_VERSION = "1.0.0"
CONTINUATION_PLAN_VERSION = "1.0.0"
CONTINUATION_PARENT_SHA = "bff1a405ffb9d8d6de01c4abc3d567dcb02d000b"
CONTINUATION_PARENT_BRANCH = "phase/9cc-mlp-convergence-diagnostic"
CONTINUATION_BRANCH = "phase/9cc-continuation-15000"
CONTINUATION_PARENT_MANIFEST_FINGERPRINT = (
    "6e64f33e64de9c3d864d75828a6916d95afa9fcbadc75c14359b884cab83ab10"
)
CONTINUATION_PARENT_CHECKPOINT_SHA256 = {
    "scratch_mlp": (
        "1b3d6ac9a3d2d6e90687abf1838529c412807b6c41492cc497448e83d150072f"
    ),
    "ssl_mlp": (
        "2ffb2fc03f8901455d8b99696bcc97964b69614370c125cafb6aaf6d073c0239"
    ),
}
CONTINUATION_START_UPDATE = 9000
CONTINUATION_TARGET_UPDATE = 15000
CONTINUATION_MILESTONES = (9000, 12000, 15000)


class Phase9CCContinuationError(ValueError):
    """Stable fail-closed continuation error."""


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_head() -> str:
    return _git("rev-parse", "HEAD")


def _git_branch() -> str:
    return _git("branch", "--show-current")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise Phase9CCContinuationError(
            f"phase9cc.continuation.parent_checkpoint_unreadable:{path}"
        ) from exc
    if not isinstance(payload, dict):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.parent_checkpoint_invalid"
        )
    return payload


def _load_config(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.config_unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.config_mapping_required"
        )
    return value


def _validate_config_against_parent(
    config_path: Path,
    parent_plan: Mapping[str, object],
    *,
    continuation_git_head: str,
) -> dict[str, object]:
    config = _load_config(config_path)
    protocol = parent_plan["protocol"]
    bindings = protocol["bindings"]
    path_bindings = {
        "raw_index": bindings["raw_index"],
        "target_index": bindings["target_index"],
        "split_manifest": bindings["split_manifest"],
        "class_weight_artifact": bindings["class_weight_artifact"],
        "train_priors": bindings["train_priors"],
    }
    for name, binding in path_bindings.items():
        configured = Path(str(config.get(name, ""))).expanduser().resolve()
        if (
            configured != Path(binding["path"]).resolve()
            or config.get(f"{name}_sha256") != binding["sha256"]
            or not configured.is_file()
            or file_sha256(configured) != binding["sha256"]
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.config_binding_mismatch:{name}"
            )
    for name in ("raw_cache_root", "target_cache_root"):
        configured = Path(str(config.get(name, ""))).expanduser().resolve()
        if configured != Path(str(bindings[name])).resolve() or not configured.is_dir():
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.config_binding_mismatch:{name}"
            )
    ssl = bindings["ssl_checkpoint"]
    checks = (
        ("ssl_checkpoint", "ssl_checkpoint_sha256", "path", "sha256"),
        (
            "ssl_encoder_export",
            "ssl_encoder_export_sha256",
            "encoder_export_path",
            "encoder_export_sha256",
        ),
    )
    for path_name, sha_name, bound_path_name, bound_sha_name in checks:
        configured = Path(str(config.get(path_name, ""))).expanduser().resolve()
        if (
            configured != Path(ssl[bound_path_name]).resolve()
            or config.get(sha_name) != ssl[bound_sha_name]
            or not configured.is_file()
            or file_sha256(configured) != ssl[bound_sha_name]
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.config_binding_mismatch:{path_name}"
            )
    if (
        config.get("ssl_source_kind") != ssl["source_kind"]
        or str(config.get("git_head")) != continuation_git_head
        or float(config.get("learning_rate", 0.0003))
        != float(protocol["schedule"]["learning_rate"])
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.config_scientific_binding_mismatch"
        )
    return config


def _production_schedule(
    parent_plan: Mapping[str, object], target_update: int
) -> tuple[tuple[str, str], ...]:
    protocol = parent_plan["protocol"]
    bindings = protocol["bindings"]
    batch_size = int(protocol["schedule"]["batch_size"])
    data = DataConfig(
        name="dilemmadata",
        index_paths=[bindings["raw_index"]["path"]],
        cache_roots=[bindings["raw_cache_root"]],
        split_manifest=bindings["split_manifest"]["path"],
        target_cache_index=bindings["target_index"]["path"],
        target_cache_root=bindings["target_cache_root"],
        require_target_sidecars=True,
        batch_size=batch_size,
        workers=0,
        epoch_size=batch_size * target_update,
        validation_epoch_size=0,
        mixture_weights={"dilemmadata": 1.0},
    )
    views = build_corpus_data_views(data)
    built = build_raw_downstream_sample_schedule(
        views.train,
        weights={"dilemmadata": 1.0},
        seed=int(protocol["schedule"]["downstream_data_order_seed"]),
        first_epoch=0,
        epochs=1,
        steps_per_epoch=target_update,
        batch_size=batch_size,
    )
    return built.identities


def build_continuation_plan(
    parent_root: Path,
    config_path: Path,
    *,
    start_update: int = CONTINUATION_START_UPDATE,
    target_update: int = CONTINUATION_TARGET_UPDATE,
    validation_milestones: tuple[int, ...] = CONTINUATION_MILESTONES,
    _bounded_protocol: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind an exact new-root continuation without performing training."""

    parent_root = parent_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    bounded = dict(_bounded_protocol or {})
    if not bounded and (
        start_update != CONTINUATION_START_UPDATE
        or target_update != CONTINUATION_TARGET_UPDATE
        or validation_milestones != CONTINUATION_MILESTONES
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.production_protocol_invalid"
        )
    telemetry_interval = int(bounded.get("telemetry_interval", 100))
    checkpoint_interval = int(bounded.get("checkpoint_interval", 1000))
    if (
        start_update <= 0
        or target_update <= start_update
        or tuple(sorted(set(validation_milestones))) != validation_milestones
        or not validation_milestones
        or len(validation_milestones) != 3
        or validation_milestones[0] != start_update
        or validation_milestones[-1] != target_update
        or telemetry_interval <= 0
        or checkpoint_interval <= 0
        or start_update % telemetry_interval
        or target_update % telemetry_interval
        or start_update % checkpoint_interval
        or target_update % checkpoint_interval
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.budget_invalid"
        )

    parent_plan = _read(parent_root / "experiment_plan.json")
    current_sha = _git_head()
    current_branch = _git_branch()
    parent_sha = str(parent_plan["protocol"]["git_head"])
    parent_verification = verify_parent_bundle(
        parent_root, expected_sha=parent_sha
    )
    parent_manifest = _read(parent_root / "manifest.json")
    expected_parent_manifest = str(
        bounded.get(
            "parent_manifest_fingerprint",
            CONTINUATION_PARENT_MANIFEST_FINGERPRINT,
        )
    )
    expected_parent_sha = str(
        bounded.get("parent_git_sha", CONTINUATION_PARENT_SHA)
    )
    if (
        parent_sha != expected_parent_sha
        or parent_manifest.get("fingerprint") != expected_parent_manifest
        or parent_verification.get("required_applied_updates") != start_update
        or tuple(parent_plan["protocol"].get("cells", ())) != PHASE9CC_CELLS
        or parent_plan["protocol"].get("model", {}).get("decoder", {}).get("kind")
        != "mlp"
        or any(
            value is not False
            for value in parent_plan["protocol"]["test_lock"].values()
        )
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.parent_contract_invalid"
        )
    _validate_config_against_parent(
        config_path,
        parent_plan,
        continuation_git_head=current_sha,
    )

    expected_checkpoint_hashes = dict(
        bounded.get(
            "parent_checkpoint_sha256",
            CONTINUATION_PARENT_CHECKPOINT_SHA256,
        )
    )
    parent_checkpoints = {}
    for cell_id in PHASE9CC_CELLS:
        path = parent_root / "cells" / cell_id / "checkpoints" / (
            f"update-{start_update}.pt"
        )
        observed_sha = file_sha256(path)
        payload = _checkpoint(path)
        progress = payload.get("progress", {})
        if (
            expected_checkpoint_hashes.get(cell_id) != observed_sha
            or parent_verification["checkpoint_hashes"].get(cell_id)
            != observed_sha
            or progress.get("applied_updates") != start_update
            or progress.get("schedule_position") != start_update
            or model_state_fingerprint(payload.get("model_state"))
            != payload.get("model_state_fingerprint")
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.parent_checkpoint_invalid:{cell_id}"
            )
        parent_checkpoints[cell_id] = {
            "path": str(path),
            "sha256": observed_sha,
            "model_state_fingerprint": payload["model_state_fingerprint"],
            "attempted_updates": progress["attempted_updates"],
            "skipped_updates": progress["skipped_updates"],
        }

    old_identities = parent_schedule(parent_plan)
    bounded_identities = bounded.get("schedule_identities")
    full_identities = (
        tuple(tuple(value) for value in bounded_identities)
        if isinstance(bounded_identities, (list, tuple))
        else _production_schedule(parent_plan, target_update)
    )
    batch_size = int(parent_plan["protocol"]["schedule"]["batch_size"])
    prefix_count = start_update * batch_size
    if (
        len(full_identities) != target_update * batch_size
        or full_identities[:prefix_count] != old_identities
        or raw_downstream_sample_schedule_fingerprint(old_identities)
        != parent_plan["protocol"]["schedule"]["sample_schedule_fingerprint"]
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.schedule_prefix_mismatch"
        )

    if not bounded and (
        current_branch != CONTINUATION_BRANCH
        or _git("rev-parse", f"origin/{CONTINUATION_PARENT_BRANCH}")
        != parent_sha
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.git_branch_binding_invalid"
        )
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
        "maximum_consecutive_skips": int(
            bounded.get(
                "maximum_consecutive_skips",
                parent_plan["protocol"]["schedule"][
                    "maximum_consecutive_skips"
                ],
            )
        ),
        "downstream_initialization_seed": parent_plan["protocol"]["schedule"][
            "downstream_initialization_seed"
        ],
        "downstream_data_order_seed": parent_plan["protocol"]["schedule"][
            "downstream_data_order_seed"
        ],
        "parent_schedule_fingerprint": parent_plan["protocol"]["schedule"][
            "sample_schedule_fingerprint"
        ],
        "full_schedule_fingerprint": (
            raw_downstream_sample_schedule_fingerprint(full_identities)
        ),
        "continuation_schedule_fingerprint": (
            raw_downstream_sample_schedule_fingerprint(
                full_identities[prefix_count:]
            )
        ),
        "sample_count": len(full_identities),
        "identity_contract_version": parent_plan["protocol"]["schedule"][
            "identity_contract_version"
        ],
        "resume_boundary": "applied_update_mid_epoch",
        "optimizer": parent_plan["protocol"]["schedule"]["optimizer"],
        "learning_rate": parent_plan["protocol"]["schedule"]["learning_rate"],
        "scheduler": parent_plan["protocol"]["schedule"]["scheduler"],
        "amp": parent_plan["protocol"]["schedule"]["amp"],
    }
    schedule = {**schedule, "fingerprint": fingerprint(schedule)}
    parent_binding = {
        "contract_version": "1.0.0",
        "root": str(parent_root),
        "git_sha": parent_sha,
        "git_branch": str(
            bounded.get("parent_git_branch", CONTINUATION_PARENT_BRANCH)
        ),
        "plan_fingerprint": parent_plan["fingerprint"],
        "protocol_fingerprint": parent_plan["protocol"]["fingerprint"],
        "manifest_path": str(parent_root / "manifest.json"),
        "manifest_sha256": file_sha256(parent_root / "manifest.json"),
        "manifest_fingerprint": parent_manifest["fingerprint"],
        "payload_sha256": file_sha256(parent_root / "payload.sha256"),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "checkpoints": parent_checkpoints,
        "schedule_fingerprint": schedule["parent_schedule_fingerprint"],
        "validation_membership_fingerprint": parent_plan["protocol"][
            "validation_membership"
        ]["evaluation_membership_fingerprint"],
        "test_lock": parent_plan["protocol"]["test_lock"],
    }
    parent_binding = {
        **parent_binding,
        "fingerprint": fingerprint(parent_binding),
    }
    protocol = {
        "contract_version": CONTINUATION_PROTOCOL_VERSION,
        "phase": "9C-C-continuation",
        "git_head": current_sha,
        "git_branch": current_branch,
        "seed": parent_plan["protocol"]["seed"],
        "cells": list(PHASE9CC_CELLS),
        "tasks": list(PHASE9CC_TASKS),
        "model": parent_plan["protocol"]["model"],
        "bindings": parent_plan["protocol"]["bindings"],
        "validation_membership": parent_plan["protocol"][
            "validation_membership"
        ],
        "schedule": schedule,
        "parent_binding": parent_binding,
        "test_lock": parent_plan["protocol"]["test_lock"],
        "claim_boundary": "one_seed_descriptive_continuation_no_plateau_verdict",
        "bounded_test_protocol": bool(bounded),
    }
    if bounded:
        protocol["bounded_schedule_identities"] = [
            list(value) for value in full_identities
        ]
    protocol = {**protocol, "fingerprint": fingerprint(protocol)}
    cells = [
        {
            "cell_id": cell_id,
            "decoder_kind": "mlp",
            "parent_checkpoint": parent_checkpoints[cell_id],
            "restore_mode": "model_optimizer_scaler_scheduler_rng_sampler",
            "encoder_reload": False,
            "validation_milestones": list(validation_milestones),
        }
        for cell_id in PHASE9CC_CELLS
    ]
    plan = {
        "contract_version": CONTINUATION_PLAN_VERSION,
        "protocol": protocol,
        "cells": cells,
        "production_started": False,
    }
    return {**plan, "fingerprint": fingerprint(plan)}


__all__ = [
    "CONTINUATION_MILESTONES",
    "CONTINUATION_BRANCH",
    "CONTINUATION_PARENT_BRANCH",
    "CONTINUATION_PARENT_CHECKPOINT_SHA256",
    "CONTINUATION_PARENT_MANIFEST_FINGERPRINT",
    "CONTINUATION_PARENT_SHA",
    "CONTINUATION_PLAN_VERSION",
    "CONTINUATION_PROTOCOL_VERSION",
    "CONTINUATION_START_UPDATE",
    "CONTINUATION_TARGET_UPDATE",
    "Phase9CCContinuationError",
    "build_continuation_plan",
    "canonical_bytes",
    "file_sha256",
    "fingerprint",
]
