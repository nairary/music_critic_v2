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
EXTENSION_PARENT_SHA = "a045c44c62dd881c2fbd667e70820aad7ca6282d"
EXTENSION_PARENT_BRANCH = "phase/9cc-continuation-15000"
EXTENSION_BRANCH = "phase/9cd-bigru-convergence-15000"
EXTENSION_PARENT_MANIFEST_FINGERPRINT = (
    "da7b663da3b39d7ebe2426278610ba54d3b59af7e8d471f8d18584e287340088"
)
EXTENSION_PARENT_REPORT_FINGERPRINT = (
    "c8abc49b4fc90bfded668daadb387613dceb7d7b6f3f620f152ed636abb3b6d0"
)
EXTENSION_PARENT_CHECKPOINT_SHA256 = {
    "scratch_mlp": "993fce2feb906ac72504192d433713c7d2847667a575e92cae22fb626da2a6a4",
    "ssl_mlp": "cf3048115173969328d710a1d0c81c05bc6260129c5c80533737a76e3ad686cc",
}
EXTENSION_PARENT_MODEL_STATE_FINGERPRINT = {
    "scratch_mlp": "aec7b30b103336335e83062b39ad746151478bfd3409700403e1619320cd6d33",
    "ssl_mlp": "6d2b39b0ee05fd791aa85d1851ec3713a93d977f1b6adce774d6ad2d11bd49ce",
}
EXTENSION_START_UPDATE = 15000
EXTENSION_TARGET_UPDATE = 21000
EXTENSION_MILESTONES = (15000, 18000, 21000)


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


def _sealed_continuation_parent(root: Path) -> dict[str, object]:
    manifest = _read(root / "manifest.json")
    unsigned = dict(manifest)
    observed = unsigned.pop("fingerprint", None)
    if observed != fingerprint(unsigned):
        raise Phase9CCContinuationError("phase9cc.extension.parent_manifest_invalid")
    for relative, digest in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise Phase9CCContinuationError(
                f"phase9cc.extension.parent_payload_invalid:{relative}"
            )
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "payload.sha256"}
    }
    if actual != set(manifest.get("files", {})):
        raise Phase9CCContinuationError("phase9cc.extension.parent_inventory_invalid")
    expected_payload = f"{file_sha256(root / 'manifest.json')}  manifest.json\n"
    if (root / "payload.sha256").read_text(encoding="utf-8") != expected_payload:
        raise Phase9CCContinuationError("phase9cc.extension.parent_digest_invalid")
    plan = _read(root / "continuation_plan.json")
    report = _read(root / "convergence_report.json")
    if (
        report.get("test_access") is not False
        or any(plan.get("protocol", {}).get("test_lock", {}).values())
        or report.get("plan_fingerprint") != plan.get("fingerprint")
    ):
        raise Phase9CCContinuationError("phase9cc.extension.parent_contract_invalid")
    target = int(plan["protocol"]["schedule"]["target_applied_update"])
    checkpoints = {
        cell_id: file_sha256(
            root / "cells" / cell_id / "checkpoints" / f"update-{target}.pt"
        )
        for cell_id in PHASE9CC_CELLS
    }
    return {
        "manifest_fingerprint": manifest["fingerprint"],
        "manifest": manifest,
        "plan": plan,
        "report": report,
        "checkpoint_hashes": checkpoints,
        "required_applied_updates": int(
            plan["protocol"]["schedule"]["target_applied_update"]
        ),
    }


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
    extension = (
        start_update == EXTENSION_START_UPDATE
        and target_update == EXTENSION_TARGET_UPDATE
        and validation_milestones == EXTENSION_MILESTONES
    )
    original = (
        start_update == CONTINUATION_START_UPDATE
        and target_update == CONTINUATION_TARGET_UPDATE
        and validation_milestones == CONTINUATION_MILESTONES
    )
    if not bounded and not (original or extension):
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

    parent_is_continuation = (parent_root / "continuation_plan.json").is_file()
    parent_plan = _read(
        parent_root
        / ("continuation_plan.json" if parent_is_continuation else "experiment_plan.json")
    )
    current_sha = _git_head()
    current_branch = _git_branch()
    parent_sha = str(parent_plan["protocol"]["git_head"])
    parent_verification = (
        _sealed_continuation_parent(parent_root)
        if parent_is_continuation
        else verify_parent_bundle(parent_root, expected_sha=parent_sha)
    )
    parent_manifest = _read(parent_root / "manifest.json")
    expected_parent_manifest = str(
        bounded.get(
            "parent_manifest_fingerprint",
            EXTENSION_PARENT_MANIFEST_FINGERPRINT
            if extension
            else CONTINUATION_PARENT_MANIFEST_FINGERPRINT,
        )
    )
    expected_parent_sha = str(
        bounded.get(
            "parent_git_sha", EXTENSION_PARENT_SHA if extension else CONTINUATION_PARENT_SHA
        )
    )
    if (
        parent_sha != expected_parent_sha
        or parent_manifest.get("fingerprint") != expected_parent_manifest
        or (
            extension
            and not bounded
            and parent_verification["report"].get("fingerprint")
            != EXTENSION_PARENT_REPORT_FINGERPRINT
        )
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
            EXTENSION_PARENT_CHECKPOINT_SHA256
            if extension
            else CONTINUATION_PARENT_CHECKPOINT_SHA256,
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
        expected_state_fingerprint = (
            EXTENSION_PARENT_MODEL_STATE_FINGERPRINT.get(cell_id)
            if extension and not bounded
            else payload.get("model_state_fingerprint")
        )
        if (
            expected_checkpoint_hashes.get(cell_id) != observed_sha
            or parent_verification["checkpoint_hashes"].get(cell_id)
            != observed_sha
            or progress.get("applied_updates") != start_update
            or progress.get("schedule_position") != start_update
            or model_state_fingerprint(payload.get("model_state"))
            != payload.get("model_state_fingerprint")
            or payload.get("model_state_fingerprint") != expected_state_fingerprint
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

    parent_bounded_identities = parent_plan["protocol"].get(
        "bounded_schedule_identities"
    )
    if bounded and isinstance(parent_bounded_identities, list):
        old_identities = tuple(tuple(value) for value in parent_bounded_identities)
    elif bounded:
        old_identities = parent_schedule(parent_plan)
    else:
        old_identities = _production_schedule(parent_plan, start_update)
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
        != (
            parent_plan["protocol"]["schedule"].get("full_schedule_fingerprint")
            or parent_plan["protocol"]["schedule"].get("sample_schedule_fingerprint")
        )
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.schedule_prefix_mismatch"
        )

    if not bounded and (
        current_branch != (EXTENSION_BRANCH if extension else CONTINUATION_BRANCH)
        or _git(
            "rev-parse",
            f"origin/{EXTENSION_PARENT_BRANCH if extension else CONTINUATION_PARENT_BRANCH}",
        )
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
        "parent_schedule_fingerprint": (
            parent_plan["protocol"]["schedule"].get("full_schedule_fingerprint")
            or parent_plan["protocol"]["schedule"].get("sample_schedule_fingerprint")
        ),
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
        "require_zero_skips": bool(extension and not bounded),
    }
    schedule = {**schedule, "fingerprint": fingerprint(schedule)}
    parent_binding = {
        "contract_version": "1.0.0",
        "root": str(parent_root),
        "git_sha": parent_sha,
        "git_branch": str(
            bounded.get(
                "parent_git_branch",
                EXTENSION_PARENT_BRANCH if extension else CONTINUATION_PARENT_BRANCH,
            )
        ),
        "kind": "phase9cc_continuation" if parent_is_continuation else "phase9cc",
        "plan_fingerprint": parent_plan["fingerprint"],
        "protocol_fingerprint": parent_plan["protocol"]["fingerprint"],
        "manifest_path": str(parent_root / "manifest.json"),
        "manifest_sha256": file_sha256(parent_root / "manifest.json"),
        "manifest_fingerprint": parent_manifest["fingerprint"],
        "payload_sha256": file_sha256(parent_root / "payload.sha256"),
        "report_path": (
            str(parent_root / "convergence_report.json")
            if parent_is_continuation
            else None
        ),
        "report_sha256": (
            file_sha256(parent_root / "convergence_report.json")
            if parent_is_continuation
            else None
        ),
        "report_fingerprint": (
            parent_verification["report"]["fingerprint"]
            if parent_is_continuation
            else None
        ),
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
        "continuation_generation": 2 if parent_is_continuation else 1,
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
