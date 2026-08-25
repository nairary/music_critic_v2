"""Execution, reports, and independent verification for Phase 9C-D."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch

from music_critic.experiments.phase8b2.schedule import (
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cb.runner import verify_bundle as verify_phase9cb
from music_critic.experiments.phase9cc.runner import _read, _rows, _write
from music_critic.experiments.phase9cc.training import model_state_fingerprint
from music_critic.experiments.phase9cc_continuation.contracts import (
    Phase9CCContinuationError,
    file_sha256,
    fingerprint,
)
from music_critic.experiments.phase9cc_continuation.runner import (
    _preflight_cell,
    _run_evaluation,
    _validation_row,
    model_contract_metadata_from_payload,
    verify_bundle as verify_mlp_bundle,
)
from music_critic.experiments.phase9cc_continuation.training import (
    CONTINUATION_CHECKPOINT_VERSION,
    run_cell_training,
    schedule_identities,
)

from .contracts import (
    CELLS,
    MILESTONES,
    MLP_MANIFEST_FINGERPRINT,
    MLP_REPORT_FINGERPRINT,
    MLP_SHA,
    PARENT_CHECKPOINTS,
    PARENT_MANIFEST_FINGERPRINT,
    PARENT_SHA,
    PLAN_VERSION,
    PROTOCOL_VERSION,
    START_UPDATE,
    TARGET_UPDATE,
    build_plan,
)


METRICS = (
    "mean_normalized_nll",
    "mean_macro_f1",
    "mean_balanced_accuracy",
    "mean_accuracy",
    "mean_prediction_entropy",
)
TASK_METRICS = (
    "normalized_nll",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "prediction_entropy",
)


def _verify_fingerprint(value: Mapping[str, object], error: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("fingerprint", None)
    if observed != fingerprint(unsigned):
        raise Phase9CCContinuationError(error)


def _delta(right: Mapping[str, object], left: Mapping[str, object], names=METRICS):
    return {
        name: (
            None
            if right.get(name) is None or left.get(name) is None
            else float(right[name]) - float(left[name])
        )
        for name in names
    }


def _row_with_source(row: Mapping[str, object], root: Path, source: str):
    return {**row, "artifact_source": source, "artifact_root": str(root)}


def evaluate_milestones(root: Path, plan: Mapping[str, object], cell, *, device: str):
    directory = root / "cells" / str(cell["cell_id"])
    start = int(plan["protocol"]["schedule"]["start_applied_update"])
    rows = []
    for update in plan["protocol"]["schedule"]["validation_milestones"]:
        checkpoint = (
            Path(cell["parent_checkpoint"]["path"])
            if update == start
            else directory / "checkpoints" / f"update-{update}.pt"
        )
        if not checkpoint.is_file():
            raise Phase9CCContinuationError(
                f"phase9cd.validation.checkpoint_missing:{cell['cell_id']}:{update}"
            )
        report_path = directory / "milestones" / f"update-{update}.json"
        _run_evaluation(plan, checkpoint, report_path, device=device)
        rows.append(
            _validation_row(
                directory,
                checkpoint,
                report_path,
                update,
                checkpoint_source="parent" if update == start else "continuation",
            )
        )
    result = {
        "contract_version": "1.0.0",
        "cell_id": cell["cell_id"],
        "split": "validation",
        "milestones": rows,
        "test_access": False,
    }
    result = {**result, "fingerprint": fingerprint(result)}
    _write(directory / "validation_milestones.json", result)
    return result


def _slope(rows):
    points = [(float(row["applied_updates"]), float(row["mean_objective_loss"])) for row in rows]
    if len(points) < 2:
        raise Phase9CCContinuationError("phase9cd.train_slope_unavailable")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def aggregate(root: Path, plan: Mapping[str, object]):
    protocol = plan["protocol"]
    parent_root = Path(protocol["parent_binding"]["root"])
    mlp_root = Path(protocol["mlp_reference"]["root"])
    start = int(protocol["schedule"]["start_applied_update"])
    target = int(protocol["schedule"]["target_applied_update"])
    combined = {}
    indexed = {}
    for cell_id in CELLS:
        parent_report_path = parent_root / "cells" / cell_id / "validation_report.json"
        parent_report = _read(parent_report_path)
        parent_payload = torch.load(
            Path(next(cell for cell in plan["cells"] if cell["cell_id"] == cell_id)["parent_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=True,
        )
        parent_row = {
            "update": start,
            "checkpoint_source": "parent",
            "checkpoint_path": str(next(cell for cell in plan["cells"] if cell["cell_id"] == cell_id)["parent_checkpoint"]["path"]),
            "checkpoint_sha256": next(cell for cell in plan["cells"] if cell["cell_id"] == cell_id)["parent_checkpoint"]["sha256"],
            "model_state_fingerprint": model_state_fingerprint(parent_payload["model_state"]),
            "validation_report_path": str(parent_report_path),
            "validation_report_sha256": file_sha256(parent_report_path),
            "validation_report_fingerprint": parent_report["fingerprint"],
            "validation_membership_fingerprint": parent_report["membership_fingerprint"],
            "aggregate": parent_report["aggregate"],
            "tasks": parent_report["tasks"],
        }
        continuation = _read(root / "cells" / cell_id / "validation_milestones.json")["milestones"]
        rows = [_row_with_source(parent_row, parent_root, "parent")]
        rows.extend(
            _row_with_source(row, root, "continuation")
            for row in continuation
            if int(row["update"]) != start
        )
        combined[cell_id] = rows
        indexed[cell_id] = {int(row["update"]): row for row in rows}
    milestones = list(protocol["schedule"]["validation_milestones"])
    transitions = list(zip(milestones, milestones[1:], strict=True)) + [(milestones[0], milestones[-1])]
    within = {
        cell_id: {
            f"{left}_to_{right}": _delta(indexed[cell_id][right]["aggregate"], indexed[cell_id][left]["aggregate"])
            for left, right in transitions
        }
        for cell_id in CELLS
    }
    best = {}
    for cell_id in CELLS:
        best_update = min(milestones, key=lambda update: float(indexed[cell_id][update]["aggregate"]["mean_normalized_nll"]))
        best[cell_id] = {
            "selection_metric": "mean_normalized_nll_descriptive_only",
            "best_milestone": best_update,
            "final_milestone": target,
            "final_minus_best": _delta(indexed[cell_id][target]["aggregate"], indexed[cell_id][best_update]["aggregate"]),
        }
    report = {
        "contract_version": "1.0.0",
        "plan_fingerprint": plan["fingerprint"],
        "parent_binding_fingerprint": protocol["parent_binding"]["fingerprint"],
        "milestone_inventory": milestones,
        "cells": {},
        "within_cell_milestone_deltas": within,
        "ssl_minus_scratch_gaps": {
            str(update): _delta(indexed["ssl_onset_bigru"][update]["aggregate"], indexed["scratch_onset_bigru"][update]["aggregate"])
            for update in milestones
        },
        "best_milestone_and_final_vs_best": best,
        "automatic_plateau_verdict": None,
        "test_access": False,
        "scientific_superiority_claimed": False,
        "statistical_significance_claimed": False,
    }
    for cell_id in CELLS:
        training = _read(root / "cells" / cell_id / "training_report.json")
        telemetry = _rows(root / "cells" / cell_id / "train_telemetry.jsonl")
        report["cells"][cell_id] = {
            "training": training,
            "milestones": combined[cell_id],
            "train_loss_moving_averages": telemetry,
            "train_loss_continuation_slope": _slope(telemetry),
            "update_accounting": {name: training[name] for name in ("applied_updates", "attempted_updates", "skipped_updates")},
        }
    report = {**report, "fingerprint": fingerprint(report)}
    _write(root / "bigru_convergence_report.json", report)

    mlp = _read(mlp_root / "convergence_report.json")
    mlp_index = {
        cell_id: {int(row["update"]): row for row in mlp["cells"][cell_id]["milestones"]}
        for cell_id in ("scratch_mlp", "ssl_mlp")
    }
    pairings = {
        "bigru_minus_mlp_under_scratch": ("scratch_onset_bigru", "scratch_mlp"),
        "bigru_minus_mlp_under_ssl": ("ssl_onset_bigru", "ssl_mlp"),
        "ssl_minus_scratch_with_mlp": ("ssl_mlp", "scratch_mlp"),
        "ssl_minus_scratch_with_bigru": ("ssl_onset_bigru", "scratch_onset_bigru"),
    }
    comparisons = {}
    for name, (right, left) in pairings.items():
        comparisons[name] = {}
        for update in milestones:
            right_row = indexed[right][update] if right in indexed else mlp_index[right][update]
            left_row = indexed[left][update] if left in indexed else mlp_index[left][update]
            comparisons[name][str(update)] = {
                "aggregate": _delta(right_row["aggregate"], left_row["aggregate"]),
                "tasks": {
                    task: _delta(right_row["tasks"][task], left_row["tasks"][task], TASK_METRICS)
                    for task in right_row["tasks"]
                },
            }
    decoder = {
        "contract_version": "1.0.0",
        "plan_fingerprint": plan["fingerprint"],
        "bigru_report_fingerprint": report["fingerprint"],
        "mlp_reference": protocol["mlp_reference"],
        "milestone_inventory": milestones,
        "comparisons": comparisons,
        "decoder_selected_by_test": False,
        "test_access": False,
        "scientific_superiority_claimed": False,
        "statistical_significance_claimed": False,
    }
    decoder = {**decoder, "fingerprint": fingerprint(decoder)}
    _write(root / "decoder_comparison_report.json", decoder)
    return report


def _write_manifest(root: Path):
    excluded = {"manifest.json", "payload.sha256"}
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    value = {"contract_version": "1.0.0", "files": files, "file_count": len(files)}
    value = {**value, "fingerprint": fingerprint(value)}
    _write(root / "manifest.json", value)
    (root / "payload.sha256").write_text(
        f"{file_sha256(root / 'manifest.json')}  manifest.json\n", encoding="utf-8"
    )
    return value


def verify_bundle(root: Path, *, expected_sha: str | None = None):
    root = root.resolve()
    plan = _read(root / "continuation_plan.json")
    _verify_fingerprint(plan, "phase9cd.verify.plan_fingerprint_invalid")
    protocol = plan.get("protocol", {})
    if (
        plan.get("contract_version") != PLAN_VERSION
        or protocol.get("contract_version") != PROTOCOL_VERSION
        or tuple(protocol.get("cells", ())) != CELLS
        or tuple(cell.get("cell_id") for cell in plan.get("cells", ())) != CELLS
        or any(cell.get("decoder_kind") != "onset_bigru" for cell in plan["cells"])
        or any(cell.get("encoder_reload") is not False for cell in plan["cells"])
        or any(protocol.get("test_lock", {}).values())
    ):
        raise Phase9CCContinuationError("phase9cd.verify.protocol_invalid")
    _verify_fingerprint(protocol, "phase9cd.verify.protocol_fingerprint_invalid")
    if expected_sha is not None and protocol.get("git_head") != expected_sha:
        raise Phase9CCContinuationError("phase9cd.verify.git_head_mismatch")
    schedule = protocol["schedule"]
    if not protocol.get("bounded_test_protocol") and (
        schedule.get("start_applied_update") != START_UPDATE
        or schedule.get("target_applied_update") != TARGET_UPDATE
        or schedule.get("validation_milestones") != list(MILESTONES)
    ):
        raise Phase9CCContinuationError("phase9cd.verify.production_budget_invalid")
    parent = protocol["parent_binding"]
    mlp = protocol["mlp_reference"]
    _verify_fingerprint(parent, "phase9cd.verify.parent_binding_invalid")
    _verify_fingerprint(mlp, "phase9cd.verify.mlp_binding_invalid")
    parent_root = Path(parent["root"])
    mlp_root = Path(mlp["root"])
    parent_verified = verify_phase9cb(parent_root, expected_sha=parent["git_sha"])
    if not protocol.get("bounded_test_protocol"):
        verify_mlp_bundle(mlp_root, expected_sha=mlp["git_sha"])
    if (
        parent_verified["manifest_fingerprint"] != parent["manifest_fingerprint"]
        or file_sha256(Path(parent["manifest_path"])) != parent["manifest_sha256"]
        or file_sha256(mlp_root / "manifest.json") != mlp["manifest_sha256"]
        or file_sha256(mlp_root / "convergence_report.json") != mlp["report_sha256"]
        or (not protocol.get("bounded_test_protocol") and (
            parent["manifest_fingerprint"] != PARENT_MANIFEST_FINGERPRINT
            or mlp["manifest_fingerprint"] != MLP_MANIFEST_FINGERPRINT
            or mlp["report_fingerprint"] != MLP_REPORT_FINGERPRINT
        ))
    ):
        raise Phase9CCContinuationError("phase9cd.verify.external_binding_invalid")
    identities = schedule_identities(plan)
    if raw_downstream_sample_schedule_fingerprint(identities) != schedule["full_schedule_fingerprint"]:
        raise Phase9CCContinuationError("phase9cd.verify.schedule_invalid")
    start = int(schedule["start_applied_update"])
    target = int(schedule["target_applied_update"])
    interval = int(schedule["telemetry_interval_applied"])
    checkpoint_interval = int(schedule["checkpoint_interval_applied"])
    checkpoint_hashes = {}
    for cell in plan["cells"]:
        cell_id = cell["cell_id"]
        parent_checkpoint = Path(cell["parent_checkpoint"]["path"])
        if file_sha256(parent_checkpoint) != cell["parent_checkpoint"]["sha256"]:
            raise Phase9CCContinuationError(f"phase9cd.verify.parent_checkpoint_invalid:{cell_id}")
        if not protocol.get("bounded_test_protocol") and cell["parent_checkpoint"]["sha256"] != PARENT_CHECKPOINTS[cell_id]:
            raise Phase9CCContinuationError(f"phase9cd.verify.parent_checkpoint_exact_invalid:{cell_id}")
        directory = root / "cells" / cell_id
        preflight = _read(directory / "preflight_evidence.json")
        _verify_fingerprint(preflight, f"phase9cd.verify.preflight_invalid:{cell_id}")
        training = _read(directory / "training_report.json")
        _verify_fingerprint(training, f"phase9cd.verify.training_invalid:{cell_id}")
        telemetry = _rows(directory / "train_telemetry.jsonl")
        expected_rows = list(range(start + interval, target + 1, interval))
        if (
            preflight.get("passed") is not True
            or preflight.get("logit_replay", {}).get("candidate_identities_exact") is not True
            or preflight.get("metric_comparison", {}).get("within_tolerance") is not True
            or training.get("complete") is not True
            or training.get("applied_updates") != target
            or training.get("sample_schedule_position") != target
            or training.get("encoder_export_reloaded") is not False
            or [row.get("applied_updates") for row in telemetry] != expected_rows
            or len(expected_rows) != len(set(expected_rows))
        ):
            raise Phase9CCContinuationError(f"phase9cd.verify.cell_invalid:{cell_id}")
        checkpoints = {
            int(path.stem.split("-", 1)[1]): path
            for path in (directory / "checkpoints").glob("update-*.pt")
        }
        expected_checkpoints = set(range(start + checkpoint_interval, target + 1, checkpoint_interval))
        if set(checkpoints) != expected_checkpoints:
            raise Phase9CCContinuationError(f"phase9cd.verify.checkpoint_inventory_invalid:{cell_id}")
        for update, path in checkpoints.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("metadata", {}).get("phase9cc_continuation_checkpoint_version") != CONTINUATION_CHECKPOINT_VERSION
                or payload.get("metadata", {}).get("cell_id") != cell_id
                or payload.get("progress", {}).get("applied_updates") != update
                or model_contract_metadata_from_payload(payload) != payload.get("metadata", {}).get("model_contract")
                or model_state_fingerprint(payload.get("model_state")) != payload.get("model_state_fingerprint")
            ):
                raise Phase9CCContinuationError(f"phase9cd.verify.checkpoint_invalid:{cell_id}:{update}")
        validation = _read(directory / "validation_milestones.json")
        _verify_fingerprint(validation, f"phase9cd.verify.validation_invalid:{cell_id}")
        if [row.get("update") for row in validation.get("milestones", ())] != list(schedule["validation_milestones"]):
            raise Phase9CCContinuationError(f"phase9cd.verify.validation_inventory_invalid:{cell_id}")
        checkpoint_hashes[cell_id] = file_sha256(checkpoints[target])
    bigru = _read(root / "bigru_convergence_report.json")
    decoder = _read(root / "decoder_comparison_report.json")
    _verify_fingerprint(bigru, "phase9cd.verify.bigru_report_invalid")
    _verify_fingerprint(decoder, "phase9cd.verify.decoder_report_invalid")
    if (
        bigru.get("automatic_plateau_verdict") is not None
        or bigru.get("test_access") is not False
        or decoder.get("test_access") is not False
        or decoder.get("decoder_selected_by_test") is not False
        or bigru.get("milestone_inventory") != list(schedule["validation_milestones"])
    ):
        raise Phase9CCContinuationError("phase9cd.verify.report_claim_invalid")
    manifest = _read(root / "manifest.json")
    _verify_fingerprint(manifest, "phase9cd.verify.manifest_invalid")
    for relative, digest in manifest["files"].items():
        if not (root / relative).is_file() or file_sha256(root / relative) != digest:
            raise Phase9CCContinuationError(f"phase9cd.verify.payload_invalid:{relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name not in {"manifest.json", "payload.sha256"}}
    if actual != set(manifest["files"]):
        raise Phase9CCContinuationError("phase9cd.verify.file_inventory_invalid")
    return {
        "status": "verified",
        "cell_count": len(CELLS),
        "start_applied_update": start,
        "final_applied_update": target,
        "additional_applied_updates": target - start,
        "checkpoint_hashes": checkpoint_hashes,
        "parent_manifest_fingerprint": parent["manifest_fingerprint"],
        "mlp_manifest_fingerprint": mlp["manifest_fingerprint"],
        "manifest_fingerprint": manifest["fingerprint"],
        "test_access": False,
    }


def execute(root: Path, plan: Mapping[str, object], *, device: str | None = None):
    production = not plan["protocol"].get("bounded_test_protocol")
    resolved = device or ("cuda:0" if production else "cpu")
    if production and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090"
        or resolved != "cuda:0"
    ):
        raise Phase9CCContinuationError("phase9cd.hardware.rtx3090_cuda0_required")
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "continuation_plan.json"
    if plan_path.is_file() and _read(plan_path) != plan:
        raise Phase9CCContinuationError("phase9cd.resume.plan_mismatch")
    if not plan_path.is_file():
        _write(plan_path, plan)
        _write(root / "protocol.json", plan["protocol"])
        _write(root / "parent_binding.json", plan["protocol"]["parent_binding"])
    if (root / "manifest.json").is_file():
        return verify_bundle(root, expected_sha=plan["protocol"]["git_head"])
    for cell in plan["cells"]:
        _preflight_cell(root, plan, cell, device=resolved)
    for cell in plan["cells"]:
        directory = root / "cells" / cell["cell_id"]
        report_path = directory / "training_report.json"
        if not report_path.is_file() or _read(report_path).get("complete") is not True:
            run_cell_training(
                plan,
                cell,
                directory,
                action="resume" if (directory / "train_telemetry.jsonl").exists() or (directory / "checkpoints").exists() else "run",
                device=resolved,
            )
    for cell in plan["cells"]:
        evaluate_milestones(root, plan, cell, device=resolved)
    report = aggregate(root, plan)
    return {"status": "execution_complete_pending_manifest", "bigru_convergence_report_fingerprint": report["fingerprint"], "test_access": False}


def finalize(root: Path, *, expected_sha: str):
    plan = _read(root / "continuation_plan.json")
    if plan["protocol"].get("git_head") != expected_sha:
        raise Phase9CCContinuationError("phase9cd.finalize.git_head_mismatch")
    if not (root / "execution.log").is_file():
        raise Phase9CCContinuationError("phase9cd.finalize.execution_log_missing")
    aggregate(root, plan)
    _write_manifest(root)
    return verify_bundle(root, expected_sha=expected_sha)


__all__ = ["aggregate", "execute", "finalize", "verify_bundle"]
