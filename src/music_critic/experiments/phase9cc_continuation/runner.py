"""Preflight, execution, aggregation, and verification for continuation."""

from __future__ import annotations

import copy
from dataclasses import replace
import gc
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import torch

from music_critic.experiments.dilemmadata.supervised_smoke import (
    DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE,
    DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE,
    _prediction_evidence,
    _prediction_replay_diagnostic,
)
from music_critic.experiments.phase8b2.schedule import (
    raw_downstream_sample_schedule_fingerprint,
)
from music_critic.experiments.phase9cc.contracts import (
    PHASE9CC_CELLS,
    PHASE9CC_TASKS,
)
from music_critic.experiments.phase9cc.runner import (
    _read,
    _rows,
    _verify_metrics,
    _write,
    verify_bundle as verify_parent_bundle,
)
from music_critic.experiments.phase9cc.training import model_state_fingerprint
from music_critic.experiments.phase9cc.training import _schedule as parent_schedule
from music_critic.training import engine as training_engine
from music_critic.training.device import move_multisource_batch
from music_critic.training.models import model_contract_metadata
from music_critic.models import (
    DilemmadataHierarchicalModel,
    dilemmadata_config_from_model_contract,
)

from .contracts import (
    CONTINUATION_PARENT_CHECKPOINT_SHA256,
    CONTINUATION_PARENT_MANIFEST_FINGERPRINT,
    CONTINUATION_PARENT_SHA,
    CONTINUATION_PLAN_VERSION,
    CONTINUATION_PROTOCOL_VERSION,
    Phase9CCContinuationError,
    build_continuation_plan,
    file_sha256,
    fingerprint,
)
from .training import (
    CONTINUATION_CHECKPOINT_VERSION,
    _restore_parent,
    continuation_training_config,
    run_cell_training,
    schedule_identities,
)


_METRICS = (
    "mean_normalized_nll",
    "mean_macro_f1",
    "mean_balanced_accuracy",
    "mean_accuracy",
    "mean_prediction_entropy",
)
_TASK_METRICS = (
    "normalized_nll",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "prediction_entropy",
)


def _evaluation_command(
    plan: Mapping[str, object], checkpoint: Path, output: Path, device: str
) -> list[str]:
    bindings = plan["protocol"]["bindings"]
    return [
        sys.executable,
        "-m",
        "music_critic.evaluation.dilemmadata_run",
        "--checkpoint",
        str(checkpoint),
        "--raw-index",
        bindings["raw_index"]["path"],
        "--raw-cache-root",
        bindings["raw_cache_root"],
        "--target-index",
        bindings["target_index"]["path"],
        "--target-cache-root",
        bindings["target_cache_root"],
        "--split-manifest",
        bindings["split_manifest"]["path"],
        "--split",
        "validation",
        "--batch-size",
        str(plan["protocol"]["schedule"]["batch_size"]),
        "--workers",
        "0",
        "--device",
        device,
        "--train-priors",
        bindings["train_priors"]["path"],
        "--output",
        str(output),
    ]


def _run_evaluation(
    plan: Mapping[str, object],
    checkpoint: Path,
    report_path: Path,
    *,
    device: str,
) -> dict[str, object]:
    log_path = report_path.with_suffix(".log")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.is_file():
        process = subprocess.run(
            _evaluation_command(plan, checkpoint, report_path, device),
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise Phase9CCContinuationError(
                "phase9cc.continuation.validation.subprocess_failed:"
                f"{checkpoint}:{process.returncode}"
            )
    return _read(report_path)


def _metric_comparison(
    parent: Mapping[str, object], replay: Mapping[str, object]
) -> dict[str, object]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    compared = 0

    def compare(left: object, right: object, name: str) -> None:
        nonlocal maximum_absolute, maximum_relative, compared
        if left is None or right is None:
            if left is not right:
                raise Phase9CCContinuationError(
                    f"phase9cc.continuation.preflight.metric_mismatch:{name}"
                )
            return
        left_value = float(left)
        right_value = float(right)
        difference = abs(left_value - right_value)
        relative = difference / max(abs(left_value), abs(right_value), 1e-12)
        maximum_absolute = max(maximum_absolute, difference)
        maximum_relative = max(maximum_relative, relative)
        compared += 1
        if not math.isclose(
            left_value,
            right_value,
            abs_tol=DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE,
            rel_tol=DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE,
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.preflight.metric_mismatch:{name}"
            )

    for name in _METRICS:
        compare(parent["aggregate"].get(name), replay["aggregate"].get(name), name)
    if set(parent["tasks"]) != set(replay["tasks"]):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.preflight.task_inventory_mismatch"
        )
    for task_id in PHASE9CC_TASKS:
        left = parent["tasks"][task_id]
        right = replay["tasks"][task_id]
        for name in _TASK_METRICS:
            compare(left.get(name), right.get(name), f"{task_id}:{name}")
        if left.get("true_class_support") != right.get("true_class_support"):
            raise Phase9CCContinuationError(
                "phase9cc.continuation.preflight.support_mismatch"
            )
        left_distribution = left.get("predicted_class_distribution", ())
        right_distribution = right.get("predicted_class_distribution", ())
        if len(left_distribution) != len(right_distribution):
            raise Phase9CCContinuationError(
                "phase9cc.continuation.preflight.distribution_mismatch"
            )
        for index, (left_value, right_value) in enumerate(
            zip(left_distribution, right_distribution, strict=True)
        ):
            compare(
                left_value,
                right_value,
                f"{task_id}:predicted_class_distribution:{index}",
            )
    return {
        "contract_version": "1.0.0",
        "comparison_dtype": "float64_report_scalars",
        "absolute_tolerance": DILEMMADATA_CUDA_REPLAY_ABSOLUTE_TOLERANCE,
        "relative_tolerance": DILEMMADATA_CUDA_REPLAY_RELATIVE_TOLERANCE,
        "compared_scalar_count": compared,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "support_exact": True,
        "within_tolerance": True,
    }


def _prediction_sequence(
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    output: Path,
    *,
    device: str,
) -> tuple[object, ...]:
    config = continuation_training_config(plan, cell, output, device=device)
    (
        _unused,
        resolved_device,
        runtime,
        model,
        optimizer,
        scheduler,
        scaler,
        _memory,
        _weights,
    ) = training_engine._prepare(config)
    _restore_parent(
        plan,
        cell,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    model.eval()
    rows = []
    with torch.no_grad():
        for cpu_batch in runtime.validation_loader():
            batch = move_multisource_batch(cpu_batch, resolved_device)
            _, predictions = model.predict(batch.raw_graph_batch)
            for prediction in predictions:
                rows.append(
                    replace(
                        prediction,
                        candidate_node_type_codes=(
                            prediction.candidate_node_type_codes.detach().cpu()
                        ),
                        global_entity_indices=(
                            prediction.global_entity_indices.detach().cpu()
                        ),
                        sample_indices=prediction.sample_indices.detach().cpu(),
                        candidate_offsets_by_node_type=(
                            prediction.candidate_offsets_by_node_type.detach().cpu()
                        ),
                        candidate_counts_by_node_type=(
                            prediction.candidate_counts_by_node_type.detach().cpu()
                        ),
                        logits=prediction.logits.detach().float().cpu(),
                    )
                )
    del model, optimizer, scaler, runtime
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return tuple(rows)


def _parent_milestone(
    plan: Mapping[str, object], cell_id: str, update: int
) -> dict[str, object]:
    root = Path(plan["protocol"]["parent_binding"]["root"])
    milestone = _read(root / "cells" / cell_id / "validation_milestones.json")
    rows = [row for row in milestone["milestones"] if row["update"] == update]
    if len(rows) != 1:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.preflight.parent_milestone_missing"
        )
    return rows[0]


def _preflight_cell(
    root: Path,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    device: str,
) -> dict[str, object]:
    cell_id = str(cell["cell_id"])
    start = int(plan["protocol"]["schedule"]["start_applied_update"])
    directory = root / "cells" / cell_id
    evidence_path = directory / "preflight_evidence.json"
    if evidence_path.is_file():
        evidence = _read(evidence_path)
        unsigned = dict(evidence)
        observed = unsigned.pop("fingerprint", None)
        replay_path = directory / str(evidence.get("replay_validation_report_path", ""))
        parent_report_path = Path(
            str(evidence.get("parent_validation_report_path", ""))
        )
        if (
            observed != fingerprint(unsigned)
            or evidence.get("passed") is not True
            or evidence.get("parent_checkpoint_sha256")
            != file_sha256(Path(cell["parent_checkpoint"]["path"]))
            or not replay_path.is_file()
            or evidence.get("replay_validation_report_sha256")
            != file_sha256(replay_path)
            or not parent_report_path.is_file()
            or evidence.get("parent_validation_report_sha256")
            != file_sha256(parent_report_path)
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.preflight.evidence_invalid:{cell_id}"
            )
        return evidence
    checkpoint = Path(cell["parent_checkpoint"]["path"])
    report_path = directory / "milestones" / f"update-{start}.json"
    replay_report = _run_evaluation(
        plan, checkpoint, report_path, device=device
    )
    unsigned_replay_report = dict(replay_report)
    observed_replay_fingerprint = unsigned_replay_report.pop("fingerprint", None)
    if observed_replay_fingerprint != fingerprint(unsigned_replay_report):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.preflight.validation_fingerprint_invalid"
        )
    parent_row = _parent_milestone(plan, cell_id, start)
    parent_root = Path(plan["protocol"]["parent_binding"]["root"])
    parent_report_path = (
        parent_root / "cells" / cell_id / parent_row["validation_report_path"]
    )
    parent_report = _read(parent_report_path)
    if (
        replay_report.get("split") != "validation"
        or replay_report.get("membership_fingerprint")
        != plan["protocol"]["validation_membership"][
            "evaluation_membership_fingerprint"
        ]
        or parent_report.get("membership_fingerprint")
        != replay_report.get("membership_fingerprint")
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.preflight.membership_mismatch"
        )
    metric_comparison = _metric_comparison(parent_report, replay_report)
    reference = _prediction_sequence(
        plan, cell, directory / "preflight-reference", device=device
    )
    replay = _prediction_sequence(
        plan, cell, directory / "preflight-replay", device=device
    )
    replay_diagnostic = _prediction_replay_diagnostic(reference, replay)
    prediction_evidence = _prediction_evidence(reference)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identities = schedule_identities(plan)
    batch_size = int(plan["protocol"]["schedule"]["batch_size"])
    next_identities = identities[start * batch_size : (start + 1) * batch_size]
    evidence = {
        "contract_version": "1.0.0",
        "cell_id": cell_id,
        "update": start,
        "parent_checkpoint_path": str(checkpoint),
        "parent_checkpoint_sha256": file_sha256(checkpoint),
        "parent_model_state_fingerprint": payload["model_state_fingerprint"],
        "reloaded_model_state_fingerprint": model_state_fingerprint(
            payload["model_state"]
        ),
        "parent_validation_report_path": str(parent_report_path),
        "parent_validation_report_sha256": file_sha256(parent_report_path),
        "replay_validation_report_path": str(report_path.relative_to(directory)),
        "replay_validation_report_sha256": file_sha256(report_path),
        "validation_membership_fingerprint": replay_report[
            "membership_fingerprint"
        ],
        "metric_comparison": metric_comparison,
        "prediction_evidence": prediction_evidence,
        "logit_replay": replay_diagnostic,
        "next_train_sample_identities": [list(value) for value in next_identities],
        "next_train_sample_identity_fingerprint": fingerprint(next_identities),
        "test_lock": copy.deepcopy(plan["protocol"]["test_lock"]),
        "passed": True,
    }
    evidence = {**evidence, "fingerprint": fingerprint(evidence)}
    _write(evidence_path, evidence)
    return evidence


def _validation_row(
    directory: Path,
    checkpoint: Path,
    report_path: Path,
    update: int,
    *,
    checkpoint_source: str,
) -> dict[str, object]:
    report = _read(report_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        "update": update,
        "checkpoint_source": checkpoint_source,
        "checkpoint_path": (
            str(checkpoint)
            if checkpoint_source == "parent"
            else str(checkpoint.relative_to(directory))
        ),
        "checkpoint_sha256": file_sha256(checkpoint),
        "model_state_fingerprint": model_state_fingerprint(payload["model_state"]),
        "checkpoint_declared_model_state_fingerprint": payload.get(
            "model_state_fingerprint"
        ),
        "validation_report_path": str(report_path.relative_to(directory)),
        "validation_report_sha256": file_sha256(report_path),
        "validation_report_fingerprint": report.get("fingerprint"),
        "validation_membership_fingerprint": report.get(
            "membership_fingerprint"
        ),
        "aggregate": report.get("aggregate"),
        "tasks": report.get("tasks"),
    }


def evaluate_milestones(
    root: Path,
    plan: Mapping[str, object],
    cell: Mapping[str, object],
    *,
    device: str,
) -> dict[str, object]:
    directory = root / "cells" / str(cell["cell_id"])
    start = int(plan["protocol"]["schedule"]["start_applied_update"])
    rows = []
    for update in plan["protocol"]["schedule"]["validation_milestones"]:
        checkpoint_source = "parent" if update == start else "continuation"
        checkpoint = (
            Path(cell["parent_checkpoint"]["path"])
            if update == start
            else directory / "checkpoints" / f"update-{update}.pt"
        )
        if not checkpoint.is_file():
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.validation.checkpoint_missing:{update}"
            )
        report_path = directory / "milestones" / f"update-{update}.json"
        _run_evaluation(plan, checkpoint, report_path, device=device)
        rows.append(
            _validation_row(
                directory,
                checkpoint,
                report_path,
                update,
                checkpoint_source=checkpoint_source,
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


def _metric_delta(
    right: Mapping[str, object], left: Mapping[str, object]
) -> dict[str, float | None]:
    return {
        name: (
            None
            if right.get(name) is None or left.get(name) is None
            else float(right[name]) - float(left[name])
        )
        for name in _METRICS
    }


def _slope(rows: list[Mapping[str, object]]) -> float:
    points = [
        (float(row["applied_updates"]), float(row["mean_objective_loss"]))
        for row in rows
    ]
    if len(points) < 2:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.aggregate.train_slope_unavailable"
        )
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    result = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / denominator
    if not math.isfinite(result):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.aggregate.train_slope_non_finite"
        )
    return result


def aggregate(root: Path, plan: Mapping[str, object]) -> dict[str, object]:
    parent_root = Path(plan["protocol"]["parent_binding"]["root"])
    combined = {}
    training = {}
    telemetry = {}
    start = int(plan["protocol"]["schedule"]["start_applied_update"])
    target = int(plan["protocol"]["schedule"]["target_applied_update"])
    for cell_id in PHASE9CC_CELLS:
        parent_validation = _read(
            parent_root / "cells" / cell_id / "validation_milestones.json"
        )["milestones"]
        continuation_validation = _read(
            root / "cells" / cell_id / "validation_milestones.json"
        )["milestones"]
        combined[cell_id] = [
            {
                **row,
                "artifact_source": "parent",
                "artifact_root": str(parent_root),
            }
            for row in parent_validation
            if int(row["update"]) < start
        ] + [
            {
                **row,
                "artifact_source": "continuation",
                "artifact_root": str(root),
            }
            for row in continuation_validation
        ]
        training[cell_id] = _read(
            root / "cells" / cell_id / "training_report.json"
        )
        parent_telemetry = _rows(
            parent_root / "cells" / cell_id / "train_telemetry.jsonl"
        )
        continuation_telemetry = _rows(
            root / "cells" / cell_id / "train_telemetry.jsonl"
        )
        telemetry[cell_id] = parent_telemetry + continuation_telemetry
    indexed = {
        cell_id: {int(row["update"]): row for row in rows}
        for cell_id, rows in combined.items()
    }
    continuation_milestones = tuple(
        int(value)
        for value in plan["protocol"]["schedule"]["validation_milestones"]
    )
    transitions = (
        (continuation_milestones[0], continuation_milestones[1]),
        (continuation_milestones[1], continuation_milestones[-1]),
        (continuation_milestones[0], continuation_milestones[-1]),
    )
    within = {
        cell_id: {
            f"{left}_to_{right}": _metric_delta(
                indexed[cell_id][right]["aggregate"],
                indexed[cell_id][left]["aggregate"],
            )
            for left, right in transitions
        }
        for cell_id in PHASE9CC_CELLS
    }
    gaps = {
        str(update): _metric_delta(
            indexed["ssl_mlp"][update]["aggregate"],
            indexed["scratch_mlp"][update]["aggregate"],
        )
        for update in continuation_milestones
    }
    best = {}
    for cell_id in PHASE9CC_CELLS:
        best_update = min(
            indexed[cell_id],
            key=lambda update: float(
                indexed[cell_id][update]["aggregate"]["mean_normalized_nll"]
            ),
        )
        best[cell_id] = {
            "selection_metric": "mean_normalized_nll_descriptive_only",
            "best_milestone": best_update,
            "final_milestone": target,
            "final_minus_best": _metric_delta(
                indexed[cell_id][target]["aggregate"],
                indexed[cell_id][best_update]["aggregate"],
            ),
        }
    report = {
        "contract_version": "1.0.0",
        "plan_fingerprint": plan["fingerprint"],
        "parent_binding_fingerprint": plan["protocol"]["parent_binding"][
            "fingerprint"
        ],
        "milestone_inventory": sorted(indexed["scratch_mlp"]),
        "cells": {
            cell_id: {
                "training": training[cell_id],
                "milestones": combined[cell_id],
                "train_loss_moving_averages": telemetry[cell_id],
                "train_loss_continuation_slope": _slope(
                    [
                        row
                        for row in telemetry[cell_id]
                        if int(row["applied_updates"]) >= start
                    ]
                ),
                "update_accounting": {
                    name: training[cell_id][name]
                    for name in (
                        "applied_updates",
                        "attempted_updates",
                        "skipped_updates",
                    )
                },
            }
            for cell_id in PHASE9CC_CELLS
        },
        "within_cell_milestone_deltas": within,
        "ssl_minus_scratch_gaps": gaps,
        "best_milestone_and_final_vs_best": best,
        "automatic_plateau_verdict": None,
        "test_access": False,
        "scientific_superiority_claimed": False,
        "statistical_significance_claimed": False,
    }
    report = {**report, "fingerprint": fingerprint(report)}
    _write(root / "convergence_report.json", report)
    return report


def _write_manifest(root: Path) -> dict[str, object]:
    excluded = {"manifest.json", "payload.sha256"}
    files = {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    manifest = {
        "contract_version": "1.0.0",
        "files": files,
        "file_count": len(files),
    }
    manifest = {**manifest, "fingerprint": fingerprint(manifest)}
    _write(root / "manifest.json", manifest)
    (root / "payload.sha256").write_text(
        f"{file_sha256(root / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    return manifest


def _verify_fingerprint(value: Mapping[str, object], error: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("fingerprint", None)
    if observed != fingerprint(unsigned):
        raise Phase9CCContinuationError(error)


def verify_bundle(
    root: Path, *, expected_sha: str | None = None
) -> dict[str, object]:
    plan = _read(root / "continuation_plan.json")
    _verify_fingerprint(
        plan, "phase9cc.continuation.verify.plan_fingerprint_invalid"
    )
    protocol = plan.get("protocol")
    if (
        not isinstance(protocol, dict)
        or plan.get("contract_version") != CONTINUATION_PLAN_VERSION
        or protocol.get("contract_version") != CONTINUATION_PROTOCOL_VERSION
        or tuple(protocol.get("cells", ())) != PHASE9CC_CELLS
        or tuple(cell.get("cell_id") for cell in plan.get("cells", ()))
        != PHASE9CC_CELLS
        or {cell.get("decoder_kind") for cell in plan["cells"]} != {"mlp"}
        or any(cell.get("encoder_reload") is not False for cell in plan["cells"])
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.protocol_invalid"
        )
    _verify_fingerprint(
        protocol, "phase9cc.continuation.verify.protocol_fingerprint_invalid"
    )
    if _read(root / "protocol.json") != protocol:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.protocol_artifact_mismatch"
        )
    if expected_sha is not None and protocol.get("git_head") != expected_sha:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.git_head_mismatch"
        )
    if any(value is not False for value in protocol["test_lock"].values()):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.test_access"
        )
    schedule = protocol["schedule"]
    if not protocol.get("bounded_test_protocol") and (
        schedule.get("start_applied_update") != 9000
        or schedule.get("target_applied_update") != 15000
        or schedule.get("additional_applied_updates") != 6000
        or schedule.get("batch_size") != 2
        or schedule.get("telemetry_interval_applied") != 100
        or schedule.get("checkpoint_interval_applied") != 1000
        or schedule.get("validation_milestones") != [9000, 12000, 15000]
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.production_protocol_invalid"
        )
    _verify_fingerprint(
        protocol["parent_binding"],
        "phase9cc.continuation.verify.parent_binding_fingerprint_invalid",
    )
    if _read(root / "parent_binding.json") != protocol["parent_binding"]:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.parent_binding_artifact_mismatch"
        )
    parent = protocol["parent_binding"]
    parent_root = Path(parent["root"])
    parent_result = verify_parent_bundle(
        parent_root, expected_sha=parent["git_sha"]
    )
    if (
        parent_result["manifest_fingerprint"] != parent["manifest_fingerprint"]
        or file_sha256(parent_root / "manifest.json")
        != parent["manifest_sha256"]
        or file_sha256(parent_root / "payload.sha256")
        != parent["payload_sha256"]
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.parent_manifest_mismatch"
        )
    config_path = Path(parent["config_path"])
    if (
        not config_path.is_file()
        or file_sha256(config_path) != parent["config_sha256"]
        or (
            not protocol.get("bounded_test_protocol")
            and (
                parent.get("git_sha") != CONTINUATION_PARENT_SHA
                or parent.get("manifest_fingerprint")
                != CONTINUATION_PARENT_MANIFEST_FINGERPRINT
                or {
                    name: value["sha256"]
                    for name, value in parent["checkpoints"].items()
                }
                != CONTINUATION_PARENT_CHECKPOINT_SHA256
            )
        )
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.parent_exact_binding_invalid"
        )
    if not protocol.get("bounded_test_protocol"):
        rebuilt = build_continuation_plan(
            parent_root,
            config_path,
            start_update=int(schedule["start_applied_update"]),
            target_update=int(schedule["target_applied_update"]),
            validation_milestones=tuple(schedule["validation_milestones"]),
        )
        if rebuilt != plan:
            raise Phase9CCContinuationError(
                "phase9cc.continuation.verify.plan_rebuild_mismatch"
            )
    identities = schedule_identities(plan)
    batch_size = int(schedule["batch_size"])
    start = int(schedule["start_applied_update"])
    target = int(schedule["target_applied_update"])
    telemetry_interval = int(schedule["telemetry_interval_applied"])
    checkpoint_interval = int(schedule["checkpoint_interval_applied"])
    parent_plan = _read(parent_root / "experiment_plan.json")
    old_identities = parent_schedule(parent_plan)
    prefix_count = start * batch_size
    if (
        identities[:prefix_count] != old_identities
        or raw_downstream_sample_schedule_fingerprint(old_identities)
        != schedule["parent_schedule_fingerprint"]
        or raw_downstream_sample_schedule_fingerprint(identities[prefix_count:])
        != schedule["continuation_schedule_fingerprint"]
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.schedule_prefix_invalid"
        )
    expected_checkpoint_updates = set(
        range(start + checkpoint_interval, target + 1, checkpoint_interval)
    )
    schedule_fingerprints = set()
    data_fingerprints = set()
    validation_memberships = set()
    checkpoint_hashes = {}
    for cell in plan["cells"]:
        cell_id = str(cell["cell_id"])
        parent_checkpoint = Path(cell["parent_checkpoint"]["path"])
        if (
            file_sha256(parent_checkpoint)
            != cell["parent_checkpoint"]["sha256"]
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.parent_checkpoint_mismatch:{cell_id}"
            )
        directory = root / "cells" / cell_id
        preflight = _read(directory / "preflight_evidence.json")
        _verify_fingerprint(
            preflight,
            f"phase9cc.continuation.verify.preflight_invalid:{cell_id}",
        )
        if (
            preflight.get("passed") is not True
            or preflight.get("update") != start
            or preflight.get("parent_checkpoint_sha256")
            != cell["parent_checkpoint"]["sha256"]
            or preflight.get("validation_membership_fingerprint")
            != protocol["validation_membership"][
                "evaluation_membership_fingerprint"
            ]
            or preflight.get("logit_replay", {}).get(
                "candidate_identities_exact"
            )
            is not True
            or preflight.get("logit_replay", {}).get("all_logits_finite")
            is not True
            or any(
                task.get("within_elementwise_tolerance") is not True
                for task in preflight.get("logit_replay", {}).get("tasks", ())
            )
            or preflight.get("metric_comparison", {}).get("within_tolerance")
            is not True
            or any(
                value is not False
                for value in preflight.get("test_lock", {}).values()
            )
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.preflight_invalid:{cell_id}"
            )
        training = _read(directory / "training_report.json")
        _verify_fingerprint(
            training,
            f"phase9cc.continuation.verify.training_fingerprint_invalid:{cell_id}",
        )
        if (
            training.get("complete") is not True
            or training.get("start_applied_update") != start
            or training.get("target_applied_update") != target
            or training.get("applied_updates") != target
            or training.get("attempted_updates")
            != target + int(training.get("skipped_updates", -1))
            or training.get("sample_schedule_position") != target
            or training.get("actual_sample_schedule_fingerprint")
            != schedule["full_schedule_fingerprint"]
            or training.get("encoder_export_reloaded") is not False
            or training.get("restore_mode")
            != "model_optimizer_scaler_scheduler_rng_sampler"
            or training.get("test_access") is not False
            or not isinstance(training.get("resume_evidence"), dict)
            or training["resume_evidence"].get(
                "loader_advanced_before_rng_restore"
            )
            is not True
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.training_invalid:{cell_id}"
            )
        telemetry = _rows(directory / "train_telemetry.jsonl")
        expected_updates = list(
            range(start + telemetry_interval, target + 1, telemetry_interval)
        )
        if (
            [row.get("applied_updates") for row in telemetry] != expected_updates
            or len({row.get("applied_updates") for row in telemetry})
            != len(expected_updates)
            or fingerprint(telemetry) != training["telemetry_fingerprint"]
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.telemetry_invalid:{cell_id}"
            )
        for row in telemetry:
            if any(
                not math.isfinite(float(row[name]))
                for name in (
                    "mean_objective_loss",
                    "learning_rate",
                    "grad_scaler_scale",
                    "mean_gradient_norm_before_clip",
                    "last_gradient_norm_before_clip",
                )
            ):
                raise Phase9CCContinuationError(
                    f"phase9cc.continuation.verify.telemetry_non_finite:{cell_id}"
                )
            consumed = int(row["sample_count_consumed"])
            if row.get("sample_schedule_prefix_fingerprint") != (
                raw_downstream_sample_schedule_fingerprint(
                    identities[:consumed]
                )
            ):
                raise Phase9CCContinuationError(
                    f"phase9cc.continuation.verify.telemetry_schedule_invalid:{cell_id}"
                )
        checkpoints = {
            int(path.stem.split("-", 1)[1]): path
            for path in (directory / "checkpoints").glob("update-*.pt")
        }
        if set(checkpoints) != expected_checkpoint_updates:
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.checkpoint_inventory_invalid:{cell_id}"
            )
        for update, path in checkpoints.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            metadata = payload.get("metadata", {})
            progress = payload.get("progress", {})
            if (
                metadata.get("phase9cc_continuation_checkpoint_version")
                != CONTINUATION_CHECKPOINT_VERSION
                or metadata.get("plan_fingerprint") != plan["fingerprint"]
                or metadata.get("cell_id") != cell_id
                or metadata.get("parent_checkpoint_sha256")
                != cell["parent_checkpoint"]["sha256"]
                or metadata.get("schedule_fingerprint")
                != schedule["full_schedule_fingerprint"]
                or metadata.get("model_contract")
                != model_contract_metadata_from_payload(payload)
                or progress.get("applied_updates") != update
                or progress.get("schedule_position") != update
                or progress.get("attempted_updates")
                != update + int(progress.get("skipped_updates", -1))
                or progress.get("schedule_prefix_fingerprint")
                != raw_downstream_sample_schedule_fingerprint(
                    identities[: update * batch_size]
                )
                or model_state_fingerprint(payload.get("model_state"))
                != payload.get("model_state_fingerprint")
            ):
                raise Phase9CCContinuationError(
                    "phase9cc.continuation.verify.checkpoint_invalid:"
                    f"{cell_id}:{update}"
                )
        validation = _read(directory / "validation_milestones.json")
        _verify_fingerprint(
            validation,
            f"phase9cc.continuation.verify.validation_fingerprint_invalid:{cell_id}",
        )
        rows = validation.get("milestones", [])
        if (
            validation.get("test_access") is not False
            or [row.get("update") for row in rows]
            != schedule["validation_milestones"]
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.milestones_invalid:{cell_id}"
            )
        for row in rows:
            update = int(row["update"])
            checkpoint = parent_checkpoint if update == start else checkpoints[update]
            report_path = directory / row["validation_report_path"]
            report = _read(report_path)
            checkpoint_payload = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
            unsigned_report = dict(report)
            report_fingerprint = unsigned_report.pop("fingerprint", None)
            if (
                row.get("checkpoint_sha256") != file_sha256(checkpoint)
                or row.get("validation_report_sha256") != file_sha256(report_path)
                or row.get("validation_report_fingerprint")
                != report.get("fingerprint")
                or report_fingerprint != fingerprint(unsigned_report)
                or row.get("validation_membership_fingerprint")
                != protocol["validation_membership"][
                    "evaluation_membership_fingerprint"
                ]
                or row.get("model_state_fingerprint")
                != checkpoint_payload.get("model_state_fingerprint")
                or row.get("checkpoint_declared_model_state_fingerprint")
                != checkpoint_payload.get("model_state_fingerprint")
            ):
                raise Phase9CCContinuationError(
                    "phase9cc.continuation.verify.validation_binding_invalid:"
                    f"{cell_id}:{update}"
                )
            _verify_metrics(cell_id, update, row)
            validation_memberships.add(row["validation_membership_fingerprint"])
        schedule_fingerprints.add(training["actual_sample_schedule_fingerprint"])
        data_fingerprints.add(fingerprint(training["data_fingerprints"]))
        checkpoint_hashes[cell_id] = file_sha256(checkpoints[target])
    if (
        len(schedule_fingerprints) != 1
        or len(data_fingerprints) != 1
        or len(validation_memberships) != 1
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.cell_pairing_invalid"
        )
    report = _read(root / "convergence_report.json")
    _verify_fingerprint(
        report, "phase9cc.continuation.verify.convergence_fingerprint_invalid"
    )
    if (
        report.get("plan_fingerprint") != plan["fingerprint"]
        or (
            not protocol.get("bounded_test_protocol")
            and report.get("milestone_inventory")
            != [0, 1000, 3000, 6000, 9000, 12000, 15000]
        )
        or report.get("automatic_plateau_verdict") is not None
        or report.get("test_access") is not False
        or report.get("scientific_superiority_claimed") is not False
        or report.get("statistical_significance_claimed") is not False
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.convergence_claim_invalid"
        )
    for cell_id in PHASE9CC_CELLS:
        cell_report = report.get("cells", {}).get(cell_id, {})
        milestones = cell_report.get("milestones", ())
        if (
            [row.get("update") for row in milestones]
            != report["milestone_inventory"]
            or not math.isfinite(
                float(cell_report.get("train_loss_continuation_slope"))
            )
        ):
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.convergence_cell_invalid:{cell_id}"
            )
        for row in milestones:
            _verify_metrics(cell_id, int(row["update"]), row)
    manifest = _read(root / "manifest.json")
    _verify_fingerprint(
        manifest, "phase9cc.continuation.verify.manifest_fingerprint_invalid"
    )
    for relative, digest in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise Phase9CCContinuationError(
                f"phase9cc.continuation.verify.artifact_hash_invalid:{relative}"
            )
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "payload.sha256"}
    }
    if actual_files != set(manifest.get("files", {})):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.file_inventory_invalid"
        )
    expected_payload = f"{file_sha256(root / 'manifest.json')}  manifest.json\n"
    if (root / "payload.sha256").read_text(encoding="utf-8") != expected_payload:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.verify.payload_digest_invalid"
        )
    return {
        "status": "verified",
        "cell_count": 2,
        "start_applied_update": start,
        "final_applied_update": target,
        "additional_applied_updates": target - start,
        "checkpoint_hashes": checkpoint_hashes,
        "parent_manifest_fingerprint": parent["manifest_fingerprint"],
        "manifest_fingerprint": manifest["fingerprint"],
        "resume_boundary": "applied_update_mid_epoch",
        "test_access": False,
    }


def model_contract_metadata_from_payload(
    payload: Mapping[str, object],
) -> object:
    """Return the independently state-checked contract already in payload."""

    metadata = payload.get("metadata", {})
    contract = metadata.get("model_contract")
    state = payload.get("model_state")
    if not isinstance(contract, dict) or not isinstance(state, dict):
        return None
    try:
        model = DilemmadataHierarchicalModel(
            dilemmadata_config_from_model_contract(contract, state)
        )
        model.load_state_dict(state, strict=True)
    except Exception:
        return None
    return model_contract_metadata(model)


def execute(
    root: Path,
    plan: Mapping[str, object],
    *,
    device: str | None = None,
) -> dict[str, object]:
    production = not bool(plan["protocol"].get("bounded_test_protocol"))
    resolved_device = device or ("cuda:0" if production else "cpu")
    if production and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3090"
        or resolved_device != "cuda:0"
    ):
        raise Phase9CCContinuationError(
            "phase9cc.continuation.hardware.rtx3090_cuda0_required"
        )
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "continuation_plan.json"
    if plan_path.is_file():
        if _read(plan_path) != plan:
            raise Phase9CCContinuationError(
                "phase9cc.continuation.resume.plan_mismatch"
            )
    else:
        _write(plan_path, plan)
        _write(root / "protocol.json", plan["protocol"])
        _write(root / "parent_binding.json", plan["protocol"]["parent_binding"])
    if (root / "manifest.json").is_file():
        return verify_bundle(root, expected_sha=plan["protocol"]["git_head"])

    # Every parent checkpoint must reproduce before either cell may optimize.
    for cell in plan["cells"]:
        _preflight_cell(root, plan, cell, device=resolved_device)
    for cell in plan["cells"]:
        directory = root / "cells" / str(cell["cell_id"])
        report_path = directory / "training_report.json"
        report = _read(report_path) if report_path.is_file() else None
        if report is None or report.get("complete") is not True:
            run_cell_training(
                plan,
                cell,
                directory,
                action=(
                    "resume"
                    if (directory / "train_telemetry.jsonl").exists()
                    or (directory / "checkpoints").exists()
                    else "run"
                ),
                device=resolved_device,
            )
    for cell in plan["cells"]:
        evaluate_milestones(root, plan, cell, device=resolved_device)
    report = aggregate(root, plan)
    return {
        "status": "execution_complete_pending_manifest",
        "convergence_report_fingerprint": report["fingerprint"],
        "test_access": False,
    }


def finalize(root: Path, *, expected_sha: str) -> dict[str, object]:
    plan = _read(root / "continuation_plan.json")
    if plan["protocol"].get("git_head") != expected_sha:
        raise Phase9CCContinuationError(
            "phase9cc.continuation.finalize.git_head_mismatch"
        )
    if not (root / "execution.log").is_file():
        raise Phase9CCContinuationError(
            "phase9cc.continuation.finalize.execution_log_missing"
        )
    aggregate(root, plan)
    _write_manifest(root)
    return verify_bundle(root, expected_sha=expected_sha)


__all__ = ["aggregate", "execute", "finalize", "verify_bundle"]
