"""Bounded sequential two-arm TRAIN/recovery workflow with immutable resume.

Data authority and experiment admission belong to the caller. This module binds
those admitted inputs, reuses the reviewed update engine, and never discovers a
corpus. A resume imports only explicitly hashed, confined artifacts into a fresh
attempt directory; held-out metrics are recomputed from their bound full states.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
import stat
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Callable

import torch

from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.ssl.group_masking_accounting import validate_successful_row
from music_critic.ssl.group_masking_pilot import (
    POLICIES, GroupMaskingPilot, GroupPilotConfig, _runtime_binding, _sha,
    tree_fingerprint,
)
from music_critic.ssl.group_masking_scientific_eval import (
    compare_theory_endpoints, evaluate_scientific_validation,
)

WORKFLOW_SCHEMA = "WS2GroupMaskingScientificWorkflow@1.0.0"
RESUME_SCHEMA = "WS2GroupMaskingScientificResume@1.0.0"
MAX_JSON_BYTES = 128 * 1024**2
MAX_CHECKPOINT_BYTES = 256 * 1024**2
# Each entry names the next operation, never a checkpoint-selection candidate.
PHASES = (
    ("evaluation", "baseline", None),
    ("training", POLICIES[0], "pilot"),
    ("evaluation", POLICIES[0], "pilot"),
    ("training", POLICIES[0], "recovery"),
    ("evaluation", POLICIES[0], "recovery"),
    ("training", POLICIES[1], "pilot"),
    ("evaluation", POLICIES[1], "pilot"),
    ("training", POLICIES[1], "recovery"),
    ("evaluation", POLICIES[1], "recovery"),
)
RECOVERY_TRANSITION = {
    "kind": "full_endpoint_harness_then_fresh_Adam",
    "preserve": "all_shared_theory_and_SSL_tensors_including_inactive_modules",
    "optimizer_state": "explicit_reset_to_empty",
    "loss": "existing_theory_only_lambda_SSL_0_lambda_theory_1",
    "schedule": "identical_predeclared_recovery_schedule_for_both_endpoints",
}


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("group_workflow.duplicate_json_key")
        result[key] = value
    return result


def _decode(raw):
    def reject(_):
        raise ValueError("group_workflow.nonfinite_json")
    return json.loads(raw, object_pairs_hook=_unique_json, parse_constant=reject)


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_bytes(path, raw):
    """Atomic exclusive publication; failed writes never replace prior bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink()


def _publish_json(root, relative, value):
    raw = _bytes(value)
    _publish_bytes(root / relative, raw)
    return {"path": relative, "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _read_bound(path, expected_sha256, expected_size_bytes, maximum):
    _sha(expected_sha256)
    if type(expected_size_bytes) is not int or not 0 < expected_size_bytes <= maximum:
        raise ValueError("group_workflow.artifact_size_bound")
    path = Path(path)
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("group_workflow.symlink_artifact_denied")
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size_bytes:
        raise ValueError("group_workflow.regular_bound_artifact_required")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("group_workflow.artifact_changed_during_read")
        raw = stream.read(expected_size_bytes + 1)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("group_workflow.artifact_changed_during_read")
    if len(raw) != expected_size_bytes or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("group_workflow.artifact_mismatch")
    return raw


def _confined_artifact(root, ref, *, checkpoint=False):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256", "size_bytes"}:
        raise ValueError("group_workflow.artifact_reference_invalid")
    relative = ref["path"]
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("group_workflow.artifact_path_invalid")
    parts = PurePosixPath(relative)
    prefix = "checkpoints" if checkpoint else "evaluations"
    if (parts.is_absolute() or len(parts.parts) != 2 or parts.parts[0] != prefix
            or any(part in {"", ".", ".."} for part in parts.parts)
            or str(parts) != relative):
        raise ValueError("group_workflow.artifact_path_invalid")
    root = Path(root).resolve(strict=True)
    candidate = root / relative
    if any((root.joinpath(*parts.parts[:i])).is_symlink() for i in range(1, len(parts.parts) + 1)):
        raise ValueError("group_workflow.symlink_artifact_denied")
    if not candidate.resolve(strict=True).is_relative_to(root):
        raise ValueError("group_workflow.artifact_path_escape")
    raw = _read_bound(candidate, ref["sha256"], ref["size_bytes"],
                      MAX_CHECKPOINT_BYTES if checkpoint else MAX_JSON_BYTES)
    return candidate, raw


def _key(policy, stage):
    return "baseline" if policy == "baseline" else policy + "." + stage


def make_theory_recovery_pilot(endpoint: GroupMaskingPilot, config: GroupPilotConfig, *,
                                schedule: tuple[str, ...], parent_checkpoint: dict):
    """Retain the complete endpoint while explicitly resetting Adam for recovery.

    This is a stage transition, not a fake model-only warmstart. The new stage
    keeps the reviewed GroupMaskingPilot step/checkpoint/resume implementation;
    its binding explicitly names the endpoint artifact and complete state.
    """
    if endpoint.failed or endpoint.completed_updates != endpoint.config.updates:
        raise ValueError("group_workflow.completed_endpoint_required")
    if config.lambda_ssl != 0 or config.lambda_theory != 1:
        raise ValueError("group_workflow.theory_only_recovery_required")
    if (config.seed != endpoint.config.seed or config.learning_rate != endpoint.config.learning_rate
            or config.ssl_config() != endpoint.config.ssl_config()):
        raise ValueError("group_workflow.recovery_protocol_drift")
    if not isinstance(schedule, tuple) or len(schedule) != config.updates:
        raise ValueError("group_workflow.complete_recovery_schedule_required")
    for entry in schedule:
        _sha(entry)
    if set(parent_checkpoint) != {"path", "sha256", "size_bytes"}:
        raise ValueError("group_workflow.parent_checkpoint_binding_required")
    _sha(parent_checkpoint["sha256"])
    if type(parent_checkpoint["size_bytes"]) is not int or not 0 < parent_checkpoint["size_bytes"] <= MAX_CHECKPOINT_BYTES:
        raise ValueError("group_workflow.parent_checkpoint_size_invalid")
    endpoint._validate_runtime()
    endpoint._validate_finite_state()
    recovery = GroupMaskingPilot.__new__(GroupMaskingPilot)
    recovery.config, recovery.policy, recovery.schedule = config, endpoint.policy, schedule
    recovery.device = endpoint.device
    recovery.harness = deepcopy(endpoint.harness)
    recovery.optimizer = torch.optim.Adam(recovery.harness.optimizer_parameters(), lr=config.learning_rate)
    recovery.initial_fingerprint = model_state_fingerprint(recovery.harness)
    recovery.binding = {
        "contract": config.contract(), "policy": endpoint.policy, "schedule": list(schedule),
        "source": deepcopy(endpoint.binding["source"]), "initial_state": recovery.initial_fingerprint,
        "transfer_report": deepcopy(endpoint.binding["transfer_report"]),
        "optimizer_inventory": recovery.harness.parameter_inventory(),
        "device": str(recovery.device), "runtime": _runtime_binding(recovery.device),
        "stage_transition": {**deepcopy(RECOVERY_TRANSITION),
            "parent_checkpoint_sha256": parent_checkpoint["sha256"],
            "parent_checkpoint_size_bytes": parent_checkpoint["size_bytes"],
            "parent_binding_fingerprint": endpoint.binding_fingerprint,
            "parent_full_state_fingerprint": recovery.initial_fingerprint},
    }
    recovery.binding_fingerprint = tree_fingerprint(recovery.binding)
    recovery.completed_updates, recovery.rows, recovery.failed, recovery.last_attempt = 0, [], False, None
    recovery._validate_finite_state()
    return recovery


def _metric_control(left, right):
    """Beat-minus-random absolute differences; no winner/selection rule."""
    metrics = ("accuracy", "balanced_accuracy", "macro_f1_observed_validation_classes", "masked_cross_entropy")
    # Existing comparison validates coverage and common held-out bindings.
    checked = compare_theory_endpoints(left, right, right)
    return {"membership_fingerprint": checked["membership_fingerprint"],
            "orientation": "beat_minus_independent_note", "per_head": {
                task: {"support": row["support"], "metrics": {
                    metric: row["metrics"][metric]["endpoint_minus_baseline"] for metric in metrics}}
                for task, row in checked["per_head"].items()}}


def _accounting(operations):
    stages = {}
    for op in operations:
        name = op["key"]
        if op["kind"] == "validation":
            name += ".validation_replay" if op["purpose"] == "resume_verification" else ".validation"
        row = stages.setdefault(name, {"attempts": 0, "confirmed_updates": 0, "replayed_confirmed_updates": 0,
            "replayed_after_unknown_attempts": 0, "observed_encoder_forwards": 0,
            "optimizer_outcomes": {"applied": 0, "not_applied": 0, "unknown": 0},
            "ssl_draws_from_confirmed_rows": 0, "theory_draws_from_confirmed_rows": 0,
            "operation_wall_seconds": 0.0})
        row["attempts"] += 1
        row["operation_wall_seconds"] += op["wall_seconds"]
        receipt = op["receipt"]
        row["observed_encoder_forwards"] += receipt.get("observed_encoder_forwards", 0)
        if op["kind"] == "training":
            row["optimizer_outcomes"][receipt.get("optimizer_outcome", "not_applied")] += 1
            row["replayed_after_unknown_attempts"] += int("unknown" in op["prior_optimizer_outcomes"])
            if op["row"] is not None:
                row["confirmed_updates"] += 1
                row["replayed_confirmed_updates"] += int(op["replay"])
                row["ssl_draws_from_confirmed_rows"] += op["row"]["counts"]["ssl_samples"]
                row["theory_draws_from_confirmed_rows"] += op["row"]["counts"]["theory_samples"]
    return {"stages": stages,
            "confirmed_updates": sum(r["confirmed_updates"] for r in stages.values()),
            "replayed_confirmed_updates": sum(r["replayed_confirmed_updates"] for r in stages.values()),
            "observed_encoder_forwards": sum(r["observed_encoder_forwards"] for r in stages.values()),
            "accounting_boundary": "completed_receipts_only_device_work_after_last_durable_receipt_may_be_unknown",
            "rollback_claim": False}


class _Workflow:
    def __init__(self, *, warmstart_factory, source_binding, data, pilot_config, recovery_config,
                 output_root, contract_binding, device, checkpoint_interval, fixture_mode,
                 attempt_id, event_sink):
        self.owned_output_created = False
        self.started = time.monotonic()
        self.bootstrap_stage = "admission"
        if type(checkpoint_interval) is not int or not 1 <= checkpoint_interval <= 200:
            raise ValueError("group_workflow.checkpoint_interval_invalid")
        if recovery_config.lambda_ssl != 0 or recovery_config.lambda_theory != 1:
            raise ValueError("group_workflow.theory_only_recovery_required")
        if (pilot_config.seed != recovery_config.seed or pilot_config.learning_rate != recovery_config.learning_rate
                or pilot_config.ssl_config() != recovery_config.ssl_config()):
            raise ValueError("group_workflow.recovery_protocol_drift")
        if not fixture_mode and (source_binding.get("kind") != "C2_model_only"
                or pilot_config != GroupPilotConfig()
                or recovery_config != GroupPilotConfig(updates=100, lambda_ssl=0, lambda_theory=1)):
            raise ValueError("group_workflow.fixed_scientific_protocol_required")
        self.root = Path(output_root)
        if self.root.exists():
            raise FileExistsError("group_workflow.fresh_attempt_output_required")
        self.attempt_id = attempt_id or self.root.name
        if not isinstance(self.attempt_id, str) or not self.attempt_id or len(self.attempt_id) > 128:
            raise ValueError("group_workflow.attempt_id_invalid")
        self.warmstart_factory, self.source_binding, self.data = warmstart_factory, source_binding, data
        self.configs = {"pilot": pilot_config, "recovery": recovery_config}
        self.schedules = {stage: tuple(data.schedule_bindings(stage)) for stage in self.configs}
        for stage, config in self.configs.items():
            if len(self.schedules[stage]) != config.updates:
                raise ValueError("group_workflow.complete_schedule_required")
            for entry in self.schedules[stage]:
                _sha(entry)
        self.device, self.interval, self.event_sink = device, checkpoint_interval, event_sink
        self.phase, self.checkpoints, self.evaluations, self.operations, self.attempt_ids = 0, {}, {}, [], [self.attempt_id]
        self.manifest_index, self.last_manifest, self.parent_manifest = 0, None, None
        self.operation_started = {}
        self.root.mkdir(parents=True, exist_ok=False)
        self.owned_output_created = True
        for name in ("receipts.jsonl", "execution.jsonl", "workflow-receipts.jsonl"):
            with (self.root / name).open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(self.root)
        self.bootstrap_stage = "warmstart_and_initial_harness"
        self._append("workflow-receipts.jsonl", {"schema": WORKFLOW_SCHEMA,
            "attempt_id": self.attempt_id, "status": "started", "stage": self.bootstrap_stage,
            "source_validation": "pending", "confirmed_updates": 0,
            "optimizer_outcome": "not_applied", "observed_encoder_forwards": 0})
        self.base = self._pilot(POLICIES[0])
        self.bootstrap_stage = "binding"
        self.binding = _json_value({"schema": WORKFLOW_SCHEMA, "contract": contract_binding,
            "source": source_binding, "data": data.binding,
            "pilot": pilot_config.contract(), "recovery": recovery_config.contract(),
            "recovery_transition": RECOVERY_TRANSITION,
            "runtime": self.base.binding["runtime"], "initial_state": self.base.initial_fingerprint,
            "checkpoint_interval": checkpoint_interval, "fixture_mode": fixture_mode,
            "selection": "fixed_endpoints_no_selection", "test_access": "closed"})
        self.binding_fingerprint = tree_fingerprint(self.binding)
        _publish_json(self.root, "plan.json", {"binding": self.binding, "binding_fingerprint": self.binding_fingerprint,
            "attempt_id": self.attempt_id, "external_resource_guard_required": True})
        self.bootstrap_stage = "ready"
        self._append("workflow-receipts.jsonl", {"schema": WORKFLOW_SCHEMA,
            "attempt_id": self.attempt_id, "status": "completed", "stage": "initial_binding",
            "source_validation": "verified_by_warmstart_and_pilot", "confirmed_updates": 0,
            "binding_fingerprint": self.binding_fingerprint,
            "optimizer_outcome": "not_applied", "observed_encoder_forwards": 0})

    def _pilot(self, policy):
        result = GroupMaskingPilot(self.warmstart_factory(), self.configs["pilot"], policy=policy,
            schedule=self.schedules["pilot"], source_binding=self.source_binding, device=self.device)
        if hasattr(self, "base") and result.initial_fingerprint != self.base.initial_fingerprint:
            raise ValueError("group_workflow.paired_initial_state_mismatch")
        return result

    def _append(self, name, value):
        with (self.root / name).open("ab") as stream:
            stream.write(_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())

    def _operation(self, kind, key, *, update=None, purpose="primary"):
        prior = [op for op in self.operations if op["kind"] == kind and op["key"] == key and op["update"] == update]
        op = {"attempt_id": self.attempt_id, "operation_id": sum(x["attempt_id"] == self.attempt_id for x in self.operations),
              "kind": kind, "key": key, "update": update, "purpose": purpose,
              "replay": any(x["row"] is not None for x in prior) if kind == "training" else bool(prior),
              "prior_optimizer_outcomes": [x["receipt"]["optimizer_outcome"] for x in prior] if kind == "training" else [],
              "receipt": {}, "row": None, "wall_seconds": 0.0}
        self.operation_started[(op["attempt_id"], op["operation_id"])] = time.monotonic()
        self.operations.append(op)
        return op

    def _receipt(self, op, receipt):
        op["receipt"] = deepcopy(receipt)
        op["wall_seconds"] = time.monotonic() - self.operation_started[(op["attempt_id"], op["operation_id"])]
        event = {"attempt_id": op["attempt_id"], "operation_id": op["operation_id"],
                 "kind": op["kind"], "key": op["key"], "purpose": op["purpose"], "receipt": receipt}
        self._append("receipts.jsonl", event)
        if self.event_sink is not None:
            self.event_sink(deepcopy(event))

    def _save(self, key, arm):
        name = "checkpoints/" + key + f"-{arm.completed_updates:06d}.pt"
        evidence = arm.save(self.root / name)
        _fsync_directory((self.root / name).parent)
        self.checkpoints[key] = {**evidence, "path": name}

    def _manifest(self, reason, *, terminal=False):
        payload = {"schema": RESUME_SCHEMA, "binding": self.binding,
            "binding_fingerprint": self.binding_fingerprint, "phase": self.phase,
            "checkpoints": deepcopy(self.checkpoints), "evaluations": deepcopy(self.evaluations),
            "operations": deepcopy(self.operations), "attempt_ids": list(self.attempt_ids),
            "attempt_id": self.attempt_id, "terminal_attempt": terminal, "reason": reason,
            "parent_manifest": self.parent_manifest,
            "post_manifest_compute": "not_observed_by_this_immutable_manifest" if not terminal else "attempt_closed"}
        payload = _json_value(payload)
        payload["payload_fingerprint"] = tree_fingerprint(payload)
        relative = f"workflow-{self.manifest_index:06d}.json"
        self.manifest_index += 1
        self.last_manifest = _publish_json(self.root, relative, payload)
        return self.last_manifest

    def _load_arm(self, key):
        if key == "baseline":
            arm = self._pilot(POLICIES[0])
        else:
            policy, stage = key.rsplit(".", 1)
            if stage == "pilot":
                arm = self._pilot(policy)
            else:
                parent_key = _key(policy, "pilot")
                endpoint = self._load_arm(parent_key)
                arm = make_theory_recovery_pilot(endpoint, self.configs["recovery"],
                    schedule=self.schedules["recovery"], parent_checkpoint=self.checkpoints[parent_key])
                del endpoint
        ref = self.checkpoints[key]
        path, _ = _confined_artifact(self.root, ref, checkpoint=True)
        arm.resume(path, expected_sha256=ref["sha256"], expected_size_bytes=ref["size_bytes"])
        if key == "baseline" and arm.completed_updates != 0:
            raise ValueError("group_workflow.baseline_checkpoint_not_initial")
        if key != "baseline":
            for row in arm.rows:
                matches = [op for op in self.operations if op["kind"] == "training" and op["key"] == key
                           and op["update"] == row["update"] and op["row"] == row]
                if not matches:
                    raise ValueError("group_workflow.checkpoint_execution_prefix_mismatch")
        return arm

    def _evaluate(self, key, arm, *, purpose="primary"):
        op = self._operation("validation", key, purpose=purpose)
        result = evaluate_scientific_validation(arm.harness,
            batch_factory=self.data.validation_batches, config=self.configs["pilot"],
            expected_membership=self.data.validation_membership,
            expected_batch_bindings=self.data.validation_bindings,
            on_receipt=lambda receipt: self._receipt(op, receipt),
            runtime_validator=arm._validate_runtime,
            train_seen_tuples=getattr(self.data, "train_seen_tuples", None),
            resource_limits=getattr(self.data, "validation_resource_limits", None))
        for previous in self.operations:
            if (previous["kind"] == "validation" and previous["key"] == key
                    and previous["receipt"].get("status") == "completed"
                    and (previous["receipt"]["observed_encoder_forwards"] != result["observed_encoder_forwards"]
                         or previous["receipt"]["confirmed_batches"] != result["confirmed_batches"])):
                raise ValueError("group_workflow.evaluation_receipt_replay_mismatch")
        if purpose == "primary":
            self.evaluations[key] = _publish_json(self.root, "evaluations/" + key + ".json", result)
        return result

    def _validate_operations(self, operations, attempt_ids):
        if not isinstance(operations, list) or len(operations) > 10000:
            raise ValueError("group_workflow.operation_history_bound")
        expected_keys = {"attempt_id", "operation_id", "kind", "key", "update", "purpose", "replay",
                         "prior_optimizer_outcomes", "receipt", "row", "wall_seconds"}
        seen_ids, earlier_rows, earlier_outcomes = set(), {}, {}
        next_ids, previous_phase, previous_attempt = {}, {}, -1
        validation_seen = set()
        training_receipt_keys = {"schema", "update", "completed_updates_before", "batch_binding",
            "status", "stage", "observed_encoder_forwards", "optimizer_outcome", "confirmed_updates"}
        training_stages = {"started", "runtime_validation", "factory", "batch_validation", "masking",
            "device_transfer", "forward", "backward", "optimizer_step", "post_step_validation", "accounting", "completed"}
        stage_keys = {_key(p, s) for p in POLICIES for s in self.configs}
        for op in operations:
            if not isinstance(op, dict) or set(op) != expected_keys or op["attempt_id"] not in attempt_ids:
                raise ValueError("group_workflow.operation_schema_invalid")
            identity = (op["attempt_id"], op["operation_id"])
            if type(op["operation_id"]) is not int or op["operation_id"] < 0 or identity in seen_ids:
                raise ValueError("group_workflow.operation_identity_invalid")
            if op["operation_id"] != next_ids.get(op["attempt_id"], 0):
                raise ValueError("group_workflow.operation_sequence_invalid")
            next_ids[op["attempt_id"]] = op["operation_id"] + 1
            attempt_position = attempt_ids.index(op["attempt_id"])
            if attempt_position < previous_attempt:
                raise ValueError("group_workflow.attempt_sequence_invalid")
            previous_attempt = attempt_position
            seen_ids.add(identity)
            receipt = op["receipt"]
            if type(op["wall_seconds"]) not in (float, int) or not 0 <= op["wall_seconds"] < float("inf"):
                raise ValueError("group_workflow.operation_wall_accounting_invalid")
            if (not isinstance(receipt, dict) or receipt.get("status") not in {"started", "in_progress", "completed", "failed"}
                    or type(receipt.get("observed_encoder_forwards")) is not int or receipt["observed_encoder_forwards"] < 0
                    or receipt.get("optimizer_outcome") not in {"applied", "not_applied", "unknown"}
                    or type(op["replay"]) is not bool):
                raise ValueError("group_workflow.operation_receipt_invalid")
            if op["kind"] == "training":
                if op["key"] not in stage_keys or op["purpose"] != "primary":
                    raise ValueError("group_workflow.operation_stage_invalid")
                policy, stage = op["key"].rsplit(".", 1)
                config = self.configs[stage]
                update = op["update"]
                if type(update) is not int or not 1 <= update <= config.updates:
                    raise ValueError("group_workflow.operation_update_invalid")
                maximum_forwards = 2 * int(config.lambda_ssl > 0) + int(config.lambda_theory > 0)
                allowed_keys = training_receipt_keys | ({"error_type", "reason"} if receipt["status"] == "failed" else set())
                if (set(receipt) != allowed_keys or receipt.get("stage") not in training_stages
                        or (receipt["status"] == "failed" and (not isinstance(receipt["reason"], str) or not isinstance(receipt["error_type"], str)))
                        or receipt.get("schema") != "WS2GroupMaskingAttempt@1.0.0" or receipt.get("update") != update
                        or receipt.get("completed_updates_before") != update - 1
                        or receipt.get("batch_binding") != self.schedules[stage][update - 1]
                        or receipt["observed_encoder_forwards"] > maximum_forwards
                        or type(receipt.get("confirmed_updates")) is not int
                        or receipt["confirmed_updates"] not in {update - 1, update}):
                    raise ValueError("group_workflow.training_receipt_invalid")
                if ((receipt["optimizer_outcome"] in {"applied", "unknown"} and receipt["observed_encoder_forwards"] != maximum_forwards)
                        or (receipt["optimizer_outcome"] == "unknown" and receipt["stage"] != "optimizer_step")
                        or (receipt["optimizer_outcome"] == "applied" and receipt["stage"] not in {"post_step_validation", "accounting", "completed"})
                        or (receipt["status"] == "completed" and receipt["stage"] != "completed")):
                    raise ValueError("group_workflow.optimizer_receipt_stage_invalid")
                draw = (op["key"], update)
                if op["replay"] != (draw in earlier_rows) or op["prior_optimizer_outcomes"] != earlier_outcomes.get(draw, []):
                    raise ValueError("group_workflow.replay_accounting_invalid")
                if op["row"] is not None:
                    validate_successful_row(op["row"], config=config, policy=policy,
                        scheduled_binding=self.schedules[stage][update - 1], update_index=update - 1)
                    if (receipt["confirmed_updates"] != update or receipt["optimizer_outcome"] != "applied"
                            or receipt["observed_encoder_forwards"] != maximum_forwards):
                        raise ValueError("group_workflow.confirmed_receipt_mismatch")
                    if draw in earlier_rows and earlier_rows[draw] != op["row"]:
                        raise ValueError("group_workflow.replayed_row_mismatch")
                    earlier_rows[draw] = op["row"]
                elif receipt["confirmed_updates"] != update - 1 or receipt["status"] == "completed":
                    raise ValueError("group_workflow.confirmed_row_missing")
                earlier_outcomes.setdefault(draw, []).append(receipt["optimizer_outcome"])
            elif op["kind"] == "validation":
                if (op["key"] not in stage_keys | {"baseline"} or op["update"] is not None
                        or op["purpose"] not in {"primary", "resume_verification"}
                        or op["row"] is not None or op["prior_optimizer_outcomes"] != []
                        or receipt.get("schema") != "WS2GroupMaskingValidationReceipt@1.0.0"
                        or receipt["optimizer_outcome"] != "not_applied"
                        or receipt["observed_encoder_forwards"] > 5 * len(self.data.validation_bindings)
                        or type(receipt.get("confirmed_batches")) is not int
                        or not 0 <= receipt["confirmed_batches"] <= len(self.data.validation_bindings)
                        or receipt["observed_encoder_forwards"] < receipt["confirmed_batches"]
                        or (receipt["status"] == "completed" and (receipt["confirmed_batches"] != len(self.data.validation_bindings)
                            or receipt.get("stage") != "completed"))):
                    raise ValueError("group_workflow.validation_receipt_invalid")
                if op["replay"] != (op["key"] in validation_seen):
                    raise ValueError("group_workflow.validation_replay_accounting_invalid")
                validation_seen.add(op["key"])
            else:
                raise ValueError("group_workflow.operation_kind_invalid")
            if op["purpose"] == "primary":
                matches = [i for i, (kind, policy, stage) in enumerate(PHASES)
                           if ("validation" if kind == "evaluation" else kind) == op["kind"] and _key(policy, stage) == op["key"]]
                if not matches or matches[0] < previous_phase.get(op["attempt_id"], -1):
                    raise ValueError("group_workflow.operation_phase_order_invalid")
                previous_phase[op["attempt_id"]] = matches[0]

    def _validate_phase(self):
        if type(self.phase) is not int or not 0 <= self.phase <= len(PHASES):
            raise ValueError("group_workflow.phase_invalid")
        required_evals = {_key(policy, stage) for kind, policy, stage in PHASES[:self.phase] if kind == "evaluation"}
        required_stages = {_key(policy, stage) for kind, policy, stage in PHASES[:self.phase] if kind == "training"}
        allowed_stages = set(required_stages)
        if self.phase < len(PHASES) and PHASES[self.phase][0] == "training":
            allowed_stages.add(_key(*PHASES[self.phase][1:]))
        if set(self.evaluations) != required_evals or not required_stages <= set(self.checkpoints) or set(self.checkpoints) - {"baseline"} - allowed_stages:
            raise ValueError("group_workflow.phase_artifact_mismatch")
        if "baseline" not in self.checkpoints:
            raise ValueError("group_workflow.baseline_checkpoint_required")
        allowed_evals = set(required_evals)
        if self.phase < len(PHASES) and PHASES[self.phase][0] == "evaluation":
            allowed_evals.add(_key(*PHASES[self.phase][1:]))
        for op in self.operations:
            if op["kind"] == "training" and op["key"] not in allowed_stages:
                raise ValueError("group_workflow.operation_after_manifest_phase")
            if op["kind"] == "validation" and op["key"] not in allowed_evals:
                raise ValueError("group_workflow.operation_after_manifest_phase")
        for key, ref in self.evaluations.items():
            _, raw = _confined_artifact(self.root, ref)
            report = _decode(raw)
            completed = [op for op in self.operations if op["kind"] == "validation" and op["key"] == key
                         and op["receipt"]["status"] == "completed"]
            if any(op["receipt"]["observed_encoder_forwards"] != report.get("observed_encoder_forwards")
                   or op["receipt"]["confirmed_batches"] != report.get("confirmed_batches") for op in completed):
                raise ValueError("group_workflow.evaluation_receipt_missing_or_inconsistent")
            if not any(op["kind"] == "validation" and op["key"] == key and op["purpose"] == "primary"
                       and op["receipt"]["status"] == "completed"
                       and op["receipt"]["observed_encoder_forwards"] == report.get("observed_encoder_forwards")
                       and op["receipt"]["confirmed_batches"] == report.get("confirmed_batches") for op in self.operations):
                raise ValueError("group_workflow.evaluation_receipt_missing_or_inconsistent")
        for key in self.checkpoints:
            arm = self._load_arm(key)
            if key in required_stages and arm.completed_updates != arm.config.updates:
                raise ValueError("group_workflow.completed_stage_checkpoint_required")
            del arm

    def import_resume(self, path, expected_sha256, expected_size_bytes):
        raw = _read_bound(path, expected_sha256, expected_size_bytes, MAX_JSON_BYTES)
        payload = _decode(raw)
        expected_keys = {"schema", "binding", "binding_fingerprint", "phase", "checkpoints", "evaluations",
            "operations", "attempt_ids", "attempt_id", "terminal_attempt", "reason", "parent_manifest",
            "post_manifest_compute", "payload_fingerprint"}
        if not isinstance(payload, dict) or set(payload) != expected_keys or payload["schema"] != RESUME_SCHEMA:
            raise ValueError("group_workflow.resume_schema_invalid")
        if payload["payload_fingerprint"] != tree_fingerprint({k: v for k, v in payload.items() if k != "payload_fingerprint"}):
            raise ValueError("group_workflow.resume_payload_mismatch")
        if payload["binding_fingerprint"] != self.binding_fingerprint or tree_fingerprint(payload["binding"]) != self.binding_fingerprint:
            raise ValueError("group_workflow.resume_binding_mismatch")
        ids = payload["attempt_ids"]
        if (not isinstance(ids, list) or not 1 <= len(ids) < 32 or len(ids) != len(set(ids))
                or any(not isinstance(item, str) or not item or len(item) > 128 for item in ids)
                or payload["attempt_id"] != ids[-1] or self.attempt_id in ids
                or type(payload["terminal_attempt"]) is not bool
                or payload["reason"] not in {"checkpoint", "stage_boundary", "evaluation_boundary", "failure", "completed"}
                or payload["terminal_attempt"] != (payload["reason"] in {"failure", "completed"})
                or payload["post_manifest_compute"] != ("attempt_closed" if payload["terminal_attempt"] else "not_observed_by_this_immutable_manifest")):
            raise ValueError("group_workflow.resume_attempt_lineage_invalid")
        parent = payload["parent_manifest"]
        if parent is not None:
            if (not isinstance(parent, dict) or set(parent) != {"sha256", "size_bytes", "attempt_id", "terminal_attempt", "post_manifest_compute"}
                    or parent["attempt_id"] not in ids[:-1]
                    or type(parent["size_bytes"]) is not int or not 0 < parent["size_bytes"] <= MAX_JSON_BYTES
                    or type(parent["terminal_attempt"]) is not bool
                    or parent["post_manifest_compute"] != ("attempt_closed" if parent["terminal_attempt"] else "not_observed_by_this_immutable_manifest")):
                raise ValueError("group_workflow.parent_manifest_reference_invalid")
            _sha(parent["sha256"])
        self._validate_operations(payload["operations"], ids)
        if not isinstance(payload["checkpoints"], dict) or not isinstance(payload["evaluations"], dict):
            raise ValueError("group_workflow.resume_artifacts_invalid")
        stage_keys = {_key(p, s) for p in POLICIES for s in self.configs} | {"baseline"}
        if not set(payload["checkpoints"]) <= stage_keys or not set(payload["evaluations"]) <= stage_keys:
            raise ValueError("group_workflow.resume_artifact_keys_invalid")
        # All referenced bytes are read and verified before any is interpreted.
        copies = {}
        for kind in ("checkpoints", "evaluations"):
            for ref in payload[kind].values():
                _, content = _confined_artifact(Path(path).parent, ref, checkpoint=kind == "checkpoints")
                if ref["path"] in copies and copies[ref["path"]] != content:
                    raise ValueError("group_workflow.conflicting_artifact_references")
                copies[ref["path"]] = content
        for relative, content in copies.items():
            _publish_bytes(self.root / relative, content)
        self.phase = payload["phase"]
        self.checkpoints, self.evaluations, self.operations = payload["checkpoints"], payload["evaluations"], payload["operations"]
        self.attempt_ids = ids + [self.attempt_id]
        self.parent_manifest = {"sha256": expected_sha256, "size_bytes": expected_size_bytes,
            "attempt_id": payload["attempt_id"], "terminal_attempt": payload["terminal_attempt"],
            "post_manifest_compute": payload["post_manifest_compute"]}
        self._validate_phase()
        for key, ref in self.evaluations.items():
            _, content = _confined_artifact(self.root, ref)
            expected = _decode(content)
            arm = self._load_arm(key)
            actual = self._evaluate(key, arm, purpose="resume_verification")
            if actual != expected:
                raise ValueError("group_workflow.resume_evaluation_replay_mismatch")
            del arm
        self._manifest("stage_boundary")

    def _train(self, policy, stage):
        key = _key(policy, stage)
        if key in self.checkpoints:
            arm = self._load_arm(key)
        elif stage == "pilot":
            arm = self._pilot(policy)
            if policy == POLICIES[0]:
                self.checkpoints[key] = deepcopy(self.checkpoints["baseline"])
            else:
                self._save(key, arm)
            self._manifest("stage_boundary")
        else:
            parent_key = _key(policy, "pilot")
            endpoint = self._load_arm(parent_key)
            arm = make_theory_recovery_pilot(endpoint, self.configs[stage],
                schedule=self.schedules[stage], parent_checkpoint=self.checkpoints[parent_key])
            del endpoint
            self._save(key, arm)
            self._manifest("stage_boundary")
        while arm.completed_updates < arm.config.updates:
            update = arm.completed_updates + 1
            op = self._operation("training", key, update=update)
            try:
                row = arm.step_from_factory(lambda: self.data.load_update(stage, update - 1),
                                             on_attempt=lambda receipt: self._receipt(op, receipt))
            except BaseException:
                if arm.last_attempt is not None:
                    op["receipt"] = deepcopy(arm.last_attempt)
                # A journal callback may fail after the model update was fully
                # checked. Keep that confirmed row, without saving poisoned state.
                if arm.completed_updates == update:
                    op["row"] = deepcopy(arm.rows[-1])
                    self._append("execution.jsonl", {"key": key, "attempt_id": self.attempt_id,
                        "operation_id": op["operation_id"], "replay": op["replay"], "row": op["row"]})
                raise
            op["row"] = row
            self._append("execution.jsonl", {"key": key, "attempt_id": self.attempt_id,
                "operation_id": op["operation_id"], "replay": op["replay"], "row": row})
            prior = [item["row"] for item in self.operations[:-1] if item["kind"] == "training"
                     and item["key"] == key and item["update"] == update and item["row"] is not None]
            if prior and any(item != row for item in prior):
                raise ValueError("group_workflow.replayed_row_mismatch")
            if update % self.interval == 0 or update == arm.config.updates:
                self._save(key, arm)
                self._manifest("checkpoint")
        return arm

    def execute(self):
        if not self.checkpoints:
            self._save("baseline", self.base)
            self._manifest("stage_boundary")
        while self.phase < len(PHASES):
            kind, policy, stage = PHASES[self.phase]
            if kind == "training":
                arm = self._train(policy, stage)
                self.phase += 1
                self._manifest("stage_boundary")
            else:
                key = _key(policy, stage)
                arm = self._load_arm(key)
                self._evaluate(key, arm)
                self.phase += 1
                self._manifest("evaluation_boundary")
            del arm
        reports = {}
        for key, ref in self.evaluations.items():
            _, content = _confined_artifact(self.root, ref)
            reports[key] = _decode(content)
        comparisons = {policy: compare_theory_endpoints(reports["baseline"],
            reports[_key(policy, "pilot")], reports[_key(policy, "recovery")]) for policy in POLICIES}
        control = {stage: _metric_control(reports[_key(POLICIES[0], stage)], reports[_key(POLICIES[1], stage)])
                   for stage in ("pilot", "recovery")}
        result = {"schema": WORKFLOW_SCHEMA, "status": "completed", "binding": self.binding,
            "binding_fingerprint": self.binding_fingerprint, "attempt_id": self.attempt_id,
            "attempt_ids": self.attempt_ids, "source": self.source_binding,
            "checkpoints": self.checkpoints, "evaluations": self.evaluations,
            "forgetting_recovery": comparisons, "beat_minus_random_control": control,
            "compute": _accounting(self.operations),
            "current_attempt_compute": _accounting([op for op in self.operations if op["attempt_id"] == self.attempt_id]),
            "logical_updates": {policy: {stage: config.updates for stage, config in self.configs.items()} for policy in POLICIES},
            "selection": "fixed_endpoints_no_selection", "test_access": "closed",
            "scientific_superiority_claim": False, "external_resource_guard_required": True,
            "parent_manifest": self.parent_manifest,
            "elapsed_wall_seconds": time.monotonic() - self.started,
            "data_access_observations": deepcopy(getattr(self.data, "access_observations", None)),
            "resume_accounting_scope": {
                "known_parent_operations": sum(op["attempt_id"] != self.attempt_id for op in self.operations),
                "parent_post_manifest_work_unknown": bool(self.parent_manifest and not self.parent_manifest["terminal_attempt"]),
                "replay_definition": "same_stage_update_previously_confirmed_in_imported_history",
                "unsealed_parent_journals": "retain_separately_under_external_resource_guard"}}
        result["resume_manifest"] = self._manifest("completed", terminal=True)
        _publish_json(self.root, "result.json", result)
        return result


def run_scientific_workflow(*, warmstart_factory: Callable, source_binding: dict, data,
    pilot_config: GroupPilotConfig, recovery_config: GroupPilotConfig, output_root,
    contract_binding: dict, device="cpu", checkpoint_interval=50, fixture_mode=False,
    resume_manifest=None, expected_resume_sha256=None, expected_resume_size_bytes=None,
    attempt_id=None, event_sink: Callable[[dict], None] | None = None):
    """Execute one admitted bounded attempt; errors retain receipts and re-raise.

    The CLI owns experiment-record admission and the external resource wrapper.
    CPU fixtures may explicitly use smaller budgets; the scientific path is
    fixed to 200 mixed updates and 100 theory-only recovery updates per arm.
    """
    if (resume_manifest is None) != (expected_resume_sha256 is None and expected_resume_size_bytes is None):
        raise ValueError("group_workflow.complete_resume_reference_required")
    workflow = _Workflow.__new__(_Workflow)
    try:
        workflow.__init__(warmstart_factory=warmstart_factory, source_binding=source_binding, data=data,
            pilot_config=pilot_config, recovery_config=recovery_config, output_root=output_root,
            contract_binding=contract_binding, device=device, checkpoint_interval=checkpoint_interval,
            fixture_mode=fixture_mode, attempt_id=attempt_id, event_sink=event_sink)
        if resume_manifest is not None:
            workflow.import_resume(resume_manifest, expected_resume_sha256, expected_resume_size_bytes)
        return workflow.execute()
    except BaseException as exc:
        # Failure evidence never masquerades as a result or a rollback. The
        # journal exists before model-only warmstart starts. Admission failures
        # must not write into an existing/foreign output directory.
        if getattr(workflow, "owned_output_created", False):
            try:
                resume_ref = workflow._manifest("failure", terminal=True) if "baseline" in workflow.checkpoints else None
                _publish_json(workflow.root, "failure.json", {"schema": WORKFLOW_SCHEMA, "status": "failed",
                    "error_type": type(exc).__name__, "reason": str(exc), "phase": workflow.phase,
                    "bootstrap_stage": workflow.bootstrap_stage,
                    "source_validation": "pending_or_failed" if not hasattr(workflow, "binding_fingerprint") else "verified_at_initial_binding",
                    "attempt_id": workflow.attempt_id, "binding_fingerprint": getattr(workflow, "binding_fingerprint", None),
                    "compute": _accounting(workflow.operations),
                    "current_attempt_compute": _accounting([op for op in workflow.operations if op["attempt_id"] == workflow.attempt_id]),
                    "last_operation": workflow.operations[-1] if workflow.operations else None,
                    "resume_manifest": resume_ref, "successful_result_published": False, "rollback_claim": False,
                    "elapsed_wall_seconds": time.monotonic() - workflow.started,
                    "data_access_observations": deepcopy(getattr(workflow.data, "access_observations", None))})
            except BaseException:
                pass  # Keep the original exception if storage itself failed.
        raise
