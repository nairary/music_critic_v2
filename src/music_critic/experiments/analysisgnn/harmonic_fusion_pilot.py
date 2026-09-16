"""Bounded reference-path paired pilot admission and immutable attempt execution.

No corpus discovery, cache population, TEST dispatch, optimization or resource
reservation is performed here. Real inputs and completed gate records must be
provided and byte-bound by the coordinator before the public CLI can train.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import time

from .contracts import fingerprint

REQUEST_SCHEMA = "HarmonicFusionPilotRequest@1.0.0"
JOURNAL_SCHEMA = "HarmonicFusionPilotJournal@1.0.0"
ARMS = ("control", "fusion")
PIN_KEYS = {"path", "sha256", "size_bytes"}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _fail(reason):
    raise ValueError("fusion_pilot." + reason)


def _keys(value, expected, reason):
    if type(value) is not dict or set(value) != set(expected):
        _fail(reason)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key")
        result[key] = value
    return result


def decode_json(content):
    def reject(_):
        _fail("nonfinite_json")
    return json.loads(content, object_pairs_hook=_object, parse_constant=reject)


def checked_bytes(pin):
    _keys(pin, PIN_KEYS, "artifact_pin_required")
    if (type(pin["path"]) is not str or not Path(pin["path"]).is_absolute()
            or type(pin["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
            or type(pin["size_bytes"]) is not int or pin["size_bytes"] <= 0):
        _fail("artifact_pin_invalid")
    path = Path(pin["path"])
    if path.is_symlink() or not path.is_file():
        _fail("artifact_regular_file_required")
    data = path.read_bytes()
    if len(data) != pin["size_bytes"] or sha256(data).hexdigest() != pin["sha256"]:
        _fail("artifact_hash_mismatch")
    return data


def checked_json(pin):
    return decode_json(checked_bytes(pin))


def artifact_pin(path):
    path = Path(path).absolute()
    content = path.read_bytes()
    return {"path": str(path), "sha256": sha256(content).hexdigest(), "size_bytes": len(content)}


def _write_json(path, payload):
    from .harmonic_fusion_run import _write_json_once
    _write_json_once(path, payload)


def runtime_observations(device):
    """Capture the policy that must match gates and every continuation attempt."""
    import torch
    from .corrected_training import environment_fingerprint
    result = environment_fingerprint()
    result.pop("fingerprint")
    # TorchVersion is a str subclass; preserve its text/canonical fingerprint
    # while presenting the plain JSON string required by runtime admission.
    if isinstance(result.get("torch"), str):
        result["torch"] = str(result["torch"])
    result.update(device=device, torch_num_threads=torch.get_num_threads(),
                  torch_num_interop_threads=torch.get_num_interop_threads(),
                  native_threads={key: os.environ.get(key) for key in THREAD_ENV},
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  cudnn_deterministic=torch.backends.cudnn.deterministic,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    return result


def verify_source_manifest(pin, repository_root):
    manifest = checked_json(pin)
    if type(manifest) is not dict or type(manifest.get("files")) is not list:
        _fail("source_manifest_invalid")
    root = Path(repository_root).resolve()
    seen = set()
    for row in manifest["files"]:
        _keys(row, PIN_KEYS, "source_row_invalid")
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or row["path"] in seen:
            _fail("source_path_invalid")
        seen.add(row["path"])
        checked_bytes({**row, "path": str(root / rel)})
    required = {str(p.relative_to(root)) for p in (root / "src").rglob("*.py")}
    required.update(str(p.relative_to(root)) for p in (root / "scripts").rglob("*.py"))
    required.update({"pyproject.toml", "scripts/run_harmonic_fusion_real_pilot.py",
                     "configs/experiments/harmonic_fusion_pilot.json"})
    if not required <= seen:
        _fail("source_manifest_incomplete")
    return manifest


def _record(pin, *, terminal=False):
    from music_critic.experiments.registry import validate_record, canonical_json_bytes
    content = checked_bytes(pin)
    value = decode_json(content)
    validate_record(value, filename=Path(pin["path"]).name)
    if content != canonical_json_bytes(value) + b"\n":
        _fail("canonical_record_required")
    if terminal:
        if value["status"] not in {"completed", "failed", "aborted", "invalid"}:
            _fail("parent_record_not_terminal")
    elif value["status"] != "planned":
        _fail("planned_record_required")
    return value


def validate_runtime_policy(runtime):
    _keys(runtime, {"python", "platform", "torch", "numpy", "packages", "torch_cuda_build",
        "cudnn_version", "cuda_available", "cuda_device_count", "gpu", "cublas_workspace_config",
        "device", "torch_num_threads", "torch_num_interop_threads", "native_threads",
        "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic", "cudnn_allow_tf32",
        "matmul_allow_tf32"}, "runtime_policy_keys_invalid")
    if (runtime["device"] != "cuda" or runtime["cuda_available"] is not True
            or type(runtime["cuda_device_count"]) is not int or runtime["cuda_device_count"] != 1
            or runtime["native_threads"] != {key: "1" for key in THREAD_ENV}
            or type(runtime["torch_num_threads"]) is not int or runtime["torch_num_threads"] != 1
            or type(runtime["torch_num_interop_threads"]) is not int or runtime["torch_num_interop_threads"] != 1
            or runtime["deterministic_algorithms"] is not True
            or runtime["cudnn_benchmark"] is not False or runtime["cudnn_deterministic"] is not True
            or any(type(runtime[key]) is not bool for key in ("cudnn_allow_tf32", "matmul_allow_tf32"))
            or runtime["cublas_workspace_config"] not in {":4096:8", ":16:8"}):
        _fail("runtime_policy_invalid")
    if (any(type(runtime[key]) is not str or not runtime[key] for key in (
            "python", "platform", "torch", "numpy", "torch_cuda_build", "gpu"))
            or type(runtime["packages"]) is not dict
            or set(runtime["packages"]) != {"numpy", "torch", "torch-geometric"}
            or any(type(value) is not str or not value for value in runtime["packages"].values())
            or type(runtime["cudnn_version"]) is not int or runtime["cudnn_version"] <= 0):
        _fail("runtime_identity_required")
    fingerprint(runtime)


def verify_gate(gate, *, role, bindings, runtime=None, repository_root=None):
    fields = {"record", "report"}
    if role == "cuda_synthetic" and type(gate) is dict and "historical_source_manifest" in gate:
        fields.add("historical_source_manifest")
    _keys(gate, fields, "gate_artifacts_required")
    record = _record(gate["record"], terminal=True)
    report = checked_json(gate["report"])
    if (record["status"] != "completed" or record["validity"].get("valid") is not True
            or record["kind"] != "engineering_gate" or report.get("passed") is not True
            or report.get("device") != "cuda"):
        _fail("completed_cuda_gate_required")
    selected = ("source_manifest_sha256", "runtime_fingerprint") if role.startswith("cuda_") else (
        "source_manifest_sha256", "runtime_fingerprint", "data_manifest_sha256",
        "source_checkpoint_sha256", "source_model_state_fingerprint", "reference_path")
    expected = {"role": role, **{key: bindings[key] for key in selected}}
    if record.get("protocol", {}).get("fusion_pilot_bindings") != expected:
        if (role != "cuda_synthetic" or "historical_source_manifest" not in gate
                or repository_root is None or record.get("protocol", {}).get("fusion_pilot_bindings") is not None):
            _fail("gate_binding_mismatch")
        from .harmonic_fusion_gate_reuse import verify_historical_gate_source
        verify_historical_gate_source(record=record, report=report,
            manifest_pin=gate["historical_source_manifest"], repository_root=repository_root,
            current_source_manifest_sha256=bindings["source_manifest_sha256"])
    if not any(row.get("sha256") == gate["report"]["sha256"]
               and row.get("size_bytes") == gate["report"]["size_bytes"] for row in record["artifacts"]):
        _fail("gate_report_not_registered")
    for key in ("test_inputs_accessed", "test_targets_accessed", "test_metrics_computed"):
        if report.get(key) is not False or record["data_access"].get(key) is not False:
            _fail("gate_test_policy_mismatch")
    if role.startswith("cuda_"):
        if report.get("real_data_accessed") is not False:
            _fail("synthetic_gate_scope_mismatch")
        if role == "cuda_fresh_process" and report.get("fresh_process_resume") is not True:
            _fail("fresh_process_gate_required")
    elif (report.get("real_data_accessed") is not True
          or report.get("reference_smoke_passed") is not True
          or report.get("source_hashes_verified") is not True
          or report.get("raw_only_prediction_verified") is not True
          or report.get("target_alignment_verified") is not True):
        _fail("real_train_reference_gate_required")
    if (type(report.get("actual_updates")) is not int or report["actual_updates"] <= 0
            or report.get("actual_draws") != report["actual_updates"] * 2):
        _fail("gate_compute_evidence_required")
    if runtime is not None:
        observed = report.get("environment", {})
        for key in ("python", "packages", "torch_cuda_build", "cublas_workspace_config"):
            if observed.get(key) != runtime[key]:
                _fail("gate_observed_runtime_mismatch")
    if role.startswith("cuda_"):
        if (report["actual_updates"] != 24 or report["actual_draws"] != 48
                or type(report.get("arms")) is not list
                or [row.get("arm") for row in report["arms"]] != list(ARMS)):
            _fail("paired_synthetic_gate_required")
        if report.get("config", {}).get("validation_mode") != "required":
            _fail("synthetic_validation_required")
    if role == "cuda_synthetic":
        if report.get("schema_version") != "HarmonicFusionSyntheticTrainerGate@1.1.0":
            _fail("synthetic_gate_schema_mismatch")
        from .harmonic_fusion_run import _require_learning
        for row in report["arms"]:
            if (row.get("actual_updates") != 12 or row.get("actual_draws") != 24
                    or row.get("committed_final_update") != 6 or row.get("sampler_position") != 12
                    or row.get("validation_updates") != [0, 3, 6]
                    or row.get("interrupted_matches_uninterrupted") is not True):
                _fail("synthetic_trajectory_incomplete")
            _require_learning(row["learning_diagnostics"], enabled=row["arm"] == "fusion")
        fused = report["arms"][1]
        export = fused.get("fusion_export")
        if (type(export) is not dict or export.get("parent_training_checkpoint") != fused.get("final_training_checkpoint")
                or not re.fullmatch(r"[0-9a-f]{64}", str(export.get("sha256")))
                or type(export.get("size_bytes")) is not int or export["size_bytes"] <= 0):
            _fail("fusion_export_evidence_required")
    elif role == "cuda_fresh_process":
        if report.get("schema_version") != "HarmonicFusionFreshProcessGate@1.0.0":
            _fail("fresh_process_schema_mismatch")
        pids = [report.get("parent_process_id"), *[row.get("worker_process_id") for row in report["arms"]]]
        if any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != 3:
            _fail("distinct_worker_processes_required")
        for row in report["arms"]:
            ready = row.get("worker_runtime_readiness", {})
            if (row.get("whole_state_equal") is not True or ready.get("device_type") != "cuda"
                    or ready.get("cuda_device_count") != 1 or ready.get("verified_cuda_ordinals") != [0]
                    or ready.get("saved_rng_applied") is not False):
                _fail("fresh_cuda_restore_not_verified")
    else:
        from .harmonic_fusion_reference import verify_reference_report
        verify_reference_report(record, report, bindings)
    return record["experiment_id"]


def stable_bindings(request):
    """Common immutable comparison identity used by reference and pilot plans."""
    return {
        "comparison_id": request["comparison_id"],
        "data_manifest_sha256": request["data_manifest"]["sha256"],
        "config_sha256": request["config"]["sha256"],
        "source_manifest_sha256": request["source_manifest"]["sha256"],
        "source_checkpoint_sha256": request["source_checkpoint"]["sha256"],
        "source_checkpoint_size_bytes": request["source_checkpoint"]["size_bytes"],
        "source_model_state_fingerprint": request["source_model_state_fingerprint"],
        "runtime_fingerprint": fingerprint(request["runtime"]),
        "reference_path": "sequential_historical_C2",
    }


def admit_request(request, *, repository_root):
    """Verify plans, source and completed gate evidence before raw/tensor reads."""
    _keys(request, {"schema_version", "comparison_id", "attempt_id", "data_manifest", "config",
                    "source_checkpoint", "source_model_state_fingerprint", "source_manifest",
                    "runtime", "gates", "experiment_records", "parent", "max_wall_seconds"},
          "request_keys_invalid")
    if request["schema_version"] != REQUEST_SCHEMA:
        _fail("request_schema_invalid")
    validate_runtime_policy(request["runtime"])
    for key in ("comparison_id", "attempt_id"):
        if type(request[key]) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", request[key]):
            _fail("request_id_invalid")
    if request["max_wall_seconds"] != 7200 or type(request["max_wall_seconds"]) is not int:
        _fail("fixed_pair_wall_budget_required")
    from .harmonic_fusion_training import FusionTrainingConfig
    config_raw = checked_json(request["config"])
    if config_raw != asdict(FusionTrainingConfig()):
        _fail("fixed_p512_config_required")
    verify_source_manifest(request["source_manifest"], repository_root)
    if (type(request["source_model_state_fingerprint"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", request["source_model_state_fingerprint"])):
        _fail("source_model_state_pin_required")
    # Shape-check source pin without deserializing or reading its tensors here.
    _keys(request["source_checkpoint"], PIN_KEYS, "source_checkpoint_pin_required")
    stable = stable_bindings(request)
    _keys(request["experiment_records"], ARMS, "paired_plans_required")
    ids, plans = {}, {}
    for arm in ARMS:
        record = _record(request["experiment_records"][arm])
        if (record.get("protocol", {}).get("fusion_pilot_bindings") != {"arm": arm, **stable}
                or record["seeds"] != [17]
                or record["budget"].get("planned") != 512
                or record["budget"].get("planned_draws") != 1024):
            _fail("planned_pair_binding_mismatch")
        policy = record["data_access"]["declared_policy"]
        if policy["validation_allowed"] is not True or any(policy[k] is not False for k in (
                "test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")):
            _fail("planned_access_policy_mismatch")
        ids[arm] = record["experiment_id"]
        plans[arm] = record
    if len(set(ids.values())) != 2:
        _fail("distinct_arm_records_required")
    required = {"cuda_synthetic", "real_train_reference"}
    if request["parent"] is not None:
        required.add("cuda_fresh_process")
    if (type(request["gates"]) is not dict or not required <= set(request["gates"])
            or not set(request["gates"]) <= {"cuda_synthetic", "cuda_fresh_process", "real_train_reference"}):
        _fail("required_gates_missing")
    for role, gate in request["gates"].items():
        verify_gate(gate, role=role, bindings=stable, runtime=request["runtime"], repository_root=repository_root)
    from .harmonic_fusion_data import FrozenFusionData
    data = FrozenFusionData(checked_json(request["data_manifest"]))
    from .full_orbit_training import FullOrbitSampler
    sampler = FullOrbitSampler(data.table, seed=17)
    draws = [asdict(sampler.peek(i)) for i in range(1024)]
    data.require_train_records({row["record_id"] for row in draws})
    bindings = {**data.bindings, **stable, "device": "cuda", "draw_schedule_fingerprint": fingerprint(draws)}
    parent = read_parent(request["parent"], bindings=bindings) if request["parent"] is not None else None
    if parent is not None:
        for arm in ARMS:
            if parent["experiment_ids"][arm] not in plans[arm].get("parent_ids", []):
                _fail("child_parent_record_link_missing")
    return {"data": data, "config": FusionTrainingConfig(), "bindings": bindings,
            "experiment_ids": ids, "parent": parent}


def read_parent(parent, *, bindings):
    """A frozen journal survives hard kill; only complete hash-chained rows count."""
    _keys(parent, {"journal", "experiment_records"}, "parent_contract_invalid")
    content = checked_bytes(parent["journal"])
    lines = content.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith(b"\n"))
    if incomplete_tail:
        lines.pop()
    state = None
    previous = None
    for index, line in enumerate(lines):
        row = decode_json(line)
        digest = row.pop("fingerprint", None)
        if (row.get("schema_version") != JOURNAL_SCHEMA or row.get("index") != index
                or row.get("previous") != previous or fingerprint(row) != digest):
            _fail("parent_journal_invalid")
        state = row["state"]
        previous = digest
    if state is None or state["bindings_fingerprint"] != fingerprint(bindings):
        _fail("parent_binding_mismatch")
    _keys(parent["experiment_records"], ARMS, "parent_records_required")
    for arm in ARMS:
        record = _record(parent["experiment_records"][arm], terminal=True)
        if record["experiment_id"] != state["experiment_ids"][arm]:
            _fail("parent_record_id_mismatch")
        if not any(row.get("sha256") == parent["journal"]["sha256"]
                   and row.get("size_bytes") == parent["journal"]["size_bytes"] for row in record["artifacts"]):
            _fail("parent_journal_not_registered")
    state = deepcopy(state)
    state["unconfirmed_parent_tail"] = (state["unconfirmed_parent_tail"] or incomplete_tail
        or state["event_status"] == "started" or any(row["update_in_flight"] or row["validation_in_flight"]
                                                     for row in state["arms"].values()))
    return state


class AttemptJournal:
    def __init__(self, root, *, attempt_id, bindings, experiment_ids, parent):
        self.path = root / "progress.jsonl"
        self.path.touch(exist_ok=False)
        self.index = 0
        self.previous = None
        self.started = time.monotonic()
        self.state = {"attempt_id": attempt_id, "bindings_fingerprint": fingerprint(bindings),
                      "experiment_ids": experiment_ids, "arm": None, "stage": "initialization",
                      "event_status": "started", "status": "running",
                      "arms": {arm: {"cursor": 0, "checkpoint": None, "validations": {},
                                     "attempted_updates": 0, "applied_updates": 0,
                                     "reader_calls": {"train": 0, "validation": 0},
                                     "validation_records_completed": 0, "status": "pending"} for arm in ARMS},
                      "parent_costs": None if parent is None else cumulative_costs(parent),
                      "unconfirmed_parent_tail": False if parent is None else parent["unconfirmed_parent_tail"]}
        for state in self.state["arms"].values():
            state["update_in_flight"] = False
            state["validation_in_flight"] = False
        if parent is not None:
            if parent["attempt_id"] == attempt_id:
                _fail("new_attempt_id_required")
            for arm in ARMS:
                self.state["arms"][arm]["checkpoint"] = deepcopy(parent["arms"][arm]["checkpoint"])
                self.state["arms"][arm]["validations"] = deepcopy(parent["arms"][arm]["validations"])
        self.append("initialization", "started")

    def append(self, stage, status, **detail):
        self.state.update(stage=stage, event_status=status)
        row = {"schema_version": JOURNAL_SCHEMA, "index": self.index, "previous": self.previous,
               "stage": stage, "status": status, "detail": detail, "state": deepcopy(self.state),
               "elapsed_wall_seconds": time.monotonic() - self.started}
        row["fingerprint"] = fingerprint(row)
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.index += 1
        self.previous = row["fingerprint"]


def cumulative_costs(state):
    prior = state.get("parent_costs") or {"attempted_updates": 0, "applied_updates": 0,
                                         "train_reader_calls": 0, "validation_reader_calls": 0}
    result = dict(prior)
    for arm in state["arms"].values():
        result["attempted_updates"] += arm["attempted_updates"]
        result["applied_updates"] += arm["applied_updates"]
        for split in ("train", "validation"):
            result[split + "_reader_calls"] += arm["reader_calls"][split]
    return result


def _execute_attempt(root, *, attempt_id, experiment_ids, config, bindings, table, class_weights,
                     reader, validation_factory, session_factory, parent=None,
                     verify_inputs=lambda: None, max_wall_seconds=7200, synthetic=False, started_at=None,
                     execution_context=None):
    """Mechanics seam tested on tiny fixtures; production CLI fixes all arguments."""
    from .harmonic_fusion_resume import load_training_checkpoint, save_training_checkpoint
    from .harmonic_fusion_checkpoint import create_model_payload
    from music_critic.experiments.pop909_cl_frozen.checkpoint import create_model_only_payload
    from .harmonic_fusion_run import _json_bytes, _write_bytes_once
    from .full_orbit_training import FullOrbitSampler
    import torch
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic() if started_at is None else started_at
    journal = AttemptJournal(root, attempt_id=attempt_id, bindings=bindings,
                             experiment_ids=experiment_ids, parent=parent)
    session = None
    current = None
    initial_cursor = 0
    summaries = {}

    def deadline():
        if time.monotonic() - started >= max_wall_seconds:
            raise TimeoutError("fusion_pilot.pair_wall_budget_exceeded")

    def synchronize_counts():
        if current is not None and session is not None:
            current["cursor"] = session.applied_updates
            current["applied_updates"] = session.applied_updates - initial_cursor

    def checkpoint(active):
        deadline()
        key = [active.applied_updates, len(active.validations), active.pending_validation]
        old = current["checkpoint"]
        if old is not None and old["key"] == key:
            return old
        journal.append("checkpoint", "started", key=key)
        path = root / f"{journal.state['arm']}-u{key[0]:06d}-v{key[1]}.pt"
        evidence = save_training_checkpoint(active, path)
        pin = {"path": str(path), **evidence}
        descriptor = {"key": key, "artifact": pin}
        current["checkpoint"] = descriptor
        if key[2] is None and active.validations and active.validations[-1]["update"] == key[0]:
            current["validations"][str(key[0])] = descriptor
        journal.append("checkpoint", "completed", checkpoint=descriptor)
        return descriptor

    def progress(event):
        deadline()
        synchronize_counts()
        stage, status = event["stage"], event["status"]
        if stage == "update" and status == "started":
            current["attempted_updates"] += 1
            current["update_in_flight"] = True
        if stage == "update" and status == "completed":
            current["update_in_flight"] = False
        if stage == "validation":
            current["validation_in_flight"] = status == "started"
        if stage == "validation_record" and status == "completed":
            current["validation_records_completed"] += 1
        # Save the pending boundary before invoking potentially failing full validation.
        if stage == "validation" and status == "started":
            checkpoint(session)
        journal.append(stage, status, event=event)
        if stage == "validation" and status == "completed":
            checkpoint(session)

    def observed_reader(record_id, *, split):
        deadline()
        if split not in ("train", "validation"):
            _fail("TEST_closed")
        current["reader_calls"][split] += 1
        journal.append("reader", "started", split=split, record_id=record_id)
        result = reader(record_id, split=split)
        journal.append("reader", "completed", split=split, record_id=record_id)
        return result

    try:
        journal.append("resolve", "started")
        _write_json(root / "resolved.json", {"config": config.to_dict(), "bindings": bindings,
                                            "attempt_id": attempt_id, "experiment_ids": experiment_ids,
                                            "synthetic": synthetic, "max_wall_seconds": max_wall_seconds,
                                            "execution_context": deepcopy(execution_context)})
        journal.append("resolve", "completed")
        journal.append("verify_inputs", "started")
        verify_inputs()
        journal.append("verify_inputs", "completed")
        for arm in ARMS:
            deadline()
            journal.state["arm"] = arm
            current = journal.state["arms"][arm]
            session = None
            initial_cursor = 0
            journal.append("initialize_arm", "started")
            if current["checkpoint"] is None:
                session = session_factory(arm == "fusion")
            else:
                pin = current["checkpoint"]["artifact"]
                session = load_training_checkpoint(pin["path"], expected_sha256=pin["sha256"],
                    expected_size_bytes=pin["size_bytes"], config=config, bindings=bindings, table=table,
                    device=bindings.get("device", "cpu"), expected_enabled=arm == "fusion")
            initial_cursor = session.applied_updates
            if (current["checkpoint"] is not None and current["checkpoint"]["key"] !=
                    [session.applied_updates, len(session.validations), session.pending_validation]):
                _fail("resume_descriptor_mismatch")
            synchronize_counts()
            current["status"] = "running"
            journal.append("initialize_arm", "completed", restored_cursor=initial_cursor)
            validate = validation_factory(observed_reader, progress)
            session.run_until(config.total_updates, observed_reader, class_weights,
                              validate=validate, checkpoint=checkpoint, progress=progress)
            synchronize_counts()
            checkpoint(session)
            expected = FullOrbitSampler(table, seed=config.seed)
            draws = [asdict(expected.peek(i)) for i in range(config.total_updates * config.batch_size)]
            actual = [draw for row in session.history for draw in row["draws"]]
            if actual != draws or session.pending_validation is not None:
                _fail("final_schedule_mismatch")
            if session.best_update is None:
                _fail("best_validation_unavailable")
            best = current["validations"].get(str(session.best_update))
            if best is None:
                _fail("selected_checkpoint_missing")
            result = {"arm": arm, "source_model_state": session.source_model_state,
                      "applied_updates": session.applied_updates, "draws": len(actual),
                      "draw_schedule_fingerprint": fingerprint(actual),
                      "validations": deepcopy(session.validations), "best_update": session.best_update,
                      "best_checkpoint": deepcopy(best), "final_checkpoint": deepcopy(current["checkpoint"])}
            tensors = [] if arm == "control" else [p.detach().cpu() for p in session.model.fusion.parameters()]
            result["learning_observations"] = {
                "base_state_changed": model_state_fingerprint_for_base(session.model.base) != session.source_model_state,
                "fusion_nonzero_scalars": sum(int(p.ne(0).sum()) for p in tensors),
                "fusion_l2_from_zero_initialization": math.sqrt(sum(float(p.double().square().sum()) for p in tensors)),
                "fusion_optimizer_steps": {name: value for name,value in session.optimizer_steps.items() if name.startswith("fusion.")},
                "quality_improvement_required": False}
            buffer = BytesIO()
            export = create_model_payload(session.model) if arm == "fusion" else create_model_only_payload(session.model.base)
            torch.save(export, buffer)
            path = root / f"{arm}-final-model.pt"
            _write_bytes_once(path, buffer.getvalue())
            result["final_model"] = artifact_pin(path)
            # Drop the live GPU model before reconstructing the selected model.
            session = None
            selected_pin = best["artifact"]
            selected = load_training_checkpoint(selected_pin["path"], expected_sha256=selected_pin["sha256"],
                expected_size_bytes=selected_pin["size_bytes"], config=config, bindings=bindings,
                table=table, device=bindings.get("device", "cpu"), expected_enabled=arm == "fusion")
            if (selected.applied_updates != result["best_update"] or selected.pending_validation is not None
                    or not selected.validations or selected.validations[-1]["update"] != result["best_update"]):
                _fail("best_descriptor_mismatch")
            buffer = BytesIO()
            export = create_model_payload(selected.model) if arm == "fusion" else create_model_only_payload(selected.model.base)
            torch.save(export, buffer)
            path = root / f"{arm}-best-model.pt"
            _write_bytes_once(path, buffer.getvalue())
            result["best_model"] = artifact_pin(path)
            del selected, export
            summaries[arm] = result
            _write_json(root / f"{arm}-summary.json", result)
            current["status"] = "completed"
            journal.append("arm", "completed")
        if summaries["control"]["source_model_state"] != summaries["fusion"]["source_model_state"]:
            _fail("paired_source_mismatch")
        if summaries["control"]["draw_schedule_fingerprint"] != summaries["fusion"]["draw_schedule_fingerprint"]:
            _fail("paired_draws_mismatch")
        deadline()
        def scores(summary):
            values = {row["update"]: row["metrics"]["corrected_primary_macro_score"] for row in summary["validations"]}
            return {"final": values[config.total_updates], "best": values[summary["best_update"]]}
        control, fusion = scores(summaries["control"]), scores(summaries["fusion"])
        comparison = {"scores": {"control": control, "fusion": fusion},
                      "fusion_minus_control": {key: None if control[key] is None or fusion[key] is None
                          else fusion[key] - control[key] for key in control},
                      "one_seed_descriptive_only": True, "automatic_baseline_promotion": False}
        _write_json(root / "comparison.json", comparison)
        journal.state["status"] = "completed"
        journal.append("attempt", "completed")
        report = {"schema_version": "HarmonicFusionPilotAttempt@1.0.0", "passed": True,
                  "status": "completed", "state": deepcopy(journal.state), "arms": summaries,
                  "cumulative_costs": cumulative_costs(journal.state), "wall_seconds": time.monotonic() - started,
                  "attempted_draw_slots": 2 * cumulative_costs(journal.state)["attempted_updates"],
                  "confirmed_applied_draw_slots": 2 * cumulative_costs(journal.state)["applied_updates"],
                  "synthetic": synthetic, "real_data_accessed": not synthetic,
                  "device": bindings.get("device", "cpu"),
                  "execution_context_fingerprint": fingerprint(execution_context),
                  "process_resources": process_resources(bindings.get("device", "cpu")),
                  "test_inputs_accessed": False, "test_targets_accessed": False,
                  "test_metrics_computed": False, "scientific_acceptance": False,
                  "registry_finalization": "coordinator_must_finalize_before_decision_or_next_run"}
        report["fingerprint"] = fingerprint(report)
        content = _json_bytes(report)
        manifest = {p.name: artifact_pin(p) for p in sorted(root.iterdir()) if p.is_file()}
        manifest["report.json"] = {"path": str(root / "report.json"), "sha256": sha256(content).hexdigest(),
                                   "size_bytes": len(content)}
        _write_json(root / "artifact-manifest.json", manifest)
        _write_bytes_once(root / "report.json", content)
        return report
    except BaseException as exc:
        synchronize_counts()
        journal.state["status"] = "failed"
        failure = {"exception_type": type(exc).__name__, "message": str(exc),
                   "stage": journal.state["stage"], "arm": journal.state["arm"]}
        try:
            journal.append("attempt", "failed", failure=failure)
            receipt = {"schema_version": "HarmonicFusionPilotAttempt@1.0.0", "passed": False,
                       "status": "failed", "failure": failure, "state": deepcopy(journal.state),
                       "cumulative_costs": cumulative_costs(journal.state), "wall_seconds": time.monotonic() - started,
                       "attempted_draw_slots": 2 * cumulative_costs(journal.state)["attempted_updates"],
                       "confirmed_applied_draw_slots": 2 * cumulative_costs(journal.state)["applied_updates"],
                       "synthetic": synthetic, "scientific_acceptance": False}
            receipt.update(device=bindings.get("device", "cpu"),
                           execution_context_fingerprint=fingerprint(execution_context),
                           process_resources=process_resources(bindings.get("device", "cpu")))
            receipt["fingerprint"] = fingerprint(receipt)
            _write_json(root / "failure-receipt.json", receipt)
        except BaseException as receipt_error:
            exc.add_note("Failure receipt could not be completed: " + repr(receipt_error))
            raise exc from receipt_error
        raise


def model_state_fingerprint_for_base(model):
    from .corrected_model import model_state_fingerprint
    return model_state_fingerprint(model)


def process_resources(device):
    import resource
    import sys
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "scope": "current_process_high_water_not_whole_host_or_hard_limit"}


def prepare_pilot_runtime(runtime):
    """Explicit coordinator setup; no model construction or training."""
    validate_runtime_policy(runtime)
    if runtime.get("device") != "cuda" or runtime.get("cuda_device_count") != 1:
        _fail("one_cuda_device_required")
    import torch
    from .harmonic_fusion_resume import prepare_training_resume_runtime
    if any(os.environ.get(name) != "1" for name in THREAD_ENV):
        _fail("native_thread_limit_required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if runtime.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
        _fail("cublas_policy_required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != runtime["cublas_workspace_config"]:
        _fail("cublas_policy_mismatch")
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"])
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = runtime["matmul_allow_tf32"]
    prepare_training_resume_runtime("cuda", expected_cuda_device_count=1)
    observed_runtime = runtime_observations("cuda")
    if observed_runtime != runtime:
        _fail("runtime_policy_mismatch")
    return observed_runtime


def run_request(request, output_root, *, repository_root):
    """Production dispatch: exactly512/arm on a source/gate/runtime-bound CUDA path."""
    started = time.monotonic()
    admitted = admit_request(request, repository_root=repository_root)
    observed_runtime = prepare_pilot_runtime(request["runtime"])
    from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
    from .corrected_model import model_state_fingerprint
    from .harmonic_fusion_training import FusionTrainingSession
    from .harmonic_fusion_pilot_evaluation import make_full_validation_callback
    data = admitted["data"]
    bindings = admitted["bindings"]
    base = None

    def factory(enabled):
        nonlocal base
        if base is None:
            pin = request["source_checkpoint"]
            loaded = load_model_checkpoint(pin["path"], expected_sha256=pin["sha256"], expected_size_bytes=pin["size_bytes"])
            base = loaded.model
            if model_state_fingerprint(base) != bindings["source_model_state_fingerprint"]:
                _fail("source_state_mismatch")
        return FusionTrainingSession(base=base, enabled=enabled, config=admitted["config"],
                                     bindings=bindings, table=data.table, device="cuda")

    def validation_factory(reader, progress):
        return make_full_validation_callback(assignments=data.assignments, reader=reader,
            class_weights=data.class_weights, train_seen_tuples=data.train_seen_tuples,
            native_train_tuple_source_sha256=data.bindings["native_train_joint_tuple_sha256"],
            binding_fingerprint=fingerprint(bindings), progress=progress)

    def verify_inputs():
        data.verify_bindings()
        checked_bytes(request["source_checkpoint"])

    return _execute_attempt(output_root, attempt_id=request["attempt_id"],
        experiment_ids=admitted["experiment_ids"], config=admitted["config"], bindings=bindings,
        table=data.table, class_weights=data.class_weights, reader=data.reader,
        validation_factory=validation_factory, session_factory=factory,
        parent=admitted["parent"], verify_inputs=verify_inputs,
        max_wall_seconds=request["max_wall_seconds"], started_at=started,
        execution_context={"request": deepcopy(request), "observed_runtime": observed_runtime})
