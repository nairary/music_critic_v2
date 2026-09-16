"""Registered one-update-per-arm TRAIN reference check, without evaluation.

This non-resumable engineering protocol retains the exact P512 optimizer,
schedule and sampler. It deliberately does not enter the scientific trainer's
VALIDATION lifecycle and never exports a checkpoint or a musical score.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import time

from .contracts import fingerprint
from .harmonic_fusion_training import FusionTrainingConfig, FusionTrainingSession
from . import harmonic_fusion_pilot as pilot


REQUEST_SCHEMA = "HarmonicFusionReferenceRequest@1.0.0"
REPORT_SCHEMA = "HarmonicFusionReferenceTrainSmoke@1.0.0"
JOURNAL_SCHEMA = "HarmonicFusionReferenceJournal@1.0.0"
VALIDATION_POLICY = "not_part_of_train_reference_smoke"
_GATE_BINDINGS = ("source_manifest_sha256", "runtime_fingerprint", "data_manifest_sha256",
                  "source_checkpoint_sha256", "source_model_state_fingerprint", "reference_path")
_ACCESS = {"validation_inputs_accessed": False, "validation_targets_accessed": False,
           "validation_metrics_computed": False, "test_inputs_accessed": False,
           "test_targets_accessed": False, "test_metrics_computed": False}


def _fail(reason):
    raise ValueError("fusion_reference." + reason)


def reference_protocol(bindings):
    """Exact planned-record contract, separate from the P512 scientific run."""
    return {"schema_version": "HarmonicFusionReferenceProtocol@1.0.0",
            "bindings": deepcopy(bindings), "updates_per_arm": 1,
            "actual_budget_updates": 2, "actual_budget_draws": 4, "draw_positions": [0, 1],
            "validation_policy": VALIDATION_POLICY, "resumable": False}


def reference_gate_bindings(bindings):
    return {"role": "real_train_reference", **{key: bindings[key] for key in _GATE_BINDINGS}}


def verify_reference_report(record, report, bindings):
    """Accept only this executable's complete two-arm TRAIN reference contract."""
    if (record.get("protocol", {}).get("fusion_pilot_bindings") != reference_gate_bindings(bindings)
            or fingerprint(record.get("protocol", {}).get("reference_train_smoke")) != fingerprint(reference_protocol(bindings))
            or record.get("seeds") != [17]
            or type(record.get("budget", {}).get("planned")) is not int or record["budget"]["planned"] != 2
            or type(record["budget"].get("planned_draws")) is not int or record["budget"]["planned_draws"] != 4):
        _fail("reference_report_protocol_mismatch")
    policy = record.get("data_access", {}).get("declared_policy", {})
    if any(policy.get(key) is not False for key in ("validation_allowed", "test_inputs_allowed",
            "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")):
        _fail("reference_report_access_policy_mismatch")
    if (report.get("schema_version") != REPORT_SCHEMA or report.get("device") != "cuda"
            or report.get("experiment_id") != record.get("experiment_id")
            or any(report.get(key) is not True for key in ("passed", "reference_smoke_passed", "real_data_accessed",
                "source_hashes_verified", "raw_only_prediction_verified", "target_alignment_verified"))
            or any(report.get(key) is not False for key in (*_ACCESS, "synthetic", "resumable",
                "scientific_acceptance", "unconfirmed_in_flight_tail"))
            or report.get("reference_validation_policy") != VALIDATION_POLICY
            or fingerprint(report.get("config")) != fingerprint(FusionTrainingConfig().to_dict())):
        _fail("reference_report_contract_mismatch")
    for key, expected in {"actual_updates": 2, "actual_draws": 4, "attempted_updates": 2,
            "attempted_draw_slots": 4, "reader_calls": 4, "reader_completed": 4, "validation_reader_calls": 0}.items():
        if type(report.get(key)) is not int or report[key] != expected:
            _fail("reference_report_counts_mismatch")
    observed_bindings = report.get("bindings", {})
    if (type(observed_bindings) is not dict or any(observed_bindings.get(key) != value for key, value in bindings.items())
            or fingerprint(report.get("environment")) != bindings["runtime_fingerprint"]):
        _fail("reference_report_binding_mismatch")
    arms = report.get("arms")
    states = report.get("state", {}).get("arms", {})
    if (type(arms) is not list or len(arms) != 2 or any(type(row) is not dict for row in arms)
            or [row.get("arm") for row in arms] != list(pilot.ARMS)
            or type(states) is not dict or set(states) != set(pilot.ARMS)):
        _fail("reference_report_paired_arms_required")
    shared = None
    for row in arms:
        state = states[row["arm"]]
        if (type(state) is not dict or row.get("source_model_state") != bindings["source_model_state_fingerprint"]
                or row.get("validation_updates") != [] or state.get("update_in_flight") is not False):
            _fail("reference_report_arm_mismatch")
        for key, expected in {"actual_updates": 1, "actual_draws": 2, "sampler_position": 2,
                              "raw_prediction_calls": 1, "target_alignment_checks": 1}.items():
            if type(row.get(key)) is not int or row[key] != expected:
                _fail("reference_report_arm_counts_mismatch")
        for key, expected in {"attempted_updates": 1, "applied_updates": 1, "reader_calls": 2,
                "reader_completed": 2, "raw_prediction_calls": 1, "target_alignment_checks": 1}.items():
            if type(state.get(key)) is not int or state[key] != expected:
                _fail("reference_report_journal_counts_mismatch")
        draws = row.get("draws")
        if type(draws) is not list or len(draws) != 2:
            _fail("reference_report_draws_mismatch")
        for index, draw in enumerate(draws):
            if (type(draw) is not dict or set(draw) != {"orbit_epoch", "orbit_index", "record_id", "component_id", "shift_pc"}
                    or type(draw["orbit_epoch"]) is not int or draw["orbit_epoch"] != 0
                    or type(draw["orbit_index"]) is not int or draw["orbit_index"] != index
                    or type(draw["shift_pc"]) is not int or not 0 <= draw["shift_pc"] < 12
                    or any(type(draw[key]) is not str or not draw[key] for key in ("record_id", "component_id"))):
                _fail("reference_report_draws_mismatch")
        digest = fingerprint(draws)
        if (row.get("draw_schedule_fingerprint") != digest or (shared is not None and shared != digest)
                or report.get("draw_schedule_fingerprint") != digest
                or observed_bindings.get("reference_draw_schedule_fingerprint") != digest):
            _fail("reference_report_draw_schedule_mismatch")
        shared = digest
    return record["experiment_id"]


def _pin_shape(pin):
    pilot._keys(pin, pilot.PIN_KEYS, "artifact_pin_required")
    if (type(pin["path"]) is not str or not Path(pin["path"]).is_absolute()
            or type(pin["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
            or type(pin["size_bytes"]) is not int or pin["size_bytes"] <= 0):
        _fail("artifact_pin_invalid")


def admit_reference_request(request, *, repository_root):
    """Authenticate metadata and registration; no CUDA or checkpoint tensors."""
    pilot._keys(request, {"schema_version", "comparison_id", "data_manifest", "config",
        "source_checkpoint", "source_model_state_fingerprint", "source_manifest", "runtime",
        "gate", "experiment_record", "max_wall_seconds"}, "reference_request_keys_invalid")
    if request["schema_version"] != REQUEST_SCHEMA:
        _fail("request_schema_invalid")
    if (type(request["comparison_id"]) is not str
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", request["comparison_id"])):
        _fail("comparison_id_invalid")
    if type(request["max_wall_seconds"]) is not int or request["max_wall_seconds"] != 600:
        _fail("fixed_wall_budget_required")
    for key in ("config", "data_manifest", "source_manifest", "source_checkpoint", "experiment_record"):
        _pin_shape(request[key])
    if (type(request["source_model_state_fingerprint"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", request["source_model_state_fingerprint"])):
        _fail("source_state_pin_required")
    pilot.validate_runtime_policy(request["runtime"])
    config = FusionTrainingConfig()
    if fingerprint(pilot.checked_json(request["config"])) != fingerprint(asdict(config)):
        _fail("fixed_p512_config_required")
    pilot.verify_source_manifest(request["source_manifest"], repository_root)
    stable = pilot.stable_bindings(request)
    record = pilot._record(request["experiment_record"])
    if (record["kind"] != "engineering_gate" or record["seeds"] != [17]
            or type(record["budget"].get("planned")) is not int or record["budget"]["planned"] != 2
            or type(record["budget"].get("planned_draws")) is not int or record["budget"]["planned_draws"] != 4
            or record.get("protocol", {}).get("fusion_pilot_bindings") != reference_gate_bindings(stable)
            or fingerprint(record.get("protocol", {}).get("reference_train_smoke")) != fingerprint(reference_protocol(stable))):
        _fail("planned_reference_contract_mismatch")
    policy = record["data_access"]["declared_policy"]
    if any(policy[key] is not False for key in ("validation_allowed", "test_inputs_allowed",
            "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")):
        _fail("train_only_policy_required")
    pilot.verify_gate(request["gate"], role="cuda_synthetic", bindings=stable,
                      runtime=request["runtime"], repository_root=repository_root)
    from .harmonic_fusion_data import FrozenFusionData
    from .full_orbit_training import FullOrbitSampler
    data = FrozenFusionData(pilot.checked_json(request["data_manifest"]))
    sampler = FullOrbitSampler(data.table, seed=config.seed)
    draws = [asdict(sampler.peek(index)) for index in range(2)]
    data.require_train_records({row["record_id"] for row in draws})
    return {"data": data, "config": config, "experiment_id": record["experiment_id"],
            "stable_bindings": stable, "draws": draws,
            "bindings": {**data.bindings, **stable, "device": "cuda",
                         "reference_draw_schedule_fingerprint": fingerprint(draws)}}


class _ReferenceStepSession(FusionTrainingSession):
    """TRAIN-only engineering session; forbidden as a full-resume artifact.

    The unchanged base ``step`` performs all preparation, forward, target
    alignment and AdamW work. Only validation scheduling is inapplicable to
    this explicitly registered one-step protocol. No validation is fabricated.
    """
    @property
    def pending_validation(self):
        return None

    def step(self, reader, class_weights):
        if self.applied_updates != 0:
            _fail("single_update_budget_exhausted")
        return super().step(reader, class_weights)


class _Journal:
    def __init__(self, root, *, experiment_id, bindings):
        self.path = root / "progress.jsonl"
        self.path.touch(exist_ok=False)
        self.started = time.monotonic()
        self.index, self.previous = 0, None
        self.state = {"experiment_id": experiment_id, "bindings_fingerprint": fingerprint(bindings),
            "status": "running", "stage": "initialization", "event_status": "started", "arm": None,
            "arms": {arm: {"attempted_updates": 0, "applied_updates": 0,
                "reader_calls": 0, "reader_completed": 0, "raw_prediction_calls": 0,
                "target_alignment_checks": 0, "update_in_flight": False} for arm in pilot.ARMS}}
        self.append("initialization", "started")

    def append(self, stage, status, **detail):
        self.state.update(stage=stage, event_status=status)
        row = {"schema_version": JOURNAL_SCHEMA, "index": self.index, "previous": self.previous,
               "stage": stage, "status": status, "detail": detail, "state": deepcopy(self.state),
               "elapsed_wall_seconds": time.monotonic() - self.started}
        row["fingerprint"] = fingerprint(row)
        import os
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.index += 1
        self.previous = row["fingerprint"]


def _execute_reference(output_root, *, experiment_id, config, bindings, table, class_weights,
                       reader, session_factory, verify_inputs, setup_runtime, execution_context=None,
                       synthetic=False, max_wall_seconds=600, started_at=None):
    """Private CPU fixture seam; the public path requires registered real CUDA."""
    from music_critic.graph import validate_raw_graph_batch
    from .corrected_training import ACTIVE_HEADS, align_target_sidecars_after_prediction
    from .corrected_model import model_state_fingerprint
    from .full_orbit_training import FullOrbitSampler
    from .harmonic_fusion_run import _json_bytes, _write_bytes_once
    import torch
    root = Path(output_root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    journal = _Journal(root, experiment_id=experiment_id, bindings=bindings)
    started = time.monotonic() if started_at is None else started_at
    runtime, source_verified, session = None, False, None
    arms, handles = [], []

    def deadline():
        if time.monotonic() - started >= max_wall_seconds:
            raise TimeoutError("fusion_reference.wall_budget_exceeded")

    def counts():
        rows = journal.state["arms"].values()
        return {key: sum(row[key] for row in rows) for key in (
            "attempted_updates", "applied_updates", "reader_calls", "reader_completed")}

    def envelope(passed):
        total = counts()
        return {"schema_version": REPORT_SCHEMA, "passed": passed,
            "experiment_id": experiment_id, "state": deepcopy(journal.state),
            "actual_updates": total["applied_updates"], "actual_draws": 2 * total["applied_updates"],
            "attempted_updates": total["attempted_updates"], "attempted_draw_slots": 2 * total["attempted_updates"],
            "reader_calls": total["reader_calls"], "reader_completed": total["reader_completed"],
            "validation_reader_calls": 0, "device": bindings.get("device", "cpu"),
            "real_data_accessed": (False if synthetic or not total["reader_calls"] else
                                   True if total["reader_completed"] else None),
            "unconfirmed_in_flight_tail": any(row["update_in_flight"] for row in journal.state["arms"].values()),
            "synthetic": synthetic, "scientific_acceptance": False,
            "reference_validation_policy": VALIDATION_POLICY, "resumable": False,
            "source_hashes_verified": source_verified and total["reader_completed"] == 4,
            "raw_only_prediction_verified": all(row["raw_prediction_calls"] == 1 for row in journal.state["arms"].values()),
            "target_alignment_verified": all(row["target_alignment_checks"] == 1 for row in journal.state["arms"].values()),
            "config": config.to_dict(), "bindings": deepcopy(bindings), "environment": deepcopy(runtime),
            "execution_context_fingerprint": fingerprint(execution_context),
            "wall_seconds": time.monotonic() - started, **_ACCESS}

    try:
        pilot._write_json(root / "resolved.json", {"config": config.to_dict(), "bindings": bindings,
            "experiment_id": experiment_id, "execution_context": deepcopy(execution_context),
            "synthetic": synthetic, "max_wall_seconds": max_wall_seconds,
            "reference_validation_policy": VALIDATION_POLICY, "command": list(sys.argv)})
        deadline()
        journal.append("runtime", "started")
        runtime = setup_runtime()
        journal.append("runtime", "completed", observed_runtime=runtime)
        deadline()
        journal.append("verify_inputs", "started")
        verify_inputs()
        source_verified = True
        journal.append("verify_inputs", "completed")
        expected = FullOrbitSampler(table, seed=config.seed)
        draws = [asdict(expected.peek(index)) for index in range(2)]
        for arm in pilot.ARMS:
            deadline()
            journal.state["arm"] = arm
            current = journal.state["arms"][arm]
            journal.append("initialize_arm", "started")
            session = session_factory(arm == "fusion")
            if (not isinstance(session, _ReferenceStepSession) or session.config != config
                    or session.applied_updates != 0 or session.sampler.position != 0
                    or session.validations or session.model.enabled != (arm == "fusion")
                    or session.source_model_state != bindings["source_model_state_fingerprint"]):
                _fail("fresh_reference_session_required")
            initial = model_state_fingerprint(session.model)
            loaded, alignment_summary = [], {}
            journal.append("initialize_arm", "completed", source_model_state=session.source_model_state)

            def observed_reader(record_id, *, split):
                deadline()
                if split != "train" or len(loaded) >= 2 or record_id != draws[len(loaded)]["record_id"]:
                    _fail("fixed_train_draw_required")
                current["reader_calls"] += 1
                journal.append("reader", "started", split=split, record_id=record_id)
                raw, sidecar = reader(record_id, split=split)
                loaded.append((raw, sidecar))
                current["reader_completed"] += 1
                journal.append("reader", "completed", split=split, record_id=record_id)
                return raw, sidecar

            def before_forward(module, args):
                deadline()
                if len(args) != 1 or len(loaded) != 2:
                    _fail("raw_only_forward_required")
                validate_raw_graph_batch(args[0], sample_count=2)
                journal.append("raw_forward", "started")

            def after_forward(module, args, output):
                deadline()
                if current["raw_prediction_calls"] or set(output.logits) != set(ACTIVE_HEADS):
                    _fail("one_native_forward_required")
                if not all(bool(torch.isfinite(value).all()) for value in output.logits.values()):
                    _fail("finite_native_logits_required")
                current["raw_prediction_calls"] += 1
                journal.append("raw_forward", "completed")
                # Observe the real forward output and the same explicit sidecars
                # before the unchanged update performs its own native loss join.
                aligned = align_target_sidecars_after_prediction(output, args[0],
                    [row[1] for row in loaded], shifts=tuple(row["shift_pc"] for row in draws))
                alignment_summary.update({task: {"target_rows": len(value.valid_mask),
                    "valid_rows": int(value.valid_mask.sum())} for task, value in aligned.heads.items()})
                current["target_alignment_checks"] += 1
                journal.append("target_alignment", "completed", heads=alignment_summary)

            def optimizer_applied(optimizer, args, kwargs):
                # The disabled FP32 scaler executes AdamW exactly once. Capture
                # that effect before later scheduler/history/journal errors.
                current["applied_updates"] += 1
                journal.append("optimizer", "completed")

            handles = [session.model.register_forward_pre_hook(before_forward),
                       session.model.register_forward_hook(after_forward),
                       session.optimizer.register_step_post_hook(optimizer_applied)]
            current["attempted_updates"] += 1
            current["update_in_flight"] = True
            journal.append("update", "started", draws=draws)
            result = session.step(observed_reader, class_weights)
            for handle in handles:
                handle.remove()
            handles = []
            if (result["draws"] != draws or session.applied_updates != 1 or session.sampler.position != 2
                    or session.validations or current["applied_updates"] != 1
                    or current["raw_prediction_calls"] != 1 or current["target_alignment_checks"] != 1):
                _fail("one_step_reference_contract_failed")
            current["update_in_flight"] = False
            journal.append("update", "completed", update=result)
            arms.append({"arm": arm, "actual_updates": 1, "actual_draws": 2,
                "draws": result["draws"], "draw_schedule_fingerprint": fingerprint(result["draws"]),
                "source_model_state": session.source_model_state, "sampler_position": session.sampler.position,
                "raw_prediction_calls": current["raw_prediction_calls"],
                "target_alignment_checks": current["target_alignment_checks"], "alignment": alignment_summary,
                "training_loss": result["loss"], "lr_used": result["lr_used"], "lr_next": result["lr_next"],
                "model_state_changed": model_state_fingerprint(session.model) != initial,
                "optimizer_steps": deepcopy(session.optimizer_steps), "validation_updates": []})
            session = None
            journal.append("arm", "completed")
        deadline()
        if counts() != {"attempted_updates": 2, "applied_updates": 2, "reader_calls": 4, "reader_completed": 4}:
            _fail("total_budget_mismatch")
        journal.state["status"] = "completed"
        journal.append("attempt", "completed")
        report = {**envelope(True), "reference_smoke_passed": True, "arms": arms,
            "draw_schedule_fingerprint": fingerprint(draws),
            "process_resources": pilot.process_resources(bindings.get("device", "cpu")),
            "registry_finalization": "coordinator_must_finalize_before_decision_or_next_run"}
        report["fingerprint"] = fingerprint(report)
        content = _json_bytes(report)
        manifest = {path.name: pilot.artifact_pin(path) for path in sorted(root.iterdir()) if path.is_file()}
        manifest["report.json"] = {"path": str(root / "report.json"), "sha256": sha256(content).hexdigest(),
                                   "size_bytes": len(content)}
        pilot._write_json(root / "artifact-manifest.json", manifest)
        _write_bytes_once(root / "report.json", content)
        return report
    except BaseException as exc:
        failure = {"exception_type": type(exc).__name__, "message": str(exc),
                   "stage": journal.state["stage"], "arm": journal.state["arm"]}
        journal.state["status"] = "failed"
        try:
            journal.append("attempt", "failed", failure=failure)
            receipt = {**envelope(False), "reference_smoke_passed": False, "failure": failure, "arms": arms}
            receipt["fingerprint"] = fingerprint(receipt)
            pilot._write_json(root / "failure-receipt.json", receipt)
        except BaseException as receipt_error:
            exc.add_note("Failure receipt incomplete: " + repr(receipt_error))
            raise exc from receipt_error
        raise
    finally:
        for handle in handles:
            handle.remove()


def run_reference_request(request, output_root, *, repository_root):
    """Coordinator dispatch only: two real TRAIN updates on the reserved CUDA."""
    started = time.monotonic()
    admitted = admit_reference_request(request, repository_root=repository_root)
    from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
    from .corrected_model import model_state_fingerprint
    base = None

    def factory(enabled):
        nonlocal base
        if base is None:
            pin = request["source_checkpoint"]
            base = load_model_checkpoint(pin["path"], expected_sha256=pin["sha256"],
                                         expected_size_bytes=pin["size_bytes"]).model
            if model_state_fingerprint(base) != request["source_model_state_fingerprint"]:
                _fail("source_state_mismatch")
        return _ReferenceStepSession(base=base, enabled=enabled, config=admitted["config"],
            bindings=admitted["bindings"], table=admitted["data"].table, device="cuda")

    def verify_inputs():
        pilot.verify_source_manifest(request["source_manifest"], repository_root)
        pilot.checked_bytes(request["config"])
        pilot.checked_bytes(request["source_checkpoint"])
        # FrozenFusionData authenticated metadata at admission. Its reader
        # rechecks metadata identities and hashes exactly the four requested
        # TRAIN payload sets. Do not verify/read unused VALIDATION warm files.

    return _execute_reference(output_root, experiment_id=admitted["experiment_id"],
        config=admitted["config"], bindings=admitted["bindings"], table=admitted["data"].table,
        class_weights=admitted["data"].class_weights, reader=admitted["data"].reader,
        session_factory=factory, verify_inputs=verify_inputs,
        setup_runtime=lambda: pilot.prepare_pilot_runtime(request["runtime"]),
        execution_context={"request": deepcopy(request)}, max_wall_seconds=request["max_wall_seconds"],
        started_at=started)
