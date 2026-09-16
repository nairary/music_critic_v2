#!/usr/bin/env python3
"""Finalize one bound TRAIN reference gate from complete supervised evidence.

This reads JSON/journal artifacts only. The request's immutable planned copy
is retained; only its identical, still-planned registry record is replaced.
Run ledger regeneration separately after successful finalization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import stat
import tempfile

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.harmonic_fusion_dispatch import GPU_UUID
from music_critic.experiments.analysisgnn.harmonic_fusion_pilot import (
    artifact_pin, checked_bytes, checked_json, decode_json, stable_bindings, validate_runtime_policy,
)
from music_critic.experiments.analysisgnn.harmonic_fusion_reference import (
    JOURNAL_SCHEMA, REQUEST_SCHEMA, verify_reference_report,
)
from music_critic.experiments.registry import canonical_json_bytes, seal_record, validate_record


def _fail(reason):
    raise ValueError("fusion_reference_finalization." + reason)


def _regular(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        _fail("regular_evidence_file_required")
    return path


def _number(value, *, minimum=0, maximum=math.inf, integer=False):
    return (type(value) is int if integer else type(value) in (int, float)) and (
        math.isfinite(value) and minimum <= value <= maximum)


def _resource(receipt, *, request_path, report_path, repository_root):
    if (receipt.get("schema") != "ResearchPilotResourceReceipt@1.0.0"
            or type(receipt.get("child_returncode")) is not int or receipt["child_returncode"] != 0
            or receipt.get("process_status") != "exited_zero" or receipt.get("reason") != "child_exit"
            or receipt.get("gpu_uuid") != GPU_UUID):
        _fail("successful_supervised_child_required")
    life = receipt.get("child_lifecycle", {})
    if (life.get("ownership_lost") is not False
            or life.get("ownership_confirmed_before_cleanup") is not True
            or life.get("leader_reaped_after_cleanup") is not True):
        _fail("confirmed_cleanup_before_reap_required")
    if (not _number(receipt.get("wall_limit_seconds"), minimum=0.001, maximum=600)
            or not _number(receipt.get("wall_seconds"), maximum=receipt["wall_limit_seconds"])
            or not _number(receipt.get("gpu_memory_ceiling_mib"), minimum=1, maximum=16384)
            or not _number(receipt.get("host_available_floor_mib"), minimum=8192)
            or not _number(receipt.get("max_swap_growth_mib"), maximum=256)
            or not _number(receipt.get("resource_samples"), minimum=1, integer=True)):
        _fail("supervisor_resource_limits_invalid")
    if (not _number(receipt.get("sampled_gpu_memory_peak_mib"), maximum=receipt["gpu_memory_ceiling_mib"])
            or not _number(receipt.get("sampled_host_available_min_kib"),
                           minimum=receipt["host_available_floor_mib"] * 1024)
            or not _number(receipt.get("initial_swap_used_kib"))
            or not _number(receipt.get("sampled_peak_swap_used_kib"))
            or receipt["sampled_peak_swap_used_kib"] - receipt["initial_swap_used_kib"]
                > receipt["max_swap_growth_mib"] * 1024):
        _fail("sampled_resource_limit_exceeded")
    command = receipt.get("command")
    helpers = {str(Path(__file__).absolute().with_name(name)) for name in (
        "run-pinned-request.py", "run_harmonic_fusion_pinned_request.py")}
    if (type(command) is not list or len(command) != 5
            or any(type(value) is not str for value in command)
            or not Path(command[0]).is_absolute() or command[1] not in helpers
            or command[2:] != ["reference", str(request_path), str(report_path.parent)]
            or receipt.get("cwd") != str(repository_root)):
        _fail("supervisor_command_not_bound_to_request_and_output")


def _progress(content, report):
    if not content or not content.endswith(b"\n"):
        _fail("complete_progress_journal_required")
    previous = None
    last = None
    for index, line in enumerate(content.splitlines()):
        row = decode_json(line)
        if type(row) is not dict:
            _fail("progress_journal_invalid")
        digest = row.pop("fingerprint", None)
        if (row.get("schema_version") != JOURNAL_SCHEMA or row.get("index") != index
                or row.get("previous") != previous or fingerprint(row) != digest
                or not _number(row.get("elapsed_wall_seconds"))):
            _fail("progress_journal_invalid")
        previous, last = digest, row
    if (last["stage"] != "attempt" or last["status"] != "completed"
            or fingerprint(last["state"]) != fingerprint(report["state"])
            or last["state"].get("experiment_id") != report["experiment_id"]
            or last["state"].get("status") != "completed"):
        _fail("completed_progress_report_state_mismatch")


def _atomic_complete(path, expected, content, snapshots):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # All finalizers use the exclusive per-record lock. Recheck both the
        # planned target and every authenticated artifact before publication.
        for evidence, before in snapshots.items():
            if _regular(evidence).read_bytes() != before:
                _fail("evidence_changed_before_publication")
        if _regular(path).read_bytes() != expected:
            _fail("planned_record_changed_before_publication")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize_reference(*, request_path, record_path, report_path, resource_receipt_path):
    request_path, record_path, report_path, resource_receipt_path = (
        _regular(path) for path in (request_path, record_path, report_path, resource_receipt_path))
    lock = record_path.with_name("." + record_path.name + ".finalize.lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        snapshots = {}

        def read(path):
            path = _regular(path)
            pin = artifact_pin(path)
            content = checked_bytes(pin)
            snapshots[path] = content
            return decode_json(content), pin

        planned, planned_pin = read(record_path)
        validate_record(planned, filename=record_path.name)
        expected = snapshots[record_path]
        if expected != canonical_json_bytes(planned) + b"\n":
            _fail("canonical_planned_record_required")
        if planned["status"] != "planned" or planned["kind"] != "engineering_gate":
            _fail("planned_record_required_terminal_is_immutable")
        request, request_pin = read(request_path)
        if request.get("schema_version") != REQUEST_SCHEMA or request.get("max_wall_seconds") != 600:
            _fail("bound_reference_request_required")
        original_pin = request["experiment_record"]
        original_path = _regular(original_pin["path"])
        original_bytes = checked_bytes(original_pin)
        original = checked_json(original_pin)
        validate_record(original, filename=original_path.name)
        if (original_path == record_path or original_bytes != expected
                or original["experiment_id"] != planned["experiment_id"]):
            _fail("separate_identical_immutable_planned_copy_required")
        snapshots[original_path] = original_bytes
        validate_runtime_policy(request["runtime"])
        bindings = stable_bindings(request)
        report, report_pin = read(report_path)
        if report.get("fingerprint") != fingerprint({key: value for key, value in report.items() if key != "fingerprint"}):
            _fail("report_self_fingerprint_mismatch")
        verify_reference_report(planned, report, bindings)
        if (report.get("execution_context_fingerprint") != fingerprint({"request": request})
                or not _number(report.get("wall_seconds"), maximum=600)):
            _fail("report_request_execution_context_mismatch")
        config_path = Path(request["config"]["path"])
        if (not config_path.is_absolute()
                or config_path.parts[-3:] != ("configs", "experiments", "harmonic_fusion_pilot.json")):
            _fail("repository_config_path_required")
        repository_root = config_path.parents[2]
        resource, resource_pin = read(resource_receipt_path)
        _resource(resource, request_path=request_path, report_path=report_path, repository_root=repository_root)
        manifest_path = report_path.with_name("artifact-manifest.json")
        manifest, manifest_pin = read(manifest_path)
        if type(manifest) is not dict or set(manifest) != {"report.json", "progress.jsonl", "resolved.json"}:
            _fail("complete_reference_artifact_manifest_required")
        members = {}
        for name, pin in manifest.items():
            path = report_path.parent / name
            if pin.get("path") != str(path):
                _fail("manifest_member_path_mismatch")
            content = checked_bytes(pin)
            snapshots[_regular(path)] = content
            members[name] = content
        if members["report.json"] != snapshots[report_path]:
            _fail("manifest_report_mismatch")
        _progress(members["progress.jsonl"], report)
        resolved = decode_json(members["resolved.json"])
        expected_command = [str(repository_root / "scripts/run_harmonic_fusion_reference_gate.py"),
            "--request", str(request_path), "--request-sha256", request_pin["sha256"],
            "--request-size-bytes", str(request_pin["size_bytes"]), "--execute", "--output-root", str(report_path.parent)]
        if (resolved.get("execution_context") != {"request": request}
                or resolved.get("experiment_id") != planned["experiment_id"]
                or resolved.get("command") != expected_command
                or resolved.get("synthetic") is not False
                or fingerprint(resolved.get("bindings")) != fingerprint(report["bindings"])):
            _fail("resolved_request_binding_mismatch")
        completed = deepcopy(planned)
        completed.update(status="completed", validity={"valid": True}, environment=deepcopy(report["environment"]),
            finalized_at=datetime.now(timezone.utc).isoformat())
        completed["budget"].update(actual=2, actual_draws=4, attempted_updates=2, attempted_draw_slots=4,
            wall_seconds=report["wall_seconds"], supervised_wall_seconds=resource["wall_seconds"])
        completed["results"] = {"actual_updates": 2, "actual_draws": 4,
            "reference_smoke_passed": True, "source_hashes_verified": True,
            "raw_only_prediction_verified": True, "target_alignment_verified": True,
            "draw_schedule_fingerprint": report["draw_schedule_fingerprint"],
            "report_fingerprint": report["fingerprint"], "sampled_gpu_memory_peak_mib": resource["sampled_gpu_memory_peak_mib"]}
        completed["data_access"].update(access_evidence_status="verified_primary_evidence",
            validation_accessed=False, test_inputs_accessed=False, test_targets_accessed=False,
            test_metrics_computed=False, test_used_for_selection=False,
            train_inputs_accessed=True, train_targets_accessed=True)
        completed["claims"] = {"supported": [
            "Both equally initialized arms completed one TRAIN update and two draws on the bound CUDA runtime.",
            "Current source hashes, raw-only prediction, target alignment and the sampled resource limits were verified."],
            "not_supported": ["musical benefit", "VALIDATION performance", "TEST performance", "fresh-process resume"]}
        completed["artifacts"] = [{"role": role, "sha256": pin["sha256"], "size_bytes": pin["size_bytes"],
            "availability": "local_verified", "locator": os.path.relpath(pin["path"], repository_root)}
            for role, pin in (("reference_report", report_pin), ("reference_artifact_manifest", manifest_pin),
                              ("reference_progress", manifest["progress.jsonl"]), ("supervisor_resource_receipt", resource_pin))]
        completed["import_provenance"].update(mode="bound_reference_gate_finalization",
            planned_record_sha256=planned_pin["sha256"], reference_request_sha256=request_pin["sha256"],
            resource_receipt_sha256=resource_pin["sha256"], checkpoint_tensor_reads=0)
        completed = seal_record(completed)
        validate_record(completed, filename=record_path.name)
        _atomic_complete(record_path, expected, canonical_json_bytes(completed) + b"\n", snapshots)
        return {"experiment_id": completed["experiment_id"], "status": "completed",
                "record": artifact_pin(record_path), "record_fingerprint": completed["record_fingerprint"],
                "actual_updates": 2, "actual_draws": 4, "immutable_planned_copy_preserved": True}
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "record", "report", "resource-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    result = finalize_reference(request_path=args.request, record_path=args.record,
        report_path=args.report, resource_receipt_path=args.resource_receipt)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
