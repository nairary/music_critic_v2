"""Coordinator-operated, source-bound Dilemmadata SSL pilot and recovery.

Preflight resolves metadata only. Sizing reads raw historical cache pieces;
preparation additionally reads TRAIN/VALIDATION targets. Run/resume also load
the strict C2 model-only artifact. Every action requires its own matching
planned experiment. No TEST entrypoint exists.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile

import torch

from music_critic.experiments.registry import load_record
from music_critic.ssl.deterministic_runtime import deterministic_cuda_evidence_runtime
from music_critic.ssl.group_masking_pilot import GroupPilotConfig, tree_fingerprint
from music_critic.ssl.theory_warmstart import TheoryWarmstartSSLModel, load_model_only_theory_warmstart


PLAN_SCHEMA = "WS2GroupMaskingScientificPlan@1.0.0"


def implementation_fingerprint(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    paths = sorted((root / "src").rglob("*.py")) + [root / "scripts/run_group_masking_scientific.py"]
    return tree_fingerprint({str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths})


def _require(condition, reason):
    if not condition:
        raise ValueError("group_scientific." + reason)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _artifact_binding(value, *, maximum=64 * 1024**2):
    _require(type(value) is dict and set(value) == {"sha256", "size_bytes"}, "artifact_binding_required")
    _require(_sha(value["sha256"]) and type(value["size_bytes"]) is int
             and 0 < value["size_bytes"] <= maximum, "artifact_binding_invalid")
    return value


def read_bound(path, binding, *, maximum=64 * 1024**2):
    """Read regular bytes under an external hash and an explicit resource bound."""
    _artifact_binding(binding, maximum=maximum)
    _require(path is not None, "artifact_path_required")
    path = Path(path)
    before = path.lstat()
    _require(not stat.S_ISLNK(before.st_mode), "symlink_artifact_forbidden")
    _require(stat.S_ISREG(before.st_mode), "regular_artifact_required")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        actual = os.fstat(stream.fileno())
        _require(stat.S_ISREG(actual.st_mode) and (actual.st_dev, actual.st_ino) == (before.st_dev, before.st_ino),
                 "regular_artifact_required")
        raw = stream.read(binding["size_bytes"] + 1)
    _require(len(raw) == binding["size_bytes"] and sha256(raw).hexdigest() == binding["sha256"],
             "artifact_bytes_mismatch")
    return raw


def _json_bytes(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def invalid_constant(_):
        raise ValueError("group_scientific.nonfinite_json")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _read_plan(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    _require(len(raw) <= 1024**2, "plan_size_bound")
    plan = _json_bytes(raw)
    _require(type(plan) is dict and set(plan) == {
        "schema", "pilot", "recovery", "checkpoint_interval", "source_checkpoint",
        "data_request", "prepared_data", "resume",
    } and plan["schema"] == PLAN_SCHEMA, "plan_schema_invalid")
    # Production budgets and objectives are the predeclared scientific question.
    # Tiny mechanics tests use the runner API; the real CLI has no bypass flag.
    pilot = GroupPilotConfig(**plan["pilot"])
    recovery = GroupPilotConfig(**plan["recovery"])
    _require(tree_fingerprint(asdict(pilot)) == tree_fingerprint(asdict(GroupPilotConfig())),
             "scientific_pilot_config_changed")
    expected_recovery = replace(pilot, updates=100, lambda_ssl=0., lambda_theory=1.)
    _require(tree_fingerprint(asdict(recovery)) == tree_fingerprint(asdict(expected_recovery)),
             "scientific_recovery_config_changed")
    _require(type(plan["checkpoint_interval"]) is int and 1 <= plan["checkpoint_interval"] <= 100,
             "checkpoint_interval_invalid")
    source = plan["source_checkpoint"]
    _require(type(source) is dict and set(source) == {
        "kind", "locator", "parent_experiment_id", "sha256", "size_bytes", "state_fingerprint",
    } and source["kind"] == "C2_model_only" and _sha(source["state_fingerprint"])
        and type(source["parent_experiment_id"]) is str and bool(source["parent_experiment_id"])
        and type(source["locator"]) is str and bool(source["locator"]), "strict_C2_binding_required")
    _artifact_binding({k: source[k] for k in ("sha256", "size_bytes")}, maximum=256 * 1024**2)
    _artifact_binding(plan["data_request"])
    if plan["prepared_data"] is not None:
        _artifact_binding(plan["prepared_data"])
    if plan["resume"] is not None:
        resume = plan["resume"]
        _require(type(resume) is dict and set(resume) == {"sha256", "size_bytes", "parent_experiment_id"}
                 and type(resume["parent_experiment_id"]) is str and bool(resume["parent_experiment_id"]),
                 "resume_binding_invalid")
        _artifact_binding({k: resume[k] for k in ("sha256", "size_bytes")})
    return plan, raw, pilot, recovery


def preflight(*, plan_path, record_path, action, data_request_path, prepared_data_path=None,
              resume_manifest=None):
    """No source conversion, checkpoint read or CUDA initialization here."""
    _require(action in {"size", "prepare", "run", "resume"}, "action_invalid")
    plan, raw, pilot, recovery = _read_plan(plan_path)
    record = load_record(record_path)
    code = implementation_fingerprint()
    preparation = action == "prepare"
    sizing = action == "size"
    engineering = preparation or sizing
    _require(record["status"] == "planned", "planned_record_required")
    _require(record["kind"] == ("engineering_gate" if engineering else "scientific_comparison")
             and record["phase"] == "WS2-SSL-GROUP", "experiment_kind_or_phase_invalid")
    protocol = record["protocol"]
    expected = {"implementation_fingerprint": code, "scientific_plan_sha256": sha256(raw).hexdigest(),
                "data_request": plan["data_request"], "prepared_data": plan["prepared_data"],
                "source_checkpoint": plan["source_checkpoint"], "resume": plan["resume"],
                "pilot_config": pilot.contract(), "recovery_config": recovery.contract(),
                "operation": action}
    _require(set(expected) <= set(protocol)
             and all(tree_fingerprint(protocol[k]) == tree_fingerprint(v) for k, v in expected.items()),
             "planned_protocol_binding_mismatch")
    policy = record["data_access"]["declared_policy"]
    _require(policy["validation_allowed"] is True and all(policy[k] is False for k in (
        "test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")),
        "TRAIN_VALIDATION_only_policy_required")
    _require(plan["source_checkpoint"]["parent_experiment_id"] in record["parent_ids"], "C2_parent_missing")
    if engineering:
        _require(plan["prepared_data"] is None and plan["resume"] is None,
                 "size_must_compile_fresh_report" if sizing else "prepare_must_compile_fresh_binding")
    else:
        _artifact_binding(plan["prepared_data"])
        _require((action == "resume") == (plan["resume"] is not None), "resume_mode_mismatch")
        if action == "resume":
            _require(plan["resume"]["parent_experiment_id"] in record["parent_ids"], "resume_parent_missing")
            previous = _json_bytes(read_bound(resume_manifest, {k: plan["resume"][k] for k in ("sha256", "size_bytes")}))
            _require(type(previous) is dict and previous.get("attempt_id") == plan["resume"]["parent_experiment_id"],
                     "resume_parent_identity_mismatch")
    # Parse exactly the externally hash-verified bytes, without a second read
    # that could substitute a different request between validation and parsing.
    request_value = _json_bytes(read_bound(data_request_path, plan["data_request"]))
    from music_critic.ssl import group_masking_scientific_data as data_api
    if type(request_value) is dict and request_value.get("schema") != data_api.REQUEST_SCHEMA:
        from music_critic.ssl import group_masking_scientific_cached_data as data_api
    if sizing:
        from music_critic.ssl.group_masking_scientific_cached_data import REQUEST_SCHEMA
        _require(type(request_value) is dict and request_value.get("schema") == REQUEST_SCHEMA,
                 "sizing_requires_cached_request")
    request = data_api.ScientificDataRequest(request_value)
    metadata = (data_api.preflight_data(request, pilot, recovery, sizing=True) if sizing
                else data_api.preflight_data(request, pilot, recovery))
    prepared = None if engineering else _json_bytes(read_bound(prepared_data_path, plan["prepared_data"]))
    return {"plan": plan, "plan_sha256": sha256(raw).hexdigest(), "record": record,
            "implementation_fingerprint": code, "pilot": pilot, "recovery": recovery,
            "request": request, "metadata": metadata, "prepared": prepared, "data_api": data_api}


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_once(path, value):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink()
    return {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _execute_size(checked, output):
    """Publish an independently registered raw-only sizing attempt, once."""
    output.mkdir(parents=True, exist_ok=False)
    receipts = output / "sizing.jsonl"
    receipts.touch(exist_ok=False)
    _sync_directory(output)
    last_receipt = None

    def receipt(value):
        nonlocal last_receipt
        _append(receipts, value)
        last_receipt = value

    previous_threads = torch.get_num_threads()
    try:
        receipt({"status": "started", "operation": "size", "experiment_id": checked["record"]["experiment_id"],
                 "raw_sources_read": False, "checkpoint_loaded": False, "optimizer_updates": 0})
        torch.set_num_threads(1)
        sizing = checked["data_api"].size_scientific_data(
            checked["request"], checked["pilot"], checked["recovery"], on_receipt=receipt)
        evidence = write_json_once(output / "sizing-report.json", sizing)
        report = {"status": "completed", "operation": "size", "sizing_report": evidence,
                  "plan_sha256": checked["plan_sha256"], "implementation_fingerprint": checked["implementation_fingerprint"],
                  "experiment_id": checked["record"]["experiment_id"],
                  "access_observations": sizing["access_observations"],
                  "TEST_access": False, "checkpoint_loaded": False, "optimizer_updates": 0,
                  "next_action": "Finalize sizing record; choose explicit resource limits and bind class weights in a new planned preparation."}
        receipt({"status": "report_published", "operation": "size", "sizing_report": evidence,
                 "access_observations": sizing["access_observations"]})
        write_json_once(output / "result.json", report)
        return report
    except BaseException as exc:
        failure = {"status": "failed", "operation": "size", "type": type(exc).__name__, "reason": str(exc),
                   "last_receipt": last_receipt, "access_observations_complete": False,
                   "checkpoint_loaded": False, "optimizer_updates": 0}
        # Preserve the original failure if a receipt disk itself fails. A saved
        # sizing report alone does not declare a completed attempt without result.
        for publish in (lambda: _append(receipts, failure),
                        lambda: write_json_once(output / "failure.json", failure)):
            try:
                publish()
            except BaseException:
                pass
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight-size", "size", "preflight-prepare", "prepare", "preflight-run", "run", "resume"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--experiment-record", required=True, type=Path)
    parser.add_argument("--data-request", required=True, type=Path)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    action = args.action.removeprefix("preflight-")
    if action == "size":
        _require(args.device == "cpu", "sizing_cpu_only")
    checked = preflight(plan_path=args.plan, record_path=args.experiment_record, action=action,
                        data_request_path=args.data_request, prepared_data_path=args.prepared_data,
                        resume_manifest=args.resume_manifest)
    if args.action.startswith("preflight-"):
        print(json.dumps({"status": "metadata_preflight_passed", "operation": action,
                          "raw_sources_read": False, "checkpoint_loaded": False, "CUDA_initialized": False,
                          "implementation_fingerprint": checked["implementation_fingerprint"]}))
        return 0
    _require(args.output is not None, "fresh_output_required")
    data_api = checked["data_api"]
    if action == "size":
        report = _execute_size(checked, args.output)
        print(json.dumps(report))
        return 0
    if action == "prepare":
        args.output.mkdir(parents=True, exist_ok=False)
        receipts = args.output / "preparation.jsonl"
        receipts.touch(exist_ok=False)
        _sync_directory(args.output)
        _append(receipts, {"status": "started", "experiment_id": checked["record"]["experiment_id"]})
        try:
            data = data_api.prepare_scientific_data(checked["request"], checked["pilot"], checked["recovery"],
                                           on_receipt=lambda r: _append(receipts, r))
            evidence = write_json_once(args.output / "prepared-binding.json", data.binding)
            report = {"status": "completed", "operation": "prepare", "prepared_data": evidence,
                      "plan_sha256": checked["plan_sha256"], "implementation_fingerprint": checked["implementation_fingerprint"],
                      "experiment_id": checked["record"]["experiment_id"], "TEST_access": False,
                      "checkpoint_loaded": False, "optimizer_updates": 0,
                      "next_action": "Finalize preparation record; bind these bytes in a new planned run before model execution."}
            write_json_once(args.output / "result.json", report)
        except BaseException as exc:
            _append(receipts, {"status": "failed", "type": type(exc).__name__, "reason": str(exc)})
            write_json_once(args.output / "failure.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc)})
            raise
        print(json.dumps(report))
        return 0
    data = data_api.load_bound_scientific_data(checked["request"], checked["prepared"], checked["pilot"], checked["recovery"])
    source = checked["plan"]["source_checkpoint"]
    read_bound(args.source_checkpoint, {k: source[k] for k in ("sha256", "size_bytes")}, maximum=256 * 1024**2)
    from music_critic.ssl.group_masking_scientific_runner import run_scientific_workflow
    contract = {"implementation_fingerprint": checked["implementation_fingerprint"],
                "data_request": checked["plan"]["data_request"], "prepared_data": checked["plan"]["prepared_data"],
                "source_checkpoint": source, "pilot_config": checked["pilot"].contract(),
                "recovery_config": checked["recovery"].contract(), "checkpoint_interval": checked["plan"]["checkpoint_interval"]}
    def factory():
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(checked["pilot"].seed)
            destination = TheoryWarmstartSSLModel(checked["pilot"].ssl_config())
            result, _ = load_model_only_theory_warmstart(destination, args.source_checkpoint,
                expected_sha256=source["sha256"], expected_size_bytes=source["size_bytes"],
                expected_state_fingerprint=source["state_fingerprint"])
            return result
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    context = deterministic_cuda_evidence_runtime() if args.device.startswith("cuda") else nullcontext()
    try:
        with context:
            if args.device.startswith("cuda"):
                _require(torch.cuda.is_available(), "CUDA_unavailable")
            resume = checked["plan"]["resume"]
            report = run_scientific_workflow(warmstart_factory=factory, source_binding=source, data=data,
                pilot_config=checked["pilot"], recovery_config=checked["recovery"], output_root=args.output,
                contract_binding=contract, device=args.device, checkpoint_interval=checked["plan"]["checkpoint_interval"],
                resume_manifest=args.resume_manifest if resume else None,
                expected_resume_sha256=resume["sha256"] if resume else None,
                expected_resume_size_bytes=resume["size_bytes"] if resume else None,
                attempt_id=checked["record"]["experiment_id"])
    finally:
        torch.set_num_threads(previous_threads)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
