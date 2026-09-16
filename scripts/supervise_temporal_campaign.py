#!/usr/bin/env python3
"""Own one temporal run, enforce resource bounds, and seal its terminal record.

Run this foreground inside tmux/nohup (or another durable parent). The child is
a separate owned process group; the supervisor requires Linux waitid/WNOWAIT.
The child reads an immutable planned record copy, while this process advances
the master registry record. A machine crash/SIGKILL still requires recovery.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

from music_critic.experiments.registry import (
    canonical_json, load_record, load_registry, render_markdown_ledger,
    seal_record, validate_record,
)
from scripts.research_pilot_resource_wrapper import host_memory, load_ownership

SCHEMA = "TemporalCampaignSupervisor@1.0.0"
TERMINAL = {"completed", "failed", "aborted", "invalid"}
MAX_JSON_BYTES = 16 * 1024**2
MAX_JOURNAL_LINE = 65536


def require(value, reason):
    if not value:
        raise ValueError("temporal_supervisor." + reason)


def utc():
    return datetime.now(timezone.utc).isoformat()


def regular(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_forbidden")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "regular_file_required")
    return path


def digest(path):
    path = regular(path)
    h, size = sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block); size += len(block)
    return {"sha256": h.hexdigest(), "size_bytes": size}


def read_json(path):
    path = regular(path)
    require(path.stat().st_size <= MAX_JSON_BYTES, "json_size_bound")
    def bad(token):
        raise ValueError("nonfinite JSON: " + token)
    return json.loads(path.read_text(), parse_constant=bad)


def write_json(path, value):
    with Path(path).open("x") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".supervisor-tmp")
    with temporary.open("x") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def validate_config(config):
    require(type(config) is dict, "configuration")
    for key in ("cwd", "output_root", "run_output", "record_path", "project_root", "planned_record_input"):
        require(type(config.get(key)) is str and Path(config[key]).is_absolute(), "absolute_" + key)
        require(not any(p.is_symlink() for p in (Path(config[key]), *Path(config[key]).parents)), "symlink_" + key)
    require(type(config.get("command")) is list and config["command"]
            and all(type(s) is str and s and "\x00" not in s for s in config["command"]), "command_list")
    require(type(config.get("env", {})) is dict and all(type(k) is str and type(v) is str
            and "\x00" not in k+v and "=" not in k for k, v in config.get("env", {}).items()), "environment")
    require(config.get("mode") in ("train", "gate", "memory"), "mode")
    root, output = Path(config["output_root"]), Path(config["run_output"])
    require(output != root and output.is_relative_to(root), "run_output_inside_evidence_root")
    require(Path(config["record_path"]).parent == Path(config["project_root"]) / "docs/experiments/records",
            "master_registry_location")
    require(Path(config["planned_record_input"]) != Path(config["record_path"]), "separate_planned_input")
    prefix = Path(config.get("artifact_locator_prefix", ""))
    require(str(prefix) not in ("", ".") and not prefix.is_absolute() and ".." not in prefix.parts,
            "portable_artifact_locator")
    limits = {"gpu_memory_mib": 16384, "host_available_mib": 8192, "swap_growth_mib": 256,
              "wall_seconds": 36*3600, "poll_seconds": 5., "grace_seconds": 30.}
    limits.update(config.get("limits", {}))
    require(set(limits) == {"gpu_memory_mib", "host_available_mib", "swap_growth_mib", "wall_seconds",
                           "poll_seconds", "grace_seconds"}, "limit_fields")
    require(all(type(v) in (int, float) and math.isfinite(v) for v in limits.values()), "finite_limits")
    require(0 < limits["gpu_memory_mib"] <= 16384 and limits["host_available_mib"] >= 8192
            and 0 <= limits["swap_growth_mib"] <= 256 and 0 < limits["wall_seconds"] <= 36*3600
            and 5 <= limits["poll_seconds"] <= 10 and 0 < limits["grace_seconds"] <= 30, "resource_limits")
    start, stop = config.get("start_update", 0), config.get("expected_updates", 80000 if config["mode"] == "train" else 32)
    require(type(start) is int and start >= 0 and type(stop) is int and 0 <= stop <= 80000,
            "update_budget")
    return dict(config, limits=limits, start_update=start, expected_updates=stop)


def read_training_journal(path, *, start_update=0):
    """Count the complete valid applied-update prefix; truncated work is unknown."""
    result = {"rows": 0, "last_update": start_update, "draws_lower_bound": 0,
              "partial": False, "issue": None, "tail": None}
    if not Path(path).exists():
        return dict(result, partial=True, issue="training journal absent")
    with regular(path).open("rb") as stream:
        while True:
            line = stream.readline(MAX_JOURNAL_LINE + 1)
            if not line:
                break
            try:
                require(len(line) <= MAX_JOURNAL_LINE and line.endswith(b"\n"), "incomplete_or_oversized_line")
                row = json.loads(line)
                require(type(row) is dict and type(row.get("update")) is int
                        and row["update"] == result["last_update"] + 1, "journal_update_sequence")
                require(type(row.get("draws")) is list and len(row["draws"]) == 2, "journal_batch_two")
                require(all(type(d) is dict and type(d.get("record_id")) is str
                            and type(d.get("shift_pc")) is int and 0 <= d["shift_pc"] < 12
                            for d in row["draws"]), "journal_draw_identity")
                require(all(type(row.get(k)) in (float, int) and math.isfinite(row[k])
                            for k in ("loss", "lr", "seconds")), "journal_finite_scalars")
                require(row["seconds"] >= 0 and row["lr"] >= 0, "journal_scalar_range")
            except (ValueError, TypeError, KeyError) as exc:
                result.update(partial=True, issue=str(exc)); break
            result.update(rows=result["rows"]+1, last_update=row["update"], tail=row)
    result["draws_lower_bound"] = result["rows"] * 2
    return result


def validation_summaries(run_output):
    rows = []
    for path in sorted(Path(run_output).glob("validation-[0-9][0-9][0-9][0-9][0-9][0-9].json")):
        result = read_json(path)
        require(type(result.get("update")) is int and path.stem == f"validation-{result['update']:06d}",
                "validation_identity")
        row = {"update": result["update"]}
        for scope in ("common_support", "expanded_support"):
            score = result.get(scope, {}).get("corrected_primary_macro_score")
            require(type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1, "validation_score")
            row[scope + "_primary_score"] = score
        require(result.get("validation_records") == 162 and result.get("test_accessed") is False,
                "validation_population_or_test_flag")
        row["model_state_fingerprint"] = result.get("evaluated_model_state_fingerprint")
        rows.append(row)
    return rows


def inspect_result(config, receipt):
    output = Path(config["run_output"])
    report_path = output / config.get("report_filename", "report.json")
    report = read_json(report_path) if report_path.exists() else None
    mode = config["mode"]
    journal = read_training_journal(output / "training_metrics.jsonl", start_update=config["start_update"]) if mode == "train" else None
    validations = validation_summaries(output) if mode == "train" else []
    status = "aborted" if receipt["process_status"] == "aborted" else "failed"
    reason = receipt["reason"]
    accounting = {"applied_updates": None, "draws": None,
                  "applied_updates_lower_bound": journal["rows"] if journal is not None else None,
                  "draws_lower_bound": journal["draws_lower_bound"] if journal is not None else None,
                  "unlogged_work_possible": True}
    if mode == "train" and report is not None:
        matches = (journal is not None and not journal["partial"]
            and report.get("start_update") == config["start_update"]
            and report.get("applied_updates") == journal["rows"]
            and report.get("final_update") == journal["last_update"]
            and report.get("cumulative_draws") == 2*journal["last_update"])
        if matches:
            accounting.update(applied_updates=journal["rows"], draws=journal["draws_lower_bound"],
                              unlogged_work_possible=False)
        if report.get("completed") is True:
            require(matches and journal["last_update"] == config["expected_updates"], "completion_journal_mismatch")
            expected_validations = list(range((config["start_update"]//5000 + 1)*5000,
                                              config["expected_updates"]+1, 5000))
            observed = [v["update"] for v in validations if v["update"] > config["start_update"]]
            require(observed == expected_validations, "completion_validation_coverage")
            latest = read_json(output / "latest-checkpoint.json")
            require(latest.get("update") == config["expected_updates"], "completion_checkpoint_update")
            checkpoint = regular(latest["path"])
            require(checkpoint.parent == output and digest(checkpoint) ==
                    {k: latest[k] for k in ("sha256", "size_bytes")}, "completion_checkpoint_binding")
            require(report.get("test_accessed") is False, "completion_test_flag")
            if receipt["process_status"] == "exited_zero":
                status, reason = "completed", "complete_journal_validation_and_checkpoint_evidence"
        elif report.get("interrupted") is True and matches:
            status, reason = "aborted", "gracefully_interrupted_with_applied_update_accounting"
    elif report is not None and mode in ("gate", "memory"):
        if report.get("passed") is True and receipt["process_status"] == "exited_zero":
            status, reason = "completed", "bounded_gate_report_passed"
        declared = report.get("actual_optimizer_updates", report.get("applied_updates"))
        if type(declared) is int and declared >= 0:
            require(declared <= config["expected_updates"], "gate_update_budget_exceeded")
            accounting.update(applied_updates=declared, draws=report.get("actual_draws"),
                              unlogged_work_possible=False)
    test_flag = report.get("test_accessed") if isinstance(report, dict) else None
    validation_access = (True if bool(validations) or (output / "startup-data-verification.json").exists()
                         else report.get("validation_accessed") if isinstance(report, dict) else None)
    access = {"validation_accessed": validation_access,
              **{key: False if test_flag is False else None for key in (
                  "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed", "test_used_for_selection")}}
    return dict(status=status, reason=reason, accounting=accounting, validation_history=validations,
                best=report.get("best") if isinstance(report, dict) else None,
                report_present=report is not None, access=access,
                journal_integrity={k: v for k, v in journal.items() if k != "tail"} if journal else None,
                evidence_scope="producer journals, completion receipt and source-bound reports; no OS file-access tracing or quality-superiority claim")


def resource_violation(sample, initial_swap_kib, limits):
    if sample["gpu_memory_mib"] > limits["gpu_memory_mib"]:
        return "gpu_memory_ceiling"
    if sample["available_kib"] < limits["host_available_mib"] * 1024:
        return "host_memory_floor"
    if sample["swap_used_kib"] - initial_swap_kib > limits["swap_growth_mib"] * 1024:
        return "swap_growth"
    return None


def stop_gracefully(child, ownership, lifecycle, seconds):
    # Leader-only TERM lets its trainer finish a step/checkpoint while workers
    # remain available. The accepted cleanup kills only the owned group later.
    if ownership._observe_child_exit(child) is None:
        require(os.getpgid(child.pid) == child.pid, "owned_process_group_changed")
        os.kill(child.pid, signal.SIGTERM)
        lifecycle["trainer_sigterm_sent"] = True
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and ownership._observe_child_exit(child) is None:
            time.sleep(min(.25, max(0., deadline-time.monotonic())))
    return ownership._cleanup_owned_group(child, lifecycle)


def transition(record_path, expected_fingerprint, status, *, additions=None):
    current = load_record(record_path)
    require(current["status"] not in TERMINAL, "terminal_record_immutable")
    require(current["record_fingerprint"] == expected_fingerprint, "record_changed_concurrently")
    require(status in ({"running", *TERMINAL} if current["status"] == "planned" else TERMINAL), "record_transition")
    value = deepcopy(current)
    value["status"] = status
    if additions:
        for key, data in additions.items():
            value[key] = data
    value = validate_record(seal_record(value), filename=record_path)
    atomic_text(record_path, canonical_json(value)+"\n")
    return value


def retained_artifacts(config):
    root, prefix = Path(config["output_root"]), Path(config["artifact_locator_prefix"])
    artifacts = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "artifact_symlink_forbidden")
        if path.is_dir():
            continue
        if path.name.endswith(".lock") or path.name.endswith(".supervisor-tmp"):
            continue
        artifacts.append(dict(role="temporal_run_evidence", **digest(path), availability="external",
                              locator=(prefix/path.relative_to(root)).as_posix()))
    return artifacts


def refresh_project(project_root, record):
    root = Path(project_root)
    ledger = render_markdown_ledger(load_registry(root/"docs/experiments/records"))
    atomic_text(root/"docs/EXPERIMENT_LEDGER.md", ledger)
    status = root/"docs/STATUS.md"
    marker = f"<!-- temporal-supervisor:{record['experiment_id']}:{record['record_fingerprint']} -->"
    existing = status.read_text() if status.exists() else "# Status\n"
    if marker not in existing:
        note = (f"\n{marker}\n- {utc()}: `{record['experiment_id']}` — `{record['status']}`; "
                "terminal resource and experiment evidence recorded automatically.\n")
        atomic_text(status, existing+note)


def finalize(config, current_record, receipt):
    supervisor = Path(config["output_root"])/"supervisor"
    try:
        result = inspect_result(config, receipt)
    except Exception as exc:
        lower = None
        if config["mode"] == "train":
            try:
                lower = read_training_journal(Path(config["run_output"])/"training_metrics.jsonl",
                                              start_update=config["start_update"])
            except Exception:
                pass
        result = dict(status="invalid", reason=f"terminal_evidence_verification_failed:{type(exc).__name__}:{exc}",
            accounting={"applied_updates": None, "draws": None, "unlogged_work_possible": True,
                "applied_updates_lower_bound": lower["rows"] if lower else None,
                "draws_lower_bound": lower["draws_lower_bound"] if lower else None},
            validation_history=[], access={key: None for key in ("validation_accessed", "test_inputs_accessed",
                "test_targets_accessed", "test_metrics_computed", "test_used_for_selection")})
    write_json(supervisor/"terminal-summary.json", result)
    access = deepcopy(current_record["data_access"])
    access.update(result["access"])
    values = list(result["access"].values())
    access["access_evidence_status"] = ("verified_primary_evidence" if all(type(v) is bool for v in values)
        else "pending_primary_evidence" if all(v is None for v in values) else "partial_primary_evidence")
    access["evidence_scope"] = "Bounded producer report and durable validation files; not an OS access trace."
    results = dict(current_record["results"], terminal_supervision=result)
    budget = dict(current_record["budget"], actual=result["accounting"],
                  actual_wall_seconds=receipt["wall_seconds"])
    validity = dict(current_record["validity"], reason=result["reason"],
                    automated_terminal_evidence_checked=result["status"] == "completed")
    artifacts = list(current_record["artifacts"])
    identities = {(a["locator"], a["sha256"]) for a in artifacts}
    for artifact in retained_artifacts(config):
        if (artifact["locator"], artifact["sha256"]) not in identities:
            artifacts.append(artifact)
    record = transition(Path(config["record_path"]), current_record["record_fingerprint"], result["status"],
        additions=dict(data_access=access, results=results, budget=budget, validity=validity, artifacts=artifacts))
    refresh_project(config["project_root"], record)
    return record


def supervise(config, *, ownership=None):
    config = validate_config(config)
    ownership = ownership or load_ownership(Path(config["project_root"])/"tests/fixtures/research_supervision/accepted_context_wrapper.py")
    require(ownership.GPU_UUID.fullmatch(config.get("gpu_uuid", "")) is not None, "gpu_uuid")
    ownership._require_nonreaping_runtime()
    record_path = regular(config["record_path"])
    initial = load_record(record_path)
    require(initial["status"] == "planned", "planned_master_record_required")
    planned = load_record(regular(config["planned_record_input"]))
    require(planned == initial, "immutable_planned_input_mismatch")
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    out = root/"supervisor"
    out.mkdir(exist_ok=False)
    write_json(out/"launch-config.json", {k: v for k, v in config.items() if k != "env"} |
               {"environment_keys": sorted(config.get("env", {})), "environment_overrides_sha256": sha256(canonical_json(config.get("env", {})).encode()).hexdigest()})
    current, child, code = initial, None, None
    started, started_at = time.monotonic(), utc()
    deadline = started+config["limits"]["wall_seconds"]
    status, reason, detail = "failed", "supervisor_failed", None
    lifecycle = dict(ownership_lost=False, trainer_sigterm_sent=False,
        ownership_confirmed_before_cleanup=False, group_cleanup_attempted=False,
        group_signal_sent=False, group_already_absent=False, leader_reaped_after_cleanup=False)
    received = {"signal": None}
    handlers, samples, initial_swap = {}, [], None
    lock_path = record_path.with_suffix(".supervisor.lock")
    with lock_path.open("a") as lock, (out/"stdout.log").open("xb") as stdout, (out/"stderr.log").open("xb") as stderr:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda sig, _: received.update(signal=sig))
            def sample():
                nonlocal initial_swap
                value = dict(elapsed_seconds=time.monotonic()-started,
                    gpu_memory_mib=ownership.query_gpu_memory(config["gpu_uuid"], config.get("nvidia_smi", "nvidia-smi"), 2.),
                    **host_memory())
                if initial_swap is None:
                    initial_swap = value["swap_used_kib"]
                samples.append(value)
                with (out/"resources.jsonl").open("a") as handle:
                    handle.write(canonical_json(value)+"\n")
                violation = resource_violation(value, initial_swap, config["limits"])
                if violation:
                    raise ownership.ResourceStop(violation, "sampled external resource bound")
            sample()
            env = dict(os.environ, **config.get("env", {}))
            env.update(CUDA_VISIBLE_DEVICES=config["gpu_uuid"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
            child = subprocess.Popen(config["command"], cwd=config["cwd"], env=env,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            current = transition(record_path, initial["record_fingerprint"], "running")
            write_json(out/"started.json", dict(child_pid=child.pid, started_at=started_at,
                       planned_record_fingerprint=initial["record_fingerprint"], running_record_fingerprint=current["record_fingerprint"]))
            while True:
                if received["signal"] is not None:
                    raise ownership.ResourceStop("supervisor_signal", str(received["signal"]))
                if time.monotonic() >= deadline:
                    raise ownership.ResourceStop("wall_timeout", "36h or configured external deadline")
                code = ownership._observe_child_exit(child)
                if code is not None:
                    status, reason = ("exited_zero" if code == 0 else "failed"), "child_exit"
                    break
                sample()
                time.sleep(min(config["limits"]["poll_seconds"], max(0., deadline-time.monotonic())))
        except ownership.ResourceStop as exc:
            reason, detail = exc.reason, str(exc)
            lifecycle["ownership_lost"] = reason in {"child_ownership_lost", "child_ownership_context_unavailable"}
            status = "aborted" if reason in {"supervisor_signal", "wall_timeout", "gpu_memory_ceiling", "host_memory_floor", "swap_growth"} else "failed"
        except Exception as exc:
            status, reason, detail = "failed", "supervisor_exception", f"{type(exc).__name__}:{exc}"
        finally:
            if child is not None and not lifecycle["ownership_lost"]:
                try:
                    code = (ownership._cleanup_owned_group(child, lifecycle) if status == "exited_zero"
                            else stop_gracefully(child, ownership, lifecycle, config["limits"]["grace_seconds"]))
                except Exception as exc:
                    status, reason, detail = "failed", "cleanup_failed", f"{type(exc).__name__}:{exc}"
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            receipt = dict(schema_version=SCHEMA, process_status=status, reason=reason, detail=detail,
                started_at=started_at, ended_at=utc(), wall_seconds=time.monotonic()-started,
                child_returncode=code, child_pid=child.pid if child else None,
                resource_limits=config["limits"], resource_samples=len(samples), lifecycle=lifecycle,
                gpu_peak_mib=max((s["gpu_memory_mib"] for s in samples), default=None),
                host_available_min_kib=min((s["available_kib"] for s in samples), default=None),
                swap_growth_peak_kib=max((s["swap_used_kib"]-initial_swap for s in samples), default=None),
                supervisor_sha256=digest(Path(__file__))["sha256"],
                limitations="sampled whole-GPU/host bounds, not hard allocation limits; crash or SIGKILL requires external recovery")
            write_json(out/"resource-receipt.json", receipt)
            terminal = finalize(config, current, receipt)
    return 0 if terminal["status"] == "completed" else 1


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    return supervise(read_json(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
