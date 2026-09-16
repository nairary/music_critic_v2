"""Lifecycle and immutable evidence for the opt-in performance replay."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable

from music_critic.experiments.registry import (
    TERMINAL_NON_COMPLETED_STATUSES,
    canonical_json_bytes, file_sha256, load_record, render_registry, seal_record,
    validate_record, write_record_once,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError("performance.repository_git_unavailable")
    return result.stdout.strip()


def source_manifest(root: Path) -> dict[str, object]:
    """Hash executable source, including opt-in files not yet committed."""
    selected = [root / "pyproject.toml"]
    for directory in ("src", "scripts"):
        selected.extend(sorted((root / directory).rglob("*.py")))
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in selected if path.is_file()
    }
    return {"files": files, "fingerprint": sha256(canonical_json_bytes(files)).hexdigest()}


def _transition(path: Path, record: dict[str, object]) -> dict[str, object]:
    sealed = seal_record(record)
    validate_record(sealed, filename=path.name)
    if not path.exists():
        write_record_once(path, sealed)
        return json.loads(canonical_json_bytes(sealed))
    old = load_record(path)
    allowed = {"planned": {"running", "failed", "aborted", "invalid"},
               "running": {"completed", "failed", "aborted", "invalid"}}
    if sealed["status"] not in allowed.get(old["status"], set()):
        raise ValueError("performance.immutable_terminal_record")
    temporary = path.with_suffix(".json.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json_bytes(sealed).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return json.loads(canonical_json_bytes(sealed))


def _artifact_rows(root: Path, output: Path, experiment_id: str) -> list[dict[str, object]]:
    artifacts = []
    if not output.is_dir():
        return artifacts
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("performance.evidence_symlink_forbidden")
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        try:
            locator = path.relative_to(root).as_posix()
            availability = "local_ignored"
        except ValueError:
            locator = f"external:{experiment_id}/{relative}"
            availability = "external_evidence_root"
        artifacts.append({"role": relative, "sha256": file_sha256(path),
                          "size_bytes": path.stat().st_size,
                          "availability": availability, "locator": locator})
    return artifacts


def _validated_parent_ids(
    records: Path, experiment_id: str, parent_experiment_id: str | None,
) -> list[str]:
    if parent_experiment_id is None:
        return []
    # Same identifier grammar as the registry; reject paths before any read.
    if not isinstance(parent_experiment_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", parent_experiment_id
    ) is None:
        raise ValueError("performance.parent_experiment_id_invalid")
    if parent_experiment_id == experiment_id:
        raise ValueError("performance.parent_experiment_self_reference")
    path = records / f"{parent_experiment_id}.json"
    if path.is_symlink():
        raise ValueError("performance.parent_record_invalid")
    if not path.is_file():
        raise ValueError("performance.parent_record_missing")
    try:
        parent = load_record(path)
    except ValueError as exc:
        raise ValueError("performance.parent_record_invalid") from exc
    if parent["status"] not in TERMINAL_NON_COMPLETED_STATUSES | {"completed"}:
        raise ValueError("performance.parent_record_not_terminal")
    return [parent_experiment_id]


_PREFETCH_BINDINGS = (
    "comparison_kind", "planned_arm_count", "worker_counts",
    "worker_configs_fingerprint", "schedule_descriptor_fingerprint",
    "reservation_manifest_fingerprint",
)
_CACHE_BINDINGS = ("preparation_strategy", "compact_cache_global_max_bytes", "cache_policies",
                   "cache_policies_fingerprint", "cache_start_policy", "cache_budget_scope")
_ARM_STRATEGY_BINDINGS = ("include_original_baseline", "arm_strategies")


def _validate_protocol(protocol: dict[str, object]) -> None:
    """Reject malformed new declarations before registering or executing work."""
    for key in ("question", "hypothesis", "decision_use"):
        if key in protocol and (type(protocol[key]) is not str or not protocol[key].strip()):
            raise ValueError(f"performance.protocol_text_invalid:{key}")
    if "planned_arm_count" in protocol:
        for key in ("planned_arm_count", "repeats"):
            if type(protocol.get(key)) is not int or protocol[key] <= 0:
                raise ValueError(f"performance.protocol_positive_integer_required:{key}")
    if protocol.get("comparison_kind") not in ("ordered_prefetch", "ordered_prefetch_with_original"):
        return
    for key in _PREFETCH_BINDINGS:
        if key not in protocol:
            raise ValueError(f"performance.protocol_binding_missing:{key}")
    counts = protocol["worker_counts"]
    if (type(counts) is not list or not counts
        or any(type(count) is not int or count not in (0, 1, 2, 4) for count in counts)
        or len(set(counts)) != len(counts) or 0 not in counts):
        raise ValueError("performance.protocol_worker_counts_invalid")
    original = protocol["comparison_kind"] == "ordered_prefetch_with_original"
    if protocol["planned_arm_count"] != len(counts) + int(original):
        raise ValueError("performance.protocol_arm_count_mismatch")
    if any(key in protocol for key in _CACHE_BINDINGS):
        from music_critic.experiments.analysisgnn.performance_prefetch_replay import cache_policies
        from music_critic.experiments.analysisgnn.contracts import fingerprint
        if any(key not in protocol for key in _CACHE_BINDINGS) or protocol["preparation_strategy"] != "compact_cache":
            raise ValueError("performance.protocol_compact_cache_binding_incomplete")
        policies = cache_policies(protocol["preparation_strategy"], protocol["compact_cache_global_max_bytes"], counts)
        if (protocol["cache_policies"] != policies or protocol["cache_policies_fingerprint"] != fingerprint(policies)
                or protocol["cache_start_policy"] != "cold_process_local_per_arm_no_prepared_population_prewarm"
                or protocol["cache_budget_scope"] != "retained_entries_only_partitioned_across_preparing_processes_not_RSS_or_IPC"):
            raise ValueError("performance.protocol_compact_cache_binding_mismatch")
    if original:
        expected = [{"arm_key": f"integrated-workers{n}", "worker_count": n,
                     "preparation_strategy": "compact_cache", "update_strategy": "optimized"} for n in counts] + [
                    {"arm_key": "original-workers0", "worker_count": 0,
                     "preparation_strategy": "frozen", "update_strategy": "reference"}]
        if (protocol.get("include_original_baseline") is not True or protocol.get("arm_strategies") != expected
                or protocol.get("preparation_strategy") != "compact_cache"
                or protocol.get("update_strategy") != "mixed_reference_and_optimized"):
            raise ValueError("performance.protocol_original_arm_binding_mismatch")
    elif any(key in protocol for key in _ARM_STRATEGY_BINDINGS):
        raise ValueError("performance.protocol_original_arm_requires_distinct_comparison")
    for key in _PREFETCH_BINDINGS[3:]:
        if type(protocol[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", protocol[key]) is None:
            raise ValueError(f"performance.protocol_fingerprint_invalid:{key}")
    for key, minimum in (("measured_draws", 2), ("warmup_draws", 0)):
        if type(protocol.get(key)) is not int or protocol[key] < minimum or protocol[key] % 2:
            raise ValueError(f"performance.protocol_b2_draw_count_invalid:{key}")
    if ("batch_size" in protocol and (type(protocol["batch_size"]) is not int or protocol["batch_size"] != 2)
        or "batch_sizes" in protocol and canonical_json_bytes(protocol["batch_sizes"]) != b"[2]"):
        raise ValueError("performance.protocol_fixed_b2_required")


def _validate_declared_arm_count(protocol: dict[str, object], result: dict[str, object]) -> None:
    if "planned_arm_count" not in protocol:
        return
    arms = result.get("arms")
    if not isinstance(arms, list) or len(arms) != protocol["planned_arm_count"] * protocol["repeats"]:
        raise ValueError("performance.result_arm_count_mismatch")
    if protocol.get("comparison_kind") not in ("ordered_prefetch", "ordered_prefetch_with_original"):
        return
    original = protocol["comparison_kind"] == "ordered_prefetch_with_original"
    strategies = {row["arm_key"]: row for row in protocol["arm_strategies"]} if original else {}
    expected = {(repeat, key) for repeat in range(protocol["repeats"])
                for key in (strategies if original else protocol["worker_counts"])}
    observed = set()
    for arm in arms:
        if (not isinstance(arm, dict) or type(arm.get("repeat")) is not int
            or type(arm.get("worker_count")) is not int):
            raise ValueError("performance.result_worker_arm_identity_invalid")
        identity = arm["repeat"], arm.get("arm_key") if original else arm["worker_count"]
        if identity not in expected or identity in observed:
            raise ValueError("performance.result_worker_arm_coverage_mismatch")
        observed.add(identity)
        if type(arm.get("batch_size")) is not int or arm["batch_size"] != 2:
            raise ValueError("performance.result_fixed_b2_required")
        wanted_kind = ("original_baseline" if original and arm["arm_key"] == "original-workers0" else
                       "baseline" if arm["worker_count"] == 0 else "optimized")
        if arm.get("kind") != wanted_kind:
            raise ValueError("performance.result_worker_arm_kind_mismatch")
        if original:
            strategy = strategies[arm["arm_key"]]
            if (arm["worker_count"] != strategy["worker_count"] or arm.get("data_strategy") != strategy["preparation_strategy"]
                    or arm.get("update_strategy") != strategy["update_strategy"]):
                raise ValueError("performance.result_original_arm_strategy_mismatch")
        window = arm.get("window")
        if (not isinstance(window, dict) or type(window.get("records")) is not int
            or window["records"] != protocol["measured_draws"]):
            raise ValueError("performance.result_worker_arm_draw_count_mismatch")
    if observed != expected:
        raise ValueError("performance.result_worker_arm_coverage_mismatch")


def run_registered_replay(
    *, experiment_id: str, output_root: Path, protocol: dict[str, object],
    execute: Callable[[], dict[str, object]], repository_root: Path | None = None,
    parent_experiment_id: str | None = None,
) -> dict[str, object]:
    """Plan before executing; retain all failures, never reuse an attempt ID.

    ``execute`` creates a fresh output directory. Source changes during the run
    invalidate evidence. This wrapper does not assert GPU/quality observations.
    An optional parent must be a sealed terminal record before planning starts;
    failed/aborted/invalid attempts are valid parents for a new retry ID.
    Data strategies declared in the planned protocol require matching result
    fields; older protocols without those declarations remain supported.
    Ordered-prefetch comparisons additionally bind every worker arm/repeat and
    its measured B2 budget; no arm may substitute for a missing reference.
    """
    root = (repository_root or Path(__file__).resolve().parents[4]).resolve()
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    if not isinstance(protocol, dict):
        raise ValueError("performance.protocol_object_required")
    _validate_protocol(protocol)
    output = Path(output_root).resolve()
    records = root / "docs/experiments/records"
    record_path = records / f"{experiment_id}.json"
    parent_ids = _validated_parent_ids(records, experiment_id, parent_experiment_id)
    if record_path.exists() or output.exists():
        raise ValueError("performance.fresh_experiment_and_output_required")
    if output == root or root in output.parents and not (
        (root / "outputs") in output.parents or (root / "artifacts/experiments") in output.parents
    ):
        raise ValueError("performance.output_must_be_ignored_or_external")
    manifest = source_manifest(root)
    source_state = {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(_git(root, "status", "--porcelain")),
        "diff_sha256": sha256(_git(root, "diff", "--binary", "HEAD").encode()).hexdigest(),
        "source_manifest_fingerprint": manifest["fingerprint"],
    }
    policy = {field: False for field in (
        "validation_allowed", "test_inputs_allowed", "test_targets_allowed",
        "test_metrics_allowed", "test_selection_allowed",
    )}
    policy["source"] = "docs/PERFORMANCE_REPLAY.md"
    access = {field: None for field in (
        "validation_accessed", "test_inputs_accessed", "test_targets_accessed",
        "test_metrics_computed", "test_used_for_selection",
    )}
    planned_draws = None
    if "measured_draws" in protocol and "repeats" in protocol:
        if "planned_arm_count" in protocol:
            planned_draws = protocol["measured_draws"] * protocol["planned_arm_count"] * protocol["repeats"]
        elif "batch_sizes" in protocol:
            planned_draws = protocol["measured_draws"] * (1 + len(protocol["batch_sizes"])) * protocol["repeats"]
    record: dict[str, object] = {
        "schema_version": "1.0.0", "experiment_id": experiment_id,
        "parent_ids": parent_ids,
        "kind": "engineering_gate", "phase": "9E-P1", "status": "planned",
        "title": f"C0 isolated performance replay ({protocol.get('mode', 'unknown')})",
        "question": protocol.get("question", "Can preparation and synchronization overhead be reduced with exact B2 training parity?"),
        "hypothesis": protocol.get("hypothesis", "Sealed graph reuse and CPU candidate metadata improve throughput without changing B2 states."),
        "decision_use": protocol.get("decision_use", "Engineering throughput and equivalence only; no model-quality selection."),
        "repository": source_state, "seeds": [17], "protocol": protocol,
        "budget": {"planned": planned_draws,
                   "planned_per_arm": protocol.get("measured_draws"), "actual": None,
                   "unit": "measured_record_draws_across_all_arms", "repeats": protocol.get("repeats")},
        "data_access": {**access, "declared_policy": policy,
                        "access_evidence_status": "pending_primary_evidence"},
        "results": {}, "validity": {"valid": None},
        "claims": {"supported": [], "not_supported": ["musical quality", "C2 augmentation benefit", "TEST performance"],
                   "caveats": ["Fixture measurements do not establish production GPU speed or memory capacity."]},
        "artifacts": [], "import_provenance": {"mode": "local_execution"},
        "started_at": None, "created_at": _now(),
    }
    lifecycle = [_transition(record_path, record)]
    record.update(status="running", started_at=_now())
    lifecycle.append(_transition(record_path, record))
    started = time.perf_counter()
    failure: BaseException | None = None
    result: dict[str, object] = {}
    try:
        # Match the persisted JSON representation (including integer mapping
        # keys) and reject NaN/non-serializable results inside the failure path.
        returned = json.loads(json.dumps(execute(), allow_nan=False))
        if not isinstance(returned, dict):
            raise ValueError("performance.result_object_required")
        result = returned
        for key, expected in {
            "experiment_id": experiment_id, "mode": protocol.get("mode"),
            "device": protocol.get("device"), "repeats": protocol.get("repeats"),
            "measured_draws_per_arm": protocol.get("measured_draws"),
            "warmup_draws_per_arm": protocol.get("warmup_draws"),
        }.items():
            if key in result and expected is not None and result[key] != expected:
                raise ValueError(f"performance.result_binding_mismatch:{key}")
        for key in ("reference_data_strategy", "optimized_data_strategy", "update_strategy", *_PREFETCH_BINDINGS, *_CACHE_BINDINGS, *_ARM_STRATEGY_BINDINGS):
            if key not in protocol:
                continue
            if key not in result:
                raise ValueError(f"performance.result_binding_missing:{key}")
            if canonical_json_bytes(result[key]) != canonical_json_bytes(protocol[key]):
                raise ValueError(f"performance.result_binding_mismatch:{key}")
        _validate_declared_arm_count(protocol, result)
        if "arms" in result:
            arms = result["arms"]
            if not isinstance(arms, list) or any(
                not isinstance(arm, dict) or not isinstance(arm.get("window"), dict)
                or type(arm["window"].get("records")) is not int
                or arm["window"]["records"] < 0 for arm in arms
            ):
                raise ValueError("performance.result_arm_accounting_invalid")
            record["budget"]["actual"] = sum(arm["window"]["records"] for arm in arms)
        if result.get("equivalence_passed") is not True:
            record.update(status="invalid", validity={"valid": False, "reason": "exact_b2_equivalence_not_proven"})
        elif source_manifest(root)["fingerprint"] != manifest["fingerprint"]:
            record.update(status="invalid", validity={"valid": False, "reason": "source_changed_during_execution"})
        else:
            record.update(status="completed", validity={"valid": True})
        record["results"] = result
        observations = result.get("data_access")
        if isinstance(observations, dict) and all(type(observations.get(key)) is bool for key in access):
            if any(observations[key] for key in access):
                raise ValueError("performance.reported_forbidden_data_access")
            record["data_access"] = {**observations, "declared_policy": policy,
                                     "access_evidence_status": "verified_primary_evidence"}
        validate_record(seal_record(record), filename=record_path.name)
    except BaseException as exc:
        failure = exc
        record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      validity={"valid": False, "reason": f"{type(exc).__name__}: {exc}"})
        # A malformed result cannot prevent its own failure from being sealed.
        try:
            validate_record(seal_record(record), filename=record_path.name)
        except ValueError:
            record["results"] = {"result_metadata_rejected": True}
            record["data_access"] = {**access, "declared_policy": policy,
                                     "access_evidence_status": "pending_primary_evidence"}
    finally:
        evidence = {"source_manifest": manifest, "repository": source_state,
                    "parent_ids": parent_ids,
                    "argv": sys.argv, "python_executable": sys.executable,
                    "protocol": protocol, "lifecycle": lifecycle,
                    "failure": None if failure is None else repr(failure)}
        try:
            output.mkdir(parents=True, exist_ok=True)
            with (output / "execution-evidence.json").open("x", encoding="utf-8") as stream:
                json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            record["artifacts"] = _artifact_rows(root, output, experiment_id)
        except BaseException as exc:
            # Evidence failure must not strand an otherwise terminal attempt
            # as 'running'. Never overwrite a pre-existing evidence file.
            record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed", validity={"valid": False,
                          "reason": f"evidence_finalization_failed:{type(exc).__name__}: {exc}"})
            record["artifacts"] = []
            if failure is None:
                failure = exc
        record["ended_at"] = _now()
        record["budget"]["wall_seconds"] = time.perf_counter() - started
        sealed = _transition(record_path, record)
        ledger = root / "docs/EXPERIMENT_LEDGER.md"
        ledger.write_text(render_registry(records), encoding="utf-8")
    if failure is not None:
        raise failure
    if sealed["status"] != "completed":
        raise ValueError(f"performance.experiment_invalid:{sealed['validity']['reason']}")
    return result
