"""Fresh-process B2 native-data memory/equality gate, never a training runner.

Consumes a completed, artifact-bound TRAIN index. Each case/strategy gets its
own spawn child, two applied updates and no input/output checkpoint. The parent
samples its own child's RSS; kernel high-water marks and CUDA allocator peaks
are separately labelled. Finite extrema are not a universal OOM proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import gc
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import re
import resource
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from music_critic.experiments.analysisgnn import performance_replay as existing
from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn.performance_data import FrozenProductionRecordLoader, PERFORMANCE_DATA_SCHEMA
from music_critic.experiments.analysisgnn.performance_evidence import (
    _artifact_rows, _git, _now, _transition, _validated_parent_ids, source_manifest,
)
from music_critic.experiments.analysisgnn.performance_measurement import raw_graph_counts
from music_critic.experiments.analysisgnn.performance_memory import (
    SizeIndexRow, TrainRecordMetadata, select_extreme_b2_cases,
)
from music_critic.experiments.analysisgnn.performance_training import (
    attempt_performance_update, prepare_performance_class_weights,
)
from music_critic.experiments.registry import DATA_ACCESS_FIELDS, canonical_json_bytes, load_record, render_registry


SCHEMA = "AnalysisGNNFreshProcessMemoryReplay@1.0.0"
LIMITS = (
    "Observed one-dimensional TRAIN extrema and their finite ordered B2 pairs do not exhaust all joint memory maxima.",
    "The package is original CPU preparation plus reference/optimized update; no graph cache, prefetch workers or IPC payloads.",
    "Parent sampled RSS peaks are lower bounds at the declared interval; fresh-child kernel high-water marks are separate.",
    "Kernel RSS high-water marks include import/setup/state hashing; phase-labelled samples do not isolate allocation ownership.",
    "CUDA free memory is whole-device state, while allocated/reserved/peak counters concern this child's Torch allocator.",
    "State fingerprints establish exact comparison of captured states, not independently reloadable checkpoint evidence.",
    "Children rehash common code/split identity and actual used pair payloads, not all TRAIN payloads on every child launch.",
    "The index producer executable manifest fingerprint is a sealed producer claim; this consumer authenticates four index artifacts, not an independent producer executable audit.",
    "No universal OOM safety, aggregate prefetch RAM readiness, throughput, C2 equivalence or music quality claim.",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("memory_replay." + reason)


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _write(path: Path, value, *, replace: bool = False) -> None:
    destination = path.with_suffix(path.suffix + ".partial") if replace else path
    with destination.open("xb") as stream:
        stream.write(canonical_json_bytes(_json(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        destination.replace(path)


def _append(path: Path, value) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(_json(value)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class MemoryIndexInputs:
    experiment_id: str
    records_root: Path
    size_index: Path
    bindings_initial: Path
    bindings_final: Path
    extreme_cases: Path


def _decode_row(value: dict) -> SizeIndexRow:
    metadata = dict(value["metadata"])
    metadata["source_versions"] = tuple(tuple(pair) for pair in metadata["source_versions"])
    row = SizeIndexRow(metadata=TrainRecordMetadata(**metadata),
        nodes_by_type=tuple(tuple(pair) for pair in value["nodes_by_type"]),
        edges_by_relation=tuple((tuple(pair[0]), pair[1]) for pair in value["edges_by_relation"]),
        graph_versions=tuple(tuple(pair) for pair in value["graph_versions"]),
        unique_cpu_tensor_storage_bytes=value["unique_cpu_tensor_storage_bytes"],
        accounted_decoded_entry_bytes=value["accounted_decoded_entry_bytes"],
        accounted_compact_entry_bytes=value["accounted_compact_entry_bytes"],
        cache_representation=value["cache_representation"])
    _require(_json(row.to_dict()) == value, "index_row_derived_fields_mismatch")
    return row


def load_completed_memory_index(inputs: MemoryIndexInputs, *, synthetic: bool = False) -> dict:
    """Authenticate exact artifacts and independently regenerate the finite cases."""
    _require(isinstance(inputs, MemoryIndexInputs) and type(synthetic) is bool, "invalid_index_inputs")
    _require(isinstance(inputs.experiment_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
             inputs.experiment_id) is not None, "invalid_index_id")
    record_path = Path(inputs.records_root) / f"{inputs.experiment_id}.json"
    _require(not record_path.is_symlink(), "regular_index_record_required")
    record = load_record(record_path)
    protocol = record.get("protocol", {})
    expected_mode = "synthetic" if synthetic else "production"
    _require(record["status"] == "completed" and record["validity"].get("valid") is True
             and record["kind"] == "data_audit" and protocol.get("mode") == expected_mode
             and protocol.get("split") == "train"
             and protocol.get("schema_version") == "AnalysisGNNPerformanceMemoryIndexRun@1.0.0",
             "completed_train_memory_index_required")
    paths = {"size-index.jsonl": inputs.size_index, "source-bindings-initial.json": inputs.bindings_initial,
             "source-bindings-final.json": inputs.bindings_final, "extreme-b2-cases.json": inputs.extreme_cases}
    payloads, hashes = {}, {}
    for role, supplied in paths.items():
        path = Path(supplied)
        _require(path.is_file() and not path.is_symlink(), "regular_index_artifact_required")
        payload = path.read_bytes()
        digest = sha256(payload).hexdigest()
        matches = [row for row in record["artifacts"] if row["role"] == role]
        _require(len(matches) == 1 and matches[0]["sha256"] == digest
                 and matches[0]["size_bytes"] == len(payload), "index_artifact_mismatch:" + role)
        payloads[role], hashes[role] = payload, digest
    initial = json.loads(payloads["source-bindings-initial.json"])
    _require(initial == json.loads(payloads["source-bindings-final.json"]), "index_source_binding_mismatch")
    rows = tuple(_decode_row(json.loads(line)) for line in payloads["size-index.jsonl"].splitlines() if line.strip())
    expected_count = protocol.get("expected_train_records")
    _require(type(expected_count) is int and expected_count > 0 and len(rows) == expected_count
             and (synthetic or expected_count == 1295), "complete_index_required")
    ids = initial["record_ids"]
    selection = select_extreme_b2_cases(rows, expected_train_ids=ids)
    _require(json.loads(payloads["extreme-b2-cases.json"]) == _json(asdict(selection)), "extreme_cases_mismatch")
    bindings = initial["bindings"]
    _require(bindings.get("schema") == PERFORMANCE_DATA_SCHEMA and set(bindings["record_bindings"]) == set(ids)
             and set(initial["components"]) == set(ids), "full_index_bindings_required")
    for row in rows:
        metadata = row.metadata
        _require(metadata.binding_seal == bindings["record_bindings"][row.record_id]
                 and metadata.source_component_id == initial["components"][row.record_id]
                 and metadata.source_seal == bindings["source_seal"] and metadata.code_seal == bindings["code_seal"]
                 and metadata.split_sha256 == bindings["split_sha256"], "row_source_binding_mismatch")
    results = record["results"]
    _require(results.get("supplied_index_fingerprint") == selection.supplied_index_fingerprint
             and results.get("indexed_records") == len(rows)
             and results.get("size_index_sha256") == hashes["size-index.jsonl"], "index_result_mismatch")
    _require(1 <= len(selection.selected_rows) <= 8 and 1 <= len(selection.ordered_b2_cases) <= 64,
             "finite_extreme_limit_exceeded")
    return {"index_experiment_id": inputs.experiment_id, "index_record_fingerprint": record["record_fingerprint"],
            "producer_identity": {"record_fingerprint": record["record_fingerprint"],
                "repository_source_manifest_fingerprint": record["repository"].get("source_manifest_fingerprint"),
                "scope": "sealed producer identity claim; producer executable artifacts not independently audited here"},
            "artifact_sha256": hashes, "selection": _json(asdict(selection)), "binding_manifest": bindings,
            "row_counts": {row.record_id: {"nodes": row.nodes, "edges": row.edges} for row in rows}}


def read_process_memory(pid: int) -> dict:
    """Current RSS from /proc or bounded ps; kernel VmHWM is separately returned."""
    _require(type(pid) is int and pid > 0, "invalid_monitored_pid")
    try:
        if sys.platform.startswith("linux"):
            fields = {}
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    name, value, unit = line.split()
                    _require(unit == "kB", "unexpected_proc_memory_units")
                    fields[name.rstrip(":")] = int(value) * 1024
            return {"rss_bytes": fields.get("VmRSS"), "kernel_hwm_bytes": fields.get("VmHWM"), "backend": "proc_status"}
        if sys.platform == "darwin":
            result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
                                    check=False, timeout=1)
            return {"rss_bytes": int(result.stdout.strip()) * 1024 if result.returncode == 0 and result.stdout.strip() else None,
                    "kernel_hwm_bytes": None, "backend": "ps_rss_kib"}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"rss_bytes": None, "kernel_hwm_bytes": None, "backend": "unavailable_or_exited"}
    return {"rss_bytes": None, "kernel_hwm_bytes": None, "backend": "unsupported_platform"}


def _resources(device: str) -> dict:
    current = read_process_memory(os.getpid())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    current.update(pid=os.getpid(), monotonic_ns=time.perf_counter_ns(),
                   kernel_ru_maxrss_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                   kernel_scope="fresh_child_process_lifetime", cuda=None)
    if device == "cuda":
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        current["cuda"] = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total,
            "peak_scope": "since_reset_before_native_preparation_after_initial_state_hash"}
    return current


def _state_fingerprints(model, optimizer, scheduler, scaler) -> dict:
    snapshot = existing._snapshot(model, optimizer, scheduler, scaler)
    result = {name: existing.recursive_state_fingerprint(value) for name, value in snapshot.items()}
    result["full_state"] = existing.recursive_state_fingerprint(snapshot)
    del snapshot
    gc.collect()
    return result


def _bind_pair(paths, full_manifest: dict, record_ids):
    loader = FrozenProductionRecordLoader(paths=paths, split="train")
    common = loader.validate_all_bindings()
    for key in ("schema", "source_seal", "code_seal", "split_sha256"):
        _require(common.get(key) == full_manifest.get(key), "common_source_identity_mismatch:" + key)
    _require(set(full_manifest["record_bindings"]) == set(loader._train_ids),
             "full_train_membership_mismatch")
    pair_ids = sorted(set(record_ids))
    _require(all(record_id in full_manifest["record_bindings"] for record_id in pair_ids), "unbound_pair_record")
    subset = {**full_manifest, "record_bindings": {record_id: full_manifest["record_bindings"][record_id] for record_id in pair_ids}}
    for record_id in pair_ids:
        _require(loader.check_binding(record_id, split="train") == subset["record_bindings"][record_id], "pair_binding_mismatch")
    _require(loader.validate_all_bindings(expected=subset) == subset, "pair_source_binding_mismatch")
    return loader, subset


def _child(directory: str, case: dict, strategy: str, device: str, paths, binding_manifest: dict,
           expected_counts: dict, synthetic_factory, expected_runner_sha: str) -> None:
    root = Path(directory)
    applied = 0
    stage = "startup"
    update = None

    def mark(name: str, number=None, *, snapshot=True):
        nonlocal stage, update
        stage, update = name, number
        _write(root / "phase.json", {"stage": name, "update": number, "pid": os.getpid()}, replace=True)
        if snapshot:
            _append(root / "resources.jsonl", {"stage": name, "update": number, **_resources(device)})

    try:
        _require(sha256(Path(__file__).read_bytes()).hexdigest() == expected_runner_sha, "runner_changed_before_child")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        ids = tuple(case["record_ids"])
        _require(len(ids) == 2 and strategy in ("baseline", "optimized"), "invalid_child_case")
        # Before any model/CUDA allocation, authenticate actual warm pair bytes.
        mark("source_prebinding", snapshot=False)
        loader = subset = source = None
        if synthetic_factory is None:
            loader, subset = _bind_pair(paths, binding_manifest, ids)
        else:
            source = synthetic_factory()

        def load(record_id):
            if source is not None:
                return source(record_id)
            loader.check_binding(record_id, split="train")
            def forbidden(*args, **kwargs):
                raise RuntimeError("memory_replay.cold_conversion_forbidden")
            with patch.object(existing.reference, "_resolve_selected_corpus_record", forbidden):
                result = existing.reference.load_production_record(record_id, split="train", paths=paths)
            loader.check_binding(record_id, split="train")
            return result

        mark("model_setup", snapshot=False)
        random.seed(17)
        np.random.seed(17)
        torch.random.default_generator.manual_seed(17)
        if device == "cuda":
            _require(torch.cuda.is_available(), "cuda_unavailable")
            torch.cuda.manual_seed_all(17)
        model = existing.CorrectedAnalysisGNNModel().to(device).train()
        random.seed(17001)
        np.random.seed(17001)
        torch.random.default_generator.manual_seed(17001)
        if device == "cuda":
            torch.cuda.manual_seed_all(17001)
        optimizer, scheduler = existing.build_orbit_matched_control_optimizer_scheduler(model)
        scaler, scaler_api = existing._scaler()
        weights = existing.reference.load_frozen_class_weights()
        class_weight_payload_fingerprint = weights.payload_fingerprint
        if strategy == "optimized":
            weights = prepare_performance_class_weights(weights, device)
        initial = _state_fingerprints(model, optimizer, scheduler, scaler)
        _write(root / "initial-state-fingerprints.json", initial)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mark("before_preparation")
        for number in (1, 2):
            mark("native_preparation", number, snapshot=False)
            loaded = [load(record_id) for record_id in ids]
            raw = existing.reference.combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
            sidecars = tuple(row[1] for row in loaded)
            raw = existing.reference.transpose_raw_graph_batch(raw, (0, 0))
            del loaded
            counts = raw_graph_counts(raw)
            _require(counts["nodes"] == expected_counts["nodes"] and counts["edges"] == expected_counts["edges"]
                     and counts["coarse_lengths"] == case["coarse_lengths"], "prepared_geometry_differs_from_index")
            mark("after_preparation", number)
            raw = existing.reference.move_raw_graph_batch(raw, device)
            mark("after_host_to_device", number)
            mark("update", number, snapshot=False)
            kwargs = dict(model=model, raw_graph_batch=raw, sidecars=sidecars, shifts=(0, 0), class_weights=weights,
                          optimizer=optimizer, scheduler=scheduler, scaler=scaler)
            outcome = (existing.reference.attempt_applied_update(**kwargs) if strategy == "baseline"
                       else attempt_performance_update(**kwargs))
            _require(outcome.applied and outcome.finite_logits and outcome.finite_gradients, "unapplied_or_nonfinite_update")
            applied += 1
            # Persist actual application before fallible telemetry or CPU state copies.
            _append(root / "applied-updates.jsonl", {"update": number, "applied": True, "pid": os.getpid()})
            mark("after_update", number)
            scalar = lambda value: None if value is None else float(value.detach().cpu())
            trace = {"update": number, "applied": True, "total": scalar(outcome.loss.total),
                "primary": scalar(outcome.loss.primary), "auxiliary": scalar(outcome.loss.auxiliary),
                "heads": {name: {"weighted_ce": scalar(value.weighted_ce), "unweighted_ce": scalar(value.unweighted_ce),
                    "valid_rows": value.valid_row_count, "masked_rows": value.masked_row_count,
                    "unsupported_rows": value.unsupported_row_count} for name, value in outcome.loss.heads.items()},
                "gradient_norm": outcome.gradient_norm, "scale_before": outcome.scaler_scale_before,
                "scale_after": outcome.scaler_scale_after, "record_ids": list(ids), "applied_shifts": [0, 0]}
            _append(root / "updates.jsonl", trace)
            del outcome, kwargs, raw, sidecars
            gc.collect()
            mark("state_hashing", number, snapshot=False)
            trace["state_fingerprints"] = _state_fingerprints(model, optimizer, scheduler, scaler)
            _append(root / "states.jsonl", trace)
        mark("source_finalization", snapshot=False)
        if loader is not None:
            _require(loader.validate_all_bindings(expected=subset) == subset, "source_changed_after_updates")
        mark("completed")
        _write(root / "result.json", {"status": "completed", "pid": os.getpid(), "strategy": strategy,
            "case": case, "applied_updates": applied, "initial": initial, "scaler_api": scaler_api,
            "class_weight_payload_fingerprint": class_weight_payload_fingerprint,
            "device": device, "source_scope": "common_identity_and_used_pair_payloads_before_after",
            "environment": {"python": sys.version, "torch": str(torch.__version__), "platform": platform.platform(),
                            "cuda_build": torch.version.cuda, "torch_threads": torch.get_num_threads(),
                            "torch_interop_threads": torch.get_num_interop_threads(),
                            "thread_environment": {name: os.environ.get(name) for name in (
                                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                            "cuda_device_name": torch.cuda.get_device_name() if device == "cuda" else None,
                            "cuda_device_total_bytes": torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None},
            "final_resources": _resources(device)})
    except BaseException as exc:
        _write(root / "failure.json", {"status": "failed", "stage": stage, "update": update,
            "applied_updates": applied, "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "pid": os.getpid()})
        raise


def _stop_owned(process) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=3)
    if process.is_alive():
        process.kill()
        process.join(timeout=3)


def run_monitored_child(*, directory: Path, args: tuple, timeout_seconds: float, sample_interval_seconds: float) -> dict:
    """Monitor only the child spawned here; timeout never signals another PID."""
    process = mp.get_context("spawn").Process(target=_child, args=(str(directory), *args))
    started = time.monotonic()
    samples = 0
    peak = 0
    kernel_peak = None
    phase_peaks = {}
    try:
        process.start()
        while process.is_alive():
            _require(time.monotonic() - started <= timeout_seconds, "owned_child_timeout")
            memory = read_process_memory(process.pid)
            try:
                phase = json.loads((directory / "phase.json").read_text())
            except (OSError, ValueError):
                phase = {"stage": "spawn_import", "update": None}
            row = {"monotonic_ns": time.perf_counter_ns(), "pid": process.pid, **memory,
                   "stage": phase["stage"], "update": phase.get("update")}
            _append(directory / "parent-rss-samples.jsonl", row)
            if memory["rss_bytes"] is not None:
                samples += 1
                peak = max(peak, memory["rss_bytes"])
                phase_peaks[phase["stage"]] = max(phase_peaks.get(phase["stage"], 0), memory["rss_bytes"])
            if memory["kernel_hwm_bytes"] is not None:
                kernel_peak = max(kernel_peak or 0, memory["kernel_hwm_bytes"])
            process.join(timeout=sample_interval_seconds)
        process.join()
        _require(process.exitcode == 0, f"owned_child_failed:exit={process.exitcode}")
        _require(samples > 0, "no_valid_parent_rss_samples")
        result = json.loads((directory / "result.json").read_text())
        _require(result["status"] == "completed" and result["applied_updates"] == 2, "child_incomplete")
        return result
    finally:
        if process.pid is not None:
            _stop_owned(process)
            _write(directory / "parent-monitor.json", {"pid": process.pid, "exitcode": process.exitcode,
                "rss_sample_count": samples, "sampled_rss_peak_bytes": peak if samples else None,
                "observed_kernel_hwm_bytes": kernel_peak, "phase_sampled_rss_peaks": phase_peaks,
                "native_preparation_sampled_rss_peak_bytes": phase_peaks.get("native_preparation"),
                "native_preparation_peak_scope": "parent samples matching native_preparation marker; null means no sample caught the phase",
                "sample_interval_seconds": sample_interval_seconds, "wall_seconds": time.monotonic() - started,
                "scope": "own_fresh_child_only; sampled_peak_lower_bound; phase_from_latest_child_marker"})


def compare_case(directory: Path) -> dict:
    results = {kind: json.loads((directory / kind / "result.json").read_text()) for kind in ("baseline", "optimized")}
    states = {kind: [json.loads(line) for line in (directory / kind / "states.jsonl").read_text().splitlines()]
              for kind in results}
    _require(all(result["status"] == "completed" and result["applied_updates"] == 2 for result in results.values()),
             "incomplete_case_cannot_pass")
    _require(all(results[kind].get("strategy") == kind for kind in results)
             and results["baseline"]["case"] == results["optimized"]["case"]
             and results["baseline"]["device"] == results["optimized"]["device"]
             and results["baseline"]["pid"] != results["optimized"]["pid"], "case_or_fresh_process_mismatch")
    _require(results["baseline"].get("class_weight_payload_fingerprint") is not None
             and results["baseline"].get("class_weight_payload_fingerprint") == results["optimized"].get("class_weight_payload_fingerprint"),
             "class_weight_payload_mismatch")
    _require(results["baseline"]["initial"] == results["optimized"]["initial"], "initial_state_mismatch")
    _require(len(states["baseline"]) == len(states["optimized"]) == 2
             and [row.get("update") for row in states["baseline"]] == [1, 2]
             and all(row.get("applied") is True and row.get("record_ids") == results["baseline"]["case"]["record_ids"]
                     and row.get("applied_shifts") == [0, 0] for row in states["baseline"])
             and states["baseline"] == states["optimized"], "loss_or_full_state_mismatch")
    return {"case": results["baseline"]["case"], "equivalence_passed": True,
            "initial_fingerprints": results["baseline"]["initial"],
            "after_update_fingerprints": [row["state_fingerprints"] for row in states["baseline"]],
            "state_trace_fingerprint": existing.recursive_state_fingerprint(states["baseline"])}


def _applied_accounting(output: Path) -> dict:
    """Do not guess whether a child interrupted inside optimizer.step applied it."""
    known = 0
    uncertain = []
    for directory in sorted(output.glob("case-*/*")):
        if not directory.is_dir():
            continue
        journal = directory / "applied-updates.jsonl"
        applied = []
        if journal.exists():
            for line in journal.read_text().splitlines():
                try:
                    row = json.loads(line)
                    _require(row.get("applied") is True and row.get("update") == len(applied) + 1,
                             "applied_journal_invalid")
                    applied.append(row["update"])
                except (ValueError, AttributeError):
                    uncertain.append(directory.relative_to(output).as_posix())
                    break
        known += len(applied)
        phase_path = directory / "phase.json"
        if phase_path.exists():
            try:
                phase = json.loads(phase_path.read_text())
                if phase["stage"] == "update" and phase.get("update") not in applied:
                    uncertain.append(directory.relative_to(output).as_posix())
            except (ValueError, KeyError, TypeError):
                uncertain.append(directory.relative_to(output).as_posix())
    return {"actual": None if uncertain else known, "known_applied_updates": known,
            "uncertain_applied_children": sorted(set(uncertain))}


def run_memory_replay(*, output_root: Path, index_inputs: MemoryIndexInputs, paths=None, device: str,
                      child_timeout_seconds: float, rss_sample_interval_seconds: float,
                      synthetic: bool = False, synthetic_source_factory=None) -> dict:
    """Execution core; the registered wrapper owns planning before this call."""
    _require(device in ("cpu", "cuda") and type(synthetic) is bool, "invalid_mode")
    _require((synthetic and device == "cpu" and paths is None and synthetic_source_factory is not None)
             or (not synthetic and device == "cuda" and isinstance(paths, ProductionArtifactPaths)
                 and synthetic_source_factory is None), "explicit_production_or_cpu_fixture_required")
    for value in (child_timeout_seconds, rss_sample_interval_seconds):
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0, "positive_timeout_interval_required")
    _require(rss_sample_interval_seconds <= child_timeout_seconds, "sample_interval_exceeds_timeout")
    root = Path(output_root)
    validated = load_completed_memory_index(index_inputs, synthetic=synthetic)
    _write(root / "validated-index.json", validated)
    cases = validated["selection"]["ordered_b2_cases"]
    _write(root / "resolved-case-plan.json", {"schema_version": SCHEMA, "case_count": len(cases),
        "child_count": 2 * len(cases), "exact_planned_applied_updates": 4 * len(cases),
        "ordered_b2_cases": cases, "case_fingerprint": existing.recursive_state_fingerprint(cases),
        "index_record_fingerprint": validated["index_record_fingerprint"],
        "scope": "Resolved from authenticated index before first child; registered budget remains a 256-update upper bound."})
    runner_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    comparisons = []
    for number, case in enumerate(validated["selection"]["ordered_b2_cases"]):
        directory = root / f"case-{number:03d}"
        directory.mkdir()
        expected = {name: sum(validated["row_counts"][record][name] for record in case["record_ids"])
                    for name in ("nodes", "edges")}
        for strategy in ("baseline", "optimized"):
            if device == "cuda":
                existing._assert_idle_gpu()
            child_dir = directory / strategy
            child_dir.mkdir()
            run_monitored_child(directory=child_dir, args=(case, strategy, device, paths,
                validated["binding_manifest"], expected, synthetic_source_factory, runner_sha),
                timeout_seconds=child_timeout_seconds, sample_interval_seconds=rss_sample_interval_seconds)
        comparison = compare_case(directory)
        _write(directory / "comparison.json", comparison)
        comparisons.append(comparison)
    _require(sha256(Path(__file__).read_bytes()).hexdigest() == runner_sha, "runner_changed_during_execution")
    return {"schema_version": SCHEMA, "equivalence_passed": True, "case_count": len(comparisons),
        "child_count": 2 * len(comparisons), "applied_updates": 4 * len(comparisons),
        "index_record_fingerprint": validated["index_record_fingerprint"], "input_artifact_sha256": validated["artifact_sha256"],
        "case_fingerprint": existing.recursive_state_fingerprint(validated["selection"]["ordered_b2_cases"]),
        "comparisons": comparisons, "limits": list(LIMITS), "device": device,
        "mode": "synthetic" if synthetic else "production"}


def run_registered_memory_replay(*, experiment_id: str, parent_experiment_id: str,
    output_root: Path, index_inputs: MemoryIndexInputs, paths=None, device: str,
    child_timeout_seconds: float, rss_sample_interval_seconds: float,
    repository_root: Path | None = None, synthetic: bool = False, synthetic_source_factory=None, command=None) -> dict:
    """Immutable engineering lifecycle, planned before index authentication/spawn."""
    _require(isinstance(experiment_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_id)
             is not None, "invalid_experiment_id")
    root = (repository_root or Path(__file__).resolve().parents[4]).resolve()
    output = Path(output_root).resolve()
    records = root / "docs/experiments/records"
    record_path = records / f"{experiment_id}.json"
    _require(not record_path.exists() and not record_path.is_symlink() and not output.exists(), "fresh_id_and_output_required")
    _require(output != root and not (root in output.parents and not ((root / "outputs") in output.parents
             or (root / "artifacts/experiments") in output.parents)), "ignored_or_external_output_required")
    parents = _validated_parent_ids(records, experiment_id, parent_experiment_id)
    _require(bool(parents), "terminal_parent_required")
    manifest = source_manifest(root)
    config = {"schema_version": SCHEMA, "mode": "synthetic" if synthetic else "production", "device": device,
        "index_experiment_id": index_inputs.experiment_id, "batch_size": 2, "updates_per_child": 2,
        "child_timeout_seconds": child_timeout_seconds, "rss_sample_interval_seconds": rss_sample_interval_seconds,
        "index_inputs": {field.name: str(getattr(index_inputs, field.name)) for field in fields(index_inputs)},
        "paths": None if paths is None else {field.name: str(getattr(paths, field.name)) for field in fields(paths)}}
    protocol = {key: value for key, value in config.items() if key not in ("index_inputs", "paths")}
    protocol["config_fingerprint"] = sha256(canonical_json_bytes(config)).hexdigest()
    policy = {key: False for key in ("validation_allowed", "test_inputs_allowed", "test_targets_allowed",
                                   "test_metrics_allowed", "test_selection_allowed")}
    policy["source"] = "docs/PERFORMANCE_REPLAY.md; fresh-process TRAIN B2 memory gate"
    record = {"schema_version": "1.0.0", "experiment_id": experiment_id, "parent_ids": parents,
        "kind": "engineering_gate", "phase": "9E-P1", "status": "planned", "title": "Fresh-process B2 memory and update equality gate",
        "question": "Do two applied updates agree on finite extremal TRAIN pairs, and what memory does each fresh process use?",
        "hypothesis": "Original preparation and optimized update preserve exact state on observed size extrema.",
        "decision_use": "Finite native-data B2 package memory/equality evidence, not global prefetch or OOM acceptance.",
        "repository": {"git_commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(_git(root, "status", "--porcelain")),
            "diff_sha256": sha256(_git(root, "diff", "--binary", "HEAD").encode()).hexdigest(),
            "source_manifest_fingerprint": manifest["fingerprint"]},
        "seeds": [17, 17001], "protocol": protocol,
        "budget": {"planned": 256, "planning_scope": "upper_bound_all_verified_extreme_pairs",
                   "actual": None, "unit": "applied_updates_in_fresh_B2_children"},
        "data_access": {**{key: None for key in DATA_ACCESS_FIELDS}, "declared_policy": policy,
                        "access_evidence_status": "pending_primary_evidence"},
        "results": {}, "validity": {"valid": None}, "claims": {"supported": [], "not_supported": [
            "universal OOM safety", "prefetch aggregate memory", "B4/B8 readiness", "C2 equivalence", "checkpoint integrity"],
            "caveats": list(LIMITS)}, "artifacts": [], "import_provenance": {"mode": "local_synthetic" if synthetic else "local_execution"},
        "created_at": _now(), "started_at": None}
    lifecycle = [_transition(record_path, record)]
    started, owned, error, result = time.perf_counter(), False, None, {}
    try:
        output.mkdir(parents=True, exist_ok=False)
        owned = True
        record.update(status="running", started_at=_now())
        lifecycle.append(_transition(record_path, record))
        for name, value in (("lifecycle-start.json", lifecycle), ("resolved-config.json", config), ("source-manifest-initial.json", manifest)):
            _write(output / name, value)
        result = run_memory_replay(output_root=output, index_inputs=index_inputs, paths=paths, device=device,
            child_timeout_seconds=child_timeout_seconds, rss_sample_interval_seconds=rss_sample_interval_seconds,
            synthetic=synthetic, synthetic_source_factory=synthetic_source_factory)
        final_manifest = source_manifest(root)
        _write(output / "source-manifest-final.json", final_manifest)
        _require(final_manifest == manifest, "source_changed_during_execution")
        _write(output / "summary.json", result)
        record.update(status="completed", results=result, validity={"valid": True})
        record["budget"].update(actual=result["applied_updates"])
        record["claims"]["supported"] = ["Two exact applied updates per strategy on all declared finite B2 cases, with measured child memory."]
    except BaseException as exc:
        error = exc
        record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      validity={"valid": False, "reason": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if owned:
                accounting = _applied_accounting(output)
                record["budget"].update(accounting)
                if record["status"] == "completed":
                    _require(accounting["actual"] == result["applied_updates"], "completed_accounting_mismatch")
                _write(output / "execution-evidence.json", {"lifecycle": lifecycle, "protocol": protocol,
                    "argv": list(sys.argv if command is None else command), "failure": None if error is None else repr(error),
                    "access_observation_limits": "Authenticated TRAIN index and guarded pair readers; no independent OS I/O trace."})
                record["artifacts"] = _artifact_rows(root, output, experiment_id)
        except BaseException as exc:
            record.update(status="aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          validity={"valid": False, "reason": f"evidence_finalization_failed:{type(exc).__name__}: {exc}"})
            record["artifacts"], record["claims"]["supported"] = [], []
            record["results"] = {}
            if error is None:
                error = exc
        record["ended_at"] = _now()
        record["budget"]["wall_seconds"] = time.perf_counter() - started
        _transition(record_path, record)
        (root / "docs/EXPERIMENT_LEDGER.md").write_text(render_registry(records), encoding="utf-8")
    if error is not None:
        raise error
    return result
