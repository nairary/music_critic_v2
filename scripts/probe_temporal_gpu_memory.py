"""Bounded TRAIN-only worst-size temporal CUDA forward/backward admission.

No checkpoint load, optimizer update, validation or TEST payload access. The
parent registers this experiment and supplies a completed whole-TRAIN sizing
record. A separate process guard bounds host RSS and elapsed time.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from music_critic.experiments.registry import load_record
from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.frozen_training_data import FrozenTrainingData
from music_critic.experiments.analysisgnn.performance_training import prepare_performance_class_weights, performance_gradient_diagnostics
from music_critic.experiments.analysisgnn.temporal_slices.campaign import components, scientific_config
from music_critic.experiments.analysisgnn.temporal_slices.cache import (
    read_temporal_record, collate_temporal_graphs, transpose_temporal_historical,
)
from music_critic.experiments.analysisgnn.temporal_slices.alignment import align_temporal_targets, temporal_supervised_loss
from scripts.size_temporal_graphs import dimensions


SCHEMA = "TemporalWorstSizeGPUAdmission@1.0.0"
GPU_LIMIT_BYTES = 16 * 1024**3
MAX_FORWARDS = 6
DESKTOP_EXECUTABLES = frozenset({
    "/usr/libexec/gnome-remote-desktop-daemon",
    "/usr/bin/nautilus",
    "/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration",
})
DESKTOP_MEMORY_LIMIT_MIB = 600


def require(value, reason):
    if not value:
        raise ValueError("temporal_gpu_probe." + reason)


def assert_probe_gpu_idle():
    """Permit only this probe and the three explicitly observed desktop apps."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("temporal_gpu_probe.gpu_process_listing_unavailable") from exc
    observations, seen = [], set()
    for line in listing.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        require(len(fields) == 3 and fields[0].isdigit() and fields[2].isdigit(), "gpu_process_listing_invalid")
        pid, reported, memory_mib = int(fields[0]), fields[1], int(fields[2])
        require(pid > 0 and pid not in seen, "gpu_process_pid_invalid")
        seen.add(pid)
        is_self = pid == os.getpid()
        require(is_self or reported in DESKTOP_EXECUTABLES, "other_gpu_compute_process_active")
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
        except OSError as exc:
            raise ValueError("temporal_gpu_probe.gpu_process_identity_unverified") from exc
        require(Path(reported).is_absolute() and executable == reported, "gpu_process_executable_mismatch")
        require(is_self or memory_mib <= DESKTOP_MEMORY_LIMIT_MIB, "desktop_gpu_memory_exceeded")
        observations.append({"pid": pid, "process_name": reported, "proc_exe": executable,
                             "used_memory_mib": memory_mib, "allowed_as": "self" if is_self else "desktop"})
    return observations


def bound_json(path, expected_sha256):
    content = Path(path).read_bytes()
    require(sha256(content).hexdigest() == expected_sha256, "artifact_hash")
    return json.loads(content)


def verify_sources(config):
    manifest = bound_json(config["source_manifest"], config["source_manifest_sha256"])
    root = Path(config["source_root"]).resolve()
    paths = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "source_path")
        actual = (root / relative).resolve()
        require(actual.is_relative_to(root) and actual not in paths, "source_path_escape_or_duplicate")
        paths.add(actual)
        require(sha256(actual.read_bytes()).hexdigest() == row["sha256"], "source_changed")
    require(Path(__file__).resolve() in paths, "probe_source_not_bound")
    return len(paths)


def select_probe(sizing, manifest):
    """Top-two union, duplicate B2 per work, then largest pair at shifts 0/6."""
    require(type(sizing) is dict and sizing.get("schema") == "TemporalFullTrainSizing@1.0.0"
            and sizing.get("completed") is True and sizing.get("records") == 1295,
            "completed_whole_train_sizing_required")
    require(sizing.get("validation_payload_accessed") is False and sizing.get("test_payload_accessed") is False,
            "sizing_access_boundary")
    # Structural rejection precedes even FrozenTrainingData metadata access.
    rows = manifest.get("records", ())
    require(all(r.get("split") in ("train", "validation") for r in rows), "test_declaration_forbidden")
    by_id = {r["record_id"]: r for r in rows}
    require(len(by_id) == len(rows), "duplicate_manifest_identity")
    selected = {}
    top_bytes = None
    for field, key in (("top_by_tensor_bytes", "total_tensor_bytes"), ("top_by_slice_nodes", "slice_nodes")):
        ranking = sizing.get(field)
        require(type(ranking) is list and len(ranking) >= 2, "two_ranked_works_required")
        require(len({r["record_id"] for r in ranking}) == len(ranking), "duplicate_sizing_identity")
        require(all(type(r.get(key)) is int and r[key] > 0 for r in ranking), "positive_sizing_counts")
        require(all(a[key] >= b[key] for a, b in zip(ranking, ranking[1:])), "sizing_rank_order")
        if field == "top_by_tensor_bytes":
            top_bytes = tuple(r["record_id"] for r in ranking[:2])
        for row in ranking[:2]:
            rid = row["record_id"]
            source = by_id.get(rid)
            require(source is not None and source["split"] == "train", "selected_work_not_train")
            require(row["piece_sha256"] == source["piece"]["sha256"]
                    and row["component_id"] == source["source_component_id"], "sizing_data_binding")
            if rid in selected:
                require(selected[rid] == row, "conflicting_sizing_rows")
            selected[rid] = row
    jobs = [{"kind": "duplicate_largest", "record_ids": [rid, rid], "shifts": [0, 0]}
            for rid in sorted(selected)]
    jobs += [{"kind": "largest_pair_identity", "record_ids": list(top_bytes), "shifts": [0, 0]},
             {"kind": "largest_pair_shift6", "record_ids": list(top_bytes), "shifts": [6, 6]}]
    require(4 <= len(jobs) <= MAX_FORWARDS, "forward_budget")
    return selected, jobs


def check_configuration(config):
    require(type(config.get("max_forwards")) is int and config["max_forwards"] == MAX_FORWARDS, "max_forwards")
    require(type(config.get("gpu_limit_bytes")) is int and config["gpu_limit_bytes"] == GPU_LIMIT_BYTES, "gpu_ceiling")
    record = load_record(Path(config["record"]))
    require(record["status"] == "planned" and record["experiment_id"] == config["experiment_id"], "planned_record")
    require(record["budget"]["planned"] == MAX_FORWARDS, "planned_forward_budget")
    source_count = verify_sources(config)
    manifest = bound_json(config["data_manifest"], config["data_manifest_sha256"])
    sizing = bound_json(config["sizing_report"], config["sizing_report_sha256"])
    parent = load_record(Path(config["sizing_record"]))
    require(parent["status"] == "completed", "completed_sizing_record")
    require(any(a["sha256"] == config["sizing_report_sha256"]
                and a["size_bytes"] == Path(config["sizing_report"]).stat().st_size
                for a in parent["artifacts"]), "sizing_record_artifact_binding")
    selected, jobs = select_probe(sizing, manifest)
    return manifest, selected, jobs, source_count


def check_dimensions(observed, expected):
    for field in ("total_tensor_bytes", "slice_nodes", "base_nodes", "base_edges", "extension_edges"):
        require(observed[field] == expected[field], "rebuilt_sizing_mismatch_" + field)


def validate_ceiling(allocated, reserved, limit=GPU_LIMIT_BYTES):
    require(type(allocated) is int and type(reserved) is int and 0 <= allocated <= reserved <= limit,
            "cuda_memory_ceiling_exceeded")


def execute_probe(model, graph, sidecars, shifts, weights):
    """One forward/backward, no optimizer reference or step in this function."""
    model.zero_grad(set_to_none=True)
    output = model(graph)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in output.logits.values()), "nonfinite_logits")
    alignment = align_temporal_targets(output, sidecars, shifts=shifts)
    loss = temporal_supervised_loss(output, alignment, weights)
    require(loss.total is not None and loss.fp32_boundary and bool(torch.isfinite(loss.total)), "nonfinite_loss")
    loss.total.backward()
    finite, norm = performance_gradient_diagnostics(model)
    require(finite and math.isfinite(norm), "nonfinite_gradients")
    require(any(p.grad is not None for p in model.parameters()), "missing_gradients")
    return {"loss": float(loss.total.detach()), "gradient_norm": norm,
            "finite_logits": True, "finite_gradients": True,
            "valid_rows": {h: int(row.valid_mask.sum()) for h, row in alignment.heads.items()}}


def run(config, out):
    manifest, selected, jobs, source_count = check_configuration(config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write = lambda path, value: path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    write(out / "config.json", config)
    write(out / "selection.json", {"selected": selected, "jobs": jobs, "max_forwards": MAX_FORWARDS})
    # Admission has no other application's GPU process as an implicit exception.
    idle = assert_probe_gpu_idle()
    require(torch.cuda.is_available(), "cuda_required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    require(total_memory >= GPU_LIMIT_BYTES, "insufficient_device_capacity")
    torch.cuda.set_per_process_memory_fraction(GPU_LIMIT_BYTES / total_memory, device)
    data = FrozenTrainingData(manifest)
    data.require_train_records(selected)
    eligible = {}
    for entry in data.table:
        eligible.setdefault(entry.record_id, set()).add(entry.shift_pc)
    skipped = []
    admitted = []
    for job in jobs:
        valid = all(shift in eligible[rid] for rid, shift in zip(job["record_ids"], job["shifts"]))
        if not valid and job["kind"] == "largest_pair_shift6":
            skipped.append({**job, "reason": "shift6_not_eligible_for_both_works"})
        else:
            require(valid, "shift_not_eligible")
            admitted.append(job)
    jobs = admitted
    parts = components("cuda")
    model = parts[0]
    weights = prepare_performance_class_weights(data.class_weights, device)
    initial = model_state_fingerprint(model)
    # Reserve the two FP32 AdamW moment tensors without performing an update.
    # Thus activation sizing includes the optimizer's steady-state storage.
    moments = [torch.empty_like(p) for p in model.parameters() for _ in range(2)]
    reserve_bytes = sum(t.numel() * t.element_size() for t in moments)
    started = time.perf_counter()
    measurements = []
    attempted = 0
    attempted_jobs = 0
    report = {"schema_version": SCHEMA, "experiment_id": config["experiment_id"],
              "passed": False, "completed": False, "optimizer_updates": 0,
              "model_forwards": 0, "max_model_forwards": MAX_FORWARDS,
              "validation_payload_accessed": False, "test_payload_accessed": False,
              "gpu_limit_bytes": GPU_LIMIT_BYTES, "optimizer_moment_reserve_bytes": reserve_bytes,
              "gpu_name": torch.cuda.get_device_name(device), "gpu_total_bytes": total_memory,
              "torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
              "source_entries_verified": source_count, "data_manifest_fingerprint": data.manifest_fingerprint,
              "scientific_config": scientific_config(), "selected_records": sorted(selected),
              "skipped_optional_jobs": skipped,
              "idle_exempt_processes": idle,
              "limitations": "Untrained forward/backward capacity probe; excludes driver allocations and CPU/IPC memory; no optimizer step or quality evidence."}
    try:
        for job in jobs:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            singletons, sidecars = [], []
            for rid in job["record_ids"]:
                graph, sidecar = read_temporal_record(data, rid, split="train")
                check_dimensions(dimensions(graph), selected[rid])
                singletons.append(graph)
                sidecars.append(sidecar)
            graph = transpose_temporal_historical(collate_temporal_graphs(singletons), job["shifts"])
            del singletons
            torch.cuda.reset_peak_memory_stats(device)
            attempted_jobs += 1
            begin = time.perf_counter()
            graph = graph.to(device)
            attempted += 1
            row = {**job, **execute_probe(model, graph, sidecars, job["shifts"], weights)}
            torch.cuda.synchronize(device)
            row.update(seconds=time.perf_counter()-begin,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            validate_ceiling(row["cuda_peak_allocated_bytes"], row["cuda_peak_reserved_bytes"])
            measurements.append(row)
            with (out / "measurements.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            print(json.dumps({"event": "memory_probe", "forward": attempted, **row}), flush=True)
            del graph, sidecars
        require(model_state_fingerprint(model) == initial, "model_parameters_changed_without_updates")
        require(all(not optimizer.state for optimizer in (parts[1],)), "optimizer_state_unexpected")
        verify_sources(config)
        report.update(passed=True, completed=True, model_unchanged=True)
    except BaseException as exc:
        report.update(failure={"type": type(exc).__name__, "reason": str(exc)},
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        raise
    finally:
        report.update(model_forwards=attempted, attempted_jobs=attempted_jobs, completed_forwards=len(measurements),
                      elapsed_seconds=time.perf_counter()-started, measurements=measurements)
        if report["passed"]:
            report.update(cuda_peak_allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in measurements),
                          cuda_peak_reserved_bytes=max(r["cuda_peak_reserved_bytes"] for r in measurements))
        write(out / "report.json", report)
        model.zero_grad(set_to_none=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
