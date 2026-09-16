"""Opt-in bounded Dilemma local-control dispatcher; execution belongs to coordinator.

Preflight reads only authenticated metadata/index identities. Execution opens the
selected TRAIN/VALIDATION raw caches, then targets after raw features/prediction.
All outputs are exclusive, and a durable event prefix survives process failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import heapq
import json
import os
from pathlib import Path, PurePosixPath
import platform
import tempfile
import time

import torch

from music_critic.experiments.context_diagnostics import (
    FileBinding, Member, LabeledFeature, FEATURE_CONTRACT, LOCAL_FEATURE_SHA256,
    fingerprint, fit_local_control, local_features, raw_candidates,
)
from music_critic.experiments.context_local_contract import load_authoritative_metadata
from music_critic.experiments.context_local_exports import (
    SelectedExport, aligned_target_rows, checked_json, read_raw, read_targets, resolve_binding,
)
from music_critic.experiments.context_local_reference import (
    ALLOWED_TASKS, TASK_LEVELS, FrozenC2Reference, build_reference_view, ordered_classes,
)
from music_critic.experiments.context_local_uncertainty import paired_component_report
from music_critic.experiments.registry import load_record, seal_record, validate_record, write_record_once


CONFIG_SCHEMA = "ContextLocalControlConfig@1.0.0"
CHECKPOINT = {"sha256": "a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797",
              "size_bytes": 14844299,
              "model_contract_sha256": "c12380dc6a9e4923bf80e843d010698fa7dd8fdefb62e9d3c0262d0c1ff32748",
              "model_state_sha256": "82ed08304b565a1447062a95112c8a51b47ce75c8f46f07b858443236ba83f39"}
TASKS = ("local_key", "root", "quality", "primary_degree", "cadence", "metrical_strength")
BUDGET = {"train_pieces": 128, "validation_pieces": 32, "max_notes_per_piece": 16384,
          "max_candidates_per_level": 16384, "max_target_entities_per_piece": 20000,
          "max_forwards": 32, "training_updates": 0, "cpu_threads": 1,
          "max_wall_seconds": 600, "max_gpu_allocated_bytes": 4 * 1024**3}
BASE_R2_MANIFEST_SHA256 = "22f4890cb83bed3953b94df89a01ea0b72de39628e47beffafdfc47356760def"
SIZING_EVIDENCE = {"experiment_id": "EXP-PILOT-RAW-SIZING-001", "status": "completed",
    "record_sha256": "dd3180a59a307de2db9a8f7cb9107120910fb35b8c2c5cd72ffcec524a3d1990",
    "record_fingerprint": "ac0e39509fb06fc4b386685a669c7fa01ff8850d354719c7ce3aaecda3e555c5",
    "receipt_sha256": "385fd300e17625af1936e6b663848e569139e76998b028aca2f95c114daab876",
    "selected_records": 160, "selected_max_notes": 9164, "selected_max_beats": 2822,
    "selected_max_onsets": 4661, "selected_max_target_entities": 16970,
    "selected_max_relations": 32859, "selected_max_local_window_notes": 47,
    "provenance": "coordinator registered sizing; compact record and receipt verified before source seal",
    "external_host_available_memory_floor_bytes": 8 * 1024**3}


def _fail(reason):
    raise ValueError("context_local_dispatch." + reason)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload):
    descriptor, temporary = tempfile.mkstemp(prefix=".context-local-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber publication on the same filesystem.
    finally:
        os.unlink(temporary)


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        _fail("config_budget")
    config = checked_json(FileBinding(str(path), sha256(raw).hexdigest(), len(raw)), max_bytes=64*1024)
    keys = {"schema_version", "experiment_id", "tasks", "seed", "bootstrap_draws", "device",
            "budget", "checkpoint", "metadata_bindings", "cache_index_binding", "binding_audit"}
    if not isinstance(config, dict) or set(config) != keys or config["schema_version"] != CONFIG_SCHEMA:
        _fail("config_schema")
    if (not isinstance(config["experiment_id"], str)
            or not config["experiment_id"].startswith("EXP-CTX-C2-LOCAL-")
            or "/" in config["experiment_id"] or "\\" in config["experiment_id"]):
        _fail("local_experiment_id_required")
    if (config["tasks"] != list(TASKS) or type(config["seed"]) is not int or config["seed"] != 17
            or type(config["bootstrap_draws"]) is not int or config["bootstrap_draws"] != 200
            or config["device"] not in ("cpu", "cuda:0")
            or fingerprint(config["budget"]) != fingerprint(BUDGET)
            or fingerprint(config["checkpoint"]) != fingerprint(CHECKPOINT)):
        _fail("fixed_protocol_required")
    audit = config["binding_audit"]
    if (not isinstance(audit, dict) or set(audit) != {"experiment_id", "record_binding"}
            or audit["experiment_id"] != "EXP-RESEARCH-DILEMMA-BINDINGS-001"):
        _fail("registered_cache_audit_required")
    return config


def _index_rows(binding: FileBinding) -> list[dict]:
    from music_critic.experiments.context_local_exports import _object_pairs, _finite_float
    raw = binding.read(max_bytes=4*1024*1024)
    rows = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line, object_pairs_hook=_object_pairs,
                                   parse_float=_finite_float,
                                   parse_constant=lambda value: _fail("nonfinite_index")))
    return rows


def _cache_binding(value: dict, *, record_id: str, filename: str, cache_root: Path) -> tuple[FileBinding, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        _fail("native_cache_index_binding")
    path = PurePosixPath(value["path"])
    token = sha256(record_id.encode()).hexdigest()[:24]
    if (str(path) != value["path"] or ".." in path.parts or "\\" in value["path"]
            or tuple(path.parts[-2:]) != (token, filename)):
        _fail("native_cache_locator_layout")
    # Original absolute server paths are evidence, never opened directly. Only
    # the verified record token/name are resolved below the operator's cache root.
    actual = resolve_binding({"locator": f"{token}/{filename}", "sha256": value["sha256"],
                              "size_bytes": value["size_bytes"]}, cache_root)
    return actual, str(path.parent.parent)


def export_identity(exports):
    """Portable selected payload binding; relocation cannot change evidence bytes."""
    return [{"record_id": e.record_id, "member": asdict(e.member), "dialect": e.dialect,
             **{key: {"sha256": getattr(e, key).sha256, "size_bytes": getattr(e, key).size_bytes}
                for key in ("raw", "targets", "metadata")}}
            for e in sorted(exports, key=lambda e: e.record_id)]


def preflight(config: dict, *, input_root: Path, cache_root: Path) -> tuple[dict, object, tuple[SelectedExport, ...]]:
    bindings = {key: resolve_binding(value, input_root) for key, value in config["metadata_bindings"].items()}
    metadata = load_authoritative_metadata(bindings)
    index_binding = resolve_binding(config["cache_index_binding"], input_root)
    audit_binding = resolve_binding(config["binding_audit"]["record_binding"], input_root)
    audit = validate_record(checked_json(audit_binding, max_bytes=128*1024))
    if (audit["status"] != "completed" or audit["experiment_id"] != config["binding_audit"]["experiment_id"]
            or not any(a["sha256"] == index_binding.sha256 and a["size_bytes"] == index_binding.size_bytes
                       for a in audit["artifacts"])):
        _fail("cache_index_not_in_completed_audit")
    if any(audit["data_access"].get(key) is True for key in (
            "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed", "test_used_for_selection")):
        _fail("audit_TEST_access_incompatible")
    rows = _index_rows(index_binding)
    by_id = {}
    roots = set()
    normalized = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"record_id", "split", "source_component_id", "raw", "sidecar", "metadata"}:
            _fail("native_cache_index_schema")
        identity = row["record_id"]
        if identity in by_id or identity not in metadata.assignments:
            _fail("duplicate_or_unknown_cache_record")
        assignment = metadata.assignments[identity]
        if (row["split"] not in ("train", "validation") or row["split"] != assignment["split"]
                or row["source_component_id"] != assignment["source_component_id"]):
            _fail("cache_official_assignment_mismatch")
        values = {}
        for key, name in (("raw", "piece.json"), ("sidecar", "target_sidecar.json"), ("metadata", "metadata.json")):
            values[key], origin = _cache_binding(row[key], record_id=identity, filename=name, cache_root=cache_root)
            roots.add(origin)
        normalized[identity] = values
        by_id[identity] = row
    required = {key for key, row in metadata.assignments.items() if row["split"] in ("train", "validation")}
    if set(by_id) != required or len(roots) != 1:
        _fail("complete_TRAINVAL_cache_index_required")
    exports = tuple(SelectedExport(row["record_id"], Member(metadata.record_to_piece[row["record_id"]],
                       row["component_id"], row["split"]), metadata.assignments[row["record_id"]]["dialect"],
                       normalized[row["record_id"]]["raw"], normalized[row["record_id"]]["sidecar"],
                       normalized[row["record_id"]]["metadata"]) for row in metadata.selected)
    package = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(package)): sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.rglob("*.py"))}
    script = Path(__file__).resolve().parents[3] / "scripts/context_local_control.py"
    protocol = {"schema_version": "ContextLocalControlProtocol@1.0.0", "config_fingerprint": fingerprint(config),
                "base_r2_manifest_sha256": BASE_R2_MANIFEST_SHA256,
                "source_sha256": source_hashes, "dispatcher_script_sha256": sha256(script.read_bytes()).hexdigest(),
                "selection": metadata.to_dict(), "cache_index_sha256": index_binding.sha256,
                "selected_exports_fingerprint": fingerprint(export_identity(exports)),
                "cache_audit_record_fingerprint": audit["record_fingerprint"],
                "original_cache_prefix_sha256": fingerprint(sorted(roots)),
                "checkpoint": config["checkpoint"], "device": config["device"],
                "parent_selection_provenance": "C2 best step 115000 and seed 17 declared by parent; not independently extracted from checkpoint by this dispatcher",
                "classes": ordered_classes(TASKS), "feature_contract": FEATURE_CONTRACT,
                "local_feature_sha256": LOCAL_FEATURE_SHA256, "budget": BUDGET,
                "resource_sizing_evidence": SIZING_EVIDENCE,
                "bootstrap": {"draws": 200, "seed": 17, "unit": "official validation component",
                              "pairing": "same component multiplicities, same eligible source observations"},
                "source_provenance_limit": "registered historical cache byte/metadata audit; original raw conversion not repeated",
                "claim_limit": "descriptive local-control comparison; capacity differs; no causal context attribution, new model selection or music-quality verdict",
                "test_policy": "assignment identities only; no TEST raw, expanded targets, inference or metrics"}
    protocol["protocol_fingerprint"] = fingerprint(protocol)
    return protocol, metadata, exports


def make_planned_record(config, protocol) -> dict:
    record = {"schema_version": "1.0.0", "experiment_id": config["experiment_id"],
              "kind": "diagnostic", "phase": "9E-context-local-control", "status": "planned",
              "title": "Bounded Dilemma TRAIN-only nearest-centroid versus preselected frozen C2",
              "question": "Which errors persist under a fixed raw local control on exactly shared VALIDATION observations?",
              "hypothesis": "A local control may separate local recognition failures from errors requiring further context probes; no superiority is presumed.",
              "decision_use": "Error localization for later preregistered ablations, not selection using POP or a music-quality claim.",
              "repository": {"git_commit": "da8c9dcd999a4b522f87f9a20bf3506ad26497b7", "dirty": True,
                             "base_r2_manifest_sha256": BASE_R2_MANIFEST_SHA256,
                             "package_sources_fingerprint": fingerprint(protocol["source_sha256"])},
              "parent_ids": [config["binding_audit"]["experiment_id"], "EXP-RESEARCH-DILEMMA-METADATA-002", "EXP-PILOT-RAW-SIZING-001", "EXP-CTX-C2-SUITE-GPU-001"],
              "created_at": _now(), "seeds": [17], "budget": {"planned": BUDGET},
              "protocol": {"context_local_fingerprint": protocol["protocol_fingerprint"],
                           "selection_fingerprint": protocol["selection"]["fingerprint"],
                           "config_fingerprint": fingerprint(config)},
              "data_access": {"declared_policy": {"source": "2026-09-13 coordinator continuation; fixed TRAIN/VALIDATION only",
                  "validation_allowed": True, "test_inputs_allowed": False, "test_targets_allowed": False,
                  "test_metrics_allowed": False, "test_selection_allowed": False},
                  "validation_accessed": None, "test_inputs_accessed": None, "test_targets_accessed": None,
                  "test_metrics_computed": None, "test_used_for_selection": None,
                  "access_evidence_status": "pending_primary_evidence"},
              "results": {}, "validity": {"valid": None, "reason": None},
              "claims": {"supported": [], "not_supported": [protocol["claim_limit"], protocol["source_provenance_limit"]]},
              "artifacts": [], "import_provenance": {"mode": "planned_not_launched", "source": protocol["source_provenance_limit"]}}
    return validate_record(seal_record(record))


def piece_features(piece, tasks=TASKS, check_budget=lambda: None):
    """Same one-qn features, with a raw interval sweep avoiding irrelevant notes."""
    candidates = {level: raw_candidates(piece, level) for level in set(TASK_LEVELS[t] for t in tasks)}
    if (len(piece.notes) > BUDGET["max_notes_per_piece"]
            or any(len(rows) > BUDGET["max_candidates_per_level"] for rows in candidates.values())):
        _fail("raw_feature_budget")
    windows = sorted({(row.start, row.end) for rows in candidates.values() for row in rows})
    notes = sorted((n for n in piece.notes if not n.is_percussion and not n.is_grace),
                   key=lambda n: (n.onset_qn, n.note_id))
    active, ends, vectors, cursor = {}, [], {}, 0
    for start, end in windows:
        check_budget()
        while cursor < len(notes) and notes[cursor].onset_qn < end:
            n = notes[cursor]
            active[n.note_id] = n
            heapq.heappush(ends, (n.onset_qn + n.duration_qn, n.note_id))
            cursor += 1
        while ends and ends[0][0] <= start:
            _, identity = heapq.heappop(ends)
            active.pop(identity, None)
        from music_critic.experiments.context_diagnostics import Candidate
        vectors[(start, end)] = local_features(replace(piece, notes=tuple(active.values())), Candidate("window", start, end))
    return {task: {row.candidate_id: vectors[(row.start, row.end)] for row in candidates[TASK_LEVELS[task]]}
            for task in tasks}, len(windows)


class Progress:
    def __init__(self, stream):
        self.stream, self.start, self.sequence = stream, time.monotonic(), 0
        self.state = {"stage": "initializing", "piece_id": None, "split": None,
                      "raw_pieces_started": 0, "raw_pieces_completed": 0, "target_pieces_started": 0,
                      "validation_raw_started": 0, "validation_targets_started": 0,
                      "target_pieces_completed": 0, "forwards_started": 0, "forwards_completed": 0,
                      "forwards_completion_unknown": 0, "local_fits_completed": 0, "applied_updates": 0}

    def check_budget(self):
        if time.monotonic() - self.start > BUDGET["max_wall_seconds"]:
            _fail("cooperative_wall_budget_exceeded")

    def emit(self, event, detail=None):
        self.sequence += 1
        if self.sequence > 2000:
            _fail("progress_budget")
        if detail:
            self.state.update(detail)
        row = {"schema_version": "ContextLocalProgress@1.0.0", "sequence": self.sequence,
               "event": event, "wall_seconds": time.monotonic() - self.start, **self.state}
        line = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if len(line.encode()) > 4096:
            _fail("progress_row_budget")
        self.stream.write(line + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())


def execute(config, protocol, exports, *, checkpoint: Path, record_path: Path, output_root: Path):
    record = load_record(record_path)
    policy = record["data_access"]["declared_policy"]
    if (record["status"] != "planned" or record["experiment_id"] != config["experiment_id"]
            or record.get("protocol", {}).get("context_local_fingerprint") != protocol["protocol_fingerprint"]
            or policy["validation_allowed"] is not True
            or any(policy[k] for k in policy if k.startswith("test_"))):
        _fail("matching_planned_record_required")
    if Counter(e.member.split for e in exports) != {"train": BUDGET["train_pieces"], "validation": BUDGET["validation_pieces"]}:
        _fail("selected_count_mismatch")
    if (len({e.record_id for e in exports}) != len(exports)
            or len({e.member.piece_id for e in exports}) != len(exports)
            or {e.member.component_id for e in exports if e.member.split == "train"}
               & {e.member.component_id for e in exports if e.member.split == "validation"}
            or fingerprint(export_identity(exports)) != protocol.get("selected_exports_fingerprint")
            or fingerprint(config) != protocol.get("config_fingerprint")
            or fingerprint({k: v for k, v in protocol.items() if k != "protocol_fingerprint"}) != protocol["protocol_fingerprint"]):
        _fail("selected_protocol_identity_mismatch")
    output_root.mkdir()  # Entire evidence root must be fresh; an earlier attempt is never overwritten.
    _write_json(output_root / "planned-record.json", record)
    _write_json(output_root / "protocol.json", protocol)
    classes = ordered_classes(TASKS)
    train = {task: [] for task in TASKS}
    observations = {task: [] for task in TASKS}
    coverage = {task: Counter() for task in TASKS}
    members = [e.member for e in exports if e.member.split == "validation"]
    controls, per_piece, reference = {}, [], None
    started = _now()
    status, error, report = "completed", None, None
    with (output_root / "progress.jsonl").open("x", encoding="utf-8") as stream:
        progress = Progress(stream)
        try:
            torch.set_num_threads(1)
            for export in sorted(exports, key=lambda e: (e.member.split != "train", e.record_id)):
                progress.check_budget()
                if export.member.split == "validation" and reference is None:
                    for task in TASKS:
                        progress.emit("local_fit_started", {"stage": "fit_train_only", "task_id": task})
                        controls[task] = fit_local_control(train[task], classes[task]) if train[task] else None
                        progress.state["local_fits_completed"] += controls[task] is not None
                        train[task].clear()
                        progress.check_budget()
                    progress.emit("checkpoint_load_started", {"stage": "checkpoint_load"})
                    reference = FrozenC2Reference(FileBinding(str(checkpoint), CHECKPOINT["sha256"], CHECKPOINT["size_bytes"]),
                        CHECKPOINT["model_contract_sha256"], CHECKPOINT["model_state_sha256"], config["device"], progress.emit)
                    if config["device"] != "cpu":
                        torch.cuda.reset_peak_memory_stats(0)
                progress.state["raw_pieces_started"] += 1
                progress.state["validation_raw_started"] += export.member.split == "validation"
                progress.emit("raw_read_started", {"stage": "raw_read", "piece_id": export.member.piece_id,
                    "record_id": export.record_id, "split": export.member.split})
                piece = read_raw(export)
                progress.state["raw_pieces_completed"] += 1
                progress.emit("features_started", {"stage": "raw_features"})
                features, window_count = piece_features(piece, check_budget=progress.check_budget)
                progress.emit("raw_view_started", {"stage": "raw_graph_or_frozen_prediction"})
                view = (reference.predict(piece, TASKS) if export.member.split == "validation"
                        else build_reference_view(piece, TASKS))
                progress.state["target_pieces_started"] += 1
                progress.state["validation_targets_started"] += export.member.split == "validation"
                progress.emit("targets_started", {"stage": "target_read_after_raw"})
                sidecar = read_targets(export)
                progress.state["target_pieces_completed"] += 1
                targets, join = aligned_target_rows(view, sidecar, TASKS, {t: set(features[t]) for t in TASKS})
                for task in TASKS:
                    for target in targets[task]:
                        coverage[task][export.member.split + ":source_observations"] += 1
                        if not target.available:
                            coverage[task][export.member.split + ":masked:" + target.reason] += 1
                            continue
                        coverage[task][export.member.split + ":eligible"] += 1
                        if export.member.split == "train":
                            train[task].append(LabeledFeature(export.member, target.observation_id,
                                                             features[task][target.candidate_id], target))
                        elif controls[task] is not None:
                            observations[task].append({"piece_id": piece.piece_id, "component_id": export.member.component_id,
                                "observation_id": target.observation_id, "candidate_id": target.candidate_id,
                                "gold": target.value, "local": controls[task].predict(features[task][target.candidate_id]),
                                "reference": view.predictions[task][target.candidate_id],
                                "target_provenance": target.provenance, "target_confidence": target.confidence})
                if export.member.split == "validation":
                    _write_json(output_root / (sha256(export.record_id.encode()).hexdigest()[:24] + ".predictions.json"),
                        {"record_id": export.record_id, "piece_id": piece.piece_id, "input_fingerprint": view.input_fingerprint,
                         "logits": view.logits, "predictions": view.predictions,
                         "local_features": features, "endpoint_candidates": view.evidence.get("excluded_endpoint_candidates"),
                         "source_bindings": {"raw_sha256": export.raw.sha256, "target_sha256": export.targets.sha256}})
                per_piece.append({"record_id": export.record_id, "piece_id": piece.piece_id,
                    "component_id": export.member.component_id, "split": export.member.split,
                    "raw_sha256": export.raw.sha256, "sidecar_sha256": export.targets.sha256,
                    "model_input_fingerprint": view.input_fingerprint, "unique_raw_windows": window_count,
                    "alignment": join})
                progress.emit("piece_completed", {"stage": "piece_completed"})
                if config["device"] != "cpu" and reference is not None:
                    if torch.cuda.max_memory_allocated(0) > BUDGET["max_gpu_allocated_bytes"]:
                        _fail("post_forward_gpu_peak_budget_exceeded")
            if (reference is None or reference.forwards_started != BUDGET["max_forwards"]
                    or reference.forwards_completed != BUDGET["max_forwards"] or reference.forwards_unknown):
                _fail("forward_completion_mismatch")
            progress.emit("comparison_started", {"stage": "paired_component_uncertainty"})
            comparisons = {}
            for task in TASKS:
                progress.check_budget()
                reason = ("no_available_TRAIN_targets" if controls[task] is None else
                          "no_eligible_VALIDATION_observations" if not observations[task] else None)
                comparisons[task] = {"status": "estimated" if reason is None else "not_estimable",
                    "reason": reason,
                    "coverage": dict(sorted(coverage[task].items())),
                    "local_model": asdict(controls[task]) if controls[task] is not None else None,
                    "observations": observations[task],
                    "comparison": paired_component_report(observations[task], classes[task], members) if controls[task] is not None else None}
            report = {"schema_version": "ContextLocalControlResult@1.0.0", "experiment_id": config["experiment_id"],
                      "protocol_fingerprint": protocol["protocol_fingerprint"], "per_piece": per_piece,
                      "tasks": comparisons, "checkpoint_evidence": reference.checkpoint_evidence,
                      "accounting": dict(progress.state), "wall_seconds": time.monotonic()-progress.start,
                      "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if config["device"] != "cpu" else None,
                      "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": config["device"]},
                      "claim_limit": protocol["claim_limit"], "source_provenance_limit": protocol["source_provenance_limit"],
                      "TEST": {"raw": False, "expanded_targets": False, "forwards": 0, "metrics": False}}
            report["result_fingerprint"] = fingerprint(report)
            progress.emit("result_write_started", {"stage": "result_write"})
            _write_json(output_root / "result.json", report)
            progress.emit("result_written", {"stage": "result_written"})
        except BaseException as exc:
            status = "aborted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            error = {"type": type(exc).__name__, "message": str(exc)[:1000], "stage": progress.state["stage"]}
            if reference is not None:
                progress.state.update(forwards_started=reference.forwards_started,
                    forwards_completed=reference.forwards_completed, forwards_completion_unknown=reference.forwards_unknown)
            try:
                progress.emit("execution_" + status, {"error": error})
            except (OSError, ValueError):
                pass
        receipt = {"schema_version": "ContextLocalExecutionReceipt@1.0.0", "status": status,
                   "started_at": started, "ended_at": _now(), "error": error,
                   "accounting": dict(progress.state), "wall_seconds": time.monotonic()-progress.start,
                   "runtime_limit": "cooperative boundary checks; coordinator must enforce external 600s timeout; hard kill leaves journal prefix"}
        _write_json(output_root / "execution-receipt.json", receipt)
    proposal = terminal_record(record, protocol, receipt, report if status == "completed" else None, output_root)
    write_record_once(output_root / (record["experiment_id"] + ".json"), proposal)
    return receipt


def terminal_record(planned, protocol, receipt, report, output_root):
    """Immutable terminal proposal; coordinator validates/imports into the tracked registry."""
    terminal = json.loads(json.dumps(planned))
    terminal.update(status=receipt["status"], started_at=receipt["started_at"], ended_at=receipt["ended_at"])
    terminal["budget"]["actual"] = {"wall_seconds": receipt["wall_seconds"], **receipt["accounting"]}
    terminal["data_access"].update(validation_accessed=receipt["accounting"]["validation_raw_started"] > 0,
        test_inputs_accessed=False, test_targets_accessed=False, test_metrics_computed=False,
        test_used_for_selection=False, access_evidence_status="verified_primary_evidence",
        evidence="hash-bound TRAIN/VALIDATION selection and durable execution journal; read-start counters conservatively count access")
    terminal["validity"] = {"valid": True if report else False,
        "reason": None if report else receipt["error"]["type"] + ": " + receipt["error"]["message"]}
    def compact_point(point):
        return {arm: {key: values[key] for key in ("accuracy", "observed_class_macro_f1") if key in values}
                for arm, values in point.items()}
    terminal["results"] = ({"result_fingerprint": report["result_fingerprint"],
        "tasks": {task: {"status": value["status"], "coverage": value["coverage"],
            "point": compact_point(value["comparison"]["point"]) if value["comparison"] else None}
            for task, value in report["tasks"].items()},
        "claim_limit": protocol["claim_limit"], "TEST_closed": True} if report else
        {"error": receipt["error"], "partial_artifacts_are_not_comparison_evidence": True})
    terminal["claims"]["supported"] = (["Fixed raw local-control comparison completed on shared selected VALIDATION observations."]
                                           if report else ["Attempt stopped; durable partial accounting retained."])
    terminal["artifacts"] = [{"role": "local_control_" + p.stem, "sha256": sha256(p.read_bytes()).hexdigest(),
        "size_bytes": p.stat().st_size, "availability": "local_ignored",
        "locator": p.name}
        for p in sorted(output_root.iterdir()) if p.is_file()]
    terminal["import_provenance"] = {"mode": "terminal_proposal_requires_coordinator_import",
        "source": protocol["source_provenance_limit"],
        "artifact_locator_base": "execution evidence bundle; coordinator binds its retained portable bundle prefix at import",
        "planned_record_fingerprint": planned["record_fingerprint"],
        "external_supervisor_receipt": "coordinator attaches verified wrapper receipt before registry finalization"}
    terminal["next_action"] = "Coordinator checks terminal proposal and resource receipt, finalizes tracked record and regenerates ledger before another substantial run."
    return validate_record(seal_record(terminal))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "plan", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    protocol, metadata, exports = preflight(config, input_root=args.input_root, cache_root=args.cache_root)
    if args.mode == "preflight":
        _write_json(args.output, protocol)
    elif args.mode == "plan":
        write_record_once(args.output, make_planned_record(config, protocol))
    else:
        if args.record is None or args.checkpoint is None:
            parser.error("run requires --record and --checkpoint")
        receipt = execute(config, protocol, exports, checkpoint=args.checkpoint,
                          record_path=args.record, output_root=args.output)
        return 0 if receipt["status"] == "completed" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
