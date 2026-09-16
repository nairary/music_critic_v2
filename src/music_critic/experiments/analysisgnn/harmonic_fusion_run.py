"""Source-free executable gate with learning checks and durable failure receipts."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
import time

import torch

from .contracts import fingerprint
from .corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from .corrected_training import (
    build_source_free_fixture, environment_fingerprint, load_frozen_class_weights,
    move_raw_graph_batch,
)
from .full_orbit_training import FullOrbitPair
from .harmonic_fusion_training import FusionTrainingConfig, evaluate_validation
from .harmonic_fusion_resume import (
    create_training_payload, load_training_checkpoint, preserve_caller_rng,
    save_training_checkpoint,
)
from .harmonic_fusion_checkpoint import create_model_payload, load_fusion_checkpoint
from .performance_replay import recursive_state_fingerprint


def _json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_bytes_once(path, content):
    """Prepare/fsync complete bytes before exclusive publication of the name."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            # A temporary-name cleanup failure cannot undo the already atomic
            # publication or replace the original pre-publication exception.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _write_json_once(path, payload):
    _write_bytes_once(path, _json_bytes(payload))


def _manifest(output_root, *, pending_files=None, filename="artifact-manifest.json"):
    manifest = {}
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != filename:
            content = path.read_bytes()
            manifest[path.name] = {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}
    for name, content in (pending_files or {}).items():
        manifest[name] = {"sha256": sha256(content).hexdigest(), "size_bytes": len(content)}
    _write_json_once(output_root / filename, manifest)


class _ProgressReceipt:
    """Append each boundary before continuing; completed trajectories stay visible."""

    def __init__(self, output_root, reads):
        self.path = output_root / "progress.jsonl"
        with self.path.open("x", encoding="utf-8"):
            pass
        self.reads = reads
        self.trajectories = []
        self.current = None
        self.session = None
        self.stage = "initialization"
        self.event_index = 0
        self.phase("initialization", "started")

    def _append(self, event):
        row = {"event_index": self.event_index, "arm": None if self.current is None else self.current["arm"],
               "trajectory": None if self.current is None else self.current["trajectory"], **event}
        row["fingerprint"] = fingerprint(row)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.event_index += 1

    def phase(self, stage, status="started"):
        self.stage = stage
        self._append({"stage": stage, "status": status})

    def begin(self, arm, trajectory, start_update):
        self.session = None
        self.current = {"arm": arm, "trajectory": trajectory, "start_update": start_update,
                        "end_update": start_update, "status": "running", "attempted_updates": 0,
                        "applied_updates": 0, "attempted_draws": 0, "applied_draws": 0}
        self.trajectories.append(self.current)
        self.phase("initialize_trajectory")

    def attach(self, session):
        if session.applied_updates != self.current["start_update"]:
            raise ValueError("fusion_gate.trajectory_start_mismatch")
        self.session = session

    def _sync(self):
        if self.session is not None:
            self.current["end_update"] = self.session.applied_updates
            self.current["applied_updates"] = self.session.applied_updates - self.current["start_update"]
            self.current["applied_draws"] = self.current["applied_updates"] * self.session.config.batch_size

    def event(self, event):
        self.stage = event["stage"]
        if event["stage"] == "update" and event["status"] == "started":
            self.current["attempted_updates"] += 1
            self.current["attempted_draws"] += len(event["draws"])
        self._sync()
        self._append(deepcopy(event))

    def finish(self):
        self._sync()
        self.current["status"] = "completed"
        self.phase("trajectory", "completed")
        self.session = None

    def snapshot(self):
        self._sync()
        return {"schema_version": "HarmonicFusionGateProgress@1.0.0",
                "trajectories": deepcopy(self.trajectories),
                **{key: sum(row[key] for row in self.trajectories) for key in (
                    "attempted_updates", "applied_updates", "attempted_draws", "applied_draws")},
                "actual_draws": self.reads["train"], "reader_calls": dict(self.reads),
                "draw_accounting": "attempted_schedule_slots_applied_B2_slots_and_actual_reader_calls_separate"}

    def fail(self, exc):
        self._sync()
        if self.current is not None and self.current["status"] != "completed":
            self.current["status"] = "failed"
        failure = {"stage": self.stage, "arm": None if self.current is None else self.current["arm"],
                   "trajectory": None if self.current is None else self.current["trajectory"],
                   "exception_type": type(exc).__name__, "message": str(exc)}
        self._append({"stage": self.stage, "status": "failed", "failure": failure,
                      "progress": self.snapshot()})
        return failure


def _learning_diagnostics(session, raw_graph_batch, initial_model_state, initial_fusion_state):
    """Report label-eligible per-head changes and check active fusion computation.

    Identity-validation diagnostics establish only that this fixture exercised
    learning. The raw residual probe adds no target-reader call or selection.
    """
    validations = {row["update"]: row["metrics"] for row in session.validations}
    before = validations[0]["per_head"]
    after = validations[session.config.total_updates]["per_head"]
    modes = [(module, module.training) for module in session.model.modules()]
    try:
        with preserve_caller_rng(), torch.inference_mode():
            session.model.eval()
            raw = move_raw_graph_batch(raw_graph_batch, session.device)
            baseline = session.model.base.predict(raw)
            fused = session.model.predict(raw)
            residuals = {task: float((fused.logits[task] - baseline.logits[task]).abs().max().cpu())
                         if fused.logits[task].numel() else 0.0 for task in fused.logits}
    finally:
        for module, mode in modes:
            module.training = mode
    per_head = {}
    for task in before:
        first, final = before[task], after[task]
        active = first["support"] > 0 and final["support"] > 0
        delta = final["masked_cross_entropy"] - first["masked_cross_entropy"] if active else None
        per_head[task] = {
            "initial_support": first["support"], "final_support": final["support"],
            "initial_masked_cross_entropy": first["masked_cross_entropy"],
            "final_masked_cross_entropy": final["masked_cross_entropy"],
            "masked_cross_entropy_delta": delta,
            "initial_accuracy": first["accuracy"], "final_accuracy": final["accuracy"],
            "initial_macro_f1": first["macro_f1_observed_validation_classes"],
            "final_macro_f1": final["macro_f1_observed_validation_classes"],
            "supervision_available": active,
            "actual_fusion_residual_max_abs": residuals[task],
        }
    changed_heads = [task for task, row in per_head.items()
                     if row["masked_cross_entropy_delta"] is not None and row["masked_cross_entropy_delta"] != 0]
    improved_heads = [task for task, row in per_head.items()
                      if row["masked_cross_entropy_delta"] is not None and row["masked_cross_entropy_delta"] < 0]
    fusion_change = None
    if session.model.enabled:
        final_fusion = {name: value.detach().cpu() for name, value in session.model.fusion.state_dict().items()}
        deltas = {name: value - initial_fusion_state[name] for name, value in final_fusion.items()}
        fusion_change = {
            "initial_state_fingerprint": model_state_fingerprint(initial_fusion_state),
            "final_state_fingerprint": model_state_fingerprint(final_fusion),
            "changed_parameters": sum(int(value.ne(0).sum()) for value in deltas.values()),
            "delta_l2": float(torch.sqrt(sum(value.double().square().sum() for value in deltas.values()))),
            "nonzero_output_head_count": sum(value > 0 for value in residuals.values()),
            "optimizer_steps": {name: count for name, count in session.optimizer_steps.items()
                                if name.startswith("fusion.")},
        }
    return {"schema_version": "HarmonicFusionFixtureLearningDiagnostics@1.0.0",
            "model_state_changed": model_state_fingerprint(session.model) != initial_model_state,
            "changed_supervised_heads": changed_heads, "improved_supervised_heads": improved_heads,
            "per_head": per_head, "fusion_weight_change": fusion_change,
            "claim": "nonzero_fixture_learning_and_active_fusion_only_not_musical_effectiveness"}


def _require_learning(diagnostics, *, enabled):
    if not diagnostics["model_state_changed"]:
        raise ValueError("fusion_gate.missing_nonzero_learning_signal")
    if enabled:
        change = diagnostics["fusion_weight_change"]
        if (not change["changed_parameters"] or change["delta_l2"] <= 0
                or not change["nonzero_output_head_count"] or not any(change["optimizer_steps"].values())):
            raise ValueError("fusion_gate.missing_fusion_learning_signal")


def run_synthetic_gate(output_root, *, device="cpu"):
    """24 actual updates/48 draws: 6 uninterrupted + 3+3 restored per arm.

    A failed run retains an exclusive failure receipt and append-only progress,
    including earlier completed trajectories. Exceptions always propagate.
    No original checkpoint or prior result is overwritten.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reads = {"train": 0, "validation": 0}
    progress = _ProgressReceipt(output_root, reads)
    try:
        return _run_gate(output_root, device=device, started=started, reads=reads, progress=progress)
    except BaseException as exc:
        try:
            failure = progress.fail(exc)
            result = {"schema_version": "HarmonicFusionSyntheticGateFailure@1.0.0", "passed": False,
                      "scope": "generated_fixture_mechanics_only", "device": device, "failure": failure,
                      "progress": progress.snapshot(), "wall_seconds": time.perf_counter() - started,
                      "actual_updates": progress.snapshot()["applied_updates"], "actual_draws": reads["train"],
                      "test_inputs_accessed": False, "test_targets_accessed": False,
                      "test_metrics_computed": False, "real_data_accessed": False, "quality_claims": False}
            result["fingerprint"] = fingerprint(result)
            _write_json_once(output_root / "failure-receipt.json", result)
            filename = ("failure-artifact-manifest.json" if (output_root / "artifact-manifest.json").exists()
                        else "artifact-manifest.json")
            _manifest(output_root, filename=filename)
        except BaseException as receipt_error:
            exc.add_note(f"Failure receipt could not be fully written: {type(receipt_error).__name__}: {receipt_error}")
            raise exc from receipt_error
        raise


def _run_gate(output_root, *, device, started, reads, progress):
    config = FusionTrainingConfig(total_updates=6, warmup_updates=2,
                                  validation_interval=3, checkpoint_interval=3)
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    table = tuple(FullOrbitPair(sidecar["record_id"], shift, sidecar["source_component_id"])
                  for shift in range(12))
    bindings = {"source_kind": "generated_fixture_no_real_checkpoint",
                "class_weight_fingerprint": weights.payload_fingerprint,
                "sidecar_fingerprint": sidecar["fingerprint"],
                "source_code": {Path(name).name: sha256(Path(name).read_bytes()).hexdigest()
                    for name in (__file__, __file__.replace("_run.py", "_training.py"),
                                 __file__.replace("_run.py", "_resume.py"))}}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def reader(record_id, *, split):
        if split not in reads or record_id != sidecar["record_id"]:
            raise ValueError("fusion_gate.source_free_read_boundary")
        reads[split] += 1
        return batch.raw_graph_batch, deepcopy(sidecar)

    def validate(model, *, split):
        return evaluate_validation(model, records=(sidecar["record_id"],), reader=reader,
                                   class_weights=weights, split=split)

    with preserve_caller_rng():
        torch.manual_seed(17)
        base = CorrectedAnalysisGNNModel()
    rows = []
    for enabled in (False, True):
        from .harmonic_fusion_training import FusionTrainingSession
        arm = "fusion" if enabled else "control"
        fresh = lambda: FusionTrainingSession(base=base, enabled=enabled, config=config,
                                              bindings=bindings, table=table, device=device)
        progress.begin(arm, "uninterrupted", 0)
        full = fresh()
        progress.attach(full)
        initial_model_state = model_state_fingerprint(full.model)
        initial_fusion_state = ({name: value.detach().cpu().clone() for name, value in full.model.fusion.state_dict().items()}
                                if enabled else None)
        full.run_until(6, reader, weights, validate=validate, progress=progress.event)
        progress.phase("learning_check")
        diagnostics = _learning_diagnostics(full, batch.raw_graph_batch, initial_model_state, initial_fusion_state)
        _write_json_once(output_root / f"{arm}-learning-diagnostics.json", diagnostics)
        _require_learning(diagnostics, enabled=enabled)
        progress.phase("snapshot")
        expected = create_training_payload(full)
        full_state = model_state_fingerprint(full.model)
        progress.finish()
        del full
        progress.begin(arm, "interrupted", 0)
        interrupted = fresh()
        progress.attach(interrupted)
        interrupted.run_until(3, reader, weights, validate=validate, progress=progress.event)
        progress.phase("checkpoint_save")
        checkpoint = output_root / f"{arm}-update-000003.pt"
        evidence = save_training_checkpoint(interrupted, checkpoint)
        progress.finish()
        del interrupted
        progress.begin(arm, "restored", 3)
        progress.phase("restore")
        restored = load_training_checkpoint(checkpoint, expected_sha256=evidence["sha256"],
            expected_size_bytes=evidence["size_bytes"], config=config, bindings=bindings,
            table=table, device=device, expected_enabled=enabled)
        progress.attach(restored)
        restored.run_until(6, reader, weights, validate=validate, progress=progress.event)
        progress.phase("trajectory_comparison")
        actual = create_training_payload(restored)
        if recursive_state_fingerprint(expected) != recursive_state_fingerprint(actual):
            raise ValueError("fusion_gate.interrupted_trajectory_mismatch")
        progress.phase("checkpoint_save")
        final_evidence = save_training_checkpoint(restored, output_root / f"{arm}-update-000006.pt")
        fusion_export = None
        if enabled:
            progress.phase("model_export")
            export = output_root / "fusion-final-model-only.pt"
            with export.open("xb") as stream:
                torch.save(create_model_payload(restored.model), stream)
            content = export.read_bytes()
            fusion_export = {"sha256": sha256(content).hexdigest(), "size_bytes": len(content),
                             "parent_training_checkpoint": final_evidence}
            loaded = load_fusion_checkpoint(export, expected_sha256=fusion_export["sha256"],
                                            expected_size_bytes=fusion_export["size_bytes"])
            if model_state_fingerprint(loaded.model) != full_state:
                raise ValueError("fusion_gate.model_export_dropped_weights")
        progress.finish()
        arm_progress = [row for row in progress.snapshot()["trajectories"] if row["arm"] == arm]
        rows.append({"arm": arm, "interrupted_matches_uninterrupted": True,
                     "actual_updates": sum(row["applied_updates"] for row in arm_progress),
                     "actual_draws": sum(row["applied_draws"] for row in arm_progress),
                     "committed_final_update": restored.applied_updates,
                     "sampler_position": restored.sampler.position,
                     "model_state_fingerprint": full_state,
                     "final_training_checkpoint": final_evidence,
                     "fusion_export": fusion_export, "learning_diagnostics": diagnostics,
                     "validation_updates": [row["update"] for row in restored.validations]})
        del restored, expected, actual
    progress.phase("report")
    accounting = progress.snapshot()
    if (accounting["attempted_updates"] != 24
            or accounting["applied_updates"] != 24 or accounting["attempted_draws"] != 48
            or accounting["applied_draws"] != reads["train"] or reads["train"] != 48):
        raise ValueError("fusion_gate.actual_budget_mismatch")
    result = {"schema_version": "HarmonicFusionSyntheticTrainerGate@1.1.0",
              "passed": True, "scope": "generated_fixture_mechanics_only", "device": device,
              "config": config.to_dict(), "bindings": bindings, "arms": rows, "progress": accounting,
              "actual_updates": accounting["applied_updates"], "actual_draws": reads["train"], "reader_calls": reads,
              "test_inputs_accessed": False, "test_targets_accessed": False,
              "test_metrics_computed": False, "real_data_accessed": False,
              "wall_seconds": time.perf_counter() - started,
              "torch_num_threads": torch.get_num_threads(),
              "torch_num_interop_threads": torch.get_num_interop_threads(),
              "environment": environment_fingerprint(),
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "quality_claims": False, "real_train_C2_equivalence": False}
    result["fingerprint"] = fingerprint(result)
    # report.json is the final success commit. Any serialization, artifact
    # enumeration, manifest write or report fsync failure precedes its name.
    report_bytes = _json_bytes(result)
    _manifest(output_root, pending_files={"report.json": report_bytes})
    _write_bytes_once(output_root / "report.json", report_bytes)
    return result
