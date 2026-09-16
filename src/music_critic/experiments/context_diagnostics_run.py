"""Bounded context-suite CLI: mechanics by default, registered frozen inference opt-in."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import time
from typing import TextIO

import torch
from torch_geometric import __version__ as pyg_version
from torch_geometric.data import Batch

from music_critic.experiments.analysisgnn.corrected_model import corrected_model_contract
from music_critic.experiments.analysisgnn.corrected_training import move_raw_graph_batch
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.context_diagnostics import (
    FEATURE_CONTRACT, fingerprint, local_features, Candidate, restrict_note_context, require_sha,
)
from music_critic.experiments.context_diagnostics_pairs import (
    build_context_pair_suite, context_pair_to_dict, validate_context_pair,
)
from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
from music_critic.experiments.registry import load_record
from music_critic.graph import build_raw_graph, graph_fingerprint, model_input_fingerprint, validate_raw_graph_batch


TASKS = ("local_key", "root", "quality", "primary_degree", "cadence")


@dataclass
class SuiteProgress:
    """Small append-only accounting journal; no logits or corpus data.

    A start is durably announced immediately before calling predict. A process
    can die between that announcement and the call, so starts are attempts,
    not proof of physical GPU execution. Completion requires a returned predict
    and, on CUDA, a successful device synchronization. Unclosed starts remain
    unknown, never successful or zero-cost. A hard kill leaves a valid prefix.
    """
    stream: TextIO | None = field(default=None, repr=False)
    stage: str = "initializing"
    pair_id: str | None = None
    side: str | None = None
    view: str | None = None
    forwards_started: int = 0
    forwards_completed: int = 0
    views_recorded: int = 0
    prediction_returned: bool | None = None
    sequence: int = 0
    failure_stage: str | None = None
    terminal_error_recorded: bool = False

    def snapshot(self) -> dict:
        return {"schema": "ContextSuiteProgress@1.0.0", "sequence": self.sequence,
                "stage": self.stage, "failure_stage": self.failure_stage,
                "pair_id": self.pair_id, "side": self.side, "view": self.view,
                "forwards_started": self.forwards_started,
                "forwards_completed": self.forwards_completed,
                "forwards_completion_unknown": self.forwards_started-self.forwards_completed,
                "prediction_returned": self.prediction_returned,
                "views_recorded": self.views_recorded, "applied_updates": 0, "budget": 20}

    def emit(self, event: str, stage: str | None = None, **detail) -> None:
        if self.sequence >= 200:
            raise ValueError("context_diagnostics.progress_event_budget")
        if stage is not None:
            self.stage = stage
        self.sequence += 1
        row = {**self.snapshot(), "event": event, "detail": detail}
        serialized = json.dumps(row, sort_keys=True, allow_nan=False)
        if len(serialized.encode("utf-8")) > 4096:
            raise ValueError("context_diagnostics.progress_row_budget")
        if self.stream is not None:
            self.stream.write(serialized + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def failure(self, exc: BaseException) -> None:
        if not self.terminal_error_recorded:
            self.failure_stage = self.stage
            self.terminal_error_recorded = True
            self.emit("execution_aborted" if isinstance(exc, KeyboardInterrupt) else "execution_failed",
                      error_type=type(exc).__name__, error=str(exc)[:1000])


def _predict_accounted(model, batch, device: str, progress: SuiteProgress):
    if progress.forwards_started >= 20:
        raise ValueError("context_diagnostics.forward_budget_exceeded")
    progress.forwards_started += 1
    progress.prediction_returned = False
    progress.emit("forward_started", "model_predict")
    with torch.inference_mode():
        output = model.predict(batch)
    progress.prediction_returned = True
    if device != "cpu":
        progress.emit("prediction_returned", "cuda_forward_synchronization")
        torch.cuda.synchronize(device)
    progress.forwards_completed += 1
    progress.emit("forward_completed", "observation_validation")
    return output


def suite_manifest() -> dict:
    pairs = build_context_pair_suite()
    checks = [validate_context_pair(pair) for pair in pairs]
    if not all(check.mechanically_valid for check in checks):
        raise ValueError("context_diagnostics.invalid_synthetic_pair")
    payload = {"schema": "ContextSyntheticSuite@1.0.0",
               "pairs": [context_pair_to_dict(pair) for pair in pairs],
               "checks": [{"checks": [list(item) for item in check.checks]} for check in checks],
               "musician_review": "not_reviewed", "functional_labels": "not_transferred",
               "preferred_side": None, "local_feature_contract": FEATURE_CONTRACT}
    payload["suite_sha256"] = fingerprint(payload)
    return payload


def checkpoint_suite_protocol(checkpoint_sha256: str, size_bytes: int, device: str) -> dict:
    require_sha(checkpoint_sha256)
    if type(size_bytes) is not int or not 0 < size_bytes <= 64*1024*1024:
        raise ValueError("context_diagnostics.checkpoint_budget")
    if device not in {"cpu", "cuda:0"}:
        raise ValueError("context_diagnostics.explicit_cpu_or_cuda0_required")
    source_root = Path(__file__).resolve().parents[1]
    # Bind all package code: heads, loaders and transitive raw feature helpers
    # can change behavior without changing the architecture/state fingerprints.
    files = sorted(str(path.relative_to(source_root)) for path in source_root.rglob("*.py"))
    with torch.random.fork_rng(devices=[]):
        model_contract_sha = corrected_model_contract()["fingerprint"]
    classes = {task: list(get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels) for task in TASKS}
    source_hashes = {name: sha256((source_root / name).read_bytes()).hexdigest() for name in files}
    wrapper_path = Path(__file__).resolve().parents[3] / "scripts/context_diagnostics_resource_wrapper.py"
    return {"schema": "ContextFrozenSuiteProtocol@1.0.0", "split": "synthetic",
            "checkpoint": {"sha256": checkpoint_sha256, "size_bytes": size_bytes},
            "device": device, "suite_sha256": suite_manifest()["suite_sha256"],
            "model_contract_sha256": model_contract_sha,
            "ordered_classes": classes, "class_registry_sha256": fingerprint(classes),
            "source_sha256": source_hashes,
            "external_resource_wrapper": {
                "path": "scripts/context_diagnostics_resource_wrapper.py",
                "sha256": sha256(wrapper_path.read_bytes()).hexdigest(),
                "required_for_gpu_gate": True, "max_wall_seconds": 600,
                "max_driver_memory_mib": 4096,
                "memory_scope": "sampled whole-device NVIDIA memory.used; not a hard allocation cap",
            },
            "raw_feature_sources_sha256": fingerprint({name: value for name, value in source_hashes.items() if name.startswith("graph/")}),
            "source_snapshot_manifest_sha256": "9491ad6acece3cae0bdf5b15ed7b6017f7dee366359b6d758f349c6e25e3afdf",
            "tasks": list(TASKS), "views": ["full", "one_qn_note_evidence_same_clock_grid"],
            "restriction": "Keep original attacks/durations of notes overlapping focus; retain raw clock grid. Sustains retain their own history/release. Not a receptive-field cutoff.",
            "seed": 17, "seed_scope": "declared parent training seed, not independently extracted from model-only payload; synthetic construction uses no RNG; dropout disabled",
            "progress_contract": {"schema": "ContextSuiteProgress@1.0.0", "max_events": 200,
                                  "max_row_bytes": 4096, "updates": 0,
                                  "start": "durable announced attempt before model.predict",
                                  "completion": "predict returned and CUDA synchronization succeeded",
                                  "unfinished": "completion unknown; completed count is a lower bound after hard kill"},
            "forward_budget": 20, "training_updates": 0,
            "test_access": False, "claim": "sensitivity only; no transferred functional labels or musical preference"}


def run_checkpoint_suite(*, checkpoint: Path, checkpoint_sha256: str, size_bytes: int,
                         device: str, experiment_record: Path, progress: SuiteProgress | None = None) -> dict:
    progress = progress if progress is not None else SuiteProgress()
    try:
        return _run_checkpoint_suite(checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
                                     size_bytes=size_bytes, device=device, experiment_record=experiment_record,
                                     progress=progress)
    except (Exception, KeyboardInterrupt) as exc:
        # Direct API callers retain the same exception type plus accounting;
        # the CLI also persists it in its exclusive failure receipt.
        try:
            progress.failure(exc)
        except (OSError, ValueError) as journal_error:
            exc.add_note(f"Progress journal write failed: {journal_error}")
        exc.context_diagnostic_progress = progress.snapshot()
        raise


def _run_checkpoint_suite(*, checkpoint: Path, checkpoint_sha256: str, size_bytes: int,
                          device: str, experiment_record: Path, progress: SuiteProgress) -> dict:
    progress.emit("stage_started", "protocol_preflight")
    protocol = checkpoint_suite_protocol(checkpoint_sha256, size_bytes, device)
    progress.emit("stage_started", "experiment_record_gate")
    record = load_record(experiment_record)
    policy = record["data_access"]["declared_policy"]
    if (record["status"] != "planned" or not record["experiment_id"].startswith("EXP-CTX-C2-")
        or record.get("protocol", {}).get("context_binding_fingerprint") != fingerprint(protocol)
        or any(policy[key] for key in ("test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed"))):
        raise ValueError("context_diagnostics.matching_planned_record_required")
    # This is the first checkpoint read; all execution and policy gates precede it.
    progress.emit("stage_started", "checkpoint_load")
    loaded = load_model_checkpoint(checkpoint, expected_sha256=checkpoint_sha256, expected_size_bytes=size_bytes)
    progress.emit("stage_started", "device_setup")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise ValueError("context_diagnostics.registered_cuda_unavailable")
    model = loaded.model.to(device).eval()
    if device != "cpu":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    observations = []
    for pair in build_context_pair_suite():
        row = {"pair_id": pair.pair_id, "family": pair.family, "sides": {}}
        for side in ("left", "right"):
            progress.pair_id, progress.side, progress.view = pair.pair_id, side, None
            progress.emit("stage_started", "view_construction")
            piece = getattr(pair, side)
            focus = Candidate("focus", pair.focus_start_qn, pair.focus_end_qn)
            views = {"full": piece, "one_qn_note_evidence_same_clock_grid":
                     restrict_note_context(piece, pair.focus_start_qn, pair.focus_end_qn)}
            side_rows = {}
            for view_id, viewed in views.items():
                progress.view = view_id
                progress.emit("stage_started", "raw_graph_build")
                graph = build_raw_graph(viewed)
                batch = Batch.from_data_list([graph])
                validate_raw_graph_batch(batch, sample_count=1)
                progress.emit("stage_started", "batch_transfer")
                moved = move_raw_graph_batch(batch, device)
                output = _predict_accounted(model, moved, device, progress)
                task_rows = {}
                for task in TASKS:
                    prediction = next(p for p in output.predictions if p.task_id == task)
                    level = "onset" if task == "cadence" else "beat"
                    candidate_id = (f"onset:{pair.focus_start_qn.num}_{pair.focus_start_qn.den}" if level == "onset"
                                    else next(beat.beat_id for beat in viewed.beats
                                              if beat.start_qn <= pair.focus_start_qn < beat.start_qn+beat.duration_qn))
                    graph_index = list(graph[level].entity_id).index(candidate_id)
                    indices = prediction.global_entity_indices.detach().cpu().tolist()
                    if len(indices) != len(set(indices)) or any(i < 0 or i >= graph[level].num_nodes for i in indices):
                        raise ValueError("context_diagnostics.invalid_prediction_candidates")
                    logits = prediction.logits[indices.index(graph_index)].detach().float().cpu()
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError("context_diagnostics.nonfinite_logits")
                    task_rows[task] = {"candidate_id": candidate_id, "logits": logits.tolist(),
                                       "probabilities": logits.softmax(-1).tolist(), "prediction": int(logits.argmax())}
                side_rows[view_id] = {"graph_sha256": graph_fingerprint(graph),
                                      "model_input_sha256": model_input_fingerprint(graph),
                                      "local_features": local_features(viewed, focus), "tasks": task_rows}
                progress.views_recorded += 1
                progress.emit("observation_validated", "observation_recorded")
            row["sides"][side] = side_rows
        progress.emit("stage_started", "pair_summary")
        row["sensitivity"] = {}
        for view_id in protocol["views"]:
            row["sensitivity"][view_id] = {}
            for task in TASKS:
                left = row["sides"]["left"][view_id]["tasks"][task]
                right = row["sides"]["right"][view_id]["tasks"][task]
                row["sensitivity"][view_id][task] = {
                    "argmax_changed": left["prediction"] != right["prediction"],
                    "probability_total_variation": sum(abs(a-b) for a,b in zip(left["probabilities"], right["probabilities"]))/2,
                    "correctness": None,
                }
        observations.append(row)
    progress.emit("stage_started", "suite_summary")
    if device != "cpu":
        torch.cuda.synchronize(device)
    result = {"schema": "ContextFrozenSuiteReport@1.0.0", "experiment_id": record["experiment_id"],
              "protocol": protocol, "checkpoint_evidence": loaded.evidence,
              "observations": observations, "model_forwards": progress.forwards_completed,
              "actual": {"forward_window_wall_seconds": time.perf_counter()-started, "device": device,
                         "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if device != "cpu" else None},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": str(torch.__version__), "pyg": str(pyg_version), "cuda": torch.version.cuda,
                              "device_name": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
                              "torch_threads": torch.get_num_threads(),
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "TEST": {"inputs": False, "targets": False, "metrics": False},
              "correctness_claim": None, "quality_preference_claim": None}
    if progress.forwards_completed != protocol["forward_budget"]:
        raise ValueError("context_diagnostics.forward_budget_mismatch")
    progress.emit("suite_completed", "suite_completed")
    result["progress"] = progress.snapshot()
    result["report_fingerprint"] = fingerprint(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="synthetic")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--suite", action="store_true", help="generate five synthetic pairs; no model forward")
    mode.add_argument("--checkpoint-suite", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="emit exact protocol; no checkpoint read")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-size-bytes", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--progress", type=Path, help="new append-only progress JSONL (default OUTPUT.progress.jsonl)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split != "synthetic":
        raise ValueError("context_diagnostics.suite_synthetic_only_TEST_closed")
    progress_path = args.progress or args.output.with_name(args.output.name+".progress.jsonl")
    if progress_path.resolve() == args.output.resolve():
        raise ValueError("context_diagnostics.progress_output_must_differ")
    if args.checkpoint_suite and progress_path.exists():
        raise FileExistsError(progress_path)
    # Reserve output before any model/source access; preserve failure receipts.
    with args.output.open("x", encoding="utf-8") as stream, ExitStack() as resources:
        progress = SuiteProgress()
        try:
            if args.suite:
                result = suite_manifest()
            elif args.preflight:
                result = checkpoint_suite_protocol(args.checkpoint_sha256, args.checkpoint_size_bytes, args.device)
            else:
                if args.checkpoint is None or args.experiment_record is None:
                    raise ValueError("context_diagnostics.explicit_checkpoint_and_record_required")
                progress.stream = resources.enter_context(progress_path.open("x", encoding="utf-8"))
                result = run_checkpoint_suite(checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
                        size_bytes=args.checkpoint_size_bytes, device=args.device,
                        experiment_record=args.experiment_record, progress=progress)
            progress.emit("stage_started", "output_serialization")
            serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
            progress.emit("stage_started", "output_write")
            stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            progress.emit("result_written", "result_written")
        except (Exception, KeyboardInterrupt) as exc:
            try:
                progress.failure(exc)
            except (OSError, ValueError) as journal_error:
                exc.add_note(f"Progress journal write failed: {journal_error}")
            # Only this still-open, exclusively created output is reset if a
            # success write stopped mid-JSON. Existing evidence cannot be opened
            # here. The durable journal retains the failure stage/accounting.
            stream.seek(0)
            stream.truncate()
            json.dump({"schema": "ContextDiagnosticFailure@1.0.0",
                       "status": "aborted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       "error_type": type(exc).__name__, "error": str(exc),
                       "progress": progress.snapshot()}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
