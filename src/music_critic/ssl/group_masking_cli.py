"""Small executable fixture gate for the opt-in WS2 paired masking adapter.

The wall budget is checked before each update. A hard timeout, including a
stalled update, must be enforced by the external coordinator resource wrapper.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.registry import load_record
from music_critic.ssl.deterministic_runtime import deterministic_cuda_evidence_runtime
from music_critic.ssl.group_masking_fixture import build_group_pilot_fixture
from music_critic.ssl.group_masking_pilot import (
    POLICIES, GroupPilotConfig, GroupMaskingPilot, bind_train_batch, tree_fingerprint,
)
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartSSLModel, export_theory_warmstart_state, load_theory_warmstart,
    load_model_only_theory_warmstart,
)


def implementation_fingerprint(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    paths = sorted((root / "src").rglob("*.py")) + [root / "scripts/run_group_masking_pilot.py"]
    return tree_fingerprint({str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths})


def _json_write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _jsonl_append_durable(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _attempt_totals(attempts):
    receipts = [observation["receipt"] for observation in attempts.values()]
    return {
        "attempted_updates": len(receipts),
        "observed_completed_updates": sum(
            receipt["confirmed_updates"] - receipt["completed_updates_before"]
            for receipt in receipts),
        "observed_encoder_forwards": sum(receipt["observed_encoder_forwards"] for receipt in receipts),
        "optimizer_outcomes": {outcome: sum(receipt["optimizer_outcome"] == outcome for receipt in receipts)
                               for outcome in ("applied", "not_applied", "unknown")},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-fixture", "cuda-fixture"), required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    document = json.loads(config_bytes)
    if document["schema"] != "WS2GroupMaskingPilotPlan@1.0.0":
        raise ValueError("group_pilot.plan_schema_invalid")
    config = replace(GroupPilotConfig(**document["pilot"]), updates=2)
    code_fp = implementation_fingerprint()
    record = None
    if args.mode == "cuda-fixture" and args.source_checkpoint is None:
        raise ValueError("group_pilot.cuda_gate_requires_bound_C2")
    if args.source_checkpoint is not None or args.mode == "cuda-fixture":
        if args.experiment_record is None:
            raise ValueError("group_pilot.planned_record_required_before_checkpoint_or_cuda")
        record = load_record(args.experiment_record)
        if (record["status"] != "planned" or record["kind"] != "engineering_gate"
                or not record["experiment_id"].startswith("EXP-SSL-GROUP-")):
            raise ValueError("group_pilot.planned_engineering_record_required")
        protocol = record["protocol"]
        if (protocol["implementation_fingerprint"] != code_fp
                or protocol["config_file_sha256"] != sha256(config_bytes).hexdigest()
                or protocol["fixture_gate_config"] != config.contract()
                or protocol["source_checkpoint"] != document["source_checkpoint"]):
            raise ValueError("group_pilot.planned_protocol_binding_mismatch")
        policy = record["data_access"]["declared_policy"]
        if any(policy[k] for k in ("validation_allowed", "test_inputs_allowed", "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")):
            raise ValueError("group_pilot.fixture_access_policy_invalid")
    # No checkpoint, CUDA initialization or graph construction above this gate.
    args.output.mkdir(parents=True, exist_ok=False)
    _json_write(args.output / "plan.json", {"config": document, "gate_config": config.contract(),
        "implementation_fingerprint": code_fp, "mode": args.mode,
        "wall_budget": {"seconds": 600, "enforcement": "pre_step_elapsed_time_check",
                        "hard_timeout": "external_coordinator_resource_wrapper_required"},
        "experiment_id": None if record is None else record["experiment_id"]})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    execution_rows = []
    attempts = {}
    (args.output / "execution.jsonl").touch(exist_ok=False)
    (args.output / "attempts.jsonl").touch(exist_ok=False)

    def execute_update(arm, pass_name):
        # This deliberately makes no claim of interrupting an in-flight update.
        if time.monotonic() - started > 600:
            raise TimeoutError("group_pilot.fixture_gate_wall_budget")
        attempt_id = f"execution-attempt-{len(attempts) + 1:06d}"

        def record_attempt(receipt):
            observation = {"execution_attempt_id": attempt_id, "policy": arm.policy,
                           "pass": pass_name, "receipt": deepcopy(receipt)}
            # Retain observations even if the journal write itself fails. The
            # failure report can then describe what was seen without rollback claims.
            attempts[attempt_id] = observation
            _jsonl_append_durable(args.output / "attempts.jsonl", observation)

        try:
            row = arm.step_from_factory(lambda: batch, on_attempt=record_attempt)
        except BaseException:
            # A callback can fail after an update was confirmed. Use the pilot's
            # current count without inferring whether an interrupted Adam applied.
            if arm.last_attempt is not None:
                receipt = deepcopy(arm.last_attempt)
                receipt["confirmed_updates"] = arm.completed_updates
                observation = {"execution_attempt_id": attempt_id, "policy": arm.policy,
                               "pass": pass_name, "receipt": receipt}
                if observation != attempts.get(attempt_id):
                    attempts[attempt_id] = observation
                    try:
                        _jsonl_append_durable(args.output / "attempts.jsonl", observation)
                    except BaseException:
                        pass  # Preserve the original update or receipt-write failure.
            raise
        observation = {"execution_attempt_id": attempt_id, "policy": arm.policy,
                       "pass": pass_name, "row": row}
        _jsonl_append_durable(args.output / "execution.jsonl", observation)
        execution_rows.append(observation)
        return row
    device = "cuda:0" if args.mode == "cuda-fixture" else "cpu"
    context = deterministic_cuda_evidence_runtime() if args.mode == "cuda-fixture" else nullcontext()
    try:
        with context:
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise ValueError("group_pilot.cuda_unavailable")
                torch.empty(1, device=device)
                torch.cuda.reset_peak_memory_stats(device)
            batch = build_group_pilot_fixture()
            schedule = (bind_train_batch(batch, config),) * config.updates
            source = document["source_checkpoint"]
            if args.source_checkpoint is None:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(3)
                    export = export_theory_warmstart_state(CorrectedAnalysisGNNModel())
                source = {"kind": "synthetic", "state_fingerprint": export["state_fingerprint"]}
            def make_arm(policy):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(config.seed)
                    destination = TheoryWarmstartSSLModel(config.ssl_config())
                    if args.source_checkpoint is None:
                        warmstart = load_theory_warmstart(destination, export)
                    else:
                        warmstart, _ = load_model_only_theory_warmstart(destination, args.source_checkpoint,
                            expected_sha256=source["sha256"], expected_size_bytes=source["size_bytes"],
                            expected_state_fingerprint=source["state_fingerprint"])
                return GroupMaskingPilot(warmstart, config, policy=policy, schedule=schedule,
                                        source_binding=source, device=device)
            arms = {}
            initial = None
            for policy in POLICIES:
                arm = make_arm(policy)
                if initial is not None and arm.initial_fingerprint != initial:
                    raise ValueError("group_pilot.paired_full_initial_state_mismatch")
                initial = arm.initial_fingerprint
                # Full paired starts include fresh SSL and inactive state.
                _json_write(args.output / (policy + "-binding.json"), arm.binding)
                for _ in range(config.updates):
                    execute_update(arm, "uninterrupted")
                evidence = arm.save(args.output / (policy + "-final.pt"))
                uninterrupted_rows = list(arm.rows)
                final_model_fp = tree_fingerprint(arm.harness.state_dict())
                final_optimizer_fp = tree_fingerprint(arm.optimizer.state_dict())
                del arm
                interrupted = make_arm(policy)
                execute_update(interrupted, "before_interruption")
                middle = interrupted.save(args.output / (policy + "-update1.pt"))
                del interrupted
                resumed = make_arm(policy)
                resumed.resume(middle["path"], expected_sha256=middle["sha256"], expected_size_bytes=middle["size_bytes"])
                execute_update(resumed, "after_resume")
                if (resumed.rows != uninterrupted_rows or tree_fingerprint(resumed.harness.state_dict()) != final_model_fp
                        or tree_fingerprint(resumed.optimizer.state_dict()) != final_optimizer_fp):
                    raise ValueError("group_pilot.interrupted_resume_not_exact")
                arms[policy] = {"rows": uninterrupted_rows, "final_checkpoint": evidence,
                                "initial_fingerprint": initial, "resume_exact": True}
                del resumed
            for a, b in zip(arms[POLICIES[0]]["rows"], arms[POLICIES[1]]["rows"], strict=True):
                if a["paired_masks"] != b["paired_masks"] or a["batch_binding"] != b["batch_binding"]:
                    raise ValueError("group_pilot.paired_schedule_or_mask_evidence_changed")
            totals = _attempt_totals(attempts)
            report = {"schema": "WS2GroupMaskingFixtureGate@1.1.0", "status": "completed",
                "source": source, "implementation_fingerprint": code_fp, "arms": arms,
                "synthetic_batches_only": True, "quality_evidence": False, "TEST_access": False,
                "VALIDATION_access": False, "device": device, "amp": False,
                **totals,
                "recorded_successful_updates": len(execution_rows),
                "actual_updates_including_resume_replays": totals["observed_completed_updates"],
                "actual_encoder_forwards_including_resume_replays": totals["observed_encoder_forwards"],
                "actual_ssl_draws_including_resume_replays": sum(r["row"]["counts"]["ssl_samples"] for r in execution_rows),
                "actual_theory_draws_including_resume_replays": sum(r["row"]["counts"]["theory_samples"] for r in execution_rows),
                "comparison_updates": sum(r["pass"] == "uninterrupted" for r in execution_rows),
                "resume_replay_updates": sum(r["pass"] != "uninterrupted" for r in execution_rows),
                "wall_seconds": time.monotonic() - started,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.startswith("cuda") else None,
                "scientific_data_factory_ready": False, "scientific_validation_evaluator_ready": False}
            _json_write(args.output / "result.json", report)
            print(json.dumps({"status": "completed", "result": str(args.output / "result.json"),
                              "quality_evidence": False, "resume_exact": True}))
            return 0
    except BaseException as exc:
        _json_write(args.output / "failure.json", {"status": "failed", "type": type(exc).__name__,
            "reason": str(exc), "wall_seconds": time.monotonic() - started,
            **_attempt_totals(attempts), "recorded_successful_updates": len(execution_rows),
            "last_attempt": next(reversed(attempts.values())) if attempts else None,
            "instruction": "Retain evidence; coordinator must finalize launched record before retry with new ID."})
        raise
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    sys.exit(main())
