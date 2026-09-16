"""Concrete coordinator setup for the already agreed reference gate and P512 pair.

This creates plans/requests, never launches training or changes terminal records.
The fixed remote locators and hashes were supplied by the coordinator. Source,
handoff, runtime and fresh output roots remain explicit command arguments.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from .harmonic_fusion_pilot import (ARMS, THREAD_ENV, artifact_pin, checked_bytes,
    checked_json, stable_bindings, validate_runtime_policy, verify_source_manifest,
    runtime_observations, admit_request)
from .harmonic_fusion_run import _write_json_once
from ..registry import seal_record, canonical_json_bytes, write_record_once

GPU_UUID = "GPU-1d7c6d9f-751f-7ea8-a951-d1b8a7991b42"
C2_SHA = "a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797"
C2_STATE = "82ed08304b565a1447062a95112c8a51b47ce75c8f46f07b858443236ba83f39"
CACHE_SHA = "5ad21b1f9bad7cfeac2a098866724b5d80d08361d0e5d3188b5fb298086cc6ee"
WEIGHTS_SHA = "12b8b812e138af45e0ca2f7926bf2de3a53872368f165b3ead24b6aa9140dd34"


def pin(path, digest, size):
    return {"path": str(Path(path).absolute()), "sha256": digest, "size_bytes": size}


def source_manifest(repository_root, output):
    root = Path(repository_root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents, *(root / folder for folder in ("src", "scripts", "tests", "configs")))):
        raise ValueError("fusion_dispatch.source_symlink_forbidden")
    files = [root / "pyproject.toml"]
    for folder in ("src", "scripts", "tests", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("fusion_dispatch.source_symlink_forbidden")
            if path.is_file():
                files.append(path)
    rows = [{**artifact_pin(path), "path": str(path.relative_to(root))} for path in sorted(files)]
    _write_json_once(Path(output), {"files": rows})
    result = artifact_pin(output)
    verify_source_manifest(result, root)
    return result


def planned_record(experiment_id, *, protocol, updates, draws, validation, parents):
    record = {"schema_version": "1.0.0", "experiment_id": experiment_id,
        "kind": "paired_validation_comparison" if validation else "engineering_gate",
        "phase": "9E-FUSION", "title": "Bound registered fusion " + ("P512 arm" if validation else "TRAIN reference smoke"),
        "status": "planned", "question": "Can the already agreed bound protocol execute on its declared inputs?",
        "hypothesis": "The registered fixed protocol remains valid under the observed target environment.",
        "decision_use": "No promotion without terminal evidence; no TEST access.",
        "repository": {"git_commit": None, "dirty": True,
            "source_manifest_sha256": protocol["fusion_pilot_bindings"]["source_manifest_sha256"]},
        "seeds": [17], "budget": {"planned": updates, "planned_draws": draws,
            "batch_size": 2, "actual": None}, "protocol": protocol, "parent_ids": list(parents),
        "data_access": {"access_evidence_status": "not_applicable_planning_record",
            **{key: None for key in ("validation_accessed", "test_inputs_accessed", "test_targets_accessed",
                                   "test_metrics_computed", "test_used_for_selection")},
            "declared_policy": {"validation_allowed": validation, "test_inputs_allowed": False,
                "test_targets_allowed": False, "test_metrics_allowed": False, "test_selection_allowed": False,
                "source": "docs/workstreams/2026-09-13/02_LOGIT_FUSION_DISPATCH_004.md"}},
        "results": {}, "validity": {"valid": None, "reason": "planned_not_executed"},
        "claims": {"supported": [], "not_supported": ["execution", "musical benefit", "TEST performance"]},
        "artifacts": [], "import_provenance": {"mode": "coordinator_bound_setup"}}
    return seal_record(record)


def capture_runtime(output):
    """Target-only metadata probe; zero model construction/forward/update."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID or any(os.environ.get(k) != "1" for k in THREAD_ENV):
        raise ValueError("fusion_dispatch.reserved_uuid_and_native_threads_required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("fusion_dispatch.cublas_environment_required")
    import torch
    from .harmonic_fusion_resume import prepare_training_resume_runtime
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    prepare_training_resume_runtime("cuda", expected_cuda_device_count=1)
    runtime = runtime_observations("cuda")
    validate_runtime_policy(runtime)
    _write_json_once(Path(output), runtime)
    return artifact_pin(output)


def prepare_reference(*, server_root, repository_root, handoff_root, runtime_pin,
                      output_root, records_root, comparison_id, experiment_id):
    from .harmonic_fusion_prepare import prepare_data_manifest
    from .harmonic_fusion_reference import (REQUEST_SCHEMA, reference_protocol,
        reference_gate_bindings, admit_reference_request)
    root, source, handoff = map(lambda p: Path(p).absolute(), (server_root, repository_root, handoff_root))
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "split_assignments": pin("/home/humtech/music_critic_v2/outputs/phase9eb3/analysisgnn-multitask-contract-01290f5/split_assignments.jsonl", "056de0942918976981401db5017f6e567bc8949bae91ae395f0bd669adbfd51a", 528617),
        "shift_eligibility": pin("/home/humtech/music_critic_v2/outputs/phase9eb5a/analysisgnn-transposition-a6a2796/record_shift_eligibility.jsonl", "3ab036e5f6c58104ccbdfc41d259e02e3f7a9108142c62ff99bfab8001fb6195", 6594901),
        "class_weights": pin(source / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json", WEIGHTS_SHA, 303658),
        "joint_tuple_counts": pin("/home/humtech/music_critic_v2/outputs/phase9eb4/analysisgnn-class-balance-671097b/joint_tuple_counts.jsonl", "32cd9a4396bdb68c2077b68ca0efd0cd31d400e7c3385ca000d9b1206e38686b", 3031726)}
    _write_json_once(output / "metadata-pins.json", metadata)
    prepare_data_manifest(cache_index=pin(root / "evidence/EXP-RESEARCH-DILEMMA-BINDINGS-001/cache-bindings.jsonl", CACHE_SHA, 1233467),
        metadata_pins=artifact_pin(output / "metadata-pins.json"), output=output / "data-manifest.json")
    checkpoint = pin(root / "c2-model-only.pt", C2_SHA, 14844299)
    checked_bytes(checkpoint)  # byte hash only; no pickle/tensor load
    request = {"schema_version": REQUEST_SCHEMA, "comparison_id": comparison_id,
        "data_manifest": artifact_pin(output / "data-manifest.json"),
        "config": artifact_pin(source / "configs/experiments/harmonic_fusion_pilot.json"),
        "source_checkpoint": checkpoint, "source_model_state_fingerprint": C2_STATE,
        "source_manifest": source_manifest(source, output / "source-manifest.json"),
        "runtime": checked_json(runtime_pin), "gate": {
            "record": artifact_pin(handoff / "cuda002-history/EXP-FUSION-CUDA-SYNTHETIC-002.json"),
            "report": artifact_pin(handoff / "cuda002-history/report.json"),
            "historical_source_manifest": artifact_pin(handoff / "cuda002-history/deployment-manifest.json")},
        "max_wall_seconds": 600}
    stable = stable_bindings(request)
    record = planned_record(experiment_id, protocol={"fusion_pilot_bindings": reference_gate_bindings(stable),
        "reference_train_smoke": reference_protocol(stable)}, updates=2, draws=4, validation=False,
        parents=["EXP-FUSION-CUDA-SYNTHETIC-002"])
    path = Path(records_root) / (experiment_id + ".json")
    write_record_once(path, record)
    planned_copy = output / "planned-records" / path.name
    write_record_once(planned_copy, record)
    request["experiment_record"] = artifact_pin(planned_copy)
    admit_reference_request(request, repository_root=source)
    _write_json_once(output / "reference-request.json", request)
    return artifact_pin(output / "reference-request.json")


def prepare_pair(*, reference_request_pin, reference_record_pin, reference_report_pin,
                 repository_root, output_root, records_root, attempt_id, control_id, fusion_id):
    reference = checked_json(reference_request_pin)
    request = {key: deepcopy(reference[key]) for key in ("comparison_id", "data_manifest", "config",
        "source_checkpoint", "source_model_state_fingerprint", "source_manifest", "runtime")}
    request.update(schema_version="HarmonicFusionPilotRequest@1.0.0", attempt_id=attempt_id,
        gates={"cuda_synthetic": reference["gate"], "real_train_reference": {
            "record": reference_record_pin, "report": reference_report_pin}}, parent=None,
        experiment_records={}, max_wall_seconds=7200)
    from .harmonic_fusion_pilot import _record, verify_gate
    completed = _record(reference_record_pin, terminal=True)
    if completed["experiment_id"] != _record(reference["experiment_record"])["experiment_id"]:
        raise ValueError("fusion_dispatch.reference_record_identity_mismatch")
    verify_gate(request["gates"]["real_train_reference"], role="real_train_reference",
        bindings=stable_bindings(request), runtime=request["runtime"], repository_root=repository_root)
    if control_id == fusion_id:
        raise ValueError("fusion_dispatch.distinct_arm_ids_required")
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=False)
    for arm, experiment_id in zip(ARMS, (control_id, fusion_id), strict=True):
        path = Path(records_root) / (experiment_id + ".json")
        record = planned_record(experiment_id,
            protocol={"fusion_pilot_bindings": {"arm": arm, **stable_bindings(request)}},
            updates=512, draws=1024, validation=True, parents=[completed["experiment_id"]])
        write_record_once(path, record)
        planned_copy = output / "planned-records" / path.name
        write_record_once(planned_copy, record)
        request["experiment_records"][arm] = artifact_pin(planned_copy)
    admit_request(request, repository_root=repository_root)
    _write_json_once(output / "pilot-request.json", request)
    return artifact_pin(output / "pilot-request.json")
