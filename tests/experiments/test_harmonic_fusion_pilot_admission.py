"""Public admission with real registry/hash checks and fabricated gate metadata.

Every CUDA and real-TRAIN observation below is an explicitly fabricated unit
fixture. Nothing in this module is an engineering gate or scientific evidence.
Only FrozenFusionData is replaced: no corpus, checkpoint, or CUDA is needed.
"""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_data as data_module
from music_critic.experiments.analysisgnn import harmonic_fusion_pilot as pilot
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.harmonic_fusion_training import FusionTrainingConfig
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, seal_record, validate_record,
)


ROOT = Path(__file__).resolve().parents[2]
TEST_FLAGS = {key: False for key in (
    "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed")}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return pilot.artifact_pin(path)


def _registered_artifact(pin):
    return {"role": "fabricated_unit_fixture", "sha256": pin["sha256"],
            "size_bytes": pin["size_bytes"], "availability": "unit_fixture_only",
            "locator": "unit-fixtures/" + Path(pin["path"]).name}


def _record(experiment_id, *, status="planned", bindings, artifacts=(), parent_ids=()):
    planned = status == "planned"
    record = {
        "schema_version": "1.0.0", "experiment_id": experiment_id,
        "kind": "paired_validation_comparison" if planned else "engineering_gate",
        "phase": "9E-FUSION", "title": "Fabricated admission unit fixture",
        "status": status, "question": "Does metadata admission enforce its bindings?",
        "hypothesis": "Malformed bindings are rejected before data construction.",
        "decision_use": "Unit testing only; no execution or scientific evidence.",
        "repository": {"git_commit": "a" * 40, "dirty": True}, "seeds": [17],
        "budget": {"planned": 512, "planned_draws": 1024} if planned else {"actual": 24},
        "protocol": {"fusion_pilot_bindings": bindings}, "parent_ids": list(parent_ids),
        "data_access": {
            "access_evidence_status": "not_applicable_planning_record" if planned else "verified_primary_evidence",
            **{key: None if planned else False for key in DATA_ACCESS_FIELDS},
            "declared_policy": {"validation_allowed": True, "test_inputs_allowed": False,
                "test_targets_allowed": False, "test_metrics_allowed": False,
                "test_selection_allowed": False, "source": "fabricated unit fixture"}},
        "results": {} if planned else {"fixture_only": True},
        "validity": {"valid": None if planned else True, "reason": "fabricated unit fixture"},
        "claims": {"supported": [], "not_supported": ["CUDA gate", "real TRAIN access", "quality"]},
        "artifacts": [_registered_artifact(pin) for pin in artifacts],
        "import_provenance": {"mode": "fabricated_unit_fixture"},
    }
    return seal_record(record)


def _write_record(path, value):
    value = seal_record(value)
    validate_record(value, filename=path.name)
    return _write_json(path, value)


def _runtime():
    # This is a metadata acceptance fixture, never an observed CUDA environment.
    return {"python": "fixture-python", "platform": "fixture-platform",
        "torch": "fixture-torch", "numpy": "fixture-numpy",
        "packages": {"torch": "fixture-torch", "numpy": "fixture-numpy", "torch-geometric": "fixture-pyg"},
        "torch_cuda_build": "fixture-cuda", "cudnn_version": 1,
        "cuda_available": True, "cuda_device_count": 1, "gpu": "fixture-gpu",
        "cublas_workspace_config": ":4096:8", "device": "cuda", "torch_num_threads": 1,
        "torch_num_interop_threads": 1, "native_threads": {key: "1" for key in pilot.THREAD_ENV},
        "deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cudnn_allow_tf32": False, "matmul_allow_tf32": False}


def _gate_report(role, runtime, stable=None):
    # Native report field shapes; all observations are deliberately fabricated.
    report = {"fixture_only": True, "passed": True, "device": "cuda", **TEST_FLAGS,
        "actual_updates": 24, "actual_draws": 48, "real_data_accessed": False,
        "environment": deepcopy(runtime), "config": {"validation_mode": "required"}}
    if role == "cuda_synthetic":
        report["schema_version"] = "HarmonicFusionSyntheticTrainerGate@1.1.0"
        report["arms"] = []
        for arm in pilot.ARMS:
            checkpoint = {"sha256": ("e" if arm == "control" else "f") * 64, "size_bytes": 100}
            report["arms"].append({"arm": arm, "actual_updates": 12, "actual_draws": 24,
                "committed_final_update": 6, "sampler_position": 12, "validation_updates": [0, 3, 6],
                "interrupted_matches_uninterrupted": True, "final_training_checkpoint": checkpoint,
                "learning_diagnostics": {"model_state_changed": True,
                    "fusion_weight_change": {"changed_parameters": 1, "delta_l2": 0.1,
                        "nonzero_output_head_count": 18, "optimizer_steps": {"fusion.weight": 6}}},
                "fusion_export": None if arm == "control" else {"sha256": "a" * 64,
                    "size_bytes": 50, "parent_training_checkpoint": deepcopy(checkpoint)}})
    elif role == "cuda_fresh_process":
        report.update(schema_version="HarmonicFusionFreshProcessGate@1.0.0",
                      fresh_process_resume=True, parent_process_id=100)
        report["arms"] = [{"arm": arm, "whole_state_equal": True, "worker_process_id": 101 + index,
            "worker_runtime_readiness": {"device_type": "cuda", "cuda_device_count": 1,
                "verified_cuda_ordinals": [0], "saved_rng_applied": False}}
            for index, arm in enumerate(pilot.ARMS)]
    else:
        from music_critic.experiments.analysisgnn import harmonic_fusion_reference as reference
        draws = [{"orbit_epoch": 0, "orbit_index": index, "record_id": "fixture-a",
                  "component_id": "component-fixture-a", "shift_pc": index} for index in range(2)]
        schedule = fingerprint(draws)
        report.update(schema_version=reference.REPORT_SCHEMA, actual_updates=2, actual_draws=4,
            real_data_accessed=True, reference_smoke_passed=True, experiment_id="FIXTURE-GATE-" + role,
            source_hashes_verified=True, raw_only_prediction_verified=True, target_alignment_verified=True,
            attempted_updates=2, attempted_draw_slots=4, reader_calls=4, reader_completed=4,
            validation_reader_calls=0, synthetic=False, resumable=False, scientific_acceptance=False,
            unconfirmed_in_flight_tail=False, reference_validation_policy=reference.VALIDATION_POLICY,
            config=FusionTrainingConfig().to_dict(), draw_schedule_fingerprint=schedule,
            bindings={**stable, "reference_draw_schedule_fingerprint": schedule}, **reference._ACCESS)
        report["arms"] = [{"arm": arm, "actual_updates": 1, "actual_draws": 2, "sampler_position": 2,
            "source_model_state": stable["source_model_state_fingerprint"], "raw_prediction_calls": 1,
            "target_alignment_checks": 1, "validation_updates": [], "draws": deepcopy(draws),
            "draw_schedule_fingerprint": schedule} for arm in pilot.ARMS]
        report["state"] = {"arms": {arm: {"attempted_updates": 1, "applied_updates": 1, "reader_calls": 2,
            "reader_completed": 2, "raw_prediction_calls": 1, "target_alignment_checks": 1,
            "update_in_flight": False} for arm in pilot.ARMS}}
    return report


@pytest.fixture
def admission(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    files = {"pyproject.toml": b"# unit fixture\n", "src/model.py": b"# unit model\n",
        "scripts/run_harmonic_fusion_real_pilot.py": (ROOT / "scripts/run_harmonic_fusion_real_pilot.py").read_bytes(),
        "configs/experiments/harmonic_fusion_pilot.json": canonical_json_bytes(asdict(FusionTrainingConfig())) + b"\n"}
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    source = _write_json(tmp_path / "source-manifest.json", {"files": [
        {**pilot.artifact_pin(repo / name), "path": name} for name in sorted(files)]})
    config = pilot.artifact_pin(repo / "configs/experiments/harmonic_fusion_pilot.json")
    data_pin = _write_json(tmp_path / "data-manifest.json", {"fixture_only": True})
    # Deliberately absent: successful metadata admission must never open a checkpoint.
    checkpoint = {"path": str(tmp_path / "absent-source.pt"), "sha256": "c" * 64, "size_bytes": 10}
    runtime = _runtime()
    stable = {"comparison_id": "fixture-pair", "data_manifest_sha256": data_pin["sha256"],
        "config_sha256": config["sha256"], "source_manifest_sha256": source["sha256"],
        "source_checkpoint_sha256": checkpoint["sha256"], "source_checkpoint_size_bytes": checkpoint["size_bytes"],
        "source_model_state_fingerprint": "d" * 64, "runtime_fingerprint": fingerprint(runtime),
        "reference_path": "sequential_historical_C2"}
    request = {"schema_version": pilot.REQUEST_SCHEMA, "comparison_id": "fixture-pair",
        "attempt_id": "fixture-attempt", "data_manifest": data_pin, "config": config,
        "source_checkpoint": checkpoint, "source_model_state_fingerprint": "d" * 64,
        "source_manifest": source, "runtime": runtime, "gates": {}, "experiment_records": {},
        "parent": None, "max_wall_seconds": 7200}
    for arm in pilot.ARMS:
        name = "FIXTURE-PLAN-" + arm
        request["experiment_records"][arm] = _write_record(tmp_path / f"{name}.json",
            _record(name, bindings={"arm": arm, **stable}))
    for role in ("cuda_synthetic", "cuda_fresh_process", "real_train_reference"):
        report_pin = _write_json(tmp_path / f"{role}-report.json", _gate_report(role, runtime, stable))
        selected = ("source_manifest_sha256", "runtime_fingerprint") if role.startswith("cuda_") else (
            "source_manifest_sha256", "runtime_fingerprint", "data_manifest_sha256",
            "source_checkpoint_sha256", "source_model_state_fingerprint", "reference_path")
        name = "FIXTURE-GATE-" + role
        gate_record = _record(name, status="completed",
            bindings={"role": role, **{key: stable[key] for key in selected}}, artifacts=(report_pin,))
        if role == "real_train_reference":
            from music_critic.experiments.analysisgnn.harmonic_fusion_reference import reference_protocol
            gate_record["protocol"]["reference_train_smoke"] = reference_protocol(stable)
            gate_record["budget"] = {"planned": 2, "planned_draws": 4, "actual": 2}
            gate_record["data_access"]["declared_policy"]["validation_allowed"] = False
        record_pin = _write_record(tmp_path / f"{name}.json", gate_record)
        request["gates"][role] = {"record": record_pin, "report": report_pin}

    table = tuple(FullOrbitPair(record, shift, "component-" + record)
                  for record in ("fixture-a", "fixture-b") for shift in range(12))
    events = []

    class FixtureData:
        def __init__(self, manifest):
            assert manifest == {"fixture_only": True}
            events.append(("construct_data", deepcopy(manifest)))
            self.table = table
            self.bindings = {"fixture_data_binding": fingerprint(manifest)}

        def require_train_records(self, record_ids):
            events.append(("require_train_records", set(record_ids)))
            assert record_ids == {"fixture-a", "fixture-b"}

    monkeypatch.setattr(data_module, "FrozenFusionData", FixtureData)

    def denied(*args, **kwargs):
        pytest.fail("metadata admission accessed CUDA or checkpoint tensors")

    monkeypatch.setattr(torch, "load", denied)
    monkeypatch.setattr(torch.cuda, "init", denied)
    monkeypatch.setattr(torch.cuda, "is_available", denied)
    return SimpleNamespace(root=tmp_path, repo=repo, request=request, stable=stable, events=events, table=table)


def _admit(fixture):
    return pilot.admit_request(fixture.request, repository_root=fixture.repo)


def _replace_report(fixture, role, mutate):
    gate = fixture.request["gates"][role]
    report = pilot.checked_json(gate["report"])
    mutate(report)
    gate["report"] = _write_json(Path(gate["report"]["path"]), report)
    record = pilot.checked_json(gate["record"])
    record["artifacts"] = [_registered_artifact(gate["report"])]
    gate["record"] = _write_record(Path(gate["record"]["path"]), record)


def test_public_admission_binds_canonical_plans_reports_and_full_draw_prefix(admission):
    result = _admit(admission)
    assert admission.request["max_wall_seconds"] == 7200
    sampler = FullOrbitSampler(admission.table, seed=17)
    expected = [asdict(sampler.peek(i)) for i in range(1024)]
    assert result["bindings"] == {"fixture_data_binding": fingerprint({"fixture_only": True}),
        **admission.stable, "device": "cuda", "draw_schedule_fingerprint": fingerprint(expected)}
    assert asdict(result["config"]) == asdict(FusionTrainingConfig())
    assert result["experiment_ids"] == {arm: "FIXTURE-PLAN-" + arm for arm in pilot.ARMS}
    assert result["parent"] is None
    assert admission.events == [("construct_data", {"fixture_only": True}),
                                ("require_train_records", {"fixture-a", "fixture-b"})]


@pytest.mark.parametrize("wall_seconds", [2400, 600, 7199, 7201, 7200.0, "7200", True, None])
def test_public_admission_rejects_other_pair_wall_budgets_before_artifact_reads(admission, monkeypatch, wall_seconds):
    admission.request["max_wall_seconds"] = wall_seconds

    def forbidden(*args, **kwargs):
        pytest.fail("invalid pair wall budget must fail before artifact reads")

    monkeypatch.setattr(pilot, "checked_bytes", forbidden)
    with pytest.raises(ValueError, match="fixed_pair_wall_budget_required"):
        _admit(admission)
    assert admission.events == []


def test_real_cli_defaults_to_public_admission_without_tensor_or_cuda_access(admission, capsys):
    entrypoint = admission.repo / "scripts/run_harmonic_fusion_real_pilot.py"
    spec = importlib.util.spec_from_file_location("fixture_real_pilot_cli", entrypoint)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    request_pin = _write_json(admission.root / "request.json", admission.request)
    assert cli.main(["--request", request_pin["path"], "--request-sha256", request_pin["sha256"],
                     "--request-size-bytes", str(request_pin["size_bytes"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["admission_valid"] is True
    assert report["executed"] is report["cuda_initialized"] is False
    assert report["raw_graphs_loaded"] == report["checkpoint_tensors_loaded"] == 0
    assert report["bindings"]["source_manifest_sha256"] == admission.stable["source_manifest_sha256"]


def test_initial_pair_does_not_require_repeated_fresh_process_synthetic(admission):
    del admission.request["gates"]["cuda_fresh_process"]
    assert _admit(admission)["parent"] is None


def test_resume_requires_completed_fresh_process_gate_before_parent_or_data_reads(admission):
    del admission.request["gates"]["cuda_fresh_process"]
    admission.request["parent"] = {"deliberately_unread": True}
    with pytest.raises(ValueError, match="required_gates_missing"):
        _admit(admission)
    assert admission.events == []


@pytest.mark.parametrize("mutation,reason", [
    ("changed_file", "artifact_hash_mismatch"),
    ("new_source", "source_manifest_incomplete"),
    ("missing_source", "source_manifest_incomplete"),
    ("wrong_config", "fixed_p512_config_required"),
    ("runtime_policy", "runtime_policy_invalid"),
    ("plan_binding", "planned_pair_binding_mismatch"),
    ("noncanonical_record", "canonical_record_required"),
    ("unsealed_record", "fingerprint_mismatch"),
])
def test_public_admission_rejects_source_config_runtime_or_record_drift_before_data(admission, mutation, reason):
    if mutation == "changed_file":
        (admission.repo / "src/model.py").write_text("# changed after seal\n")
    elif mutation == "new_source":
        (admission.repo / "src/unlisted.py").write_text("# unbound code\n")
    elif mutation == "missing_source":
        manifest = pilot.checked_json(admission.request["source_manifest"])
        manifest["files"] = [row for row in manifest["files"] if row["path"] != "src/model.py"]
        admission.request["source_manifest"] = _write_json(admission.root / "incomplete.json", manifest)
    elif mutation == "wrong_config":
        config = {**asdict(FusionTrainingConfig()), "total_updates": 256}
        admission.request["config"] = _write_json(admission.root / "wrong-config.json", config)
    elif mutation == "runtime_policy":
        admission.request["runtime"]["deterministic_algorithms"] = False
    else:
        pin = admission.request["experiment_records"]["fusion"]
        record = pilot.checked_json(pin)
        if mutation == "plan_binding":
            record["protocol"]["fusion_pilot_bindings"]["source_model_state_fingerprint"] = "e" * 64
            pin = _write_record(Path(pin["path"]), record)
        elif mutation == "unsealed_record":
            record["title"] = "Changed without resealing"
            pin = _write_json(Path(pin["path"]), record)
        else:
            Path(pin["path"]).write_text(json.dumps(record, indent=2) + "\n")
            pin = pilot.artifact_pin(pin["path"])
        admission.request["experiment_records"]["fusion"] = pin
    with pytest.raises(ValueError, match=reason):
        _admit(admission)
    assert admission.events == []


@pytest.mark.parametrize("role", ["cuda_synthetic", "cuda_fresh_process", "real_train_reference"])
def test_bound_record_cannot_hide_different_observed_gate_runtime(admission, role):
    _replace_report(admission, role, lambda report: report["environment"].update(torch_cuda_build="different-build"))
    with pytest.raises(ValueError, match="gate_observed_runtime_mismatch"):
        _admit(admission)
    assert admission.events == []


def test_unrelated_one_step_cuda_report_does_not_satisfy_synthetic_gate(admission):
    _replace_report(admission, "cuda_synthetic", lambda report: report.update(
        schema_version="UnrelatedCudaOneStep@1.0.0", actual_updates=1, actual_draws=2))
    with pytest.raises(ValueError, match="paired_synthetic_gate_required"):
        _admit(admission)
    assert admission.events == []


def _attach_parent(fixture, *, link_children):
    admitted = _admit(fixture)
    fixture.events.clear()
    root = fixture.root / "parent"
    root.mkdir()
    ids = {arm: "FIXTURE-PARENT-" + arm for arm in pilot.ARMS}
    journal = pilot.AttemptJournal(root, attempt_id="fixture-parent-attempt", bindings=admitted["bindings"],
                                   experiment_ids=ids, parent=None)
    journal.state["status"] = "failed"
    journal.append("exception", "failed", reason="fabricated unit fixture")
    pin = pilot.artifact_pin(journal.path)
    parents = {arm: _write_record(root / f"{name}.json", _record(name, status="failed",
        bindings={"arm": arm, **fixture.stable}, artifacts=(pin,))) for arm, name in ids.items()}
    fixture.request["parent"] = {"journal": pin, "experiment_records": parents}
    if link_children:
        for arm in pilot.ARMS:
            child = fixture.request["experiment_records"][arm]
            record = pilot.checked_json(child)
            record["parent_ids"] = [ids[arm]]
            fixture.request["experiment_records"][arm] = _write_record(Path(child["path"]), record)
    return ids


def test_public_continuation_accepts_registered_parent_journal_and_explicit_links(admission):
    ids = _attach_parent(admission, link_children=True)
    result = _admit(admission)
    assert result["parent"]["experiment_ids"] == ids
    assert result["parent"]["attempt_id"] == "fixture-parent-attempt"
    assert result["parent"]["status"] == "failed"


def test_public_continuation_rejects_missing_child_parent_record_link(admission):
    _attach_parent(admission, link_children=False)
    with pytest.raises(ValueError, match="child_parent_record_link_missing"):
        _admit(admission)


def test_public_continuation_rejects_unregistered_parent_journal(admission):
    _attach_parent(admission, link_children=True)
    pin = admission.request["parent"]["experiment_records"]["control"]
    record = pilot.checked_json(pin)
    record["artifacts"] = []
    admission.request["parent"]["experiment_records"]["control"] = _write_record(Path(pin["path"]), record)
    with pytest.raises(ValueError, match="parent_journal_not_registered"):
        _admit(admission)
