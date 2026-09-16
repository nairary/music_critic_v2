"""Fabricated native-shaped metadata only; no CUDA/real-TRAIN execution."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_reference as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.harmonic_fusion_dispatch import planned_record
from music_critic.experiments.analysisgnn.harmonic_fusion_pilot import artifact_pin, stable_bindings
from music_critic.experiments.analysisgnn.harmonic_fusion_training import FusionTrainingConfig
from music_critic.experiments.registry import canonical_json_bytes, load_record
from tests.experiments.test_harmonic_fusion_pilot_admission import _gate_report, _runtime


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fusion_reference_finalizer", ROOT / "scripts/finalize_harmonic_fusion_reference.py")
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return artifact_pin(path)


def _write_report(fixture):
    fixture.report.pop("fingerprint", None)
    fixture.report["fingerprint"] = fingerprint(fixture.report)
    _write(fixture.report_path, fixture.report)
    manifest = {name: artifact_pin(fixture.output / name) for name in (
        "report.json", "resolved.json", "progress.jsonl")}
    _write(fixture.output / "artifact-manifest.json", manifest)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    repository = tmp_path / "source"
    config = _write(repository / "configs/experiments/harmonic_fusion_pilot.json", asdict(FusionTrainingConfig()))
    metadata = _write(tmp_path / "metadata.json", {"fixture_only": True})
    request = {"schema_version": reference.REQUEST_SCHEMA, "comparison_id": "fixture-pair",
        "data_manifest": metadata, "config": config, "source_manifest": metadata,
        "source_checkpoint": {"path": str(tmp_path / "never-read.pt"), "sha256": "a" * 64, "size_bytes": 10},
        "source_model_state_fingerprint": "b" * 64, "runtime": _runtime(),
        "gate": {}, "experiment_record": {}, "max_wall_seconds": 600}
    stable = stable_bindings(request)
    identity = "FIXTURE-REFERENCE-FINALIZATION"
    record = planned_record(identity, protocol={"fusion_pilot_bindings": reference.reference_gate_bindings(stable),
        "reference_train_smoke": reference.reference_protocol(stable)}, updates=2, draws=4,
        validation=False, parents=[])
    record_path = repository / "docs/experiments/records" / (identity + ".json")
    _write(record_path, record)
    original = tmp_path / "plan/planned-records" / record_path.name
    request["experiment_record"] = _write(original, record)
    request_path = tmp_path / "plan/reference-request.json"
    request_pin = _write(request_path, request)
    output = tmp_path / "attempt"
    report = _gate_report("real_train_reference", request["runtime"], stable)
    report.update(experiment_id=identity, wall_seconds=5.0,
                  execution_context_fingerprint=fingerprint({"request": request}))
    report["state"].update(experiment_id=identity, status="completed", stage="attempt", event_status="completed")
    _write(output / "resolved.json", {"experiment_id": identity, "execution_context": {"request": request},
        "synthetic": False, "bindings": report["bindings"], "config": report["config"],
        "max_wall_seconds": 600, "reference_validation_policy": reference.VALIDATION_POLICY,
        "command": [str(repository / "scripts/run_harmonic_fusion_reference_gate.py"),
            "--request", str(request_path), "--request-sha256", request_pin["sha256"],
            "--request-size-bytes", str(request_pin["size_bytes"]), "--execute", "--output-root", str(output)]})
    row = {"schema_version": reference.JOURNAL_SCHEMA, "index": 0, "previous": None,
        "stage": "attempt", "status": "completed", "detail": {},
        "state": deepcopy(report["state"]), "elapsed_wall_seconds": 5.0}
    row["fingerprint"] = fingerprint(row)
    _write(output / "progress.jsonl", row)
    resource = {"schema": "ResearchPilotResourceReceipt@1.0.0", "child_returncode": 0,
        "process_status": "exited_zero", "reason": "child_exit", "gpu_uuid": finalizer.GPU_UUID,
        "child_lifecycle": {"ownership_lost": False, "ownership_confirmed_before_cleanup": True,
                            "leader_reaped_after_cleanup": True},
        "wall_limit_seconds": 600.0, "wall_seconds": 7.0, "gpu_memory_ceiling_mib": 16384,
        "host_available_floor_mib": 8192, "max_swap_growth_mib": 256,
        "resource_samples": 10, "sampled_gpu_memory_peak_mib": 1400,
        "sampled_host_available_min_kib": 12000 * 1024, "initial_swap_used_kib": 100,
        "sampled_peak_swap_used_kib": 100, "cwd": str(repository),
        "command": ["/fixture/python", str(Path(finalizer.__file__).with_name("run_harmonic_fusion_pinned_request.py")),
                    "reference", str(request_path), str(output)]}
    resource_path = tmp_path / "resource-receipt.json"
    _write(resource_path, resource)
    fixture = SimpleNamespace(repository=repository, request=request, request_path=request_path,
        record_path=record_path, original=original, output=output, report=report,
        report_path=output / "report.json", resource=resource, resource_path=resource_path)
    _write_report(fixture)
    fixture.planned_bytes = record_path.read_bytes()

    def deny(*args, **kwargs):
        pytest.fail("finalization must not read tensors or use CUDA")

    monkeypatch.setattr(torch, "load", deny)
    monkeypatch.setattr(torch.cuda, "is_available", deny)
    monkeypatch.setattr(torch.cuda, "init", deny)
    return fixture


def _finalize(fixture):
    return finalizer.finalize_reference(request_path=fixture.request_path, record_path=fixture.record_path,
        report_path=fixture.report_path, resource_receipt_path=fixture.resource_path)


def test_valid_native_report_and_supervised_evidence_finalize_only_planned_record(evidence):
    result = _finalize(evidence)
    record = load_record(evidence.record_path)
    assert result["status"] == record["status"] == "completed"
    assert record["budget"]["actual"] == record["results"]["actual_updates"] == 2
    assert record["budget"]["actual_draws"] == record["results"]["actual_draws"] == 4
    assert record["environment"] == evidence.request["runtime"]
    assert record["data_access"]["access_evidence_status"] == "verified_primary_evidence"
    assert all(record["data_access"][key] is False for key in (
        "validation_accessed", "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed", "test_used_for_selection"))
    assert {row["role"] for row in record["artifacts"]} == {
        "reference_report", "reference_artifact_manifest", "reference_progress", "supervisor_resource_receipt"}
    for row in record["artifacts"]:
        pin = artifact_pin(evidence.repository / row["locator"])
        assert (pin["sha256"], pin["size_bytes"]) == (row["sha256"], row["size_bytes"])
    assert evidence.original.read_bytes() == evidence.planned_bytes
    assert not list(evidence.record_path.parent.glob("*.lock"))


@pytest.mark.parametrize("mutation", ["gpu_peak", "gpu_limit", "wall", "host", "swap", "no_samples", "uuid", "lost_ownership", "not_reaped", "child_error"])
def test_resource_failure_never_finalizes_or_modifies_original_evidence(evidence, mutation):
    receipt = evidence.resource
    if mutation == "gpu_peak": receipt["sampled_gpu_memory_peak_mib"] = 16385
    elif mutation == "gpu_limit": receipt["gpu_memory_ceiling_mib"] = 16385
    elif mutation == "wall": receipt["wall_seconds"] = 601
    elif mutation == "host": receipt["sampled_host_available_min_kib"] = 8192 * 1024 - 1
    elif mutation == "swap": receipt["sampled_peak_swap_used_kib"] = 100 + 256 * 1024 + 1
    elif mutation == "no_samples": receipt["resource_samples"] = 0
    elif mutation == "uuid": receipt["gpu_uuid"] = "wrong-device"
    elif mutation == "lost_ownership": receipt["child_lifecycle"]["ownership_lost"] = True
    elif mutation == "not_reaped": receipt["child_lifecycle"]["leader_reaped_after_cleanup"] = False
    else: receipt["child_returncode"] = 1
    _write(evidence.resource_path, receipt)
    with pytest.raises(ValueError):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == evidence.original.read_bytes() == evidence.planned_bytes


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_supervisor_command_must_bind_exact_helper_kind_request_and_output(evidence, index):
    evidence.resource["command"][index] = "/wrong/value"
    _write(evidence.resource_path, evidence.resource)
    with pytest.raises(ValueError, match="supervisor_command_not_bound"):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == evidence.planned_bytes


@pytest.mark.parametrize("mutation", ["failed", "runtime", "counts", "self_fingerprint", "context"])
def test_report_failure_or_unbound_report_cannot_be_finalized(evidence, mutation):
    if mutation == "failed": evidence.report["passed"] = False
    elif mutation == "runtime": evidence.report["environment"]["torch_cuda_build"] = "other-build"
    elif mutation == "counts": evidence.report["actual_updates"] = 1
    elif mutation == "context": evidence.report["execution_context_fingerprint"] = "f" * 64
    _write_report(evidence)
    if mutation == "self_fingerprint":
        evidence.report["fingerprint"] = "f" * 64
        _write(evidence.report_path, evidence.report)
    with pytest.raises(ValueError):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == evidence.planned_bytes


def test_terminal_collision_refuses_any_second_write(evidence):
    _finalize(evidence)
    completed = evidence.record_path.read_bytes()
    with pytest.raises(ValueError, match="terminal_is_immutable"):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == completed
    assert evidence.original.read_bytes() == evidence.planned_bytes


def test_missing_manifest_member_preserves_planned_record(evidence):
    manifest_path = evidence.output / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    del manifest["progress.jsonl"]
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="complete_reference_artifact_manifest_required"):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == evidence.planned_bytes


def test_atomic_publication_failure_preserves_planned_record(evidence, monkeypatch):
    def fail(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(finalizer.os, "replace", fail)
    with pytest.raises(OSError, match="publication failure"):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == evidence.original.read_bytes() == evidence.planned_bytes


def test_evidence_mutation_during_finalization_is_rejected(evidence, monkeypatch):
    original = finalizer._atomic_complete
    def mutate(path, expected, content, snapshots):
        evidence.resource_path.write_text("{}\n")
        return original(path, expected, content, snapshots)
    monkeypatch.setattr(finalizer, "_atomic_complete", mutate)
    with pytest.raises(ValueError, match="evidence_changed_before_publication"):
        _finalize(evidence)
    assert evidence.record_path.read_bytes() == evidence.planned_bytes


def test_cli_has_the_four_concrete_finalization_arguments(evidence, capsys):
    assert finalizer.main(["--request", str(evidence.request_path), "--record", str(evidence.record_path),
                          "--report", str(evidence.report_path), "--resource-receipt", str(evidence.resource_path)]) == 0
    assert json.loads(capsys.readouterr().out)["actual_updates"] == 2
