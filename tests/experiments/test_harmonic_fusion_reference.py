"""CPU-only reference gate mechanics and metadata admission; no real data."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_reference as reference
from music_critic.experiments.analysisgnn import harmonic_fusion_pilot as pilot
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import build_source_free_fixture, load_frozen_class_weights
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair
from music_critic.experiments.analysisgnn.harmonic_fusion_resume import create_training_payload
from music_critic.experiments.analysisgnn.harmonic_fusion_training import FusionTrainingConfig, FusionTrainingSession


@pytest.fixture(scope="module")
def fixture_recipe():
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    table = tuple(FullOrbitPair(sidecar["record_id"], shift, sidecar["source_component_id"]) for shift in range(12))
    torch.manual_seed(734)
    base = CorrectedAnalysisGNNModel()
    config = FusionTrainingConfig()
    bindings = {"source_kind": "generated_unit_fixture", "device": "cpu",
        "source_model_state_fingerprint": model_state_fingerprint(base),
        "class_weight_fingerprint": weights.payload_fingerprint}
    def reader(record_id, *, split):
        assert split == "train" and record_id == sidecar["record_id"]
        return batch.raw_graph_batch, deepcopy(sidecar)
    def factory(enabled):
        return reference._ReferenceStepSession(base=base, enabled=enabled, config=config,
                                               bindings=bindings, table=table)
    return dict(experiment_id="fixture-reference", config=config, bindings=bindings, table=table,
        class_weights=weights, reader=reader, session_factory=factory, verify_inputs=lambda: None,
        setup_runtime=lambda: {"device": "cpu", "source_kind": "fixture"}, synthetic=True), base


@pytest.fixture(scope="module")
def complete(tmp_path_factory, fixture_recipe):
    root = tmp_path_factory.mktemp("reference-native-cpu") / "attempt"
    kwargs, _ = fixture_recipe
    report = reference._execute_reference(root, **kwargs)
    return root, report


def _receipt(root):
    assert not (root / "report.json").exists()
    return json.loads((root / "failure-receipt.json").read_text())


def test_actual_native_reference_pair_uses_p512_first_draws_without_validation(complete):
    _, report = complete
    assert report["schema_version"] == reference.REPORT_SCHEMA
    assert report["passed"] and report["reference_smoke_passed"]
    assert report["actual_updates"] == 2 and report["actual_draws"] == 4
    assert report["attempted_updates"] == 2 and report["reader_calls"] == report["reader_completed"] == 4
    assert report["config"] == FusionTrainingConfig().to_dict()
    assert report["source_hashes_verified"] and report["raw_only_prediction_verified"] and report["target_alignment_verified"]
    assert report["reference_validation_policy"] == reference.VALIDATION_POLICY
    assert report["real_data_accessed"] is False and report["synthetic"] is True
    assert report["resumable"] is report["scientific_acceptance"] is report["unconfirmed_in_flight_tail"] is False
    assert all(report[key] is False for key in reference._ACCESS)
    assert report["validation_reader_calls"] == 0
    assert [arm["arm"] for arm in report["arms"]] == ["control", "fusion"]
    assert report["arms"][0]["draws"] == report["arms"][1]["draws"]
    for arm in report["arms"]:
        assert arm["actual_updates"] == arm["raw_prediction_calls"] == arm["target_alignment_checks"] == 1
        assert arm["actual_draws"] == arm["sampler_position"] == 2
        assert arm["model_state_changed"] and arm["validation_updates"] == []
        assert arm["lr_used"] == 0.0003 / 64 and arm["lr_next"] == 0.0003 * 2 / 64
        assert len(arm["alignment"]) == 18 and any(row["valid_rows"] for row in arm["alignment"].values())
        assert set(arm["optimizer_steps"].values()) <= {0, 1}
        assert 1 in arm["optimizer_steps"].values()


def test_reference_outputs_are_immutable_and_hash_chained(complete, fixture_recipe):
    root, report = complete
    before = (root / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        reference._execute_reference(root, **fixture_recipe[0])
    assert (root / "report.json").read_bytes() == before
    previous = None
    for index, line in enumerate((root / "progress.jsonl").read_text().splitlines()):
        row = json.loads(line)
        digest = row.pop("fingerprint")
        assert row["index"] == index and row["previous"] == previous and fingerprint(row) == digest
        assert row["elapsed_wall_seconds"] >= 0
        previous = digest
    assert row["status"] == "completed" and row["stage"] == "attempt"
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    for pin in manifest.values():
        assert pilot.checked_bytes(pin)
    assert not list(root.glob("*.pt"))


def test_reference_session_cannot_take_second_update_or_be_full_resume(fixture_recipe):
    kwargs, _ = fixture_recipe
    session = kwargs["session_factory"](True)
    with pytest.raises(ValueError):
        create_training_payload(session)
    session.step(kwargs["reader"], kwargs["class_weights"])
    with pytest.raises(ValueError, match="single_update_budget"):
        session.step(kwargs["reader"], kwargs["class_weights"])
    with pytest.raises(ValueError):
        create_training_payload(session)


@pytest.mark.parametrize("failure_call,applied", [(1, 0), (3, 1)])
def test_reader_failure_retains_spent_work_and_no_success(tmp_path, fixture_recipe, failure_call, applied):
    kwargs, _ = fixture_recipe
    calls = 0
    def reader(record_id, *, split):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("injected reader failure")
        return kwargs["reader"](record_id, split=split)
    root = tmp_path / "failure"
    with pytest.raises(OSError, match="reader failure"):
        reference._execute_reference(root, **{**kwargs, "reader": reader})
    report = _receipt(root)
    assert report["actual_updates"] == applied and report["actual_draws"] == 2 * applied
    assert report["reader_calls"] == failure_call and report["reader_completed"] == failure_call - 1
    assert report["attempted_updates"] == applied + 1
    assert report["unconfirmed_in_flight_tail"] and report["source_hashes_verified"] is False


@pytest.mark.parametrize("failure_stage", ["runtime", "verify_inputs", "resolved"])
def test_early_failures_have_zero_work_receipt(tmp_path, fixture_recipe, monkeypatch, failure_stage):
    kwargs, _ = fixture_recipe
    def fail(*args, **kwargs):
        raise OSError("injected setup failure")
    if failure_stage == "resolved":
        original = pilot._write_json
        def write(path, data):
            if Path(path).name == "resolved.json":
                fail()
            return original(path, data)
        monkeypatch.setattr(pilot, "_write_json", write)
    else:
        kwargs = {**kwargs, "setup_runtime" if failure_stage == "runtime" else "verify_inputs": fail}
    root = tmp_path / "early"
    with pytest.raises(OSError, match="setup failure"):
        reference._execute_reference(root, **kwargs)
    receipt = _receipt(root)
    assert receipt["actual_updates"] == receipt["reader_calls"] == receipt["attempted_updates"] == 0
    assert receipt["source_hashes_verified"] is False


def test_failure_after_adamw_counts_the_applied_effect(tmp_path, fixture_recipe):
    kwargs, _ = fixture_recipe
    def factory(enabled):
        session = kwargs["session_factory"](enabled)
        def fail():
            raise RuntimeError("scheduler failed after AdamW")
        session.scheduler.step = fail
        return session
    root = tmp_path / "after-adam"
    with pytest.raises(RuntimeError, match="after AdamW"):
        reference._execute_reference(root, **{**kwargs, "session_factory": factory})
    report = _receipt(root)
    assert report["actual_updates"] == report["attempted_updates"] == 1
    assert report["actual_draws"] == report["reader_completed"] == 2
    assert report["unconfirmed_in_flight_tail"]


def test_prediction_alignment_failure_occurs_before_optimizer(tmp_path, fixture_recipe, monkeypatch):
    from music_critic.experiments.analysisgnn import corrected_training
    def fail(*args, **kwargs):
        raise RuntimeError("injected target join failure")
    monkeypatch.setattr(corrected_training, "align_target_sidecars_after_prediction", fail)
    root = tmp_path / "alignment"
    with pytest.raises(RuntimeError, match="target join"):
        reference._execute_reference(root, **fixture_recipe[0])
    report = _receipt(root)
    assert report["actual_updates"] == 0 and report["reader_completed"] == 2
    assert report["state"]["arms"]["control"]["raw_prediction_calls"] == 1
    assert report["target_alignment_verified"] is False


def test_manifest_failure_cannot_publish_success(tmp_path, fixture_recipe, monkeypatch):
    original = pilot._write_json
    def write(path, data):
        if Path(path).name == "artifact-manifest.json":
            raise OSError("manifest failure")
        return original(path, data)
    monkeypatch.setattr(pilot, "_write_json", write)
    root = tmp_path / "manifest"
    with pytest.raises(OSError, match="manifest failure"):
        reference._execute_reference(root, **fixture_recipe[0])
    report = _receipt(root)
    assert report["actual_updates"] == 2 and report["reader_completed"] == 4
    assert report["passed"] is False


def test_deadline_prevents_runtime_or_reads(tmp_path, fixture_recipe):
    kwargs, _ = fixture_recipe
    def deny():
        raise AssertionError("must not initialize")
    with pytest.raises(TimeoutError, match="wall_budget"):
        reference._execute_reference(tmp_path / "timeout", **{**kwargs, "setup_runtime": deny}, max_wall_seconds=-1)
    assert _receipt(tmp_path / "timeout")["actual_updates"] == 0


def _metadata_recipe(tmp_path, monkeypatch, fixture_recipe):
    kwargs, _ = fixture_recipe
    pin = {"path": str(tmp_path / "not-opened"), "sha256": "a" * 64, "size_bytes": 12}
    request = {"schema_version": reference.REQUEST_SCHEMA, "comparison_id": "fixture-comparison",
        **{key: deepcopy(pin) for key in ("data_manifest", "config", "source_checkpoint", "source_manifest", "experiment_record")},
        "source_model_state_fingerprint": kwargs["bindings"]["source_model_state_fingerprint"],
        "runtime": {"fixture": True}, "gate": {"record": deepcopy(pin), "report": deepcopy(pin)}, "max_wall_seconds": 600}
    # Pure metadata fakes never represent canonical records or a CUDA result.
    monkeypatch.setattr(pilot, "validate_runtime_policy", lambda runtime: None)
    monkeypatch.setattr(pilot, "verify_source_manifest", lambda *args: None)
    monkeypatch.setattr(pilot, "verify_gate", lambda *args, **kw: None)
    monkeypatch.setattr(pilot, "checked_json", lambda pin: asdict(FusionTrainingConfig()))
    stable = pilot.stable_bindings(request)
    record = {"experiment_id": "fixture-planned", "kind": "engineering_gate", "seeds": [17],
        "budget": {"planned": 2, "planned_draws": 4}, "protocol": {
            "fusion_pilot_bindings": reference.reference_gate_bindings(stable),
            "reference_train_smoke": reference.reference_protocol(stable)},
        "data_access": {"declared_policy": {key: False for key in ("validation_allowed", "test_inputs_allowed",
            "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")}}}
    monkeypatch.setattr(pilot, "_record", lambda pin: record)
    from music_critic.experiments.analysisgnn import harmonic_fusion_data
    class MetadataData:
        table = kwargs["table"]
        bindings = kwargs["bindings"]
        def require_train_records(self, record_ids):
            assert set(record_ids) == {self.table[0].record_id}
    monkeypatch.setattr(harmonic_fusion_data, "FrozenFusionData", lambda manifest: MetadataData())
    return request, record


def test_metadata_admission_does_not_initialize_cuda_read_raw_or_load_checkpoint(tmp_path, fixture_recipe, monkeypatch):
    request, _ = _metadata_recipe(tmp_path, monkeypatch, fixture_recipe)
    def deny(*args, **kwargs):
        raise AssertionError("metadata-only admission")
    monkeypatch.setattr(torch.cuda, "init", deny)
    monkeypatch.setattr(torch, "load", deny)
    result = reference.admit_reference_request(request, repository_root=tmp_path)
    assert len(result["draws"]) == 2 and result["config"] == FusionTrainingConfig()


@pytest.mark.parametrize("mutation", ["extra", "schema", "wall", "config", "checkpoint_pin", "planned_budget",
    "validation_policy", "seed", "protocol_config", "binding", "kind"])
def test_reference_admission_rejects_protocol_drift(tmp_path, fixture_recipe, monkeypatch, mutation):
    request, record = _metadata_recipe(tmp_path, monkeypatch, fixture_recipe)
    if mutation == "extra": request["extra"] = True
    if mutation == "schema": request["schema_version"] = "different"
    if mutation == "wall": request["max_wall_seconds"] = 601
    if mutation == "checkpoint_pin": request["source_checkpoint"]["size_bytes"] = True
    if mutation == "config": monkeypatch.setattr(pilot, "checked_json", lambda pin: {**asdict(FusionTrainingConfig()), "validation_mode": "synthetic_disabled"})
    if mutation == "planned_budget": record["budget"]["planned"] = 4
    if mutation == "validation_policy": record["data_access"]["declared_policy"]["validation_allowed"] = True
    if mutation == "seed": record["seeds"] = [18]
    if mutation == "protocol_config": record["protocol"]["reference_train_smoke"]["bindings"]["config_sha256"] = "b" * 64
    if mutation == "binding": record["protocol"]["fusion_pilot_bindings"]["source_checkpoint_sha256"] = "b" * 64
    if mutation == "kind": record["kind"] = "training"
    with pytest.raises(ValueError):
        reference.admit_reference_request(request, repository_root=tmp_path)


def test_cli_defaults_to_metadata_and_requires_output_root_for_execution(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts/run_harmonic_fusion_reference_gate.py"
    spec = importlib.util.spec_from_file_location("reference_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pin = tmp_path / "request.json"
    pin.write_text("{}")
    artifact = pilot.artifact_pin(pin)
    args = ["--request", str(pin), "--request-sha256", artifact["sha256"], "--request-size-bytes", str(artifact["size_bytes"])]
    monkeypatch.setattr(reference, "admit_reference_request", lambda *args, **kwargs: {
        "config": FusionTrainingConfig(), "bindings": {}, "experiment_id": "fixture", "draws": []})
    def deny(*args, **kwargs):
        raise AssertionError("execution not requested")
    monkeypatch.setattr(reference, "run_reference_request", deny)
    assert module.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["executed"] is False and result["cuda_initialized"] is False
    assert result["checkpoint_tensors_loaded"] == result["raw_graphs_loaded"] == 0
    with pytest.raises(SystemExit):
        module.main([*args, "--execute"])


def _fabricated_cuda_report_metadata(complete):
    # Transform a CPU unit result's metadata only. This dictionary is never
    # written as an artifact or represented as an observed CUDA gate.
    report = deepcopy(complete[1])
    stable = {"comparison_id": "fixture-comparison", "data_manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64, "source_manifest_sha256": "c" * 64,
        "source_checkpoint_sha256": "d" * 64, "source_checkpoint_size_bytes": 10,
        "source_model_state_fingerprint": report["bindings"]["source_model_state_fingerprint"],
        "runtime_fingerprint": fingerprint(report["environment"]), "reference_path": "sequential_historical_C2"}
    report.update(device="cuda", synthetic=False, real_data_accessed=True)
    report["bindings"].update(stable)
    report["bindings"]["reference_draw_schedule_fingerprint"] = report["draw_schedule_fingerprint"]
    record = {"experiment_id": report["experiment_id"], "seeds": [17], "budget": {"planned": 2, "planned_draws": 4},
        "protocol": {"fusion_pilot_bindings": reference.reference_gate_bindings(stable),
                     "reference_train_smoke": reference.reference_protocol(stable)},
        "data_access": {"declared_policy": {key: False for key in ("validation_allowed", "test_inputs_allowed",
            "test_targets_allowed", "test_metrics_allowed", "test_selection_allowed")}}}
    return record, report, stable


def test_reference_report_verifier_accepts_complete_native_shape(complete):
    record, report, stable = _fabricated_cuda_report_metadata(complete)
    assert reference.verify_reference_report(record, report, stable) == record["experiment_id"]


@pytest.mark.parametrize("mutation", ["protocol", "budget", "policy", "source", "environment", "config", "scope",
    "validation", "unknown", "updates", "draws", "reader_calls", "arm_order", "arm_count", "native_forward",
    "native_alignment", "journal", "draw_position", "draw_hash", "draw_shift"])
def test_reference_report_verifier_rejects_incomplete_or_unbound_evidence(complete, mutation):
    record, report, stable = _fabricated_cuda_report_metadata(complete)
    if mutation == "protocol": record["protocol"]["reference_train_smoke"]["bindings"]["config_sha256"] = "e" * 64
    if mutation == "budget": record["budget"]["planned"] = True
    if mutation == "policy": record["data_access"]["declared_policy"]["validation_allowed"] = True
    if mutation == "source": report["bindings"]["source_checkpoint_sha256"] = "e" * 64
    if mutation == "environment": report["environment"]["device"] = "different"
    if mutation == "config": report["config"]["total_updates"] = 1
    if mutation == "scope": report["synthetic"] = True
    if mutation == "validation": report["validation_inputs_accessed"] = True
    if mutation == "unknown": report["unconfirmed_in_flight_tail"] = True
    if mutation == "updates": report["actual_updates"] = True
    if mutation == "draws": report["actual_draws"] = 6
    if mutation == "reader_calls": report["reader_completed"] = 3
    if mutation == "arm_order": report["arms"].reverse()
    if mutation == "arm_count": report["arms"][1]["actual_updates"] = 0
    if mutation == "native_forward": report["arms"][1]["raw_prediction_calls"] = 0
    if mutation == "native_alignment": report["arms"][1]["target_alignment_checks"] = 0
    if mutation == "journal": report["state"]["arms"]["control"]["applied_updates"] = 2
    if mutation == "draw_position": report["arms"][1]["draws"][0]["orbit_index"] = 3
    if mutation == "draw_hash": report["arms"][1]["draw_schedule_fingerprint"] = "e" * 64
    if mutation == "draw_shift": report["arms"][1]["draws"][0]["shift_pc"] = 12
    with pytest.raises(ValueError, match="fusion_reference"):
        reference.verify_reference_report(record, report, stable)
