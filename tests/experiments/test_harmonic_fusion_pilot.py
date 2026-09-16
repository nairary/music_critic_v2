"""Real bounded dispatcher mechanics on tiny fixtures; no corpus or CUDA."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest
import torch

from music_critic.experiments.analysisgnn import harmonic_fusion_pilot as pilot
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import build_source_free_fixture, load_frozen_class_weights
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair
from music_critic.experiments.analysisgnn.harmonic_fusion_training import FusionTrainingConfig, FusionTrainingSession
from music_critic.experiments.analysisgnn.harmonic_fusion_resume import load_training_checkpoint, create_training_payload
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint


@pytest.fixture(scope="module")
def mechanics(tmp_path_factory):
    root = tmp_path_factory.mktemp("fusion-pilot")
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    table = tuple(FullOrbitPair(sidecar["record_id"], shift, sidecar["source_component_id"]) for shift in (0, 1, 2))
    torch.manual_seed(334)
    base = CorrectedAnalysisGNNModel()
    config = FusionTrainingConfig(total_updates=4, warmup_updates=2, validation_interval=2, checkpoint_interval=2)
    bindings = {"data": "synthetic-only", "device": "cpu", "source_model_state_fingerprint": model_state_fingerprint(base),
                "class_weight_fingerprint": weights.payload_fingerprint}
    def reader(record_id, *, split):
        assert split in ("train", "validation") and record_id == sidecar["record_id"]
        return batch.raw_graph_batch, deepcopy(sidecar)
    def validation_factory(read, progress):
        def validate(model, *, split):
            raw, _ = read(sidecar["record_id"], split=split)
            with torch.no_grad():
                model(raw)
            return {"corrected_primary_macro_score": 0.5}
        return validate
    kwargs = dict(config=config, bindings=bindings, table=table, class_weights=weights,
                  reader=reader, validation_factory=validation_factory, synthetic=True,
                  execution_context={"source_kind":"generated_unit_fixture"},
                  session_factory=lambda enabled: FusionTrainingSession(base=base, enabled=enabled,
                      config=config, bindings=bindings, table=table),
                  experiment_ids={"control": "fixture-control", "fusion": "fixture-fusion"})
    full = pilot._execute_attempt(root / "full", attempt_id="full", **kwargs)
    counter = 0
    def fail_reader(record_id, *, split):
        nonlocal counter
        if split == "train":
            counter += 1
            if counter == 7:
                raise RuntimeError("injected fourth update read failure")
        return reader(record_id, split=split)
    with pytest.raises(RuntimeError, match="injected fourth"):
        pilot._execute_attempt(root / "failed", attempt_id="failed", **{**kwargs, "reader": fail_reader})
    failure = json.loads((root / "failed/failure-receipt.json").read_text())
    parent = deepcopy(failure["state"])
    parent["unconfirmed_parent_tail"] = False
    resumed = pilot._execute_attempt(root / "resumed", attempt_id="resumed", parent=parent, **kwargs)
    yield root, kwargs, full, failure, resumed
    shutil.rmtree(root)


def test_pair_completes_fixed_budget_earliest_best_and_distinct_final_exports(mechanics):
    root, _, report, _, _ = mechanics
    assert report["passed"] and report["synthetic"] and not report["scientific_acceptance"]
    assert report["device"] == "cpu" and report["process_resources"]["cuda_peak_allocated_bytes"] is None
    resolved=json.loads((root/"full/resolved.json").read_text())
    assert fingerprint(resolved["execution_context"]) == report["execution_context_fingerprint"]
    assert resolved["max_wall_seconds"] == 7200
    assert report["cumulative_costs"] == {"attempted_updates": 8, "applied_updates": 8,
                                          "train_reader_calls": 16, "validation_reader_calls": 6}
    for arm in ("control", "fusion"):
        summary = report["arms"][arm]
        assert summary["applied_updates"] == 4 and summary["draws"] == 8
        assert [row["update"] for row in summary["validations"]] == [0, 2, 4]
        assert summary["best_update"] == 0
        assert summary["best_model"]["path"] != summary["final_model"]["path"]
        for name in ("best_model", "final_model"):
            assert pilot.checked_bytes(summary[name])
    manifest = json.loads((root / "full/artifact-manifest.json").read_text())
    for pin in manifest.values():
        pilot.checked_bytes(pin)


def test_failed_prefix_retained_and_resume_counts_replayed_work(mechanics):
    root, kwargs, full, failure, resumed = mechanics
    assert failure["passed"] is False and not (root / "failed/report.json").exists()
    assert failure["cumulative_costs"]["applied_updates"] == 3
    assert failure["cumulative_costs"]["attempted_updates"] == 4
    assert failure["state"]["arms"]["control"]["checkpoint"]["key"][0] == 2
    assert resumed["cumulative_costs"]["applied_updates"] == 9  # one applied update repeated after durableu2
    assert resumed["cumulative_costs"]["attempted_updates"] == 10
    for arm in ("control", "fusion"):
        states = []
        for result in (full, resumed):
            pin = result["arms"][arm]["final_checkpoint"]["artifact"]
            restored = load_training_checkpoint(pin["path"], expected_sha256=pin["sha256"],
                expected_size_bytes=pin["size_bytes"], config=kwargs["config"], bindings=kwargs["bindings"],
                table=kwargs["table"], expected_enabled=arm == "fusion")
            states.append(recursive_state_fingerprint(create_training_payload(restored)))
        assert states[0] == states[1]


def test_existing_attempt_root_rejected_without_overwrite(mechanics):
    root, kwargs, *_ = mechanics
    before = (root / "full/report.json").read_bytes()
    with pytest.raises(FileExistsError):
        pilot._execute_attempt(root / "full", attempt_id="different", **kwargs)
    assert (root / "full/report.json").read_bytes() == before


def test_boundary_validation_failure_checkpoints_pending_for_recovery(tmp_path, mechanics):
    _, kwargs, *_ = mechanics
    def validation_factory(read, progress):
        def validate(model, *, split):
            raise RuntimeError("validation failed")
        return validate
    with pytest.raises(RuntimeError, match="validation failed"):
        pilot._execute_attempt(tmp_path / "pending", attempt_id="pending", **{**kwargs,"validation_factory":validation_factory})
    receipt = json.loads((tmp_path / "pending/failure-receipt.json").read_text())
    cp = receipt["state"]["arms"]["control"]["checkpoint"]
    assert cp["key"] == [0, 0, 0]
    assert receipt["cumulative_costs"]["attempted_updates"] == 0
    shutil.rmtree(tmp_path / "pending")


@pytest.mark.parametrize("content", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_ambiguous_json_rejected(content):
    with pytest.raises(ValueError):
        pilot.decode_json(content)


@pytest.mark.parametrize("change", ["hash", "size", "relative", "boolsize", "symlink"])
def test_artifact_pins_fail_closed_before_decode(tmp_path, change):
    path = tmp_path / "data.json"
    path.write_text('{"value": 1}')
    pin = pilot.artifact_pin(path)
    if change == "hash": pin["sha256"] = "f" * 64
    if change == "size": pin["size_bytes"] += 1
    if change == "boolsize": pin["size_bytes"] = True
    if change == "relative": pin["path"] = "data.json"
    if change == "symlink":
        alias = tmp_path / "alias.json"; alias.symlink_to(path); pin["path"] = str(alias)
    with pytest.raises(ValueError):
        pilot.checked_json(pin)


def test_source_manifest_requires_all_runtime_sources(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text("# model\n")
    path = tmp_path / "manifest.json";path.write_text('{"files":[]}')
    with pytest.raises(ValueError, match="source_manifest_incomplete"):
        pilot.verify_source_manifest(pilot.artifact_pin(path), tmp_path)


def test_parent_hash_chain_and_bindings_rejected_before_checkpoint_io(tmp_path, mechanics, monkeypatch):
    root, kwargs, _, failure, _ = mechanics
    parent = {"journal":pilot.artifact_pin(root / "failed/progress.jsonl"),
              "experiment_records":kwargs["experiment_ids"]}
    monkeypatch.setattr(pilot, "_record", lambda pin, **kw: {"experiment_id": pin,
        "artifacts":[parent["journal"]]})
    state = pilot.read_parent(parent, bindings=kwargs["bindings"])
    assert state["arms"]["control"]["cursor"] == 3
    with pytest.raises(ValueError, match="parent_binding_mismatch"):
        pilot.read_parent(parent, bindings={"other":"binding"})
    rows = (root / "failed/progress.jsonl").read_bytes().splitlines(keepends=True)
    path = tmp_path / "tampered.jsonl"; path.write_bytes(b"".join(rows[1:]))
    with pytest.raises(ValueError, match="parent_journal_invalid"):
        pilot.read_parent({**parent,"journal":pilot.artifact_pin(path)},bindings=kwargs["bindings"])
    path = tmp_path / "truncated.jsonl";path.write_bytes(b"".join(rows)+b'{"interrupted":')
    parent = {**parent,"journal":pilot.artifact_pin(path)}
    recovered = pilot.read_parent(parent,bindings=kwargs["bindings"])
    assert recovered["unconfirmed_parent_tail"] is True


def test_missing_gate_admission_never_constructs_data_or_reads_source(tmp_path, monkeypatch):
    config = tmp_path / "config.json";config.write_text(json.dumps(asdict(FusionTrainingConfig())))
    monkeypatch.setattr(pilot, "verify_source_manifest", lambda *args: None)
    monkeypatch.setattr(pilot, "validate_runtime_policy", lambda *args: None)
    stable = {"comparison_id":"pair", "data_manifest_sha256":"a"*64,
              "config_sha256":pilot.artifact_pin(config)["sha256"], "source_manifest_sha256":"b"*64,
              "source_checkpoint_sha256":"c"*64, "source_checkpoint_size_bytes":10,
              "source_model_state_fingerprint":"d"*64,"runtime_fingerprint":fingerprint({}),
              "reference_path":"sequential_historical_C2"}
    def record(pin, **kwargs):
        return {"protocol":{"fusion_pilot_bindings":{"arm":pin,**stable}},"seeds":[17],
                "experiment_id":pin,"budget":{"planned":512,"planned_draws":1024},
                "data_access":{"declared_policy":{"validation_allowed":True,
                **{k:False for k in ("test_inputs_allowed","test_targets_allowed","test_metrics_allowed","test_selection_allowed")}}}}
    monkeypatch.setattr(pilot,"_record",record)
    request = {"schema_version":pilot.REQUEST_SCHEMA,"comparison_id":"pair","attempt_id":"first",
        "data_manifest":{"sha256":"a"*64},"config":pilot.artifact_pin(config),
        "source_manifest":{"sha256":"b"*64},"source_checkpoint":{"path":"/never-read.pt","sha256":"c"*64,"size_bytes":10},
        "source_model_state_fingerprint":"d"*64,"runtime":{},"gates":{},
        "experiment_records":{"control":"control","fusion":"fusion"},"parent":None,"max_wall_seconds":7200}
    with pytest.raises(ValueError, match="required_gates_missing"):
        pilot.admit_request(request, repository_root=tmp_path)


@pytest.mark.parametrize("mutation", ["not_completed", "invalid", "not_registered", "wrong_source", "TEST", "no_compute", "wrong_scope"])
def test_gate_requires_bound_primary_report_and_terminal_validity(tmp_path, monkeypatch, mutation):
    report = {"passed":True,"device":"cuda","real_data_accessed":False,"actual_updates":24,"actual_draws":48,
              **{k:False for k in ("test_inputs_accessed","test_targets_accessed","test_metrics_computed")}}
    if mutation == "TEST": report["test_targets_accessed"] = True
    if mutation == "wrong_scope": report["real_data_accessed"] = True
    if mutation == "no_compute": report.pop("actual_updates")
    path = tmp_path / "report.json";path.write_text(json.dumps(report))
    pin = pilot.artifact_pin(path)
    bindings = {"source_manifest_sha256":"a"*64,"runtime_fingerprint":"b"*64}
    record = {"experiment_id":"gate","status":"completed","validity":{"valid":True},"kind":"engineering_gate",
        "protocol":{"fusion_pilot_bindings":{"role":"cuda_synthetic",**bindings}},"artifacts":[pin],
        "data_access":{k:False for k in ("test_inputs_accessed","test_targets_accessed","test_metrics_computed")}}
    if mutation == "not_completed": record["status"] = "failed"
    if mutation == "invalid": record["validity"]["valid"] = False
    if mutation == "not_registered": record["artifacts"] = []
    if mutation == "wrong_source": record["protocol"]["fusion_pilot_bindings"]["source_manifest_sha256"] = "c"*64
    monkeypatch.setattr(pilot,"_record",lambda *args,**kwargs:record)
    with pytest.raises(ValueError):
        pilot.verify_gate({"record":{},"report":pin},role="cuda_synthetic",bindings=bindings)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_hard_kill_after_nested_reader_completion_retains_inflight_uncertainty(tmp_path, mechanics, monkeypatch, split):
    root, kwargs, *_ = mechanics
    source = root / "failed/progress.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    boundary = next(i for i,row in enumerate(rows) if row["stage"] == "reader" and row["status"] == "completed"
                    and row["detail"]["split"] == split)
    path = tmp_path / "killed.jsonl";path.write_text("\n".join(json.dumps(r) for r in rows[:boundary+1])+"\n")
    pin = pilot.artifact_pin(path)
    monkeypatch.setattr(pilot,"_record",lambda record,**kw:{"experiment_id":record,"artifacts":[pin]})
    parent = pilot.read_parent({"journal":pin,"experiment_records":kwargs["experiment_ids"]},bindings=kwargs["bindings"])
    assert parent["event_status"] == "completed" and parent["unconfirmed_parent_tail"] is True
    newroot = tmp_path / "next";newroot.mkdir()
    child = pilot.AttemptJournal(newroot,attempt_id="next",bindings=kwargs["bindings"],
                                 experiment_ids=kwargs["experiment_ids"],parent=parent)
    child.append("attempt","completed")
    pin = pilot.artifact_pin(child.path)
    descendant = pilot.read_parent({"journal":pin,"experiment_records":kwargs["experiment_ids"]},bindings=kwargs["bindings"])
    assert descendant["unconfirmed_parent_tail"] is True


def test_parent_journal_must_be_registered_in_terminal_record(mechanics, monkeypatch):
    root, kwargs, *_ = mechanics
    monkeypatch.setattr(pilot,"_record",lambda record,**kw:{"experiment_id":record,"artifacts":[]})
    with pytest.raises(ValueError,match="parent_journal_not_registered"):
        pilot.read_parent({"journal":pilot.artifact_pin(root/"failed/progress.jsonl"),
                           "experiment_records":kwargs["experiment_ids"]},bindings=kwargs["bindings"])


def test_success_manifest_failure_leaves_no_success_report(tmp_path, mechanics, monkeypatch):
    _, kwargs, *_ = mechanics
    original = pilot._write_json
    def fail_manifest(path, payload):
        if Path(path).name == "artifact-manifest.json":
            raise OSError("injected manifest error")
        return original(path,payload)
    monkeypatch.setattr(pilot,"_write_json",fail_manifest)
    with pytest.raises(OSError,match="manifest error"):
        pilot._execute_attempt(tmp_path/"manifest-failure",attempt_id="manifest-failure",**kwargs)
    receipt = json.loads((tmp_path/"manifest-failure/failure-receipt.json").read_text())
    assert not (tmp_path/"manifest-failure/report.json").exists()
    assert receipt["cumulative_costs"]["applied_updates"] == 8
    assert receipt["passed"] is False
    shutil.rmtree(tmp_path/"manifest-failure")


def test_expired_budget_stops_before_reader_or_model(tmp_path, mechanics):
    _, kwargs, *_ = mechanics
    def deny(*args, **kwargs):
        raise AssertionError("expired run must not construct model or read data")
    with pytest.raises(TimeoutError,match="wall_budget"):
        pilot._execute_attempt(tmp_path/"timeout",attempt_id="timeout",max_wall_seconds=-1,
                               **{**kwargs,"reader":deny,"session_factory":deny})
    receipt = json.loads((tmp_path/"timeout/failure-receipt.json").read_text())
    assert receipt["cumulative_costs"]["attempted_updates"] == 0


@pytest.mark.parametrize("kind", ["cuda_synthetic", "cuda_fresh_process"])
def test_gate_accepts_only_native_complete_report_shapes(tmp_path, monkeypatch, kind):
    # Pure metadata fixture, not a CUDA observation or experiment artifact.
    report = {"passed":True,"device":"cuda","real_data_accessed":False,"actual_updates":24,"actual_draws":48,
              "config":{"validation_mode":"required"},"arms":[],
              **{key:False for key in ("test_inputs_accessed","test_targets_accessed","test_metrics_computed")}}
    if kind == "cuda_synthetic":
        report["schema_version"]="HarmonicFusionSyntheticTrainerGate@1.1.0"
        checkpoint={"sha256":"e"*64,"size_bytes":100}
        for arm in ("control","fusion"):
            report["arms"].append({"arm":arm,"actual_updates":12,"actual_draws":24,"committed_final_update":6,
                "sampler_position":12,"validation_updates":[0,3,6],"interrupted_matches_uninterrupted":True,
                "learning_diagnostics":{"model_state_changed":True,"fusion_weight_change":{"changed_parameters":1,
                    "delta_l2":0.1,"nonzero_output_head_count":1,"optimizer_steps":{"fusion.weight":6}}},
                "final_training_checkpoint":checkpoint,
                "fusion_export":{"sha256":"f"*64,"size_bytes":50,"parent_training_checkpoint":checkpoint}})
    if kind == "cuda_fresh_process":
        report.update(schema_version="HarmonicFusionFreshProcessGate@1.0.0",fresh_process_resume=True,parent_process_id=11)
        for index,arm in enumerate(("control","fusion")):
            report["arms"].append({"arm":arm,"whole_state_equal":True,"worker_process_id":12+index,
                "worker_runtime_readiness":{"device_type":"cuda","cuda_device_count":1,
                    "verified_cuda_ordinals":[0],"saved_rng_applied":False}})
    path = tmp_path/"fixture.json";path.write_text(json.dumps(report));pin=pilot.artifact_pin(path)
    bindings={"source_manifest_sha256":"a"*64,"runtime_fingerprint":"b"*64}
    monkeypatch.setattr(pilot,"_record",lambda *args,**kw:{"experiment_id":"fixture", "status":"completed",
        "validity":{"valid":True},"kind":"engineering_gate", "artifacts":[pin],
        "protocol":{"fusion_pilot_bindings":{"role":kind,**bindings}},
        "data_access":{key:False for key in ("test_inputs_accessed","test_targets_accessed","test_metrics_computed")}})
    assert pilot.verify_gate({"record":{},"report":pin},role=kind,bindings=bindings) == "fixture"
    report["arms"][1]["whole_state_equal" if kind == "cuda_fresh_process" else "interrupted_matches_uninterrupted"] = False
    path.write_text(json.dumps(report));pin=pilot.artifact_pin(path)
    with pytest.raises(ValueError):
        pilot.verify_gate({"record":{},"report":pin},role=kind,bindings=bindings)


def test_exported_control_uses_old_format_and_fusion_preserves_new_weights(mechanics):
    from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
    from music_critic.experiments.analysisgnn.harmonic_fusion_checkpoint import load_fusion_checkpoint
    _, _, report, *_ = mechanics
    for arm, loader in (("control",load_model_checkpoint),("fusion",load_fusion_checkpoint)):
        pin=report["arms"][arm]["final_model"]
        loaded=loader(pin["path"],expected_sha256=pin["sha256"],expected_size_bytes=pin["size_bytes"])
        assert loaded.model is not None
    assert report["arms"]["fusion"]["learning_observations"]["fusion_nonzero_scalars"] > 0


def test_resume_descriptor_cannot_mislabel_checkpoint_cursor(tmp_path, mechanics):
    _, kwargs, _, failure, _ = mechanics
    parent=deepcopy(failure["state"]);parent["unconfirmed_parent_tail"]=False
    parent["arms"]["control"]["checkpoint"]["key"][0]=4
    with pytest.raises(ValueError,match="resume_descriptor_mismatch"):
        pilot._execute_attempt(tmp_path/"bad-descriptor",attempt_id="bad-descriptor",parent=parent,**kwargs)
    report=json.loads((tmp_path/"bad-descriptor/failure-receipt.json").read_text())
    assert report["state"]["arms"]["control"]["attempted_updates"] == 0
    assert report["state"]["arms"]["control"]["applied_updates"] == 0
    assert report["cumulative_costs"]["applied_updates"] == failure["cumulative_costs"]["applied_updates"]


def test_second_arm_initialization_failure_does_not_charge_prior_arm_updates(tmp_path, mechanics):
    _, kwargs, *_ = mechanics
    def factory(enabled):
        if enabled:
            raise RuntimeError("second arm initialization failed")
        return kwargs["session_factory"](enabled)
    with pytest.raises(RuntimeError, match="second arm initialization failed"):
        pilot._execute_attempt(tmp_path/"second-arm", attempt_id="second-arm",
                               **{**kwargs, "session_factory": factory})
    report=json.loads((tmp_path/"second-arm/failure-receipt.json").read_text())
    assert report["cumulative_costs"]["applied_updates"] == 4
    assert report["state"]["arms"]["fusion"]["applied_updates"] == 0
    assert report["state"]["arms"]["fusion"]["attempted_updates"] == 0


def test_resolved_metadata_write_failure_retains_zero_work_receipt(tmp_path, mechanics, monkeypatch):
    _, kwargs, *_ = mechanics
    original=pilot._write_json
    def write(path, payload):
        if path.name == "resolved.json":
            raise OSError("resolved metadata storage failed")
        return original(path, payload)
    monkeypatch.setattr(pilot, "_write_json", write)
    with pytest.raises(OSError, match="resolved metadata storage failed"):
        pilot._execute_attempt(tmp_path/"resolve-failure", attempt_id="resolve-failure", **kwargs)
    report=json.loads((tmp_path/"resolve-failure/failure-receipt.json").read_text())
    assert report["failure"]["stage"] == "resolve"
    assert report["cumulative_costs"] == {"applied_updates":0, "attempted_updates":0,
        "train_reader_calls":0, "validation_reader_calls":0}
