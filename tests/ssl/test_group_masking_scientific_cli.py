"""Scientific CLI admission and wiring, using temporary synthetic metadata only."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_critic.experiments.registry import seal_record
from music_critic.ssl import group_masking_scientific_cli as cli
from music_critic.ssl import group_masking_scientific_data as data_module
from music_critic.ssl.group_masking_pilot import GroupPilotConfig


ROOT = Path(__file__).resolve().parents[2]


def _dump(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return {"sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    pilot = GroupPilotConfig()
    recovery = replace(pilot, updates=100, lambda_ssl=0., lambda_theory=1.)
    source = json.loads((ROOT / "configs/ssl_group/random_vs_beat.json").read_text())["source_checkpoint"]
    request = tmp_path / "request.json"
    request_bound = _dump(request, {"schema": data_module.REQUEST_SCHEMA, "synthetic_test_metadata": True})
    plan = {"schema": cli.PLAN_SCHEMA, "pilot": asdict(pilot), "recovery": asdict(recovery),
            "checkpoint_interval": 50, "source_checkpoint": source, "data_request": request_bound,
            "prepared_data": None, "resume": None}
    plan_path = tmp_path / "plan.json"
    record_path = tmp_path / "EXP-SSL-SCI-CLI-FIXTURE-001.json"
    base = json.loads((ROOT / "tests/fixtures/ssl_group/planned_record_template.json").read_text())
    base.update(experiment_id=record_path.stem, phase="WS2-SSL-GROUP", status="planned")
    reads = []
    def load_request(payload):
        reads.append(payload)
        return SimpleNamespace(synthetic=True)
    monkeypatch.setattr(data_module, "ScientificDataRequest", load_request)
    monkeypatch.setattr(data_module, "preflight_data", lambda *a, **k: {"metadata_only": True})
    def seal(action="prepare"):
        _dump(plan_path, plan)
        record = deepcopy(base)
        record["kind"] = "engineering_gate" if action in {"size", "prepare"} else "scientific_comparison"
        record["data_access"]["declared_policy"].update(validation_allowed=True,
            test_inputs_allowed=False, test_targets_allowed=False, test_metrics_allowed=False, test_selection_allowed=False)
        record["protocol"] = {
            "implementation_fingerprint": cli.implementation_fingerprint(),
            "scientific_plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
            "data_request": plan["data_request"], "prepared_data": plan["prepared_data"],
            "source_checkpoint": source, "resume": plan["resume"],
            "pilot_config": pilot.contract(), "recovery_config": recovery.contract(), "operation": action}
        _dump(record_path, seal_record(record))
        return record
    seal()
    return SimpleNamespace(plan=plan, plan_path=plan_path, record_path=record_path,
        request=request, tmp=tmp_path, seal=seal, reads=reads)


def _preflight(item, action="prepare", **kwargs):
    return cli.preflight(plan_path=item.plan_path, record_path=item.record_path, action=action,
                         data_request_path=item.request, **kwargs)


def test_metadata_preflight_never_opens_model_or_sources(inputs, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_model_only_theory_warmstart", lambda *a, **k: pytest.fail("checkpoint load"))
    monkeypatch.setattr(data_module, "prepare_scientific_data", lambda *a, **k: pytest.fail("source prepare"))
    assert cli.main(["preflight-prepare", "--plan", str(inputs.plan_path), "--experiment-record",
        str(inputs.record_path), "--data-request", str(inputs.request)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["raw_sources_read"] is report["checkpoint_loaded"] is report["CUDA_initialized"] is False
    assert inputs.reads == [{"schema": data_module.REQUEST_SCHEMA, "synthetic_test_metadata": True}]


@pytest.mark.parametrize("damage,reason", [("unknown_code", "protocol_binding"), ("not_planned", "planned_record"),
    ("TEST", "TRAIN_VALIDATION"), ("no_validation", "TRAIN_VALIDATION"), ("wrong_kind", "kind_or_phase")])
def test_fail_closed_before_request_parse(inputs, damage, reason):
    record = json.loads(inputs.record_path.read_text())
    if damage == "unknown_code":
        record["protocol"]["implementation_fingerprint"] = "f" * 64
    elif damage == "not_planned":
        record["status"] = "running"
    elif damage == "TEST":
        record["data_access"]["declared_policy"]["test_inputs_allowed"] = True
    elif damage == "no_validation":
        record["data_access"]["declared_policy"]["validation_allowed"] = False
    else:
        record["kind"] = "unrelated"
    _dump(inputs.record_path, seal_record(record))
    with pytest.raises(ValueError, match=reason):
        _preflight(inputs)
    assert not inputs.reads


@pytest.mark.parametrize("field,value,reason", [("pilot", {"updates": 199}, "pilot_config_changed"),
    ("recovery", {"lambda_ssl": .1}, "recovery_config_changed"),
    ("checkpoint_interval", True, "checkpoint_interval_invalid"),
    ("data_request", None, "artifact_binding_required")])
def test_incomplete_or_changed_scientific_question_rejected(inputs, field, value, reason):
    if isinstance(value, dict):
        inputs.plan[field].update(value)
    else:
        inputs.plan[field] = value
    _dump(inputs.plan_path, inputs.plan)
    with pytest.raises(ValueError, match=reason):
        _preflight(inputs)
    assert not inputs.reads


def test_request_sha_mismatch_precedes_parser(inputs):
    inputs.request.write_text("changed metadata")
    with pytest.raises(ValueError, match="artifact_bytes_mismatch"):
        _preflight(inputs)
    assert not inputs.reads


def test_prepared_binding_required_for_run(inputs):
    inputs.seal("run")
    with pytest.raises(ValueError, match="artifact_binding_required"):
        _preflight(inputs, "run")
    assert not inputs.reads


def test_prepare_publishes_binding_and_durable_receipts(inputs, monkeypatch):
    out = inputs.tmp / "prepared"
    binding = {"schema": "synthetic_prepared_binding", "fingerprint": "a" * 64}
    def prepare(*args, on_receipt):
        assert json.loads((out / "preparation.jsonl").read_text().splitlines()[0])["status"] == "started"
        on_receipt({"status": "source_verified", "synthetic": True})
        return SimpleNamespace(binding=binding)
    monkeypatch.setattr(data_module, "prepare_scientific_data", prepare)
    assert cli.main(["prepare", "--plan", str(inputs.plan_path), "--experiment-record", str(inputs.record_path),
                     "--data-request", str(inputs.request), "--output", str(out)]) == 0
    result = json.loads((out / "result.json").read_text())
    assert result["optimizer_updates"] == 0 and result["checkpoint_loaded"] is False
    assert cli.read_bound(out / "prepared-binding.json", result["prepared_data"])
    with pytest.raises(FileExistsError):
        cli.write_json_once(out / "result.json", {"bad": True})
    assert json.loads((out / "result.json").read_text()) == result


def test_prepare_failure_never_publishes_success(inputs, monkeypatch):
    def fail(*args, **kwargs):
        raise KeyboardInterrupt("synthetic source interrupted")
    monkeypatch.setattr(data_module, "prepare_scientific_data", fail)
    out = inputs.tmp / "failure"
    with pytest.raises(KeyboardInterrupt):
        cli.main(["prepare", "--plan", str(inputs.plan_path), "--experiment-record", str(inputs.record_path),
                  "--data-request", str(inputs.request), "--output", str(out)])
    assert not (out / "result.json").exists()
    assert json.loads((out / "failure.json").read_text())["type"] == "KeyboardInterrupt"


def test_bound_read_rejects_nonregular_oversized_and_symlink(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"test")
    binding = {"sha256": sha256(b"test").hexdigest(), "size_bytes": 4}
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        cli.read_bound(link, binding)
    with pytest.raises(ValueError, match="binding_invalid"):
        cli.read_bound(source, binding, maximum=3)
    with pytest.raises(ValueError, match="duplicate_json_key"):
        cli._json_bytes(b'{"a":1,"a":2}')


def test_resume_declared_parent_must_match_actual_bound_manifest(inputs):
    prepared = inputs.tmp / "bound-data.json"
    inputs.plan["prepared_data"] = _dump(prepared, {"synthetic_binding": True})
    manifest = inputs.tmp / "workflow.json"
    binding = _dump(manifest, {"attempt_id": "EXP-ACTUAL-PARENT-001"})
    parent = inputs.plan["source_checkpoint"]["parent_experiment_id"]
    inputs.plan["resume"] = {**binding, "parent_experiment_id": parent}
    inputs.seal("resume")
    with pytest.raises(ValueError, match="resume_parent_identity_mismatch"):
        _preflight(inputs, "resume", prepared_data_path=prepared, resume_manifest=manifest)
    assert not inputs.reads


def test_run_cli_uses_strict_C2_factory_and_full_bound_runner(inputs, monkeypatch):
    import torch
    from music_critic.ssl import group_masking_scientific_runner as runner
    from music_critic.ssl.theory_warmstart import TheoryWarmstartSSLModel
    prepared = inputs.tmp / "bound-data.json"
    inputs.plan["prepared_data"] = _dump(prepared, {"synthetic_binding": True})
    # This wiring test does not deserialize a model; the separately tested
    # strict loader is spied on so the exact admitted C2 arguments are checked.
    source = inputs.tmp / "model.pt"
    source.write_bytes(b"synthetic-placeholder-only")
    inputs.plan["source_checkpoint"].update(sha256=sha256(source.read_bytes()).hexdigest(),
                                            size_bytes=source.stat().st_size)
    inputs.seal("run")
    data = object()
    monkeypatch.setattr(data_module, "load_bound_scientific_data", lambda *a, **k: data)
    loaded = []
    sentinel = object()
    def loader(destination, path, **kwargs):
        assert isinstance(destination, TheoryWarmstartSSLModel)
        assert path == source
        assert kwargs == {"expected_sha256": inputs.plan["source_checkpoint"]["sha256"],
                          "expected_size_bytes": source.stat().st_size,
                          "expected_state_fingerprint": inputs.plan["source_checkpoint"]["state_fingerprint"]}
        loaded.append(True)
        return sentinel, {}
    monkeypatch.setattr(cli, "load_model_only_theory_warmstart", loader)
    observed = []
    def run(**kwargs):
        assert kwargs["data"] is data
        assert kwargs["pilot_config"].updates == 200 and kwargs["recovery_config"].updates == 100
        assert kwargs["checkpoint_interval"] == 50 and kwargs["device"] == "cpu"
        assert kwargs["warmstart_factory"]() is sentinel
        assert kwargs["attempt_id"] == inputs.record_path.stem
        assert kwargs["contract_binding"]["prepared_data"] == inputs.plan["prepared_data"]
        observed.append(True)
        return {"status": "completed"}
    monkeypatch.setattr(runner, "run_scientific_workflow", run)
    before_threads = torch.get_num_threads()
    assert cli.main(["run", "--plan", str(inputs.plan_path), "--experiment-record", str(inputs.record_path),
                     "--data-request", str(inputs.request), "--prepared-data", str(prepared),
                     "--source-checkpoint", str(source), "--output", str(inputs.tmp / "execution")]) == 0
    assert loaded == observed == [True]
    assert torch.get_num_threads() == before_threads


@pytest.fixture
def cached_sizing_inputs(tmp_path, monkeypatch):
    from tests.ssl.test_group_masking_scientific_cached_data import make_cached_request
    directory = tmp_path / "synthetic-data"
    directory.mkdir()
    request = make_cached_request(directory, monkeypatch).to_dict()
    request["validation_limits"] = request["class_weights"] = None
    request_path = tmp_path / "size-request.json"
    request_bound = _dump(request_path, request)
    pilot = GroupPilotConfig()
    recovery = replace(pilot, updates=100, lambda_ssl=0., lambda_theory=1.)
    source = json.loads((ROOT / "configs/ssl_group/random_vs_beat.json").read_text())["source_checkpoint"]
    plan = {"schema": cli.PLAN_SCHEMA, "pilot": asdict(pilot), "recovery": asdict(recovery),
            "checkpoint_interval": 50, "source_checkpoint": source, "data_request": request_bound,
            "prepared_data": None, "resume": None}
    plan_path = tmp_path / "size-plan.json"
    record_path = tmp_path / "EXP-SSL-SIZE-CLI-FIXTURE-001.json"
    base = json.loads((ROOT / "tests/fixtures/ssl_group/planned_record_template.json").read_text())

    def seal(action="size"):
        _dump(plan_path, plan)
        record = deepcopy(base)
        record.update(experiment_id=record_path.stem, phase="WS2-SSL-GROUP", status="planned",
                      kind="engineering_gate" if action in {"size", "prepare"} else "scientific_comparison")
        record["data_access"]["declared_policy"].update(validation_allowed=True,
            test_inputs_allowed=False, test_targets_allowed=False, test_metrics_allowed=False, test_selection_allowed=False)
        record["protocol"] = {"implementation_fingerprint": cli.implementation_fingerprint(),
            "scientific_plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
            "data_request": plan["data_request"], "prepared_data": plan["prepared_data"],
            "source_checkpoint": source, "resume": plan["resume"],
            "pilot_config": pilot.contract(), "recovery_config": recovery.contract(), "operation": action}
        _dump(record_path, seal_record(record))
    seal()
    return SimpleNamespace(plan=plan, plan_path=plan_path, record_path=record_path,
        request=request_path, tmp=tmp_path, seal=seal)


def _size_args(item, *, action="size", output=None):
    args = [action, "--plan", str(item.plan_path), "--experiment-record", str(item.record_path),
            "--data-request", str(item.request)]
    if output is not None:
        args += ["--output", str(output)]
    return args


def test_cached_size_preflight_is_metadata_only_with_null_limits_and_weights(cached_sizing_inputs, monkeypatch, capsys):
    from music_critic.ssl import group_masking_scientific_cached_data as cached
    monkeypatch.setattr(cached.CacheFile, "read", lambda *a: pytest.fail("cache payload read"))
    monkeypatch.setattr(cached.reference, "load_frozen_class_weights", lambda *a: pytest.fail("class weights read"))
    monkeypatch.setattr(cli, "load_model_only_theory_warmstart", lambda *a, **k: pytest.fail("checkpoint read"))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    assert cli.main(_size_args(cached_sizing_inputs, action="preflight-size")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["operation"] == "size"
    assert report["raw_sources_read"] is report["checkpoint_loaded"] is report["CUDA_initialized"] is False


def test_cached_size_cli_publishes_real_fixture_counts_and_fsynced_receipts(cached_sizing_inputs, monkeypatch):
    from music_critic.ssl import group_masking_scientific_cached_data as cached
    output = cached_sizing_inputs.tmp / "sized"
    original_read, original_fsync = cached.CacheFile.read, cli.os.fsync
    verified, synced = [], []
    def read(item):
        first = json.loads((output / "sizing.jsonl").read_text().splitlines()[0])
        assert first["status"] == "started" and first["operation"] == "size"
        assert item.path.name != "target_sidecar.json"
        verified.append(item.path.name)
        return original_read(item)
    def fsync(fd):
        synced.append(fd)
        return original_fsync(fd)
    monkeypatch.setattr(cached.CacheFile, "read", read)
    monkeypatch.setattr(cached.reference, "load_frozen_class_weights", lambda *a: pytest.fail("class weights read"))
    monkeypatch.setattr(cli, "load_model_only_theory_warmstart", lambda *a, **k: pytest.fail("checkpoint read"))
    monkeypatch.setattr(cli.os, "fsync", fsync)
    threads = cli.torch.get_num_threads()
    assert cli.main(_size_args(cached_sizing_inputs, output=output)) == 0
    result = json.loads((output / "result.json").read_text())
    sizing = cli._json_bytes(cli.read_bound(output / "sizing-report.json", result["sizing_report"]))
    assert result["operation"] == "size" and result["status"] == "completed"
    assert result["optimizer_updates"] == 0 and result["checkpoint_loaded"] is False
    assert len(sizing["records"]) == 3
    assert result["access_observations"]["raw_files_verified"] == verified.count("piece.json") == 3
    assert result["access_observations"]["sidecar_files_verified"] == 0
    assert result["access_observations"]["test_source_reads"] == 0
    assert synced and cli.torch.get_num_threads() == threads
    assert not (output / "prepared-binding.json").exists()
    with pytest.raises(FileExistsError):
        cli.main(_size_args(cached_sizing_inputs, output=output))


def test_cached_size_interrupt_preserves_partial_receipts_without_success(cached_sizing_inputs, monkeypatch):
    from music_critic.ssl import group_masking_scientific_cached_data as cached
    output = cached_sizing_inputs.tmp / "sizing-interrupted"
    original = cached.CacheFile.read
    raw_reads = []
    def interrupt(item):
        if item.path.name == "piece.json":
            raw_reads.append(item.path)
            if len(raw_reads) == 2:
                raise KeyboardInterrupt("synthetic interruption after one completed raw piece")
        return original(item)
    monkeypatch.setattr(cached.CacheFile, "read", interrupt)
    threads = cli.torch.get_num_threads()
    with pytest.raises(KeyboardInterrupt):
        cli.main(_size_args(cached_sizing_inputs, output=output))
    assert not (output / "result.json").exists()
    assert not (output / "sizing-report.json").exists()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["operation"] == "size" and failure["type"] == "KeyboardInterrupt"
    assert failure["last_receipt"]["access_observations"]["raw_files_verified"] == 1
    assert failure["access_observations_complete"] is False
    assert failure["checkpoint_loaded"] is False and failure["optimizer_updates"] == 0
    receipts = [json.loads(line) for line in (output / "sizing.jsonl").read_text().splitlines()]
    assert receipts[0]["status"] == "started" and receipts[-1]["status"] == "failed"
    assert cli.torch.get_num_threads() == threads


@pytest.mark.parametrize("damage,reason", [
    ("wrong_kind", "experiment_kind_or_phase_invalid"),
    ("wrong_operation", "planned_protocol_binding_mismatch"),
    ("running", "planned_record_required"),
    ("TEST", "TRAIN_VALIDATION_only_policy_required"),
])
def test_size_has_separate_planned_admission_before_request_bytes(cached_sizing_inputs, monkeypatch, damage, reason):
    record = json.loads(cached_sizing_inputs.record_path.read_text())
    if damage == "wrong_kind":
        record["kind"] = "scientific_comparison"
    elif damage == "wrong_operation":
        record["protocol"]["operation"] = "prepare"
    elif damage == "running":
        record["status"] = "running"
    else:
        record["data_access"]["declared_policy"]["test_targets_allowed"] = True
    _dump(cached_sizing_inputs.record_path, seal_record(record))
    monkeypatch.setattr(cli, "read_bound", lambda *a, **k: pytest.fail("request bytes opened before admission"))
    with pytest.raises(ValueError, match=reason):
        cli.main(_size_args(cached_sizing_inputs, action="preflight-size"))


def test_size_request_sha_checked_before_cached_metadata(cached_sizing_inputs, monkeypatch):
    from music_critic.ssl import group_masking_scientific_cached_data as cached
    cached_sizing_inputs.request.write_text("forged request bytes")
    monkeypatch.setattr(cached, "preflight_data", lambda *a, **k: pytest.fail("metadata read before request hash"))
    with pytest.raises(ValueError, match="artifact_bytes_mismatch"):
        cli.main(_size_args(cached_sizing_inputs, action="preflight-size"))


def test_size_requires_cached_backend_and_cpu(inputs, monkeypatch):
    inputs.seal("size")
    with pytest.raises(ValueError, match="sizing_requires_cached_request"):
        cli.main(_size_args(inputs, action="preflight-size"))
    assert not inputs.reads
    monkeypatch.setattr(cli, "preflight", lambda **k: pytest.fail("noncpu sizing reached preflight"))
    with pytest.raises(ValueError, match="sizing_cpu_only"):
        cli.main(_size_args(inputs, action="preflight-size") + ["--device", "cuda:0"])


def test_size_does_not_relax_prepare_null_input_requirements(cached_sizing_inputs):
    cached_sizing_inputs.seal("prepare")
    with pytest.raises(ValueError, match="invalid_artifact_binding"):
        cli.main(_size_args(cached_sizing_inputs, action="preflight-prepare"))


def test_size_forbids_existing_prepared_or_resume_bindings(cached_sizing_inputs):
    cached_sizing_inputs.plan["prepared_data"] = {"sha256": "a" * 64, "size_bytes": 2}
    cached_sizing_inputs.seal("size")
    with pytest.raises(ValueError, match="size_must_compile_fresh_report"):
        cli.main(_size_args(cached_sizing_inputs, action="preflight-size"))
