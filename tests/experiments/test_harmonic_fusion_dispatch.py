"""Setup tests use fabricated metadata only, never a target runtime or corpus."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import pytest

from music_critic.experiments.analysisgnn import harmonic_fusion_dispatch as setup
from music_critic.experiments.analysisgnn import harmonic_fusion_data as data_module
from music_critic.experiments.analysisgnn import harmonic_fusion_pilot as pilot
from music_critic.experiments.analysisgnn import harmonic_fusion_prepare as prepare
from music_critic.experiments.analysisgnn import harmonic_fusion_reference as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.registry import (DATA_ACCESS_FIELDS, canonical_json_bytes,
    seal_record, validate_record, load_record, write_record_once)
from test_harmonic_fusion_pilot_admission import (
    admission, _gate_report, _record, _registered_artifact, _write_json, _write_record,
)


@pytest.mark.parametrize("validation,updates,draws", [(False,2,4),(True,512,1024)])
def test_generated_plan_is_canonical_closed_test_and_has_no_execution_claims(tmp_path, validation, updates, draws):
    record=setup.planned_record("FIXTURE-PLAN", protocol={"fusion_pilot_bindings":{
        "source_manifest_sha256":"a"*64}}, updates=updates, draws=draws,
        validation=validation, parents=["FIXTURE-PARENT"])
    validate_record(record, filename="FIXTURE-PLAN.json")
    path=tmp_path/"FIXTURE-PLAN.json";write_record_once(path,record)
    assert load_record(path)==record
    assert record["budget"]["actual"] is None and record["status"]=="planned"
    assert record["data_access"]["declared_policy"]["validation_allowed"] is validation
    assert record["data_access"]["test_inputs_accessed"] is None
    with pytest.raises(Exception):write_record_once(path,record)


def test_source_seal_covers_runtime_config_and_fixture_but_excludes_bytecode(tmp_path):
    root=tmp_path/"source"
    names=["pyproject.toml","src/model.py","scripts/run_harmonic_fusion_real_pilot.py",
        "configs/experiments/harmonic_fusion_pilot.json","tests/fixture.json"]
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("{}\n")
    cache=root/"src/__pycache__";cache.mkdir();(cache/"test.pyc").write_bytes(b"cache")
    result=setup.source_manifest(root,tmp_path/"source.json")
    assert {r["path"] for r in setup.checked_json(result)["files"]}==set(names)
    (root/"src/model.py").write_text("changed\n")
    with pytest.raises(ValueError,match="artifact_hash_mismatch"):
        setup.verify_source_manifest(result,root)


def test_source_symlink_rejected(tmp_path):
    root=tmp_path/"source";(root/"src").mkdir(parents=True)
    (root/"pyproject.toml").write_text("x")
    (root/"src/link.py").symlink_to(root/"pyproject.toml")
    with pytest.raises(ValueError,match="symlink_forbidden"):
        setup.source_manifest(root,tmp_path/"source.json")


def test_runtime_probe_rejects_unreserved_process_before_cuda(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    with pytest.raises(ValueError,match="reserved_uuid"):
        setup.capture_runtime(tmp_path/"runtime.json")
    assert not (tmp_path/"runtime.json").exists()


def test_dispatch_cli_runtime_forwards_explicit_path_without_training(tmp_path,monkeypatch,capsys):
    path=Path(__file__).resolve().parents[2]/"scripts/prepare_harmonic_fusion_dispatch.py"
    spec=importlib.util.spec_from_file_location("setup_cli",path)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    observed=[]
    monkeypatch.setattr(setup,"capture_runtime",lambda output:observed.append(output) or {"fixture":True})
    assert cli.main(["runtime","--output",str(tmp_path/"runtime.json")])==0
    assert observed==[tmp_path/"runtime.json"]
    assert json.loads(capsys.readouterr().out)=={"fixture":True}


@pytest.fixture
def dispatch(admission, monkeypatch):
    """Actual registration/admission, fabricated metadata, no payload access.

    The imported fixture forbids torch.load and CUDA probes. Only the remote
    data-manifest adapter, FrozenFusionData, and checkpoint byte authentication
    are replaced. Registry hashes, source inventory, config/runtime policy,
    sampler scheduling, and both gate validators run normally. These gate
    observations are invented unit fixtures, never scientific evidence.
    """
    item = admission
    server = item.root / "server"
    handoff = item.root / "handoff"
    records = item.root / "docs/experiments/records"
    history = handoff / "cuda002-history"
    history.mkdir(parents=True)
    source = setup.source_manifest(item.repo, item.root / "setup-source.json")
    runtime = item.request["runtime"]
    runtime_pin = _write_json(item.root / "runtime.json", runtime)
    report_pin = _write_json(history / "report.json", _gate_report("cuda_synthetic", runtime))
    bindings = {"role": "cuda_synthetic", "source_manifest_sha256": source["sha256"],
                "runtime_fingerprint": fingerprint(runtime)}
    _write_record(history / "EXP-FUSION-CUDA-SYNTHETIC-002.json",
        _record("EXP-FUSION-CUDA-SYNTHETIC-002", status="completed", bindings=bindings,
                artifacts=(report_pin,)))
    _write_json(history / "deployment-manifest.json", {"fixture_only": True})
    events = []

    class FixtureData:
        def __init__(self, manifest):
            assert manifest == {"fixture_only": True}
            events.append(("metadata_admitted", manifest))
            self.table = item.table
            self.bindings = {"fixture_data_binding": fingerprint(manifest)}

        def require_train_records(self, record_ids):
            assert record_ids and record_ids <= {"fixture-a", "fixture-b"}
            events.append(("train_records_required", set(record_ids)))

        def reader(self, *args, **kwargs):
            pytest.fail("setup must not open raw or target payloads")

        def verify_bindings(self):
            pytest.fail("setup must not hash warm payload files")

    def manifest_fixture(*, cache_index, metadata_pins, output):
        assert cache_index == setup.pin(server / "evidence/EXP-RESEARCH-DILEMMA-BINDINGS-001/cache-bindings.jsonl",
                                        setup.CACHE_SHA, 1233467)
        metadata = pilot.checked_json(metadata_pins)
        assert set(metadata) == {"split_assignments", "shift_eligibility", "class_weights", "joint_tuple_counts"}
        assert metadata["class_weights"] == setup.pin(item.repo / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json",
                                                       setup.WEIGHTS_SHA, 303658)
        events.append(("manifest_adapter", deepcopy(metadata)))
        return {"output": _write_json(output, {"fixture_only": True})}

    original_checked_bytes = setup.checked_bytes

    def checkpoint_auth_fixture(pin):
        if pin["path"] == str(server / "c2-model-only.pt"):
            assert pin == setup.pin(server / "c2-model-only.pt", setup.C2_SHA, 14844299)
            assert not Path(pin["path"]).exists()
            events.append(("checkpoint_auth_fixture", deepcopy(pin)))
            return b"fabricated byte authentication; no checkpoint exists"
        return original_checked_bytes(pin)

    monkeypatch.setattr(data_module, "FrozenFusionData", FixtureData)
    monkeypatch.setattr(prepare, "prepare_data_manifest", manifest_fixture)
    monkeypatch.setattr(setup, "checked_bytes", checkpoint_auth_fixture)
    return SimpleNamespace(root=item.root, repo=item.repo, records=records, events=events,
        reference_kwargs={"server_root": server, "repository_root": item.repo,
            "handoff_root": handoff, "runtime_pin": runtime_pin,
            "output_root": item.root / "reference-setup", "records_root": records,
            "comparison_id": "FIXTURE-COMPARISON", "experiment_id": "FIXTURE-REFERENCE"})


def _prepare_reference(item):
    pin = setup.prepare_reference(**item.reference_kwargs)
    return pin, pilot.checked_json(pin)


def _finalize_fixture(path, *, report_pin=None, status="completed"):
    """Simulate coordinator lifecycle finalization of a disposable unit record."""
    record = load_record(path)
    assert record["status"] == "planned"
    record.update(status=status, results={"fixture_only": True},
                  validity={"valid": status == "completed", "reason": "fabricated unit finalization"})
    record["budget"]["actual"] = record["budget"]["planned"]
    record["data_access"].update(access_evidence_status="verified_primary_evidence",
                                **{key: False for key in DATA_ACCESS_FIELDS})
    if report_pin is not None:
        record["artifacts"] = [_registered_artifact(report_pin)]
    record = seal_record(record)
    validate_record(record, filename=path.name)
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    return pilot.artifact_pin(path)


def _prepare_completed_reference(item):
    request_pin, request = _prepare_reference(item)
    report = _gate_report("real_train_reference", request["runtime"], pilot.stable_bindings(request))
    report.update(actual_updates=2, actual_draws=4, experiment_id="FIXTURE-REFERENCE")
    report_pin = _write_json(item.root / "reference-report.json", report)
    record_pin = _finalize_fixture(item.records / "FIXTURE-REFERENCE.json", report_pin=report_pin)
    return request_pin, request, record_pin, report_pin


def _pair_kwargs(item, request_pin, record_pin, report_pin):
    return {"reference_request_pin": request_pin, "reference_record_pin": record_pin,
        "reference_report_pin": report_pin, "repository_root": item.repo,
        "output_root": item.root / "pair-setup", "records_root": item.records,
        "attempt_id": "FIXTURE-ATTEMPT", "control_id": "FIXTURE-CONTROL", "fusion_id": "FIXTURE-FUSION"}


def test_prepare_reference_registers_canonical_plan_and_independent_immutable_copy(dispatch):
    pin, request = _prepare_reference(dispatch)
    admitted = reference.admit_reference_request(request, repository_root=dispatch.repo)
    assert admitted["experiment_id"] == "FIXTURE-REFERENCE"
    assert len(admitted["draws"]) == 2
    assert pin == pilot.artifact_pin(dispatch.root / "reference-setup/reference-request.json")
    tracked = dispatch.records / "FIXTURE-REFERENCE.json"
    planned = Path(request["experiment_record"]["path"])
    assert tracked != planned and tracked.stat().st_ino != planned.stat().st_ino
    record = load_record(tracked)
    assert tracked.read_bytes() == planned.read_bytes() == canonical_json_bytes(record) + b"\n"
    assert record["protocol"]["fusion_pilot_bindings"] == reference.reference_gate_bindings(pilot.stable_bindings(request))
    assert record["protocol"]["reference_train_smoke"] == reference.reference_protocol(pilot.stable_bindings(request))
    assert record["budget"] == {"planned": 2, "planned_draws": 4, "batch_size": 2, "actual": None}
    assert record["parent_ids"] == ["EXP-FUSION-CUDA-SYNTHETIC-002"]
    assert request["max_wall_seconds"] == 600
    assert [event[0] for event in dispatch.events].count("checkpoint_auth_fixture") == 1


def test_prepare_pair_accepts_finalized_reference_and_keeps_all_planned_copies_usable(dispatch):
    request_pin, reference_request, record_pin, report_pin = _prepare_completed_reference(dispatch)
    saved_reference = pilot.checked_bytes(reference_request["experiment_record"])
    assert pilot._record(record_pin, terminal=True)["status"] == "completed"
    assert pilot._record(reference_request["experiment_record"])["status"] == "planned"
    reference.admit_reference_request(reference_request, repository_root=dispatch.repo)
    pair_pin = setup.prepare_pair(**_pair_kwargs(dispatch, request_pin, record_pin, report_pin))
    pair = pilot.checked_json(pair_pin)
    admitted = pilot.admit_request(pair, repository_root=dispatch.repo)
    assert admitted["experiment_ids"] == {"control": "FIXTURE-CONTROL", "fusion": "FIXTURE-FUSION"}
    assert pair["parent"] is None and pair["max_wall_seconds"] == 7200
    assert set(pair["gates"]) == {"cuda_synthetic", "real_train_reference"}
    for arm, experiment_id in admitted["experiment_ids"].items():
        tracked = dispatch.records / (experiment_id + ".json")
        planned_pin = pair["experiment_records"][arm]
        planned = Path(planned_pin["path"])
        record = load_record(tracked)
        original = planned.read_bytes()
        assert tracked != planned and tracked.stat().st_ino != planned.stat().st_ino
        assert tracked.read_bytes() == original == canonical_json_bytes(record) + b"\n"
        assert record["protocol"]["fusion_pilot_bindings"] == {"arm": arm, **pilot.stable_bindings(pair)}
        assert record["budget"] == {"planned": 512, "planned_draws": 1024, "batch_size": 2, "actual": None}
        assert record["parent_ids"] == ["FIXTURE-REFERENCE"]
        _finalize_fixture(tracked)
        assert pilot.checked_bytes(planned_pin) == original
        assert pilot._record(planned_pin)["status"] == "planned"
    assert pilot.checked_bytes(reference_request["experiment_record"]) == saved_reference
    assert pilot.admit_request(pair, repository_root=dispatch.repo)["experiment_ids"] == admitted["experiment_ids"]


def test_existing_reference_setup_is_not_overwritten(dispatch):
    pin, request = _prepare_reference(dispatch)
    before = pilot.checked_bytes(pin)
    plan = pilot.checked_bytes(request["experiment_record"])
    with pytest.raises(FileExistsError):
        setup.prepare_reference(**dispatch.reference_kwargs)
    assert pilot.checked_bytes(pin) == before
    assert pilot.checked_bytes(request["experiment_record"]) == plan


@pytest.mark.parametrize("mutation,error", [
    ("unfinalized", "parent_record_not_terminal"),
    ("failed", "completed_cuda_gate_required"),
    ("wrong_identity", "reference_record_identity_mismatch"),
    ("report_tampered", "artifact_hash_mismatch"),
    ("same_arm_ids", "distinct_arm_ids_required"),
])
def test_prepare_pair_rejects_invalid_reference_before_registering_arms(dispatch, mutation, error):
    request_pin, request, record_pin, report_pin = _prepare_completed_reference(dispatch)
    kwargs = _pair_kwargs(dispatch, request_pin, record_pin, report_pin)
    if mutation == "unfinalized":
        kwargs["reference_record_pin"] = request["experiment_record"]
    elif mutation in {"failed", "wrong_identity"}:
        record = pilot.checked_json(record_pin)
        if mutation == "failed":
            record["status"] = "failed"
            path = Path(record_pin["path"])
        else:
            record["experiment_id"] = "FIXTURE-OTHER-REFERENCE"
            path = dispatch.records / "FIXTURE-OTHER-REFERENCE.json"
        kwargs["reference_record_pin"] = _write_record(path, record)
    elif mutation == "report_tampered":
        Path(report_pin["path"]).write_bytes(b"altered fixture report")
    else:
        kwargs["fusion_id"] = kwargs["control_id"]
    with pytest.raises(ValueError, match=error):
        setup.prepare_pair(**kwargs)
    assert not kwargs["output_root"].exists()
    assert not (dispatch.records / "FIXTURE-CONTROL.json").exists()
    assert not (dispatch.records / "FIXTURE-FUSION.json").exists()


def test_setup_cli_prepares_reference_and_pair_with_explicit_paths(dispatch, capsys):
    path = Path(__file__).resolve().parents[2] / "scripts/prepare_harmonic_fusion_dispatch.py"
    spec = importlib.util.spec_from_file_location("fixture_dispatch_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = ["reference"]
    for key, value in dispatch.reference_kwargs.items():
        if key == "runtime_pin":
            args.extend(["--runtime", value["path"]])
        else:
            args.extend(["--" + key.replace("_", "-"), str(value)])
    assert cli.main(args) == 0
    request_pin = json.loads(capsys.readouterr().out)
    request = pilot.checked_json(request_pin)
    report = _gate_report("real_train_reference", request["runtime"], pilot.stable_bindings(request))
    report["experiment_id"] = "FIXTURE-REFERENCE"
    report_pin = _write_json(dispatch.root / "reference-report.json", report)
    record_pin = _finalize_fixture(dispatch.records / "FIXTURE-REFERENCE.json", report_pin=report_pin)
    args = ["pair"]
    for key, value in _pair_kwargs(dispatch, request_pin, record_pin, report_pin).items():
        if key.endswith("_pin"):
            key, value = key.removesuffix("_pin"), value["path"]
        args.extend(["--" + key.replace("_", "-"), str(value)])
    assert cli.main(args) == 0
    pair_pin = json.loads(capsys.readouterr().out)
    assert pilot.admit_request(pilot.checked_json(pair_pin), repository_root=dispatch.repo)["experiment_ids"] == {
        "control": "FIXTURE-CONTROL", "fusion": "FIXTURE-FUSION"}


def test_reference_setup_collision_preserves_terminal_record_and_old_planned_copy(dispatch):
    _, request, record_pin, _ = _prepare_completed_reference(dispatch)
    terminal = pilot.checked_bytes(record_pin)
    planned = pilot.checked_bytes(request["experiment_record"])
    retry = {**dispatch.reference_kwargs, "output_root": dispatch.root / "colliding-reference-setup"}
    with pytest.raises(ValueError, match="immutable_collision"):
        setup.prepare_reference(**retry)
    assert pilot.checked_bytes(record_pin) == terminal
    assert pilot.checked_bytes(request["experiment_record"]) == planned
    assert not (retry["output_root"] / "reference-request.json").exists()


def test_pair_setup_collision_preserves_terminal_arm_and_old_planned_copy(dispatch):
    request_pin, _, record_pin, report_pin = _prepare_completed_reference(dispatch)
    kwargs = _pair_kwargs(dispatch, request_pin, record_pin, report_pin)
    pair_pin = setup.prepare_pair(**kwargs)
    pair = pilot.checked_json(pair_pin)
    planned = pilot.checked_bytes(pair["experiment_records"]["control"])
    terminal_pin = _finalize_fixture(dispatch.records / "FIXTURE-CONTROL.json")
    terminal = pilot.checked_bytes(terminal_pin)
    retry = {**kwargs, "output_root": dispatch.root / "colliding-pair-setup"}
    with pytest.raises(ValueError, match="immutable_collision"):
        setup.prepare_pair(**retry)
    assert pilot.checked_bytes(terminal_pin) == terminal
    assert pilot.checked_bytes(pair["experiment_records"]["control"]) == planned
    assert not (retry["output_root"] / "pilot-request.json").exists()


@pytest.mark.parametrize("symlink_location", ["repository", "src", "scripts", "tests", "configs"])
def test_source_seal_rejects_symlinked_walk_roots_before_publication(tmp_path, symlink_location):
    root = tmp_path / "repository"
    root.mkdir()
    for folder in ("src", "scripts", "tests", "configs"):
        (root / folder).mkdir()
    for name in ("pyproject.toml", "src/model.py", "scripts/run_harmonic_fusion_real_pilot.py",
                 "configs/experiments/harmonic_fusion_pilot.json", "tests/unit-fixture.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    linked = root if symlink_location == "repository" else root / symlink_location
    external = tmp_path / ("outside-" + symlink_location)
    linked.rename(external)
    linked.symlink_to(external, target_is_directory=True)
    output = tmp_path / "source-manifest.json"
    with pytest.raises(ValueError, match="symlink_forbidden"):
        setup.source_manifest(root, output)
    assert not output.exists()
