"""Fresh CPU child fixtures and sealed synthetic index tests; no corpus/GPU."""

from copy import deepcopy
from dataclasses import asdict, fields, replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from music_critic.experiments.analysisgnn import performance_memory_replay as mr
from music_critic.experiments.analysisgnn import performance_memory as pm
from music_critic.experiments.analysisgnn.performance_data import PreparedRecord, PERFORMANCE_DATA_SCHEMA
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, file_sha256, load_record, render_registry,
    seal_record, validate_record, write_record_once,
)


class FixtureSource:
    def __call__(self, record_id):
        return mr.existing._fixture_callback(record_id)


class FailingSecondUpdateSource:
    def __init__(self):
        self.calls = 0

    def __call__(self, record_id):
        self.calls += 1
        if self.calls == 3:
            raise FileNotFoundError("fixture second preparation failed")
        return mr.existing._fixture_callback(record_id)


class SlowSource:
    def __call__(self, record_id):
        time.sleep(30)
        return mr.existing._fixture_callback(record_id)


def _write(path, value):
    path.write_bytes(canonical_json_bytes(json.loads(json.dumps(value))) + b"\n")


@pytest.fixture
def sealed_index(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init",), ("config", "user.name", "Fixture"),
                 ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "pyproject.toml").write_text("# isolated memory replay fixture\n")
    subprocess.run(["git", "-C", str(root), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    batch, sidecar = FixtureSource()("synthetic:a")
    payload = json.dumps(sidecar, indent=3).encode()
    metadata = pm.TrainRecordMetadata(record_id="synthetic:a", source_component_id="component-a", split="train",
        canonical_sha256="a" * 64, sidecar_sha256=sha256(payload).hexdigest(), metadata_sha256="b" * 64,
        binding_seal="c" * 64, source_seal="d" * 64, code_seal="e" * 64, split_sha256="f" * 64,
        source_versions=(("fixture", "synthetic-v1"),))
    row = pm.build_size_index_row(PreparedRecord(batch.raw_graph_batch.to_data_list()[0], sidecar, metadata.binding_seal),
        metadata=metadata, compact_sidecar_bytes=payload, cache_representation="source_json_bytes")
    selected = pm.select_extreme_b2_cases([row], expected_train_ids=[metadata.record_id])
    artifact_root = root / "outputs/index"
    artifact_root.mkdir(parents=True)
    bindings = {"record_ids": [row.record_id], "components": {row.record_id: metadata.source_component_id},
        "bindings": {"schema": PERFORMANCE_DATA_SCHEMA, "source_seal": metadata.source_seal,
            "code_seal": metadata.code_seal, "split_sha256": metadata.split_sha256,
            "record_bindings": {row.record_id: metadata.binding_seal}}}
    artifacts = {"size-index.jsonl": row.to_dict(), "source-bindings-initial.json": bindings,
                 "source-bindings-final.json": bindings, "extreme-b2-cases.json": asdict(selected)}
    for name, value in artifacts.items():
        _write(artifact_root / name, value)
    record = {"schema_version": "1.0.0", "experiment_id": "EXP-INDEX", "kind": "data_audit", "phase": "9E-P1",
        "title": "Synthetic completed index", "status": "completed", "question": "Fixture?",
        "hypothesis": "Fixture", "decision_use": "Synthetic test only", "repository": {}, "seeds": [], "budget": {},
        "protocol": {"schema_version": "AnalysisGNNPerformanceMemoryIndexRun@1.0.0", "mode": "synthetic",
                     "split": "train", "expected_train_records": 1},
        "data_access": {**{name: None for name in DATA_ACCESS_FIELDS},
            "access_evidence_status": "pending_primary_evidence", "declared_policy": {
                "source": "synthetic test", "validation_allowed": False, "test_inputs_allowed": False,
                "test_targets_allowed": False, "test_metrics_allowed": False, "test_selection_allowed": False}},
        "results": {"supplied_index_fingerprint": selected.supplied_index_fingerprint, "indexed_records": 1,
                    "size_index_sha256": file_sha256(artifact_root / "size-index.jsonl")},
        "validity": {"valid": True}, "claims": {"supported": [], "not_supported": ["production readiness"], "caveats": []},
        "artifacts": [{"role": name, "sha256": file_sha256(artifact_root / name), "size_bytes": (artifact_root / name).stat().st_size,
                       "availability": "local_ignored", "locator": (artifact_root / name).relative_to(root).as_posix()}
                      for name in artifacts], "import_provenance": {"mode": "synthetic_test"}}
    records_root = root / "docs/experiments/records"
    write_record_once(records_root / "EXP-INDEX.json", seal_record(record))
    inputs = mr.MemoryIndexInputs("EXP-INDEX", records_root, artifact_root / "size-index.jsonl",
        artifact_root / "source-bindings-initial.json", artifact_root / "source-bindings-final.json",
        artifact_root / "extreme-b2-cases.json")
    return root, inputs, row


def _run(sealed_index, **kwargs):
    root, inputs, row = sealed_index
    options = dict(experiment_id="EXP-MEMORY", parent_experiment_id="EXP-INDEX", output_root=root / "outputs/memory",
        index_inputs=inputs, device="cpu", child_timeout_seconds=45, rss_sample_interval_seconds=0.02,
        repository_root=root, synthetic=True, synthetic_source_factory=FixtureSource)
    options.update(kwargs)
    return mr.run_registered_memory_replay(**options)


def _record(sealed_index):
    return load_record(sealed_index[0] / "docs/experiments/records/EXP-MEMORY.json")


def _reseal_index(inputs, *, change=None, artifacts=()):
    """Create intentionally corrupt but internally sealed fixture evidence."""
    path = inputs.records_root / "EXP-INDEX.json"
    record = load_record(path)
    for role in artifacts:
        item = next(item for item in record["artifacts"] if item["role"] == role)
        source = inputs.size_index.parent / role
        item.update(sha256=file_sha256(source), size_bytes=source.stat().st_size)
    if change is not None:
        change(record)
    _write(path, seal_record(record))


def test_authentication_regenerates_exact_finite_cases(sealed_index):
    root, inputs, row = sealed_index
    result = mr.load_completed_memory_index(inputs, synthetic=True)
    assert result["selection"]["ordered_b2_cases"] == [{"record_ids": [row.record_id, row.record_id],
        "coarse_lengths": [row.coarse_length, row.coarse_length], "attention_padding_proxy": 2 * row.coarse_length**2}]
    assert result["row_counts"] == {row.record_id: {"nodes": row.nodes, "edges": row.edges}}
    assert not (root / "docs/experiments/records/EXP-MEMORY.json").exists()
    with pytest.raises(ValueError, match="completed_train_memory_index"):
        mr.load_completed_memory_index(inputs)  # Synthetic evidence never enables production.


@pytest.mark.parametrize("role", ["size-index.jsonl", "source-bindings-initial.json", "source-bindings-final.json", "extreme-b2-cases.json"])
def test_artifact_tamper_fails_before_payload_interpretation(sealed_index, role):
    inputs = sealed_index[1]
    path = inputs.size_index.parent / role
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="index_artifact_mismatch"):
        mr.load_completed_memory_index(inputs, synthetic=True)


@pytest.mark.parametrize("change", [
    lambda record: record["protocol"].update(split="test"),
    lambda record: record["protocol"].update(split="validation"),
    lambda record: record["protocol"].update(schema_version="OtherIndex@1.0.0"),
    lambda record: record.update(kind="engineering_gate"),
    lambda record: record.update(status="failed", validity={"valid": False, "reason": "synthetic failure"}),
])
def test_non_production_kind_or_non_train_index_rejected(sealed_index, change):
    inputs = sealed_index[1]
    _reseal_index(inputs, change=change)
    with pytest.raises(ValueError, match="completed_train_memory_index"):
        mr.load_completed_memory_index(inputs, synthetic=True)


@pytest.mark.parametrize("split", ["validation", "test", None])
def test_even_resealed_nontrain_rows_rejected(sealed_index, split):
    inputs = sealed_index[1]
    row = json.loads(inputs.size_index.read_text())
    row["metadata"]["split"] = split
    _write(inputs.size_index, row)
    _reseal_index(inputs, artifacts=["size-index.jsonl"])
    with pytest.raises(ValueError, match="train_only"):
        mr.load_completed_memory_index(inputs, synthetic=True)


@pytest.mark.parametrize("problem,match", [("count", "complete_index"), ("derived", "derived_fields"),
    ("cases", "extreme_cases_mismatch"), ("binding", "row_source_binding_mismatch"),
    ("membership", "membership"), ("results", "index_result_mismatch"), ("duplicate", "duplicate")])
def test_resealed_semantic_corruption_still_fails(sealed_index, problem, match):
    inputs = sealed_index[1]
    roles = []
    change = None
    if problem == "count":
        change = lambda record: record["protocol"].update(expected_train_records=2)
    elif problem == "results":
        change = lambda record: record["results"].update(indexed_records=8)
    elif problem in ("binding", "membership"):
        binding = json.loads(inputs.bindings_initial.read_text())
        if problem == "binding":
            binding["bindings"]["code_seal"] = "0" * 64
        else:
            binding["record_ids"].append("excluded:record")
        _write(inputs.bindings_initial, binding)
        _write(inputs.bindings_final, binding)
        roles = ["source-bindings-initial.json", "source-bindings-final.json"]
    elif problem == "cases":
        cases = json.loads(inputs.extreme_cases.read_text())
        cases["ordered_b2_cases"][0]["attention_padding_proxy"] += 1
        _write(inputs.extreme_cases, cases)
        roles = ["extreme-b2-cases.json"]
    elif problem == "duplicate":
        inputs.size_index.write_bytes(inputs.size_index.read_bytes() * 2)
        roles = ["size-index.jsonl"]
        change = lambda record: record["protocol"].update(expected_train_records=2)
    else:
        row = json.loads(inputs.size_index.read_text())
        row["nodes"] += 1
        _write(inputs.size_index, row)
        roles = ["size-index.jsonl"]
    _reseal_index(inputs, change=change, artifacts=roles)
    with pytest.raises(ValueError, match=match):
        mr.load_completed_memory_index(inputs, synthetic=True)


def test_real_cpu_spawn_two_strategies_equal_with_explicit_synthetic_rss_backend(sealed_index, monkeypatch):
    # Mac sandbox forbids ps. This assertion covers sampling/lifecycle and real
    # fresh-child CPU updates, not measured parent-observed memory acceptance.
    monkeypatch.setattr(mr, "read_process_memory", lambda pid: {
        "rss_bytes": 123456, "kernel_hwm_bytes": None, "backend": "synthetic_test_sample"})
    authenticate = mr.load_completed_memory_index
    def checked(inputs, **kwargs):
        assert _record(sealed_index)["status"] == "running"
        history = json.loads((sealed_index[0] / "outputs/memory/lifecycle-start.json").read_text())
        assert [row["status"] for row in history] == ["planned", "running"]
        for row in history:
            validate_record(row)
        return authenticate(inputs, **kwargs)
    monkeypatch.setattr(mr, "load_completed_memory_index", checked)
    result = _run(sealed_index)
    record = _record(sealed_index)
    assert result["equivalence_passed"] is True and result["case_count"] == 1
    assert result["child_count"] == 2 and result["applied_updates"] == record["budget"]["actual"] == 4
    assert record["status"] == "completed"
    assert record["budget"]["planned"] == 256
    output = sealed_index[0] / "outputs/memory"
    case_plan = json.loads((output / "resolved-case-plan.json").read_text())
    assert case_plan["exact_planned_applied_updates"] == 4 and case_plan["case_count"] == 1
    pids = []
    for kind in ("baseline", "optimized"):
        directory = output / "case-000" / kind
        child = json.loads((directory / "result.json").read_text())
        monitor = json.loads((directory / "parent-monitor.json").read_text())
        states = [json.loads(line) for line in (directory / "states.jsonl").read_text().splitlines()]
        resources = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
        pids.append(child["pid"])
        assert monitor["pid"] == child["pid"] and monitor["exitcode"] == 0
        assert monitor["rss_sample_count"] > 0 and monitor["sampled_rss_peak_bytes"] > 0
        assert child["final_resources"]["kernel_ru_maxrss_bytes"] > 0
        assert [row["update"] for row in states] == [1, 2]
        assert {"model", "optimizer", "scheduler", "scaler", "rng", "training", "full_state"} == set(states[0]["state_fingerprints"])
        assert states[0]["state_fingerprints"]["model"] != states[1]["state_fingerprints"]["model"]
        assert states[0]["state_fingerprints"]["optimizer"] != states[1]["state_fingerprints"]["optimizer"]
        assert all(row["cuda"] is None for row in resources)
        assert {"before_preparation", "after_preparation", "after_update"} <= {row["stage"] for row in resources}
    assert len(set(pids + [os.getpid()])) == 3
    assert not list(output.rglob("*.pt")) and not list(output.rglob("*.ckpt"))
    for artifact in record["artifacts"]:
        assert file_sha256(sealed_index[0] / artifact["locator"]) == artifact["sha256"]
    assert all(record["data_access"][name] is None for name in DATA_ACCESS_FIELDS)
    assert (sealed_index[0] / "docs/EXPERIMENT_LEDGER.md").read_text() == render_registry(sealed_index[1].records_root)
    before = (sealed_index[1].records_root / "EXP-MEMORY.json").read_bytes()
    with pytest.raises(ValueError, match="fresh_id_and_output"):
        _run(sealed_index)
    assert (sealed_index[1].records_root / "EXP-MEMORY.json").read_bytes() == before


def test_unavailable_parent_rss_cannot_pass_memory_gate(sealed_index, monkeypatch):
    monkeypatch.setattr(mr, "read_process_memory", lambda pid: {
        "rss_bytes": None, "kernel_hwm_bytes": None, "backend": "synthetic_unavailable"})
    with pytest.raises(ValueError, match="no_valid_parent_rss_samples"):
        _run(sealed_index)
    record = _record(sealed_index)
    assert record["status"] == "failed" and record["budget"]["actual"] == 2
    assert not record["claims"]["supported"]


def test_linux_proc_rss_and_kernel_hwm_units(monkeypatch):
    monkeypatch.setattr(mr.sys, "platform", "linux")
    def read(path):
        assert path == Path("/proc/123/status")
        return "Name:\tfixture\nVmRSS:\t1024 kB\nVmHWM:\t2048 kB\n"
    monkeypatch.setattr(Path, "read_text", read)
    assert mr.read_process_memory(123) == {"rss_bytes": 1024**2, "kernel_hwm_bytes": 2 * 1024**2, "backend": "proc_status"}


def test_current_process_kernel_highwater_is_available_without_ps():
    observation = mr._resources("cpu")
    assert observation["pid"] == os.getpid() and observation["kernel_ru_maxrss_bytes"] > 0
    assert observation["cuda"] is None and observation["kernel_scope"] == "fresh_child_process_lifetime"


def test_child_failure_retains_one_applied_update_without_equality_claim(sealed_index):
    with pytest.raises(ValueError, match="owned_child_failed"):
        _run(sealed_index, synthetic_source_factory=FailingSecondUpdateSource)
    record = _record(sealed_index)
    directory = sealed_index[0] / "outputs/memory/case-000/baseline"
    failure = json.loads((directory / "failure.json").read_text())
    assert record["status"] == "failed" and record["budget"]["actual"] == 1
    assert failure["applied_updates"] == 1 and failure["stage"] == "native_preparation"
    assert not record["claims"]["supported"] and not record["results"]
    assert (directory / "states.jsonl").exists() and (directory / "parent-monitor.json").exists()
    assert not (directory.parent / "comparison.json").exists()


def test_timeout_stops_only_owned_child_and_retains_failed_evidence(sealed_index):
    with pytest.raises(ValueError, match="owned_child_timeout"):
        _run(sealed_index, synthetic_source_factory=SlowSource, child_timeout_seconds=0.3)
    record = _record(sealed_index)
    monitor = json.loads((sealed_index[0] / "outputs/memory/case-000/baseline/parent-monitor.json").read_text())
    assert record["status"] == "failed" and record["budget"]["actual"] == 0
    assert monitor["pid"] != os.getpid() and monitor["exitcode"] != 0
    assert not record["claims"]["supported"]


def test_common_identity_then_explicit_subset_payload_binding(monkeypatch):
    calls = []
    full = {"schema": PERFORMANCE_DATA_SCHEMA, "source_seal": "a", "code_seal": "b", "split_sha256": "c",
            "record_bindings": {"train-a": "aa", "train-b": "bb", "unused-train": "cc"}}
    class Loader:
        def __init__(self, *, paths, split):
            assert paths == "fixture_paths" and split == "train"
            self.ids = set()
            self._train_ids = frozenset(full["record_bindings"])
        def validate_all_bindings(self, *, expected=None):
            calls.append(("validate", deepcopy(expected)))
            return {**full, "record_bindings": {key: full["record_bindings"][key] for key in sorted(self.ids)}}
        def check_binding(self, record_id, *, split):
            assert split == "train"
            calls.append(("check", record_id))
            self.ids.add(record_id)
            return full["record_bindings"][record_id]
    monkeypatch.setattr(mr, "FrozenProductionRecordLoader", Loader)
    loader, subset = mr._bind_pair("fixture_paths", full, ["train-b", "train-a"])
    assert list(subset["record_bindings"]) == ["train-a", "train-b"]
    assert calls[:3] == [("validate", None), ("check", "train-a"), ("check", "train-b")]
    assert calls[3] == ("validate", subset)
    assert "unused-train" not in loader.ids
    with pytest.raises(ValueError, match="common_source_identity"):
        mr._bind_pair("fixture_paths", {**full, "code_seal": "wrong"}, ["train-a", "train-a"])


@pytest.mark.parametrize("problem", ["omitted_unselected", "substituted_unselected"])
def test_full_index_membership_matches_actual_train_before_any_pair_payload_read(monkeypatch, problem):
    actual_ids = frozenset({"train-a", "train-b", "unused-train"})
    common = {"schema": PERFORMANCE_DATA_SCHEMA, "source_seal": "a", "code_seal": "b", "split_sha256": "c"}
    supplied = {**common, "record_bindings": {"train-a": "aa", "train-b": "bb"}}
    if problem == "substituted_unselected":
        supplied["record_bindings"]["excluded-record"] = "cc"
        assert len(supplied["record_bindings"]) == len(actual_ids)
    calls = []
    class Loader:
        _train_ids = actual_ids
        def __init__(self, *, paths, split):
            assert paths == "fixture_paths" and split == "train"
        def validate_all_bindings(self, *, expected=None):
            assert expected is None
            calls.append("common_identity_only")
            return {**common, "record_bindings": {}}
        def check_binding(self, record_id, *, split):
            pytest.fail("membership guard must reject before any actual selected pair payload read")
    monkeypatch.setattr(mr, "FrozenProductionRecordLoader", Loader)
    with pytest.raises(ValueError, match="full_train_membership_mismatch"):
        mr._bind_pair("fixture_paths", supplied, ["train-a", "train-b"])
    assert calls == ["common_identity_only"]


@pytest.mark.parametrize("current_stage,journal_updates,expected", [
    ("native_preparation", [1], 1), ("update", [1], None), ("update", [1, 2], 2), ("after_update", [1, 2], 2)])
def test_interruption_inside_update_keeps_actual_unknown(tmp_path, current_stage, journal_updates, expected):
    directory = tmp_path / "case-000/baseline"
    directory.mkdir(parents=True)
    _write(directory / "phase.json", {"stage": current_stage, "update": 2})
    for number in journal_updates:
        mr._append(directory / "applied-updates.jsonl", {"update": number, "applied": True})
    result = mr._applied_accounting(tmp_path)
    assert result["actual"] == expected and result["known_applied_updates"] == len(journal_updates)
    assert bool(result["uncertain_applied_children"]) == (expected is None)


@pytest.mark.parametrize("corruption,match", [("rng", "loss_or_full_state"), ("loss", "loss_or_full_state"),
    ("initial", "initial_state"), ("incomplete", "incomplete_case"), ("same_pid", "fresh_process"),
    ("case", "fresh_process"), ("strategy", "fresh_process"), ("weights", "class_weight_payload"),
    ("duplicate_update", "loss_or_full_state")])
def test_comparison_requires_two_completed_same_case_full_states(tmp_path, corruption, match):
    case = {"record_ids": ["a", "a"], "coarse_lengths": [3, 3], "attention_padding_proxy": 18}
    results = {kind: {"status": "completed", "applied_updates": 2, "strategy": kind, "pid": index + 100,
        "device": "cpu", "case": deepcopy(case), "initial": {"full_state": "initial"},
        "class_weight_payload_fingerprint": "a" * 64} for index, kind in enumerate(("baseline", "optimized"))}
    states = {kind: [{"update": number, "applied": True, "total": 1.25, "record_ids": ["a", "a"],
        "applied_shifts": [0, 0], "state_fingerprints": {"rng": "r", "model": "m", "optimizer": "o", "full_state": "f"}}
        for number in (1, 2)] for kind in results}
    if corruption == "rng":
        states["optimized"][0]["state_fingerprints"]["rng"] = "changed"
    elif corruption == "loss":
        states["optimized"][0]["total"] = 2.5
    elif corruption == "initial":
        results["optimized"]["initial"]["full_state"] = "changed"
    elif corruption == "incomplete":
        results["optimized"]["applied_updates"] = 1
    elif corruption == "same_pid":
        results["optimized"]["pid"] = results["baseline"]["pid"]
    elif corruption == "case":
        results["optimized"]["case"]["record_ids"] = ["b", "b"]
    elif corruption == "strategy":
        results["optimized"]["strategy"] = "baseline"
    elif corruption == "weights":
        results["optimized"]["class_weight_payload_fingerprint"] = "b" * 64
    elif corruption == "duplicate_update":
        for rows in states.values():
            rows[1]["update"] = 1
    for kind in results:
        (tmp_path / kind).mkdir()
        _write(tmp_path / kind / "result.json", results[kind])
        for row in states[kind]:
            mr._append(tmp_path / kind / "states.jsonl", row)
    with pytest.raises(ValueError, match=match):
        mr.compare_case(tmp_path)


def test_symlink_index_artifact_and_parent_id_are_rejected(sealed_index):
    root, inputs, row = sealed_index
    linked = root / "linked-index"
    linked.symlink_to(inputs.size_index)
    with pytest.raises(ValueError, match="regular_index_artifact"):
        mr.load_completed_memory_index(replace(inputs, size_index=linked), synthetic=True)
    with pytest.raises(ValueError, match="parent_experiment_id_invalid"):
        _run(sealed_index, parent_experiment_id="../EXP-INDEX")
    assert not (inputs.records_root / "EXP-MEMORY.json").exists()


@pytest.mark.parametrize("field,value", [("child_timeout_seconds", 0), ("child_timeout_seconds", True),
    ("rss_sample_interval_seconds", -1), ("rss_sample_interval_seconds", 46)])
def test_explicit_finite_positive_process_bounds(sealed_index, field, value):
    with pytest.raises(ValueError, match="positive_timeout|interval_exceeds"):
        _run(sealed_index, **{field: value})
    assert _record(sealed_index)["status"] == "failed"


def test_registered_rejection_of_index_tamper_preserves_planned_history(sealed_index):
    inputs = sealed_index[1]
    inputs.size_index.write_bytes(b"not-json\n")
    with pytest.raises(ValueError, match="index_artifact_mismatch"):
        _run(sealed_index)
    assert _record(sealed_index)["status"] == "failed"
    history = json.loads((sealed_index[0] / "outputs/memory/lifecycle-start.json").read_text())
    assert [row["status"] for row in history] == ["planned", "running"]
    assert not list((sealed_index[0] / "outputs/memory").glob("case-*"))


def test_production_cli_requires_concrete_paths_and_has_no_fixture_or_checkpoint_switch(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_performance_memory_replay.py"
    spec = importlib.util.spec_from_file_location("memory_replay_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = []
    monkeypatch.setattr(module, "run_registered_memory_replay", lambda **kwargs: captured.append(kwargs) or {})
    args = []
    for name, value in {"experiment-id": "EXP-A", "parent-experiment-id": "EXP-B", "index-experiment-id": "EXP-INDEX",
        "output-root": "/fixture/output", "index-records-root": "/fixture/records", "size-index": "/fixture/index",
        "bindings-initial": "/fixture/initial", "bindings-final": "/fixture/final", "extreme-cases": "/fixture/cases",
        "child-timeout-seconds": "30", "rss-sample-interval-seconds": "0.02",
        **{field.name.replace("_", "-"): "/fixture/" + field.name for field in fields(mr.ProductionArtifactPaths)}}.items():
        args += ["--" + name, value]
    assert module.main(args) == 0
    assert captured[0]["device"] == "cuda" and "synthetic" not in captured[0]
    for forbidden in ("--synthetic", "--checkpoint", "--device"):
        with pytest.raises(SystemExit):
            module.main(args + [forbidden, "cpu"])
