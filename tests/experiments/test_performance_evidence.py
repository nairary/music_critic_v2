from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import subprocess

import pytest

from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
from music_critic.experiments.registry import (
    load_record, seal_record, validate_registry, validate_record, write_record_once,
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init",), ("config", "user.name", "Fixture"),
                 ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "pyproject.toml").write_text("# fixture\n")
    subprocess.run(["git", "-C", str(root), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    return root


def _run(root, execute, name="EXP-PERF-TEST"):
    return run_registered_replay(
        experiment_id=name, output_root=root / "outputs" / name,
        protocol={"mode": "fixture", "measured_draws": 2, "repeats": 1},
        execute=execute, repository_root=root,
    )


def test_planned_running_record_exists_before_work_and_completed_is_immutable(repo):
    path = repo / "docs/experiments/records/EXP-PERF-TEST.json"

    def execute():
        assert load_record(path)["status"] == "running"
        return {"equivalence_passed": True}

    assert _run(repo, execute)["equivalence_passed"]
    record = load_record(path)
    assert record["status"] == "completed"
    assert len(record["artifacts"]) == 1
    old = path.read_bytes()
    with pytest.raises(ValueError, match="fresh_experiment"):
        _run(repo, execute)
    assert path.read_bytes() == old
    assert len(validate_registry(path.parent)) == 1
    assert (repo / "docs/EXPERIMENT_LEDGER.md").is_file()


@pytest.mark.parametrize("error,status", [(RuntimeError("fixture failed"), "failed"),
                                          (KeyboardInterrupt(), "aborted")])
def test_failures_and_interrupts_are_sealed_not_lost(repo, error, status):
    def execute():
        raise error
    with pytest.raises(type(error)):
        _run(repo, execute)
    record = load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")
    assert record["status"] == status
    assert record["validity"]["reason"]
    assert record["artifacts"]
    assert record["data_access"]["test_targets_accessed"] is None


@pytest.mark.parametrize("change_source", [False, True])
def test_missing_equivalence_or_changed_source_invalidates_run(repo, change_source):
    def execute():
        if change_source:
            (repo / "pyproject.toml").write_text("# modified during run\n")
        return {"equivalence_passed": change_source}
    with pytest.raises(ValueError, match="experiment_invalid"):
        _run(repo, execute)
    assert load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")["status"] == "invalid"


def test_tracked_output_is_rejected_before_any_execution(repo):
    with pytest.raises(ValueError, match="output_must_be_ignored"):
        run_registered_replay(experiment_id="EXP-PERF-NO", output_root=repo / "bad",
                              protocol={}, execute=lambda: pytest.fail("executed"),
                              repository_root=repo)
    assert not (repo / "docs/experiments/records").exists()


@pytest.mark.parametrize("problem", ["reserved_file", "symlink", "output_is_file"])
def test_evidence_failure_still_finalizes_record(repo, problem):
    output = repo / "outputs/EXP-PERF-TEST"
    def execute():
        output.parent.mkdir(parents=True, exist_ok=True)
        if problem == "output_is_file":
            output.write_text("occupied")
        else:
            output.mkdir()
            if problem == "reserved_file":
                (output / "execution-evidence.json").write_text("do not overwrite")
            else:
                (output / "bad-link").symlink_to(repo / "pyproject.toml")
        return {"equivalence_passed": True}
    with pytest.raises((OSError, ValueError)):
        _run(repo, execute)
    record = load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")
    assert record["status"] == "failed"
    assert "evidence_finalization_failed" in record["validity"]["reason"]
    if problem == "reserved_file":
        assert (output / "execution-evidence.json").read_text() == "do not overwrite"


@pytest.mark.parametrize("payload", [
    [],
    {"equivalence_passed": True, "experiment_id": "DIFFERENT"},
    {"equivalence_passed": True, "arms": [None]},
    {"equivalence_passed": True, "bad": float("nan")},
    {"equivalence_passed": True, "bad": object()},
    {"equivalence_passed": True, "data_access": {
        "validation_accessed": True, "test_inputs_accessed": False,
        "test_targets_accessed": False, "test_metrics_computed": False,
        "test_used_for_selection": False}},
])
def test_bad_results_finalize_failed_instead_of_remaining_running(repo, payload):
    with pytest.raises((ValueError, TypeError)):
        _run(repo, lambda: payload)
    assert load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")["status"] == "failed"


@pytest.mark.parametrize("field", ["reference_data_strategy", "optimized_data_strategy"])
@pytest.mark.parametrize("problem", ["missing", "mismatch"])
def test_declared_data_strategy_requires_matching_result_and_seals_failure(repo, field, problem):
    strategies = {"reference_data_strategy": "frozen", "optimized_data_strategy": "frozen"}
    result = {"equivalence_passed": True, **strategies}
    if problem == "missing":
        del result[field]
    else:
        result[field] = "prepared_cache"
    output = repo / "outputs/EXP-STRATEGY"
    category = f"result_binding_{problem}:{field}"
    with pytest.raises(ValueError, match=category):
        run_registered_replay(
            experiment_id="EXP-STRATEGY", output_root=output,
            protocol={"mode": "fixture", **strategies}, execute=lambda: result,
            repository_root=repo,
        )
    record = load_record(repo / "docs/experiments/records/EXP-STRATEGY.json")
    assert record["status"] == "failed" and record["validity"]["valid"] is False
    assert category in record["validity"]["reason"]
    assert record["protocol"][field] == "frozen"
    assert record["artifacts"]
    evidence = json.loads((output / "execution-evidence.json").read_text())
    assert [row["status"] for row in evidence["lifecycle"]] == ["planned", "running"]
    assert all(row["protocol"][field] == "frozen" for row in evidence["lifecycle"])
    assert category in evidence["failure"]


@pytest.mark.parametrize("optimized_strategy", ["prepared_cache", "frozen"])
def test_matching_declared_data_strategies_are_bound_in_completed_record(repo, optimized_strategy):
    strategies = {"reference_data_strategy": "frozen", "optimized_data_strategy": optimized_strategy}
    result = {"equivalence_passed": True, **strategies}
    returned = run_registered_replay(
        experiment_id="EXP-STRATEGY", output_root=repo / "outputs/EXP-STRATEGY",
        protocol={"mode": "fixture", **strategies}, execute=lambda: result,
        repository_root=repo,
    )
    record = load_record(repo / "docs/experiments/records/EXP-STRATEGY.json")
    assert record["status"] == "completed"
    assert record["results"] == returned == result
    for key, value in strategies.items():
        assert record["protocol"][key] == record["results"][key] == value


@pytest.mark.parametrize("reported_strategies", [{}, {
    "reference_data_strategy": "frozen", "optimized_data_strategy": "prepared_cache",
}])
def test_historical_protocol_without_data_strategies_remains_supported(repo, reported_strategies):
    result = {"equivalence_passed": True, **reported_strategies}
    assert _run(repo, lambda: result) == result
    record = load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")
    assert record["status"] == "completed"
    assert "reference_data_strategy" not in record["protocol"]
    assert "optimized_data_strategy" not in record["protocol"]


def test_bound_protocol_is_detached_from_callback_mutation(repo):
    protocol = {"mode": "fixture", "measured_draws": 2, "repeats": 1}
    def execute():
        protocol["measured_draws"] = 999
        return {"equivalence_passed": True}
    run_registered_replay(experiment_id="EXP-PERF-TEST", output_root=repo / "outputs/EXP-PERF-TEST",
                          protocol=protocol, execute=execute, repository_root=repo)
    record = load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")
    assert record["protocol"]["measured_draws"] == 2


def test_interrupt_during_artifact_hashing_is_terminal(repo, monkeypatch):
    from music_critic.experiments.analysisgnn import performance_evidence
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(performance_evidence, "_artifact_rows", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(repo, lambda: {"equivalence_passed": True})
    assert load_record(repo / "docs/experiments/records/EXP-PERF-TEST.json")["status"] == "aborted"


def test_lifecycle_snapshots_remain_sealed_after_actual_budget_is_known(repo):
    import json
    _run(repo, lambda: {"equivalence_passed": True, "arms": [{"window": {"records": 2}}]})
    evidence = json.loads((repo / "outputs/EXP-PERF-TEST/execution-evidence.json").read_text())
    for row in evidence["lifecycle"]:
        validate_record(row)
        assert row["budget"]["actual"] is None


def _write_parent(repo, status):
    _run(repo, lambda: {"equivalence_passed": True}, name="EXP-PARENT-TEMPLATE")
    records = repo / "docs/experiments/records"
    parent = load_record(records / "EXP-PARENT-TEMPLATE.json")
    parent.update(experiment_id="EXP-PARENT", status=status)
    if status != "completed":
        parent.update(results={}, validity={"valid": False, "reason": "synthetic_parent"})
    path = records / "EXP-PARENT.json"
    write_record_once(path, seal_record(parent))
    return path


@pytest.mark.parametrize("status", ["completed", "failed", "aborted", "invalid"])
def test_terminal_parent_link_is_present_before_execution_and_in_all_evidence(repo, status):
    parent_path = _write_parent(repo, status)
    parent_bytes = parent_path.read_bytes()
    record_path = repo / "docs/experiments/records/EXP-CHILD.json"
    output = repo / "outputs/EXP-CHILD"

    def execute():
        running = load_record(record_path)
        assert running["status"] == "running"
        assert running["parent_ids"] == ["EXP-PARENT"]
        return {"equivalence_passed": True}

    run_registered_replay(
        experiment_id="EXP-CHILD", output_root=output,
        protocol={"mode": "fixture"}, execute=execute, repository_root=repo,
        parent_experiment_id="EXP-PARENT",
    )
    terminal = load_record(record_path)
    assert terminal["status"] == "completed" and terminal["parent_ids"] == ["EXP-PARENT"]
    evidence = json.loads((output / "execution-evidence.json").read_text())
    assert evidence["parent_ids"] == ["EXP-PARENT"]
    assert [row["status"] for row in evidence["lifecycle"]] == ["planned", "running"]
    for row in evidence["lifecycle"]:
        validate_record(row)
        assert row["parent_ids"] == ["EXP-PARENT"]
    assert parent_path.read_bytes() == parent_bytes


@pytest.mark.parametrize("parent_id,category", [
    ("EXP-MISSING", "parent_record_missing"),
    ("EXP-CHILD", "parent_experiment_self_reference"),
    ("", "parent_experiment_id_invalid"),
    ("../EXP-PARENT", "parent_experiment_id_invalid"),
    ("/tmp/EXP-PARENT", "parent_experiment_id_invalid"),
    ("EXP\\PARENT", "parent_experiment_id_invalid"),
    ("EXP PARENT", "parent_experiment_id_invalid"),
    ("P" * 129, "parent_experiment_id_invalid"),
])
def test_missing_or_invalid_parent_is_rejected_before_planning_and_execution(repo, parent_id, category):
    with pytest.raises(ValueError, match=category):
        run_registered_replay(
            experiment_id="EXP-CHILD", output_root=repo / "outputs/EXP-CHILD",
            protocol={}, execute=lambda: pytest.fail("executed with invalid parent"),
            repository_root=repo, parent_experiment_id=parent_id,
        )
    assert not (repo / "docs/experiments/records").exists()
    assert not (repo / "outputs/EXP-CHILD").exists()


@pytest.mark.parametrize("status", ["planned", "running"])
def test_nonterminal_parent_is_rejected_before_child_registration(repo, status):
    parent_path = _write_parent(repo, status)
    before = parent_path.read_bytes()
    with pytest.raises(ValueError, match="parent_record_not_terminal"):
        run_registered_replay(
            experiment_id="EXP-CHILD", output_root=repo / "outputs/EXP-CHILD",
            protocol={}, execute=lambda: pytest.fail("executed before parent finished"),
            repository_root=repo, parent_experiment_id="EXP-PARENT",
        )
    assert parent_path.read_bytes() == before
    assert not (parent_path.parent / "EXP-CHILD.json").exists()
    assert not (repo / "outputs/EXP-CHILD").exists()


@pytest.mark.parametrize("problem", ["fingerprint", "noncanonical", "symlink"])
def test_parent_requires_a_valid_canonical_sealed_record(repo, problem):
    path = _write_parent(repo, "failed")
    if problem == "fingerprint":
        path.write_bytes(path.read_bytes().replace(b'synthetic_parent', b'synthetic_changed'))
    elif problem == "noncanonical":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path.unlink()
        path.symlink_to(path.parent / "EXP-PARENT-TEMPLATE.json")
    with pytest.raises(ValueError, match="parent_record_invalid"):
        run_registered_replay(
            experiment_id="EXP-CHILD", output_root=repo / "outputs/EXP-CHILD",
            protocol={}, execute=lambda: pytest.fail("executed with unsealed parent"),
            repository_root=repo, parent_experiment_id="EXP-PARENT",
        )
    assert not (path.parent / "EXP-CHILD.json").exists()
    assert not (repo / "outputs/EXP-CHILD").exists()


@pytest.mark.parametrize("parent_id", [None, "EXP-PARENT"])
def test_cli_forwards_optional_parent_without_running_replay(repo, monkeypatch, parent_id):
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_performance_replay.py"
    spec = importlib.util.spec_from_file_location("performance_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def registered(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "run_registered_replay", registered)
    argv = ["--experiment-id", "EXP-CHILD", "--output-root", str(repo / "outputs/EXP-CHILD"),
            "--mode", "fixture"]
    if parent_id is not None:
        argv.extend(("--parent-experiment-id", parent_id))
    assert module.main(argv) == 0
    assert captured["parent_experiment_id"] == parent_id
    assert not (repo / "outputs/EXP-CHILD").exists()


def _prefetch_protocol(counts=(0, 1, 2, 4)):
    return {
        "mode": "fixture", "comparison_kind": "ordered_prefetch", "batch_sizes": [2],
        "measured_draws": 4, "warmup_draws": 2, "repeats": 2,
        "planned_arm_count": len(counts), "worker_counts": list(counts),
        "worker_configs_fingerprint": "a" * 64,
        "schedule_descriptor_fingerprint": "b" * 64,
        "reservation_manifest_fingerprint": "c" * 64,
        "reference_data_strategy": "frozen", "optimized_data_strategy": "frozen",
        "update_strategy": "optimized",
        "question": "Can ordered CPU preparation overlap the fixed parent update?",
        "hypothesis": "CPU workers reduce waits with exact B2 state parity.",
        "decision_use": "Compare CPU overlap only, with no musical-quality selection.",
    }


def _prefetch_result(protocol):
    arms = [{"repeat": repeat, "worker_count": count, "batch_size": 2,
             "kind": "baseline" if count == 0 else "optimized",
             "window": {"records": protocol["measured_draws"]}}
            for repeat in range(protocol["repeats"]) for count in protocol["worker_counts"]]
    return {**deepcopy(protocol), "equivalence_passed": True, "arms": list(reversed(arms))}


def _run_prefetch(repo, protocol, result):
    return run_registered_replay(
        experiment_id="EXP-PREFETCH", output_root=repo / "outputs/EXP-PREFETCH",
        protocol=protocol, execute=lambda: result, repository_root=repo,
    )


@pytest.mark.parametrize("counts", [(0, 2), (4, 0, 1, 2)])
def test_prefetch_declared_arms_budget_narrative_and_all_lifecycle_bindings(repo, counts):
    protocol = _prefetch_protocol(counts)
    result = _prefetch_result(protocol)
    expected_budget = 4 * len(counts) * 2
    record_path = repo / "docs/experiments/records/EXP-PREFETCH.json"

    def execute():
        running = load_record(record_path)
        assert running["status"] == "running"
        assert running["budget"]["planned"] == expected_budget
        assert running["budget"]["actual"] is None
        return result

    assert run_registered_replay(
        experiment_id="EXP-PREFETCH", output_root=repo / "outputs/EXP-PREFETCH",
        protocol=protocol, execute=execute, repository_root=repo,
    ) == result
    completed = load_record(record_path)
    assert completed["status"] == "completed"
    assert completed["budget"]["planned"] == completed["budget"]["actual"] == expected_budget
    for field in ("question", "hypothesis", "decision_use"):
        assert completed[field] == protocol[field]
    evidence = json.loads((repo / "outputs/EXP-PREFETCH/execution-evidence.json").read_text())
    assert [row["status"] for row in evidence["lifecycle"]] == ["planned", "running"]
    for row in evidence["lifecycle"]:
        validate_record(row)
        assert row["protocol"] == protocol
        assert row["budget"]["planned"] == expected_budget and row["budget"]["actual"] is None


@pytest.mark.parametrize("field", [
    "comparison_kind", "planned_arm_count", "worker_counts", "worker_configs_fingerprint",
    "schedule_descriptor_fingerprint", "reservation_manifest_fingerprint", "update_strategy",
])
@pytest.mark.parametrize("problem", ["missing", "mismatch"])
def test_prefetch_declared_bindings_missing_or_mismatched_result_seals_failure(repo, field, problem):
    protocol = _prefetch_protocol()
    result = _prefetch_result(protocol)
    if problem == "missing":
        del result[field]
    else:
        result[field] = "wrong"
    category = f"result_binding_{problem}:{field}"
    with pytest.raises(ValueError, match=category):
        _run_prefetch(repo, protocol, result)
    record = load_record(repo / "docs/experiments/records/EXP-PREFETCH.json")
    assert record["status"] == "failed" and record["validity"]["valid"] is False
    assert category in record["validity"]["reason"] and record["artifacts"]
    evidence = json.loads((repo / "outputs/EXP-PREFETCH/execution-evidence.json").read_text())
    assert category in evidence["failure"]
    assert [row["status"] for row in evidence["lifecycle"]] == ["planned", "running"]


@pytest.mark.parametrize("updates", [
    {"planned_arm_count": 0}, {"planned_arm_count": True}, {"planned_arm_count": 4.0},
    {"planned_arm_count": 3}, {"repeats": 0}, {"repeats": True},
    {"worker_counts": [1, 2, 4]}, {"worker_counts": [0, 2, 2, 4]},
    {"worker_counts": [False, 1, 2, 4]}, {"worker_counts": [0, 1, 2, 3]},
    {"worker_counts": []}, {"worker_counts": "0,1,2,4"},
    {"measured_draws": 3}, {"measured_draws": 0}, {"warmup_draws": -2},
    {"warmup_draws": True}, {"batch_size": 4}, {"batch_sizes": [2, 4]},
    {"batch_sizes": [2.0]}, {"worker_configs_fingerprint": "unknown"},
    {"schedule_descriptor_fingerprint": 123}, {"reservation_manifest_fingerprint": "G" * 64},
    {"question": " "}, {"hypothesis": None}, {"decision_use": 123},
])
def test_invalid_prefetch_protocol_fails_before_source_io_planning_and_execution(repo, monkeypatch, updates):
    from music_critic.experiments.analysisgnn import performance_evidence
    def forbidden(*args):
        pytest.fail("invalid protocol reached source capture or execution")
    monkeypatch.setattr(performance_evidence, "source_manifest", forbidden)
    protocol = {**_prefetch_protocol(), **updates}
    with pytest.raises(ValueError, match="performance.protocol_"):
        run_registered_replay(experiment_id="EXP-PREFETCH", output_root=repo / "outputs/EXP-PREFETCH",
                              protocol=protocol, execute=forbidden, repository_root=repo)
    assert not (repo / "docs/experiments/records").exists()
    assert not (repo / "outputs/EXP-PREFETCH").exists()


@pytest.mark.parametrize("field", [
    "planned_arm_count", "worker_counts", "worker_configs_fingerprint",
    "schedule_descriptor_fingerprint", "reservation_manifest_fingerprint", "repeats",
    "measured_draws", "warmup_draws",
])
def test_incomplete_prefetch_protocol_fails_before_planning(repo, field):
    protocol = _prefetch_protocol()
    del protocol[field]
    with pytest.raises(ValueError, match="performance.protocol_"):
        _run_prefetch(repo, protocol, {})
    assert not (repo / "docs/experiments/records").exists()


@pytest.mark.parametrize("problem", [
    "missing_arms", "too_few", "too_many", "duplicate", "missing_reference",
    "wrong_repeat", "bool_repeat", "bool_worker", "wrong_batch", "float_batch",
    "wrong_kind", "wrong_draw_count", "bool_draw_count", "no_window",
    "bool_worker_binding", "float_arm_count_binding",
])
def test_prefetch_requires_complete_exact_worker_arms_and_b2_accounting(repo, problem):
    protocol = _prefetch_protocol()
    result = _prefetch_result(protocol)
    arm = result["arms"][0]
    if problem == "missing_arms":
        del result["arms"]
    elif problem == "too_few":
        result["arms"].pop()
    elif problem == "too_many":
        result["arms"].append(deepcopy(arm))
    elif problem == "duplicate":
        result["arms"][0] = deepcopy(result["arms"][1])
    elif problem == "missing_reference":
        next(row for row in result["arms"] if row["worker_count"] == 0)["worker_count"] = 4
    elif problem == "wrong_repeat":
        arm["repeat"] = 2
    elif problem == "bool_repeat":
        arm["repeat"] = True
    elif problem == "bool_worker":
        arm["worker_count"] = False
    elif problem == "wrong_batch":
        arm["batch_size"] = 4
    elif problem == "float_batch":
        arm["batch_size"] = 2.0
    elif problem == "wrong_kind":
        arm["kind"] = "baseline"
    elif problem == "wrong_draw_count":
        arm["window"]["records"] = 2
    elif problem == "bool_draw_count":
        arm["window"]["records"] = True
    elif problem == "no_window":
        del arm["window"]
    elif problem == "bool_worker_binding":
        result["worker_counts"][0] = False  # Python equality would treat False == 0.
    elif problem == "float_arm_count_binding":
        result["planned_arm_count"] = 4.0
    with pytest.raises(ValueError, match="performance.result_"):
        _run_prefetch(repo, protocol, result)
    record = load_record(repo / "docs/experiments/records/EXP-PREFETCH.json")
    assert record["status"] == "failed" and record["validity"]["valid"] is False
    assert record["artifacts"]


@pytest.mark.parametrize("equivalence", [False, None, 1])
def test_prefetch_arm_coverage_does_not_relax_exact_equivalence(repo, equivalence):
    protocol = _prefetch_protocol()
    result = _prefetch_result(protocol)
    result["equivalence_passed"] = equivalence
    with pytest.raises(ValueError, match="experiment_invalid:exact_b2_equivalence_not_proven"):
        _run_prefetch(repo, protocol, result)
    record = load_record(repo / "docs/experiments/records/EXP-PREFETCH.json")
    assert record["status"] == "invalid" and record["budget"]["actual"] == 32


def test_legacy_budget_and_narrative_defaults_remain_unchanged(repo):
    protocol = {"mode": "fixture", "measured_draws": 2, "batch_sizes": [2, 4], "repeats": 3}
    run_registered_replay(experiment_id="EXP-LEGACY", output_root=repo / "outputs/EXP-LEGACY",
                          protocol=protocol, execute=lambda: {"equivalence_passed": True}, repository_root=repo)
    record = load_record(repo / "docs/experiments/records/EXP-LEGACY.json")
    assert record["budget"]["planned"] == 18 and record["budget"]["actual"] is None
    assert record["question"] == "Can preparation and synchronization overhead be reduced with exact B2 training parity?"
    assert record["hypothesis"] == "Sealed graph reuse and CPU candidate metadata improve throughput without changing B2 states."
    assert record["decision_use"] == "Engineering throughput and equivalence only; no model-quality selection."


@pytest.mark.parametrize("actual_arms", [2, 3, 4])
def test_optional_arm_count_overrides_legacy_budget_without_prefetch_schema(repo, actual_arms):
    protocol = {"mode": "fixture", "planned_arm_count": 3, "measured_draws": 2,
                "batch_sizes": [2], "repeats": 1}
    result = {"equivalence_passed": True, "planned_arm_count": 3,
              "arms": [{"window": {"records": 2}} for _ in range(actual_arms)]}
    if actual_arms == 3:
        _run_prefetch(repo, protocol, result)
    else:
        with pytest.raises(ValueError, match="result_arm_count_mismatch"):
            _run_prefetch(repo, protocol, result)
    record = load_record(repo / "docs/experiments/records/EXP-PREFETCH.json")
    assert record["budget"]["planned"] == 6
    assert record["status"] == ("completed" if actual_arms == 3 else "failed")
    assert record["budget"]["actual"] == (6 if actual_arms == 3 else None)
