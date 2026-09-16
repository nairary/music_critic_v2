from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from music_critic.experiments.analysisgnn import performance_prefetch_replay as replay
from music_critic.experiments.analysisgnn import performance_replay as existing
from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
from music_critic.experiments.registry import load_record


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_prefetch_fixture_gate.py"


def test_import_as_spawn_main_has_no_heavy_import_or_execution(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "numpy", "music_critic")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("__mp_main__", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location("prefetch_fixture_driver", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init",), ("config", "user.name", "Fixture"),
                 ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "pyproject.toml").write_text("# synthetic fixture\n")
    (root / "scripts").mkdir()
    (root / "scripts" / SCRIPT.name).write_bytes(SCRIPT.read_bytes())
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    run_registered_replay(experiment_id="EXP-PARENT", output_root=root / "outputs/parent",
                          protocol={"mode": "fixture"}, execute=lambda: {"equivalence_passed": True},
                          repository_root=root)
    return root


def _record(repo):
    return load_record(repo / "docs/experiments/records/EXP-FIXTURE.json")


def _resume(device="cpu", passed=True):
    results = []
    for count in (0, 1, 2, 4):
        branch = {"state": "a" * 64, "rng": "b" * 64, "trace": "c" * 64,
                  "logits": "d" * 64, "committed": 3, "pending_at_resume": None}
        results.append({"worker_count": count, "resume_exact": passed,
                        "uninterrupted": branch,
                        "resumed": {**branch, "pending_at_resume": 2 if count else 0}})
    return {"scope": "synthetic_full_model_checkpoint_gate_no_throughput_claim", "passed": passed,
            "device": device, "worker_counts": [0, 1, 2, 4], "results": results}


def _install_gates(monkeypatch, repo, *, throughput_passed=True, resume_result=None,
                   resume_error=None, device="cpu", events=None):
    events = [] if events is None else events

    def throughput(output, **kwargs):
        assert _record(repo)["status"] == "running"
        events.append("throughput")
        assert kwargs["worker_counts"] == (0, 1, 2, 4)
        assert kwargs["mode"] == "fixture" and kwargs["device"] == device
        assert (kwargs["measured_draws"], kwargs["warmup_draws"], kwargs["repeats"]) == (6, 2, 1)
        output.mkdir(parents=True)
        bindings = replay.protocol_bindings(mode="fixture", descriptor=kwargs["descriptor"],
            reservation_manifest=None, config=kwargs["config"], worker_counts=kwargs["worker_counts"])
        return {**bindings, "completed": True, "equivalence_passed": throughput_passed,
            "experiment_id": kwargs["experiment_id"], "mode": "fixture", "device": device,
            "repeats": 1, "measured_draws_per_arm": 6, "warmup_draws_per_arm": 2,
            "update_strategy": "optimized", "actual_attempted_updates": 16,
            "actual_applied_updates": 16, "actual_measured_draws": 24,
            "arms": [{"repeat": 0, "worker_count": count, "batch_size": 2,
                      "kind": "baseline" if count == 0 else "optimized", "window": {"records": 6}}
                     for count in kwargs["worker_counts"]]}

    def resume(output, **kwargs):
        assert _record(repo)["status"] == "running"
        assert kwargs == {"worker_counts": (0, 1, 2, 4), "measured_draws": 6,
                          "resume_after_updates": 1, "device": device}
        assert (output.parent / "throughput-summary.json").is_file()
        pending = json.loads((output.parent / "summary.json").read_text())
        assert pending["completed"] is False and pending["equivalence_passed"] is False
        assert pending["resume_status"] == "running"
        events.append("resume")
        output.mkdir()
        (output / "worker-1.pt").write_bytes(b"synthetic own checkpoint")
        if resume_error is not None:
            raise resume_error
        return deepcopy(resume_result) if resume_result is not None else _resume(device)

    monkeypatch.setattr(replay, "run_prefetch_replay", throughput)
    monkeypatch.setattr(replay, "run_fixture_resume_gate", resume)
    if device == "cpu":
        import torch
        monkeypatch.setattr(existing, "_assert_idle_gpu", lambda: pytest.fail("CPU gate touched GPU"))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU gate queried CUDA"))
    return events


def _run(driver, repo, **kwargs):
    return driver.run_registered_fixture_gate(experiment_id="EXP-FIXTURE", parent_experiment_id="EXP-PARENT",
        output_root=repo / "outputs/gate", device=kwargs.pop("device", "cpu"), repository_root=repo, **kwargs)


def test_combined_gate_is_registered_before_work_and_budgets_resume_separately(driver, repo, monkeypatch):
    events = _install_gates(monkeypatch, repo)
    result = _run(driver, repo)
    assert events == ["throughput", "resume"]
    record = _record(repo)
    assert record["status"] == "completed" and record["parent_ids"] == ["EXP-PARENT"]
    assert record["budget"]["planned"] == record["budget"]["actual"] == 24
    assert record["budget"]["unit"] == "measured_record_draws_across_all_arms"
    assert result["actual_attempted_updates"] == result["actual_applied_updates"] == 40
    assert result["actual_measured_draws"] == 24
    assert result["actual_training_draws_including_warmup_and_resume"] == 80
    assert result["supplemental_resume_budget"] == {
        "planned_updates": 24, "planned_record_draws": 48, "actual_updates": 24, "actual_record_draws": 48,
        "scope": "supplemental_synthetic_resume_excludes_throughput_windows"}
    assert result["resume_equivalence_checked"] is True and result["equivalence_passed"] is True
    assert result["canonical_checkpoint_input_used"] is False and result["own_fixture_checkpoint_input_used"] is True
    assert record["results"] == result
    artifacts = {row["role"]: row for row in record["artifacts"]}
    checkpoint = artifacts["own-checkpoints/worker-1.pt"]
    assert checkpoint["sha256"] == sha256(b"synthetic own checkpoint").hexdigest()
    assert "summary.json" in artifacts and "throughput-summary.json" in artifacts
    evidence = json.loads((repo / "outputs/gate/execution-evidence.json").read_text())
    assert [row["status"] for row in evidence["lifecycle"]] == ["planned", "running"]
    assert f"scripts/{SCRIPT.name}" in evidence["source_manifest"]["files"]
    assert all(row["protocol"]["supplemental_resume_budget"]["planned_updates"] == 24
               for row in evidence["lifecycle"])


@pytest.mark.parametrize("error,status", [(RuntimeError("partial resume failed"), "failed"),
                                         (KeyboardInterrupt(), "aborted")])
def test_partial_resume_failure_keeps_checkpoint_and_terminal_evidence(driver, repo, monkeypatch, error, status):
    _install_gates(monkeypatch, repo, resume_error=error)
    with pytest.raises(type(error)):
        _run(driver, repo)
    record = _record(repo)
    assert record["status"] == status
    assert "own-checkpoints/worker-1.pt" in {row["role"] for row in record["artifacts"]}
    summary = json.loads((repo / "outputs/gate/summary.json").read_text())
    assert summary["completed"] is False and summary["equivalence_passed"] is False
    assert summary["resume_status"] == "failed"
    assert summary["supplemental_resume_budget"]["actual_updates"] is None
    assert summary["supplemental_resume_budget"]["actual_record_draws"] is None
    assert summary["actual_attempted_updates"] is None and summary["actual_applied_updates"] is None
    assert summary["actual_training_draws_including_warmup_and_resume"] is None
    assert summary["completed_throughput_applied_updates"] == 16
    assert json.loads((repo / "outputs/gate/throughput-summary.json").read_text())["equivalence_passed"] is True


def test_failed_resume_equivalence_cannot_complete_registered_gate(driver, repo, monkeypatch):
    _install_gates(monkeypatch, repo, resume_result=_resume(passed=False))
    with pytest.raises(ValueError, match="experiment_invalid"):
        _run(driver, repo)
    record = _record(repo)
    assert record["status"] == "invalid"
    assert record["results"]["supplemental_resume_budget"]["actual_updates"] == 24
    assert record["results"]["resume_result"]["passed"] is False


def test_resume_not_run_after_failed_throughput_equivalence(driver, repo, monkeypatch):
    events = _install_gates(monkeypatch, repo, throughput_passed=False)
    with pytest.raises(ValueError, match="experiment_invalid"):
        _run(driver, repo)
    assert events == ["throughput"]
    record = _record(repo)
    assert record["status"] == "invalid" and record["results"]["resume_equivalence_checked"] is False
    assert record["results"]["supplemental_resume_budget"]["actual_updates"] == 0
    assert not (repo / "outputs/gate/own-checkpoints").exists()


@pytest.mark.parametrize("problem", ["missing_arm", "duplicate_arm", "wrong_count", "no_pending", "non_boolean"])
def test_resume_accounting_rejects_incomplete_gate(driver, repo, monkeypatch, problem):
    result = _resume()
    if problem == "missing_arm":
        result["results"].pop()
    elif problem == "duplicate_arm":
        result["results"][1]["worker_count"] = 0
    elif problem == "wrong_count":
        result["results"][0]["resumed"]["committed"] = 2
    elif problem == "no_pending":
        result["results"][1]["resumed"]["pending_at_resume"] = 0
    else:
        result["passed"] = 1
    _install_gates(monkeypatch, repo, resume_result=result)
    with pytest.raises(ValueError, match="prefetch_fixture_gate.resume_"):
        _run(driver, repo)
    assert _record(repo)["status"] == "failed"


def test_cuda_guards_wrap_both_gates_after_planning(driver, repo, monkeypatch):
    import torch
    for name in driver.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    events = []
    def idle():
        assert _record(repo)["status"] == "running"
        events.append("idle")
    monkeypatch.setattr(existing, "_assert_idle_gpu", idle)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _install_gates(monkeypatch, repo, device="cuda", events=events)
    result = _run(driver, repo, device="cuda")
    assert events == ["idle", "throughput", "idle", "resume"]
    assert result["resume_result"]["device"] == "cuda"
    assert result["fixture_gate_environment"]["inherited_thread_environment"] == {
        name: "1" for name in driver.THREAD_ENVIRONMENT}


@pytest.mark.parametrize("failure", ["busy", "unavailable"])
def test_cuda_not_idle_or_unavailable_seals_failure_before_model_work(driver, repo, monkeypatch, failure):
    import torch
    for name in driver.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    def idle():
        assert _record(repo)["status"] == "running"
        if failure == "busy":
            raise RuntimeError("GPU busy")
    monkeypatch.setattr(existing, "_assert_idle_gpu", idle)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(replay, "run_prefetch_replay", lambda *a, **kw: pytest.fail("model work started"))
    with pytest.raises(RuntimeError, match="busy|cuda_unavailable"):
        _run(driver, repo, device="cuda")
    assert _record(repo)["status"] == "failed"


@pytest.mark.parametrize("variable", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"])
def test_cuda_cli_rejects_missing_thread_limits_before_torch_import(driver, monkeypatch, variable):
    import builtins
    for name in driver.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv(variable)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "torch", "Torch imported before CUDA environment validation"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as exc:
        driver.main(["--experiment-id", "EXP-GATE", "--parent-experiment-id", "EXP-PARENT",
                     "--output-root", "/ignored/unused"])
    assert exc.value.code == 2 and variable not in __import__("os").environ


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cli_fixed_fixture_interface_and_thread_configuration(driver, monkeypatch, capsys, device):
    for name in driver.THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    events = []
    monkeypatch.setattr(driver, "_configure_parent_threads", lambda torch: events.append("configure"))
    def gate(**kwargs):
        assert events == ["configure"]
        events.append("gate")
        assert kwargs == {"experiment_id": "EXP-GATE", "parent_experiment_id": "EXP-PARENT",
                          "output_root": Path("/ignored/unused"), "device": device}
        return {"equivalence_passed": True}
    monkeypatch.setattr(driver, "run_registered_fixture_gate", gate)
    args = ["--experiment-id", "EXP-GATE", "--parent-experiment-id", "EXP-PARENT",
            "--output-root", "/ignored/unused"]
    if device == "cpu":
        args += ["--device", "cpu"]
    assert driver.main(args) == 0
    assert events == ["configure", "gate"]
    assert json.loads(capsys.readouterr().out)["equivalence_passed"] is True
    with pytest.raises(SystemExit):
        driver.main(args + ["--split-manifest", "/forbidden/production"])


@pytest.mark.parametrize("interop", [1, 4])
def test_parent_interop_configuration_only_changes_when_required(driver, interop):
    calls = []
    class Torch:
        set_num_threads = staticmethod(lambda value: calls.append(("intra", value)))
        get_num_interop_threads = staticmethod(lambda: interop)
        set_num_interop_threads = staticmethod(lambda value: calls.append(("inter", value)))
    driver._configure_parent_threads(Torch)
    assert calls == [("intra", 1)] + ([("inter", 1)] if interop != 1 else [])


def test_parent_and_fresh_identity_required_before_any_gate_work(driver, repo, monkeypatch):
    monkeypatch.setattr(replay, "run_prefetch_replay", lambda *a, **kw: pytest.fail("work started"))
    for parent in (None, "EXP-NO-PARENT"):
        with pytest.raises(ValueError, match="terminal_parent_required|parent_record_missing"):
            driver.run_registered_fixture_gate(experiment_id="EXP-FIXTURE", parent_experiment_id=parent,
                output_root=repo / "outputs/gate", device="cpu", repository_root=repo)
    assert not (repo / "docs/experiments/records/EXP-FIXTURE.json").exists()
    with pytest.raises(ValueError, match="fresh_experiment"):
        driver.run_registered_fixture_gate(experiment_id="EXP-FIXTURE", parent_experiment_id="EXP-PARENT",
            output_root=repo / "outputs/parent", device="cpu", repository_root=repo)
