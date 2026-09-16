"""Synthetic sequential reservation gate; no pool, GPU or corpus execution."""

from dataclasses import asdict, fields, replace
import importlib.util
import json
from pathlib import Path
import random
import pickle
import subprocess
import sys

import pytest

from music_critic.experiments.analysisgnn import performance_prefetch as pp
from music_critic.experiments.analysisgnn.corrected_training import (
    ProductionArtifactPaths, build_source_free_fixture,
)
from music_critic.experiments.analysisgnn.performance_prefetch_replay import (
    build_prefetch_schedule, jobs_from_prefetch_descriptor,
)
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, file_sha256, load_record, render_registry,
    seal_record, validate_record, write_record_once,
)


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_prefetch_reservations.py"
    name = "test_prefetch_reservation_cli"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for args in (("init",), ("config", "user.name", "Fixture"),
                 ("config", "user.email", "fixture@example.invalid")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "pyproject.toml").write_text("# isolated reservation fixture\n")
    subprocess.run(["git", "-C", str(root), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    record = {"schema_version": "1.0.0", "experiment_id": "EXP-PARENT", "kind": "data_audit",
        "phase": "9E-P1", "title": "Synthetic parent", "status": "completed", "question": "Fixture?",
        "hypothesis": "Fixture", "decision_use": "Synthetic test only", "repository": {}, "seeds": [], "budget": {},
        "data_access": {**{name: None for name in DATA_ACCESS_FIELDS},
            "access_evidence_status": "pending_primary_evidence", "declared_policy": {
                "source": "synthetic test", "validation_allowed": False, "test_inputs_allowed": False,
                "test_targets_allowed": False, "test_metrics_allowed": False, "test_selection_allowed": False}},
        "results": {"synthetic": True}, "validity": {"valid": True}, "claims": {
            "supported": [], "not_supported": ["production readiness"], "caveats": []},
        "artifacts": [], "import_provenance": {"mode": "synthetic_test"}}
    write_record_once(root / "docs/experiments/records/EXP-PARENT.json", seal_record(record))
    return root


class FakePreparer:
    def __init__(self, owner, manifest=None):
        self.owner = owner
        self.base_source_fingerprint = "b" * 64
        self.source_fingerprint = "c" * 64 if manifest is None else "d" * 64
        self.manifest = manifest

    def __call__(self, job):
        assert self.owner.prebound
        self.owner.calls.append(job)
        if job.position == self.owner.fail_position:
            raise FileNotFoundError("synthetic missing warm cache")
        if self.owner.consume_rng:
            random.random()
        if self.owner.bad_return_job:
            job = replace(job, reserved_bytes=job.reserved_bytes - 1)
        sidecars = self.owner.sidecars
        if job.position == self.owner.oversize_position:
            sidecars = ({"oversized": b"x" * (job.reserved_bytes + 1)}, {})
        return pp.PreparedCPUJob(job, self.owner.graph, sidecars)

    def validate_jobs(self, jobs):
        self.owner.validated_jobs = tuple(jobs)
        assert self.manifest == pp.make_reservation_manifest(source_fingerprint=self.base_source_fingerprint,
            evidence_id=self.manifest["evidence_id"], measured_serialized_bytes=self.manifest["measured_serialized_bytes"])
        assert set(self.manifest["measured_serialized_bytes"]) == {pp.reservation_job_fingerprint(job) for job in jobs}
        assert all(self.manifest["measured_serialized_bytes"][pp.reservation_job_fingerprint(job)] <= job.reserved_bytes
                   for job in jobs)


class SyntheticReservationSource:
    def __init__(self):
        batch, sidecar = build_source_free_fixture()
        self.graph, self.sidecars = batch.raw_graph_batch, (sidecar, sidecar)
        self.prebound = False
        self.fail_position = self.oversize_position = None
        self.calls = []
        self.events = []
        self.validated_jobs = None
        self.consume_rng = self.bad_return_job = self.missing_binding = False
        self.change_final = self.change_reservation = False

    def schedule(self, **kwargs):
        self.events.append("schedule")
        if self.change_reservation:
            kwargs["reserved_bytes"] -= 1
        return build_prefetch_schedule(mode="fixture", **kwargs)

    def prebind(self, jobs):
        self.events.append("prebind")
        self.prebound = True
        bindings = {record: "a" * 64 for job in jobs for record in job.record_ids}
        if self.missing_binding:
            bindings.pop(next(iter(bindings)))
        return {"record_bindings": bindings, "fixture_source": "source-v1"}

    def preparer(self, bindings, *, reservation_manifest=None):
        self.events.append("bootstrap" if reservation_manifest is None else "final_preparer")
        return FakePreparer(self, reservation_manifest)

    def finalize(self, initial):
        self.events.append("finalize")
        if self.change_final:
            initial["fixture_source"] = "changed"
        return initial


def _run(cli, repo, source, **kwargs):
    options = dict(experiment_id="EXP-RESERVATIONS", parent_experiment_id="EXP-PARENT",
        output_root=repo / "outputs/EXP-RESERVATIONS", repository_root=repo,
        measured_draws=4, warmup_draws=2, max_job_payload_bytes=1024**2,
        buffer_byte_cap=2 * 1024**2, max_pending_batches=4, synthetic=True,
        synthetic_source_factory=lambda: source)
    options.update(kwargs)
    return cli.run_registered_reservations(**options)


def _record(repo):
    return load_record(repo / "docs/experiments/records/EXP-RESERVATIONS.json")


def test_plan_before_schedule_and_prebind_exact_final_reservations_without_pool(cli, repo, monkeypatch):
    source = SyntheticReservationSource()
    output = repo / "outputs/EXP-RESERVATIONS"
    schedule = source.schedule
    def observe_schedule(**kwargs):
        assert _record(repo)["status"] == "running"
        lifecycle = json.loads((output / "lifecycle-start.json").read_text())
        assert [item["status"] for item in lifecycle] == ["planned", "running"]
        for item in lifecycle:
            validate_record(item)
        return schedule(**kwargs)
    source.schedule = observe_schedule
    monkeypatch.setattr(pp, "OrderedPrefetch", lambda *a, **kw: pytest.fail("pool constructed"))
    result = _run(cli, repo, source)
    record = _record(repo)
    assert record["status"] == "completed" and record["budget"]["actual"] == 3
    assert result["warmup_jobs"] == 1 and result["future_measured_jobs"] == 2
    assert result["planned_max_admitted_jobs"] == 2  # Global cap, not four independently reserved jobs.
    assert result["retained_per_job_reserved_bytes"] == 1024**2
    jobs = jobs_from_prefetch_descriptor(json.loads((output / "schedule.json").read_text()))
    assert tuple(source.calls) == source.validated_jobs == jobs
    assert [job.draw_ids for job in jobs] == [(0, 1), (2, 3), (4, 5)]
    assert all(job.reserved_bytes == 1024**2 for job in jobs)
    rows = [json.loads(line) for line in (output / "payload-sizes.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["warmup", "measured", "measured"]
    assert all(row["serialized_bytes"] < row["reserved_bytes"] for row in rows)
    manifest = json.loads((output / "reservation-manifest.json").read_text())
    assert manifest == pp.make_reservation_manifest(source_fingerprint="b" * 64, evidence_id="EXP-RESERVATIONS",
        measured_serialized_bytes={row["job_fingerprint"]: row["serialized_bytes"] for row in rows})
    assert result["final_preparer_source_fingerprint"] == "d" * 64
    assert source.events == ["schedule", "prebind", "bootstrap", "final_preparer", "finalize"]
    for artifact in record["artifacts"]:
        assert file_sha256(repo / artifact["locator"]) == artifact["sha256"]
    assert (repo / "docs/EXPERIMENT_LEDGER.md").read_text() == render_registry(repo / "docs/experiments/records")
    assert all(record["data_access"][name] is None for name in DATA_ACCESS_FIELDS)
    assert "RSS bound" in record["claims"]["not_supported"]


@pytest.mark.parametrize("problem,match,actual", [
    ("fail_position", "missing warm", 1), ("oversize_position", "exceeds_reservation", 1),
    ("missing_binding", "complete_scheduled_bindings", None), ("consume_rng", "global_rng", 0),
    ("bad_return_job", "prepared_job_mismatch", 0), ("change_final", "source_bindings_changed", 3),
    ("change_reservation", "reservation_changed", None),
])
def test_failed_size_or_binding_gate_retains_partial_evidence_and_no_accepted_manifest(cli, repo, problem, match, actual):
    source = SyntheticReservationSource()
    setattr(source, problem, 1 if problem.endswith("position") else True)
    before = random.getstate()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError), match=match):
        _run(cli, repo, source)
    record = _record(repo)
    assert record["status"] == "failed" and record["budget"]["actual"] == actual
    assert not (repo / "outputs/EXP-RESERVATIONS/reservation-manifest.json").exists()
    assert all(record["data_access"][name] is None for name in DATA_ACCESS_FIELDS)
    assert random.getstate() == before
    assert record["artifacts"]


def test_terminal_ids_parent_and_output_are_never_overwritten(cli, repo):
    source = SyntheticReservationSource()
    parent_path = repo / "docs/experiments/records/EXP-PARENT.json"
    parent = parent_path.read_bytes()
    _run(cli, repo, source)
    record_path = repo / "docs/experiments/records/EXP-RESERVATIONS.json"
    original = record_path.read_bytes()
    with pytest.raises(ValueError, match="fresh_experiment"):
        _run(cli, repo, source)
    assert record_path.read_bytes() == original and parent_path.read_bytes() == parent
    with pytest.raises(ValueError, match="fresh_experiment"):
        _run(cli, repo, source, experiment_id="EXP-OTHER")


@pytest.mark.parametrize("status", ["planned", "running"])
def test_parent_must_be_terminal(cli, repo, status):
    path = repo / "docs/experiments/records/EXP-PARENT.json"
    parent = load_record(path)
    parent.update(status=status, results={}, validity={"valid": None})
    path.write_bytes(canonical_json_bytes(seal_record(parent)) + b"\n")
    with pytest.raises(ValueError, match="parent_record_not_terminal"):
        _run(cli, repo, SyntheticReservationSource())


@pytest.mark.parametrize("kwargs,match", [
    ({"split": "test"}, "train_only"), ({"split": "validation"}, "train_only"),
    ({"measured_draws": 3}, "invalid_measured_draws"), ({"warmup_draws": -2}, "invalid_warmup_draws"),
    ({"max_job_payload_bytes": 0}, "positive_explicit_payload_cap"),
    ({"max_job_payload_bytes": 3 * 1024**2}, "reservation_exceeds_global_cap"),
    ({"max_pending_batches": 0}, "invalid_max_pending"),
])
def test_invalid_configuration_precedes_source_manifest_and_factory(cli, repo, monkeypatch, kwargs, match):
    monkeypatch.setattr(cli, "source_manifest", lambda *a: pytest.fail("read source"))
    with pytest.raises(ValueError, match=match):
        _run(cli, repo, None, **kwargs)
    assert not (repo / "docs/experiments/records/EXP-RESERVATIONS.json").exists()


def test_source_edit_after_sizing_is_failed_even_if_complete_artifacts_exist(cli, repo):
    source = SyntheticReservationSource()
    finalize = source.finalize
    def edit(initial):
        (repo / "pyproject.toml").write_text("# changed source\n")
        return finalize(initial)
    source.finalize = edit
    with pytest.raises(ValueError, match="source_changed_during_execution"):
        _run(cli, repo, source)
    assert _record(repo)["status"] == "failed"
    assert (repo / "outputs/EXP-RESERVATIONS/reservation-manifest.json").is_file()


def test_descriptor_warmup_boundary_must_match_registered_protocol(cli, repo):
    source = SyntheticReservationSource()
    def swap(**kwargs):
        return build_prefetch_schedule(mode="fixture", measured_draws=kwargs["warmup_draws"],
                                      warmup_draws=kwargs["measured_draws"], reserved_bytes=kwargs["reserved_bytes"])
    source.schedule = swap
    with pytest.raises(ValueError, match="descriptor_protocol_mismatch"):
        _run(cli, repo, source)
    assert not source.prebound and source.calls == []
    assert _record(repo)["status"] == "failed"


@pytest.mark.parametrize("error,status", [(RuntimeError("synthetic startup"), "failed"), (KeyboardInterrupt(), "aborted")])
def test_factory_failure_is_terminal_after_durable_planning(cli, repo, error, status):
    def failed_factory():
        assert _record(repo)["status"] == "running"
        raise error
    with pytest.raises(type(error)):
        _run(cli, repo, None, synthetic_source_factory=failed_factory)
    record = _record(repo)
    assert record["status"] == status and record["budget"]["actual"] is None
    assert record["artifacts"]


def test_production_schedule_reuses_exact_frozen_builder_once(cli, monkeypatch):
    from music_critic.experiments.analysisgnn import performance_replay
    calls = []
    def schedule(mode, paths, count):
        calls.append((mode, paths, count))
        return [{"record_id": f"fixture:{i % 2}", "schedule_shift_pc": i % 12, "applied_shift_pc": 0}
                for i in range(count)]
    monkeypatch.setattr(performance_replay, "_schedule", schedule)
    paths = ProductionArtifactPaths(**{field.name: Path("/synthetic") / field.name for field in fields(ProductionArtifactPaths)})
    source = cli.ProductionReservationSource(paths)
    jobs, descriptor = source.schedule(measured_draws=200, warmup_draws=2, reserved_bytes=1024**2)
    assert calls == [("production", paths, 202)]
    assert len(jobs) == 101 and jobs[-1].draw_ids == (200, 201)
    assert jobs_from_prefetch_descriptor(descriptor) == jobs


def test_production_source_prebinds_unique_ids_before_preparer(cli, monkeypatch):
    events = []
    class FakeLoader:
        def __init__(self, *, paths, split):
            assert split == "train"
        def check_binding(self, record_id, *, split):
            assert split == "train"
            events.append(record_id)
        def validate_all_bindings(self, *, expected=None):
            return {"record_bindings": {record: "a" * 64 for record in events}}
    monkeypatch.setattr(cli, "FrozenProductionRecordLoader", FakeLoader)
    jobs, _ = build_prefetch_schedule(mode="fixture", measured_draws=4, warmup_draws=2, reserved_bytes=1024**2)
    source = cli.ProductionReservationSource(None)
    initial = source.prebind(jobs)
    assert events == sorted({record for job in jobs for record in job.record_ids})
    assert source.finalize(initial) == initial
    with pytest.raises(ValueError, match="train_only"):
        cli.ProductionReservationSource(None, split="test")


@pytest.mark.parametrize("compact", [False, True])
def test_cli_requires_explicit_size_and_global_buffer_policy(cli, monkeypatch, compact):
    called = []
    monkeypatch.setattr(cli, "run_registered_reservations", lambda **kwargs: called.append(kwargs) or {"parse": True})
    args = ["--experiment-id", "EXP-CLI", "--parent-experiment-id", "EXP-PARENT", "--output-root", "outputs/fixture",
        "--measured-draws", "800", "--warmup-draws", "160", "--max-job-payload-bytes", "1048576",
        "--buffer-byte-cap", "2097152", "--max-pending-batches", "4"]
    for field in fields(ProductionArtifactPaths):
        args.extend(["--" + field.name.replace("_", "-"), f"/synthetic/{field.name}"])
    if compact:
        args.extend(["--preparation-strategy", "compact_cache", "--compact-cache-global-max-bytes", "8388608",
                     "--compact-cache-preparing-processes", "1"])
    assert cli.main(args) == 0
    assert called[0]["max_job_payload_bytes"] == 1048576 and called[0]["buffer_byte_cap"] == 2097152
    assert called[0]["preparation_strategy"] == ("compact_cache" if compact else "frozen")
    assert called[0]["compact_cache_global_max_bytes"] == (8388608 if compact else None)
    assert called[0]["compact_cache_preparing_processes"] == (1 if compact else None)
    for flag in ("--max-job-payload-bytes", "--buffer-byte-cap", "--max-pending-batches"):
        omitted = list(args)
        index = omitted.index(flag)
        del omitted[index:index + 2]
        with pytest.raises(SystemExit):
            cli.main(omitted)
    with pytest.raises(SystemExit):
        cli.main(args + ["--split", "validation"])
    assert len(called) == 1


@pytest.mark.parametrize("kwargs,match", [
    ({"preparation_strategy": "other"}, "invalid_preparation_strategy"),
    ({"compact_cache_global_max_bytes": 100}, "cache_options_require"),
    ({"compact_cache_preparing_processes": 1}, "cache_options_require"),
    ({"preparation_strategy": "compact_cache"}, "explicit_positive_compact_cache_budget"),
    ({"preparation_strategy": "compact_cache", "compact_cache_global_max_bytes": True}, "explicit_positive"),
    ({"preparation_strategy": "compact_cache", "compact_cache_global_max_bytes": 100}, "explicit_compact_cache_process"),
    ({"preparation_strategy": "compact_cache", "compact_cache_global_max_bytes": 100,
      "compact_cache_preparing_processes": True}, "explicit_compact_cache_process"),
    ({"preparation_strategy": "compact_cache", "compact_cache_global_max_bytes": 100,
      "compact_cache_preparing_processes": 3}, "explicit_compact_cache_process"),
    ({"preparation_strategy": "compact_cache", "compact_cache_global_max_bytes": 1,
      "compact_cache_preparing_processes": 4}, "compact_cache_process_budget_empty"),
])
def test_invalid_compact_policy_fails_before_source_or_registration(cli, repo, monkeypatch, kwargs, match):
    monkeypatch.setattr(cli, "source_manifest", lambda *a: pytest.fail("source read"))
    with pytest.raises(ValueError, match=match):
        _run(cli, repo, None, **kwargs)
    assert not (repo / "docs/experiments/records/EXP-RESERVATIONS.json").exists()


class CompactSyntheticSource(SyntheticReservationSource):
    """Only a lifecycle/routing spy; concrete compact semantics are tested below."""
    def __init__(self):
        super().__init__()
        self.policies, self.resets = [], 0

    def preparer(self, initial, *, reservation_manifest=None, cache_policy=None):
        self.policies.append(cache_policy)
        prepare = super().preparer(initial, reservation_manifest=reservation_manifest)
        prepare.cache_diagnostics = lambda: {"synthetic": True, "calls": len(self.calls)}
        def reset():
            self.resets += 1
        prepare.reset_cache = reset
        return prepare


@pytest.mark.parametrize("fail", [False, True])
def test_integrated_policy_is_planned_and_measured_once_with_cleanup(cli, repo, fail):
    source = CompactSyntheticSource()
    if fail:
        source.fail_position = 1
    kwargs = {"preparation_strategy": "compact_cache", "compact_cache_global_max_bytes": 8 * 1024**2,
              "compact_cache_preparing_processes": 1}
    if fail:
        with pytest.raises(FileNotFoundError):
            _run(cli, repo, source, **kwargs)
    else:
        result = _run(cli, repo, source, **kwargs)
        assert result["cache_diagnostics"] == {"synthetic": True, "calls": 3}
        assert result["measured_jobs"] == 3
        assert len(source.calls) == 3  # No second pass for a different capacity.
        assert result["reservation_compatibility"]["instance_source_fingerprint_includes_cache_policy"]
    record = _record(repo)
    policy = {"schema": "AnalysisGNNProcessLocalCompactCache@1.0.0",
              "global_max_bytes": 8 * 1024**2, "preparing_processes": 1}
    assert source.policies == [policy] * (1 if fail else 2)
    assert source.resets == 1
    assert record["protocol"]["cache_policy"] == policy
    assert record["protocol"]["preparation_strategy"] == "compact_cache"
    assert record["protocol"]["workers_for_this_gate"] == 0
    output = repo / "outputs/EXP-RESERVATIONS"
    lifecycle = json.loads((output / "lifecycle-start.json").read_text())
    assert all(row["protocol"]["cache_policy"] == policy for row in lifecycle)
    assert all(cli.COMPACT_SIZE_LIMIT in row["claims"]["caveats"] for row in lifecycle)
    assert "capacity-independent serialized payload bound" in record["claims"]["not_supported"]
    assert record["status"] == ("failed" if fail else "completed")


def test_default_does_not_add_cache_policy_or_call_cache_methods(cli, repo):
    result = _run(cli, repo, SyntheticReservationSource())
    assert "cache_policy" not in result and "preparation_strategy" not in result
    assert "cache_policy" not in _record(repo)["protocol"]


def test_production_source_passes_compact_policy_to_both_preparers(cli, monkeypatch):
    calls = []
    monkeypatch.setattr(pp, "WarmTrainBatchPreparer", lambda *args, **kw: calls.append((args, kw)))
    policy = cli._cache_policy("compact_cache", 8 * 1024**2, 4)
    source = cli.ProductionReservationSource(None)
    source.preparer({"record_bindings": {}}, cache_policy=policy)
    source.preparer({"record_bindings": {}}, cache_policy=policy, reservation_manifest={"explicit": True})
    source.preparer({"record_bindings": {}})
    assert [row[1].get("cache_policy") for row in calls] == [policy, policy, None]
    assert all(row[1]["identity_only"] is True for row in calls)


def _warm_helper():
    helper_path = Path(__file__).with_name("test_performance_prefetch.py")
    spec = importlib.util.spec_from_file_location("reservation_warm_test_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def test_registered_integrated_gate_uses_current_concrete_preparer(cli, repo, tmp_path, monkeypatch):
    from music_critic.experiments.analysisgnn import performance_replay
    paths, record_id, _, _ = _warm_helper()._warm_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(performance_replay, "_schedule", lambda mode, paths, count: [
        {"record_id": record_id, "schedule_shift_pc": i % 12, "applied_shift_pc": 0} for i in range(count)])
    source = cli.ProductionReservationSource(paths)
    source.schedule = lambda **kwargs: build_prefetch_schedule(mode="fixture", **kwargs)
    result = _run(cli, repo, source, preparation_strategy="compact_cache",
                  compact_cache_global_max_bytes=8*1024**2, compact_cache_preparing_processes=1)
    output = repo / "outputs/EXP-RESERVATIONS"
    bindings = json.loads((output / "source-bindings.json").read_text())
    manifest = json.loads((output / "reservation-manifest.json").read_text())
    consumer = pp.WarmTrainBatchPreparer(paths, bindings, cache_policy=result["cache_policy"],
                                        reservation_manifest=manifest)
    assert result["base_source_fingerprint"] == consumer.base_source_fingerprint
    assert result["final_preparer_source_fingerprint"] == consumer.source_fingerprint
    assert result["cache_diagnostics"]["hits"] == 5
    assert result["cache_diagnostics"]["misses"] == 1
    assert _record(repo)["status"] == "completed"
    assert _record(repo)["protocol"]["mode"] == "synthetic"
    assert len((output / "payload-sizes.jsonl").read_text().splitlines()) == 3


def test_concrete_cache_values_match_across_budgets_but_measured_sizes_are_not_universal_bounds(cli, tmp_path, monkeypatch):
    # Existing source-free warm fixture writes only synthetic TRAIN JSON in tmp.
    # It exercises the concrete loader, verified compact sidecar bytes, graph
    # construction, direct collation and native protocol-5 CPU serialization.
    from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
    paths, record_id, _, bindings = _warm_helper()._warm_fixture(tmp_path, monkeypatch)
    source = cli.ProductionReservationSource(paths)
    jobs = tuple(pp.PrefetchJob(i, (2*i, 2*i+1), (record_id, record_id),
                               (i, (i+5) % 12), (0, 0), 2_000_000) for i in range(2))
    def contents(payload):
        raw = payload.raw_graph_batch
        return recursive_state_fingerprint({"job": asdict(payload.job), "raw": raw.to_dict(),
            "slices": raw._slice_dict, "increments": raw._inc_dict,
            "num_graphs": raw.num_graphs, "sidecars": payload.sidecars})
    policies = [cli._cache_policy("compact_cache", total, processes)
                for total in (4, 8*1024**2) for processes in (1, 4)]
    candidates = [source.preparer(bindings, cache_policy=policy) for policy in policies]
    frozen = source.preparer(bindings)
    expected = [contents(frozen(job)) for job in jobs]
    assert len({item.base_source_fingerprint for item in candidates}) == 1
    assert len({item.source_fingerprint for item in candidates}) == len(policies)
    assert frozen.base_source_fingerprint != candidates[0].base_source_fingerprint
    sizes_by_policy, observations = [], []
    for candidate in candidates:
        observed_sizes = {}
        try:
            for job, expected_contents in zip(jobs, expected, strict=True):
                encoded = pp._prepare_bytes(candidate, job)
                decoded = pickle.loads(encoded)  # Own tiny synthetic CPU output only.
                assert contents(decoded) == expected_contents
                observed_sizes[pp.reservation_job_fingerprint(job)] = len(encoded)
            observations.append(candidate.cache_diagnostics())
        finally:
            candidate.reset_cache()
        sizes_by_policy.append(observed_sizes)
    # Small budgets bypass every entry; larger budgets exercise within-job and
    # later-job hits. Equal values do not imply equal pickle alias topology or
    # length: sizes below are observations, never universal capacity bounds.
    assert all(row["oversized_bypasses"] == 4 for row in observations[:2])
    assert all(row["hits"] == 3 and row["misses"] == 1 for row in observations[2:])
    assert sizes_by_policy[0] != sizes_by_policy[2]
    # One observed history supplies the manifest. Other capacities retain the
    # algorithm/population identity and enforce actual size <= R at runtime.
    manifest = pp.make_reservation_manifest(source_fingerprint=candidates[0].base_source_fingerprint,
        evidence_id="synthetic-integrated-size", measured_serialized_bytes=sizes_by_policy[2])
    for policy in policies:
        consumer = source.preparer(bindings, cache_policy=policy, reservation_manifest=manifest)
        consumer.validate_jobs(jobs)
        assert consumer.cache_diagnostics()["entries"] == 0
        assert len(pp._prepare_bytes(consumer, jobs[0])) <= jobs[0].reserved_bytes
        consumer.reset_cache()
    with pytest.raises(ValueError, match="reservation_manifest_binding_mismatch"):
        source.preparer(bindings, reservation_manifest=manifest)
    changed = {**manifest, "source_fingerprint": "0" * 64}
    with pytest.raises(ValueError, match="reservation_manifest_binding_mismatch"):
        source.preparer(bindings, cache_policy=policies[0], reservation_manifest=changed)
    # A claimed measured length cannot override the actual serializer cap.
    # The job fingerprint omits R; the full schedule still binds it separately.
    too_small = replace(jobs[0], reserved_bytes=1024)
    small_manifest = pp.make_reservation_manifest(source_fingerprint=candidates[0].base_source_fingerprint,
        evidence_id="synthetic-dishonest-size", measured_serialized_bytes={pp.reservation_job_fingerprint(too_small): 1})
    consumer = source.preparer(bindings, cache_policy=policies[0], reservation_manifest=small_manifest)
    consumer.validate_jobs((too_small,))
    with pytest.raises(pp.PrefetchError, match="result_exceeds_reservation"):
        pp._prepare_bytes(consumer, too_small)
    consumer.reset_cache()
