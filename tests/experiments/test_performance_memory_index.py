"""Synthetic temporary-repository lifecycle and TRAIN-only indexing tests."""

from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from music_critic.experiments.analysisgnn import performance_data as pd
from music_critic.experiments.analysisgnn.corrected_training import (
    ProductionArtifactPaths, build_source_free_fixture,
)
from music_critic.experiments.analysisgnn.performance_memory import TrainRecordMetadata
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, file_sha256, load_record, render_registry, seal_record,
    validate_record, write_record_once,
)


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[2] / "scripts/run_analysisgnn_performance_memory_index.py"
    name = "test_memory_index_cli"
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
    (root / "pyproject.toml").write_text("# isolated synthetic source\n")
    subprocess.run(["git", "-C", str(root), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", "fixture"], check=True, capture_output=True)
    record = {
        "schema_version": "1.0.0", "experiment_id": "EXP-PARENT", "kind": "engineering_gate",
        "phase": "9E-P1", "title": "Synthetic terminal parent", "status": "completed",
        "question": "Synthetic test?", "hypothesis": "Synthetic test", "decision_use": "Unit test only",
            "repository": {}, "seeds": [], "budget": {}, "data_access": {
            **{name: None for name in DATA_ACCESS_FIELDS}, "access_evidence_status": "pending_primary_evidence",
            "declared_policy": {"source": "synthetic test", "validation_allowed": False,
                "test_inputs_allowed": False, "test_targets_allowed": False,
                "test_metrics_allowed": False, "test_selection_allowed": False}},
        "results": {"synthetic": True}, "validity": {"valid": True},
        "claims": {"supported": [], "not_supported": ["real data"], "caveats": []},
        "artifacts": [], "import_provenance": {"mode": "synthetic_test"},
    }
    write_record_once(root / "docs/experiments/records/EXP-PARENT.json", seal_record(record))
    return root


class SyntheticIndexSource:
    def __init__(self):
        batch, sidecar = build_source_free_fixture()
        self.graph = batch.raw_graph_batch.to_data_list()[0]
        self.sidecar = sidecar
        self.ids = ["a", "b"]
        self.events = []
        self.prebound = False
        self.fail_on = None
        self.change_final = False
        self.nontrain = False
        self.bad_bytes = False

    def prebind(self):
        self.events.append("prebind")
        self.prebound = True
        return {"record_ids": self.ids, "synthetic_source_seal": "d" * 64}

    def read(self, record_id):
        assert self.prebound
        self.events.append(record_id)
        if record_id == self.fail_on:
            raise FileNotFoundError("synthetic missing warm entry")
        sidecar = deepcopy(self.sidecar)
        sidecar["record_id"] = record_id
        payload = json.dumps(sidecar, indent=2).encode()
        metadata = TrainRecordMetadata(
            record_id=record_id, source_component_id=f"component-{record_id}", split="train",
            canonical_sha256="a" * 64, sidecar_sha256=sha256(payload).hexdigest(),
            metadata_sha256="b" * 64, binding_seal="c" * 64, source_seal="d" * 64,
            code_seal="e" * 64, split_sha256="f" * 64,
            source_versions=(("fixture", "synthetic-v1"),),
        )
        if self.nontrain:
            object.__setattr__(metadata, "split", "validation")
        if self.bad_bytes:
            payload += b" "
        return pd.PreparedRecord(deepcopy(self.graph), sidecar, metadata.binding_seal), metadata, payload

    def finalize(self, initial):
        self.events.append("finalize")
        return {**initial, "changed": True} if self.change_final else initial


def _run(cli, repo, source, **kwargs):
    return cli.run_registered_memory_index(
        experiment_id="EXP-INDEX", output_root=repo / "outputs/EXP-INDEX",
        parent_experiment_id="EXP-PARENT", repository_root=repo,
        synthetic=True, synthetic_source_factory=lambda: source,
        synthetic_expected_count=2, **kwargs,
    )


def _terminal(repo):
    return load_record(repo / "docs/experiments/records/EXP-INDEX.json")


def test_planned_running_and_source_manifest_exist_before_loader_factory(cli, repo):
    source = SyntheticIndexSource()
    parent = (repo / "docs/experiments/records/EXP-PARENT.json").read_bytes()
    output = repo / "outputs/EXP-INDEX"

    def factory():
        assert _terminal(repo)["status"] == "running"
        lifecycle = json.loads((output / "lifecycle-start.json").read_text())
        assert [record["status"] for record in lifecycle] == ["planned", "running"]
        for record in lifecycle:
            validate_record(record)
            assert record["budget"]["actual"] is None
        assert (output / "source-manifest-initial.json").is_file()
        return source

    result = cli.run_registered_memory_index(
        experiment_id="EXP-INDEX", output_root=output, parent_experiment_id="EXP-PARENT",
        repository_root=repo, synthetic=True, synthetic_source_factory=factory,
        synthetic_expected_count=2, command=["python", "synthetic_cli"],
    )
    record = _terminal(repo)
    assert record["status"] == "completed" and record["budget"]["actual"] == 2
    assert result["indexed_records"] == result["indexed_components"] == 2
    assert source.events == ["prebind", "a", "b", "finalize"]
    assert record["results"] == result
    assert (repo / "docs/EXPERIMENT_LEDGER.md").read_text() == render_registry(repo / "docs/experiments/records")
    assert (repo / "docs/experiments/records/EXP-PARENT.json").read_bytes() == parent
    for artifact in record["artifacts"]:
        assert file_sha256(repo / artifact["locator"]) == artifact["sha256"]
    evidence = json.loads((output / "execution-evidence.json").read_text())
    assert evidence["argv"] == ["python", "synthetic_cli"]
    assert all(record["data_access"][name] is None for name in DATA_ACCESS_FIELDS)
    assert "GPU memory capacity" in record["claims"]["not_supported"]


@pytest.mark.parametrize("split", ["test", "validation", "TRAIN", None])
def test_forbidden_split_fails_before_loader_or_any_source_read(cli, repo, monkeypatch, split):
    def forbidden(*args, **kwargs):
        pytest.fail("forbidden split reached source")
    monkeypatch.setattr(cli, "source_manifest", forbidden)
    with pytest.raises(ValueError, match="train_only"):
        cli.run_registered_memory_index(
            experiment_id="EXP-INDEX", output_root=repo / "outputs/EXP-INDEX",
            parent_experiment_id="EXP-PARENT", repository_root=repo, split=split,
            synthetic=True, synthetic_source_factory=forbidden, synthetic_expected_count=2,
        )
    assert not (repo / "docs/experiments/records/EXP-INDEX.json").exists()
    with pytest.raises(ValueError, match="train_only"):
        cli.ProductionIndexSource(None, split=split)


def test_fresh_ids_outputs_and_terminal_record_are_immutable(cli, repo):
    source = SyntheticIndexSource()
    _run(cli, repo, source)
    path = repo / "docs/experiments/records/EXP-INDEX.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="fresh_experiment"):
        _run(cli, repo, source)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="fresh_experiment"):
        cli.run_registered_memory_index(experiment_id="NEW-ID", output_root=repo / "outputs/EXP-INDEX",
            parent_experiment_id="EXP-PARENT", repository_root=repo, synthetic=True,
            synthetic_source_factory=lambda: pytest.fail("executed"), synthetic_expected_count=2)


@pytest.mark.parametrize("status", ["planned", "running"])
def test_parent_must_be_sealed_and_terminal_before_index_start(cli, repo, status):
    path = repo / "docs/experiments/records/EXP-PARENT.json"
    parent = load_record(path)
    parent.update(status=status, results={}, validity={"valid": None})
    path.write_bytes(canonical_json_bytes(seal_record(parent)) + b"\n")
    with pytest.raises(ValueError, match="parent_record_not_terminal"):
        _run(cli, repo, SyntheticIndexSource())
    assert not (repo / "outputs/EXP-INDEX").exists()


def test_corrupted_parent_fingerprint_rejected(cli, repo):
    path = repo / "docs/experiments/records/EXP-PARENT.json"
    parent = json.loads(path.read_text())
    parent["title"] = "changed without reseal"
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError, match="parent_record_invalid"):
        _run(cli, repo, SyntheticIndexSource())


def test_partial_failure_keeps_flushed_rows_and_terminal_unknown_access(cli, repo):
    source = SyntheticIndexSource()
    source.fail_on = "b"
    with pytest.raises(FileNotFoundError, match="missing warm"):
        _run(cli, repo, source)
    record = _terminal(repo)
    assert record["status"] == "failed" and record["budget"]["actual"] == 1
    output = repo / "outputs/EXP-INDEX"
    rows = [json.loads(line) for line in (output / "size-index.jsonl").read_text().splitlines()]
    assert [row["record_id"] for row in rows] == ["a"]
    assert not (output / "extreme-b2-cases.json").exists()
    assert all(record["data_access"][name] is None for name in DATA_ACCESS_FIELDS)
    assert any(artifact["role"] == "size-index.jsonl" for artifact in record["artifacts"])


@pytest.mark.parametrize("error", [RuntimeError("synthetic startup failure"), KeyboardInterrupt()])
def test_factory_failure_still_seals_before_any_rows(cli, repo, error):
    def factory():
        assert _terminal(repo)["status"] == "running"
        raise error
    with pytest.raises(type(error)):
        cli.run_registered_memory_index(experiment_id="EXP-INDEX", output_root=repo / "outputs/EXP-INDEX",
            parent_experiment_id="EXP-PARENT", repository_root=repo, synthetic=True,
            synthetic_source_factory=factory, synthetic_expected_count=2)
    record = _terminal(repo)
    assert record["status"] == ("aborted" if isinstance(error, KeyboardInterrupt) else "failed")
    assert record["budget"]["actual"] is None and record["artifacts"]


@pytest.mark.parametrize("ids", [["a"], ["a", "a"], ["a", "b", "c"]])
def test_incomplete_or_duplicate_membership_fails_before_graph_reads(cli, repo, ids):
    source = SyntheticIndexSource()
    source.ids = ids
    with pytest.raises(ValueError, match="complete_train_membership_required"):
        _run(cli, repo, source)
    assert source.events == ["prebind"]
    assert _terminal(repo)["status"] == "failed"


@pytest.mark.parametrize("problem,match", [("bad_bytes", "sidecar_hash_mismatch"),
                                           ("nontrain", "train_only"),
                                           ("change_final", "source_bindings_changed")])
def test_bad_row_hash_split_or_final_source_is_not_completed(cli, repo, problem, match):
    source = SyntheticIndexSource()
    setattr(source, problem, True)
    with pytest.raises(ValueError, match=match):
        _run(cli, repo, source)
    assert _terminal(repo)["status"] == "failed"


def test_source_change_during_index_is_terminal_failure(cli, repo):
    source = SyntheticIndexSource()
    finalize = source.finalize
    def changed(initial):
        (repo / "pyproject.toml").write_text("# changed source\n")
        return finalize(initial)
    source.finalize = changed
    with pytest.raises(ValueError, match="source_changed_during_execution"):
        _run(cli, repo, source)
    assert _terminal(repo)["status"] == "failed"


def test_finalize_cannot_mutate_initial_manifest_to_hide_binding_change(cli, repo):
    source = SyntheticIndexSource()
    def mutate(initial):
        initial["synthetic_source_seal"] = "0" * 64
        return initial
    source.finalize = mutate
    with pytest.raises(ValueError, match="source_bindings_changed"):
        _run(cli, repo, source)
    output = repo / "outputs/EXP-INDEX"
    assert json.loads((output / "source-bindings-initial.json").read_text())["synthetic_source_seal"] == "d" * 64
    assert _terminal(repo)["status"] == "failed"


def test_evidence_hash_failure_seals_terminal_without_overwriting_partial_files(cli, repo, monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic artifact failure")
    monkeypatch.setattr(cli, "_artifact_rows", fail)
    with pytest.raises(RuntimeError, match="artifact failure"):
        _run(cli, repo, SyntheticIndexSource())
    record = _terminal(repo)
    assert record["status"] == "failed" and "evidence_finalization_failed" in record["validity"]["reason"]
    assert record["claims"]["supported"] == []
    assert len((repo / "outputs/EXP-INDEX/size-index.jsonl").read_text().splitlines()) == 2


def test_production_adapter_rejects_split_rehash_mismatch_before_payload(cli, tmp_path, monkeypatch):
    class FakeFrozen:
        def __init__(self, *, paths, split):
            assert split == "train"
            self._split = SimpleNamespace(path=tmp_path / "split.jsonl", digest="a" * 64)
    monkeypatch.setattr(pd, "FrozenProductionRecordLoader", FakeFrozen)
    monkeypatch.setattr(pd, "_read_stable", lambda path: (b"", SimpleNamespace(digest="b" * 64)))
    with pytest.raises(ValueError, match="split_hash_mismatch"):
        cli.ProductionIndexSource(None)


def test_production_adapter_prebinds_all_train_and_never_passes_excluded_ids(cli, monkeypatch, tmp_path):
    synthetic = SyntheticIndexSource()
    assignments = [{"record_id": str(i), "source_component_id": "component", "split": "train"}
                   for i in range(1295)]
    assignments.extend({"record_id": split, "source_component_id": split, "split": split}
                       for split in ("validation", "test"))
    payload = b"\n".join(json.dumps(row).encode() for row in assignments)
    digest = sha256(payload).hexdigest()
    class FakeFrozen:
        def __init__(self, *, paths, split):
            self._split = SimpleNamespace(path=tmp_path / "split.jsonl", digest=digest)
            self._train_ids = frozenset(str(i) for i in range(1295))
            self.checked = []
            self.code_seal, self.source_seal = "e" * 64, "d" * 64
            self._records = {}
        def check_binding(self, record_id, *, split):
            assert split == "train" and record_id in self._train_ids
            self.checked.append(record_id)
        def validate_all_bindings(self, *, expected=None):
            return {"record_bindings": {record_id: "c" * 64 for record_id in self._train_ids}}
        def _load(self, record_id, *, split):
            assert len(self.checked) == 1295 and split == "train"
            prepared, metadata, sidecar = synthetic.read(record_id)
            self._records[record_id] = tuple(SimpleNamespace(digest=value) for value in (
                metadata.canonical_sha256, metadata.sidecar_sha256, metadata.metadata_sha256))
            return prepared, sidecar
    synthetic.prebound = True
    monkeypatch.setattr(pd, "FrozenProductionRecordLoader", FakeFrozen)
    monkeypatch.setattr(pd, "_read_stable", lambda path: (payload, SimpleNamespace(digest=digest)))
    source = cli.ProductionIndexSource(None)
    initial = source.prebind()
    assert len(initial["record_ids"]) == len(source.loader.checked) == 1295
    assert not {"validation", "test"} & set(source.loader.checked)
    prepared, metadata, sidecar = source.read("0")
    assert metadata.record_id == "0" and metadata.split == "train"
    assert prepared.binding_seal == metadata.binding_seal
    assert sha256(sidecar).hexdigest() == metadata.sidecar_sha256
    assert source.finalize(initial) == initial
    for record_id in ("test", "validation"):
        with pytest.raises(ValueError, match="record_not_train"):
            source.read(record_id)


def test_cli_exposes_only_train_and_no_synthetic_override(cli, monkeypatch):
    called = []
    def run(**kwargs):
        called.append(kwargs)
        return {"synthetic_cli_parse": True}
    monkeypatch.setattr(cli, "run_registered_memory_index", run)
    args = ["--experiment-id", "EXP-CLI", "--parent-experiment-id", "EXP-PARENT", "--output-root", "outputs/fixture"]
    for field in fields(ProductionArtifactPaths):
        args.extend(["--" + field.name.replace("_", "-"), f"/synthetic/{field.name}"])
    assert cli.main(args) == 0
    assert isinstance(called[0]["paths"], ProductionArtifactPaths)
    assert called[0]["split"] == "train"
    for forbidden in (["--split", "test"], ["--split", "validation"], ["--synthetic"]):
        with pytest.raises(SystemExit):
            cli.main(args + forbidden)
    assert len(called) == 1
