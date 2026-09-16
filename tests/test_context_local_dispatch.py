"""Synthetic dispatcher gates, exact sweep equivalence and retained failures.

Config tests use production constants. Only execution fixtures scale counts and
replace the checkpoint identity with newly generated random weights.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from music_critic.data import RationalTime, piece_to_dict
from music_critic.experiments import context_local_dispatch as dispatch
from music_critic.experiments.context_diagnostics import FileBinding, fingerprint, local_features, raw_candidates
from music_critic.experiments.context_diagnostics_pairs import build_context_pair_suite
from music_critic.experiments.context_local_contract import MetadataSelection
from music_critic.experiments.context_local_reference import TASK_LEVELS, FrozenC2Reference
from music_critic.experiments.registry import canonical_json_bytes, load_record, seal_record, validate_record, write_record_once
from test_context_local_exports import synthetic_sidecar, _rehash
from test_context_local_reference import checkpoint


def test_prospective_sizing_limits_keep_selected_members_and_reference_aligned():
    from music_critic.experiments import context_local_reference as reference
    assert dispatch.BUDGET["max_notes_per_piece"] == reference.MAX_NOTES == 16384
    assert dispatch.BUDGET["max_candidates_per_level"] == reference.MAX_CANDIDATES_PER_LEVEL == 16384
    assert dispatch.BUDGET["train_pieces"] == 128 and dispatch.BUDGET["validation_pieces"] == 32
    assert dispatch.SIZING_EVIDENCE["selected_max_notes"] < reference.MAX_NOTES
    assert dispatch.SIZING_EVIDENCE["selected_max_target_entities"] < 20000
    assert dispatch.SIZING_EVIDENCE["selected_max_local_window_notes"] < 256


def _config():
    return {"schema_version": dispatch.CONFIG_SCHEMA,
            "experiment_id": "EXP-CTX-C2-LOCAL-SYNTHETIC-UNIT", "tasks": list(dispatch.TASKS),
            "seed": 17, "bootstrap_draws": 200, "device": "cpu",
            "budget": deepcopy(dispatch.BUDGET), "checkpoint": deepcopy(dispatch.CHECKPOINT),
            "metadata_bindings": {}, "cache_index_binding": {},
            "binding_audit": {"experiment_id": "EXP-RESEARCH-DILEMMA-BINDINGS-001", "record_binding": {}}}


def _write(path, value):
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, allow_nan=False).encode()
    path.write_bytes(raw)
    return FileBinding(str(path), sha256(raw).hexdigest(), len(raw))


def _declaration(binding, root):
    return {"locator": str(Path(binding.path).relative_to(root)),
            "sha256": binding.sha256, "size_bytes": binding.size_bytes}


def _seed_protocol(config, selection):
    protocol = {"source_sha256": {"synthetic.py": "a" * 64}, "selection": selection,
                "claim_limit": "synthetic mechanics only", "source_provenance_limit": "generated fixtures only",
                "config_fingerprint": fingerprint(config)}
    protocol["protocol_fingerprint"] = fingerprint(protocol)
    return protocol


def _replace_index(env, rows):
    raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    binding = _write(env.inputs / "index.jsonl", raw)
    env.config["cache_index_binding"] = _declaration(binding, env.inputs)
    protocol = _seed_protocol(env.config, env.metadata.to_dict())
    audit = dispatch.make_planned_record(env.config, protocol)
    audit.update(experiment_id="EXP-RESEARCH-DILEMMA-BINDINGS-001", parent_ids=[], status="completed",
                 results={"synthetic_index_registered": True}, validity={"valid": True, "reason": None},
                 artifacts=[{"role": "fixture_index", "locator": "index.jsonl", "sha256": binding.sha256,
                             "size_bytes": binding.size_bytes, "availability": "local_ignored"}])
    audit = validate_record(seal_record(audit))
    audit_binding = _write(env.inputs / "audit.json", audit)
    env.config["binding_audit"]["record_binding"] = _declaration(audit_binding, env.inputs)
    return audit


def _environment(tmp_path, monkeypatch, *, masked_validation=False):
    inputs, cache = tmp_path / "inputs", tmp_path / "cache"
    inputs.mkdir(); cache.mkdir()
    assignments, pieces, selected, index = {}, {}, [], []
    for identity, split in (("train-a", "train"), ("train-b", "train"), ("validation-a", "validation"), ("closed-test", "test")):
        token = sha256(identity.encode()).hexdigest()[:24]
        component = "component:" + identity
        piece_id = "piece:dilemmadata-dlc-" + token
        assignments[identity] = {"record_id": identity, "split": split,
                                 "source_component_id": component, "dialect": "dlc"}
        pieces[identity] = piece_id
        if split == "test":
            continue  # TEST is identity-only: no directory or payload exists.
        piece = replace(build_context_pair_suite()[0].left, dataset_name="dilemmadata",
                        piece_id=piece_id, source_group_id=component, split=split)
        root = cache / token
        root.mkdir()
        raw = _write(root / "piece.json", piece_to_dict(piece))
        sidecar = synthetic_sidecar(piece, record_id=identity)
        if masked_validation and split == "validation":
            for entity in sidecar["entities"]:
                for state in entity["targets"].values():
                    state.update(available=False, masked=True, missing_reason="synthetic_missing",
                                 canonical_value=None, source_value=None)
            _rehash(sidecar)
        targets = _write(root / "target_sidecar.json", sidecar)
        metadata = _write(root / "metadata.json", {"record_id": identity, "split": split,
                          "piece_sha256": raw.sha256, "sidecar_sha256": targets.sha256})
        row = {"record_id": identity, "split": split, "source_component_id": component}
        for key, binding in (("raw", raw), ("sidecar", targets), ("metadata", metadata)):
            row[key] = {"path": "/unopened/server/cache/" + token + "/" + Path(binding.path).name,
                        "sha256": binding.sha256, "size_bytes": binding.size_bytes}
        index.append(row)
        selected.append({"record_id": identity, "component_id": component, "split": split})
    content = {"assignments": assignments, "selected": selected, "record_to_piece": pieces,
               "evidence": {"source_free_fixture": True}}
    metadata = MetadataSelection(assignments, tuple(selected), fingerprint(content), content["evidence"], pieces)
    monkeypatch.setattr(dispatch, "load_authoritative_metadata", lambda bindings: metadata)
    env = SimpleNamespace(inputs=inputs, cache=cache, metadata=metadata, config=_config(), rows=index)
    _replace_index(env, index)
    return env


@pytest.mark.parametrize("order", ["original", "reverse", "shuffled"])
def test_interval_sweep_equals_original_exact_edge_windows(order):
    piece = replace(build_context_pair_suite()[0].left, split="validation")
    base, end = piece.notes[0], piece.duration_qn
    tiny = RationalTime(10**25 - 1, 3 * 10**25)
    specifications = [
        ("expires", RationalTime(0), RationalTime(1), False, False),
        ("held", RationalTime(0), end, False, False),
        ("boundary", RationalTime(1), RationalTime(1), False, False),
        ("fraction", RationalTime(1, 3), RationalTime(1, 3), False, False),
        ("near-fraction", tiny, RationalTime(1, 3) - tiny, False, False),
        ("clipped-last-window", end - RationalTime(1, 3), RationalTime(1, 3), False, False),
        ("percussion", RationalTime(1, 2), RationalTime(1), True, False),
        ("grace", RationalTime(1, 4), RationalTime(0), False, True),
        ("endpoint", end, RationalTime(0), False, True),
    ]
    notes = [replace(base, note_id="note:" + name, onset_qn=start, duration_qn=duration,
                     is_percussion=drum, is_grace=grace, pitch=60 + i % 3)
             for i, (name, start, duration, drum, grace) in enumerate(specifications)]
    assert notes[0].pitch == notes[3].pitch  # Distinct overlapping unison notes retain multiplicity.
    if order == "reverse": notes.reverse()
    if order == "shuffled": random.Random(17).shuffle(notes)
    piece = replace(piece, notes=tuple(notes))
    checks = []
    actual, count = dispatch.piece_features(piece, check_budget=lambda: checks.append(True))
    expected = {task: {row.candidate_id: local_features(piece, row)
                       for row in raw_candidates(piece, TASK_LEVELS[task])} for task in dispatch.TASKS}
    assert actual == expected
    windows = {(row.start, row.end) for level in set(TASK_LEVELS.values()) for row in raw_candidates(piece, level)}
    assert count == len(checks) == len(windows)
    assert "note:endpoint" not in actual["metrical_strength"]


def test_sweep_budget_interrupts_before_evaluating_windows(monkeypatch):
    piece = replace(build_context_pair_suite()[0].left, split="train")
    monkeypatch.setattr(dispatch, "local_features", lambda *args: pytest.fail("budget first"))
    def fail(): raise RuntimeError("injected wall boundary")
    with pytest.raises(RuntimeError, match="wall boundary"):
        dispatch.piece_features(piece, check_budget=fail)


def test_config_accepts_exact_fixed_protocol_without_cache_or_checkpoint_reads(tmp_path):
    config = _config()
    assert dispatch.load_config(Path(_write(tmp_path / "config.json", config).path)) == config


@pytest.mark.parametrize("field,value,reason", [
    ("tasks", ["quality"], "fixed_protocol"), ("seed", True, "fixed_protocol"),
    ("seed", 18, "fixed_protocol"), ("bootstrap_draws", 1, "fixed_protocol"),
    ("device", "cuda:1", "fixed_protocol"), ("experiment_id", "EXP-OTHER-001", "local_experiment_id"),
    ("experiment_id", "EXP-CTX-C2-LOCAL-../escape", "local_experiment_id"),
    ("schema_version", "unknown", "config_schema"),
])
def test_config_rejects_drift(tmp_path, field, value, reason):
    config = _config(); config[field] = value
    path = Path(_write(tmp_path / "config.json", config).path)
    with pytest.raises(ValueError, match=reason): dispatch.load_config(path)


@pytest.mark.parametrize("field,key,value", [("budget", "max_forwards", 31),
    ("budget", "max_forwards", 32.0), ("checkpoint", "sha256", "b" * 64)])
def test_fixed_budget_and_checkpoint_have_exact_json_identity(tmp_path, field, key, value):
    config = _config(); config[field][key] = value
    with pytest.raises(ValueError, match="fixed_protocol"):
        dispatch.load_config(Path(_write(tmp_path / "config.json", config).path))


def test_config_duplicate_json_and_oversize_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(ValueError, match="duplicate_json_key"): dispatch.load_config(path)
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="config_budget"): dispatch.load_config(path)


def test_preflight_binds_complete_index_without_opening_raw_targets_or_checkpoint(tmp_path, monkeypatch):
    env = _environment(tmp_path, monkeypatch)
    native = FileBinding.read
    def read(binding, **kwargs):
        assert env.cache not in Path(binding.path).parents, "preflight opened cache payload"
        return native(binding, **kwargs)
    monkeypatch.setattr(FileBinding, "read", read)
    monkeypatch.setattr(dispatch, "FrozenC2Reference", lambda *a, **kw: pytest.fail("preflight loaded checkpoint"))
    protocol, metadata, exports = dispatch.preflight(env.config, input_root=env.inputs, cache_root=env.cache)
    assert len(exports) == 3 and all(export.member.split != "test" for export in exports)
    assert metadata is env.metadata
    assert protocol["selected_exports_fingerprint"] == fingerprint(dispatch.export_identity(exports))
    assert protocol["protocol_fingerprint"] == fingerprint({k: v for k, v in protocol.items() if k != "protocol_fingerprint"})
    assert all(env.cache in Path(export.raw.path).parents for export in exports)


@pytest.mark.parametrize("mutation,reason", [
    ("missing", "complete_TRAINVAL"), ("duplicate", "duplicate_or_unknown"),
    ("test", "cache_official_assignment"), ("component", "cache_official_assignment"),
    ("locator", "native_cache_locator_layout"), ("mixed_roots", "complete_TRAINVAL"),
])
def test_cache_index_gates_fail_before_payload_reads(tmp_path, monkeypatch, mutation, reason):
    env = _environment(tmp_path, monkeypatch)
    rows = deepcopy(env.rows)
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(rows[0])
    elif mutation == "test": rows[0]["split"] = "test"
    elif mutation == "component": rows[0]["source_component_id"] = "component:wrong"
    elif mutation == "locator": rows[0]["raw"]["path"] = "/unopened/server/cache/../piece.json"
    else: rows[0]["raw"]["path"] = rows[0]["raw"]["path"].replace("/server/", "/different-server/")
    _replace_index(env, rows)
    with pytest.raises(ValueError, match=reason):
        dispatch.preflight(env.config, input_root=env.inputs, cache_root=env.cache)


@pytest.mark.parametrize("mutation", ["index_digest", "audit_status", "audit_artifact", "audit_test_access"])
def test_index_requires_completed_matching_audit(tmp_path, monkeypatch, mutation):
    env = _environment(tmp_path, monkeypatch)
    if mutation == "index_digest":
        (env.inputs / "index.jsonl").write_bytes(b"{}\n")
    else:
        audit = json.loads((env.inputs / "audit.json").read_text())
        if mutation == "audit_status":
            audit.update(status="planned", validity={"valid": None, "reason": None})
        elif mutation == "audit_artifact": audit["artifacts"] = []
        else:
            audit["data_access"].update(test_inputs_accessed=True, test_targets_accessed=False,
                test_metrics_computed=False, test_used_for_selection=False, validation_accessed=False,
                access_evidence_status="verified_primary_evidence")
            audit["data_access"]["declared_policy"]["test_inputs_allowed"] = True
        bound = _write(env.inputs / "audit.json", seal_record(audit))
        env.config["binding_audit"]["record_binding"] = _declaration(bound, env.inputs)
    with pytest.raises(ValueError, match="file_binding_mismatch|cache_index_not_in_completed_audit|audit_TEST_access_incompatible"):
        dispatch.preflight(env.config, input_root=env.inputs, cache_root=env.cache)


def _execution_fixture(tmp_path, monkeypatch, checkpoint, *, masked_validation=False):
    binding, contract, state = checkpoint
    monkeypatch.setattr(dispatch, "BUDGET", {**dispatch.BUDGET, "train_pieces": 2,
                        "validation_pieces": 1, "max_forwards": 1})
    monkeypatch.setattr(dispatch, "CHECKPOINT", {"sha256": binding.sha256, "size_bytes": binding.size_bytes,
                        "model_contract_sha256": contract, "model_state_sha256": state})
    env = _environment(tmp_path, monkeypatch, masked_validation=masked_validation)
    protocol, _, exports = dispatch.preflight(env.config, input_root=env.inputs, cache_root=env.cache)
    record_path = tmp_path / (env.config["experiment_id"] + ".json")
    write_record_once(record_path, dispatch.make_planned_record(env.config, protocol))
    env.protocol, env.exports, env.record_path = protocol, exports, record_path
    env.checkpoint, env.output = Path(binding.path), tmp_path / "execution"
    return env


@pytest.fixture
def execution(tmp_path, monkeypatch, checkpoint):
    return _execution_fixture(tmp_path, monkeypatch, checkpoint)


def _execute(env):
    return dispatch.execute(env.config, env.protocol, env.exports, checkpoint=env.checkpoint,
                            record_path=env.record_path, output_root=env.output)


def test_synthetic_execution_preserves_raw_prediction_target_order_and_terminal_proposal(execution, monkeypatch):
    env = execution
    original_record = env.record_path.read_bytes()
    events = []
    raw_reader, target_reader, predict = dispatch.read_raw, dispatch.read_targets, FrozenC2Reference.predict
    def read_raw(export):
        events.append((export.member.split, "raw", export.record_id)); return raw_reader(export)
    def read_targets(export):
        if export.member.split == "validation": assert events[-1][1] == "predict"
        events.append((export.member.split, "target", export.record_id)); return target_reader(export)
    def forward(reference, piece, tasks):
        assert piece.split == "validation"
        result = predict(reference, piece, tasks)
        events.append((piece.split, "predict", piece.piece_id)); return result
    monkeypatch.setattr(dispatch, "read_raw", read_raw)
    monkeypatch.setattr(dispatch, "read_targets", read_targets)
    monkeypatch.setattr(FrozenC2Reference, "predict", forward)
    receipt = _execute(env)
    assert receipt["status"] == "completed"
    accounting = receipt["accounting"]
    assert accounting["raw_pieces_completed"] == accounting["target_pieces_completed"] == 3
    assert accounting["forwards_started"] == accounting["forwards_completed"] == 1
    assert accounting["forwards_completion_unknown"] == accounting["applied_updates"] == 0
    assert accounting["local_fits_completed"] == 6
    assert [kind for split, kind, _ in events if split == "validation"] == ["raw", "predict", "target"]
    assert all(split == "train" for split, _, _ in events[:4])
    proposal = load_record(env.output / env.record_path.name)
    assert proposal["status"] == "completed" and proposal["validity"]["valid"] is True
    assert proposal["data_access"]["validation_accessed"] is True
    assert proposal["data_access"]["test_inputs_accessed"] is False
    assert env.record_path.read_bytes() == original_record
    report = json.loads((env.output / "result.json").read_text())
    assert len(report["tasks"]) == 6
    assert all(row["status"] == "estimated" for row in report["tasks"].values())
    assert all(obs["gold"] == obs["local"] == 0 for row in report["tasks"].values() for obs in row["observations"])
    with pytest.raises(FileExistsError): _execute(env)


@pytest.mark.parametrize("mutation,reason", [("record", "matching_planned"), ("config", "selected_protocol"),
    ("protocol", "selected_protocol"), ("duplicate", "selected_protocol"), ("cross_component", "selected_protocol")])
def test_execution_binding_failure_precedes_payload_and_output(execution, monkeypatch, mutation, reason):
    env = execution
    if mutation == "record":
        record = json.loads(env.record_path.read_text()); record["protocol"]["context_local_fingerprint"] = "0" * 64
        env.record_path.write_bytes(canonical_json_bytes(seal_record(record)) + b"\n")
    elif mutation == "config": env.config["seed"] = 18
    elif mutation == "protocol": env.protocol["claim_limit"] = "unbound drift"
    elif mutation == "duplicate": env.exports = (env.exports[0], env.exports[0], env.exports[2])
    else:
        env.exports = tuple(replace(e, member=replace(e.member, component_id=env.exports[0].member.component_id))
                            if e.member.split == "validation" else e for e in env.exports)
    monkeypatch.setattr(dispatch, "read_raw", lambda *a: pytest.fail("binding failed before payload"))
    with pytest.raises(ValueError, match=reason): _execute(env)
    assert not env.output.exists()


@pytest.mark.parametrize("interruption", [RuntimeError("injected forward"), KeyboardInterrupt()])
def test_forward_failure_retains_started_unknown_and_unread_validation_targets(execution, monkeypatch, interruption):
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    def fail(self, batch): raise interruption
    monkeypatch.setattr(CorrectedAnalysisGNNModel, "predict", fail)
    receipt = _execute(execution)
    assert receipt["status"] == ("aborted" if isinstance(interruption, KeyboardInterrupt) else "failed")
    assert receipt["accounting"]["forwards_started"] == receipt["accounting"]["forwards_completion_unknown"] == 1
    assert receipt["accounting"]["forwards_completed"] == 0
    assert receipt["accounting"]["validation_raw_started"] == 1
    assert receipt["accounting"]["validation_targets_started"] == 0
    proposal = load_record(execution.output / execution.record_path.name)
    assert proposal["status"] == receipt["status"] and proposal["validity"]["valid"] is False
    assert proposal["results"]["partial_artifacts_are_not_comparison_evidence"] is True
    assert not (execution.output / "result.json").exists()


@pytest.mark.parametrize("stage", ["result_write", "journal_after_result"])
def test_post_comparison_io_failure_cannot_promote_terminal_comparison(execution, monkeypatch, stage):
    if stage == "result_write":
        native = dispatch._write_json
        def fail(path, payload):
            if path.name == "result.json":
                path.write_bytes(b"partial")
                raise OSError("injected result write")
            return native(path, payload)
        monkeypatch.setattr(dispatch, "_write_json", fail)
    else:
        native = dispatch.Progress.emit
        def fail(progress, event, detail=None):
            if event == "result_written": raise OSError("injected journal write")
            return native(progress, event, detail)
        monkeypatch.setattr(dispatch.Progress, "emit", fail)
    receipt = _execute(execution)
    assert receipt["status"] == "failed"
    assert receipt["accounting"]["forwards_completed"] == 1
    assert (execution.output / "result.json").exists()
    proposal = load_record(execution.output / execution.record_path.name)
    assert proposal["status"] == "failed" and proposal["validity"]["valid"] is False
    assert proposal["results"]["partial_artifacts_are_not_comparison_evidence"] is True
    assert not any("comparison completed" in claim for claim in proposal["claims"]["supported"])


def test_atomic_json_failure_never_publishes_partial_destination(tmp_path):
    output = tmp_path / "new.json"
    with pytest.raises(ValueError): dispatch._write_json(output, {"partial": [1, float("nan")]})
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_json_never_overwrites_existing_evidence(tmp_path):
    output = tmp_path / "immutable.json"
    output.write_bytes(b"retained previous attempt")
    with pytest.raises(FileExistsError): dispatch._write_json(output, {"new": "attempt"})
    assert output.read_bytes() == b"retained previous attempt"
    assert list(tmp_path.iterdir()) == [output]


def test_all_masked_validation_is_not_estimable_with_trained_local_model(tmp_path, monkeypatch, checkpoint):
    env = _execution_fixture(tmp_path, monkeypatch, checkpoint, masked_validation=True)
    receipt = _execute(env)
    assert receipt["status"] == "completed"
    assert receipt["accounting"]["local_fits_completed"] == 6
    report = json.loads((env.output / "result.json").read_text())
    for task in report["tasks"].values():
        assert task["status"] == "not_estimable"
        assert task["reason"] == "no_eligible_VALIDATION_observations"
        assert task["local_model"] is not None
        assert task["observations"] == []
        assert task["coverage"]["validation:source_observations"] > 0
        assert task["comparison"]["point"]["local"]["accuracy"] is None
        assert task["comparison"]["point"]["reference"]["accuracy"] is None
