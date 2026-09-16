from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.adapters import dilemmadata as adapter
from music_critic.data import dumps_piece
from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import materialize_target_sidecar
from music_critic.ssl import group_masking_scientific_cached_data as subject
from music_critic.ssl.group_masking_pilot import GroupPilotConfig, bind_train_batch
from tests.ssl.test_group_masking_scientific_data import make_disk_request, _file, _jsonl


def make_cached_request(tmp_path, monkeypatch, *, single_validation=False, single_train=False):
    """Three tiny authentic-format cache entries; the TEST entry has metadata only."""
    native = make_disk_request(tmp_path, monkeypatch)
    payload = native.to_dict()
    paths = reference.ProductionArtifactPaths(
        corpus_root=Path(payload["source_root"]),
        b2_source_inventory=Path(payload["metadata"]["source_inventory"]["path"]),
        b3_root=Path(payload["metadata"]["split_assignments"]["path"]).parent)
    assignments = reference.frozen_split_assignments(paths)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for record_id, row in assignments.items():
        if row["split"] == "test":
            continue
        record = reference._resolve_selected_corpus_record(record_id, split=row["split"], paths=paths)
        accepted = adapter.convert_dilemmadata_record(record)
        assert isinstance(accepted, adapter.DilemmadataAccepted)
        sidecar = materialize_target_sidecar(accepted)
        piece = accepted.piece
        if ((single_validation and row["split"] == "validation")
                or (single_train and record_id == "an:training:same")):
            piece = replace(piece, notes=piece.notes[:1])
        directory = cache / sha256(record_id.encode()).hexdigest()[:24]
        directory.mkdir()
        raw_path, sidecar_path, metadata_path = (directory / filename for filename in (
            "piece.json", "target_sidecar.json", "metadata.json"))
        raw_path.write_text(dumps_piece(piece))
        sidecar_path.write_text(json.dumps(sidecar, sort_keys=True))
        metadata_path.write_text(json.dumps({"record_id": record_id, "split": row["split"],
            "piece_sha256": _file(raw_path)["sha256"], "sidecar_sha256": _file(sidecar_path)["sha256"]}))
        rows.append({"record_id": record_id, "split": row["split"],
                     "source_component_id": row["source_component_id"],
                     "raw": _file(raw_path), "sidecar": _file(sidecar_path), "metadata": _file(metadata_path)})
    index = tmp_path / "cache-bindings.jsonl"
    _jsonl(index, rows)
    components = tmp_path / "source_components.jsonl"
    _jsonl(components, [{"source_component_id": row["source_component_id"], "record_ids": [record]}
                        for record, row in assignments.items()])
    return subject.ScientificDataRequest({
        "schema": subject.REQUEST_SCHEMA, "cache_root": str(cache), "cache_index": _file(index),
        "split_assignments": payload["metadata"]["split_assignments"],
        "source_components": _file(components), "class_weights": payload["metadata"]["class_weights"],
        "validation_limits": payload["validation_limits"]})


@pytest.fixture
def configs():
    pilot = GroupPilotConfig(updates=1)
    return pilot, replace(pilot, lambda_ssl=0, lambda_theory=1)


@pytest.fixture
def data_request(tmp_path, monkeypatch):
    return make_cached_request(tmp_path, monkeypatch)


def _mutate_index(data_request, transform):
    payload = data_request.to_dict()
    path = Path(payload["cache_index"]["path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    _jsonl(path, rows)
    payload["cache_index"] = _file(path)
    return subject.ScientificDataRequest(payload)


def test_cached_prepare_exact_replay_and_full_validation(data_request, configs, monkeypatch):
    monkeypatch.setattr(reference, "_resolve_selected_corpus_record", lambda *a, **kw: pytest.fail("raw conversion forbidden"))
    receipts = []
    data = subject.prepare_scientific_data(data_request, *configs, on_receipt=receipts.append)
    assert data.access_observations["source_conversion_calls"] == data.access_observations["test_source_reads"] == 0
    assert data.access_observations["raw_files_verified"] == 9  # 2 eligibility + 4 pilot + 2 recovery + 1 validation
    assert data.access_observations["sidecar_files_verified"] == 5
    assert len(data.validation_membership) == len(data.validation_bindings) == 1
    assert all(row["eligible"] for row in data.binding["eligibility"].values())
    assert data.binding["preflight"]["compiler_policy"] == subject.COMPILER_POLICY
    assert data.binding["validation_policy_availability"][0]["beat_pitch_descendants"]["available"] is True
    # The first two completed eligibility records did not read a target payload.
    completed = [row for row in receipts if row["status"] == "completed"]
    assert completed[1]["access_observations"]["sidecar_files_verified"] == 0
    restored = subject.load_bound_scientific_data(data_request, data.binding, *configs)
    for stage, config in zip(("pilot", "recovery"), configs):
        batch = restored.load_update(stage, 0)
        assert bind_train_batch(batch, config) == data.schedule_bindings(stage)[0]
    assert len(list(restored.validation_batches())) == 1


def test_preflight_only_reads_index_component_split_weights(data_request, configs, monkeypatch):
    monkeypatch.setattr(subject.CacheFile, "read", lambda *a, **kw: pytest.fail("cache payload read"))
    report = subject.preflight_data(data_request, *configs)
    assert report["source_bytes_verified"] is False
    assert report["split_record_counts"] == {"train": 2, "validation": 1, "test": 1}


def test_train_structural_exclusion_is_shared_and_validation_not_filtered(tmp_path, monkeypatch, configs):
    request = make_cached_request(tmp_path, monkeypatch, single_validation=True, single_train=True)
    data = subject.prepare_scientific_data(request, *configs)
    assert data.binding["eligibility"]["an:training:same"] == {"eligible": False, "reason": "fewer_than_two_raw_notes"}
    selected = {record for schedule in data.binding["schedules"].values() for row in schedule for ids in row.values() for record in ids}
    assert selected == {"dlc:demo:same"}
    assert len(list(data.validation_batches())) == 1
    availability = data.binding["validation_policy_availability"][0]
    assert any(row["available"] is False for row in availability.values())


def test_singleton_validation_resource_limits_are_explicit(data_request, configs):
    payload = data_request.to_dict()
    payload["validation_limits"]["max_notes_per_piece"] = 1
    with pytest.raises(ValueError, match="record_resource_bound_notes"):
        subject.prepare_scientific_data(subject.ScientificDataRequest(payload), *configs)


def test_conservative_train_budget_exclusion_leaves_no_hidden_replacement(data_request, configs):
    pilot = replace(configs[0], max_notes_per_batch=2)
    recovery = replace(configs[1], max_notes_per_batch=2)
    with pytest.raises(ValueError, match="no_jointly_eligible_train_records"):
        subject.prepare_scientific_data(data_request, pilot, recovery)


@pytest.mark.parametrize("split", ["TEST", "test", "TeSt"])
def test_test_request_deny_before_read(tmp_path, split):
    with pytest.raises(ValueError, match="test_access_forbidden"):
        subject.load_data_request(tmp_path / "missing", split=split)


def test_test_record_never_reaches_payload_reader(data_request, configs, monkeypatch):
    data = subject.BoundScientificData(data_request, *configs)
    monkeypatch.setattr(subject.CacheFile, "read", lambda *a, **kw: pytest.fail("cache payload read"))
    with pytest.raises(ValueError, match="record_not_in_bound_membership"):
        data._record("synthetic:test", split="train", targets=True)
    with pytest.raises(ValueError, match="test_access_forbidden"):
        data._record("synthetic:test", split="test", targets=True)
    with pytest.raises(ValueError, match="test_access_forbidden"):
        data.validation_batches(split="test")


@pytest.mark.parametrize("mutation", ["missing_validation", "test_included", "wrong_split", "wrong_component"])
def test_cache_full_split_closure_before_payloads(data_request, configs, mutation):
    def change(rows):
        if mutation == "missing_validation":
            rows[:] = [row for row in rows if row["split"] == "train"]
        elif mutation == "test_included":
            rows.append({**rows[0], "record_id": "synthetic:test", "split": "test"})
        else:
            rows[0]["split" if mutation == "wrong_split" else "source_component_id"] = "invalid"
    updated = _mutate_index(data_request, change)
    with pytest.raises(ValueError, match="complete_train_validation_cache_required|cache_split_identity_mismatch"):
        subject.preflight_data(updated, *configs)


def test_whole_component_partition_is_required(data_request, configs):
    payload = data_request.to_dict()
    path = Path(payload["source_components"]["path"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["record_ids"].append("synthetic:test")
    rows[-1]["record_ids"] = ["synthetic:outside-paper"]
    _jsonl(path, rows)
    payload["source_components"] = _file(path)
    with pytest.raises(ValueError, match="complete_split_component_closure_mismatch"):
        subject.preflight_data(subject.ScientificDataRequest(payload), *configs)


def test_cache_paths_bound_to_record_token_and_root(data_request, configs):
    updated = _mutate_index(data_request, lambda rows: rows[0]["raw"].update({"path": "/private/tmp/outside.json"}))
    with pytest.raises(ValueError, match="cache_path_identity_or_confinement_mismatch"):
        subject.preflight_data(updated, *configs)


def test_large_sidecar_descriptor_is_allowed_with_explicit_128mib_limit(data_request, configs):
    # Metadata only: this does not allocate or create a 72 MB fixture.
    updated = _mutate_index(data_request, lambda rows: rows[0]["sidecar"].update({"size_bytes": 71_714_612}))
    assert subject.preflight_data(updated, *configs)["cache_file_limit_bytes"] == 128 * 1024 * 1024
    updated = _mutate_index(updated, lambda rows: rows[0]["sidecar"].update({"size_bytes": 128 * 1024 * 1024 + 1}))
    with pytest.raises(ValueError, match="invalid_cache_artifact_binding"):
        subject.preflight_data(updated, *configs)


def test_cache_byte_drift_fails_before_canonical_parser(data_request, configs, monkeypatch):
    data = subject.BoundScientificData(data_request, *configs)
    row = next(iter(data._cache.values()))
    path = Path(row["raw"]["path"])
    original = path.read_bytes()
    path.write_bytes(original.replace(b"60", b"61", 1))
    monkeypatch.setattr(subject, "loads_piece", lambda *a, **kw: pytest.fail("canonical parser reached"))
    with pytest.raises(ValueError, match="artifact_binding_mismatch"):
        data._record(row["record_id"], split=row["split"], targets=False)


def test_rehashed_false_deterministic_schedule_refused(data_request, configs):
    data = subject.prepare_scientific_data(data_request, *configs)
    value = data.binding
    value["schedules"]["recovery"][0]["theory"][0] = "synthetic:test"
    value["fingerprint"] = fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
    with pytest.raises(ValueError, match="prepared_deterministic_schedule_mismatch"):
        subject.load_bound_scientific_data(data_request, value, *configs)


def test_validation_membership_forgery_cannot_drop_hard_piece(data_request, configs):
    data = subject.prepare_scientific_data(data_request, *configs)
    value = data.binding
    value["validation_membership"] = []
    value["fingerprint"] = fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
    with pytest.raises(ValueError, match="prepared_validation_membership_invalid"):
        subject.load_bound_scientific_data(data_request, value, *configs)


def test_raw_eligibility_scan_never_reads_target_bytes(data_request, configs, monkeypatch):
    data = subject.BoundScientificData(data_request, *configs)
    original = data._read
    def raw_only(row, role):
        assert role != "sidecar"
        return original(row, role)
    monkeypatch.setattr(data, "_read", raw_only)
    for record in data.preflight["train_record_ids"]:
        data._record(record, split="train", targets=False)
    assert data.access_observations["sidecar_files_verified"] == 0


def test_raw_sizing_requires_no_target_weights_or_preselected_validation_limits(data_request, configs, monkeypatch):
    payload = data_request.to_dict()
    payload["validation_limits"] = payload["class_weights"] = None
    monkeypatch.setattr(reference, "load_frozen_class_weights", lambda *a, **kw: pytest.fail("class weights read"))
    original = subject.BoundScientificData._read
    def no_targets(self, row, role):
        assert role != "sidecar"
        return original(self, row, role)
    monkeypatch.setattr(subject.BoundScientificData, "_read", no_targets)
    request = subject.ScientificDataRequest(payload)
    result = subject.size_scientific_data(request, *configs)
    assert len(result["records"]) == 3
    assert result["access_observations"]["sidecar_files_verified"] == 0
    assert result["access_observations"]["raw_files_verified"] == 3
    assert result["max_counts_by_split"]["validation"]["notes"] == 2
    assert result["limits_not_selected_automatically"] is True
    with pytest.raises(ValueError, match="invalid_artifact_binding"):
        subject.prepare_scientific_data(request, *configs)


def test_rehashed_false_validation_availability_fails_against_raw(data_request, configs):
    data = subject.prepare_scientific_data(data_request, *configs)
    value = data.binding
    value["validation_policy_availability"][0]["independent_note_pitch"]["plan_fingerprint"] = "0" * 64
    value["fingerprint"] = fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
    restored = subject.load_bound_scientific_data(data_request, value, *configs)
    with pytest.raises(ValueError, match="validation_policy_availability_mismatch"):
        list(restored.validation_batches())
