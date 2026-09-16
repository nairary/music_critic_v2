from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from music_critic.adapters import dilemmadata as adapter
from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.ssl import group_masking_scientific_data as subject
from music_critic.ssl.group_masking_pilot import GroupPilotConfig, bind_train_batch
from tests.adapters.test_dilemmadata import CORPUS, _discover


def _file(path):
    payload = path.read_bytes()
    return {"path": str(path), "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def make_disk_request(tmp_path, monkeypatch):
    """Generated tiny metadata universe with the actual sequential native reader.

    Only the frozen split constant/count contract is replaced in this test.
    The original reference resolver, raw conversion, target materialization,
    graph builder and policy preparation run unchanged against copied fixtures.
    Artificial component identities are synthetic and never scientific evidence.
    """
    root = tmp_path / "corpus"
    shutil.copytree(CORPUS, root)
    validation_source = root / "pitch_arrays/AN/validation/same-alt_joint.tsv"
    validation_source.write_text(validation_source.read_text().replace("\t60\tTrue\tC\t0\t", "\t61\tTrue\tC\t1\t"))
    discovered = _discover(root).records
    records = []
    for index, row in enumerate(discovered):
        records.append(adapter._bind_record(replace(
            row, source_group_id=f"dilemmadata-component:fixture-{index}",
            lineage_group_id=f"dilemmadata-lineage:fixture-{index}",
            corpus_identity=adapter.DilemmadataCorpusIdentity(), record_binding_sha256="")))
    inventory, paper, assignments, sizes = [], [], [], {}
    for record in records:
        split = "validation" if record.record_id == "an:validation:same-alt" else "train"
        inventory.append({
            "record_id": record.record_id, "annotation_path": record.relative_path,
            "score_path": record.score_relative_path, "dialect": record.dialect,
            "collection": record.collection, "source_piece_id": record.piece_name,
            "source_resolution": record.source_resolution, "source_group_id": record.source_group_id,
            "source_fingerprints": {
                "physical_annotation_sha256": record.physical_source_sha256,
                "raw_projection_sha256": record.raw_projection_sha256,
                "grouping_sha256": record.grouping_fingerprint,
                "record_binding_sha256": record.record_binding_sha256,
                "score_sha256": record.score_sha256,
            }})
        paper.append({"record_id": record.record_id, "piece_id": record.piece_id,
                      "source_component_id": record.source_group_id,
                      "raw_equivalence_id": record.raw_equivalence_id})
        assignments.append({"record_id": record.record_id, "source_component_id": record.source_group_id,
                            "split": split})
        sizes[record.record_id] = {
            "annotation_size_bytes": record.path.stat().st_size,
            "score_size_bytes": None if record.score_path is None else record.score_path.stat().st_size}
    # The complete split includes TEST metadata; its nonexistent source must stay unopened.
    inventory.append({"record_id": "synthetic:test", "source_group_id": "component:test"})
    paper.append({"record_id": "synthetic:test", "source_component_id": "component:test",
                  "piece_id": "piece:test", "raw_equivalence_id": "raw:test"})
    assignments.append({"record_id": "synthetic:test", "source_component_id": "component:test", "split": "test"})
    b2, b3 = tmp_path / "b2", tmp_path / "b3"
    b2.mkdir(); b3.mkdir()
    _jsonl(b2 / "source_inventory.jsonl", inventory)
    _jsonl(b2 / "quarantine_records.jsonl", [])
    (b2 / "audit_summary.json").write_text(json.dumps({"snapshot": {
        "actual_path": str(root), "content_fingerprint": records[0].corpus_identity.content_fingerprint,
        "file_count": records[0].corpus_identity.installation_file_count}}))
    _jsonl(b3 / "paper_candidate_records.jsonl", paper)
    _jsonl(b3 / "split_assignments.jsonl", assignments)
    weights = tmp_path / "weights.json"
    shutil.copyfile(Path(__file__).parents[1] / "fixtures/analysisgnn/phase9eb5b_training_policy.json", weights)
    metadata = {
        "source_inventory": b2 / "source_inventory.jsonl", "quarantine_records": b2 / "quarantine_records.jsonl",
        "audit_summary": b2 / "audit_summary.json", "split_assignments": b3 / "split_assignments.jsonl",
        "paper_candidate_records": b3 / "paper_candidate_records.jsonl",
        "dlc_metadata": root / "processing/DLC/distant_listening_corpus.metadata.tsv", "class_weights": weights}
    payload = {"schema": subject.REQUEST_SCHEMA, "source_root": str(root),
               "validation_limits": {"max_notes_per_piece": 8192, "max_nodes_per_piece": 20000, "max_edges_per_piece": 200000},
               "metadata": {key: _file(path) for key, path in metadata.items()}, "source_sizes": sizes,
               "pilot_schedule": [{"ssl": ["an:training:same", "dlc:demo:same"],
                                   "theory": ["dlc:demo:same", "an:training:same"]}],
               "recovery_schedule": [{"theory": ["an:training:same", "dlc:demo:same"]}]}
    monkeypatch.setattr(reference, "CORRECTED_SPLIT_ASSIGNMENTS_SHA256", payload["metadata"]["split_assignments"]["sha256"])

    def fixture_assignments(paths):
        file = paths.b3_root / "split_assignments.jsonl"
        if sha256(file.read_bytes()).hexdigest() != reference.CORRECTED_SPLIT_ASSIGNMENTS_SHA256:
            raise ValueError("fixture_split_changed")
        return {row["record_id"]: row for row in map(json.loads, file.read_text().splitlines())}
    monkeypatch.setattr(reference, "frozen_split_assignments", fixture_assignments)
    return subject.ScientificDataRequest(payload)


@pytest.fixture
def configs():
    pilot = GroupPilotConfig(updates=1)
    return pilot, replace(pilot, lambda_ssl=0, lambda_theory=1)


@pytest.fixture
def data_request(tmp_path, monkeypatch):
    return make_disk_request(tmp_path, monkeypatch)


def test_reference_disk_prepare_frozen_replay_and_no_cache(data_request, configs):
    receipts = []
    data = subject.prepare_scientific_data(data_request, *configs, on_receipt=receipts.append)
    binding = data.binding
    assert set(binding["records"]) == set(data_request.payload["source_sizes"])
    assert len(data.validation_membership) == len(data.validation_bindings) == 1
    assert data.preflight["split_record_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert data.access_observations["raw_records_loaded"] == 8
    assert data.access_observations["target_sidecars_materialized"] == 5
    assert data.access_observations["test_source_reads"] == 0
    assert {row["status"] for row in receipts} == {"started", "completed"}
    restored = subject.load_bound_scientific_data(data_request, binding, *configs)
    pilot = restored.load_update("pilot", 0)
    assert bind_train_batch(pilot, configs[0]) == data.schedule_bindings("pilot")[0]
    recovery = restored.load_update("recovery", 0)
    assert recovery.ssl is None
    assert bind_train_batch(recovery, configs[1]) == data.schedule_bindings("recovery")[0]
    assert len(list(restored.validation_batches())) == 1
    assert not list(Path(data_request.payload["source_root"]).parent.rglob("piece.json"))
    # A returned JSON copy cannot silently alter this provider's binding.
    binding["schedule_bindings"]["pilot"][0] = "0" * 64
    assert restored.schedule_bindings("pilot")[0] != "0" * 64


def test_preflight_opens_only_metadata(data_request, configs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("source resolver accessed by metadata preflight")
    monkeypatch.setattr(reference, "_resolve_selected_corpus_record", forbidden)
    report = subject.preflight_data(data_request, *configs)
    assert report["source_bytes_verified"] is False
    assert report["test_source_access"] is False


@pytest.mark.parametrize("split", ["test", "TEST", "TeSt"])
def test_test_deny_before_even_request_read(tmp_path, split):
    with pytest.raises(ValueError, match="test_access_forbidden"):
        subject.load_data_request(tmp_path / "nonexistent", split=split)


def test_provider_test_deny_before_reader(data_request, configs, monkeypatch):
    data = subject.BoundScientificData(data_request, *configs)
    monkeypatch.setattr(data, "_record", lambda *args, **kwargs: pytest.fail("source read"))
    with pytest.raises(ValueError, match="test_access_forbidden"):
        data.load_update("pilot", 0, split="test")
    with pytest.raises(ValueError, match="test_access_forbidden"):
        data.validation_batches(split="test")


def _rewrite_metadata(data_request, name, transform, monkeypatch=None):
    payload = data_request.to_dict()
    path = Path(payload["metadata"][name]["path"])
    rows = list(map(json.loads, path.read_text().splitlines()))
    transform(rows)
    _jsonl(path, rows)
    payload["metadata"][name] = _file(path)
    if name == "split_assignments" and monkeypatch is not None:
        monkeypatch.setattr(reference, "CORRECTED_SPLIT_ASSIGNMENTS_SHA256", payload["metadata"][name]["sha256"])
    return subject.ScientificDataRequest(payload)


@pytest.mark.parametrize("field", ["ssl", "theory"])
def test_schedule_validation_or_test_refused_before_source(data_request, configs, field):
    payload = data_request.to_dict()
    payload["pilot_schedule"][0][field][0] = "synthetic:test"
    with pytest.raises(ValueError, match="scheduled_record_not_train"):
        subject.preflight_data(subject.ScientificDataRequest(payload), *configs)


def test_entire_component_cross_split_is_checked(data_request, configs, monkeypatch):
    payload = data_request.to_dict()
    # Join the TEST row to a TRAIN component coherently in all three metadata files.
    train_component = "dilemmadata-component:fixture-0"
    for name, field in (("split_assignments", "source_component_id"),
                        ("paper_candidate_records", "source_component_id"),
                        ("source_inventory", "source_group_id")):
        data_request = _rewrite_metadata(subject.ScientificDataRequest(payload), name,
            lambda rows: next(row for row in rows if row["record_id"] == "synthetic:test").update({field: train_component}), monkeypatch)
        payload = data_request.to_dict()
    with pytest.raises(ValueError, match="component_cross_split_leakage"):
        subject.preflight_data(data_request, *configs)


@pytest.mark.parametrize("relative", ["../escape.tsv", "/absolute.tsv", "pitch_arrays/../../x", "x\\y.tsv"])
def test_path_confinement_before_parser(data_request, configs, relative):
    modified = _rewrite_metadata(data_request, "source_inventory",
        lambda rows: rows[0].update({"annotation_path": relative}))
    with pytest.raises(ValueError, match="source_path_invalid"):
        subject.preflight_data(modified, *configs)


def test_symlink_escape_rejected_at_preflight(data_request, configs, tmp_path):
    root = Path(data_request.payload["source_root"])
    (root / "escape").symlink_to(tmp_path)
    modified = _rewrite_metadata(data_request, "source_inventory",
        lambda rows: rows[0].update({"annotation_path": "escape/secret.tsv"}))
    with pytest.raises(ValueError, match="source_path_escapes_root"):
        subject.preflight_data(modified, *configs)


@pytest.mark.parametrize("size", [True, 0, -1, subject.MAX_SOURCE_BYTES + 1])
def test_source_size_bounds_fail_closed(data_request, configs, size):
    payload = data_request.to_dict()
    payload["source_sizes"]["an:training:same"]["annotation_size_bytes"] = size
    with pytest.raises(ValueError, match="source_size_binding_invalid"):
        subject.preflight_data(subject.ScientificDataRequest(payload), *configs)


def test_full_validation_cannot_be_silently_removed(data_request, configs):
    payload = data_request.to_dict()
    payload["source_sizes"].pop("an:validation:same-alt")
    with pytest.raises(ValueError, match="selected_source_sizes_incomplete"):
        subject.preflight_data(subject.ScientificDataRequest(payload), *configs)


def test_source_drift_rejected_before_reference_parser(data_request, configs, monkeypatch):
    data = subject.BoundScientificData(data_request, *configs)
    source = data._source_files("an:training:same")[0].path
    text = source.read_bytes()
    source.write_bytes(text.replace(b"60", b"61", 1))
    monkeypatch.setattr(reference, "_resolve_selected_corpus_record", lambda *args, **kwargs: pytest.fail("parser reached"))
    with pytest.raises(ValueError, match="artifact_binding_mismatch"):
        data.load_update("pilot", 0)


def test_changed_metadata_refused_during_execution(data_request, configs):
    data = subject.BoundScientificData(data_request, *configs)
    path = Path(data_request.payload["metadata"]["paper_candidate_records"]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact_size_mismatch"):
        data.load_update("pilot", 0)


def test_rehashed_forged_schedule_fails_on_actual_batch(data_request, configs):
    prepared = subject.prepare_scientific_data(data_request, *configs)
    value = prepared.binding
    value["schedule_bindings"]["pilot"][0] = "0" * 64
    value["fingerprint"] = fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
    restored = subject.load_bound_scientific_data(data_request, value, *configs)
    with pytest.raises(ValueError, match="update_content_binding_mismatch"):
        restored.load_update("pilot", 0)


def test_actual_both_policy_failure_has_no_replacement(data_request, configs, monkeypatch):
    records = []
    original = subject.BoundScientificData._record
    def record(self, record_id, **kwargs):
        records.append(record_id)
        return original(self, record_id, **kwargs)
    monkeypatch.setattr(subject.BoundScientificData, "_record", record)
    def unavailable(*args, **kwargs):
        raise ValueError("actual_beat_policy_unavailable")
    monkeypatch.setattr(subject, "paired_mask_bindings", unavailable)
    with pytest.raises(ValueError, match="actual_beat_policy_unavailable"):
        subject.prepare_scientific_data(data_request, *configs)
    assert records == ["an:training:same", "dlc:demo:same", "dlc:demo:same", "an:training:same"]


def test_duplicate_json_keys_rejected_before_metadata(tmp_path):
    path = tmp_path / "data_request.json"
    path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="duplicate_json_key"):
        subject.load_data_request(path)
