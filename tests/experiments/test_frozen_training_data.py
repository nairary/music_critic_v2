"""Synthetic files only: strict Dilemma metadata/payload preparation boundary."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from music_critic.data import dumps_piece, load_piece
from music_critic.experiments.analysisgnn import frozen_training_data as data
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.full_orbit_training import build_full_orbit_table, full_orbit_table_contract
from music_critic.tasks import collate_multisource_samples, prepare_multisource_sample


ROOT = Path(__file__).resolve().parents[2]


def _write(path, value):
    payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _jsonl(rows):
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def _record(root, rid, split, component):
    piece = replace(load_piece(ROOT / "tests/fixtures/data/canonical_piece_v2.json"),
                    dataset_name="dilemmadata", source_group_id=component,
                    piece_id=data._piece_id(rid, "dlc"), split=None, annotations=(), targets=(), quality_flags=())
    sidecar = {"schema_version": data.TARGET_SIDECAR_VERSION, "record_id": rid,
               "source_component_id": component, "dialect": "dlc", "entities": [],
               "relations": [], "entity_counts": {}, "relation_counts": {},
               "repair_evidence_fingerprint": None}
    sidecar["fingerprint"] = fingerprint(sidecar)
    row = {"record_id": rid, "split": split, "source_component_id": component,
           "piece": _write(root / "piece.json", dumps_piece(piece).encode()),
           "sidecar": _write(root / "sidecar.json", sidecar)}
    row["metadata"] = _write(root / "metadata.json", {"record_id": rid, "split": split,
        "piece_sha256": row["piece"]["sha256"], "sidecar_sha256": row["sidecar"]["sha256"]})
    return row, piece, sidecar


@pytest.fixture
def source(tmp_path, monkeypatch):
    assignments, eligibility, records, components = [], [], [], {}
    for split, count, component_count in (("train", 1295, 1209), ("validation", 162, 147), ("test", 162, 151)):
        for i in range(count):
            rid, component = f"synthetic:{split}:{i:04}", f"component:{split}:{i % component_count:04}"
            assignments.append({"record_id": rid, "split": split, "source_component_id": component})
            if split == "train":
                components.setdefault(component, []).append(rid)
                for shift in range(12):
                    eligibility.append({"record_id": rid, "source_component_id": component,
                        "shift_pc": shift, "signed_semitones": (shift if shift <= 6 else shift - 12),
                        "dialect": "dlc", "corrected_valid": True, "graph_range_valid": True,
                        "all_targets_closed": True, "round_trip_valid": True, "corrected_invalid_reasons": []})
            if split == "validation" or split == "train" and i == 0:
                records.append(_record(tmp_path / "exports" / rid, rid, split, component)[0])
    weights_path = ROOT / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json"
    tuples = [{"split": "train", "mode": "corrected_harmonic_event",
               "tuple": {task: f"native:{task}" for task in data.PAPER_DEFINED_JOINT_COMPONENTS}},
              {"split": "validation", "mode": "corrected_harmonic_event",
               "tuple": {task: f"validation:{task}" for task in data.PAPER_DEFINED_JOINT_COMPONENTS}}]
    manifest = {"schema_version": data.DATA_SCHEMA,
        "split_assignments": _write(tmp_path / "split_assignments.jsonl", _jsonl(assignments)),
        "shift_eligibility": _write(tmp_path / "eligibility.jsonl", _jsonl(eligibility)),
        "class_weights": _write(tmp_path / "weights.json", weights_path.read_bytes()),
        "joint_tuple_counts": _write(tmp_path / "tuples.jsonl", _jsonl(tuples)), "records": records}
    monkeypatch.setattr(data.reference, "CORRECTED_SPLIT_ASSIGNMENTS_SHA256", manifest["split_assignments"]["sha256"])
    monkeypatch.setattr(data, "FROZEN_JOINT_TUPLE_SHA256", manifest["joint_tuple_counts"]["sha256"])
    table = build_full_orbit_table(components, {rid: tuple(range(12)) for ids in components.values() for rid in ids})
    monkeypatch.setattr(data, "FROZEN_ORBIT_TABLE_FINGERPRINT", full_orbit_table_contract(table)["table_fingerprint"])
    return manifest


def _deny(*args, **kwargs):
    pytest.fail("forbidden file or corpus operation")


def _replace_file(row, key, payload, root):
    row[key] = _write(root / f"changed-{key}.json", payload)
    if key in ("piece", "sidecar"):
        row["metadata"] = _write(root / "changed-metadata.json", {"record_id": row["record_id"],
            "split": row["split"], "piece_sha256": row["piece"]["sha256"],
            "sidecar_sha256": row["sidecar"]["sha256"]})


def test_production_pins_match_authoritative_tracked_contracts():
    # Constants themselves are checked outside the fixture's synthetic override.
    b5a = json.loads((ROOT / "tests/fixtures/analysisgnn/phase9eb5a_transposition_audit.json").read_text())
    b5b = json.loads((ROOT / "tests/fixtures/analysisgnn/phase9eb5b_training_policy.json").read_text())
    b5h = json.loads((ROOT / "tests/fixtures/analysisgnn/phase9eb5h_full_orbit_profile.json").read_text())
    assert data.FROZEN_JOINT_TUPLE_SHA256 == b5a["input_fingerprints"]["b4_artifact_sha256"]["joint_tuple_counts.jsonl"]
    assert data.FROZEN_CLASS_WEIGHT_FINGERPRINT == b5b["class_weight_payload"]["fingerprint"]
    assert data.FROZEN_ORBIT_TABLE_FINGERPRINT == b5h["orbit_table"]["table_fingerprint"]


def test_metadata_preflight_has_no_cache_payload_read_and_native_train_tuples_only(source, monkeypatch):
    original = data.read_bound_bytes
    calls = []
    def read(binding):
        assert "/exports/" not in binding["path"]
        calls.append(binding["path"])
        return original(binding)
    monkeypatch.setattr(data, "read_bound_bytes", read)
    monkeypatch.setattr(data.reference, "load_production_record", _deny)
    loader = data.FrozenTrainingData(source)
    assert len(loader.assignments) == 1619
    assert len(loader.validation_record_ids) == 162
    assert len(loader.table) == 15540  # Explicit synthetic full12 table.
    assert loader.train_seen_tuples == {tuple(f"native:{task}" for task in data.PAPER_DEFINED_JOINT_COMPONENTS)}
    assert len(calls) == 5  # Four metadata files and post-reference weight rehash.


@pytest.mark.parametrize("split", ["train", "validation"])
def test_reader_exactly_reuses_reference_raw_preparation_and_preserves_masks(source, split):
    loader = data.FrozenTrainingData(source)
    row = next(item for item in source["records"] if item["split"] == split)
    piece = load_piece(row["piece"]["path"])
    expected = collate_multisource_samples((prepare_multisource_sample(piece),)).raw_graph_batch
    raw, sidecar = loader.reader(row["record_id"], split=split)
    assert raw.to_dict().keys() == expected.to_dict().keys()
    for node_type in raw.node_types:
        for field, value in raw[node_type].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, expected[node_type][field])
    assert sidecar == json.loads(Path(row["sidecar"]["path"]).read_text())
    assert piece.split is None
    sidecar["entities"].append({"masked": False})
    raw["note"].x_cat.fill_(0)
    second_raw, second_sidecar = loader.reader(row["record_id"], split=split)
    assert torch.equal(second_raw["note"].x_cat, expected["note"].x_cat)
    assert second_sidecar["entities"] == []


@pytest.mark.parametrize("record_id,split", [("synthetic:test:0000", "test"), ("synthetic:test:0000", "train"),
    ("synthetic:validation:0000", "train"), ("synthetic:train:0001", "train"), ("missing", "validation")])
def test_record_and_split_rejection_precedes_all_io(source, monkeypatch, record_id, split):
    loader = data.FrozenTrainingData(source)
    monkeypatch.setattr(data, "read_bound_bytes", _deny)
    monkeypatch.setattr(data, "_identity", _deny)
    with pytest.raises(ValueError, match="fusion_data"):
        loader.reader(record_id, split=split)


def test_declared_test_payload_and_forbidden_allowed_splits_fail_before_metadata(source, monkeypatch):
    monkeypatch.setattr(data, "read_bound_bytes", _deny)
    with pytest.raises(ValueError, match="train_validation_only"):
        data.FrozenTrainingData(source, allowed_splits=("test",))
    source["records"][0]["split"] = "test"
    with pytest.raises(ValueError, match="record_split_forbidden"):
        data.FrozenTrainingData(source)


def test_missing_validation_and_scheduled_train_exports_fail_closed(source):
    loader = data.FrozenTrainingData(source)
    assert loader.require_train_records(["synthetic:train:0000"] * 2) == ("synthetic:train:0000",)
    with pytest.raises(ValueError, match="warm_export_not_declared"):
        loader.require_train_records(["synthetic:train:0001"])
    source["records"].pop()
    with pytest.raises(ValueError, match="complete_validation_required"):
        data.FrozenTrainingData(source)


def test_manifest_identity_is_portable_and_callers_cannot_mutate_internal_membership(source, tmp_path):
    first = data.FrozenTrainingData(source)
    moved = deepcopy(source)
    root = tmp_path / "moved"
    for key in data._METADATA:
        binding = moved[key]
        binding["path"] = _write(root / key, Path(binding["path"]).read_bytes())["path"]
    for row in moved["records"]:
        for key in data._FILES:
            row[key]["path"] = str(root / "unread-exports" / row["record_id"] / key)
    moved["records"].reverse()
    second = data.FrozenTrainingData(moved)
    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert first.bindings == second.bindings
    exported = first.assignments
    exported["synthetic:test:0000"]["split"] = "train"
    assert first.assignments["synthetic:test:0000"]["split"] == "test"
    source["records"][0]["split"] = "test"
    assert first.require_train_records(["synthetic:train:0000"])


@pytest.mark.parametrize("key", ["split_assignments", "joint_tuple_counts"])
def test_caller_hash_does_not_override_frozen_source_pin(source, key):
    source[key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen_.*hash_mismatch"):
        data.FrozenTrainingData(source)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("size_bytes", 1), ("size_bytes", True),
                                       ("path", "relative/piece.json")])
def test_bad_payload_pin_fails_before_decode(source, monkeypatch, field, value):
    source["records"][0]["piece"][field] = value
    monkeypatch.setattr(data, "loads_piece", _deny)
    with pytest.raises(ValueError, match="fusion_data"):
        loader = data.FrozenTrainingData(source)
        loader.reader("synthetic:train:0000", split="train")


@pytest.mark.parametrize("change", ["metadata", "sidecar_id", "sidecar_component", "sidecar_fingerprint", "piece_split", "piece_group", "piece_id"])
def test_self_consistent_wrong_record_payload_rejected(source, tmp_path, change):
    row = source["records"][0]
    if change == "metadata":
        payload = json.loads(Path(row["metadata"]["path"]).read_text()); payload["split"] = "validation"
        _replace_file(row, "metadata", payload, tmp_path)
    elif change.startswith("sidecar"):
        payload = json.loads(Path(row["sidecar"]["path"]).read_text())
        field = {"sidecar_id": "record_id", "sidecar_component": "source_component_id", "sidecar_fingerprint": "fingerprint"}[change]
        payload[field] = "wrong"
        _replace_file(row, "sidecar", payload, tmp_path)
    else:
        piece = load_piece(row["piece"]["path"])
        field = {"piece_split": "split", "piece_group": "source_group_id", "piece_id": "piece_id"}[change]
        value = {"piece_split": "test", "piece_group": "wrong", "piece_id": "piece:other"}[change]
        piece = replace(piece, **{field: value})
        _replace_file(row, "piece", dumps_piece(piece).encode(), tmp_path)
    loader = data.FrozenTrainingData(source)
    with pytest.raises(ValueError, match="fusion_data"):
        loader.reader(row["record_id"], split="train")


def test_full_binding_verification_reads_only_declared_exports_without_graphs(source, monkeypatch):
    loader = data.FrozenTrainingData(source)
    monkeypatch.setattr(data, "loads_piece", _deny)
    report = loader.verify_bindings()
    assert report["verified_records"] == 163 and report["record_files_verified"] == 489
    assert report["test_payloads_accessed"] is False and report["graph_builds"] == 0
    Path(source["records"][0]["piece"]["path"]).write_text("modified")
    with pytest.raises(ValueError, match="artifact_size_mismatch"):
        loader.verify_bindings()


def test_metadata_drift_after_construction_blocks_before_graph_decode(source, monkeypatch):
    loader = data.FrozenTrainingData(source)
    Path(source["shift_eligibility"]["path"]).write_bytes(b"changed")
    monkeypatch.setattr(data, "loads_piece", _deny)
    with pytest.raises(ValueError, match="metadata_changed"):
        loader.reader("synthetic:train:0000", split="train")


def test_missing_warm_file_never_invokes_corpus_conversion(source, monkeypatch):
    loader = data.FrozenTrainingData(source)
    Path(source["records"][0]["piece"]["path"]).unlink()
    monkeypatch.setattr(data.reference, "load_production_record", _deny)
    with pytest.raises(ValueError, match="artifact_unavailable"):
        loader.reader("synthetic:train:0000", split="train")


def test_symlink_payload_rejected(source, tmp_path):
    row = source["records"][0]
    link = tmp_path / "link.json"
    link.symlink_to(row["piece"]["path"])
    row["piece"]["path"] = str(link)
    loader = data.FrozenTrainingData(source)
    with pytest.raises(ValueError, match="symlink_forbidden"):
        loader.reader(row["record_id"], split="train")


def test_duplicate_json_keys_rejected_even_with_matching_hash(source, tmp_path):
    row = source["records"][0]
    _replace_file(row, "metadata", b'{"record_id":"one","record_id":"two"}', tmp_path)
    loader = data.FrozenTrainingData(source)
    with pytest.raises(ValueError, match="duplicate_json_key"):
        loader.reader(row["record_id"], split="train")


def test_eligibility_incomplete_or_semantically_changed_cannot_pass_caller_hash(source, tmp_path):
    payload = Path(source["shift_eligibility"]["path"]).read_bytes().splitlines()
    source["shift_eligibility"] = _write(tmp_path / "incomplete.jsonl", b"\n".join(payload[1:]))
    with pytest.raises(ValueError, match="complete_train_eligibility_required"):
        data.FrozenTrainingData(source)
    changed = json.loads(payload[0]); changed.update(corrected_valid=False,
        corrected_invalid_reasons=["synthetic"], graph_range_valid=False)
    payload[0] = json.dumps(changed).encode()
    source["shift_eligibility"] = _write(tmp_path / "changed.jsonl", b"\n".join(payload))
    with pytest.raises(ValueError, match="identity_shift_required"):
        data.FrozenTrainingData(source)


def test_valid_nonidentity_eligibility_drift_is_rejected_by_historical_orbit(source, tmp_path):
    rows = [json.loads(line) for line in Path(source["shift_eligibility"]["path"]).read_bytes().splitlines()]
    rows[1].update(corrected_valid=False, corrected_invalid_reasons=["synthetic"], graph_range_valid=False)
    source["shift_eligibility"] = _write(tmp_path / "different-orbit.jsonl", _jsonl(rows))
    with pytest.raises(ValueError, match="frozen_orbit_table_mismatch"):
        data.FrozenTrainingData(source)


def test_duplicate_and_cross_split_assignments_fail_even_under_test_hash_override(source, tmp_path, monkeypatch):
    rows = [json.loads(line) for line in Path(source["split_assignments"]["path"]).read_bytes().splitlines()]
    for mutation, error in (("duplicate", "duplicate_split_record"), ("component", "cross_split_component")):
        changed = deepcopy(rows)
        if mutation == "duplicate":
            changed.append(changed[0])
        else:
            changed[1295]["source_component_id"] = changed[0]["source_component_id"]
        source["split_assignments"] = _write(tmp_path / f"bad-{mutation}.jsonl", _jsonl(changed))
        monkeypatch.setattr(data.reference, "CORRECTED_SPLIT_ASSIGNMENTS_SHA256", source["split_assignments"]["sha256"])
        with pytest.raises(ValueError, match=error):
            data.FrozenTrainingData(source)


def test_v2_targets_cannot_enter_frozen_v1_training_even_when_rehashed(source, tmp_path):
    from music_critic.experiments.analysisgnn.multitask_contract import TARGET_SIDECAR_VERSION
    row = source["records"][0]
    payload = json.loads(Path(row["sidecar"]["path"]).read_text())
    payload["schema_version"] = TARGET_SIDECAR_VERSION
    payload["fingerprint"] = fingerprint({key: value for key, value in payload.items() if key != "fingerprint"})
    _replace_file(row, "sidecar", payload, tmp_path)
    loader = data.FrozenTrainingData(source)
    with pytest.raises(ValueError, match="sidecar_contract_mismatch"):
        loader.reader(row["record_id"], split="train")


def test_authenticated_sidecar_mask_change_does_not_change_raw_graph(source, tmp_path):
    before = data.FrozenTrainingData(source)
    raw_before, _ = before.reader("synthetic:train:0000", split="train")
    row = source["records"][0]
    payload = json.loads(Path(row["sidecar"]["path"]).read_text())
    payload["entities"] = [{"canonical_entity_id": "masked-fixture", "entity_type": "harmonic_event",
        "targets": {"quality": {"available": False, "masked": True, "canonical_value": None,
            "missing_reason": "synthetic_missing", "confidence": None}}}]
    payload["fingerprint"] = fingerprint({key: value for key, value in payload.items() if key != "fingerprint"})
    _replace_file(row, "sidecar", payload, tmp_path)
    after = data.FrozenTrainingData(source)
    raw_after, sidecar = after.reader("synthetic:train:0000", split="train")
    assert after.manifest_fingerprint != before.manifest_fingerprint
    for node_type in raw_before.node_types:
        for field, value in raw_before[node_type].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, raw_after[node_type][field])
    assert sidecar["entities"][0]["targets"]["quality"]["masked"] is True
    assert sidecar["entities"][0]["targets"]["quality"]["canonical_value"] is None


def test_self_consistent_reweighted_class_payload_cannot_replace_frozen_weights(source, tmp_path):
    from music_critic.experiments.analysisgnn.training_policy import build_class_weight_payload
    fixture = json.loads(Path(source["class_weights"]["path"]).read_text())
    rows = {head["task_id"]: [{"class_id": row["class_id"], "class_value": row["class_value"],
        "canonical_target_row_count": row["train_canonical_target_row_count"],
        "component_count": row["train_component_count"], "support_tier": row["train_support_tier"]}
        for row in head["classes"]] for head in fixture["class_weight_payload"]["heads"]}
    rows["quality"][0]["canonical_target_row_count"] += 1
    fixture["class_weight_payload"] = build_class_weight_payload(rows)
    source["class_weights"] = _write(tmp_path / "reweighted.json", fixture)
    with pytest.raises(ValueError, match="frozen_class_weights_mismatch"):
        data.FrozenTrainingData(source)
