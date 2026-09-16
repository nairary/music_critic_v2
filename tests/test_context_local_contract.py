"""Bounded synthetic metadata only; no raw corpus or checkpoint payloads."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.experiments import context_local_contract as contract
from music_critic.experiments.analysisgnn.contracts import fingerprint as b3_fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import production_task_registry, vocabularies_payload
from music_critic.experiments.context_diagnostics import FileBinding


def _piece_id(record, dialect):
    label = "an" if dialect == "an_joint" else "dlc"
    return f"piece:dilemmadata-{label}-{sha256(record.encode()).hexdigest()[:24]}"


def _seal(value):
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    return {**payload, "fingerprint": b3_fingerprint(payload)}


@pytest.fixture
def metadata_fixture(monkeypatch, tmp_path):
    full_counts = {"an_joint": 1, "dlc": 6, "total": 7}
    paper_counts = {"an_joint": 1, "dlc": 5, "total": 6}
    for name, value in {
        "EXPECTED_FULL_COUNTS": full_counts, "EXPECTED_PAPER_COUNTS": paper_counts,
        "EXPECTED_SPLIT_COUNTS": {"train": 3, "validation": 2, "test": 1},
        "EXPECTED_COMPONENT_COUNTS": {"train": 2, "validation": 1, "test": 1},
        "EXPECTED_EXCLUSION_COUNT": 1, "SELECTED_RECORD_COUNTS": {"train": 2, "validation": 1},
    }.items():
        monkeypatch.setattr(contract, name, value)
    inventory = [("an-a", "an_joint", "A", "train"), ("dlc-b", "dlc", "B", "train"),
                 ("dlc-c", "dlc", "B", "train"), ("dlc-d", "dlc", "C", "validation"),
                 ("dlc-e", "dlc", "C", "validation"), ("dlc-f", "dlc", "D", "test")]
    records = [{"record_id": record, "piece_id": _piece_id(record, dialect),
                "dialect": dialect, "source_component_id": component, "raw_projection_sha256": "a" * 64,
                "raw_equivalence_id": "raw-" + record, "task_availability": {"quality": True}}
               for record, dialect, component, split in inventory]
    assignments = [{"record_id": record, "dialect": dialect, "source_component_id": component,
                    "split": split, "assignment_algorithm": contract.ASSIGNMENT_ALGORITHM,
                    "assignment_namespace": contract.ASSIGNMENT_NAMESPACE}
                   for record, dialect, component, split in inventory]
    full_records = [row[0] for row in inventory] + ["dlc-x"]
    grouped = {"A": ["an-a", "dlc-x"], "B": ["dlc-b", "dlc-c"], "C": ["dlc-d", "dlc-e"], "D": ["dlc-f"]}
    components = [{"version": contract.SOURCE_COMPONENT_VERSION, "source_component_id": group,
                   "record_ids": members, "source_evidence": [{"record_id": name} for name in members]}
                  for group, members in grouped.items()]
    data = {
        "full_raw_manifest.json": _seal({"universe_id": contract.FULL_RAW_UNIVERSE_ID, "counts": full_counts, "record_ids": full_records}),
        "paper_candidate_manifest.json": _seal({"universe_id": contract.PAPER_CANDIDATE_UNIVERSE_ID,
                                                  "counts": paper_counts, "record_ids": [row[0] for row in inventory]}),
        "source_components.jsonl": components,
        "overlap_exclusions.jsonl": [{"record_id": "dlc-x", "matched_an_record_id": "an-a", "dialect": "dlc", "source_component_id": "A"}],
        "paper_candidate_records.jsonl": records,
        "split_assignments.jsonl": assignments,
        "task_registry.json": production_task_registry(),
        "vocabularies.json": vocabularies_payload(),
    }
    def write(values=None):
        payloads = data if values is None else values
        bindings = {}
        expected = {}
        for name, value in payloads.items():
            raw = (("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in value) + "\n")
                   if name.endswith(".jsonl") else json.dumps(value, sort_keys=True, ensure_ascii=False)).encode()
            path = tmp_path / name
            path.write_bytes(raw)
            expected[name] = sha256(raw).hexdigest()
            bindings[name] = FileBinding(str(path), expected[name], len(raw))
        monkeypatch.setattr(contract, "EXPECTED_METADATA_SHA256", expected)
        return bindings
    return data, write


def test_frozen_hashes_and_counts_match_tracked_b3_fixture():
    path = Path(__file__).parent / "fixtures/analysisgnn/phase9eb3_multitask_contract.json"
    fixture = json.loads(path.read_text())
    assert len(contract.EXPECTED_METADATA_SHA256) == 8
    assert contract.EXPECTED_METADATA_SHA256 == {name: fixture["artifact_sha256"][name] for name in contract.EXPECTED_METADATA_SHA256}
    assert contract.EXPECTED_FULL_COUNTS == fixture["universes"]["full_raw"]
    assert contract.EXPECTED_PAPER_COUNTS == fixture["universes"]["paper_candidate"]
    assert contract.EXPECTED_SPLIT_COUNTS == fixture["split"]["record_counts"]
    assert contract.EXPECTED_COMPONENT_COUNTS == fixture["split"]["component_counts"]
    assert contract.EXPECTED_EXCLUSION_COUNT == 14
    assert contract.SELECTED_RECORD_COUNTS == {"train": 128, "validation": 32}
    assert contract.SELECTION_SEED == 17


def test_full_vs_paper_closure_and_no_TEST_selection(metadata_fixture):
    _, write = metadata_fixture
    result = contract.load_authoritative_metadata(write())
    assert len(result.assignments) == 6
    assert "dlc-x" not in result.assignments
    assert result.assignments["dlc-f"]["split"] == "test"
    assert len(result.selected) == 3
    assert [row["split"] for row in result.selected] == ["train", "train", "validation"]
    assert "dlc-f" not in result.selected_records
    assert result.evidence["official_component_counts"]["validation"] == 1
    assert result.evidence["selection"]["unit"] == "record"
    assert result.evidence["access_scope"]["test_raw_or_targets_opened"] is False
    for row in result.selected:
        assert result.record_to_piece[row["record_id"]].startswith("piece:dilemmadata-")


def test_selection_has_independent_seed17_sha_oracle(metadata_fixture):
    data, write = metadata_fixture
    result = contract.load_authoritative_metadata(write())
    expected = []
    for split, count in (("train", 2), ("validation", 1)):
        ids = [row["record_id"] for row in data["split_assignments.jsonl"] if row["split"] == split]
        def key(record):
            raw = json.dumps(["music-critic-v2.context-local-record-selection-v1", 17, split, record],
                             separators=(",", ":"), ensure_ascii=False).encode()
            return sha256(raw).hexdigest(), record
        expected.extend(sorted(ids, key=key)[:count])
    assert [row["record_id"] for row in result.selected] == expected
    with pytest.raises(ValueError, match="TRAIN_or_VALIDATION"):
        contract.selection_rank("test", "dlc-f")


def test_labels_availability_iteration_order_and_paths_do_not_choose_records(metadata_fixture, tmp_path):
    data, write = metadata_fixture
    first = contract.load_authoritative_metadata(write())
    changed = deepcopy(data)
    for name, value in changed.items():
        if name.endswith(".jsonl"):
            value.reverse()
    for row in changed["paper_candidate_records.jsonl"]:
        row["task_availability"] = {"quality": False, "cadence": True}
    second = contract.load_authoritative_metadata(write(changed))
    assert first.selected == second.selected
    bindings = write()
    moved = tmp_path / "moved"
    moved.mkdir()
    alternative = {}
    for name, binding in bindings.items():
        raw = Path(binding.path).read_bytes()
        path = moved / name
        path.write_bytes(raw)
        alternative[name] = FileBinding(str(path), binding.sha256, binding.size_bytes)
    assert first.to_dict() == contract.load_authoritative_metadata(alternative).to_dict()


def test_result_is_deeply_immutable_and_json_export_detached(metadata_fixture):
    _, write = metadata_fixture
    result = contract.load_authoritative_metadata(write())
    with pytest.raises(TypeError):
        result.assignments["an-a"]["split"] = "test"
    with pytest.raises(TypeError):
        result.selected[0]["split"] = "test"
    with pytest.raises(TypeError):
        result.evidence["selection"]["seed"] = 1
    exported = result.to_dict()
    exported["assignments"]["an-a"]["split"] = "test"
    assert result.assignments["an-a"]["split"] == "train"
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_hash", "relative", "oversize", "bool_size"])
def test_all_binding_declarations_gate_before_first_file_read(metadata_fixture, monkeypatch, change):
    _, write = metadata_fixture
    bindings = write()
    first = next(iter(bindings))
    old = bindings[first]
    if change == "missing":
        del bindings[first]
    elif change == "extra":
        bindings["unrequested.json"] = old
    elif change == "wrong_hash":
        bindings[first] = FileBinding(old.path, "0" * 64, old.size_bytes)
    elif change == "relative":
        bindings[first] = FileBinding("metadata.json", old.sha256, old.size_bytes)
    elif change == "oversize":
        bindings[first] = FileBinding(old.path, old.sha256, contract.MAX_METADATA_FILE_BYTES + 1)
    else:
        bindings[first] = FileBinding(old.path, old.sha256, True)
    monkeypatch.setattr(FileBinding, "read", lambda *args, **kwargs: pytest.fail("read before complete declaration gate"))
    with pytest.raises(ValueError):
        contract.load_authoritative_metadata(bindings)


def test_tampered_bytes_and_wrong_size_rejected(metadata_fixture):
    _, write = metadata_fixture
    bindings = write()
    binding = bindings["full_raw_manifest.json"]
    Path(binding.path).write_bytes(b"{}")
    with pytest.raises(ValueError, match="file_binding_mismatch"):
        contract.load_authoritative_metadata(bindings)
    bindings = write()
    old = bindings["full_raw_manifest.json"]
    bindings["full_raw_manifest.json"] = FileBinding(old.path, old.sha256, old.size_bytes + 1)
    with pytest.raises(ValueError, match="file_binding_mismatch"):
        contract.load_authoritative_metadata(bindings)


@pytest.mark.parametrize("case", ["duplicate_split", "namespace", "cross_split", "excluded_in_split", "missing_split",
                                   "wrong_component", "wrong_piece", "duplicate_piece", "missing_exclusion", "wrong_peer",
                                   "duplicate_component_member", "source_evidence", "unknown_split", "universe", "internal_hash",
                                   "source_membership", "paper_membership", "component_count", "source_evidence_not_objects"])
def test_metadata_invariants_fail_closed_even_with_fixture_hashes(metadata_fixture, case):
    data, write = metadata_fixture
    data = deepcopy(data)
    split = data["split_assignments.jsonl"]
    if case == "duplicate_split":
        split.append(deepcopy(split[0]))
    elif case == "namespace":
        split[0]["assignment_namespace"] = "different"
    elif case == "cross_split":
        split[2]["split"] = "validation"
    elif case == "excluded_in_split":
        split[0]["record_id"] = "dlc-x"
    elif case == "missing_split":
        split.pop()
    elif case == "wrong_component":
        split[0]["source_component_id"] = "B"
    elif case == "wrong_piece":
        data["paper_candidate_records.jsonl"][0]["piece_id"] = "piece:other"
    elif case == "duplicate_piece":
        data["paper_candidate_records.jsonl"][1]["piece_id"] = data["paper_candidate_records.jsonl"][0]["piece_id"]
    elif case == "missing_exclusion":
        data["overlap_exclusions.jsonl"] = []
    elif case == "wrong_peer":
        data["overlap_exclusions.jsonl"][0]["matched_an_record_id"] = "dlc-b"
    elif case == "duplicate_component_member":
        data["source_components.jsonl"][1]["record_ids"].append("an-a")
        data["source_components.jsonl"][1]["source_evidence"].append({"record_id": "an-a"})
    elif case == "source_evidence":
        data["source_components.jsonl"][0]["source_evidence"].pop()
    elif case == "source_evidence_not_objects":
        data["source_components.jsonl"][0]["source_evidence"] = ["not_an_object"]
    elif case == "unknown_split":
        split[0]["split"] = "other"
    elif case == "universe":
        data["paper_candidate_manifest.json"]["universe_id"] = "other"
        data["paper_candidate_manifest.json"] = _seal(data["paper_candidate_manifest.json"])
    elif case == "internal_hash":
        data["paper_candidate_manifest.json"]["fingerprint"] = "0" * 64
    elif case == "source_membership":
        data["source_components.jsonl"][0]["record_ids"].remove("dlc-x")
        data["source_components.jsonl"][0]["source_evidence"].pop()
    elif case == "paper_membership":
        data["paper_candidate_records.jsonl"].pop()
    else:
        data["source_components.jsonl"].append({"version": contract.SOURCE_COMPONENT_VERSION, "source_component_id": "extra",
                                                  "record_ids": [], "source_evidence": []})
    with pytest.raises(ValueError):
        contract.load_authoritative_metadata(write(data))


def test_duplicate_JSON_keys_rejected_after_outer_hash(metadata_fixture, monkeypatch):
    _, write = metadata_fixture
    bindings = write()
    name = "full_raw_manifest.json"
    raw = b'{"universe_id":"one","universe_id":"two"}'
    Path(bindings[name].path).write_bytes(raw)
    expected = dict(contract.EXPECTED_METADATA_SHA256)
    expected[name] = sha256(raw).hexdigest()
    monkeypatch.setattr(contract, "EXPECTED_METADATA_SHA256", expected)
    bindings[name] = FileBinding(bindings[name].path, expected[name], len(raw))
    with pytest.raises(ValueError, match="duplicate_json_key"):
        contract.load_authoritative_metadata(bindings)
