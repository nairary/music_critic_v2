"""Synthetic filesystem/admission checks; no real musical payloads."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import gzip
import json
from pathlib import Path

import pytest

from scripts.build_native_theory_enrichment import build_admissions, export_native, safe_child
from music_critic.data import piece_to_dict
from tests.tasks.test_multisource_contract import _hook_piece


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return sha256(data).hexdigest()


def fixture(tmp_path):
    base = _hook_piece()
    piece = piece_to_dict(replace(base, split=None))
    h = write_json(tmp_path / "cache" / "one.json", piece)
    index = {"header": {"dataset_id": "hooktheory", "index_fingerprint": "index"}, "records": [
        {"dataset_id": "hooktheory", "piece_id": base.piece_id, "source_identity": "one",
         "source_group_id": "family", "canonical_relative_path": "one.json", "canonical_sha256": h,
         "source_sha256": "b" * 64, "suggested_split": "train"}]}
    split = {"index_fingerprints": [["hooktheory", "index"]], "assignments": [
        {"piece_id": base.piece_id, "dataset_id": "hooktheory", "split": "train", "component_fingerprint": "family"}]}
    metadata = {"one": {"clip_id": "one", "metadata": {
        "split": "TRAIN", "hooktheory": {"artist": "Artist", "song": "Song"}}}}
    return index, split, metadata


def test_native_and_current_train_required(tmp_path):
    index, split, metadata = fixture(tmp_path)
    assert build_admissions([index], split, metadata)[0]["admitted"]
    metadata["one"]["metadata"]["split"] = "VALID"
    assert not build_admissions([index], split, metadata)[0]["admitted"]
    metadata["one"]["metadata"]["split"] = "TRAIN"
    split["assignments"][0]["split"] = "test"
    assert not build_admissions([index], split, metadata)[0]["admitted"]


def test_family_heldout_blocks_otherwise_train_clip(tmp_path):
    index, split, metadata = fixture(tmp_path)
    metadata["other"] = deepcopy(metadata["one"])
    metadata["other"]["metadata"]["split"] = "TEST"
    row = build_admissions([index], split, metadata)[0]
    assert not row["admitted"] and "native_family_heldout" in row["reasons"]


def test_current_family_conflict_blocks_other_clip(tmp_path):
    index, split, metadata = fixture(tmp_path)
    extra = deepcopy(index["records"][0]); extra["piece_id"] = "piece:other"; extra["source_identity"] = "other"
    index["records"].append(extra)
    metadata["other"] = deepcopy(metadata["one"])
    split["assignments"].append({"piece_id": "piece:other", "dataset_id": "hooktheory", "split": "validation", "component_fingerprint": "othercomponent"})
    assert not build_admissions([index], split, metadata)[0]["admitted"]


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/escape.json"])
def test_path_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError): safe_child(tmp_path, path)


def test_metadata_fingerprint_mismatch_rejected(tmp_path):
    index, split, metadata = fixture(tmp_path)
    split["index_fingerprints"][0][1] = "other"
    with pytest.raises(ValueError): build_admissions([index], split, metadata)


def test_export_skips_heldout_before_payload_and_is_immutable(tmp_path):
    index, split, metadata = fixture(tmp_path)
    metadata["one"]["metadata"]["split"] = "TEST"
    (tmp_path / "cache" / "one.json").unlink()
    out = tmp_path / "output"
    summary = export_native([index], split, metadata, {"hooktheory": tmp_path / "cache"}, out, [])
    assert summary["completed"] == 0 and summary["failed"] == 0 and summary["skipped"] == 1
    with pytest.raises(FileExistsError):
        export_native([index], split, metadata, {"hooktheory": tmp_path / "cache"}, out, [])


def test_hash_bound_gzip_roundtrip_and_bad_hash_failure(tmp_path):
    index, split, metadata = fixture(tmp_path)
    out = tmp_path / "output"
    summary = export_native([index], split, metadata, {"hooktheory": tmp_path / "cache"}, out, [])
    assert summary["completed"] == 1 and summary["note_count"] == 1
    row = json.loads((out / "index.jsonl").read_text())
    data = Path(row["bundle"]["path"]).read_bytes()
    assert row["status"] == "completed"
    assert sha256(data).hexdigest() == row["bundle"]["sha256"]
    assert json.loads(gzip.decompress(data))["coverage"]["unique_notes_accounted"] == 1
    index["records"][0]["canonical_sha256"] = "0" * 64
    summary = export_native([index], split, metadata, {"hooktheory": tmp_path / "cache"}, tmp_path / "bad", [])
    assert summary["failed"] == 1 and summary["completed"] == 0
    assert "hash mismatch" in (tmp_path / "bad" / "failures.jsonl").read_text()


def test_export_preserves_clipped_zero_chord_without_rejecting_piece(tmp_path):
    index, split, metadata = fixture(tmp_path)
    base = _hook_piece()
    annotations = tuple(replace(a, end_qn=a.start_qn) if a.annotation_type == "theory.chord" else a
                        for a in base.annotations)
    payload = piece_to_dict(replace(base, annotations=annotations, split=None))
    index["records"][0]["canonical_sha256"] = write_json(tmp_path / "cache" / "one.json", payload)
    out = tmp_path / "output"
    summary = export_native([index], split, metadata, {"hooktheory": tmp_path / "cache"}, out, [])
    assert summary["completed"] == 1 and summary["failed"] == 0
    row = json.loads((out / "index.jsonl").read_text())
    bundle = json.loads(gzip.decompress(Path(row["bundle"]["path"]).read_bytes()))
    assert bundle["harmonies"] == []
    assert bundle["objects"][0]["applicability"] == "nonsounding_native_annotation"
    assert all(r["harmony_id"] is None for r in bundle["note_contexts"])
    packet_row = json.loads((out / "astra_queue" / "index.jsonl").read_text())
    packet = json.loads(gzip.decompress(Path(packet_row["path"]).read_bytes()))
    assert packet["objects"] == bundle["objects"]
