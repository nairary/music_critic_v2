"""Admission guard witnesses do not open synthetic source payloads."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_an_enrichment import checked_bytes, validate_admission, write_new


def fixture(tmp_path):
    rid = "an:training:one"
    row = dict(record_id=rid, status="admitted", current_split="train", native_split="training",
        source_component_id="g", component_record_ids=[rid], canonical_expected={"sha256":"a"*64},
        source_path=str(tmp_path / "pitch_arrays/AN/training/one_joint.tsv"))
    current = {rid:dict(split="train", source_component_id="g", piece=row["canonical_expected"])}
    assignments = {rid:dict(split="train", source_component_id="g")}
    components = {"g":dict(record_ids=[rid])}
    return row,current,assignments,components,tmp_path


def test_admission_does_not_require_or_open_payload(tmp_path):
    args = fixture(tmp_path)
    path = validate_admission(*args)
    assert not path.exists()


@pytest.mark.parametrize("split", ["test", "validation"])
def test_current_heldout_component_rejected(tmp_path, split):
    args = fixture(tmp_path)
    args[0]["component_record_ids"].append("other")
    args[3]["g"]["record_ids"].append("other")
    args[2]["other"] = dict(split=split)
    with pytest.raises(ValueError, match="heldout"): validate_admission(*args)


def test_native_heldout_component_rejected_even_if_current_train(tmp_path):
    args = fixture(tmp_path)
    args[0]["component_record_ids"].append("an:test:other")
    args[3]["g"]["record_ids"].append("an:test:other")
    args[2]["an:test:other"] = dict(split="train")
    with pytest.raises(ValueError, match="native AN heldout"): validate_admission(*args)


def test_unmatched_source_locator_rejected(tmp_path):
    args = fixture(tmp_path); args[0]["source_path"] = str(tmp_path / "other.tsv")
    with pytest.raises(ValueError, match="locator"): validate_admission(*args)


def test_immutable_write_and_authentication(tmp_path):
    from hashlib import sha256
    path = tmp_path / "once.json"; write_new(path, {"known": False})
    b = path.read_bytes(); expected = dict(sha256=sha256(b).hexdigest(), size_bytes=len(b))
    assert checked_bytes(path, expected) == b
    with pytest.raises(FileExistsError): write_new(path, {})
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="hash/size"): checked_bytes(path, expected)
