from scripts.build_tavern_targets import admit_families, write_new, run
from types import SimpleNamespace
from hashlib import sha1
import json
import gzip

import pytest


def test_admission_rejects_current_and_native_heldout_and_component_conflicts(tmp_path):
    rows = [
        {"record_id": "an:training:tavern-beethoven-woo-63-a", "split": "train", "source_component_id": "c1"},
        {"record_id": "an:training:tavern-beethoven-woo-63-b", "split": "train", "source_component_id": "c1"},
        {"record_id": "an:test:tavern-beethoven-woo-70-a", "split": "train", "source_component_id": "c2"},
        {"record_id": "an:training:tavern-mozart-k179-a", "split": "validation", "source_component_id": "c3"},
        {"record_id": "an:training:tavern-mozart-k025-a", "split": "train", "source_component_id": "c4"},
        {"record_id": "dlc:other", "split": "test", "source_component_id": "c4"},
    ]
    result = admit_families({"records": rows})
    assert result["beethoven-woo-63"]["admitted"]
    assert not result["beethoven-woo-70"]["admitted"]
    assert not result["mozart-k179"]["admitted"]
    assert not result["mozart-k025"]["admitted"]
    path = tmp_path / "out.json"
    write_new(path, b"test")
    with pytest.raises(FileExistsError):
        write_new(path, b"other")


def test_disagreeing_joined_versions_keep_distinct_raw_notes(tmp_path):
    root = tmp_path / "source"
    texts = {
        "Beethoven/B063/Krn/B063_00_01_score.krn": "**kern\n*M2/4\n2c\n*-\n",
        "Beethoven/B063/Joined/B063_00_01a_a.krn": "**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n2T\t2I\t2e\n*-\t*-\t*-\n",
        "Beethoven/B063/Joined/B063_00_01a_b.krn": "**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n2T\t2I\t2g\n*-\t*-\t*-\n",
    }
    tree = {"sha": "a" * 40, "tree": []}
    for path, text in texts.items():
        payload = text.encode()
        write_new(root / path, payload)
        tree["tree"].append({"path": path, "type": "blob", "sha": sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()})
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree))
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"records": [{"record_id": "an:training:tavern-beethoven-woo-63-a", "split": "train", "source_component_id": "one"}]}))
    result = run(SimpleNamespace(output=str(tmp_path / "release"), data_manifest=str(manifest),
                 source_tree=str(tree_path), source_root=str(root), download=False, workers=1))
    assert not result["failures"]
    assert result["counts"]["released_phrases"] == 2
    assert result["counts"]["unique_source_phrases"] == 1
    assert result["counts"]["canonical_notes"] == 2
    for record in result["records"]:
        assert record["raw_source_origin"] == "joined_score_projection"
        with open(record["enrichment"]["path"], "rb") as handle:
            bundle = json.loads(gzip.decompress(handle.read()))
        assert len(bundle["provenance"]["source_note_bindings"]) == 1
