from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("package_training_data", Path(__file__).parents[1] / "scripts/package_training_data.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


@pytest.fixture
def manifest(tmp_path):
    def file(name):
        p = tmp_path / name
        p.write_bytes(b"synthetic raw payload " + name.encode())
        return {"path": str(p), "sha256": pack.digest(p), "size_bytes": p.stat().st_size}
    row = {"record_id": "fixture-a", "split": "train", "source_component_id": "component-a"}
    row.update({k: file(k) for k in ("piece", "sidecar", "metadata")})
    m = {"schema_version": "HarmonicFusionDilemmaData@1.0.0", "records": [row]}
    m.update({k: file(k) for k in ("split_assignments", "shift_eligibility", "class_weights", "joint_tuple_counts")})
    return m


def test_roundtrip_preserves_bytes_original_manifest_and_payload_identity(tmp_path, manifest):
    original = tmp_path / "manifest.json"
    original.write_text(json.dumps(manifest))
    before = original.read_bytes()
    out = tmp_path / "bundle"
    receipt = pack.package(original, "/srv/music-critic/data/prepared", out, files_per_part=3)
    assert receipt["files"] == 7 and not receipt["test_payload_packaged"]
    assert original.read_bytes() == before == (out / "original-manifest.json").read_bytes()
    relocated, plan = pack.plan(manifest, "/srv/music-critic/data/prepared")
    decoded = {}
    for part in receipt["parts"]:
        assert pack.digest(out / part["path"]) == part["sha256"]
        with tarfile.open(out / part["path"]) as archive:
            for member in archive:
                assert member.isfile() and not Path(member.name).is_absolute()
                decoded[member.name] = archive.extractfile(member).read()
    for binding, name in plan:
        assert hashlib.sha256(decoded[name]).hexdigest() == binding["sha256"]
    assert json.loads((out / "data-manifest.json").read_text()) == relocated


@pytest.mark.parametrize("split", ["test", "unknown"])
def test_forbidden_split_rejected_before_payload_io(tmp_path, manifest, split):
    manifest["records"][0]["split"] = split
    manifest["records"][0]["piece"]["path"] = "/absent/payload"
    with pytest.raises(ValueError, match="splits are forbidden"):
        pack.plan(manifest, "/srv/music-critic/data")


def test_duplicate_and_traversal_ids_are_rejected(manifest):
    duplicate = deepcopy(manifest)
    duplicate["records"].append(deepcopy(duplicate["records"][0]))
    with pytest.raises(ValueError, match="duplicate record"):
        pack.plan(duplicate, "/srv/music-critic/data")
    manifest["records"][0]["record_id"] = "../escape"
    with pytest.raises(ValueError, match="record ID"):
        pack.plan(manifest, "/srv/music-critic/data")


def test_source_ids_with_unicode_colons_and_spaces_have_portable_paths(manifest):
    manifest["records"][0]["record_id"] = "an:test:Béla op. 1 (a)"
    relocated, files = pack.plan(manifest, "/srv/music-critic/data")
    assert relocated["records"][0]["record_id"] == "an:test:Béla op. 1 (a)"
    assert all(":" not in name and " " not in name for _, name in files)


def test_corrupt_source_cannot_produce_completed_receipt(tmp_path, manifest):
    Path(manifest["records"][0]["piece"]["path"]).write_bytes(b"corrupt")
    original = tmp_path / "manifest.json"
    original.write_text(json.dumps(manifest))
    out = tmp_path / "bundle"
    with pytest.raises(ValueError, match="hash/size mismatch"):
        pack.package(original, "/srv/music-critic/data", out)
    assert not (out / "receipt.json").exists()


def test_symlink_payload_rejected(tmp_path, manifest):
    binding = manifest["records"][0]["piece"]
    link = tmp_path / "alias"
    link.symlink_to(binding["path"])
    binding["path"] = str(link)
    original = tmp_path / "manifest.json"
    original.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonsymlink"):
        pack.package(original, "/srv/music-critic/data", tmp_path / "bundle")
