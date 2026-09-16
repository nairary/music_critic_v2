import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts import package_training_data as pack
from scripts.restore_training_data import restore


@pytest.fixture
def bundle(tmp_path):
    def binding(name):
        path = tmp_path / name
        path.write_bytes(b"fixture " + name.encode())
        return {"path": str(path), "sha256": pack.digest(path), "size_bytes": path.stat().st_size}
    row = {"record_id": "an:example:Béla op.1", "split": "train"}
    row.update({k: binding(k) for k in ("piece", "sidecar", "metadata")})
    manifest = {"schema_version": "HarmonicFusionDilemmaData@1.0.0", "records": [row]}
    manifest.update({k: binding(k) for k in ("split_assignments", "shift_eligibility", "class_weights", "joint_tuple_counts")})
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    root = tmp_path / "package"
    pack.package(source, "/srv/music-critic/data/prepared", root, files_per_part=3)
    return root


def test_restore_to_new_root_preserves_bytes_and_can_verify_existing_payload(tmp_path, bundle):
    target = tmp_path / "restored"
    sha = pack.digest(bundle / "receipt.json")
    result = restore(bundle, target, expected_receipt_sha256=sha)
    assert result["verified_files"] == 7 and result["records"] == 1
    assert not result["test_payload_opened"]
    assert (target / "original-manifest.json").read_bytes() == (bundle / "original-manifest.json").read_bytes()
    relocated = json.loads((target / "data-manifest.json").read_text())
    assert relocated["records"][0]["record_id"] == "an:example:Béla op.1"
    assert all(Path(relocated["records"][0][k]["path"]).is_relative_to(target) for k in ("piece", "sidecar", "metadata"))
    assert restore(bundle, target, expected_receipt_sha256=sha, verify_only=True)["verified_files"] == 7
    with pytest.raises(FileExistsError):
        restore(bundle, target, expected_receipt_sha256=sha)
    Path(relocated["records"][0]["piece"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="payload hash"):
        restore(bundle, target, expected_receipt_sha256=sha, verify_only=True)


def test_receipt_or_part_corruption_rejected_before_destination_creation(tmp_path, bundle):
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="receipt SHA256"):
        restore(bundle, target, expected_receipt_sha256="0" * 64)
    sha = pack.digest(bundle / "receipt.json")
    (bundle / "payload-0000.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="part hash"):
        restore(bundle, target, expected_receipt_sha256=sha)
    assert not target.exists()


def test_test_split_rejected_before_archive_or_destination_io(tmp_path, bundle):
    original_path = bundle / "original-manifest.json"
    original = json.loads(original_path.read_text())
    original["records"][0]["split"] = "test"
    original_path.write_text(json.dumps(original))
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["original_manifest_sha256"] = pack.digest(original_path)
    receipt_path.write_text(json.dumps(receipt))
    for path in bundle.glob("*.tar.gz"):
        path.unlink()
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="splits are forbidden"):
        restore(bundle, target, expected_receipt_sha256=pack.digest(receipt_path))
    assert not target.exists()


@pytest.mark.parametrize("unsafe", ["../outside", "symlink", "duplicate"])
def test_archive_members_cannot_escape_or_duplicate_payload(tmp_path, bundle, unsafe):
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    p = bundle / receipt["parts"][0]["path"]
    with tarfile.open(p) as archive:
        member = next(iter(archive))
        content = archive.extractfile(member).read()
    with tarfile.open(p, "w:gz") as archive:
        if unsafe == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
            archive.addfile(member)
        elif unsafe == "duplicate":
            archive.addfile(member, io.BytesIO(content))
            archive.addfile(member, io.BytesIO(content))
        else:
            member.name = unsafe
            archive.addfile(member, io.BytesIO(content))
    receipt["parts"][0].update(sha256=pack.digest(p), size_bytes=p.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    target = tmp_path / "restored"
    with pytest.raises(ValueError, match="archive member"):
        restore(bundle, target, expected_receipt_sha256=pack.digest(receipt_path))
    assert not (tmp_path / "outside").exists()
    assert not (target / "restore-receipt.json").exists()
