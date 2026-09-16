"""Restore or verify an authenticated TRAIN/VALIDATION package using stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

try:
    from scripts.package_training_data import digest, plan
except ModuleNotFoundError:
    from package_training_data import digest, plan


def restore(package_root, destination, *, expected_receipt_sha256, verify_only=False):
    package_root, destination = Path(package_root), Path(destination)
    receipt_path = package_root / "receipt.json"
    if digest(receipt_path) != expected_receipt_sha256:
        raise ValueError("receipt SHA256 mismatch")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema_version") != "TrainingDataRelocation@1.0.0" or
            receipt.get("test_payload_packaged") is not False):
        raise ValueError("unsupported package or TEST payload")
    original_path = package_root / "original-manifest.json"
    if digest(original_path) != receipt["original_manifest_sha256"]:
        raise ValueError("original manifest SHA256 mismatch")
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    relocated, bindings = plan(original, str(destination))
    packaged_manifest, _ = plan(original, receipt["destination"])
    packaged_path = package_root / "data-manifest.json"
    if (digest(packaged_path) != receipt["relocated_manifest_sha256"] or
            json.loads(packaged_path.read_text()) != packaged_manifest):
        raise ValueError("packaged manifest changed more than paths")
    expected = {name: binding for binding, name in bindings}
    if (len(expected) != receipt["files"] or
            len(original["records"]) != receipt["records"] or
            sum(b["size_bytes"] for b in expected.values()) != receipt["payload_bytes"]):
        raise ValueError("package accounting mismatch")

    def verify_payload(path, binding):
        if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)) or
                path.stat().st_size != binding["size_bytes"] or
                digest(path) != binding["sha256"]):
            raise ValueError("restored payload hash/size/type mismatch")

    if verify_only:
        for name, binding in expected.items():
            verify_payload(destination / name, binding)
    else:
        names = set()
        for part in receipt["parts"]:
            name = part["path"]
            if Path(name).name != name or name in names or not name.endswith(".tar.gz"):
                raise ValueError("unsafe or duplicate part name")
            names.add(name)
            path = package_root / name
            if (path.is_symlink() or path.stat().st_size != part["size_bytes"] or
                    digest(path) != part["sha256"]):
                raise ValueError("part hash/size mismatch")
        # Never overwrite an existing destination, including a partial restore.
        destination.mkdir(parents=True, exist_ok=False)
        seen = set()
        for part in receipt["parts"]:
            part_count = 0
            with tarfile.open(package_root / part["path"]) as archive:
                for member in archive:
                    if not member.isfile() or member.name not in expected or member.name in seen:
                        raise ValueError("unexpected, unsafe or duplicate archive member")
                    binding = expected[member.name]
                    if member.size != binding["size_bytes"]:
                        raise ValueError("archive payload size mismatch")
                    target = destination / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256()
                    with archive.extractfile(member) as source, target.open("xb") as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(chunk)
                            h.update(chunk)
                    if h.hexdigest() != binding["sha256"]:
                        raise ValueError("archive payload SHA256 mismatch")
                    seen.add(member.name)
                    part_count += 1
            if part_count != part["files"]:
                raise ValueError("part file count mismatch")
        if seen != set(expected):
            raise ValueError("missing archive payload")
        (destination / "original-manifest.json").write_bytes(original_bytes)
        (destination / "data-manifest.json").write_text(
            json.dumps(relocated, sort_keys=True, separators=(",", ":")) + "\n"
        )
    result = {"schema_version": "TrainingDataRestore@1.0.0",
              "package_receipt_sha256": expected_receipt_sha256,
              "original_manifest_sha256": receipt["original_manifest_sha256"],
              "destination": str(destination), "verified_files": len(expected),
              "verified_bytes": receipt["payload_bytes"], "records": receipt["records"],
              "test_payload_opened": False, "verify_only": verify_only}
    if not verify_only:
        result["restored_manifest_sha256"] = digest(destination / "data-manifest.json")
        (destination / "restore-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(restore(args.package, args.destination,
                             expected_receipt_sha256=args.expected_receipt_sha256,
                             verify_only=args.verify_only), sort_keys=True))
