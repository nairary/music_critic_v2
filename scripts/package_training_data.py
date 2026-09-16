"""Package authenticated TRAIN/VALIDATION payloads without decoding music or models.

Original manifests remain immutable. Relocation changes operational paths only.
Run with the standard library on the source host; no Torch environment is needed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plan(manifest, destination):
    """Validate all splits and names before any referenced file can be opened."""
    metadata = ("split_assignments", "shift_eligibility", "class_weights", "joint_tuple_counts")
    if set(manifest) != {"schema_version", "records", *metadata}:
        raise ValueError("unexpected manifest fields")
    if manifest["schema_version"] != "HarmonicFusionDilemmaData@1.0.0":
        raise ValueError("unsupported manifest schema")
    root = Path(destination)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("absolute destination required")
    out = deepcopy(manifest)
    files, ids = [], set()

    def bind(source, target, relative):
        if set(source) != {"path", "sha256", "size_bytes"}:
            raise ValueError("invalid binding fields")
        p = Path(source["path"])
        if not p.is_absolute() or ".." in p.parts:
            raise ValueError("invalid source path")
        if not re.fullmatch("[0-9a-f]{64}", source["sha256"]):
            raise ValueError("invalid SHA256")
        if type(source["size_bytes"]) is not int or source["size_bytes"] <= 0:
            raise ValueError("invalid size")
        target["path"] = str(root / relative)
        files.append((source, relative))

    for original, relocated in zip(manifest["records"], out["records"], strict=True):
        if original["split"] not in ("train", "validation"):
            raise ValueError("TEST and unknown splits are forbidden")
        rid = original["record_id"]
        if (not isinstance(rid, str) or not rid or rid in (".", "..") or
                any(c in rid for c in ("/", "\\", "\x00")) or rid in ids):
            raise ValueError("invalid or duplicate record ID")
        ids.add(rid)
        directory = hashlib.sha256(rid.encode("utf-8")).hexdigest()
        for key in ("piece", "sidecar", "metadata"):
            bind(original[key], relocated[key], f"records/{directory}/{key}.json")
    if not ids:
        raise ValueError("empty manifest")
    for key in metadata:
        bind(manifest[key], out[key], f"metadata/{key}.json")
    return out, files


def package(manifest_path, destination, output, *, files_per_part=192):
    manifest_path, output = Path(manifest_path), Path(output)
    original_bytes = manifest_path.read_bytes()
    original = json.loads(original_bytes)
    relocated, files = plan(original, destination)
    if files_per_part < 1:
        raise ValueError("positive files_per_part required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "original-manifest.json").write_bytes(original_bytes)
    (output / "data-manifest.json").write_text(json.dumps(relocated, sort_keys=True, separators=(",", ":")) + "\n")
    parts = []
    for start in range(0, len(files), files_per_part):
        path = output / f"payload-{start // files_per_part:04d}.tar.gz"
        members = files[start:start + files_per_part]
        with tarfile.open(path, "w:gz", compresslevel=1) as archive:
            for binding, name in members:
                source = Path(binding["path"])
                if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_file():
                    raise ValueError(f"not a regular nonsymlink file: {source}")
                before = source.stat()
                if before.st_size != binding["size_bytes"] or digest(source) != binding["sha256"]:
                    raise ValueError(f"source hash/size mismatch: {source}")
                info = archive.gettarinfo(str(source), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with source.open("rb") as f:
                    archive.addfile(info, f)
                after = source.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
                ):
                    raise ValueError(f"source changed during packaging: {source}")
        parts.append({"path": path.name, "sha256": digest(path), "size_bytes": path.stat().st_size,
                      "files": len(members)})
        print(json.dumps({"event": "part_ready", **parts[-1]}), flush=True)
    receipt = {"schema_version": "TrainingDataRelocation@1.0.0", "parts": parts,
               "original_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
               "relocated_manifest_sha256": digest(output / "data-manifest.json"),
               "destination": str(destination), "files": len(files), "records": len(original["records"]),
               "payload_bytes": sum(x[0]["size_bytes"] for x in files), "test_payload_packaged": False}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.manifest, args.destination, args.output)), flush=True)
