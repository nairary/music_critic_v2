#!/usr/bin/env python3
"""Read only authenticated TRAIN/VALIDATION sidecars from the prepared export."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

from music_critic.experiments.analysisgnn.roman_composition import audit, harmonic_rows


def checked_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError("unsafe archive member")
    return path


def checked_bytes(path, digest, size=None):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink forbidden")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest or (size is not None and len(raw) != size):
        raise ValueError("hash/size mismatch: " + str(path))
    return raw


def read_rows(package, manifest_sha256, receipt_sha256, progress):
    manifest = json.loads(checked_bytes(package / "data-manifest.json", manifest_sha256))
    receipt = json.loads(checked_bytes(package / "receipt.json", receipt_sha256))
    if receipt["relocated_manifest_sha256"] != manifest_sha256 or receipt["test_payload_packaged"] is not False:
        raise ValueError("untrusted manifest/TEST package")
    root = PurePosixPath(receipt["destination"])
    expected, ids, components = {}, set(), {}
    for record in manifest["records"]:
        if record["split"] not in ("train", "validation"):
            raise ValueError("TEST is forbidden")
        if record["record_id"] in ids:
            raise ValueError("duplicate record")
        ids.add(record["record_id"])
        previous = components.setdefault(record["source_component_id"], record["split"])
        if previous != record["split"]:
            raise ValueError("component crosses split")
        name = str(PurePosixPath(record["sidecar"]["path"]).relative_to(root))
        checked_member(name)
        if name in expected:
            raise ValueError("duplicate sidecar binding")
        expected[name] = record
    if Counter(r["split"] for r in manifest["records"]) != {"train": 1295, "validation": 162}:
        raise ValueError("incomplete frozen corpus")
    seen, rows = set(), []
    for part in receipt["parts"]:
        checked_member(part["path"])
        archive = package / part["path"]
        checked_bytes(archive, part["sha256"], part["size_bytes"])
        with tarfile.open(archive, mode="r|gz") as handle:
            for member in handle:
                checked_member(member.name)
                if not member.isfile() or member.name in seen:
                    raise ValueError("duplicate or nonregular member")
                seen.add(member.name)
                if member.name not in expected:
                    continue
                record = expected[member.name]
                binding = record["sidecar"]
                raw = handle.extractfile(member).read()
                if len(raw) != binding["size_bytes"] or sha256(raw).hexdigest() != binding["sha256"]:
                    raise ValueError("sidecar integrity")
                rows.extend(harmonic_rows(record, json.loads(raw)))
        progress(dict(part=part["path"], harmonic_events=len(rows)))
    if expected.keys() - seen:
        raise ValueError("missing sidecars")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def progress(value):
        print(json.dumps(value), flush=True)
    rows = read_rows(args.package, args.manifest_sha256, args.receipt_sha256, progress)
    with (args.output / "harmonic-rows.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = audit(rows)
    result["elapsed_seconds"] = time.monotonic() - started
    result["manifest_sha256"] = args.manifest_sha256
    result["receipt_sha256"] = args.receipt_sha256
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    progress(dict(completed=True, elapsed_seconds=result["elapsed_seconds"]))


if __name__ == "__main__":
    main()
