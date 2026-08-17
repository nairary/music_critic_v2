"""Crash-safe Phase 9C-A artifacts, plots, archives, and verification."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tarfile
import tempfile
from typing import Iterable, Mapping, Sequence
import zlib

from .contracts import PHASE9C_ARTIFACT_VERSION, Phase9CContractError, fingerprint


MANIFEST_NAME = "artifact_manifest.json"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise Phase9CContractError(f"phase9c.artifact.immutable_collision:{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_once(path: Path, value: object) -> None:
    write_bytes_once(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n",
    )


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase9CContractError(f"phase9c.artifact.json_invalid:{path}") from exc


def publish_staged_cell(
    staging: Path,
    destination: Path,
    *,
    cell_id: str,
    protocol_fingerprint: str,
) -> dict[str, object]:
    if destination.exists():
        return verify_completed_cell(
            destination,
            cell_id=cell_id,
            protocol_fingerprint=protocol_fingerprint,
        )
    hashes = {
        path.relative_to(staging).as_posix(): file_sha256(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "cell_manifest.json"
    }
    manifest_payload = {
        "contract_version": PHASE9C_ARTIFACT_VERSION,
        "cell_id": cell_id,
        "protocol_fingerprint": protocol_fingerprint,
        "status": "complete",
        "artifact_sha256": hashes,
    }
    manifest = {**manifest_payload, "fingerprint": fingerprint(manifest_payload)}
    write_json_once(staging / "cell_manifest.json", manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return manifest


def verify_completed_cell(
    directory: Path,
    *,
    cell_id: str,
    protocol_fingerprint: str,
) -> dict[str, object]:
    manifest = read_json(directory / "cell_manifest.json")
    if not isinstance(manifest, dict):
        raise Phase9CContractError("phase9c.artifact.cell_manifest_invalid")
    payload = dict(manifest)
    observed = payload.pop("fingerprint", None)
    actual = {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "cell_manifest.json"
    }
    if (
        observed != fingerprint(payload)
        or payload.get("contract_version") != PHASE9C_ARTIFACT_VERSION
        or payload.get("cell_id") != cell_id
        or payload.get("protocol_fingerprint") != protocol_fingerprint
        or payload.get("status") != "complete"
        or payload.get("artifact_sha256") != actual
    ):
        raise Phase9CContractError(f"phase9c.artifact.completed_cell_invalid:{cell_id}")
    return manifest


def _png_chunk(name: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload))


def line_plot_png(values: Sequence[float], *, width: int = 480, height: int = 240) -> bytes:
    """Render a small dependency-free RGB line plot."""

    if not values or any(not isinstance(value, (int, float)) for value in values):
        raise Phase9CContractError("phase9c.artifact.plot_values_invalid")
    pixels = bytearray([255] * width * height * 3)
    lower, upper = min(values), max(values)
    span = upper - lower or 1.0
    points = []
    for index, value in enumerate(values):
        x = 16 + round(index * (width - 33) / max(1, len(values) - 1))
        y = height - 17 - round((float(value) - lower) * (height - 33) / span)
        points.append((x, y))

    def darken(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = b"\x20\x5a\xa8"

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            darken(round(x0 + (x1 - x0) * step / steps), round(y0 + (y1 - y0) * step / steps))
    for x, y in points:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                darken(x + dx, y + dy)
    raw = b"".join(b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3]) for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def write_comparison_tables(root: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "variant_id",
        "transfer_mode",
        "primary_score",
        "mean_macro_f1",
        "mean_task_nll",
        "selected_epoch",
    ]
    temporary = root / ".comparison_table.csv.partial"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    csv_payload = temporary.read_bytes()
    temporary.unlink()
    write_bytes_once(root / "comparison_table.csv", csv_payload)
    header = "| " + " | ".join(fields) + " |\n"
    divider = "| " + " | ".join("---" for _ in fields) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row[field]) for field in fields) + " |\n"
        for row in rows
    )
    write_bytes_once(root / "comparison_table.md", (header + divider + body).encode("utf-8"))


def build_artifact_manifest(root: Path) -> dict[str, object]:
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME and ".staging" not in path.parts
    }
    payload = {
        "contract_version": PHASE9C_ARTIFACT_VERSION,
        "hash": "sha256",
        "regular_files_only": True,
        "files": files,
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def verify_bundle(root: Path) -> dict[str, object]:
    required = {
        "experiment_plan.json",
        "protocol.json",
        "data_semantic_projection.json",
        "profile_report.json",
        "bootstrap_report.json",
        "selection_report.json",
        "final_comparison_report.json",
        "comparison_table.csv",
        "comparison_table.md",
        "claim_boundaries.json",
        "curves/loss.png",
        "curves/primary_validation_metric.png",
    }
    for path in root.rglob("*"):
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
            raise Phase9CContractError("phase9c.verify.non_regular_member")
    manifest = read_json(root / MANIFEST_NAME)
    if not isinstance(manifest, dict):
        raise Phase9CContractError("phase9c.verify.manifest_invalid")
    payload = dict(manifest)
    observed = payload.pop("fingerprint", None)
    current = build_artifact_manifest(root)
    if observed != fingerprint(payload) or manifest != current:
        raise Phase9CContractError("phase9c.verify.artifact_corruption")
    if not required.issubset(manifest["files"]):
        raise Phase9CContractError("phase9c.verify.required_artifact_missing")
    return {
        "status": "verified",
        "bundle_fingerprint": manifest["fingerprint"],
        "file_count": len(manifest["files"]),
        "test_access": False,
    }


def safe_extract_members(archive: Path) -> tuple[str, ...]:
    """Validate a tar as regular-file-only without extracting it."""

    names = []
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise Phase9CContractError("phase9c.verify.unsafe_tar_member")
            names.append(member.name)
    return tuple(names)


def create_evidence_tar(root: Path, destination: Path) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as handle:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".staging" in path.parts:
                continue
            info = handle.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                handle.addfile(info, source)
    os.replace(temporary, destination)
    digest = file_sha256(destination)
    write_bytes_once(destination.with_suffix(destination.suffix + ".sha256"), f"{digest}  {destination.name}\n".encode("ascii"))
    return {"archive": str(destination), "sha256": digest}


__all__ = [
    "MANIFEST_NAME",
    "build_artifact_manifest",
    "create_evidence_tar",
    "file_sha256",
    "line_plot_png",
    "publish_staged_cell",
    "read_json",
    "safe_extract_members",
    "verify_bundle",
    "verify_completed_cell",
    "write_bytes_once",
    "write_comparison_tables",
    "write_json_once",
]
