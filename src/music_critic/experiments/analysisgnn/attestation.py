"""Offline verification of downloaded public AnalysisGNN evidence."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import BinaryIO

from music_critic.experiments.analysisgnn.contracts import (
    ANALYSISGNN_COMMIT,
    ANALYSISGNN_RUN_PATH,
    ANALYSISGNN_RUN_SOURCE_COMMIT,
    HISTORICAL_ARTIFACT,
    HistoricalAttestation,
    canonical_json,
)


ATTESTATION_FILE_DIGESTS = {
    "config.yaml": "1bce62b022ae8f05a55fae108fbbfb5106b8151afc2b53e2bfb433372b09e09e",
    "model-rhsjiz03-v0/model.ckpt": "a557d0046e2c03c19514e1351a3cd0f2b49c31b991c370307345a7f1c6a65f31",
    "output.log": "5b41967f07d4559cd31f18a8d76185a7b252a83a0b4515971e71262a92c7d11f",
    "requirements.txt": "e33910c8bb6b66828ee898521aa44278ab96af2e87558f3b3b0b7da55f3dc668",
    "wandb-summary.json": "53f24c85da6c75103996bb5b0d12047ebbd286dfce9607f29d510a46af1a946e",
}


def _digest(stream: BinaryIO) -> str:
    value = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def attest_historical_directory(root: str | Path) -> HistoricalAttestation:
    """Verify bytes only; checkpoint deserialization is deliberately unnecessary."""

    root = Path(root)
    actual: dict[str, str] = {}
    for filename, expected in ATTESTATION_FILE_DIGESTS.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            actual[filename] = _digest(stream)
        if actual[filename] != expected:
            raise ValueError(f"historical evidence digest mismatch: {filename}")
    checkpoint = root / "model-rhsjiz03-v0" / "model.ckpt"
    attestation = HistoricalAttestation(
        source_commit=ANALYSISGNN_COMMIT,
        run_path=ANALYSISGNN_RUN_PATH,
        run_source_commit=ANALYSISGNN_RUN_SOURCE_COMMIT,
        artifact_path=HISTORICAL_ARTIFACT,
        artifact_version=0,
        original_filename="epoch=98-step=8910.ckpt",
        epoch=98,
        global_step=8910,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=actual["model-rhsjiz03-v0/model.ckpt"],
        config_sha256=actual["config.yaml"],
        requirements_sha256=actual["requirements.txt"],
        summary_sha256=actual["wandb-summary.json"],
        output_log_sha256=actual["output.log"],
    )
    return attestation


def write_attestation(root: str | Path, output: str | Path) -> HistoricalAttestation:
    attestation = attest_historical_directory(root)
    payload = {**asdict(attestation), "attestation_fingerprint": attestation.fingerprint}
    Path(output).write_text(canonical_json(payload, indent=2) + "\n", encoding="utf-8")
    return attestation


def load_reported_metrics(root: str | Path) -> dict[str, object]:
    value = json.loads((Path(root) / "wandb-summary.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("W&B summary must be a JSON object")
    return value


__all__ = [
    "ATTESTATION_FILE_DIGESTS",
    "attest_historical_directory",
    "load_reported_metrics",
    "write_attestation",
]
