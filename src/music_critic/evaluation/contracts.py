"""Versioned deterministic contracts shared by Phase 6D-A evaluation."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


EVALUATION_CONTRACT_VERSION = "1.2.0"
EVALUATION_ARTIFACT_VERSION = "1.2.0"
TRAIN_PRIOR_CONTRACT_VERSION = "1.0.0"
PROFILER_CONTRACT_VERSION = "1.1.0"
MACRO_SUMMARY_CONTRACT_VERSION = "1.0.0"


class EvaluationContractError(ValueError):
    """A stable fail-closed evaluation contract violation."""


def canonical_json_bytes(value: object, *, indent: int | None = 2) -> bytes:
    """Encode finite JSON deterministically with one terminal newline."""

    _validate_finite(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_fingerprint(value: object) -> str:
    """Fingerprint the compact canonical JSON representation."""

    return sha256(canonical_json_bytes(value, indent=None)).hexdigest()


def write_json_atomic(path: str | Path, value: object) -> None:
    """Atomically replace one deterministic JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_finite(value: object, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationContractError(
            f"evaluation.json.non_finite:{path}"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def metric_value(
    value: float | int | None,
    *,
    category: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return one explicit defined/undefined scalar metric."""

    if value is None:
        if not category or not reason:
            raise EvaluationContractError(
                "evaluation.metric.undefined_reason_missing"
            )
        return {
            "value": None,
            "undefined": {
                "category": category,
                "reason": reason,
            },
        }
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationContractError(
            "evaluation.metric.non_finite"
        )
    return {"value": value, "undefined": None}


__all__ = [
    "EVALUATION_ARTIFACT_VERSION",
    "EVALUATION_CONTRACT_VERSION",
    "PROFILER_CONTRACT_VERSION",
    "MACRO_SUMMARY_CONTRACT_VERSION",
    "TRAIN_PRIOR_CONTRACT_VERSION",
    "EvaluationContractError",
    "canonical_fingerprint",
    "canonical_json_bytes",
    "metric_value",
    "write_json_atomic",
]
