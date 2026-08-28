"""719-record common-subset materialization with source-first splitting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Iterable, Sequence

from music_critic.adapters.dilemmadata import (
    DilemmadataAccepted,
    DilemmadataCorpusRecord,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
)
from music_critic.adapters.dilemmadata_targets import (
    DilemmadataTargetAccepted,
    build_dilemmadata_target_sidecar,
    load_dilemmadata_target_metadata_index,
)
from music_critic.data import CanonicalPiece, dumps_piece, loads_piece
from music_critic.experiments.analysisgnn.contracts import (
    EXPECTED_COMMON_REGISTRY_FINGERPRINT,
    EXPECTED_DIALECT_COUNTS,
    EXPECTED_RAW_INDEX_FINGERPRINT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_FINGERPRINT,
    canonical_json,
    fingerprint,
)
from music_critic.tasks.dilemmadata_common import (
    DilemmadataCommonHarmonicProjection,
    build_dilemmadata_common_harmonic_projection,
    dumps_dilemmadata_common_projection,
    loads_dilemmadata_common_projection,
)
from music_critic.tasks.multisource import (
    TargetBundle,
    dumps_target_bundle,
    loads_target_bundle,
    target_bundle_fingerprint,
)


@dataclass(frozen=True, slots=True)
class CommonDatasetRecord:
    record_id: str
    piece_id: str
    dialect: str
    source_group_id: str
    split: str
    raw_projection_sha256: str
    target_bundle_fingerprint: str
    common_projection_fingerprint: str


@dataclass(frozen=True, slots=True)
class CommonDatasetManifest:
    contract_version: str
    source_split_fingerprint: str
    raw_index_fingerprint: str
    common_registry_fingerprint: str
    record_count: int
    dialect_counts: dict[str, int]
    split_counts: dict[str, int]
    assignment_fingerprint: str
    records_fingerprint: str
    records: tuple[CommonDatasetRecord, ...]
    manifest_fingerprint: str


def _component_fingerprint(records: Sequence[DilemmadataCorpusRecord]) -> str:
    return fingerprint(
        [
            {
                "dataset_id": "dilemmadata",
                "lineage_group_id": row.lineage_group_id,
                "piece_id": row.piece_id,
                "source_group_id": row.source_group_id,
            }
            for row in sorted(records, key=lambda item: item.piece_id)
        ]
    )


def _largest_remainder(total: int) -> dict[str, int]:
    ratios = {"test": Decimal("0.1"), "train": Decimal("0.8"), "validation": Decimal("0.1")}
    exact = {key: Decimal(total) * ratio for key, ratio in ratios.items()}
    quotas = {
        key: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for key, value in exact.items()
    }
    remaining = total - sum(quotas.values())
    order = sorted(quotas, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def accepted_split_assignments(
    records: Sequence[DilemmadataCorpusRecord],
) -> dict[str, str]:
    """Exact Phase 9E-A source-component algorithm, before augmentation."""

    components: dict[str, list[DilemmadataCorpusRecord]] = defaultdict(list)
    for record in records:
        components[record.source_group_id].append(record)
    component_rows = tuple(
        tuple(sorted(rows, key=lambda row: row.piece_id))
        for _group, rows in sorted(components.items())
    )
    quotas = _largest_remainder(len(component_rows))
    schedule = tuple(split for split in sorted(quotas) for _ in range(quotas[split]))
    ordered = sorted(
        component_rows,
        key=lambda rows: (
            fingerprint({"component": _component_fingerprint(rows), "seed": 9_001}),
            tuple(row.piece_id for row in rows),
        ),
    )
    return {
        record.piece_id: split
        for rows, split in zip(ordered, schedule, strict=True)
        for record in rows
    }


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _manifest_payload(
    rows: tuple[CommonDatasetRecord, ...], assignment_fingerprint: str
) -> dict[str, object]:
    return {
        "assignment_fingerprint": assignment_fingerprint,
        "common_registry_fingerprint": EXPECTED_COMMON_REGISTRY_FINGERPRINT,
        "contract_version": "phase9eb1-common-dataset-v1",
        "dialect_counts": dict(Counter(row.dialect for row in rows)),
        "raw_index_fingerprint": EXPECTED_RAW_INDEX_FINGERPRINT,
        "record_count": len(rows),
        "records": [asdict(row) for row in rows],
        "records_fingerprint": fingerprint([asdict(row) for row in rows]),
        "source_split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
        "split_counts": dict(Counter(row.split for row in rows)),
    }


def prepare_common_dataset(
    corpus_root: str | Path,
    output_root: str | Path,
) -> CommonDatasetManifest:
    """Validate the release and materialize target-neutral + sidecar JSON."""

    output_root = Path(output_root)
    discovery = discover_dilemmadata_corpus(corpus_root, require_valid=True)
    metadata = load_dilemmadata_target_metadata_index(
        discovery.root, discovery.records
    )
    accepted: list[tuple[DilemmadataAccepted, DilemmadataTargetAccepted, DilemmadataCommonHarmonicProjection]] = []
    for record in discovery.records:
        raw = convert_dilemmadata_record(record)
        if not isinstance(raw, DilemmadataAccepted):
            continue
        targets = build_dilemmadata_target_sidecar(raw, metadata_index=metadata)
        if not isinstance(targets, DilemmadataTargetAccepted):
            raise RuntimeError(f"accepted raw record lost its target sidecar: {record.record_id}")
        projection = build_dilemmadata_common_harmonic_projection(raw, targets)
        accepted.append((raw, targets, projection))
    if len(accepted) != EXPECTED_RECORD_COUNT:
        raise RuntimeError(f"expected 719 accepted records, observed {len(accepted)}")
    assignments = accepted_split_assignments(tuple(raw.record for raw, _, _ in accepted))
    rows: list[CommonDatasetRecord] = []
    for raw, targets, projection in sorted(accepted, key=lambda values: values[0].record.piece_id):
        record = raw.record
        record_root = output_root / "records" / record.piece_id
        _write(record_root / "piece.json", dumps_piece(raw.piece))
        _write(record_root / "targets.json", dumps_target_bundle(targets.target_bundle))
        _write(
            record_root / "common_projection.json",
            dumps_dilemmadata_common_projection(projection),
        )
        rows.append(
            CommonDatasetRecord(
                record_id=record.record_id,
                piece_id=record.piece_id,
                dialect=record.dialect,
                source_group_id=record.source_group_id,
                split=assignments[record.piece_id],
                raw_projection_sha256=record.raw_projection_sha256,
                target_bundle_fingerprint=target_bundle_fingerprint(targets.target_bundle),
                common_projection_fingerprint=projection.projection_fingerprint,
            )
        )
    materialized = tuple(rows)
    dialect_counts = dict(Counter(row.dialect for row in materialized))
    split_counts = dict(Counter(row.split for row in materialized))
    if dialect_counts != EXPECTED_DIALECT_COUNTS or split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError("common dataset count contract changed")
    if any(
        len({row.split for row in materialized if row.source_group_id == group}) != 1
        for group in {row.source_group_id for row in materialized}
    ):
        raise RuntimeError("source group crosses split boundary")
    assignment_fingerprint = fingerprint(
        [[row.piece_id, row.source_group_id, row.split] for row in materialized]
    )
    payload = _manifest_payload(materialized, assignment_fingerprint)
    manifest_fingerprint = fingerprint(payload)
    payload["manifest_fingerprint"] = manifest_fingerprint
    _write(output_root / "manifest.json", canonical_json(payload, indent=2) + "\n")
    return CommonDatasetManifest(
        **{**payload, "records": materialized}  # type: ignore[arg-type]
    )


def load_common_manifest(path: str | Path) -> CommonDatasetManifest:
    import json

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = tuple(CommonDatasetRecord(**row) for row in value.pop("records"))
    expected = value.pop("manifest_fingerprint")
    if fingerprint({**value, "records": [asdict(row) for row in rows]}) != expected:
        raise ValueError("common dataset manifest fingerprint mismatch")
    return CommonDatasetManifest(**value, records=rows, manifest_fingerprint=expected)


def load_common_record(
    cache_root: str | Path,
    row: CommonDatasetRecord,
) -> tuple[CanonicalPiece, TargetBundle, DilemmadataCommonHarmonicProjection]:
    root = Path(cache_root) / "records" / row.piece_id
    piece = loads_piece((root / "piece.json").read_text(encoding="utf-8"))
    targets = loads_target_bundle((root / "targets.json").read_text(encoding="utf-8"))
    projection = loads_dilemmadata_common_projection(
        (root / "common_projection.json").read_text(encoding="utf-8")
    )
    if (
        piece.piece_id != row.piece_id
        or target_bundle_fingerprint(targets) != row.target_bundle_fingerprint
        or projection.projection_fingerprint != row.common_projection_fingerprint
    ):
        raise ValueError("cached common record differs from its manifest")
    return piece, targets, projection


__all__ = [
    "CommonDatasetManifest",
    "CommonDatasetRecord",
    "accepted_split_assignments",
    "load_common_manifest",
    "load_common_record",
    "prepare_common_dataset",
]
