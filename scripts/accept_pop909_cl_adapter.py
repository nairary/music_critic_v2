#!/usr/bin/env python3
"""Streaming Phase 4B production-adapter acceptance for pinned POP909-CL."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

from music_critic.adapters import (
    POP909_CL_ADAPTER_VERSION,
    POP909_CL_CONTENT_FINGERPRINT,
    Pop909ClAdapterConfig,
    convert_pop909_cl_file,
    discover_pop909_cl_corpus,
)
from music_critic.data import dumps_piece, loads_piece, piece_to_dict
from music_critic.graph import build_raw_graph, graph_fingerprint


ACCEPTANCE_SCHEMA_VERSION = "1.0.1"


def _raw_piece(piece: Any) -> Any:
    return replace(
        piece,
        annotations=(),
        targets=(),
        provenance=tuple(
            record
            for record in piece.provenance
            if not record.provenance_id.startswith("prov:pop909-cl")
        ),
        quality_flags=tuple(
            flag
            for flag in piece.quality_flags
            if not flag.code.startswith("pop909_cl.")
        ),
    )


def _anomaly_row(anomaly: Any) -> dict[str, Any]:
    return {
        "anomaly_id": anomaly.anomaly_id,
        "category": anomaly.category,
        "tick": anomaly.tick,
        "pitch": anomaly.pitch,
        "velocity": anomaly.velocity,
        "channel": anomaly.channel,
        "message_type": anomaly.message_type,
        "ordinal": anomaly.ordinal,
        "source_track_index": anomaly.source_track_index,
        "source_path": anomaly.source_path,
        "source_sha256": anomaly.source_sha256,
        "affected_block_onsets": list(anomaly.affected_block_onsets),
        "affected_span_ids": list(anomaly.affected_span_ids),
        "affected_interval": {
            "start_tick": anomaly.affected_interval_start_tick,
            "end_tick": anomaly.affected_interval_end_tick,
            "basis": anomaly.affected_interval_basis,
        },
    }


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _raw_canonical_fingerprint(piece: Any) -> str:
    value = piece_to_dict(piece)
    for field in (
        "piece_id",
        "source_group_id",
        "source_path",
        "annotations",
        "targets",
        "provenance",
    ):
        value.pop(field, None)
    for flag in value["quality_flags"]:
        flag["entity_ids"] = [
            "<piece>" if item.startswith("piece:") else item
            for item in flag["entity_ids"]
        ]
        flag["provenance_id"] = None
    return _fingerprint(value)


def _target_evidence(piece: Any) -> dict[str, dict[str, object]]:
    return {
        target.task: {
            "available_count": sum(target.mask),
            "masked_count": len(target.mask) - sum(target.mask),
            "values_fingerprint": _fingerprint(target.values),
            "mask_fingerprint": _fingerprint(target.mask),
            "values_and_masks_fingerprint": _fingerprint(
                {
                    "values": target.values,
                    "mask": target.mask,
                }
            ),
        }
        for target in piece.targets
    }


def _load_expectations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("acceptance_schema_version") != ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported acceptance manifest: {path}")
    return payload


def build_acceptance_report(root: Path, expectations: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    discovery = discover_pop909_cl_corpus(root)
    statuses: Counter[str] = Counter()
    missing_target_ids: list[str] = []
    quarantine_rows: list[dict[str, Any]] = []
    fatal_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    total_blocks = 0
    ambiguous_blocks = 0
    unsupported_blocks = 0
    n_spans = 0
    trailing_spans = 0
    boundary_available = 0
    bass_available = 0
    root_available = 0
    quality_available = 0
    inversion_available = 0
    chord_instruments = 0
    validated = 0
    round_trips = 0
    raw_equal = 0
    graph_equal = 0
    accepted_identity_rows: list[dict[str, Any]] = []

    for record in discovery.records:
        try:
            visible = convert_pop909_cl_file(record)
            hidden = convert_pop909_cl_file(
                record,
                config=Pop909ClAdapterConfig(include_targets=False),
            )
            statuses[visible.status] += 1
            evidence = visible.chord_evidence
            chord_instruments += int(bool(visible.instrument_resolution.chord_track_indices))
            total_blocks += len(evidence.blocks)
            ambiguous_blocks += sum(
                block.normalization_status == "ambiguous"
                for block in evidence.blocks
            )
            unsupported_blocks += sum(
                block.normalization_status == "unsupported"
                for block in evidence.blocks
            )
            n_spans += len(evidence.no_chord_spans)
            trailing_spans += sum(
                span.kind == "trailing_unannotated"
                for span in evidence.trailing_spans
            )
            boundary_available += len(evidence.blocks)
            bass_available += len(evidence.blocks)
            root_available += sum(block.root_available for block in evidence.blocks)
            quality_available += sum(
                block.quality_available for block in evidence.blocks
            )
            inversion_available += sum(
                block.inversion_available for block in evidence.blocks
            )
            anomaly_rows.extend(
                _anomaly_row(anomaly) for anomaly in evidence.pairing_anomalies
            )
            if visible.status == "quarantined":
                if hidden.status != "quarantined":
                    fatal_rows.append(
                        {
                            "song_id": record.song_id,
                            "category": "hidden_quarantine_mismatch",
                        }
                    )
                quarantine_rows.append(
                    {
                        "song_id": record.song_id,
                        "source_path": record.relative_path,
                        "source_sha256": record.sha256,
                        "source_group_id": record.source_group_id,
                        "lineage_group_id": record.lineage_group_id,
                        "category": visible.category,
                        "source_error_type": visible.source_error_type,
                        "source_error": visible.source_error,
                    }
                )
                continue
            if hidden.status == "quarantined":
                fatal_rows.append(
                    {
                        "song_id": record.song_id,
                        "category": "visible_hidden_status_mismatch",
                    }
                )
                continue
            if visible.status == "accepted_missing_targets":
                missing_target_ids.append(record.song_id)
            if visible.validation_report.errors or hidden.validation_report.errors:
                fatal_rows.append(
                    {
                        "song_id": record.song_id,
                        "category": "canonical_validation",
                    }
                )
                continue
            validated += 1
            visible_payload = dumps_piece(visible.piece)
            hidden_payload = dumps_piece(hidden.piece)
            if (
                loads_piece(visible_payload) != visible.piece
                or dumps_piece(loads_piece(visible_payload)) != visible_payload
                or loads_piece(hidden_payload) != hidden.piece
                or dumps_piece(loads_piece(hidden_payload)) != hidden_payload
            ):
                fatal_rows.append(
                    {
                        "song_id": record.song_id,
                        "category": "serialization_round_trip",
                    }
                )
                continue
            round_trips += 1
            if _raw_piece(visible.piece) != hidden.piece:
                fatal_rows.append(
                    {
                        "song_id": record.song_id,
                        "category": "target_hiding_raw_mismatch",
                    }
                )
                continue
            raw_equal += 1
            visible_fingerprint = graph_fingerprint(
                build_raw_graph(visible.piece, assume_valid=True)
            )
            hidden_fingerprint = graph_fingerprint(
                build_raw_graph(hidden.piece, assume_valid=True)
            )
            if visible_fingerprint != hidden_fingerprint:
                fatal_rows.append(
                    {
                        "song_id": record.song_id,
                        "category": "target_hiding_graph_mismatch",
                    }
                )
                continue
            graph_equal += 1
            graph = build_raw_graph(
                visible.piece, assume_valid=True
            )
            accepted_identity_rows.append(
                {
                    "song_id": visible.record.song_id,
                    "piece_id": visible.piece.piece_id,
                    "source_sha256": visible.record.sha256,
                    "raw_input_equivalence_id": (
                        visible.record.source_group_id
                    ),
                    "score_projection_sha256": (
                        visible.score_projection_sha256
                    ),
                    "raw_canonical_fingerprint": (
                        _raw_canonical_fingerprint(visible.piece)
                    ),
                    "graph_fingerprint": graph_fingerprint(graph),
                    "node_counts": {
                        node_type: int(graph[node_type].num_nodes)
                        for node_type in graph.node_types
                    },
                    "edge_counts": {
                        "|".join(edge_type): int(
                            graph[edge_type].edge_index.shape[1]
                        )
                        for edge_type in graph.edge_types
                    },
                    "chord_block_count": len(evidence.blocks),
                    "target_bundle_fingerprint": _fingerprint(
                        [asdict(target) for target in visible.piece.targets]
                    ),
                    "targets": _target_evidence(visible.piece),
                    "diagnostics": {
                        "ambiguous_block_count": sum(
                            block.normalization_status == "ambiguous"
                            for block in evidence.blocks
                        ),
                        "unsupported_block_count": sum(
                            block.normalization_status == "unsupported"
                            for block in evidence.blocks
                        ),
                        "pairing_anomaly_count": len(
                            evidence.pairing_anomalies
                        ),
                        "repeated_pitch_block_count": (
                            evidence.repeated_pitch_block_count
                        ),
                        "mixed_end_block_count": (
                            evidence.mixed_end_block_count
                        ),
                        "overlap_count": evidence.overlap_count,
                    },
                }
            )
        except Exception as exc:
            fatal_rows.append(
                {
                    "song_id": record.song_id,
                    "category": f"{type(exc).__module__}.{type(exc).__name__}",
                    "message": " ".join(str(exc).split())[:500],
                }
            )

    anomaly_rows.sort(
        key=lambda row: (
            row["source_path"],
            row["tick"],
            row["ordinal"],
            row["category"],
        )
    )
    anomaly_fingerprint = sha256(
        json.dumps(
            anomaly_rows,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    by_raw_input: dict[str, list[dict[str, Any]]] = {}
    for row in accepted_identity_rows:
        by_raw_input.setdefault(
            str(row["raw_input_equivalence_id"]), []
        ).append(row)
    duplicate_clusters = []
    for rows in by_raw_input.values():
        if len(rows) < 2:
            continue
        ordered_rows = sorted(rows, key=lambda row: row["song_id"])
        task_ids = sorted(
            {
                task_id
                for row in ordered_rows
                for task_id in row["targets"]
            }
        )
        duplicate_clusters.append(
            {
                "song_ids": [
                    row["song_id"] for row in ordered_rows
                ],
                "piece_ids": [
                    row["piece_id"] for row in ordered_rows
                ],
                "source_sha256": [
                    row["source_sha256"] for row in ordered_rows
                ],
                "score_projection_sha256": ordered_rows[0][
                    "score_projection_sha256"
                ],
                "score_projection_equal": len(
                    {
                        row["score_projection_sha256"]
                        for row in ordered_rows
                    }
                )
                == 1,
                "canonical_raw_equal": len(
                    {
                        row["raw_canonical_fingerprint"]
                        for row in ordered_rows
                    }
                )
                == 1,
                "graph_fingerprints": [
                    row["graph_fingerprint"]
                    for row in ordered_rows
                ],
                "graph_fingerprints_equal": len(
                    {
                        row["graph_fingerprint"]
                        for row in ordered_rows
                    }
                )
                == 1,
                "node_counts": [
                    row["node_counts"] for row in ordered_rows
                ],
                "edge_counts": [
                    row["edge_counts"] for row in ordered_rows
                ],
                "chord_block_counts": [
                    row["chord_block_count"]
                    for row in ordered_rows
                ],
                "target_bundle_fingerprints": [
                    row["target_bundle_fingerprint"]
                    for row in ordered_rows
                ],
                "target_values_and_masks_equal": {
                    task_id: len(
                        {
                            row["targets"][task_id][
                                "values_and_masks_fingerprint"
                            ]
                            for row in ordered_rows
                        }
                    )
                    == 1
                    for task_id in task_ids
                },
                "target_values_equal": {
                    task_id: len(
                        {
                            row["targets"][task_id][
                                "values_fingerprint"
                            ]
                            for row in ordered_rows
                        }
                    )
                    == 1
                    for task_id in task_ids
                },
                "availability_masks_equal": {
                    task_id: len(
                        {
                            row["targets"][task_id][
                                "mask_fingerprint"
                            ]
                            for row in ordered_rows
                        }
                    )
                    == 1
                    for task_id in task_ids
                },
                "target_available_counts": {
                    task_id: [
                        row["targets"][task_id][
                            "available_count"
                        ]
                        for row in ordered_rows
                    ]
                    for task_id in task_ids
                },
                "target_masked_counts": {
                    task_id: [
                        row["targets"][task_id]["masked_count"]
                        for row in ordered_rows
                    ]
                    for task_id in task_ids
                },
                "diagnostics": [
                    row["diagnostics"] for row in ordered_rows
                ],
            }
        )
    duplicate_clusters.sort(key=lambda row: row["song_ids"])
    observed = {
        "logical_files": len(discovery.records),
        "accepted": statuses["accepted"] + statuses["accepted_missing_targets"],
        "accepted_with_targets": statuses["accepted"],
        "accepted_missing_targets": statuses["accepted_missing_targets"],
        "quarantined": statuses["quarantined"],
        "quarantine_count": len(quarantine_rows),
        "chord_instruments": chord_instruments,
        "missing_target_song_ids": sorted(missing_target_ids),
        "quarantine_song_ids": sorted(row["song_id"] for row in quarantine_rows),
        "chord_blocks": total_blocks,
        "root_available": root_available,
        "quality_available": quality_available,
        "bass_available": bass_available,
        "boundary_available": boundary_available,
        "inversion_available": inversion_available,
        "ambiguous_blocks": ambiguous_blocks,
        "unsupported_blocks": unsupported_blocks,
        "derived_n_spans": n_spans,
        "trailing_masked_spans": trailing_spans,
        "anomaly_evidence_fingerprint": anomaly_fingerprint,
        "validated_pieces": validated,
        "deterministic_round_trips": round_trips,
        "target_hidden_raw_equal": raw_equal,
        "target_hidden_graph_equal": graph_equal,
        "accepted_record_count": len(accepted_identity_rows),
        "unique_record_piece_id_count": len(
            {row["piece_id"] for row in accepted_identity_rows}
        ),
        "unique_raw_input_equivalence_count": len(by_raw_input),
        "raw_input_duplicate_cluster_count": len(
            duplicate_clusters
        ),
        "raw_input_duplicate_record_count": sum(
            len(row["song_ids"]) for row in duplicate_clusters
        ),
        "duplicate_cluster_song_ids": [
            row["song_ids"] for row in duplicate_clusters
        ],
        "duplicate_cluster_evidence": duplicate_clusters,
    }
    expected = expectations["expected"]
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    manifest_adapter_version = expectations.get("adapter_version")
    if manifest_adapter_version != POP909_CL_ADAPTER_VERSION:
        mismatches["adapter_version"] = {
            "expected": manifest_adapter_version,
            "observed": POP909_CL_ADAPTER_VERSION,
        }
    manifest_fingerprint = expectations.get("corpus", {}).get(
        "content_fingerprint"
    )
    if manifest_fingerprint != discovery.content_fingerprint:
        mismatches["manifest_content_fingerprint"] = {
            "expected": manifest_fingerprint,
            "observed": discovery.content_fingerprint,
        }
    return {
        "acceptance_schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "adapter_version": POP909_CL_ADAPTER_VERSION,
        "corpus_content_fingerprint": discovery.content_fingerprint,
        "expected_corpus_content_fingerprint": POP909_CL_CONTENT_FINGERPRINT,
        "ready": (
            not fatal_rows
            and not mismatches
            and discovery.content_fingerprint == POP909_CL_CONTENT_FINGERPRINT
        ),
        "observed": observed,
        "mismatches": mismatches,
        "fatal_failure_count": len(fatal_rows),
        "fatal_failure_samples": fatal_rows[:10],
        "quarantine": quarantine_rows,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/fixtures/pop909_cl/production_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve(strict=False)
    if output == root or output.is_relative_to(root):
        parser.error("output must be outside the POP909-CL dataset root")
    report = build_acceptance_report(root, _load_expectations(args.manifest))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "ready": report["ready"],
                "observed": report["observed"],
                "fatal_failure_count": report["fatal_failure_count"],
                "mismatches": report["mismatches"],
                "duration_seconds": report["duration_seconds"],
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
