#!/usr/bin/env python3
"""Full-corpus Phase 9B.1 Dilemmadata raw-adapter acceptance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

from music_critic.adapters import (
    DILEMMADATA_ACCEPTANCE_REPORT_VERSION,
    DILEMMADATA_ADAPTER_VERSION,
    DILEMMADATA_DATASET_NAME,
    DILEMMADATA_PRIMARY_RECORD_COUNT,
    DILEMMADATA_RAW_PROJECTION_VERSION,
    DILEMMADATA_RELEASE_COMMIT,
    DilemmadataAccepted,
    DilemmadataAdapterConfig,
    DilemmadataCorpusIdentity,
    DilemmadataQuarantine,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
)
from music_critic.data import dumps_piece, loads_piece
from music_critic.graph import (
    build_raw_graph,
    graph_fingerprint,
    model_input_fingerprint,
)
from music_critic.tasks import (
    CORPUS_CACHE_INPUT_IDENTITY_VERSION,
    CanonicalCorpusInput,
    CorpusCacheConfig,
    CorpusQuarantineRecord,
    cache_canonical_corpus,
    corpus_cache_key,
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
    dumps_corpus_index,
    load_cached_piece,
    plan_group_hash_split,
    validate_split_manifest,
)


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


def _maximum_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _outside_corpus(root: Path, path: Path, name: str) -> None:
    if path == root or path.is_relative_to(root):
        raise ValueError(f"{name} must be outside the Dilemmadata root")


def _adapter_config() -> tuple[DilemmadataAdapterConfig, dict[str, object]]:
    config = DilemmadataAdapterConfig()
    return config, {
        **asdict(config),
        "raw_projection_version": DILEMMADATA_RAW_PROJECTION_VERSION,
        "cache_input_identity_version": CORPUS_CACHE_INPUT_IDENTITY_VERSION,
    }


def _build_report_summary(report: Any) -> dict[str, object]:
    value = asdict(report)
    value.pop("quarantine")
    value["quarantined_count"] = report.quarantined_count
    return value


def _source_identity(release_version: str | None) -> str:
    return f"dilemmadata:{release_version}:{DILEMMADATA_RELEASE_COMMIT}"


def _cache_mutation_evidence(index: Any, record: Any) -> dict[str, object]:
    common = {
        "source_identity": record.record_id,
        "source_sha256": record.physical_source_sha256,
        "adapter_name": "dilemmadata",
        "adapter_version": DILEMMADATA_ADAPTER_VERSION,
        "adapter_config_fingerprint": index.header.adapter_config_fingerprint,
        "cache_input_sha256": record.raw_projection_sha256,
    }
    baseline = corpus_cache_key(**common)
    physical = "0" * 64 if record.physical_source_sha256 != "0" * 64 else "f" * 64
    raw = "1" * 64 if record.raw_projection_sha256 != "1" * 64 else "e" * 64
    target_only = corpus_cache_key(**{**common, "source_sha256": physical})
    raw_changed = corpus_cache_key(**{**common, "cache_input_sha256": raw})
    return {
        "contract_kind": "deterministic_cache_key_mutation_probe",
        "physical_or_target_sidecar_mutation_keeps_raw_projection": {
            "cache_key_equal": baseline == target_only,
            "expected": True,
        },
        "raw_projection_mutation": {
            "cache_key_changed": baseline != raw_changed,
            "expected": True,
        },
    }


def _smoke_manifest(index: Any, accepted: dict[str, dict[str, object]]):
    source_counts = Counter(row.source_group_id for row in index.records)
    lineage_counts = Counter(row.lineage_group_id for row in index.records)
    candidates: dict[str, list[Any]] = {"an_joint": [], "dlc": []}
    for row in index.records:
        metadata = accepted[row.piece_id]
        if (
            source_counts[row.source_group_id] == 1
            and lineage_counts[row.lineage_group_id] == 1
        ):
            candidates[str(metadata["dialect"])].append(row)
    for dialect in candidates:
        candidates[dialect].sort(
            key=lambda row: (int(accepted[row.piece_id]["note_count"]), row.piece_id)
        )
        if len(candidates[dialect]) < 2:
            raise RuntimeError(f"not enough singleton {dialect} records for SSL smoke")
    selected = (*candidates["an_joint"][:2], *candidates["dlc"][:2])
    selected_ids = {row.piece_id for row in selected}
    assignments = {
        (row.dataset_id, row.piece_id): (
            "train" if row.piece_id in selected_ids else "validation"
        )
        for row in index.records
    }
    manifest = create_split_manifest(
        (index,),
        assignments,
        seed=9_001,
        policy="phase9b1_real_two_an_two_dlc_smoke",
        policy_config={"an_joint": 2, "dlc": 2, "selection": "smallest_singletons"},
    )
    validate_split_manifest(manifest, (index,))
    return manifest, tuple(selected)


def _run_ssl_smoke(
    *,
    repo_root: Path,
    work_dir: Path,
    cache_config: CorpusCacheConfig,
    index_path: Path,
    index: Any,
    accepted: dict[str, dict[str, object]],
) -> dict[str, object]:
    manifest, selected = _smoke_manifest(index, accepted)
    split_path = work_dir / "dilemmadata.ssl-smoke.split.json"
    dump_split_manifest(manifest, split_path)
    output = work_dir / "ssl-one-step"
    command = [
        sys.executable,
        "-m",
        "music_critic.ssl.run",
        "+phase8b_objective=onset_only",
        "+phase8b_masking=onset_only",
        "experiment=one_batch",
        "experiment.steps=1",
        "experiment.overwrite_output=true",
        "model=hierarchical",
        "model.hidden_dim=8",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=2",
        "model.ffn_multiplier=2",
        "model.dropout=0",
        "data=dilemmadata",
        f"data.index_paths=[{index_path}]",
        f"data.cache_roots=[{cache_config.root}]",
        f"data.split_manifest={split_path}",
        "data.batch_size=4",
        "data.epoch_size=4",
        "data.validation_epoch_size=1",
        "data.workers=0",
        "device=cpu",
        "optimizer.learning_rate=0.02",
        "optimizer.weight_decay=0",
        "ssl.mask_rate=0.5",
        "ssl.decoder_views=1",
        "ssl.decoder_remask_prob=0",
        "ssl.projector_hidden_dim=8",
        "ssl.decoder_hidden_dim=8",
        f"output_dir={output}",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        return {
            "passed": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2_000:],
            "selected_records": [
                {
                    "record_id": row.source_identity,
                    "piece_id": row.piece_id,
                    "dialect": accepted[row.piece_id]["dialect"],
                }
                for row in selected
            ],
        }
    report = json.loads((output / "one_batch_report.json").read_text(encoding="utf-8"))
    gradient = report.get("gradient_coverage") or {}
    encoder = (gradient.get("groups") or {}).get("online_encoder") or {}
    final_loss = report.get("final", {}).get("total_ssl_loss")
    exact_composition = Counter(
        str(accepted[row.piece_id]["dialect"]) for row in selected
    ) == {"an_joint": 2, "dlc": 2}
    passed = bool(
        report.get("mechanics_acceptance", {}).get("passed")
        and exact_composition
        and report.get("accounting", {}).get("optimizer_step_applied_count") == 1
        and final_loss is not None
        and math.isfinite(float(final_loss))
        and encoder.get("finite_gradient_count") == encoder.get("with_gradient_count")
        and int(encoder.get("nonzero_gradient_count", 0)) > 0
        and int(encoder.get("changed_parameter_count", 0)) > 0
        and report.get("final", {}).get("retained_prediction_tensor_count") == 0
    )
    return {
        "passed": passed,
        "returncode": completed.returncode,
        "selected_records": [
            {
                "record_id": row.source_identity,
                "piece_id": row.piece_id,
                "dialect": accepted[row.piece_id]["dialect"],
                "note_count": accepted[row.piece_id]["note_count"],
            }
            for row in selected
        ],
        "split_manifest_fingerprint": manifest.manifest_fingerprint,
        "mechanics_acceptance": report.get("mechanics_acceptance"),
        "optimizer_step_applied_count": report.get("accounting", {}).get(
            "optimizer_step_applied_count"
        ),
        "final_loss": final_loss,
        "loss_decreased": report.get("loss_decreased"),
        "online_encoder_gradient_evidence": encoder,
        "retained_prediction_tensor_count": report.get("final", {}).get(
            "retained_prediction_tensor_count"
        ),
        "retained_cuda_tensor_count": report.get("final", {}).get(
            "retained_cuda_tensor_count"
        ),
        "mask_plan_fingerprints": report.get("final", {}).get(
            "mask_plan_fingerprints"
        ),
        "input_sample_identities": report.get("final", {}).get(
            "input_sample_identities"
        ),
    }


def build_acceptance_report(
    root: Path,
    *,
    work_dir: Path,
    run_ssl_smoke: bool = True,
) -> dict[str, object]:
    started = time.monotonic()
    root = root.resolve()
    work_dir = work_dir.resolve(strict=False)
    _outside_corpus(root, work_dir, "work directory")
    work_dir.mkdir(parents=True, exist_ok=True)
    discovery = discover_dilemmadata_corpus(root, identity=DilemmadataCorpusIdentity())
    config, cache_adapter_config = _adapter_config()
    cache_config = CorpusCacheConfig(work_dir / "cache")
    status_counts: Counter[str] = Counter()
    dialect_status_counts: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    provenance_kind_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    quarantined: list[CorpusQuarantineRecord] = []
    failure_samples: list[dict[str, object]] = []
    accepted: dict[str, dict[str, object]] = {}
    raw_clusters: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    fatal: list[dict[str, object]] = []
    processed_bytes = 0

    def inputs():
        nonlocal processed_bytes
        for record in discovery.records:
            processed_bytes += record.path.stat().st_size
            try:
                outcome = convert_dilemmadata_record(record, config=config)
            except Exception as exc:
                fatal.append(
                    {
                        "record_id": record.record_id,
                        "category": f"{type(exc).__module__}.{type(exc).__name__}",
                        "message": " ".join(str(exc).split())[:240],
                    }
                )
                continue
            status_counts[outcome.status] += 1
            dialect_status_counts[(record.dialect, outcome.status)] += 1
            if isinstance(outcome, DilemmadataQuarantine):
                category_counts.update(outcome.categories)
                if len(failure_samples) < 32:
                    failure_samples.append(
                        {
                            "record_id": record.record_id,
                            "source_path": record.relative_path,
                            "categories": list(outcome.categories),
                            "messages": list(outcome.messages[:4]),
                        }
                    )
                quarantined.append(
                    CorpusQuarantineRecord(
                        dataset_id=record.dataset_name,
                        source_identity=record.record_id,
                        source_relative_path=record.relative_path,
                        category=outcome.categories[0],
                        message=(
                            "; ".join(outcome.messages)
                            if outcome.messages
                            else ", ".join(outcome.categories)
                        ),
                    )
                )
                continue
            assert isinstance(outcome, DilemmadataAccepted)
            piece = outcome.piece
            payload = dumps_piece(piece)
            if loads_piece(payload) != piece or dumps_piece(loads_piece(payload)) != payload:
                fatal.append(
                    {"record_id": record.record_id, "category": "serialization_round_trip"}
                )
                continue
            graph = build_raw_graph(piece, assume_valid=True)
            graph_sha = graph_fingerprint(graph)
            model_sha = model_input_fingerprint(graph)
            warning_counts.update(issue.code for issue in outcome.validation_report.warnings)
            quality_flag_counts.update(flag.code for flag in piece.quality_flags)
            provenance_kind_counts.update(row.kind for row in piece.provenance)
            totals.update(
                {
                    "source_note_rows": outcome.statistics.source_note_row_count,
                    "canonical_notes": outcome.statistics.canonical_note_count,
                    "tie_continuation_rows": outcome.statistics.tie_continuation_row_count,
                    "tie_merges": outcome.statistics.tie_merge_count,
                    "grace_notes": outcome.statistics.grace_note_count,
                    "meter_events": outcome.statistics.meter_event_count,
                    "bars": outcome.statistics.bar_count,
                    "beats": outcome.statistics.beat_count,
                    "pickup_bars": outcome.statistics.pickup_bar_count,
                    "incomplete_bars": outcome.statistics.incomplete_bar_count,
                    "graph_nodes": sum(int(graph[node].num_nodes) for node in graph.node_types),
                    "graph_edges": sum(
                        int(graph[edge].edge_index.shape[1]) for edge in graph.edge_types
                    ),
                }
            )
            canonical_sha = sha256(payload.encode("utf-8")).hexdigest()
            accepted[piece.piece_id] = {
                "record_id": record.record_id,
                "dialect": record.dialect,
                "note_count": len(piece.notes),
                "canonical_sha256": canonical_sha,
                "graph_fingerprint": graph_sha,
                "model_input_fingerprint": model_sha,
            }
            raw_clusters[record.raw_equivalence_id].append(
                {
                    "record_id": record.record_id,
                    "canonical_sha256": canonical_sha,
                    "graph_fingerprint": graph_sha,
                    "model_input_fingerprint": model_sha,
                }
            )
            yield CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=record.lineage_group_id,
                source_identity=record.record_id,
                source_relative_path=record.relative_path,
                source_sha256=outcome.record.physical_source_sha256,
                suggested_split=record.suggested_split,
                cache_input_sha256=record.raw_projection_sha256,
            )

    index, build_report = cache_canonical_corpus(
        inputs(),
        cache_config=cache_config,
        dataset_id=DILEMMADATA_DATASET_NAME,
        adapter_name="dilemmadata",
        adapter_version=DILEMMADATA_ADAPTER_VERSION,
        adapter_config=cache_adapter_config,
        source_identity=_source_identity(discovery.release_version),
        source_fingerprint=discovery.content_fingerprint,
        creation_policy="offline_full_corpus",
        quarantine=quarantined,
    )
    index_path = work_dir / "dilemmadata.index.json"
    dump_corpus_index(index, index_path)

    record_by_id = {row.record_id: row for row in discovery.records}

    def cached_inputs():
        for indexed in index.records:
            source = record_by_id[indexed.source_identity]
            yield CanonicalCorpusInput(
                piece=load_cached_piece(indexed, cache_config),
                lineage_group_id=indexed.lineage_group_id,
                source_identity=indexed.source_identity,
                source_relative_path=indexed.source_relative_path,
                source_sha256=indexed.source_sha256,
                suggested_split=indexed.suggested_split,
                cache_input_sha256=source.raw_projection_sha256,
            )

    rebuilt_index, rebuilt_report = cache_canonical_corpus(
        cached_inputs(),
        cache_config=cache_config,
        dataset_id=DILEMMADATA_DATASET_NAME,
        adapter_name="dilemmadata",
        adapter_version=DILEMMADATA_ADAPTER_VERSION,
        adapter_config=cache_adapter_config,
        source_identity=_source_identity(discovery.release_version),
        source_fingerprint=discovery.content_fingerprint,
        creation_policy="offline_full_corpus",
        quarantine=quarantined,
    )
    cached_rebuilt_equal = dumps_corpus_index(index) == dumps_corpus_index(rebuilt_index)
    split = plan_group_hash_split(
        (index,), seed=9_001, ratios={"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    validate_split_manifest(split, (index,))
    split_path = work_dir / "dilemmadata.split.json"
    dump_split_manifest(split, split_path)
    component_count = len({row.component_fingerprint for row in split.assignments})
    split_counts = Counter(row.split for row in split.assignments)
    split_component_counts = Counter()
    for component in {row.component_fingerprint for row in split.assignments}:
        split_component_counts[
            next(row.split for row in split.assignments if row.component_fingerprint == component)
        ] += 1

    cluster_rows = []
    for raw_id, rows in sorted(raw_clusters.items()):
        if len(rows) < 2:
            continue
        cluster_rows.append(
            {
                "raw_equivalence_id": raw_id,
                "record_ids": sorted(row["record_id"] for row in rows),
                "canonical_equal": len({row["canonical_sha256"] for row in rows}) == 1,
                "graph_equal": len({row["graph_fingerprint"] for row in rows}) == 1,
                "model_input_equal": len(
                    {row["model_input_fingerprint"] for row in rows}
                )
                == 1,
            }
        )
    cache_mutation = (
        _cache_mutation_evidence(index, discovery.records[0])
        if discovery.records
        else {"contract_kind": "unavailable"}
    )
    ssl_smoke = (
        _run_ssl_smoke(
            repo_root=Path(__file__).resolve().parents[1],
            work_dir=work_dir,
            cache_config=cache_config,
            index_path=index_path,
            index=index,
            accepted=accepted,
        )
        if run_ssl_smoke
        else {"passed": None, "skipped": True}
    )
    outcome_count = sum(status_counts.values()) + len(fatal)
    accepted_count = status_counts["accepted"]
    ready = bool(
        not fatal
        and outcome_count == DILEMMADATA_PRIMARY_RECORD_COUNT
        and accepted_count == len(index.records) == build_report.accepted_count
        and build_report.raw_only_piece_count == accepted_count
        and build_report.cache_miss_count == accepted_count
        and rebuilt_report.cache_hit_count == accepted_count
        and rebuilt_report.cache_miss_count == 0
        and cached_rebuilt_equal
        and component_count == len({row.source_group_id for row in index.records})
        and cache_mutation["physical_or_target_sidecar_mutation_keeps_raw_projection"][
            "cache_key_equal"
        ]
        and cache_mutation["raw_projection_mutation"]["cache_key_changed"]
        and (ssl_smoke.get("passed") is True if run_ssl_smoke else True)
    )
    semantic = {
        "acceptance_report_version": DILEMMADATA_ACCEPTANCE_REPORT_VERSION,
        "adapter_version": DILEMMADATA_ADAPTER_VERSION,
        "corpus_identity": {
            "content_fingerprint": discovery.content_fingerprint,
            "installation_file_count": discovery.installation_file_count,
            "primary_record_count": len(discovery.records),
        },
        "outcomes": {
            "discovered_count": len(discovery.records),
            "fatal_count": len(fatal),
            "status_counts": dict(sorted(status_counts.items())),
            "dialect_status_counts": {
                f"{dialect}:{status}": count
                for (dialect, status), count in sorted(dialect_status_counts.items())
            },
            "failure_category_counts": dict(sorted(category_counts.items())),
        },
        "totals": dict(sorted(totals.items())),
        "canonical_validation": {
            "validated_piece_count": len(index.records),
            "validation_error_count": 0,
            "deterministic_round_trip_count": len(index.records),
            "raw_graph_build_count": len(index.records),
            "graph_fingerprint_count": len(index.records),
            "model_input_fingerprint_count": len(index.records),
        },
        "cache": {
            "first": _build_report_summary(build_report),
            "rebuilt": _build_report_summary(rebuilt_report),
            "cached_rebuilt_equal": cached_rebuilt_equal,
            "mutation_evidence": cache_mutation,
        },
        "grouping_and_split": {
            "discovery_component_count": discovery.component_count,
            "discovery_multi_record_component_count": (
                discovery.multi_record_component_count
            ),
            "discovery_explicit_overlap_count": discovery.explicit_overlap_count,
            "discovery_suggested_split_conflict_count": (
                discovery.suggested_split_conflict_count
            ),
            "accepted_manifest_component_count": component_count,
            "accepted_source_group_count": len(
                {row.source_group_id for row in index.records}
            ),
            "split_record_counts": dict(sorted(split_counts.items())),
            "split_component_counts": dict(sorted(split_component_counts.items())),
            "manifest_fingerprint": split.manifest_fingerprint,
        },
        "raw_identity_recheck": {
            "duplicate_cluster_count": len(cluster_rows),
            "canonical_mismatch_count": sum(not row["canonical_equal"] for row in cluster_rows),
            "graph_mismatch_count": sum(not row["graph_equal"] for row in cluster_rows),
            "model_input_mismatch_count": sum(
                not row["model_input_equal"] for row in cluster_rows
            ),
            "bounded_cluster_samples": cluster_rows[:16],
        },
        "warnings_and_defaults": {
            "validation_warning_counts": dict(sorted(warning_counts.items())),
            "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
            "provenance_kind_counts": dict(sorted(provenance_kind_counts.items())),
        },
        "ssl_smoke": ssl_smoke,
        "ready": ready,
    }
    return {
        **semantic,
        "semantic_fingerprint": _fingerprint(semantic),
        "processed_primary_tsv_bytes": processed_bytes,
        "installation_byte_count": discovery.installation_byte_count,
        "maximum_rss_bytes": _maximum_rss_bytes(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "fatal_failure_count": len(fatal),
        "fatal_failure_samples": fatal[:32],
        "quarantine_samples": failure_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-ssl-smoke", action="store_true")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    output = arguments.output.resolve(strict=False)
    try:
        _outside_corpus(root, output, "output")
        report = build_acceptance_report(
            root,
            work_dir=arguments.work_dir,
            run_ssl_smoke=not arguments.skip_ssl_smoke,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "ready": report["ready"],
                "semantic_fingerprint": report["semantic_fingerprint"],
                "outcomes": report["outcomes"],
                "duration_seconds": report["duration_seconds"],
                "maximum_rss_bytes": report["maximum_rss_bytes"],
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
