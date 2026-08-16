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
    DILEMMADATA_AN_RECORD_COUNT,
    DILEMMADATA_CONTENT_FINGERPRINT,
    DILEMMADATA_CORPUS_IDENTITY_VERSION,
    DILEMMADATA_DATASET_NAME,
    DILEMMADATA_DLC_RECORD_COUNT,
    DILEMMADATA_GROUPING_VERSION,
    DILEMMADATA_INSTALLATION_FILE_COUNT,
    DILEMMADATA_PRIMARY_RECORD_COUNT,
    DILEMMADATA_PRODUCTION_MANIFEST_VERSION,
    DILEMMADATA_RAW_PROJECTION_VERSION,
    DILEMMADATA_RECORD_BINDING_VERSION,
    DILEMMADATA_RELEASE_COMMIT,
    DILEMMADATA_RELEASE_VERSION,
    DilemmadataAccepted,
    DilemmadataAdapterConfig,
    DilemmadataCorpusIdentity,
    DilemmadataQuarantine,
    convert_dilemmadata_record,
    discover_dilemmadata_corpus,
)
from music_critic.data import SCHEMA_VERSION, dumps_piece, loads_piece
from music_critic.graph import (
    FEATURE_REGISTRY_VERSION,
    GRAPH_BUILDER_VERSION,
    GRAPH_SCHEMA_VERSION,
    MODEL_INPUT_FINGERPRINT_VERSION,
    build_raw_graph,
    graph_fingerprint,
    model_input_fingerprint,
)
from music_critic.tasks import (
    CORPUS_CACHE_INPUT_IDENTITY_VERSION,
    MULTISOURCE_CACHE_VERSION,
    MULTISOURCE_CORPUS_INDEX_VERSION,
    SPLIT_MANIFEST_VERSION,
    TARGET_ENCODING_REGISTRY_VERSION,
    TARGET_ONTOLOGY_VERSION,
    CanonicalCorpusInput,
    CorpusCacheConfig,
    CorpusQuarantineRecord,
    cache_canonical_corpus,
    corpus_cache_key,
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
    dumps_corpus_index,
    plan_group_hash_split,
    validate_split_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "tests" / "fixtures" / "dilemmadata" / "production_manifest.json"
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


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    accounting = report.get("accounting") or {}
    final_loss = report.get("final", {}).get("total_ssl_loss")
    exact_composition = Counter(
        str(accepted[row.piece_id]["dialect"]) for row in selected
    ) == {"an_joint": 2, "dlc": 2}
    passed = bool(
        report.get("mechanics_acceptance", {}).get("passed")
        and exact_composition
        and accounting.get("optimizer_step_attempt_count") == 1
        and accounting.get("optimizer_step_applied_count") == 1
        and accounting.get("optimizer_step_skipped_count") == 0
        and final_loss is not None
        and math.isfinite(float(final_loss))
        and encoder.get("finite_gradient_count") == encoder.get("with_gradient_count")
        and int(encoder.get("nonzero_gradient_count", 0)) > 0
        and int(encoder.get("changed_parameter_count", 0)) > 0
        and report.get("final", {}).get("retained_prediction_tensor_count") == 0
        and report.get("final", {}).get("retained_cuda_tensor_count") == 0
        and sum(int(accepted[row.piece_id]["target_count"]) for row in selected) == 0
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
        "optimizer_step_attempted_count": accounting.get(
            "optimizer_step_attempt_count"
        ),
        "optimizer_step_applied_count": accounting.get("optimizer_step_applied_count"),
        "optimizer_step_skipped_count": accounting.get("optimizer_step_skipped_count"),
        "theory_target_access_count": sum(
            int(accepted[row.piece_id]["target_count"]) for row in selected
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


def _discovery_projection(discovery: Any) -> dict[str, object]:
    return {
        "content_fingerprint": discovery.content_fingerprint,
        "installation_file_count": discovery.installation_file_count,
        "release_version": discovery.release_version,
        "component_count": discovery.component_count,
        "multi_record_component_count": discovery.multi_record_component_count,
        "explicit_overlap_count": discovery.explicit_overlap_count,
        "suggested_split_conflict_count": discovery.suggested_split_conflict_count,
        "record_bindings": [
            [row.record_id, row.record_binding_sha256] for row in discovery.records
        ],
    }


def _artifact_snapshot(index: Any, cache_config: CorpusCacheConfig) -> tuple[object, ...]:
    cache_root = (cache_config.root / cache_config.namespace).resolve()
    rows: list[object] = []
    for record in index.records:
        path = (cache_root / record.canonical_relative_path).resolve()
        stat = path.stat()
        rows.append(
            (
                record.canonical_relative_path,
                stat.st_size,
                stat.st_mtime_ns,
                _hash_file(path),
            )
        )
    return tuple(rows)


def _run_source_build(
    discovery: Any,
    *,
    config: DilemmadataAdapterConfig,
    cache_adapter_config: dict[str, object],
    cache_config: CorpusCacheConfig,
    collect_full_evidence: bool,
) -> dict[str, object]:
    status_counts: Counter[str] = Counter()
    dialect_status_counts: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    provenance_kind_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    quarantined: list[CorpusQuarantineRecord] = []
    quarantine_projection: list[dict[str, object]] = []
    failure_samples: list[dict[str, object]] = []
    accepted: dict[str, dict[str, object]] = {}
    raw_clusters: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    fatal: list[dict[str, object]] = []
    outcome_projection: list[dict[str, object]] = []
    processed_bytes = 0

    def inputs():
        nonlocal processed_bytes
        for record in discovery.records:
            processed_bytes += record.path.stat().st_size
            try:
                outcome = convert_dilemmadata_record(record, config=config)
            except Exception as exc:
                fatal_row = {
                    "record_id": record.record_id,
                    "category": f"{type(exc).__module__}.{type(exc).__name__}",
                    "message": " ".join(str(exc).split())[:240],
                }
                fatal.append(fatal_row)
                outcome_projection.append(
                    {
                        "record_id": record.record_id,
                        "dialect": record.dialect,
                        "status": "fatal",
                        "category": fatal_row["category"],
                    }
                )
                continue
            status_counts[outcome.status] += 1
            dialect_status_counts[(record.dialect, outcome.status)] += 1
            if isinstance(outcome, DilemmadataQuarantine):
                category_counts.update(outcome.categories)
                row = {
                    "record_id": record.record_id,
                    "dialect": record.dialect,
                    "categories": list(outcome.categories),
                }
                quarantine_projection.append(row)
                outcome_projection.append({**row, "status": "quarantined"})
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
                outcome_projection.append(
                    {
                        "record_id": record.record_id,
                        "dialect": record.dialect,
                        "status": "fatal",
                        "category": "serialization_round_trip",
                    }
                )
                continue
            canonical_sha = sha256(payload.encode("utf-8")).hexdigest()
            outcome_projection.append(
                {
                    "record_id": record.record_id,
                    "dialect": record.dialect,
                    "status": "accepted",
                    "canonical_sha256": canonical_sha,
                    "statistics": asdict(outcome.statistics),
                    "warning_codes": [
                        issue.code for issue in outcome.validation_report.warnings
                    ],
                }
            )
            graph_sha = ""
            model_sha = ""
            if collect_full_evidence:
                graph = build_raw_graph(piece, assume_valid=True)
                graph_sha = graph_fingerprint(graph)
                model_sha = model_input_fingerprint(graph)
                warning_counts.update(
                    issue.code for issue in outcome.validation_report.warnings
                )
                quality_flag_counts.update(flag.code for flag in piece.quality_flags)
                provenance_kind_counts.update(row.kind for row in piece.provenance)
                totals.update(
                    {
                        "source_note_rows": outcome.statistics.source_note_row_count,
                        "canonical_notes": outcome.statistics.canonical_note_count,
                        "tie_continuation_rows": (
                            outcome.statistics.tie_continuation_row_count
                        ),
                        "tie_merges": outcome.statistics.tie_merge_count,
                        "grace_notes": outcome.statistics.grace_note_count,
                        "meter_events": outcome.statistics.meter_event_count,
                        "bars": outcome.statistics.bar_count,
                        "beats": outcome.statistics.beat_count,
                        "pickup_bars": outcome.statistics.pickup_bar_count,
                        "incomplete_bars": outcome.statistics.incomplete_bar_count,
                        "graph_nodes": sum(
                            int(graph[node].num_nodes) for node in graph.node_types
                        ),
                        "graph_edges": sum(
                            int(graph[edge].edge_index.shape[1])
                            for edge in graph.edge_types
                        ),
                    }
                )
            accepted[piece.piece_id] = {
                "record_id": record.record_id,
                "dialect": record.dialect,
                "note_count": len(piece.notes),
                "target_count": len(piece.targets),
                "canonical_sha256": canonical_sha,
                "graph_fingerprint": graph_sha,
                "model_input_fingerprint": model_sha,
            }
            if collect_full_evidence:
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
                lineage_group_id=outcome.record.lineage_group_id,
                source_identity=outcome.record.record_id,
                source_relative_path=outcome.record.relative_path,
                source_sha256=outcome.record.physical_source_sha256,
                suggested_split=outcome.record.suggested_split,
                cache_input_sha256=outcome.record.raw_projection_sha256,
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
    return {
        "index": index,
        "build_report": build_report,
        "status_counts": status_counts,
        "dialect_status_counts": dialect_status_counts,
        "category_counts": category_counts,
        "warning_counts": warning_counts,
        "quality_flag_counts": quality_flag_counts,
        "provenance_kind_counts": provenance_kind_counts,
        "totals": totals,
        "quarantine_projection": quarantine_projection,
        "failure_samples": failure_samples,
        "accepted": accepted,
        "raw_clusters": raw_clusters,
        "fatal": fatal,
        "processed_bytes": processed_bytes,
        "semantic_projection_fingerprint": _fingerprint(outcome_projection),
    }


def _normalized_ssl_smoke(value: dict[str, object]) -> dict[str, object]:
    ssl = dict(value)
    applied = ssl.get("optimizer_step_applied_count")
    skipped = ssl.get("optimizer_step_skipped_count")
    attempted = ssl.get("optimizer_step_attempted_count")
    if (
        attempted is None
        and isinstance(applied, int)
        and not isinstance(applied, bool)
        and isinstance(skipped, int)
        and not isinstance(skipped, bool)
    ):
        attempted = applied + skipped
        ssl["optimizer_step_attempted_count"] = attempted
    selected = Counter(row["dialect"] for row in ssl.get("selected_records", []))
    gradient = ssl.get("online_encoder_gradient_evidence") or {}
    mechanics = ssl.get("mechanics_acceptance") or {}
    final_loss = ssl.get("final_loss")
    ssl["passed"] = bool(
        mechanics.get("passed") is True
        and selected == {"an_joint": 2, "dlc": 2}
        and attempted == 1
        and applied == 1
        and skipped == 0
        and final_loss is not None
        and math.isfinite(float(final_loss))
        and gradient.get("finite_gradient_count")
        == gradient.get("with_gradient_count")
        and int(gradient.get("nonzero_gradient_count", 0)) > 0
        and int(gradient.get("changed_parameter_count", 0)) > 0
        and ssl.get("theory_target_access_count") == 0
        and ssl.get("retained_prediction_tensor_count") == 0
        and ssl.get("retained_cuda_tensor_count") == 0
    )
    return ssl


def _report_intrinsic_ready(report: dict[str, object]) -> bool:
    outcomes = report["outcomes"]
    statuses = outcomes["status_counts"]
    cache = report["cache"]
    first = cache["first"]
    second = cache["second_source"]
    grouping = report["grouping_and_split"]
    validation = report["canonical_validation"]
    mutation = cache["mutation_evidence"]
    accepted = statuses.get("accepted", 0)
    quarantined = statuses.get("quarantined", 0)
    return bool(
        outcomes["discovered_count"] == DILEMMADATA_PRIMARY_RECORD_COUNT
        and outcomes["fatal_count"] == 0
        and report["fatal_failure_count"] == 0
        and accepted + quarantined == DILEMMADATA_PRIMARY_RECORD_COUNT
        and accepted == first["accepted_count"] == second["accepted_count"]
        and first["raw_only_piece_count"] == accepted
        and first["cache_hit_count"] == 0
        and first["cache_miss_count"] == accepted
        and second["cache_hit_count"] == accepted
        and second["cache_miss_count"] == 0
        and cache["corpus_index_byte_identical"] is True
        and cache["discovery_projection_identical"] is True
        and cache["quarantine_identity_categories_identical"] is True
        and cache["source_build_semantic_projection_identical"] is True
        and cache["immutable_artifacts_unchanged"] is True
        and grouping["discovery_component_count"] == 1_507
        and grouping["discovery_multi_record_component_count"] == 126
        and grouping["discovery_explicit_overlap_count"] == 98
        and grouping["discovery_suggested_split_conflict_count"] == 5
        and grouping["accepted_manifest_component_count"]
        == grouping["accepted_source_group_count"]
        and grouping["non_empty_train_validation_test"] is True
        and validation["validated_piece_count"] == accepted
        and validation["validation_error_count"] == 0
        and validation["deterministic_round_trip_count"] == accepted
        and validation["raw_graph_build_count"] == accepted
        and mutation["physical_or_target_sidecar_mutation_keeps_raw_projection"][
            "cache_key_equal"
        ]
        is True
        and mutation["raw_projection_mutation"]["cache_key_changed"] is True
        and report["ssl_smoke"]["passed"] is True
    )


def _revalidate_report(report: dict[str, object]) -> dict[str, object]:
    revalidated = dict(report)
    ssl = report.get("ssl_smoke")
    if not isinstance(ssl, dict):
        raise ValueError("acceptance report has invalid SSL smoke evidence")
    revalidated["ssl_smoke"] = _normalized_ssl_smoke(ssl)
    revalidated["intrinsic_ready"] = _report_intrinsic_ready(revalidated)
    revalidated["ready"] = False
    revalidated["manifest_check"] = {"checked": False, "passed": False}
    revalidated.pop("semantic_acceptance_fingerprint", None)
    revalidated.pop("semantic_fingerprint", None)
    return revalidated


def _manifest_core(report: dict[str, object]) -> dict[str, object]:
    cache = report["cache"]
    grouping = report["grouping_and_split"]
    ssl = report["ssl_smoke"]
    gradient = ssl.get("online_encoder_gradient_evidence") or {}
    selected = Counter(row["dialect"] for row in ssl.get("selected_records", []))
    return {
        "manifest_version": DILEMMADATA_PRODUCTION_MANIFEST_VERSION,
        "contracts": report["contracts"],
        "pinned_corpus_identity": report["corpus_identity"],
        "outcomes": report["outcomes"],
        "accepted_totals": report["totals"],
        "grouping": {
            "discovery_component_count": grouping["discovery_component_count"],
            "discovery_multi_record_component_count": grouping[
                "discovery_multi_record_component_count"
            ],
            "discovery_explicit_overlap_count": grouping[
                "discovery_explicit_overlap_count"
            ],
            "discovery_suggested_split_conflict_count": grouping[
                "discovery_suggested_split_conflict_count"
            ],
            "accepted_component_count": grouping["accepted_manifest_component_count"],
        },
        "cache": {
            "first_hit_count": cache["first"]["cache_hit_count"],
            "first_miss_count": cache["first"]["cache_miss_count"],
            "second_source_hit_count": cache["second_source"]["cache_hit_count"],
            "second_source_miss_count": cache["second_source"]["cache_miss_count"],
            "index_fingerprint": cache["first"]["index_fingerprint"],
            "corpus_index_byte_identical": cache["corpus_index_byte_identical"],
            "quarantine_identity_categories_identical": cache[
                "quarantine_identity_categories_identical"
            ],
            "source_build_semantic_projection_identical": cache[
                "source_build_semantic_projection_identical"
            ],
            "immutable_artifacts_unchanged": cache[
                "immutable_artifacts_unchanged"
            ],
        },
        "split": {
            "fingerprint": grouping["manifest_fingerprint"],
            "record_counts": grouping["split_record_counts"],
            "component_counts": grouping["split_component_counts"],
            "non_empty_train_validation_test": grouping[
                "non_empty_train_validation_test"
            ],
        },
        "ssl_smoke": {
            "composition": dict(sorted(selected.items())),
            "mechanics_passed": ssl.get("passed") is True,
            "optimizer_step_attempted_count": ssl.get(
                "optimizer_step_attempted_count"
            ),
            "optimizer_step_applied_count": ssl.get("optimizer_step_applied_count"),
            "optimizer_step_skipped_count": ssl.get("optimizer_step_skipped_count"),
            "finite_loss": (
                ssl.get("final_loss") is not None
                and math.isfinite(float(ssl["final_loss"]))
            ),
            "loss_decreased": ssl.get("loss_decreased") is True,
            "nonzero_encoder_gradients": int(
                gradient.get("nonzero_gradient_count", 0)
            )
            > 0,
            "changed_encoder_parameters": int(
                gradient.get("changed_parameter_count", 0)
            )
            > 0,
            "theory_target_access_count": ssl.get("theory_target_access_count"),
            "retained_prediction_tensor_count": ssl.get(
                "retained_prediction_tensor_count"
            ),
            "retained_cuda_tensor_count": ssl.get("retained_cuda_tensor_count"),
        },
    }


def manifest_projection(report: dict[str, object]) -> dict[str, object]:
    """Return the deterministic, path/runtime/RSS/loss-value-free contract."""

    core = _manifest_core(report)
    return {
        **core,
        "semantic_acceptance_fingerprint": _fingerprint(core),
    }


def check_manifest_projection(
    actual: dict[str, object], manifest_path: Path
) -> tuple[bool, str | None, str, str]:
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read production manifest: {type(exc).__name__}") from exc
    required = {
        "manifest_version",
        "contracts",
        "pinned_corpus_identity",
        "outcomes",
        "accepted_totals",
        "grouping",
        "cache",
        "split",
        "ssl_smoke",
        "semantic_acceptance_fingerprint",
    }
    for label, value in (("expected", expected), ("actual", actual)):
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError(f"{label} production manifest has an invalid shape")
        if value["manifest_version"] != DILEMMADATA_PRODUCTION_MANIFEST_VERSION:
            raise ValueError(f"{label} production manifest version is unsupported")
        core = dict(value)
        semantic = core.pop("semantic_acceptance_fingerprint")
        if semantic != _fingerprint(core):
            raise ValueError(
                f"{label} production manifest semantic fingerprint is inconsistent"
            )
    expected_fingerprint = _fingerprint(expected)
    actual_fingerprint = _fingerprint(actual)
    if actual == expected:
        return True, None, expected_fingerprint, actual_fingerprint
    return (
        False,
        (
            "manifest mismatch: expected "
            f"{expected_fingerprint}, actual {actual_fingerprint}"
        ),
        expected_fingerprint,
        actual_fingerprint,
    )


def check_manifest(
    report: dict[str, object], manifest_path: Path
) -> tuple[bool, str | None, str, str]:
    return check_manifest_projection(manifest_projection(report), manifest_path)


def _apply_manifest_check(
    report: dict[str, object], manifest_path: Path
) -> dict[str, object]:
    actual = manifest_projection(report)
    semantic_fingerprint = actual["semantic_acceptance_fingerprint"]
    stored = report.get("semantic_acceptance_fingerprint")
    if stored is not None and stored != semantic_fingerprint:
        raise ValueError("report semantic acceptance fingerprint is internally inconsistent")
    matches, message, expected_fingerprint, actual_fingerprint = (
        check_manifest_projection(actual, manifest_path)
    )
    checked = dict(report)
    checked["semantic_acceptance_fingerprint"] = semantic_fingerprint
    checked["semantic_fingerprint"] = semantic_fingerprint
    checked["manifest_check"] = {
        "checked": True,
        "passed": matches,
        "manifest_version": DILEMMADATA_PRODUCTION_MANIFEST_VERSION,
        "expected_manifest_fingerprint": expected_fingerprint,
        "actual_manifest_fingerprint": actual_fingerprint,
        "message": message,
    }
    checked["ready"] = bool(checked.get("intrinsic_ready") and matches)
    return checked


def build_acceptance_report(
    root: Path,
    *,
    work_dir: Path,
    run_ssl_smoke: bool = True,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    root = root.resolve()
    work_dir = work_dir.resolve(strict=False)
    _outside_corpus(root, work_dir, "work directory")
    work_dir.mkdir(parents=True, exist_ok=True)
    identity = DilemmadataCorpusIdentity()
    first_discovery = discover_dilemmadata_corpus(root, identity=identity)
    config, cache_adapter_config = _adapter_config()
    cache_config = CorpusCacheConfig(work_dir / "cache")
    first = _run_source_build(
        first_discovery,
        config=config,
        cache_adapter_config=cache_adapter_config,
        cache_config=cache_config,
        collect_full_evidence=True,
    )
    index = first["index"]
    build_report = first["build_report"]
    index_path = work_dir / "dilemmadata.index.json"
    dump_corpus_index(index, index_path)
    before_second = _artifact_snapshot(index, cache_config)

    second_discovery = discover_dilemmadata_corpus(root, identity=identity)
    second = _run_source_build(
        second_discovery,
        config=config,
        cache_adapter_config=cache_adapter_config,
        cache_config=cache_config,
        collect_full_evidence=False,
    )
    rebuilt_index = second["index"]
    rebuilt_report = second["build_report"]
    after_second = _artifact_snapshot(rebuilt_index, cache_config)
    corpus_index_byte_identical = (
        dumps_corpus_index(index) == dumps_corpus_index(rebuilt_index)
    )
    discovery_equal = _discovery_projection(first_discovery) == _discovery_projection(
        second_discovery
    )
    quarantine_equal = (
        first["quarantine_projection"] == second["quarantine_projection"]
    )
    source_semantic_equal = (
        first["semantic_projection_fingerprint"]
        == second["semantic_projection_fingerprint"]
    )
    immutable_artifacts_unchanged = before_second == after_second
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
    for raw_id, rows in sorted(first["raw_clusters"].items()):
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
        _cache_mutation_evidence(index, first_discovery.records[0])
        if first_discovery.records
        else {"contract_kind": "unavailable"}
    )
    ssl_smoke = (
        _run_ssl_smoke(
            repo_root=Path(__file__).resolve().parents[1],
            work_dir=work_dir,
            cache_config=cache_config,
            index_path=index_path,
            index=index,
            accepted=first["accepted"],
        )
        if run_ssl_smoke
        else {"passed": None, "skipped": True}
    )
    status_counts = first["status_counts"]
    dialect_status_counts = first["dialect_status_counts"]
    category_counts = first["category_counts"]
    non_empty_splits = all(split_counts[name] > 0 for name in ("train", "validation", "test"))
    report: dict[str, object] = {
        "acceptance_report_version": DILEMMADATA_ACCEPTANCE_REPORT_VERSION,
        "adapter_version": DILEMMADATA_ADAPTER_VERSION,
        "contracts": {
            "acceptance_report_version": DILEMMADATA_ACCEPTANCE_REPORT_VERSION,
            "adapter_version": DILEMMADATA_ADAPTER_VERSION,
            "canonical_schema_version": SCHEMA_VERSION,
            "corpus_cache_input_identity_version": (
                CORPUS_CACHE_INPUT_IDENTITY_VERSION
            ),
            "corpus_identity_version": DILEMMADATA_CORPUS_IDENTITY_VERSION,
            "feature_registry_version": FEATURE_REGISTRY_VERSION,
            "graph_builder_version": GRAPH_BUILDER_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "grouping_version": DILEMMADATA_GROUPING_VERSION,
            "model_input_fingerprint_version": MODEL_INPUT_FINGERPRINT_VERSION,
            "multisource_cache_version": MULTISOURCE_CACHE_VERSION,
            "multisource_corpus_index_version": MULTISOURCE_CORPUS_INDEX_VERSION,
            "production_manifest_version": DILEMMADATA_PRODUCTION_MANIFEST_VERSION,
            "raw_projection_version": DILEMMADATA_RAW_PROJECTION_VERSION,
            "record_binding_version": DILEMMADATA_RECORD_BINDING_VERSION,
            "split_manifest_version": SPLIT_MANIFEST_VERSION,
            "target_encoding_registry_version": TARGET_ENCODING_REGISTRY_VERSION,
            "target_ontology_version": TARGET_ONTOLOGY_VERSION,
        },
        "corpus_identity": {
            "version": DILEMMADATA_CORPUS_IDENTITY_VERSION,
            "release_version": DILEMMADATA_RELEASE_VERSION,
            "release_commit": DILEMMADATA_RELEASE_COMMIT,
            "installation_file_count": DILEMMADATA_INSTALLATION_FILE_COUNT,
            "content_fingerprint": DILEMMADATA_CONTENT_FINGERPRINT,
            "primary_record_count": DILEMMADATA_PRIMARY_RECORD_COUNT,
            "an_record_count": DILEMMADATA_AN_RECORD_COUNT,
            "dlc_record_count": DILEMMADATA_DLC_RECORD_COUNT,
        },
        "outcomes": {
            "discovered_count": len(first_discovery.records),
            "fatal_count": len(first["fatal"]),
            "status_counts": dict(sorted(status_counts.items())),
            "dialect_status_counts": {
                f"{dialect}:{status}": count
                for (dialect, status), count in sorted(dialect_status_counts.items())
            },
            "failure_category_counts": dict(sorted(category_counts.items())),
        },
        "totals": dict(sorted(first["totals"].items())),
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
            "second_source": _build_report_summary(rebuilt_report),
            "corpus_index_byte_identical": corpus_index_byte_identical,
            "discovery_projection_identical": discovery_equal,
            "quarantine_identity_categories_identical": quarantine_equal,
            "source_build_semantic_projection_identical": source_semantic_equal,
            "immutable_artifacts_unchanged": immutable_artifacts_unchanged,
            "first_source_semantic_fingerprint": first[
                "semantic_projection_fingerprint"
            ],
            "second_source_semantic_fingerprint": second[
                "semantic_projection_fingerprint"
            ],
            "mutation_evidence": cache_mutation,
        },
        "grouping_and_split": {
            "discovery_component_count": first_discovery.component_count,
            "discovery_multi_record_component_count": (
                first_discovery.multi_record_component_count
            ),
            "discovery_explicit_overlap_count": first_discovery.explicit_overlap_count,
            "discovery_suggested_split_conflict_count": (
                first_discovery.suggested_split_conflict_count
            ),
            "accepted_manifest_component_count": component_count,
            "accepted_source_group_count": len(
                {row.source_group_id for row in index.records}
            ),
            "split_record_counts": dict(sorted(split_counts.items())),
            "split_component_counts": dict(sorted(split_component_counts.items())),
            "non_empty_train_validation_test": non_empty_splits,
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
            "validation_warning_counts": dict(
                sorted(first["warning_counts"].items())
            ),
            "quality_flag_counts": dict(
                sorted(first["quality_flag_counts"].items())
            ),
            "provenance_kind_counts": dict(
                sorted(first["provenance_kind_counts"].items())
            ),
        },
        "ssl_smoke": ssl_smoke,
        "intrinsic_ready": False,
        "ready": False,
        "manifest_check": {"checked": False, "passed": False},
        "processed_primary_tsv_bytes": first["processed_bytes"],
        "second_source_processed_primary_tsv_bytes": second["processed_bytes"],
        "installation_byte_count": first_discovery.installation_byte_count,
        "maximum_rss_bytes": _maximum_rss_bytes(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "fatal_failure_count": len(first["fatal"]) + len(second["fatal"]),
        "fatal_failure_samples": [*first["fatal"][:16], *second["fatal"][:16]],
        "quarantine_samples": first["failure_samples"],
    }
    report = _revalidate_report(report)
    projection = manifest_projection(report)
    report["semantic_acceptance_fingerprint"] = projection[
        "semantic_acceptance_fingerprint"
    ]
    report["semantic_fingerprint"] = projection["semantic_acceptance_fingerprint"]
    if manifest_path is not None:
        return _apply_manifest_check(report, manifest_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-ssl-smoke", action="store_true")
    parser.add_argument(
        "--check",
        nargs="?",
        const=DEFAULT_MANIFEST,
        type=Path,
        metavar="MANIFEST",
        help=(
            "compare the deterministic semantic projection with MANIFEST "
            "(default: committed production manifest)"
        ),
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        metavar="PATH",
        help="write the actual compact projection for explicit manifest bootstrapping",
    )
    parser.add_argument(
        "--report-input",
        type=Path,
        help="check a previously generated full report without rerunning the corpus",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output.resolve(strict=False)
    try:
        if arguments.report_input is not None:
            if arguments.root is not None or arguments.work_dir is not None:
                raise ValueError("--report-input cannot be combined with --root/--work-dir")
            if arguments.check is None and arguments.write_manifest is None:
                raise ValueError("--report-input requires --check or --write-manifest")
            if arguments.skip_ssl_smoke:
                raise ValueError("--report-input cannot change SSL execution")
            loaded = json.loads(arguments.report_input.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("report input must contain one JSON object")
            report = _revalidate_report(loaded)
            projection = manifest_projection(report)
            report["semantic_acceptance_fingerprint"] = projection[
                "semantic_acceptance_fingerprint"
            ]
            report["semantic_fingerprint"] = projection[
                "semantic_acceptance_fingerprint"
            ]
            if arguments.check is not None:
                report = _apply_manifest_check(report, arguments.check)
        else:
            if arguments.root is None or arguments.work_dir is None:
                raise ValueError("--root and --work-dir are required for a source build")
            root = arguments.root.resolve()
            _outside_corpus(root, output, "output")
            if arguments.write_manifest is not None:
                _outside_corpus(
                    root,
                    arguments.write_manifest.resolve(strict=False),
                    "manifest output",
                )
            report = build_acceptance_report(
                root,
                work_dir=arguments.work_dir,
                run_ssl_smoke=not arguments.skip_ssl_smoke,
                manifest_path=arguments.check,
            )
        if arguments.write_manifest is not None:
            manifest_output = arguments.write_manifest.resolve(strict=False)
            manifest_output.parent.mkdir(parents=True, exist_ok=True)
            manifest_output.write_text(
                json.dumps(
                    manifest_projection(report),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
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
                "intrinsic_ready": report["intrinsic_ready"],
                "semantic_acceptance_fingerprint": report[
                    "semantic_acceptance_fingerprint"
                ],
                "manifest_check": report["manifest_check"],
                "outcomes": report["outcomes"],
                "duration_seconds": report["duration_seconds"],
                "maximum_rss_bytes": report["maximum_rss_bytes"],
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    bootstrapped = arguments.write_manifest is not None and report["intrinsic_ready"]
    return 0 if report["ready"] or bootstrapped else 1


if __name__ == "__main__":
    raise SystemExit(main())
