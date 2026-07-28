"""Bounded, explicitly enabled CPU performance evidence."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from music_critic.data import dump_piece, load_piece
from music_critic.evaluation.contracts import (
    PROFILER_CONTRACT_VERSION,
    EvaluationContractError,
    canonical_fingerprint,
    write_json_atomic,
)
from music_critic.graph import build_raw_graph, validate_raw_graph_batch
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    aggregate_task_losses,
    join_task_supervision,
)
from music_critic.tasks import (
    MultiSourceBatch,
    align_sample_targets,
    collate_multisource_samples,
    prepare_multisource_sample,
    tensorize_aligned_targets,
)
from music_critic.training.data import _bounded_samples
from music_critic.training.device import move_multisource_batch


_STAGES = (
    "canonical_artifact_read",
    "graph_construction",
    "target_alignment_tensorization",
    "collation",
    "device_transfer",
    "model_forward",
    "loss_construction",
    "backward",
    "optimizer_step",
    "validation_forward",
)


def _plain(config: object) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        value = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, dict):
        value = dict(config)
    else:
        raise EvaluationContractError("profiler.config.type_invalid")
    if not isinstance(value, dict):
        raise EvaluationContractError("profiler.config.mapping_invalid")
    return value


def _validate(config: dict[str, Any]) -> None:
    if config.get("enabled") is not True:
        raise EvaluationContractError("profiler.explicit_enable_required")
    for name in ("max_batches", "hidden_dim", "local_gnn_layers"):
        value = config.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise EvaluationContractError(f"profiler.{name}_invalid")
    accepted = {
        "dataset_values": {"hooktheory", "pop909_cl", "mixed"},
        "model_values": {
            "feature_only",
            "local_gnn",
            "hierarchical",
        },
    }
    for name, choices in accepted.items():
        values = config.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(value not in choices for value in values)
            or len(values) != len(set(values))
        ):
            raise EvaluationContractError(f"profiler.{name}_invalid")
    for name, minimum in (("batch_sizes", 1), ("worker_values", 0)):
        values = config.get(name)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                for value in values
            )
            or len(values) != len(set(values))
        ):
            raise EvaluationContractError(f"profiler.{name}_invalid")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "observation_count": 0,
            "mean_seconds": 0.0,
            "p50_seconds": 0.0,
            "p90_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
        }
    return {
        "observation_count": len(values),
        "mean_seconds": math.fsum(values) / len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p90_seconds": _percentile(values, 0.90),
        "p95_seconds": _percentile(values, 0.95),
        "p99_seconds": _percentile(values, 0.99),
    }


def _measure(call: Callable[[], Any]) -> tuple[Any, float]:
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


def _model(name: str, config: dict[str, Any]) -> torch.nn.Module:
    common = {
        "hidden_dim": int(config["hidden_dim"]),
        "dropout": 0.0,
    }
    if name in {"feature_only", "local_gnn"}:
        return LocalHeterogeneousBaseline(
            LocalBaselineConfig(
                variant=name,
                gnn_layers=(
                    0
                    if name == "feature_only"
                    else int(config["local_gnn_layers"])
                ),
                **common,
            )
        )
    return HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            local_gnn_layers=int(config["local_gnn_layers"]),
            transformer_layers=int(config["transformer_layers"]),
            attention_heads=int(config["attention_heads"]),
            **common,
        )
    )


def _selected_samples(name: str) -> tuple[Any, ...]:
    train, _validation = _bounded_samples()
    if name == "mixed":
        return train
    selected = tuple(
        sample for sample in train if sample.dataset_id == name
    )
    if not selected:
        raise EvaluationContractError(
            f"profiler.dataset.empty:{name}"
        )
    return selected


def _scheduled_samples(
    samples: tuple[Any, ...], *, batch_size: int, max_batches: int
) -> tuple[Any, ...]:
    count = batch_size * max_batches
    return tuple(samples[index % len(samples)] for index in range(count))


def _loader(
    samples: tuple[Any, ...], *, batch_size: int, workers: int
) -> DataLoader[Any]:
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
        collate_fn=collate_multisource_samples,
    )


def _graph_target_stages(
    pieces: tuple[Any, ...],
    samples: tuple[Any, ...],
    *,
    batch_size: int,
) -> tuple[dict[str, list[float]], tuple[MultiSourceBatch, ...]]:
    timings = {name: [] for name in _STAGES}
    for piece in pieces:
        _graph, elapsed = _measure(lambda piece=piece: build_raw_graph(piece))
        timings["graph_construction"].append(elapsed)
    for start in range(0, len(samples), batch_size):
        group = samples[start : start + batch_size]

        def align_and_tensorize() -> object:
            aligned = tuple(
                align_sample_targets(
                    sample.canonical_piece,
                    sample.raw_graph,
                    sample,
                )
                for sample in group
            )
            graph_batch = Batch.from_data_list(
                [sample.raw_graph for sample in group]
            )
            validate_raw_graph_batch(
                graph_batch, sample_count=len(group)
            )
            return tensorize_aligned_targets(aligned, graph_batch)

        _targets, elapsed = _measure(align_and_tensorize)
        timings["target_alignment_tensorization"].append(elapsed)
        _batch, elapsed = _measure(
            lambda group=group: collate_multisource_samples(group)
        )
        timings["collation"].append(elapsed)
    return timings, tuple(
        collate_multisource_samples(
            samples[start : start + batch_size]
        )
        for start in range(0, len(samples), batch_size)
    )


def _profile_cell(
    config: dict[str, Any],
    *,
    dataset_name: str,
    model_name: str,
    batch_size: int,
    workers: int,
    scratch: Path,
) -> dict[str, Any]:
    base = _selected_samples(dataset_name)
    samples = _scheduled_samples(
        base,
        batch_size=batch_size,
        max_batches=int(config["max_batches"]),
    )
    canonical_paths = []
    for index, sample in enumerate(samples):
        path = scratch / f"{dataset_name}-{index}.json"
        dump_piece(sample.canonical_piece, path, indent=None)
        canonical_paths.append(path)
    canonical_read = []
    pieces = []
    for path in canonical_paths:
        piece, elapsed = _measure(lambda path=path: load_piece(path))
        pieces.append(piece)
        canonical_read.append(elapsed)
    stage_values, _direct_batches = _graph_target_stages(
        tuple(pieces),
        samples,
        batch_size=batch_size,
    )
    stage_values["canonical_artifact_read"] = canonical_read

    try:
        loaded_batches, loader_seconds = _measure(
            lambda: tuple(
                _loader(
                    samples,
                    batch_size=batch_size,
                    workers=workers,
                )
            )
        )
    except (OSError, RuntimeError) as exc:
        return {
            "status": "unavailable",
            "reason": f"worker configuration unavailable: {exc}",
            "dataset": dataset_name,
            "model": model_name,
            "batch_size": batch_size,
            "workers": workers,
        }
    if not loaded_batches:
        raise EvaluationContractError("profiler.loader.empty")
    # The complete loader traversal provides worker-aware collation evidence.
    stage_values["collation"].append(loader_seconds)
    model = _model(model_name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")
    batch_times = []
    totals = {
        "samples": 0,
        "batches": 0,
        "nodes": 0,
        "edges": 0,
        "eligible_target_rows": 0,
    }
    for cpu_batch in loaded_batches:
        batch_started = perf_counter()
        batch, elapsed = _measure(
            lambda cpu_batch=cpu_batch: move_multisource_batch(
                cpu_batch, device, non_blocking=False
            )
        )
        stage_values["device_transfer"].append(elapsed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction_result, elapsed = _measure(
            lambda: model.predict(batch.raw_graph_batch)
        )
        stage_values["model_forward"].append(elapsed)
        predictions = prediction_result[1]

        def build_loss() -> Any:
            supervisions = join_task_supervision(
                predictions, batch.target_batches
            )
            return (
                supervisions,
                aggregate_task_losses(supervisions),
            )

        loss_result, elapsed = _measure(build_loss)
        stage_values["loss_construction"].append(elapsed)
        supervisions, loss_report = loss_result
        total = loss_report.total_loss
        if total is not None:
            _unused, elapsed = _measure(total.backward)
            stage_values["backward"].append(elapsed)
            _unused, elapsed = _measure(optimizer.step)
            stage_values["optimizer_step"].append(elapsed)
        model.eval()
        with torch.no_grad():
            _unused, elapsed = _measure(
                lambda: model.predict(batch.raw_graph_batch)
            )
        stage_values["validation_forward"].append(elapsed)
        batch_times.append(perf_counter() - batch_started)
        totals["samples"] += len(batch.dataset_ids)
        totals["batches"] += 1
        totals["nodes"] += sum(
            value for _name, value in batch.statistics.node_counts
        )
        totals["edges"] += sum(
            value for _name, value in batch.statistics.edge_counts
        )
        totals["eligible_target_rows"] += sum(
            int(item.per_row_loss.shape[0]) for item in supervisions
        )
    elapsed_total = math.fsum(batch_times)
    throughputs = {
        "samples_per_second": totals["samples"] / elapsed_total,
        "batches_per_second": totals["batches"] / elapsed_total,
        "nodes_per_second": totals["nodes"] / elapsed_total,
        "edges_per_second": totals["edges"] / elapsed_total,
        "eligible_target_rows_per_second": (
            totals["eligible_target_rows"] / elapsed_total
        ),
    }
    descriptor = {
        "dataset": dataset_name,
        "model": model_name,
        "batch_size": batch_size,
        "workers": workers,
        "sample_identities": [
            [sample.dataset_id, sample.piece_id] for sample in samples
        ],
        "model_config": (
            model.config.to_dict()
            if hasattr(model.config, "to_dict")
            else asdict(model.config)
        ),
    }
    fingerprints = {
        "dataset": canonical_fingerprint(
            descriptor["sample_identities"]
        ),
        "model": canonical_fingerprint(descriptor["model_config"]),
        "batch": canonical_fingerprint(
            {"batch_size": batch_size}
        ),
        "worker": canonical_fingerprint({"workers": workers}),
    }
    return {
        "status": "completed",
        "dataset": dataset_name,
        "model": model_name,
        "batch_size": batch_size,
        "workers": workers,
        "counts": totals,
        "throughput": throughputs,
        "batch_time": _summary(batch_times),
        "cpu_peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "stages": {
            name: _summary(stage_values[name]) for name in _STAGES
        },
        "fingerprints": fingerprints,
        "dataset_model_batch_worker_fingerprint": (
            canonical_fingerprint(descriptor)
        ),
    }


def run_profiler(config: object) -> dict[str, Any]:
    """Run only a finite declared synthetic matrix and write one report."""

    resolved = _plain(config)
    _validate(resolved)
    torch.manual_seed(int(resolved["seed"]))
    output = Path(resolved["output_path"]).resolve()
    if output.exists():
        raise EvaluationContractError(
            "profiler.output.collision"
        )
    cells = []
    with TemporaryDirectory(
        prefix="music-critic-phase6d-profiler-"
    ) as temporary:
        scratch = Path(temporary)
        for dataset_name in resolved["dataset_values"]:
            for model_name in resolved["model_values"]:
                for batch_size in resolved["batch_sizes"]:
                    for workers in resolved["worker_values"]:
                        cells.append(
                            _profile_cell(
                                resolved,
                                dataset_name=dataset_name,
                                model_name=model_name,
                                batch_size=batch_size,
                                workers=workers,
                                scratch=scratch,
                            )
                        )
    report = {
        "profiler_contract_version": PROFILER_CONTRACT_VERSION,
        "mode": "explicit_bounded_synthetic",
        "enabled": True,
        "normal_training_instrumented": False,
        "cuda_synchronization_used": False,
        "configuration": resolved,
        "matrix_fingerprint": canonical_fingerprint(
            {
                "datasets": resolved["dataset_values"],
                "models": resolved["model_values"],
                "batch_sizes": resolved["batch_sizes"],
                "workers": resolved["worker_values"],
                "max_batches": resolved["max_batches"],
            }
        ),
        "cells": cells,
        "cpu_peak_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "retained_per_batch_history": False,
    }
    write_json_atomic(output, report)
    return report


__all__ = ["run_profiler"]
