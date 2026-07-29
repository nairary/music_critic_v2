"""Bounded, explicitly enabled CPU performance evidence."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from itertools import chain
import math
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch

from music_critic.data import dump_piece, load_piece
from music_critic.evaluation.contracts import (
    PROFILER_CONTRACT_VERSION,
    EvaluationContractError,
    canonical_fingerprint,
    write_json_atomic,
)
from music_critic.graph import (
    build_raw_graph,
    graph_fingerprint,
    validate_raw_graph_batch,
)
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    aggregate_task_losses,
    join_task_supervision,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    MultiSourceBatch,
    align_sample_targets,
    collate_multisource_samples,
    load_cached_piece,
    load_corpus_index,
    load_split_manifest,
    prepare_multisource_sample,
    project_multisource_targets,
    tensorize_aligned_targets,
)
from music_critic.tasks.collator import _batch_statistics
from music_critic.tasks.multisource import _sample_from_projection
from music_critic.training.data import _bounded_samples
from music_critic.training.device import move_multisource_batch


_PREPARATION_STAGES = (
    "canonical_artifact_read",
    "graph_construction",
    "target_alignment_tensorization",
    "collation",
)
_TRAIN_COMPUTE_STAGES = (
    "device_transfer",
    "model_forward",
    "loss_construction",
    "backward",
    "optimizer_step",
)
_STAGE_UNITS = {
    "canonical_artifact_read": "per_sample",
    "graph_construction": "per_sample",
    "target_alignment_tensorization": "per_batch",
    "collation": "per_batch",
    **{name: "per_batch" for name in _TRAIN_COMPUTE_STAGES},
    "validation_forward": "per_batch",
}


@dataclass(frozen=True, slots=True)
class _CellSource:
    dataset: Dataset[Any]
    scheduled_indices: tuple[int, ...]
    selected_identities: tuple[tuple[str, str], ...]
    scheduled_identities: tuple[tuple[str, str], ...]
    read_piece: Callable[[int], Any]
    evidence: dict[str, Any]


class _FixedIndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class _CanonicalPathDataset(Dataset[Any]):
    """Synthetic loader plumbing that starts at a canonical artifact."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        return prepare_multisource_sample(load_piece(self.paths[index]))


@dataclass(frozen=True, slots=True)
class _ProductionContext:
    dataset: MultiCorpusDataset
    selected_by_dataset: dict[str, tuple[int, ...]]
    record_sources: dict[int, tuple[Any, CorpusCacheConfig]]
    evidence: dict[str, Any]


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


def _positive_integer(config: dict[str, Any], name: str) -> None:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvaluationContractError(f"profiler.{name}_invalid")


def _validate(config: dict[str, Any]) -> None:
    if config.get("enabled") is not True:
        raise EvaluationContractError("profiler.explicit_enable_required")
    for name in (
        "max_batches",
        "hidden_dim",
        "local_gnn_layers",
        "transformer_layers",
        "attention_heads",
        "production_max_samples_per_dataset",
    ):
        _positive_integer(config, name)
    if int(config["production_max_samples_per_dataset"]) > 32:
        raise EvaluationContractError(
            "profiler.production_max_samples_per_dataset_unbounded"
        )
    if config.get("input_mode") not in {
        "synthetic",
        "production_read_only",
    }:
        raise EvaluationContractError("profiler.input_mode_invalid")
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
    index_paths = config.get("production_index_paths")
    cache_roots = config.get("production_cache_roots")
    manifest = config.get("production_split_manifest")
    if not isinstance(index_paths, list) or not isinstance(cache_roots, list):
        raise EvaluationContractError("profiler.production_paths_invalid")
    if config["input_mode"] == "synthetic":
        if index_paths or cache_roots or manifest:
            raise EvaluationContractError(
                "profiler.synthetic_production_paths_forbidden"
            )
        return
    if (
        not index_paths
        or len(index_paths) != len(cache_roots)
        or not isinstance(manifest, str)
        or not manifest
        or not isinstance(config.get("production_split"), str)
        or not config["production_split"]
    ):
        raise EvaluationContractError("profiler.production_paths_invalid")
    for value in [*index_paths, *cache_roots, manifest]:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise EvaluationContractError(
                "profiler.production_paths_must_be_absolute"
            )


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


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "observation_count": 0,
            "mean_seconds": None,
            "p50_seconds": None,
            "p90_seconds": None,
            "p95_seconds": None,
            "p99_seconds": None,
        }
    return {
        "observation_count": len(values),
        "mean_seconds": math.fsum(values) / len(values),
        "p50_seconds": _percentile(values, 0.50),
        "p90_seconds": _percentile(values, 0.90),
        "p95_seconds": _percentile(values, 0.95),
        "p99_seconds": _percentile(values, 0.99),
    }


def _measured_stage(
    values: list[float], *, name: str, measurement_pass: str
) -> dict[str, Any]:
    return {
        "status": "measured",
        "unit": _STAGE_UNITS[name],
        "measurement_pass": measurement_pass,
        "timing": _summary(values),
    }


def _unavailable_stage(
    *,
    name: str,
    measurement_pass: str,
    category: str,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value": None,
        "unit": _STAGE_UNITS[name],
        "measurement_pass": measurement_pass,
        "reason": {"category": category, "message": message},
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


def _scheduled_indices(
    selected: Sequence[int], *, batch_size: int, max_batches: int
) -> tuple[int, ...]:
    count = batch_size * max_batches
    return tuple(selected[index % len(selected)] for index in range(count))


def _loader(
    source: _CellSource, *, batch_size: int, workers: int
) -> DataLoader[Any]:
    return DataLoader(
        source.dataset,
        batch_size=batch_size,
        sampler=_FixedIndexSampler(source.scheduled_indices),
        num_workers=workers,
        persistent_workers=False,
        collate_fn=collate_multisource_samples,
    )


def _assemble_batch(
    samples: tuple[Any, ...],
    raw_graph_batch: Batch,
    targets: tuple[Any, ...],
) -> MultiSourceBatch:
    statistics = _batch_statistics(
        samples=samples,
        raw_graph_batch=raw_graph_batch,
        targets=targets,
    )
    return MultiSourceBatch(
        raw_graph_batch=raw_graph_batch,
        target_batches=targets,
        dataset_ids=tuple(sample.dataset_id for sample in samples),
        piece_ids=tuple(sample.piece_id for sample in samples),
        source_group_ids=tuple(
            sample.source_group_id for sample in samples
        ),
        lineage_group_ids=tuple(
            sample.lineage_group_id for sample in samples
        ),
        diagnostics_cpu=tuple(sample.diagnostics for sample in samples),
        statistics=statistics,
    )


def _exclusive_preparation_pass(
    source: _CellSource, *, batch_size: int
) -> tuple[dict[str, Any], tuple[MultiSourceBatch, ...]]:
    """One serial chain; every measured stage consumes prior-stage output."""

    values = {name: [] for name in _PREPARATION_STAGES}
    batches = []
    indices = source.scheduled_indices
    for start in range(0, len(indices), batch_size):
        group_indices = indices[start : start + batch_size]
        pieces = []
        for index in group_indices:
            piece, elapsed = _measure(
                lambda index=index: source.read_piece(index)
            )
            pieces.append(piece)
            values["canonical_artifact_read"].append(elapsed)
        graphs = []
        for piece in pieces:
            graph, elapsed = _measure(
                lambda piece=piece: build_raw_graph(piece)
            )
            graphs.append(graph)
            values["graph_construction"].append(elapsed)

        def align_and_tensorize() -> tuple[
            tuple[Any, ...], Batch, tuple[Any, ...]
        ]:
            samples = tuple(
                _sample_from_projection(
                    project_multisource_targets(piece),
                    raw_graph=graph,
                    raw_graph_fingerprint=graph_fingerprint(graph),
                )
                for piece, graph in zip(pieces, graphs, strict=True)
            )
            aligned = tuple(
                align_sample_targets(
                    sample.canonical_piece,
                    sample.raw_graph,
                    sample,
                )
                for sample in samples
            )
            raw_graph_batch = Batch.from_data_list(
                [sample.raw_graph for sample in samples]
            )
            validate_raw_graph_batch(
                raw_graph_batch, sample_count=len(samples)
            )
            targets = tensorize_aligned_targets(
                aligned, raw_graph_batch
            )
            return samples, raw_graph_batch, targets

        prepared, elapsed = _measure(align_and_tensorize)
        values["target_alignment_tensorization"].append(elapsed)
        samples, raw_graph_batch, targets = prepared
        batch, elapsed = _measure(
            lambda: _assemble_batch(samples, raw_graph_batch, targets)
        )
        values["collation"].append(elapsed)
        batches.append(batch)
    return (
        {
            "name": "serial_exclusive_preparation",
            "status": "measured",
            "boundary": (
                "canonical read -> graph construction -> target projection, "
                "alignment and tensorization -> metadata/statistics assembly; "
                "each output is passed to the next stage"
            ),
            "stage_sum_interpretation": (
                "exclusive within this pass only; never combined with loader "
                "or compute passes"
            ),
            "stages": {
                name: _measured_stage(
                    values[name],
                    name=name,
                    measurement_pass="serial_exclusive_preparation",
                )
                for name in _PREPARATION_STAGES
            },
        },
        tuple(batches),
    )


def _prepare_unmeasured(
    source: _CellSource, *, batch_size: int
) -> tuple[MultiSourceBatch, ...]:
    samples = tuple(source.dataset[index] for index in source.scheduled_indices)
    return tuple(
        collate_multisource_samples(samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    )


def _batch_counts(
    batch: MultiSourceBatch, supervisions: Sequence[Any]
) -> dict[str, int]:
    return {
        "samples": len(batch.dataset_ids),
        "batches": 1,
        "nodes": sum(
            value for _name, value in batch.statistics.node_counts
        ),
        "edges": sum(
            value for _name, value in batch.statistics.edge_counts
        ),
        "eligible_target_rows": sum(
            int(item.per_row_loss.shape[0]) for item in supervisions
        ),
    }


def _add_counts(left: dict[str, int], right: dict[str, int]) -> None:
    for name, value in right.items():
        left[name] += value


def _rates(counts: dict[str, int], elapsed: float) -> dict[str, float]:
    return {
        f"{name}_per_second": value / elapsed
        for name, value in counts.items()
    }


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cpu_batch: MultiSourceBatch,
    *,
    timings: dict[str, list[float]] | None,
) -> tuple[MultiSourceBatch, tuple[Any, ...]]:
    batch, elapsed = _measure(
        lambda: move_multisource_batch(
            cpu_batch, torch.device("cpu"), non_blocking=False
        )
    )
    if timings is not None:
        timings["device_transfer"].append(elapsed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction_result, elapsed = _measure(
        lambda: model.predict(batch.raw_graph_batch)
    )
    if timings is not None:
        timings["model_forward"].append(elapsed)
    predictions = prediction_result[1]

    def build_loss() -> tuple[tuple[Any, ...], Any]:
        supervisions = join_task_supervision(
            predictions, batch.target_batches
        )
        return supervisions, aggregate_task_losses(supervisions)

    loss_result, elapsed = _measure(build_loss)
    if timings is not None:
        timings["loss_construction"].append(elapsed)
    supervisions, loss_report = loss_result
    total = loss_report.total_loss
    if total is not None:
        _unused, elapsed = _measure(total.backward)
        if timings is not None:
            timings["backward"].append(elapsed)
        _unused, elapsed = _measure(optimizer.step)
        if timings is not None:
            timings["optimizer_step"].append(elapsed)
    return batch, supervisions


def _prepared_compute_pass(
    config: dict[str, Any],
    model_name: str,
    batches: tuple[MultiSourceBatch, ...],
) -> tuple[dict[str, Any], torch.nn.Module]:
    model = _model(model_name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values = {name: [] for name in _TRAIN_COMPUTE_STAGES}
    batch_times = []
    counts = {
        "samples": 0,
        "batches": 0,
        "nodes": 0,
        "edges": 0,
        "eligible_target_rows": 0,
    }
    for batch in batches:
        started = perf_counter()
        moved, supervisions = _training_step(
            model, optimizer, batch, timings=values
        )
        batch_times.append(perf_counter() - started)
        _add_counts(counts, _batch_counts(moved, supervisions))
    elapsed = math.fsum(batch_times)
    return (
        {
            "name": "prepared_batch_compute",
            "status": "measured",
            "boundary": (
                "starts with an already prepared CPU MultiSourceBatch and "
                "ends after optimizer.step; excludes all dataset/DataLoader work"
            ),
            "elapsed_seconds": elapsed,
            "batch_time": _summary(batch_times),
            "counts": counts,
            "throughput": _rates(counts, elapsed),
            "stages": {
                name: _measured_stage(
                    values[name],
                    name=name,
                    measurement_pass="prepared_batch_compute",
                )
                for name in _TRAIN_COMPUTE_STAGES
            },
        },
        model,
    )


def _validation_compute_pass(
    model: torch.nn.Module,
    batches: tuple[MultiSourceBatch, ...],
) -> dict[str, Any]:
    values = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            _unused, elapsed = _measure(
                lambda batch=batch: model.predict(
                    batch.raw_graph_batch
                )
            )
            values.append(elapsed)
    return {
        "name": "prepared_validation_compute",
        "status": "measured",
        "boundary": (
            "separate inference-only pass over prepared CPU batches; not "
            "summed with training compute stages"
        ),
        "stage": _measured_stage(
            values,
            name="validation_forward",
            measurement_pass="prepared_validation_compute",
        ),
    }


def _loader_traversal_pass(
    source: _CellSource, *, batch_size: int, workers: int
) -> dict[str, Any]:
    loader = _loader(source, batch_size=batch_size, workers=workers)
    started = perf_counter()
    iterator = iter(loader)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise EvaluationContractError("profiler.loader.empty") from exc
    first_latency = perf_counter() - started
    batch_count = 1
    sample_count = len(first.dataset_ids)
    for batch in iterator:
        batch_count += 1
        sample_count += len(batch.dataset_ids)
    elapsed = perf_counter() - started
    component_attribution = (
        {
            "status": "not_applicable",
            "reason": {
                "category": "single_process_loader",
                "message": "num_workers=0 has no worker startup or IPC",
            },
        }
        if workers == 0
        else {
            "status": "unavailable",
            "value": None,
            "reason": {
                "category": "worker_components_overlap",
                "message": (
                    "worker startup, IPC, prefetch and item/collate work "
                    "overlap and are not attributed as exclusive stages"
                ),
            },
        }
    )
    return {
        "name": "full_loader_traversal",
        "status": "measured",
        "boundary": (
            "timer starts before iter(loader), includes first-batch startup "
            "and every DataLoader iteration, and stops after exhaustion; "
            "no model compute"
        ),
        "first_batch_latency_seconds": first_latency,
        "elapsed_seconds": elapsed,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "samples_per_second": sample_count / elapsed,
        "worker_component_attribution": component_attribution,
    }


def _end_to_end_pass(
    config: dict[str, Any],
    source: _CellSource,
    *,
    model_name: str,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    model = _model(model_name, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _loader(source, batch_size=batch_size, workers=workers)
    started = perf_counter()
    iterator = iter(loader)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise EvaluationContractError("profiler.loader.empty") from exc
    first_batch_ready = perf_counter() - started
    counts = {
        "samples": 0,
        "batches": 0,
        "nodes": 0,
        "edges": 0,
        "eligible_target_rows": 0,
    }
    for cpu_batch in chain((first,), iterator):
        moved, supervisions = _training_step(
            model, optimizer, cpu_batch, timings=None
        )
        _add_counts(counts, _batch_counts(moved, supervisions))
    elapsed = perf_counter() - started
    return {
        "name": "end_to_end_loader_and_training_compute",
        "status": "measured",
        "boundary": (
            "timer starts before iter(loader), includes loader startup, "
            "canonical reads/sample preparation/collation, batch delivery, "
            "and the complete training compute step through optimizer.step"
        ),
        "first_batch_ready_seconds": first_batch_ready,
        "elapsed_seconds": elapsed,
        "counts": counts,
        "throughput": _rates(counts, elapsed),
    }


def _synthetic_sources(
    scratch: Path,
    *,
    batch_size: int,
    max_batches: int,
) -> dict[str, _CellSource]:
    train, _validation = _bounded_samples()
    paths = []
    identities = []
    for index, sample in enumerate(train):
        path = scratch / f"synthetic-{index}.json"
        dump_piece(sample.canonical_piece, path, indent=None)
        paths.append(path)
        identities.append((sample.dataset_id, sample.piece_id))
    dataset = _CanonicalPathDataset(paths)
    result = {}
    for name in ("hooktheory", "pop909_cl", "mixed"):
        selected = tuple(
            index
            for index, identity in enumerate(identities)
            if name == "mixed" or identity[0] == name
        )
        scheduled = _scheduled_indices(
            selected,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        result[name] = _CellSource(
            dataset=dataset,
            scheduled_indices=scheduled,
            selected_identities=tuple(identities[index] for index in selected),
            scheduled_identities=tuple(
                identities[index] for index in scheduled
            ),
            read_piece=lambda index, paths=tuple(paths): load_piece(
                paths[index]
            ),
            evidence={
                "mode": "synthetic",
                "selected_unique_sample_count": len(selected),
                "canonical_artifacts": "temporary synthetic fixtures",
            },
        )
    return result


def _production_context(config: dict[str, Any]) -> _ProductionContext:
    indices = tuple(
        load_corpus_index(path)
        for path in config["production_index_paths"]
    )
    indexed = tuple(
        IndexedMultiSourceDataset(
            index,
            cache_config=CorpusCacheConfig(root=Path(root)),
        )
        for index, root in zip(
            indices, config["production_cache_roots"], strict=True
        )
    )
    manifest = load_split_manifest(config["production_split_manifest"])
    dataset = MultiCorpusDataset(
        indexed,
        manifest,
        split=config["production_split"],
    )
    limit = int(config["production_max_samples_per_dataset"])
    selected_by_dataset = {}
    record_sources = {}
    for view, (_dataset_id, start, end) in zip(
        dataset.views, dataset.global_ranges, strict=True
    ):
        ranked = sorted(
            range(start, end),
            key=lambda index: (
                canonical_fingerprint(
                    {
                        "policy": "phase6d_profiler_subset_v1",
                        "seed": int(config["seed"]),
                        "split": config["production_split"],
                        "identity": list(dataset.record_identity(index)),
                    }
                ),
                dataset.record_identity(index),
            ),
        )
        selected = tuple(
            sorted(ranked[:limit])
        )
        selected_by_dataset[view.dataset_id] = selected
        for global_index in selected:
            local_index = global_index - start
            record_index = view.record_indices[local_index]
            record_sources[global_index] = (
                view.dataset.index.records[record_index],
                view.dataset.cache_config,
            )
    missing = {
        name
        for name in config["dataset_values"]
        if name != "mixed" and not selected_by_dataset.get(name)
    }
    if missing:
        raise EvaluationContractError(
            "profiler.production_dataset_empty:" + ",".join(sorted(missing))
        )
    nonempty = {
        name: values
        for name, values in selected_by_dataset.items()
        if values
    }
    if "mixed" in config["dataset_values"] and len(nonempty) < 2:
        raise EvaluationContractError(
            "profiler.production_mixed_requires_two_datasets"
        )
    selection_evidence = {
        name: [list(dataset.record_identity(index)) for index in values]
        for name, values in sorted(nonempty.items())
    }
    return _ProductionContext(
        dataset=dataset,
        selected_by_dataset=nonempty,
        record_sources=record_sources,
        evidence={
            "mode": "production_read_only",
            "split": config["production_split"],
            "subset_policy": "phase6d_profiler_subset_v1",
            "max_samples_per_dataset": limit,
            "selected_identities_by_dataset": selection_evidence,
            "subset_fingerprint": canonical_fingerprint(
                selection_evidence
            ),
            "index_fingerprints": [
                [index.header.dataset_id, index.header.index_fingerprint]
                for index in indices
            ],
            "split_manifest_fingerprint": manifest.manifest_fingerprint,
            "cache_write_count": 0,
            "checkpoint_read_count": 0,
            "canonical_artifact_policy": (
                "only deterministically selected indexed artifacts are read; "
                "no cache directory or corpus artifact scan"
            ),
        },
    )


def _production_source(
    context: _ProductionContext,
    dataset_name: str,
    *,
    batch_size: int,
    max_batches: int,
) -> _CellSource:
    if dataset_name == "mixed":
        by_dataset = [
            values
            for _name, values in sorted(
                context.selected_by_dataset.items()
            )
        ]
        selected = tuple(
            value
            for position in range(max(map(len, by_dataset)))
            for values in by_dataset
            if position < len(values)
            for value in (values[position],)
        )
    else:
        selected = context.selected_by_dataset[dataset_name]
    scheduled = _scheduled_indices(
        selected, batch_size=batch_size, max_batches=max_batches
    )

    def read_piece(index: int) -> Any:
        record, cache_config = context.record_sources[index]
        return load_cached_piece(record, cache_config)

    return _CellSource(
        dataset=context.dataset,
        scheduled_indices=scheduled,
        selected_identities=tuple(
            context.dataset.record_identity(index) for index in selected
        ),
        scheduled_identities=tuple(
            context.dataset.record_identity(index) for index in scheduled
        ),
        read_piece=read_piece,
        evidence=context.evidence,
    )


def _profile_cell(
    config: dict[str, Any],
    *,
    source: _CellSource,
    dataset_name: str,
    model_name: str,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    if workers == 0:
        preparation, batches = _exclusive_preparation_pass(
            source, batch_size=batch_size
        )
    else:
        preparation = {
            "name": "serial_exclusive_preparation",
            "status": "unavailable",
            "value": None,
            "reason": {
                "category": "multiprocess_exact_stage_attribution_unavailable",
                "message": (
                    "workers>0 overlaps item preparation, IPC and prefetch; "
                    "no worker time is assigned to exclusive preparation stages"
                ),
            },
            "stages": {
                name: _unavailable_stage(
                    name=name,
                    measurement_pass="serial_exclusive_preparation",
                    category=(
                        "multiprocess_exact_stage_attribution_unavailable"
                    ),
                    message=(
                        "workers>0 exact stage attribution is intentionally "
                        "not inferred"
                    ),
                )
                for name in _PREPARATION_STAGES
            },
        }
        batches = _prepare_unmeasured(source, batch_size=batch_size)
    prepared_compute, trained_model = _prepared_compute_pass(
        config, model_name, batches
    )
    validation_compute = _validation_compute_pass(
        trained_model, batches
    )
    try:
        loader_traversal = _loader_traversal_pass(
            source, batch_size=batch_size, workers=workers
        )
        end_to_end = _end_to_end_pass(
            config,
            source,
            model_name=model_name,
            batch_size=batch_size,
            workers=workers,
        )
    except (OSError, RuntimeError) as exc:
        return {
            "status": "unavailable",
            "reason": {
                "category": "worker_configuration_unavailable",
                "message": str(exc),
            },
            "dataset": dataset_name,
            "model": model_name,
            "batch_size": batch_size,
            "workers": workers,
            "exclusive_preparation": preparation,
            "prepared_compute": prepared_compute,
            "prepared_validation": validation_compute,
        }
    model_config = (
        trained_model.config.to_dict()
        if hasattr(trained_model.config, "to_dict")
        else asdict(trained_model.config)
    )
    descriptor = {
        "input_mode": config["input_mode"],
        "dataset": dataset_name,
        "model": model_name,
        "batch_size": batch_size,
        "workers": workers,
        "sample_identities": [
            list(identity) for identity in source.scheduled_identities
        ],
        "model_config": model_config,
    }
    return {
        "status": "completed",
        "dataset": dataset_name,
        "model": model_name,
        "batch_size": batch_size,
        "workers": workers,
        "source_evidence": source.evidence,
        "selected_unique_identities": [
            list(identity) for identity in source.selected_identities
        ],
        "scheduled_identities": [
            list(identity) for identity in source.scheduled_identities
        ],
        "measurement_passes": [
            preparation,
            prepared_compute,
            validation_compute,
            loader_traversal,
            end_to_end,
        ],
        "exclusive_preparation": preparation,
        "prepared_compute": prepared_compute,
        "prepared_validation": validation_compute,
        "worker_and_loader_evidence": loader_traversal,
        "end_to_end": end_to_end,
        "memory_evidence": {
            "scope": "process_level_high_water_mark",
            "isolated_cell": False,
            "ru_maxrss_kib_at_cell_end": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "interpretation": (
                "cumulative process high-water mark; not an isolated per-cell "
                "allocation or delta"
            ),
        },
        "fingerprints": {
            "dataset": canonical_fingerprint(
                [list(value) for value in source.scheduled_identities]
            ),
            "model": canonical_fingerprint(model_config),
            "batch": canonical_fingerprint(
                {"batch_size": batch_size}
            ),
            "worker": canonical_fingerprint({"workers": workers}),
        },
        "dataset_model_batch_worker_fingerprint": (
            canonical_fingerprint(descriptor)
        ),
    }


def run_profiler(config: object) -> dict[str, Any]:
    """Run one finite declared matrix outside training and checkpoint state."""

    resolved = _plain(config)
    _validate(resolved)
    torch.manual_seed(int(resolved["seed"]))
    output = Path(resolved["output_path"]).resolve()
    if output.exists():
        raise EvaluationContractError("profiler.output.collision")
    cells = []
    with TemporaryDirectory(
        prefix="music-critic-phase6d-profiler-"
    ) as temporary:
        scratch = Path(temporary)
        production = (
            _production_context(resolved)
            if resolved["input_mode"] == "production_read_only"
            else None
        )
        for dataset_name in resolved["dataset_values"]:
            for model_name in resolved["model_values"]:
                for batch_size in resolved["batch_sizes"]:
                    sources = (
                        _synthetic_sources(
                            scratch,
                            batch_size=batch_size,
                            max_batches=int(resolved["max_batches"]),
                        )
                        if production is None
                        else None
                    )
                    source = (
                        sources[dataset_name]
                        if sources is not None
                        else _production_source(
                            production,
                            dataset_name,
                            batch_size=batch_size,
                            max_batches=int(resolved["max_batches"]),
                        )
                    )
                    for workers in resolved["worker_values"]:
                        cells.append(
                            _profile_cell(
                                resolved,
                                source=source,
                                dataset_name=dataset_name,
                                model_name=model_name,
                                batch_size=batch_size,
                                workers=workers,
                            )
                        )
    report = {
        "profiler_contract_version": PROFILER_CONTRACT_VERSION,
        "mode": (
            "explicit_bounded_synthetic"
            if resolved["input_mode"] == "synthetic"
            else "explicit_bounded_production_read_only"
        ),
        "enabled": True,
        "normal_training_instrumented": False,
        "checkpoint_loaded": False,
        "checkpoint_determinism_affected": False,
        "cuda_synchronization_used": False,
        "configuration": resolved,
        "matrix_fingerprint": canonical_fingerprint(
            {
                "input_mode": resolved["input_mode"],
                "datasets": resolved["dataset_values"],
                "models": resolved["model_values"],
                "batch_sizes": resolved["batch_sizes"],
                "workers": resolved["worker_values"],
                "max_batches": resolved["max_batches"],
            }
        ),
        "cells": cells,
        "memory_evidence": {
            "scope": "process_level_high_water_mark",
            "isolated_cells": False,
            "ru_maxrss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "retained_per_batch_history": False,
    }
    write_json_atomic(output, report)
    return report


__all__ = [
    "_end_to_end_pass",
    "_exclusive_preparation_pass",
    "run_profiler",
]
