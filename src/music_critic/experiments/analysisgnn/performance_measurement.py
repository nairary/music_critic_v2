"""Measurement helpers for isolated replays, never canonical-run instrumentation.

Wall spans may overlap; CUDA event spans measure elapsed stream time rather than
kernel busy time. Neither set should be added to infer a critical path. Profiling
ranges and CUDA events are opt-in diagnostics; normal CPU timing does not touch
CUDA. Callers own phase selection, experiment registration and replay state.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import sys
import time
from typing import Iterator, Mapping, Sequence

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"performance.invalid_{name}")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"performance.invalid_{name}")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"performance.invalid_{name}")
    return result


class StageTimer:
    """Record host spans and optionally deferred CUDA event durations.

    ``resolve`` is the only synchronization boundary. With
    ``synchronize_cuda=False``, incomplete events remain explicitly pending.
    CPU-only fallback is reported as unavailable rather than zero CUDA time.
    """

    def __init__(
        self, *, diagnostic: bool = False, cuda_events: bool = False,
        cuda_device: object = None,
    ) -> None:
        if type(diagnostic) is not bool or type(cuda_events) is not bool:
            raise ValueError("performance.invalid_timer_flags")
        self.diagnostic = diagnostic
        self.cuda_requested = cuda_events
        self.cuda_enabled = cuda_events and torch.cuda.is_available()
        self.cuda_device = cuda_device
        self._rows: list[dict[str, object]] = []
        self._events: list[tuple[int, object, object]] = []
        self._active = 0

    @contextmanager
    def span(
        self, name: str, *, phase: str = "train", update: int | None = None,
    ) -> Iterator[None]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("performance.invalid_stage")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("performance.invalid_phase")
        if update is not None:
            _integer(update, "update")
        row: dict[str, object] = {
            "stage": name, "phase": phase, "update": update,
            "utc_started": _utc_now(), "monotonic_started_ns": time.perf_counter_ns(),
            "wall_seconds": None, "status": "running", "cuda_seconds": None,
            "cuda_status": ("pending" if self.cuda_enabled else
                            "unavailable" if self.cuda_requested else "disabled"),
        }
        index = len(self._rows)
        self._rows.append(row)
        cuda_start = cuda_end = None
        if self.cuda_enabled:
            with torch.cuda.device(self.cuda_device):
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
        self._active += 1
        try:
            scope = torch.profiler.record_function(name) if self.diagnostic else nullcontext()
            with scope:
                yield
        except BaseException as exc:
            row["status"] = "error"
            row["error_type"] = type(exc).__name__
            raise
        else:
            row["status"] = "completed"
        finally:
            if cuda_end is not None:
                with torch.cuda.device(self.cuda_device):
                    cuda_end.record()
                self._events.append((index, cuda_start, cuda_end))
            row["wall_seconds"] = (time.perf_counter_ns() - row["monotonic_started_ns"]) / 1e9
            self._active -= 1

    def resolve(self, *, synchronize_cuda: bool = True) -> list[dict[str, object]]:
        if type(synchronize_cuda) is not bool:
            raise ValueError("performance.invalid_synchronization_flag")
        if self._active:
            raise ValueError("performance.cannot_resolve_active_span")
        pending = [events for events in self._events
                   if self._rows[events[0]]["cuda_status"] == "pending"]
        if pending and synchronize_cuda:
            torch.cuda.synchronize(self.cuda_device)
        for index, start, end in pending:
            if synchronize_cuda or end.query():
                self._rows[index]["cuda_seconds"] = start.elapsed_time(end) / 1000.0
                self._rows[index]["cuda_status"] = "resolved"
        return [dict(row) for row in self._rows]


@dataclass(frozen=True)
class EqualDrawBenchmarkPlan:
    measured_draws: int = 800
    batch_sizes: tuple[int, ...] = (2, 4, 8)
    fixture_smoke: bool = False

    def __post_init__(self) -> None:
        _integer(self.measured_draws, "measured_draws", minimum=1)
        if type(self.fixture_smoke) is not bool:
            raise ValueError("performance.invalid_fixture_smoke")
        if not isinstance(self.batch_sizes, tuple) or not self.batch_sizes:
            raise ValueError("performance.invalid_batch_sizes")
        for batch in self.batch_sizes:
            _integer(batch, "batch_size", minimum=1)
            if self.measured_draws % batch:
                raise ValueError("performance.draws_not_divisible_by_batch")
            updates = self.measured_draws // batch
            if not self.fixture_smoke and not 100 <= updates <= 500:
                raise ValueError("performance.measured_updates_outside_100_500")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("performance.duplicate_batch_size")

    @property
    def updates_by_batch(self) -> dict[int, int]:
        return {batch: self.measured_draws // batch for batch in self.batch_sizes}


def _sample_lengths(store: object, records: int) -> list[int]:
    count = _integer(store.num_nodes, "node_count")
    pointer = getattr(store, "ptr", None)
    if pointer is None:
        if records != 1:
            raise ValueError("performance.batched_store_requires_ptr")
        return [count]
    if not isinstance(pointer, torch.Tensor) or pointer.device.type != "cpu":
        raise ValueError("performance.count_raw_graph_on_host_before_h2d")
    if pointer.dtype != torch.long or pointer.ndim != 1 or pointer.numel() != records + 1:
        raise ValueError("performance.invalid_graph_pointer")
    values = pointer.tolist()
    if values[0] != 0 or values[-1] != count or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("performance.invalid_graph_pointer")
    return [b - a for a, b in zip(values, values[1:])]


def raw_graph_counts(raw_graph: object) -> dict[str, object]:
    """Count host graph structure without inputs/targets or GPU scalar reads.

    Coarse padding is exact for the frozen hierarchy layout: one song token,
    then all bar tokens, then track tokens. It is not a node-padding estimate.
    """
    records = _integer(getattr(raw_graph, "num_graphs", 1), "record_count", minimum=1)
    nodes = {kind: _integer(raw_graph[kind].num_nodes, "node_count")
             for kind in raw_graph.node_types}
    if not {"song", "bar", "track"}.issubset(nodes) or nodes["song"] != records:
        raise ValueError("performance.raw_hierarchy_required")
    edges = {}
    for kind in raw_graph.edge_types:
        tensor = raw_graph[kind].edge_index
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != 2:
            raise ValueError("performance.invalid_edge_index")
        edges["/".join(kind)] = tensor.shape[1]
    bars = _sample_lengths(raw_graph["bar"], records)
    tracks = _sample_lengths(raw_graph["track"], records)
    lengths = [1 + bar + track for bar, track in zip(bars, tracks, strict=True)]
    padded = records * max(lengths)
    return {
        "records": records, "nodes": sum(nodes.values()), "edges": sum(edges.values()),
        "nodes_by_type": nodes, "edges_by_type": edges, "coarse_lengths": lengths,
        "coarse_tokens": sum(lengths), "padded_coarse_tokens": padded,
        "coarse_padding_tokens": padded - sum(lengths),
        "coarse_attention_padded_cells": records * max(lengths) ** 2,
        "coarse_layout": "song_then_bars_then_tracks",
    }


def resource_snapshot() -> dict[str, object]:
    """Current process CPU counters and peak RSS; no external monitor needed."""
    times = os.times()
    result: dict[str, object] = {
        "utc": _utc_now(), "monotonic_ns": time.perf_counter_ns(),
        "cpu_user_seconds": times.user, "cpu_system_seconds": times.system,
        "logical_cpu_count": os.cpu_count(), "max_rss_bytes": None,
        "minor_faults": None, "major_faults": None,
        "voluntary_context_switches": None, "involuntary_context_switches": None,
        "scope": "current_process_only",
    }
    try:
        import resource
    except ImportError:  # Windows has no stdlib resource module.
        return result
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(
        cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
        max_rss_bytes=usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt,
        voluntary_context_switches=usage.ru_nvcsw,
        involuntary_context_switches=usage.ru_nivcsw,
    )
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_window(
    *, wall_seconds: float, counts: Sequence[Mapping[str, object]],
    update_seconds: Sequence[float], resource_before: Mapping[str, object] | None = None,
    resource_after: Mapping[str, object] | None = None, cuda_device: object = None,
) -> dict[str, object]:
    """Summarize one measured window; callers reset CUDA peaks beforehand.

    GPU peaks are optional and include only what the caller's reset/window
    boundaries include. Utilization is deliberately not inferred from peaks.
    """
    wall = _finite(wall_seconds, "wall_seconds", positive=True)
    if not counts or len(counts) != len(update_seconds):
        raise ValueError("performance.one_count_and_duration_per_update_required")
    durations = [_finite(value, "update_seconds", positive=True) for value in update_seconds]
    totals = {name: sum(_integer(row[name], name, minimum=1 if name == "records" else 0)
                       for row in counts) for name in ("records", "nodes", "edges")}
    result: dict[str, object] = {
        "wall_seconds": wall, "updates": len(counts), **totals,
        "updates_per_second": len(counts) / wall,
        **{f"{name}_per_second": value / wall for name, value in totals.items()},
        "update_p50_seconds": _quantile(durations, 0.5),
        "update_p95_seconds": _quantile(durations, 0.95),
        "peak_allocated_bytes": None, "peak_reserved_bytes": None,
        "cpu_process_percent": None, "cpu_logical_capacity_percent": None,
        "resource_before": None if resource_before is None else dict(resource_before),
        "resource_after": None if resource_after is None else dict(resource_after),
    }
    if (resource_before is None) != (resource_after is None):
        raise ValueError("performance.resource_snapshots_must_be_paired")
    if resource_before is not None and resource_after is not None:
        cpu = 0.0
        for name in ("cpu_user_seconds", "cpu_system_seconds"):
            difference = _finite(resource_after[name], name) - _finite(resource_before[name], name)
            if difference < 0:
                raise ValueError("performance.resource_counter_went_backwards")
            cpu += difference
        result["cpu_process_percent"] = 100 * cpu / wall
        capacity = resource_after.get("logical_cpu_count")
        if capacity is not None:
            result["cpu_logical_capacity_percent"] = 100 * cpu / wall / _integer(capacity, "cpu_count", minimum=1)
    if cuda_device is not None and torch.cuda.is_available():
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(cuda_device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(cuda_device)
    return result
