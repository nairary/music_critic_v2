"""Synthetic measurement contracts; no model, corpus or training invocation."""

from contextlib import contextmanager, nullcontext
from datetime import datetime
import math

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from music_critic.experiments.analysisgnn.performance_measurement import (
    EqualDrawBenchmarkPlan,
    StageTimer,
    raw_graph_counts,
    resource_snapshot,
    summarize_window,
)


def _graph(bars=2, tracks=1, notes=3):
    graph = HeteroData()
    for kind, count in {"song": 1, "bar": bars, "track": tracks, "note": notes}.items():
        graph[kind].num_nodes = count
    graph["note", "next", "note"].edge_index = torch.tensor(
        [list(range(max(0, notes - 1))), list(range(1, notes))], dtype=torch.long,
    )
    return graph


def test_cpu_timer_metadata_failure_and_no_cuda_or_profiler_touch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled diagnostic path executed")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.profiler, "record_function", forbidden)
    timer = StageTimer()
    with timer.span("load", phase="warmup", update=0):
        with pytest.raises(ValueError, match="active_span"):
            timer.resolve()
    with pytest.raises(RuntimeError, match="synthetic"):
        with timer.span("forward", update=1):
            raise RuntimeError("synthetic")
    rows = timer.resolve()
    assert [(row["stage"], row["phase"], row["update"], row["status"])
            for row in rows] == [("load", "warmup", 0, "completed"),
                                 ("forward", "train", 1, "error")]
    assert rows[1]["error_type"] == "RuntimeError"
    assert all(row["wall_seconds"] >= 0 and row["monotonic_started_ns"] > 0 for row in rows)
    assert all(datetime.fromisoformat(row["utc_started"]).utcoffset().total_seconds() == 0
               for row in rows)
    assert all(row["cuda_seconds"] is None and row["cuda_status"] == "disabled" for row in rows)
    rows[0]["status"] = "caller_changed_copy"
    assert timer.resolve()[0]["status"] == "completed"


def test_explicit_cuda_request_falls_back_without_false_zero_time(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    timer = StageTimer(cuda_events=True)
    with timer.span("cpu_only"):
        pass
    row = timer.resolve(synchronize_cuda=False)[0]
    assert row["cuda_status"] == "unavailable" and row["cuda_seconds"] is None


def test_cuda_events_are_deferred_and_synchronize_once_per_window(monkeypatch):
    events = []
    synchronizations = []

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.ready = False
            self.recorded = False
            events.append(self)

        def record(self):
            self.recorded = True

        def query(self):
            return self.ready

        def elapsed_time(self, end):
            assert self.recorded and end.recorded and end.ready
            return 25.0

    def synchronize(device):
        synchronizations.append(device)
        for event in events:
            event.ready = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    timer = StageTimer(cuda_events=True, cuda_device="cuda:0")
    for update in range(2):
        with timer.span("forward", update=update):
            pass
    assert synchronizations == []
    assert all(row["cuda_status"] == "pending" for row in timer.resolve(synchronize_cuda=False))
    assert synchronizations == []
    rows = timer.resolve()
    assert synchronizations == ["cuda:0"]
    assert all(row["cuda_status"] == "resolved" and row["cuda_seconds"] == 0.025 for row in rows)
    timer.resolve()
    assert synchronizations == ["cuda:0"]


def test_profiler_ranges_are_only_created_when_requested(monkeypatch):
    seen = []

    @contextmanager
    def record(name):
        seen.append(("enter", name))
        yield
        seen.append(("exit", name))

    monkeypatch.setattr(torch.profiler, "record_function", record)
    with StageTimer(diagnostic=True).span("backward"):
        assert seen == [("enter", "backward")]
    assert seen == [("enter", "backward"), ("exit", "backward")]


@pytest.mark.parametrize("kwargs", [{"name": ""}, {"name": "x", "phase": ""},
                                   {"name": "x", "update": -1}, {"name": "x", "update": True}])
def test_invalid_stage_metadata_rejected(kwargs):
    with pytest.raises(ValueError):
        with StageTimer().span(**kwargs):
            pass


def test_equal_draw_budget_is_not_equal_update_budget():
    plan = EqualDrawBenchmarkPlan()
    assert plan.updates_by_batch == {2: 400, 4: 200, 8: 100}
    assert {batch * updates for batch, updates in plan.updates_by_batch.items()} == {800}
    assert EqualDrawBenchmarkPlan(8, fixture_smoke=True).updates_by_batch == {2: 4, 4: 2, 8: 1}


@pytest.mark.parametrize("kwargs", [
    {"measured_draws": 0}, {"measured_draws": True}, {"measured_draws": 802},
    {"measured_draws": 8}, {"measured_draws": 1600}, {"batch_sizes": ()},
    {"batch_sizes": (2, 2)}, {"batch_sizes": (0,)}, {"batch_sizes": (True,)},
    {"fixture_smoke": 1},
])
def test_invalid_benchmark_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        EqualDrawBenchmarkPlan(**kwargs)


def test_graph_counts_match_pyg_and_exact_frozen_coarse_padding():
    first, second = _graph(), _graph(bars=4, tracks=2, notes=5)
    graph = Batch.from_data_list([first, second])
    counts = raw_graph_counts(graph)
    assert counts["records"] == 2
    assert counts["nodes"] == 19
    assert counts["edges"] == 6
    assert counts["nodes_by_type"] == {"song": 2, "bar": 6, "track": 3, "note": 8}
    assert counts["coarse_lengths"] == [4, 7]
    assert counts["coarse_tokens"] == 11
    assert counts["padded_coarse_tokens"] == 14
    assert counts["coarse_padding_tokens"] == 3
    assert counts["coarse_attention_padded_cells"] == 98
    assert raw_graph_counts(_graph(bars=0, tracks=0, notes=0))["coarse_lengths"] == [1]


def test_invalid_pointer_is_not_silently_used_for_padding():
    graph = Batch.from_data_list([_graph(), _graph()])
    graph["bar"].ptr = torch.tensor([0, 5, 4])
    with pytest.raises(ValueError, match="invalid_graph_pointer"):
        raw_graph_counts(graph)


def test_window_rates_percentiles_and_cpu_denominators():
    counts = [raw_graph_counts(_graph()), raw_graph_counts(_graph())]
    before = {"cpu_user_seconds": 1.0, "cpu_system_seconds": 0.5, "logical_cpu_count": 28}
    after = {"cpu_user_seconds": 4.0, "cpu_system_seconds": 0.76, "logical_cpu_count": 28}
    result = summarize_window(wall_seconds=2.0, counts=counts, update_seconds=[0.5, 1.5],
                              resource_before=before, resource_after=after)
    assert result["records_per_second"] == 1.0
    assert result["nodes_per_second"] == 7.0
    assert result["edges_per_second"] == 2.0
    assert result["update_p50_seconds"] == 1.0
    assert result["update_p95_seconds"] == pytest.approx(1.45)
    assert result["cpu_process_percent"] == pytest.approx(163.0)
    assert result["cpu_logical_capacity_percent"] == pytest.approx(163.0 / 28)
    assert result["peak_allocated_bytes"] is None and result["peak_reserved_bytes"] is None


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True])
def test_invalid_window_denominator_rejected(value):
    with pytest.raises(ValueError):
        summarize_window(wall_seconds=value, counts=[raw_graph_counts(_graph())], update_seconds=[1.0])


@pytest.mark.parametrize("durations", [[], [0], [-1], [math.nan], [True], [1, 2]])
def test_invalid_update_denominator_rejected(durations):
    with pytest.raises(ValueError):
        summarize_window(wall_seconds=1, counts=[raw_graph_counts(_graph())], update_seconds=durations)


def test_resource_counter_backwards_and_unpaired_snapshots_rejected():
    args = dict(wall_seconds=1, counts=[raw_graph_counts(_graph())], update_seconds=[1])
    before = {"cpu_user_seconds": 2, "cpu_system_seconds": 0}
    with pytest.raises(ValueError, match="paired"):
        summarize_window(**args, resource_before=before)
    with pytest.raises(ValueError, match="went_backwards"):
        summarize_window(**args, resource_before=before,
                         resource_after={"cpu_user_seconds": 1, "cpu_system_seconds": 0})


def test_resource_snapshot_is_portable_process_evidence():
    snapshot = resource_snapshot()
    assert snapshot["cpu_user_seconds"] >= 0 and snapshot["cpu_system_seconds"] >= 0
    assert snapshot["monotonic_ns"] > 0
    assert snapshot["max_rss_bytes"] is None or snapshot["max_rss_bytes"] > 0
    assert snapshot["scope"] == "current_process_only"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only event contract")
def test_actual_cuda_events_resolve_after_window():
    timer = StageTimer(cuda_events=True)
    with timer.span("tiny_kernel"):
        value = torch.ones(8, device="cuda") + 1
    row = timer.resolve()[0]
    assert row["cuda_status"] == "resolved" and row["cuda_seconds"] >= 0
    assert torch.equal(value.cpu(), torch.full((8,), 2.0))
