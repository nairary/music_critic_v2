from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import sleep

import pytest
import torch

from music_critic.evaluation.config import ProfilerConfig
from music_critic.evaluation.contracts import EvaluationContractError
from music_critic.evaluation import profiler
from music_critic.evaluation.profiler import run_profiler
from music_critic.tasks import (
    CanonicalCorpusInput,
    CorpusCacheConfig,
    cache_canonical_corpus,
    collate_multisource_samples,
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
)
from music_critic.training.data import _bounded_samples, _hook_piece


def _tiny_config(tmp_path: Path, *, workers: int = 0) -> ProfilerConfig:
    return ProfilerConfig(
        enabled=True,
        output_path=str(tmp_path / f"report-{workers}.json"),
        max_batches=1,
        dataset_values=["hooktheory"],
        model_values=["feature_only"],
        batch_sizes=[1],
        worker_values=[workers],
        hidden_dim=8,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=1,
    )


def _production_fixture(root: Path, dataset_id: str):
    cache = CorpusCacheConfig(root / f"{dataset_id}-cache")
    inputs = []
    for ordinal in range(2):
        piece = replace(
            _hook_piece(f"{dataset_id}-{ordinal}", ordinal + 1),
            dataset_name=dataset_id,
        )
        inputs.append(
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=piece.source_group_id,
                source_identity=piece.piece_id,
                source_relative_path=f"{piece.piece_id}.json",
                source_sha256=sha256(
                    f"{dataset_id}:{ordinal}".encode()
                ).hexdigest(),
                suggested_split=None,
            )
        )
    index, _report = cache_canonical_corpus(
        inputs,
        cache_config=cache,
        dataset_id=dataset_id,
        adapter_name="phase6d_profiler_test",
        adapter_version="1.0.0",
        adapter_config={},
        source_identity=f"{dataset_id}-fixture",
        source_fingerprint=sha256(dataset_id.encode()).hexdigest(),
        creation_policy="phase6d_profiler_test",
    )
    path = root / f"{dataset_id}.index.json"
    dump_corpus_index(index, path)
    return index, path, cache


def test_profiler_requires_explicit_enable(tmp_path: Path) -> None:
    with pytest.raises(
        EvaluationContractError,
        match="explicit_enable_required",
    ):
        run_profiler(
            ProfilerConfig(output_path=str(tmp_path / "report.json"))
        )
    assert not (tmp_path / "report.json").exists()


def test_tiny_profiler_separates_measurement_passes_and_rates(
    tmp_path: Path,
) -> None:
    report = run_profiler(_tiny_config(tmp_path))
    cell = report["cells"][0]

    assert cell["status"] == "completed"
    assert cell["exclusive_preparation"]["status"] == "measured"
    assert set(cell["exclusive_preparation"]["stages"]) == {
        "canonical_artifact_read",
        "graph_construction",
        "target_alignment_tensorization",
        "collation",
    }
    for stage in cell["exclusive_preparation"]["stages"].values():
        assert stage["status"] == "measured"
        assert stage["timing"]["observation_count"] == 1
    assert all(
        value > 0
        for value in cell["prepared_compute"]["throughput"].values()
    )
    assert all(
        value > 0
        for value in cell["end_to_end"]["throughput"].values()
    )
    assert (
        cell["worker_and_loader_evidence"]["name"]
        == "full_loader_traversal"
    )
    assert cell["memory_evidence"]["scope"] == (
        "process_level_high_water_mark"
    )
    assert cell["memory_evidence"]["isolated_cell"] is False
    assert report["normal_training_instrumented"] is False
    assert report["checkpoint_loaded"] is False
    assert report["retained_per_batch_history"] is False


def test_workers_zero_exclusive_chain_aligns_each_sample_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    actual = profiler.align_sample_targets

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(profiler, "align_sample_targets", counted)
    report = run_profiler(_tiny_config(tmp_path))

    cell = report["cells"][0]
    assert calls == len(cell["scheduled_identities"])
    alignment = cell["exclusive_preparation"]["stages"][
        "target_alignment_tensorization"
    ]
    collation = cell["exclusive_preparation"]["stages"]["collation"]
    assert alignment["unit"] == "per_batch"
    assert collation["unit"] == "per_batch"
    assert alignment["timing"]["observation_count"] == 1
    assert collation["timing"]["observation_count"] == 1


def test_workers_positive_has_structured_unavailable_stage_attribution(
    tmp_path: Path,
) -> None:
    report = run_profiler(_tiny_config(tmp_path, workers=1))
    cell = report["cells"][0]

    assert cell["status"] == "completed"
    decomposition = cell["exclusive_preparation"]
    assert decomposition["status"] == "unavailable"
    assert decomposition["value"] is None
    assert decomposition["reason"]["category"] == (
        "multiprocess_exact_stage_attribution_unavailable"
    )
    assert all(
        stage["status"] == "unavailable"
        and stage["value"] is None
        for stage in decomposition["stages"].values()
    )
    worker = cell["worker_and_loader_evidence"][
        "worker_component_attribution"
    ]
    assert worker["status"] == "unavailable"
    assert worker["reason"]["category"] == "worker_components_overlap"


def test_end_to_end_timer_includes_loader_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, _validation = _bounded_samples()
    batch = collate_multisource_samples((train[0],))
    delay = 0.03

    class DelayedLoader:
        def __iter__(self):
            sleep(delay)
            yield batch

    monkeypatch.setattr(
        profiler, "_loader", lambda *args, **kwargs: DelayedLoader()
    )
    monkeypatch.setattr(
        profiler,
        "_model",
        lambda *args, **kwargs: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        profiler,
        "_training_step",
        lambda _model, _optimizer, cpu_batch, timings: (cpu_batch, ()),
    )
    source = object()
    result = profiler._end_to_end_pass(
        {},
        source,
        model_name="feature_only",
        batch_size=1,
        workers=0,
    )

    assert result["elapsed_seconds"] >= delay
    assert result["first_batch_ready_seconds"] >= delay


def test_production_mode_reads_only_fingerprinted_bounded_subset(
    tmp_path: Path,
) -> None:
    hook, hook_path, hook_cache = _production_fixture(
        tmp_path, "hooktheory"
    )
    pop, pop_path, pop_cache = _production_fixture(
        tmp_path, "pop909_cl"
    )
    assignments = {
        (record.dataset_id, record.piece_id): "validation"
        for index in (hook, pop)
        for record in index.records
    }
    manifest = create_split_manifest(
        (hook, pop),
        assignments,
        seed=42,
        policy="phase6d_profiler_fixture",
    )
    manifest_path = tmp_path / "global.split.json"
    dump_split_manifest(manifest, manifest_path)
    before = {
        path: sha256(path.read_bytes()).hexdigest()
        for cache in (hook_cache, pop_cache)
        for path in cache.root.rglob("*")
        if path.is_file()
    }

    report = run_profiler(
        ProfilerConfig(
            enabled=True,
            output_path=str(tmp_path / "production-report.json"),
            input_mode="production_read_only",
            max_batches=1,
            dataset_values=["hooktheory", "pop909_cl", "mixed"],
            model_values=["feature_only"],
            batch_sizes=[1],
            worker_values=[0],
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=1,
            production_index_paths=[
                str(hook_path.resolve()),
                str(pop_path.resolve()),
            ],
            production_cache_roots=[
                str(hook_cache.root.resolve()),
                str(pop_cache.root.resolve()),
            ],
            production_split_manifest=str(manifest_path.resolve()),
            production_max_samples_per_dataset=1,
        )
    )
    after = {
        path: sha256(path.read_bytes()).hexdigest()
        for cache in (hook_cache, pop_cache)
        for path in cache.root.rglob("*")
        if path.is_file()
    }

    assert report["mode"] == "explicit_bounded_production_read_only"
    assert {cell["dataset"] for cell in report["cells"]} == {
        "hooktheory",
        "pop909_cl",
        "mixed",
    }
    assert all(
        cell["source_evidence"]["cache_write_count"] == 0
        and cell["source_evidence"]["checkpoint_read_count"] == 0
        and cell["source_evidence"]["subset_fingerprint"]
        for cell in report["cells"]
    )
    assert before == after
