from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from omegaconf import OmegaConf

from music_critic.evaluation.config import EvaluationDataConfig
from music_critic.evaluation.data import build_evaluation_data_runtime
from music_critic.tasks import (
    CanonicalCorpusInput,
    CorpusCacheConfig,
    cache_canonical_corpus,
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
)
from music_critic.training.data import _hook_piece


def _index(root: Path, dataset_id: str):
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
        adapter_name="phase6d_test",
        adapter_version="1.0.0",
        adapter_config={},
        source_identity=f"{dataset_id}-fixture",
        source_fingerprint=sha256(dataset_id.encode()).hexdigest(),
        creation_policy="phase6d_test",
    )
    path = root / f"{dataset_id}.index.json"
    dump_corpus_index(index, path)
    return index, path, cache


def test_single_dataset_runtime_derives_bound_view_from_global_manifest(
    tmp_path: Path,
) -> None:
    hook, hook_path, hook_cache = _index(tmp_path, "hooktheory")
    pop, _pop_path, _pop_cache = _index(tmp_path, "pop909_cl")
    assignments = {
        (record.dataset_id, record.piece_id): (
            "train" if ordinal == 0 else "validation"
        )
        for index in (hook, pop)
        for ordinal, record in enumerate(index.records)
    }
    manifest = create_split_manifest(
        (hook, pop),
        assignments,
        seed=42,
        policy="phase6d_global_fixture",
    )
    manifest_path = tmp_path / "global.split.json"
    dump_split_manifest(manifest, manifest_path)

    runtime = build_evaluation_data_runtime(
        OmegaConf.structured(
            EvaluationDataConfig(
                name="hooktheory",
                index_paths=[str(hook_path)],
                cache_roots=[str(hook_cache.root)],
                split_manifest=str(manifest_path),
                batch_size=1,
            )
        ),
        split="validation",
        seed=42,
    )

    assert runtime.bindings["split_manifest_fingerprint"] == (
        manifest.manifest_fingerprint
    )
    assert runtime.bindings["effective_split_manifest_fingerprint"] != (
        manifest.manifest_fingerprint
    )
    assert {
        dataset_id
        for loader in (runtime.train_loader, runtime.evaluation_loader)
        for batch in loader()
        for dataset_id in batch.dataset_ids
    } == {"hooktheory"}
