from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from music_critic.tasks import (
    CanonicalCorpusInput,
    CorpusCacheConfig,
    cache_canonical_corpus,
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
    load_split_manifest,
    validate_split_manifest,
)
from music_critic.training.config import DataConfig
from music_critic.training.data import _hook_piece, build_data_runtime
from music_critic.training.make_split import main as make_split_main


def _build_index(root: Path, dataset_id: str, count: int):
    config = CorpusCacheConfig(root)
    inputs = []
    for ordinal in range(count):
        piece = replace(
            _hook_piece(f"{dataset_id}-{ordinal}", ordinal % 7 + 1),
            dataset_name=dataset_id,
        )
        payload = f"{dataset_id}:{ordinal}".encode()
        inputs.append(
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=piece.source_group_id,
                source_identity=piece.piece_id,
                source_relative_path=f"{piece.piece_id}.json",
                source_sha256=sha256(payload).hexdigest(),
                suggested_split="train",
            )
        )
    index, report = cache_canonical_corpus(
        inputs,
        cache_config=config,
        dataset_id=dataset_id,
        adapter_name="phase6c_test",
        adapter_version="1.0.0",
        adapter_config={},
        source_identity=f"{dataset_id}-bounded",
        source_fingerprint=sha256(dataset_id.encode()).hexdigest(),
        creation_policy="bounded_test",
    )
    return index, report, config


def test_split_cli_closes_source_and_lineage_and_validates(
    tmp_path: Path,
) -> None:
    alpha, _, _ = _build_index(tmp_path / "alpha-cache", "alpha", 5)
    beta, _, _ = _build_index(tmp_path / "beta-cache", "beta", 5)
    alpha_path = tmp_path / "alpha.index.json"
    beta_path = tmp_path / "beta.index.json"
    output = tmp_path / "split.json"
    dump_corpus_index(alpha, alpha_path)
    dump_corpus_index(beta, beta_path)

    make_split_main(
        [
            "--index",
            str(alpha_path),
            "--index",
            str(beta_path),
            "--ratio",
            "train=0.8",
            "--ratio",
            "validation=0.1",
            "--ratio",
            "test=0.1",
            "--seed",
            "42",
            "--output",
            str(output),
        ]
    )
    manifest = load_split_manifest(output)
    validate_split_manifest(manifest, (alpha, beta))
    by_source: dict[str, set[str]] = {}
    by_lineage: dict[str, set[str]] = {}
    for row in manifest.assignments:
        by_source.setdefault(row.source_group_id, set()).add(row.split)
        by_lineage.setdefault(row.lineage_group_id, set()).add(row.split)
    assert all(len(values) == 1 for values in by_source.values())
    assert all(len(values) == 1 for values in by_lineage.values())


def test_real_cache_runtime_enforces_train_validation_isolation(
    tmp_path: Path,
) -> None:
    alpha, _, alpha_cache = _build_index(
        tmp_path / "alpha-cache", "alpha", 4
    )
    beta, _, beta_cache = _build_index(
        tmp_path / "beta-cache", "beta", 4
    )
    alpha_path = tmp_path / "alpha.index.json"
    beta_path = tmp_path / "beta.index.json"
    split_path = tmp_path / "global.split.json"
    dump_corpus_index(alpha, alpha_path)
    dump_corpus_index(beta, beta_path)
    assignments = {
        (record.dataset_id, record.piece_id): (
            "train" if index < 2 else "validation"
        )
        for corpus in (alpha, beta)
        for index, record in enumerate(corpus.records)
    }
    manifest = create_split_manifest(
        (alpha, beta),
        assignments,
        seed=42,
        policy="phase6c_bounded_test",
        policy_config={"purpose": "split_isolation"},
    )
    validate_split_manifest(manifest, (alpha, beta))
    dump_split_manifest(manifest, split_path)
    config = DataConfig(
        name="mixed",
        index_paths=[str(alpha_path), str(beta_path)],
        cache_roots=[
            str(alpha_cache.root),
            str(beta_cache.root),
        ],
        split_manifest=str(split_path),
        batch_size=2,
        epoch_size=4,
        validation_epoch_size=4,
        mixture_weights={"alpha": 1.0, "beta": 1.0},
    )
    runtime = build_data_runtime(config, seed=42)
    train = {
        (dataset, piece)
        for batch in runtime.train_loader(0)
        for dataset, piece in zip(
            batch.dataset_ids, batch.piece_ids, strict=True
        )
    }
    validation = {
        (dataset, piece)
        for batch in runtime.validation_loader(0)
        for dataset, piece in zip(
            batch.dataset_ids, batch.piece_ids, strict=True
        )
    }
    assert train
    assert validation
    assert train.isdisjoint(validation)
    assert runtime.fingerprints["kind"] == "corpus_cache"
