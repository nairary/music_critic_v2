from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from hashlib import sha256
import json
import shutil

import pytest
from torch.utils.data import DataLoader

from music_critic.adapters import dilemmadata as raw_adapter_module
from music_critic.adapters import dilemmadata_targets as target_adapter_module
from music_critic.tasks import dilemmadata_cache as target_cache_module
from music_critic.graph import (
    graph_fingerprint,
    graph_to_dict,
    model_input_fingerprint,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    DILEMMADATA_TARGET_CACHE_INDEX_VERSION,
    DILEMMADATA_TARGET_CACHE_VERSION,
    DilemmadataTargetCacheConfig,
    DilemmadataTargetCacheError,
    IndexedMultiSourceDataset,
    build_dilemmadata_corpus_cache,
    build_dilemmadata_target_cache,
    check_dilemmadata_target_cache,
    collate_multisource_samples,
    load_dilemmadata_target_bundle,
)
from tests.adapters.test_dilemmadata import (
    CORPUS,
    _fixture_identity,
    _set_cell,
)


def _build(root: Path):
    corpus = root / "corpus"
    shutil.copytree(CORPUS, corpus)
    raw_config = CorpusCacheConfig(root / "raw-cache")
    raw_index, _ = build_dilemmadata_corpus_cache(
        corpus,
        cache_config=raw_config,
        identity=_fixture_identity(corpus),
    )
    target_config = DilemmadataTargetCacheConfig(root / "target-cache")
    target_index, report = build_dilemmadata_target_cache(
        corpus,
        raw_index=raw_index,
        raw_cache_config=raw_config,
        target_cache_config=target_config,
        identity=_fixture_identity(corpus),
    )
    return corpus, raw_config, raw_index, target_config, target_index, report


def _batch_signature(batch) -> tuple[object, ...]:
    def normalized(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(item) for item in value]
        return value

    graph_values = tuple(
        normalized(graph_to_dict(graph))
        for graph in batch.raw_graph_batch.to_data_list()
    )
    return (
        batch.dataset_ids,
        batch.piece_ids,
        tuple(
            sha256(
                json.dumps(
                    graph,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for graph in graph_values
        ),
        graph_values,
        tuple(
            (
                item.task_id,
                item.entity_indices.tolist(),
                (
                    item.values.tolist()
                    if hasattr(item.values, "tolist")
                    else item.values
                ),
                item.source_entry_indices.tolist(),
                item.source_entry_counts_by_sample.tolist(),
            )
            for item in batch.target_batches
        ),
    )


def test_target_cache_miss_hit_and_strict_artifact_check(tmp_path: Path) -> None:
    (
        corpus,
        raw_config,
        raw_index,
        target_config,
        target_index,
        first,
    ) = _build(tmp_path)
    assert target_index.index_version == DILEMMADATA_TARGET_CACHE_INDEX_VERSION
    assert target_index.cache_version == DILEMMADATA_TARGET_CACHE_VERSION
    assert first.cache_miss_count == len(raw_index.records)
    assert first.cache_hit_count == 0

    second_index, second = build_dilemmadata_target_cache(
        corpus,
        raw_index=raw_index,
        raw_cache_config=raw_config,
        target_cache_config=target_config,
        identity=_fixture_identity(corpus),
    )
    assert second_index == target_index
    assert second.cache_hit_count == len(raw_index.records)
    assert second.cache_miss_count == 0
    assert check_dilemmadata_target_cache(
        target_index,
        raw_index=raw_index,
        cache_config=target_config,
    )["ready"] is True

    record = target_index.records[0]
    bundle = load_dilemmadata_target_bundle(record, target_config)
    assert (bundle.dataset_id, bundle.piece_id) == (
        record.dataset_id,
        record.piece_id,
    )


def test_target_cache_fails_closed_on_stale_raw_index_and_corruption(
    tmp_path: Path,
) -> None:
    (
        corpus,
        raw_config,
        raw_index,
        target_config,
        target_index,
        _,
    ) = _build(tmp_path)
    source = corpus / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 0, "onset_div", "120")
    changed_raw, _ = build_dilemmadata_corpus_cache(
        corpus,
        cache_config=CorpusCacheConfig(tmp_path / "changed-raw"),
        identity=_fixture_identity(corpus),
    )
    with pytest.raises(DilemmadataTargetCacheError) as stale:
        check_dilemmadata_target_cache(
            target_index,
            raw_index=changed_raw,
            cache_config=target_config,
        )
    assert stale.value.category == "dilemmadata.target_cache.raw_index_stale"

    record = target_index.records[0]
    artifact = (
        target_config.root
        / target_config.namespace
        / record.artifact_relative_path
    )
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(DilemmadataTargetCacheError) as corrupt:
        load_dilemmadata_target_bundle(record, target_config)
    assert corrupt.value.category == (
        "dilemmadata.target_cache.artifact_fingerprint_mismatch"
    )


def test_self_consistent_physical_indexes_share_semantic_projection(
    tmp_path: Path,
) -> None:
    _, _, raw_index, target_config, target_index, _ = _build(tmp_path)
    original = check_dilemmadata_target_cache(
        target_index,
        raw_index=raw_index,
        cache_config=target_config,
    )
    rewritten_records = []
    for record in target_index.records:
        artifact = (
            target_config.root
            / target_config.namespace
            / record.artifact_relative_path
        )
        payload = artifact.read_bytes() + b"\n"
        artifact.write_bytes(payload)
        rewritten_records.append(
            replace(record, artifact_sha256=sha256(payload).hexdigest())
        )
    provisional = object.__new__(type(target_index))
    for name, value in (
        ("index_version", target_index.index_version),
        ("cache_version", target_index.cache_version),
        ("dataset_id", target_index.dataset_id),
        ("raw_index_fingerprint", target_index.raw_index_fingerprint),
        ("metadata_index_fingerprint", target_index.metadata_index_fingerprint),
        ("records", tuple(rewritten_records)),
        ("index_fingerprint", ""),
    ):
        object.__setattr__(provisional, name, value)
    rewritten_index = replace(
        target_index,
        records=tuple(rewritten_records),
        index_fingerprint=target_cache_module._target_index_fingerprint(provisional),
    )
    rewritten = check_dilemmadata_target_cache(
        rewritten_index,
        raw_index=raw_index,
        cache_config=target_config,
    )
    assert rewritten_index.index_fingerprint != target_index.index_fingerprint
    assert rewritten["record_count"] == len(target_index.records)
    assert rewritten["target_bundle_fingerprint"] == (
        original["target_bundle_fingerprint"]
    )


def test_target_only_mutation_changes_only_target_cache_identity(
    tmp_path: Path,
) -> None:
    (
        corpus,
        raw_config,
        raw_index,
        _,
        before_index,
        _,
    ) = _build(tmp_path)
    source = corpus / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 0, "a_quality", "m")
    after_index, _ = build_dilemmadata_target_cache(
        corpus,
        raw_index=raw_index,
        raw_cache_config=raw_config,
        target_cache_config=DilemmadataTargetCacheConfig(
            tmp_path / "mutated-target-cache"
        ),
        identity=_fixture_identity(corpus),
    )
    before = {row.piece_id: row for row in before_index.records}
    after = {row.piece_id: row for row in after_index.records}
    changed = [
        piece_id
        for piece_id in before
        if before[piece_id].target_bundle_fingerprint
        != after[piece_id].target_bundle_fingerprint
    ]
    assert len(changed) == 1
    for piece_id in before:
        assert before[piece_id].raw_index_fingerprint == after[piece_id].raw_index_fingerprint
        assert before[piece_id].raw_cache_key == after[piece_id].raw_cache_key
        assert before[piece_id].canonical_artifact_sha256 == after[piece_id].canonical_artifact_sha256
    assert before_index.raw_index_fingerprint == after_index.raw_index_fingerprint
    assert before_index.index_fingerprint != after_index.index_fingerprint


def test_lazy_sidecar_workers_are_identical_without_source_or_oracle_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _,
        raw_config,
        raw_index,
        target_config,
        target_index,
        _,
    ) = _build(tmp_path)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("source/oracle access is forbidden at runtime")

    monkeypatch.setattr(raw_adapter_module, "convert_dilemmadata_record", forbidden)
    monkeypatch.setattr(raw_adapter_module, "discover_dilemmadata_corpus", forbidden)
    monkeypatch.setattr(
        target_adapter_module, "build_dilemmadata_target_sidecar", forbidden
    )
    monkeypatch.setattr(
        raw_adapter_module,
        "reconstruct_dilemmadata_alignment_evidence",
        forbidden,
    )
    dataset = IndexedMultiSourceDataset(
        raw_index,
        cache_config=raw_config,
        target_cache_index=target_index,
        target_cache_config=target_config,
        require_target_sidecars=True,
    )
    raw_only = IndexedMultiSourceDataset(
        raw_index, cache_config=raw_config
    )
    for index in range(len(dataset)):
        raw_sample = raw_only[index]
        target_sample = dataset[index]
        assert raw_sample.raw_graph_fingerprint == (
            target_sample.raw_graph_fingerprint
        )
        assert graph_fingerprint(raw_sample.raw_graph) == graph_fingerprint(
            target_sample.raw_graph
        )
        assert model_input_fingerprint(raw_sample.raw_graph) == (
            model_input_fingerprint(target_sample.raw_graph)
        )

    signatures = []
    for workers in (0, 2):
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_multisource_samples,
        )
        signatures.append(tuple(_batch_signature(batch) for batch in loader))
    assert signatures[0] == signatures[1]
    available_task_ids = {
        item.task_id
        for batch in DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_multisource_samples,
        )
        for item in batch.target_batches
        if bool(item.availability_mask.any())
    }
    assert available_task_ids
    assert all(
        task_id.startswith("dilemmadata.")
        for task_id in available_task_ids
    )


def test_required_target_cache_rejects_missing_or_partial_inventory(
    tmp_path: Path,
) -> None:
    _, raw_config, raw_index, target_config, target_index, _ = _build(tmp_path)
    with pytest.raises(Exception, match="required_target_cache_missing"):
        IndexedMultiSourceDataset(
            raw_index,
            cache_config=raw_config,
            require_target_sidecars=True,
        )
    partial = type(target_index).__new__(type(target_index))
    for name in (
        "index_version",
        "cache_version",
        "dataset_id",
        "raw_index_fingerprint",
        "metadata_index_fingerprint",
        "index_fingerprint",
    ):
        object.__setattr__(partial, name, getattr(target_index, name))
    object.__setattr__(partial, "records", target_index.records[:-1])
    with pytest.raises(Exception, match="coverage_mismatch"):
        IndexedMultiSourceDataset(
            raw_index,
            cache_config=raw_config,
            target_cache_index=partial,
            target_cache_config=target_config,
            require_target_sidecars=True,
        )
