"""Tiny generated fixture contracts; no corpus, checkpoint or TEST reader."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from music_critic.adapters.pdmx import PDMX_SCHEMA_PROFILE_ID, PdmxAdapterConfig, load_pdmx_piece
from music_critic.pdmx_pipeline import (
    LineageLink, PdmxPipelineError, PipelineConfig, PdmxWindowDataset,
    SelectionConfig, SourceManifest, SourceRecord, WindowConfig, plan_selection,
    preprocess_pdmx, read_publication,
)
from music_critic.pdmx_pipeline.manifest import source_manifest_from_dict, strict_json
from music_critic.pdmx_pipeline.pipeline import bounded_map
from music_critic.ssl.data import collate_ssl_samples
from music_critic.tasks.corpus import CorpusContractError, load_cached_piece


def _payload(pitch=60):
    return {"resolution": 3, "absolute_time": False,
            "barlines": [{"time": 0}, {"time": 12}, {"time": 24}],
            "time_signatures": [{"time": 0, "numerator": 4, "denominator": 4}],
            "tempos": [{"time": 0, "qpm": 120}, {"time": 13, "qpm": 100}],
            "tracks": [{"program": 0, "is_drum": False, "notes": [
                {"time": 9, "duration": 6, "pitch": pitch},
                {"time": 16, "duration": 2, "pitch": pitch + 1}]}]}


def _sources(tmp_path, count=3):
    root = tmp_path / "sources"
    root.mkdir(parents=True)
    rows = []
    for i in range(count):
        payload = json.dumps(_payload(60 + i)).encode()
        relative = f"source-{i}.json"
        (root / relative).write_bytes(payload)
        rows.append(SourceRecord(f"s{i}", relative, sha256(payload).hexdigest(),
                                 "synthetic", "generated-in-test", synthetic=True))
    manifest = SourceManifest(tuple(rows), PDMX_SCHEMA_PROFILE_ID, "fixture")
    return root, manifest, {r.source_id: "train" for r in rows}


def _config(**kwargs):
    return PipelineConfig(SelectionConfig(("synthetic",)), window=WindowConfig(1, 1), **kwargs)


def _run(tmp_path, *, count=3, **kwargs):
    source, manifest, splits = _sources(tmp_path, count)
    result = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out",
                             config=_config(**kwargs))
    return source, manifest, splits, result


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_manifest_roundtrip_unknown_release_and_explicit_fixture(tmp_path):
    _, manifest, _ = _sources(tmp_path)
    assert manifest.release_identity is None
    assert source_manifest_from_dict(json.loads(json.dumps(manifest.to_dict()))) == manifest
    assert replace(manifest, records=tuple(reversed(manifest.records))).fingerprint == manifest.fingerprint
    with pytest.raises(PdmxPipelineError, match="production_bindings_missing"):
        replace(manifest, mode="production")
    with pytest.raises(PdmxPipelineError, match="fixture_source_not_declared"):
        replace(manifest, records=(replace(manifest.records[0], synthetic=False),))


def test_hash_and_lineage_components_precede_filtering(tmp_path):
    _, manifest, splits = _sources(tmp_path)
    a, b, c = manifest.records
    b = replace(b, sha256=a.sha256, license_id="excluded")
    manifest = replace(manifest, records=(a, b, c), lineage_links=(LineageLink("s1", "s2", "supplied:lineage"),))
    plan = plan_selection(manifest, _config().selection, splits)
    assert len(set(dict(plan.components).values())) == 1
    assert plan.exclusions == (("s1", "pdmx.filter_license"),)
    with pytest.raises(PdmxPipelineError, match="split_component_crossing"):
        plan_selection(manifest, _config().selection, {**splits, "s2": "validation"})


@pytest.mark.parametrize("field,value", [("relative_path", "../a"), ("relative_path", "/tmp/a"),
                                        ("relative_path", "a//b"), ("sha256", "unknown")])
def test_manifest_invalid_path_or_hash(tmp_path, field, value):
    _, manifest, _ = _sources(tmp_path, 1)
    with pytest.raises(PdmxPipelineError):
        replace(manifest.records[0], **{field: value})


def test_selection_declared_rating_budget_and_missing_facts(tmp_path):
    _, manifest, splits = _sources(tmp_path, 4)
    rows = tuple(replace(row, rating=rating) for row, rating in zip(manifest.records, [None, 2, 5, 5]))
    plan = plan_selection(replace(manifest, records=rows), SelectionConfig(("synthetic",), min_rating=3, max_records=1), splits)
    assert [r.source_id for r in plan.records] == ["s2"]
    assert dict(plan.exclusions) == {"s0": "pdmx.filter_rating_missing", "s1": "pdmx.filter_rating", "s3": "pdmx.filter_record_budget"}
    no_facts = replace(manifest, records=(replace(manifest.records[0], license_evidence=None),))
    assert plan_selection(no_facts, _config().selection, {"s0": "train"}).exclusions[0][1] == "pdmx.filter_license_evidence_missing"


def test_test_guard_before_source_cache_or_index_reader(tmp_path, monkeypatch):
    source, manifest, splits = _sources(tmp_path, 1)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("reader must remain unopened")
    with pytest.raises(PdmxPipelineError, match="split_locked"):
        preprocess_pdmx(manifest, {"s0": "test"}, source_root=source, output_root=tmp_path / "out",
                        config=_config(), reader=forbidden)
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(PdmxPipelineError, match="split_locked"):
        PdmxWindowDataset.from_output(tmp_path / "missing", split="test")
    assert not calls


def test_workers_are_deterministic_and_raw_windows_collate(tmp_path):
    source, manifest, splits = _sources(tmp_path, 4)
    splits["s3"] = "validation"
    first = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "one", config=_config())
    relocated = tmp_path / "relocated-sources"
    shutil.copytree(source, relocated)
    second = preprocess_pdmx(manifest, splits, source_root=relocated, output_root=tmp_path / "two", config=_config(workers=2, max_in_flight=2))
    assert _files(first.root) == _files(second.root)
    assert first.report["input_records"] == first.report["input_groups"] == 4
    assert first.report["windows"] == 8
    assert first.report["graph_sizes"] == {"status": "not_computed", "denominator_graphs": 0}
    dataset = PdmxWindowDataset.from_output(first.root, split="train")
    assert len(dataset) == 6
    batch = collate_ssl_samples([dataset[0], dataset[1]])
    assert batch.sample_count == 2
    assert len(PdmxWindowDataset.from_output(first.root, split="validation")) == 2
    for row in first.index.records:
        piece = load_cached_piece(row, first.cache_config)
        assert piece.targets == piece.annotations == ()


def test_resume_is_byte_identical_and_never_reparses_completed_source(tmp_path):
    source, manifest, splits = _sources(tmp_path)
    first = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "complete", config=_config())
    def interrupt(_):
        raise RuntimeError("injected interruption after commit")
    with pytest.raises(RuntimeError, match="injected interruption"):
        preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "resume", config=_config(), on_record_committed=interrupt)
    namespaces = list((tmp_path / "resume" / "pdmx-v1.0.0").iterdir())
    assert not (namespaces[0] / "publication.json").exists()
    read = []
    def reader(path, **kwargs):
        read.append(kwargs["source_id"])
        return load_pdmx_piece(path, **kwargs)
    resumed = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "resume", config=_config(), reader=reader)
    assert read == ["s1", "s2"]
    assert resumed.cache_hit_records == 1 and resumed.cache_hit_windows == 2
    assert _files(first.root) == _files(resumed.root)


def test_partial_or_corrupt_artifact_is_not_cache_hit(tmp_path):
    source, manifest, splits, result = _run(tmp_path, count=1)
    (result.root / "journal" / ".partial-orphan").write_text("incomplete")
    again = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=_config())
    assert again.cache_hit_records == 1
    record = result.index.records[0]
    path = result.cache_config.root / result.cache_config.namespace / record.canonical_relative_path
    path.write_text("incomplete")
    with pytest.raises(CorpusContractError, match="fingerprint_mismatch"):
        preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=_config())


@pytest.mark.parametrize("artifact", ["journal", "shard", "publication"])
def test_corrupt_metadata_rejected(tmp_path, artifact):
    source, manifest, splits, result = _run(tmp_path, count=1)
    path = result.root / "publication.json" if artifact == "publication" else next((result.root / ("shards" if artifact == "shard" else "journal")).glob("*.json"))
    path.write_text('{"fingerprint":"wrong"}')
    if artifact == "publication":
        with pytest.raises(PdmxPipelineError, match="artifact_corrupt"):
            read_publication(result.root, split="train")
    else:
        with pytest.raises(PdmxPipelineError):
            preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=_config())


def test_changed_contract_uses_fresh_namespace(tmp_path):
    source, manifest, splits, result = _run(tmp_path, count=1)
    config = replace(_config(), window=WindowConfig(2, 2))
    changed = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=config)
    assert changed.root != result.root and changed.cache_hit_records == 0
    assert len(changed.index.records) == 1 and len(result.index.records) == 2
    assert result.root.exists()
    changed_split = preprocess_pdmx(manifest, {"s0": "validation"}, source_root=source, output_root=tmp_path / "out", config=config)
    assert changed_split.root != changed.root


def test_changed_source_bytes_cannot_resume_old_evidence(tmp_path):
    source, manifest, splits, result = _run(tmp_path, count=1)
    (source / manifest.records[0].relative_path).write_text("changed")
    with pytest.raises(PdmxPipelineError, match="resume_source_changed"):
        preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=_config())


@pytest.mark.parametrize("initial_reason", [
    "pdmx.source_unreadable", "pdmx.source_hash_mismatch", "pdmx.source_bytes_limit",
])
def test_restored_unverified_source_requires_new_output_and_preserves_prior_evidence(tmp_path, initial_reason):
    source, manifest, splits = _sources(tmp_path, 1)
    path = source / manifest.records[0].relative_path
    expected = path.read_bytes()
    config = replace(_config(), adapter=PdmxAdapterConfig(max_source_bytes=len(expected)))
    if initial_reason == "pdmx.source_unreadable":
        path.unlink()
    elif initial_reason == "pdmx.source_hash_mismatch":
        path.write_bytes(b"not the expected source")
    else:
        path.write_bytes(b"x" * (len(expected) + 1))
    output = tmp_path / "first-attempt"
    first = preprocess_pdmx(manifest, splits, source_root=source, output_root=output, config=config)
    assert first.report["accepted_records"] == 0
    assert first.report["reasons"] == {initial_reason: 1}
    assert first.cache_hit_records == first.cache_hit_windows == 0
    shard = strict_json(next((first.root / "shards").glob("*.json")).read_text())
    assert shard["source_hash_observed"] is False
    evidence = _files(first.root)
    assert "publication.json" in evidence
    assert any(name.startswith("journal/") for name in evidence)
    assert any(name.startswith("shards/") for name in evidence)
    reads = []

    def forbidden_reader(*args, **kwargs):
        reads.append(True)
        raise AssertionError("old attempt must not reparse source")

    unchanged = preprocess_pdmx(manifest, splits, source_root=source, output_root=output,
                                config=config, reader=forbidden_reader)
    assert unchanged.report == first.report
    assert unchanged.cache_hit_records == unchanged.cache_hit_windows == 0
    assert _files(first.root) == evidence

    path.write_bytes(expected)
    with pytest.raises(PdmxPipelineError, match="resume_source_changed") as error:
        preprocess_pdmx(manifest, splits, source_root=source, output_root=output,
                        config=config, reader=forbidden_reader)
    assert "new output root" in str(error.value)
    assert _files(first.root) == evidence
    assert not reads

    retry_output = tmp_path / "explicit-retry"
    retry = preprocess_pdmx(manifest, splits, source_root=source, output_root=retry_output, config=config)
    assert retry.report["accepted_records"] == 1 and retry.report["quarantined_records"] == 0
    assert retry.report["windows"] == 2
    assert retry.cache_hit_records == retry.cache_hit_windows == 0
    retry_evidence = _files(retry.root)
    cached = preprocess_pdmx(manifest, splits, source_root=source, output_root=retry_output,
                             config=config, reader=forbidden_reader)
    assert cached.report == retry.report
    assert cached.cache_hit_records == 1 and cached.cache_hit_windows == 2
    assert _files(retry.root) == retry_evidence
    assert _files(first.root) == evidence
    assert not reads


@pytest.mark.parametrize("programming_error", [TypeError, RuntimeError])
def test_bad_empty_oversize_isolated_programming_exceptions_propagate(tmp_path, programming_error):
    source, manifest, splits = _sources(tmp_path, 4)
    payloads = [b"not JSON", b'{"resolution":3,"tracks":[]}', b" " * 4096]
    rows = list(manifest.records)
    for i, payload in enumerate(payloads):
        (source / rows[i].relative_path).write_bytes(payload)
        rows[i] = replace(rows[i], sha256=sha256(payload).hexdigest())
    manifest = replace(manifest, records=tuple(rows))
    config = replace(_config(), adapter=PdmxAdapterConfig(max_source_bytes=2048))
    result = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=config)
    assert result.report["accepted_records"] == 1
    assert result.report["quarantined_records"] == 3
    assert result.report["reasons"] == {"empty_score": 1, "invalid_json": 1, "pdmx.source_bytes_limit": 1}
    repeated = preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out", config=config)
    assert repeated.report == result.report
    # The unchanged oversize rejection is re-observed, not claimed as hash-verified cache hit.
    assert repeated.cache_hit_records == 3
    assert result.report["resource_rejected_records"] == 1
    def broken(*args, **kwargs):
        raise programming_error("programming defect")
    with pytest.raises(programming_error, match="programming defect"):
        preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "broken", config=config, reader=broken)


def test_invalid_decimal_json_is_quarantined_without_stopping_valid_source(tmp_path):
    source, manifest, splits = _sources(tmp_path, 2)
    bad = (b'{"resolution":480,"tempos":[{"time":0,"qpm":1e999999999999999999999999999999}],'
           b'"tracks":[{"notes":[{"time":0,"duration":480,"pitch":60}]}]}')
    row = manifest.records[0]
    (source / row.relative_path).write_bytes(bad)
    manifest = replace(manifest, records=(replace(row, sha256=sha256(bad).hexdigest()), manifest.records[1]))
    result = preprocess_pdmx(manifest, splits, source_root=source,
                             output_root=tmp_path / "out", config=_config())
    assert result.report["reasons"] == {"invalid_json": 1}
    assert result.report["quarantined_records"] == 1
    assert result.report["accepted_records"] == 1 and result.report["windows"] == 2
    assert {record.source_identity for record in result.index.records} == {"s1"}
    assert (result.root / "publication.json").is_file()


def test_lazy_reader_opens_one_window_at_indexing_only(tmp_path):
    _, _, _, result = _run(tmp_path, count=2)
    reads = []
    def reader(row, cache):
        reads.append(row.piece_id)
        return load_cached_piece(row, cache)
    dataset = PdmxWindowDataset(result.index, result.split_manifest, cache_config=result.cache_config, split="train", reader=reader)
    assert not reads
    dataset[2]
    assert reads == [dataset.records[2].piece_id]
    with pytest.raises(IndexError):
        dataset[-1]
    assert len(reads) == 1
    with pytest.raises(PdmxPipelineError, match="split_locked"):
        PdmxWindowDataset(result.index, result.split_manifest, cache_config=result.cache_config, split="test", reader=reader)
    assert len(reads) == 1


def test_graph_sizes_only_actual_graphs(tmp_path):
    _, _, _, result = _run(tmp_path, count=2, graph_statistics=True)
    stats = result.report["graph_sizes"]
    assert stats["status"] == "computed" and stats["denominator_graphs"] == 4
    assert stats["nodes"]["p50"] > 0 and stats["edges"]["p99"] >= stats["edges"]["p50"]
    assert result.report["throughput"] == result.report["corpus_distribution"] == "not_computed"


def test_bounded_generated_sequence_does_not_eagerly_consume():
    generated = []
    def values():
        for i in range(60):
            generated.append(i)
            yield i
    iterator = bounded_map(lambda i: i * i, values(), workers=2, max_in_flight=3)
    assert generated == []
    assert next(iterator) == 0
    assert len(generated) == 3
    assert next(iterator) == 1
    assert len(generated) == 4
    assert list(iterator) == [i * i for i in range(2, 60)]


def test_cli_requires_explicit_inputs_and_never_scans_on_import(tmp_path):
    from scripts.preprocess_pdmx import main
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_json_manifest_duplicate_or_nonfinite_rejected():
    for payload in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(PdmxPipelineError):
            strict_json(payload)


@pytest.mark.parametrize("version", ["PDMX_PROJECTION_VERSION", "PDMX_ADAPTER_VERSION", "WINDOW_VERSION", "PDMX_SCHEMA_UPSTREAM_REVISION"])
def test_loader_rejects_stale_pdmx_versions(tmp_path, monkeypatch, version):
    import music_critic.pdmx_pipeline.pipeline as module
    _, _, _, result = _run(tmp_path, count=1)
    monkeypatch.setattr(module, version, "future-contract")
    with pytest.raises(PdmxPipelineError, match="cached_contract_stale"):
        PdmxWindowDataset.from_output(result.root, split="train")
    with pytest.raises(PdmxPipelineError, match="cached_contract_stale"):
        PdmxWindowDataset(result.index, result.split_manifest, cache_config=result.cache_config, split="train")


def test_loader_rejects_stale_shared_graph_contract(tmp_path, monkeypatch):
    import music_critic.tasks.corpus as module
    _, _, _, result = _run(tmp_path, count=1)
    monkeypatch.setattr(module, "GRAPH_BUILDER_VERSION", "changed")
    with pytest.raises(CorpusContractError, match="contract_mismatch"):
        PdmxWindowDataset(result.index, result.split_manifest, cache_config=result.cache_config, split="train")


def test_empty_production_manifest_metadata_is_not_fixture_provenance(tmp_path):
    # No real source exists or is opened: verifies future mode metadata only.
    manifest = SourceManifest((), PDMX_SCHEMA_PROFILE_ID, "production",
                              release_identity="declared-source-release", release_sha256="a" * 64,
                              license_artifact_sha256="b" * 64)
    result = preprocess_pdmx(manifest, {}, source_root=tmp_path / "absent", output_root=tmp_path / "out", config=_config())
    assert result.index.header.creation_policy == "pdmx_production_explicit"
    assert result.report["scope"] == "declared_production_preprocessing"
    assert result.report["production_ready"] is False


def test_cli_fixture_end_to_end(tmp_path, capsys):
    from scripts.preprocess_pdmx import main
    source, manifest, splits = _sources(tmp_path, 1)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest.to_dict()))
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    assert main(["--source-root", str(source), "--manifest", str(tmp_path / "manifest.json"),
                 "--splits", str(tmp_path / "splits.json"), "--config", "configs/pdmx/synthetic.json",
                 "--output", str(tmp_path / "out")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["report"]["windows"] == 1
    assert Path(output["publication"]).exists()


def test_graph_identity_excludes_semantic_metadata_split_and_group(tmp_path):
    source, manifest, splits, result = _run(tmp_path, count=1)
    before = PdmxWindowDataset.from_output(result.root, split="train")
    expected = [before[i].raw_graph_fingerprint for i in range(len(before))]
    payload = _payload()
    payload.update(song_length=999999, annotations=[{"time": 999999, "annotation": "Roman V"}],
                   lyrics=[{"time": 999999, "text": "semantic"}], metadata={"title": "changed title"})
    payload["tracks"][0]["role"] = "melody"
    encoded = json.dumps(payload).encode()
    row = manifest.records[0]
    (source / row.relative_path).write_bytes(encoded)
    updated = replace(row, sha256=sha256(encoded).hexdigest())
    shadow = SourceRecord("shadow", "excluded-absent.json", updated.sha256, synthetic=True)
    changed = replace(manifest, records=(updated, shadow))
    rebuilt = preprocess_pdmx(changed, {"s0": "validation", "shadow": "validation"}, source_root=source,
                              output_root=tmp_path / "out", config=_config())
    assert result.index.records[0].source_group_id != rebuilt.index.records[0].source_group_id
    after = PdmxWindowDataset.from_output(rebuilt.root, split="validation")
    assert [after[i].raw_graph_fingerprint for i in range(len(after))] == expected


def test_reader_must_bind_the_verified_source_bytes(tmp_path):
    source, manifest, splits = _sources(tmp_path, 1)
    def stale_reader(path, **kwargs):
        piece = load_pdmx_piece(path, **kwargs)
        return replace(piece, provenance=tuple(
            replace(prov, checksum_sha256="a" * 64) if prov.kind == "source" else prov
            for prov in piece.provenance))
    with pytest.raises(PdmxPipelineError, match="reader_source_binding_invalid"):
        preprocess_pdmx(manifest, splits, source_root=source, output_root=tmp_path / "out",
                        config=_config(), reader=stale_reader)
    assert not list((tmp_path / "out").rglob("publication.json"))
