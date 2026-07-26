from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest
import torch

import music_critic.tasks.corpus as corpus_module
import music_critic.adapters as adapters_module
from music_critic.adapters import (
    HookTheoryAdapterConfig,
    Pop909ClCorpusRecord,
    convert_hooktheory_record,
    convert_pop909_cl_file,
)
from music_critic.tasks import (
    MIXTURE_SAMPLER_VERSION,
    MULTISOURCE_CACHE_VERSION,
    MULTISOURCE_CORPUS_INDEX_VERSION,
    CanonicalCorpusInput,
    CorpusCacheConfig,
    CorpusContractError,
    DatasetContractError,
    DatasetView,
    DeterministicQuotaSampler,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    MultiSourceDataLoaderConfig,
    build_hooktheory_corpus_cache,
    build_pop909_cl_corpus_cache,
    cache_canonical_corpus,
    collate_multisource_samples,
    corpus_cache_key,
    create_split_manifest,
    dataset_view_report,
    dumps_corpus_index,
    dumps_split_manifest,
    loads_corpus_index,
    loads_split_manifest,
    make_multisource_dataloader,
    multisource_worker_seed,
    plan_group_hash_split,
    prepare_multisource_sample,
    validate_split_manifest,
)
from music_critic.graph import graph_to_dict


def _hook_piece(
    clip_id: str,
    *,
    dataset_id: str,
    source_group_id: str | None = None,
    include_targets: bool = True,
):
    record = {
        "hash": clip_id,
        "split": "train",
        "json": {
            "endBeat": 5,
            "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
            "tempos": [{"beat": 1, "bpm": 120}],
            "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
            "notes": [
                {
                    "beat": 1,
                    "duration": 1,
                    "sd": "1",
                    "octave": 0,
                    "isRest": False,
                }
            ],
            "chords": [
                {
                    "beat": 1,
                    "duration": 2,
                    "root": 1,
                    "type": 5,
                    "inversion": 0,
                    "adds": [],
                    "omits": [],
                    "alterations": [],
                    "suspensions": [],
                    "borrowed": None,
                    "isRest": False,
                    "applied": 0,
                    "alternate": "",
                    "pedal": None,
                }
            ],
        },
    }
    return convert_hooktheory_record(
        clip_id,
        record,
        config=HookTheoryAdapterConfig(
            dataset_name=dataset_id, include_targets=include_targets
        ),
        structure_row={
            "audio_path": f"audio/{clip_id}.mp3",
            "ori_uid": source_group_id or f"source:{clip_id}",
        },
        source_path="4_merged.json",
    )


def _pop_piece(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "001.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=4, denominator=4, time=0
                ),
                mido.MetaMessage("end_of_track", time=1_920),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", channel=0, program=0, time=0),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=1_920),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    chord = mido.MidiTrack(
        [mido.Message("program_change", channel=1, program=0, time=0)]
    )
    for pitch in (60, 64, 67):
        chord.append(
            mido.Message("note_on", channel=1, note=pitch, velocity=70, time=0)
        )
    for index, pitch in enumerate((60, 64, 67)):
        chord.append(
            mido.Message(
                "note_off",
                channel=1,
                note=pitch,
                velocity=0,
                time=1_920 if index == 0 else 0,
            )
        )
    chord.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(chord)
    midi.save(path)
    result = convert_pop909_cl_file(
        Pop909ClCorpusRecord(
            song_id="001",
            path=path,
            relative_path="POP909_processed/001.mid",
            corpus_relative_path="001.mid",
            sha256=sha256(path.read_bytes()).hexdigest(),
            source_group_id="pop909-cl:001",
            lineage_group_id="pop909-lineage:001",
        )
    )
    assert result.status == "accepted"
    return result.piece


def _build_index(
    root: Path,
    dataset_id: str,
    count: int,
    *,
    linked: bool = False,
    include_targets: bool = True,
):
    config = CorpusCacheConfig(root)
    inputs = []
    for ordinal in range(count):
        clip_id = f"{dataset_id}-{ordinal}"
        source_group = "source:linked" if linked and ordinal < 2 else f"source:{clip_id}"
        lineage = "lineage:linked" if linked and 1 <= ordinal < 3 else f"lineage:{clip_id}"
        piece = _hook_piece(
            clip_id,
            dataset_id=dataset_id,
            source_group_id=source_group,
            include_targets=include_targets,
        )
        if not linked:
            lineage = piece.source_group_id
        elif lineage != piece.source_group_id:
            provenance = piece.provenance[0]
            piece = replace(
                piece,
                provenance=(
                    replace(
                        provenance,
                        details=tuple(
                            sorted(
                                (
                                    *provenance.details,
                                    ("lineage_group_id", lineage),
                                )
                            )
                        ),
                    ),
                    *piece.provenance[1:],
                ),
            )
        source_payload = f"source:{dataset_id}:{ordinal}".encode()
        inputs.append(
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=lineage,
                source_identity=clip_id,
                source_relative_path=f"sources/{clip_id}.json",
                source_sha256=sha256(source_payload).hexdigest(),
                suggested_split="train",
            )
        )
    return (
        *cache_canonical_corpus(
            inputs,
            cache_config=config,
            dataset_id=dataset_id,
            adapter_name="test_hook",
            adapter_version="1.0.0",
            adapter_config={"include_targets": include_targets},
            source_identity=f"{dataset_id}-fixture",
            source_fingerprint=sha256(dataset_id.encode()).hexdigest(),
            creation_policy="bounded_test",
        ),
        config,
    )


@pytest.fixture()
def corpus_pair(tmp_path: Path):
    alpha, alpha_report, config = _build_index(tmp_path, "alpha", 3)
    beta, beta_report, _ = _build_index(tmp_path, "beta", 2)
    alpha_manifest = create_split_manifest(
        (alpha,),
        {("alpha", row.piece_id): "train" for row in alpha.records},
        seed=7,
    )
    beta_manifest = create_split_manifest(
        (beta,),
        {("beta", row.piece_id): "train" for row in beta.records},
        seed=7,
    )
    alpha_view = DatasetView(
        IndexedMultiSourceDataset(alpha, cache_config=config),
        alpha_manifest,
        split="train",
    )
    beta_view = DatasetView(
        IndexedMultiSourceDataset(beta, cache_config=config),
        beta_manifest,
        split="train",
    )
    return alpha, beta, alpha_report, beta_report, config, alpha_view, beta_view


def test_contract_versions_are_pinned() -> None:
    assert MULTISOURCE_CORPUS_INDEX_VERSION == "1.0.0"
    assert MULTISOURCE_CACHE_VERSION == "1.0.0"
    assert MIXTURE_SAMPLER_VERSION == "1.0.0"


def test_index_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 2)
    payload = dumps_corpus_index(index)
    assert payload == dumps_corpus_index(index)
    assert loads_corpus_index(payload) == index


def test_index_records_are_sorted(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 3)
    assert tuple(row.piece_id for row in index.records) == tuple(
        sorted(row.piece_id for row in index.records)
    )


def test_index_fingerprint_is_input_order_invariant(tmp_path: Path) -> None:
    pieces = tuple(
        _hook_piece(f"clip-{ordinal}", dataset_id="alpha")
        for ordinal in range(2)
    )
    inputs = tuple(
        CanonicalCorpusInput(
            piece=piece,
            lineage_group_id=piece.source_group_id,
            source_identity=piece.piece_id,
            source_relative_path=f"sources/{piece.piece_id}.json",
            source_sha256=sha256(piece.piece_id.encode()).hexdigest(),
        )
        for piece in pieces
    )
    kwargs = {
        "dataset_id": "alpha",
        "adapter_name": "test",
        "adapter_version": "1.0.0",
        "adapter_config": {},
        "source_identity": "bounded",
        "source_fingerprint": sha256(b"bounded").hexdigest(),
    }
    first, _ = cache_canonical_corpus(
        inputs, cache_config=CorpusCacheConfig(tmp_path / "first"), **kwargs
    )
    second, _ = cache_canonical_corpus(
        reversed(inputs),
        cache_config=CorpusCacheConfig(tmp_path / "second"),
        **kwargs,
    )
    assert first.header.index_fingerprint == second.header.index_fingerprint


def test_index_rejects_absolute_source_path(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 1)
    with pytest.raises(CorpusContractError, match="POSIX relative"):
        replace(index.records[0], source_relative_path="/private/source.json")


def test_index_rejects_absolute_artifact_path(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 1)
    with pytest.raises(CorpusContractError, match="POSIX relative"):
        replace(index.records[0], canonical_relative_path="/cache/piece.json")


def test_index_rejects_path_traversal(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 1)
    with pytest.raises(CorpusContractError, match="normalized POSIX"):
        replace(index.records[0], canonical_relative_path="../escape.json")


def test_duplicate_piece_identity_is_rejected(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 1)
    value = json.loads(dumps_corpus_index(index))
    value["records"].append(value["records"][0])
    value["header"]["record_count"] = 2
    with pytest.raises(CorpusContractError):
        loads_corpus_index(json.dumps(value), require_current=False)


def test_tampered_index_fingerprint_is_rejected(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 1)
    value = json.loads(dumps_corpus_index(index))
    value["header"]["source_identity"] = "tampered"
    with pytest.raises(CorpusContractError, match="fingerprint"):
        loads_corpus_index(json.dumps(value))


def test_stale_contract_version_is_rejected(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 1)
    value = json.loads(dumps_corpus_index(index))
    value["header"]["index_version"] = "0.9.0"
    # Re-fingerprinting stale fixtures is intentionally not a public bypass.
    with pytest.raises(CorpusContractError):
        loads_corpus_index(json.dumps(value))


def test_cache_key_changes_with_source_content(tmp_path: Path) -> None:
    first, _, _ = _build_index(tmp_path / "a", "alpha", 1)
    piece = _hook_piece("alpha-0", dataset_id="alpha")
    changed, _ = cache_canonical_corpus(
        (
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=piece.source_group_id,
                source_identity="alpha-0",
                source_relative_path="sources/alpha-0.json",
                source_sha256=sha256(b"changed").hexdigest(),
            ),
        ),
        cache_config=CorpusCacheConfig(tmp_path / "b"),
        dataset_id="alpha",
        adapter_name="test_hook",
        adapter_version="1.0.0",
        adapter_config={"include_targets": True},
        source_identity="alpha-fixture",
        source_fingerprint=sha256(b"alpha").hexdigest(),
    )
    assert first.records[0].cache_key != changed.records[0].cache_key


def test_cache_key_changes_with_adapter_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {
        "source_identity": "piece",
        "source_sha256": sha256(b"source").hexdigest(),
        "adapter_name": "adapter",
        "adapter_version": "1.0.0",
        "adapter_config_fingerprint": sha256(b"config").hexdigest(),
    }
    baseline = corpus_cache_key(**kwargs)
    assert corpus_cache_key(**{**kwargs, "adapter_version": "1.0.1"}) != baseline
    assert (
        corpus_cache_key(
            **{
                **kwargs,
                "adapter_config_fingerprint": sha256(b"changed").hexdigest(),
            }
        )
        != baseline
    )
    monkeypatch.setattr(corpus_module, "SCHEMA_VERSION", "test-future-schema")
    assert corpus_cache_key(**kwargs) != baseline


def test_target_change_changes_artifact_identity(tmp_path: Path) -> None:
    with_targets, _, _ = _build_index(
        tmp_path / "with", "alpha", 1, include_targets=True
    )
    raw_only, _, _ = _build_index(
        tmp_path / "raw", "alpha", 1, include_targets=False
    )
    assert (
        with_targets.records[0].canonical_sha256
        != raw_only.records[0].canonical_sha256
    )
    assert (
        with_targets.records[0].canonical_relative_path
        != raw_only.records[0].canonical_relative_path
    )
    with_sample = IndexedMultiSourceDataset(
        with_targets, cache_config=CorpusCacheConfig(tmp_path / "with")
    )[0]
    raw_sample = IndexedMultiSourceDataset(
        raw_only, cache_config=CorpusCacheConfig(tmp_path / "raw")
    )[0]
    assert with_sample.raw_graph_fingerprint == raw_sample.raw_graph_fingerprint


def test_cache_artifact_corruption_is_rejected(tmp_path: Path) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 1)
    artifact = config.root / config.namespace / index.records[0].canonical_relative_path
    artifact.write_text("{}", encoding="utf-8")
    dataset = IndexedMultiSourceDataset(index, cache_config=config)
    with pytest.raises(
        DatasetContractError, match="canonical artifact SHA-256 differs"
    ):
        dataset[0]


def test_partial_cache_artifact_is_not_a_cache_hit(tmp_path: Path) -> None:
    index, report, config = _build_index(tmp_path, "alpha", 1)
    artifact = config.root / config.namespace / index.records[0].canonical_relative_path
    artifact.unlink()
    artifact.with_name(f".{artifact.name}.partial").write_text(
        "incomplete", encoding="utf-8"
    )
    with pytest.raises(DatasetContractError, match="cannot read canonical"):
        IndexedMultiSourceDataset(index, cache_config=config)[0]
    assert report.cache_miss_count == 1


def test_dataset_constructor_is_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 1)
    artifact = config.root / config.namespace / index.records[0].canonical_relative_path
    original = Path.read_bytes

    def fail_artifact_reads(path: Path):
        if path == artifact:
            raise AssertionError("artifact read during construction")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_artifact_reads)
    dataset = IndexedMultiSourceDataset(index, cache_config=config)
    assert len(dataset) == 1


def test_dataset_reads_one_item_and_prepares_bound_sample(tmp_path: Path) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 2)
    sample = IndexedMultiSourceDataset(index, cache_config=config)[1]
    assert sample.piece_id == index.records[1].piece_id
    assert sample.dataset_id == "alpha"
    repeated = IndexedMultiSourceDataset(index, cache_config=config)[1]
    assert repeated.raw_graph_fingerprint == sample.raw_graph_fingerprint
    assert repeated.target_bundle == sample.target_bundle


def test_dataset_rejects_negative_index(tmp_path: Path) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 1)
    with pytest.raises(DatasetContractError, match="must lie"):
        IndexedMultiSourceDataset(index, cache_config=config)[-1]


def test_dataset_pickle_round_trip_preserves_lazy_contract(tmp_path: Path) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 1)
    dataset = IndexedMultiSourceDataset(index, cache_config=config)
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[0].piece_id == dataset[0].piece_id


def test_sample_pickle_restores_private_graph_binding(tmp_path: Path) -> None:
    piece = _hook_piece("clip", dataset_id="alpha")
    sample = prepare_multisource_sample(piece)
    restored = pickle.loads(pickle.dumps(sample))
    assert restored.piece_id == sample.piece_id
    assert restored.raw_graph_fingerprint == sample.raw_graph_fingerprint


def test_raw_only_piece_remains_loadable(tmp_path: Path) -> None:
    index, _, config = _build_index(
        tmp_path, "alpha", 1, include_targets=False
    )
    sample = IndexedMultiSourceDataset(index, cache_config=config)[0]
    assert sample.target_bundle == ()
    assert all(not row.family_present for row in sample.target_availability)
    batch = collate_multisource_samples((sample,))
    assert len(batch.target_batches) == 18
    assert all(target.entry_count == 0 for target in batch.target_batches)


def test_explicit_split_manifest_round_trips(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 2)
    manifest = create_split_manifest(
        (index,),
        {
            ("alpha", index.records[0].piece_id): "train",
            ("alpha", index.records[1].piece_id): "valid",
        },
        seed=11,
    )
    assert loads_split_manifest(dumps_split_manifest(manifest)) == manifest


def test_split_manifest_rejects_missing_assignment(tmp_path: Path) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 2)
    with pytest.raises(DatasetContractError, match="cover every piece"):
        create_split_manifest(
            (index,),
            {("alpha", index.records[0].piece_id): "train"},
            seed=1,
        )


def test_split_manifest_rejects_cross_split_source_lineage_closure(
    tmp_path: Path,
) -> None:
    index, _, _ = _build_index(tmp_path, "alpha", 3, linked=True)
    with pytest.raises(DatasetContractError, match="crosses splits"):
        create_split_manifest(
            (index,),
            {
                ("alpha", index.records[0].piece_id): "train",
                ("alpha", index.records[1].piece_id): "train",
                ("alpha", index.records[2].piece_id): "test",
            },
            seed=1,
        )


def test_split_manifest_rejects_index_fingerprint_mismatch(tmp_path: Path) -> None:
    first, _, _ = _build_index(tmp_path / "a", "alpha", 1)
    second, _, _ = _build_index(tmp_path / "b", "alpha", 2)
    manifest = create_split_manifest(
        (first,),
        {("alpha", first.records[0].piece_id): "train"},
        seed=1,
    )
    with pytest.raises(DatasetContractError, match="different corpus index"):
        validate_split_manifest(manifest, (second,))


def test_suggested_split_is_not_an_implicit_manifest(tmp_path: Path) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 1)
    dataset = IndexedMultiSourceDataset(index, cache_config=config)
    with pytest.raises(DatasetContractError):
        DatasetView(
            dataset,
            create_split_manifest(
                (index,),
                {("alpha", index.records[0].piece_id): "valid"},
                seed=1,
            ),
            split="train",
        )[-1]


def test_group_hash_planner_is_input_order_invariant(tmp_path: Path) -> None:
    alpha, _, _ = _build_index(tmp_path, "alpha", 7)
    first = plan_group_hash_split(
        (alpha,), seed=91, ratios={"train": 0.8, "valid": 0.1, "test": 0.1}
    )
    # Invalid input ordering is rejected before planning instead of silently
    # making planner output depend on caller ordering.
    with pytest.raises(CorpusContractError):
        replace(
            alpha,
            records=tuple(reversed(alpha.records)),
        )
    second = plan_group_hash_split(
        (alpha,), seed=91, ratios={"test": 0.1, "train": 0.8, "valid": 0.1}
    )
    assert first.assignments == second.assignments


def test_group_hash_planner_is_target_blind(tmp_path: Path) -> None:
    with_targets, _, _ = _build_index(
        tmp_path / "with", "alpha", 4, include_targets=True
    )
    raw_only, _, _ = _build_index(
        tmp_path / "raw", "alpha", 4, include_targets=False
    )
    first = plan_group_hash_split(
        (with_targets,), seed=2, ratios={"train": 3, "test": 1}
    )
    second = plan_group_hash_split(
        (raw_only,), seed=2, ratios={"train": 3, "test": 1}
    )
    assert tuple((row.piece_id, row.split) for row in first.assignments) == tuple(
        (row.piece_id, row.split) for row in second.assignments
    )


def test_multi_corpus_uses_stable_dataset_ranges(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    mixed = MultiCorpusDataset((beta_view, alpha_view))
    assert tuple(row[0] for row in mixed.global_ranges) == ("alpha", "beta")
    assert mixed[0].dataset_id == "alpha"
    assert mixed[mixed.global_ranges[1][1]].dataset_id == "beta"


def test_multi_corpus_rejects_duplicate_dataset_view(corpus_pair) -> None:
    *_, alpha_view, _ = corpus_pair
    with pytest.raises(DatasetContractError, match="unique"):
        MultiCorpusDataset((alpha_view, alpha_view))


def test_dataset_view_report_uses_index_metadata(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    report = dataset_view_report(MultiCorpusDataset((alpha_view, beta_view)))
    assert report.piece_count == 5
    assert report.constituent_counts == (("alpha", 3), ("beta", 2))


def test_sampler_largest_remainder_quotas_are_exact(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    mixed = MultiCorpusDataset((alpha_view, beta_view))
    sampler = DeterministicQuotaSampler(
        mixed, weights={"beta": 1, "alpha": 2}, seed=5, epoch_size=10
    )
    assert sampler.quotas == (("alpha", 7), ("beta", 3))
    assert len(tuple(sampler)) == 10


def test_sampler_same_seed_and_epoch_replay(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    mixed = MultiCorpusDataset((alpha_view, beta_view))
    first = DeterministicQuotaSampler(
        mixed, weights={"alpha": 1, "beta": 1}, seed=9, epoch_size=8
    )
    second = DeterministicQuotaSampler(
        mixed, weights={"beta": 1, "alpha": 1}, seed=9, epoch_size=8
    )
    assert tuple(first) == tuple(second)
    assert first.last_evidence == second.last_evidence


def test_sampler_epoch_changes_schedule(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    sampler = DeterministicQuotaSampler(
        MultiCorpusDataset((alpha_view, beta_view)),
        weights={"alpha": 1, "beta": 1},
        seed=9,
        epoch_size=8,
    )
    first = tuple(sampler)
    sampler.set_epoch(1)
    assert tuple(sampler) != first


def test_sampler_does_not_repeat_before_local_exhaustion(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    mixed = MultiCorpusDataset((alpha_view, beta_view))
    sampler = DeterministicQuotaSampler(
        mixed, weights={"alpha": 100, "beta": 1}, seed=3, epoch_size=4
    )
    selected = tuple(sampler)
    alpha_start, alpha_end = mixed.global_ranges[0][1:]
    alpha_values = [value for value in selected if alpha_start <= value < alpha_end]
    assert len(alpha_values[:3]) == len(set(alpha_values[:3]))


def test_sampler_rejects_missing_weight(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    with pytest.raises(DatasetContractError, match="cover"):
        DeterministicQuotaSampler(
            MultiCorpusDataset((alpha_view, beta_view)),
            weights={"alpha": 1},
            seed=1,
            epoch_size=3,
        )


@pytest.mark.parametrize("epoch_size", (0, -1))
def test_sampler_rejects_invalid_epoch_size(corpus_pair, epoch_size: int) -> None:
    *_, alpha_view, beta_view = corpus_pair
    with pytest.raises(DatasetContractError, match="epoch_size"):
        DeterministicQuotaSampler(
            MultiCorpusDataset((alpha_view, beta_view)),
            weights={"alpha": 1, "beta": 1},
            seed=1,
            epoch_size=epoch_size,
        )


def test_sampler_rejects_zero_or_negative_weights(corpus_pair) -> None:
    *_, alpha_view, beta_view = corpus_pair
    mixed = MultiCorpusDataset((alpha_view, beta_view))
    for weight in (0, -1):
        with pytest.raises(DatasetContractError, match="positive"):
            DeterministicQuotaSampler(
                mixed,
                weights={"alpha": 1, "beta": weight},
                seed=1,
                epoch_size=3,
            )


def _loader_evidence(mixed, *, workers: int):
    sampler = DeterministicQuotaSampler(
        mixed, weights={"alpha": 1, "beta": 1}, seed=17, epoch_size=4
    )
    config = MultiSourceDataLoaderConfig(
        batch_size=2,
        num_workers=workers,
        seed=23,
        prefetch_factor=2 if workers else None,
        multiprocessing_context="spawn" if workers else None,
    )
    evidence = []
    for batch in make_multisource_dataloader(
        mixed, sampler=sampler, config=config
    ):
        def normalized(value):
            if isinstance(value, torch.Tensor):
                return value.item() if value.numel() == 1 else value.tolist()
            if isinstance(value, dict):
                return {key: normalized(item) for key, item in value.items()}
            if isinstance(value, list):
                return [normalized(item) for item in value]
            return value

        graph_fingerprints = tuple(
            sha256(
                json.dumps(
                    normalized(graph_to_dict(graph)),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for graph in batch.raw_graph_batch.to_data_list()
        )
        targets = tuple(
            (
                target.task_id,
                target.values.tolist()
                if isinstance(target.values, torch.Tensor)
                else target.values,
                target.availability_mask.tolist(),
                target.entity_indices.tolist(),
                target.entity_index_mask.tolist(),
            )
            for target in batch.target_batches
        )
        evidence.append(
            (
                batch.dataset_ids,
                batch.piece_ids,
                graph_fingerprints,
                targets,
            )
        )
    return tuple(evidence)


def test_dataloader_num_workers_zero_and_two_have_schedule_parity(
    corpus_pair,
) -> None:
    *_, alpha_view, beta_view = corpus_pair
    mixed = MultiCorpusDataset((alpha_view, beta_view))
    assert _loader_evidence(mixed, workers=0) == _loader_evidence(
        mixed, workers=2
    )


def test_worker_exception_retains_structured_category(tmp_path: Path) -> None:
    index, _, config = _build_index(tmp_path, "alpha", 1)
    manifest = create_split_manifest(
        (index,), {("alpha", index.records[0].piece_id): "train"}, seed=1
    )
    view = DatasetView(
        IndexedMultiSourceDataset(index, cache_config=config),
        manifest,
        split="train",
    )
    mixed = MultiCorpusDataset((view,))
    sampler = DeterministicQuotaSampler(
        mixed, weights={"alpha": 1}, seed=1, epoch_size=1
    )
    artifact = config.root / config.namespace / index.records[0].canonical_relative_path
    artifact.write_text("corrupt", encoding="utf-8")
    loader = make_multisource_dataloader(
        mixed,
        sampler=sampler,
        config=MultiSourceDataLoaderConfig(
            batch_size=1,
            num_workers=2,
            seed=1,
            prefetch_factor=2,
            multiprocessing_context="spawn",
        ),
    )
    with pytest.raises(DatasetContractError) as caught:
        next(iter(loader))
    assert "corpus_cache.artifact_fingerprint_mismatch" in str(caught.value)


def test_dataloader_worker_options_are_explicit() -> None:
    with pytest.raises(DatasetContractError, match="require num_workers"):
        MultiSourceDataLoaderConfig(
            batch_size=2,
            num_workers=0,
            seed=1,
            persistent_workers=True,
        )


def test_worker_seed_uses_pytorch_worker_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.initial_seed", lambda: 2**32 + 101)
    assert multisource_worker_seed(0) == 101
    monkeypatch.setattr("torch.initial_seed", lambda: 2**32 + 102)
    assert multisource_worker_seed(1) == 102


def test_bounded_pop_piece_uses_dataset_to_loader_path(tmp_path: Path) -> None:
    piece = _pop_piece(tmp_path / "source")
    config = CorpusCacheConfig(tmp_path / "cache")
    index, report = cache_canonical_corpus(
        (
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id="pop909-lineage:001",
                source_identity="001",
                source_relative_path="POP909_processed/001.mid",
                source_sha256=sha256(b"bounded-pop").hexdigest(),
            ),
        ),
        cache_config=config,
        dataset_id=piece.dataset_name,
        adapter_name="pop909_cl",
        adapter_version="1.0.0",
        adapter_config={"include_targets": True},
        source_identity="bounded-pop",
        source_fingerprint=sha256(b"bounded-pop-source").hexdigest(),
        creation_policy="bounded_test",
    )
    manifest = create_split_manifest(
        (index,), {(piece.dataset_name, piece.piece_id): "train"}, seed=1
    )
    view = DatasetView(
        IndexedMultiSourceDataset(index, cache_config=config),
        manifest,
        split="train",
    )
    mixed = MultiCorpusDataset((view,))
    sampler = DeterministicQuotaSampler(
        mixed, weights={piece.dataset_name: 1}, seed=1, epoch_size=1
    )
    batch = next(
        iter(
            make_multisource_dataloader(
                mixed,
                sampler=sampler,
                config=MultiSourceDataLoaderConfig(
                    batch_size=1, num_workers=0, seed=1
                ),
            )
        )
    )
    assert batch.piece_ids == (piece.piece_id,)
    assert report.accepted_count == 1
    no_chord = next(
        target
        for target in batch.target_batches
        if target.task_id == "pop909_cl.chord.no_chord"
    )
    boundary = next(
        target
        for target in batch.target_batches
        if target.task_id == "pop909_cl.chord.boundary"
    )
    assert set(no_chord.values.tolist()) <= {0}
    assert set(boundary.values.tolist()) <= {0}


def test_pop_song_172_quarantine_never_becomes_dataset_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "172.mid"
    source.write_bytes(b"bounded-routing-fixture")
    record = Pop909ClCorpusRecord(
        song_id="172",
        path=source,
        relative_path="POP909_processed/172.mid",
        corpus_relative_path="172.mid",
        sha256=sha256(source.read_bytes()).hexdigest(),
        source_group_id="pop909-cl:172",
        lineage_group_id="pop909-lineage:172",
    )

    class FakeQuarantine:
        category = "midi_adapter.meter_change_inside_bar"
        source_error = "bounded song 172 quarantine"

    monkeypatch.setattr(
        adapters_module, "Pop909ClQuarantine", FakeQuarantine
    )
    monkeypatch.setattr(
        adapters_module,
        "discover_pop909_cl_corpus",
        lambda _root: SimpleNamespace(
            records=(record,),
            corpus_root=tmp_path,
            content_fingerprint=sha256(b"bounded-discovery").hexdigest(),
        ),
    )
    monkeypatch.setattr(
        adapters_module,
        "convert_pop909_cl_file",
        lambda _record, config: FakeQuarantine(),
    )
    index, report = build_pop909_cl_corpus_cache(
        tmp_path,
        cache_config=CorpusCacheConfig(tmp_path / "cache"),
        limit=1,
    )
    assert index.records == ()
    assert report.accepted_count == 0
    assert report.quarantined_count == 1
    assert report.quarantine[0].source_identity == "172"


def test_mixed_hook_pop_raw_only_dataloader_batch_is_valid(tmp_path: Path) -> None:
    hook, _, config = _build_index(tmp_path / "cache", "hook", 1)
    raw, _, _ = _build_index(
        tmp_path / "cache", "raw", 1, include_targets=False
    )
    pop_piece = _pop_piece(tmp_path / "pop-source")
    pop, _ = cache_canonical_corpus(
        (
            CanonicalCorpusInput(
                piece=pop_piece,
                lineage_group_id="pop909-lineage:001",
                source_identity="001",
                source_relative_path="POP909_processed/001.mid",
                source_sha256=sha256(b"mixed-pop").hexdigest(),
            ),
        ),
        cache_config=config,
        dataset_id=pop_piece.dataset_name,
        adapter_name="pop909_cl",
        adapter_version="1.0.0",
        adapter_config={"include_targets": True},
        source_identity="mixed-pop",
        source_fingerprint=sha256(b"mixed-pop-source").hexdigest(),
    )

    def view(index):
        manifest = create_split_manifest(
            (index,),
            {(row.dataset_id, row.piece_id): "train" for row in index.records},
            seed=1,
        )
        return DatasetView(
            IndexedMultiSourceDataset(index, cache_config=config),
            manifest,
            split="train",
        )

    mixed = MultiCorpusDataset((view(raw), view(pop), view(hook)))
    sampler = DeterministicQuotaSampler(
        mixed,
        weights={"hook": 1, "pop909_cl": 1, "raw": 1},
        seed=1,
        epoch_size=3,
    )
    batch = next(
        iter(
            make_multisource_dataloader(
                mixed,
                sampler=sampler,
                config=MultiSourceDataLoaderConfig(
                    batch_size=3, num_workers=0, seed=1
                ),
            )
        )
    )
    assert set(batch.dataset_ids) == {"hook", "pop909_cl", "raw"}
    assert batch.statistics.sample_count == 3
    assert len(batch.target_batches) == 18


def test_hooktheory_stream_builder_reports_unusable_records(tmp_path: Path) -> None:
    valid_piece = _hook_piece("valid", dataset_id="hooktheory")
    record = {
        "hash": "valid",
        "split": "train",
        "json": {
            "endBeat": 5,
            "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
            "tempos": [{"beat": 1, "bpm": 120}],
            "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
            "notes": [
                {
                    "beat": 1,
                    "duration": 1,
                    "sd": "1",
                    "octave": 0,
                    "isRest": False,
                }
            ],
            "chords": [],
        },
    }
    raw = tmp_path / "4_merged.json"
    raw.write_text(
        json.dumps(
            {
                "valid": record,
                "unusable": {"hash": "unusable", "split": "train", "json": None},
            }
        ),
        encoding="utf-8",
    )
    index, report = build_hooktheory_corpus_cache(
        raw,
        cache_config=CorpusCacheConfig(tmp_path / "cache"),
        limit=2,
    )
    assert len(index.records) == 1
    assert report.accepted_count == 1
    assert report.quarantined_count == 1
    assert valid_piece.dataset_name == index.header.dataset_id
