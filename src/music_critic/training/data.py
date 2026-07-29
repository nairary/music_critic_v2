"""Bounded and production-cache data runtimes for Phase 6C."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mido
import torch
from torch.utils.data import DataLoader, Sampler

from music_critic.data.validation_membership import (
    FixedValidationMembership,
    fixed_validation_membership,
)
from music_critic.adapters import (
    HookTheoryAdapterConfig,
    Pop909ClCorpusRecord,
    convert_hooktheory_record,
    convert_pop909_cl_file,
    inspect_pop909_cl_instruments,
    pop909_cl_raw_input_group_id,
    project_pop909_cl_score_bytes,
)
from music_critic.tasks import (
    CorpusCacheConfig,
    DeterministicQuotaSampler,
    IndexedMultiSourceDataset,
    MultiCorpusDataset,
    MultiSourceBatch,
    MultiSourceDataLoaderConfig,
    collate_multisource_samples,
    dataset_view_report,
    load_corpus_index,
    load_split_manifest,
    make_multisource_dataloader,
    prepare_multisource_sample,
    seed_multisource_worker,
)
from music_critic.training.config import DataConfig


@dataclass(frozen=True, slots=True)
class ValidationMembership:
    """Fixed no-replacement validation membership and evidence."""

    identities: tuple[tuple[str, str], ...]
    membership_fingerprint: str
    dataset_counts: dict[str, int]
    full_view_count: int
    selected_count: int
    subset_limit: int


@dataclass(frozen=True, slots=True)
class DataRuntime:
    """Epoch factories plus immutable data/split fingerprint evidence."""

    first_train_batch: MultiSourceBatch
    train_loader: Callable[[int], Iterable[MultiSourceBatch]]
    validation_loader: Callable[[], Iterable[MultiSourceBatch]]
    validation_membership: ValidationMembership
    fingerprints: dict[str, object]
    mixture_statistics: dict[str, object]


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _training_membership(
    selection: FixedValidationMembership,
) -> ValidationMembership:
    return ValidationMembership(
        identities=selection.identities,
        membership_fingerprint=selection.membership_fingerprint,
        dataset_counts=selection.dataset_counts,
        full_view_count=selection.full_view_count,
        selected_count=selection.selected_count,
        subset_limit=selection.subset_limit,
    )


class _FixedIndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _fixed_validation_loader(
    dataset: Any,
    indices: Sequence[int],
    *,
    batch_size: int,
    workers: int,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=_FixedIndexSampler(indices),
        num_workers=workers,
        collate_fn=collate_multisource_samples,
        worker_init_fn=seed_multisource_worker,
        generator=generator,
        persistent_workers=False,
    )


def _hook_piece(piece_id: str, root: int):
    record = {
        "hash": piece_id,
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
                    "root": root,
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
        piece_id,
        record,
        config=HookTheoryAdapterConfig(dataset_name="hooktheory"),
        structure_row={
            "audio_path": f"audio/{piece_id}.mp3",
            "ori_uid": f"hook-source:{piece_id}",
        },
        source_path="4_merged.json",
    )


def _pop_piece(
    root: Path,
    song_id: str,
    pitches: tuple[int, ...],
):
    path = root / f"{song_id}.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage(
                    "time_signature",
                    numerator=4,
                    denominator=4,
                    time=0,
                ),
                mido.MetaMessage("end_of_track", time=1_920),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message(
                    "program_change", channel=0, program=0, time=0
                ),
                mido.Message(
                    "note_on", channel=0, note=60, velocity=80, time=0
                ),
                mido.Message(
                    "note_off",
                    channel=0,
                    note=60,
                    velocity=0,
                    time=1_920,
                ),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    chord = mido.MidiTrack(
        [mido.Message("program_change", channel=1, program=0, time=0)]
    )
    for pitch in pitches:
        chord.append(
            mido.Message(
                "note_on",
                channel=1,
                note=pitch,
                velocity=70,
                time=0,
            )
        )
    for index, pitch in enumerate(pitches):
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
    resolution = inspect_pop909_cl_instruments(midi)
    source_group_id = pop909_cl_raw_input_group_id(
        sha256(
            project_pop909_cl_score_bytes(midi, resolution)
        ).hexdigest()
    )
    record = Pop909ClCorpusRecord(
        song_id=song_id,
        path=path,
        relative_path=f"POP909_processed/{song_id}.mid",
        corpus_relative_path=f"{song_id}.mid",
        sha256=sha256(path.read_bytes()).hexdigest(),
        source_group_id=source_group_id,
        lineage_group_id=f"pop909-lineage:{song_id}",
    )
    result = convert_pop909_cl_file(record)
    if result.status != "accepted" or result.piece is None:
        raise RuntimeError("bounded POP909-CL fixture was not accepted")
    return result.piece


def _bounded_samples() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    with TemporaryDirectory(prefix="music-critic-phase6c-bounded-") as tmp:
        root = Path(tmp)
        train_hook = _hook_piece("bounded-train-hook", 1)
        train_pop = _pop_piece(root, "901", (60, 64, 67))
        train_raw = replace(
            _hook_piece("bounded-train-raw", 5),
            annotations=(),
            targets=(),
        )
        validation_hook = _hook_piece("bounded-validation-hook", 3)
        validation_pop = _pop_piece(root, "902", (62, 65, 69))
        validation_raw = replace(
            _hook_piece("bounded-validation-raw", 7),
            annotations=(),
            targets=(),
        )
        train = tuple(
            prepare_multisource_sample(piece)
            for piece in (train_hook, train_pop, train_raw)
        )
        validation = tuple(
            prepare_multisource_sample(piece)
            for piece in (
                validation_hook,
                validation_pop,
                validation_raw,
            )
        )
    return train, validation


def _bounded_epoch(
    samples: tuple[Any, ...],
    *,
    batch_size: int,
    epoch_size: int,
    seed: int,
    epoch: int,
) -> tuple[MultiSourceBatch, ...]:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    order = []
    while len(order) < epoch_size:
        order.extend(
            torch.randperm(len(samples), generator=generator).tolist()
        )
    selected = tuple(samples[index] for index in order[:epoch_size])
    return tuple(
        collate_multisource_samples(
            selected[start : start + batch_size]
        )
        for start in range(0, len(selected), batch_size)
    )


def _bounded_runtime(config: DataConfig | Any, seed: int) -> DataRuntime:
    train, validation = _bounded_samples()
    first = collate_multisource_samples(train[: config.batch_size])
    validation_identities = tuple(
        (sample.dataset_id, sample.piece_id) for sample in validation
    )
    validation_selection = fixed_validation_membership(
        validation_identities,
        limit=config.validation_epoch_size,
        seed=seed,
    )
    validation_indices = validation_selection.indices
    membership = _training_membership(validation_selection)

    def train_loader(epoch: int):
        return _bounded_epoch(
            train,
            batch_size=config.batch_size,
            epoch_size=config.epoch_size,
            seed=seed,
            epoch=epoch,
        )

    def validation_loader():
        selected = tuple(validation[index] for index in validation_indices)
        return tuple(
            collate_multisource_samples(
                selected[start : start + config.batch_size]
            )
            for start in range(0, len(selected), config.batch_size)
        )

    identities = {
        "train": [
            [sample.dataset_id, sample.piece_id] for sample in train
        ],
        "validation": [
            [sample.dataset_id, sample.piece_id]
            for sample in validation
        ],
    }
    return DataRuntime(
        first_train_batch=first,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_membership=membership,
        fingerprints={
            "kind": "bounded",
            "bounded_fixture_fingerprint": _fingerprint(identities),
            "split_fingerprint": _fingerprint(identities),
            "validation_membership_fingerprint": (
                membership.membership_fingerprint
            ),
        },
        mixture_statistics={
            "requested_weights": dict(config.mixture_weights),
            "train_dataset_counts": dict(
                sorted(Counter(item.dataset_id for item in train).items())
            ),
            "validation_dataset_counts": dict(
                membership.dataset_counts
            ),
            "validation_membership": asdict(membership),
        },
    )


def _selected_groups(dataset: MultiCorpusDataset) -> tuple[set[str], set[str]]:
    sources = set()
    lineages = set()
    for view in dataset.views:
        for index in view.record_indices:
            record = view.dataset.index.records[index]
            sources.add(record.source_group_id)
            lineages.add(record.lineage_group_id)
    return sources, lineages


def _corpus_runtime(config: DataConfig | Any, seed: int) -> DataRuntime:
    if (
        not config.index_paths
        or len(config.index_paths) != len(config.cache_roots)
        or not config.split_manifest
    ):
        raise ValueError("training.data.corpus_paths_incomplete")
    indices = tuple(load_corpus_index(path) for path in config.index_paths)
    indexed = tuple(
        IndexedMultiSourceDataset(
            index,
            cache_config=CorpusCacheConfig(Path(cache_root)),
        )
        for index, cache_root in zip(
            indices, config.cache_roots, strict=True
        )
    )
    manifest = load_split_manifest(config.split_manifest)
    train = MultiCorpusDataset(
        indexed, manifest, split=config.train_split
    )
    validation = MultiCorpusDataset(
        indexed, manifest, split=config.validation_split
    )
    train_sources, train_lineages = _selected_groups(train)
    val_sources, val_lineages = _selected_groups(validation)
    if train_sources & val_sources or train_lineages & val_lineages:
        raise ValueError("training.data.split_isolation_failed")
    weights = dict(config.mixture_weights)
    validation_identities = tuple(
        validation.record_identity(index)
        for index in range(len(validation))
    )
    validation_selection = fixed_validation_membership(
        validation_identities,
        limit=config.validation_epoch_size,
        seed=seed,
    )
    validation_indices = validation_selection.indices
    membership = _training_membership(validation_selection)

    def loader(dataset, epoch_size: int, epoch: int):
        sampler = DeterministicQuotaSampler(
            dataset,
            weights=weights,
            seed=seed,
            epoch_size=epoch_size,
        )
        sampler.set_epoch(epoch)
        return make_multisource_dataloader(
            dataset,
            sampler=sampler,
            config=MultiSourceDataLoaderConfig(
                batch_size=config.batch_size,
                num_workers=config.workers,
                seed=seed,
            ),
        )

    def train_loader(epoch: int):
        return loader(train, config.epoch_size, epoch)

    def validation_loader():
        return _fixed_validation_loader(
            validation,
            validation_indices,
            batch_size=config.batch_size,
            workers=config.workers,
            seed=seed + 10_000,
        )

    first = next(iter(train_loader(0)))
    return DataRuntime(
        first_train_batch=first,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_membership=membership,
        fingerprints={
            "kind": "corpus_cache",
            "index_fingerprints": [
                [index.header.dataset_id, index.header.index_fingerprint]
                for index in sorted(
                    indices, key=lambda item: item.header.dataset_id
                )
            ],
            "split_manifest_fingerprint": manifest.manifest_fingerprint,
            "train_composition_fingerprint": train.composition_fingerprint,
            "validation_composition_fingerprint": (
                validation.composition_fingerprint
            ),
            "validation_membership_fingerprint": (
                membership.membership_fingerprint
            ),
        },
        mixture_statistics={
            "requested_weights": weights,
            "train": asdict(dataset_view_report(train)),
            "validation": asdict(dataset_view_report(validation)),
            "validation_membership": asdict(membership),
        },
    )


def build_data_runtime(
    config: DataConfig | Any,
    *,
    seed: int,
) -> DataRuntime:
    if config.name == "bounded":
        return _bounded_runtime(config, seed)
    if config.name in {"hooktheory", "pop909_cl", "mixed"}:
        return _corpus_runtime(config, seed)
    raise ValueError(f"training.data.unknown:{config.name}")


__all__ = [
    "DataRuntime",
    "ValidationMembership",
    "build_data_runtime",
]
