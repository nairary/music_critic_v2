"""Deterministic, per-sample Phase 7A note-pitch mask planning."""

from __future__ import annotations

import math
from collections.abc import Sequence

from torch_geometric.data import Batch, HeteroData

from music_critic.graph import validate_raw_graph, validate_raw_graph_batch
from music_critic.ssl.contracts import (
    MASK_POLICY_VERSION,
    UNIFORM_NOTE_MASK_POLICY,
    CollateralFeatureMask,
    MaskPlan,
    MaskStage,
    SSLContractError,
    SampleIdentity,
    StableSeed,
    canonical_sha256,
    validate_global_seed,
    validate_mask_rate,
    validate_non_negative_integer,
)
from music_critic.ssl.field_registry import (
    MASKABLE_FIELD_REGISTRY_FINGERPRINT,
    MASKABLE_FIELD_REGISTRY_VERSION,
    NOTE_PITCH_GROUP,
    NOTE_PITCH_GROUP_NAME,
    SSL_MASKABLE_FIELD_REGISTRY,
)


DEFAULT_ENCODER_MASK_RATE = 0.30
TRACK_CONTAINS_NOTE_EDGE = ("track", "contains_note", "note")


def derive_stable_seed(
    *,
    namespace: str,
    global_seed: int,
    dataset_id: str,
    piece_id: str,
    epoch: int,
    view_index: int,
    extra: object = None,
) -> StableSeed:
    """Derive a portable unsigned seed without Python's process-random hash."""

    if (
        not isinstance(namespace, str)
        or not namespace
        or namespace != namespace.strip()
    ):
        raise SSLContractError("seed namespace must be a non-empty trimmed string")
    validate_global_seed(global_seed)
    identity = SampleIdentity(dataset_id, piece_id)
    validate_non_negative_integer(epoch, name="epoch")
    validate_non_negative_integer(view_index, name="view_index")
    digest = canonical_sha256(
        {
            "namespace": namespace,
            "global_seed": global_seed,
            "sample_identity": identity.to_dict(),
            "epoch": epoch,
            "view_index": view_index,
            "extra": extra,
        }
    )
    return StableSeed(value=int(digest[:16], 16), sha256=digest)


def _canonical_epoch(stage: MaskStage, epoch: int) -> int:
    validate_non_negative_integer(epoch, name="epoch")
    if stage not in {"train", "validation"}:
        raise SSLContractError("stage must be train or validation")
    return epoch if stage == "train" else 0


def _selected_count(node_count: int, requested_rate: float) -> int:
    if node_count == 0 or requested_rate == 0:
        return 0
    if requested_rate == 1:
        return node_count
    return max(1, int(math.floor(node_count * requested_rate)))


def _base_order(
    *,
    node_count: int,
    global_seed: int,
    identity: SampleIdentity,
    stage: MaskStage,
    encoder_view_index: int,
) -> tuple[int, ...]:
    """Create a stable sample-specific permutation from per-index SHA-256 keys."""

    common = {
        "namespace": "music_critic.ssl.encoder_mask.base_order.v1",
        "global_seed": global_seed,
        "sample_identity": identity.to_dict(),
        "stage": stage,
        "encoder_view_index": encoder_view_index,
        "mask_policy": UNIFORM_NOTE_MASK_POLICY,
        "mask_policy_version": MASK_POLICY_VERSION,
        "primary_feature_group": NOTE_PITCH_GROUP_NAME,
        "maskable_field_registry_version": MASKABLE_FIELD_REGISTRY_VERSION,
        "maskable_field_registry_fingerprint": (
            MASKABLE_FIELD_REGISTRY_FINGERPRINT
        ),
    }
    return tuple(
        sorted(
            range(node_count),
            key=lambda local_index: (
                canonical_sha256(
                    {
                        **common,
                        "local_node_index": local_index,
                    }
                ),
                local_index,
            ),
        )
    )


def _sample_local_indices(
    *,
    node_count: int,
    requested_rate: float,
    global_seed: int,
    identity: SampleIdentity,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
) -> tuple[int, ...]:
    selected_count = _selected_count(node_count, requested_rate)
    if selected_count == 0:
        return ()
    if selected_count == node_count:
        return tuple(range(node_count))
    order = _base_order(
        node_count=node_count,
        global_seed=global_seed,
        identity=identity,
        stage=stage,
        encoder_view_index=encoder_view_index,
    )
    offset_seed = derive_stable_seed(
        namespace="music_critic.ssl.encoder_mask.offset.v1",
        global_seed=global_seed,
        dataset_id=identity.dataset_id,
        piece_id=identity.piece_id,
        epoch=0,
        view_index=encoder_view_index,
        extra={
            "stage": stage,
            "mask_policy_version": MASK_POLICY_VERSION,
        },
    )
    # Advancing by exactly one position per train epoch guarantees a changed
    # subset for adjacent epochs whenever 0 < selected_count < node_count.
    offset = (offset_seed.value + epoch) % node_count
    rotated = order[offset:] + order[:offset]
    return tuple(sorted(rotated[:selected_count]))


def _validated_ptr(
    graph: HeteroData,
    node_type: str,
    *,
    sample_count: int,
) -> tuple[int, ...]:
    if isinstance(graph, Batch):
        raw = tuple(
            int(value)
            for value in graph[node_type].ptr.detach().cpu().tolist()
        )
    else:
        raw = (0, int(graph[node_type].num_nodes))
    if (
        len(raw) != sample_count + 1
        or raw[0] != 0
        or raw[-1] != int(graph[node_type].num_nodes)
        or any(left > right for left, right in zip(raw, raw[1:]))
    ):
        raise SSLContractError(f"{node_type}.ptr is incompatible with the batch")
    return raw


def _note_owner_tracks(graph: HeteroData) -> tuple[int, ...]:
    note_count = int(graph["note"].num_nodes)
    track_count = int(graph["track"].num_nodes)
    owners = [-1] * note_count
    edge_index = graph[TRACK_CONTAINS_NOTE_EDGE].edge_index.detach().cpu()
    track_indices, note_indices = edge_index.tolist()
    for track_index, note_index in zip(
        track_indices,
        note_indices,
        strict=True,
    ):
        if not 0 <= track_index < track_count or not 0 <= note_index < note_count:
            raise SSLContractError("track-note ownership edge is out of range")
        if owners[note_index] != -1:
            raise SSLContractError("a note has more than one owner track")
        owners[note_index] = track_index
    if any(owner < 0 for owner in owners):
        raise SSLContractError("every note must have exactly one owner track")
    return tuple(owners)


def _one_plan(
    *,
    dataset_id: str,
    piece_id: str,
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    requested_mask_rate: float,
    global_seed: int,
    note_start: int,
    note_end: int,
    track_start: int,
    track_end: int,
    owner_track_by_global_note: tuple[int, ...],
) -> MaskPlan:
    identity = SampleIdentity(dataset_id, piece_id)
    note_count = note_end - note_start
    selected = _sample_local_indices(
        node_count=note_count,
        requested_rate=requested_mask_rate,
        global_seed=global_seed,
        identity=identity,
        stage=stage,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
    )
    owner_tracks = []
    for local_note_index in selected:
        owner_global_index = owner_track_by_global_note[
            note_start + local_note_index
        ]
        if not track_start <= owner_global_index < track_end:
            raise SSLContractError(
                "selected note owner track escaped its source sample"
            )
        owner_tracks.append(owner_global_index - track_start)
    owner_track_set = set(owner_tracks)
    selected_set = set(selected)
    peer_note_indices = tuple(
        local_note_index
        for local_note_index in range(note_count)
        if local_note_index not in selected_set
        and (
            owner_track_by_global_note[note_start + local_note_index]
            - track_start
        )
        in owner_track_set
    )
    peer_relative_collateral = CollateralFeatureMask(
        reason=NOTE_PITCH_GROUP.peer_note_collateral_reason,
        node_type="note",
        local_node_indices=peer_note_indices,
        features=NOTE_PITCH_GROUP.peer_note_collateral_fields,
    )
    track_statistics_collateral = CollateralFeatureMask(
        reason=NOTE_PITCH_GROUP.collateral_reason,
        node_type="track",
        local_node_indices=tuple(sorted(set(owner_tracks))),
        features=NOTE_PITCH_GROUP.collateral_fields,
    )
    stable_seed = derive_stable_seed(
        namespace="music_critic.ssl.encoder_mask.plan.v1",
        global_seed=global_seed,
        dataset_id=dataset_id,
        piece_id=piece_id,
        epoch=epoch,
        view_index=encoder_view_index,
        extra={
            "stage": stage,
            "mask_policy": UNIFORM_NOTE_MASK_POLICY,
            "mask_policy_version": MASK_POLICY_VERSION,
            "primary_feature_group": NOTE_PITCH_GROUP_NAME,
            "maskable_field_registry_fingerprint": (
                MASKABLE_FIELD_REGISTRY_FINGERPRINT
            ),
        },
    )
    realized_rate = len(selected) / note_count if note_count else 0.0
    return MaskPlan.create(
        mask_policy=UNIFORM_NOTE_MASK_POLICY,
        mask_policy_version=MASK_POLICY_VERSION,
        dataset_id=dataset_id,
        piece_id=piece_id,
        stage=stage,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        selected_node_type="note",
        selected_local_node_indices=selected,
        primary_feature_group=NOTE_PITCH_GROUP_NAME,
        collateral_feature_masks=(
            peer_relative_collateral,
            track_statistics_collateral,
        ),
        requested_mask_rate=requested_mask_rate,
        maskable_node_count=note_count,
        realized_mask_rate=realized_rate,
        global_seed=global_seed,
        stable_seed=stable_seed.value,
        stable_seed_sha256=stable_seed.sha256,
    )


def build_mask_plan(
    graph: HeteroData,
    *,
    dataset_id: str,
    piece_id: str,
    global_seed: int,
    epoch: int,
    encoder_view_index: int = 0,
    requested_mask_rate: float = DEFAULT_ENCODER_MASK_RATE,
    stage: MaskStage = "train",
) -> MaskPlan:
    """Build one target-blind plan for an immutable raw graph."""

    if isinstance(graph, Batch):
        raise SSLContractError(
            "build_mask_plan requires one graph; use build_batched_mask_plans"
        )
    validate_raw_graph(graph)
    validate_global_seed(global_seed)
    canonical_epoch = _canonical_epoch(stage, epoch)
    validate_non_negative_integer(
        encoder_view_index,
        name="encoder_view_index",
    )
    rate = validate_mask_rate(requested_mask_rate)
    # Resolve every semantic field before sampling. This fails closed if the
    # raw registry ever changes without a Phase 7A registry/version update.
    SSL_MASKABLE_FIELD_REGISTRY.resolve_group(NOTE_PITCH_GROUP_NAME)
    owners = _note_owner_tracks(graph)
    return _one_plan(
        dataset_id=dataset_id,
        piece_id=piece_id,
        stage=stage,
        epoch=canonical_epoch,
        encoder_view_index=encoder_view_index,
        requested_mask_rate=rate,
        global_seed=global_seed,
        note_start=0,
        note_end=int(graph["note"].num_nodes),
        track_start=0,
        track_end=int(graph["track"].num_nodes),
        owner_track_by_global_note=owners,
    )


def build_batched_mask_plans(
    graph_batch: Batch,
    *,
    dataset_ids: Sequence[str],
    piece_ids: Sequence[str],
    global_seed: int,
    epoch: int,
    encoder_view_index: int = 0,
    requested_mask_rate: float = DEFAULT_ENCODER_MASK_RATE,
    stage: MaskStage = "train",
) -> tuple[MaskPlan, ...]:
    """Build independent local-index plans from PyG ``ptr`` and ownership."""

    if not isinstance(graph_batch, Batch):
        raise SSLContractError("build_batched_mask_plans requires a PyG Batch")
    if isinstance(dataset_ids, (str, bytes)) or isinstance(piece_ids, (str, bytes)):
        raise SSLContractError("batch identities must be sequences of strings")
    datasets = tuple(dataset_ids)
    pieces = tuple(piece_ids)
    sample_count = int(graph_batch.num_graphs)
    if (
        not datasets
        or len(datasets) != sample_count
        or len(pieces) != sample_count
    ):
        raise SSLContractError(
            "dataset_ids and piece_ids must match the non-empty graph batch"
        )
    identities = tuple(
        SampleIdentity(dataset_id, piece_id)
        for dataset_id, piece_id in zip(datasets, pieces)
    )
    validate_raw_graph_batch(graph_batch, sample_count=sample_count)
    validate_global_seed(global_seed)
    canonical_epoch = _canonical_epoch(stage, epoch)
    validate_non_negative_integer(
        encoder_view_index,
        name="encoder_view_index",
    )
    rate = validate_mask_rate(requested_mask_rate)
    SSL_MASKABLE_FIELD_REGISTRY.resolve_group(NOTE_PITCH_GROUP_NAME)
    note_ptr = _validated_ptr(
        graph_batch,
        "note",
        sample_count=sample_count,
    )
    track_ptr = _validated_ptr(
        graph_batch,
        "track",
        sample_count=sample_count,
    )
    owners = _note_owner_tracks(graph_batch)
    return tuple(
        _one_plan(
            dataset_id=identity.dataset_id,
            piece_id=identity.piece_id,
            stage=stage,
            epoch=canonical_epoch,
            encoder_view_index=encoder_view_index,
            requested_mask_rate=rate,
            global_seed=global_seed,
            note_start=note_ptr[sample_index],
            note_end=note_ptr[sample_index + 1],
            track_start=track_ptr[sample_index],
            track_end=track_ptr[sample_index + 1],
            owner_track_by_global_note=owners,
        )
        for sample_index, identity in enumerate(identities)
    )


def build_batch_mask_plans(*args: object, **kwargs: object) -> tuple[MaskPlan, ...]:
    """Compatibility alias for :func:`build_batched_mask_plans`."""

    return build_batched_mask_plans(*args, **kwargs)


def build_mask_plans_for_batch(
    batch: object,
    *,
    global_seed: int,
    epoch: int,
    encoder_view_index: int = 0,
    requested_mask_rate: float = DEFAULT_ENCODER_MASK_RATE,
    stage: MaskStage = "train",
) -> tuple[MaskPlan, ...]:
    """Build plans directly from a target-bearing batch without reading targets."""

    graph_batch = getattr(batch, "raw_graph_batch", None)
    dataset_ids = getattr(batch, "dataset_ids", None)
    piece_ids = getattr(batch, "piece_ids", None)
    if (
        not isinstance(graph_batch, Batch)
        or dataset_ids is None
        or piece_ids is None
    ):
        raise SSLContractError(
            "batch must expose raw_graph_batch, dataset_ids, and piece_ids"
        )
    return build_batched_mask_plans(
        graph_batch,
        dataset_ids=dataset_ids,
        piece_ids=piece_ids,
        global_seed=global_seed,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        requested_mask_rate=requested_mask_rate,
        stage=stage,
    )


__all__ = [
    "DEFAULT_ENCODER_MASK_RATE",
    "TRACK_CONTAINS_NOTE_EDGE",
    "build_batch_mask_plans",
    "build_batched_mask_plans",
    "build_mask_plan",
    "build_mask_plans_for_batch",
    "derive_stable_seed",
]
