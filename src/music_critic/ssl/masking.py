"""Deterministic, per-sample Phase 7A note-pitch mask planning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hmac
import math
import secrets
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, HeteroData

from music_critic.device import resolve_runtime_device
from music_critic.graph import (
    BATCH_BASE_NODE_ATTRIBUTES,
    BATCH_CANDIDATE_NODE_ATTRIBUTES,
    BATCH_EDGE_ATTRIBUTES,
    BATCH_GLOBAL_ATTRIBUTES,
    MANDATORY_EDGE_TYPES,
    MANDATORY_NODE_TYPES,
    validate_raw_graph,
    validate_raw_graph_batch,
)
from music_critic.ssl.contracts import (
    MASK_POLICY_VERSION,
    PREPARED_MASK_BINDING_CONTRACT_VERSION,
    UNIFORM_NOTE_MASK_POLICY,
    CollateralFeatureMask,
    MaskPlan,
    MaskStage,
    SSLContractError,
    SampleIdentity,
    StableSeed,
    canonical_sha256,
    is_sha256,
    mask_plan_fingerprint,
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
from music_critic.ssl.views import (
    FeatureMaskOverlay,
    build_feature_mask_overlay,
)


DEFAULT_ENCODER_MASK_RATE = 0.30
PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION = "1.2.0"
TRACK_CONTAINS_NOTE_EDGE = ("track", "contains_note", "note")

_PREPARED_BINDING_ATTESTATION_KEY = secrets.token_bytes(32)


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
    """Create the exact SHA-key order with fixed-width stable radix passes."""

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
    keyed = [
        (
            bytes.fromhex(
                canonical_sha256(
                    {
                        **common,
                        "local_node_index": local_index,
                    }
                )
            ),
            local_index,
        )
        for local_index in range(node_count)
    ]
    # SHA-256 keys are fixed-width.  Stable least-significant-byte passes
    # reproduce lexicographic digest ordering in O(32 * N).  ``keyed`` starts
    # in local-index order, so a hypothetical digest collision retains the
    # former ``(digest, local_index)`` tie-break exactly.
    for byte_position in range(31, -1, -1):
        buckets: list[list[tuple[bytes, int]]] = [
            [] for _ in range(256)
        ]
        for item in keyed:
            buckets[item[0][byte_position]].append(item)
        keyed = [
            item
            for bucket in buckets
            for item in bucket
        ]
    return tuple(local_index for _, local_index in keyed)


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
    selected = set(rotated[:selected_count])
    return tuple(
        local_index
        for local_index in range(node_count)
        if local_index in selected
    )


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


def _binding_payload(
    *,
    dataset_ids: tuple[str, ...],
    piece_ids: tuple[str, ...],
    stage: MaskStage,
    epoch: int,
    encoder_view_index: int,
    global_seed: int,
    requested_mask_rate: float,
    sample_count: int,
    node_counts: tuple[tuple[str, int], ...],
    node_ptrs: tuple[tuple[str, tuple[int, ...]], ...],
    edge_counts: tuple[tuple[tuple[str, str, str], int], ...],
    validated_structure_sha256: str,
    note_track_ownership_sha256: str,
    ordered_plan_fingerprints: tuple[str, ...],
    feature_overlay_fingerprint: str,
    selected_global_note_indices: tuple[int, ...],
    hierarchy_profile_version: str | None = None,
    hierarchy_policy_config_fingerprint: str | None = None,
    hierarchy_resolution_fingerprints: tuple[str, ...] = (),
) -> dict[str, object]:
    contract_version = (
        PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
        if hierarchy_profile_version is not None
        else PREPARED_MASK_BINDING_CONTRACT_VERSION
    )
    payload = {
        "contract_version": contract_version,
        "sample_identities": [
            {"dataset_id": dataset_id, "piece_id": piece_id}
            for dataset_id, piece_id in zip(
                dataset_ids,
                piece_ids,
                strict=True,
            )
        ],
        "stage": stage,
        "epoch": epoch,
        "encoder_view_index": encoder_view_index,
        "global_seed": global_seed,
        "requested_mask_rate": requested_mask_rate,
        "sample_count": sample_count,
        "node_counts": [
            {"node_type": node_type, "count": count}
            for node_type, count in node_counts
        ],
        "node_ptrs": [
            {"node_type": node_type, "ptr": list(ptr)}
            for node_type, ptr in node_ptrs
        ],
        "edge_counts": [
            {
                "edge_type": list(edge_type),
                "count": count,
            }
            for edge_type, count in edge_counts
        ],
        "validated_structure_sha256": validated_structure_sha256,
        "note_track_ownership_sha256": note_track_ownership_sha256,
        "ordered_plan_fingerprints": list(ordered_plan_fingerprints),
        "feature_overlay_fingerprint": feature_overlay_fingerprint,
        "selected_global_note_indices": list(
            selected_global_note_indices
        ),
    }
    if hierarchy_profile_version is not None:
        payload["shared_attestation_contract_version"] = (
            PREPARED_MASK_BINDING_CONTRACT_VERSION
        )
        payload["hierarchy_profile"] = {
            "profile_version": hierarchy_profile_version,
            "policy_config_fingerprint": (
                hierarchy_policy_config_fingerprint
            ),
            "ordered_resolution_fingerprints": list(
                hierarchy_resolution_fingerprints
            ),
        }
    return payload


def _semantic_attestation(fingerprint: str) -> str:
    return hmac.new(
        _PREPARED_BINDING_ATTESTATION_KEY,
        fingerprint.encode("ascii"),
        digestmod="sha256",
    ).hexdigest()


def _runtime_error(detail: str) -> SSLContractError:
    return SSLContractError(
        f"ssl.prepared_binding.runtime_input_changed:{detail}"
    )


def _typed_metadata(value: object, *, location: str) -> object:
    """Return a type-preserving JSON value without inspecting tensor data."""

    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _runtime_error(
                f"metadata_non_finite:{location}"
            )
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "value": [
                _typed_metadata(
                    item,
                    location=f"{location}[{index}]",
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "value": [
                _typed_metadata(
                    item,
                    location=f"{location}[{index}]",
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, Mapping):
        items: list[list[object]] = []
        keys = tuple(value)
        if not all(isinstance(key, str) for key in keys):
            raise _runtime_error(
                f"metadata_key_type:{location}"
            )
        for key in sorted(keys):
            items.append(
                [
                    key,
                    _typed_metadata(
                        value[key],
                        location=f"{location}.{key}",
                    ),
                ]
            )
        return {"type": "mapping", "value": items}
    raise _runtime_error(
        f"metadata_type:{location}:{type(value).__name__}"
    )


def _attribute_names(
    store: object,
    *,
    location: str,
) -> tuple[str, ...]:
    keys = tuple(store.keys())
    if not all(isinstance(key, str) for key in keys):
        raise _runtime_error(f"attribute_name_type:{location}")
    return tuple(sorted(keys))


@dataclass(frozen=True, slots=True)
class _StoreRuntimeEvidence:
    location: str
    store: object = field(repr=False, compare=False)
    object_id: int
    store_type: str
    attributes: tuple[str, ...]

    def private_payload(self) -> dict[str, object]:
        return {
            "location": self.location,
            "object_id": self.object_id,
            "store_type": self.store_type,
            "attributes": list(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class _TensorRuntimeEvidence:
    location: str
    tensor: Tensor = field(repr=False, compare=False)
    object_id: int
    version: int
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None

    @classmethod
    def capture(
        cls,
        tensor: Tensor,
        *,
        location: str,
    ) -> _TensorRuntimeEvidence:
        return cls(
            location=location,
            tensor=tensor,
            object_id=id(tensor),
            version=int(tensor._version),
            shape=tuple(int(size) for size in tensor.shape),
            dtype=str(tensor.dtype),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
        )

    def private_payload(self) -> dict[str, object]:
        return {
            "location": self.location,
            "object_id": self.object_id,
            "version": self.version,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": {
                "type": self.device_type,
                "index": self.device_index,
            },
        }


@dataclass(frozen=True, slots=True)
class _RuntimeGraphEvidence:
    graph: HeteroData = field(repr=False, compare=False)
    graph_object_id: int
    graph_type: str
    sample_count: int
    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]
    stores: tuple[_StoreRuntimeEvidence, ...]
    tensors: tuple[_TensorRuntimeEvidence, ...]
    non_tensor_metadata_sha256: str

    def private_payload(self) -> dict[str, object]:
        return {
            "graph_object_id": self.graph_object_id,
            "graph_type": self.graph_type,
            "sample_count": self.sample_count,
            "node_types": list(self.node_types),
            "edge_types": [list(value) for value in self.edge_types],
            "stores": [
                value.private_payload() for value in self.stores
            ],
            "tensors": [
                value.private_payload() for value in self.tensors
            ],
            "non_tensor_metadata_sha256": (
                self.non_tensor_metadata_sha256
            ),
        }


def _capture_runtime_graph_evidence(
    graph: HeteroData,
) -> _RuntimeGraphEvidence:
    """Capture live metadata only; never materialize tensor values."""

    if not isinstance(graph, Batch):
        raise _runtime_error("graph_type")
    node_items = tuple(graph.node_items())
    edge_items = tuple(graph.edge_items())
    node_types = tuple(key for key, _ in node_items)
    edge_types = tuple(key for key, _ in edge_items)
    if not all(isinstance(value, str) for value in node_types):
        raise _runtime_error("node_type_value")
    if not all(
        isinstance(value, tuple)
        and len(value) == 3
        and all(isinstance(part, str) for part in value)
        for value in edge_types
    ):
        raise _runtime_error("edge_type_value")
    if node_types != MANDATORY_NODE_TYPES:
        raise _runtime_error("node_types")
    if edge_types != MANDATORY_EDGE_TYPES:
        raise _runtime_error("edge_types")

    global_attributes = _attribute_names(
        graph._global_store,
        location="global",
    )
    if set(global_attributes) != BATCH_GLOBAL_ATTRIBUTES:
        raise _runtime_error("attribute_set:global")
    for node_type, store in node_items:
        attributes = _attribute_names(
            store,
            location=f"node:{node_type}",
        )
        expected_attributes = (
            BATCH_CANDIDATE_NODE_ATTRIBUTES
            if node_type in {"beat", "onset"}
            else BATCH_BASE_NODE_ATTRIBUTES
        )
        if set(attributes) != expected_attributes:
            raise _runtime_error(
                f"attribute_set:node:{node_type}"
            )
    for edge_type, store in edge_items:
        attributes = _attribute_names(
            store,
            location="edge:" + "|".join(edge_type),
        )
        if set(attributes) != BATCH_EDGE_ATTRIBUTES:
            raise _runtime_error(
                "attribute_set:edge:" + "|".join(edge_type)
            )

    stores_with_locations: list[tuple[str, object]] = [
        ("global", graph._global_store)
    ]
    stores_with_locations.extend(
        (f"node:{node_type}", store)
        for node_type, store in node_items
    )
    stores_with_locations.extend(
        ("edge:" + "|".join(edge_type), store)
        for edge_type, store in edge_items
    )
    stores: list[_StoreRuntimeEvidence] = []
    tensors: list[_TensorRuntimeEvidence] = []
    metadata: list[list[object]] = []
    for location, store in stores_with_locations:
        attributes = _attribute_names(store, location=location)
        stores.append(
            _StoreRuntimeEvidence(
                location=location,
                store=store,
                object_id=id(store),
                store_type=(
                    f"{type(store).__module__}."
                    f"{type(store).__qualname__}"
                ),
                attributes=attributes,
            )
        )
        for name in attributes:
            value = store[name]
            value_location = f"{location}:{name}"
            if isinstance(value, Tensor):
                tensors.append(
                    _TensorRuntimeEvidence.capture(
                        value,
                        location=value_location,
                    )
                )
            else:
                metadata.append(
                    [
                        value_location,
                        _typed_metadata(
                            value,
                            location=value_location,
                        ),
                    ]
                )
    return _RuntimeGraphEvidence(
        graph=graph,
        graph_object_id=id(graph),
        graph_type=(
            f"{type(graph).__module__}.{type(graph).__qualname__}"
        ),
        sample_count=int(graph.num_graphs),
        node_types=node_types,
        edge_types=edge_types,
        stores=tuple(stores),
        tensors=tuple(tensors),
        non_tensor_metadata_sha256=canonical_sha256(metadata),
    )


def _validate_runtime_graph_evidence(
    graph: HeteroData,
    expected: _RuntimeGraphEvidence,
) -> None:
    if graph is not expected.graph or id(graph) != expected.graph_object_id:
        raise _runtime_error("graph_object_identity")
    current = _capture_runtime_graph_evidence(graph)
    for name in (
        "graph_type",
        "sample_count",
        "node_types",
        "edge_types",
    ):
        if getattr(current, name) != getattr(expected, name):
            raise _runtime_error(name)
    if len(current.stores) != len(expected.stores):
        raise _runtime_error("store_count")
    for actual_store, expected_store in zip(
        current.stores,
        expected.stores,
        strict=True,
    ):
        if actual_store.location != expected_store.location:
            raise _runtime_error("store_locations")
        if actual_store.attributes != expected_store.attributes:
            raise _runtime_error(
                f"attribute_set:{expected_store.location}"
            )
        if actual_store.store_type != expected_store.store_type:
            raise _runtime_error(
                f"store_type:{expected_store.location}"
            )
        if (
            actual_store.store is not expected_store.store
            or actual_store.object_id != expected_store.object_id
        ):
            raise _runtime_error(
                f"store_object_identity:{expected_store.location}"
            )
    if (
        current.non_tensor_metadata_sha256
        != expected.non_tensor_metadata_sha256
    ):
        raise _runtime_error("non_tensor_metadata")
    if len(current.tensors) != len(expected.tensors):
        raise _runtime_error("tensor_count")
    for actual, captured in zip(
        current.tensors,
        expected.tensors,
        strict=True,
    ):
        if actual.location != captured.location:
            raise _runtime_error("tensor_locations")
        if actual.tensor is not captured.tensor:
            raise _runtime_error(
                f"tensor_object_identity:{captured.location}"
            )
        for name in (
            "object_id",
            "version",
            "shape",
            "dtype",
            "device_type",
            "device_index",
        ):
            if getattr(actual, name) != getattr(captured, name):
                raise _runtime_error(
                    f"tensor_{name}:{captured.location}"
                )


def _validate_transferred_runtime_evidence(
    source: _RuntimeGraphEvidence,
    moved: _RuntimeGraphEvidence,
) -> None:
    """Require transfer to preserve the validated semantic input surface."""

    for name in (
        "graph_type",
        "sample_count",
        "node_types",
        "edge_types",
        "non_tensor_metadata_sha256",
    ):
        if getattr(moved, name) != getattr(source, name):
            raise _runtime_error(f"transfer_{name}")
    source_store_surface = tuple(
        (store.location, store.store_type, store.attributes)
        for store in source.stores
    )
    moved_store_surface = tuple(
        (store.location, store.store_type, store.attributes)
        for store in moved.stores
    )
    if moved_store_surface != source_store_surface:
        raise _runtime_error("transfer_store_surface")
    source_tensor_surface = tuple(
        (tensor.location, tensor.shape, tensor.dtype)
        for tensor in source.tensors
    )
    moved_tensor_surface = tuple(
        (tensor.location, tensor.shape, tensor.dtype)
        for tensor in moved.tensors
    )
    if moved_tensor_surface != source_tensor_surface:
        raise _runtime_error("transfer_tensor_surface")


def _runtime_attestation(
    *,
    fingerprint: str,
    graph_evidence: _RuntimeGraphEvidence,
    selected_indices_evidence: _TensorRuntimeEvidence,
) -> str:
    payload = {
        "binding_fingerprint": fingerprint,
        "graph_evidence": graph_evidence.private_payload(),
        "selected_indices_evidence": (
            selected_indices_evidence.private_payload()
        ),
    }
    message = canonical_sha256(payload).encode("ascii")
    return hmac.new(
        _PREPARED_BINDING_ATTESTATION_KEY,
        message,
        digestmod="sha256",
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedMaskBinding:
    """CPU-prepared plans bound to one validated raw batch and transfer."""

    contract_version: str
    dataset_ids: tuple[str, ...]
    piece_ids: tuple[str, ...]
    stage: MaskStage
    epoch: int
    encoder_view_index: int
    global_seed: int
    requested_mask_rate: float
    sample_count: int
    node_counts: tuple[tuple[str, int], ...]
    node_ptrs: tuple[tuple[str, tuple[int, ...]], ...]
    edge_counts: tuple[tuple[tuple[str, str, str], int], ...]
    validated_structure_sha256: str
    note_track_ownership_sha256: str
    mask_plans: tuple[object, ...]
    ordered_plan_fingerprints: tuple[str, ...]
    feature_overlay: FeatureMaskOverlay
    selected_global_note_indices: tuple[int, ...]
    hierarchy_profile_version: str | None
    hierarchy_policy_config: object | None = field(
        repr=False,
        compare=True,
    )
    hierarchy_resolutions: tuple[object, ...] = field(
        repr=False,
        compare=True,
    )
    hierarchy_resolution_fingerprints: tuple[str, ...]
    selected_global_note_indices_tensor: Tensor = field(
        repr=False,
        compare=False,
    )
    fingerprint: str
    _bound_graph: HeteroData = field(repr=False, compare=False)
    _semantic_attestation: str = field(repr=False, compare=False)
    _runtime_graph_evidence: _RuntimeGraphEvidence = field(
        repr=False,
        compare=False,
    )
    _selected_indices_evidence: _TensorRuntimeEvidence = field(
        repr=False,
        compare=False,
    )
    _runtime_attestation: str = field(repr=False, compare=False)

    @classmethod
    def _create(
        cls,
        *,
        dataset_ids: tuple[str, ...],
        piece_ids: tuple[str, ...],
        stage: MaskStage,
        epoch: int,
        encoder_view_index: int,
        global_seed: int,
        requested_mask_rate: float,
        sample_count: int,
        node_counts: tuple[tuple[str, int], ...],
        node_ptrs: tuple[tuple[str, tuple[int, ...]], ...],
        edge_counts: tuple[tuple[tuple[str, str, str], int], ...],
        validated_structure_sha256: str,
        note_track_ownership_sha256: str,
        mask_plans: tuple[object, ...],
        feature_overlay: FeatureMaskOverlay,
        selected_global_note_indices: tuple[int, ...],
        hierarchy_profile_version: str | None = None,
        hierarchy_policy_config: object | None = None,
        hierarchy_resolutions: tuple[object, ...] = (),
        bound_graph: HeteroData,
    ) -> PreparedMaskBinding:
        if not isinstance(bound_graph, Batch):
            raise SSLContractError(
                "prepared binding construction requires a PyG Batch"
            )
        if (
            hierarchy_profile_version is None
            and cls is not PreparedMaskBinding
        ) or (
            hierarchy_profile_version is not None
            and cls is not PreparedHierarchyMaskBinding
        ):
            raise SSLContractError(
                "prepared binding envelope type is incompatible"
            )
        (
            canonical_node_counts,
            canonical_node_ptrs,
            canonical_edge_counts,
            canonical_structure_sha256,
            canonical_ownership_sha256,
        ) = _cpu_graph_evidence(
            bound_graph,
            sample_count=sample_count,
        )
        if (
            node_counts != canonical_node_counts
            or node_ptrs != canonical_node_ptrs
            or edge_counts != canonical_edge_counts
            or validated_structure_sha256
            != canonical_structure_sha256
            or note_track_ownership_sha256
            != canonical_ownership_sha256
        ):
            raise SSLContractError(
                "prepared binding construction evidence is non-canonical"
            )
        hierarchy_resolution_fingerprints: tuple[str, ...] = ()
        hierarchy_policy_config_fingerprint: str | None = None
        if hierarchy_profile_version is None:
            if (
                hierarchy_policy_config is not None
                or hierarchy_resolutions
            ):
                raise SSLContractError(
                    "Phase 7A binding cannot contain hierarchy evidence"
                )
            canonical_plans: tuple[object, ...] = (
                build_batched_mask_plans(
                    bound_graph,
                    dataset_ids=dataset_ids,
                    piece_ids=piece_ids,
                    global_seed=global_seed,
                    epoch=epoch,
                    encoder_view_index=encoder_view_index,
                    requested_mask_rate=requested_mask_rate,
                    stage=stage,
                )
            )
        else:
            from music_critic.ssl.hierarchical_masking import (
                HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
                HierarchyMaskPolicyConfig,
                HierarchyMaskResolution,
                build_batched_hierarchy_mask_resolutions,
            )

            if (
                hierarchy_profile_version
                != HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
                or type(hierarchy_policy_config)
                is not HierarchyMaskPolicyConfig
                or not isinstance(hierarchy_resolutions, tuple)
                or not all(
                    type(resolution) is HierarchyMaskResolution
                    for resolution in hierarchy_resolutions
                )
            ):
                raise SSLContractError(
                    "prepared hierarchy binding evidence is incompatible"
                )
            canonical_resolutions = (
                build_batched_hierarchy_mask_resolutions(
                    bound_graph,
                    dataset_ids=dataset_ids,
                    piece_ids=piece_ids,
                    global_seed=global_seed,
                    epoch=epoch,
                    encoder_view_index=encoder_view_index,
                    requested_mask_rate=requested_mask_rate,
                    stage=stage,
                    policy_config=hierarchy_policy_config,
                )
            )
            if (
                hierarchy_resolutions != canonical_resolutions
                or any(
                    resolution.plan is None
                    for resolution in canonical_resolutions
                )
            ):
                raise SSLContractError(
                    "prepared hierarchy binding resolutions are non-canonical"
                )
            canonical_plans = tuple(
                resolution.plan
                for resolution in canonical_resolutions
                if resolution.plan is not None
            )
            hierarchy_resolution_fingerprints = tuple(
                resolution.fingerprint
                for resolution in canonical_resolutions
            )
            hierarchy_policy_config_fingerprint = (
                hierarchy_policy_config.fingerprint
            )
        if mask_plans != canonical_plans:
            raise SSLContractError(
                "prepared binding construction plans are non-canonical"
            )
        canonical_overlay = build_feature_mask_overlay(
            bound_graph,
            canonical_plans,
        )
        note_ptr = dict(canonical_node_ptrs)["note"]
        canonical_selected_global_indices = tuple(
            note_ptr[sample_index] + local_index
            for sample_index, plan in enumerate(canonical_plans)
            for local_index in plan.selected_local_node_indices
        )
        if (
            feature_overlay != canonical_overlay
            or selected_global_note_indices
            != canonical_selected_global_indices
        ):
            raise SSLContractError(
                "prepared binding construction overlay is non-canonical"
            )
        ordered_plan_fingerprints = tuple(
            plan.fingerprint for plan in mask_plans
        )
        payload = _binding_payload(
            dataset_ids=dataset_ids,
            piece_ids=piece_ids,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            requested_mask_rate=requested_mask_rate,
            sample_count=sample_count,
            node_counts=node_counts,
            node_ptrs=node_ptrs,
            edge_counts=edge_counts,
            validated_structure_sha256=validated_structure_sha256,
            note_track_ownership_sha256=(
                note_track_ownership_sha256
            ),
            ordered_plan_fingerprints=ordered_plan_fingerprints,
            feature_overlay_fingerprint=feature_overlay.fingerprint,
            selected_global_note_indices=(
                selected_global_note_indices
            ),
            hierarchy_profile_version=hierarchy_profile_version,
            hierarchy_policy_config_fingerprint=(
                hierarchy_policy_config_fingerprint
            ),
            hierarchy_resolution_fingerprints=(
                hierarchy_resolution_fingerprints
            ),
        )
        fingerprint = canonical_sha256(payload)
        selected_tensor = torch.tensor(
            selected_global_note_indices,
            dtype=torch.long,
            device="cpu",
        )
        runtime_graph_evidence = _capture_runtime_graph_evidence(
            bound_graph
        )
        selected_indices_evidence = _TensorRuntimeEvidence.capture(
            selected_tensor,
            location="binding:selected_global_note_indices_tensor",
        )
        return cls(
            contract_version=(
                PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
                if hierarchy_profile_version is not None
                else PREPARED_MASK_BINDING_CONTRACT_VERSION
            ),
            dataset_ids=dataset_ids,
            piece_ids=piece_ids,
            stage=stage,
            epoch=epoch,
            encoder_view_index=encoder_view_index,
            global_seed=global_seed,
            requested_mask_rate=requested_mask_rate,
            sample_count=sample_count,
            node_counts=node_counts,
            node_ptrs=node_ptrs,
            edge_counts=edge_counts,
            validated_structure_sha256=validated_structure_sha256,
            note_track_ownership_sha256=(
                note_track_ownership_sha256
            ),
            mask_plans=mask_plans,
            ordered_plan_fingerprints=ordered_plan_fingerprints,
            feature_overlay=feature_overlay,
            selected_global_note_indices=(
                selected_global_note_indices
            ),
            hierarchy_profile_version=hierarchy_profile_version,
            hierarchy_policy_config=hierarchy_policy_config,
            hierarchy_resolutions=hierarchy_resolutions,
            hierarchy_resolution_fingerprints=(
                hierarchy_resolution_fingerprints
            ),
            selected_global_note_indices_tensor=selected_tensor,
            fingerprint=fingerprint,
            _bound_graph=bound_graph,
            _semantic_attestation=_semantic_attestation(fingerprint),
            _runtime_graph_evidence=runtime_graph_evidence,
            _selected_indices_evidence=selected_indices_evidence,
            _runtime_attestation=_runtime_attestation(
                fingerprint=fingerprint,
                graph_evidence=runtime_graph_evidence,
                selected_indices_evidence=selected_indices_evidence,
            ),
        )

    def __post_init__(self) -> None:
        _validate_prepared_mask_binding_contract(self)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic public evidence, excluding runtime binding."""

        payload = _binding_payload(
            dataset_ids=self.dataset_ids,
            piece_ids=self.piece_ids,
            stage=self.stage,
            epoch=self.epoch,
            encoder_view_index=self.encoder_view_index,
            global_seed=self.global_seed,
            requested_mask_rate=self.requested_mask_rate,
            sample_count=self.sample_count,
            node_counts=self.node_counts,
            node_ptrs=self.node_ptrs,
            edge_counts=self.edge_counts,
            validated_structure_sha256=(
                self.validated_structure_sha256
            ),
            note_track_ownership_sha256=(
                self.note_track_ownership_sha256
            ),
            ordered_plan_fingerprints=(
                self.ordered_plan_fingerprints
            ),
            feature_overlay_fingerprint=(
                self.feature_overlay.fingerprint
            ),
            selected_global_note_indices=(
                self.selected_global_note_indices
            ),
            hierarchy_profile_version=(
                self.hierarchy_profile_version
            ),
            hierarchy_policy_config_fingerprint=(
                None
                if self.hierarchy_policy_config is None
                else getattr(
                    self.hierarchy_policy_config,
                    "fingerprint",
                    None,
                )
            ),
            hierarchy_resolution_fingerprints=(
                self.hierarchy_resolution_fingerprints
            ),
        )
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class PreparedHierarchyMaskBinding(PreparedMaskBinding):
    """Phase 8A envelope reusing the Phase 7A attestation kernel."""


def _validate_prepared_mask_binding_contract(
    binding: PreparedMaskBinding,
) -> None:
    hierarchy_mode = binding.hierarchy_profile_version is not None
    expected_type = (
        PreparedHierarchyMaskBinding
        if hierarchy_mode
        else PreparedMaskBinding
    )
    expected_contract_version = (
        PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION
        if hierarchy_mode
        else PREPARED_MASK_BINDING_CONTRACT_VERSION
    )
    if (
        type(binding) is not expected_type
        or binding.contract_version != expected_contract_version
    ):
        raise SSLContractError(
            "prepared mask binding contract version is incompatible"
        )
    if (
        not isinstance(binding.dataset_ids, tuple)
        or not isinstance(binding.piece_ids, tuple)
        or not binding.dataset_ids
        or len(binding.dataset_ids) != len(binding.piece_ids)
        or len(binding.dataset_ids) != binding.sample_count
    ):
        raise SSLContractError(
            "prepared mask binding identities are incompatible"
        )
    for dataset_id, piece_id in zip(
        binding.dataset_ids,
        binding.piece_ids,
        strict=True,
    ):
        SampleIdentity(dataset_id, piece_id)
    if binding.stage not in {"train", "validation"}:
        raise SSLContractError("prepared mask binding stage is invalid")
    validate_non_negative_integer(binding.epoch, name="epoch")
    if binding.stage == "validation" and binding.epoch != 0:
        raise SSLContractError(
            "prepared validation bindings require canonical epoch zero"
        )
    if binding.encoder_view_index != 0:
        raise SSLContractError(
            "prepared Phase 7A bindings require encoder view zero"
        )
    validate_global_seed(binding.global_seed)
    rate = validate_mask_rate(binding.requested_mask_rate)
    if (
        not isinstance(binding.requested_mask_rate, float)
        or rate != binding.requested_mask_rate
    ):
        raise SSLContractError(
            "prepared binding mask rate must use canonical float form"
        )
    validate_non_negative_integer(
        binding.sample_count,
        name="sample_count",
    )
    if binding.sample_count == 0:
        raise SSLContractError("prepared binding batch must be non-empty")

    if tuple(name for name, _ in binding.node_counts) != (
        MANDATORY_NODE_TYPES
    ):
        raise SSLContractError(
            "prepared binding node counts have incompatible ordering"
        )
    node_count_by_type = dict(binding.node_counts)
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for count in node_count_by_type.values()
    ):
        raise SSLContractError(
            "prepared binding node counts must be non-negative integers"
        )
    if tuple(name for name, _ in binding.node_ptrs) != (
        MANDATORY_NODE_TYPES
    ):
        raise SSLContractError(
            "prepared binding node ptrs have incompatible ordering"
        )
    for node_type, ptr in binding.node_ptrs:
        if (
            not isinstance(ptr, tuple)
            or len(ptr) != binding.sample_count + 1
            or ptr[0] != 0
            or ptr[-1] != node_count_by_type[node_type]
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in ptr
            )
            or any(left > right for left, right in zip(ptr, ptr[1:]))
        ):
            raise SSLContractError(
                f"prepared binding {node_type}.ptr is invalid"
            )
    if tuple(edge_type for edge_type, _ in binding.edge_counts) != (
        MANDATORY_EDGE_TYPES
    ):
        raise SSLContractError(
            "prepared binding edge counts have incompatible ordering"
        )
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for _, count in binding.edge_counts
    ):
        raise SSLContractError(
            "prepared binding edge counts must be non-negative integers"
        )
    if (
        not is_sha256(binding.validated_structure_sha256)
        or not is_sha256(binding.note_track_ownership_sha256)
    ):
        raise SSLContractError(
            "prepared binding structure evidence must use SHA-256"
        )
    if hierarchy_mode:
        from music_critic.ssl.hierarchical_masking import (
            HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
            HierarchicalMaskPlan,
            HierarchyMaskPolicyConfig,
            HierarchyMaskResolution,
            hierarchy_mask_resolution_fingerprint,
            hierarchy_policy_config_fingerprint,
            hierarchical_mask_plan_fingerprint,
        )

        if (
            binding.hierarchy_profile_version
            != HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
            or type(binding.hierarchy_policy_config)
            is not HierarchyMaskPolicyConfig
            or not isinstance(binding.hierarchy_resolutions, tuple)
            or len(binding.hierarchy_resolutions)
            != binding.sample_count
            or not all(
                type(resolution) is HierarchyMaskResolution
                for resolution in binding.hierarchy_resolutions
            )
            or binding.hierarchy_resolution_fingerprints
            != tuple(
                hierarchy_mask_resolution_fingerprint(resolution)
                for resolution in binding.hierarchy_resolutions
            )
            or binding.hierarchy_policy_config.fingerprint
            != hierarchy_policy_config_fingerprint(
                binding.hierarchy_policy_config
            )
            or any(
                resolution.plan is None
                for resolution in binding.hierarchy_resolutions
            )
        ):
            raise SSLContractError(
                "prepared hierarchy binding profile is invalid"
            )
        def prepared_plan_fingerprint(plan: object) -> str:
            if type(plan) is MaskPlan:
                return mask_plan_fingerprint(plan)
            if type(plan) is HierarchicalMaskPlan:
                if not plan.available:
                    raise SSLContractError(
                        "prepared hierarchy binding contains "
                        "an unavailable plan"
                    )
                return hierarchical_mask_plan_fingerprint(plan)
            raise SSLContractError(
                "prepared hierarchy binding contains an invalid plan type"
            )

        if tuple(
            resolution.plan
            for resolution in binding.hierarchy_resolutions
        ) != binding.mask_plans:
            raise SSLContractError(
                "prepared hierarchy plans differ from their resolutions"
            )
    else:
        if (
            binding.hierarchy_policy_config is not None
            or binding.hierarchy_resolutions
            or binding.hierarchy_resolution_fingerprints
        ):
            raise SSLContractError(
                "prepared Phase 7A binding contains hierarchy evidence"
            )

        def prepared_plan_fingerprint(plan: object) -> str:
            if type(plan) is not MaskPlan:
                raise SSLContractError(
                    "prepared Phase 7A binding requires MaskPlan values"
                )
            return mask_plan_fingerprint(plan)

    if (
        not isinstance(binding.mask_plans, tuple)
        or len(binding.mask_plans) != binding.sample_count
    ):
        raise SSLContractError(
            "prepared binding requires one plan per sample"
        )
    if (
        not isinstance(binding.ordered_plan_fingerprints, tuple)
        or binding.ordered_plan_fingerprints
        != tuple(plan.fingerprint for plan in binding.mask_plans)
        or not all(
            is_sha256(value)
            for value in binding.ordered_plan_fingerprints
        )
    ):
        raise SSLContractError(
            "prepared binding ordered plan fingerprints are invalid"
        )
    note_ptr = dict(binding.node_ptrs)["note"]
    expected_selected: list[int] = []
    for sample_index, plan in enumerate(binding.mask_plans):
        if (
            plan.fingerprint != prepared_plan_fingerprint(plan)
            or plan.dataset_id != binding.dataset_ids[sample_index]
            or plan.piece_id != binding.piece_ids[sample_index]
            or plan.stage != binding.stage
            or plan.epoch != binding.epoch
            or plan.encoder_view_index != binding.encoder_view_index
            or plan.global_seed != binding.global_seed
            or plan.requested_mask_rate != binding.requested_mask_rate
            or plan.maskable_node_count
            != note_ptr[sample_index + 1] - note_ptr[sample_index]
        ):
            raise SSLContractError(
                "prepared binding contains a non-canonical bound plan"
            )
        expected_selected.extend(
            note_ptr[sample_index] + local_index
            for local_index in plan.selected_local_node_indices
        )
    if (
        not isinstance(binding.selected_global_note_indices, tuple)
        or binding.selected_global_note_indices
        != tuple(expected_selected)
    ):
        raise SSLContractError(
            "prepared binding selected global note indices are invalid"
        )
    if (
        not isinstance(binding.feature_overlay, FeatureMaskOverlay)
        or binding.feature_overlay.graph_count != binding.sample_count
        or dict(binding.feature_overlay.node_counts)
        != node_count_by_type
        or binding.feature_overlay.mask_plan_fingerprints
        != binding.ordered_plan_fingerprints
    ):
        raise SSLContractError(
            "prepared binding feature overlay is incompatible"
        )
    binding.feature_overlay.__post_init__()

    payload = _binding_payload(
        dataset_ids=binding.dataset_ids,
        piece_ids=binding.piece_ids,
        stage=binding.stage,
        epoch=binding.epoch,
        encoder_view_index=binding.encoder_view_index,
        global_seed=binding.global_seed,
        requested_mask_rate=binding.requested_mask_rate,
        sample_count=binding.sample_count,
        node_counts=binding.node_counts,
        node_ptrs=binding.node_ptrs,
        edge_counts=binding.edge_counts,
        validated_structure_sha256=(
            binding.validated_structure_sha256
        ),
        note_track_ownership_sha256=(
            binding.note_track_ownership_sha256
        ),
        ordered_plan_fingerprints=(
            binding.ordered_plan_fingerprints
        ),
        feature_overlay_fingerprint=(
            binding.feature_overlay.fingerprint
        ),
        selected_global_note_indices=(
            binding.selected_global_note_indices
        ),
        hierarchy_profile_version=binding.hierarchy_profile_version,
        hierarchy_policy_config_fingerprint=(
            None
            if binding.hierarchy_policy_config is None
            else getattr(
                binding.hierarchy_policy_config,
                "fingerprint",
                None,
            )
        ),
        hierarchy_resolution_fingerprints=(
            binding.hierarchy_resolution_fingerprints
        ),
    )
    if (
        not is_sha256(binding.fingerprint)
        or binding.fingerprint != canonical_sha256(payload)
        or not isinstance(binding._semantic_attestation, str)
        or not hmac.compare_digest(
            binding._semantic_attestation,
            _semantic_attestation(binding.fingerprint),
        )
    ):
        raise SSLContractError(
            "prepared mask binding semantic attestation is invalid"
        )
    selected_tensor = binding.selected_global_note_indices_tensor
    if (
        not isinstance(selected_tensor, Tensor)
        or selected_tensor.dtype != torch.long
        or selected_tensor.ndim != 1
        or int(selected_tensor.shape[0])
        != len(binding.selected_global_note_indices)
    ):
        raise SSLContractError(
            "prepared binding selected-index tensor is invalid"
        )
    if not isinstance(binding._bound_graph, Batch):
        raise SSLContractError(
            "prepared binding must remain bound to a PyG Batch"
        )
    if not isinstance(binding._runtime_attestation, str):
        raise SSLContractError(
            "prepared mask binding runtime attestation is invalid"
        )
    if not isinstance(
        binding._runtime_graph_evidence,
        _RuntimeGraphEvidence,
    ):
        raise SSLContractError(
            "prepared mask binding runtime evidence is invalid"
        )
    if not isinstance(
        binding._selected_indices_evidence,
        _TensorRuntimeEvidence,
    ):
        raise SSLContractError(
            "prepared binding selected-index evidence is invalid"
        )
    _validate_runtime_graph_evidence(
        binding._bound_graph,
        binding._runtime_graph_evidence,
    )
    selected_evidence = _TensorRuntimeEvidence.capture(
        selected_tensor,
        location="binding:selected_global_note_indices_tensor",
    )
    captured_selected_evidence = binding._selected_indices_evidence
    if selected_evidence.location != captured_selected_evidence.location:
        raise _runtime_error("selected_indices_location")
    if selected_evidence.tensor is not captured_selected_evidence.tensor:
        raise _runtime_error("selected_indices_object_identity")
    for name in (
        "object_id",
        "version",
        "shape",
        "dtype",
        "device_type",
        "device_index",
    ):
        if getattr(selected_evidence, name) != getattr(
            captured_selected_evidence,
            name,
        ):
            raise _runtime_error(f"selected_indices_{name}")
    expected_runtime_attestation = _runtime_attestation(
        fingerprint=binding.fingerprint,
        graph_evidence=binding._runtime_graph_evidence,
        selected_indices_evidence=binding._selected_indices_evidence,
    )
    if not hmac.compare_digest(
        binding._runtime_attestation,
        expected_runtime_attestation,
    ):
        raise SSLContractError(
            "prepared mask binding runtime attestation is invalid"
        )


def _cpu_graph_evidence(
    graph_batch: Batch,
    *,
    sample_count: int,
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[tuple[str, str, str], int], ...],
    str,
    str,
]:
    for store in graph_batch.stores:
        for value in store.values():
            if isinstance(value, Tensor) and value.device.type != "cpu":
                raise SSLContractError(
                    "prepared MaskPlans must be constructed from a CPU raw batch"
                )
    node_counts = tuple(
        (
            node_type,
            int(graph_batch[node_type].num_nodes),
        )
        for node_type in MANDATORY_NODE_TYPES
    )
    node_ptrs = tuple(
        (
            node_type,
            tuple(
                int(value)
                for value in graph_batch[node_type].ptr.detach().tolist()
            ),
        )
        for node_type in MANDATORY_NODE_TYPES
    )
    node_batches = tuple(
        (
            node_type,
            tuple(
                int(value)
                for value in (
                    graph_batch[node_type].batch.detach().tolist()
                )
            ),
        )
        for node_type in MANDATORY_NODE_TYPES
    )
    edge_values = tuple(
        (
            edge_type,
            tuple(
                tuple(int(value) for value in row)
                for row in (
                    graph_batch[edge_type]
                    .edge_index.detach()
                    .tolist()
                )
            ),
        )
        for edge_type in MANDATORY_EDGE_TYPES
    )
    edge_counts = tuple(
        (edge_type, len(values[0]))
        for edge_type, values in edge_values
    )
    structure_sha256 = canonical_sha256(
        {
            "sample_count": sample_count,
            "node_counts": [
                [node_type, count]
                for node_type, count in node_counts
            ],
            "node_ptrs": [
                [node_type, list(ptr)]
                for node_type, ptr in node_ptrs
            ],
            "node_batches": [
                [node_type, list(membership)]
                for node_type, membership in node_batches
            ],
            "edges": [
                [
                    list(edge_type),
                    [list(row) for row in values],
                ]
                for edge_type, values in edge_values
            ],
        }
    )
    ownership_values = dict(edge_values)[TRACK_CONTAINS_NOTE_EDGE]
    ownership_sha256 = canonical_sha256(
        {
            "note_ptr": list(dict(node_ptrs)["note"]),
            "track_ptr": list(dict(node_ptrs)["track"]),
            "track_contains_note": [
                list(row) for row in ownership_values
            ],
        }
    )
    return (
        node_counts,
        node_ptrs,
        edge_counts,
        structure_sha256,
        ownership_sha256,
    )


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
        local_node_indices=tuple(
            local_track_index
            for local_track_index in range(track_end - track_start)
            if local_track_index in owner_track_set
        ),
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


def prepare_mask_binding(
    batch: object,
    *,
    global_seed: int,
    epoch: int,
    requested_mask_rate: float = DEFAULT_ENCODER_MASK_RATE,
    stage: MaskStage = "train",
    encoder_view_index: int = 0,
) -> PreparedMaskBinding:
    """Prepare canonical plans and their graph binding on validated CPU input."""

    from music_critic.ssl.data import SSLBatch, validate_ssl_batch

    if not isinstance(batch, SSLBatch):
        raise SSLContractError(
            "prepare_mask_binding requires a raw-only SSLBatch"
        )
    if encoder_view_index != 0:
        raise SSLContractError(
            "prepared Phase 7A bindings require encoder view zero"
        )
    validate_ssl_batch(batch)
    graph_batch = batch.raw_graph_batch
    if not isinstance(graph_batch, Batch):
        raise SSLContractError(
            "prepare_mask_binding requires a PyG Batch"
        )
    (
        node_counts,
        node_ptrs,
        edge_counts,
        structure_sha256,
        ownership_sha256,
    ) = _cpu_graph_evidence(
        graph_batch,
        sample_count=batch.sample_count,
    )
    plans = build_batched_mask_plans(
        graph_batch,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        global_seed=global_seed,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        requested_mask_rate=requested_mask_rate,
        stage=stage,
    )
    feature_overlay = build_feature_mask_overlay(graph_batch, plans)
    canonical_epoch = plans[0].epoch
    canonical_rate = plans[0].requested_mask_rate
    note_ptr = dict(node_ptrs)["note"]
    selected_global_indices = tuple(
        note_ptr[sample_index] + local_index
        for sample_index, plan in enumerate(plans)
        for local_index in plan.selected_local_node_indices
    )
    return PreparedMaskBinding._create(
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        stage=stage,
        epoch=canonical_epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        requested_mask_rate=canonical_rate,
        sample_count=batch.sample_count,
        node_counts=node_counts,
        node_ptrs=node_ptrs,
        edge_counts=edge_counts,
        validated_structure_sha256=structure_sha256,
        note_track_ownership_sha256=ownership_sha256,
        mask_plans=plans,
        feature_overlay=feature_overlay,
        selected_global_note_indices=selected_global_indices,
        bound_graph=graph_batch,
    )


def prepare_hierarchy_mask_binding(
    batch: object,
    *,
    policy_config: object,
    global_seed: int,
    epoch: int,
    requested_mask_rate: float = DEFAULT_ENCODER_MASK_RATE,
    stage: MaskStage = "train",
    encoder_view_index: int = 0,
) -> PreparedMaskBinding | PreparedHierarchyMaskBinding:
    """Prepare a Phase 8A view through the shared attested binding kernel.

    An independent-only configuration delegates to ``prepare_mask_binding``
    exactly, preserving the complete Phase 7A portable binding and runtime
    behavior.  Every other configuration binds its mixture-resolution
    evidence in the hierarchy profile while reusing the same full graph
    attestation, opaque token, and transfer implementation.
    """

    from music_critic.ssl.data import SSLBatch, validate_ssl_batch
    from music_critic.ssl.hierarchical_masking import (
        HIERARCHY_PREPARED_BINDING_PROFILE_VERSION,
        INDEPENDENT_NOTE_PITCH,
        HierarchyMaskPolicyConfig,
        HierarchyMaskUnavailableError,
        build_batched_hierarchy_mask_resolutions,
        validate_hierarchy_policy_config,
    )

    if not isinstance(batch, SSLBatch):
        raise SSLContractError(
            "prepare_hierarchy_mask_binding requires a raw-only SSLBatch"
        )
    if type(policy_config) is not HierarchyMaskPolicyConfig:
        raise SSLContractError(
            "prepare_hierarchy_mask_binding requires an exact "
            "HierarchyMaskPolicyConfig"
        )
    canonical_config = validate_hierarchy_policy_config(
        policy_config
    )
    if encoder_view_index != 0:
        raise SSLContractError(
            "prepared Phase 8A bindings require encoder view zero"
        )
    if canonical_config.enabled_policies() == (
        INDEPENDENT_NOTE_PITCH,
    ):
        return prepare_mask_binding(
            batch,
            global_seed=global_seed,
            epoch=epoch,
            requested_mask_rate=requested_mask_rate,
            stage=stage,
            encoder_view_index=encoder_view_index,
        )
    validate_ssl_batch(batch)
    graph_batch = batch.raw_graph_batch
    if not isinstance(graph_batch, Batch):
        raise SSLContractError(
            "prepare_hierarchy_mask_binding requires a PyG Batch"
        )
    (
        node_counts,
        node_ptrs,
        edge_counts,
        structure_sha256,
        ownership_sha256,
    ) = _cpu_graph_evidence(
        graph_batch,
        sample_count=batch.sample_count,
    )
    resolutions = build_batched_hierarchy_mask_resolutions(
        graph_batch,
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        global_seed=global_seed,
        epoch=epoch,
        encoder_view_index=encoder_view_index,
        requested_mask_rate=requested_mask_rate,
        stage=stage,
        policy_config=canonical_config,
    )
    if any(resolution.plan is None for resolution in resolutions):
        raise HierarchyMaskUnavailableError(resolutions)
    plans = tuple(
        resolution.plan
        for resolution in resolutions
        if resolution.plan is not None
    )
    feature_overlay = build_feature_mask_overlay(
        graph_batch,
        plans,
    )
    canonical_epoch = plans[0].epoch
    canonical_rate = plans[0].requested_mask_rate
    note_ptr = dict(node_ptrs)["note"]
    selected_global_indices = tuple(
        note_ptr[sample_index] + local_index
        for sample_index, plan in enumerate(plans)
        for local_index in plan.selected_local_node_indices
    )
    return PreparedHierarchyMaskBinding._create(
        dataset_ids=batch.dataset_ids,
        piece_ids=batch.piece_ids,
        stage=stage,
        epoch=canonical_epoch,
        encoder_view_index=encoder_view_index,
        global_seed=global_seed,
        requested_mask_rate=canonical_rate,
        sample_count=batch.sample_count,
        node_counts=node_counts,
        node_ptrs=node_ptrs,
        edge_counts=edge_counts,
        validated_structure_sha256=structure_sha256,
        note_track_ownership_sha256=ownership_sha256,
        mask_plans=plans,
        feature_overlay=feature_overlay,
        selected_global_note_indices=selected_global_indices,
        hierarchy_profile_version=(
            HIERARCHY_PREPARED_BINDING_PROFILE_VERSION
        ),
        hierarchy_policy_config=canonical_config,
        hierarchy_resolutions=resolutions,
        bound_graph=graph_batch,
    )


prepare_hierarchical_mask_binding = prepare_hierarchy_mask_binding


def _validate_prepared_mask_binding_runtime(
    batch: object,
    binding: PreparedMaskBinding,
    *,
    expected_mask_rate: float | None = None,
) -> None:
    """Validate a prepared binding without reading accelerator graph values."""

    from music_critic.ssl.data import SSLBatch

    if not isinstance(batch, SSLBatch):
        raise SSLContractError(
            "prepared mask binding requires a raw-only SSLBatch"
        )
    if not isinstance(binding, PreparedMaskBinding):
        raise SSLContractError(
            "prepared mask binding has an invalid type"
        )
    _validate_prepared_mask_binding_contract(binding)
    if (
        binding._bound_graph is not batch.raw_graph_batch
        or binding.dataset_ids != batch.dataset_ids
        or binding.piece_ids != batch.piece_ids
        or binding.sample_count != batch.sample_count
        or sum(count for _, count in binding.node_counts)
        != batch.node_count
        or sum(count for _, count in binding.edge_counts)
        != batch.edge_count
    ):
        raise SSLContractError(
            "prepared mask binding does not belong to this SSLBatch"
        )
    graph = batch.raw_graph_batch
    if (
        not isinstance(graph, Batch)
        or int(graph.num_graphs) != binding.sample_count
        or tuple(graph.node_types) != MANDATORY_NODE_TYPES
        or tuple(graph.edge_types) != MANDATORY_EDGE_TYPES
    ):
        raise SSLContractError(
            "prepared mask binding graph schema is incompatible"
        )
    for node_type, count in binding.node_counts:
        store = graph[node_type]
        if (
            int(store.num_nodes) != count
            or not isinstance(store.ptr, Tensor)
            or store.ptr.dtype != torch.long
            or store.ptr.ndim != 1
            or int(store.ptr.shape[0]) != binding.sample_count + 1
        ):
            raise SSLContractError(
                f"prepared mask binding {node_type} shape is incompatible"
            )
    for edge_type, count in binding.edge_counts:
        edge_index = graph[edge_type].edge_index
        if (
            not isinstance(edge_index, Tensor)
            or edge_index.dtype != torch.long
            or edge_index.ndim != 2
            or tuple(edge_index.shape) != (2, count)
        ):
            raise SSLContractError(
                "prepared mask binding edge shape is incompatible: "
                + "|".join(edge_type)
            )
    graph_device = graph["note"].x_cat.device
    if (
        binding.selected_global_note_indices_tensor.device
        != graph_device
    ):
        raise SSLContractError(
            "prepared binding indices and raw graph use different devices"
        )
    if expected_mask_rate is not None:
        rate = validate_mask_rate(expected_mask_rate)
        if rate != binding.requested_mask_rate:
            raise SSLContractError(
                "prepared binding mask rate differs from the SSL model"
            )


def _prepared_input_token_attestation(
    *,
    batch: object,
    graph: object,
    binding: PreparedMaskBinding,
    expected_mask_rate: float | None,
) -> str:
    message = canonical_sha256(
        {
            "batch_object": id(batch),
            "graph_object": id(graph),
            "binding_object": id(binding),
            "binding_fingerprint": binding.fingerprint,
            "binding_runtime_attestation": (
                binding._runtime_attestation
            ),
            "expected_mask_rate": expected_mask_rate,
        }
    ).encode("ascii")
    return hmac.new(
        _PREPARED_BINDING_ATTESTATION_KEY,
        message,
        digestmod="sha256",
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class _ValidatedPreparedInput:
    """Opaque, process-local capability for one attested prepared input."""

    batch: object = field(repr=False, compare=False)
    graph: HeteroData = field(repr=False, compare=False)
    binding: PreparedMaskBinding = field(repr=False, compare=False)
    expected_mask_rate: float | None
    _attestation: str = field(repr=False, compare=False)


def validate_prepared_mask_binding(
    batch: object,
    binding: PreparedMaskBinding,
    *,
    expected_mask_rate: float | None = None,
) -> _ValidatedPreparedInput:
    """Validate live evidence and issue one opaque prepared-input capability."""

    _validate_prepared_mask_binding_runtime(
        batch,
        binding,
        expected_mask_rate=expected_mask_rate,
    )
    graph = batch.raw_graph_batch
    return _ValidatedPreparedInput(
        batch=batch,
        graph=graph,
        binding=binding,
        expected_mask_rate=expected_mask_rate,
        _attestation=_prepared_input_token_attestation(
            batch=batch,
            graph=graph,
            binding=binding,
            expected_mask_rate=expected_mask_rate,
        ),
    )


def _verify_prepared_input_token(
    graph: object,
    token: object,
) -> None:
    """Re-attest a private capability immediately before encoder work."""

    if type(token) is not _ValidatedPreparedInput:
        raise SSLContractError(
            "ssl.prepared_binding.validated_token_invalid:type"
        )
    if graph is not token.graph:
        raise SSLContractError(
            "ssl.prepared_binding.validated_token_invalid:graph"
        )
    expected_attestation = _prepared_input_token_attestation(
        batch=token.batch,
        graph=token.graph,
        binding=token.binding,
        expected_mask_rate=token.expected_mask_rate,
    )
    if (
        not isinstance(token._attestation, str)
        or not hmac.compare_digest(
            token._attestation,
            expected_attestation,
        )
    ):
        raise SSLContractError(
            "ssl.prepared_binding.validated_token_invalid:attestation"
        )
    _validate_prepared_mask_binding_runtime(
        token.batch,
        token.binding,
        expected_mask_rate=token.expected_mask_rate,
    )


def move_ssl_batch_with_prepared_binding(
    batch: object,
    binding: PreparedMaskBinding,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> tuple[object, PreparedMaskBinding]:
    """Move one raw batch and its prepared index sidecar as one trusted step."""

    from music_critic.ssl.data import (
        _require_ssl_tensor_device,
        move_ssl_batch,
    )

    validate_prepared_mask_binding(batch, binding)
    target_device = resolve_runtime_device(device)
    moved_batch = move_ssl_batch(
        batch,
        target_device,
        non_blocking=non_blocking,
    )
    moved_indices = binding.selected_global_note_indices_tensor.to(
        device=target_device,
        non_blocking=non_blocking,
        copy=True,
    )
    _require_ssl_tensor_device(
        moved_indices,
        device=target_device,
        location="binding:selected_global_note_indices_tensor",
    )
    moved_runtime_graph_evidence = _capture_runtime_graph_evidence(
        moved_batch.raw_graph_batch
    )
    _validate_transferred_runtime_evidence(
        binding._runtime_graph_evidence,
        moved_runtime_graph_evidence,
    )
    moved_selected_indices_evidence = _TensorRuntimeEvidence.capture(
        moved_indices,
        location="binding:selected_global_note_indices_tensor",
    )
    moved_binding = replace(
        binding,
        selected_global_note_indices_tensor=moved_indices,
        _bound_graph=moved_batch.raw_graph_batch,
        _runtime_graph_evidence=moved_runtime_graph_evidence,
        _selected_indices_evidence=moved_selected_indices_evidence,
        _runtime_attestation=_runtime_attestation(
            fingerprint=binding.fingerprint,
            graph_evidence=moved_runtime_graph_evidence,
            selected_indices_evidence=moved_selected_indices_evidence,
        ),
    )
    validate_prepared_mask_binding(moved_batch, moved_binding)
    return moved_batch, moved_binding


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
    "PREPARED_HIERARCHY_MASK_BINDING_CONTRACT_VERSION",
    "PreparedHierarchyMaskBinding",
    "PreparedMaskBinding",
    "TRACK_CONTAINS_NOTE_EDGE",
    "build_batch_mask_plans",
    "build_batched_mask_plans",
    "build_mask_plan",
    "build_mask_plans_for_batch",
    "derive_stable_seed",
    "move_ssl_batch_with_prepared_binding",
    "prepare_hierarchical_mask_binding",
    "prepare_hierarchy_mask_binding",
    "prepare_mask_binding",
    "validate_prepared_mask_binding",
]
