"""Semantic field groups for leakage-safe Phase 7A encoder masking."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from music_critic.graph.feature_registry import (
    RAW_FEATURE_REGISTRY,
    FeatureRegistry,
    FeatureSpec,
)
from music_critic.ssl.contracts import (
    FeatureKind,
    MaskedFeature,
    SSLContractError,
    canonical_sha256,
)


MASKABLE_FIELD_REGISTRY_VERSION = "1.0.0"
NOTE_PITCH_GROUP_NAME = "note_pitch_group"
OWNER_TRACK_PITCH_STATISTICS_REASON = "owner_track_pitch_statistics"
OWNER_TRACK_PEER_RELATIVE_PITCH_REASON = (
    "owner_track_peer_relative_pitch"
)


@dataclass(frozen=True, slots=True)
class ResolvedFeatureColumn:
    """A semantic field resolved to a raw-registry column by name."""

    field: MaskedFeature
    column_index: int
    raw_feature_spec: FeatureSpec

    def __post_init__(self) -> None:
        if (
            isinstance(self.column_index, bool)
            or not isinstance(self.column_index, int)
            or self.column_index < 0
        ):
            raise SSLContractError("resolved feature column must be non-negative")
        if (
            self.raw_feature_spec.node_type != self.field.node_type
            or self.raw_feature_spec.kind != self.field.kind
            or self.raw_feature_spec.name != self.field.feature_name
        ):
            raise SSLContractError(
                "resolved raw feature does not match its semantic mask field"
            )
        if (
            self.field.mask_availability
            and not self.raw_feature_spec.has_availability_mask
        ):
            raise SSLContractError(
                "masked field requires an unavailable raw availability column"
            )

    @property
    def value_tensor_name(self) -> str:
        return "x_cat" if self.field.kind == "categorical" else "x_cont"

    @property
    def availability_tensor_name(self) -> str:
        return (
            "x_cat_available"
            if self.field.kind == "categorical"
            else "x_cont_available"
        )


@dataclass(frozen=True, slots=True)
class MaskableFieldGroup:
    """One primary node feature group plus its explicit collateral closure."""

    name: str
    selected_node_type: str
    primary_fields: tuple[MaskedFeature, ...]
    collateral_reason: str
    collateral_fields: tuple[MaskedFeature, ...]
    peer_note_collateral_reason: str
    peer_note_collateral_fields: tuple[MaskedFeature, ...]

    def __post_init__(self) -> None:
        if self.name != NOTE_PITCH_GROUP_NAME:
            raise SSLContractError("Phase 7A supports only note_pitch_group")
        if self.selected_node_type != "note":
            raise SSLContractError("note_pitch_group must select note nodes")
        if (
            not self.primary_fields
            or not self.collateral_fields
            or not self.peer_note_collateral_fields
        ):
            raise SSLContractError(
                "maskable group requires primary and collateral fields"
            )
        if any(
            field.node_type != self.selected_node_type
            for field in self.primary_fields
        ):
            raise SSLContractError(
                "primary fields must belong to the selected node type"
            )
        if any(field.node_type != "track" for field in self.collateral_fields):
            raise SSLContractError(
                "Phase 7A collateral pitch statistics must belong to tracks"
            )
        if any(
            field.node_type != "note"
            for field in self.peer_note_collateral_fields
        ):
            raise SSLContractError(
                "Phase 7A peer-relative collateral fields must belong to notes"
            )
        for name, fields in (
            ("primary", self.primary_fields),
            ("collateral", self.collateral_fields),
            (
                "peer note collateral",
                self.peer_note_collateral_fields,
            ),
        ):
            identities = tuple(
                (field.node_type, field.kind, field.feature_name)
                for field in fields
            )
            if len(identities) != len(set(identities)):
                raise SSLContractError(f"{name} maskable fields must be unique")
        if self.collateral_reason != OWNER_TRACK_PITCH_STATISTICS_REASON:
            raise SSLContractError("collateral mask reason is incompatible")
        if (
            self.peer_note_collateral_reason
            != OWNER_TRACK_PEER_RELATIVE_PITCH_REASON
        ):
            raise SSLContractError(
                "peer-note collateral mask reason is incompatible"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "selected_node_type": self.selected_node_type,
            "primary_fields": [
                field.to_dict() for field in self.primary_fields
            ],
            "collateral_reason": self.collateral_reason,
            "collateral_fields": [
                field.to_dict() for field in self.collateral_fields
            ],
            "peer_note_collateral_reason": (
                self.peer_note_collateral_reason
            ),
            "peer_note_collateral_fields": [
                field.to_dict()
                for field in self.peer_note_collateral_fields
            ],
        }


class MaskableFieldRegistry:
    """Immutable SSL registry validated against the raw feature registry."""

    def __init__(
        self,
        *,
        version: str,
        raw_registry: FeatureRegistry,
        groups: tuple[MaskableFieldGroup, ...],
    ) -> None:
        if not isinstance(version, str) or not version:
            raise SSLContractError("maskable field registry version is invalid")
        if not groups:
            raise SSLContractError("maskable field registry cannot be empty")
        names = tuple(group.name for group in groups)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise SSLContractError(
                "maskable field groups must be uniquely sorted by name"
            )
        self._version = version
        self._raw_registry = raw_registry
        self._groups = groups
        for group in groups:
            for field in (
                *group.primary_fields,
                *group.collateral_fields,
                *group.peer_note_collateral_fields,
            ):
                resolve_feature_column(field, raw_registry=raw_registry)

    @property
    def version(self) -> str:
        return self._version

    @property
    def raw_registry_version(self) -> str:
        return self._raw_registry.version

    @property
    def groups(self) -> tuple[MaskableFieldGroup, ...]:
        return self._groups

    def group(self, name: str) -> MaskableFieldGroup:
        for group in self._groups:
            if group.name == name:
                return group
        raise SSLContractError(f"unknown maskable feature group {name!r}")

    def resolve_group(
        self, name: str
    ) -> tuple[tuple[ResolvedFeatureColumn, ...], tuple[ResolvedFeatureColumn, ...]]:
        group = self.group(name)
        return (
            tuple(
                resolve_feature_column(field, raw_registry=self._raw_registry)
                for field in group.primary_fields
            ),
            tuple(
                resolve_feature_column(field, raw_registry=self._raw_registry)
                for field in group.collateral_fields
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "raw_feature_registry": {
                "version": self._raw_registry.version,
                "specs": [asdict(spec) for spec in self._raw_registry.specs],
            },
            "groups": [group.to_dict() for group in self.groups],
        }


def resolve_feature_column(
    field: MaskedFeature,
    *,
    raw_registry: FeatureRegistry = RAW_FEATURE_REGISTRY,
) -> ResolvedFeatureColumn:
    """Resolve a field by semantic name, never by a hard-coded numeric index."""

    specs = raw_registry.for_node(field.node_type, field.kind)
    matches = tuple(spec for spec in specs if spec.name == field.feature_name)
    if len(matches) != 1:
        raise SSLContractError(
            f"raw registry must contain exactly one "
            f"{field.node_type}.{field.feature_name} {field.kind} field"
        )
    names = raw_registry.names(field.node_type, field.kind)
    return ResolvedFeatureColumn(
        field=field,
        column_index=names.index(field.feature_name),
        raw_feature_spec=matches[0],
    )


NOTE_PITCH_GROUP = MaskableFieldGroup(
    name=NOTE_PITCH_GROUP_NAME,
    selected_node_type="note",
    primary_fields=(
        MaskedFeature("note", "categorical", "pitch"),
        MaskedFeature("note", "categorical", "pitch_class"),
        MaskedFeature("note", "categorical", "octave"),
        MaskedFeature("note", "continuous", "track_relative_pitch"),
    ),
    collateral_reason=OWNER_TRACK_PITCH_STATISTICS_REASON,
    collateral_fields=(
        MaskedFeature("track", "continuous", "mean_pitch"),
        MaskedFeature("track", "continuous", "pitch_std"),
        MaskedFeature("track", "continuous", "min_pitch"),
        MaskedFeature("track", "continuous", "max_pitch"),
    ),
    peer_note_collateral_reason=(
        OWNER_TRACK_PEER_RELATIVE_PITCH_REASON
    ),
    peer_note_collateral_fields=(
        MaskedFeature(
            "note",
            "continuous",
            "track_relative_pitch",
        ),
    ),
)

SSL_MASKABLE_FIELD_REGISTRY = MaskableFieldRegistry(
    version=MASKABLE_FIELD_REGISTRY_VERSION,
    raw_registry=RAW_FEATURE_REGISTRY,
    groups=(NOTE_PITCH_GROUP,),
)


def maskable_field_registry_fingerprint(
    registry: MaskableFieldRegistry = SSL_MASKABLE_FIELD_REGISTRY,
) -> str:
    """Bind semantic masks to the complete ordered raw feature contract."""

    return canonical_sha256(registry.to_dict())


MASKABLE_FIELD_REGISTRY_FINGERPRINT = maskable_field_registry_fingerprint()


__all__ = [
    "MASKABLE_FIELD_REGISTRY_FINGERPRINT",
    "MASKABLE_FIELD_REGISTRY_VERSION",
    "NOTE_PITCH_GROUP",
    "NOTE_PITCH_GROUP_NAME",
    "OWNER_TRACK_PEER_RELATIVE_PITCH_REASON",
    "OWNER_TRACK_PITCH_STATISTICS_REASON",
    "SSL_MASKABLE_FIELD_REGISTRY",
    "MaskableFieldGroup",
    "MaskableFieldRegistry",
    "ResolvedFeatureColumn",
    "maskable_field_registry_fingerprint",
    "resolve_feature_column",
]
