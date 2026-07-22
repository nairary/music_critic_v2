"""Versioned, raw-inference-safe graph feature declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FEATURE_REGISTRY_VERSION = "1.0.0"

NodeType = Literal["song", "track", "bar", "beat", "onset", "note"]
FeatureKind = Literal["categorical", "continuous"]
Normalization = Literal["none", "log1p", "zscore_within_track"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One stable model-facing feature column."""

    name: str
    node_type: NodeType
    kind: FeatureKind
    vocabulary_size: int | None = None
    unknown_id: int | None = None
    normalization: Normalization = "none"
    has_availability_mask: bool = True
    raw_inference_safe: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("feature names must be non-empty and trimmed")
        if not self.raw_inference_safe:
            raise ValueError("the raw feature registry cannot contain unsafe features")
        if self.kind == "categorical":
            if self.vocabulary_size is None or self.vocabulary_size <= 0:
                raise ValueError("categorical features require a vocabulary size")
            if self.unknown_id is not None and not (
                0 <= self.unknown_id < self.vocabulary_size
            ):
                raise ValueError("unknown_id must be inside the vocabulary")
        elif self.vocabulary_size is not None or self.unknown_id is not None:
            raise ValueError("continuous features cannot declare a vocabulary")


class FeatureRegistry:
    """Immutable registry with deterministic per-node column ordering."""

    def __init__(self, version: str, specs: tuple[FeatureSpec, ...]) -> None:
        if not version:
            raise ValueError("registry version must be non-empty")
        identities = [(spec.node_type, spec.name) for spec in specs]
        if len(identities) != len(set(identities)):
            raise ValueError("feature names must be unique within each node type")
        self._version = version
        self._specs = specs

    @property
    def version(self) -> str:
        return self._version

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        return self._specs

    def for_node(
        self, node_type: NodeType, kind: FeatureKind | None = None
    ) -> tuple[FeatureSpec, ...]:
        return tuple(
            spec
            for spec in self._specs
            if spec.node_type == node_type and (kind is None or spec.kind == kind)
        )

    def names(self, node_type: NodeType, kind: FeatureKind) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.for_node(node_type, kind))


def _cat(
    node_type: NodeType,
    name: str,
    vocabulary_size: int,
    *,
    unknown_id: int | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        node_type=node_type,
        kind="categorical",
        vocabulary_size=vocabulary_size,
        unknown_id=unknown_id,
    )


def _cont(
    node_type: NodeType,
    name: str,
    *,
    normalization: Normalization = "none",
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        node_type=node_type,
        kind="continuous",
        normalization=normalization,
    )


RAW_FEATURE_REGISTRY = FeatureRegistry(
    FEATURE_REGISTRY_VERSION,
    (
        _cont("song", "duration_qn"),
        _cont("song", "track_count", normalization="log1p"),
        _cont("song", "bar_count", normalization="log1p"),
        _cont("song", "beat_count", normalization="log1p"),
        _cont("song", "onset_count", normalization="log1p"),
        _cont("song", "note_count", normalization="log1p"),
        _cont("song", "tempo_mean_us_per_qn"),
        _cont("song", "tempo_min_us_per_qn"),
        _cont("song", "tempo_max_us_per_qn"),
        _cont("song", "tempo_change_count", normalization="log1p"),
        _cont("song", "meter_change_count", normalization="log1p"),
        _cat("track", "program", 128, unknown_id=0),
        _cat("track", "channel", 16, unknown_id=0),
        _cat("track", "is_percussion", 2),
        _cont("track", "source_track_index"),
        _cont("track", "note_count", normalization="log1p"),
        _cont("track", "mean_pitch"),
        _cont("track", "pitch_std"),
        _cont("track", "min_pitch"),
        _cont("track", "max_pitch"),
        _cont("track", "note_density", normalization="log1p"),
        _cont("track", "polyphony_ratio"),
        _cont("track", "active_bar_ratio"),
        _cont("track", "mean_duration_qn"),
        _cont("track", "mean_velocity"),
        _cat("bar", "meter_numerator", 256, unknown_id=0),
        _cat("bar", "meter_denominator_log2", 128, unknown_id=0),
        _cat("bar", "is_pickup", 2),
        _cat("bar", "is_incomplete", 2),
        _cont("bar", "index"),
        _cont("bar", "start_qn"),
        _cont("bar", "duration_qn"),
        _cont("bar", "metric_offset_qn"),
        _cont("bar", "tempo_us_per_qn"),
        _cont("bar", "starting_note_count", normalization="log1p"),
        _cont("bar", "active_note_count", normalization="log1p"),
        _cont("bar", "onset_count", normalization="log1p"),
        _cont("bar", "active_track_count", normalization="log1p"),
        _cat("beat", "meter_numerator", 256, unknown_id=0),
        _cat("beat", "meter_denominator_log2", 128, unknown_id=0),
        _cat("beat", "is_downbeat", 2),
        _cont("beat", "index_in_bar"),
        _cont("beat", "start_qn"),
        _cont("beat", "duration_qn"),
        _cont("beat", "position_in_bar_qn"),
        _cont("beat", "strength"),
        _cont("beat", "tempo_us_per_qn"),
        _cont("beat", "starting_note_count", normalization="log1p"),
        _cont("beat", "active_note_count", normalization="log1p"),
        _cont("beat", "active_track_count", normalization="log1p"),
        _cont("onset", "start_qn"),
        _cont("onset", "position_in_bar_qn"),
        _cont("onset", "starting_note_count", normalization="log1p"),
        _cont("onset", "active_note_count", normalization="log1p"),
        _cont("onset", "active_track_count", normalization="log1p"),
        _cont("onset", "onsets_in_beat", normalization="log1p"),
        _cat("note", "pitch", 128),
        _cat("note", "pitch_class", 12),
        _cat("note", "octave", 11),
        _cat("note", "program", 128, unknown_id=0),
        _cat("note", "channel", 16, unknown_id=0),
        _cat("note", "is_percussion", 2),
        _cat("note", "is_grace", 2),
        _cont("note", "onset_qn"),
        _cont("note", "duration_qn"),
        _cont("note", "velocity"),
        _cont("note", "position_in_bar_qn"),
        _cont("note", "track_relative_pitch", normalization="zscore_within_track"),
    ),
)


__all__ = [
    "FEATURE_REGISTRY_VERSION",
    "FeatureKind",
    "FeatureRegistry",
    "FeatureSpec",
    "NodeType",
    "Normalization",
    "RAW_FEATURE_REGISTRY",
]
