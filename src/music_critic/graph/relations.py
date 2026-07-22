"""Stable raw-graph node and relation names."""

from __future__ import annotations

from typing import TypeAlias


GRAPH_SCHEMA_VERSION = "1.0.0"
GRAPH_BUILDER_VERSION = "1.0.0"

EdgeType: TypeAlias = tuple[str, str, str]

MANDATORY_NODE_TYPES = ("song", "track", "bar", "beat", "onset", "note")

CONTAINMENT_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("song", "contains_track", "track"),
    ("track", "belongs_to_song", "song"),
    ("song", "contains_bar", "bar"),
    ("bar", "belongs_to_song", "song"),
    ("track", "contains_note", "note"),
    ("note", "belongs_to_track", "track"),
    ("bar", "contains_beat", "beat"),
    ("beat", "belongs_to_bar", "bar"),
    ("bar", "contains_onset", "onset"),
    ("onset", "belongs_to_bar", "bar"),
    ("bar", "contains_note", "note"),
    ("note", "belongs_to_bar", "bar"),
    ("beat", "contains_onset", "onset"),
    ("onset", "belongs_to_beat", "beat"),
    ("onset", "starts_note", "note"),
    ("note", "in_onset", "onset"),
)

TEMPORAL_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("bar", "next_bar", "bar"),
    ("bar", "previous_bar", "bar"),
    ("beat", "next_beat", "beat"),
    ("beat", "previous_beat", "beat"),
    ("onset", "next_onset", "onset"),
    ("onset", "previous_onset", "onset"),
    ("note", "next_in_track", "note"),
    ("note", "previous_in_track", "note"),
)

SUSTAINED_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("note", "active_at", "beat"),
    ("beat", "has_active_note", "note"),
)

MANDATORY_EDGE_TYPES = (
    *CONTAINMENT_EDGE_TYPES,
    *TEMPORAL_EDGE_TYPES,
    *SUSTAINED_EDGE_TYPES,
)

REVERSE_EDGE_TYPES: dict[EdgeType, EdgeType] = {
    ("song", "contains_track", "track"): ("track", "belongs_to_song", "song"),
    ("song", "contains_bar", "bar"): ("bar", "belongs_to_song", "song"),
    ("track", "contains_note", "note"): ("note", "belongs_to_track", "track"),
    ("bar", "contains_beat", "beat"): ("beat", "belongs_to_bar", "bar"),
    ("bar", "contains_onset", "onset"): ("onset", "belongs_to_bar", "bar"),
    ("bar", "contains_note", "note"): ("note", "belongs_to_bar", "bar"),
    ("beat", "contains_onset", "onset"): ("onset", "belongs_to_beat", "beat"),
    ("onset", "starts_note", "note"): ("note", "in_onset", "onset"),
    ("bar", "next_bar", "bar"): ("bar", "previous_bar", "bar"),
    ("beat", "next_beat", "beat"): ("beat", "previous_beat", "beat"),
    ("onset", "next_onset", "onset"): ("onset", "previous_onset", "onset"),
    ("note", "next_in_track", "note"): (
        "note",
        "previous_in_track",
        "note",
    ),
    ("note", "active_at", "beat"): ("beat", "has_active_note", "note"),
}


__all__ = [
    "CONTAINMENT_EDGE_TYPES",
    "EdgeType",
    "GRAPH_BUILDER_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "MANDATORY_EDGE_TYPES",
    "MANDATORY_NODE_TYPES",
    "REVERSE_EDGE_TYPES",
    "SUSTAINED_EDGE_TYPES",
    "TEMPORAL_EDGE_TYPES",
]
