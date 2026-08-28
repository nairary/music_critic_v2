"""Exact-timing adapter to the AnalysisGNN/GraphMuse graph surface.

Only this experiment-local module depends on GraphMuse.  It does not alter the
Music Critic graph builder and never reads target columns while constructing
topology or input features.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch

from music_critic.data import CanonicalBar, CanonicalPiece, RationalTime
from music_critic.experiments.analysisgnn.contracts import (
    BASE_FEATURE_NAMES,
    EDGE_TYPES,
    INVERSION_VOCABULARY,
    SEMITONES_BY_TRANSPOSITION,
    TRANSPOSITIONS,
    canonical_json,
    graph_schema_fingerprint,
)
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_QUALITY_TASK,
    DILEMMADATA_COMMON_FAMILY_BY_TASK,
    DilemmadataCommonHarmonicProjection,
)
from music_critic.tasks.multisource import TargetBundle

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


_STEPS = ("C", "D", "E", "F", "G", "A", "B")
_NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_INTERVAL_NUMBER = {
    "P1": 1,
    "m2": 2,
    "M2": 2,
    "m3": 3,
    "M3": 3,
    "P4": 4,
    "A4": 4,
    "P5": 5,
    "m6": 6,
    "M6": 6,
    "m7": 7,
    "M7": 7,
}
_KEY_FIFTH_DELTA = {
    "P1": 0,
    "m2": 7,
    "M2": 2,
    "m3": -3,
    "M3": 4,
    "P4": -1,
    "A4": 8,
    "P5": 1,
    "m6": -4,
    "M6": 3,
    "m7": -2,
    "M7": 5,
}
_PITCH_CLASSES = tuple(
    sorted(
        pitch
        for pitches in (
            ("C", "B#", "D--"),
            ("C#", "B##", "D-"),
            ("D", "C##", "E--"),
            ("D#", "E-", "F--"),
            ("E", "D##", "F-"),
            ("F", "E#", "G--"),
            ("F#", "E##", "G-"),
            ("G", "F##", "A--"),
            ("G#", "A-"),
            ("A", "G##", "B--"),
            ("A#", "B-", "C--"),
            ("B", "A##", "C-"),
        )
        for pitch in pitches
    )
)
_PITCH_TO_INDEX = {name: index for index, name in enumerate(_PITCH_CLASSES)}
_QUALITY_VOCABULARY = DILEMMADATA_COMMON_FAMILY_BY_TASK[
    COMMON_QUALITY_TASK
].vocabulary
if _QUALITY_VOCABULARY is None or len(_QUALITY_VOCABULARY) != 50:
    raise RuntimeError("Phase 9E-A quality vocabulary is not the frozen 50-class set")
_QUALITY_TO_INDEX = {value: index for index, value in enumerate(_QUALITY_VOCABULARY)}
_INVERSION_TO_INDEX = {value: index for index, value in enumerate(INVERSION_VOCABULARY)}


class AnalysisGNNGraphError(ValueError):
    """Raised when a source cannot satisfy the pinned graph contract."""


@dataclass(frozen=True, slots=True)
class GraphEntry:
    task: str
    entry_index: int
    entity_id: str
    label: int
    mask: bool


def _as_float(value: RationalTime) -> float:
    return value.num / value.den


def _contains(start: RationalTime, end: RationalTime, onset: RationalTime) -> bool:
    return onset == start if start == end else start <= onset < end


def _bar_for_onset(piece: CanonicalPiece, onset: RationalTime) -> CanonicalBar:
    candidates = tuple(
        bar
        for bar in piece.bars
        if _contains(bar.start_qn, bar.start_qn + bar.duration_qn, onset)
    )
    if len(candidates) != 1:
        raise AnalysisGNNGraphError(
            f"note onset {onset!r} has {len(candidates)} exact bar memberships"
        )
    return candidates[0]


def _active_key_fifths(piece: CanonicalPiece, onset: RationalTime) -> int:
    active = [event for event in piece.key_signature_events if event.onset_qn <= onset]
    return active[-1].fifths if active else 0


def _render_spelling(step: str, alter: int) -> str:
    if alter < -2 or alter > 2:
        raise AnalysisGNNGraphError("AnalysisGNN spelling vocabulary supports alterations -2..2")
    return step + ("#" * alter if alter > 0 else "-" * -alter)


def _canonical_spelling(pitch: int) -> tuple[str, int]:
    names = (("C", 0), ("C", 1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
             ("F", 1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0))
    return names[pitch % 12]


def _transpose_spelling(
    step: str | None,
    alter: int | None,
    pitch: int,
    transposition: str,
) -> tuple[str, int, int]:
    if step not in _STEPS or alter is None or not -2 <= alter <= 2:
        step, alter = _canonical_spelling(pitch)
    # Preserve the public AnalysisGNN pitch augmentation boundary exactly.
    shifted_pitch = (pitch + SEMITONES_BY_TRANSPOSITION[transposition]) % 128
    shifted_pitch_class = (pitch + SEMITONES_BY_TRANSPOSITION[transposition]) % 12
    step_index = (_STEPS.index(step) + _INTERVAL_NUMBER[transposition] - 1) % 7
    shifted_step = _STEPS[step_index]
    # Spelling transposition is independent of the public MIDI-128 wrap.
    shifted_alter = (shifted_pitch_class - _NATURAL_PC[shifted_step]) % 12
    if shifted_alter > 6:
        shifted_alter -= 12
    # Rare double-accidental boundary cases are enharmonically normalized to
    # the same 35-token public vocabulary; this is a declared substitution.
    if shifted_alter < -2 or shifted_alter > 2:
        shifted_step, shifted_alter = _canonical_spelling(shifted_pitch_class)
    return shifted_step, shifted_alter, shifted_pitch


def _transpose_fifths(fifths: int, transposition: str) -> int:
    value = fifths + _KEY_FIFTH_DELTA[transposition]
    while value > 7:
        value -= 12
    while value < -7:
        value += 12
    return value


def _note_features(piece: CanonicalPiece, note: Any, pitch: int) -> np.ndarray:
    bar = _bar_for_onset(piece, note.onset_qn)
    offset = note.onset_qn - bar.start_qn
    duration_ratio = _as_float(note.duration_qn) / _as_float(bar.duration_qn)
    onset_ratio = _as_float(offset) / _as_float(bar.duration_qn)
    # Preserve the public descriptor exactly: despite its historical name,
    # `is_down_beat` marks every integer metric beat, not only bar starts.
    is_downbeat = float(any(beat.start_qn == note.onset_qn for beat in piece.beats))
    pc = np.zeros(12, dtype=np.float32)
    pc[pitch % 12] = 1.0
    octave = np.zeros(10, dtype=np.float32)
    octave_index = min(max(pitch // 12, 0), 9)
    octave[octave_index] = 1.0
    out = np.concatenate(
        (
            np.asarray([1.0 - math.tanh(duration_ratio), onset_ratio, is_downbeat], dtype=np.float32),
            pc,
            octave,
        )
    )
    if out.shape != (len(BASE_FEATURE_NAMES),):
        raise AssertionError("base feature width differs from the pinned 25")
    return out


def _integer_resolution(values: Iterable[RationalTime]) -> int:
    resolution = 1
    for value in values:
        resolution = math.lcm(resolution, value.den)
    return resolution


def _ticks(value: RationalTime, resolution: int) -> int:
    return value.num * (resolution // value.den)


def _entry_labels(
    notes: tuple[Any, ...],
    source: TargetBundle,
    projection: DilemmadataCommonHarmonicProjection,
) -> tuple[dict[str, torch.Tensor], tuple[GraphEntry, ...]]:
    span_by_id = {span.annotation_id: span for span in source.alignment_spans}
    task_specs = (
        (COMMON_QUALITY_TASK, "quality", _QUALITY_TO_INDEX),
        (COMMON_INVERSION_TASK, "inversion", _INVERSION_TO_INDEX),
    )
    tensors: dict[str, torch.Tensor] = {}
    rows: list[GraphEntry] = []
    for common_task, name, vocabulary in task_specs:
        target = next(row for row in projection.targets if row.task_id == common_task)
        labels = torch.full((len(notes),), -1, dtype=torch.long)
        entries = torch.full((len(notes),), -1, dtype=torch.long)
        for entry_index, entry in enumerate(target.entries):
            available = entry.state in {"exact", "coarsened"}
            label = vocabulary[str(entry.common_value)] if available else -1
            rows.append(GraphEntry(name, entry_index, entry.entity_id, label, available))
            span = span_by_id.get(entry.entity_id)
            if span is None:
                if available:
                    raise AnalysisGNNGraphError(f"available {name} entry has no alignment span")
                continue
            for note_index, note in enumerate(notes):
                if _contains(span.start_qn, span.end_qn, note.onset_qn):
                    if entries[note_index] >= 0 and entries[note_index] != entry_index:
                        raise AnalysisGNNGraphError(f"overlapping {name} supervision spans")
                    entries[note_index] = entry_index
                    labels[note_index] = label
        tensors[name] = labels
        tensors[f"{name}_entry_index"] = entries
        tensors[f"{name}_mask"] = labels.ne(-1)
    return tensors, tuple(rows)


def build_analysisgnn_graph(
    piece: CanonicalPiece,
    source_targets: TargetBundle,
    projection: DilemmadataCommonHarmonicProjection,
    *,
    transposition: str = "P1",
) -> tuple["HeteroData", tuple[GraphEntry, ...]]:
    """Build one native note/beat/measure graph without target leakage."""

    if transposition not in TRANSPOSITIONS:
        raise AnalysisGNNGraphError(f"unsupported transposition {transposition!r}")
    if (
        piece.piece_id != source_targets.piece_id
        or piece.piece_id != projection.piece_id
        or piece.targets
        or piece.annotations
    ):
        raise AnalysisGNNGraphError("raw piece and target sidecars are not identity-aligned")
    if not piece.notes or not piece.bars or not piece.beats:
        raise AnalysisGNNGraphError("AnalysisGNN requires non-empty notes, bars, and beats")

    try:
        from graphmuse.utils.graph import create_score_graph
    except ImportError as exc:  # pragma: no cover - exercised by environment smoke
        raise RuntimeError(
            "pinned GraphMuse is required; run scripts/prepare_phase9eb1_environment.sh"
        ) from exc

    ordered = tuple(sorted(piece.notes, key=lambda note: (note.onset_qn, note.pitch, note.note_id)))
    timing = [piece.duration_qn]
    for note in ordered:
        timing.extend((note.onset_qn, note.duration_qn))
    resolution = _integer_resolution(timing)
    dtype = np.dtype([("onset_div", "i8"), ("duration_div", "i8"), ("pitch", "i8")])
    note_array = np.empty(len(ordered), dtype=dtype)
    features: list[np.ndarray] = []
    pitch_spelling: list[int] = []
    key_signature: list[int] = []
    for index, note in enumerate(ordered):
        step, alter, pitch = _transpose_spelling(
            note.spelling_step, note.spelling_alter, note.pitch, transposition
        )
        if not 0 <= pitch <= 127:
            raise AnalysisGNNGraphError("transposition moves a note outside MIDI [0, 127]")
        note_array[index] = (
            _ticks(note.onset_qn, resolution),
            _ticks(note.duration_qn, resolution),
            pitch,
        )
        features.append(_note_features(piece, note, pitch))
        pitch_spelling.append(_PITCH_TO_INDEX[_render_spelling(step, alter)])
        key_signature.append(
            _transpose_fifths(_active_key_fifths(piece, note.onset_qn), transposition) + 7
        )

    if (
        note_array["onset_div"].max(initial=0) > np.iinfo(np.int32).max
        or note_array["duration_div"].max(initial=0) > np.iinfo(np.int32).max
    ):
        raise AnalysisGNNGraphError("exact integer timing exceeds GraphMuse int32 topology")

    graph = create_score_graph(np.stack(features), note_array, sort=False, add_reverse=True)
    graph["note"].pitch_spelling = torch.tensor(pitch_spelling, dtype=torch.long)
    graph["note"].key_signature = torch.tensor(key_signature, dtype=torch.long)

    def add_hierarchy(kind: str, spans: tuple[Any, ...]) -> None:
        membership: list[int] = []
        for note in ordered:
            matches = [
                index
                for index, span in enumerate(spans)
                if _contains(span.start_qn, span.start_qn + span.duration_qn, note.onset_qn)
            ]
            if len(matches) != 1:
                raise AnalysisGNNGraphError(
                    f"note {note.note_id!r} has {len(matches)} exact {kind} memberships"
                )
            membership.append(matches[0])
        cluster = torch.tensor(membership, dtype=torch.long)
        note_index = torch.arange(len(ordered), dtype=torch.long)
        hierarchy_index = torch.arange(len(spans), dtype=torch.long)
        graph["note"][f"{kind}_cluster"] = cluster
        graph["note", "connects", kind].edge_index = torch.stack((note_index, cluster))
        graph[kind, "connects", "note"].edge_index = torch.stack((cluster, note_index))
        graph[kind, "next", kind].edge_index = torch.stack(
            (hierarchy_index[:-1], hierarchy_index[1:])
        )
        graph[kind].index = hierarchy_index
        graph[kind].x = torch.zeros((len(spans), len(BASE_FEATURE_NAMES)), dtype=torch.float32)
        graph[kind].x.index_add_(0, cluster, graph["note"].x)

    add_hierarchy("measure", piece.bars)
    add_hierarchy("beat", piece.beats)
    labels, entries = _entry_labels(ordered, source_targets, projection)
    for name, value in labels.items():
        graph["note"][name] = value
    graph.phase9eb1_schema_fingerprint = graph_schema_fingerprint()
    graph.record_id = source_targets.analysis_view_id
    graph.transposition = transposition
    if set(graph.edge_types) != set(EDGE_TYPES):
        raise AnalysisGNNGraphError("constructed graph edge schema differs from pinned contract")
    return graph, entries


def graph_fingerprint(graph: "HeteroData") -> str:
    """Fingerprint graph tensors, schema, source identity, and view."""

    digest = sha256()
    header = {
        "edge_types": sorted("|".join(edge) for edge in graph.edge_types),
        "node_types": sorted(graph.node_types),
        "record_id": graph.record_id,
        "schema_fingerprint": graph.phase9eb1_schema_fingerprint,
        "transposition": graph.transposition,
    }
    digest.update(canonical_json(header).encode("utf-8"))
    for node_type in sorted(graph.node_types):
        store = graph[node_type]
        for key in sorted(store.keys()):
            value = store[key]
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(f"{node_type}\0{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode())
                digest.update(tensor.numpy().tobytes())
    for edge_type in sorted(graph.edge_types):
        tensor = graph[edge_type].edge_index.detach().cpu().contiguous()
        digest.update(("|".join(edge_type) + "\0").encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


__all__ = [
    "AnalysisGNNGraphError",
    "GraphEntry",
    "build_analysisgnn_graph",
    "graph_fingerprint",
]
