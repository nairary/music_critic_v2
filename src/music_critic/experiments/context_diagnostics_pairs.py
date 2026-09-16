"""Bounded raw-symbolic context probes, without transferred theory labels.

These fixtures establish mechanical contrasts, not correct musical analyses or
preferences. ``hypothesis_only`` and ``not_reviewed`` are deliberate: no human
musician has reviewed this suite. Track continuity is not a gold voice label.
The module uses only the standard library and the existing canonical contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json

from music_critic.data import (
    SCHEMA_VERSION,
    CanonicalBar,
    CanonicalBeat,
    CanonicalNote,
    CanonicalPiece,
    CanonicalTrack,
    MeterEvent,
    PieceMetadata,
    ProvenanceRecord,
    RationalTime,
    TempoEvent,
    piece_to_dict,
    validate_or_raise,
    validate_piece,
)


CONTEXT_PAIR_SUITE_VERSION = "1.0.0"
_ZERO = RationalTime(0)
_ONE = RationalTime(1)
_COMMON_INVARIANTS = ("equal_duration", "equal_raw_onset_boundaries", "equal_focus_sounding_notes")


@dataclass(frozen=True, slots=True)
class RawContextCandidate:
    """An exact note-onset span; no annotation or target supplies a boundary."""

    piece_id: str
    candidate_id: str
    start_qn: RationalTime
    end_qn: RationalTime
    onset_note_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContextDiagnosticPair:
    pair_id: str
    family: str
    left: CanonicalPiece
    right: CanonicalPiece
    focus_start_qn: RationalTime
    focus_end_qn: RationalTime
    declared_invariants: tuple[str, ...]
    hypothesis: str
    limitations: tuple[str, ...]
    interpretation_status: str = field(default="hypothesis_only", init=False)
    musician_review_status: str = field(default="not_reviewed", init=False)
    preferred_side: None = field(default=None, init=False)
    functional_label_transfer_allowed: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class ContextPairValidation:
    checks: tuple[tuple[str, bool], ...]

    @property
    def mechanically_valid(self) -> bool:
        return bool(self.checks) and all(passed for _, passed in self.checks)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)


@dataclass(frozen=True, slots=True)
class ContextCrop:
    piece: CanonicalPiece
    source_piece_id: str
    source_start_qn: RationalTime
    source_end_qn: RationalTime
    intervention: str = field(default="bar_aligned_raw_context_crop", init=False)
    distribution_shift_warning: str = field(
        default=("Cropping changes available context, attacks of clipped notes and "
                 "piece/track summaries; this is an intervention on a frozen model, "
                 "not an independently trained local-only baseline."),
        init=False,
    )


def _guard_partition(piece: CanonicalPiece) -> None:
    if piece.split is not None and piece.split.lower() not in {"train", "validation"}:
        raise ValueError("context diagnostics accept only TRAIN, VALIDATION or synthetic unsplit pieces")
    if piece.split is None and piece.metadata.source_format != "synthetic":
        raise ValueError("unsplit context diagnostics require a synthetic piece")


def raw_context_candidates(piece: CanonicalPiece) -> tuple[RawContextCandidate, ...]:
    """Return sounding-note onset candidates, ignoring annotations and targets.

    Endpoints are the next raw note onset or piece end, not a claim that the
    sounding set is constant throughout the span. Grace-only onsets are excluded.
    Corpus membership authorization must additionally be enforced by the harness.
    """
    _guard_partition(piece)
    onsets: dict[RationalTime, list[str]] = {}
    for note in piece.notes:
        if note.duration_qn > _ZERO and not note.is_percussion:
            onsets.setdefault(note.onset_qn, []).append(note.note_id)
    ordered = sorted(onsets)
    return tuple(
        RawContextCandidate(
            piece.piece_id, f"onset:{index:04d}", start,
            ordered[index + 1] if index + 1 < len(ordered) else piece.duration_qn,
            tuple(sorted(onsets[start])),
        )
        for index, start in enumerate(ordered)
    )


def sounding_note_signature(
    piece: CanonicalPiece,
    start_qn: RationalTime | None = None,
    end_qn: RationalTime | None = None,
) -> tuple[tuple[object, ...], ...]:
    """Exact audible event multiset, independent of note and track identifiers.

    Notes are clipped to an optional half-open interval. Multiplicity, MIDI pitch,
    timing, velocity, channel, program and percussion identity remain observable.
    This is an event identity check, not a synthesized-audio equivalence claim.
    """
    _guard_partition(piece)
    start = _ZERO if start_qn is None else start_qn
    end = piece.duration_qn if end_qn is None else end_qn
    if not isinstance(start, RationalTime) or not isinstance(end, RationalTime):
        raise TypeError("context intervals require exact RationalTime endpoints")
    if not _ZERO <= start < end <= piece.duration_qn:
        raise ValueError("context interval must be nonempty and inside the piece")
    rows = []
    for note in piece.notes:
        left, right = max(start, note.onset_qn), min(end, note.onset_qn + note.duration_qn)
        if left < right:
            # Optional integer values use an explicit missing sentinel for sorting.
            rows.append((
                left, right - left, note.pitch,
                -1 if note.velocity is None else note.velocity,
                -1 if note.channel is None else note.channel,
                -1 if note.program is None else note.program,
                note.is_percussion,
            ))
    return tuple(sorted(rows))


def _motion_signature(piece: CanonicalPiece) -> tuple[tuple[int, ...], ...]:
    trajectories = []
    for track in piece.tracks:
        notes = sorted((n for n in piece.notes if n.track_id == track.track_id),
                       key=lambda n: (n.onset_qn, n.pitch, n.note_id))
        trajectories.append(tuple(b.pitch - a.pitch for a, b in zip(notes, notes[1:])))
    return tuple(trajectories)


def validate_context_pair(pair: ContextDiagnosticPair) -> ContextPairValidation:
    """Recompute declared invariants; mechanical validity implies no quality rank."""
    left, right = pair.left, pair.right
    _guard_partition(left)
    _guard_partition(right)
    focus = pair.focus_start_qn, pair.focus_end_qn
    valid_interval = all(
        isinstance(v, RationalTime) for v in focus
    ) and _ZERO <= focus[0] < focus[1] <= min(left.duration_qn, right.duration_qn)
    checks = [
        ("canonical_left", validate_piece(left).is_valid),
        ("canonical_right", validate_piece(right).is_valid),
        ("distinct_piece_ids", left.piece_id != right.piece_id),
        ("same_source_group", left.source_group_id == right.source_group_id),
        ("same_dataset_and_partition", (left.dataset_name, left.split) == (right.dataset_name, right.split)),
        ("raw_only_no_annotations_or_targets", not (left.annotations or left.targets or right.annotations or right.targets)),
        ("valid_focus_interval", valid_interval),
        ("required_invariants_declared", set(_COMMON_INVARIANTS) <= set(pair.declared_invariants)),
        ("no_transferred_functional_labels", not pair.functional_label_transfer_allowed),
        ("no_automatic_preference", pair.preferred_side is None),
    ]
    invariant_values = {
        "equal_duration": left.duration_qn == right.duration_qn,
        "equal_raw_onset_boundaries": tuple((c.start_qn, c.end_qn) for c in raw_context_candidates(left))
        == tuple((c.start_qn, c.end_qn) for c in raw_context_candidates(right)),
        "equal_focus_sounding_notes": valid_interval and sounding_note_signature(left, *focus)
        == sounding_note_signature(right, *focus),
        "equal_all_sounding_notes": sounding_note_signature(left) == sounding_note_signature(right),
        "changed_context_notes": sounding_note_signature(left) != sounding_note_signature(right),
        "changed_trackwise_pitch_motion": _motion_signature(left) != _motion_signature(right),
        "changed_track_ids": tuple(t.track_id for t in left.tracks) != tuple(t.track_id for t in right.tracks),
        "changed_track_count": len(left.tracks) != len(right.tracks),
    }
    for name in pair.declared_invariants:
        if name not in invariant_values:
            raise ValueError(f"unknown context-pair invariant: {name}")
        checks.append((name, invariant_values[name]))
    for side, piece in (("left", left), ("right", right)):
        checks.append((f"focus_is_raw_beat_{side}", sum(
            beat.start_qn == focus[0] and beat.start_qn + beat.duration_qn == focus[1]
            for beat in piece.beats
        ) == 1))
    return ContextPairValidation(tuple(checks))


def _make_piece(pair_id: str, side: str, chords: tuple[tuple[int, ...], ...]) -> CanonicalPiece:
    provenance_id = "prov:context-pair"
    tracks = tuple(CanonicalTrack(
        f"track:{index:03d}", index, None, None, 0, 0, False, provenance_id,
    ) for index in range(len(chords[0])))
    notes = tuple(CanonicalNote(
        note_id=f"note:{beat:03d}-{track:03d}", track_id=tracks[track].track_id,
        pitch=pitch, onset_qn=RationalTime(beat), duration_qn=_ONE,
        velocity=80, channel=0, program=0, is_percussion=False, is_grace=False,
        spelling_step=None, spelling_alter=None, staff=None, voice=None,
        articulations=None, dynamic=None, source_onset_ticks=None,
        source_duration_ticks=None, source_onset_seconds=None,
        source_duration_seconds=None, provenance_id=provenance_id,
    ) for beat, chord in enumerate(chords) for track, pitch in enumerate(chord))
    piece = CanonicalPiece(
        schema_version=SCHEMA_VERSION, piece_id=f"piece:{pair_id}-{side}",
        dataset_name="context-diagnostics-synthetic", source_group_id=pair_id,
        split=None, source_path=None, source_resolution=None, duration_qn=RationalTime(4),
        metadata=PieceMetadata("synthetic", None, None, None, None, None, None, None, None),
        tracks=tracks, notes=notes,
        bars=(CanonicalBar("bar:000", 0, _ZERO, RationalTime(4), "meter:000", _ZERO,
                           False, False, None, provenance_id),),
        beats=tuple(CanonicalBeat(
            f"beat:{index:03d}", "bar:000", "meter:000", index,
            RationalTime(index), _ONE, RationalTime(index), index == 0, None, provenance_id,
        ) for index in range(4)),
        tempo_events=(TempoEvent("tempo:000", _ZERO, 500000, provenance_id),),
        meter_events=(MeterEvent("meter:000", _ZERO, 4, 4, provenance_id),),
        key_signature_events=(), annotations=(), targets=(),
        provenance=(ProvenanceRecord(
            provenance_id, "synthetic", "context_diagnostics_pairs", pair_id, None,
            CONTEXT_PAIR_SUITE_VERSION, None, None, (),
            (("review_status", "not_reviewed"),),
        ),),
        quality_flags=(),
    )
    validate_or_raise(piece)
    return piece


def _retrack(piece: CanonicalPiece, mapping: dict[str, str], indices: dict[str, int]) -> CanonicalPiece:
    tracks = tuple(sorted((replace(
        t, track_id=mapping[t.track_id], source_track_index=indices[mapping[t.track_id]],
    ) for t in piece.tracks), key=lambda t: (t.source_track_index, t.track_id)))
    order = {t.track_id: i for i, t in enumerate(tracks)}
    notes = tuple(sorted((replace(n, track_id=mapping[n.track_id]) for n in piece.notes),
                         key=lambda n: (n.onset_qn, order[n.track_id], n.pitch, n.duration_qn, n.note_id)))
    return replace(piece, tracks=tracks, notes=notes)


def build_context_pair_suite() -> tuple[ContextDiagnosticPair, ...]:
    """Return five fixed pairs covering four families; no RNG, corpus or labels."""
    common = _COMMON_INVARIANTS
    specs = (
        (
            "harmonic-surroundings", "harmonic_surroundings",
            ((48, 52, 55, 60), (55, 59, 62, 65), (48, 52, 55, 60), (48, 52, 55, 60)),
            ((45, 48, 52, 57), (55, 59, 62, 65), (45, 48, 52, 57), (45, 48, 52, 57)),
            "The same G-B-D-F sonority may receive different contextual analyses in these surroundings.",
            ("No local-key, function or preferred continuation label is assigned.",),
        ),
        (
            "dissonance-context", "dissonance_preparation_resolution",
            ((48, 55, 60, 65), (48, 55, 60, 65), (48, 55, 60, 64), (48, 55, 60, 64)),
            ((48, 55, 60, 69), (48, 55, 60, 65), (48, 55, 60, 69), (48, 55, 60, 69)),
            "Rearticulated F followed by stepwise descent may be treated differently from leap approach/departure.",
            ("The F is reattacked, not tied; this does not establish a suspension label.",
             "Preparation and resolution are varied jointly, not isolated causal effects."),
        ),
        (
            "voice-motion", "vertical_sonority_voice_motion",
            ((48, 67), (50, 65), (52, 64), (53, 62)),
            ((55, 60), (50, 65), (45, 69), (43, 72)),
            "The preserved focal vertical may be interpreted differently under altered approach/departure motion.",
            ("Only the focal vertical is invariant; surrounding pitches change.",
             "Tracks are constructed raw streams, not annotated or inferred perceptual voices."),
        ),
    )
    pairs = [ContextDiagnosticPair(
        name, family, _make_piece(name, "left", left), _make_piece(name, "right", right),
        _ONE, RationalTime(2), common + ("changed_context_notes", "changed_trackwise_pitch_motion"),
        hypothesis, limitations,
    ) for name, family, left, right, hypothesis, limitations in specs]
    chords = ((48, 52, 55, 60), (55, 59, 62, 65), (48, 52, 55, 60), (48, 52, 55, 60))
    left = _make_piece("track-renumber", "left", chords)
    right = _make_piece("track-renumber", "right", chords)
    mapping = {t.track_id: f"track:{17 + i:03d}" for i, t in enumerate(right.tracks)}
    indices = {mapping[t.track_id]: 3 - i for i, t in enumerate(right.tracks)}
    right = _retrack(right, mapping, indices)
    pairs.append(ContextDiagnosticPair(
        "track-renumber", "track_repartition", left, right, _ONE, RationalTime(2),
        common + ("equal_all_sounding_notes", "changed_track_ids"),
        "Renumbering and reordering tracks should test identifier sensitivity with identical sounding events.",
        ("Invariance of model predictions must be measured, not assumed from mechanical equivalence.",),
    ))
    left = _make_piece("track-split", "left", chords)
    right = _make_piece("track-split", "right", chords)
    left = replace(left, tracks=left.tracks[:1], notes=tuple(sorted(
        (replace(n, track_id="track:000") for n in left.notes),
        key=lambda n: (n.onset_qn, n.pitch, n.duration_qn, n.note_id),
    )))
    pairs.append(ContextDiagnosticPair(
        "track-split", "track_repartition", left, right, _ONE, RationalTime(2),
        common + ("equal_all_sounding_notes", "changed_track_count"),
        "Partitioning a polyphonic track tests reliance on raw grouping while sounding events remain identical.",
        ("Track topology and pooled track statistics change; this is not a quality degradation.",),
    ))
    for pair in pairs:
        result = validate_context_pair(pair)
        if not result.mechanically_valid:
            raise ValueError(f"invalid generated pair {pair.pair_id}: {result.failed_checks}")
    return tuple(pairs)


def context_pair_to_dict(pair: ContextDiagnosticPair) -> dict[str, object]:
    """Serialize mechanical evidence and hypotheses outside model inputs."""
    result = validate_context_pair(pair)
    return {
        "schema": f"ContextDiagnosticPair@{CONTEXT_PAIR_SUITE_VERSION}",
        "pair_id": pair.pair_id, "family": pair.family,
        "left": piece_to_dict(pair.left), "right": piece_to_dict(pair.right),
        "focus_start_qn": {"num": pair.focus_start_qn.num, "den": pair.focus_start_qn.den},
        "focus_end_qn": {"num": pair.focus_end_qn.num, "den": pair.focus_end_qn.den},
        "declared_invariants": list(pair.declared_invariants),
        "mechanically_valid": result.mechanically_valid,
        "checks": dict(result.checks), "hypothesis": pair.hypothesis,
        "limitations": list(pair.limitations), "interpretation_status": pair.interpretation_status,
        "musician_review_status": pair.musician_review_status,
        "preferred_side": pair.preferred_side,
        "functional_label_transfer_allowed": pair.functional_label_transfer_allowed,
    }


def context_pair_fingerprint(pair: ContextDiagnosticPair) -> str:
    payload = json.dumps(context_pair_to_dict(pair), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def crop_raw_context(piece: CanonicalPiece, start_qn: RationalTime, end_qn: RationalTime) -> ContextCrop:
    """Crop complete raw bars, clipping sounding notes with exact rational overlap.

    Partial bars are rejected. Retained bars/beats are rebased, active raw meter,
    tempo and key events are carried to zero, and annotations/targets are dropped.
    Parent dataset/group/partition remain fixed. This helper neither loads a
    corpus nor establishes membership authorization for a real-data experiment.
    """
    _guard_partition(piece)
    if not isinstance(start_qn, RationalTime) or not isinstance(end_qn, RationalTime):
        raise TypeError("crop endpoints require RationalTime")
    if not _ZERO <= start_qn < end_qn <= piece.duration_qn:
        raise ValueError("crop interval must be nonempty and inside the piece")
    # Strip target sidecars before canonical validation: they are never inputs.
    raw = replace(piece, annotations=(), targets=(), quality_flags=())
    validate_or_raise(raw)
    if start_qn not in {b.start_qn for b in raw.bars} or end_qn not in {
        b.start_qn + b.duration_qn for b in raw.bars
    }:
        raise ValueError("context crop requires complete raw bar boundaries")
    suffix = f"crop-{start_qn.num}-{start_qn.den}-{end_qn.num}-{end_qn.den}"
    provenance_id = f"prov:{suffix}"
    if any(p.provenance_id == provenance_id for p in raw.provenance):
        raise ValueError("repeated crop provenance identity; use the original source piece")
    bars = tuple(replace(b, start_qn=b.start_qn - start_qn, index=i) for i, b in enumerate(
        b for b in raw.bars if start_qn <= b.start_qn < end_qn
    ))
    bar_ids = {b.bar_id for b in bars}
    beats = tuple(replace(b, start_qn=b.start_qn - start_qn) for b in raw.beats if b.bar_id in bar_ids)
    notes = []
    for n in raw.notes:
        left, right = max(start_qn, n.onset_qn), min(end_qn, n.onset_qn + n.duration_qn)
        if left < right or (n.is_grace and start_qn <= n.onset_qn < end_qn):
            notes.append(replace(
                n, onset_qn=left - start_qn, duration_qn=right - left,
                source_onset_ticks=None, source_duration_ticks=None,
                source_onset_seconds=None, source_duration_seconds=None,
                provenance_id=provenance_id,
            ))
    used_tracks = {n.track_id for n in notes}
    tracks = tuple(t for t in raw.tracks if t.track_id in used_tracks)
    order = {t.track_id: i for i, t in enumerate(tracks)}
    notes.sort(key=lambda n: (n.onset_qn, order[n.track_id], n.pitch, n.duration_qn, n.note_id))

    def cropped_events(events: tuple) -> tuple:
        active = tuple(e for e in events if e.onset_qn <= start_qn)
        interior = tuple(e for e in events if start_qn < e.onset_qn < end_qn)
        return ((replace(active[-1], onset_qn=_ZERO),) if active else ()) + tuple(
            replace(e, onset_qn=e.onset_qn - start_qn) for e in interior
        )

    provenance = ProvenanceRecord(
        provenance_id, "derivation", "context_diagnostics_pairs.crop_raw_context", piece.piece_id,
        None, CONTEXT_PAIR_SUITE_VERSION, None, None,
        tuple(p.provenance_id for p in raw.provenance),
        (("distribution_shift", True), ("source_end_qn", f"{end_qn.num}/{end_qn.den}"),
         ("source_start_qn", f"{start_qn.num}/{start_qn.den}")),
    )
    cropped = replace(
        raw, piece_id=f"{piece.piece_id}-{suffix}", source_path=None, source_resolution=None,
        duration_qn=end_qn - start_qn, tracks=tracks, notes=tuple(notes), bars=bars, beats=beats,
        meter_events=cropped_events(raw.meter_events), tempo_events=cropped_events(raw.tempo_events),
        key_signature_events=cropped_events(raw.key_signature_events), provenance=raw.provenance + (provenance,),
    )
    validate_or_raise(cropped)
    return ContextCrop(cropped, piece.piece_id, start_qn, end_qn)
