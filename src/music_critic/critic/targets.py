"""Target-side verification and exact synthetic intervention support."""

from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
from typing import Mapping

from music_critic.data.schema import CanonicalPiece
from music_critic.data.validation import validate_or_raise

from music_critic.experiments.music_interventions import NotePatch
from .contracts import ASPECTS, LocalCandidate, local_candidates


@dataclass(frozen=True)
class VerifiedPatch:
    original_id: str
    variant_id: str
    work_id: str
    patches: tuple[NotePatch, ...]
    aspects: tuple[str, ...]
    kind: str
    exhaustive: bool = True
    support_ids: tuple[str, ...] = ()
    swap_boundaries: tuple[Fraction, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.exhaustive) is not bool:
            raise ValueError("exhaustive must be a boolean")
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.original_id, self.variant_id, self.work_id, self.kind)):
            raise ValueError("known piece/work/kind identity required")
        if not isinstance(self.patches, tuple) or any(not isinstance(p, NotePatch) for p in self.patches):
            raise ValueError("patches must be a tuple of NotePatch")
        if (not isinstance(self.aspects, tuple) or len(set(self.aspects)) != len(self.aspects)
                or set(self.aspects) - set((*ASPECTS, "arrangement"))):
            raise ValueError("unknown or duplicate aspect")
        if not isinstance(self.swap_boundaries, tuple) or any(not isinstance(t, Fraction) for t in self.swap_boundaries):
            raise ValueError("swap boundaries require exact Fraction values")


@dataclass(frozen=True)
class VerifiedDiff:
    changed_note_ids: tuple[str, ...]
    deleted_note_ids: tuple[str, ...]
    exhaustive: bool


@dataclass(frozen=True)
class LocalTargets:
    candidates: tuple[LocalCandidate, ...]
    values: tuple[float, ...]
    mask: tuple[bool, ...]

    def __post_init__(self):
        if not (len(self.candidates) == len(self.values) == len(self.mask)):
            raise ValueError("local target lengths disagree")
        if any(type(v) is not bool for v in self.mask):
            raise ValueError("local availability must be boolean")
        if any(v not in (0.0, 1.0) for v in self.values):
            raise ValueError("intervention presence targets must be binary")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate local target candidate")


@dataclass(frozen=True)
class PairTargets:
    work_id: str
    sample_ids: tuple[str, str]
    pair_mask: tuple[bool, ...]
    preference: tuple[int, ...]
    local_targets: tuple[Mapping[str, LocalTargets], Mapping[str, LocalTargets]]
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.work_id or len(self.sample_ids) != 2 or not all(self.sample_ids):
            raise ValueError("pair work/sample identities required")
        if len(self.pair_mask) != len(ASPECTS) or len(self.preference) != len(ASPECTS):
            raise ValueError("pair aspect count mismatch")
        if len(self.local_targets) != 2:
            raise ValueError("two local target views required")
        for mask, direction in zip(self.pair_mask, self.preference):
            if type(mask) is not bool or type(direction) is not int:
                raise ValueError("pair mask/direction types invalid")
            if direction not in ((-1, 1) if mask else (0,)):
                raise ValueError("unknown preference must be masked")

    def swapped(self) -> "PairTargets":
        return replace(self, sample_ids=self.sample_ids[::-1],
                       preference=tuple(-value for value in self.preference),
                       local_targets=self.local_targets[::-1])


class InterventionTargetBuilder:
    def build(
        self, original: CanonicalPiece, variant: CanonicalPiece, verified_patch: VerifiedPatch,
    ) -> PairTargets:
        from .features import geometry_from_piece

        proof = verified_patch
        diff = verify_actual_diff(original, variant, proof)
        if not diff.changed_note_ids and not diff.deleted_note_ids:
            raise ValueError("intervention requires an actual edit")
        swap_boundaries = _verified_swap_boundaries(original, variant, proof)
        before = {n.note_id: n for n in original.notes}
        after = {n.note_id: n for n in variant.notes}
        intervals: list[tuple[Fraction, Fraction]] = []
        attacks: set[tuple[str, Fraction]] = set()
        changed = set(diff.changed_note_ids)
        for patch in proof.patches:
            old, new = before[patch.note_id], after.get(patch.note_id)
            affected = (old,) if patch.delete else (new,)
            if patch.written_fields & {"onset_qn", "duration_qn"}:
                affected = (old, new)
                attacks.update((n.track_id, n.onset_qn.to_fraction()) for n in affected)
            if patch.delete and old.duration_qn.to_fraction() == 0:
                attacks.add((old.track_id, old.onset_qn.to_fraction()))
            for note in affected:
                start = note.onset_qn.to_fraction()
                end = (note.onset_qn + note.duration_qn).to_fraction()
                if start < end:
                    intervals.append((start, end))
        local_views = []
        for edited, piece in ((False, original), (True, variant)):
            view = {}
            for namespace, candidates in local_candidates(geometry_from_piece(piece)).items():
                aspect, candidate_type = namespace.split("/")
                values, masks = [], []
                for candidate in candidates:
                    if candidate_type in {"note", "event"}:
                        positive = bool(set(candidate.note_ids) & changed)
                        if candidate_type == "event":
                            positive |= any((after[key].track_id, candidate.start) in attacks
                                            for key in candidate.note_ids if key in after)
                    elif candidate_type in {"slice", "beat"}:
                        positive = any(candidate.start < end and start < candidate.end
                                       for start, end in intervals)
                    else:
                        positive = candidate.start in swap_boundaries
                    positive = edited and positive
                    # A grace note has event support, never a sounding-note or
                    # zero-length interval target. Its absence is not a negative.
                    eligible = candidate_type in {"event", "boundary"} or candidate.start < candidate.end
                    observed = (eligible and aspect in proof.aspects
                                and (positive or (proof.exhaustive and candidate.complete)))
                    values.append(float(positive) if observed else 0.0)
                    masks.append(observed)
                view[namespace] = LocalTargets(candidates, tuple(values), tuple(masks))
            local_views.append(view)
        mask = tuple(aspect in proof.aspects for aspect in ASPECTS)
        return PairTargets(proof.work_id, (original.piece_id, variant.piece_id), mask,
                           tuple(1 if available else 0 for available in mask), tuple(local_views),
                           {**proof.provenance, "quality_preference": "unreviewed",
                            "weak_direction": "original_preferred_assumption",
                            "exhaustive_diff": diff.exhaustive,
                            "changed_note_ids": diff.changed_note_ids,
                            "deleted_note_ids": diff.deleted_note_ids})


def _verified_swap_boundaries(original, variant, proof) -> frozenset[Fraction]:
    """A named boundary cannot certify an arbitrary time edit as a bar swap."""
    if proof.kind != "bar_order_swap":
        if proof.swap_boundaries or "structure" in proof.aspects:
            raise ValueError("structure targets require a verified bar swap")
        return frozenset()
    if len(proof.swap_boundaries) != 3:
        raise ValueError("bar swap requires entrance, join and exit")
    start, middle, end = proof.swap_boundaries
    length = middle - start
    bars = [bar for bar in original.bars if bar.start_qn.to_fraction() in (start, middle)]
    if (length <= 0 or end - middle != length or len(bars) != 2
            or any(bar.duration_qn.to_fraction() != length or bar.is_incomplete or bar.is_pickup for bar in bars)
            or bars[0].meter_event_id != bars[1].meter_event_id
            or bars[0].metric_offset_qn != bars[1].metric_offset_qn):
        raise ValueError("bar swap requires adjacent complete equal metric bars")
    before = {n.note_id: n for n in original.notes}
    after = {n.note_id: n for n in variant.notes}
    affected = {n.note_id for n in original.notes if start <= n.onset_qn.to_fraction() < end}
    if not affected or affected != {p.note_id for p in proof.patches}:
        raise ValueError("bar swap must permute all notes in both bars")
    for note in original.notes:
        onset = note.onset_qn.to_fraction()
        release = (note.onset_qn + note.duration_qn).to_fraction()
        if any(onset < boundary < release for boundary in proof.swap_boundaries):
            raise ValueError("bar swap crosses a sustained note")
    for patch in proof.patches:
        onset = before[patch.note_id].onset_qn.to_fraction()
        expected = onset + (length if onset < middle else -length)
        if patch.written_fields != {"onset_qn"} or after[patch.note_id].onset_qn.to_fraction() != expected:
            raise ValueError("bar swap is not the declared permutation")
    def pattern(lo, hi):
        return sorted((n.track_id, n.pitch, n.onset_qn.to_fraction() - lo, n.duration_qn.to_fraction(),
                       n.velocity, str(n.staff), str(n.voice)) for n in original.notes
                      if lo <= n.onset_qn.to_fraction() < hi)
    if pattern(start, middle) == pattern(middle, end):
        raise ValueError("bar swap has identical musical contents")
    return frozenset(proof.swap_boundaries)


def verify_actual_diff(
    original: CanonicalPiece, variant: CanonicalPiece, proof: VerifiedPatch,
) -> VerifiedDiff:
    """Verify exact edits independently of the generator's eligibility IDs."""
    validate_or_raise(original)
    validate_or_raise(variant)
    if (original.piece_id, variant.piece_id) != (proof.original_id, proof.variant_id):
        raise ValueError("patch piece identity mismatch")
    for name in ("duration_qn", "tracks", "bars", "beats", "meter_events",
                 "tempo_events", "key_signature_events", "source_group_id", "split"):
        if getattr(original, name) != getattr(variant, name):
            raise ValueError("unverified raw piece context change")
    before = {note.note_id: note for note in original.notes}
    after = {note.note_id: note for note in variant.notes}
    patches = {patch.note_id: patch for patch in proof.patches}
    if len(patches) != len(proof.patches) or patches.keys() - before.keys():
        raise ValueError("duplicate or unknown patch note")
    if after.keys() - before.keys():
        raise ValueError("added notes are not supported by note patches")
    changed, deleted = [], []
    for key, old in before.items():
        new, patch = after.get(key), patches.get(key)
        if patch is None:
            if proof.exhaustive and new != old:
                raise ValueError("unreported note difference in exhaustive patch")
            continue
        if patch.delete:
            if new is not None:
                raise ValueError("declared deletion is absent")
            deleted.append(key)
            continue
        if new is None:
            raise ValueError("undeclared deletion of edited note")
        written = patch.written_fields
        if all(getattr(old, name) == getattr(patch, name) for name in written):
            raise ValueError("patch has no actual edit")
        allowed = set(written) | {"provenance_id"}
        if "pitch" in written:
            allowed.update(("spelling_step", "spelling_alter"))
            if new.spelling_step is not None or new.spelling_alter is not None:
                raise ValueError("stale spelling after pitch edit")
        if written & {"onset_qn", "duration_qn"}:
            timing = ("source_onset_ticks", "source_duration_ticks",
                      "source_onset_seconds", "source_duration_seconds")
            allowed.update(timing)
            if any(getattr(new, name) is not None for name in timing):
                raise ValueError("stale source timing after time edit")
        for field in fields(old):
            name = field.name
            expected = getattr(patch, name) if name in written else getattr(old, name)
            if (name in written or name not in allowed) and getattr(new, name) != expected:
                raise ValueError("materialized note does not match verified patch")
        changed.append(key)
    if variant.targets or any(a.layer == "target_alignment" for a in variant.annotations):
        raise ValueError("variant retains stale supervised targets")
    return VerifiedDiff(tuple(sorted(changed)), tuple(sorted(deleted)), proof.exhaustive)
