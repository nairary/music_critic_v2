"""Exact, fail-closed TAVERN Humdrum notes and native phrase-context targets.

The parser performs no I/O. Callers authenticate bytes and admit work families
before opening music. Native ``**function`` belongs to a phrase-model span,
not to each sounding note. Score-only parsing owns canonical raw inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
from pathlib import PurePosixPath
import re

from music_critic.data import (
    CanonicalBar, CanonicalBeat, CanonicalNote, CanonicalPiece, CanonicalTrack,
    MeterEvent, PieceMetadata, ProvenanceRecord, QualityFlag, RationalTime,
    TempoEvent, validate_or_raise,
)

VERSION = "tavern-exact@2"
TARGET_REVISION = "tavern-annotator-metadata-only@1"


class TavernError(ValueError):
    """Unverifiable notation or unsupported source construct."""


def require(ok, reason):
    if not ok:
        raise TavernError(reason)


def rational(value):
    value = Fraction(value)
    return {"num": value.numerator, "den": value.denominator}


def rt(value):
    return RationalTime.from_fraction(Fraction(value))


def native_field(value, evidence, *, origin="source_native", reason="source_missing"):
    return {"value": value, "available": value is not None,
            "origin": origin if value is not None else "unknown", "confidence": None,
            "evidence": evidence, "unknown_reason": None if value is not None else reason}


_RECIP = re.compile(r"(0+|[1-9][0-9]*)(?:%([1-9][0-9]*))?(\.*)")
_PITCH = re.compile(r"([a-g]+|[A-G]+)([#n-]*)")


def duration(token):
    match = _RECIP.search(token)
    require(match is not None, f"missing reciprocal duration: {token}")
    n, multiplier, dots = match.groups()
    value = Fraction(4 * (2 ** len(n))) if set(n) == {"0"} else Fraction(4, int(n))
    if multiplier:
        value *= int(multiplier)
    value *= Fraction(2 ** (len(dots) + 1) - 1, 2 ** len(dots))
    return value, match


@dataclass
class Note:
    onset: Fraction
    duration: Fraction
    pitch: int
    step: str
    alter: int
    track: int
    voice: str
    grace: bool
    evidence: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Span:
    start: Fraction
    end: Fraction
    label: str
    key: str | None
    evidence: dict


@dataclass
class Parsed:
    notes: list[Note]
    functions: list[Span]
    harmonies: list[Span]
    meters: list[tuple[Fraction, int, int]]
    barlines: list[tuple[Fraction, str]]
    duration: Fraction
    tracks: int
    metadata: dict
    source_path: str
    source_sha256: str


@dataclass
class _Spine:
    kind: str
    root: int
    branch: str
    remaining: Fraction = Fraction(0)
    key: str | None = None


def parse_joined(text: str, *, source_path: str, source_sha256: str) -> Parsed:
    """Parse rhythmic TAVERN spines; unsupported or inconsistent files fail.

    Spines split/merge/exchange with exact clocks. Ties retain one attack and
    every contributing token. Repeats stay in written score order, explicitly
    not a realized performance. Grace rows have zero duration.
    """
    require(re.fullmatch(r"[0-9a-f]{64}", source_sha256) is not None, "invalid source SHA")
    spines, notes, functions, harmonies, meters, bars = [], [], [], [], [], []
    metadata, ties = {}, {}
    now, track_count, terminated = Fraction(0), 0, False
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("!!"):
            if line.startswith("!!!") and ":" in line:
                key, val = line[3:].split(":", 1)
                metadata[key] = val.strip()
            continue
        tokens = line.split("\t")
        if not spines:
            require(not terminated and all(t.startswith("**") for t in tokens),
                    f"line {line_number}: expected exclusive spine interpretations")
            for token in tokens:
                require(token in ("**kern", "**harm", "**function"), f"unsupported spine {token}")
                root = track_count if token == "**kern" else -1
                track_count += int(token == "**kern")
                spines.append(_Spine(token, root, str(root)))
            continue
        require(len(tokens) == len(spines), f"line {line_number}: spine count mismatch")
        if all(t.startswith("!") for t in tokens):
            continue
        if all(t.startswith("=") for t in tokens):
            numbers = {m.group(1) for t in tokens if (m := re.match(r"=+([0-9]+[a-z]?)", t))}
            require(len(numbers) <= 1, f"line {line_number}: inconsistent bar numbers")
            bars.append((now, next(t for s, t in zip(spines, tokens) if s.kind == "**kern")))
            continue
        if all(t.startswith("*") for t in tokens):
            paths = {"*^", "*v", "*x", "*-", "*+"}
            if any(t in paths for t in tokens):
                require("*+" not in tokens, "unsupported added spine")
                exchanges = [i for i, t in enumerate(tokens) if t == "*x"]
                require(len(exchanges) in (0, 2), "unsupported exchange cardinality")
                if exchanges:
                    a, b = exchanges
                    spines[a], spines[b] = spines[b], spines[a]
                new, i = [], 0
                while i < len(tokens):
                    token, spine = tokens[i], spines[i]
                    if token == "*^":
                        left, right = deepcopy(spine), deepcopy(spine)
                        left.branch += "a"; right.branch += "b"
                        new.extend((left, right))
                    elif token == "*v":
                        siblings = [spine]
                        while i + 1 < len(tokens) and tokens[i + 1] == "*v":
                            i += 1; siblings.append(spines[i])
                        require(len(siblings) >= 2, "isolated spine join")
                        require(len({(s.kind, s.root, s.remaining, s.key) for s in siblings}) == 1,
                                "ambiguous spine join")
                        joined = deepcopy(spine)
                        joined.branch = str(joined.root)
                        new.append(joined)
                    elif token != "*-":
                        new.append(spine)
                    else:
                        require(spine.remaining == 0, "termination before duration complete")
                    i += 1
                spines = new
                terminated = not spines
                continue
            row_meters = set()
            for spine, token in zip(spines, tokens):
                if re.fullmatch(r"\*[A-Ga-g][#-]*:", token):
                    spine.key = token[1:-1]
                match = re.fullmatch(r"\*M([0-9]+)/([0-9]+)", token)
                if match and spine.kind == "**kern":
                    row_meters.add(tuple(map(int, match.groups())))
                require(not any(x in token.lower() for x in ("tb", "rscale", "omet", "met(")),
                        f"unsupported time interpretation {token}")
            require(len(row_meters) <= 1, "conflicting meters")
            if row_meters:
                n, d = row_meters.pop()
                require(n > 0 and d > 0 and d & (d - 1) == 0, "unsupported meter")
                if not meters or meters[-1][1:] != (n, d):
                    meters.append((now, n, d))
            continue
        require(not any(t.startswith(("*", "!", "=")) for t in tokens), "mixed record types")
        grace_row = any(("q" in t or "Q" in t) for s, t in zip(spines, tokens) if s.kind == "**kern")
        for column, (spine, token) in enumerate(zip(spines, tokens), 1):
            if token == ".":
                continue
            require(spine.remaining == 0 or grace_row, f"line {line_number}: overlapping spine events")
            evidence = {"source_path": source_path, "source_sha256": source_sha256,
                        "line": line_number, "column": column, "token": token}
            if spine.kind != "**kern":
                d, match = duration(token)
                require(not grace_row, "analysis event on grace row")
                require(match.start() == 0, "analysis reciprocal must prefix label")
                label = token[match.end():]
                require(bool(label), "empty analytical token")
                span = Span(now, now + d, label, spine.key, evidence)
                (functions if spine.kind == "**function" else harmonies).append(span)
                spine.remaining = d
                continue
            subtokens = token.split(" ")
            durations = []
            for subtoken in subtokens:
                require(bool(subtoken), "empty chord member")
                is_grace = "q" in subtoken or "Q" in subtoken
                require(is_grace == grace_row, "mixed grace and timed events")
                sd = Fraction(0) if is_grace else duration(subtoken)[0]
                durations.append(sd)
                if "r" in subtoken:
                    continue
                pitch_match = _PITCH.search(subtoken)
                require(pitch_match is not None, f"unsupported note {subtoken}")
                letters, accidental = pitch_match.groups()
                require(len(set(letters)) == 1, "invalid kern pitch spelling")
                require(not ("#" in accidental and "-" in accidental) and
                        not ("n" in accidental and accidental != "n"), "invalid accidental")
                step = letters[0].upper()
                alter = accidental.count("#") - accidental.count("-")
                octave = (3 + len(letters)) if letters.islower() else (4 - len(letters))
                pitch = (octave + 1) * 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step] + alter
                require(0 <= pitch <= 127, "pitch outside MIDI")
                tie_key = (spine.root, step, alter, octave)
                source = {**evidence, "subtoken": subtoken, "onset_qn": rational(now), "duration_qn": rational(sd)}
                if "]" in subtoken or "_" in subtoken:
                    require(tie_key in ties, "orphan tie continuation")
                    existing = ties[tie_key]
                    require(existing.onset + existing.duration == now, "noncontiguous tie")
                    existing.duration += sd
                    existing.evidence.append(source)
                    if "]" in subtoken:
                        del ties[tie_key]
                else:
                    require(tie_key not in ties, "ambiguous overlapping tied pitch")
                    note = Note(now, sd, pitch, step, alter, spine.root, spine.branch, is_grace, [source])
                    notes.append(note)
                    if "[" in subtoken:
                        require(not is_grace, "grace tie unsupported")
                        ties[tie_key] = note
            require(len(set(durations)) == 1, "unequal chord member durations")
            if not grace_row:
                spine.remaining = durations[0]
        if not grace_row:
            positive = [s.remaining for s in spines if s.remaining > 0]
            require(bool(positive), "data row has no duration")
            step = min(positive)
            now += step
            for spine in spines:
                spine.remaining = max(Fraction(0), spine.remaining - step)
    require(terminated, "unterminated spines")
    require(not ties, "unclosed ties")
    require(bool(notes), "no pitched notes")
    require(meters and meters[0][0] == 0, "initial meter required")
    require(max(n.onset + n.duration for n in notes) <= now, "notes exceed score clock")
    return Parsed(notes, functions, harmonies, meters, bars, now, track_count,
                  metadata, source_path, source_sha256)


def raw_signature(parsed: Parsed):
    return sorted((n.onset, n.duration, n.pitch, n.step, n.alter, n.track, n.grace) for n in parsed.notes)


def canonical_piece(parsed: Parsed, *, phrase_id: str, family_id: str) -> CanonicalPiece:
    """Create raw canonical notes only from score-only source bytes."""
    require(not parsed.harmonies and not parsed.functions, "score-only source required")
    meters = tuple(MeterEvent(f"meter:{i}", rt(t), n, d, "prov:source")
                   for i, (t, n, d) in enumerate(parsed.meters))
    bars, beats = [], []
    boundaries = sorted(set([Fraction(0), parsed.duration] + [t for t, _ in parsed.barlines]))
    for a, b in zip(boundaries, boundaries[1:]):
        if a == b:
            continue
        meter = max((m for m in meters if m.onset_qn.to_fraction() <= a), key=lambda m: m.onset_qn)
        require(not any(a < m.onset_qn.to_fraction() < b for m in meters), "meter change inside source bar")
        nominal = Fraction(4 * meter.numerator, meter.denominator)
        require(b - a <= nominal, "overfull source bar")
        pickup = a == 0 and b - a < nominal
        offset = nominal - (b - a) if pickup else Fraction(0)
        bar_id = f"bar:{len(bars):06d}"
        display = next((v.lstrip("=") for t, v in parsed.barlines if t == a), None)
        bars.append(CanonicalBar(bar_id, len(bars), rt(a), rt(b-a), meter.meter_event_id,
                                 rt(offset), pickup, b-a < nominal, display, "prov:source"))
        unit, cursor, position = Fraction(4, meter.denominator), a, offset
        while cursor < b:
            length = min(unit - position % unit, b - cursor)
            beats.append(CanonicalBeat(f"beat:{len(beats):08d}", bar_id, meter.meter_event_id,
                                        int(position // unit), rt(cursor), rt(length), rt(position),
                                        position == 0, None, "prov:source"))
            cursor += length; position += length
    ordered = ordered_notes(parsed)
    notes = tuple(CanonicalNote(
        f"note:{i:08d}", f"track:{n.track}", n.pitch, rt(n.onset), rt(n.duration),
        None, None, None, False, n.grace, n.step, n.alter, None, None,
        None, None, None, None, None, None, "prov:source",
    ) for i, (_, n) in enumerate(ordered))
    piece = CanonicalPiece(
        "2.0.0", f"piece:tavern-{phrase_id}", "tavern", family_id, "train", parsed.source_path,
        None, rt(parsed.duration), PieceMetadata("other", parsed.metadata.get("OTL"),
        (parsed.metadata["COM"],) if "COM" in parsed.metadata else None, "TAVERN", None,
        None, ("classical",), "CC BY-SA 4.0", None),
        tuple(CanonicalTrack(f"track:{i}", i, None, None, None, None, False, "prov:source")
              for i in range(parsed.tracks)), notes, tuple(bars), tuple(beats),
        (TempoEvent("tempo:default", rt(0), 500000, "prov:default"),), meters, (), (), (),
        (ProvenanceRecord("prov:default", "default", "music_critic.tavern_adapter", None,
         None, VERSION, None, None, (), (("tempo_bpm", 120),)),
         ProvenanceRecord("prov:source", "conversion", "music_critic.tavern_adapter", phrase_id,
         parsed.source_path, VERSION, parsed.source_sha256, None, (),
         (("exact_score_measure_boundaries_version", VERSION),
          ("repeat_expansion", False), ("time_basis", "written_score_order")))),
        (QualityFlag("tavern.default_tempo", "info", "120 BPM is an explicit rendering default, not a source performance tempo.", (), "prov:default"),),
    )
    validate_or_raise(piece)
    return piece


def ordered_notes(parsed):
    return sorted(enumerate(parsed.notes), key=lambda x: (
        x[1].onset, x[1].track, x[1].pitch, x[1].duration, x[0]))


def source_note_bindings(piece, parsed):
    """Retain every original token, including tied segments, for each raw ID."""
    ordered = ordered_notes(parsed)
    require(len(ordered) == len(piece.notes), "source-note binding length mismatch")
    return [{"note_id": note.note_id, "source_track": source.track,
             "source_voice_branch": source.voice, "tokens": source.evidence}
            for note, (_, source) in zip(piece.notes, ordered)]


def project_score_spines(text):
    """Remove analysis spines and analysis-only rows before raw parsing.

    The returned line map points back to original Joined lines. This is a raw
    score version present inside Joined, never a repair of its adjacent Krn.
    Cross-kind exchanges and newly added spines fail closed.
    """
    selected, lines, line_map = [], [], []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        if raw.startswith("!!"):
            lines.append(raw); line_map.append(number)
            continue
        tokens = raw.strip().split("\t")
        if not selected:
            require(all(t.startswith("**") for t in tokens), "projection needs exclusive interpretations")
            selected = [t == "**kern" for t in tokens]
            require(any(selected), "projection has no score spines")
        require(len(tokens) == len(selected), "projection spine count mismatch")
        keep = [t for t, take in zip(tokens, selected) if take]
        if not all(t == "." for t in keep):
            lines.append("\t".join(keep)); line_map.append(number)
        if all(t.startswith("*") for t in tokens):
            require("*+" not in tokens, "unsupported added spine in projection")
            exchanges = [i for i, t in enumerate(tokens) if t == "*x"]
            require(not exchanges or len(exchanges) == 2 and selected[exchanges[0]] == selected[exchanges[1]],
                    "unsupported cross-kind spine exchange")
            new, i = [], 0
            while i < len(tokens):
                t, take = tokens[i], selected[i]
                if t == "*^":
                    new.extend((take, take))
                elif t == "*v":
                    group = [take]
                    while i + 1 < len(tokens) and tokens[i + 1] == "*v":
                        i += 1; group.append(selected[i])
                    require(len(group) >= 2 and len(set(group)) == 1, "unsupported projection join")
                    new.append(take)
                elif t != "*-":
                    new.append(take)
                i += 1
            selected = new
    return "\n".join(lines) + "\n", line_map


def build_enrichment(piece: CanonicalPiece, analyses: list[tuple[str, Parsed]], *,
                     canonical_binding: dict, source_bindings: list[dict], admission: dict) -> dict:
    """Exact note intersections with independently retained analytical views."""
    harmony_rows, contexts, bindings = [], [], []
    score_signature = sorted((n.onset_qn.to_fraction(), n.duration_qn.to_fraction(), n.pitch,
                              n.spelling_step, n.spelling_alter, int(n.track_id.split(":")[1]), n.is_grace)
                             for n in piece.notes)
    for annotator, analysis in analyses:
        require(raw_signature(analysis) == score_signature, "Joined/score-only note disagreement")
        require(analysis.duration == piece.duration_qn.to_fraction(), "Joined/score duration disagreement")
        for kind, spans in (("function", analysis.functions), ("harm", analysis.harmonies)):
            require(all(a.end <= b.start for a, b in zip(spans, spans[1:])), f"overlapping native {kind} spans")
        edges = sorted(set([Fraction(0), analysis.duration] +
                           [t for s in analysis.functions + analysis.harmonies for t in (s.start, s.end)]))
        view_spans = []
        for a, b in zip(edges, edges[1:]):
            h = next((s for s in analysis.harmonies if s.start <= a and b <= s.end), None)
            f = next((s for s in analysis.functions if s.start <= a and b <= s.end), None)
            hid = f"harmony:{annotator}:{len(view_spans):05d}"
            fields = {
                "tavern.roman": native_field(h.label if h else None, [h.evidence] if h else []),
                "tavern.local_key": native_field(h.key if h else None, [h.evidence] if h else []),
                "tavern.phrase_function": native_field(f.label if f else None, [f.evidence] if f else []),
            }
            row = {"id": hid, "view_id": annotator, "start_qn": rational(a), "end_qn": rational(b), "fields": fields,
                   "function_scope": "native_phrase_model_span",
                   "native_function_span": {"start_qn": rational(f.start), "end_qn": rational(f.end)} if f else None}
            harmony_rows.append(row); view_spans.append((a, b, row))
        for note in piece.notes:
            start = note.onset_qn.to_fraction(); end = start + note.duration_qn.to_fraction()
            matched = False
            for a, b, h in view_spans:
                left, right = max(a, start), min(b, end)
                if left >= right and not (start == end and a <= start < b):
                    continue
                matched = True
                contexts.append({"id": f"context:{annotator}:{len(contexts):08d}", "note_id": note.note_id,
                    "start_qn": rational(left), "end_qn": rational(right), "harmony_id": h["id"],
                    "view_id": annotator, "fields": {
                        **h["fields"],
                        "tavern.note_function": native_field(None, [], reason="phrase_function_is_not_individual_note_function"),
                        "tavern.source_pitch_spelling": native_field(f"{note.spelling_step}:{note.spelling_alter}",
                            [{"canonical_note_id": note.note_id}], origin="source_derived"),
                    }})
            if not matched and start == end:
                contexts.append({"id": f"context:{annotator}:{len(contexts):08d}", "note_id": note.note_id,
                    "start_qn": rational(start), "end_qn": rational(end), "harmony_id": None,
                    "view_id": annotator, "fields": {
                        name: native_field(None, [], reason="point_has_no_half_open_source_context")
                        for name in ("tavern.roman", "tavern.local_key", "tavern.phrase_function", "tavern.note_function")}})
        bindings.append({"annotator": annotator, "path": analysis.source_path, "sha256": analysis.source_sha256,
                         "native_function_spans": len(analysis.functions), "native_harmony_spans": len(analysis.harmonies)})
    require(bool(contexts), "no note contexts")
    return {"schema": "EventTheoryEnrichment@1", "dataset_id": "tavern", "piece_id": piece.piece_id,
            "current_split": "train", "canonical_piece": canonical_binding, "source_bindings": source_bindings,
            "harmonies": harmony_rows, "note_contexts": contexts, "relations": [],
            "coverage": {"canonical_notes": len(piece.notes), "notes_with_context": len({r["note_id"] for r in contexts}),
                         "note_context_rows": len(contexts), "analytical_views": len(analyses),
                         "individual_note_functions_reviewed": 0},
            "provenance": {"adapter": VERSION, "target_revision": TARGET_REVISION,
                           "native_analyses": bindings, "admission": admission,
                           "license": "CC BY-SA 4.0", "source_repository": "https://github.com/jcdevaney/TAVERN",
                           "function_scope": "phrase-model context, not individual-note role",
                           "time_basis": "phrase_local_written_score_order"}}
