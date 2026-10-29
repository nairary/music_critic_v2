"""Frozen raw C3 tiles plus explicitly oracle H geometry and masked targets.

Candidate contexts depend on raw note/slice geometry and H boundaries, never on
which note rows happen to be labelled. Raw C3 encoding precedes the addition of
H/context nodes; no harmonic labels or alternative-view IDs enter its features.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

import torch

from music_critic.relation_probe.data import checked_read, exact, pair
from .data import candidate_labels
from .evidence import atomic_json, binding

BASE_NODES = ("note", "onset", "beat", "bar", "track", "song", "slice")
NODE_TYPES = BASE_NODES + ("context", "harmony")
EXTRA_EDGES = (
    ("context", "observed_note", "note"), ("note", "has_context", "context"),
    ("context", "observed_slice", "slice"), ("slice", "has_context", "context"),
    ("harmony", "geometric_contains", "context"), ("context", "in_harmony_interval", "harmony"),
)
INPUT_WIDTH = 266
LIMITS = dict(core_bars=4, halo_bars=1, max_notes_per_tile=2048,
              max_contexts_per_tile=8192, max_h_per_tile=512, max_flat_edges=300000)


class UnsupportedHGeometry(ValueError):
    """Explicit coverage exclusion; no candidate or event is truncated."""


def edge_schema():
    from music_critic.graph.relations import MANDATORY_EDGE_TYPES
    from music_critic.experiments.analysisgnn.temporal_slices.graph import EDGES
    result = tuple(MANDATORY_EDGE_TYPES) + tuple(EDGES) + EXTRA_EDGES
    if len(result) != len(set(result)) or len(result) > 64:
        raise ValueError("H/member edge schema must be unique and fit 64 type slots")
    return result


def _scope_tracks(h, piece):
    """Absent source scope declares an answer domain, not ensemble harmony."""
    all_tracks = tuple(t.track_id for t in piece.tracks)
    requested = h.get("scope_track_ids")
    if requested is None:
        return frozenset(all_tracks), h.get("participant_scope", {
            "kind": "all_observed_tracks_candidate_domain",
            "provenance": "source_H_has_no_explicit_participant_track_scope",
        })
    if (not isinstance(requested, (tuple, list)) or not requested
            or len(set(requested)) != len(requested) or not set(requested) <= set(all_tracks)):
        raise UnsupportedHGeometry("unsupported_explicit_H_track_scope")
    return frozenset(requested), h.get("participant_scope", {"kind": "explicit_tracks", "track_ids": requested})


def _raw_context_features(note, start, tile_start, track_index):
    onset = note.onset_qn.to_fraction()
    duration = float(note.duration_qn.to_fraction())
    age = float(start - onset)
    return [note.pitch / 127, math.sin(note.pitch % 12 * math.pi / 6),
            math.cos(note.pitch % 12 * math.pi / 6), duration / (1 + duration),
            float(start - tile_start) / 32, track_index / 16,
            float(note.is_percussion), float(note.is_grace),
            age / (1 + abs(age)), float(onset - tile_start) / 32]


def _candidate_geometry(piece, tile, graph, h):
    """Raw active-note x raw-time intervals, clipped only at oracle H bounds."""
    a, b = exact(h["start"]), exact(h["end"])
    left, right = tile.global_start.to_fraction(), tile.global_end.to_fraction()
    if not left <= a < b <= right:
        raise UnsupportedHGeometry("whole_H_does_not_fit_tile")
    tracks, scope = _scope_tracks(h, piece)
    times = [t + left for t, _ in graph.slots[0].time_slots]
    if not times or times[0] > a or times[-1] < b:
        raise UnsupportedHGeometry("raw_slice_grid_does_not_cover_H")
    # Added H boundaries do not modify the raw C3 graph or imply note attacks.
    grid = sorted({a, b, *(t for t in times if a < t < b)})
    notes = {n.note_id: n for n in piece.notes}
    note_index = {identity: i for i, (_, identity) in enumerate(graph.keys["note"])}
    contexts = []
    for note_id in note_index:
        note = notes[note_id]
        if note.track_id not in tracks or note.is_percussion:
            continue
        onset = note.onset_qn.to_fraction()
        end = (note.onset_qn + note.duration_qn).to_fraction()
        # This task has interval labels. Zero-duration grace contexts are an
        # explicitly unsupported domain rather than fabricated intervals.
        if onset == end or onset >= b or end <= a:
            continue
        for start, stop in zip(grid, grid[1:]):
            u, v = max(start, onset), min(stop, end)
            if u < v:
                slot = bisect_right(times, u) - 1
                if not 0 <= slot < len(times) - 1 or v > times[slot + 1]:
                    raise ValueError("context crosses its source raw slice")
                contexts.append((note_id, u, v, note_index[note_id], slot))
    contexts.sort(key=lambda row: (row[1], row[2], row[0]))
    return contexts, scope


def assemble_tile(piece, tile, graph, encoded, harmonies):
    """Pure synthetic-test seam after raw encoding; returns CPU tensor payload.

    ``encoded`` must be frozen C3's seven 128d node states. This seam makes
    changing labels while holding geometry fixed directly testable.
    """
    if len(harmonies) > LIMITS["max_h_per_tile"]:
        raise UnsupportedHGeometry("H_count_limit")
    schema = edge_schema()
    edge_id = {e: i for i, e in enumerate(schema)}
    offsets, features, types = {}, [], []
    cursor = 0
    for type_id, name in enumerate(BASE_NODES):
        values = encoded[name].detach().to(device="cpu", dtype=torch.float32)
        expected = graph.base[name].num_nodes if name != "slice" else graph.slice_features.shape[0]
        if values.shape != (expected, 128) or not torch.isfinite(values).all():
            raise ValueError("C3 node states must be finite, aligned and 128d")
        offsets[name] = cursor
        cursor += values.shape[0]
        features.append(torch.cat((values, values.new_zeros((values.shape[0], 138))), dim=1))
        types.extend([type_id] * values.shape[0])
    raw_edges, raw_edge_types = [], []
    for edge in schema[:-len(EXTRA_EDGES)]:
        source, _, target = edge
        values = graph.edges[edge] if edge in graph.edges else graph.base[edge].edge_index
        values = values.detach().cpu().clone()
        values[0] += offsets[source]
        values[1] += offsets[target]
        raw_edges.append(values)
        raw_edge_types.extend([edge_id[edge]] * values.shape[1])

    notes = {n.note_id: n for n in piece.notes}
    tracks = {t.track_id: i for i, t in enumerate(piece.tracks)}
    context_nodes, context_features, context_identity = {}, [], []
    extra_edges, extra_types, pending, exclusions = [], [], [], []

    def add(source, target, edge):
        extra_edges.append((source, target))
        extra_types.append(edge_id[edge])

    for h in harmonies:
        try:
            geometry, scope = _candidate_geometry(piece, tile, graph, h)
        except UnsupportedHGeometry as error:
            exclusions.append({"h_id": h["id"], "reason": str(error)})
            continue
        if not geometry:
            exclusions.append({"h_id": h["id"], "reason": "no_positive_duration_pitched_candidates"})
            continue
        ids = []
        for note_id, a, b, ni, si in geometry:
            identity = (note_id, a, b)
            if identity not in context_nodes:
                if len(context_nodes) >= LIMITS["max_contexts_per_tile"]:
                    raise UnsupportedHGeometry("context_count_limit")
                node = cursor + len(context_features)
                context_nodes[identity] = node
                context_identity.append(dict(note_id=note_id, start=pair(a), end=pair(b)))
                raw = torch.tensor(_raw_context_features(
                    notes[note_id], a, tile.global_start.to_fraction(), tracks[notes[note_id].track_id]
                ), dtype=torch.float32)
                context_features.append(torch.cat((
                    encoded["note"][ni].detach().cpu().float(),
                    encoded["slice"][si].detach().cpu().float(), raw,
                )))
                add(node, offsets["note"] + ni, EXTRA_EDGES[0])
                add(offsets["note"] + ni, node, EXTRA_EDGES[1])
                add(node, offsets["slice"] + si, EXTRA_EDGES[2])
                add(offsets["slice"] + si, node, EXTRA_EDGES[3])
            ids.append(context_nodes[identity])
        pending.append((h, geometry, ids, scope))

    if context_features:
        features.append(torch.stack(context_features))
        types.extend([7] * len(context_features))
    h_offset = cursor + len(context_features)
    queries, h_features = [], []
    for index, (h, geometry, ids, scope) in enumerate(pending):
        source = h_offset + index
        # All geometric candidates contribute, including unknown/nonmembers.
        h_features.append(torch.stack([context_features[i - cursor] for i in ids]).mean(dim=0))
        ys, masks = candidate_labels(h, [(note_id, a, b) for note_id, a, b, _, _ in geometry])
        for target in ids:
            add(source, target, EXTRA_EDGES[4])
            add(target, source, EXTRA_EDGES[5])
        queries.append(dict(
            h_id=h["id"], source=source, candidates=torch.tensor(ids, dtype=torch.long),
            y=torch.tensor(ys, dtype=torch.float32), mask=torch.tensor(masks, dtype=torch.bool),
            start=h["start"], end=h["end"], view_id=h.get("view_id"),
            participant_scope=scope, candidate_geometry=[context_identity[i - cursor] for i in ids],
            label_provenance=h.get("labels", []),
        ))
    if h_features:
        features.append(torch.stack(h_features))
        types.extend([8] * len(h_features))
    # Alternative analyses can be observationally identical. A model without
    # view-ID inputs cannot satisfy contradictory labels for the same question.
    equivalent = {}
    for query in queries:
        key = (tuple(query['start']), tuple(query['end']), tuple(query['candidates'].tolist()))
        equivalent.setdefault(key, []).append(query)
    for group in equivalent.values():
        if len(group) < 2:
            continue
        known_positive = torch.stack([q['mask'] & (q['y'] == 1) for q in group]).any(0)
        known_negative = torch.stack([q['mask'] & (q['y'] == 0) for q in group]).any(0)
        conflict = known_positive & known_negative
        for query in group:
            query['equivalent_view_conflicts'] = int((query['mask'] & conflict).sum())
            query['mask'] &= ~conflict
    if extra_edges:
        raw_edges.append(torch.tensor(extra_edges, dtype=torch.long).t().contiguous())
    raw_edge_types.extend(extra_types)
    if len(raw_edge_types) > LIMITS["max_flat_edges"]:
        raise UnsupportedHGeometry("flat_edge_count_limit")
    geometry_rows = [{k: h[k] for k in ("id", "start", "end", "view_id", "scope_track_ids", "participant_scope") if k in h}
                     for h, _, _, _ in pending]
    payload = dict(
        schema="HMemberTile@1", x=torch.cat(features), node_type=torch.tensor(types, dtype=torch.long),
        edge_index=torch.cat(raw_edges, dim=1), edge_type=torch.tensor(raw_edge_types, dtype=torch.long),
        queries=queries, exclusions=exclusions,
        oracle_geometry_sha256=sha256(json.dumps(geometry_rows, sort_keys=True).encode()).hexdigest(),
        mode="oracle_H_geometry_frozen_raw_C3", target_fields_are_inputs=False,
    )
    if not torch.isfinite(payload["x"]).all():
        raise ValueError("nonfinite cache features")
    return payload


def _check_raw(piece):
    if piece.annotations or piece.targets or piece.key_signature_events:
        raise ValueError("cache requires a sanitized raw piece without theory sidecars")
    if any(n.voice is not None or n.staff is not None or n.spelling_step is not None
           or n.spelling_alter is not None for n in piece.notes):
        raise ValueError("source-only note fields must be sanitized before C3")


def build_cache(dataset_dir, output_dir, checkpoint_path, checkpoint_sha, checkpoint_size, device="cuda"):
    """Build one fresh immutable cache from admitted preparation bindings."""
    from music_critic.data import loads_piece
    from music_critic.pll.contracts import ContextProfile, MaskQuery
    from music_critic.pll.encoder import PLLContextEncoder
    from music_critic.pll.graph import build_tile_graph
    from music_critic.pll.masking import build_observed_tiles
    from music_critic.pll.observed_view import ObservedMusicView

    started = time.monotonic()
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    manifest_bytes = (dataset_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    seen, families = set(), {}
    for work in manifest["works"]:
        if (not isinstance(work["id"], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", work["id"])
                or work["id"] in seen or work["split"] not in {"train", "selection", "dev"}):
            raise ValueError("unique work IDs and internal TRAIN splits required")
        seen.add(work["id"])
        if work["family"] in families and families[work["family"]] != work["split"]:
            raise ValueError("family split leakage")
        families[work["family"]] = work["split"]
    output_dir.mkdir(parents=True, exist_ok=False)
    counts, rows, exclusions, errors = Counter(), [], [], []
    result = dict(schema="HMemberCache@1", status="running", tiles=rows, counts={},
                  exclusions=exclusions, errors=errors, input_width=INPUT_WIDTH,
                  node_types=len(NODE_TYPES), edge_types=64, node_schema=NODE_TYPES,
                  edge_schema=edge_schema(), limits=LIMITS,
                  dataset_manifest_sha256=sha256(manifest_bytes).hexdigest(),
                  checkpoint={"sha256": checkpoint_sha, "size_bytes": checkpoint_size},
                  test_locked=True, human_gold=False, mode="oracle_H_geometry_frozen_raw_C3")

    def flush(status):
        result.update(status=status, counts=dict(counts), wall_seconds=time.monotonic() - started)
        atomic_json(output_dir / "manifest.json", result)

    flush("running")
    try:
        encoder = PLLContextEncoder.from_pretrained(
            checkpoint_path, expected_sha256=checkpoint_sha,
            expected_size_bytes=checkpoint_size, freeze_policy="frozen",
        ).to(device)
        encoder.requires_grad_(False)
        encoder.eval()
        for work in manifest["works"]:
            piece = loads_piece(checked_read(work["raw"], dataset_dir))
            _check_raw(piece)
            harmonies = json.loads(checked_read(work["targets"], dataset_dir))
            if len({h["id"] for h in harmonies}) != len(harmonies):
                raise ValueError("duplicate H identity within work")
            counts[work["split"] + "_source_h"] += len(harmonies)
            counts["zero_duration_pitched_notes_outside_candidate_domain"] += sum(
                not n.is_percussion and n.duration_qn.to_fraction() == 0 for n in piece.notes
            )
            profile = ContextProfile(context_bars=max(32, len(piece.bars)),
                                     core_bars=LIMITS["core_bars"], halo_bars=LIMITS["halo_bars"],
                                     max_notes_per_tile=LIMITS["max_notes_per_tile"])
            start = piece.bars[0].start_qn
            end = piece.bars[-1].start_qn + piece.bars[-1].duration_qn
            view = ObservedMusicView(piece, MaskQuery(start, end, tuple(t.track_id for t in piece.tracks)),
                                     tuple(range(len(piece.bars))))
            try:
                tiles = build_observed_tiles(view, profile)
            except ValueError as error:
                exclusions.extend(dict(work=work["id"], h_id=h["id"], reason=str(error)) for h in harmonies)
                flush("running")
                continue
            assigned = set()
            for ordinal, tile in enumerate(tiles):
                core = [b for b in piece.bars if b.bar_id in tile.core_bar_ids]
                a = core[0].start_qn.to_fraction()
                b = (core[-1].start_qn + core[-1].duration_qn).to_fraction()
                selected = [h for h in harmonies if a <= exact(h["start"]) < b]
                if not selected:
                    continue
                assigned.update(h["id"] for h in selected)
                try:
                    graph = build_tile_graph(tile, profile)
                except ValueError as error:
                    exclusions.extend(dict(work=work["id"], tile=ordinal, h_id=h["id"], reason=str(error)) for h in selected)
                    continue
                flags = torch.tensor(tile.boundary_flags, dtype=torch.float32, device=device).reshape(-1, 2)
                with torch.no_grad():
                    values = encoder._encode_graph(graph.to(device), flags)
                encoded = {k: v.detach().cpu() for k, v in values.items()}
                try:
                    payload = assemble_tile(piece, tile, graph, encoded, selected)
                except UnsupportedHGeometry as error:
                    exclusions.extend(dict(work=work["id"], tile=ordinal, h_id=h["id"], reason=str(error)) for h in selected)
                    continue
                exclusions.extend(dict(work=work["id"], tile=ordinal, **e) for e in payload.pop("exclusions"))
                if not payload["queries"]:
                    continue
                identifier = f'{work["id"]}-{ordinal:04d}'
                payload.update(id=identifier, work=work["id"], family=work["family"], split=work["split"],
                               parent_raw_sha256=work["raw"]["sha256"], parent_targets_sha256=work["targets"]["sha256"])
                path = output_dir / (identifier + ".pt")
                torch.save(payload, path)
                rows.append(dict(id=identifier, work=work["id"], family=work["family"], split=work["split"],
                                 **binding(path, path.name)))
                for query in payload["queries"]:
                    counts['equivalent_view_conflicts_masked'] += query.get('equivalent_view_conflicts', 0)
                    available = query["mask"]
                    positive = int(query["y"][available].sum())
                    negative = int(available.sum()) - positive
                    prefix = work["split"] + "_"
                    counts[prefix + "cached_h"] += 1
                    counts[prefix + "candidates"] += len(available)
                    counts[prefix + "positive"] += positive
                    counts[prefix + "negative"] += negative
                    counts[prefix + "unknown"] += int((~available).sum())
                    counts[prefix + "mixed_h"] += int(positive > 0 and negative > 0)
            exclusions.extend(dict(work=work["id"], h_id=h["id"], reason="H_start_outside_raw_core_domain")
                              for h in harmonies if h["id"] not in assigned)
            counts["works_processed"] += 1
            flush("running")
            print(json.dumps({"cached_work": work["id"], "tiles": len(rows), "excluded_h": len(exclusions)}), flush=True)
        flush("completed")
    except Exception as error:
        errors.append(dict(type=type(error).__name__, reason=str(error)))
        flush("failed")
        raise
    return result
