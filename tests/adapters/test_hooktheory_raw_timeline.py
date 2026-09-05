"""Synthetic coverage of the versioned annotation-independent raw timeline."""

from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
from hashlib import sha256
import json

import pytest

from music_critic.adapters.hooktheory import (
    HOOKTHEORY_ADAPTER_VERSION,
    HOOKTHEORY_RAW_INPUT_VERSION,
    HookTheoryAdapterConfig,
    HookTheoryAdapterError,
    convert_hooktheory_record,
    hooktheory_raw_input_bytes,
    hooktheory_raw_input_fingerprint,
)
from music_critic.data import RationalTime, dumps_piece, validate_piece
from music_critic.graph import build_raw_graph, dumps_graph, graph_fingerprint
from music_critic.tasks.corpus import (
    CanonicalCorpusInput,
    CorpusCacheConfig,
    build_hooktheory_corpus_cache,
    cache_canonical_corpus,
    corpus_cache_key,
    load_cached_piece,
)


def _chord(**overrides):
    return {
        "beat": 1, "duration": 2, "root": 1, "type": 5,
        "inversion": 0, "adds": [], "omits": [], "alterations": [],
        "suspensions": [], "borrowed": None, "isRest": False, **overrides,
    }


def _record():
    return {
        "hash": "raw-timeline", "split": "train",
        "json": {
            "endBeat": 5,
            "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
            "tempos": [{"beat": 1, "bpm": 120}],
            "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
            "notes": [{"beat": 1, "duration": 2, "sd": "1", "octave": 0}],
            "chords": [_chord()],
        },
    }


def _convert(record, include_targets=True):
    return convert_hooktheory_record(
        record["hash"], record,
        config=HookTheoryAdapterConfig("hooktheory-fixture", include_targets),
    )


def _target(piece, task):
    return next(value for value in piece.targets if value.task == task)


@pytest.mark.parametrize("include_targets", [False, True])
@pytest.mark.parametrize("source_duration", [None, 2, 5])
@pytest.mark.parametrize("chords", [
    [], [_chord(root=5)], [_chord(duration=100)],
    [_chord(beat=30, duration=4)], [_chord(beat=5, duration=4)],
    [_chord(duration=100), _chord(beat=7, duration=1)],
])
def test_chord_changes_preserve_exact_raw_bytes_cache_identity_and_graph(
    include_targets, source_duration, chords,
):
    record = _record()
    if source_duration is None:
        record["json"].pop("endBeat")
    else:
        record["json"]["endBeat"] = source_duration
    before = _convert(record)
    record["json"]["chords"] = chords
    source_snapshot = deepcopy(record)
    after = _convert(record, include_targets)
    assert record == source_snapshot
    assert not validate_piece(after).errors
    assert (before.duration_qn, before.bars, before.beats, before.notes) == (
        after.duration_qn, after.bars, after.beats, after.notes,
    )
    assert hooktheory_raw_input_bytes(before) == hooktheory_raw_input_bytes(after)
    assert hooktheory_raw_input_fingerprint(before) == hooktheory_raw_input_fingerprint(after)
    graphs = [build_raw_graph(piece) for piece in (before, after)]
    assert dumps_graph(graphs[0]) == dumps_graph(graphs[1])
    assert graph_fingerprint(graphs[0]) == graph_fingerprint(graphs[1])
    # This is the mandatory raw identity, distinct from target-bearing artifacts.
    cache_keys = [corpus_cache_key(
        source_identity="raw-timeline", source_sha256=sha256(dumps_piece(piece).encode()).hexdigest(),
        adapter_name="hooktheory", adapter_version=HOOKTHEORY_ADAPTER_VERSION,
        adapter_config_fingerprint=sha256(HOOKTHEORY_RAW_INPUT_VERSION.encode()).hexdigest(),
        cache_input_sha256=hooktheory_raw_input_fingerprint(piece),
    ) for piece in (before, after)]
    assert cache_keys[0] == cache_keys[1]
    if not include_targets:
        assert after.targets == after.annotations == ()


def test_partial_overlap_clips_exactly_and_outside_masks_every_chord_target():
    record = _record()
    record["json"]["chords"] = [
        _chord(beat=Decimal("3.5"), duration=Decimal("8.5")),
        _chord(beat=5), _chord(beat=40),
    ]
    piece = _convert(record)
    chords = [span for span in piece.annotations if span.annotation_type == "theory.chord"]
    assert [(span.start_qn, span.end_qn) for span in chords] == [
        (RationalTime(5, 2), RationalTime(4)),
        (RationalTime(4), RationalTime(4)),
        (RationalTime(4), RationalTime(4)),
    ]
    for target in piece.targets:
        if target.task.startswith("theory.chord."):
            assert target.mask == (True, False, False)
            assert target.values[1:] == (None, None)
            assert target.source[1:] == target.provenance[1:] == (None, None)
    reasons = [flag for flag in piece.quality_flags if "raw_timeline" in flag.code]
    assert [flag.code for flag in reasons].count("hooktheory.chord_outside_raw_timeline") == 2
    assert any(flag.code == "hooktheory.chord_clipped_to_raw_timeline" for flag in reasons)
    assert any("source index 2" in flag.message and "all targets masked" in flag.message for flag in reasons)
    assert dumps_piece(piece) != dumps_piece(_convert(_record()))


@pytest.mark.parametrize("end_beat,expected", [(None, 0), (1, 0), (5, 4)])
def test_empty_melody_never_uses_chords_or_unused_late_key_to_invent_time(end_beat, expected):
    record = _record()
    record["json"]["notes"] = []
    if end_beat is None:
        record["json"].pop("endBeat")
    else:
        record["json"]["endBeat"] = end_beat
    empty = _convert(record)
    record["json"]["chords"] = [_chord(duration=100), _chord(beat=60)]
    record["json"]["keys"].append({"beat": 20, "tonic": "D", "scale": "minor"})
    piece = _convert(record)
    hidden = _convert(record, False)
    assert piece.duration_qn == RationalTime(expected)
    assert hooktheory_raw_input_bytes(piece) == hooktheory_raw_input_bytes(empty)
    assert hooktheory_raw_input_bytes(hidden) == hooktheory_raw_input_bytes(empty)
    assert dumps_graph(build_raw_graph(piece)) == dumps_graph(build_raw_graph(empty))
    assert "hooktheory.empty_melody" in {flag.code for flag in piece.quality_flags}
    key = _target(piece, "theory.local_key.tonic_pc")
    assert key.mask == (expected > 0, False)
    if not expected:
        assert piece.bars == piece.beats == ()
        assert all(not any(target.mask) for target in piece.targets)
    assert not validate_piece(piece).errors


def test_rest_timing_and_compound_meter_remain_exact_observable_time():
    record = _record()
    record["json"].pop("endBeat")
    record["json"]["notes"] = [{"beat": 3, "duration": 3, "isRest": True}]
    record["json"]["meters"].append({"beat": 3, "numBeats": 6, "beatUnit": 3})
    record["json"]["chords"] = [_chord(duration=100)]
    piece = _convert(record)
    assert piece.duration_qn == RationalTime(7, 2)
    assert piece.notes == ()
    assert piece.bars[-1].duration_qn == RationalTime(3, 2)
    assert not validate_piece(piece).errors


def test_raw_projection_excludes_semantic_identity_and_preserves_input_changes():
    piece = _convert(_record())
    changed_metadata = replace(piece, split="val", source_group_id="another-group")
    assert hooktheory_raw_input_bytes(changed_metadata) == hooktheory_raw_input_bytes(piece)
    raw = json.loads(hooktheory_raw_input_bytes(piece))
    assert raw["raw_input_version"] == HOOKTHEORY_RAW_INPUT_VERSION
    assert raw["adapter_version"] == HOOKTHEORY_ADAPTER_VERSION
    assert not {"targets", "annotations", "quality_flags", "provenance", "split"} & raw.keys()
    changed_note = replace(piece.notes[0], pitch=61)
    assert hooktheory_raw_input_fingerprint(replace(piece, notes=(changed_note,))) != hooktheory_raw_input_fingerprint(piece)
    changed_track = replace(piece.tracks[0], source_track_index=4)
    assert hooktheory_raw_input_fingerprint(replace(piece, tracks=(changed_track,))) != hooktheory_raw_input_fingerprint(piece)


def test_new_adapter_identity_creates_new_cache_and_preserves_sealed_old_artifact(tmp_path):
    record = _record()
    piece = _convert(record)
    assert next(p.version for p in piece.provenance if p.provenance_id == "prov:conversion") == HOOKTHEORY_ADAPTER_VERSION
    config = HookTheoryAdapterConfig("hooktheory-fixture")
    assert asdict(config)["adapter_version"] == HOOKTHEORY_ADAPTER_VERSION
    old_piece = replace(piece, provenance=tuple(
        replace(p, version="1") if p.provenance_id == "prov:conversion" else p
        for p in piece.provenance
    ))
    with pytest.raises(HookTheoryAdapterError, match="historical pieces must be rebuilt"):
        hooktheory_raw_input_bytes(old_piece)
    cache = CorpusCacheConfig(tmp_path / "cache")
    old_index, _ = cache_canonical_corpus(
        [CanonicalCorpusInput(
            piece=old_piece, lineage_group_id=piece.source_group_id,
            source_identity=record["hash"], source_relative_path="4_merged.json#raw-timeline",
            source_sha256="a" * 64, suggested_split="train",
        )],
        cache_config=cache, dataset_id="hooktheory-fixture", adapter_name="hooktheory",
        adapter_version="2B.1", adapter_config={"dataset_name": "hooktheory-fixture", "include_targets": True},
        source_identity="4_merged.json", source_fingerprint="b" * 64,
    )
    old_path = cache.root / cache.namespace / old_index.records[0].canonical_relative_path
    old_bytes = old_path.read_bytes()
    source = tmp_path / "4_merged.json"
    source.write_text(json.dumps({record["hash"]: record}))
    new_index, report = build_hooktheory_corpus_cache(
        source, cache_config=cache, dataset_id="hooktheory-fixture",
    )
    assert new_index.header.adapter_version == HOOKTHEORY_ADAPTER_VERSION
    assert new_index.header.adapter_config_fingerprint != old_index.header.adapter_config_fingerprint
    assert new_index.records[0].cache_key != old_index.records[0].cache_key
    assert report.cache_miss_count == 1
    assert old_path.read_bytes() == old_bytes
    assert load_cached_piece(old_index.records[0], cache) == old_piece
    repeated, report = build_hooktheory_corpus_cache(
        source, cache_config=cache, dataset_id="hooktheory-fixture",
    )
    assert repeated == new_index
    assert report.cache_hit_count == 1


def test_stream_cache_uses_raw_identity_and_keeps_distinct_target_artifacts(tmp_path):
    cache = CorpusCacheConfig(tmp_path / "cache")
    source = tmp_path / "4_merged.json"
    record = _record()
    indexes = []
    artifact_bytes = {}
    expected_pieces = []
    graph_hashes = []
    for chords in ([_chord()], [], [_chord(root=5)], [_chord(duration=100)]):
        record["json"]["chords"] = chords
        source.write_text(json.dumps({record["hash"]: record}))
        for include_targets in (True, False):
            index, _ = build_hooktheory_corpus_cache(
                source, cache_config=cache, dataset_id="hooktheory-fixture",
                include_targets=include_targets,
            )
            indexes.append(index)
            row = index.records[0]
            path = cache.root / cache.namespace / row.canonical_relative_path
            if path in artifact_bytes:
                assert path.read_bytes() == artifact_bytes[path]
            artifact_bytes[path] = path.read_bytes()
            piece = load_cached_piece(row, cache)
            expected_pieces.append((row, piece))
            graph_hashes.append(graph_fingerprint(build_raw_graph(piece)))
            assert bool(piece.targets) is include_targets
            conversion = next(p for p in piece.provenance if p.provenance_id == "prov:conversion")
            assert dict(conversion.details)["include_targets"] is include_targets
            assert piece.duration_qn == RationalTime(4)
    assert len({index.records[0].cache_key for index in indexes}) == 1
    assert len({index.header.adapter_config_fingerprint for index in indexes}) == 1
    assert len({index.records[0].source_sha256 for index in indexes}) == 4
    assert len({index.records[0].canonical_sha256 for index in indexes}) > 1
    assert len({index.header.index_fingerprint for index in indexes}) > 1
    assert len(set(graph_hashes)) == 1
    baseline_key = indexes[0].records[0].cache_key
    # Both sounding pitch and observable duration must invalidate raw caches.
    for changes in ({"sd": "3"}, {"duration": 5}):
        changed = _record()
        changed["json"]["notes"][0].update(changes)
        source.write_text(json.dumps({changed["hash"]: changed}))
        changed_index, _ = build_hooktheory_corpus_cache(
            source, cache_config=cache, dataset_id="hooktheory-fixture",
        )
        assert changed_index.records[0].cache_key != baseline_key
    for path, payload in artifact_bytes.items():
        assert path.read_bytes() == payload
    for row, expected in expected_pieces:
        assert load_cached_piece(row, cache) == expected
