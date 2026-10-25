"""Synthetic MusicRender-shaped objects; no real corpus access."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from music_critic.adapters.pdmx import (
    PDMX_METER_DENOMINATOR_LOG2_VOCABULARY_SIZE, PDMX_METER_NUMERATOR_VOCABULARY_SIZE,
    PDMX_RAW_FEATURE_REGISTRY_VERSION,
    PDMX_SCHEMA_UPSTREAM_REVISION, PdmxAdapterConfig, PdmxAdapterError,
    load_pdmx_piece,
)
from music_critic.data import RationalTime, validate_piece
from music_critic.graph import RAW_FEATURE_REGISTRY, build_raw_graph, graph_fingerprint


def _source():
    return {
        "resolution": 12, "absolute_time": False,
        "tracks": [{"__class__.__name__": "Track", "program": 0, "is_drum": False,
                    "notes": [{"__class__.__name__": "Note", "time": 1,
                               "pitch": 60, "duration": 11}]}],
    }


def _load(tmp_path, data, **kwargs):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_pdmx_piece(path, source_id="fixture-A", source_group_id="group-fixture-A",
                           split="train", **kwargs)


def test_exact_ticks_optional_metadata_and_empty_theory(tmp_path):
    piece = _load(tmp_path, _source())
    assert not validate_piece(piece).errors
    assert piece.notes[0].onset_qn == RationalTime(1, 12)
    assert piece.notes[0].duration_qn == RationalTime(11, 12)
    assert piece.notes[0].source_onset_ticks == 1
    assert piece.notes[0].source_duration_ticks == 11
    assert piece.notes[0].velocity is None
    assert piece.notes[0].voice is piece.notes[0].staff is None
    assert piece.notes[0].spelling_step is None
    assert piece.targets == piece.annotations == ()
    assert piece.metadata.title is None
    assert piece.tempo_events[0].microseconds_per_quarter == 500_000
    assert {flag.message for flag in piece.quality_flags} == {"meter", "tempo"}
    assert len(PDMX_SCHEMA_UPSTREAM_REVISION) == 40
    graph = build_raw_graph(piece)
    assert graph["note"].num_nodes == 1
    assert ("note", "next_in_track", "note") in graph.edge_types


def test_integral_json_decimals_preserve_exact_projection_and_source_receipt(tmp_path):
    data = _source()
    data["resolution"] = 480
    data["tracks"][0]["notes"][0].update(time=13060, duration=480, velocity=80)
    data["tempos"] = [{"time": 0, "qpm": 120}, {"time": 13060, "qpm": 89.5}]
    data["time_signatures"] = [{"time": 0, "numerator": 4, "denominator": 4}]
    data["key_signatures"] = [{"time": 0, "fifths": -2, "mode": "major"}]
    data["barlines"] = [{"time": 0}, {"time": 1920}]
    data["beats"] = [{"time": 0, "is_downbeat": True}]
    integer_piece = _load(tmp_path, data)

    def decimal_numbers(value):
        if type(value) is int:
            return float(value)  # Tiny synthetic values are exactly representable.
        if isinstance(value, dict):
            return {key: decimal_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decimal_numbers(item) for item in value]
        return value

    decimal_piece = _load(tmp_path, decimal_numbers(data))
    assert decimal_piece.notes == integer_piece.notes
    assert decimal_piece.notes[0].onset_qn == RationalTime(653, 24)
    assert decimal_piece.notes[0].source_onset_ticks == 13060
    assert type(decimal_piece.notes[0].source_onset_ticks) is int
    for field in ("tracks", "source_resolution", "duration_qn", "bars", "beats", "tempo_events",
                  "meter_events", "key_signature_events", "quality_flags"):
        assert getattr(decimal_piece, field) == getattr(integer_piece, field)
    assert decimal_piece.tempo_events[1].onset_qn == RationalTime(653, 24)
    assert decimal_piece.tempo_events[1].microseconds_per_quarter == 670391
    source = next(p for p in decimal_piece.provenance if p.provenance_id == "prov:source")
    integer_source = next(p for p in integer_piece.provenance if p.provenance_id == "prov:source")
    assert source.checksum_sha256 != integer_source.checksum_sha256
    projection = next(p for p in decimal_piece.provenance if p.provenance_id == "prov:projection")
    details = dict(projection.details)
    assert projection.version == details["adapter_version"] == "1.0.1"
    assert details["integer_number_policy"] == "finite-integral-decimal-exact-int-v1"
    assert details["max_decimal_integer_digits"] == 4300
    assert decimal_piece.quality_flags == ()  # Numeric normalization is not a musical default.
    assert decimal_piece.targets == decimal_piece.annotations == ()


def test_decimal_ticks_do_not_pass_through_binary_float(tmp_path):
    path = tmp_path / "exact-decimal.json"
    path.write_text(
        '{"resolution":9007199254740992,"tracks":[{"notes":'
        '[{"time":9007199254740993.0,"duration":1.0,"pitch":60}]}]}'
    )
    piece = load_pdmx_piece(path, source_id="synthetic", source_group_id="synthetic", split="train")
    assert piece.notes[0].source_onset_ticks == 9007199254740993
    assert piece.notes[0].onset_qn == RationalTime(9007199254740993, 9007199254740992)
    assert piece.notes[0].duration_qn == RationalTime(1, 9007199254740992)


@pytest.mark.parametrize("token,expected", [
    ("1.306e4", 13060), ("-0.0", 0), ("1.0000", 1), ("0e999999999", 0),
])
def test_integral_json_number_notations_are_exact(tmp_path, token, expected):
    path = tmp_path / "exact-notation.json"
    path.write_text(json.dumps(_source()).replace('"time": 1', '"time": ' + token))
    piece = load_pdmx_piece(path, source_id="synthetic", source_group_id="synthetic", split="train")
    assert piece.notes[0].source_onset_ticks == expected
    assert type(piece.notes[0].source_onset_ticks) is int


@pytest.mark.parametrize("field,value", [
    ("time", 1.5), ("time", -1.0), ("time", True), ("time", "1.0"), ("time", None),
    ("duration", 1.5), ("duration", -1.0), ("duration", None),
    ("pitch", 60.5), ("pitch", 128.0), ("pitch", -1.0),
    ("velocity", 128.0), ("velocity", -1.0), ("velocity", False),
])
def test_decimal_compatibility_does_not_relax_integer_validity(tmp_path, field, value):
    data = _source()
    data["tracks"][0]["notes"][0][field] = value
    with pytest.raises(PdmxAdapterError, match="invalid_schema"):
        _load(tmp_path, data)


@pytest.mark.parametrize("token,category", [
    ("1.000000000000000000000000000001", "invalid_schema"),
    ("1e-999999999", "invalid_schema"),
    ("1e999999999", "resource_numeric_expansion"),
    ("NaN", "invalid_json"), ("Infinity", "invalid_json"), ("-Infinity", "invalid_json"),
])
def test_bad_decimal_ticks_reject_before_truncation_or_numeric_expansion(tmp_path, token, category):
    path = tmp_path / "bad-decimal.json"
    path.write_text(json.dumps(_source()).replace('"time": 1', '"time": ' + token))
    with pytest.raises(PdmxAdapterError) as caught:
        load_pdmx_piece(path, source_id="synthetic", source_group_id="synthetic", split="train")
    assert caught.value.category == category


@pytest.mark.parametrize("fifths", [-8.0, 14.0, 15.0, 241.0, True, "-2"])
def test_invalid_key_fifths_are_not_normalized_into_another_signature(tmp_path, fifths):
    data = _source()
    data["key_signatures"] = [{"time": 0.0, "fifths": fifths}]
    with pytest.raises(PdmxAdapterError, match="invalid_schema"):
        _load(tmp_path, data)


def test_null_key_fifths_remain_unavailable_with_integral_decimal_time(tmp_path):
    data = _source()
    data["key_signatures"] = [{"time": 0.0, "fifths": None, "root": 0}]
    piece = _load(tmp_path, data)
    assert piece.key_signature_events == ()
    assert piece.targets == piece.annotations == ()


def test_meter_change_is_exact_bar_boundary_and_tempo_decimal_rounds_half_up(tmp_path):
    data = _source()
    data["tracks"][0]["notes"][0].update(time=1, duration=60)
    data["time_signatures"] = [
        {"time": 0, "numerator": 4, "denominator": 4},
        {"time": 30, "numerator": 6, "denominator": 8},
    ]
    data["tempos"] = [{"time": 0, "qpm": 120}, {"time": 13, "qpm": 89.5}]
    piece = _load(tmp_path, data)
    assert piece.bars[0].duration_qn == RationalTime(5, 2)
    assert piece.bars[0].is_incomplete and not piece.bars[0].is_pickup
    assert piece.bars[1].start_qn == RationalTime(5, 2)
    assert piece.bars[1].meter_event_id == piece.meter_events[1].meter_event_id
    assert piece.tempo_events[1].onset_qn == RationalTime(13, 12)
    assert piece.tempo_events[1].microseconds_per_quarter == 670391
    assert piece.notes[0].duration_qn == RationalTime(5)
    assert not validate_piece(piece).errors


def test_source_pickup_downbeat_and_final_incomplete(tmp_path):
    data = _source()
    data["tracks"][0]["notes"][0].update(time=0, duration=72)
    data["barlines"] = [{"time": 0}, {"time": 12}, {"time": 60}]
    data["beats"] = [{"time": 12, "is_downbeat": True}]
    piece = _load(tmp_path, data)
    assert [bar.duration_qn for bar in piece.bars] == [RationalTime(1), RationalTime(4), RationalTime(1)]
    assert piece.bars[0].is_pickup
    assert piece.bars[0].metric_offset_qn == RationalTime(3)
    assert piece.beats[0].index_in_bar == 3
    assert not piece.beats[0].is_downbeat
    assert piece.bars[-1].is_incomplete


def test_downbeats_are_explicit_boundaries_without_barlines(tmp_path):
    data = _source()
    data["tracks"][0]["notes"][0].update(time=0, duration=60)
    data["beats"] = [{"time": 12, "is_downbeat": True}]
    piece = _load(tmp_path, data)
    assert piece.bars[0].is_pickup
    assert piece.bars[1].start_qn == RationalTime(1)


def test_annotations_roles_and_annotation_derived_length_cannot_change_raw_graph(tmp_path):
    base = _source()
    first = _load(tmp_path, base)
    changed = deepcopy(base)
    changed["annotations"] = [{"time": 99999, "annotation": {"__class__.__name__": "ExecuteMe", "root": "F#"}}]
    changed["lyrics"] = [{"time": 100000, "lyric": "ignored"}]
    changed["song_length"] = 100001
    changed["tracks"][0]["annotations"] = [{"theory": "V/V"}]
    changed["tracks"][0]["role"] = "melody"
    changed["tracks"][0]["notes"][0].update(voice=44, staff=9)
    second = _load(tmp_path, changed)
    assert second.duration_qn == first.duration_qn
    assert second.notes == first.notes
    assert second.annotations == second.targets == ()
    assert graph_fingerprint(build_raw_graph(first)) == graph_fingerprint(build_raw_graph(second))


def test_spelling_key_signature_and_grace_are_optional_raw_facts(tmp_path):
    data = _source()
    data["tracks"][0]["notes"] += [
        {"time": 12, "pitch": 61, "duration": 0, "is_grace": True, "pitch_str": "Db4"}
    ]
    data["key_signatures"] = [{"time": 0, "fifths": -2, "mode": "major"}]
    piece = _load(tmp_path, data)
    assert piece.notes[-1].is_grace and piece.notes[-1].duration_qn == RationalTime(0)
    assert (piece.notes[-1].spelling_step, piece.notes[-1].spelling_alter) == ("D", -1)
    assert piece.key_signature_events[0].fifths == -2
    assert piece.targets == ()
    build_raw_graph(piece)


def test_point_only_grace_has_metric_container_without_fake_duration(tmp_path):
    data = _source()
    data["tracks"][0]["notes"][0].update(time=0, duration=0, is_grace=True)
    piece = _load(tmp_path, data)
    assert piece.notes[0].duration_qn == RationalTime(0)
    assert piece.duration_qn == RationalTime(4)
    assert "point-only-timeline" in {flag.message for flag in piece.quality_flags}


@pytest.mark.parametrize("split", ["test", "TEST", None, "", "dev"])
def test_forbidden_split_precedes_any_source_reader(monkeypatch, split):
    def forbidden(*args, **kwargs):
        raise AssertionError("source reader was invoked")
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(PdmxAdapterError, match="split_forbidden"):
        load_pdmx_piece(Path("not-present"), source_id="id", source_group_id="group", split=split)


@pytest.mark.parametrize("change,category", [
    ({"resolution": 0}, "invalid_schema"),
    ({"resolution": 1.5}, "invalid_schema"),
    ({"absolute_time": True}, "unsupported_absolute_time"),
    ({"tracks": []}, "empty_score"),
    ({"tracks": [{"chords": [{"pitches": [60, 64, 67]}]}]}, "empty_score"),
    ({"time_signatures": [{"time": 0, "numerator": 3, "denominator": 3}]}, "invalid_meter"),
    ({"tempos": [{"time": 0, "qpm": 0}]}, "invalid_tempo"),
    ({"metadata": []}, "invalid_schema"),
    ({"tempos": [{"time": 0, "qpm": 120}, {"time": 0, "qpm": 130}]}, "conflicting_tempo"),
    ({"key_signatures": [{"time": 200, "fifths": 1}]}, "key_outside_raw_timeline"),
    ({"key_signatures": [{"time": 0, "fifths": 1}, {"time": 0, "fifths": 2}]}, "conflicting_key_signature"),
])
def test_invalid_sources_have_stable_categories(tmp_path, change, category):
    data = _source()
    data.update(change)
    with pytest.raises(PdmxAdapterError) as caught:
        _load(tmp_path, data)
    assert caught.value.category == category


@pytest.mark.parametrize("change", [{"time": 0.5}, {"duration": -1}, {"pitch": 128}, {"duration": 0}])
def test_invalid_notes_are_rejected(tmp_path, change):
    data = _source()
    data["tracks"][0]["notes"][0].update(change)
    with pytest.raises(PdmxAdapterError):
        _load(tmp_path, data)


def test_unsupported_internal_irregular_source_bar_is_not_silently_lost(tmp_path):
    data = _source()
    data["tracks"][0]["notes"][0].update(time=0, duration=144)
    data["barlines"] = [{"time": 48}, {"time": 60}]
    with pytest.raises(PdmxAdapterError, match="unsupported_metric_geometry"):
        _load(tmp_path, data)


@pytest.mark.parametrize("field,limit", [("max_source_bytes", 5), ("max_notes", 1), ("max_tracks", 1), ("max_bars", 1), ("max_beats", 1)])
def test_resources_are_explicitly_bounded(tmp_path, field, limit):
    data = _source()
    data["tracks"][0]["notes"][0]["duration"] = 192
    data["tracks"].append(deepcopy(data["tracks"][0]))
    config = replace(PdmxAdapterConfig(), **{field: limit})
    with pytest.raises(PdmxAdapterError) as caught:
        _load(tmp_path, data, config=config)
    assert caught.value.category == f"resource_{field}"


@pytest.mark.parametrize("payload", ["{", '{"resolution":12,"resolution":24}', '{"value":NaN}', '[1,2,3]'])
def test_invalid_json_never_deserializes_objects(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(PdmxAdapterError):
        load_pdmx_piece(path, source_id="id", source_group_id="g", split="train")


def test_note_identity_uses_original_array_ordinal_not_sorted_position(tmp_path):
    data = _source()
    data["tracks"][0]["notes"] = [{"time": 12, "duration": 12, "pitch": 64}, {"time": 0, "duration": 12, "pitch": 60}]
    piece = _load(tmp_path, data)
    assert piece.notes[0].note_id.endswith("-000000001")
    assert piece.notes[1].note_id.endswith("-000000000")


def test_unexpected_programming_error_propagates(tmp_path, monkeypatch):
    import music_critic.adapters.pdmx as pdmx
    def broken(*args, **kwargs):
        raise RuntimeError("programming error")
    monkeypatch.setattr(pdmx, "_timeline", broken)
    with pytest.raises(RuntimeError, match="programming error"):
        _load(tmp_path, _source())


def test_unbounded_decimal_exponent_is_rejected_before_fraction_expansion(tmp_path):
    path = tmp_path / "huge-tempo.json"
    payload = json.dumps(_source())[:-1] + ', "tempos":[{"time":0,"qpm":1e-999999999}]}'
    path.write_text(payload)
    with pytest.raises(PdmxAdapterError, match="invalid_tempo"):
        load_pdmx_piece(path, source_id="id", source_group_id="g", split="validation")


def test_decimal_exponent_outside_parser_range_is_invalid_json(tmp_path):
    path = tmp_path / "unrepresentable-tempo.json"
    path.write_text(
        '{"resolution":480,"tempos":[{"time":0,"qpm":1e999999999999999999999999999999}],'
        '"tracks":[{"notes":[{"time":0,"duration":480,"pitch":60}]}]}',
        encoding="utf-8",
    )
    with pytest.raises(PdmxAdapterError) as caught:
        load_pdmx_piece(path, source_id="id", source_group_id="g", split="validation")
    assert caught.value.category == "invalid_json"


def test_adapter_meter_envelope_matches_actual_raw_feature_registry():
    assert RAW_FEATURE_REGISTRY.version == PDMX_RAW_FEATURE_REGISTRY_VERSION
    meter_specs = [spec for spec in RAW_FEATURE_REGISTRY.specs if spec.name in {
        "meter_numerator", "meter_denominator_log2"
    }]
    assert {(spec.node_type, spec.name) for spec in meter_specs} == {
        (node, feature) for node in ("bar", "beat")
        for feature in ("meter_numerator", "meter_denominator_log2")
    }
    for spec in meter_specs:
        expected = PDMX_METER_NUMERATOR_VOCABULARY_SIZE if spec.name == "meter_numerator" else PDMX_METER_DENOMINATOR_LOG2_VOCABULARY_SIZE
        assert spec.vocabulary_size == expected
        assert spec.unknown_id is None


@pytest.mark.parametrize("numerator,denominator", [(256, 4), (1, 2**128)])
def test_canonical_meter_outside_raw_graph_vocabulary_is_source_rejection(tmp_path, numerator, denominator):
    data = _source()
    data["time_signatures"] = [{"time": 0, "numerator": numerator, "denominator": denominator}]
    with pytest.raises(PdmxAdapterError) as caught:
        _load(tmp_path, data)
    assert caught.value.category == "unsupported_meter"


@pytest.mark.parametrize("numerator,denominator", [(255, 4), (1, 2**127)])
def test_supported_raw_meter_vocabulary_edges_build_graph(tmp_path, numerator, denominator):
    data = _source()
    data["resolution"] = max(12, denominator // 4)
    data["tracks"][0]["notes"][0].update(time=0, duration=1)
    data["time_signatures"] = [{"time": 0, "numerator": numerator, "denominator": denominator}]
    piece = _load(tmp_path, data)
    assert build_raw_graph(piece)["note"].num_nodes == 1
