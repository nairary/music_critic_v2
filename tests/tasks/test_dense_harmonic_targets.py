"""Synthetic exact-context contracts, not musical quality measurements."""
from copy import deepcopy
from dataclasses import replace

import pytest

from music_critic.data import RationalTime
from music_critic.tasks.dense_harmonic_targets import (
    COORDINATE_CONTRACT, DENSE_FIELD_VOCABULARIES, DenseHarmonicError,
    build_dense_harmonic_targets, collate_dense_harmonic_targets,
    dumps_dense_harmonic_targets, loads_dense_harmonic_targets,
)
from tests.tasks.test_multisource_contract import _hook_piece


def q(num, den=1):
    return {"num": num, "den": den}


def fixture():
    base = _hook_piece()
    note = replace(base.notes[0], note_id="note:n0", onset_qn=RationalTime(0),
                   duration_qn=RationalTime(4), pitch=60, spelling_step="C",
                   spelling_alter=0, staff=1, voice=1)
    piece = replace(base, notes=(note,), annotations=(), targets=())
    values = {"localkey_tpc": "0", "root_tpc": "0", "bass_note_tpc": "0",
              "chord_tones": "(0, 4, 1)", "added_tones": "()",
              "label": "I", "localkey": "C", "chord_type": "M",
              "unfolded_harmony_index": "0"}
    h0 = {"id": "h0", "start_qn": q(0), "end_qn": q(2), "values": values,
          "availability": {key: True for key in values}, "source_row_ids": ["l2"]}
    h1 = deepcopy(h0)
    h1.update(id="h1", start_qn=q(2), end_qn=q(3))
    h1["values"].update(root_tpc="-1", bass_note_tpc="-1", chord_tones="(-1, 3, 0)",
                         label="IV", unfolded_harmony_index="1")
    segment = {"source_event_id": "l2", "note_id": "note:n0", "start_qn": q(0),
               "end_qn": q(4), "binding_available": True, "mask_reason": None,
               "part_id": "part1", "staff": 1, "voice": 1, "harmony_source_index": "0",
               "values": {"pitch": 60, "tpc": 0, "valid_chord_label": True,
                          "note_degree": "1", "tpc_is_in_label": True,
                          "tpc_is_root": True, "tpc_is_bass": True}}
    packet = {"schema": "DenseHarmonicSourcePacket@1", "time_unit": "quarter_note", "dataset_id": piece.dataset_name,
              "piece_id": piece.piece_id, "current_split": "train",
              "source_version_id": "synthetic-source@1", "coordinate_contract": dict(COORDINATE_CONTRACT),
              "harmonies": [h0, h1], "note_segments": [segment]}
    return piece, packet


def test_held_note_gets_changing_context_and_explicit_unknown_tail():
    piece, packet = fixture()
    result = build_dense_harmonic_targets(piece, packet)
    rows = result["rows"]
    assert len(rows) == 3
    assert [row["note_id"] for row in rows] == ["note:n0"] * 3
    assert [(row["start_qn"], row["end_qn"]) for row in rows] == [
        (q(0), q(2)), (q(2), q(3)), (q(3), q(4))]
    assert [row["fields"]["chord_degree"]["value"] for row in rows] == ["1", "5", None]
    assert [row["fields"]["is_root"]["value"] for row in rows] == [True, False, None]
    assert rows[0]["fields"]["source_is_root"]["value"] is True
    assert not rows[1]["fields"]["source_is_root"]["available"]
    assert rows[2]["fields"]["role"]["reason"] == "no_harmony_coverage"


def repitch(piece, packet, pitch, step, alter, tpc):
    piece = replace(piece, notes=(replace(piece.notes[0], pitch=pitch,
                                         spelling_step=step, spelling_alter=alter),))
    packet["note_segments"][0]["values"].update(pitch=pitch, tpc=tpc)
    return piece


def test_added_tone_is_member_and_compound_degree_needs_explicit_evidence():
    piece, packet = fixture()
    piece = repitch(piece, packet, 62, "D", 0, 2)
    harmony = packet["harmonies"][0]
    harmony["values"]["added_tones"] = "(2,)"
    fields = build_dense_harmonic_targets(piece, packet)["rows"][0]["fields"]
    assert fields["role"]["value"] == "added"
    assert fields["core_member"]["value"] is False
    assert fields["added_member"]["value"] is True
    assert fields["chord_member"]["value"] is True
    assert fields["chord_degree"]["value"] == "2"
    assert not fields["compound_member"]["available"]
    harmony["values"]["explicit_compound_member_by_tpc"] = {"2": "9"}
    harmony["availability"]["explicit_compound_member_by_tpc"] = True
    fields = build_dense_harmonic_targets(piece, packet)["rows"][0]["fields"]
    assert fields["compound_member"]["value"] == "9"
    assert fields["compound_member"]["origin"] == "source_native"


def test_missing_added_set_does_not_create_nonmember_negative():
    piece, packet = fixture()
    piece = repitch(piece, packet, 62, "D", 0, 2)
    packet["harmonies"][0]["availability"]["added_tones"] = False
    fields = build_dense_harmonic_targets(piece, packet)["rows"][0]["fields"]
    assert fields["core_member"]["available"] and fields["core_member"]["value"] is False
    assert not fields["added_member"]["available"]
    assert not fields["chord_member"]["available"]
    assert not fields["role"]["available"]


@pytest.mark.parametrize("step,alter,tpc,degree", [("F", 1, 6, "#4"), ("G", -1, -6, "b5")])
def test_spelling_is_not_collapsed_to_pitch_class(step, alter, tpc, degree):
    piece, packet = fixture()
    piece = repitch(piece, packet, 66, step, alter, tpc)
    fields = build_dense_harmonic_targets(piece, packet)["rows"][0]["fields"]
    assert fields["interval_from_root_pc"]["value"] == 6
    assert fields["chord_degree"]["value"] == degree


def test_false_native_value_is_available_and_scale_degree_is_independent():
    piece, packet = fixture()
    packet["note_segments"][0]["values"].update(note_degree="5", tpc_is_root=False)
    result = build_dense_harmonic_targets(piece, packet)
    fields = result["rows"][0]["fields"]
    assert fields["source_note_degree"]["value"] == "5"
    assert fields["chord_degree"]["value"] == "1"
    assert fields["source_is_root"]["value"] is False and fields["source_is_root"]["available"]
    batch = collate_dense_harmonic_targets(((piece, result),))
    assert batch.values["source_is_root"].tolist() == [0, -1, -1]
    assert batch.masks["source_is_root"].tolist() == [True, False, False]


def test_overlapping_harmonies_are_explicit_unknown_conflicts():
    piece, packet = fixture()
    packet["harmonies"][1]["start_qn"] = q(1)
    rows = build_dense_harmonic_targets(piece, packet)["rows"]
    conflict = next(r for r in rows if r["start_qn"] == q(1))
    assert conflict["harmony_id"] is None
    assert conflict["harmony_ids"] == ["h0", "h1"]
    assert all(not f["available"] for f in conflict["fields"].values())
    assert conflict["context_status"] == "conflicting_harmony_context"


def test_source_segment_and_repair_gaps_remain_unavailable():
    piece, packet = fixture()
    packet["note_segments"][0]["start_qn"] = q(1)
    packet["note_segments"][0]["binding_available"] = False
    rows = build_dense_harmonic_targets(piece, packet)["rows"]
    assert rows[0]["context_status"] == "no_source_note_coverage"
    assert rows[1]["context_status"] == "source_binding_masked"
    assert all(not f["available"] for r in rows for f in r["fields"].values())


def test_tied_source_rows_keep_one_canonical_note_and_both_native_contexts():
    piece, packet = fixture()
    first = packet["note_segments"][0]
    first["end_qn"] = q(2)
    second = deepcopy(first)
    second.update(source_event_id="l3", start_qn=q(2), end_qn=q(4), harmony_source_index="1")
    second["values"]["tpc_is_root"] = False
    packet["note_segments"].append(second)
    result = build_dense_harmonic_targets(piece, packet)
    assert {r["note_id"] for r in result["rows"]} == {"note:n0"}
    assert [r["fields"]["source_is_root"]["value"] for r in result["rows"]] == [True, False, None]


def test_roundtrip_collation_indices_and_unchanged_raw_graph():
    from music_critic.graph import build_raw_graph, graph_fingerprint

    piece, packet = fixture()
    before = graph_fingerprint(build_raw_graph(piece))
    bundle = build_dense_harmonic_targets(piece, packet)
    restored = loads_dense_harmonic_targets(dumps_dense_harmonic_targets(bundle), piece=piece)
    assert restored == bundle
    batch = collate_dense_harmonic_targets(((piece, restored), (piece, restored)))
    assert batch.local_note_indices.tolist() == [0] * 6
    assert batch.global_note_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert batch.sample_indices.tolist() == [0, 0, 0, 1, 1, 1]
    assert batch.values["role"].tolist() == [0, 0, -1] * 2
    assert batch.masks["role"].tolist() == [True, True, False] * 2
    assert batch.exact_intervals[1] == (RationalTime(2), RationalTime(3))
    assert graph_fingerprint(build_raw_graph(piece)) == before


def test_grace_point_is_retained_without_invented_duration():
    piece, packet = fixture()
    piece = replace(piece, notes=(replace(piece.notes[0], duration_qn=RationalTime(0), is_grace=True),))
    packet["note_segments"][0]["end_qn"] = q(0)
    (row,) = build_dense_harmonic_targets(piece, packet)["rows"]
    assert row["start_qn"] == row["end_qn"] == q(0)
    assert row["context_status"] == "zero_duration_note"
    assert all(not f["available"] for f in row["fields"].values())


def test_interior_grace_point_remains_unknown_without_harmony_assignment():
    piece, packet = fixture()
    note = replace(piece.notes[0], onset_qn=RationalTime(1), duration_qn=RationalTime(0), is_grace=True)
    piece = replace(piece, notes=(note,))
    packet["note_segments"][0].update(start_qn=q(1), end_qn=q(1))
    bundle = build_dense_harmonic_targets(piece, packet)
    (row,) = bundle["rows"]
    assert row["start_qn"] == row["end_qn"] == q(1)
    assert row["harmony_id"] is None and row["harmony_ids"] == []
    assert row["context_status"] == "zero_duration_note"
    assert all(not f["available"] for f in row["fields"].values())
    restored = loads_dense_harmonic_targets(dumps_dense_harmonic_targets(bundle), piece=piece)
    assert restored == bundle
    batch = collate_dense_harmonic_targets(((piece, restored),))
    assert batch.exact_intervals == ((RationalTime(1), RationalTime(1)),)
    assert not any(mask.any().item() for mask in batch.masks.values())


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(piece_id="other-piece"),
    lambda p: p.update(current_split="test"),
    lambda p: p["coordinate_contract"].update(chord_tones="absolute_pc"),
    lambda p: p["harmonies"][0].update(end_qn=q(0)),
    lambda p: p["harmonies"][0].update(start_qn={"num": 0.5, "den": 1}),
    lambda p: p["note_segments"][0].update(note_id="note:missing"),
    lambda p: p["note_segments"][0].update(end_qn=q(5)),
    lambda p: p["note_segments"][0]["values"].update(pitch=65),
    lambda p: p["note_segments"][0]["values"].update(tpc=1),
    lambda p: p["note_segments"][0].update(source_event_id=""),
    lambda p: p["note_segments"][0].update(availability={"tpc": 1}),
    lambda p: p["note_segments"].append(deepcopy(p["note_segments"][0])),
])
def test_invalid_source_binding_or_geometry_fails(mutation):
    piece, packet = fixture()
    mutation(packet)
    with pytest.raises(DenseHarmonicError):
        build_dense_harmonic_targets(piece, packet)


def test_loader_rejects_forged_note_geometry_even_with_matching_ids():
    import json

    piece, packet = fixture()
    bundle = build_dense_harmonic_targets(piece, packet)
    bundle["canonical_notes"][0]["duration_qn"] = q(5)
    bundle["rows"][-1]["end_qn"] = q(5)
    with pytest.raises(DenseHarmonicError, match="geometry"):
        loads_dense_harmonic_targets(json.dumps(bundle), piece=piece)


def polyphonic_fixture():
    piece, packet = fixture()
    template = piece.notes[0]
    notes = (
        replace(template, duration_qn=RationalTime(1)),
        replace(template, note_id="note:n1", onset_qn=RationalTime(2), duration_qn=RationalTime(1), pitch=64, spelling_step="E"),
        replace(template, note_id="note:n2", onset_qn=RationalTime(1, 2), duration_qn=RationalTime(2), pitch=67, spelling_step="G", staff=2),
    )
    segments = []
    for i, (note, tpc) in enumerate(zip(notes, (0, 4, 1))):
        segment = deepcopy(packet["note_segments"][0])
        segment.update(source_event_id=f"l{i}", note_id=note.note_id,
                       start_qn=q(note.onset_qn.num, note.onset_qn.den),
                       end_qn=q((note.onset_qn.to_fraction() + note.duration_qn.to_fraction()).numerator,
                                (note.onset_qn.to_fraction() + note.duration_qn.to_fraction()).denominator),
                       part_id="p1" if i < 2 else "p2", staff=note.staff)
        segment["values"].update(pitch=note.pitch, tpc=tpc)
        segments.append(segment)
    packet["note_segments"] = segments
    return replace(piece, notes=(notes[0], notes[2], notes[1])), packet


def test_physical_voice_and_interpart_links_have_exact_geometry_and_numeric_batch():
    piece, packet = polyphonic_fixture()
    bundle = build_dense_harmonic_targets(piece, packet)
    voice = bundle["voice_links"]
    assert len(voice) == 3
    assert voice[0]["available"] and voice[0]["target_note_id"] == "note:n1"
    assert voice[0]["pitch_difference"] == 4
    assert voice[0]["gap_qn"] == q(1)
    assert voice[2]["reason"] == "no_successor"
    sounding = bundle["interpart_links"]
    assert [(row["start_qn"], row["end_qn"]) for row in sounding] == [(q(1, 2), q(1)), (q(2), q(5, 2))]
    assert all(row["origin"] == "source_derived" for row in sounding)
    batch = collate_dense_harmonic_targets(((piece, bundle), (piece, bundle)))
    assert batch.voice_links.source_note_indices.tolist() == [0, 1, 2, 3, 4, 5]
    assert batch.voice_links.target_note_indices.tolist() == [2, -1, -1, 5, -1, -1]
    assert batch.voice_links.masks.tolist() == [True, False, False] * 2
    assert batch.interpart_links.pitch_differences.tolist() == [7, -3] * 2


@pytest.mark.parametrize("change", ["masked", "voice_mismatch", "part_missing", "coverage_gap"])
def test_untrusted_source_identity_cannot_produce_physical_links(change):
    piece, packet = polyphonic_fixture()
    segment = packet["note_segments"][1]
    if change == "masked":
        segment["binding_available"] = False
    elif change == "voice_mismatch":
        segment["voice"] = 9
    elif change == "part_missing":
        segment["part_id"] = None
    else:
        segment["end_qn"] = q(5, 2)
    bundle = build_dense_harmonic_targets(piece, packet)
    assert not bundle["source_note_identities"][2]["voice_available"]
    assert all(not r["available"] or r["target_note_id"] != "note:n1" for r in bundle["voice_links"])
    assert all("note:n1" not in (r["source_note_id"], r["target_note_id"]) for r in bundle["interpart_links"])


def test_same_onset_polyphony_is_ambiguous_for_next_note_not_an_arbitrary_edge():
    piece, packet = polyphonic_fixture()
    extra = replace(piece.notes[2], note_id="note:n3", pitch=67, spelling_step="G")
    segment = deepcopy(packet["note_segments"][1])
    segment.update(note_id=extra.note_id, source_event_id="l3")
    segment["values"].update(pitch=67, tpc=1)
    packet["note_segments"].append(segment)
    bundle = build_dense_harmonic_targets(replace(piece, notes=(*piece.notes, extra)), packet)
    assert bundle["voice_links"][0]["reason"] == "ambiguous_next_onset"
    assert not bundle["voice_links"][0]["available"]


def test_loader_rejects_forged_physical_link():
    import json

    piece, packet = polyphonic_fixture()
    bundle = build_dense_harmonic_targets(piece, packet)
    bundle["interpart_links"][0]["end_qn"] = q(2)
    with pytest.raises(DenseHarmonicError, match="physical link"):
        loads_dense_harmonic_targets(json.dumps(bundle), piece=piece)


def test_explicit_tpc_mask_blocks_derivation_but_not_independent_harmony_fields():
    piece, packet = fixture()
    packet["note_segments"][0]["availability"] = {"tpc": False, "valid_chord_label": False}
    fields = build_dense_harmonic_targets(piece, packet)["rows"][0]["fields"]
    for field in ("role", "chord_member", "chord_degree", "key_degree", "source_is_root"):
        assert not fields[field]["available"]
    assert fields["root_pc"]["available"]


@pytest.mark.parametrize("mutation", [
    lambda b: b["harmonies"][0].update(end_qn=q(1)),
    lambda b: b["rows"][0].update(harmony_id="missing"),
    lambda b: b.update(current_split="test"),
    lambda b: b.update(time_unit="measured_seconds"),
    lambda b: b["coordinate_contract"].update(tpc="pitch_class"),
    lambda b: b["harmonies"].append(deepcopy(b["harmonies"][0])),
    lambda b: b["source_note_identities"][0].update(voice_available="true"),
])
def test_loader_rejects_corrupted_context_contract(mutation):
    import json

    piece, packet = fixture()
    bundle = build_dense_harmonic_targets(piece, packet)
    mutation(bundle)
    with pytest.raises(DenseHarmonicError):
        loads_dense_harmonic_targets(json.dumps(bundle), piece=piece)


def test_source_non_quarter_note_clock_is_rejected():
    piece, packet = fixture()
    packet["time_unit"] = "measured_seconds"
    with pytest.raises(DenseHarmonicError, match="time unit"):
        build_dense_harmonic_targets(piece, packet)
