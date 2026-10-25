"""Synthetic packet geometry and provenance checks, without corpus reads."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import pytest

from scripts.build_corpus400_analysis_packets import build_analysis_packets, render_segment_markdown, write_analysis_packets


def q(value):
    value = Fraction(value)
    return {"num": value.numerator, "den": value.denominator}


def inspection(count=12):
    duration = max(16, (count + 3) // 4 * 4)
    return {"schema": "Corpus400FullInspectionPacket@1", "candidate_id": "candidate:synthetic",
        "work_family_id": "family:synthetic", "primary_genre": "classical", "identity": {},
        "canonical_piece": {"path": "synthetic.json", "sha256": "a" * 64, "size_bytes": 1},
        "native_theory": None, "duration_qn": q(duration), "note_count": count,
        "inspection_scope": "entire_materialized_file", "cropped": False, "all_notes_included": True,
        "tracks": [{"track_id": "track:0", "name": "Synthetic"}],
        "bars": [{"bar_id": f"bar:{i}", "start_qn": q(i * 4), "duration_qn": q(4)} for i in range(duration // 4)],
        "meter_events": [], "key_signature_events": [], "source_extent": {"status": "unverified"},
        "note_columns": ["note_id", "track_id", "onset_qn", "duration_qn", "pitch", "is_percussion", "is_grace"],
        "note_rows": [[f"note:{i}", "track:0", q(i), q(1), 60 + i % 12, False, False] for i in range(count)]}


def test_all_notes_have_exactly_one_core_owner_and_all_time_is_partitioned():
    result = build_analysis_packets(inspection(), max_notes=5, context_qn="1")
    segments = result["segments"]
    assert len(segments) > 1
    assert Counter(n for s in segments for n in s["core_note_ids"]) == Counter(f"note:{i}" for i in range(12))
    assert all(len(s["note_rows"]) <= 5 for s in segments)
    intervals = [(Fraction(p["start_qn"]["num"], p["start_qn"]["den"]),
                  Fraction(p["end_qn"]["num"], p["end_qn"]["den"])) for p in result["partitions"]]
    assert intervals[0][0] == 0 and intervals[-1][1] == 16
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    assert result["overview"]["annotation_complete"] is False


def test_large_simultaneous_onset_uses_bounded_shards_without_losing_notes():
    packet = inspection(12)
    for row in packet["note_rows"]: row[2] = q(0)
    result = build_analysis_packets(packet, max_notes=5, context_qn="0")
    assert len(result["partitions"]) == 1
    assert len(result["segments"]) >= 3
    assert all(len(s["note_rows"]) <= 5 for s in result["segments"])
    assert Counter(n for s in result["segments"] for n in s["core_note_ids"]) == Counter(f"note:{i}" for i in range(12))
    assert any(s["context_note_coverage"]["omitted_ids"] for s in result["segments"])


def native(packet):
    fields = {"source.zero": {"value": 0, "available": True, "origin": "source_native",
        "confidence": None, "evidence": ["fixture:0"], "unknown_reason": None},
        "source.false": {"value": False, "available": True, "origin": "source_native",
        "confidence": None, "evidence": ["fixture:false"], "unknown_reason": None},
        "source.missing": {"value": None, "available": False, "origin": "unknown",
        "confidence": None, "evidence": [], "unknown_reason": "not supplied"}}
    return {"schema": "EventTheoryEnrichment@1", "current_split": "train",
        "canonical_piece": deepcopy(packet["canonical_piece"]),
        "harmonies": [{"id": "h:1", "start_qn": q(0), "end_qn": packet["duration_qn"], "fields": fields}],
        "note_contexts": [{"id": f"ctx:{r[0]}", "note_id": r[0], "harmony_id": "h:1", "view_id": "native:a",
            "start_qn": r[2], "end_qn": q(Fraction(r[2]["num"], r[2]["den"]) + Fraction(r[3]["num"], r[3]["den"])),
            "fields": deepcopy(fields)} for r in packet["note_rows"]], "objects": [], "relations": []}


def test_native_field_tokens_preserve_false_zero_missing_and_view_without_mutation():
    packet = inspection(); bundle = native(packet); before = deepcopy((packet, bundle))
    result = build_analysis_packets(packet, bundle, max_notes=5)
    seen = {}
    for segment in result["segments"]:
        for row in segment["native"]["note_contexts"]:
            restored = {k: v for k, v in row.items() if k != "fieldset_id"}
            restored["fields"] = segment["native_fieldsets"][row["fieldset_id"]]
            seen[row["id"]] = restored
    assert list(sorted(seen.values(), key=lambda r: r["id"])) == sorted(bundle["note_contexts"], key=lambda r: r["id"])
    assert (packet, bundle) == before


def test_sustained_note_and_fractional_context_keep_original_intervals():
    packet = inspection()
    packet["note_rows"][0][3] = q(Fraction(25, 3))
    bundle = native(packet)
    result = build_analysis_packets(packet, bundle, max_notes=5, context_qn="1/3")
    matches = [(s, row) for s in result["segments"] for row in s["note_rows"] if row[0] == "note:0"]
    assert len(matches) > 1
    assert all(row[3] == q(Fraction(25, 3)) for _, row in matches)
    assert sum("note:0" in s["core_note_ids"] for s in result["segments"]) == 1
    assert any("note:0" in s["core_lifetimes_beyond_context_ids"] for s in result["segments"])


def test_markdown_and_index_include_events_after_180_and_every_native_field():
    packet = inspection(201); result = build_analysis_packets(packet, native(packet), max_notes=70)
    rendered = "\n".join(render_segment_markdown(s) for s in result["segments"])
    assert "note:200" in rendered and "source.zero" in rendered and "source.missing" in rendered
    assert result["overview"]["all_notes_have_one_core_owner"] is True
    assert len(result["entity_index"]["notes"]) == 201


@pytest.mark.parametrize("change,reason", [
    (lambda p: p.update(cropped=True), "full inspection"),
    (lambda p: p["note_rows"].append(p["note_rows"][0]), "duplicate"),
    (lambda p: p["note_rows"][0].__setitem__(2, q(-1)), "outside"),
])
def test_invalid_inspection_is_rejected(change, reason):
    packet = inspection(); change(packet)
    with pytest.raises(ValueError, match=reason): build_analysis_packets(packet)


def test_native_binding_or_note_identity_mismatch_rejected():
    packet = inspection(); bundle = native(packet)
    bundle["canonical_piece"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="binding"): build_analysis_packets(packet, bundle)
    bundle = native(packet); bundle["note_contexts"][0]["note_id"] = "unknown"
    with pytest.raises(ValueError, match="context note"): build_analysis_packets(packet, bundle)


def test_unbarred_silence_and_terminal_grace_are_not_lost():
    packet = inspection(1); packet["bars"] = []
    packet["note_rows"].append(["note:grace", "track:0", q(16), q(0), 60, False, True]); packet["note_count"] = 2
    result = build_analysis_packets(packet, max_notes=1, context_qn="0")
    assert Counter(n for s in result["segments"] for n in s["core_note_ids"]) == Counter(["note:0", "note:grace"])
    assert all(not b["source_bar"] for s in result["segments"] for b in s["bar_timeline"])
    assert result["partitions"][-1]["end_qn"] == q(16)


def test_output_is_immutable_and_every_segment_has_json_markdown_and_binding(tmp_path):
    result = build_analysis_packets(inspection(), max_notes=5)
    summary = write_analysis_packets(tmp_path / "out", result)
    assert summary["annotation_complete"] is False
    rows = [json.loads(line) for line in (tmp_path / "out/segments/index.jsonl").read_text().splitlines()]
    assert len(rows) == len(result["segments"])
    assert all(Path(row["json"]["path"]).is_file() and Path(row["markdown"]["path"]).is_file() for row in rows)
    with pytest.raises(FileExistsError): write_analysis_packets(tmp_path / "out", result)


def test_markdown_uses_declared_note_id_column_without_collapsing_same_track():
    packet = inspection()
    packet["note_columns"][0], packet["note_columns"][1] = packet["note_columns"][1], packet["note_columns"][0]
    for row in packet["note_rows"]: row[0], row[1] = row[1], row[0]
    result = build_analysis_packets(packet)
    rendered = render_segment_markdown(result["segments"][0])
    assert all(f"CORE note:{i} |" in rendered for i in range(12))


def test_registered_index_hash_gate_rejects_before_reading_packet(tmp_path, monkeypatch):
    from music_critic.experiments.registry import seal_record, write_record_once
    from tests.experiments.test_registry import _record
    from scripts.build_corpus400_analysis_packets import run
    monkeypatch.chdir(tmp_path)
    index = tmp_path / "materialized.jsonl"
    index.write_text(json.dumps({"candidate_id": "one", "status": "raw_materialized",
        "inspection_packet": {"path": str(tmp_path / "never-opened"), "sha256": "a" * 64, "size_bytes": 1}}))
    base = _record(experiment_id="EXP-SYNTHETIC-PACKETS-001", status="planned")
    base.update(phase="tonal-relations-T08", kind="corpus400_analysis_packetization",
        protocol={"materialized_index": {"path": index.name, "sha256": "0" * 64, "size_bytes": index.stat().st_size}})
    record = tmp_path / "EXP-SYNTHETIC-PACKETS-001.json"
    write_record_once(record, seal_record(base))
    with pytest.raises(ValueError, match="binding hash"):
        run(index, tmp_path / "out", record)
    assert not (tmp_path / "out").exists()


def test_registered_synthetic_runtime_preserves_extras_binding_without_opening_it(tmp_path, monkeypatch):
    from hashlib import sha256
    from music_critic.experiments.registry import seal_record, write_record_once
    from tests.experiments.test_registry import _record
    from scripts.build_corpus400_analysis_packets import run
    monkeypatch.chdir(tmp_path)
    packet = inspection()
    packet["source_analytical_extras"] = {"path": "not-opened.json", "sha256": "b" * 64, "size_bytes": 1}
    raw = json.dumps(packet).encode(); Path("inspection.json").write_bytes(raw)
    packet_binding = {"path": "inspection.json", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}
    row = {"candidate_id": packet["candidate_id"], "status": "raw_materialized", "inspection_packet": packet_binding,
        "canonical_piece": packet["canonical_piece"], "native_theory": None,
        "source_analytical_extras": packet["source_analytical_extras"]}
    raw = (json.dumps(row) + "\n").encode(); Path("index.jsonl").write_bytes(raw)
    record = _record(experiment_id="EXP-SYNTHETIC-PACKETS-001", status="planned")
    record.update(phase="tonal-relations-T08", kind="corpus400_analysis_packetization",
        protocol={"materialized_index": {"path": "index.jsonl", "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}})
    write_record_once("EXP-SYNTHETIC-PACKETS-001.json", seal_record(record))
    result = run("index.jsonl", "out", "EXP-SYNTHETIC-PACKETS-001.json", max_notes=5)
    assert result["success"] == 1 and result["failed"] == 0 and result["annotation_complete"] == 0
    output_row = json.loads(Path("out/index.jsonl").read_text())
    overview = json.loads(Path(output_row["overview"]["path"]).read_text())
    assert overview["source_analytical_extras"] == packet["source_analytical_extras"]


def test_direct_cli_help():
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "scripts/build_corpus400_analysis_packets.py", "--help"],
        cwd=Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--materialized-index" in result.stdout
