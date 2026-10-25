"""Synthetic Humdrum examples; no corpus music is embedded here."""
from fractions import Fraction
from dataclasses import replace

import pytest

from music_critic.adapters.tavern import parse_joined, TavernError, canonical_piece, build_enrichment, project_score_spines, raw_signature
from music_critic.data import validate_piece
from music_critic.tasks.event_theory_enrichment import validate_enrichment, build_vocabulary, encode_enrichment


def test_exact_triplets_and_native_function_context():
    result = parse_joined(
        "**function\t**harm\t**kern\n*M4/4\t*M4/4\t*M4/4\n"
        "*C:\t*C:\t*C:\n2T\t2I\t6c\n.\t.\t6e\n.\t.\t6g\n*-\t*-\t*-\n",
        source_path="Beethoven/B063/Joined/B063_00_01a_a.krn", source_sha256="a" * 64,
    )
    assert [(n.onset, n.duration, n.pitch) for n in result.notes] == [
        (Fraction(0), Fraction(2, 3), 60),
        (Fraction(2, 3), Fraction(2, 3), 64),
        (Fraction(4, 3), Fraction(2, 3), 67),
    ]
    assert result.functions[0].label == "T"
    assert result.functions[0].end == 2
    assert result.harmonies[0].label == "I"


def test_unsupported_spines_fail_closed():
    with pytest.raises(TavernError, match="unsupported"):
        parse_joined("**kern\t**foo\n4c\t1x\n*-\t*-\n",
                     source_path="x", source_sha256="a" * 64)


def parse(text):
    return parse_joined(text, source_path="synthetic.krn", source_sha256="b" * 64)


def test_tied_note_retains_attack_across_native_function_change_and_grace():
    score = parse("**kern\n*M2/4\n=1\n[4c\n4c]\n=2\n8dq\n2e\n==\n*-\n")
    joined = parse("**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n"
                   "*C:\t*C:\t*C:\n=1\t=1\t=1\n4T\t4I\t[4c\n4D\t4V\t4c]\n"
                   "=2\t=2\t=2\n.\t.\t8dq\n2T\t2I\t2e\n==\t==\t==\n*-\t*-\t*-\n")
    piece = canonical_piece(score, phrase_id="synthetic", family_id="family:synthetic")
    assert len(piece.notes) == 3
    assert piece.notes[0].duration_qn.num == 2
    bundle = build_enrichment(piece, [("a", joined), ("b", joined)], canonical_binding={}, source_bindings=[], admission={})
    validate_enrichment(piece, bundle)
    rows = [r for r in bundle["note_contexts"] if r["note_id"] == piece.notes[0].note_id and r["view_id"] == "a"]
    assert [r["fields"]["tavern.phrase_function"]["value"] for r in rows] == ["T", "D"]
    assert all(not r["fields"]["tavern.note_function"]["available"] for r in rows)
    encoded = encode_enrichment(piece, bundle, build_vocabulary([bundle]))
    assert not encoded["oov_counts"]


def test_spine_split_merge_and_orphan_tie_guard():
    score = parse("**kern\n*M2/4\n=1\n*^\n4c\t4e\n4d\t4f\n*v\t*v\n==\n*-\n")
    assert len(score.notes) == 4
    with pytest.raises(TavernError, match="orphan"):
        parse("**kern\n*M2/4\n2c]\n*-\n")


def test_no_extend_native_label_across_unlabelled_region():
    score = parse("**kern\n*M2/4\n2c\n*-\n")
    joined = parse("**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n4T\t4I\t2c\n.\t.\t.\n*-\t*-\t*-\n")
    piece = canonical_piece(score, phrase_id="gap", family_id="family:synthetic")
    bundle = build_enrichment(piece, [("a", joined)], canonical_binding={}, source_bindings=[], admission={})
    validate_enrichment(piece, bundle)
    assert len(bundle["note_contexts"]) == 2
    assert not bundle["note_contexts"][1]["fields"]["tavern.phrase_function"]["available"]


def test_durationless_grace_and_same_pitch_order():
    score = parse("**kern\n*M2/4\nccq\n*^\n2c\t4c\n.\t4e\n*v\t*v\n*-\n")
    assert score.notes[0].duration == 0
    piece = canonical_piece(score, phrase_id="durationless-grace", family_id="family:synthetic")
    assert [n.duration_qn.num for n in piece.notes if n.pitch == 60] == [1, 2]


def test_exact_fractional_pickup_requires_source_provenance_and_retains_dilemma_exception():
    piece = canonical_piece(parse("**kern\n*M4/4\n8c\n=1\n1e\n*-\n"),
                            phrase_id="pickup", family_id="family:synthetic")
    assert piece.bars[0].metric_offset_qn.to_fraction() == Fraction(7, 2)
    assert validate_piece(piece).is_valid
    source = piece.provenance[1]
    without = replace(piece, provenance=(piece.provenance[0], replace(source, details=())))
    assert not validate_piece(without).is_valid
    dilemmadata = replace(piece, provenance=(piece.provenance[0], replace(source,
        source="music_critic.dilemmadata_adapter", details=(("raw_repair_evidence_version", "1.2.0"),))))
    assert validate_piece(dilemmadata).is_valid


def test_projection_reparses_without_analysis_only_rows_and_preserves_source_lines():
    text = "**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n4T\t4I\t2c\n4D\t4V\t.\n*-\t*-\t*-\n"
    projected, lines = project_score_spines(text)
    assert "4D" not in projected and "4T" not in projected
    assert lines == [1, 2, 3, 5]
    assert raw_signature(parse(projected)) == raw_signature(parse(text))
    assert parse(projected).duration == parse(text).duration == 2


def test_trailing_grace_gets_explicit_unknown_point_context():
    score = parse("**kern\n*M2/4\n2c\ngq\n*-\n")
    joined = parse("**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n2T\t2I\t2c\n.\t.\tgq\n*-\t*-\t*-\n")
    piece = canonical_piece(score, phrase_id="trailing-grace", family_id="family:synthetic")
    assert all(n.staff is None for n in piece.notes)
    bundle = build_enrichment(piece, [("a", joined)], canonical_binding={}, source_bindings=[], admission={})
    validate_enrichment(piece, bundle)
    assert bundle["note_contexts"][-1]["harmony_id"] is None
    assert not bundle["note_contexts"][-1]["fields"]["tavern.phrase_function"]["available"]


def test_annotator_is_view_metadata_and_never_a_categorical_target():
    score = parse("**kern\n*M2/4\n2c\n*-\n")
    joined = parse("**function\t**harm\t**kern\n*M2/4\t*M2/4\t*M2/4\n2T\t2I\t2c\n*-\t*-\t*-\n")
    piece = canonical_piece(score, phrase_id="annotator-metadata", family_id="family:synthetic")
    bundle = build_enrichment(piece, [("a", joined), ("b", joined)],
                              canonical_binding={}, source_bindings=[], admission={})
    for scope in ("harmonies", "note_contexts"):
        assert {row["view_id"] for row in bundle[scope]} == {"a", "b"}
        assert all("tavern.annotator" not in row["fields"] for row in bundle[scope])
    assert [view["annotator"] for view in bundle["provenance"]["native_analyses"]] == ["a", "b"]
    vocabulary = build_vocabulary([bundle])
    encoded = encode_enrichment(piece, bundle, vocabulary)
    for scope in ("harmonies", "note_contexts"):
        assert "tavern.annotator" not in vocabulary["fields"][scope]
        assert "tavern.annotator" not in encoded[scope]["values"]
        assert "tavern.annotator" not in encoded[scope]["masks"]
