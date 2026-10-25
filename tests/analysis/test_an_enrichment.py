"""Synthetic AN projection witnesses; no production corpus access."""
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import pytest

from music_critic.adapters.dilemmadata import (
    DilemmadataRawTargetAlignmentEvidence, DilemmadataSourceRowBinding,
    _alignment_evidence_fingerprint,
)
from music_critic.analysis.an_enrichment import build_an_enrichment
from music_critic.data import RationalTime, dumps_piece, load_piece
from music_critic.tasks.event_theory_enrichment import build_vocabulary, encode_enrichment


def fixture():
    p = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    n = replace(p.notes[0], note_id="note:an", onset_qn=RationalTime(0),
                duration_qn=RationalTime(2), pitch=60, is_percussion=False)
    p = replace(p, notes=(n,), targets=(), annotations=(), split="train")
    digest = sha256(dumps_piece(p).encode()).hexdigest()
    e = DilemmadataRawTargetAlignmentEvidence("1.1.0", "an:training:one", p.piece_id,
        "b" * 64, "c" * 64, digest, ("s_midi",),
        (DilemmadataSourceRowBinding(0, 2, n.onset_qn, n.note_id, False),), "")
    e = replace(e, fingerprint=_alignment_evidence_fingerprint(e))
    rows = [dict(s_offset_frac="0", s_duration_frac="2", s_midi="60", s_isOnset="True",
        valid_chord_label="True", a_annotationNumber="0", a_romanNumeral="V7/V",
        a_isOnset="False", note_degree="b7", a_degree2="None")]
    admission = dict(status="admitted", record_id=e.record_id, current_split="train",
                     native_split="training", source_component_id=p.source_group_id)
    kwargs = dict(canonical_binding=dict(path="piece.json", sha256=digest, size_bytes=1),
                  source_bindings=[])
    return p, e, rows, admission, kwargs


def build(args):
    return build_an_enrichment(*args[:4], **args[4])


def test_native_facts_do_not_invent_harmony_or_boundary_negatives():
    b = build(fixture())
    assert not b["harmonies"] and not b["relations"]
    c, = b["note_contexts"]
    assert c["applicability"] == "source_note_annotation"
    assert c["fields"]["an.a_romanNumeral"]["value"] == "V7/V"
    assert c["fields"]["an.a_romanNumeral"]["origin"] == "source_native"
    assert c["fields"]["an.a_isOnset"]["unknown_reason"] == "native_boundary_positive_unlabeled"
    assert c["fields"]["an.a_degree2"]["available"] is False


@pytest.mark.parametrize("name,value", [("native_split", "test"), ("native_split", "validation"),
                                       ("current_split", "test"), ("status", "excluded")])
def test_dual_train_guard(name, value):
    args = fixture(); args[3][name] = value
    with pytest.raises(ValueError, match="TRAIN"): build(args)


def test_missing_tail_keeps_independent_unknown_context():
    args = fixture(); args[2][0]["s_duration_frac"] = "1"
    b = build(args)
    assert len(b["note_contexts"]) == 2
    tail = b["note_contexts"][1]
    assert tail["start_qn"] == {"num": 1, "den": 1}
    assert all(not x["available"] for x in tail["fields"].values())


@pytest.mark.parametrize("name,value", [("s_midi", "61"), ("s_duration_frac", "3"),
                                       ("s_offset_frac", "1/2"), ("s_isOnset", "False")])
def test_corrupt_row_binding_rejected(name, value):
    args = fixture(); args[2][0][name] = value
    with pytest.raises(ValueError): build(args)


def test_invalid_source_and_repair_masks_are_not_known_labels():
    args = fixture(); args[2][0]["valid_chord_label"] = "False"
    assert not any(f["available"] for f in build(args)["note_contexts"][0]["fields"].values())
    p, e, rows, a, kw = fixture()
    e = replace(e, rows=(replace(e.rows[0], repair_mask_scope="note"),))
    e = replace(e, fingerprint=_alignment_evidence_fingerprint(e))
    assert not any(f["available"] for f in build((p,e,rows,a,kw))["note_contexts"][0]["fields"].values())


def test_numeric_uses_masks_and_exact_note_ids():
    args = fixture(); b = build(args); vocab = build_vocabulary([b])
    numeric = encode_enrichment(args[0], b, vocab)
    assert not numeric["oov_counts"]
    assert numeric["note_contexts"]["note_indices"] == [0]
    assert numeric["note_contexts"]["masks"]["an.a_romanNumeral"] == [True]
    assert numeric["note_contexts"]["masks"]["an.a_degree2"] == [False]
    assert numeric["note_contexts"]["values"]["an.a_degree2"] == [-1]


def test_exact_fraction_offset_and_point_role_mask():
    p,e,rows,a,kw = fixture()
    p = replace(p, notes=(replace(p.notes[0], onset_qn=RationalTime(1,3),
                                 duration_qn=RationalTime(0), is_grace=True),))
    digest = sha256(dumps_piece(p).encode()).hexdigest()
    e = replace(e, canonical_piece_sha256=digest,
        rows=(replace(e.rows[0], onset_qn=RationalTime(1,3)),))
    e = replace(e, fingerprint=_alignment_evidence_fingerprint(e))
    rows[0]["s_duration_frac"]="0"; kw["canonical_binding"]["sha256"] = digest
    kw["offset"] = Fraction(1,3)
    b = build((p,e,rows,a,kw)); c, = b["note_contexts"]
    assert c["start_qn"] == c["end_qn"] == {"num":1,"den":3}
    assert not c["fields"]["an.note_degree"]["available"]
