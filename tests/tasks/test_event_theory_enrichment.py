"""Synthetic contracts for event-bound auxiliary targets, not music accuracy."""
from copy import deepcopy
from dataclasses import replace

import pytest

from music_critic.data import RationalTime
from tests.tasks.test_multisource_contract import _hook_piece
from music_critic.tasks.event_theory_enrichment import (
    EnrichmentError, build_vocabulary, encode_enrichment, validate_enrichment,
    load_enrichment,
    merge_interpretations,
)


def fixture():
    base = _hook_piece()
    note = replace(base.notes[0], note_id="note:n0", onset_qn=RationalTime(0),
                   duration_qn=RationalTime(4))
    piece = replace(base, notes=(note,), split="train", annotations=(), targets=())
    def field(value, available=True):
        return dict(value=value, available=available, origin="source_native" if available else "unknown",
                    confidence=None, evidence=["fixture:1"], unknown_reason=None if available else "not_annotated")
    bundle = dict(schema="EventTheoryEnrichment@1", dataset_id=piece.dataset_name,
                  piece_id=piece.piece_id, current_split="train", harmonies=[], relations=[],
                  note_contexts=[dict(id="c0", note_id="note:n0", start_qn={"num":0,"den":1},
                                      end_qn={"num":2,"den":1}, harmony_id=None,
                                      fields={"native.member":field(False)}),
                                 dict(id="c1", note_id="note:n0", start_qn={"num":2,"den":1},
                                      end_qn={"num":4,"den":1}, harmony_id=None,
                                      fields={"native.member":field(None,False)})])
    return piece, bundle


def test_false_is_known_and_unknown_is_minus_one_without_changing_notes():
    piece, bundle = fixture()
    before = deepcopy(bundle)
    validate_enrichment(piece, bundle)
    vocab = build_vocabulary([bundle])
    encoded = encode_enrichment(piece, bundle, vocab)
    assert encoded["note_contexts"]["note_indices"] == [0,0]
    assert encoded["note_contexts"]["values"]["native.member"] == [0,-1]
    assert encoded["note_contexts"]["masks"]["native.member"] == [True,False]
    assert encoded["note_contexts"]["intervals"][1][0] == {"num":2,"den":1}
    assert bundle == before


def test_harmony_only_evidence_is_not_marked_eligible_for_raw_note_training():
    piece,bundle=fixture()
    piece=replace(piece,notes=())
    bundle['note_contexts']=[]
    encoded=encode_enrichment(piece,bundle,build_vocabulary([bundle]))
    assert encoded['raw_event_train_eligible'] is False


@pytest.mark.parametrize("change,match", [
    (lambda b: b.update(current_split="test"), "TRAIN"),
    (lambda b: b.update(piece_id="other"), "identity"),
    (lambda b: b["note_contexts"][0].update(note_id="unknown"), "unknown canonical"),
    (lambda b: b["note_contexts"][0].update(end_qn={"num":5,"den":1}), "outside canonical"),
    (lambda b: b["note_contexts"][0].update(end_qn={"num":4,"den":2}), "reduced"),
    (lambda b: b["note_contexts"][0].update(end_qn=2.0), "rational"),
    (lambda b: b["note_contexts"].pop(), "tail"),
    (lambda b: b["note_contexts"][1].update(start_qn={"num":3,"den":1}), "gap"),
    (lambda b: b["note_contexts"][0]["fields"]["native.member"].update(origin="algorithmic_candidate"), "candidate"),
    (lambda b: b["note_contexts"][1]["fields"]["native.member"].update(value=False), "must be null"),
])
def test_invalid_supervision_is_rejected(change, match):
    piece, bundle = fixture()
    change(bundle)
    with pytest.raises(EnrichmentError, match=match):
        validate_enrichment(piece, bundle)


def test_frozen_vocabulary_masks_oov_and_detects_tampering():
    piece, bundle = fixture()
    vocab = build_vocabulary([bundle])
    bundle["note_contexts"][0]["fields"]["native.member"]["value"] = True
    encoded = encode_enrichment(piece, bundle, vocab)
    assert encoded["note_contexts"]["values"]["native.member"] == [-1,-1]
    assert encoded["oov_counts"] == {"note_contexts:native.member":1}
    vocab["fields"]["note_contexts"]["native.member"].append(True)
    with pytest.raises(EnrichmentError, match="fingerprint"):
        encode_enrichment(piece, bundle, vocab)


def test_false_and_integer_zero_have_different_classes():
    piece, bundle = fixture()
    second = deepcopy(bundle)
    second["note_contexts"][0]["fields"]["native.member"]["value"] = 0
    vocab = build_vocabulary([bundle, second])
    assert encode_enrichment(piece,bundle,vocab)["note_contexts"]["values"]["native.member"][0] != \
           encode_enrichment(piece,second,vocab)["note_contexts"]["values"]["native.member"][0]


def test_gzip_bytes_are_authenticated_before_decoding(tmp_path):
    import gzip, hashlib, json
    _, bundle = fixture()
    path = tmp_path/"piece.json.gz"
    raw = gzip.compress(json.dumps(bundle).encode())
    path.write_bytes(raw)
    assert load_enrichment(path,expected_sha256=hashlib.sha256(raw).hexdigest()) == bundle
    with pytest.raises(EnrichmentError,match="hash"):
        load_enrichment(path,expected_sha256="0"*64)


def test_relations_have_numeric_endpoints_and_preserve_explicit_false():
    piece,bundle = fixture()
    relation = dict(id="r0", type="prepares", source_ids=["note:n0"], target_ids=["c1"],
                    start_qn={"num":0,"den":1}, end_qn={"num":4,"den":1},
                    value=False, available=True, origin="source_native", confidence=None,
                    evidence=["fixture"], unknown_reason=None)
    bundle["relations"] = [relation]
    encoded = encode_enrichment(piece,bundle,build_vocabulary([bundle]))
    assert encoded["relations"]["source_indices"] == [[0]]
    assert encoded["relations"]["target_indices"] == [[2]]
    assert encoded["relations"]["values"] == [0]
    assert encoded["relations"]["masks"] == [True]
    bundle["relations"][0]["target_ids"] = ["missing"]
    with pytest.raises(EnrichmentError,match="participant"):
        validate_enrichment(piece,bundle)


def test_object_geometry_and_view_identity_survive_numeric_export():
    piece,bundle=fixture()
    bundle['note_contexts'][0]['harmony_view']='analyst:a'
    bundle['objects']=[dict(id='phrase:0',kind='phrase',member_ids=['c0','c1'],
                            start_qn={'num':0,'den':1},end_qn={'num':4,'den':1})]
    numeric=encode_enrichment(piece,bundle,build_vocabulary([bundle]))
    assert numeric['note_contexts']['view_ids']==['analyst:a',None]
    obj=next(row for row in numeric['participants'] if row['id']=='phrase:0')
    assert obj['object_type']=='phrase' and obj['member_ids']==['c0','c1']
    assert obj['interval']==[{'num':0,'den':1},{'num':4,'den':1}]
    assert obj['member_indices']==[1,2]


@pytest.mark.parametrize('location', ['object','relation','harmony','relation_scope','object_members'])
def test_analysis_scope_cannot_escape_piece_or_participants(location):
    piece,bundle=fixture()
    bundle['objects']=[dict(id='phrase:0',start_qn={'num':0,'den':1},end_qn={'num':4,'den':1})]
    bundle['relations']=[dict(id='r0',type='supports',source_ids=['note:n0'],target_ids=['phrase:0'],
        start_qn={'num':0,'den':1},end_qn={'num':4,'den':1},value=True,available=True,
        origin='source_native',confidence=None,evidence=['fixture'],unknown_reason=None)]
    if location=='object':bundle['objects'][0]['end_qn']={'num':9999,'den':1}
    elif location=='relation':bundle['relations'][0]['end_qn']={'num':9999,'den':1}
    elif location=='relation_scope':bundle['relations'][0]['start_qn']={'num':1,'den':1}
    elif location=='object_members':bundle['objects'][0]['member_ids']=['missing']
    else:bundle['harmonies']=[dict(id='h0',start_qn={'num':-1,'den':1},end_qn={'num':1,'den':1},fields={})]
    with pytest.raises(EnrichmentError):validate_enrichment(piece,bundle)


def test_authored_overlay_keeps_native_fields_and_cannot_relabel_candidates():
    piece,bundle=fixture()
    overlay=dict(schema="EventTheoryInterpretations@1",piece_id=piece.piece_id,
                 model_requested="gpt-6-astra",reasoning_effort_requested="ultra",human_gold=False,
                 base_sha256="a"*64,harmony_fields=[],objects=[],relations=[],
                 note_fields=[dict(context_id="c0",fields={"analysis.role":dict(
                     value="stable_support",available=True,origin="astra_individual_context",confidence=None,
                     evidence=["individual context review"],unknown_reason=None)})])
    result=merge_interpretations(piece,bundle,overlay,base_sha256="a"*64)
    assert result["note_contexts"][0]["fields"]["analysis.role"]["value"]=="stable_support"
    assert "analysis.role" not in bundle["note_contexts"][0]["fields"]
    assert result["note_contexts"][0]["fields"]["native.member"]["value"] is False
    bundle["note_contexts"][0]["fields"]["analysis.role"] = dict(
        value=None,available=False,origin="unknown",confidence=None,evidence=[],unknown_reason="not_reviewed")
    assert merge_interpretations(piece,bundle,overlay,base_sha256="a"*64)["note_contexts"][0]["fields"]["analysis.role"]["available"]
    known=deepcopy(bundle)
    known["note_contexts"][0]["fields"]["analysis.role"]=result["note_contexts"][0]["fields"]["analysis.role"]
    with pytest.raises(EnrichmentError,match="overwrite"):
        merge_interpretations(piece,known,overlay,base_sha256="a"*64)
    overlay["note_fields"][0]["fields"]["analysis.role"]["origin"]="algorithmic_candidate"
    with pytest.raises(EnrichmentError,match="authored"):
        merge_interpretations(piece,bundle,overlay,base_sha256="a"*64)


def test_sol_overlay_preserves_origin_masks_and_rejects_false_astra_attribution():
    piece, bundle = fixture()
    field = dict(value="stable_support", available=True, origin="model_individual_context",
                 confidence=None, evidence=["synthetic:sol-decision"], unknown_reason=None)
    overlay = dict(schema="EventTheoryInterpretations@1", piece_id=piece.piece_id,
        model_requested="gpt-6.1-sol", reasoning_effort_requested="high", human_gold=False,
        base_sha256="a"*64, harmony_fields=[], objects=[], relations=[],
        note_fields=[dict(context_id="c0", fields={"analysis.role":field})])
    before = deepcopy(bundle)
    result = merge_interpretations(piece, bundle, overlay, base_sha256="a"*64)
    assert bundle == before
    assert result["provenance"]["individual_analysis"]["model_requested"] == "gpt-6.1-sol"
    assert result["provenance"]["individual_analysis"]["reasoning_effort_requested"] == "high"
    assert result["note_contexts"][0]["fields"]["analysis.role"]["origin"] == "model_individual_context"
    numeric = encode_enrichment(piece, result, build_vocabulary([result]))
    assert numeric["note_contexts"]["masks"]["analysis.role"] == [True, False]
    field["origin"] = "astra_individual_context"
    with pytest.raises(EnrichmentError, match="origin.*model"):
        merge_interpretations(piece, bundle, overlay, base_sha256="a"*64)


def test_sol_overlay_cannot_hide_false_astra_authorship_inside_objects():
    piece, bundle = fixture()
    field = dict(value="antecedent", available=True, origin="astra_individual_context",
                 confidence=None, evidence=["synthetic:sol-object"], unknown_reason=None)
    obj = dict(id="phrase:sol", object_type="phrase", member_ids=["note:n0"],
        start_qn={"num":0,"den":1}, end_qn={"num":4,"den":1}, fields={"analysis.function":field})
    overlay = dict(schema="EventTheoryInterpretations@1", piece_id=piece.piece_id,
        model_requested="gpt-6.1-sol", reasoning_effort_requested="high", human_gold=False,
        base_sha256="a"*64, harmony_fields=[], objects=[obj], relations=[], note_fields=[])
    with pytest.raises(EnrichmentError, match="origin.*model"):
        merge_interpretations(piece, bundle, overlay, base_sha256="a"*64)
    field["origin"] = "model_individual_context"
    result = merge_interpretations(piece, bundle, overlay, base_sha256="a"*64)
    assert result["objects"][0]["fields"]["analysis.function"]["origin"] == "model_individual_context"
