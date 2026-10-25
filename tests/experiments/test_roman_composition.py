import pytest

from music_critic.experiments.analysisgnn.roman_composition import (
    audit, classification_metrics, compatibility_scores, component_key,
    fit_lookup, harmonic_rows, lookup_predict, score_lookup,
)
from scripts.audit_roman_composition import checked_bytes, checked_member


def row(label="I", degree="1", quality="major", split="train", **extra):
    return dict(split=split, record_id=split, component_id=split,
                source_entity_id="row:1", entity_id="e:1",
                labels=dict(roman_numeral=label, primary_degree=degree, quality=quality), **extra)


FEATURES = ("primary_degree", "quality")


def test_collisions_never_claim_lossless_or_learn_from_validation():
    train = [row(), row(), row("Cad")]
    table = fit_lookup(train, FEATURES)
    scores = score_lookup([row("Cad", split="validation")], FEATURES, table)
    assert scores["ambiguous_tuple"] == 1
    assert scores["correct_unique"] == scores["correct_mode"] == 0
    assert table[("1", "major")] == {"I": 2, "Cad": 1}
    with pytest.raises(ValueError, match="TRAIN"):
        fit_lookup([row(split="validation")], FEATURES)


def test_missing_and_unseen_abstain():
    table = fit_lookup([row()], FEATURES)
    assert component_key(row(degree=None)["labels"], FEATURES) is None
    assert lookup_predict(table, ("5", "major")) is None
    scores = score_lookup([row(degree=None), row(degree="5"), row(label=None)], FEATURES, table)
    assert scores["unseen_tuple"] == scores["components_missing"] == scores["target_missing"] == 1
    assert scores["mode_accuracy_complete"] == 0


def test_audit_keeps_component_splits_disjoint():
    val = row(split="validation")
    val["component_id"] = "train"
    with pytest.raises(ValueError, match="crosses split"):
        audit([row(), val])
    with pytest.raises(ValueError, match="TEST"):
        audit([row(split="test")])


def test_availability_gates_source_value():
    record = dict(record_id="r", source_component_id="c", split="train")
    sidecar = dict(record_id="r", source_component_id="c", dialect="dlc_joint", entities=[dict(
        entity_type="harmonic_event", canonical_entity_id="e", source_entity_id="s",
        targets=dict(roman_numeral=dict(available=True, masked=True, canonical_value="I", source_value="I")))])
    result = list(harmonic_rows(record, sidecar))
    assert result[0]["labels"]["roman_numeral"] is None
    assert result[0]["source_values"]["roman_numeral"] == "I"
    sidecar["entities"] *= 2
    with pytest.raises(ValueError, match="duplicate harmonic"):
        list(harmonic_rows(record, sidecar))


def test_fixed_inventory_and_abstentions():
    m = classification_metrics(["I", "V"], ["I", None], ["I", "V", "vi"])
    assert m["accuracy"] == .5
    assert m["macro_f1"] == 1/3
    assert m["macro_f1_present"] == .5


def test_compatibility_marginalizes_training_tuples():
    table = fit_lookup([row(), row(), row("Cad"), row("V", "5")], FEATURES)
    probs = [{"1": .25, "5": .75}, {"major": 1.}]
    assert compatibility_scores(table, probs, ["I", "Cad", "V", "vi"]) == [.25, .25, .75, 0]
    # Same components cannot distinguish I and Cad, even with exact probabilities.


@pytest.mark.parametrize("name", ["/tmp/a", "a/../b", "a\\b", "./a", "a//b"])
def test_archive_paths_fail_closed(name):
    with pytest.raises(ValueError):
        checked_member(name)


def test_bound_bytes_reject_bad_hash_and_symlink(tmp_path):
    target = tmp_path / "input"
    target.write_text("x")
    with pytest.raises(ValueError, match="hash"):
        checked_bytes(target, "0" * 64)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        checked_bytes(link, "0" * 64)


def test_tensor_decoders_match_reference_and_use_probabilities():
    import torch
    from scripts.evaluate_roman_composition import compatibility_tensor, decode_probabilities
    table = fit_lookup([row(), row(), row("Cad"), row("V", "5")], FEATURES)
    vocabs = dict(primary_degree=("1", "5"), quality=("major",), roman_numeral=("I", "Cad", "V"))
    tensor = compatibility_tensor(table, vocabs)
    predictions = decode_probabilities(dict(primary_degree=torch.tensor([[.25, .75]]),
        quality=torch.tensor([[1.]]), roman_numeral=torch.tensor([[.01, .98, .01]])), table, vocabs, tensor)
    assert predictions == dict(direct=["Cad"], hard_lookup=["V"], soft_components=["V"],
                               direct_times_components=["Cad"])
    assert torch.allclose(tensor.sum((0, 1)), torch.ones(3, dtype=torch.float64))


def test_tensor_decoders_abstain_with_zero_compatibility():
    import torch
    from scripts.evaluate_roman_composition import compatibility_tensor, decode_probabilities
    vocabs = dict(primary_degree=("1", "5"), quality=("major",), roman_numeral=("I", "V"))
    table = fit_lookup([row()], FEATURES)
    result = decode_probabilities(dict(primary_degree=torch.tensor([[0., 1.]]),
        quality=torch.tensor([[1.]]), roman_numeral=torch.tensor([[0., 1.]])), table, vocabs,
        compatibility_tensor(table, vocabs))
    assert result == dict(direct=["V"], hard_lookup=[None], soft_components=[None], direct_times_components=[None])


def test_both_archive_readers_reject_test_before_opening_payloads(tmp_path):
    import json
    from hashlib import sha256
    from scripts.audit_roman_composition import read_rows
    from scripts.evaluate_roman_composition import read_validation
    manifest = json.dumps(dict(records=[dict(split="test")])).encode()
    digest = sha256(manifest).hexdigest()
    receipt = json.dumps(dict(relocated_manifest_sha256=digest, test_payload_packaged=False,
                             destination="/srv/data", parts=[])).encode()
    (tmp_path / "data-manifest.json").write_bytes(manifest)
    (tmp_path / "receipt.json").write_bytes(receipt)
    args = (tmp_path, digest, sha256(receipt).hexdigest())
    with pytest.raises(ValueError, match="TEST"):
        read_rows(*args, lambda _: None)
    with pytest.raises(ValueError, match="TEST"):
        next(read_validation(*args))
