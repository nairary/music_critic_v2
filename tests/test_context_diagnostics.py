from dataclasses import replace
from hashlib import sha256

import pytest

from music_critic.data import RationalTime
from music_critic.experiments.context_diagnostics import (
    Candidate, FileBinding, LabeledFeature, Member, ModelBinding, TargetRow,
    classification_metrics, fingerprint, fit_local_control, local_features,
    raw_candidates, restrict_note_context, run_context_diagnostics, validate_membership,
)
from music_critic.experiments.context_diagnostics_pairs import build_context_pair_suite


@pytest.fixture
def piece():
    return replace(build_context_pair_suite()[0].left, split="train")


def target(identity="e", candidate="b", value=0):
    return TargetRow(identity, candidate, value, value is not None,
                     None if value is not None else "missing", "synthetic_manual", .8)


def row(value, vector, identity="e", split="train"):
    return LabeledFeature(Member(identity, identity, split), identity, tuple(vector), target(identity, value=value))


def test_local_training_and_metric_manual_oracle():
    control = fit_local_control([row(0, (0.,), "a"), row(1, (2.,), "b")], ["zero", "one", "unseen"])
    assert control.means == (1.,)
    assert control.scales == (1.,)
    assert control.centroids == ((-1.,), (1.,), None)
    assert control.predict((1.,)) == 0  # exact tie -> fixed class order
    assert control.predict((2.,)) == 1
    metrics = classification_metrics([(0, 0), (0, 1), (1, 1), (1, 2)], ["a", "b", "c"])
    assert metrics["confusion_gold_rows_prediction_columns"] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert metrics["classes"][0]["precision"] == 1
    assert metrics["classes"][0]["recall"] == .5
    assert metrics["classes"][1]["f1"] == .5
    assert metrics["classes"][2]["recall"] is None
    assert metrics["observed_class_macro_f1"] == pytest.approx((2/3 + .5)/2)


@pytest.mark.parametrize("split", ["validation", "test", "TEST", "unknown"])
def test_fit_never_uses_other_splits_even_for_masked_rows(split):
    with pytest.raises(ValueError, match="TRAIN_only"):
        fit_local_control([row(None, (99,), split=split)], ["a", "b"])


def test_masked_values_excluded_from_statistics_and_training_fingerprint():
    rows = [row(0, (0,), "a"), row(1, (2,), "b")]
    expected = fit_local_control(rows, ["a", "b"])
    assert fit_local_control(rows + [row(None, (1e6,), "missing")], ["a", "b"]) == expected
    assert fit_local_control(list(reversed(rows)), ["a", "b"]) == expected


@pytest.mark.parametrize("change", [{"value": 0, "available": False, "reason": "missing"},
                                    {"value": None}, {"confidence": float("nan")},
                                    {"provenance": ""}, {"value": True}, {"available": 1}])
def test_target_contract_masks_and_confidence(change):
    with pytest.raises(ValueError):
        replace(target(), **change).validate(2)


def test_empty_metrics_are_null():
    result = classification_metrics([], ["a", "b"])
    assert result["accuracy"] is result["observed_class_macro_f1"] is None
    assert all(r["precision"] is r["recall"] is r["f1"] is None for r in result["classes"])


def test_exact_overlap_not_float_and_half_open_boundary(piece):
    end = RationalTime(1, 3)
    n = piece.notes[0]
    p = replace(piece, notes=(replace(n, pitch=60, onset_qn=RationalTime(0), duration_qn=end),
                              replace(n, note_id="second", pitch=64, onset_qn=end, duration_qn=end)))
    f = local_features(p, Candidate("x", RationalTime(0), end))
    assert f[0] == 1 and f[4] == 0 and f[19:] == pytest.approx((1, 1, 0, 1, 1/3))
    tiny = RationalTime(10**25-1, 3*10**25)
    f = local_features(p, Candidate("x", tiny, end))
    assert f[0] == 1 and f[4] == 0 and f[19] == 1
    with pytest.raises(ValueError, match="exact_timing"):
        local_features(p, Candidate("x", 0., end))


def test_interval_oracle_and_no_non_simultaneous_pairs(piece):
    n = piece.notes[0]
    p = replace(piece, notes=tuple(replace(n, note_id=str(i), pitch=pitch, onset_qn=RationalTime(0),
                                         duration_qn=RationalTime(1)) for i, pitch in enumerate((60, 64, 67))))
    f = local_features(p, Candidate("x", RationalTime(0), RationalTime(1)))
    assert f[0] == f[4] == f[7] == 1/3
    assert f[12+3] == f[12+4] == f[12+5] == 1/3
    separated = replace(p, notes=(p.notes[0], replace(p.notes[1], onset_qn=RationalTime(1))))
    assert local_features(separated, Candidate("x", RationalTime(0), RationalTime(2)))[12:19] == (0.,)*7


def test_raw_features_ignore_tracks_annotations_targets_and_order(piece):
    window = Candidate("x", RationalTime(1), RationalTime(2))
    modified = replace(piece, notes=tuple(replace(note, track_id="arbitrary", voice=9, staff=5,
                        velocity=1, channel=15, program=127) for note in reversed(piece.notes)),
                        annotations=("sentinel",), targets=("sentinel",), key_signature_events=("sentinel",))
    assert local_features(modified, window) == local_features(piece, window)
    assert raw_candidates(modified, "onset") == raw_candidates(piece, "onset")


@pytest.mark.parametrize("members", [
    [Member("a", "x", "train"), Member("a", "y", "validation")],
    [Member("a", "x", "train"), Member("b", "x", "validation")],
    [Member("a", "x", "test")], []])
def test_membership_rejects_mixing(members):
    with pytest.raises(ValueError):
        validate_membership(members)


def test_file_binding_exact_bytes_budget(tmp_path):
    p = tmp_path / "raw.json"
    p.write_bytes(b"abc")
    binding = FileBinding(str(p), sha256(b"abc").hexdigest(), 3)
    assert binding.read() == b"abc"
    for bad in (replace(binding, sha256="0"*64), replace(binding, size_bytes=2), replace(binding, size_bytes=True)):
        with pytest.raises(ValueError):
            bad.read()


def test_harness_orders_raw_reference_targets_and_identical_masks(piece):
    members = [Member("val", "g-val", "validation"), Member("train", "g-train", "train")]
    classes = ["a", "b"]
    binding = ModelBinding("0"*64, "1"*64, "2"*64, fingerprint(classes), "3"*64)
    trace = []
    def raw(member):
        trace.append((member.split, "raw"))
        return replace(piece, piece_id=member.piece_id, source_group_id=member.component_id,
                       split=member.split, targets=("sentinel",), annotations=("sentinel",))
    def predict(p, candidates):
        assert p.targets == p.annotations == ()
        trace.append((p.split, "predict"))
        return {c.candidate_id: 1 for c in candidates}
    def labels(member):
        trace.append((member.split, "targets"))
        candidate = piece.beats[0].beat_id
        return [target("e0", candidate, 0), target("e1", candidate, None), target("e2", "absent", 1)]
    report = run_context_diagnostics(members=members, membership_sha256="f"*64, classes=classes,
                level="beat", binding=binding, raw_reader=raw, target_reader=labels, reference_predictor=predict)
    assert trace == [("train", "raw"), ("train", "targets"), ("validation", "raw"),
                     ("validation", "predict"), ("validation", "targets")]
    assert report["local"]["eligible_observations"] == report["reference"]["eligible_observations"] == 1
    assert report["local"]["accuracy"] == 1 and report["reference"]["accuracy"] == 0
    assert report["coverage"]["validation_masked:missing"] == report["coverage"]["validation_unaligned"] == 1
    assert report["observations"][0]["target_confidence"] == .8
    with pytest.raises(ValueError, match="TRAINVAL_only"):
        run_context_diagnostics(members=[Member("test", "test", "test")], membership_sha256="f"*64,
            classes=classes, level="beat", binding=binding, raw_reader=lambda _: pytest.fail("read"),
            target_reader=labels, reference_predictor=predict)


def test_note_context_intervention_preserves_exact_local_feature_and_grid(piece):
    window = Candidate("focus", RationalTime(1), RationalTime(2))
    crop = restrict_note_context(piece, window.start, window.end)
    assert local_features(crop, window) == local_features(piece, window)
    assert crop.beats == piece.beats and crop.bars == piece.bars and crop.duration_qn == piece.duration_qn
    assert all(n.onset_qn < window.end and n.onset_qn+n.duration_qn > window.start for n in crop.notes)


def test_held_note_intervention_never_creates_a_new_attack(piece):
    note = replace(piece.notes[0], onset_qn=RationalTime(0), duration_qn=RationalTime(3))
    piece = replace(piece, notes=(note,))
    window = Candidate("focus", RationalTime(1), RationalTime(2))
    restricted = restrict_note_context(piece, window.start, window.end)
    assert restricted.notes == (note,)
    assert local_features(piece, window)[19:] == (1., 0., 1., 1., 1.)
    assert local_features(restricted, window) == local_features(piece, window)
