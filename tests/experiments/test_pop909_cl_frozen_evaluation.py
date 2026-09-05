from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from music_critic.adapters.pop909_cl import Pop909ClChordEvidence, Pop909ClCoverageSpan
from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.pop909_cl_frozen import run
from music_critic.experiments.pop909_cl_frozen.contracts import canonical_json, evaluation_protocol
from music_critic.experiments.pop909_cl_frozen.evaluation import (
    EvaluationPiece, SourceEvent, SourceTargets, align_event_start, evaluate_frozen,
    evaluate_from_readers, targets_from_chord_evidence,
)
from music_critic.experiments.pop909_cl_frozen.projection import (
    quality_class_labels, quality_gold_class_id, quality_subset_ids, root_class_labels,
)


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        result = CorrectedAnalysisGNNModel().eval().requires_grad_(False)
    return result


@pytest.fixture
def item(tmp_path):
    return run.synthetic_input(tmp_path)


def event(event_id="e0", start=RationalTime(0), root=0, quality="M"):
    return SourceEvent(event_id, start, root, quality, root is not None, quality is not None,
                       None if root is not None else "missing", None if quality is not None else "missing")


def controlled_predictions(monkeypatch, model, *, root=0, quality=None, trace=None):
    original = model.predict
    root_index = root_class_labels().index("C" if root == 0 else "D")
    quality_index = quality_gold_class_id("M") if quality is None else quality

    def predict(batch):
        if trace is not None:
            trace.append("predict")
        output = original(batch)
        predictions = []
        for row in output.predictions:
            if row.task_id in {"root", "quality"}:
                logits = torch.full_like(row.logits, -20)
                logits[:, root_index if row.task_id == "root" else quality_index] = 20
                row = replace(row, logits=logits)
            predictions.append(row)
        return replace(output, predictions=tuple(predictions))

    monkeypatch.setattr(model, "predict", predict)


def test_source_onset_alignment_exact_fractional_and_end_exclusive(item):
    beats = item.piece.beats
    assert align_event_start(RationalTime(1, 3), beats) == (0, None)
    assert align_event_start(RationalTime(1), beats) == (1, None)
    assert align_event_start(RationalTime(4), beats) == (None, "missing_containing_beat")
    assert align_event_start(RationalTime(-1, 3), beats) == (None, "missing_containing_beat")
    overlapping = (beats[0], replace(beats[1], start_qn=RationalTime(1, 2)))
    assert align_event_start(RationalTime(2, 3), overlapping) == (None, "ambiguous_containing_beat")
    tiny = RationalTime(10**25 - 1, 10**25)
    assert align_event_start(tiny, beats) == (0, None)
    with pytest.raises(ValueError, match="exact_source"):
        align_event_start(0.5, beats)


def test_full17_outside_prediction_retains_errors_and_independent_masks(model, item, monkeypatch):
    outside = next(i for i in range(len(quality_class_labels())) if i not in quality_subset_ids())
    controlled_predictions(monkeypatch, model, quality=outside)
    rows = (
        event("joint", RationalTime(0)),
        event("root-only", RationalTime(1), quality=None),
        event("quality-only", RationalTime(2), root=None),
        event("unsupported", RationalTime(3), quality="sus2"),
        event("unaligned", RationalTime(4)),
    )
    report = evaluate_frozen(model, [replace(item, read_targets=lambda: SourceTargets(rows))], split="synthetic")
    for view in report["views"].values():
        root, quality = view["tasks"]["root"], view["tasks"]["quality"]
        assert root["eligible_events"] == 3
        assert root["accuracy"]["value"] == 1
        assert quality["eligible_events"] == 2
        assert quality["accuracy"]["value"] == quality["observed_true_class_macro_f1"]["value"] == 0
        assert quality["outside_subset_predictions"] == 2
        major = quality_gold_class_id("M")
        assert quality["confusion"][major][outside] == 2
        assert quality["per_class"][outside]["false_positive"] == 2
        assert quality["per_class"][major]["false_negative"] == 2
        assert quality["included_class_ids"] == [major]
        assert quality["coverage"]["excluded_events"]["unsupported"] == 1
        assert view["root_quality_joint_accuracy"]["eligible_events"] == 1
        assert view["root_quality_joint_accuracy"]["value"] == 0


def test_multiple_events_share_prediction_are_not_averaged_or_dropped(model, item, monkeypatch):
    controlled_predictions(monkeypatch, model)
    targets = SourceTargets((event("a", RationalTime(1, 3)), event("b", RationalTime(2, 3), 2, "m")))
    result = evaluate_frozen(model, [replace(item, read_targets=lambda: targets)], split="synthetic")
    native = result["views"]["native"]
    assert native["resolution"]["multiple_events_same_beat"] == 1
    assert native["resolution"]["events_in_shared_beats"] == 2
    assert native["resolution"]["conflicting_label_beats"] == {"root": 1, "quality": 1}
    assert native["tasks"]["root"]["eligible_events"] == 2
    assert native["tasks"]["root"]["accuracy"]["value"] == .5
    assert len(native["event_observations"]) == 2
    assert len({r["beat_id"] for r in native["event_observations"]}) == 1


def test_candidate_first_both_views_and_target_change_invariance(model, item, monkeypatch):
    trace = []
    controlled_predictions(monkeypatch, model, trace=trace)

    def targets():
        assert trace == ["predict", "predict"]
        trace.append("targets")
        return SourceTargets((event(),))

    first = evaluate_frozen(model, [replace(item, read_targets=targets)], split="synthetic")
    second = evaluate_frozen(model, [replace(item, read_targets=lambda: SourceTargets((event(root=2, quality="m"),)))], split="synthetic")
    third = evaluate_frozen(model, [replace(item, read_targets=lambda: SourceTargets(source_status="expected_missing"))], split="synthetic")
    for view_id in ("native", "metadata_removed"):
        bindings = [r["views"][view_id]["input_bindings"] for r in (first, second, third)]
        assert bindings[0] == bindings[1] == bindings[2]
    assert first["pieces"][0]["target_fingerprint"] != second["pieces"][0]["target_fingerprint"]


def test_report_invariant_to_piece_and_target_order(model, item):
    a = replace(item, read_targets=lambda: SourceTargets((event("z"), event("a", RationalTime(1, 3)))))
    b = replace(item, piece=replace(item.piece, piece_id="piece:synthetic-second"), group_id="group:second",
                read_targets=lambda: SourceTargets((event("b", root=2),)))
    first = evaluate_frozen(model, [a, b], split="synthetic")
    a_reordered = replace(a, read_targets=lambda: SourceTargets(tuple(reversed(a.read_targets().events))))
    second = evaluate_frozen(model, [b, a_reordered], split="synthetic")
    assert canonical_json(first) == canonical_json(second)
    assert first["views"]["native"]["coverage"]["pieces"] == 2
    assert first["views"]["native"]["coverage"]["groups"] == 2
    assert first["views"]["native"]["view_fingerprint"] != first["views"]["metadata_removed"]["view_fingerprint"]


def test_empty_and_missing_denominators_are_null(model, item):
    for inputs in ([], [replace(item, read_targets=lambda: SourceTargets(source_status="expected_missing"))]):
        report = evaluate_frozen(model, inputs, split="synthetic")
        for view in report["views"].values():
            for task in view["tasks"].values():
                assert task["eligible_events"] == 0
                assert task["accuracy"] == {"value": None, "reason": "no_eligible_events"}
                assert task["observed_true_class_macro_f1"] == {"value": None, "reason": "no_observed_true_classes"}
                assert task["included_class_ids"] == []
                assert task["coverage"]["event_fraction"]["value"] is None
            assert view["coverage"]["missing_target_pieces"] == len(inputs)


def test_ambiguous_root_keeps_supported_quality(model, item):
    target = SourceEvent("ambiguous", RationalTime(0), None, "o7", False, True, "ambiguous", None)
    report = evaluate_frozen(model, [replace(item, read_targets=lambda: SourceTargets((target,)))], split="synthetic")
    tasks = report["views"]["native"]["tasks"]
    assert tasks["root"]["eligible_events"] == 0
    assert tasks["root"]["coverage"]["excluded_events"]["ambiguous"] == 1
    assert tasks["quality"]["eligible_events"] == 1


def test_source_block_original_onset_not_clipped_and_no_chord_counts():
    candidate = SimpleNamespace(root_pc=0, quality="M")
    block = SimpleNamespace(block_id="original", onset_tick=2401, ppqn=480, selected_candidate=candidate,
                            candidates=(candidate,), root_available=True, quality_available=True,
                            normalization_status="supported")
    evidence = SimpleNamespace(blocks=(block,), status="available", no_chord_spans=(
        SimpleNamespace(kind="leading_no_chord"), SimpleNamespace(kind="missing_chord_targets")), trailing_spans=())
    targets = targets_from_chord_evidence(evidence)
    assert targets.events[0].start_qn == RationalTime(2401, 480)
    assert targets.no_chord_span_count == 1


@pytest.mark.parametrize(
    "status,span_kind,expected_missing,expected_trailing",
    [
        ("expected_missing", "missing_chord_targets", 1, 0),
        ("available", "trailing_unannotated", 0, 1),
    ],
)
def test_adapter_trailing_spans_distinguish_missing_targets_in_report(
    model, item, status, span_kind, expected_missing, expected_trailing,
):
    # The adapter stores its full-piece missing-target marker in trailing_spans.
    # Use the real dataclasses/shape so missing supervision cannot become a tail.
    evidence = Pop909ClChordEvidence(
        status=status,
        ppqn=480,
        blocks=(),
        no_chord_spans=(),
        trailing_spans=(Pop909ClCoverageSpan(
            span_id=f"{span_kind}:0",
            kind=span_kind,
            start_tick=0,
            end_tick=1920,
            ppqn=480,
            available=False,
            value=None,
        ),),
        pairing_anomalies=(),
        repeated_pitch_block_count=0,
        mixed_end_block_count=0,
        overlap_count=0,
    )
    targets = targets_from_chord_evidence(evidence)
    assert targets.source_status == status
    assert targets.trailing_unannotated_span_count == expected_trailing
    report = evaluate_frozen(
        model,
        [replace(item, read_targets=lambda: targets)],
        split="synthetic",
    )
    assert report["pieces"][0]["trailing_unannotated_span_count"] == expected_trailing
    for view in report["views"].values():
        coverage = view["coverage"]
        assert coverage["missing_target_pieces"] == expected_missing
        assert coverage["missing_target_groups"] == expected_missing
        assert coverage["trailing_unannotated_spans"] == expected_trailing
        assert coverage["pieces"] == coverage["groups"] == 1
        assert coverage["source_chord_events"] == coverage["no_chord_spans"] == 0
        for task in view["tasks"].values():
            assert task["eligible_events"] == 0
            assert task["accuracy"]["value"] is None


def test_alignment_and_label_unavailability_have_independent_coverage(model, item):
    targets = SourceTargets((event("both", RationalTime(100), root=None, quality="sus2"),))
    report = evaluate_frozen(model, [replace(item, read_targets=lambda: targets)], split="synthetic")
    tasks = report["views"]["native"]["tasks"]
    for task in tasks.values():
        assert task["coverage"]["unaligned_events"] == 1
        assert task["coverage"]["otherwise_label_eligible_unaligned_events"] == 0
        assert task["coverage"]["alignment_unavailable"]["missing_containing_beat"] == {
            "events": 1, "pieces": 1, "groups": 1,
        }
    assert tasks["quality"]["coverage"]["label_unavailable"]["unsupported"]["events"] == 1


def test_checkpoint_evidence_must_bind_actual_model_before_sources(model):
    def forbidden_iterator():
        raise AssertionError("source accessed before bad binding rejected")
        yield
    with pytest.raises(ValueError, match="checkpoint_evidence_binding"):
        evaluate_frozen(model, forbidden_iterator(), split="synthetic", checkpoint_evidence={
            "model_state_fingerprint": "wrong-model", "artifact_sha256": "wrong-artifact",
        })


def test_smoke_source_extraction_is_deferred_after_both_predictions(model, tmp_path, monkeypatch):
    trace = []
    controlled_predictions(monkeypatch, model, trace=trace)
    original = run.convert_pop909_cl_file

    def convert(*args, **kwargs):
        assert trace == ["predict", "predict"]
        trace.append("target-extraction")
        return original(*args, **kwargs)

    monkeypatch.setattr(run, "convert_pop909_cl_file", convert)
    item = run.synthetic_input(tmp_path)
    assert trace == [] and item.piece.targets == () and item.piece.annotations == ()
    evaluate_frozen(model, [item], split="synthetic")
    assert trace == ["predict", "predict", "target-extraction"]


def test_synthetic_input_preserves_pre_helper_midi_bytes(item, tmp_path):
    # Observed before INT-R1-001 moved MIDI construction to the adapter helper.
    expected = {
        "001.mid": (127, "678226cc01c74815b35788997881f3a943658552b578dbde79eac86b3ad447e6"),
        "score-only.mid": (90, "21d31a9fc9823b6f83c821b26c5d032fa55e80b2f60ab4c0c883103550dda0b2"),
    }
    for name, (size, digest) in expected.items():
        content = (tmp_path / name).read_bytes()
        assert len(content) == size
        assert sha256(content).hexdigest() == digest
    assert item.group_id == f"pop909-cl-score:{expected['score-only.mid'][1]}"
    assert item.piece.targets == () and item.piece.annotations == ()


def test_zero_piece_denominators_and_same_group_coverage(model, item):
    empty = evaluate_frozen(model, [], split="synthetic")
    for task in empty["views"]["native"]["tasks"].values():
        assert task["coverage"]["piece_fraction"] == {"value": None, "reason": "no_pieces"}
        assert task["coverage"]["group_fraction"] == {"value": None, "reason": "no_groups"}
    second = replace(item, piece=replace(item.piece, piece_id="piece:other"))
    report = evaluate_frozen(model, [item, second], split="synthetic")
    for task in report["views"]["native"]["tasks"].values():
        assert task["coverage"]["total_pieces"] == 2
        assert task["coverage"]["total_groups"] == 1


def test_missing_and_ambiguous_containing_beat_remain_unavailable(model, item, monkeypatch):
    import music_critic.experiments.pop909_cl_frozen.evaluation as evaluation
    for reason in ("missing_containing_beat", "ambiguous_containing_beat"):
        monkeypatch.setattr(evaluation, "align_event_start", lambda *args: (None, reason))
        result = evaluate_frozen(model, [item], split="synthetic")
        for task in result["views"]["native"]["tasks"].values():
            assert task["eligible_events"] == 0
            assert task["coverage"]["excluded_events"][reason] == 1


@pytest.mark.parametrize("split", ["test", "TEST", "validation", "train"])
def test_guard_precedes_checkpoint_source_target_readers(split, model):
    def forbidden():
        raise AssertionError("reader accessed")
    with pytest.raises(ValueError, match="forbidden|future_registered"):
        evaluate_from_readers(split=split, checkpoint_reader=forbidden, source_reader=forbidden)

    def forbidden_iterator():
        forbidden()
        yield
    with pytest.raises(ValueError, match="forbidden|future_registered"):
        evaluate_frozen(model, forbidden_iterator(), split=split)


def test_cli_default_no_reads_and_test_denied_before_smoke(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("smoke accessed")
    monkeypatch.setattr(run, "run_smoke", forbidden)
    assert run.main([]) == 0
    assert "--smoke" in capsys.readouterr().out
    with pytest.raises(ValueError, match="test_access_forbidden"):
        run.main(["--smoke", "--split", "TEST"])


def test_synthetic_checkpoint_cli_smoke_deterministic_and_test_closed():
    first = run.run_smoke()
    second = run.run_smoke()
    assert canonical_json(first) == canonical_json(second)
    assert first["schema_version"] == "Pop909ClFrozenEvaluationReport@1.0.0"
    assert first["protocol"] == evaluation_protocol()
    assert first["views"]["native"]["coverage"]["source_chord_events"] == 1
    assert first["access_observations"]["test_inputs_accessed"] is False
    assert first["access_observations"]["test_targets_accessed"] is False
    assert first["access_observations"]["test_metrics_accessed"] is False
    assert "corrected_primary_macro_score" not in canonical_json(first)


def test_training_mode_and_duplicate_piece_rejected(model, item):
    with pytest.raises(ValueError, match="unique_piece"):
        evaluate_frozen(model, [item, item], split="synthetic")
    model.train()
    try:
        with pytest.raises(ValueError, match="cpu_eval_model"):
            evaluate_frozen(model, [item], split="synthetic")
    finally:
        model.eval()


@pytest.mark.parametrize("kwargs", [
    {"start_qn": .3}, {"root_pc": 12, "root_available": True, "root_reason": None},
    {"root_pc": 1}, {"quality": "M"}, {"quality_available": True, "quality_reason": None},
])
def test_invalid_source_masks_and_exact_time_fail_closed(kwargs):
    with pytest.raises(ValueError):
        SourceEvent(**{"event_id": "bad", "start_qn": RationalTime(0), **kwargs})
