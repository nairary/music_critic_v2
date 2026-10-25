"""Loss arithmetic and direct-gradient masks on tiny synthetic predictions."""

from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from torch.nn import functional as F

from music_critic.critic.contracts import CriticOutput, LocalCandidate, LocalPrediction, SamplePrediction
from music_critic.critic.targets import LocalTargets, PairTargets
from music_critic.critic.loss import CriticLoss, evaluate_local_interventions


def _prediction(name, scores, availability=(True,) * 4, local=None):
    return SamplePrediction(name, torch.tensor(scores, dtype=torch.float64, requires_grad=True),
                            torch.tensor(availability), local or {}, Fraction(4))


def _target(work="work", sample_ids=("a", "b"), mask=(True, False, False, False), local=({}, {})):
    return PairTargets(work, sample_ids, mask, tuple(1 if m else 0 for m in mask), local)


def test_pair_loss_is_softplus_of_edited_minus_original_and_swaps_exactly():
    left, right = _prediction("a", [2, 9, 9, 9]), _prediction("b", [1, -9, -9, -9])
    target = _target()
    result = CriticLoss()((left, right), target)
    assert result.pair.item() == pytest.approx(F.softplus(torch.tensor(-1.)).item())
    assert result.total.item() == result.pair.item()
    assert result.local is None and result.pair_support == 1 and result.local_support == 0
    assert not result.skip_update
    swapped = CriticLoss()((right, left), target.swapped())
    assert torch.equal(result.total, swapped.total)
    result.total.backward()
    assert left.raw_scores.grad[0] < 0 < right.raw_scores.grad[0]
    assert torch.equal(left.raw_scores.grad[1:], torch.zeros(3, dtype=torch.float64))


def test_batch_balances_works_within_each_aspect_then_macro_averages_aspects():
    left = tuple(_prediction(f"a{i}", [0, 1, 0, 0]) for i in range(3))
    right = tuple(_prediction(f"b{i}", [2 * i, 0, 0, 0]) for i in range(3))
    targets = (_target("A", ("a0", "b0"), (True, True, False, False)),
               _target("A", ("a1", "b1")), _target("B", ("a2", "b2")))
    result = CriticLoss()((CriticOutput(left), CriticOutput(right)), targets)
    sp = lambda x: F.softplus(torch.tensor(float(x))).item()
    expected_harmony = ((sp(0) + sp(2)) / 2 + sp(4)) / 2
    assert result.pair.item() == pytest.approx((expected_harmony + sp(-1)) / 2)
    assert result.pair_support == 4


def test_no_available_pair_or_local_targets_returns_explicit_skip():
    left = _prediction("a", [0, 0, 0, 0], (False,) * 4)
    right = _prediction("b", [0, 0, 0, 0])
    result = CriticLoss()((left, right), _target())
    assert result.total is None and result.pair is None and result.local is None
    assert result.skip_update and result.pair_support == result.local_support == 0


def _local(namespace, logits, values, mask=None, availability=None):
    candidates = tuple(LocalCandidate(f"id:{i}", Fraction(i), Fraction(i + 1)) for i in range(len(logits)))
    aspect, kind = namespace.split("/")
    prediction = LocalPrediction(aspect, kind, candidates,
                                 torch.tensor(logits, dtype=torch.float64, requires_grad=True),
                                 torch.tensor(availability or [True] * len(logits)))
    target = LocalTargets(candidates, tuple(values), tuple(mask or [True] * len(logits)))
    return prediction, target


def test_local_bce_balances_classes_and_types_and_masked_logits_have_no_direct_gradient():
    notes, note_targets = _local("melody/note", [-2, 0, 0, -50], [1, 0, 0, 1], [True, True, True, False])
    events, event_targets = _local("melody/event", [0], [1])
    left = _prediction("a", [0] * 4)
    right = _prediction("b", [0] * 4, local={"melody/note": notes, "melody/event": events})
    target = _target(mask=(False,) * 4, local=({}, {"melody/note": note_targets, "melody/event": event_targets}))
    result = CriticLoss()((left, right), target)
    sp = lambda x: F.softplus(torch.tensor(float(x))).item()
    expected = ((sp(2) + sp(0)) / 2 + sp(0)) / 2
    assert result.local.item() == pytest.approx(expected)
    assert result.total.item() == pytest.approx(0.25 * expected)
    assert result.local_support == 4 and result.pair_support == 0
    result.total.backward()
    assert notes.logits.grad[-1] == 0
    assert notes.logits.grad[0] < 0 < notes.logits.grad[1]
    assert right.raw_scores.grad is None
    swapped = CriticLoss()((right, left), target.swapped())
    assert torch.equal(swapped.total, result.total)
    control = CriticLoss(lambda_local=0)((left, right), target)
    assert control.local is not None and control.total is None and control.skip_update


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf"), True])
def test_local_coefficient_requires_finite_nonnegative_number(coefficient):
    with pytest.raises(ValueError, match="lambda_local"):
        CriticLoss(coefficient)


def test_misaligned_identity_shapes_and_observed_nonfinite_scores_fail_closed():
    left, right = _prediction("a", [0] * 4), _prediction("b", [0] * 4)
    with pytest.raises(ValueError, match="identit"):
        CriticLoss()((right, left), _target())
    with pytest.raises(ValueError, match="shape"):
        CriticLoss()((replace(left, raw_scores=left.raw_scores[:2]), right), _target())
    with pytest.raises(ValueError, match="boolean"):
        CriticLoss()((replace(left, availability=left.availability.float()), right), _target())
    with pytest.raises(ValueError, match="finite"):
        CriticLoss()((_prediction("a", [float("nan"), 0, 0, 0]), right), _target())


def test_local_candidate_alignment_and_availability_are_checked_before_loss():
    local, labels = _local("melody/note", [0, 0], [1, 0])
    left = _prediction("a", [0] * 4)
    target = _target(mask=(False,) * 4, local=({}, {"melody/note": labels}))
    for bad in (replace(local, candidates=local.candidates[::-1]),
                replace(local, logits=local.logits[:1]),
                replace(local, availability=local.availability.float()),
                replace(local, aspect="rhythm")):
        right = _prediction("b", [0] * 4, local={"melody/note": bad})
        with pytest.raises(ValueError):
            CriticLoss()((left, right), target)


def test_masked_nonfinite_positions_do_not_enter_arithmetic():
    local, labels = _local("melody/note", [0, float("nan")], [1, 0], [True, False])
    left = _prediction("a", [0, float("nan"), 0, 0])
    right = _prediction("b", [0] * 4, local={"melody/note": local})
    target = _target(local=({}, {"melody/note": labels}))
    result = CriticLoss()((left, right), target)
    assert torch.isfinite(result.total)
    result.total.backward()
    assert local.logits.grad[1] == 0


def test_local_metrics_use_observed_candidates_and_exact_metric_interval_union():
    beats, labels = _local("rhythm/beat", [1, -1, 1, 99], [1, 1, 0, 0], [True, True, True, False])
    left, right = _prediction("a", [0] * 4), _prediction("b", [0] * 4, local={"rhythm/beat": beats})
    target = _target(mask=(False,) * 4, local=({}, {"rhythm/beat": labels}))
    result = evaluate_local_interventions((left, right), target)
    metric = result.per_namespace["rhythm/beat"]
    assert metric.precision == metric.recall == 0.5
    assert metric.interval_iou == pytest.approx(1 / 3)
    assert metric.observed_support == 3 and metric.positive_support == metric.predicted_support == 2
    assert metric.work_support == metric.interval_work_support == 1
    assert result == evaluate_local_interventions((right, left), target.swapped())


def test_point_local_metrics_never_invent_interval_extent_or_absent_class():
    local, labels = _local("structure/boundary", [-1], [0])
    points = tuple(replace(c, end=c.start) for c in local.candidates)
    local, labels = replace(local, candidates=points), replace(labels, candidates=points)
    left, right = _prediction("a", [0] * 4), _prediction("b", [0] * 4, local={"structure/boundary": local})
    result = evaluate_local_interventions((left, right), _target(local=({}, {"structure/boundary": labels})))
    metric = result.per_namespace["structure/boundary"]
    assert metric.precision is metric.recall is metric.interval_iou is None
    assert metric.observed_support == 1 and metric.interval_work_support == 0


def test_pair_comparison_requires_same_horizon_and_support_policy():
    left, right = _prediction("a", [0] * 4), _prediction("b", [0] * 4)
    for altered in (replace(right, horizon=Fraction(8)), replace(right, support_policy="another-policy")):
        with pytest.raises(ValueError, match="scope"):
            CriticLoss()((left, altered), _target())


def test_zero_local_control_accepts_baseline_global_only_predictions_with_real_sidecar():
    from pathlib import Path
    from music_critic.data import load_piece
    from music_critic.experiments.music_interventions import Intervention, NotePatch, apply_interventions
    from music_critic.critic.targets import InterventionTargetBuilder, VerifiedPatch

    original = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    patch = NotePatch("note:melody-000", pitch=68)
    variant = apply_interventions(original, (Intervention("pitch", (patch,)),)).piece
    labels = InterventionTargetBuilder().build(original, variant,
        VerifiedPatch(original.piece_id, variant.piece_id, "work", (patch,), ("harmony",), "pitch"))
    left, right = _prediction(original.piece_id, [0] * 4), _prediction(variant.piece_id, [0] * 4)
    with pytest.raises(ValueError, match="namespace"):
        CriticLoss()((left, right), labels)
    result = CriticLoss(lambda_local=0)((left, right), labels)
    assert result.pair is result.total or torch.equal(result.pair, result.total)
    assert result.local is None and not result.skip_update
