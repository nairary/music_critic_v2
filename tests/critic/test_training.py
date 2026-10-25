"""Tiny synthetic cached-feature optimization; no corpus or backbone loading."""
from fractions import Fraction
from dataclasses import replace

import pytest
import torch

from music_critic.critic.contracts import (
    CriticOutput, SamplePrediction, CriticFeaturePyramid, FeatureBinding,
    FeatureSample, SampleGeometry,
)
from music_critic.critic.targets import PairTargets
from music_critic.critic.training import (
    evaluate_pairs, SelectionState, CachedPair, CriticTrainConfig, CriticTrainer,
)


def prediction(name, scores):
    return SamplePrediction(name, torch.tensor(scores, dtype=torch.float32),
                            torch.ones(4, dtype=torch.bool), {}, Fraction(4))


def pair_target(work, left, right, mask=(True, False, False, False)):
    return PairTargets(work, (left, right), mask,
                       tuple(1 if x else 0 for x in mask), ({}, {}), {})


def test_selection_metrics_balance_works_and_keep_unsupported_aspects_explicit():
    rows = [
        (prediction("a0", [2, 0, 0, 0]), prediction("a1", [1, 0, 0, 0]), pair_target("a", "a0", "a1")),
        (prediction("a0", [2, 0, 0, 0]), prediction("a2", [1, 0, 0, 0]), pair_target("a", "a0", "a2")),
        (prediction("b0", [1, 0, 0, 0]), prediction("b1", [1, 0, 0, 0]), pair_target("b", "b0", "b1")),
    ]
    result = evaluate_pairs(rows, selection_aspects=("harmony", "rhythm"), tie_tolerance=1e-6)
    assert result.pair_accuracy == 0.75  # Work a=1; work b=half, not 5/6.
    assert result.per_aspect["rhythm"] is None
    assert result.supported_aspects == ("harmony",)
    selection = SelectionState(("harmony", "rhythm"))
    assert selection.consider(result, update=3)
    assert not selection.consider(result, update=4)
    assert selection.best_update == 3


class TinyCritic(torch.nn.Module):
    """A genuine learned scalar seam for cheap optimizer/RNG contract tests."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, pyramid):
        return CriticOutput(tuple(SamplePrediction(
            s.sample_id, self.weight * self.dropout(s.embeddings["song"][0]),
            torch.ones(4, dtype=torch.bool), {}, s.geometry.duration,
        ) for s in pyramid.samples))


def features(name, value):
    sample = FeatureSample(SampleGeometry(name, (), (), {}, Fraction(4)),
                           {"song": torch.full((1, 4), float(value))},
                           f"synthetic-{name}-{value}", {"song": "synthetic"})
    return CriticFeaturePyramid((sample,), FeatureBinding("synthetic-source", "checkpoint", "v1"))


def make_bank(supported=True):
    result = []
    for i in range(3):
        left, right = f"left{i}", f"right{i}"
        target = pair_target(f"work{i}", left, right, (supported, False, False, False))
        result.append(CachedPair(f"pair{i}", f"work{i}", features(left, 2 + i),
                                 features(right, -1), target))
    return tuple(result)


def make_trainer(bank=None, seed=17):
    from music_critic.critic.loss import CriticLoss
    return CriticTrainer(
        TinyCritic(), CriticLoss(), make_bank() if bank is None else bank,
        CriticTrainConfig(seed=seed, learning_rate=0.01, batch_size=2,
                          max_updates=10, max_attempts=12, clip_grad_norm=1.0),
        bindings={"source_id": "synthetic-source", "split_id": "TRAIN-v1", "target_version": "v1"},
    )


def test_unsupported_batch_advances_sampler_without_optimizer_update():
    trainer = make_trainer(make_bank(False))
    before = trainer.model.weight.detach().clone()
    result = trainer.step()
    assert result.applied is False and result.reason == "unsupported_targets"
    assert trainer.applied_updates == 0 and trainer.attempted_updates == 1
    assert trainer.skipped_updates == 1 and trainer.draws == 2
    assert torch.equal(before, trainer.model.weight)
    assert not trainer.optimizer.state


def test_resume_reproduces_sampler_dropout_losses_and_parameters(tmp_path):
    trainer = make_trainer()
    caller_rng = torch.get_rng_state().clone()
    trainer.step()
    trainer.step()
    assert torch.equal(caller_rng, torch.get_rng_state())
    checkpoint = tmp_path / "critic.pt"
    trainer.save(checkpoint)
    expected = [trainer.step() for _ in range(3)]
    expected_weight = trainer.model.weight.detach().clone()
    resumed = make_trainer()
    resumed.load(checkpoint)
    actual = [resumed.step() for _ in range(3)]
    assert actual == expected
    assert torch.equal(expected_weight, resumed.model.weight)
    assert resumed.draws == trainer.draws


def test_resume_binding_mismatch_is_failure_atomic(tmp_path):
    trainer = make_trainer()
    trainer.step()
    checkpoint = tmp_path / "critic.pt"
    trainer.save(checkpoint)
    other = make_trainer(seed=18)
    before = other.model.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="resume_binding"):
        other.load(checkpoint)
    assert torch.equal(before, other.model.weight)
    assert torch.equal(rng, torch.get_rng_state())
    assert other.applied_updates == 0


def test_sampling_work_sequence_does_not_depend_on_variant_count_and_swaps_sides():
    bank = make_bank(False)
    expanded = bank + tuple(replace(bank[0], pair_id=f"extra{i}") for i in range(12))
    first, second = make_trainer(bank), make_trainer(expanded)
    first_rows = [first.step() for _ in range(8)]
    second_rows = [second.step() for _ in range(8)]
    assert [r.work_ids for r in first_rows] == [r.work_ids for r in second_rows]
    assert {swap for row in first_rows for swap in row.swapped} == {False, True}


def test_dev_overlap_rejected_and_train_diagnostic_cannot_select():
    trainer = make_trainer()
    with pytest.raises(ValueError, match="evaluation_split_overlap"):
        trainer.evaluate(trainer.bank)
    diagnostic = trainer.evaluate(trainer.bank, allow_train_diagnostic=True)
    with pytest.raises(ValueError, match="diagnostic_not_for_selection"):
        trainer.selection.consider(diagnostic, update=0)
    dev = tuple(replace(p, work_id=f"dev-{p.work_id}", split="head_dev",
                        targets=replace(p.targets, work_id=f"dev-{p.work_id}")) for p in trainer.bank)
    with pytest.raises(ValueError, match="evaluation_split_overlap"):
        trainer.evaluate(dev)
    dev = tuple(CachedPair(
        f"dev-pair{i}", f"dev-work{i}", features(f"dev-left{i}", 10 + i), features(f"dev-right{i}", 0),
        pair_target(f"dev-work{i}", f"dev-left{i}", f"dev-right{i}"), split="head_dev",
    ) for i in range(2))
    result = trainer.evaluate(dev)
    assert result.selection_eligible
    assert trainer.selection.consider(result, update=0)


def test_selection_support_cannot_silently_change():
    row = (prediction("a", [1, 1, 1, 1]), prediction("b", [0, 0, 0, 0]), pair_target("work", "a", "b"))
    state = SelectionState(("harmony", "rhythm"))
    state.consider(evaluate_pairs([row], selection_aspects=state.declared_aspects), update=0)
    changed = (*row[:2], pair_target("work", "a", "b", (True, True, False, False)))
    with pytest.raises(ValueError, match="selection_support_changed"):
        state.consider(evaluate_pairs([changed], selection_aspects=state.declared_aspects), update=1)


def test_selection_cohort_cannot_change_midrun():
    row = (prediction("a", [1, 1, 1, 1]), prediction("b", [0, 0, 0, 0]), pair_target("work", "a", "b"))
    state = SelectionState(("harmony",))
    state.consider(evaluate_pairs([row], selection_aspects=state.declared_aspects), update=0)
    changed = (prediction("c", [1, 1, 1, 1]), prediction("d", [0, 0, 0, 0]), pair_target("different", "c", "d"))
    with pytest.raises(ValueError, match="selection_cohort_changed"):
        state.consider(evaluate_pairs([changed], selection_aspects=state.declared_aspects), update=1)


def test_nonfinite_gradient_fails_before_optimizer_step():
    trainer = make_trainer()
    before = trainer.model.weight.detach().clone()
    hook = trainer.model.weight.register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(ValueError, match="gradient_nonfinite"):
        trainer.step()
    hook.remove()
    assert torch.equal(before, trainer.model.weight)
    assert trainer.applied_updates == 0
    assert not trainer.optimizer.state


def test_feature_mutation_or_checkpoint_tampering_cannot_resume():
    trainer = make_trainer()
    trainer.step()
    state = trainer.state_dict()
    state["model"]["weight"].add_(1)
    with pytest.raises(ValueError, match="fingerprint_mismatch"):
        make_trainer().load_state_dict(state)
    bank = make_bank()
    bank[0].left.samples[0].embeddings["song"].add_(1)
    other = make_trainer(bank)
    with pytest.raises(ValueError, match="resume_binding_mismatch"):
        other.load_state_dict(trainer.state_dict())


def test_encoder_is_excluded_and_detached_features_are_required():
    from music_critic.critic.loss import CriticLoss
    trainer = make_trainer()
    encoder = torch.nn.Linear(4, 4).requires_grad_(False)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    guarded = CriticTrainer(TinyCritic(), CriticLoss(), trainer.bank, trainer.config,
                            bindings=trainer.bindings, frozen_backbones=(encoder,))
    encoder.train()
    guarded.step()
    assert not encoder.training
    assert all(torch.equal(before[k], v) for k, v in encoder.state_dict().items())
    assert all(p.grad is None for p in encoder.parameters())
    assert not {id(p) for p in encoder.parameters()} & {
        id(p) for group in guarded.optimizer.param_groups for p in group["params"]}
    left = features("left", 1)
    left.samples[0].embeddings["song"].requires_grad_(True)
    with pytest.raises(ValueError, match="features_not_detached"):
        CachedPair("x", "work", left, features("right", 0), pair_target("work", "left", "right"))


def test_optimizer_arithmetic_failure_restores_weights_and_slots():
    from music_critic.critic.loss import CriticLoss
    baseline = make_trainer()
    trainer = CriticTrainer(TinyCritic(), CriticLoss(), baseline.bank,
                            replace(baseline.config, learning_rate=1e100), bindings=baseline.bindings)
    before = trainer.model.weight.detach().clone()
    with pytest.raises((RuntimeError, ValueError)):
        trainer.step()
    assert torch.equal(before, trainer.model.weight)
    assert not trainer.optimizer.state


def test_live_bank_mutation_cannot_be_trained_or_checkpointed():
    trainer = make_trainer()
    trainer.bank[0].left.samples[0].embeddings["song"].add_(1)
    with pytest.raises(ValueError, match="bank_mutated"):
        trainer.step()
    with pytest.raises(ValueError, match="bank_mutated"):
        trainer.state_dict()
    assert trainer.attempted_updates == 0


def test_compiled_bank_guard_checks_shared_tensors_and_nested_mutable_metadata():
    from music_critic.critic.training import _BankMutationGuard
    tensor = torch.ones(2)
    metadata = {'nested': [{'tags': ['original']}], 'vectors': tensor}
    bank = (metadata, metadata)
    guard = _BankMutationGuard(bank)
    assert guard.matches(bank)
    metadata['nested'][0]['tags'].append('changed')
    assert not guard.matches(bank)
    metadata['nested'][0]['tags'].pop()
    assert guard.matches(bank)
    metadata['vectors'] = tensor.clone()
    assert not guard.matches(bank)
    metadata['vectors'] = tensor
    tensor.add_(1)
    assert not guard.matches(bank)


def test_compiled_bank_guard_preserves_fraction_and_frozen_geometry_contract():
    from music_critic.critic.training import _BankMutationGuard
    bank = make_bank()
    guard = _BankMutationGuard(bank)
    assert guard.matches(bank)
    bank[0].left.samples[0].stages['song'] = 'changed-stage'
    assert not guard.matches(bank)


def test_live_cached_features_cannot_start_requiring_gradients():
    trainer = make_trainer()
    trainer.bank[0].left.samples[0].embeddings['song'].requires_grad_(True)
    with pytest.raises(ValueError, match='bank_mutated'):
        trainer.step()
    assert trainer.attempted_updates == 0


def test_inference_tensor_guard_checks_grad_mode_as_well_as_bytes():
    from music_critic.critic.training import _BankMutationGuard
    with torch.inference_mode():
        value = torch.ones(2)
        bank = (value,)
        guard = _BankMutationGuard(bank)
        value.requires_grad_(True)
    assert not guard.matches(bank)
