"""Per-evaluation prediction reuse on synthetic immutable feature views only."""
from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from music_critic.critic.contracts import (
    CriticFeaturePyramid, CriticOutput, FeatureBinding, FeatureSample,
    SampleGeometry, SamplePrediction,
)
from music_critic.critic.loss import CriticLoss
from music_critic.critic.targets import PairTargets
from music_critic.critic.training import (
    CachedPair, CriticTrainConfig, CriticTrainer, _fingerprint, evaluate_pairs,
)


class CountingCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.calls = []
        self.grad_modes = []
        self.on_forward = lambda _sample: None

    def forward(self, pyramid):
        self.calls.append(tuple(sample.sample_id for sample in pyramid.samples))
        self.grad_modes.append(torch.is_grad_enabled())
        predictions = []
        for sample in pyramid.samples:
            predictions.append(SamplePrediction(
                sample.sample_id, self.weight * sample.embeddings['song'][0],
                torch.ones(4, dtype=torch.bool), {}, sample.geometry.duration,
            ))
            self.on_forward(sample)
        return CriticOutput(tuple(predictions))


def view(name, values):
    sample = FeatureSample(
        SampleGeometry(name, (), (), {}, Fraction(4)),
        {'song': torch.tensor([values], dtype=torch.float32)},
        'synthetic-raw-' + name, {'song': 'synthetic'},
    )
    return CriticFeaturePyramid((sample,), FeatureBinding('synthetic', 'weights', 'v1'))


def pair(name, left, right, *, split='head_dev', mask=(True,) * 4, preference=None):
    work = split + '-work'
    target = PairTargets(work, (left.samples[0].sample_id, right.samples[0].sample_id),
                         mask, preference or tuple(int(value) for value in mask), ({}, {}), {})
    return CachedPair(name, work, left, right, target, split=split)


def trainer():
    training_pair = pair('train-pair', view('train-original', (2,) * 4),
                         view('train-variant', (-1,) * 4), split='train')
    return CriticTrainer(
        CountingCritic(), CriticLoss(lambda_local=0), (training_pair,),
        CriticTrainConfig(17, .02, 1, 4, 4, 1.),
        bindings={'source_id': 'synthetic', 'split_id': 'synthetic', 'target_version': 'v1'},
    )


def dev_bank():
    original = view('dev-original', (1, 2, 3, 4))
    first = view('dev-first', (.5, 3, 3, 1))
    second = view('dev-second', (2, 1, 3, 5))
    return (
        pair('first', original, first),
        pair('second', original, second, mask=(True, True, False, True),
             preference=(-1, 1, 0, -1)),
        pair('unknown', first, original, mask=(False,) * 4),
    )


def uncached_reference(model_trainer, bank):
    previous = model_trainer.model.training
    model_trainer.model.eval()
    try:
        with torch.no_grad():
            outputs = [model_trainer._forward_pairs((row,)) for row in bank]
        result = evaluate_pairs([
            (left.samples[0], right.samples[0], row.targets)
            for row, (left, right) in zip(bank, outputs)
        ], selection_aspects=model_trainer.config.selection_aspects,
            tie_tolerance=model_trainer.config.tie_tolerance)
        return replace(result, evaluation_fingerprint=_fingerprint(bank))
    finally:
        model_trainer.model.train(previous)


def test_repeated_views_reuse_predictions_but_preserve_pair_metrics_and_first_use_order():
    subject, bank = trainer(), dev_bank()
    expected = uncached_reference(subject, bank)
    assert len(subject.model.calls) == 2 * len(bank)
    subject.model.calls.clear()
    subject.model.grad_modes.clear()
    actual = subject.evaluate(bank)
    assert actual == expected
    assert subject.model.calls == [('dev-original',), ('dev-first',), ('dev-second',)]
    assert not any(subject.model.grad_modes)
    assert subject.model.training
    assert all(parameter.grad is None for parameter in subject.model.parameters())
    assert actual.per_aspect['melody'] == .5  # An exact score tie stays a tie.
    assert actual.pair_support == {'harmony': 2, 'rhythm': 2, 'melody': 1, 'structure': 2}


def test_reuse_does_not_survive_a_training_step_or_another_evaluation():
    subject, bank = trainer(), dev_bank()
    before = subject.evaluate(bank)
    subject.step()
    subject.model.calls.clear()
    after = subject.evaluate(bank)
    assert len(subject.model.calls) == 3
    assert after.pair_loss != before.pair_loss
    assert after == uncached_reference(subject, bank)


def test_same_sample_id_and_raw_hash_do_not_alias_distinct_feature_contents():
    subject = trainer()
    first = view('same-id', (1,) * 4)
    changed = replace(first, samples=(replace(
        first.samples[0], embeddings={'song': torch.full((1, 4), 3.)}),))
    variant = view('variant', (2,) * 4)
    bank = (pair('one', first, variant), pair('two', changed, variant))
    expected = uncached_reference(subject, bank)
    subject.model.calls.clear()
    actual = subject.evaluate(bank)
    assert actual == expected and actual.pair_accuracy == .5
    assert subject.model.calls == [('same-id',), ('variant',), ('same-id',)]


@pytest.mark.parametrize('inference_tensor', [False, True])
@pytest.mark.parametrize('mutate_on', ['dev-first', 'dev-second'])
def test_mutated_feature_view_is_rejected_before_reuse_or_return(inference_tensor, mutate_on):
    subject = trainer()
    with torch.inference_mode(inference_tensor):
        bank = dev_bank()
    original = bank[0].left.samples[0]
    def mutate(sample):
        if sample.sample_id == mutate_on:
            with torch.inference_mode(inference_tensor):
                original.embeddings['song'].add_(1)
    subject.model.on_forward = mutate
    with pytest.raises(ValueError, match='evaluation_bank_mutated'):
        subject.evaluate(bank)
    assert subject.model.training


def test_replaced_feature_mapping_entry_is_rejected_even_with_same_sample_identity():
    subject, bank = trainer(), dev_bank()
    original = bank[0].left.samples[0]
    def mutate(sample):
        if sample.sample_id == 'dev-first':
            original.embeddings['song'] = original.embeddings['song'].clone()
    subject.model.on_forward = mutate
    with pytest.raises(ValueError, match='evaluation_bank_mutated'):
        subject.evaluate(bank)


@pytest.mark.parametrize('binding', ['geometry', 'stages'])
def test_mutated_geometry_or_stage_binding_cannot_reuse_a_prediction(binding):
    subject, bank = trainer(), dev_bank()
    original = bank[0].left.samples[0]
    def mutate(sample):
        if sample.sample_id == 'dev-first':
            if binding == 'geometry':
                original.geometry.intervals['slice'] = ()
            else:
                original.stages['song'] = 'changed-stage'
    subject.model.on_forward = mutate
    with pytest.raises(ValueError, match='evaluation_bank_mutated'):
        subject.evaluate(bank)


def test_existing_training_bank_guard_runs_before_any_evaluation_forward():
    subject = trainer()
    subject.bank[0].left.samples[0].embeddings['song'].add_(1)
    with pytest.raises(ValueError, match='bank_mutated'):
        subject.evaluate(dev_bank())
    assert subject.model.calls == []
