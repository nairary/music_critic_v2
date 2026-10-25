"""Synthetic fixed-exposure curriculum contracts; no corpus or checkpoint reads."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from music_critic.critic.contracts import (
    ASPECTS, CriticFeaturePyramid, CriticOutput, FeatureBinding, FeatureSample,
    SampleGeometry, SamplePrediction,
)
from music_critic.critic.loss import CriticLoss
from music_critic.critic.targets import PairTargets
from music_critic.critic.training import (
    CachedPair, CriticTrainConfig, CriticTrainer, CurriculumStage,
)


class TinyCritic(torch.nn.Module):
    """Real optimizer/dropout seam with entirely synthetic detached inputs."""
    def __init__(self, supported=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.dropout = torch.nn.Dropout(0.2)
        self.supported = supported

    def forward(self, pyramid):
        return CriticOutput(tuple(SamplePrediction(
            sample.sample_id, self.weight * self.dropout(sample.embeddings["song"][0]),
            torch.full((4,), self.supported, dtype=torch.bool), {}, sample.geometry.duration,
        ) for sample in pyramid.samples))


def features(name, value):
    sample = FeatureSample(
        SampleGeometry(name, (), (), {}, Fraction(4)),
        {"song": torch.full((1, 4), float(value))},
        f"synthetic-{name}", {"song": "synthetic"},
    )
    return CriticFeaturePyramid((sample,), FeatureBinding("synthetic", "weights", "v1"))


def make_bank(*, split="train"):
    pairs = []
    for genre in ("jazz", "rock"):
        for number in range(2):
            work = f"{split}-{genre}-{number}"
            for aspect_index, aspect in enumerate(ASPECTS):
                for count in (1, 2, 3, 4):
                    pair_id = f"{work}-{aspect}-{count}"
                    left, right = f"{work}-original", f"{pair_id}-variant"
                    mask = tuple(index == aspect_index for index in range(4))
                    target = PairTargets(work, (left, right), mask,
                                         tuple(int(value) for value in mask), ({}, {}), {})
                    pairs.append(CachedPair(
                        pair_id, work, features(left, 2), features(right, -1), target,
                        split=split, genre=genre, edit_count=count,
                    ))
    return tuple(pairs)


def config(**changes):
    initial = CriticTrainConfig(
        seed=17, learning_rate=0.01, batch_size=2, max_updates=10,
        max_attempts=10, clip_grad_norm=1.0,
        curriculum_stages=(CurriculumStage(6, (3, 4)),
                           CurriculumStage(12, (2, 1)), CurriculumStage(20, (1,))),
    )
    return replace(initial, **changes)


def make_trainer(bank=None, *, cfg=None, supported=True, lambda_local=0):
    return CriticTrainer(
        TinyCritic(supported), CriticLoss(lambda_local),
        make_bank() if bank is None else bank, config() if cfg is None else cfg,
        bindings={"source_id": "synthetic", "split_id": "grouped-v1", "target_version": "weak-v1"},
    )


def test_stage_boundaries_and_journal_follow_exposures():
    trainer = make_trainer()
    rows = [trainer.step() for _ in range(10)]
    stages = [stage for row in rows for stage in row.source_stages]
    counts = [count for row in rows for count in row.edit_counts]
    assert stages == [0] * 6 + [1] * 6 + [2] * 8
    assert set(counts[:6]) <= {3, 4}
    assert set(counts[6:12]) <= {1, 2}
    assert counts[12:] == [1] * 8
    assert all(len(row.aspects) == len(row.genres) == 2 for row in rows)
    assert trainer.draws == 20
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        trainer.step()


def test_shuffled_control_has_exact_same_exposures_and_side_swaps():
    ordered = make_trainer()
    shuffled = make_trainer(cfg=config(sampling_order="shuffled"))
    assert Counter(ordered.exposure_plan) == Counter(shuffled.exposure_plan)
    assert ordered.exposure_plan != shuffled.exposure_plan
    assert ordered.metadata["exposure_plan_fingerprint"] != shuffled.metadata["exposure_plan_fingerprint"]
    assert ordered.exposure_plan == make_trainer().exposure_plan
    assert ordered.exposure_plan != make_trainer(cfg=config(seed=29)).exposure_plan


def test_variant_count_does_not_change_work_aspect_count_or_genre_choices():
    bank = make_bank()
    expanded = bank + tuple(replace(bank[0], pair_id=f"extra-{number}") for number in range(13))
    ordered, more_variants = make_trainer(bank), make_trainer(expanded)
    works = {pair.pair_id: pair.work_id for pair in expanded}
    signature = lambda trainer: [(works[row.pair_id], row.aspect, row.edit_count,
                                  row.genre, row.swapped, row.source_stage)
                                 for row in trainer.exposure_plan]
    assert signature(ordered) == signature(more_variants)


def test_genre_is_sampled_before_work_for_unequal_work_counts():
    bank = make_bank()
    first = make_trainer(bank)
    # More works in one genre must not alter the sequence of genres/aspects/k.
    extra = tuple(replace(
        pair, pair_id=f"extra-{pair.pair_id}", work_id=f"extra-{pair.work_id}",
        targets=replace(pair.targets, work_id=f"extra-{pair.work_id}"),
    ) for pair in bank if pair.work_id == "train-jazz-0")
    second = make_trainer(bank + extra)
    signature = lambda trainer: [(row.genre, row.aspect, row.edit_count) for row in trainer.exposure_plan]
    assert signature(first) == signature(second)


@pytest.mark.parametrize("missing", [
    lambda pair: pair.work_id == "train-jazz-0" and pair.edit_count == 4,
    lambda pair: pair.work_id == "train-rock-0" and pair.targets.pair_mask[2],
])
def test_missing_work_aspect_count_coverage_is_rejected_without_fallback(missing):
    with pytest.raises(ValueError, match="curriculum_coverage_missing"):
        make_trainer(tuple(pair for pair in make_bank() if not missing(pair)))


@pytest.mark.parametrize("changes, message", [
    ({"edit_count": None}, "curriculum_metadata_required"),
    ({"genre": None}, "curriculum_metadata_required"),
    ({"edit_count": 5}, "curriculum_unused_edit_count"),
    ({"genre": "different"}, "curriculum_work_genre_changed"),
])
def test_curriculum_metadata_is_required_and_bound_to_work(changes, message):
    bank = make_bank()
    with pytest.raises(ValueError, match=message):
        make_trainer((replace(bank[0], **changes), *bank[1:]))


@pytest.mark.parametrize("mask", [(False,) * 4, (True, True, False, False)])
def test_curriculum_requires_one_observed_aspect_per_pair(mask):
    bank = make_bank()
    target = replace(bank[0].targets, pair_mask=mask, preference=tuple(int(value) for value in mask))
    with pytest.raises(ValueError, match="curriculum_single_aspect_required"):
        make_trainer((replace(bank[0], targets=target), *bank[1:]))


def test_curriculum_requires_global_pair_only_loss():
    with pytest.raises(ValueError, match="curriculum_pair_only_required"):
        make_trainer(lambda_local=0.25)


@pytest.mark.parametrize("changes", [
    {"max_attempts": 11},
    {"curriculum_stages": (CurriculumStage(18, (1, 2, 3, 4)),)},
    {"curriculum_stages": (CurriculumStage(5, (3, 4)), CurriculumStage(20, (1, 2)))},
    {"curriculum_stages": (CurriculumStage(6, (3, 4)), CurriculumStage(6, (2, 1)),
                           CurriculumStage(20, (1,)))},
])
def test_schedule_requires_increasing_batch_aligned_boundaries_and_fixed_budget(changes):
    with pytest.raises(ValueError, match="curriculum_budget_invalid"):
        config(**changes)


@pytest.mark.parametrize("end, counts", [(0, (1,)), (True, (1,)), (2, ()), (2, (1, 1)),
                                        (2, (False,)), (2, (0,)), (2, (1.5,))])
def test_invalid_stage_values_are_rejected(end, counts):
    with pytest.raises(ValueError, match="curriculum_(boundary|counts)_invalid"):
        CurriculumStage(end, counts)


@pytest.mark.parametrize("field, value", [("edit_count", True), ("edit_count", 0),
                                        ("edit_count", 1.5), ("genre", " "), ("genre", 7)])
def test_invalid_pair_metadata_is_rejected(field, value):
    with pytest.raises(ValueError, match=f"{field}_invalid"):
        replace(make_bank()[0], **{field: value})


def test_skips_consume_fixed_exposures_and_reach_final_stage():
    trainer = make_trainer(supported=False)
    rows = [trainer.step() for _ in range(10)]
    assert all(not row.applied for row in rows)
    assert trainer.applied_updates == 0
    assert trainer.skipped_updates == trainer.attempted_updates == 10
    assert trainer.draws == 20
    assert rows[-1].source_stages == (2, 2)
    assert rows[-1].edit_counts == (1, 1)
    assert not trainer.optimizer.state
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        trainer.step()


@pytest.mark.parametrize("sampling_order", ["curriculum", "shuffled"])
def test_resume_reproduces_boundary_sampling_dropout_optimizer_and_weights(tmp_path, sampling_order):
    cfg = config(sampling_order=sampling_order)
    trainer = make_trainer(cfg=cfg)
    caller_rng = torch.get_rng_state().clone()
    trainer.step()
    trainer.step()
    assert torch.equal(torch.get_rng_state(), caller_rng)
    checkpoint = tmp_path / "synthetic.pt"
    trainer.save(checkpoint)  # Next two steps straddle the first stage boundary.
    expected = [trainer.step() for _ in range(5)]
    resumed = make_trainer(cfg=cfg)
    resumed.load(checkpoint)
    actual = [resumed.step() for _ in range(5)]
    assert actual == expected
    assert torch.equal(resumed.model.weight, trainer.model.weight)
    assert resumed.state_dict()["fingerprint"] == trainer.state_dict()["fingerprint"]


@pytest.mark.parametrize("binding_change", ["schedule", "order", "genre", "count"])
def test_resume_rejects_changed_curriculum_binding_without_mutation(binding_change):
    original = make_trainer()
    original.step()
    cfg, bank = config(), make_bank()
    if binding_change == "schedule":
        cfg = config(curriculum_stages=(CurriculumStage(8, (3, 4)), CurriculumStage(12, (2, 1)),
                                        CurriculumStage(20, (1,))))
    elif binding_change == "order":
        cfg = config(sampling_order="shuffled")
    elif binding_change == "genre":
        bank = tuple(replace(pair, genre=f"renamed-{pair.genre}") for pair in bank)
    else:
        # Preserve complete coverage while changing what authored count binds
        # to which variant. This is a different bank even if shapes match.
        bank = tuple(replace(pair, edit_count=3 if pair.edit_count == 4 else
                             4 if pair.edit_count == 3 else pair.edit_count) for pair in bank)
    changed = make_trainer(bank, cfg=cfg)
    before = changed.state_dict()["fingerprint"]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="resume_binding_mismatch"):
        changed.load_state_dict(original.state_dict())
    assert changed.state_dict()["fingerprint"] == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_curriculum_dev_requires_explicit_single_edit_count_but_diagnostic_can_use_train():
    trainer = make_trainer()
    dev = tuple(pair for pair in make_bank(split="head_dev") if pair.edit_count == 1)
    result = trainer.evaluate(dev)
    assert result.selection_eligible
    assert result.supported_aspects == ASPECTS
    with pytest.raises(ValueError, match="curriculum_single_edit_dev_required"):
        trainer.evaluate(make_bank(split="head_dev"))
    with pytest.raises(ValueError, match="curriculum_single_edit_dev_required"):
        trainer.evaluate(tuple(replace(pair, edit_count=None) for pair in dev))
    diagnostic = trainer.evaluate(trainer.bank, allow_train_diagnostic=True)
    assert not diagnostic.selection_eligible


def test_empty_curriculum_preserves_legacy_api_and_unknown_metadata():
    bank = tuple(replace(pair, edit_count=None, genre=None) for pair in make_bank())
    cfg = config(curriculum_stages=(), max_attempts=12)
    trainer = make_trainer(bank, cfg=cfg, lambda_local=0.25)
    assert trainer.exposure_plan == ()
    assert trainer.metadata["exposure_plan_fingerprint"] is None
    row = trainer.step()
    assert row.applied
    assert row.edit_counts == row.genres == (None, None)
    assert row.source_stages == ()
    with pytest.raises(ValueError, match="sampling_order_invalid"):
        replace(cfg, sampling_order="shuffled")


def test_runtime_cannot_replace_bound_exposure_plan():
    trainer = make_trainer()
    trainer.exposure_plan = tuple(reversed(trainer.exposure_plan))
    with pytest.raises(ValueError, match="exposure_plan_mutated"):
        trainer.step()
    with pytest.raises(ValueError, match="exposure_plan_mutated"):
        trainer.state_dict()
    assert trainer.draws == 0
