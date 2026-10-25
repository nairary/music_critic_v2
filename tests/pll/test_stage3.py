"""Third-segment chaining: ancestral LR reproduction, full-state preservation, replay.

The load-bearing property is that ``Segment3Trainer`` reproduces the LR multiplier
its parent segment actually stored in the AdamW param groups, bit-for-bit, because
``runtime.PLLTrainer._validate_optimizer`` compares with ``!=``. ``hcm1_stage2``'s
parent branch reconstructs a plain cosine from update zero, which is exact only for
a first-generation parent.
"""
from hashlib import sha256
import random
from types import SimpleNamespace

import pytest
import torch

from music_critic.pll.runtime import PLLTrainer, TrainingConfig
from scripts.hcm1_stage2 import SegmentTrainer, continue_full_state
from scripts.hcm1_stage3 import (Segment3Trainer, ancestry_multiplier, continue_full_state_gen3,
                                 load_segment_parent, plain_multiplier, run,
                                 schedule_ancestry, segment_multiplier)
from tests.pll.test_runtime import trainer, example


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def gen1(total=3, warmup=1):
    base = trainer()
    p = PLLTrainer(base.model, TrainingConfig(total_updates=total, warmup_updates=warmup),
                   data_id='synthetic', protocol_id='gen1')
    for _ in range(total):
        p.step((example(),))
    return p


def gen2(p, additional=4):
    c = continue_full_state(p, additional_updates=additional, protocol_id='gen2',
                            parent_sha256='a' * 64, implementation_sha256='b' * 64)
    for _ in range(additional):
        c.step((example(),))
    return c


def gen3(c, additional=5):
    return continue_full_state_gen3(c, additional_updates=additional, protocol_id='gen3',
                                    parent_sha256='c' * 64, implementation_sha256='d' * 64)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_gen3_chains_and_preserves_full_state():
    torch.manual_seed(42)
    random.seed(42)
    g2 = gen2(gen1())
    binding_before = dict(g2.binding)
    rng_before = g2._rng_state()
    g3 = gen3(g2)
    assert isinstance(g3, Segment3Trainer) and isinstance(g3, SegmentTrainer)
    equal(g2.model.state_dict(), g3.model.state_dict())
    equal(g2.optimizer.state_dict(), g3.optimizer.state_dict())
    equal(rng_before, g3._rng_state())
    assert (g3.updates, g3.draws) == (g2.updates, g2.draws) == (7, 7)
    assert g3.segment_start == 7
    assert g3.config.total_updates == 7 + 5
    # The parent is never mutated by chaining.
    assert g2.binding == binding_before and g2.config.total_updates == 7


def test_gen3_reproduces_parent_last_lr_bit_identically():
    """The exact value _validate_optimizer compares at the gen-2/gen-3 boundary."""
    g2 = gen2(gen1())
    boundary = g2.updates - 1
    g3 = gen3(g2)                      # raises pll.checkpoint_optimizer_invalid if wrong
    assert g3._lr_multiplier(boundary) == g2._lr_multiplier(boundary)
    # And it must match what gen-2 actually left in the AdamW groups.
    for group in g2.optimizer.param_groups:
        assert group['lr'] == group['base_lr'] * g3._lr_multiplier(boundary)


def test_naive_plain_cosine_reconstruction_would_have_failed():
    """Proves the ancestry logic is load-bearing and cannot be silently reverted."""
    g2 = gen2(gen1())
    boundary = g2.updates - 1
    g3 = gen3(g2)
    c = g3.parent_config
    naive = .5 * (1 + __import__('math').cos(
        __import__('math').pi * (boundary - c.warmup_updates) / (c.total_updates - c.warmup_updates)))
    actual = g2._lr_multiplier(boundary)
    assert naive != actual
    assert g3._lr_multiplier(boundary) == actual


def test_ancestry_multiplier_bit_exact_over_both_full_domains():
    g2 = gen2(gen1())
    ancestry = schedule_ancestry(g2.binding)
    assert [a['generation'] for a in ancestry] == [0, 1]
    g1 = PLLTrainer(trainer().model, TrainingConfig(total_updates=3, warmup_updates=1),
                    data_id='synthetic', protocol_id='ref')
    for u in range(0, g2.config.total_updates):
        expected = g1._lr_multiplier(u) if u < ancestry[1]['segment_start'] else g2._lr_multiplier(u)
        assert ancestry_multiplier(ancestry, u) == expected, u


def test_plain_and_segment_multipliers_match_their_trainers_exactly():
    for total, warm in ((3, 1), (10, 2), (1000, 50)):
        t = PLLTrainer(trainer().model, TrainingConfig(total_updates=total, warmup_updates=warm),
                       data_id='synthetic', protocol_id='ref')
        for u in range(total):
            assert plain_multiplier(total, warm, u) == t._lr_multiplier(u), (total, warm, u)
    g2 = gen2(gen1())
    for u in range(g2.segment_start, g2.config.total_updates):
        assert segment_multiplier(g2.segment_start, g2.config.total_updates,
                                  g2.config.warmup_updates, u) == g2._lr_multiplier(u), u


def test_schedule_ancestry_is_derived_not_hardcoded():
    """Different horizons must produce different ancestries from the binding alone."""
    small = schedule_ancestry(gen2(gen1(total=2, warmup=1), additional=3).binding)
    large = schedule_ancestry(gen2(gen1(total=5, warmup=2), additional=7).binding)
    assert small != large
    assert small == [dict(generation=0, segment_start=0, total_updates=2, warmup_updates=1),
                     dict(generation=1, segment_start=2, total_updates=5, warmup_updates=1)]
    assert large == [dict(generation=0, segment_start=0, total_updates=5, warmup_updates=2),
                     dict(generation=1, segment_start=5, total_updates=12, warmup_updates=2)]
    with pytest.raises(ValueError, match='contiguous'):
        schedule_ancestry({'training': {'total_updates': 9, 'warmup_updates': 1},
                           'continuation': {'start_update': 3, 'parent_binding': {
                               'training': {'total_updates': 4, 'warmup_updates': 1}}}})


def test_gen3_binding_records_generation_and_ancestry():
    g2 = gen2(gen1())
    g3 = gen3(g2)
    cont = g3.binding['continuation']
    assert cont['schema'] == 'HCM1NewScheduleSegment@1'
    assert cont['segment_generation'] == 3
    assert cont['start_update'] == 7 and cont['additional_updates'] == 5
    assert cont['ancestry']['schema'] == 'HCMScheduleAncestry@1'
    assert cont['ancestry']['segments'] == schedule_ancestry(g2.binding)
    assert cont['parent_binding'] == g2.binding


def test_gen3_lr_schedule_warms_up_peaks_and_reaches_exactly_zero():
    g2 = gen2(gen1())
    g3 = gen3(g2)
    assert g3.segment_start == 7 and g3.config.total_updates == 12
    assert g3._lr_multiplier(7) == 1.0          # warmup step, warmup_updates=1
    assert g3._lr_multiplier(8) == 1.0          # cosine peak
    assert g3._lr_multiplier(12) == 0.0         # exact zero at the new horizon
    mid = g3._lr_multiplier(10)
    assert 0.0 < mid < 1.0


def test_gen3_resume_replays_next_step_exactly(tmp_path):
    g2 = gen2(gen1())
    g3 = gen3(g2)
    g3.step((example(),))
    path = tmp_path / 'gen3.pt'
    digest = g3.save_checkpoint(path, sampler_state={'draws': 8})
    expected = g3.step((example(),))
    expected_weights = g3.model.state_dict()
    again = gen3(g2)
    assert again.load_checkpoint(path, expected_sha256=digest) == {'draws': 8}
    assert again.step((example(),)) == expected
    equal(expected_weights, again.model.state_dict())
    assert sha256(path.read_bytes()).hexdigest() == digest


def test_unfinished_segment_parent_rejected():
    g2 = continue_full_state(gen1(), additional_updates=4, protocol_id='gen2',
                             parent_sha256='a' * 64, implementation_sha256='b' * 64)
    g2.step((example(),))                      # updates 4 of 7, not at the boundary
    assert g2.updates != g2.config.total_updates
    with pytest.raises(ValueError, match='completed'):
        gen3(g2)


def test_first_generation_parent_is_rejected_by_stage3():
    """stage3 must not be used where stage2 is the correct tool."""
    with pytest.raises(ValueError, match='segment parent'):
        continue_full_state_gen3(gen1(), additional_updates=5, protocol_id='gen3',
                                 parent_sha256='c' * 64, implementation_sha256='d' * 64)


def test_non_frozen_parent_rejected():
    base = trainer()
    p = PLLTrainer(base.model, TrainingConfig(total_updates=3, warmup_updates=1, unfreeze_update=1),
                   data_id='synthetic', protocol_id='unfreezing')
    with pytest.raises(ValueError):
        continue_full_state_gen3(p, additional_updates=5, protocol_id='gen3',
                                 parent_sha256='c' * 64, implementation_sha256='d' * 64)


def test_strict_plain_trainer_still_refuses_a_gen3_checkpoint(tmp_path):
    """The invariant locked by test_stage2 must survive the third generation."""
    g2 = gen2(gen1())
    g3 = gen3(g2)
    path = tmp_path / 'gen3.pt'
    digest = g3.save_checkpoint(path, sampler_state={})
    plain = PLLTrainer(trainer().model, TrainingConfig(total_updates=12, warmup_updates=1),
                       data_id='synthetic', protocol_id='plain')
    with pytest.raises(ValueError, match='binding'):
        plain.load_checkpoint(path, expected_sha256=digest)


def test_gen2_trainer_also_refuses_a_gen3_checkpoint(tmp_path):
    g1 = gen1()
    g2 = gen2(g1)
    g3 = gen3(g2)
    path = tmp_path / 'gen3.pt'
    digest = g3.save_checkpoint(path, sampler_state={})
    # Match the total horizon so rejection proves the generation binding,
    # rather than merely rejecting a different config length.
    gen2_only = continue_full_state(g1, additional_updates=9, protocol_id='gen2',
                                    parent_sha256='a' * 64, implementation_sha256='b' * 64)
    assert gen2_only.config.total_updates == g3.config.total_updates
    with pytest.raises(ValueError, match='binding'):
        gen2_only.load_checkpoint(path, expected_sha256=digest)


def test_stage3_rejects_wrong_declared_segment_sha_before_parent_loading(tmp_path):
    checkpoint = tmp_path / 'segment.pt'
    checkpoint.write_bytes(b'valid checkpoint bytes for this test')
    with pytest.raises(ValueError, match='stage3 segment checkpoint SHA-256 mismatch'):
        load_segment_parent(None, None, None, None, None, None,
                            segment_checkpoint=checkpoint, segment_sha256='0' * 64,
                            segment_binding=None, segment_record=None, device='cpu')


@pytest.mark.parametrize('checkpoint_every,additional_updates', [(0, 1), (100, 0)])
def test_stage3_rejects_invalid_run_budget_before_record_transition(
        tmp_path, checkpoint_every, additional_updates):
    args = SimpleNamespace(output=tmp_path / 'output', record=tmp_path / 'records' / 'run.json',
                           wall_seconds=10, checkpoint_every=checkpoint_every,
                           additional_updates=additional_updates)
    with pytest.raises(ValueError, match='positive wall budget, checkpoint cadence'):
        run(args)
    assert not args.output.exists()
