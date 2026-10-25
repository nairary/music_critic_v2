"""Synthetic constant packing; no learned arithmetic or musical-quality claim."""
import copy
from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from music_critic.critic.geometry_plan import geometry_plan
from .test_model import fixture_sample


def test_prepared_constants_preserve_scalar_conversion_and_index_order():
    from music_critic.critic.device_geometry import prepare_constants
    plan = geometry_plan(fixture_sample().geometry)
    for dtype in (torch.float32, torch.float64):
        zero = torch.zeros(8, dtype=dtype, requires_grad=True)
        constants = prepare_constants(plan, zero)
        for row in (*plan.notes, *plan.events, *plan.slices, *plan.bars):
            assert torch.equal(constants.timing(row.timing), zero.new_tensor(row.timing))
        for row in (*plan.bars, *plan.beat_weights):
            weights = row.weights if hasattr(row, 'weights') else row
            assert torch.equal(constants.weights(weights), zero.new_tensor(weights.dense()))
        for track in plan.tracks:
            assert torch.equal(constants.indices(track.events), torch.tensor(track.events, dtype=torch.long))
        for size in (1, 2, len(plan.notes)):
            count, one = constants.pool.scalars(size)
            assert torch.equal(count, zero.new_tensor([size]).log1p())
            assert torch.equal(one, zero.new_ones(1))
        assert all(not t.requires_grad and t.grad_fn is None for t in constants.storages)


def test_constant_packing_is_per_forward_and_preserves_empty_geometry():
    from music_critic.critic.device_geometry import prepare_constants
    plan = geometry_plan(fixture_sample(empty=True).geometry)
    a = prepare_constants(plan, torch.zeros(8))
    b = prepare_constants(plan, torch.zeros(8, dtype=torch.float64))
    assert a.pool.scalars(1)[0].dtype == torch.float32
    assert b.pool.scalars(1)[0].dtype == torch.float64
    assert a.indices(()).dtype == torch.long
    assert a.indices(()).numel() == 0


def test_forward_does_not_materialize_host_float_constants_per_row(monkeypatch):
    from music_critic.critic.model import MultiscaleAspectCritic
    from .test_model import pyramid
    model = MultiscaleAspectCritic(8, 8)
    sample = fixture_sample()
    original = torch.Tensor.new_tensor
    calls = []
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'new_tensor', counted)
    model(pyramid(sample))
    assert len(calls) <= 2  # One float pack and one pool-count pack, irrespective of rows.


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif is_dataclass(left):
        for field in fields(left):
            assert_exact(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('width', [8, 128])
@pytest.mark.parametrize('delta', [0., 1e-4])
def test_full_pair_step_matches_serial_arithmetic_exactly(dtype, width, delta):
    from music_critic.critic.loss import CriticLoss
    from music_critic.critic.model import MultiscaleAspectCritic
    from music_critic.critic.targets import PairTargets
    from .row_reference import Source004Critic
    from .test_model import pyramid

    torch.manual_seed(17)
    candidate = MultiscaleAspectCritic(8, width).to(dtype=dtype)
    reference = Source004Critic(8, width).to(dtype=dtype)
    reference.load_state_dict(copy.deepcopy(candidate.state_dict()))
    original = fixture_sample('left')
    variant = replace(fixture_sample('right'), embeddings={
        key: value + delta for key, value in original.embeddings.items()})
    inputs = (pyramid(original), pyramid(variant))
    input_copy = copy.deepcopy(inputs)
    target = PairTargets('synthetic', ('left', 'right'), (True, False, False, False),
                         (1, 0, 0, 0), ({}, {}), {})
    optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.)
                  for m in (reference, candidate)]
    calls = [[], []]
    for index, model in enumerate((reference, candidate)):
        for name, layer in model.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.GRU, torch.nn.LayerNorm)):
                layer.register_forward_pre_hook(lambda _, args, name=name, index=index:
                    calls[index].append((name, tuple((tuple(t.shape), t.stride())
                                                    for t in args if isinstance(t, torch.Tensor)))))
    for _ in range(2):
        snapshots = []
        for model, optimizer in zip((reference, candidate), optimizers):
            optimizer.zero_grad(set_to_none=True)
            outputs = tuple(model(view) for view in inputs)
            loss = CriticLoss(0)(outputs, (target,))
            loss.total.backward()
            gradients = {key: None if p.grad is None else p.grad.detach().clone()
                         for key, p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            with torch.no_grad():
                after = tuple(model(view) for view in inputs)
            snapshots.append((outputs, gradients, copy.deepcopy(model.state_dict()),
                              copy.deepcopy(optimizer.state_dict()), after))
        assert_exact(*snapshots)
    assert calls[0] == calls[1]
    assert_exact(inputs, input_copy)
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    assert not tuple(candidate.buffers())


def test_packed_gathers_preserve_duplicate_unsorted_indices_and_gradient():
    from music_critic.critic.device_geometry import prepare_constants
    plan = geometry_plan(fixture_sample().geometry)
    duplicate = (2, 0, 2, 1)
    plan = replace(plan, active_tracks=duplicate)
    constants = prepare_constants(plan, torch.zeros(8))
    a = torch.randn(3, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    x, y = a[list(duplicate)], b[constants.indices(duplicate)]
    assert_exact(x, y)
    x.square().sum().backward()
    y.square().sum().backward()
    assert_exact(a.grad, b.grad)


def test_mutated_geometry_gets_new_constants_and_empty_pools_bypass_parameters():
    from music_critic.critic.device_geometry import prepare_constants
    from music_critic.critic.model import SetPool
    sample = fixture_sample()
    first = geometry_plan(sample.geometry)
    original = first.bars[0].weights.dense()
    sample.geometry.intervals['slice'] = ()
    second = geometry_plan(sample.geometry)
    a = prepare_constants(first, torch.zeros(8))
    b = prepare_constants(second, torch.zeros(8))
    assert_exact(a.weights(first.bars[0].weights), torch.tensor(original))
    assert b.weights(second.bars[0].weights).numel() == 0
    pool = SetPool(8)
    assert_exact(pool(torch.empty(0, 8), b.pool), torch.zeros(8))
    assert all(p.grad is None for p in pool.parameters())


@pytest.mark.parametrize('level', ['beat', 'onset'])
def test_many_overlapping_parents_match_serial_forward_and_backward(level):
    from fractions import Fraction
    from music_critic.critic.contracts import RawInterval
    from music_critic.critic.model import MultiscaleAspectCritic
    from .row_reference import Source004Critic
    from .test_model import pyramid
    sample = fixture_sample()
    intervals = dict(sample.geometry.intervals)
    intervals[level] = tuple(RawInterval(f'overlap{i}', Fraction(0), Fraction(4))
                             for i in range(40))
    sample = replace(sample, geometry=replace(sample.geometry, intervals=intervals),
                     embeddings={**sample.embeddings, level: torch.randn(40, 8)})
    reference, candidate = Source004Critic(8, 8), MultiscaleAspectCritic(8, 8)
    candidate.load_state_dict(reference.state_dict())
    outputs = []
    for model in (reference, candidate):
        out = model(pyramid(sample))
        out.samples[0].raw_scores.square().sum().backward()
        outputs.append((out, {k: p.grad for k, p in model.named_parameters()}))
    assert_exact(*outputs)
