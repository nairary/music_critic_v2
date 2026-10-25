from types import SimpleNamespace

import pytest
import torch

from music_critic.data import RationalTime as R
from music_critic.pll.contracts import CodecProfile, Event, ModelConfig
from music_critic.pll.codec import EventCodec
from music_critic.pll.decoder import ConditionalEventDecoder


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup():
    torch.manual_seed(4)
    cfg = ModelConfig(width=32, heads=4, event_layers=1, global_layers=1, ff_dim=64,
                      field_width=16, field_heads=4, field_ff_dim=32,
                      relation_dim=8, relation_components=2, dropout=0)
    codec = EventCodec(CodecProfile(max_events=4, max_denominator=8, max_numerator=1024))
    model = ConditionalEventDecoder(cfg, codec).eval()
    memory = SimpleNamespace(values=torch.randn(3, 32, requires_grad=True),
                             families={k: torch.randn(2, 32, requires_grad=True)
                                       for k in ('activity','rhythm','vertical','melodic')})
    return model, memory, codec


def test_joint_nll_empty_and_event_gradients():
    model, memory, codec = setup()
    events = (Event(R(0), 0, 0, 60, R(1, 3)),)
    result = model(events, memory, span=R(4), tracks=(0,))
    assert result.event_count == 1
    assert result.token_count == len(codec.encode(events, span=R(4), tracks=(0,)))
    assert result.nll.isfinite() and result.nll > 0
    torch.testing.assert_close(result.nll, -result.token_log_probs.sum())
    result.nll.backward()
    assert memory.values.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    empty = model((), memory, span=R(4), tracks=(0,))
    assert empty.event_count == 0 and empty.token_count == 1
    assert empty.nll.isfinite()


def test_future_event_and_field_changes_do_not_change_prior_predictions():
    model, memory, codec = setup()
    a = (Event(R(0), 0, 0, 60, R(1)), Event(R(1), 0, 0, 64, R(1)))
    b = (Event(R(0), 0, 0, 60, R(1000, 3)), Event(R(2), 0, 0, 70, R(2)))
    ta, tb = (codec.encode(x, span=R(4), tracks=(0,)) for x in (a,b))
    pa = model.token_distributions(ta, memory, span=R(4), tracks=(0,))
    pb = model.token_distributions(tb, memory, span=R(4), tracks=(0,))
    first_difference = next(i for i,(x,y) in enumerate(zip(ta,tb)) if x != y)
    torch.testing.assert_close(pa[:first_difference+1], pb[:first_difference+1], atol=2e-6, rtol=2e-6)
    for i in (0, first_difference):
        online = model.next_token_log_probs(ta[:i], memory, span=R(4), tracks=(0,))
        torch.testing.assert_close(online, pa[i], atol=2e-6, rtol=2e-6)


def test_previous_event_preserves_attribute_roles_not_only_token_multiset():
    model, memory, codec = setup()
    a = codec.encode((Event(R(1), 0, 0, 60, R(2)),), span=R(4), tracks=(0,))[:-1]
    b = codec.encode((Event(R(2), 0, 0, 60, R(1)),), span=R(4), tracks=(0,))[:-1]
    assert sorted(a) == sorted(b)
    pa = model.next_token_log_probs(a, memory, span=R(4), tracks=(0,))
    pb = model.next_token_log_probs(b, memory, span=R(4), tracks=(0,))
    assert (pa[:2] - pb[:2]).abs().max() > 1e-6


def test_field_capacity_must_cover_the_entire_codec_profile():
    from dataclasses import replace
    model, memory, _ = setup()
    codec = EventCodec(CodecProfile(max_events=1, max_denominator=1, max_numerator=1))
    with pytest.raises(ValueError, match='field_capacity'):
        ConditionalEventDecoder(replace(model.config, max_fields=12), codec)
    # A full one-byte-time pitched event needs thirteen fields, plus separate EOS.
    boundary = ConditionalEventDecoder(replace(model.config, max_fields=13), codec).eval()
    events = (Event(R(1), 0, 0, 60, R(1)),)
    value = boundary(events, memory, span=R(2), tracks=(0,))
    assert value.token_count == 14 and value.nll.isfinite()
    with pytest.raises(ValueError, match='field_capacity'):
        ConditionalEventDecoder(replace(model.config, max_fields=13), EventCodec())
