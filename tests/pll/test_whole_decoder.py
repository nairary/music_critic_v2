from dataclasses import replace

import pytest
import torch

from music_critic.data import RationalTime as Q
from music_critic.pll.codec import EventCodec
from music_critic.pll.contracts import Event, MaskQuery
from music_critic.pll.memory import FAMILY_LEVELS, MemoryToken
from music_critic.pll.whole_decoder import MultiBranchMemory, WholeEventDecoder
from tests.pll.whole_fixtures import small_codec, small_config


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def memory():
    values = torch.randn(6, 32, requires_grad=True)
    query = MaskQuery(Q(0), Q(4), ('track:0',))
    return MultiBranchMemory(values, {k:values for k in FAMILY_LEVELS},
        tuple(MemoryToken('context',Q(0),Q(4),None,1.) for _ in values), query,
        branches={'local':values[:2], 'global':values[2:4], 'motif':values[4:]}, diagnostics={})


def test_all_memory_branches_receive_gradients_and_future_fields_are_causal():
    torch.manual_seed(7)
    decoder = WholeEventDecoder(small_config(), EventCodec(small_codec())).eval()
    m = memory()
    events = (Event(Q(0), 0, 0, 60, Q(1)), Event(Q(1), 0, 0, 62, Q(1)))
    output = decoder(events, m, span=Q(4), tracks=(0,))
    output.nll.backward()
    for branch in ('local','global','motif'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for name,p in decoder.named_parameters()
                   if (('main.layers' in name and 'multihead_attn' in name) if branch=='local' else branch in name))
    changed = (events[0], replace(events[1], pitch=65))
    a = decoder.codec.encode(events, span=Q(4), tracks=(0,))
    b = decoder.codec.encode(changed, span=Q(4), tracks=(0,))
    prefix = next(i for i,(x,y) in enumerate(zip(a,b)) if x != y)
    pa = decoder.token_distributions(a, m, span=Q(4), tracks=(0,))
    pb = decoder.token_distributions(b, m, span=Q(4), tracks=(0,))
    torch.testing.assert_close(pa[:prefix+1], pb[:prefix+1], rtol=0, atol=0)


def test_invalid_branch_geometry_is_rejected():
    m = memory()
    with pytest.raises(ValueError, match='branch'):
        replace(m, branches={'local':torch.randn(1, 3)})
