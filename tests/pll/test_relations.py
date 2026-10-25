import torch

from music_critic.pll.contracts import ModelConfig
from music_critic.pll.relations import RelationRefiner


def test_relations_normalize_preserve_support_and_receive_gradients():
    torch.manual_seed(3)
    config = ModelConfig(width=32, heads=4, field_width=16, field_heads=4, relation_dim=8,
                         relation_components=2, dropout=0)
    module = RelationRefiner(config, vocab_size=7)
    hidden = torch.randn(3, 16, requires_grad=True)
    base = torch.randn(3, 7, requires_grad=True)
    allowed = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0], [1]*7], dtype=torch.bool)
    families = {name: torch.randn(2, 32, requires_grad=True) for name in module.family_names}
    result = module(base, hidden, ('pitch_class', 'octave', 'onset_num_byte'), allowed, families)
    torch.testing.assert_close(result.exp().sum(-1), torch.ones(3))
    assert torch.isneginf(result[~allowed]).all()
    assert torch.isfinite(result[allowed]).all()
    torch.testing.assert_close(result[2], base[2].log_softmax(-1))
    (-result[0, 0] - result[1, 1]).backward()
    assert families['vertical'].grad.abs().sum() > 0
    assert families['melodic'].grad.abs().sum() > 0
    assert module.experts['vertical'].relations.grad.abs().sum() > 0


def test_disabled_branch_is_exact_masked_decoder_distribution():
    config = ModelConfig(relations='none')
    module = RelationRefiner(config, 3)
    logits = torch.tensor([[1., 3., 9.]])
    mask = torch.tensor([[True, True, False]])
    actual = module(logits, torch.zeros(1, 128), ('event',), mask, {})
    torch.testing.assert_close(actual, logits.masked_fill(~mask, -torch.inf).log_softmax(-1))
