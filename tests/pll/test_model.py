from pathlib import Path

import pytest
import torch

from music_critic.data import RationalTime as R, load_piece
from music_critic.pll.contracts import CodecProfile, ContextProfile, MaskQuery, ModelConfig
from music_critic.pll.masking import split_masked_piece
from music_critic.pll.model import HCM1


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model(dropout=0):
    return HCM1(ModelConfig(width=32, heads=4, event_layers=1, global_layers=1, ff_dim=64,
                           field_width=16, field_heads=4, field_ff_dim=32, relation_dim=8,
                           relation_components=2, dropout=dropout),
                codec_profile=CodecProfile(max_events=8, max_denominator=12, max_numerator=1024))


def example():
    piece = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    return split_masked_piece(piece, MaskQuery(R(1), R(5), (piece.tracks[0].track_id,)))


def test_real_tiled_c3_to_likelihood_backward():
    torch.manual_seed(12)
    model = small_model()
    result = model(example())
    assert torch.isfinite(result.nll)
    result.nll.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.memory.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    assert all(p.grad is None for p in model.encoder.backbone.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_batch_reference_is_same_per_example_in_eval_and_bindings_are_checked():
    from dataclasses import replace
    model = small_model().eval()
    item = example()
    with torch.no_grad():
        singles = model(item)
        batch = model.batch_likelihood((item,item))
    torch.testing.assert_close(batch[0].nll, singles.nll, rtol=0, atol=0)
    torch.testing.assert_close(batch[1].nll, singles.nll, rtol=0, atol=0)
    with pytest.raises(ValueError, match='track'):
        model(replace(item, tracks=(1,)))


def test_public_generation_uses_query_geometry_and_reaches_eos():
    from dataclasses import replace
    from music_critic.pll.codec import NOTE, EOS
    cfg = replace(small_model().config, relations='none')
    model = HCM1(cfg, codec_profile=CodecProfile(max_events=1, max_denominator=1,
                                                max_numerator=1)).eval()
    with torch.no_grad():
        model.decoder.output.weight.zero_()
        model.decoder.output.bias.zero_()
        model.decoder.output.bias[NOTE] = 20
        model.decoder.output.bias[EOS] = -20
    item = example()
    generated = model.generate(item.context, generator=torch.Generator().manual_seed(11))
    assert len(generated) == 1
    assert generated[0].track in item.tracks
    assert R(0) <= generated[0].onset < item.span
    tokens = model.codec.encode(generated, span=item.span, tracks=item.tracks)
    assert tokens[-1] == EOS
    assert model.codec.decode(tokens, span=item.span, tracks=item.tracks) == generated
    assert model(replace(item, targets=generated)).nll.isfinite()
    with pytest.raises(ValueError, match='token_budget'):
        model.generate(item.context, max_tokens=1, generator=torch.Generator().manual_seed(11))
    model.train()
    with pytest.raises(ValueError, match='eval_mode'):
        model.generate(item.context)
