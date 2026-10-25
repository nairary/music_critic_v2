import pytest
import torch
from music_critic.critic.baselines import BeatProbe, MultilevelPoolingCritic, RawSummaryCritic
from .test_model import fixture_sample, pyramid


@pytest.mark.parametrize('model',[BeatProbe(8),MultilevelPoolingCritic(8,8),RawSummaryCritic()])
def test_controls_share_aspects_and_unavailable_boundary(model):
    out=model(pyramid(fixture_sample(),fixture_sample('empty',empty=True)))
    assert out.samples[0].raw_scores.shape==(4,)
    assert torch.isfinite(out.samples[0].raw_scores).all()
    assert not out.samples[1].availability.any()
    assert out.samples[0].musical_error_location is None
    assert out.samples[0].local=={}
    out.samples[0].raw_scores.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
