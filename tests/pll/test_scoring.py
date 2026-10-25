from dataclasses import replace
from pathlib import Path

import pytest
import torch

from music_critic.data import RationalTime as R, load_piece
from music_critic.pll.contracts import CodecProfile, MaskQuery, ModelConfig
from music_critic.pll.model import HCM1
from music_critic.pll.scoring import score_piece


def test_score_reports_empty_support_and_never_calls_it_quality():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        p=load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json')
        p=replace(p,notes=())
        cfg=ModelConfig(width=32,heads=4,event_layers=1,global_layers=1,ff_dim=64,
                        field_width=16,field_heads=4,field_ff_dim=32,relation_dim=8,dropout=0)
        model=HCM1(cfg,codec_profile=CodecProfile(max_events=2,max_denominator=4,max_numerator=64)).eval()
        query=MaskQuery(R(1),R(5),(p.tracks[0].track_id,))
        scored=score_piece(model,p,(query,))
        assert scored['requested']==scored['supported']==1
        assert scored['blocks'][0]['nats_per_event'] is None
        assert scored['blocks'][0]['nats_per_qn']>=0
        assert scored['interpretation']=='conditional_symbolic_likelihood_not_aesthetic_quality'
        too_many=replace(p,tracks=p.tracks*9)
        unsupported=score_piece(model,too_many,(query,))
        assert unsupported['supported']==0 and unsupported['blocks'][0]['nll'] is None
        with pytest.raises(ValueError,match='TEST'):
            score_piece(model,replace(p,split='TEST'),(query,))
    finally:
        torch.set_num_threads(old)
