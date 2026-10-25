"""Fit/resume on one documentation fixture; not scientific training evidence."""
from pathlib import Path
import torch

from music_critic.data import load_piece
from music_critic.experiments.music_interventions import Intervention,NotePatch,apply_interventions
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel,model_state_fingerprint
from music_critic.critic.contracts import FeatureBinding
from music_critic.critic.features import BackboneFeatureAdapter,C2_ADAPTER_VERSION
from music_critic.critic.model import MultiscaleAspectCritic
from music_critic.critic.targets import InterventionTargetBuilder,VerifiedPatch
from music_critic.critic.loss import CriticLoss
from music_critic.critic.training import CachedPair,CriticTrainer,CriticTrainConfig


def test_full_critic_fits_fixture_and_exactly_resumes_without_touching_encoder(tmp_path):
    torch.manual_seed(21)
    original=load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json')
    patch=NotePatch('note:melody-000',pitch=68)
    variant=apply_interventions(original,(Intervention('synthetic-pitch',(patch,)),)).piece
    targets=InterventionTargetBuilder().build(original,variant,VerifiedPatch(
        original.piece_id,variant.piece_id,'fixture-work',(patch,),('harmony','melody'),'pitch'))
    backbone=CorrectedAnalysisGNNModel()
    before=model_state_fingerprint(backbone)
    adapter=BackboneFeatureAdapter(backbone,FeatureBinding('synthetic','random-init',C2_ADAPTER_VERSION))
    left,right=adapter.extract((original,)),adapter.extract((variant,))
    pair=CachedPair('fixture-pair','fixture-work',left,right,targets)
    cfg=CriticTrainConfig(seed=17,learning_rate=.005,batch_size=1,max_updates=80,
                         max_attempts=80,clip_grad_norm=2.,selection_aspects=('harmony','melody'))
    bindings={'source_id':'synthetic','split_id':'synthetic-fixture','target_version':'v1'}
    model=MultiscaleAspectCritic(feature_dim=128,width=8)
    loss=CriticLoss()
    trainer=CriticTrainer(model,loss,(pair,),cfg,bindings=bindings,frozen_backbones=(backbone,))
    initial=float(loss((model(left),model(right)),(targets,)).total.detach())
    for _ in range(3): assert trainer.step().applied
    checkpoint=tmp_path/'resume.pt'; trainer.save(checkpoint)
    resumed=CriticTrainer(MultiscaleAspectCritic(feature_dim=128,width=8),CriticLoss(),(pair,),cfg,
                         bindings=bindings,frozen_backbones=(backbone,))
    resumed.load(checkpoint)
    a,b=trainer.step(),resumed.step()
    assert a==b
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,resumed.model.state_dict()[k],rtol=0,atol=0)
    for _ in range(66): trainer.step()
    result=loss((model(left),model(right)),(targets,))
    assert float(result.total.detach()) < initial*.7
    p,q=model(left).samples[0],model(right).samples[0]
    assert p.raw_scores[0]>q.raw_scores[0]
    assert p.raw_scores[2]>q.raw_scores[2]
    assert model_state_fingerprint(backbone)==before
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
