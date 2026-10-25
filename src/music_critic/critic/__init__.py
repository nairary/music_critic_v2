"""Frozen-backbone aspect scoring and synthetic intervention localization.

No weights or datasets are loaded on import. Scores are relative; localization
of interventions is not validated localization of musical errors.
"""
from .contracts import ASPECTS, CriticFeaturePyramid, FeatureBinding
from .features import BackboneFeatureAdapter
from .model import MultiscaleAspectCritic
from .targets import InterventionTargetBuilder, VerifiedPatch
from .loss import CriticLoss, evaluate_local_interventions
from .scoring import AspectScorePolicy, ReferenceRow, TrainReferenceBank
from .training import CachedPair, CriticTrainConfig, CriticTrainer
from .inference import CriticInference, load_critic_bundle, save_critic_bundle

__all__ = [
    'ASPECTS', 'CriticFeaturePyramid', 'FeatureBinding', 'BackboneFeatureAdapter',
    'MultiscaleAspectCritic', 'InterventionTargetBuilder', 'VerifiedPatch',
    'CriticLoss', 'evaluate_local_interventions', 'AspectScorePolicy', 'ReferenceRow',
    'TrainReferenceBank', 'CachedPair', 'CriticTrainConfig', 'CriticTrainer',
    'CriticInference', 'load_critic_bundle', 'save_critic_bundle',
]
