"""Single-MIDI scoring and JSON-ready exact-time exports.

Real-data use is an experiment under the repository logging protocol. This
module does not download weights, initialize a cloud run or select checkpoints.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch

from music_critic.adapters.midi import MidiAdapterConfig, load_midi_piece
from .contracts import ASPECTS, FeatureBinding
from .scoring import AspectScorePolicy


def predictor_identity(critic, feature_binding, critic_id):
    """Bind relative scales to actual head weights and their encoder features."""
    digest = sha256(json.dumps({'critic_id': critic_id, 'feature_binding': asdict(feature_binding),
        'model': type(critic).__module__ + '.' + type(critic).__qualname__,
        'definition': repr(critic)}, sort_keys=True).encode())
    for name, value in sorted(critic.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CriticInference:
    def __init__(self, adapter, critic, *, feature_binding: FeatureBinding,
                 critic_id: str, reference=None, weights=None):
        if feature_binding != adapter.binding:
            raise ValueError('critic/backbone feature binding mismatch')
        if not critic_id or critic_id.lower() == 'unknown':
            raise ValueError('critic identity required')
        self.adapter, self.critic = adapter, critic
        self.feature_binding, self.critic_id = feature_binding, critic_id
        self.reference, self.weights = reference, weights
        self.policy = AspectScorePolicy()
        self.predictor_id = predictor_identity(critic, feature_binding, critic_id)
        if reference is not None and reference.predictor_id != self.predictor_id:
            raise ValueError('reference bank belongs to a different critic or feature binding')
        self._guard = self._weight_guard()

    def _weight_guard(self):
        return tuple((name, value._version, value.data_ptr(), value.device, value.dtype)
                     for name, value in self.critic.state_dict(keep_vars=True).items())

    def score_midi(self, path):
        piece = load_midi_piece(path, config=MidiAdapterConfig(dataset_name='critic-inference'))
        return self.score_piece(piece)

    def score_piece(self, piece):
        if self.feature_binding != self.adapter.binding:
            raise ValueError('critic/backbone feature binding changed')
        if self._guard != self._weight_guard():
            raise ValueError('critic changed; bind a new inference instance and reference bank')
        pyramid = self.adapter.extract((piece,))
        was_training = self.critic.training
        try:
            self.critic.eval()
            with torch.no_grad():
                output = self.critic(pyramid.to(next(self.critic.parameters()).device))
        finally:
            self.critic.train(was_training)
        scored = self.policy.from_output(output, self.reference, self.weights,
                                         predictor_id=self.predictor_id)
        prediction = output.samples[0]
        local = {}
        for name, values in prediction.local.items():
            rows = []
            for c, logit, available in zip(values.candidates, values.logits, values.availability):
                rows.append({
                    'candidate_id': c.candidate_id,
                    'start_qn': [c.start.numerator, c.start.denominator],
                    'end_qn': [c.end.numerator, c.end.denominator],
                    'available': bool(available),
                    'intervention_evidence_logit': float(logit) if available else None,
                    'intervention_evidence_sigmoid': float(logit.sigmoid()) if available else None,
                })
            local[name] = rows
        report = {
            'schema': 'CriticInferenceReport@1.0.0', 'sample_id': prediction.sample_id,
            'critic_id': self.critic_id, 'feature_binding': asdict(self.feature_binding),
            'predictor_id': self.predictor_id,
            'raw_hash': pyramid.samples[0].raw_hash,
            'horizon_qn': [prediction.horizon.numerator, prediction.horizon.denominator],
            'support_policy': prediction.support_policy,
            'aspects': {a: {'raw_score': scored.raw_scores[a],
                            'reference_percentile': scored.percentiles[a],
                            'reason': scored.unavailable_reasons.get(a)} for a in ASPECTS},
            'overall': scored.overall, 'coverage': scored.weighted_coverage,
            'weights': dict(scored.weights), 'reference_fingerprint': scored.reference_fingerprint,
            'local': local,
            'local_claim': 'synthetic intervention evidence; not calibrated musical-error probability',
            'musical_error_location': {'available': False, 'reason': prediction.musical_error_reason},
        }
        json.dumps(report, allow_nan=False)  # Fail before emitting a nonfinite report.
        return report

    @staticmethod
    def write_report(report, path):
        """Never replace an earlier prediction artifact silently."""
        with Path(path).open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')


def save_critic_bundle(path, critic, feature_binding, *, critic_id):
    """Publish only trained-head weights and bindings, using a fresh artifact path.

    The caller owns checkpoint selection and experiment registration. This does
    not declare the supplied weights trained or scientifically validated.
    """
    from .model import MultiscaleAspectCritic
    if type(critic) is not MultiscaleAspectCritic:
        raise ValueError('v1 inference bundle requires the multiscale critic')
    if not critic_id or not feature_binding.model_state_id:
        raise ValueError('critic and actual encoder state identities required')
    payload = {'schema': 'CriticInferenceBundle@1.0.0', 'critic_id': critic_id,
        'feature_binding': asdict(feature_binding),
        'config': {'feature_dim': critic.feature_dim, 'width': critic.width},
        'state': {k:v.detach().cpu().clone() for k,v in critic.state_dict().items()},
        'predictor_id': predictor_identity(critic, feature_binding, critic_id)}
    if not all(torch.isfinite(v).all() for v in payload['state'].values()):
        raise ValueError('nonfinite critic state')
    with Path(path).open('xb') as stream:
        torch.save(payload, stream)


def load_critic_bundle(path, *, expected_binding):
    """Return an eval-mode head plus its identity; never loads backbone objects."""
    from .model import MultiscaleAspectCritic
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('schema') != 'CriticInferenceBundle@1.0.0'
            or payload.get('feature_binding') != asdict(expected_binding)):
        raise ValueError('inference bundle binding mismatch')
    # Creating modules must not advance the caller's training RNG stream.
    with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
        critic = MultiscaleAspectCritic(**payload['config'])
        critic.load_state_dict(payload['state'], strict=True)
    critic.eval()
    if predictor_identity(critic, expected_binding, payload['critic_id']) != payload['predictor_id']:
        raise ValueError('inference bundle integrity mismatch')
    return critic, payload['critic_id']
