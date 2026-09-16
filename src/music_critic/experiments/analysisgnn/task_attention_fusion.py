"""AnalysisGNN-style task attention, adapted to the twelve aligned beat heads.

Only the fusion mechanism is reproduced; entity routing, encoder and loss are
V2 contracts. Upstream reference: analysisgnn e115182f, analysis.py 409–418,
497–512, 546–569. No upstream runtime import and no target inputs.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from io import BytesIO
import torch
from torch import nn
from .corrected_model import CorrectedAnalysisGNNModel, corrected_model_contract, model_state_fingerprint
from .contracts import fingerprint
from .harmonic_logit_fusion import _specs, _validate_predictions

MODEL_SCHEMA = 'TaskAttentionFusionModel@1.0.0'
CHECKPOINT_SCHEMA = 'TaskAttentionModelOnlyCheckpoint@1.0.0'


class TaskAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        _, harmonic = _specs()
        self.task_ids = tuple(s.task_id for s in harmonic)
        self.projections = nn.ModuleDict({s.task_id: nn.Sequential(
            nn.Linear(s.output_dim, 64), nn.ReLU(), nn.LayerNorm(64)) for s in harmonic})
        self.attention = nn.MultiheadAttention(64, 4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.outputs = nn.ModuleDict({s.task_id: nn.Linear(64, s.output_dim) for s in harmonic})

    def forward(self, predictions):
        device = _validate_predictions(predictions)
        original = {p.task_id: p for p in predictions}
        if original[self.task_ids[0]].logits.shape[0] == 0:
            return predictions
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = torch.stack([self.projections[h](original[h].logits) for h in self.task_ids], dim=1)
            attended, _ = self.attention(tokens, tokens, tokens)
            refined = self.norm(tokens + attended)
            logits = {h: self.outputs[h](refined[:, i]) for i, h in enumerate(self.task_ids)}
        result = tuple(replace(p, logits=logits[p.task_id]) if p.task_id in logits else p for p in predictions)
        _validate_predictions(result)
        return result


@dataclass(frozen=True)
class TaskAttentionOutput:
    encoder: object
    predictions: tuple
    schema_version: str = MODEL_SCHEMA

    @property
    def logits(self):
        return {p.task_id: p.logits for p in self.predictions}


class TaskAttentionModel(nn.Module):
    def __init__(self, base, *, enabled=True):
        super().__init__()
        if type(base) is not CorrectedAnalysisGNNModel or type(enabled) is not bool:
            raise ValueError('task_attention.constructor')
        self.base = base
        self.enabled = enabled
        self.fusion = TaskAttentionFusion() if enabled else None
        self.training = base.training
        if self.fusion is not None:
            self.fusion.train(base.training)

    def predict(self, raw_graph_batch, *, return_layers=False):
        output = self.base.predict(raw_graph_batch, return_layers=return_layers)
        if self.fusion is None:
            return output
        return TaskAttentionOutput(output.encoder, self.fusion(output.predictions))

    def forward(self, raw_graph_batch):
        return self.predict(raw_graph_batch)


def model_contract(model):
    if type(model) is not TaskAttentionModel or model.enabled != (model.fusion is not None):
        raise ValueError('task_attention.model_contract')
    specs, harmonic = _specs()
    contract = {'schema_version': MODEL_SCHEMA, 'enabled': model.enabled,
        'base_model_contract': corrected_model_contract(model.base),
        'task_order': [s.task_id for s in specs],
        'fusion_tasks': [{'id': s.task_id, 'classes': s.output_dim} for s in harmonic],
        'projection_dim': 64, 'attention_heads': 4, 'attention_dropout': 0.1,
        'projection': 'Linear_ReLU_LayerNorm', 'fusion': 'LayerNorm(P+MHA(P,P,P))_then_task_Linear',
        'raw_logit_residual': False, 'detach': False, 'zero_initialization': False,
        'input': 'twelve_aligned_raw_beat_predictions_no_targets',
        'scope': 'official_style_fusion_mechanism_only_not_full_AnalysisGNN_reproduction',
        'parameters': {'base': sum(p.numel() for p in model.base.parameters()),
                       'fusion': 0 if model.fusion is None else sum(p.numel() for p in model.fusion.parameters())}}
    return {**contract, 'fingerprint': fingerprint(contract)}


def create_model_payload(model):
    contract = model_contract(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {'schema_version': CHECKPOINT_SCHEMA, 'model_contract': contract,
        'model_contract_fingerprint': contract['fingerprint'], 'model_state': state,
        'model_state_fingerprint': model_state_fingerprint(state)}


def load_model_checkpoint(path, *, expected_sha256, expected_size_bytes, enabled):
    from music_critic.experiments.pop909_cl_frozen.checkpoint import (
        _checked_artifact, _preserve_caller_rng, _validate_model_contract, _validate_model_state)
    with _preserve_caller_rng():
        content = _checked_artifact(path, expected_sha256, expected_size_bytes)
        payload = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
        if type(payload) is not dict or set(payload) != {'schema_version', 'model_contract',
            'model_contract_fingerprint', 'model_state', 'model_state_fingerprint'} or payload['schema_version'] != CHECKPOINT_SCHEMA:
            raise ValueError('task_attention.checkpoint_schema')
        model = TaskAttentionModel(CorrectedAnalysisGNNModel(), enabled=enabled)
        expected = model_contract(model)
        _validate_model_contract(payload['model_contract'], expected)
        if payload['model_contract_fingerprint'] != expected['fingerprint']:
            raise ValueError('task_attention.checkpoint_contract')
        _validate_model_state(payload['model_state'], model)
        if model_state_fingerprint(payload['model_state']) != payload['model_state_fingerprint']:
            raise ValueError('task_attention.checkpoint_state')
        model.load_state_dict(payload['model_state'], strict=True)
        return model.eval().requires_grad_(False)
