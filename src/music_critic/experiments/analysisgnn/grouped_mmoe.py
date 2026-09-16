"""Two dense expert banks on raw contextual candidates, before task classifiers.

Beat tasks share one bank; note and onset tasks share another bank's weights,
but never concatenate or attend across entity rows. Every task owns its gate.
The unchanged task-attention fusion follows the twelve harmonic classifiers.
"""
from dataclasses import dataclass
from io import BytesIO

import torch
from torch import nn

from .contracts import fingerprint
from .corrected_model import CorrectedAnalysisGNNModel, corrected_model_contract, model_state_fingerprint
from .harmonic_logit_fusion import _specs
from .task_attention_fusion import TaskAttentionFusion

MODEL_SCHEMA = 'GroupedMMoEAttentionModel@1.0.0'
CHECKPOINT_SCHEMA = 'GroupedMMoEAttentionModelOnlyCheckpoint@1.0.0'
STATE_SCHEMA = 'GroupedMMoEAttentionCampaignState@1.0.0'


class ExpertBank(nn.Module):
    def __init__(self, task_ids, dim=128):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
            for _ in range(4)])
        self.gates = nn.ModuleDict({task: nn.Linear(dim, 4) for task in task_ids})

    def expert_values(self, values):
        return torch.stack([expert(values) for expert in self.experts], dim=1)

    def mix(self, task, values, experts):
        weights = torch.softmax(self.gates[task](values), dim=-1)
        return (experts * weights.unsqueeze(-1)).sum(dim=1), weights


@dataclass(frozen=True)
class GroupedMMoEOutput:
    encoder: object
    predictions: tuple
    schema_version: str = MODEL_SCHEMA

    @property
    def logits(self):
        return {p.task_id: p.logits for p in self.predictions}


class GroupedMMoEAttentionModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        if type(base) is not CorrectedAnalysisGNNModel:
            raise ValueError('grouped_mmoe.base_type')
        self.base = base
        # Preserve the existing attention initialization before creating experts.
        self.fusion = TaskAttentionFusion()
        specs, beat_specs = _specs()
        self.routes = {s.task_id: s.node_types[0] for s in specs}
        beat_tasks = tuple(s.task_id for s in beat_specs)
        fine_tasks = tuple(s.task_id for s in specs if s.node_types[0] != 'beat')
        if len(fine_tasks) != 6 or set(self.routes.values()) != {'beat', 'note', 'onset'}:
            raise ValueError('grouped_mmoe.routing_inventory')
        self.banks = nn.ModuleDict({'beat': ExpertBank(beat_tasks), 'note_onset': ExpertBank(fine_tasks)})
        self.capture_routing = False
        self._routing = {}
        self.train(base.training)

    def predict_from_encoded(self, encoded):
        """Raw encoder output only; no labels, availability or gold boundaries."""
        self._routing = {}
        with torch.autocast(device_type=encoded.fused.embeddings['beat'].device.type, enabled=False):
            raw = {node: encoded.fused.embeddings[node].float() for node in ('beat', 'note', 'onset')}
            experts = {node: self.banks['beat' if node == 'beat' else 'note_onset'].expert_values(values)
                       for node, values in raw.items()}

            def transform(task, values):
                node = self.routes[task]
                bank = self.banks['beat' if node == 'beat' else 'note_onset']
                mixed, weights = bank.mix(task, values.float(), experts[node])
                if self.capture_routing:
                    w = weights.detach()
                    count = w.shape[0]
                    self._routing[task] = {'entity': node, 'count': count,
                        'mean_weights': w.mean(0) if count else w.new_zeros(4),
                        'mean_entropy': -(w * w.clamp_min(1e-12).log()).sum(-1).mean() if count else w.new_zeros(()),
                        'top1_counts': torch.bincount(w.argmax(-1), minlength=4)}
                return mixed

            predictions = self.base.task_heads(encoded.fused, feature_transform=transform)
            return GroupedMMoEOutput(encoded, self.fusion(predictions))

    def predict(self, raw_graph_batch, *, return_layers=False):
        return self.predict_from_encoded(self.base.encode(raw_graph_batch, return_layers=return_layers))

    def forward(self, raw_graph_batch):
        return self.predict(raw_graph_batch)

    def routing_report(self):
        return {task: {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in row.items()}
                for task, row in self._routing.items()}


def model_contract(model):
    if type(model) is not GroupedMMoEAttentionModel:
        raise ValueError('grouped_mmoe.model_type')
    value = {'schema_version': MODEL_SCHEMA, 'base_model_contract': corrected_model_contract(model.base),
        'groups': {'beat': [t for t,n in model.routes.items() if n == 'beat'],
                   'note_onset': [t for t,n in model.routes.items() if n != 'beat']},
        'entity_routes': model.routes, 'experts_per_group': 4, 'expert_dimensions': [128, 256, 128],
        'expert_activation': 'GELU', 'expert_dropout': 0.0,
        'gate': 'per_task_Linear128_4_softmax_dense_event_dependent',
        'expert_input': 'post_hierarchy_and_onset_BiGRU_raw_embeddings',
        'note_onset_sharing': 'shared_expert_parameters_separate_entity_rows',
        'placement': 'experts_then_existing_task_MLPs_then_existing12task_attention',
        'additional_loss': None, 'feature_residual': False, 'targets_in_prediction': False,
        'parameters': {'base': sum(p.numel() for p in model.base.parameters()),
                       'attention': sum(p.numel() for p in model.fusion.parameters()),
                       'mmoe': sum(p.numel() for p in model.banks.parameters())}}
    return {**value, 'fingerprint': fingerprint(value)}


def create_model_payload(model):
    contract = model_contract(model)
    state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return {'schema_version': CHECKPOINT_SCHEMA, 'model_contract': contract,
            'model_contract_fingerprint': contract['fingerprint'], 'model_state': state,
            'model_state_fingerprint': model_state_fingerprint(state)}


def load_model_checkpoint(path, *, expected_sha256, expected_size_bytes):
    from music_critic.experiments.pop909_cl_frozen.checkpoint import (
        _checked_artifact, _preserve_caller_rng, _validate_model_contract, _validate_model_state)
    with _preserve_caller_rng():
        content = _checked_artifact(path, expected_sha256, expected_size_bytes)
        payload = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
        if type(payload) is not dict or set(payload) != {'schema_version','model_contract',
                'model_contract_fingerprint','model_state','model_state_fingerprint'} or payload['schema_version'] != CHECKPOINT_SCHEMA:
            raise ValueError('grouped_mmoe.checkpoint_schema')
        model = GroupedMMoEAttentionModel(CorrectedAnalysisGNNModel())
        expected = model_contract(model)
        _validate_model_contract(payload['model_contract'], expected)
        if payload['model_contract_fingerprint'] != expected['fingerprint']:
            raise ValueError('grouped_mmoe.checkpoint_contract')
        _validate_model_state(payload['model_state'], model)
        if model_state_fingerprint(payload['model_state']) != payload['model_state_fingerprint']:
            raise ValueError('grouped_mmoe.checkpoint_state')
        model.load_state_dict(payload['model_state'], strict=True)
        return model.eval().requires_grad_(False)
