"""Typed, normalized soft refinement; latent components have no gold meaning."""
from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import ModelConfig

FAMILY_NAMES = ('activity', 'rhythm', 'vertical', 'melodic')
ENERGY_ROUTES = {'event': ('activity',), 'track': ('activity',),
                 'pitch_class': ('vertical',), 'octave': ('melodic',), 'drum': ('activity',)}


class _FamilyExpert(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        d, k = config.relation_dim, config.relation_components
        self.query = nn.Linear(config.field_width, d, bias=False)
        self.key = nn.Linear(config.width, d, bias=False)
        self.value = nn.Linear(config.width, d, bias=False)
        self.prefix = nn.Linear(config.field_width, d, bias=False)
        self.candidates = nn.Embedding(vocab_size, d)
        self.relations = nn.Parameter(torch.randn(k, d) / sqrt(d))
        self.mixture = nn.Linear(config.field_width, k + 1)
        self.gate = nn.Linear(config.field_width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.)
        self.log_temperature = nn.Parameter(torch.zeros(k))
        self.mode = config.relations

    def context(self, hidden: Tensor, memory: Tensor) -> Tensor:
        if memory.ndim != 2 or not len(memory):
            raise ValueError('pll.empty_relation_memory')
        weights = (self.query(hidden) @ self.key(memory).T / sqrt(self.query.out_features)).softmax(-1)
        return weights @ self.value(memory) + self.prefix(hidden)

    def forward(self, hidden: Tensor, allowed: Tensor, memory: Tensor) -> Tensor:
        context = self.context(hidden, memory)
        candidates = self.candidates.weight
        if self.mode == 'transr':
            a = context[:, None, :] + self.relations[None, :, :]
            # Expanded squared distance avoids a [tokens, components, vocab, dim] temporary.
            energy = (2 * torch.einsum('tkd,vd->tkv', a, candidates)
                      - a.square().sum(-1, keepdim=True) - candidates.square().sum(-1)[None, None, :])
        else:
            energy = torch.einsum('td,kd,vd->tkv', context, self.relations, candidates) / sqrt(candidates.shape[-1])
        energy = energy / (F.softplus(self.log_temperature)[None, :, None] + .1)
        distributions = energy.masked_fill(~allowed[:, None, :], -torch.inf).log_softmax(-1)
        uniform = -allowed.sum(-1).to(hidden.dtype).log()[:, None, None]
        uniform = uniform.expand(-1, 1, allowed.shape[-1]).masked_fill(~allowed[:, None, :], -torch.inf)
        distributions = torch.cat((distributions, uniform), dim=1)
        # All-infinite logsumexp columns have undefined backward even when the
        # resulting column is later masked. Only valid candidates carry mass.
        distributions = distributions.masked_fill(~allowed[:, None, :], 0.)
        mixture = self.mixture(hidden).log_softmax(-1)
        log_q = torch.logsumexp(distributions + mixture[:, :, None], dim=1)
        # Invalid entries are masked at the final softmax, avoiding 0 * -inf.
        return self.gate(hidden).sigmoid() * log_q.masked_fill(~allowed, 0.)


class RelationRefiner(nn.Module):
    """Per-field local mixture, not marginalization of a persistent global plan."""
    family_names = FAMILY_NAMES

    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.mode = config.relations
        self.experts = nn.ModuleDict({name: _FamilyExpert(config, vocab_size)
                                     for name in FAMILY_NAMES if name != 'rhythm'}) if self.mode != 'none' else nn.ModuleDict()
        if self.mode != 'none':
            self.rhythm_query = nn.Linear(config.field_width, config.relation_dim, bias=False)
            self.rhythm_key = nn.Linear(config.width, config.relation_dim, bias=False)
            self.rhythm_value = nn.Linear(config.width, config.field_width, bias=False)
            self.rhythm_gate = nn.Parameter(torch.tensor(-4.))

    def condition_rhythm(self, hidden: Tensor, fields: tuple[str, ...], families: dict[str, Tensor]) -> Tensor:
        if self.mode == 'none':
            return hidden
        indices = [i for i, field in enumerate(fields) if field.startswith(('onset_', 'duration_'))]
        if not indices:
            return hidden
        memory = families['rhythm']
        if memory.ndim != 2 or not len(memory):
            raise ValueError('pll.empty_rhythm_memory')
        index = torch.tensor(indices, dtype=torch.long, device=hidden.device)
        query = self.rhythm_query(hidden.index_select(0, index))
        attention = (query @ self.rhythm_key(memory).T / sqrt(query.shape[-1])).softmax(-1)
        correction = self.rhythm_gate.sigmoid() * (attention @ self.rhythm_value(memory))
        return hidden.index_add(0, index, correction)

    def forward(self, logits: Tensor, hidden: Tensor, fields: tuple[str, ...],
                allowed: Tensor, families: dict[str, Tensor]) -> Tensor:
        if logits.ndim != 2 or allowed.shape != logits.shape or allowed.dtype != torch.bool:
            raise ValueError('pll.invalid_candidate_mask')
        if len(fields) != len(logits) or not allowed.any(-1).all():
            raise ValueError('pll.empty_candidate_support')
        refined = logits
        for family, expert in self.experts.items():
            indices = [i for i, field in enumerate(fields) if family in ENERGY_ROUTES.get(field, ())]
            if indices:
                index = torch.tensor(indices, device=logits.device, dtype=torch.long)
                adjustment = expert(hidden.index_select(0, index), allowed.index_select(0, index), families[family])
                refined = refined.index_add(0, index, adjustment)
        return refined.masked_fill(~allowed, -torch.inf).log_softmax(-1)
