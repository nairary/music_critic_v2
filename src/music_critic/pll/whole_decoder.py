"""Independent local/global/repetition attention with one exact event codec."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .decoder import ConditionalEventDecoder
from .memory import ContextMemory


@dataclass(frozen=True)
class MultiBranchMemory(ContextMemory):
    branches: Mapping[str, Tensor]
    diagnostics: dict

    def __post_init__(self):
        super().__post_init__()
        if 'local' not in self.branches or set(self.branches) - {'local', 'global', 'motif'}:
            raise ValueError('pll.whole_memory_branches')
        for branch in self.branches.values():
            if (branch.ndim != 2 or not len(branch) or
                    branch.shape[1] != self.values.shape[1] or
                    branch.device != self.values.device or branch.dtype != self.values.dtype or
                    not torch.isfinite(branch).all()):
                raise ValueError('pll.whole_memory_branch_shape_or_values')


class WholeEventDecoder(ConditionalEventDecoder):
    """Retain the HCM event/field weights and add gated distant-memory paths."""

    def __init__(self, config=None, codec=None):
        super().__init__(config, codec)
        c = self.config
        self.global_attention = nn.ModuleList(
            nn.MultiheadAttention(c.width, c.heads, dropout=c.dropout, batch_first=True)
            for _ in range(c.event_layers))
        self.motif_attention = nn.ModuleList(
            nn.MultiheadAttention(c.width, c.heads, dropout=c.dropout, batch_first=True)
            for _ in range(c.event_layers))
        self.branch_norms = nn.ModuleList(nn.LayerNorm(c.width) for _ in range(c.event_layers))
        self.branch_gates = nn.ModuleList(nn.Linear(c.width, 2) for _ in range(c.event_layers))
        for gate in self.branch_gates:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.)

    def _main_states(self, main_inputs, memory, causal):
        if not isinstance(memory, MultiBranchMemory):
            raise TypeError('pll.whole_decoder_requires_multibranch_memory')
        state = main_inputs[None]
        local = memory.branches['local'][None]
        for i, layer in enumerate(self.main.layers):
            state = layer(state, local, tgt_mask=causal)
            query = self.branch_norms[i](state)
            gates = self.branch_gates[i](query).sigmoid()
            deltas = []
            for j, (name, modules) in enumerate((('global', self.global_attention),
                                                 ('motif', self.motif_attention))):
                if name in memory.branches:
                    values = memory.branches[name][None]
                    attended, _ = modules[i](query, values, values, need_weights=False)
                    deltas.append(gates[..., j:j+1] * self.dropout(attended))
            if deltas:
                state = state + sum(deltas)
        if self.main.norm is not None:
            state = self.main.norm(state)
        return state[0]
