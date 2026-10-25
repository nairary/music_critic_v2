"""Teacher-forced event decoder with a causal, nested attribute decoder."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from .codec import BOS, EOS, PAD, FIELD_NAMES, VOCAB_SIZE, EventCodec
from .contracts import Event, ModelConfig
from .relations import RelationRefiner


@dataclass(frozen=True)
class BlockLikelihood:
    log_probability: Tensor
    token_log_probs: Tensor
    event_count: int
    tokens: tuple[int, ...]
    fields: tuple[str, ...]

    @property
    def nll(self):
        return -self.log_probability

    @property
    def token_count(self):
        return len(self.tokens)


class ConditionalEventDecoder(nn.Module):
    def __init__(self, config: ModelConfig | None = None, codec: EventCodec | None = None):
        super().__init__()
        self.config, self.codec = config or ModelConfig(), codec or EventCodec()
        c = self.config
        c.validate_codec(self.codec.profile)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, c.width, padding_idx=PAD)
        self.event_position = nn.Embedding(self.codec.profile.max_events + 1, c.width)
        self.attribute_position = nn.Embedding(c.max_fields, c.width)
        # Compose values with positions before pooling. Adding two embeddings
        # and then summing would erase which time bytes belonged to which field.
        self.event_composition = nn.Sequential(nn.Linear(2 * c.width, c.width),
                                               nn.GELU(), nn.Linear(c.width, c.width))
        self.main = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            c.width, c.heads, c.ff_dim, dropout=c.dropout, batch_first=True, norm_first=True),
            c.event_layers, norm=nn.LayerNorm(c.width))
        self.main_to_field = nn.Linear(c.width, c.field_width)
        self.field_tokens = nn.Embedding(VOCAB_SIZE, c.field_width, padding_idx=PAD)
        self.field_positions = nn.Embedding(c.max_fields, c.field_width)
        self.field_types = nn.Embedding(len(FIELD_NAMES), c.field_width)
        self.field_attention = nn.MultiheadAttention(c.field_width, c.field_heads,
                                                      dropout=c.dropout, batch_first=True)
        self.field_norm = nn.LayerNorm(c.field_width)
        self.field_ff = nn.Sequential(nn.Linear(c.field_width, c.field_ff_dim), nn.GELU(),
                                      nn.Dropout(c.dropout), nn.Linear(c.field_ff_dim, c.field_width))
        self.output_norm = nn.LayerNorm(c.field_width)
        self.dropout = nn.Dropout(c.dropout)
        self.output = nn.Linear(c.field_width, VOCAB_SIZE)
        self.relations = RelationRefiner(c, VOCAB_SIZE)

    def _main_states(self, main_inputs, memory, causal):
        """Event-level memory seam; the ordinary model keeps its exact path."""
        return self.main(main_inputs[None], memory.values[None], tgt_mask=causal)[0]

    def _distributions(self, tokens, memory, *, span, tracks, include_next=False):
        tokens = tuple(tokens)
        steps = list(self.codec.trace(tokens, span=span, tracks=tracks))
        if include_next:
            state = self.codec.parse(tokens, span=span, tracks=tracks)
            if state.complete or state.step is None:
                raise ValueError('pll.no_next_token_after_eos')
            steps.append(state.step)
        if not steps:
            raise ValueError('pll.empty_prediction_request')
        if memory.values.ndim != 2 or not len(memory.values) or memory.values.shape[1] != self.config.width:
            raise ValueError('pll.invalid_context_memory')
        device = self.token_embedding.weight.device
        if memory.values.device != device:
            raise ValueError('pll.memory_device_mismatch')
        events = max(s.event_index for s in steps) + 1
        lengths = [0] * events
        groups: list[list[int]] = [[] for _ in range(events)]
        for i, step in enumerate(steps):
            lengths[step.event_index] = max(lengths[step.event_index], step.field_index + 1)
            if i < len(tokens):
                groups[step.event_index].append(tokens[i])
        length = max(lengths)
        if length > self.config.max_fields:
            raise ValueError('pll.unsupported_internal_event_length')

        # Position i can only see the embeddings of complete events < i.
        main_inputs = [self.token_embedding.weight[BOS]]
        for group in groups[:-1]:
            ids = torch.tensor(group, dtype=torch.long, device=device)
            positions = torch.arange(len(group), device=device)
            bound = self.event_composition(torch.cat((self.token_embedding(ids),
                                                     self.attribute_position(positions)), dim=-1))
            main_inputs.append(bound.sum(0) / sqrt(len(group)))
        positions = torch.arange(events, device=device)
        main_inputs = torch.stack(main_inputs) + self.event_position(positions)
        causal = torch.ones(events, events, dtype=torch.bool, device=device).triu(1)
        main_states = self._main_states(main_inputs, memory, causal)

        # Every row is one event. Padding and future targets are never visible
        # keys. Field type/grammar state comes from the consumed prefix only.
        key_ids = torch.full((events, length), PAD, dtype=torch.long, device=device)
        field_ids = torch.zeros((events, length), dtype=torch.long, device=device)
        for i, group in enumerate(groups):
            past = [BOS] + group[:max(0, lengths[i] - 1)]
            key_ids[i, :len(past)] = torch.tensor(past, device=device)
        for step in steps:
            field_ids[step.event_index, step.field_index] = FIELD_NAMES.index(step.field)
        query = (self.main_to_field(main_states)[:, None, :]
                 + self.field_positions(torch.arange(length, device=device))[None]
                 + self.field_types(field_ids))
        keys = self.field_tokens(key_ids) + self.field_positions(torch.arange(length, device=device))[None]
        key_padding = torch.arange(length, device=device)[None] >= torch.tensor(lengths, device=device)[:, None]
        field_causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
        attended, _ = self.field_attention(query, keys, keys, attn_mask=field_causal,
                                           key_padding_mask=key_padding, need_weights=False)
        hidden = self.field_norm(query + self.dropout(attended))
        hidden = self.output_norm(hidden + self.dropout(self.field_ff(hidden)))
        event_indices = torch.tensor([s.event_index for s in steps], device=device)
        field_indices = torch.tensor([s.field_index for s in steps], device=device)
        hidden = hidden[event_indices, field_indices]
        fields = tuple(s.field for s in steps)
        hidden = self.relations.condition_rhythm(hidden, fields, memory.families)
        allowed = torch.zeros(len(steps), VOCAB_SIZE, dtype=torch.bool, device=device)
        for i, step in enumerate(steps):
            allowed[i, list(step.allowed)] = True
        return self.relations(self.output(hidden), hidden, fields, allowed, memory.families), fields

    def token_distributions(self, tokens, memory, *, span, tracks):
        """One distribution per supplied token, conditional only on its prefix."""
        return self._distributions(tokens, memory, span=span, tracks=tracks)[0]

    def next_token_log_probs(self, prefix, memory, *, span, tracks):
        return self._distributions(prefix, memory, span=span, tracks=tracks, include_next=True)[0][-1]

    def forward(self, events: tuple[Event, ...], memory, *, span, tracks) -> BlockLikelihood:
        tokens = self.codec.encode(events, span=span, tracks=tracks)
        distributions, fields = self._distributions(tokens, memory, span=span, tracks=tracks)
        target = torch.tensor(tokens, dtype=torch.long, device=distributions.device)
        chosen = distributions.gather(1, target[:, None]).squeeze(1)
        return BlockLikelihood(chosen.sum(), chosen, len(events), tokens, fields)

    @torch.no_grad()
    def generate(self, memory, *, span, tracks, generator=None, max_tokens=None):
        """Exact grammar sampling reference; generation speed is not admitted."""
        if self.training:
            raise ValueError('pll.generation_requires_eval_mode')
        limit = max_tokens or (self.codec.profile.max_events * self.config.max_fields + 1)
        prefix = ()
        for _ in range(limit):
            log_probs = self.next_token_log_probs(prefix, memory, span=span, tracks=tracks)
            token = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
            prefix += (token,)
            if token == EOS:
                return self.codec.decode(prefix, span=span, tracks=tracks)
        raise ValueError('pll.generation_token_budget_exhausted')
