"""One public model: observed tiles -> hierarchical context -> joint events."""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .codec import EventCodec
from .contracts import SCHEMA, CodecProfile, ContextProfile, ModelConfig
from .decoder import ConditionalEventDecoder
from .encoder import PLLContextEncoder
from .memory import HierarchicalMemory
from .observed_view import MaskedExample, ObservedMusicView


class HCM1(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig(), *,
                 context_profile: ContextProfile = ContextProfile(),
                 codec_profile: CodecProfile = CodecProfile(), freeze_policy='frozen'):
        super().__init__()
        if context_profile.max_tracks > codec_profile.max_tracks:
            raise ValueError('pll.context_tracks_exceed_codec')
        self.config, self.context_profile, self.codec_profile = config, context_profile, codec_profile
        self.codec = EventCodec(codec_profile)
        self.encoder = PLLContextEncoder(context_profile, freeze_policy=freeze_policy)
        self.memory = HierarchicalMemory(config, context_profile)
        self.decoder = ConditionalEventDecoder(config, self.codec)

    def contract(self):
        return dict(schema=SCHEMA, model=asdict(self.config), context=asdict(self.context_profile),
                    codec=asdict(self.codec_profile), freeze_policy=self.encoder.freeze_policy,
                    initialization=self.encoder.weights_origin,
                    event_projection='onset-track-kind-midi-pitch-duration-v1',
                    mask_ownership='onset-half-open-remove-whole-event-v1',
                    likelihood='canonical-bounded-event-multiset-with-eos-v1')

    def encode(self, view: ObservedMusicView):
        if not isinstance(view, ObservedMusicView):
            raise TypeError('pll.observed_view_required')
        return self.memory(self.encoder(view), view)

    def forward(self, example: MaskedExample):
        if not isinstance(example, MaskedExample):
            raise TypeError('pll.masked_example_required')
        if example.tracks != example.context.allowed_tracks:
            raise ValueError('pll.target_query_tracks_mismatch')
        # Admission and exact grammar checks precede expensive encoder work.
        self.codec.encode(example.targets, span=example.span, tracks=example.tracks)
        memory = self.encode(example.context)
        return self.decoder(example.targets, memory, span=example.span, tracks=example.tracks)

    def batch_likelihood(self, examples):
        """Serial variable-size reference, with no cross-sample graph mixing."""
        if not isinstance(examples, tuple) or not examples:
            raise ValueError('pll.nonempty_example_tuple_required')
        return tuple(self(example) for example in examples)

    @torch.no_grad()
    def generate(self, view: ObservedMusicView, *, generator=None, max_tokens=None):
        if self.training:
            raise ValueError('pll.generation_requires_eval_mode')
        memory = self.encode(view)
        return self.decoder.generate(memory, span=view.query.span, tracks=view.allowed_tracks,
                                     generator=generator, max_tokens=max_tokens)
