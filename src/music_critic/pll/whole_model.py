"""Whole observed source -> hierarchy and repeats -> normalized masked events."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict

import torch
from torch import nn

from .codec import EventCodec
from .contracts import CodecProfile, ContextProfile, ModelConfig
from .encoder import PLLContextEncoder
from .memory import FAMILY_LEVELS
from .model import HCM1
from .motifs import MotifMemory
from .observed_view import UnsupportedContextError
from .whole_contracts import (WHOLE_SCHEMA, WholeObservedMusicView, WholePieceConfig,
                               split_whole_piece, whole_context_profile)
from .whole_decoder import MultiBranchMemory, WholeEventDecoder
from .whole_memory import WholeSongMemory


class WholePieceEncoder(PLLContextEncoder):
    """The fully frozen option includes the otherwise trainable boundary map."""

    def __init__(self, profile=None, *, backbone=None, freeze_policy='frozen', freeze_local=True):
        self.lock_local = freeze_local
        super().__init__(profile, backbone=backbone, freeze_policy=freeze_policy)

    def set_freeze_policy(self, policy):
        if self.lock_local and policy != 'frozen':
            raise ValueError('pll.whole_frozen_local_cannot_unfreeze')
        super().set_freeze_policy(policy)
        self.boundary_projection.requires_grad_(not self.lock_local)
        if self.lock_local:
            for parameter in self.parameters():
                parameter.grad = None

    def train(self, mode=True):
        return super().train(False if self.lock_local else mode)

    @classmethod
    def from_pretrained(cls, checkpoint_path, *, freeze_local=True, **kwargs):
        base = PLLContextEncoder.from_pretrained(checkpoint_path, **kwargs)
        result = cls(base.profile, backbone=base.backbone, freeze_policy=base.freeze_policy,
                     freeze_local=freeze_local)
        result.weights_origin = base.weights_origin
        return result


def _motif_receipt(result, detailed=False):
    if detailed:
        return result.diagnostics()
    weights = result.weights.detach().cpu().tolist()
    mass = {key: 0. for key in result.scale_counts}
    for candidate, weight in zip(result.candidates, weights):
        mass[f'{candidate.unit}:{candidate.scale}'] += weight
    selected = []
    for i in result.selected_indices:
        candidate = result.candidates[i]
        selected.append(dict(candidate_id=candidate.candidate_id, unit=candidate.unit,
            scale=candidate.scale, start=[candidate.start.num,candidate.start.den],
            end=[candidate.end.num,candidate.end.den], weight=weights[i]))
    return dict(scale_counts=dict(result.scale_counts),
                selected_scale_counts=dict(result.selected_scale_counts),
                weight_by_scale=mass, null_weight=weights[-1], selected=selected)


class WholePieceHCM(HCM1):
    """An explicit new model contract; old HCM checkpoints are not resumes."""

    def __init__(self, config: ModelConfig = ModelConfig(), *,
                 whole_config: WholePieceConfig = WholePieceConfig(),
                 context_profile: ContextProfile = ContextProfile(),
                 codec_profile: CodecProfile = CodecProfile(), freeze_policy='frozen'):
        nn.Module.__init__(self)
        profile = whole_context_profile(context_profile, whole_config)
        if profile.max_tracks > codec_profile.max_tracks:
            raise ValueError('pll.context_tracks_exceed_codec')
        self.config, self.context_profile, self.codec_profile = config, profile, codec_profile
        self.whole_config = whole_config
        self.codec = EventCodec(codec_profile)
        self.encoder = WholePieceEncoder(profile, freeze_policy=freeze_policy,
                                         freeze_local=whole_config.freeze_local)
        self.memory = WholeSongMemory(config, profile, whole_config)
        self.motifs = MotifMemory(config, whole_config) if whole_config.enable_motifs else None
        self.decoder = WholeEventDecoder(config, self.codec)
        self.last_diagnostics = None
        self.warm_start_binding = None

    def contract(self):
        return dict(super().contract(), schema=WHOLE_SCHEMA,
            whole_piece={**asdict(self.whole_config), 'motif_scales':list(self.whole_config.motif_scales),
                         'motif_beat_scales':list(self.whole_config.motif_beat_scales)},
            observation='complete-declared-source-mask-before-all-memories-v1',
            memory='positional-bars-blocks-song-and-observed-motif-bank-v1',
            warm_start=self.warm_start_binding)

    def split(self, piece, query):
        return split_whole_piece(piece, query, whole_config=self.whole_config,
                                 context_profile=self.context_profile)

    def encode_tiles(self, view):
        """Run only the local-encoder stage of :meth:`encode`.

        With ``freeze_local`` this stage receives no gradients, so its output is
        identical for every checkpoint of one frozen-encoder run and may be
        computed once and reused. Callers must verify that invariant against the
        retained weights instead of assuming it.
        """
        if not isinstance(view, WholeObservedMusicView):
            raise TypeError('pll.whole_observed_view_required')
        if len(view.piece.bars) > self.whole_config.max_piece_bars:
            raise UnsupportedContextError('pll.whole_piece_bar_limit')
        if len(view.piece.notes) > self.whole_config.max_piece_notes:
            raise UnsupportedContextError('pll.whole_piece_note_limit')
        with torch.no_grad() if self.whole_config.freeze_local else nullcontext():
            return self.encoder(view)

    def encode_from(self, encoded, view, *, detailed_diagnostics=False):
        """Build memories and retrieval from already-encoded tiles."""
        if not isinstance(view, WholeObservedMusicView):
            raise TypeError('pll.whole_observed_view_required')
        context = self.memory(encoded, view)
        branches = {'local':context.local.values}
        contexts = [context.local]
        if self.whole_config.enable_global:
            branches['global'] = context.global_memory.values
            contexts.append(context.global_memory)
        values = [m.values for m in contexts]
        metadata = [token for m in contexts for token in m.metadata]
        families = {name:[m.families[name] for m in contexts] for name in FAMILY_LEVELS}
        motif_diagnostics = None
        if self.motifs is not None:
            global_values = (context.global_memory.values if self.whole_config.enable_global
                             else context.local.values.new_zeros((1,self.config.width)))
            motifs = self.motifs(encoded, view, bar_values=context.bar_values,
                                  global_values=global_values)
            branches['motif'] = motifs.values
            values.append(motifs.values)
            metadata.extend(motifs.metadata)
            for name in families:
                families[name].append(motifs.values)
            motif_diagnostics = _motif_receipt(motifs, detailed_diagnostics)
        total = sum(len(v) for v in values)
        if total > self.context_profile.max_memory:
            raise UnsupportedContextError('pll.whole_combined_memory_limit')
        diagnostics = dict(source_bars=view.source_bar_count,
            observed_notes=len(view.piece.notes), local_tokens=len(branches['local']),
            global_tokens=len(branches.get('global', ())),
            motif_tokens=len(branches.get('motif', ())), motifs=motif_diagnostics)
        self.last_diagnostics = diagnostics
        return MultiBranchMemory(torch.cat(values),
            {name:torch.cat(rows) for name,rows in families.items()}, tuple(metadata), view.query,
            branches=branches, diagnostics=diagnostics)

    def encode(self, view, *, detailed_diagnostics=False):
        return self.encode_from(self.encode_tiles(view), view,
                                detailed_diagnostics=detailed_diagnostics)
