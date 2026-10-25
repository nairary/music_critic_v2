"""Frozen backbone feature export with exact raw geometry and stage identity."""
from __future__ import annotations

from fractions import Fraction

from music_critic.data import CanonicalPiece, validate_piece
from music_critic.data.raw_time_readout_slots import build_readout_slots

from .contracts import RawInterval, RawNote, SampleGeometry


def geometry_from_piece(piece: CanonicalPiece) -> SampleGeometry:
    """Read exact raw geometry only; never infer voices or consult target labels."""
    if not isinstance(piece, CanonicalPiece):
        raise TypeError("critic geometry requires a CanonicalPiece")
    report = validate_piece(piece)
    if not report.is_valid:
        raise ValueError("invalid canonical piece for critic geometry")
    notes = tuple(RawNote(
        n.note_id, n.track_id, n.onset_qn.to_fraction(),
        (n.onset_qn + n.duration_qn).to_fraction(), n.pitch, n.is_percussion,
    ) for n in piece.notes)
    slots = build_readout_slots(piece).time_slots
    onsets = sorted({n.start for n in notes})
    intervals = {
        "onset": tuple(RawInterval(f"onset:{t.numerator}_{t.denominator}", t, t)
                       for t in onsets),
        "beat": tuple(RawInterval(b.beat_id, b.start_qn.to_fraction(),
                         (b.start_qn + b.duration_qn).to_fraction()) for b in piece.beats),
        "bar": tuple(RawInterval(b.bar_id, b.start_qn.to_fraction(),
                        (b.start_qn + b.duration_qn).to_fraction(), not b.is_incomplete) for b in piece.bars),
        "slice": tuple(RawInterval(key, start, slots[i + 1][0] if i + 1 < len(slots) else start)
                          for i, (start, key) in enumerate(slots)),
    }
    return SampleGeometry(piece.piece_id, notes, tuple(t.track_id for t in piece.tracks),
                          intervals, piece.duration_qn.to_fraction())


def geometry_payload(geometry: SampleGeometry) -> dict:
    """Plain exact metadata shared by raw hashes and safe cache serialization."""
    def time(value):
        return [value.numerator, value.denominator]
    return {
        'sample_id': geometry.sample_id,
        'notes': [[n.note_id, n.track_id, time(n.start), time(n.end), n.pitch, n.is_drum]
                  for n in geometry.notes],
        'tracks': list(geometry.tracks),
        'intervals': {level: [[i.entity_id, time(i.start), time(i.end), i.complete] for i in rows]
                      for level, rows in geometry.intervals.items()},
        'duration': time(geometry.duration),
    }


# Isolated C3 implementations require a separately versioned audited adapter.
C2_ADAPTER_VERSION = 'critic-c2-features-v1'
TEMPORAL_ADAPTER_VERSION = 'critic-temporal-features-v1'
FEATURE_LEVELS = ('note_local', 'note', 'onset', 'beat', 'bar', 'track', 'song', 'slice')


def expected_stages(adapter_version: str) -> dict[str, str]:
    stages = dict.fromkeys(FEATURE_LEVELS, 'hierarchical_context')
    stages['note_local'] = 'local_gnn'
    stages.update(dict.fromkeys(('onset', 'beat', 'bar'), 'onset_bigru_context'))
    if adapter_version == C2_ADAPTER_VERSION:
        stages['slice'] = 'derived_sounding_note_mean'
    elif adapter_version == TEMPORAL_ADAPTER_VERSION:
        stages.update(dict.fromkeys(('onset', 'bar'), 'temporal_round_1'))
        stages.update(dict.fromkeys(('note', 'beat', 'slice'), 'temporal_round_2'))
    elif adapter_version == 'critic-c3-threepass-relationgate-cloud-20260918-v1':
        stages.update(dict.fromkeys(('onset', 'bar'), 'c3_relationgate_round_2'))
        stages.update(dict.fromkeys(('note', 'beat', 'slice'), 'c3_relationgate_round_3'))
    else:
        raise ValueError('unsupported feature adapter version')
    return stages


def raw_piece_hash(piece: CanonicalPiece, geometry: SampleGeometry | None = None) -> str:
    """Bind all backbone raw inputs and exact geometry, excluding theory targets."""
    from hashlib import sha256
    import json
    from music_critic.graph import build_raw_graph, graph_fingerprint
    payload = {'raw_graph': graph_fingerprint(build_raw_graph(piece)),
               'geometry': geometry_payload(geometry or geometry_from_piece(piece))}
    return sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class BackboneFeatureAdapter:
    """Freeze a supported main encoder and export one same-pass state pyramid.

    Source/checkpoint IDs are caller-provided provenance; model_state_id binds
    the actual initialized weights independently of those identifiers.
    """
    def __init__(self, model, binding):
        from dataclasses import replace
        import torch
        from music_critic.experiments.analysisgnn.corrected_model import (
            CorrectedAnalysisGNNModel, model_state_fingerprint)
        from music_critic.experiments.analysisgnn.temporal_slices.model import TemporalSliceModel
        from .c3_backbone import (C3ThreePassRelationGateModel, C3_ADAPTER_VERSION, C3_SOURCE_ID,
                                 validate_c3_source_model)
        versions = {C3ThreePassRelationGateModel: C3_ADAPTER_VERSION,
                    CorrectedAnalysisGNNModel: C2_ADAPTER_VERSION,
                    TemporalSliceModel: TEMPORAL_ADAPTER_VERSION}
        if type(model) not in versions or binding.adapter_version != versions[type(model)]:
            raise ValueError('unsupported backbone or adapter version')
        if type(model) is C3ThreePassRelationGateModel:
            if binding.source_id != C3_SOURCE_ID:
                raise ValueError('selected C3 source manifest binding required')
            validate_c3_source_model(model)
        if binding.dtype != 'float32' or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('frozen feature adapter requires float32 weights and dtype')
        state_id = model_state_fingerprint(model)
        if binding.model_state_id and binding.model_state_id != state_id:
            raise ValueError('model state binding mismatch')
        self.binding = replace(binding, model_state_id=state_id)
        self.model = model.requires_grad_(False).eval()
        self._guard = self._weight_guard()

    def _weight_guard(self):
        return tuple((name, tensor._version, tensor.data_ptr(), tuple(tensor.shape),
                      tensor.dtype, tensor.device)
                     for name, tensor in self.model.state_dict(keep_vars=True).items())

    def _check_frozen(self):
        if self.binding.adapter_version == 'critic-c3-threepass-relationgate-cloud-20260918-v1':
            from .c3_backbone import validate_c3_source_model
            validate_c3_source_model(self.model)
        if self._guard != self._weight_guard():
            raise ValueError('frozen backbone weights changed; construct a new adapter')
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('backbone is no longer frozen')

    def to(self, device):
        self._check_frozen()
        self.model.to(device)
        self._guard = self._weight_guard()
        return self

    def extract(self, raw_batch: tuple[CanonicalPiece, ...]):
        import torch
        from .contracts import CriticFeaturePyramid, FeatureSample
        from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
        if not isinstance(raw_batch, tuple) or not raw_batch or not all(
                isinstance(p, CanonicalPiece) for p in raw_batch):
            raise TypeError('extract requires a nonempty tuple of canonical raw pieces')
        self._check_frozen()
        self.model.eval()
        geometries = tuple(geometry_from_piece(p) for p in raw_batch)
        hashes = tuple(raw_piece_hash(p, g) for p, g in zip(raw_batch, geometries))
        graph = build_temporal_batch(raw_batch).to(next(self.model.parameters()).device)
        temporal = self.binding.adapter_version != C2_ADAPTER_VERSION
        # Cached constants must remain usable by autograd in newly trained heads,
        # even when the caller precomputes features inside inference_mode.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(
                graph.slice_features.device.type, enabled=False):
            if temporal:
                output, base = self.model.forward_with_base(graph)
                latest = dict(base.fused.embeddings)
                latest.update(output.embeddings)
            else:
                base = self.model.encode(graph.base)
                latest = dict(base.fused.embeddings)
            local = base.local_encoder.final_output.embeddings['note']
            samples = []
            for index, geometry in enumerate(geometries):
                embeddings = {}
                for level in FEATURE_LEVELS:
                    if level == 'slice' and not temporal:
                        notes = embeddings['note']
                        rows = []
                        for interval in geometry.intervals['slice']:
                            sounding = [j for j, n in enumerate(geometry.notes)
                                        if n.start <= interval.start < n.end]
                            rows.append(notes[sounding].mean(0) if sounding else notes.new_zeros(128))
                        embeddings[level] = torch.stack(rows)
                        continue
                    node = 'note' if level == 'note_local' else level
                    pointers = graph.pointers[node] if node == 'slice' else graph.base[node].ptr
                    start, end = int(pointers[index]), int(pointers[index + 1])
                    values = local if level == 'note_local' else latest[level]
                    embeddings[level] = values[start:end].detach().clone()
                samples.append(FeatureSample(geometry, embeddings, hashes[index],
                                             expected_stages(self.binding.adapter_version)))
        return CriticFeaturePyramid(tuple(samples), self.binding)
