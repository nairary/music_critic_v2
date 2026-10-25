"""Controls using the same raw input, score ordering and missing-aspect semantics."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ASPECTS, CriticOutput, SamplePrediction, local_candidates
from .model import AspectReadout, SetPool


def _prediction(sample, scores):
    geo = sample.geometry
    available = torch.tensor([
        any(not n.is_drum and n.end > n.start for n in geo.notes),
        bool(geo.notes) and bool(geo.intervals.get('beat')),
        any(not n.is_drum for n in geo.notes),
        bool(geo.notes) and bool(local_candidates(geo)['structure/boundary']),
    ], device=scores.device)
    return SamplePrediction(geo.sample_id, torch.where(available, scores, torch.zeros_like(scores)),
                            available, {}, geo.duration)


class BeatProbe(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.head = nn.Sequential(nn.Linear(feature_dim, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, pyramid):
        outputs = []
        for sample in pyramid.samples:
            values = sample.embeddings.get('beat')
            if values is None or not len(values):
                scores = self.head[0].weight.new_zeros(4)
                output = _prediction(sample, scores)
                outputs.append(SamplePrediction(output.sample_id, scores,
                    torch.zeros_like(output.availability), {}, output.horizon))
                continue
            values = values.detach().to(self.head[0].weight)
            intervals = sample.geometry.intervals['beat']
            if values.shape != (len(intervals), self.feature_dim):
                raise ValueError('beat feature alignment mismatch')
            weights = values.new_tensor([float(max(0, min(b.end, sample.geometry.duration)-max(b.start,0)))
                                        for b in intervals])
            if not weights.sum() > 0:
                scores = values.new_zeros(4)
            else:
                scores = (self.head(values) * (weights / weights.sum())[:,None]).sum(0)
            outputs.append(_prediction(sample,scores))
        return CriticOutput(tuple(outputs))


class MultilevelPoolingCritic(nn.Module):
    def __init__(self, feature_dim=128, width=128):
        super().__init__()
        self.projections = nn.ModuleDict({k:nn.Linear(feature_dim,width)
            for k in ('note','slice','beat','bar','track','song')})
        self.readouts = nn.ModuleDict({a:AspectReadout(width) for a in ASPECTS})

    def forward(self,pyramid):
        zero = self.projections['note'].weight.new_zeros(self.projections['note'].out_features)
        outputs=[]
        for sample in pyramid.samples:
            levels={k:F.gelu(self.projections[k](v.detach().to(zero)))
                    for k,v in sample.embeddings.items() if k in self.projections}
            outputs.append(_prediction(sample,torch.stack([self.readouts[a](levels,zero) for a in ASPECTS])))
        return CriticOutput(tuple(outputs))


class RawSummaryCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(8,32),nn.GELU(),nn.Linear(32,4))

    def forward(self,pyramid):
        outputs=[]
        for sample in pyramid.samples:
            notes=sample.geometry.notes
            durations=[float(n.end-n.start) for n in notes]
            pitched=[n.pitch/127 for n in notes if not n.is_drum]
            duration=max(float(sample.geometry.duration),1e-12)
            mean=sum(pitched)/len(pitched) if pitched else 0.
            spread=(sum((x-mean)**2 for x in pitched)/len(pitched))**.5 if pitched else 0.
            raw=self.head[0].weight.new_tensor([
                len(notes)/duration, len({n.start for n in notes})/duration,
                len(pitched)/max(1,len(notes)), mean, spread,
                max(pitched)-min(pitched) if pitched else 0.,
                sum(durations)/max(1,len(notes)), len({n.track_id for n in notes}),
            ])
            outputs.append(_prediction(sample,self.head(raw/(1+raw.abs()))))
        return CriticOutput(tuple(outputs))
