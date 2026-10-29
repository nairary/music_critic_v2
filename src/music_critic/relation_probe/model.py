"""Matched contextual MLP/translation decoders and explicit encoder modes."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

RAW_WIDTH=10


def geometry_score(source,target,relation,bias,temperature):
    return bias-(source+relation-target).square().sum(-1)/temperature


def masked_pair_loss(logits,labels,mask,weights=None):
    if labels.shape!=logits.shape or mask.shape!=logits.shape or mask.dtype!=torch.bool:
        raise ValueError('aligned labels and boolean availability required')
    if not mask.any():return logits.sum()*0
    selected=F.binary_cross_entropy_with_logits(logits[mask],labels[mask],reduction='none')
    if weights is None:return selected.mean()
    weights=weights[mask]
    return (selected*weights).sum()/weights.sum().clamp_min(1e-8)


class RelationHead(nn.Module):
    def __init__(self,input_width,decoder='mlp',width=64):
        super().__init__()
        if decoder not in ('mlp','transe'):raise ValueError('unknown relation decoder')
        self.decoder=decoder
        self.projection=nn.Linear(input_width,width)
        self.existence=nn.Linear(width,1)
        if decoder=='mlp':
            self.pair=nn.Sequential(nn.Linear(width*4,128),nn.GELU(),nn.Linear(128,1))
        else:
            self.relation=nn.Parameter(torch.randn(width)*.05)
            self.bias=nn.Parameter(torch.tensor(1.))
            self.raw_temperature=nn.Parameter(torch.tensor(0.))

    def forward(self,features,source,target,exist_source):
        u=F.normalize(self.projection(features),dim=-1,eps=1e-8)
        a,b=u[source],u[target]
        scores=(self.pair(torch.cat((a,b,b-a,a*b),-1)).squeeze(-1) if self.decoder=='mlp'
                else geometry_score(a,b,self.relation,self.bias,F.softplus(self.raw_temperature)+1e-4))
        return scores,self.existence(u[exist_source]).squeeze(-1)


class Probe(nn.Module):
    def __init__(self,mode,decoder,seed=17,checkpoint=None):
        super().__init__()
        if mode not in ('raw','frozen','finetune','random'):raise ValueError('encoder mode')
        self.mode=mode
        self.encoder=None
        if mode!='raw':
            from music_critic.pll.encoder import PLLContextEncoder
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if mode in ('frozen','finetune'):
                    if checkpoint is None:raise ValueError('byte-bound pretrained checkpoint required')
                    self.encoder=PLLContextEncoder.from_pretrained(checkpoint['path'],
                        expected_sha256=checkpoint['sha256'],expected_size_bytes=checkpoint['size_bytes'],
                        freeze_policy='frozen' if mode=='frozen' else 'full')
                else:self.encoder=PLLContextEncoder(freeze_policy='full')
            if mode=='frozen':self.freeze_encoder()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000+seed)
            self.head=RelationHead(RAW_WIDTH if mode=='raw' else 256+RAW_WIDTH,decoder)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder.set_freeze_policy('frozen')
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.mode='frozen'

    def train(self,mode=True):
        super().train(mode)
        if self.encoder is not None and self.mode=='frozen':self.encoder.eval()
        return self

    def features(self,graph,events,boundary=None):
        if self.mode=='raw':return events['raw']
        if self.mode=='frozen':
            with torch.no_grad():values=self.encoder._encode_graph(graph,boundary)
        else:values=self.encoder._encode_graph(graph,boundary)
        return torch.cat((values['note'][events['note_index']],
                          values['slice'][events['slice_index']],events['raw']),-1)
