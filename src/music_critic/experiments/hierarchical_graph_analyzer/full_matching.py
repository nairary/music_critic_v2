"""Whole-piece assignment with bounded float64 tensor arithmetic.

Small problems retain the historical exact-Fraction solver. Large problems use
CPU float64 costs, the same cardinality-first objective and stable temporal
ordering. Exact source geometry is never changed; this is detached matching.
No additional dependency or GPU matching memory is required.
"""
from .matching import (match_events, intervals, event_mask, prediction_tie_keys,
                       compatible, interval_iou, _number, _metadata)


def _assignment_tensor(cost):
    import torch
    n,m=cost.shape
    u=cost.new_zeros(n+1);v=cost.new_zeros(m+1)
    owner=torch.zeros(m+1,dtype=torch.long);way=torch.zeros(m+1,dtype=torch.long)
    for row in range(1,n+1):
        owner[0]=row;column=0
        best=cost.new_full((m+1,),float('inf'));seen=torch.zeros(m+1,dtype=torch.bool)
        while True:
            seen[column]=True;current=int(owner[column])
            reduced=cost[current-1]-u[current]-v[1:]
            improve=(~seen[1:])&(reduced<best[1:])
            best[1:]=torch.where(improve,reduced,best[1:])
            way[1:]=torch.where(improve,column,way[1:])
            candidates=best[1:].masked_fill(seen[1:],float('inf'))
            delta,next_column=candidates.min(0);next_column=int(next_column)+1
            if not bool(torch.isfinite(delta)):raise ValueError('whole matching has no finite assignment')
            selected=torch.where(seen)[0]
            u.index_add_(0,owner[selected],delta.expand(len(selected)))
            v[selected]-=delta;best[~seen]-=delta
            column=next_column
            if not int(owner[column]):break
        while column:
            previous=int(way[column]);owner[column]=owner[previous];column=previous
    return [(int(owner[j])-1,j-1) for j in range(1,m+1) if int(owner[j])]


def match_full_events(pred,targets,*,mode='training',iou_threshold=.5,exact_pair_limit=16384):
    import torch
    if mode not in ('training','metric'):raise ValueError('mode must be training or metric')
    threshold=_number(iou_threshold)
    if not 0<threshold<=1:raise ValueError('iou_threshold must be in (0,1]')
    if type(exact_pair_limit) is not int or exact_pair_limit<0:raise ValueError('nonnegative exact pair limit required')
    if len(pred['spans'])*len(targets['spans'])<=exact_pair_limit:
        return match_events(pred,targets,mode=mode,iou_threshold=iou_threshold)
    ps,ts=intervals(pred['spans']),intervals(targets['spans']);mask=event_mask(targets)
    ids=targets.get('ids',[str(i) for i in range(len(ts))])
    if len(ids)!=len(ts) or len(set(ids))!=len(ids):raise ValueError('target ids must be unique and aligned')
    ties=prediction_tie_keys(pred)
    po=sorted(range(len(ps)),key=lambda i:(ps[i],ties[i]))
    to=sorted((i for i in range(len(ts)) if mask[i]),key=lambda i:(ts[i],str(ids[i])))
    if not po or not to:return []
    p=torch.tensor([[float(x) for x in ps[i]] for i in po],dtype=torch.float64)
    t=torch.tensor([[float(x) for x in ts[i]] for i in to],dtype=torch.float64)
    intersection=(torch.minimum(t[:,None,1],p[None,:,1])-torch.maximum(t[:,None,0],p[None,:,0])).clamp_min(0)
    union=torch.maximum(t[:,None,1],p[None,:,1])-torch.minimum(t[:,None,0],p[None,:,0])
    iou=intersection/union
    allowed=torch.ones_like(iou,dtype=torch.bool)
    for name in ('view_id','scope_id'):
        pv=[_metadata(pred,name,i,targets.get(name)) for i in po]
        tv=[_metadata(targets,name,i) for i in to]
        codes={value:i for i,value in enumerate(dict.fromkeys(pv+tv))}
        allowed &= torch.tensor([codes[x] for x in tv])[:,None]==torch.tensor([codes[x] for x in pv])[None,:]
    if mode=='metric':
        allowed &= iou>=float(threshold)
        # Resolve threshold-edge cases with exact represented interval arithmetic.
        for i,j in torch.nonzero((iou-float(threshold)).abs()<1e-12).tolist():
            allowed[i,j]=compatible(pred,targets,po[j],to[i]) and interval_iou(ps[po[j]],ts[to[i]])>=threshold
    costs=((t[:,None,:]-p[None,:,:]).abs().sum(-1)/(t[:,1]-t[:,0])[:,None])+1-iou
    penalty=(float(costs.max())+1)*(len(to)+1)
    matrix=costs.new_full((len(to),len(po)+len(to)),2*penalty)
    matrix[:,:len(po)]=costs.masked_fill(~allowed,2*penalty)
    # Private dummy columns remove meaningless unmatched-row permutations.
    index=torch.arange(len(to));matrix[index,len(po)+index]=penalty
    return sorted((po[j],to[i]) for i,j in _assignment_tensor(matrix) if j<len(po) and bool(allowed[i,j]))
