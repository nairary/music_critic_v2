"""Target-side matching and masked losses for whole-work H/h/P predictions.

No target-side object or availability is accepted by the model. This module is
called only after prediction. Addressed endpoint sets are hyperedges: a plural
source/target is never expanded into pairwise positive examples.
"""
from collections import defaultdict
from fractions import Fraction
import math
from numbers import Integral

from .losses import compute_loss
from .matching import (_assignment, _number, as_list, compatible, event_mask,
                       interval_iou)
from .full_matching import match_full_events as match_events

_EMPTY = {'ids': (), 'spans': (), 'fields': {}}
_LEVELS = frozenset(('n', 'h', 'H', 'P'))


def _tensor_leaves(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def _indices(value, name, count=None, allow_missing=False):
    rows = as_list(value)
    low = -1 if allow_missing else 0
    if count is not None and len(rows) != count:
        raise ValueError(f'{name} must align with rows')
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < low for i in rows):
        raise ValueError(f'{name} requires integer indices >= {low}')
    return [int(i) for i in rows]


def _context_spans(value):
    rows = []
    for row in as_list(value):
        if len(row) != 2:
            raise ValueError('h spans require start,end')
        a, b = map(_number, row)
        if a > b:
            raise ValueError('h interval must have start <= end')
        rows.append((a, b))
    return rows


def geometry_mask(targets):
    """Keep exact point/outside targets as misses, without manufacturing spans."""
    spans=_context_spans(targets.get('spans',()))
    raw=targets.get('geometry_eligible')
    masks=as_list(raw) if raw is not None else [0<=a<b<=1 for a,b in spans]
    if len(masks)!=len(spans) or any(type(v) is not bool for v in masks):
        raise ValueError('geometry_eligible must contain aligned booleans')
    # Explicit exact metadata may declare additional unsupported cases. A
    # normalized float collapse must not reach positive-interval arithmetic.
    return [mask and 0<=a<b<=1 for mask,(a,b) in zip(masks,spans)]


def event_supervision_targets(targets):
    """Return supported event rows and original indices; never erase coverage.

    H/P constructors produce positive intervals inside the observed work. Point
    and out-of-work source objects are retained by the adapter and counted by
    full metrics, but do not receive an invented interval for localization.
    """
    count=len(targets.get('spans',()))
    event_mask(targets)
    eligible=geometry_mask(targets)
    ids=targets.get('ids',[str(i) for i in range(count)])
    if len(ids)!=count or len(set(ids))!=count:
        raise ValueError('target event ids must be unique and aligned')
    selected=[i for i,ok in enumerate(eligible) if ok]
    excluded=[i for i,ok in enumerate(eligible) if not ok]
    result=dict(targets)
    for key in ('ids','spans','exact_spans','event_mask','geometry_eligible','view_ids','scope_ids','member_ids'):
        if key in result:
            rows=as_list(result[key])
            if len(rows)!=count:raise ValueError('target event arrays must align: '+key)
            result[key]=[rows[i] for i in selected]
    result['fields']={}
    for name,field in targets.get('fields',{}).items():
        values,masks=as_list(field['values']),as_list(field['mask'])
        if len(values)!=count or len(masks)!=count or any(type(v) is not bool for v in masks):
            raise ValueError('target event fields must align: '+name)
        result['fields'][name]={'values':[values[i] for i in selected],
                                'mask':[masks[i] for i in selected]}
    if excluded:
        # A complete region cannot be certified after excluding unresolved
        # geometry. Conservatively withhold its negative/precision claims.
        result['known_negative_spans']=[]
        result['complete_scopes']=[]
    return result,selected,excluded


def _context_eligible(a, b, threshold):
    # A point belongs to its physical note and H, and must lie within the
    # corresponding interval. No artificial positive duration is manufactured.
    if a[0] == a[1] and b[0] == b[1]:
        return a[0] == b[0]
    if a[0] == a[1]:
        return b[0] <= a[0] < b[1]
    if b[0] == b[1]:
        return a[0] <= b[0] < a[1]
    return interval_iou(a, b) >= threshold


def match_contexts(pred, targets, h_matches, *, mode='training', iou_threshold=.5):
    """Match h only within the same physical note and matched harmonic event.

    Annotation subcontexts remain separate; one prediction cannot absorb two
    role targets. Geometric overlap never supplies a chord-membership label.
    """
    if mode not in ('training', 'metric'):
        raise ValueError('mode must be training or metric')
    threshold = _number(iou_threshold)
    if not 0 < threshold <= 1:
        raise ValueError('iou_threshold must be in (0,1]')
    ps = _context_spans(pred.get('spans', ()))
    ts = _context_spans(targets.get('spans', ()))
    pn = _indices(pred.get('note_index', ()), 'h.note_index', len(ps))
    ph = _indices(pred.get('H_index', ()), 'h.H_index', len(ps))
    tn = _indices(targets.get('note_index', ()), 'target h.note_index', len(ts))
    th = _indices(targets.get('H_index', ()), 'target h.H_index', len(ts), True)
    tids = targets.get('ids', [str(i) for i in range(len(ts))])
    if len(tids) != len(ts) or len(set(tids)) != len(tids):
        raise ValueError('target h ids must be unique and aligned')
    pids = pred.get('ids', [str(i) for i in range(len(ps))])
    if len(pids) != len(ps):
        raise ValueError('prediction h ids must align')
    hmap = {t: p for p, t in h_matches}
    pg, tg = defaultdict(list), defaultdict(list)
    for i, key in enumerate(zip(pn, ph)):
        pg[key].append(i)
    mask = event_mask(targets) if 'spans' in targets else []
    geometry = geometry_mask(targets)
    for t, (note, harmony) in enumerate(zip(tn, th)):
        if mask[t] and geometry[t] and harmony in hmap:
            tg[(note, hmap[harmony])].append(t)
    result = []
    for key, target_indices in sorted(tg.items()):
        predictions = sorted(pg.get(key, ()), key=lambda i: (ps[i], str(pids[i])))
        target_indices = sorted(target_indices, key=lambda i: (ts[i], str(tids[i])))
        if not predictions:
            continue
        costs, allowed = [], []
        for t in target_indices:
            width = ts[t][1] - ts[t][0] or Fraction(1)
            costs.append([(abs(ps[p][0]-ts[t][0])+abs(ps[p][1]-ts[t][1]))/width
                          for p in predictions])
            allowed.append([compatible(pred, targets, p, t) and
                            (mode == 'training' or _context_eligible(ps[p], ts[t], threshold))
                            for p in predictions])
        penalty = (max(max(x) for x in costs)+1)*(len(target_indices)+1)
        matrix = [[v if ok else 2*penalty for v,ok in zip(row, eligibility)]
                  + [penalty]*len(target_indices) for row,eligibility in zip(costs,allowed)]
        result.extend((predictions[col], target_indices[row]) for row,col in _assignment(matrix)
                      if col<len(predictions) and allowed[row][col])
    return sorted(result)


def align_full(pred, targets, *, mode='training', iou_threshold=.5):
    """Independent exact event matching followed by note-context matching."""
    matched={}
    for level in ('H','P'):
        subset,indices,_=event_supervision_targets(targets.get(level,_EMPTY))
        matched[level]=[(p,indices[t]) for p,t in match_events(pred[level],subset,
            mode=mode,iou_threshold=iou_threshold)]
    matched['h'] = match_contexts(pred.get('h', {}), targets.get('h', {}), matched['H'],
                                  mode=mode, iou_threshold=iou_threshold)
    return matched


def endpoint_key(endpoint, *, allow_unresolved=False):
    if not isinstance(endpoint, dict):
        raise ValueError('endpoint requires a mapping')
    kind, index = endpoint.get('kind'), endpoint.get('index')
    if kind not in _LEVELS and not (allow_unresolved and kind == 'unsupported'):
        raise ValueError('endpoint requires kind n/h/H/P')
    low = -1 if allow_unresolved else 0
    if isinstance(index, bool) or not isinstance(index, Integral) or index < low:
        raise ValueError('endpoint requires an integer address')
    if kind == 'unsupported' or index == -1:
        identity = endpoint.get('id')
        if not isinstance(identity, str) or not identity:
            raise ValueError('unresolved target endpoint requires stable id')
        return kind, int(index), identity
    return kind, int(index)


def endpoint_set(endpoints, *, allow_unresolved=False):
    keys = [endpoint_key(e, allow_unresolved=allow_unresolved) for e in endpoints]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('addressed endpoint sets must be nonempty and unique')
    return tuple(sorted(keys))


def mapped_endpoint(endpoint, maps):
    address = endpoint_key(endpoint, allow_unresolved=True)
    if len(address) == 3:
        return None
    kind, index = address
    if kind == 'n':
        return kind, index
    mapped = maps.get(kind, {}).get(index)
    return None if mapped is None else (kind, mapped)


def mapped_set(endpoints, maps):
    endpoint_set(endpoints, allow_unresolved=True)
    keys = [mapped_endpoint(e, maps) for e in endpoints]
    return None if any(e is None for e in keys) else tuple(sorted(keys))


def _known(row):
    mask = row.get('mask', False)
    if type(mask) is not bool:
        raise ValueError('link target mask must be bool')
    if mask and type(row.get('value')) is not bool:
        raise ValueError('known link target value must be bool')
    return mask


def participant_keys(pred):
    logits = pred.get('existence_logits')
    count = len(logits) if logits is not None else 0
    parents = _indices(pred.get('P_index', ()), 'participants.P_index', count)
    indices = _indices(pred.get('endpoint_index', ()), 'participants.endpoint_index', count)
    levels = pred.get('endpoint_level', ())
    if len(levels) != count or any(k not in _LEVELS for k in levels):
        raise ValueError('participant endpoint levels must align')
    keys = [(p, (k, i)) for p,k,i in zip(parents,levels,indices)]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate participant candidate')
    return keys


def relation_keys(pred):
    sources, targets = pred.get('sources', ()), pred.get('targets', ())
    if len(sources) != len(targets):
        raise ValueError('relation endpoint arrays must align')
    keys = [(endpoint_set(a), endpoint_set(b)) for a,b in zip(sources, targets)]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate addressed relation candidate')
    return keys


def link_alignment(pred, targets, matched):
    """Return only explicit known link labels whose full endpoints are present."""
    maps = {kind: {t:p for p,t in rows} for kind,rows in matched.items()}
    part = pred.get('participants', {})
    pkeys = participant_keys(part);pindex = {key:i for i,key in enumerate(pkeys)}
    participants=[];seen={}
    for t,row in enumerate(targets.get('participants', ())):
        if not _known(row):
            continue
        parent = row.get('P_index')
        if type(parent) is not int or parent<0:
            raise ValueError('target participant P_index must be nonnegative integer')
        endpoint = mapped_endpoint(row['endpoint'], maps)
        if parent not in maps['P'] or endpoint is None:
            continue
        key = (maps['P'][parent], endpoint)
        if key in seen and seen[key] != row['value']:
            raise ValueError('contradictory known participant labels within one view')
        seen[key]=row['value']
        if key in pindex:
            participants.append((pindex[key],t))
    relations=[];seen={}
    rkeys=relation_keys(pred.get('relations', {}));rindex={key:i for i,key in enumerate(rkeys)}
    for t,row in enumerate(targets.get('relations', ())):
        if not _known(row):
            continue
        kind=row.get('type')
        if not isinstance(kind,str) or not kind:
            raise ValueError('known relation type must be a nonempty string')
        sources=mapped_set(row['source'],maps);destinations=mapped_set(row['target'],maps)
        if sources is None or destinations is None:
            continue
        key=(sources,destinations);typed=(kind,key)
        if typed in seen and seen[typed]!=row['value']:
            raise ValueError('contradictory addressed labels within one view')
        seen[typed]=row['value']
        if key in rindex and kind in pred.get('relations', {}).get('logits', {}):
            relations.append((kind,rindex[key],t))
    return participants,relations


def _field_loss(logits, fields, matches, target_count, prediction_count, zero):
    import torch
    import torch.nn.functional as F
    components={};support={}
    for name, values in logits.items():
        if values.ndim!=2 or values.shape[1]<1 or len(values)!=prediction_count:
            raise ValueError(f'field logits must align with [N,C]: {name}')
        components[name]=zero;support[name]=0
        field=fields.get(name)
        if field is None:
            continue
        masks=as_list(field['mask']);labels=as_list(field['values'])
        if len(masks)!=target_count or len(labels)!=target_count or any(type(m) is not bool for m in masks):
            raise ValueError(f'field mask/value alignment mismatch: {name}')
        selected=[(p,t) for p,t in matches if masks[t]]
        if selected:
            pi,ti=zip(*selected);truth=[labels[t] for t in ti]
            if any(type(v) is not int or not 0<=v<values.shape[1] for v in truth):
                raise ValueError(f'known field target outside vocabulary: {name}')
            components[name]=F.cross_entropy(values[list(pi)],torch.tensor(truth,device=values.device))
            support[name]=len(selected)
    return components,support


def participant_role_fields(rows):
    """Independent role labels, only on explicitly present participants."""
    names=sorted({name for row in rows for name in row.get('fields',{})})
    fields={name:{'values':[],'mask':[]} for name in names}
    for row in rows:
        known=_known(row)
        for name in names:
            role=row.get('fields',{}).get(name,{})
            mask=role.get('mask',False)
            if type(mask) is not bool:
                raise ValueError('participant role mask must be bool')
            if mask and known and not row['value']:
                raise ValueError('absent participant cannot have a known role')
            fields[name]['values'].append(role.get('value',-1))
            fields[name]['mask'].append(mask and known and row['value'])
    return fields


def full_loss(pred, targets, weights=None):
    """Return loss/components/support/supported with unknown-only draws skipped.

    Optional nonnegative weights address H,h,P,participants,relations and
    h_localization. They do not change annotation masks or candidate construction.
    All available field heads are averaged within their own level.
    """
    import torch
    import torch.nn.functional as F
    values=list(_tensor_leaves(pred))
    if not values:
        raise ValueError('prediction requires floating tensors')
    if any(not bool(torch.isfinite(t).all()) for t in values):
        raise ValueError('nonfinite whole-graph prediction')
    zero=sum((t.sum()*0 for t in values), values[0].new_zeros(()))
    weights=dict(weights or {})
    if set(weights)-{'H','h','P','participants','relations','h_localization'}:
        raise ValueError('unknown full loss weight')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('loss weights must be finite and nonnegative')
    matched=align_full(pred,targets)
    components={};support={};scope_losses={}
    for level in ('H','P'):
        original=targets.get(level,_EMPTY)
        subset,indices,excluded=event_supervision_targets(original)
        remap={old:new for new,old in enumerate(indices)}
        result=compute_loss(pred[level],subset,matched_events=[(p,remap[t]) for p,t in matched[level]])
        mask=event_mask(original)
        result['support']['known_events']=sum(mask)
        result['support']['geometry_excluded']=sum(mask[t] for t in excluded)
        result['support']['geometry_excluded_total']=len(excluded)
        result['support']['negative_scopes_withheld']=bool(excluded)
        components.update({level+'.'+k:v for k,v in result['components'].items()})
        support[level]=result['support'];scope_losses[level]=result['loss']
    hp=pred.get('h',{});ht=targets.get('h',{});target_count=len(ht.get('spans',()))
    field_components,field_support=_field_loss(hp.get('field_logits',{}),ht.get('fields',{}),matched['h'],target_count,len(hp.get('spans',())),zero)
    components.update({'h.field:'+k:v for k,v in field_components.items()})
    hlocal=zero
    if matched['h']:
        pi,ti=zip(*matched['h']);spans=hp['spans'];gold=torch.as_tensor(ht['spans'],dtype=spans.dtype,device=spans.device)
        hlocal=F.smooth_l1_loss(spans[list(pi)],gold[list(ti)],reduction='none').sum(-1).mean()
    components['h.localization']=hlocal
    active=[v for k,v in field_components.items() if field_support[k]]
    scope_losses['h']=weights.get('h_localization',1.)*hlocal+(sum(active)/len(active) if active else zero)
    support['h']={'known_contexts':sum(event_mask(ht)) if 'spans' in ht else 0,'matched_contexts':len(matched['h']),'geometry_excluded':sum(not m for m in geometry_mask(ht)),'fields':field_support}
    parts,rels=link_alignment(pred,targets,matched)
    ppart=pred.get('participants',{});tpart=targets.get('participants',())
    ploss=zero
    if parts:
        logits=ppart['existence_logits'];pi,ti=zip(*parts)
        ploss=F.binary_cross_entropy_with_logits(logits[list(pi)],logits.new_tensor([float(tpart[t]['value']) for t in ti]))
    components['participants.existence']=ploss
    role_fields=participant_role_fields(tpart)
    positive_parts=[(p,t) for p,t in parts if tpart[t]['value']]
    roles,rsupport=_field_loss(ppart.get('field_logits',{}),role_fields,positive_parts,
                              len(tpart),len(ppart.get('existence_logits',())),zero)
    components.update({'participants.field:'+k:v for k,v in roles.items()})
    active=[v for k,v in roles.items() if rsupport[k]];scope_losses['participants']=ploss+(sum(active)/len(active) if active else zero)
    rlosses=[];rs=defaultdict(int)
    for kind,logits in pred.get('relations',{}).get('logits',{}).items():
        if logits.ndim!=1 or len(logits)!=len(relation_keys(pred['relations'])):
            raise ValueError('relation logits must align with directed candidates')
        chosen=[(p,t) for k,p,t in rels if k==kind];value=zero
        if chosen:
            pi,ti=zip(*chosen);value=F.binary_cross_entropy_with_logits(logits[list(pi)],logits.new_tensor([float(targets['relations'][t]['value']) for t in ti]));rlosses.append(value);rs[kind]=len(chosen)
        components['relations.'+kind]=value
    scope_losses['relations']=sum(rlosses)/len(rlosses) if rlosses else zero
    support['participants']={'known':sum(_known(r) for r in tpart),'matched':len(parts),'fields':rsupport}
    support['relations']={'known':sum(_known(r) for r in targets.get('relations',())),'matched':len(rels),'by_type':dict(rs)}
    active_support={
        level: bool(support[level]['matched_events'] or support[level]['object_negative'])
        for level in ('H','P')
    }
    active_support['h']=bool((matched['h'] and weights.get('h_localization',1.)>0)
                             or any(field_support.values()))
    active_support['participants']=bool(parts)
    active_support['relations']=bool(rels)
    supported=any(active and weights.get(level,1.)>0 for level,active in active_support.items())
    loss=zero+sum(weights.get(level,1.)*value for level,value in scope_losses.items())
    return {'loss':loss,'components':components,'support':support,'supported':supported,
            'matches':matched,'participant_matches':parts,'relation_matches':rels}


compute_full_loss=full_loss
