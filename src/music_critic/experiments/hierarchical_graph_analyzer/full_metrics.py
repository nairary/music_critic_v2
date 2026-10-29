"""Coverage-preserving metrics for predicted or explicit oracle whole graphs.

Unknown candidates are not false positives. Link precision/F1 is conditional on
explicit known labels; positive recovery keeps known targets with absent
endpoints/candidates in its denominator. No harmonic target count sets a budget.
"""
from collections import defaultdict
from .full_matching import match_full_events
import math

from .matching import as_list, event_mask
from .metrics import evaluate_piece, _macro_f1, _ratio
from .full_losses import (_EMPTY, _known, endpoint_set, link_alignment, mapped_set,
                          match_contexts, participant_role_fields, relation_keys, event_supervision_targets)


def _fields(pred, targets, matches):
    count=len(targets.get('spans', ()));mask=event_mask(targets) if 'spans' in targets else []
    result={}
    for name,field in targets.get('fields',{}).items():
        values,available=as_list(field['values']),as_list(field['mask'])
        if len(values)!=count or len(available)!=count or any(type(v) is not bool for v in available):
            raise ValueError('context field mask/value alignment mismatch')
        known=sum(bool(mask[t] and available[t]) for t in range(count));pairs=[]
        logits=pred.get('field_logits',{}).get(name)
        if logits is not None:
            rows=as_list(logits)
            if len(rows)!=len(pred.get('spans',())):
                raise ValueError('h logits row alignment mismatch')
            for p,t in matches:
                if available[t]:
                    row=rows[p]
                    if not row or any(not math.isfinite(x) for x in row):
                        raise ValueError('field logits must be finite and nonempty')
                    if type(values[t]) is not int or not 0<=values[t]<len(row):
                        raise ValueError('known field outside fitted vocabulary')
                    pairs.append((values[t],max(range(len(row)),key=row.__getitem__)))
        correct=sum(a==b for a,b in pairs)
        result[name]=dict(known_support=known,matched_support=len(pairs),joint_correct=correct,
                          joint_recovery=_ratio(correct,known),conditional_macro_f1=_macro_f1(pairs),pairs=pairs,
                          modeled=logits is not None)
    return result


def _binary_metrics(target_rows, matched, scores):
    known=[i for i,row in enumerate(target_rows) if _known(row)]
    positive=sum(target_rows[i]['value'] for i in known)
    pairs=[];positive_found=0
    for p,t in matched:
        prediction=scores[p]>=0;truth=target_rows[t]['value'];pairs.append((int(truth),int(prediction)))
        positive_found+=int(truth and prediction)
    tp=sum(a==b==1 for a,b in pairs);fp=sum(a==0 and b==1 for a,b in pairs)
    fn=sum(a==1 and b==0 for a,b in pairs)
    return dict(known=len(known),known_positive=positive,known_negative=len(known)-positive,
                scored=len(pairs),unscored_known=len(known)-len(pairs),positive_found=positive_found,
                positive_recall=_ratio(positive_found,positive),conditional_precision=_ratio(tp,tp+fp),
                conditional_recall=_ratio(tp,tp+fn),conditional_f1=_ratio(2*tp,2*tp+fp+fn),
                conditional_macro_f1=_macro_f1(pairs),pairs=pairs,
                precision_scope='explicit known candidate labels only; unknown excluded')


def _relation_metrics(pred, targets, matched, links):
    candidate=pred.get('relations',{});rows=targets.get('relations',())
    kinds=sorted(set(candidate.get('logits',{})) | {r.get('type') for r in rows if _known(r)})
    by_type={};all_pairs=[];maps={kind:{t:p for p,t in pairs} for kind,pairs in matched.items()}
    keys=relation_keys(candidate)
    for kind in kinds:
        selected=[i for i,row in enumerate(rows) if row.get('type')==kind]
        local_rows=[rows[i] for i in selected];local_index={i:j for j,i in enumerate(selected)}
        typed=[(p,local_index[t]) for k,p,t in links if k==kind]
        logits=as_list(candidate.get('logits',{}).get(kind,()))
        if logits and (len(logits)!=len(keys) or any(not math.isfinite(x) for x in logits)):
            raise ValueError('relation score alignment/nonfinite values')
        report=_binary_metrics(local_rows,typed,logits);queries={}
        for row in local_rows:
            if not _known(row) or not row['value']:
                continue
            source=endpoint_set(row['source'],allow_unresolved=True);q=queries.setdefault(source,{'known_targets':set(),'mapped_targets':set(),'mapped_source':mapped_set(row['source'],maps)})
            q['known_targets'].add(endpoint_set(row['target'],allow_unresolved=True))
            target=mapped_set(row['target'],maps)
            if target is not None:q['mapped_targets'].add(target)
        details=[]
        for source,q in sorted(queries.items()):
            candidates=[(logits[i],t) for i,(s,t) in enumerate(keys)
                        if logits and s==q['mapped_source']]
            # A full endpoint set is one answer; no Cartesian pair expansion.
            ranked=[t for _,t in sorted(candidates,key=lambda x:(-x[0],x[1]))]
            truth=q['mapped_targets'];denom=len(q['known_targets'])
            hits1=sum(t in truth for t in ranked[:1]);hits5=sum(t in truth for t in ranked[:5])
            ranks=[i+1 for i,t in enumerate(ranked) if t in truth]
            details.append(dict(known_targets=denom,candidates=len(ranked),recall_at1=hits1/denom,
                                recall_at5=hits5/denom,hits_at1=float(hits1>0),hits_at5=float(hits5>0),
                                reciprocal_rank=1/min(ranks) if ranks else 0.))
        report['retrieval_queries']=details;report['retrieval_query_count']=len(details)
        for metric in ('recall_at1','recall_at5','hits_at1','hits_at5','reciprocal_rank'):
            report[metric]=_ratio(sum(q[metric] for q in details),len(details))
        by_type[kind]=report;all_pairs.extend(report['pairs'])
    known=sum(v['known'] for v in by_type.values());positive=sum(v['known_positive'] for v in by_type.values());found=sum(v['positive_found'] for v in by_type.values())
    return dict(known=known,known_positive=positive,known_negative=known-positive,
                scored=sum(v['scored'] for v in by_type.values()),positive_found=found,
                positive_recall=_ratio(found,positive),by_type=by_type,pairs=all_pairs)


def evaluate_full_piece(pred, targets, *, geometry='predicted', iou_threshold=.5,
                        max_h_predictions=None, max_p_predictions=None, duration_quarters=None):
    """Evaluate one work and one analytic view after raw-only prediction.

    Default budgets retain all model-emitted candidates. Explicit maxima are
    fixed configuration limits, never derived from target counts. The model's
    construction caps and raw exclusions are retained in the report separately.
    Oracle output must be marked by its separate model diagnostic entry point.
    """
    if geometry not in ('predicted','oracle') or pred.get('geometry','predicted')!=geometry:
        raise ValueError('geometry must agree with explicit predicted/oracle output')
    duration=duration_quarters if duration_quarters is not None else targets.get('duration_quarters',1.)
    duration=float(duration)
    if not math.isfinite(duration) or duration<=0:raise ValueError('positive observed duration required')
    result={'geometry':geometry,'family_id':targets.get('family_id'),'work_id':targets.get('work_id'),
            'view_id':targets.get('view_id'),'model_coverage':dict(pred.get('coverage',{})),
            'target_coverage':dict(targets.get('coverage',{}))}
    for kind,cap,field in [('H',max_h_predictions,'H.function'),('P',max_p_predictions,'P.process')]:
        # Rate derived from raw model outputs implements retain-all unless an
        # explicit cap was supplied; it never reads annotations or gold counts.
        rate=max(1,len(pred[kind]['spans']))/duration
        t=dict(targets.get(kind,_EMPTY));t.setdefault('family_id',targets.get('family_id'));t.setdefault('work_id',targets.get('work_id'));t.setdefault('view_id',targets.get('view_id'))
        original=t
        t,indices,excluded=event_supervision_targets(original)
        result[kind]=evaluate_piece(pred[kind],t,duration_quarters=duration,candidates_per_quarter=rate,
                                   max_predictions=cap,iou_threshold=iou_threshold,function_field=field,
                                   span_scale_quarters=duration,matcher=match_full_events)
        report=result[kind]
        mask=event_mask(original)
        excluded_known=sum(mask[i] for i in excluded)
        report['matches']=[(p,indices[t]) for p,t in report['matches']]
        report['known_h']+=excluded_known
        report['known_h_recovery']=_ratio(report['recovered_h'],report['known_h'])
        report['geometry_excluded_known']=excluded_known
        report['geometry_excluded_total']=len(excluded)
        report['negative_scopes_withheld']=bool(excluded)
        for name,values in report['fields'].items():
            masks=as_list(original['fields'][name]['mask'])
            values['known_support']+=sum(mask[i] and masks[i] for i in excluded)
            values['joint_recovery']=_ratio(values['joint_correct'],values['known_support'])
            if name==field:
                report['known_f']=values['known_support']
                report['joint_h_f_recovery']=values['joint_recovery']
        if kind=='P':
            report['known_p_recovery']=report['known_h_recovery']
            report['joint_p_type_recovery']=report['joint_h_f_recovery']
            report['conditional_type_macro_f1']=report['conditional_f_macro_f1']
    matched={kind:result[kind]['matches'] for kind in ('H','P')}
    matched['h']=match_contexts(pred.get('h',{}),targets.get('h',{}),matched['H'],mode='metric',iou_threshold=iou_threshold)
    ht=targets.get('h',{});known=sum(event_mask(ht)) if 'spans' in ht else 0
    result['h']=dict(known_contexts=known,recovered_contexts=len(matched['h']),known_context_recovery=_ratio(len(matched['h']),known),fields=_fields(pred.get('h',{}),ht,matched['h']),matches=matched['h'])
    parts,rels=link_alignment(pred,targets,matched)
    ps=as_list(pred.get('participants',{}).get('existence_logits',()))
    if any(not math.isfinite(v) for v in ps):raise ValueError('nonfinite participant scores')
    target_parts=targets.get('participants',())
    result['participants']=_binary_metrics(target_parts,parts,ps)
    # Reuse independently masked categorical scoring; the dummy row geometry
    # denotes link rows only and is never used for event matching.
    role_targets={'spans':[[0,1]]*len(target_parts),
                  'fields':participant_role_fields(target_parts)}
    role_pred={'spans':[[0,1]]*len(ps),
               'field_logits':pred.get('participants',{}).get('field_logits',{})}
    result['participants']['fields']=_fields(role_pred,role_targets,
                                            [(p,t) for p,t in parts if target_parts[t]['value']])
    result['relations']=_relation_metrics(pred,targets,matched,rels)
    result['matches']=matched
    return result


def _pool_fields(reports):
    names=sorted({name for fields in reports for name in fields});out={}
    for name in names:
        rows=[r[name] for r in reports if name in r];pairs=[p for r in rows for p in r['pairs']]
        known=sum(r['known_support'] for r in rows);correct=sum(r['joint_correct'] for r in rows)
        out[name]=dict(known_support=known,matched_support=len(pairs),joint_correct=correct,
                       joint_recovery=_ratio(correct,known),conditional_macro_f1=_macro_f1(pairs),pairs=pairs)
    return out


def _pool_binary(rows):
    known=sum(r['known'] for r in rows);positive=sum(r['known_positive'] for r in rows);found=sum(r['positive_found'] for r in rows);pairs=[p for r in rows for p in r.get('pairs',())]
    tp=sum(a==b==1 for a,b in pairs);fp=sum(a==0 and b==1 for a,b in pairs);fn=sum(a==1 and b==0 for a,b in pairs)
    return dict(known=known,known_positive=positive,positive_found=found,positive_recall=_ratio(found,positive),
                scored=len(pairs),conditional_precision=_ratio(tp,tp+fp),conditional_f1=_ratio(2*tp,2*tp+fp+fn),
                conditional_macro_f1=_macro_f1(pairs))


def aggregate_full_metrics(reports):
    """Pool within source families, then macro across supported families only."""
    groups=defaultdict(list);geometries={r['geometry'] for r in reports}
    if len(geometries)>1:raise ValueError('cannot aggregate mixed geometry regimes')
    for report in reports:
        family=report.get('family_id')
        if not isinstance(family,str) or not family:raise ValueError('family_id required')
        groups[family].append(report)
    families={}
    for family,rows in sorted(groups.items()):
        summary={}
        for kind in ('H','P'):
            ev=[r[kind] for r in rows];known=sum(r['known_h'] for r in ev);recovered=sum(r['recovered_h'] for r in ev)
            knownf=sum(r['known_f'] for r in ev);correct=sum(r['joint_h_f_correct'] for r in ev);pairs=[p for r in ev for p in r['conditional_f_pairs']]
            summary[kind]=dict(known_h_recovery=_ratio(recovered,known),joint_h_f_recovery=_ratio(correct,knownf),conditional_f_macro_f1=_macro_f1(pairs),known_events=known,recovered_events=recovered,known_function=knownf,fields=_pool_fields([r['fields'] for r in ev]))
        h=[r['h'] for r in rows];known=sum(r['known_contexts'] for r in h)
        summary['h']=dict(known_context_recovery=_ratio(sum(r['recovered_contexts'] for r in h),known),known_contexts=known,fields=_pool_fields([r['fields'] for r in h]))
        summary['participants']=_pool_binary([r['participants'] for r in rows]);summary['participants']['fields']=_pool_fields([r['participants']['fields'] for r in rows]);summary['relations']=_pool_binary([r['relations'] for r in rows])
        relation_types=sorted({k for r in rows for k in r['relations']['by_type']});summary['relations']['by_type']={}
        for kind in relation_types:
            typed=[r['relations']['by_type'][kind] for r in rows if kind in r['relations']['by_type']];value=_pool_binary(typed);queries=[q for r in typed for q in r['retrieval_queries']]
            for key in ('recall_at1','recall_at5','hits_at1','hits_at5','reciprocal_rank'):
                value[key]=_ratio(sum(q[key] for q in queries),len(queries))
            value['retrieval_query_count']=len(queries);summary['relations']['by_type'][kind]=value
        families[family]=summary
    out={'geometry':next(iter(geometries),None),'families':families,'family_count':len(families)}
    def macro(path):
        vals=[]
        for f in families.values():
            value=f
            for key in path:value=value.get(key,{}) if isinstance(value,dict) else None
            if isinstance(value,(int,float)) and value is not None:vals.append(value)
        return _ratio(sum(vals),len(vals)),len(vals)
    for scope,keys in {'H':('known_h_recovery','joint_h_f_recovery','conditional_f_macro_f1'),
                       'P':('known_h_recovery','joint_h_f_recovery','conditional_f_macro_f1'),
                       'h':('known_context_recovery',),
                       'participants':('positive_recall','conditional_precision','conditional_f1','conditional_macro_f1'),
                       'relations':('positive_recall','conditional_precision','conditional_f1','conditional_macro_f1')}.items():
        out[scope]={}
        for key in keys:
            value,support=macro((scope,key));out[scope][key]=value;out[scope]['supported_families_'+key]=support
    out['P']['known_p_recovery']=out['P']['known_h_recovery'];out['P']['joint_p_type_recovery']=out['P']['joint_h_f_recovery'];out['P']['conditional_type_macro_f1']=out['P']['conditional_f_macro_f1']
    for scope in ('H','h','P','participants'):
        fields=sorted({name for f in families.values() for name in f[scope]['fields']});out[scope]['fields']={}
        for field in fields:
            out[scope]['fields'][field]={}
            for key in ('joint_recovery','conditional_macro_f1'):
                value,support=macro((scope,'fields',field,key));out[scope]['fields'][field][key]=value;out[scope]['fields'][field]['supported_families_'+key]=support
    out['relations']['by_type']={}
    for kind in sorted({k for f in families.values() for k in f['relations']['by_type']}):
        out['relations']['by_type'][kind]={}
        for key in ('positive_recall','conditional_f1','recall_at1','recall_at5','hits_at1','hits_at5','reciprocal_rank'):
            value,support=macro(('relations','by_type',kind,key));out['relations']['by_type'][kind][key]=value;out['relations']['by_type'][kind]['supported_families_'+key]=support
    return out


full_metrics=evaluate_full_piece
