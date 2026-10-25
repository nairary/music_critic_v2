"""Prepare the preselected development diagnostic after registering its record."""
from __future__ import annotations
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import time

from scripts.hcm1_diagnostic_data import (read_bound,read_validation,save_piece,fixed_queries,
    query_json,pair_query,paired_examples)


def selected_authored_rows(manifest,plan):
    import re
    ids=plan['pair_ids'];variants=plan.get('variant_ids',['H1','R1','M1','S1'])
    if len(ids)!=len(set(ids)) or not variants or any(not re.fullmatch('[HRMS][1-5]',v) for v in variants):
        raise ValueError('invalid planned pair/variant IDs')
    rows=[r for r in manifest['pairs'] if r['pair_id'] in ids]
    if len(rows)!=len(ids) or any(r['split']!='head_dev' or r['variant_id'] not in variants for r in rows):
        raise ValueError('paired metadata scope changed')
    return rows


def generation_ids(plan,cases):
    if 'generation_ids' in plan:
        ids=plan['generation_ids'];by_id={c['id']:c for c in cases}
        if len(ids)!=len(set(ids)) or any(i not in by_id or by_id[i]['kind']!='restoration' or by_id[i]['split']!='validation' for i in ids):
            raise ValueError('generation must use distinct declared validation restoration slots')
        return list(ids)
    return [c['id'] for source in ('hooktheory','pop909_cl') for c in
        [c for c in cases if c['kind']=='restoration' and c['source']==source and c['split']=='validation' and c.get('mask_bars')==1][:2]]


def prepare(plan_path,output):
    from music_critic.data import loads_piece
    from music_critic.pdmx_pipeline.windowing import iter_bar_windows,WindowConfig
    from music_critic.pll.codec import EventCodec
    from music_critic.pll.contracts import ContextProfile
    from music_critic.pll.masking import split_masked_piece
    from music_critic.critic.authored_data import load_authored_dataset, _preflight
    from music_critic.pll.cli import _admit_output
    plan=json.loads(Path(plan_path).read_text());output=Path(output)
    _admit_output(output)
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'HCM1DiagnosticPreparation@1','status':'running','accesses':[],'cases':[],'exclusions':[],
            'plan_sha256':sha256(Path(plan_path).read_bytes()).hexdigest()}
    started=time.monotonic();codec=EventCodec();profile=ContextProfile()
    def add_blocks(piece,source,split,work_id,piece_binding,prefix):
        for i,query in enumerate(fixed_queries(piece)):
            case=dict(id=f'{prefix}:{i+1}',kind='restoration',source=source,split=split,
                work_id=work_id,original=piece_binding,query=query_json(query),mask_bars=i+1)
            try:
                ex=split_masked_piece(piece,query,profile);codec.encode(ex.targets,span=ex.span,tracks=ex.tracks)
                case.update(supported=True,target_events=len(ex.targets))
            except ValueError as exc:case.update(supported=False,reason=str(exc))
            report['cases'].append(case)
    try:
        for index,row in enumerate(plan['validation']):
            prefix='validation:'+row['source']+':'+str(index)
            piece=read_validation(Path(plan['cache_root'])/row['source']/'multisource-cache-v1.0.0',row,report['accesses'])
            selected=None
            for window in iter_bar_windows(piece,WindowConfig(bar_count=16,stride=16,max_notes_per_window=1024)):
                if window.piece is None or len(window.piece.bars)<2 or not targets_preserved(piece,window):
                    report['exclusions'].append(dict(id=prefix,bar_start=window.source_bar_start,reason=window.reason or 'changed_targets_or_short'))
                    continue
                report.setdefault('windowing',[]).append(dict(id=prefix,source_bar_start=window.source_bar_start,
                    clipped_context_notes=window.clipped_note_count,targets_unchanged=True))
                selected=window.piece;break
            if selected is None:
                for length in (1,2):report['cases'].append(dict(id=prefix+':'+str(length),kind='restoration',source=row['source'],split='validation',work_id=row['component_id'],mask_bars=length,supported=False,reason='no bounded window preserving masked targets'))
                continue
            add_blocks(selected,row['source'],'validation',row['component_id'],save_piece(output,selected),prefix)
        for index,row in enumerate(plan['train']):
            if row['split']!='train':raise ValueError('TRAIN comparison preflight')
            piece=loads_piece(read_bound(plan['train_root'],row).decode())
            report['accesses'].append(dict(piece_id=piece.piece_id,declared_split='train',observed_split=piece.split,verified=piece.split=='train'))
            if piece.split!='train' or piece.piece_id!=row['piece_id']:raise ValueError('TRAIN comparison identity')
            add_blocks(piece,row['source'],'train',row['work_id'],save_piece(output,piece),'train:'+row['source']+':'+str(index))
        authored_root=Path(plan['authored_root']);manifest=json.loads(read_bound(authored_root,plan['authored_manifest']).decode())
        rows=selected_authored_rows(manifest,plan)
        subset={**manifest,'pairs':rows}
        _preflight(subset)  # Validate every scope and binding before any music payload.
        for row in rows:
            for key in ('original','variant','plan','direction_review'):
                bound=row.get(key)
                if not bound:continue
                raw=read_bound(authored_root,bound);path=output/bound['path'];path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():path.write_bytes(raw)
        subset_path=output/'authored-subset.json';subset_path.write_text(json.dumps(subset,ensure_ascii=False,sort_keys=True))
        dataset=load_authored_dataset(subset_path,sha256(subset_path.read_bytes()).hexdigest())
        by_id={r['pair_id']:r for r in rows}
        for pair in dataset.pairs:
            row=by_id[pair.pair_id];raw_plan=json.loads(read_bound(output,row['plan']).decode())
            item=next(v for v in raw_plan['variants'] if v['id']==row['variant_id'])
            case=dict(id=pair.pair_id,kind='pair',split='campaign_head_dev',source='authored',
                work_id=pair.work_id,genre=pair.genre,aspect=item['aspect'],original=row['original'],variant=row['variant'],
                description=item,source_scope=row['source_scope'],hcm_training_overlap=training_overlap(pair.original,row,plan['training_identities']),
                author_preference=list(pair.targets.preference),human_preference='unknown')
            try:
                query,changed=pair_query(pair.original,pair.variant)
                a,b,identity=paired_examples(pair.original,pair.variant,query)
                codec.encode(a.targets,span=a.span,tracks=a.tracks);codec.encode(b.targets,span=b.span,tracks=b.tracks)
                case.update(supported=True,query=query_json(query),context_sha256=identity,changed_note_ids=list(changed),
                    original_events=len(a.targets),variant_events=len(b.targets),identical_projected_targets=a.targets==b.targets)
            except ValueError as exc:case.update(supported=False,reason=str(exc))
            report['cases'].append(case)
        # Select generation IDs by fixed source/list order, not scores or outputs.
        report['generation_ids']=generation_ids(plan,report['cases'])
        report.update(status='completed',elapsed_seconds=time.monotonic()-started)
        return report
    except Exception as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)},elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        (output/'cases.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False))


def training_overlap(piece,row,identities):
    """Only observed identity/hash matches; absence does not rule out aliases."""
    ids={piece.piece_id,piece.source_group_id,row['work_id'],row['component_id']}
    matched=[]
    for item in identities:
        keys={item.get(k) for k in ('piece_id','source_group_id','work_id','component_id')}-{None}
        reasons=sorted(ids & keys)
        if item.get('canonical_sha256')==row['original']['sha256']:reasons.append('exact canonical SHA256')
        if reasons:matched.append(dict(source=item.get('source'),reasons=reasons))
    return dict(known_overlap=bool(matched),matches=matched,
                limitation='No known match is not proof of independence; cross-corpus aliases and inherited C3 exposure unknown')


def targets_preserved(source,window):
    """Clipped context at outer window edges is allowed; masked answers aren't."""
    from collections import Counter
    for query in fixed_queries(window.piece):
        def notes(piece,start,end,offset):
            return Counter((n.track_id,n.onset_qn-offset,n.pitch,n.duration_qn,n.is_percussion,n.is_grace)
                for n in piece.notes if n.track_id in query.track_ids and start<=n.onset_qn<end)
        offset=window.global_offset_qn
        if notes(source,query.start+offset,query.end+offset,offset)!=notes(window.piece,query.start,query.end,offset-offset):
            return False
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=prepare(a.plan,a.output)
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),'supported':sum(c.get('supported',False) for c in report['cases'])}))
