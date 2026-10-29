"""Bounded, raw-derived tiles, masked objectives and auditable pilot metrics."""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter,OrderedDict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json,math,random,time
import torch
from .data import checked_read,exact
from .model import Probe,masked_pair_loss


def binding(path,relative=None):
    path=Path(path);b=path.read_bytes()
    return dict(path=str(relative or path),sha256=sha256(b).hexdigest(),size_bytes=len(b))


def dump_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def exact_slot(times,t):
    k=bisect_right(times,t)-1
    return k if k>=0 and times[k]==t else None


def retrieval_at_k(order,gold,total_gold,k):
    found=len(set(order[:k])&set(gold))
    return int(found>0),found/max(1,total_gold)


def build_cache(dataset,output):
    from music_critic.data import loads_piece
    from music_critic.pll.contracts import ContextProfile,MaskQuery
    from music_critic.pll.observed_view import ObservedMusicView
    from music_critic.pll.masking import build_observed_tiles
    from music_critic.pll.graph import build_tile_graph
    dataset=Path(dataset);manifest=json.loads((dataset/'manifest.json').read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];counts=Counter();exclusions=[]
    for work in manifest['works']:
        piece=loads_piece(checked_read(work['raw'],dataset))
        targets=json.loads(checked_read(work['targets'],dataset))
        counts[work['split']+'_annotated_binary_targets']+=sum(len(t['targets']) for t in targets)
        start=piece.bars[0].start_qn;end=piece.bars[-1].start_qn+piece.bars[-1].duration_qn
        profile=ContextProfile(context_bars=max(32,len(piece.bars)),max_notes_per_tile=2048)
        view=ObservedMusicView(piece,MaskQuery(start,end,tuple(t.track_id for t in piece.tracks)),tuple(range(len(piece.bars))))
        try:tiles=build_observed_tiles(view,profile)
        except ValueError as error:
            exclusions.append(dict(work=work['id'],reason=str(error)));continue
        byid={n.note_id:n for n in piece.notes}
        for ordinal,tile in enumerate(tiles):
            core=[b for b in piece.bars if b.bar_id in tile.core_bar_ids]
            left=core[0].start_qn.to_fraction();right=(core[-1].start_qn+core[-1].duration_qn).to_fraction()
            selected=[t for t in targets if left<=exact(t['time'])<right]
            if not selected:continue
            try:graph=build_tile_graph(tile,profile)
            except ValueError as error:
                exclusions.append(dict(work=work['id'],tile=ordinal,reason=str(error)));continue
            nmap={identity:i for i,(_,identity) in enumerate(graph.keys['note'])}
            times=[t+tile.global_start.to_fraction() for t,_ in graph.slots[0].time_slots]
            event_ids={};note_index=[];slice_index=[];raw=[]
            def event(note_id,t):
                k=exact_slot(times,t)
                if note_id not in nmap or k is None:return None
                n=byid[note_id];actual=times[k]
                a=n.onset_qn.to_fraction();b=(n.onset_qn+n.duration_qn).to_fraction()
                if not (a<=actual<b or a==b==actual):return None
                key=(note_id,k)
                if key not in event_ids:
                    age=float(actual-n.onset_qn.to_fraction());dur=float(n.duration_qn.to_fraction())
                    event_ids[key]=len(raw);note_index.append(nmap[note_id]);slice_index.append(k)
                    raw.append([n.pitch/127,math.sin(n.pitch%12*math.pi/6),math.cos(n.pitch%12*math.pi/6),
                        dur/(1+dur),float(actual-tile.global_start.to_fraction())/32,
                        int(n.track_id.split(':')[-1])/16,float(n.is_percussion),float(n.is_grace),
                        age/(1+abs(age)),float(n.onset_qn.to_fraction()-tile.global_start.to_fraction())/32])
                return event_ids[key]
            src=[];dst=[];labels=[];masks=[];pair_weight=[];exist=[];exist_y=[];queries=[]
            conflicts={};target_sets={};seen_sources=set()
            for t in selected:
                key=(t['note_id'],tuple(t['time']))
                conflicts.setdefault(key,set()).add(t['label'])
                target_sets.setdefault(key,set()).add(tuple(sorted((p['note_id'],tuple(p['time'])) for p in t['targets'])))
            for t in selected:
                key=(t['note_id'],tuple(t['time']))
                if len(conflicts[key])>1:
                    counts['alternate_view_conflicts_masked']+=1;continue
                if key in seen_sources:counts['compatible_source_duplicates']+=1;continue
                seen_sources.add(key)
                if len(target_sets[key])>1:
                    counts['alternative_endpoint_sets_masked']+=1;t={**t,'targets':[]}
                source=event(t['note_id'],exact(t['time']))
                if source is None:counts['unsupported_source']+=1;continue
                exist.append(source);exist_y.append(t['label'])
                counts[work['split']+'_positive' if t['label'] else work['split']+'_none']+=1
                sn=byid[t['note_id']];at=exact(t['time'])
                candidates=[n for n in tile.piece.notes if not n.is_percussion and n.note_id!=t['note_id']
                    and at<=byid[n.note_id].onset_qn.to_fraction()<=at+8]
                candidates.sort(key=lambda n:(byid[n.note_id].onset_qn.to_fraction()-at,abs(n.pitch-sn.pitch),n.note_id))
                candidates=candidates[:32]
                gold={(p['note_id'],tuple(p['time'])) for p in t['targets']}
                counts['annotated_binary_targets']+=len(gold)
                offsets=[];gold_offsets=[];first_pair=len(src)
                for n in candidates:
                    dest=event(n.note_id,byid[n.note_id].onset_qn.to_fraction())
                    if dest is None:continue
                    key=(n.note_id,tuple([byid[n.note_id].onset_qn.num,byid[n.note_id].onset_qn.den]))
                    positive=key in gold
                    idx=len(src);offsets.append(idx)
                    if positive:gold_offsets.append(idx)
                    src.append(source);dst.append(dest);labels.append(float(positive))
                    # Known-none is an explicit source-scoped absence, never an unlabelled tail.
                    masks.append(bool(positive or not t['label']))
                    pair_weight.append(1.)
                known=sum(masks[first_pair:])
                for idx in range(first_pair,len(src)):pair_weight[idx]=1/max(1,known)
                counts['retrievable_binary_targets']+=len(gold_offsets)
                counts[work['split']+'_retrievable_binary_targets']+=len(gold_offsets)
                if gold:queries.append(dict(indices=offsets,gold=gold_offsets,total_gold=len(gold)))
            if not exist:continue
            payload=dict(schema='RelationProbeTile@1',graph=graph,events=dict(
                note_index=torch.tensor(note_index),slice_index=torch.tensor(slice_index),raw=torch.tensor(raw,dtype=torch.float32)),
                boundary=torch.tensor(tile.boundary_flags,dtype=torch.float32),
                source=torch.tensor(src,dtype=torch.long),target=torch.tensor(dst,dtype=torch.long),
                labels=torch.tensor(labels,dtype=torch.float32),mask=torch.tensor(masks,dtype=torch.bool),
                weights=torch.tensor(pair_weight),exist_source=torch.tensor(exist,dtype=torch.long),
                exist_y=torch.tensor(exist_y,dtype=torch.float32),queries=queries,
                work=work['id'],family=work['family'],split=work['split'],
                negative_policy='reviewed_none_of_outgoing_resolution_from_exact_source_context_in_selected_view; only raw candidate domain',
                parent_raw_sha256=work['raw']['sha256'],parent_targets_sha256=work['targets']['sha256'])
            name=f"{work['id']}-{ordinal:04d}.pt";torch.save(payload,output/name)
            rows.append(dict(**binding(output/name,name),work=work['id'],family=work['family'],split=work['split']))
        print(json.dumps(dict(cached_work=work['id'],tiles=len(rows))),flush=True)
    result=dict(schema='RelationProbeCache@1',tiles=rows,counts=dict(counts),exclusions=exclusions,
                dataset_manifest_sha256=sha256((dataset/'manifest.json').read_bytes()).hexdigest(),
                window=dict(core_bars=4,halo_bars=1,max_notes=2048,lookahead_qn=8,max_candidates=32),
                human_gold=False,test_locked=True)
    dump_json(output/'manifest.json',result)
    return result


class TileStore:
    def __init__(self,root,limit=64):self.root=Path(root);self.limit=limit;self.cache=OrderedDict()
    def get(self,row):
        name=row['path']
        if name in self.cache:self.cache.move_to_end(name);return self.cache[name]
        from io import BytesIO
        # These are our own generated graph objects, admitted by exact hash/size.
        item=torch.load(BytesIO(checked_read(row,self.root)),map_location='cpu',weights_only=False)
        if item['schema']!='RelationProbeTile@1' or item['split']!=row['split'] or item['family']!=row['family']:
            raise ValueError('cache identity mismatch')
        self.cache[name]=item
        while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return item


def device_item(item,device):
    return dict(graph=item['graph'].to(device),events={k:v.to(device) for k,v in item['events'].items()},
        **{k:item[k].to(device) for k in ('boundary','source','target','labels','mask','weights','exist_source','exist_y')})


def confusion_metrics(values):
    tp,fp,tn,fn=values
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
    pos=2*tp/max(1,2*tp+fp+fn);neg=2*tn/max(1,2*tn+fp+fn)
    supported=[]
    if tp+fn:supported.append(pos)
    if tn+fp:supported.append(neg)
    return dict(macro_f1=sum(supported)/max(1,len(supported)),precision=p,recall=r,tp=tp,fp=fp,tn=tn,fn=fn)


@torch.no_grad()
def evaluate(model,rows,store,device,features_cache=None):
    model.eval();conf=[0,0,0,0];perwork={};hits1=hits5=queries=0;recall1=recall5=0.;pairloss=0.;brier=0.;n=0
    for row in rows:
        item=store.get(row);batch=device_item(item,device)
        features=(features_cache[row['path']].to(device) if features_cache is not None
                  else model.features(batch['graph'],batch['events'],batch['boundary']))
        scores,presence=model.head(features,batch['source'],batch['target'],batch['exist_source'])
        pred=presence>=0;y=batch['exist_y'].bool()
        c=[int((pred&y).sum()),int((pred&~y).sum()),int((~pred&~y).sum()),int((~pred&y).sum())]
        conf=[a+b for a,b in zip(conf,c)];wc=perwork.setdefault(item['work'],[0,0,0,0])
        perwork[item['work']]=[a+b for a,b in zip(wc,c)]
        brier+=float((presence.sigmoid()-y.float()).square().sum());n+=len(y)
        pairloss+=float(masked_pair_loss(scores,batch['labels'],batch['mask'],batch['weights']))
        score_values=scores.detach().cpu().tolist()
        for query in item['queries']:
            order=sorted(query['indices'],key=lambda i:score_values[i],reverse=True)
            h1,r1=retrieval_at_k(order,query['gold'],query['total_gold'],1)
            h5,r5=retrieval_at_k(order,query['gold'],query['total_gold'],5)
            hits1+=h1;hits5+=h5;recall1+=r1;recall5+=r5;queries+=1
    metrics=confusion_metrics(conf)
    metrics.update(work_macro_f1=sum(confusion_metrics(v)['macro_f1'] for v in perwork.values())/max(1,len(perwork)),
        brier=brier/max(1,n),pair_bce=pairloss/max(1,len(rows)),recall_at_1=recall1/max(1,queries),
        recall_at_5=recall5/max(1,queries),hits_at_1=hits1/max(1,queries),hits_at_5=hits5/max(1,queries),
        retrieval_queries=queries,works=len(perwork),contexts=n,
        per_work={k:confusion_metrics(v) for k,v in perwork.items()})
    return metrics


def train_run(cache,output,mode,decoder,seed,checkpoint,steps=600,max_seconds=1200,device='cuda'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cache=Path(cache);manifest=json.loads((cache/'manifest.json').read_text());store=TileStore(cache)
    fit=[x for x in manifest['tiles'] if x['split']=='fit'];dev=[x for x in manifest['tiles'] if x['split']=='dev']
    if not fit or not dev:raise ValueError('nonempty family fit/dev required')
    if {x['family'] for x in fit}&{x['family'] for x in dev}:raise ValueError('family split leakage')
    torch.set_num_threads(4);torch.manual_seed(2000+seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Probe(mode,decoder,seed,checkpoint).to(device)
    rate=1e-3 if mode in ('raw','frozen') else 3e-4
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=rate,weight_decay=1e-4)
    rng=random.Random(seed);bywork={}
    for row in fit:bywork.setdefault(row['work'],[]).append(row)
    names=sorted(bywork);features_cache=None
    started=time.monotonic();best=-1.;completed=0;events=[]
    torch.cuda.reset_peak_memory_stats() if device.startswith('cuda') else None
    if mode=='frozen':
        features_cache={};model.eval()
        with torch.no_grad():
            for row in fit+dev:
                if time.monotonic()-started>max_seconds:raise TimeoutError('frozen encoding exceeded run budget')
                batch=device_item(store.get(row),device)
                features_cache[row['path']]=model.features(batch['graph'],batch['events'],batch['boundary']).cpu()
    # Use work-uniform draws, not label-count-uniform draws.
    log=(output/'events.jsonl').open('x')
    for step in range(1,steps+1):
        if time.monotonic()-started>max_seconds:break
        row=rng.choice(bywork[rng.choice(names)]);item=store.get(row);batch=device_item(item,device)
        model.train();optimizer.zero_grad(set_to_none=True)
        features=(features_cache[row['path']].to(device) if features_cache is not None
                  else model.features(batch['graph'],batch['events'],batch['boundary']))
        scores,presence=model.head(features,batch['source'],batch['target'],batch['exist_source'])
        presence_loss=masked_pair_loss(presence,batch['exist_y'],torch.ones_like(batch['exist_y'],dtype=torch.bool))
        loss=presence_loss+masked_pair_loss(scores,batch['labels'],batch['mask'],batch['weights'])
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
        optimizer.step();completed=step
        if step==1 or step%50==0:
            entry=dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),wall_seconds=time.monotonic()-started)
            log.write(json.dumps(entry)+'\n');log.flush();dump_json(output/'progress.json',entry)
        if step%200==0 or step==steps:
            metrics=evaluate(model,dev,store,device,features_cache);entry=dict(step=step,dev=metrics)
            log.write(json.dumps(entry)+'\n');log.flush();events.append(entry)
            payload=dict(schema='RelationProbeCheckpoint@1',mode=mode,decoder=decoder,seed=seed,step=step,
                cache_sha256=sha256((cache/'manifest.json').read_bytes()).hexdigest(),model=model.state_dict(),
                optimizer=optimizer.state_dict(),sampling_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                config=dict(steps=steps,lr=rate,max_seconds=max_seconds),parent_checkpoint=checkpoint)
            checkpoint_name=f'checkpoint-{step:06d}.pt'
            torch.save(payload,output/checkpoint_name)
            dump_json(output/'last.json',dict(path=checkpoint_name,step=step))
            if metrics['work_macro_f1']>best:
                best=metrics['work_macro_f1'];dump_json(output/'best.json',dict(path=checkpoint_name,step=step))
    log.close()
    metrics=evaluate(model,dev,store,device,features_cache)
    metrics['endpoint_support']={k:manifest['counts'].get('dev_'+k,0) for k in ('annotated_binary_targets','retrievable_binary_targets')}
    result=dict(schema='RelationProbeRunResult@1',mode=mode,decoder=decoder,seed=seed,
        planned_updates=steps,applied_updates=completed,wall_seconds=time.monotonic()-started,
        final_dev=metrics,best_work_macro_f1=best,history=events,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        lr=rate,status='completed' if completed==steps else 'aborted',
        reason='updates_complete' if completed==steps else 'run_wall_cap',human_gold=False,
        access=dict(global_train_only=True,downstream_dev=True,corpus_validation=False,test=False))
    dump_json(output/'result.json',result)
    return result
