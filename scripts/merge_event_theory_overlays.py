#!/usr/bin/env python3
"""Materialize authenticated individual interpretations without changing bases."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from music_critic.tasks.event_theory_enrichment import (
    canonical, load_bound_piece, load_enrichment, merge_interpretations, require,
)


def binding(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path.resolve()),sha256=sha256(raw).hexdigest(),size_bytes=len(raw))


def read_bound(item):
    require(Path(item['path']).stat().st_size==item['size_bytes'],'binding size mismatch')
    return load_enrichment(item['path'],expected_sha256=item['sha256'])


def save(path,value,compressed=False):
    raw=(canonical(value)+'\n').encode()
    if compressed:raw=gzip.compress(raw,mtime=0)
    with Path(path).open('xb') as stream:stream.write(raw)
    return binding(path)


def rows(path):
    path=Path(path)
    if path.suffix=='.jsonl':return [json.loads(l) for l in path.read_text().splitlines() if l]
    return json.loads(path.read_text())['records']


def merge_indexes(base_indexes,overlay_indexes,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'pieces').mkdir();(output/'source').mkdir()
    sources=[]
    for path in (Path(__file__),Path('src/music_critic/tasks/event_theory_enrichment.py')):
        copy=output/'source'/path.name;copy.write_bytes(path.read_bytes());sources.append(binding(copy))
    save(output/'inputs.json',dict(bases=[binding(p) for p in base_indexes],
        overlays=[binding(p) for p in overlay_indexes],code=sources))
    bases={}
    for index in base_indexes:
        for row in rows(index):
            if row.get('status','completed')!='completed':continue
            b=row.get('bundle',row.get('enrichment'))
            # DLC indexes use record_id; the canonical identity comes from its bundle.
            identity=row.get('piece_id') or read_bound(b)['piece_id']
            if identity in bases:
                require(bases[identity]['canonical_piece']==row['canonical_piece'],
                        'base replacement changes canonical identity')
            bases[identity]={**row,'bundle':b,'piece_id':identity}
    seen=set();counts=Counter()
    with (output/'index.jsonl').open('x') as ledger:
        for index in overlay_indexes:
            for entry in rows(index):
                require(entry.get('status','completed')=='completed','unfinished overlay index row')
                overlay=read_bound(entry['overlay']);identity=overlay['piece_id']
                require(identity in bases and identity not in seen,'missing or duplicate overlay base')
                require(entry.get('piece_id',identity)==identity,'overlay index identity mismatch')
                seen.add(identity);base=bases[identity];bundle=read_bound(base['bundle'])
                require(bundle['canonical_piece']==base['canonical_piece'],'canonical binding mismatch')
                piece=load_bound_piece(base['canonical_piece'])
                merged=merge_interpretations(piece,bundle,overlay,base_sha256=base['bundle']['sha256'])
                c=dict(pieces=1,available_authored_fields=sum(
                    field['available'] and field['origin']=='astra_individual_context'
                    for scope in ('harmonies','note_contexts') for row in merged[scope]
                    for field in row['fields'].values()),
                    available_relations=sum(r['available'] for r in merged['relations']))
                merged['provenance']['individual_analysis']['overlay']=entry['overlay']
                merged['coverage']={**merged.get('coverage',{}),**c}
                b=save(output/'pieces'/(sha256(identity.encode()).hexdigest()[:24]+'.json.gz'),merged,True)
                row=dict(status='completed',dataset_id=piece.dataset_name,piece_id=identity,current_split='train',
                         canonical_piece=base['canonical_piece'],bundle=b,overlay=entry['overlay'],
                         base=base['bundle'],coverage=c)
                if 'record_id' in base:row['record_id']=base['record_id']
                ledger.write(canonical(row)+'\n');counts.update(c)
    summary=dict(schema='MergedEventTheorySummary@1',status='completed',**counts)
    save(output/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-index',type=Path,action='append',required=True)
    p.add_argument('--overlay-index',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment-record',type=Path,required=True)
    args=p.parse_args()
    from music_critic.experiments.registry import load_record
    record=load_record(args.experiment_record)
    require(record['experiment_id']=='EXP-MULTIGENRE-THEORY-ENRICHMENT-001'
            and record['status'] in ('planned','running'),'active registered campaign required')
    if args.output.exists():raise FileExistsError('use a fresh immutable attempt')
    try:result=merge_indexes(args.base_index,args.overlay_index,args.output)
    except Exception as exc:
        if args.output.exists():save(args.output/'failure.json',dict(status='failed',exception=type(exc).__name__,reason=str(exc)))
        raise
    print(canonical(result))


if __name__=='__main__':main()
