#!/usr/bin/env python3
"""Freeze one TRAIN vocabulary and stream verified numerical target shards.

Indexes are explicit, supplied in precedence order. A later row can supersede
only the same record/piece; every replaced binding remains in the receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

from music_critic.tasks.event_theory_enrichment import (
    build_vocabulary, canonical, encode_enrichment, load_bound_piece,
    load_enrichment, require,
)


def binding(path):
    path=Path(path)
    raw=path.read_bytes()
    try:locator=str(path.resolve().relative_to(Path.cwd()))
    except ValueError:locator=str(path.resolve())
    return dict(path=locator,sha256=sha256(raw).hexdigest(),size_bytes=len(raw))


def save(path,value,*,compressed=False):
    raw=(canonical(value)+'\n').encode()
    if compressed:raw=gzip.compress(raw,mtime=0)
    with Path(path).open('xb') as f:f.write(raw)
    return binding(path)


def select_entries(indexes):
    selected={}; superseded=[]; excluded=[]
    for path in indexes:
        path=Path(path)
        if path.suffix=='.jsonl':
            rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        else:
            value=json.loads(path.read_text());rows=value['records']
        seen=set()
        for row in rows:
            if row.get('status','completed')!='completed':
                excluded.append(dict(index=str(path),record=row));continue
            key=row.get('record_id') or row.get('piece_id')
            require(bool(key) and key not in seen,'duplicate or missing identity within index')
            seen.add(key)
            b=row.get('bundle',row.get('enrichment'))
            require(isinstance(b,dict) and {'path','sha256','size_bytes'} <= b.keys(),'bundle binding missing')
            require('canonical_piece' in row,'canonical binding missing')
            item={**row,'bundle':b,'input_index':str(path)}
            if key in selected:
                require(selected[key]['canonical_piece']['sha256']==row['canonical_piece']['sha256'],
                        'replacement changes canonical identity')
                superseded.append(dict(identity=key,previous=selected[key]['bundle'],replacement=b))
                item={**selected[key],**item}
            selected[key]=item
    return list(selected.values()),superseded,excluded


def read_bundle(entry):
    b=entry['bundle'];path=Path(b['path'])
    require(path.stat().st_size==b['size_bytes'],'bundle size mismatch')
    value=load_enrichment(path,expected_sha256=b['sha256'])
    require(value.get('current_split')=='train','non-TRAIN bundle in release')
    require(value.get('canonical_piece')==entry['canonical_piece'],'canonical binding differs from index')
    return value


def finalize(indexes,output):
    started=time.perf_counter();output=Path(output)
    output.mkdir(parents=True,exist_ok=False);(output/'numeric').mkdir();(output/'source').mkdir()
    files=[Path(__file__),Path('src/music_critic/tasks/event_theory_enrichment.py')]
    source_bindings=[]
    for path in files:
        destination=output/'source'/path.name
        destination.write_bytes(path.read_bytes());source_bindings.append(binding(destination))
    entries,superseded,excluded=select_entries(indexes)
    receipt=dict(schema='EventTheoryReleaseInputs@1',indexes=[binding(p) for p in indexes],
                 selected=len(entries),superseded=superseded,excluded=excluded,source_code=source_bindings)
    save(output/'inputs.json',receipt)
    vocab=build_vocabulary(read_bundle(e) for e in entries)
    vocabulary_binding=save(output/'vocabulary.json',vocab)
    totals=Counter();by_dataset=defaultdict(Counter);fields=Counter();origins=Counter();relation_types=Counter()
    ids=set();index=output/'index.jsonl'; native_view_sets=Counter()
    with index.open('x') as ledger:
        for ordinal,entry in enumerate(entries,1):
            bundle=read_bundle(entry);piece=load_bound_piece(entry['canonical_piece'])
            identity=(piece.dataset_name,piece.piece_id)
            require(identity not in ids,'same canonical piece selected twice');ids.add(identity)
            native_view_sets[piece.dataset_name+':'+canonical(sorted(
                {t.annotation_view_id for t in piece.targets},key=lambda v:canonical(v)))]+=1
            numeric=encode_enrichment(piece,bundle,vocab)
            require(not numeric['oov_counts'],'TRAIN vocabulary unexpectedly missing a target')
            token=sha256(canonical(identity).encode()).hexdigest()[:24]
            artifact=save(output/'numeric'/(token+'.json.gz'),numeric,compressed=True)
            counts=dict(pieces=1,notes=len(piece.notes),context_rows=len(bundle['note_contexts']),
                        pieces_with_notes=int(bool(piece.notes)),harmony_only_pieces=int(not piece.notes),
                        harmony_spans=len(bundle['harmonies']),
                        available_relations=sum(r['available'] for r in bundle['relations']))
            for scope in ('harmonies','note_contexts'):
                for row in bundle[scope]:
                    for name,field in row['fields'].items():
                        if field['available']:
                            fields[scope+':'+name]+=1;origins[field['origin']]+=1
            for relation in bundle['relations']:
                if relation['available']:relation_types[relation['type']]+=1
            totals.update(counts);by_dataset[piece.dataset_name].update(counts)
            ledger.write(canonical(dict(dataset_id=piece.dataset_name,piece_id=piece.piece_id,
                source_group_id=piece.source_group_id,
                source_metadata={k:entry[k] for k in ('source_identity','work_family','source_component_id',
                                                     'family','analyses','record_id','admission',
                                                     'original_membership','original_split','global_membership',
                                                     'augmentation_split') if k in entry},
                input_index=entry['input_index'],
                raw_event_train_eligible=numeric['raw_event_train_eligible'],
                current_split='train',canonical_piece=entry['canonical_piece'],bundle=entry['bundle'],
                numeric=artifact,vocabulary_fingerprint=vocab['fingerprint'],coverage=counts))+'\n')
            if ordinal%100==0:
                ledger.flush();print(canonical(dict(encoded=ordinal,total=len(entries),elapsed=time.perf_counter()-started)),flush=True)
    summary=dict(schema='EventTheoryReleaseSummary@1',status='completed',**totals,
                 datasets={k:dict(v) for k,v in by_dataset.items()},available_by_field=dict(fields),
                 available_by_origin=dict(origins),relation_types=dict(relation_types),
                 native_annotation_view_sets=dict(native_view_sets),
                 vocabulary=vocabulary_binding,selected_input_indexes=[str(p) for p in indexes],
                 superseded_records=len(superseded),excluded_failed_records=len(excluded),
                 elapsed_seconds=time.perf_counter()-started,training_updates=0,
                 data_access='admitted TRAIN only; input receipt and source-stage admissions retained',
                 claim_boundary='Counts are target availability, not independent annotation decisions or accuracy.')
    save(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--experiment-record',type=Path,required=True)
    args=parser.parse_args()
    from music_critic.experiments.registry import load_record
    record=load_record(args.experiment_record)
    require(record['experiment_id']=='EXP-MULTIGENRE-THEORY-ENRICHMENT-001'
            and record['status'] in ('planned','running'),'active registered campaign required')
    if args.output.exists():
        raise FileExistsError('release output already exists; use a fresh attempt')
    try:
        summary=finalize(args.index,args.output)
    except Exception as exc:
        if args.output.exists():
            save(args.output/'failure.json',dict(status='failed',exception=type(exc).__name__,reason=str(exc)))
        raise
    print(canonical({k:v for k,v in summary.items() if k not in ('available_by_field',)}))


if __name__=='__main__':main()
