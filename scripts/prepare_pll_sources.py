#!/usr/bin/env python3
"""Prepare a registered bounded HCM TRAIN pilot from existing pinned caches."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile

from music_critic.data import loads_piece
from music_critic.pll.prepare import SourceCandidate, PrepareConfig, prepare_sources
from music_critic.tasks.corpus import load_corpus_index
from music_critic.tasks.loading import load_split_manifest


def binding(path):
    path=Path(path)
    digest=sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4<<20):digest.update(block)
    return dict(path=str(path.resolve()),sha256=digest.hexdigest(),size_bytes=path.stat().st_size)


def rank(value,seed):return sha256(f'{seed}:{value}'.encode()).hexdigest()


def candidates_from_indexes(cache,seed,limit):
    split=load_split_manifest(cache/'global.split.json')
    assignments={(a.dataset_id,a.piece_id):a for a in split.assignments}
    candidates=[]
    provenance=[binding(cache/'global.split.json')]
    indexes={source:load_corpus_index(cache/(source+'.index.json'),require_current=False)
             for source in ('hooktheory','pop909_cl')}
    forbidden_components={assignments[source,row.piece_id].component_fingerprint
        for source,index in indexes.items() for row in index.records
        if row.suggested_split not in (None,'train')}
    for source,index in indexes.items():
        if dict(split.index_fingerprints)[source]!=index.header.index_fingerprint:
            raise ValueError('source index differs from authenticated split')
        groups={}
        for row in index.records:
            assignment=assignments[source,row.piece_id]
            if (assignment.source_group_id,assignment.lineage_group_id)!=(row.source_group_id,row.lineage_group_id):
                raise ValueError('split/index group mismatch')
            groups.setdefault(assignment.component_fingerprint,[]).append((row,assignment))
        selected=[];native_heldout_components=0
        for component,rows in sorted(groups.items(),key=lambda item:rank(item[0],seed)):
            if len({a.split for _,a in rows})!=1:raise ValueError('component crosses splits')
            if rows[0][1].split!='train':continue
            # Both native split hints and project group assignment constrain
            # admission. A project hash split must never reopen upstream TEST.
            if component in forbidden_components:
                native_heldout_components+=1
                continue
            row,assignment=min(rows,key=lambda item:rank(item[0].piece_id,seed))
            path=cache/source/'multisource-cache-v1.0.0'/row.canonical_relative_path
            selected.append(SourceCandidate(row.piece_id,source,row.piece_id,row.source_group_id,
                assignment.component_fingerprint,'train',str(path.resolve()),row.canonical_sha256,path.stat().st_size))
            if len(selected)>=limit:break
        candidates.extend(selected)
        provenance.append({**binding(cache/(source+'.index.json')),
            'adapter_version':index.header.adapter_version,'current_adapter_parity_claim':False,
            'train_candidates':len(selected),'source_record_count':len(index.records),
            'native_heldout_components_excluded_before_payload':native_heldout_components})
    return candidates,provenance


def dilemma_candidates(root,stage,seed,limit):
    audit=dict(status='running',accesses=[],archives=[],target_sidecars_read=0)
    try:
        result=_dilemma_candidates(root,stage,seed,limit,audit)
        audit['status']='completed'
        return result
    except BaseException as exc:
        audit.update(status='failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        unknown=any('observed_split' not in a for a in audit['accesses'])
        audit['test_payloads_read']=None if unknown else sum(a['observed_split']=='test' for a in audit['accesses'])
        audit['validation_payloads_read']=None if unknown else sum(a['observed_split']=='validation' for a in audit['accesses'])
        audit['unverified_accesses']=sum('observed_split' not in a for a in audit['accesses'])
        stage.parent.mkdir(parents=True,exist_ok=True)
        with (stage.parent/'archive-access.json').open('x') as stream:
            json.dump(audit,stream,indent=2)


def _dilemma_candidates(root,stage,seed,limit,audit):
    manifest_path=root/'data-manifest.json';receipt_path=root/'receipt.json'
    manifest=json.loads(manifest_path.read_text());receipt=json.loads(receipt_path.read_text())
    if binding(manifest_path)['sha256']!=receipt['relocated_manifest_sha256'] or receipt['test_payload_packaged'] is not False:
        raise ValueError('Dilemma relocated manifest/receipt mismatch')
    groups={}
    for row in manifest['records']:groups.setdefault(row['source_component_id'],[]).append(row)
    selected=[]
    for component,rows in sorted(groups.items(),key=lambda item:rank(item[0],seed)):
        if len({r['split'] for r in rows})!=1:raise ValueError('Dilemma group crosses splits')
        if rows[0]['split']!='train':continue
        selected.append(min(rows,key=lambda row:rank(row['record_id'],seed)))
        if len(selected)>=limit:break
    expected={str(PurePosixPath(r['piece']['path']).relative_to(receipt['destination'])):r for r in selected}
    stage.mkdir(parents=True,exist_ok=False)
    candidates=[];access=audit['accesses'];archives=audit['archives'];seen=set()
    for part in receipt['parts']:
        if PurePosixPath(part['path']).is_absolute() or '..' in PurePosixPath(part['path']).parts or '\\' in part['path']:
            raise ValueError('unsafe archive path')
        path=root/part['path'];observed=binding(path)
        if observed['sha256']!=part['sha256'] or observed['size_bytes']!=part['size_bytes']:
            raise ValueError('Dilemma archive integrity mismatch')
        archives.append(observed)
        with tarfile.open(path,'r|gz') as archive:
            for member in archive:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or '\\' in member.name or not (member.isfile() or member.isdir()):
                    raise ValueError('unsafe archive member')
                if member.name not in expected:continue
                if member.name in seen:raise ValueError('duplicate selected archive member')
                seen.add(member.name);row=expected[member.name]
                raw=archive.extractfile(member).read()
                observed_hash=sha256(raw).hexdigest()
                access.append(dict(record_id=row['record_id'],member=member.name,archive=part['path'],
                                   declared_split='train',sha256=observed_hash,size_bytes=len(raw)))
                if observed_hash!=row['piece']['sha256'] or len(raw)!=row['piece']['size_bytes']:
                    raise ValueError('Dilemma selected piece integrity mismatch')
                piece=loads_piece(raw.decode())
                access[-1]['observed_split']=piece.split
                if piece.split not in (None,'train') or piece.source_group_id!=row['source_component_id']:
                    raise ValueError('Dilemma embedded split or group mismatch')
                path=stage/(observed_hash+'.json');path.write_bytes(raw)
                candidates.append(SourceCandidate(row['record_id'],piece.dataset_name,piece.piece_id,
                    piece.source_group_id,row['source_component_id'],'train',str(path.resolve()),observed_hash,len(raw)))
        print(json.dumps(dict(archive=part['path'],selected_dilemma=len(candidates))),flush=True)
        if len(seen)==len(expected):break
    if len(seen)!=len(expected):raise ValueError('Dilemma selected member absent')
    return candidates,[binding(manifest_path),binding(receipt_path)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--works-per-source',type=int,default=256)
    parser.add_argument('--windows-per-source',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20260928)
    parser.add_argument('--reuse-candidates',type=Path)
    parser.add_argument('--reuse-candidates-sha256')
    parser.add_argument('--cache',type=Path,default=Path('data/cache'))
    parser.add_argument('--dilemma',type=Path,default=Path('outputs/cloud_migration_20260916/prepared'))
    args=parser.parse_args()
    record=json.loads(Path(args.record).read_text())
    if record['status']!='running' or record['kind']!='data_preparation':
        raise ValueError('registered running preparation required')
    if args.works_per_source<1:raise ValueError('positive candidate cap required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.reuse_candidates:
        b=binding(args.reuse_candidates)
        if b['sha256']!=args.reuse_candidates_sha256:
            raise ValueError('reused candidate metadata changed')
        # Reuse only authenticated Dilemma staging. Re-resolve cached sources
        # through both project and native split metadata on every preparation.
        reused=[SourceCandidate(**r) for r in json.loads(args.reuse_candidates.read_text())]
        candidates,provenance=candidates_from_indexes(args.cache,args.seed,args.works_per_source)
        candidates.extend(c for c in reused if c.source=='dilemmadata' and c.split=='train')
        provenance.append({**b,'reuse_scope':'dilemmadata only; cached sources reselected from dual split metadata'})
    else:
        candidates,provenance=candidates_from_indexes(args.cache,args.seed,args.works_per_source)
        dl,bindings=dilemma_candidates(args.dilemma,output/'dilemma-train-inputs',args.seed,args.works_per_source)
        candidates+=dl;provenance+=bindings
    with (output/'candidates.json').open('x') as stream:json.dump([asdict(c) for c in candidates],stream,indent=2)
    config=PrepareConfig(max_windows_per_source=args.windows_per_source)
    report=prepare_sources(candidates,output/'data',config=config,
        source_weights={'dilemmadata':.45,'hooktheory':.25,'pop909_cl':.30},input_bindings=provenance,
        split_provenance='Preserve existing Dilemma source components and global index split components; '
        'TRAIN-only exact observed-window unions; no held-out payload access; cross-source near duplicates unknown. '
        'Fixed source bar geometry and historical cache adapters retained explicitly; no current-adapter parity claim. '
        'Dilemma identifiers such as an:test:* retain upstream names; eligibility is current frozen project split=train '
        'with matching source component and embedded split None/train, never a remapping of embedded held-out splits.')
    print(json.dumps({k:v for k,v in report.items() if k not in ('accesses','admitted','exclusions')}),flush=True)


if __name__=='__main__':main()
