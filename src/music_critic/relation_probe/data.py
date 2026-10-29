"""Authenticated flat admission and weak targets, separate from raw inputs."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json


def checked_read(binding, root=Path('.')):
    path=Path(binding['path'])
    path=path if path.is_absolute() else Path(root)/path
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('symlink admission forbidden')
    if not path.is_file() or path.stat().st_size!=binding['size_bytes']:
        raise ValueError('admission size mismatch')
    payload=path.read_bytes()
    if sha256(payload).hexdigest()!=binding['sha256']:
        raise ValueError('admission hash mismatch')
    return payload


def exact(t):
    return Fraction(t['num'],t['den']) if isinstance(t,dict) else Fraction(*t)


def pair(t):
    f=exact(t) if not isinstance(t,Fraction) else t
    return [f.numerator,f.denominator]


def view(row):
    return row.get('view_id',row.get('harmony_view'))


def overlaps(a,b,c,d):
    # At an exact boundary, decline a contradictory negative conservatively.
    if a==b:return c<=a<=d
    if c==d:return a<=c<=b
    return a<d and c<b


def family_partition(names, seed=17):
    names=sorted(set(names),key=lambda n:sha256(f'{seed}:{n}'.encode()).hexdigest())
    n=max(1,round(len(names)*.2))
    return {name:('dev' if i<n else 'fit') for i,name in enumerate(names)}


def validate_target_geometry(piece,bundle):
    """Validate our consumer's identity/exact endpoint contract without PyG imports."""
    if bundle.get('current_split')!='train' or piece.split not in (None,'train'):
        raise ValueError('TRAIN canonical/bundle required')
    if (piece.piece_id,piece.dataset_name)!=(bundle.get('piece_id'),bundle.get('dataset_id')):
        raise ValueError('bundle identity mismatch')
    geometry={n.note_id:(n.onset_qn.to_fraction(),(n.onset_qn+n.duration_qn).to_fraction()) for n in piece.notes}
    note_geometry=dict(geometry);duration=piece.duration_qn.to_fraction()
    for kind in ('harmonies','note_contexts','objects'):
        for row in bundle.get(kind,[]):
            identity=row['id'];left,right=exact(row['start_qn']),exact(row['end_qn'])
            if identity in geometry or not 0<=left<=right<=duration:raise ValueError('invalid target geometry/identity')
            if kind=='note_contexts':
                a,b=note_geometry[row['note_id']]
                if not a<=left<=right<=b:raise ValueError('context outside canonical note')
            geometry[identity]=(left,right)
    seen=set(geometry)
    for rel in bundle.get('relations',[]):
        if rel['id'] in seen:raise ValueError('duplicate relation identity')
        seen.add(rel['id']);left,right=exact(rel['start_qn']),exact(rel['end_qn'])
        if not 0<=left<=right<=duration:raise ValueError('relation scope outside piece')
        for side in ('source_ids','target_ids'):
            if not rel[side] or len(set(rel[side]))!=len(rel[side]):raise ValueError('invalid relation participants')
            for identity in rel[side]:
                a,b=geometry[identity]
                if not left<=a<=b<=right:raise ValueError('relation outside participant geometry')


def extract_targets(bundle, canonical_notes=None):
    if bundle.get('current_split')!='train':raise ValueError('TRAIN bundle required')
    contexts={c['id']:c for c in bundle['note_contexts']}
    notes=canonical_notes or {}
    relations={r['id']:r for r in bundle['relations']}
    rows=[];counts=Counter()
    for c in contexts.values():
        f=c['fields'].get('analysis.resolution',{})
        if not f.get('available') or f.get('annotation_state') in ('conflicting','unresolved','unreviewed'):
            counts['unknown']+=1;continue
        value=f.get('value',{})
        if not isinstance(value,dict) or value.get('status') not in ('linked','none'):
            counts['unsupported_value']+=1;continue
        label=int(value['status']=='linked')
        row=dict(context_id=c['id'],note_id=c['note_id'],view_id=view(c),
                 time=pair(c['start_qn']),end=pair(c['end_qn']),label=label,targets=[],
                 origin=f.get('origin'),confidence=f.get('confidence'))
        grounded=False
        for rid in value.get('relation_ids',[]):
            r=relations.get(rid)
            if not r or not r.get('available') or r.get('type')!='resolves_to' or r.get('value') is not True:
                counts['unavailable_relation']+=1;continue
            if view(r)!=view(c) or not set(r['source_ids']) & {c['id'],c['note_id']}:
                counts['unbound_relation']+=1;continue
            if len(r['source_ids'])!=1 or len(r['target_ids'])!=1:
                counts['nonbinary_relation']+=1;grounded=True;continue
            dest=r['target_ids'][0]
            if dest in contexts:
                t=contexts[dest]
                if view(t)!=view(c):counts['cross_view']+=1;continue
                endpoint=dict(note_id=t['note_id'],time=pair(t['start_qn']))
            elif dest in notes:
                endpoint=dict(note_id=dest,time=pair(notes[dest]['onset_qn']))
            else:counts['unsupported_endpoint']+=1;continue
            if exact(endpoint['time'])<exact(c['start_qn']) or dest==c['id']:
                counts['invalid_direction']+=1;continue
            grounded=True
            if endpoint not in row['targets']:row['targets'].append(endpoint)
        if label and not grounded:
            counts['ungrounded_linked_field']+=1;continue
        if not label:
            # A positive direct edge contradicting a none field is never used as negative.
            contradicted=any(r.get('available') and r.get('value') is True
                and r.get('type')=='resolves_to' and view(r)==view(c)
                and (c['id'] in r['source_ids'] or (c['note_id'] in r['source_ids']
                and overlaps(exact(r.get('start_qn',c['start_qn'])),exact(r.get('end_qn',c['end_qn'])),
                             exact(c['start_qn']),exact(c['end_qn'])))) for r in relations.values())
            if contradicted:counts['contradictory_none']+=1;continue
        counts['positive_contexts' if label else 'reviewed_none']+=1
        counts['binary_endpoints']+=len(row['targets'])
        rows.append(row)
    return rows,dict(counts)


def sanitize(piece):
    """Keep identity for alignment only; remove all semantic/source-only inputs."""
    from dataclasses import replace
    from music_critic.data import PieceMetadata,ProvenanceRecord
    tracks={t.track_id:f'track:{i}' for i,t in enumerate(piece.tracks)}
    return replace(piece,dataset_name='relation_probe',source_group_id='relation_probe',
        split=None,source_path=None,source_resolution=None,
        metadata=PieceMetadata('synthetic',None,None,None,None,None,None,None,None),
        notes=tuple(replace(n,track_id=tracks[n.track_id],spelling_step=None,spelling_alter=None,
            staff=None,voice=None,articulations=None,dynamic=None,source_onset_ticks=None,
            source_duration_ticks=None,source_onset_seconds=None,source_duration_seconds=None,
            provenance_id=None) for n in piece.notes),
        tracks=tuple(replace(t,track_id=tracks[t.track_id],source_track_index=i,name=None,
            instrument_name=None,provenance_id=None) for i,t in enumerate(piece.tracks)),
        bars=tuple(replace(b,display_number=None,provenance_id=None) for b in piece.bars),
        beats=tuple(replace(b,provenance_id=None) for b in piece.beats),
        meter_events=tuple(replace(m,provenance_id=None) for m in piece.meter_events),
        tempo_events=tuple(replace(t,provenance_id=None) for t in piece.tempo_events),
        key_signature_events=(),annotations=(),targets=(),
        provenance=(ProvenanceRecord('prov:relation_probe','derivation','music_critic.relation_probe',
                    None,None,None,None,None,(),()),),quality_flags=())


def prepare(admission_path, output, root=Path('.')):
    """Read exactly admitted files; export new raw/target files without mutations."""
    from music_critic.data import loads_piece,dumps_piece,CanonicalValidationError
    admission=json.loads(Path(admission_path).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    groups={}
    for e in admission['entries']:groups.setdefault(e['ordinal'],{})[e['role']]=e
    split=family_partition([g['bundle']['work'] for g in groups.values()])
    manifest=[];total=Counter();excluded=[]
    for ordinal,g in sorted(groups.items()):
        cb=checked_read(g['canonical_piece'],root);bb=checked_read(g['bundle'],root)
        cdict=json.loads(cb);bundle=json.loads(bb)
        if bundle['canonical_piece']['sha256']!=sha256(cb).hexdigest():
            raise ValueError('bundle/canonical identity mismatch')
        piece=loads_piece(cb)
        if bundle['piece_id']!=piece.piece_id:raise ValueError('piece identity mismatch')
        validate_target_geometry(piece,bundle)
        targets,counts=extract_targets(bundle,{n['note_id']:n for n in cdict['notes']})
        raw=sanitize(piece);name=f'work{ordinal:03d}'
        try:
            rawbytes=dumps_piece(raw).encode()
        except CanonicalValidationError as error:
            excluded.append(dict(id=name,family=g['bundle']['work'],reason=str(error),
                                 codes=[i.code for i in error.report.issues if i.severity=='error']))
            print(json.dumps(dict(excluded=name,reason='raw_geometry_needs_source_repair_provenance')),flush=True)
            continue
        targetbytes=json.dumps(targets,sort_keys=True).encode()
        rawpath=output/f'{name}.raw.json';rawpath.write_bytes(rawbytes)
        labelpath=output/f'{name}.targets.json';labelpath.write_bytes(targetbytes)
        row=dict(id=name,family=g['bundle']['work'],piece_id=piece.piece_id,
            split=split[g['bundle']['work']],original_global_split='train',
            raw=dict(path=rawpath.name,sha256=sha256(rawbytes).hexdigest(),size_bytes=len(rawbytes)),
            targets=dict(path=labelpath.name,sha256=sha256(targetbytes).hexdigest(),size_bytes=len(targetbytes)),
            parent_bundle_sha256=g['bundle']['sha256'],parent_canonical_sha256=g['canonical_piece']['sha256'],counts=counts)
        manifest.append(row);total.update(counts)
        print(json.dumps(dict(prepared=ordinal+1,total=len(groups),counts=counts)),flush=True)
    result=dict(schema='RelationProbeDataset@1',admission_sha256=sha256(Path(admission_path).read_bytes()).hexdigest(),
        works=manifest,excluded=excluded,counts=dict(total),test_locked=True,human_gold=False,split_seed=17)
    (output/'manifest.json').write_text(json.dumps(result,sort_keys=True)+'\n')
    return result
