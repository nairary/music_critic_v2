#!/usr/bin/env python3
"""Registered, streaming DLC TRAIN native-target enrichment, no theory inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from fractions import Fraction
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import time

from music_critic.analysis.dlc_enrichment import build_dlc_enrichment, numerical_targets
from music_critic.data import loads_piece
from music_critic.tasks.event_theory_enrichment import fraction, validate_enrichment


def binding(path, root=None):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path.relative_to(root)) if root else path.name,
                sha256=sha256(raw).hexdigest(),size_bytes=len(raw))


def read_bound(root, entry):
    p=Path(entry['path']);p=p if p.is_absolute() else Path(root)/p
    if not p.resolve().is_relative_to(Path(root).resolve()):raise ValueError('unsafe bound path')
    raw=p.read_bytes()
    if len(raw)!=entry['size_bytes'] or sha256(raw).hexdigest()!=entry['sha256']:
        raise ValueError('bound hash/size mismatch: '+str(p))
    return raw


def write_gzip_once(path, value, root=None):
    raw=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    with Path(path).open('xb') as stream:stream.write(gzip.compress(raw,compresslevel=3,mtime=0))
    return binding(path,root)


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def admit_records(rows,current):
    by_id={r['record_id']:r for r in current};heldout={r['source_component_id'] for r in current if r['split']!='train'}
    if len(by_id)!=len(current):raise ValueError('duplicate current membership')
    good=[];skip=[]
    for row in rows:
        rid=row['record_id'];r=by_id.get(rid);reason=None
        if row['split']!='train' or not r or r['split']!='train':reason='not_current_train'
        elif r['source_component_id']!=row['component_id']:reason='component_mismatch'
        elif row['component_id'] in heldout:reason='component_has_heldout'
        elif not rid.startswith('dlc:'):reason='not_dlc'
        if reason:skip.append(dict(record_id=rid,status='skipped',reason=reason))
        else:good.append(row)
    return good,skip


def parent_artifact(root,record_name,locator):
    record=json.loads((root/'docs/experiments/records'/record_name).read_text())
    if record['status']!='completed':raise ValueError('parent is not completed')
    item,=[a for a in record['artifacts'] if a['locator']==locator]
    entry=dict(path=item['locator'],sha256=item['sha256'],size_bytes=item['size_bytes'])
    return json.loads(read_bound(root,entry)),entry


def attach_t06(bundle,piece,dense):
    if dense['canonical_piece_sha256']!=bundle['canonical_piece']['sha256']:
        raise ValueError('T06 canonical hash mismatch')
    def field(value,known,evidence,origin='source_derived'):
        return dict(value=value if known else None,available=bool(known),origin=origin if known else 'unknown',
            confidence=None,evidence=evidence,unknown_reason=None if known else 'prior_T06_unknown',
            semantic_level='exact_harmonic_interval')
    def rational(value):
        return dict(num=value.numerator,den=value.denominator)
    duration=piece.duration_qn.to_fraction()
    projection=dict(method='intersection_with_canonical_piece_extent',
        canonical_extent=dict(start_qn=rational(Fraction(0)),end_qn=rational(duration)),
        clipped_harmonies=[],omitted_harmonies=[],
        interpretation='Projected intervals describe applicability to this canonical piece, not source harmony duration.')
    omitted_ids=set()
    for h in dense['harmonies']:
        start,end=fraction(h['start_qn']),fraction(h['end_qn'])
        if end<=start:raise ValueError('T06 source harmony must have a positive interval')
        left,right=max(Fraction(0),start),min(duration,end)
        if left>=right:
            omitted_ids.add(h['id'])
            projection['omitted_harmonies'].append(dict(source_harmony=deepcopy(h),
                reason='no_positive_intersection_with_canonical_piece_extent'))
            continue
        harmony=dict(id='t06:'+h['id'],start_qn=rational(left),end_qn=rational(right),
            fields={'dlc.'+k:field(v,h['availability'].get(k,False),[h['id']],'source_native')
                    for k,v in h['values'].items() if k!='unfolded_harmony_index'})
        if (left,right)!=(start,end):
            source_interval=dict(start_qn=deepcopy(h['start_qn']),end_qn=deepcopy(h['end_qn']))
            harmony.update(source_interval=source_interval,interval_projection=projection['method'])
            projection['clipped_harmonies'].append(dict(id=harmony['id'],source_interval=source_interval,
                projected_interval=dict(start_qn=rational(left),end_qn=rational(right))))
        bundle['harmonies'].append(harmony)
    for c in dense['rows']:
        if c['harmony_id'] in omitted_ids:
            raise ValueError('T06 context references harmony outside canonical piece extent')
        bundle['note_contexts'].append(dict(id='t06:'+c['row_id'],note_id=c['note_id'],
            canonical_note_index=c['note_index'],start_qn=c['start_qn'],end_qn=c['end_qn'],
            harmony_id='t06:'+c['harmony_id'] if c['harmony_id'] else None,
            applicability='exact_harmonic_interval',fields={
                'dense.'+k:{**field(f['value'],f['available'],[c['row_id']],f['origin']),
                            'unknown_reason':None if f['available'] else f['reason'],
                            'source_fields':f.get('source_fields',[])} for k,f in c['fields'].items()}))
    bundle.setdefault('provenance',{})['t06_interval_projection']=projection
    bundle['coverage']['exact_harmonic_spans']=len(bundle['harmonies'])
    bundle['coverage']['prior_T06_context_rows']=len(dense['rows'])
    bundle['coverage']['context_rows']=len(bundle['note_contexts'])
    counts=Counter();unknown=Counter()
    for context in bundle['note_contexts']:
        for name,target in context['fields'].items():
            if target['available']:counts[name]+=1
            else:unknown[target['unknown_reason']]+=1
    bundle['coverage']['available_by_field']=dict(counts)
    bundle['coverage']['unknown_by_reason']=dict(unknown)
    validate_enrichment(piece,bundle)


def astra_packet(piece,bundle,collection):
    groups=bundle['source_annotation_groups'];center=next((i for i,g in enumerate(groups)
        if g['raw_fields'].get('relativeroot') or g['raw_fields'].get('changes')),min(4,len(groups)-1))
    chosen=groups[max(0,center-3):center+9];ids={g['id'] for g in chosen}
    candidates=[c for c in bundle['note_contexts'] if c.get('source_harmony_group_id') in ids]
    contexts=candidates[:180]
    notes={n.note_id:n for n in piece.notes}
    return dict(schema='AstraNativeDLCContext@1',record_id=bundle['record_id'],collection=collection,
        piece_id=piece.piece_id,source_bindings=bundle['source_bindings'],canonical_piece=bundle['canonical_piece'],
        scope='selected source-note annotations; exact harmony boundaries not known; note spans are not chord spans',
        truncated=len(candidates)>len(contexts),candidate_event_count=len(candidates),
        annotation_groups=chosen,events=[dict(id=c['id'],note_id=c['note_id'],pitch=notes[c['note_id']].pitch,
            start_qn=c['start_qn'],end_qn=c['end_qn'],source_staff=c.get('source_staff'),source_voice=c.get('source_voice'),
            source_harmony_group_id=c.get('source_harmony_group_id'),native_note_fields={
                k:{'value':f['value'],'available':f['available']} for k,f in c['fields'].items()
                if k in ('dlc.note_degree','dlc.tpc_is_in_label','dlc.tpc_is_root','dlc.tpc_is_bass')}) for c in contexts],
        request='Individually analyze contextual harmonic functions and directed goals/preparation/prolongation. '
                'Use exact supplied note/context IDs; native annotations remain native. Do not invent score chord boundaries, '
                'unseen targets, voices, a tension number or actual note resolutions not supported by these events. '
                'Return separately attributed weak available relations and unknowns. No good/bad quality labels.')


def run(root,output,record_path,record_ids=None):
    root=root.resolve();output=output.resolve();start=time.perf_counter()
    if not output.is_relative_to(root/'outputs'):raise ValueError('output outside outputs')
    record=json.loads(record_path.read_text())
    if record['status'] not in ('planned','running') or record['experiment_id']!='EXP-MULTIGENRE-THEORY-ENRICHMENT-001':
        raise ValueError('registered enrichment campaign required')
    output.mkdir(parents=True,exist_ok=False)
    report,report_binding=parent_artifact(root,'EXP-CHANGES-ALIGNMENT-FULLTRAIN-001.json',
                                        'outputs/changes_alignment_fulltrain_001/report.json')
    artifacts,artifacts_binding=parent_artifact(root,'EXP-CHANGES-ALIGNMENT-FULLTRAIN-001.json',
                                        'outputs/changes_alignment_fulltrain_001/artifact_manifest.json')
    source_manifest=root/'outputs/changes_train_targets_002/manifest.json'
    manifest=json.loads(source_manifest.read_text()); current_path=root/'outputs/c4_pop909_20261005/dilemma-data-manifest.json'
    current=json.loads(current_path.read_text())
    requested=manifest['records']
    if record_ids is not None:
        if not set(record_ids)<={r['record_id'] for r in requested}:raise ValueError('unknown requested record ID')
        requested=[r for r in requested if r['record_id'] in set(record_ids)]
    admitted,skipped=admit_records(requested,current['records'])
    # The retained source manifest is authenticated through the completed target parent.
    source_manifest_binding=binding(source_manifest,root)
    parent_bindings=json.loads(read_bound(root,dict(path='outputs/changes_alignment_fulltrain_001/bindings.json',
                                                  **artifacts['bindings.json'])))
    expected=parent_bindings['source_target_artifacts']['manifest.json']
    expected_sha=expected['sha256'] if isinstance(expected,dict) else expected
    if source_manifest_binding['sha256']!=expected_sha:raise ValueError('source manifest parent hash mismatch')
    native_by_id={r['record_id']:r for r in report['records']};current_by_id={r['record_id']:r for r in current['records']}
    t06_index,t06_binding=parent_artifact(root,'EXP-DENSE-HARMONIC-TARGETS-001.json',
                                        'outputs/dense_harmonic_targets_20261006_001/release/index.json')
    t06={w['record_id']:w for w in t06_index['works']}
    save(output/'admission.json',dict(experiment_id=record['experiment_id'],source_manifest=source_manifest_binding,
        current_membership=binding(current_path,root),parent_report=report_binding,parent_artifacts=artifacts_binding,
        admitted_record_ids=[r['record_id'] for r in admitted],skipped=skipped,
        policy='existing source TRAIN and current TRAIN component must agree before source reads; heldout components excluded',
        family_limit='existing authenticated source component grouping; no claim of exhaustive cross-corpus title matching'))
    (output/'pieces').mkdir();(output/'astra_packets').mkdir();(output/'source').mkdir()
    for name in ('src/music_critic/analysis/dlc_enrichment.py','scripts/build_dlc_enrichment.py',
                 'src/music_critic/analysis/target_binding.py',
                 'src/music_critic/tasks/event_theory_enrichment.py'):
        p=root/name;(output/'source'/p.name).write_bytes(p.read_bytes())
    totals=Counter();available=Counter();failures=[];packet_collections=set();packet_count=0
    with (output/'index.jsonl').open('x') as ledger:
        for row in admitted:
            rid=row['record_id'];entry=dict(record_id=rid,current_split='train',source_component_id=row['component_id'])
            try:
                previous=native_by_id[rid];token=previous['directory']
                paths={name:dict(path=f'outputs/changes_alignment_fulltrain_001/{token}/{name}',
                                **artifacts[f'{token}/{name}']) for name in ('piece.json','raw_alignment.json')}
                if paths['piece.json']['sha256']!=current_by_id[rid]['piece']['sha256']:
                    raise ValueError('current canonical bytes mismatch')
                source_binding=dict(path=str(Path(manifest['root'])/row['relative_path']),sha256=row['sha256'],size_bytes=row['size_bytes'])
                source_raw=read_bound(root,source_binding)
                piece=loads_piece(read_bound(root,paths['piece.json']).decode())
                alignment=json.loads(read_bound(root,paths['raw_alignment.json']))
                rows=list(csv.DictReader(io.StringIO(source_raw.decode('utf-8-sig')),delimiter='\t'))
                conversion=next(p for p in piece.provenance if p.provenance_id=='prov:dilemmadata-conversion')
                offset=dict(conversion.details).get('source_to_canonical_offset_qn','0/1')
                bundle=build_dlc_enrichment(piece,alignment,rows,dict(record_id=rid,source_sha256=row['sha256'],
                    current_split='train',canonical_piece=paths['piece.json'],
                    source_bindings=[source_binding,paths['raw_alignment.json']],source_to_canonical_offset_qn=offset))
                if rid in t06:
                    dense=json.loads(read_bound(root,t06[rid]['dense_targets']));attach_t06(bundle,piece,dense)
                    bundle['source_bindings'].append(t06[rid]['dense_targets'])
                validate_enrichment(piece,bundle)
                numeric=numerical_targets(bundle)
                entry.update(status='completed',canonical_piece=paths['piece.json'],
                    enrichment=write_gzip_once(output/'pieces'/f'{token}.json.gz',bundle,root),
                    numeric=write_gzip_once(output/'pieces'/f'{token}.numeric.json.gz',numeric,root),coverage=bundle['coverage'])
                collection=rid.split(':')[1]
                if collection not in packet_collections and packet_count<20:
                    packet=astra_packet(piece,bundle,collection)
                    save(output/'astra_packets'/f'{token}.json',packet)
                    packet_collections.add(collection);packet_count+=1
                totals.update({k:v for k,v in bundle['coverage'].items() if type(v) is int})
                available.update(bundle['coverage']['available_by_field']);totals['completed_records']+=1
            except Exception as exc:
                entry.update(status='failed',exception=type(exc).__name__,reason=str(exc));failures.append(entry)
            ledger.write(json.dumps(entry,separators=(',',':'),ensure_ascii=False)+'\n');ledger.flush()
            if (totals['completed_records']+len(failures))%25==0:
                print(json.dumps(dict(completed=totals['completed_records'],failed=len(failures),elapsed_seconds=time.perf_counter()-start)),flush=True)
    summary=dict(schema='DLCNativeEnrichmentStage@1',experiment_id=record['experiment_id'],
        status='completed' if not failures else 'completed_with_record_failures',requested_records=len(requested),
        admitted_records=len(admitted),skipped=skipped,failures=failures,counts=dict(totals),
        available_by_field=dict(available),astra_packets=packet_count,elapsed_seconds=time.perf_counter()-start,
        data_access=dict(TRAIN=True,VALIDATION=False,TEST=False),training_updates=0,gpu_hours=0,
        claim_boundary='Native source-note annotation targets; only retained T06 has exact harmonic spans. No new semantic relations authored.')
    save(output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('failures','available_by_field')},ensure_ascii=False))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,required=True);p.add_argument('--experiment-record',type=Path,required=True)
    p.add_argument('--record-id',action='append',default=None)
    a=p.parse_args();run(a.root,a.output,a.experiment_record,a.record_id)


if __name__=='__main__':main()
