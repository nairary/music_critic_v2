"""Exact admitted TRAIN export for the isolated oracle-H pilot."""
from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path

from music_critic.data import loads_piece, dumps_piece
from music_critic.experiments.harmonic_event_reasoner.data import extract_memberships
from music_critic.experiments.harmonic_event_reasoner.evidence import binding, atomic_json, plan, transition, artifact
from music_critic.relation_probe.data import checked_read, sanitize, validate_target_geometry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--admission-only',action='store_true')
    args=parser.parse_args();root=Path.cwd();out=args.output
    record=root/'docs/experiments/records/EXP-HMEMBER-PREPARE-001.json'
    if not out.exists():
        parent=args.source/'outputs/llm_relation_probe_20261024/preparation006'
        # These are compact metadata manifests, not corpus payloads.
        admission=json.loads(checked_read(dict(path=str(parent/'admission.json'),size_bytes=49635,
            sha256='8bbba860f0e0f7b23c522e396b7555e7e420aac496c5cdccb35ae435debfa922')))
        old=json.loads(checked_read(dict(path=str(parent/'dataset/manifest.json'),size_bytes=50222,
            sha256='777862335599d9eb2d3c75c87ec5782e51197e7299d5094408055b3262e44a50')))
        cache_path=args.source/'outputs/llm_relation_probe_20261024/monitor001/evidence/cache001/manifest.json'
        cache=json.loads(checked_read(dict(path=str(cache_path),size_bytes=452836,
            sha256='c5e82b13d6b79a6b20c40e2051096cbc6e5fa8c80d840000e6b6ff0dcab8056a')))
        ids={x['work'] for x in cache['tiles']}
        works=[w for w in old['works'] if w['id'] in ids]
        if len(works)!=64:raise ValueError('prior runtime scope differs from 64 works')
        fit_families=sorted({w['family'] for w in works if w['split']=='fit'},key=lambda s:sha256(('hmember-inner-17:'+s).encode()).hexdigest())
        selection=set(fit_families[:max(1,round(.2*len(fit_families)))])
        partitions={w['id']:('dev' if w['split']=='dev' else 'selection' if w['family'] in selection else 'train') for w in works}
        entries=[e for e in admission['entries'] if f"work{e['ordinal']:03d}" in ids]
        if len(entries)!=128:raise ValueError('flat admission pair count')
        out.mkdir(parents=True,exist_ok=False)
        obj=dict(schema='HMemberAdmission@1',entries=entries,partitions=partitions,
            scope='Exactly previous 64 runtime works; corpus TRAIN only; labels weak; raw pretraining overlap unknown.',
            parent_admission=binding(parent/'admission.json'),parent_dataset=binding(parent/'dataset/manifest.json'),
            parent_cache=binding(cache_path),inner_split='family SHA256 hmember-inner-17, lowest rounded20% of fit families')
        atomic_json(out/'admission.json',obj)
        plan(record,'EXP-HMEMBER-PREPARE-001','preparation',
            dict(admission=binding(out/'admission.json',str((out/'admission.json').relative_to(root))),
                split_counts=dict(Counter(partitions.values())),inputs='sanitized raw + oracle H geometry only',
                labels='available boolean chord_membership in exact H/view scope; full interval coverage; unknown masked'),
            dict(commit='287bf856643f9e555352e1661a7bf0563772178a',dirty=True,
                 source_overlay=binding(root/'outputs/hmember_20261025/bootstrap001/source-overlay-manifest.json')),
            parents=['EXP-LLM-RELATION-PREPARE-006'],budget=dict(max_works=64))
    if args.admission_only:
        print('admission and planned record ready; no payload decoded');return
    admission=json.loads((out/'admission.json').read_text())
    from music_critic.experiments.registry import load_record
    rec=load_record(record)
    checked_read(rec['protocol']['admission'],root)
    transition(record,'running')
    dataset=out/'dataset'
    try:
        dataset.mkdir(exist_ok=False)
        groups={}
        for e in admission['entries']:groups.setdefault(e['ordinal'],{})[e['role']]=e
        works=[];total=Counter()
        for ordinal,g in sorted(groups.items()):
            cb=checked_read(g['canonical_piece'],args.source);bb=checked_read(g['bundle'],args.source)
            bundle=json.loads(bb);piece=loads_piece(cb)
            if bundle['canonical_piece']['sha256']!=sha256(cb).hexdigest():raise ValueError('canonical binding')
            validate_target_geometry(piece,bundle)
            targets,counts=extract_memberships(bundle)
            raw=sanitize(piece);name=f'work{ordinal:03d}'
            (dataset/f'{name}.raw.json').write_text(dumps_piece(raw))
            atomic_json(dataset/f'{name}.targets.json',targets)
            works.append(dict(id=name,family=g['bundle']['work'],piece_id=piece.piece_id,
                split=admission['partitions'][name],original_global_split='train',
                raw=binding(dataset/f'{name}.raw.json'),targets=binding(dataset/f'{name}.targets.json'),
                parent_bundle_sha256=g['bundle']['sha256'],parent_canonical_sha256=g['canonical_piece']['sha256'],counts=counts))
            total.update(counts)
            print(json.dumps(dict(work=name,counts=counts)),flush=True)
        result=dict(schema='HMemberDataset@1',admission=binding(out/'admission.json'),works=works,
            counts=dict(total),human_gold=False,test_locked=True,raw_pretraining_overlap='unknown',
            scope='source analytic H/view, raw all-track candidate superset; not claimed ensemble-H supervision')
        atomic_json(dataset/'manifest.json',result)
        transition(record,'completed',results=dict(counts=dict(total),works=len(works),partitions=dict(Counter(w['split'] for w in works))),
            artifacts=[artifact(out/'admission.json',root,'admission'),artifact(dataset/'manifest.json',root,'prepared_manifest')],
            observed_train_only=True)
    except BaseException as error:
        transition(record,'failed',reason=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
