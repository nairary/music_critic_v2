"""TRAIN-only target/output collision census; no model, graph or cache mutation.

Use current converter and B3 materializer, then mirror the identity-shift
alignment's masks, vocabulary membership and raw candidate IDs. Unit tests
cross-check this projection against the production alignment function.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import time

from music_critic.adapters import dilemmadata as adapter
from music_critic.data import dumps_piece
from music_critic.experiments.analysisgnn.multitask_contract import (
    LEGACY_TARGET_SIDECAR_VERSION,
    TASK_BY_ID, get_vocabulary, materialize_target_sidecar,
)
from music_critic.experiments.analysisgnn.training_policy import (
    PRIMARY_HEADS, AUXILIARY_HEADS, PAPER_DEFINED_JOINT_COMPONENTS,
)

HEADS = (*PRIMARY_HEADS, *AUXILIARY_HEADS)
JOINT = tuple(PAPER_DEFINED_JOINT_COMPONENTS)
SPLIT_SHA = "056de0942918976981401db5017f6e567bc8949bae91ae395f0bd669adbfd51a"
VOCABS = {h: get_vocabulary(TASK_BY_ID[h].vocabulary_id).labels for h in HEADS}


def digest(path):
    with Path(path).open('rb') as stream:
        return sha256(stream.read()).hexdigest()


def save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def rational(value):
    num, den = value['num'], value['den']
    if type(num) is not int or type(den) is not int or den <= 0:
        raise ValueError('invalid exact time')
    result = Fraction(num, den)
    if result.numerator != num or result.denominator != den:
        raise ValueError('noncanonical exact time')
    return result


def candidate_sets(piece):
    return {'beat': {b.beat_id for b in piece.beats},
            'note': {n.note_id for n in piece.notes},
            'onset': {f'onset:{n.onset_qn.num}_{n.onset_qn.den}' for n in piece.notes}}


def project_targets(piece, sidecar):
    """Return every active target, including unavailable/unmapped rows."""
    if sidecar.get('schema_version') != 'analysisgnn-source-native-target-sidecar-v1':
        raise ValueError('unexpected sidecar schema')
    candidates = candidate_sets(piece)
    relation = {}
    for r in sidecar['relations']:
        if r['relation'] == 'harmonic_event_to_beat':
            if r['source_entity_id'] in relation:
                raise ValueError('duplicate harmonic mapping')
            relation[r['source_entity_id']] = r['target_entity_id']
    seen = set()
    result = {h: [] for h in HEADS}
    for e in sidecar['entities']:
        eid, level = e['canonical_entity_id'], e['entity_type']
        if eid in seen:
            raise ValueError('duplicate entity')
        seen.add(eid)
        if level not in ('harmonic_event', 'onset', 'note'):
            continue
        onset = rational(e['onset_qn'])
        node = 'beat' if level == 'harmonic_event' else level
        slot = (relation.get(eid) if level == 'harmonic_event' else
                f'onset:{onset.numerator}_{onset.denominator}' if level == 'onset' else eid)
        for head, state in e.get('targets', {}).items():
            if head not in result or TASK_BY_ID[head].prediction_level != level:
                continue
            available = state.get('available') is True and state.get('masked') is False
            label = state.get('canonical_value')
            mapped = slot in candidates[node]
            known = isinstance(label, str) and label in VOCABS[head]
            result[head].append({'entity': eid, 'slot': slot, 'onset': str(onset),
                'label': label, 'valid': available and mapped and known,
                'source_available': available, 'mapped': mapped, 'known': known,
                'masked': bool(state.get('masked', True)),
                'source_entity': e.get('source_entity_id'),
                'source_row_ordinal': e.get('source_row_ordinal'),
                'source_row_ordinals': e.get('source_row_ordinals'),
                'source_value': state.get('source_value'),
                'missing_reason': state.get('missing_reason')})
    # Match production joint's shared valid entity support, not a union of labels.
    eligible = [{r['entity']: r for r in result[h] if r['valid']} for h in JOINT]
    common = set.intersection(*(set(rows) for rows in eligible))
    result['joint'] = []
    for eid in sorted(common):
        rows = [d[eid] for d in eligible]
        if len({r['slot'] for r in rows}) != 1:
            raise ValueError('joint route mismatch')
        result['joint'].append({**rows[0], 'label': tuple(r['label'] for r in rows)})
    return result


def summarize_rows(rows):
    valid = [r for r in rows if r['valid']]
    groups = defaultdict(list)
    for r in valid:
        groups[r['slot']].append(r)
    counts = Counter(target_rows=len(rows), valid_rows=len(valid),
        source_available_rows=sum(r['source_available'] for r in rows),
        masked_rows=sum(r['masked'] for r in rows),
        alignment_failure_rows=sum(not r['mapped'] for r in rows),
        unknown_available_rows=sum(r['source_available'] and not r['known'] for r in rows),
        occupied_slots=len(groups))
    classes = defaultdict(Counter)
    examples = {}
    for slot, group in groups.items():
        labels = Counter(r['label'] for r in group)
        for label, n in labels.items():
            classes[str(label)]['valid_rows'] += n
        if len(group) > 1:
            counts['repeated_slots'] += 1
            counts['rows_in_repeated_slots'] += len(group)
        if len(labels) == 1:
            continue
        by_time = defaultdict(Counter)
        for r in group:
            by_time[r['onset']][r['label']] += 1
        beat_errors = len(group) - max(labels.values())
        exact_errors = sum(sum(c.values()) - max(c.values()) for c in by_time.values())
        counts['conflict_slots'] += 1
        counts['rows_in_conflict_slots'] += len(group)
        counts['unavoidable_errors'] += beat_errors
        counts['same_time_unavoidable_errors'] += exact_errors
        counts['temporally_recoverable_errors'] += beat_errors - exact_errors
        kind = 'same_time_present' if exact_errors else 'different_times_only'
        counts[f'{kind}_conflict_slots'] += 1
        for label, n in labels.items():
            classes[str(label)]['rows_in_conflict_slots'] += n
        # One complete example per category per piece/head; global selection later.
        if kind not in examples:
            examples[kind] = {'slot': slot, 'category': kind, 'events': group,
                              'unavoidable_errors': beat_errors,
                              'same_time_unavoidable_errors': exact_errors}
    return dict(counts), {k: dict(v) for k,v in classes.items()}, examples


def train_assignments(path):
    if digest(path) != SPLIT_SHA:
        raise ValueError('frozen split hash mismatch')
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s]
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate split record')
    components = {}
    for r in rows:
        c = r['source_component_id']
        if components.setdefault(c, r['split']) != r['split']:
            raise ValueError('cross-split component')
    train = [r for r in rows if r['split'] == 'train']
    if len(train) != 1295 or len({r['source_component_id'] for r in train}) != 1209:
        raise ValueError('frozen TRAIN inventory mismatch')
    return sorted(train, key=lambda r:r['record_id'])


def selected_source(root, assignment):
    # Gate before constructing/statting/reading any selected payload.
    if assignment['split'] != 'train':
        raise ValueError('TRAIN only')
    prefix, collection, name = assignment['record_id'].split(':', 2)
    if prefix not in ('an', 'dlc') or assignment['dialect'] != ('an_joint' if prefix=='an' else 'dlc'):
        raise ValueError('dialect mismatch')
    filename = name + ('_joint.tsv' if prefix=='an' else '.tsv')
    relative = Path('pitch_arrays') / ('AN' if prefix=='an' else 'DLC') / collection / filename
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('source outside root')
    return path, relative.as_posix(), collection, name


def bind_source(root, assignment):
    path, relative, collection, name = selected_source(root, assignment)
    source_sha = digest(path)
    dialect = assignment['dialect']
    parsed = adapter._parse_raw_file(path, dialect)
    component = assignment['source_component_id']
    record = adapter.DilemmadataCorpusRecord(
        record_id=assignment['record_id'], piece_id=adapter._piece_id(assignment['record_id'], dialect),
        dialect=dialect, path=path, relative_path=relative,
        collection=name.split('-',1)[0] if dialect=='an_joint' else collection,
        piece_name=name, suggested_split=collection if dialect=='an_joint' else None,
        physical_source_sha256=source_sha, raw_projection_sha256=parsed.raw_projection_sha256,
        raw_equivalence_id='dilemmadata-raw:'+parsed.raw_projection_sha256,
        grouping_fingerprint=parsed.grouping_fingerprint, source_group_id=component,
        lineage_group_id=component.replace('dilemmadata-component:', 'dilemmadata-lineage:'),
        source_resolution=parsed.source_resolution, score_path=None, score_relative_path=None,
        score_sha256=None, raw_issue_categories=parsed.categories,
        note_row_count=parsed.note_row_count, tie_continuation_row_count=parsed.tie_continuation_row_count,
        zero_duration_row_count=parsed.zero_duration_row_count, corpus_identity=adapter.DilemmadataCorpusIdentity(),
        record_binding_version=adapter.DILEMMADATA_RECORD_BINDING_VERSION, record_binding_sha256='')
    return adapter._bind_record(record)


def run(args):
    start=time.perf_counter()
    assignments = train_assignments(args.bindings/'metadata/split_assignments.jsonl')
    historical = json.loads((args.bindings/'source-bindings.json').read_text())
    index_path=args.bindings/'cache-bindings.jsonl'
    if digest(index_path) != historical['cache_index']['sha256']:
        raise ValueError('cache binding hash mismatch')
    cache = {r['record_id']:r for r in map(json.loads,index_path.read_text().splitlines())}
    for a in assignments:
        if any(cache[a['record_id']][k] != a[k] for k in ('split','source_component_id')):
            raise ValueError('historical membership mismatch')
    output=args.output
    output.mkdir(parents=True, exist_ok=False)
    save(output/'plan.json', {'train_records':len(assignments),'split_sha256':SPLIT_SHA,
        'cache_bindings_sha256':digest(index_path), 'args':{k:str(v) for k,v in vars(args).items()},
        'identity_only':True,'graph_model_checkpoint_execution':False,
        'test_payload_accessed':False,'validation_payload_accessed':False})
    totals={h:Counter() for h in (*HEADS,'joint')}
    by_dialect={d:{h:Counter() for h in totals} for d in ('an_joint','dlc')}
    class_counts={h:defaultdict(Counter) for h in totals}
    class_components={h:defaultdict(set) for h in totals}
    conflict_components={h:set() for h in totals}
    examples=defaultdict(list)
    failures=[];matches=Counter();completed=0
    with (output/'pieces.jsonl').open('w') as results, (output/'sources.jsonl').open('w') as sources:
        for a in assignments:
            rid=a['record_id']
            try:
                record=bind_source(args.corpus,a)
                accepted=adapter.convert_dilemmadata_record(record)
                if not isinstance(accepted,adapter.DilemmadataAccepted):
                    raise ValueError(f'conversion rejected: {accepted.categories}')
                sidecar=materialize_target_sidecar(accepted, target_sidecar_version=LEGACY_TARGET_SIDECAR_VERSION)
                sidecar_sha=sha256(json.dumps(sidecar,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                piece_sha=sha256(dumps_piece(accepted.piece).encode()).hexdigest()
                bound=cache[rid]
                matches['sidecar_equal']+=sidecar_sha==bound['sidecar']['sha256']
                matches['piece_equal']+=piece_sha==bound['raw']['sha256']
                projection=project_targets(accepted.piece,sidecar)
                piece_summary={}
                for h,rows in projection.items():
                    counts,classes,ex=summarize_rows(rows)
                    counts['pieces']=1
                    counts['pieces_with_conflicts']=int(counts.get('conflict_slots',0)>0)
                    totals[h].update(counts);by_dialect[a['dialect']][h].update(counts)
                    piece_summary[h]=counts
                    if counts['pieces_with_conflicts']:
                        conflict_components[h].add(a['source_component_id'])
                    for label,c in classes.items():
                        class_counts[h][label].update(c)
                        class_counts[h][label]['pieces']+=1
                        class_components[h][label].add(a['source_component_id'])
                    for kind,e in ex.items():
                        key=f'{a["dialect"]}/{h}/{kind}'
                        if len(examples[key])<2:
                            selected={r['entity'] for r in e['events']}
                            entities=[x for x in sidecar['entities'] if x['canonical_entity_id'] in selected]
                            detail=[{'entity':x['canonical_entity_id'],'onset_qn':x['onset_qn'],
                                'source':x.get('source_entity_id'),
                                'source_rows':x.get('source_row_ordinals'),
                                'targets':{t:s.get('canonical_value') for t,s in x['targets'].items()
                                           if s.get('available') is True and s.get('masked') is False}}
                                    for x in entities]
                            examples[key].append({'record_id':rid,'source':record.relative_path,
                                'source_sha256':record.physical_source_sha256,**e,'harmonic_details':detail})
                if digest(record.path)!=record.physical_source_sha256:
                    raise ValueError('source changed during census')
                binding={'record_id':rid,'split':'train','dialect':a['dialect'],
                    'component_id':a['source_component_id'],'source_path':record.relative_path,
                    'source_sha256':record.physical_source_sha256,'size_bytes':record.path.stat().st_size,
                    'sidecar_sha256':sidecar_sha,'historical_sidecar_sha256':bound['sidecar']['sha256'],
                    'piece_sha256':piece_sha,'historical_piece_sha256':bound['raw']['sha256']}
                sources.write(json.dumps(binding,sort_keys=True)+'\n');sources.flush()
                results.write(json.dumps({'record_id':rid,'heads':piece_summary},sort_keys=True)+'\n');results.flush()
                completed+=1
                del accepted,sidecar,projection
            except Exception as exc:
                failure={'record_id':rid,'error':type(exc).__name__,'message':str(exc)}
                failures.append(failure)
                save(output/'failures.json',failures)
                # Fail closed: no silent source exclusions and no continuing on a broken contract.
                print(json.dumps({'event':'failed',**failure}),flush=True)
                break
            if completed%25==0 or completed==1:
                print(json.dumps({'event':'progress','completed':completed,'expected':len(assignments),
                    'elapsed_seconds':time.perf_counter()-start,'historical_matches':dict(matches),
                    'root_conflict_slots':totals['root']['conflict_slots']}),flush=True)
    for h in totals:
        totals[h]['components_with_conflicts']=len(conflict_components[h])
        for label,c in class_counts[h].items():
            c['components']=len(class_components[h][label])
        for label in VOCABS.get(h,()):
            class_counts[h].setdefault(label,Counter(valid_rows=0,pieces=0,components=0))
    summary={'schema':'TargetSlotCensus@1.0.0','expected_records':len(assignments),'completed_records':completed,
        'status':'failed' if failures else 'completed','failures':failures,'split_sha256':SPLIT_SHA,
        'elapsed_seconds':time.perf_counter()-start,'historical_hash_matches':dict(matches),
        'heads':{h:dict(c) for h,c in totals.items()},
        'by_dialect':{d:{h:dict(c) for h,c in hs.items()} for d,hs in by_dialect.items()},
        'class_counts':{h:{k:dict(c) for k,c in cs.items()} for h,cs in class_counts.items()},
        'joint_components':JOINT,'counting_unit':'native TRAIN entity rows, not loss weights or augmentation draws',
        'validation_payload_accessed':False,'test_payload_accessed':False}
    save(output/'summary.json',summary);save(output/'examples.json',dict(examples))
    print(json.dumps({'event':'finished','status':summary['status'],'records':completed,
        'elapsed_seconds':summary['elapsed_seconds'],'historical_matches':dict(matches)}),flush=True)
    return 1 if failures else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--bindings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
