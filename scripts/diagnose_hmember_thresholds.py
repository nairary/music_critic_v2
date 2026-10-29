#!/usr/bin/env python3
"""Registered, bounded selection-only threshold repair on fixed saved heads.

The source campaign and cache are immutable. All DEV comparisons use identical
retained prediction bytes; no checkpoint is selected by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time

from music_critic.experiments.registry import load_record
from music_critic.experiments.harmonic_event_reasoner.training import (
    _Deadline, _TileCache, _manifest, _forward, _atomic_json,
)
from music_critic.experiments.harmonic_event_reasoner.metrics import (
    _auc, choose_threshold, choose_threshold_exact, choose_threshold_fpr, score_metrics,
)


def verified_bytes(root, relative, binding):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('artifact must be contained and relative')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('artifact escapes evidence root')
    data = path.read_bytes()
    if len(data) != binding['size_bytes'] or hashlib.sha256(data).hexdigest() != binding['sha256']:
        raise ValueError('artifact SHA/size mismatch')
    return data


def threshold_policies(selection):
    """Freeze all thresholds before opening the retained DEV rows."""
    policies = {
        'coarse_replay': choose_threshold(selection),
        'exact_bounded': choose_threshold_exact(selection, min_threshold=.1, max_threshold=.9),
        'exact_full': choose_threshold_exact(selection),
    }
    try:
        policies['fpr10'] = choose_threshold_fpr(selection, max_fpr=.1)
        infeasible = None
    except ValueError as error:
        policies['fpr10'] = None
        infeasible = str(error)
    return policies, infeasible


def metric_summary(metric):
    return {k: metric[k] for k in ('primary_auc', 'eligible_h', 'eligible_works',
        'coverage', 'global', 'retrieval', 'per_work', 'threshold')}


def load_dev(data):
    rows = [json.loads(line) for line in data.splitlines() if line]
    if not rows or any(row.get('split') != 'dev' for row in rows):
        raise ValueError('duration predictions must contain DEV only')
    return rows


def validate_authority(authority):
    expected={f'EXP-HMEMBER-80K-S{s}-{k}-001' for s in (17,42,73)
              for k in ('MLP','TRANSE','SELF','GRAPH')}
    conditions=authority.get('conditions',[])
    ids=[item.get('id') for item in conditions]
    if len(ids)!=12 or set(ids)!=expected:
        raise ValueError('exactly12 unique authorized80k source conditions required')


def selection_inference(model, store, bindings, torch, deadline, output, on_query):
    rows, logit_aucs, probability_aucs = [], {}, {}
    saturation = {'known_positive': 0, 'known_negative': 0,
                  'positive_at_one': 0, 'negative_at_one': 0,
                  'positive_at_zero': 0, 'negative_at_zero': 0}
    with output.open('x') as handle, torch.no_grad():
        model.eval()
        for tile_id, binding in sorted(bindings.items()):
            if binding['split'] != 'selection':
                continue
            tile = store.get(tile_id)
            for query in tile['queries']:
                deadline.check()
                logits = _forward(model, tile, query, 'cuda')
                on_query()
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite selection logits')
                scores, raw = logits.sigmoid().cpu().tolist(), logits.cpu().tolist()
                mask = query['mask'].tolist()
                labels = [int(y) if m else 0 for y, m in zip(query['y'].tolist(), mask)]
                row = dict(work=binding['work'], family=binding['family'], h_id=query['h_id'],
                           scores=scores, y=labels, mask=mask)
                rows.append(row)
                handle.write(json.dumps(dict(row, logits=raw, split='selection'), separators=(',', ':'))+'\n')
                known_y = [y for y, m in zip(labels, mask) if m]
                a = _auc([v for v, m in zip(raw, mask) if m], known_y)
                b = _auc([v for v, m in zip(scores, mask) if m], known_y)
                if a is not None:
                    logit_aucs.setdefault(binding['work'], []).append(a)
                    probability_aucs.setdefault(binding['work'], []).append(b)
                for p, y, m in zip(scores, labels, mask):
                    if m:
                        tag = 'positive' if y else 'negative'
                        saturation['known_'+tag] += 1
                        saturation[tag+'_at_one'] += p == 1.0
                        saturation[tag+'_at_zero'] += p == 0.0
    def mean_work(values):
        return math.fsum(math.fsum(v)/len(v) for v in values.values())/len(values) if values else None
    return rows, dict(saturation=saturation, raw_logit_primary_auc=mean_work(logit_aucs),
                     fp32_probability_primary_auc=mean_work(probability_aucs))


def run(cache, campaign, authority, output, max_seconds=1490):
    validate_authority(authority)
    import torch
    from music_critic.experiments.harmonic_event_reasoner.model import EventReadout
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    campaign = Path(campaign)
    started = time.monotonic(); result = dict(status='running', completed=0, conditions=[],
        new_training_updates=0, new_dev_inference_queries=0, selection_inference_queries=0)
    _atomic_json(output/'progress.json', result)
    def counted_query():
        result['selection_inference_queries'] += 1
        if result['selection_inference_queries'] % 100 == 1:
            _atomic_json(output/'progress.json', {k:v for k,v in result.items() if k!='conditions'})
    try:
        with _Deadline(max_seconds) as deadline:
            cache_path, manifest, bindings = _manifest(cache)
            if manifest['_binding'] != authority['cache_manifest']:
                raise ValueError('cache authority differs')
            for item in authority['conditions']:
                deadline.check(); eid = item['id']
                record_path = campaign/'records'/f'{eid}.json'
                verified_bytes(campaign, 'records/'+record_path.name, item['record'])
                record = load_record(record_path)
                if record['status'] != 'completed' or record['record_fingerprint'] != item['record_fingerprint']:
                    raise ValueError('unbound/nonterminal source record')
                result_artifact = next(a for a in record['artifacts'] if a['role']=='result')
                source = json.loads(verified_bytes(campaign, result_artifact['locator'], result_artifact))
                checkpoint = source['selected_checkpoint']
                weights = verified_bytes(campaign, eid+'/'+checkpoint['path'], checkpoint)
                state = torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True)
                if (state['schema'] != 'HMemberHeadCheckpoint@1' or state['kind'] != source['kind']
                        or state['seed'] != source['seed'] or state['applied_updates'] != source['selected_update']
                        or state['cache_manifest'] != manifest['_binding']
                        or state['exposure_sha256'] != checkpoint['exposure_sha256']):
                    raise ValueError('checkpoint identity differs')
                model = EventReadout(kind=state['kind'], input_width=manifest['input_width'],
                    node_types=manifest['node_types'], edge_types=manifest['edge_types']).cuda()
                model.load_state_dict(state['model_state'], strict=True)
                store = _TileCache(bindings, torch, manifest['input_width'])
                directory=output/eid;directory.mkdir()
                selection, score_diagnostic = selection_inference(model, store, bindings, torch,
                    deadline, directory/'selection-predictions.jsonl', counted_query)
                policies, infeasible = threshold_policies(selection)
                # All new policy decisions are frozen before retained DEV labels are opened.
                policies['historical'] = source['threshold']
                _atomic_json(directory/'selected-policies.json', dict(policies=policies,
                    fpr10_infeasible=infeasible, source_checkpoint=checkpoint,
                    rule='selection only; four prespecified policy variants; DEV never chooses policy'))
                prediction_artifact = next(a for a in record['artifacts'] if a['role']=='selected_predictions')
                dev = load_dev(verified_bytes(campaign, prediction_artifact['locator'], prediction_artifact))
                replay = score_metrics(dev, source['threshold'])
                if replay != source['final']['dev']:
                    raise ValueError('retained DEV metric replay differs from source result')
                summary = dict(id=eid, kind=source['kind'], seed=source['seed'],
                    selected_update=source['selected_update'], checkpoint=checkpoint,
                    prediction_binding=prediction_artifact, score_diagnostic=score_diagnostic,
                    policies=policies, fpr10_infeasible=infeasible, metrics={})
                for name, threshold in policies.items():
                    deadline.check()
                    if threshold is not None:
                        summary['metrics'][name] = dict(selection=metric_summary(score_metrics(selection, threshold)),
                                                       dev=metric_summary(score_metrics(dev, threshold)))
                summary['selection_replay_auc_delta'] = (summary['metrics']['historical']['selection']['primary_auc']
                    - source['final']['selection']['primary_auc'])
                summary['source_selection'] = metric_summary(source['final']['selection'])
                _atomic_json(directory/'result.json', summary)
                result['conditions'].append(summary); result['completed'] += 1
                result['wall_seconds'] = time.monotonic()-started
                _atomic_json(output/'progress.json', {k:v for k,v in result.items() if k!='conditions'})
                del model,store
                torch.cuda.empty_cache()
            if result['completed'] != 12:
                raise ValueError('incomplete diagnostic condition set')
            result.update(status='completed', wall_seconds=time.monotonic()-started,
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    except BaseException as error:
        result.update(status='aborted' if isinstance(error,(TimeoutError,KeyboardInterrupt)) else 'failed',
                      error=f'{type(error).__name__}: {error}', wall_seconds=time.monotonic()-started)
        raise
    finally:
        _atomic_json(output/'result.json',result)
        _atomic_json(output/'progress.json',{k:v for k,v in result.items() if k!='conditions'})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('cache','campaign','authority','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--max-seconds',type=float,default=1490)
    args=parser.parse_args()
    run(args.cache,args.campaign,json.loads(Path(args.authority).read_text()),args.output,args.max_seconds)
