"""Opt-in from-scratch fusion, historical C2 schedule, bounded accelerated I/O.

This is a distinct campaign, not the 512-step continuation/resume contract.
No historical checkpoint is loaded. Validation stores bounded confusion counts;
checkpoints never contain per-event predictions or repeated validation histories.
"""
from dataclasses import asdict, fields
from pathlib import Path
from copy import deepcopy
import json
import random
import time
import gc
import numpy as np
import torch
from . import corrected_training as ref
from .contracts import fingerprint
from .corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from .full_orbit_training import FullOrbitSampler, build_full_orbit_optimizer_scheduler
from .harmonic_logit_fusion import HarmonicFusionModel
from .harmonic_fusion_checkpoint import create_model_payload as linear_model_payload
from .harmonic_fusion_data import FrozenFusionData
from .harmonic_fusion_resume import capture_safe_rng, preserve_caller_rng, restore_safe_rng
from .performance_data import FrozenProductionRecordLoader
from .performance_prefetch import (PrefetchJob, PrefetchConfig, OrderedPrefetch,
    WarmTrainBatchPreparer, CACHE_POLICY_SCHEMA)
from .performance_training import attempt_performance_update, prepare_performance_class_weights
from .performance_replay import _scaler, recursive_state_fingerprint
from .performance_c2_equivalence import raw_fingerprint

CAP = 477102080
CACHE = 8 * 1024**3
TOTAL = 15000


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False, separators=(',', ':'))
        handle.write('\n')


def emit(out, event, **values):
    value = {'event': event, 'time': time.time(), **values}
    with (out / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def paths_from_root(root):
    defaults = ref.ProductionArtifactPaths()
    repo = Path(ref.__file__).resolve().parents[4]
    return ref.ProductionArtifactPaths(**{
        f.name: Path(root) / getattr(defaults, f.name).relative_to(repo)
        for f in fields(defaults)})


def training_jobs(table, updates=TOTAL):
    if type(updates) is not int or not 1 <= updates <= TOTAL:
        raise ValueError('fusion_campaign.invalid_budget')
    sampler = FullOrbitSampler(table, seed=17)
    result = []
    for i in range(updates):
        draws = [sampler.peek(k) for k in range(2)]
        result.append(PrefetchJob(i, (2*i, 2*i+1), tuple(d.record_id for d in draws),
            tuple(d.shift_pc for d in draws), tuple(d.shift_pc for d in draws), CAP))
        sampler.advance_after_applied_update(2)
    return tuple(result)


def gate_jobs(table):
    eligible = {}
    for row in table:
        eligible.setdefault(row.record_id, set()).add(row.shift_pc)
    ids = sorted(r for r, shifts in eligible.items() if shifts == set(range(12)))[:2]
    if len(ids) != 2:
        raise ValueError('fusion_campaign.gate_all_shift_records_unavailable')
    return tuple(PrefetchJob(i, (2*i, 2*i+1), tuple(ids),
        (i, (i+5) % 12), (i, (i+5) % 12), CAP) for i in range(12))


def create_model_payload(model):
    from .grouped_mmoe import GroupedMMoEAttentionModel, create_model_payload as mmoe_payload
    if type(model) is GroupedMMoEAttentionModel:
        return mmoe_payload(model)
    from .task_attention_fusion import TaskAttentionModel, create_model_payload as attention_payload
    return attention_payload(model) if type(model) is TaskAttentionModel else linear_model_payload(model)


def components(device='cuda', model_kind='linear'):
    if model_kind not in ('linear', 'attention', 'control', 'mmoe'):
        raise ValueError('fusion_campaign.model_kind')
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    if device == 'cuda':
        torch.cuda.manual_seed_all(17)
    base = CorrectedAnalysisGNNModel().to(device)
    initial = model_state_fingerprint(base)
    if model_kind == 'linear':
        model = HarmonicFusionModel(base, enabled=True).to(device).train()
    elif model_kind == 'mmoe':
        from .grouped_mmoe import GroupedMMoEAttentionModel
        model = GroupedMMoEAttentionModel(base).to(device).train()
    else:
        from .task_attention_fusion import TaskAttentionModel
        model = TaskAttentionModel(base, enabled=model_kind == 'attention').to(device).train()
    random.seed(17001); np.random.seed(17001); torch.manual_seed(17001)
    if device == 'cuda':
        torch.cuda.manual_seed_all(17001)
    optimizer, scheduler = build_full_orbit_optimizer_scheduler(model)
    scaler, _ = _scaler()
    return model, optimizer, scheduler, scaler, initial


def make_pool(data, paths, jobs, out):
    loader = FrozenProductionRecordLoader(paths=paths)
    for rid in sorted({rid for job in jobs for rid in job.record_ids}):
        loader.check_binding(rid, split='train')
    bindings = loader.validate_all_bindings()
    write_json(out / 'train-source-bindings.json', bindings)
    preparer = WarmTrainBatchPreparer(paths, bindings, identity_only=False,
        cache_policy={'schema': CACHE_POLICY_SCHEMA, 'global_max_bytes': CACHE,
                      'preparing_processes': 4}, runtime_payload_cap_bytes=CAP)
    config = PrefetchConfig(workers=4, max_pending_batches=5, buffer_byte_cap=5*CAP,
        startup_timeout_seconds=180, job_timeout_seconds=180)
    pool = OrderedPrefetch(jobs, preparer, source_fingerprint=preparer.source_fingerprint, config=config)
    write_json(out / 'preparation.json', {'source_fingerprint': preparer.source_fingerprint,
        'config': asdict(config), 'cache': preparer.cache_policy,
        'payload_policy': 'runtime_capped_not_premeasured', 'per_job_cap_bytes': CAP,
        'schedule_fingerprint': fingerprint([asdict(j) for j in jobs])})
    return pool


def verify_gate(data, paths, out, model_kind='linear'):
    jobs = gate_jobs(data.table)
    pool = make_pool(data, paths, jobs, out)
    a = components(model_kind=model_kind); b = components(model_kind=model_kind)
    rng_a = capture_safe_rng(); rng_b = deepcopy(rng_a)
    weights = prepare_performance_class_weights(data.class_weights, 'cuda')
    rows = []
    try:
        for job in jobs:
            prepared = pool.peek()
            loaded = [data.reader(rid, split='train') for rid in job.record_ids]
            raw = ref.transpose_raw_graph_batch(ref.combine_single_record_raw_batches([x[0] for x in loaded]), job.applied_shifts)
            if raw_fingerprint(raw) != raw_fingerprint(prepared.raw_graph_batch):
                raise ValueError('fusion_campaign.gate_raw_mismatch')
            if fingerprint([x[1] for x in loaded]) != fingerprint(list(prepared.sidecars)):
                raise ValueError('fusion_campaign.gate_sidecar_mismatch')
            restore_safe_rng(rng_a, device_type='cuda')
            r = ref.attempt_applied_update(model=a[0], optimizer=a[1], scheduler=a[2], scaler=a[3],
                raw_graph_batch=ref.move_raw_graph_batch(raw, 'cuda'), sidecars=[x[1] for x in loaded],
                shifts=job.applied_shifts, class_weights=data.class_weights)
            rng_a = capture_safe_rng()
            restore_safe_rng(rng_b, device_type='cuda')
            p = attempt_performance_update(model=b[0], optimizer=b[1], scheduler=b[2], scaler=b[3],
                raw_graph_batch=ref.move_raw_graph_batch(prepared.raw_graph_batch, 'cuda'), sidecars=prepared.sidecars,
                shifts=job.applied_shifts, class_weights=weights)
            rng_b = capture_safe_rng()
            if not r.applied or not p.applied or float(r.loss.total) != float(p.loss.total):
                raise ValueError('fusion_campaign.gate_loss_mismatch')
            for x, y in [(a[0].state_dict(), b[0].state_dict()), (a[1].state_dict(), b[1].state_dict()),
                         (a[2].state_dict(), b[2].state_dict()), (rng_a, rng_b)]:
                if recursive_state_fingerprint(x) != recursive_state_fingerprint(y):
                    raise ValueError('fusion_campaign.gate_state_mismatch')
            pool.commit(job.position, applied=True)
            rows.append({'update': job.position+1, 'shifts': job.applied_shifts, 'loss': float(p.loss.total)})
            emit(out, 'equivalence', **rows[-1])
        return {'passed': True, 'model_kind': model_kind, 'applied_updates': 24, 'draws': 48, 'shift_coverage': list(range(12)),
                'workers': 4, 'rows': rows, 'initial_base_fingerprint': a[4],
                'scope': 'two TRAIN pieces, declared model kind, original versus accelerated updates; not historical-runtime equivalence'}
    finally:
        pool.close()
        write_json(out / 'queue-final.json', pool.diagnostics())


def validate(model, data, out, update):
    accumulator = ref.CorrectedValidationAccumulator(data.class_weights, train_seen_tuples=set(data.train_seen_tuples))
    with preserve_caller_rng(), torch.no_grad():
        model.eval()
        for index, rid in enumerate(data.validation_record_ids, 1):
            raw, sidecar = data.reader(rid, split='validation')
            raw = ref.move_raw_graph_batch(raw, 'cuda')
            output = model(raw)
            alignment = ref.align_target_sidecars_after_prediction(output, raw, (sidecar,), shifts=(0,))
            accumulator.update(output, alignment, sidecars=(sidecar,))
            if index % 40 == 0:
                emit(out, 'validation_progress', update=update, records=index)
        model.train()
    result = accumulator.finalize()
    result['confusions'] = {task: x.tolist() for task, x in accumulator.confusions.items()}
    result.update(update=update, evaluated_model_state_fingerprint=model_state_fingerprint(model),
                  validation_records=len(data.validation_record_ids), test_accessed=False)
    write_json(out / f'validation-{update:06d}.json', result)
    return result


def save_state(out, update, model, optimizer, scheduler, scaler, sampler, pool, config):
    # Compact immutable full state; no validation observations or duplicated histories.
    schema = ('TaskAttentionCampaignState@1.0.0' if config.get('model_kind') in ('attention', 'control')
              else 'FusionAcceleratedCampaignState@1.0.0')
    if config.get('model_kind') == 'mmoe':
        from .grouped_mmoe import STATE_SCHEMA
        schema = STATE_SCHEMA
    payload = {'schema_version': schema, 'update': update,
        'config': config, 'model': create_model_payload(model), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'sampler': sampler.state_dict(), 'prefetch': pool.state_dict(), 'rng': capture_safe_rng()}
    target = out / f'checkpoint-{update:06d}.pt'
    with target.open('xb') as handle:
        torch.save(payload, handle)
    return target


def train(data, paths, out, config):
    start_update, stop_update = 0, TOTAL
    model_kind = config.get('model_kind', 'linear')
    payload = None
    if 'resume' in config:
        from .fusion_accelerated_resume import continuation_jobs, load_resume_payload, restore_campaign_state
        start_update, stop_update = config['start_update'], config['stop_update']
        valid = ((start_update, stop_update) == (15000, 80000) if model_kind == 'linear'
                 else type(start_update) is int and 0 < start_update < stop_update == 80000)
        if not valid:
            raise ValueError('fusion_campaign.unapproved_continuation_budget')
        payload = load_resume_payload(config['resume'], expected_update=start_update,
                                      expected_data_sha256=config['data_manifest_sha256'], expected_model_kind=model_kind)
        jobs = continuation_jobs(data.table, start_update, stop_update)
    elif model_kind in ('attention', 'control', 'mmoe'):
        from .fusion_accelerated_resume import continuation_jobs
        stop_update = config['stop_update']
        if config.get('start_update') != 0 or stop_update != 80000:
            raise ValueError('fusion_campaign.unapproved_attention_budget')
        jobs = continuation_jobs(data.table, 0, stop_update)
    else:
        jobs = training_jobs(data.table)
    pool = make_pool(data, paths, jobs, out)
    model, optimizer, scheduler, scaler, initial = components(model_kind=model_kind)
    if 'expected_base_fingerprint' in config and initial != config['expected_base_fingerprint']:
        raise ValueError('fusion_campaign.base_initialization_changed')
    sampler = FullOrbitSampler(data.table, seed=17)
    weights = prepare_performance_class_weights(data.class_weights, 'cuda')
    if payload is not None:
        restored = restore_campaign_state(payload, model, optimizer, scheduler, scaler, sampler, device='cuda')
        write_json(out / 'resume-verification.json', restored)
        emit(out, 'resumed', update=start_update, next_update=start_update+1,
             learning_rate=optimizer.param_groups[0]['lr'])
    write_json(out / 'initial.json', {'base_state_fingerprint': model_state_fingerprint(model.base), 'from_scratch': payload is None,
        'full_state_fingerprint': model_state_fingerprint(model)})
    write_json(out / 'model-contract.json', create_model_payload(model)['model_contract'])
    started = time.monotonic(); validations = []; best = config.get('parent_best')
    if payload is not None and best is not None:
        if best['update'] != start_update:
            raise ValueError('fusion_campaign.parent_best_update_invalid')
    del payload
    try:
        for job in jobs:
            if model_kind == 'mmoe':
                next_update = start_update+job.position+1
                model.capture_routing = next_update <= start_update+2 or next_update % 1000 == 0
            t = time.monotonic(); prepared = pool.peek()
            if prepared.job != job:
                raise ValueError('fusion_campaign.job_mismatch')
            draws = [sampler.peek(k) for k in range(2)]
            if tuple(x.record_id for x in draws) != job.record_ids or tuple(x.shift_pc for x in draws) != job.applied_shifts:
                raise ValueError('fusion_campaign.sampler_mismatch')
            lr = optimizer.param_groups[0]['lr']
            outcome = attempt_performance_update(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                raw_graph_batch=ref.move_raw_graph_batch(prepared.raw_graph_batch, 'cuda'),
                sidecars=prepared.sidecars, shifts=job.applied_shifts, class_weights=weights)
            if not outcome.applied:
                raise ValueError('fusion_campaign.update_not_applied')
            sampler.advance_after_applied_update(2); pool.commit(job.position, applied=True)
            update = start_update+job.position+1
            if model_kind == 'mmoe' and model.capture_routing:
                emit(out, 'expert_routing', update=update, tasks=model.routing_report(),
                     scope='one_training_batch_all_raw_candidates_pre_optimizer_forward')
                model.capture_routing = False
            row = {'update': update, 'draws': [asdict(x) for x in draws], 'loss': float(outcome.loss.total),
                   'lr': lr, 'seconds': time.monotonic()-t}
            with (out / 'training_metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
            if update <= start_update+2 or update % 100 == 0:
                emit(out, 'training', update=update, loss=row['loss'], elapsed_seconds=time.monotonic()-started)
            if update % 1000 == 0:
                save_state(out, update, model, optimizer, scheduler, scaler, sampler, pool, config)
            if update % 5000 == 0:
                result = validate(model, data, out, update)
                score = result['corrected_primary_macro_score']
                validations.append({'update': update, 'primary_score': score})
                if best is None or score > best['primary_score']:
                    best = validations[-1]
                    with (out / f'best-model-{update:06d}.pt').open('xb') as handle:
                        torch.save(create_model_payload(model), handle)
                emit(out, 'validation', **validations[-1])
                gc.collect()
        with (out / 'final-model.pt').open('xb') as handle:
            torch.save(create_model_payload(model), handle)
        return {'completed': True, 'applied_updates': stop_update-start_update, 'draws': 2*(stop_update-start_update),
                'start_update': start_update, 'final_update': stop_update, 'cumulative_draws': 2*stop_update, 'validations': validations,
                'best': best, 'wall_seconds': time.monotonic()-started, 'test_accessed': False}
    finally:
        pool.close()
        write_json(out / 'queue-final.json', pool.diagnostics())


def main(argv=None):
    import argparse
    from hashlib import sha256
    from music_critic.experiments.registry import load_record
    from .harmonic_fusion_pilot import artifact_pin, verify_source_manifest
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    record = load_record(Path(config['record']))
    if record['status'] != 'planned' or record['experiment_id'] != config['experiment_id']:
        raise ValueError('fusion_campaign.planned_record_required')
    if config['mode'] not in ('gate', 'train'):
        raise ValueError('fusion_campaign.mode_invalid')
    if config['mode'] == 'train':
        gate = json.loads(Path(config['gate_report']).read_text())
        if gate.get('passed') is not True or sha256(Path(config['gate_report']).read_bytes()).hexdigest() != config['gate_sha256']:
            raise ValueError('fusion_campaign.completed_equivalence_gate_required')
        if config.get('model_kind') in ('attention', 'control', 'mmoe') and gate.get('model_kind') != config['model_kind']:
            raise ValueError('fusion_campaign.gate_model_kind_mismatch')
    # This source manifest is generated for the new campaign, never overwrites an older seal.
    for row in json.loads(Path(config['source_manifest']).read_text())['files']:
        p = Path(config['source_root']) / row['path']
        if sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('fusion_campaign.source_changed')
    manifest_path = Path(config['data_manifest'])
    if sha256(manifest_path.read_bytes()).hexdigest() != config['data_manifest_sha256']:
        raise ValueError('fusion_campaign.data_manifest_changed')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', config)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
    write_json(args.output / 'environment.json', {**ref.environment_fingerprint(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32, 'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32})
    if 'expected_environment' in config:
        from .fusion_accelerated_resume import verify_resume_environment
        verify_resume_environment(json.loads((args.output / 'environment.json').read_text()), config['expected_environment'])
    if 'resume' in config:
        from .fusion_accelerated_resume import verify_resume_environment
        verify_resume_environment(json.loads((args.output / 'environment.json').read_text()),
                                  config['parent_environment'])
    data = FrozenFusionData(json.loads(manifest_path.read_text()))
    paths = paths_from_root(config['data_root'])
    emit(args.output, 'preparing', mode=config['mode'])
    try:
        result = verify_gate(data, paths, args.output, config.get('model_kind', 'linear')) if config['mode'] == 'gate' else train(data, paths, args.output, config)
        write_json(args.output / 'report.json', result)
        emit(args.output, 'completed', mode=config['mode'])
    except BaseException as exc:
        write_json(args.output / 'failure.json', {'exception': type(exc).__name__, 'reason': str(exc)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
