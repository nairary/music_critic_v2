#!/usr/bin/env python3
"""Bounded, fail-first supervisor for the registered oracle-H membership pilot.

The supervisor owns process groups and immutable condition directories. Native
workers perform the cache/timing gate and the already-defined train_run. No
worker starts external resources or accesses corpus TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from music_critic.experiments.harmonic_event_reasoner.evidence import (
    artifact, atomic_json, binding, stamp, transition,
)
from music_critic.experiments.registry import load_record

KINDS = ('mlp', 'transe', 'self', 'graph')
SEEDS = (17, 42, 73)
GATE_ID = 'EXP-HMEMBER-NATIVE-GATE-001'
CHECKPOINT_SHA = 'd651511441907d4d559d0d3009c925107b9e371ca462db953f189ad4528e4a35'
CHECKPOINT_SIZE = 58805104
CAMPAIGN_SECONDS = 14400
GATE_SECONDS = 900
CONDITION_SECONDS = 600
WORKER_SECONDS = 590


def _verify_bytes(path, expected):
    path = Path(path)
    if not isinstance(expected, dict) or not isinstance(expected.get('size_bytes'), int):
        raise ValueError('authoritative SHA/size binding required')
    if path.stat().st_size != expected['size_bytes']:
        raise ValueError('authoritative artifact size mismatch')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected.get('sha256'):
        raise ValueError('authoritative artifact SHA256 mismatch')
    return dict(path=str(path.resolve()), sha256=expected['sha256'], size_bytes=len(data))


def _load_source_binding(value):
    if not value:
        raise ValueError('authoritative source-binding JSON is required')
    try:
        source = json.loads(value)
        base = Path.cwd()
    except json.JSONDecodeError:
        path = Path(value).resolve(strict=True)
        source = json.loads(path.read_text())
        base = path.parent
    if not isinstance(source, dict) or not isinstance(source.get('dataset_manifest'), dict):
        raise ValueError('source-binding requires dataset_manifest')
    archive = source.get('source_archive')
    if not isinstance(archive, dict) or not isinstance(archive.get('path'), str):
        raise ValueError('source-binding requires source_archive with path/SHA/size')
    path = Path(archive['path'])
    source['source_archive'] = dict(archive, path=str((base / path).resolve()))
    return source


def _verify_launch_bindings(dataset, source):
    return dict(dataset_manifest=_verify_bytes(Path(dataset) / 'manifest.json', source['dataset_manifest']),
                source_archive=_verify_bytes(source['source_archive']['path'], source['source_archive']))


def condition_id(kind, seed):
    return f'EXP-HMEMBER-S{seed}-{kind.upper()}-001'


def _claim_campaign(campaign):
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    lock = campaign / 'exclusive.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(json.dumps(dict(pid=os.getpid(), created_at=stamp())) + '\n')
    if (campaign / 'campaign.json').exists() or (campaign / 'progress.json').exists():
        raise FileExistsError('campaign already has state; restart is forbidden')
    return campaign


def _group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Permission failure never proves absence; polling/reaping the owned
        # leader below also resolves transient zombie-group races on macOS.
        return True


def _cleanup_owned(process, grace=5.0):
    """Only the process group created with start_new_session=True; ≤10s default."""
    started = time.monotonic()
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = started + grace
    while _group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()  # Reap the leader so its zombie does not keep the group alive.
        time.sleep(0.02)
    if _group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(0.01, 2 * grace - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        return dict(leader_reaped=False, cleanup_seconds=time.monotonic() - started)
    return dict(leader_reaped=True, cleanup_seconds=time.monotonic() - started)


def _run_owned(command, log, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('owned stage timeout must be positive')
    started = time.monotonic()
    with Path(log).open('xb') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        timed_out = False
        try:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                code = None
        except BaseException:
            _cleanup_owned(process)
            raise
        cleanup = _cleanup_owned(process)
    return dict(returncode=code, timed_out=timed_out, pid=process.pid,
                wall_seconds=time.monotonic() - started, **cleanup)


def _choose_steps(timings, selection_h, dev_h, *, available_seconds=WORKER_SECONDS):
    """Conservative prespecified timing rule; it never receives quality scores."""
    if set(timings) != set(KINDS) or min(selection_h, dev_h) < 1:
        raise ValueError('complete four-head timing and selection/dev support required')
    if not math.isfinite(available_seconds) or available_seconds <= 0:
        raise ValueError('invalid available wall budget')
    if any(not math.isfinite(timings[k][field]) or timings[k][field] <= 0
           for k in KINDS for field in ('update_seconds', 'forward_seconds')):
        raise ValueError('all timings must be finite and positive')
    slowest_update = max(timings[k]['update_seconds'] for k in KINDS)
    slowest_forward = max(timings[k]['forward_seconds'] for k in KINDS)
    if not all(math.isfinite(x) and x > 0 for x in (slowest_update, slowest_forward)):
        raise ValueError('timings must be finite and positive')
    # Step0 selection + final selection + final dev. 1.5x is a fixed timing
    # safety factor, not a promise that unseen graph sizes take equal time.
    evaluation_seconds = 1.5 * slowest_forward * (2 * selection_h + dev_h)
    checkpoint_seconds = 1.5 * 6 * max(timings[k].get('checkpoint_seconds', 0) for k in KINDS)
    training_seconds = min(WORKER_SECONDS, available_seconds) - 90 - evaluation_seconds - checkpoint_seconds
    steps = min(2000, max(0, math.floor(training_seconds / (1.5 * slowest_update))))
    return dict(steps=steps, accepted=steps >= 50, minimum_steps=50,
                evaluation_queries=2 * selection_h + dev_h,
                estimated_evaluation_seconds=evaluation_seconds, estimated_checkpoint_seconds=checkpoint_seconds,
                slowest_update_seconds=slowest_update, slowest_forward_seconds=slowest_forward,
                safety_factor=1.5, reserve_seconds=90, available_seconds=available_seconds)


def _check_support(support):
    """Post-candidate, post-mask scientific admission; no model score is used."""
    for split in ('train', 'selection', 'dev'):
        row = support[split]
        if row['mixed_h'] < 20 or len(row['mixed_works']) < 2 or len(row['mixed_families']) < 2:
            raise ValueError(f'{split}: need >=20 mixed H across >=2 works and >=2 independent families')


def _abort_remaining(records, experiment_ids, reason):
    for identifier in experiment_ids:
        path = Path(records) / (identifier + '.json')
        record = load_record(path)
        if record['status'] in {'planned', 'running'}:
            transition(path, 'aborted', reason=reason,
                       results=dict(skipped=True, reason=reason))


def _condition_result(output, kind, seed, steps, timed_out):
    """Keep last atomic exposure/checkpoint evidence if native timeout preempts result."""
    output = Path(output)
    result_path, progress_path = output / 'result.json', output / 'progress.json'
    path = result_path if result_path.exists() else progress_path
    if not path.exists():
        return dict(status='aborted' if timed_out else 'failed', kind=kind, seed=seed,
                    requested_updates=steps, applied_updates=None, error='condition result and progress missing')
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get('kind') != kind or result.get('seed') != seed:
        raise ValueError('condition evidence identity mismatch')
    applied = result.get('applied_updates')
    if isinstance(applied, bool) or not isinstance(applied, int) or not 0 <= applied <= steps:
        raise ValueError('condition evidence applied update count invalid')
    if result.get('requested_updates') != steps or result.get('status') not in {'running', 'completed', 'failed', 'aborted'}:
        raise ValueError('condition evidence protocol/status mismatch')
    if path == progress_path:
        result.update(status='aborted' if timed_out else 'failed', progress_fallback=True,
                      error='terminal result missing; preserved last atomic progress')
    return result


def _portable_gate_result(result):
    """Registry carries content identities, while local gate JSON keeps paths."""
    portable = json.loads(json.dumps(result))
    for value in portable.get('verified_source', {}).values():
        if isinstance(value, dict):
            value.pop('path', None)
    return portable


def _compact_result(result):
    keep = ('status', 'kind', 'seed', 'requested_updates', 'applied_updates',
            'attempted_updates', 'skipped_updates', 'query_exposure', 'known_pair_exposure',
            'unique_train_h', 'exposure_sha256', 'wall_seconds', 'timing', 'evaluation_queries',
            'trainable_parameters', 'peak_vram_bytes', 'train_loss_mean', 'train_loss_last',
            'threshold', 'threshold_source', 'error', 'error_type', 'checkpoints', 'last_checkpoint', 'progress_fallback')
    compact = {key: result[key] for key in keep if key in result}
    if 'final' in result:
        compact['final'] = {split: {key: metrics[key] for key in
            ('primary_auc', 'eligible_h', 'eligible_works', 'coverage', 'global', 'retrieval', 'controls') if key in metrics}
            for split, metrics in result['final'].items()}
    return compact


def supervise(args):
    records = Path(args.records).resolve()
    identifiers = [GATE_ID] + [condition_id(kind, seed) for seed in SEEDS for kind in KINDS]
    # Fail closed before touching the campaign or any record.
    for identifier in identifiers:
        record = load_record(records / (identifier + '.json'))
        if record['status'] != 'planned':
            raise ValueError(f'{identifier} must be a fresh planned record')
    source = _load_source_binding(args.source_binding)
    verified_source = _verify_launch_bindings(args.dataset, source)
    campaign = _claim_campaign(args.campaign)
    started = time.monotonic()
    progress = dict(status='running', phase='gate', started_at=stamp(), completed=0,
                    total=12, records=identifiers, caps=dict(campaign=CAMPAIGN_SECONDS,
                    gate=GATE_SECONDS, condition=CONDITION_SECONDS, worker=WORKER_SECONDS, cleanup=10))
    progress['source_binding'] = verified_source
    atomic_json(campaign / 'source-binding.json', source)
    atomic_json(campaign / 'campaign.json', progress)
    atomic_json(campaign / 'progress.json', progress)
    script = str(Path(__file__).resolve())
    active = None
    old_term = signal.getsignal(signal.SIGTERM)

    def interrupted(*_):
        raise KeyboardInterrupt('supervisor termination requested')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        active = GATE_ID
        transition(records / (active + '.json'), 'running')
        command = [sys.executable, script, 'gate', '--campaign', str(campaign),
                   '--dataset', str(Path(args.dataset).resolve()),
                   '--checkpoint', str(Path(args.checkpoint).resolve()),
                   '--source-binding', str(campaign / 'source-binding.json')]
        receipt = _run_owned(command, campaign / 'gate.log', GATE_SECONDS)
        gate_path = campaign / 'gate-result.json'
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else dict(status='failed', error='gate result missing')
        successful = receipt['returncode'] == 0 and receipt['leader_reaped'] and gate.get('status') == 'completed'
        transition(records / (active + '.json'), 'completed' if successful else ('aborted' if receipt['timed_out'] else 'failed'),
                   reason=None if successful else gate.get('error', 'gate failed or timed out'),
                   results=_portable_gate_result(gate), actual=receipt,
                   artifacts=[artifact(gate_path, campaign, 'gate_result')] if gate_path.exists() else None,
                   observed_train_only=gate.get('cache_status') == 'completed')
        active = None
        if not successful:
            raise RuntimeError('native gate did not pass; no training conditions launched')
        steps = gate['step_policy']['steps']
        if not gate['step_policy']['accepted'] or not 50 <= steps <= 2000:
            raise ValueError('gate returned an inadmissible common update budget')
        progress.update(phase='training', common_steps=steps, gate=gate)
        atomic_json(campaign / 'campaign.json', progress)
        for seed in SEEDS:
            for kind in KINDS:
                remaining = CAMPAIGN_SECONDS - (time.monotonic() - started)
                if remaining < CONDITION_SECONDS + 10:
                    raise TimeoutError('insufficient campaign wall budget for a full condition')
                active = condition_id(kind, seed)
                output = campaign / active
                progress.update(active=active, phase='training', wall_seconds=time.monotonic() - started)
                atomic_json(campaign / 'progress.json', progress)
                transition(records / (active + '.json'), 'running')
                command = [sys.executable, script, 'run', '--cache', str(campaign / 'cache001'),
                           '--output', str(output), '--kind', kind, '--seed', str(seed),
                           '--steps', str(steps), '--max-seconds', str(WORKER_SECONDS)]
                receipt = _run_owned(command, campaign / (active + '.log'), CONDITION_SECONDS)
                result_path = output / 'result.json'
                result = _condition_result(output, kind, seed, steps, receipt['timed_out'])
                successful = (receipt['returncode'] == 0 and receipt['leader_reaped']
                              and result.get('status') == 'completed' and result.get('applied_updates') == steps)
                status = 'completed' if successful else ('aborted' if receipt['timed_out'] or result.get('status') == 'aborted' else 'failed')
                artifacts = [artifact(result_path, campaign, 'result')] if result_path.exists() else []
                prediction_path = output / 'predictions.jsonl'
                if prediction_path.exists():
                    artifacts.append(artifact(prediction_path, campaign, 'predictions'))
                for checkpoint in result.get('checkpoints', []):
                    relative = Path(checkpoint['path'])
                    target = (output / relative).resolve()
                    if relative.is_absolute() or not target.is_relative_to(output.resolve()):
                        raise ValueError('checkpoint binding escapes run output')
                    _verify_bytes(target, checkpoint)
                    artifacts.append(artifact(target, campaign, 'head_checkpoint'))
                transition(records / (active + '.json'), status, results=_compact_result(result),
                    reason=None if successful else result.get('error', 'condition incomplete'),
                    actual=receipt, artifacts=artifacts, observed_train_only='cache_manifest' in result)
                active = None
                if not successful:
                    raise RuntimeError('condition failed; campaign stops before further comparisons')
                progress['completed'] += 1
        progress.update(status='completed', phase='completed', active=None, finished_at=stamp(),
                        wall_seconds=time.monotonic() - started)
        atomic_json(campaign / 'progress.json', progress)
        atomic_json(campaign / 'campaign.json', progress)
        return 0
    except BaseException as error:
        reason = f'{type(error).__name__}: {error}'
        if active:
            record = load_record(records / (active + '.json'))
            if record['status'] in {'planned', 'running'}:
                transition(records / (active + '.json'), 'aborted', reason=reason,
                           results=dict(interrupted=True, reason=reason))
        _abort_remaining(records, identifiers, 'campaign stopped: ' + reason)
        progress.update(status='aborted' if isinstance(error, (TimeoutError, KeyboardInterrupt)) else 'failed',
                        phase='terminal', active=None, error=reason, finished_at=stamp(),
                        wall_seconds=time.monotonic() - started)
        atomic_json(campaign / 'progress.json', progress)
        atomic_json(campaign / 'campaign.json', progress)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)


def gate(args):
    import torch
    from music_critic.experiments.harmonic_event_reasoner.cache import build_cache
    from music_critic.experiments.harmonic_event_reasoner.model import EventReadout, masked_membership_loss
    from music_critic.experiments.harmonic_event_reasoner.training import _manifest, _TileCache, _forward
    torch.set_num_threads(1)
    campaign = Path(args.campaign)
    result_path = campaign / 'gate-result.json'
    started = time.monotonic()
    result = dict(status='running', applied_updates=0, benchmark_split='train_only')
    atomic_json(result_path, result)
    try:
        source = _load_source_binding(args.source_binding)
        result['verified_source'] = _verify_launch_bindings(args.dataset, source)
        props = torch.cuda.get_device_properties(0)
        result['environment'] = dict(python=platform.python_version(), torch=torch.__version__,
            pyg=importlib.metadata.version('torch-geometric'), cuda=torch.version.cuda,
            device_name=props.name, total_vram_bytes=props.total_memory,
            torch_num_threads=torch.get_num_threads())
        cache = build_cache(args.dataset, campaign / 'cache001', args.checkpoint,
                            CHECKPOINT_SHA, CHECKPOINT_SIZE)
        result['cache_status'] = cache['status']
        result['cache_wall_seconds'] = time.monotonic() - started
        result['cache_manifest'] = binding(campaign / 'cache001' / 'manifest.json')
        _, manifest, tiles = _manifest(campaign / 'cache001')
        if cache['status'] != 'completed':
            raise ValueError('completed cache required')
        loader = _TileCache(tiles, torch, manifest.get('input_width', 266))
        support = {split: dict(total_h=0, known_h=0, mixed_h=0, mixed_works=set(), mixed_families=set())
                   for split in ('train', 'selection', 'dev')}
        seen, supervised_tiles = set(), set()
        for tile_id, entry in sorted(tiles.items()):
            tile = loader.get(tile_id)
            for query in tile['queries']:
                identity = (entry['work'], query['h_id'])
                if identity in seen:
                    raise ValueError('duplicate (work, h_id) across candidate cache')
                seen.add(identity)
                row = support[entry['split']]
                row['total_h'] += 1
                known = int(query['mask'].sum())
                positives = int(query['y'][query['mask']].sum())
                if known:
                    row['known_h'] += 1
                    supervised_tiles.add(tile_id)
                if 0 < positives < known:
                    row['mixed_h'] += 1
                    row['mixed_works'].add(entry['work'])
                    row['mixed_families'].add(entry['family'])
        for row in support.values():
            row['mixed_works'] = sorted(row['mixed_works'])
            row['mixed_families'] = sorted(row['mixed_families'])
        result['post_candidate_support'] = support
        result['support_policy'] = dict(min_mixed_h=20, min_mixed_works=2, min_mixed_families=2)
        atomic_json(result_path, result)
        _check_support(support)
        train = sorted((row for row in tiles.values() if row['split'] == 'train' and row['id'] in supervised_tiles),
                       key=lambda row: (row['size_bytes'], row['id']))
        if not train:
            raise ValueError('no train tiles for timing gate')
        positions = sorted(set([0, len(train) // 2, max(0, len(train) - 3), max(0, len(train) - 2), len(train) - 1]))
        benchmark = []
        for position in positions:
            tile = loader.get(train[position]['id'])
            supported = [(i, q) for i, q in enumerate(tile['queries']) if q['mask'].any()]
            if supported:
                index, _ = max(supported, key=lambda value: (len(value[1]['candidates']), value[1]['h_id']))
                benchmark.append((train[position]['id'], index))
        if not benchmark:
            raise ValueError('selected TRAIN timing tiles contain no supervised H')
        result['benchmark_queries'] = [dict(tile=tile, query_index=index) for tile, index in benchmark]
        timings = {}
        for kind in KINDS:
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            model = EventReadout(input_width=manifest.get('input_width', 266), kind=kind,
                node_types=manifest.get('node_types', 9), edge_types=manifest.get('edge_types', 64)).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            updates, forwards = [], []
            for tile_id, index in benchmark:
                tile = loader.get(tile_id)
                query = tile['queries'][index]
                for repeat in range(4):
                    model.train()
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    logits = _forward(model, tile, query, 'cuda')
                    loss = masked_membership_loss(logits, query['y'].cuda(), query['mask'].cuda())
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite TRAIN timing loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - begin
                    result['applied_updates'] += 1
                    if repeat:
                        updates.append(elapsed)
                    model.eval()
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        begin = time.monotonic()
                        logits = _forward(model, tile, query, 'cuda')
                        logits.sigmoid().cpu().tolist()
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - begin
                        if repeat:
                            forwards.append(elapsed)
            torch.cuda.synchronize()
            begin = time.monotonic()
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint_path = campaign / ('gate-timing-' + kind + '.pt')
            with checkpoint_path.open('xb') as handle:
                torch.save(dict(kind=kind, model_state=state, timing_only=True), handle)
                handle.flush()
                os.fsync(handle.fileno())
            checkpoint_binding = binding(checkpoint_path)
            checkpoint_seconds = time.monotonic() - begin
            timings[kind] = dict(update_seconds=max(updates), forward_seconds=max(forwards),
                                 checkpoint_seconds=checkpoint_seconds, checkpoint_binding=checkpoint_binding,
                                 timed_updates=len(updates), timed_forwards=len(forwards))
            result['timings'] = timings
            atomic_json(result_path, result)
            del model, optimizer
            torch.cuda.empty_cache()
        selection_h = support['selection']['total_h']
        dev_h = support['dev']['total_h']
        remaining_per_condition = (CAMPAIGN_SECONDS - (time.monotonic() - started) - 120) / 12
        policy = _choose_steps(timings, selection_h, dev_h,
                               available_seconds=min(WORKER_SECONDS, remaining_per_condition))
        result.update(step_policy=policy, wall_seconds=time.monotonic() - started,
                      cache_counts=cache['counts'], dev_metrics_computed=False,
                      timing_limitation='TRAIN tile timing extrapolation; supervisor still enforces native run caps')
        if not policy['accepted']:
            raise RuntimeError('full evaluation plus at least 50 matched updates cannot fit the prespecified cap')
        result['status'] = 'completed'
        atomic_json(result_path, result)
        return 0
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}',
                      wall_seconds=time.monotonic() - started)
        atomic_json(result_path, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    parent = commands.add_parser('supervise')
    for key in ('campaign', 'dataset', 'checkpoint', 'records'):
        parent.add_argument('--' + key, required=True)
    parent.add_argument('--source-binding', required=True)
    native_gate = commands.add_parser('gate')
    for key in ('campaign', 'dataset', 'checkpoint'):
        native_gate.add_argument('--' + key, required=True)
    native_gate.add_argument('--source-binding', required=True)
    worker = commands.add_parser('run')
    for key in ('cache', 'output'):
        worker.add_argument('--' + key, required=True)
    worker.add_argument('--kind', choices=KINDS, required=True)
    worker.add_argument('--seed', type=int, required=True)
    worker.add_argument('--steps', type=int, required=True)
    worker.add_argument('--max-seconds', type=float, default=WORKER_SECONDS)
    args = parser.parse_args()
    if args.command == 'supervise':
        return supervise(args)
    if args.command == 'gate':
        return gate(args)
    import torch
    torch.set_num_threads(1)
    from music_critic.experiments.harmonic_event_reasoner.training import train_run
    train_run(args.cache, args.output, args.kind, args.seed, args.steps, args.max_seconds)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
