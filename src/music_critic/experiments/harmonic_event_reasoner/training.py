"""Bounded training of query readouts over authenticated frozen-C3 caches.

Only locally produced, trusted cache files may be passed here: PyTorch pickle
loading is intentional after SHA/size binding, not a sandbox for unknown files.
No data split or checkpoint is selected using exploratory dev metrics.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from contextlib import AbstractContextManager
import hashlib
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import re
import signal
import threading
import time
from typing import Any

from .metrics import choose_threshold, choose_threshold_exact, score_metrics
from .reproducibility import deterministic_runtime

__all__ = ["train_run"]

CHECKPOINT_UPDATES = (50, 200, 500, 1000, 2000)


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


class _Deadline(AbstractContextManager):
    """Unix main-thread alarm plus checks; enclosing supervisor owns hard kill."""
    def __init__(self, seconds: float):
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('max_seconds must be finite and positive')
        self.seconds = seconds
        self.started = time.monotonic()
        self.old_handler = None

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise TimeoutError('run wall deadline exceeded')

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
            raise RuntimeError('bounded training requires Unix main-thread execution')
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            raise RuntimeError('nested wall timers are unsupported')
        self.old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, self.seconds - (time.monotonic() - self.started)))
        return self

    def _alarm(self, *_):
        raise TimeoutError('run wall deadline exceeded')

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old_handler)


class _WorkSampler:
    """A private RNG makes the exact work/H schedule independent of model RNG."""
    def __init__(self, queries: Mapping[str, list[tuple[str, int]]], seed: int):
        if not queries or any(not values for values in queries.values()):
            raise ValueError('sampler requires supported queries in every work')
        self.queries = {work: sorted(values) for work, values in sorted(queries.items())}
        self.works = list(self.queries)
        self.rng = random.Random(seed)

    def next(self) -> tuple[str, str, int]:
        work = self.rng.choice(self.works)
        tile, index = self.rng.choice(self.queries[work])
        return work, tile, index


def _manifest(cache: str | Path) -> tuple[Path, dict, dict[str, dict]]:
    path = Path(cache)
    if path.is_dir():
        path = path / 'manifest.json'
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('input_width', 266) != 266:
        raise ValueError('H/member raw-duration feature contract requires input_width=266')
    if (not isinstance(manifest, dict) or manifest.get('schema') != 'HMemberCache@1'
            or manifest.get('status') != 'completed'
            or manifest.get('mode') != 'oracle_H_geometry_frozen_raw_C3'
            or manifest.get('test_locked') is not True
            or manifest.get('human_gold') is not False):
        raise ValueError('only completed HMemberCache@1 oracle raw-C3 weak-label TRAIN caches are admitted')
    manifest['_binding'] = dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    tiles = manifest.get('tiles')
    if not isinstance(tiles, list) or not tiles:
        raise ValueError('cache manifest must contain a nonempty tiles list')
    by_id, work_split, family_split, work_family = {}, {}, {}, {}
    paths = set()
    for value in tiles:
        if not isinstance(value, dict):
            raise ValueError('tile bindings must be mappings')
        value = dict(value)
        for field in ('id', 'work', 'family', 'split', 'path', 'sha256'):
            if not isinstance(value.get(field), str) or not value[field].strip():
                raise ValueError(f'tile binding requires nonempty {field}')
        if value['id'] in by_id:
            raise ValueError('duplicate tile id')
        if value['split'] not in {'train', 'selection', 'dev'}:
            raise ValueError('only internal train/selection/dev splits are permitted')
        if work_family.setdefault(value['work'], value['family']) != value['family']:
            raise ValueError('work belongs to conflicting families')
        for group, key in ((work_split, 'work'), (family_split, 'family')):
            if group.setdefault(value[key], value['split']) != value['split']:
                raise ValueError(f'{key} crosses data splits')
        relative = Path(value['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('cache tile path must be contained and relative')
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(path.parent) or resolved in paths:
            raise ValueError('cache tile path escapes cache or is duplicated')
        paths.add(resolved)
        if not re.fullmatch(r'[0-9a-f]{64}', value['sha256']):
            raise ValueError('invalid tile SHA256')
        size = value.get('size_bytes', value.get('bytes'))
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('tile size_bytes must be a positive integer')
        value.update(size_bytes=size, resolved_path=resolved)
        by_id[value['id']] = value
    if set(work_split.values()) != {'train', 'selection', 'dev'}:
        raise ValueError('train, selection and dev must all be represented')
    return path, manifest, by_id


def _query(q, node_count, torch):
    if not isinstance(q, dict) or not isinstance(q.get('h_id'), str) or not q['h_id']:
        raise ValueError('query must have a nonempty h_id')
    source = q.get('source')
    if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < node_count:
        raise ValueError('query source must be a valid integer node index')
    candidates = torch.as_tensor(q.get('candidates'))
    mask = torch.as_tensor(q.get('mask'))
    if candidates.numel() == 0 and isinstance(q.get('candidates'), list):
        candidates = candidates.long()
    if mask.numel() == 0 and isinstance(q.get('mask'), list):
        mask = mask.bool()
    labels = torch.as_tensor(q.get('y'))
    if candidates.ndim != 1 or candidates.dtype != torch.long:
        raise ValueError('query candidates must be a long vector')
    if mask.dtype != torch.bool or mask.shape != candidates.shape or labels.shape != candidates.shape:
        raise ValueError('query masks must be boolean and shapes aligned')
    if candidates.numel() and ((candidates < 0).any() or (candidates >= node_count).any()):
        raise ValueError('query candidate out of range')
    if candidates.unique().numel() != candidates.numel():
        raise ValueError('query candidate indices must be unique')
    known = labels[mask]
    if not torch.isfinite(known).all() or not ((known == 0) | (known == 1)).all():
        raise ValueError('known query labels must be finite binary values')
    return dict(h_id=q['h_id'], source=source, candidates=candidates.long(),
                y=labels.float(), mask=mask)


class _TileCache:
    def __init__(self, bindings, torch, input_width, *, max_items=2, max_bytes=512 * 1024**2):
        self.bindings, self.torch, self.input_width = bindings, torch, input_width
        self.max_items, self.max_bytes = max_items, max_bytes
        self.items = OrderedDict()
        self.resident_bytes = self.peak_bytes = 0

    def get(self, tile_id):
        if tile_id in self.items:
            self.items.move_to_end(tile_id)
            return self.items[tile_id][0]
        binding = self.bindings[tile_id]
        if binding['size_bytes'] > self.max_bytes:
            raise ValueError('tile exceeds bounded CPU cache capacity')
        data = binding['resolved_path'].read_bytes()
        if len(data) != binding['size_bytes'] or hashlib.sha256(data).hexdigest() != binding['sha256']:
            raise ValueError('cache tile SHA/size mismatch')
        # Verified bytes, not a second file read: avoid check/load TOCTOU.
        tile = self.torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        if not isinstance(tile, dict):
            raise ValueError('cache tile must be a mapping')
        for field in ('x', 'node_type', 'edge_index', 'edge_type'):
            if not isinstance(tile.get(field), self.torch.Tensor):
                raise ValueError(f'cache tile requires tensor {field}')
        x = tile['x']
        if x.ndim != 2 or x.shape[1] != self.input_width or not x.is_floating_point():
            raise ValueError('cache feature shape/type mismatch')
        if not self.torch.isfinite(x).all():
            raise ValueError('nonfinite cached features')
        tile['x'] = x.float().detach()
        if tile['node_type'].shape != (len(x),) or tile['node_type'].dtype != self.torch.long:
            raise ValueError('cache node type shape/type mismatch')
        edges, types = tile['edge_index'], tile['edge_type']
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != self.torch.long:
            raise ValueError('cache edge index shape/type mismatch')
        if types.shape != (edges.shape[1],) or types.dtype != self.torch.long:
            raise ValueError('cache edge type shape/type mismatch')
        if not isinstance(tile.get('queries'), list):
            raise ValueError('cache tile requires query list')
        tile['queries'] = [_query(q, len(x), self.torch) for q in tile['queries']]
        size = sum(tile[field].numel() * tile[field].element_size()
                   for field in ('x', 'node_type', 'edge_index', 'edge_type'))
        size += sum(q[field].numel() * q[field].element_size()
                    for q in tile['queries'] for field in ('candidates', 'y', 'mask'))
        if size > self.max_bytes:
            raise ValueError('decoded tile exceeds bounded CPU cache capacity')
        while self.items and (len(self.items) >= self.max_items or self.resident_bytes + size > self.max_bytes):
            _, (_, old_size) = self.items.popitem(last=False)
            self.resident_bytes -= old_size
        self.items[tile_id] = tile, size
        self.resident_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
        return tile


def _forward(model, tile, query, device):
    return model(*(tile[key].to(device) for key in ('x', 'node_type', 'edge_index', 'edge_type')),
                 query['source'], query['candidates'].to(device))


def _evaluate(model, tile_cache, bindings, split, device, torch, deadline, *, retain_logits=False):
    rows = []
    model.eval()
    with torch.no_grad():
        for tile_id, binding in sorted(bindings.items()):
            if binding['split'] != split:
                continue
            deadline.check()
            tile = tile_cache.get(tile_id)
            for query in tile['queries']:
                deadline.check()
                logits = _forward(model, tile, query, device)
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite evaluation logits')
                scores = logits.sigmoid().cpu().tolist()
                mask = query['mask'].tolist()
                labels = [int(y) if known else 0 for y, known in zip(query['y'].tolist(), mask)]
                row = dict(work=binding['work'], family=binding['family'],
                    h_id=query['h_id'], scores=scores, y=labels, mask=mask,
                    duration_scores=tile['x'][query['candidates'], 259].clamp(0, 1).tolist())
                if retain_logits:
                    row['logits'] = logits.cpu().tolist()
                rows.append(row)
    deadline.check()
    return rows


def _save_checkpoint(output, model, torch, progress, config, cache_binding, exposure_sha256,
                     *, reason):
    """Atomically publish head weights once; no optimizer/resume claim.

    A final update which coincides with a fixed checkpoint reuses its reference
    only after byte binding and state equality are verified. Existing files
    never get overwritten. Timeout cleanup is bounded by the caller/supervisor.
    """
    updates = progress['applied_updates']
    if updates <= 0:
        raise ValueError('trained head checkpoint requires an applied update')
    name = f'checkpoint-{updates:06d}.pt'
    path = output / name
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = dict(schema='HMemberHeadCheckpoint@1', kind=progress['kind'], seed=progress['seed'],
        applied_updates=updates, config=dict(config), model_state=state,
        cache_manifest=dict(cache_binding), exposure_sha256=exposure_sha256,
        known_pair_exposure=progress['known_pair_exposure'], query_exposure=progress['query_exposure'],
        reason=reason, resumable=False,
        claim='Weak source/LLM membership agreement; oracle H geometry, frozen raw C3; not human gold or raw H discovery.')
    if path.exists():
        previous = next((item for item in progress['checkpoints'] if item['path'] == name), None)
        raw = path.read_bytes()
        if (previous is None or previous['sha256'] != hashlib.sha256(raw).hexdigest()
                or previous['size_bytes'] != len(raw)):
            raise ValueError('existing checkpoint is not the recorded immutable artifact')
        old = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        for field in ('schema', 'kind', 'seed', 'applied_updates', 'config', 'cache_manifest',
                      'exposure_sha256', 'known_pair_exposure', 'query_exposure'):
            if old.get(field) != payload[field]:
                raise ValueError('existing checkpoint metadata differs at the same update')
        if (old['model_state'].keys() != state.keys()
                or any(not torch.equal(old['model_state'][key], value) for key, value in state.items())):
            raise ValueError('existing checkpoint weights differ at the same update')
        progress['last_checkpoint'] = previous
        return previous
    temp = output / ('.' + name + '.tmp')
    try:
        with temp.open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses to replace an existing path.
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    raw = path.read_bytes()
    artifact = dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
                    applied_updates=updates, exposure_sha256=exposure_sha256)
    progress['checkpoints'].append(artifact)
    progress['last_checkpoint'] = artifact
    return artifact


def _selection_schedule(values, steps):
    values=tuple(values)
    if values and (any(type(v) is not int or not 0 < v <= steps for v in values)
                   or tuple(sorted(set(values))) != values or values[-1] != steps):
        raise ValueError('selection updates must be sorted unique positive integers including final step')
    return values


def _select_curve_point(points):
    if not points or any(not isinstance(p['selection'].get('primary_auc'), (int,float))
            or not math.isfinite(p['selection']['primary_auc']) for p in points):
        raise ValueError('supported finite selection AUROC required at every checkpoint')
    return max(points, key=lambda p:(p['selection']['primary_auc'], -p['update']))


def _restore_selected(output, selected, model, torch, progress, manifest):
    reference=selected['checkpoint'];raw=(output/reference['path']).read_bytes()
    if len(raw)!=reference['size_bytes'] or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('selected checkpoint SHA/size mismatch')
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if (state.get('schema')!='HMemberHeadCheckpoint@1' or state.get('kind')!=progress['kind']
        or state.get('seed')!=progress['seed'] or state.get('applied_updates')!=selected['update']
        or state.get('cache_manifest')!=manifest['_binding']
        or state.get('exposure_sha256')!=selected['exposure_sha256']):
        raise ValueError('selected checkpoint protocol identity mismatch')
    model.load_state_dict(state['model_state'],strict=True)


def train_run(cache, output, kind, seed, steps, max_seconds=600, device='cuda',
              *, selection_updates=(), deterministic=False, class_weights=None,
              threshold_policy='legacy', retain_logits=False) -> dict:
    """Run one immutable condition; raise on failure with terminal JSON evidence.

    Fixed policy: 64d, three rounds, Adam 1e-3, gradient norm cap 1. Step0 uses
    full selection only. By default dev uses final fixed-update weights. With
    selection_updates, one continuous optimizer reaches the requested step,
    then highest selection-AUROC weights (earliest tie) receive the only dev
    evaluation. Thresholds use selection labels alone. A Unix main-thread
    timer includes loading/evaluation; the external supervisor must enforce a
    process-level kill for a stalled native kernel. Fixed/final head weights are
    retained without best-checkpoint selection or a continuation guarantee.
    Deterministic kernels are opt-in; strict CUDA requires a launch-provided
    CUBLAS_WORKSPACE_CONFIG. Curve selection rows are retained from the original
    winning evaluation, with no extra selection or dev forwards. Optional
    class weights use known-count BCE normalization. Exact thresholds change
    only the operating point; checkpoint selection remains selection AUROC.
    Optional raw logits are retained beside probabilities for diagnostics only;
    ranking and selection metrics continue to use the same probabilities.
    """
    if kind not in {'mlp', 'transe', 'self', 'graph'}:
        raise ValueError('unsupported readout kind')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    if class_weights is not None:
        if (not isinstance(class_weights, (tuple, list)) or len(class_weights) != 2
                or any(isinstance(w, bool) or not isinstance(w, Real)
                       or not math.isfinite(w) or w <= 0 for w in class_weights)):
            raise ValueError('class_weights must be two finite positive numbers (negative, positive)')
        class_weights = tuple(float(w) for w in class_weights)
    if threshold_policy not in ('legacy', 'exact'):
        raise ValueError('threshold_policy must be legacy or exact')
    select_threshold = choose_threshold if threshold_policy == 'legacy' else choose_threshold_exact
    if type(retain_logits) is not bool:
        raise ValueError('retain_logits must be boolean')
    evaluation_options = {'retain_logits': True} if retain_logits else {}
    if type(deterministic) is not bool:
        raise ValueError('deterministic must be boolean')
    selection_updates=_selection_schedule(selection_updates,steps)
    deadline = _Deadline(max_seconds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    progress = dict(status='running', phase='initializing', kind=kind, seed=seed,
                    requested_updates=steps, applied_updates=0, attempted_updates=0,
                    skipped_updates=0, known_pair_exposure=0, query_exposure=0,
                    checkpoints=[], last_checkpoint=None,
                    gradient_clipping_count=0, gradient_clipping_fraction=0.0)
    config = dict(hidden=64, rounds=3, learning_rate=1e-3, optimizer='Adam',
                  class_weights=list(class_weights) if class_weights is not None else None,
                  threshold_policy=threshold_policy, retain_logits=retain_logits,
                  gradient_norm_cap=1.0, step0_dev=False, device=str(device), max_seconds=max_seconds,
                  requested_updates=steps, checkpoint_updates=sorted(set(CHECKPOINT_UPDATES)|set(selection_updates)),
                  checkpoint_policy='fixed_updates_and_final_before_evaluation; partial_on_failure; no_best_selection')
    if selection_updates:
        config.update(selection_updates=list(selection_updates),
            checkpoint_policy='fixed_updates_and_final_retained; evaluation_checkpoint_selected_on_inner_selection_only',
            selection_policy='maximum inner-selection primary AUROC; earliest update on tie; dev once after selection')
    model, manifest, exposure = None, None, hashlib.sha256()
    selected_restored=False
    _atomic_json(output / 'progress.json', progress)
    try:
        with deadline, deterministic_runtime(device, enabled=deterministic) as runtime:
            config['runtime'] = dict(runtime)
            progress['runtime'] = dict(runtime)
            import torch
            from .model import EventReadout, masked_membership_loss
            if str(device).startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('requested CUDA device unavailable')
            torch.manual_seed(seed)
            if str(device).startswith('cuda'):
                from music_critic.cuda_memory import initialize_cuda_memory_statistics
                from music_critic.device import resolve_runtime_device

                torch.cuda.manual_seed_all(seed)
                cuda_statistics = initialize_cuda_memory_statistics(
                    resolve_runtime_device(str(device)))
            path, manifest, bindings = _manifest(cache)
            width = manifest.get('input_width', 266)
            config.update(input_width=width, node_types=manifest.get('node_types', 9),
                          edge_types=manifest.get('edge_types', 64))
            model = EventReadout(input_width=width, kind=kind, hidden=64, rounds=3,
                node_types=manifest.get('node_types', 9), edge_types=manifest.get('edge_types', 64)).to(device)
            tile_cache = _TileCache(bindings, torch, width)
            by_work, split_known, seen = defaultdict(list), defaultdict(int), set()
            for tile_id, binding in sorted(bindings.items()):
                deadline.check()
                tile = tile_cache.get(tile_id)
                for index, query in enumerate(tile['queries']):
                    identity = binding['work'], query['h_id']
                    if identity in seen:
                        raise ValueError('duplicate (work, h_id) across cache tiles')
                    seen.add(identity)
                    known = int(query['mask'].sum())
                    split_known[binding['split']] += known
                    if binding['split'] == 'train' and known:
                        by_work[binding['work']].append((tile_id, index))
            if any(split_known[split] == 0 for split in ('train', 'selection', 'dev')):
                raise ValueError('every split requires known target support')
            progress.update(phase='step0_selection', cache_manifest=manifest['_binding'])
            _atomic_json(output / 'progress.json', progress)
            cache_seconds = time.monotonic() - deadline.started
            evaluation_started = time.monotonic()
            initial_rows = _evaluate(model, tile_cache, bindings, 'selection', device, torch, deadline, **evaluation_options)
            initial_evaluation_seconds = time.monotonic() - evaluation_started
            initial_threshold = select_threshold(initial_rows)
            initial = dict(selection=score_metrics(initial_rows, initial_threshold), dev=None,
                           dev_omission='fixed protocol: step0 selection only to bound evaluation cost')
            sampler = _WorkSampler(by_work, seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses, sampled, curve = [], set(), []
            selected_rows = None
            curve_eval_seconds=0.0
            progress['phase'] = 'training'
            model.train()
            training_started = time.monotonic()
            for step in range(steps):
                deadline.check()
                work, tile_id, query_index = sampler.next()
                tile = tile_cache.get(tile_id)
                query = tile['queries'][query_index]
                progress['attempted_updates'] += 1
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, tile, query, device)
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite training logits')
                loss = masked_membership_loss(logits, query['y'].to(device), query['mask'].to(device),
                                              class_weights=class_weights)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                was_clipped = bool(gradient_norm > 1.0)
                optimizer.step()
                progress['applied_updates'] += 1
                progress['gradient_clipping_count'] += int(was_clipped)
                progress['gradient_clipping_fraction'] = progress['gradient_clipping_count'] / progress['applied_updates']
                losses.append(float(loss.detach().cpu()))
                identity = [work, tile_id, query['h_id']]
                exposure.update((json.dumps(identity, separators=(',', ':')) + '\n').encode())
                sampled.add((work, query['h_id']))
                progress['query_exposure'] += 1
                progress['known_pair_exposure'] += int(query['mask'].sum())
                progress['exposure_sha256'] = exposure.hexdigest()
                deadline.check()
                if progress['applied_updates'] in config['checkpoint_updates']:
                    _save_checkpoint(output, model, torch, progress, config, manifest['_binding'],
                                     exposure.hexdigest(), reason='fixed_update')
                    _atomic_json(output / 'progress.json', progress)
                if progress['applied_updates'] in selection_updates:
                    curve_started=time.monotonic()
                    progress['phase']='curve_selection'
                    _atomic_json(output/'progress.json',progress)
                    evaluated=_evaluate(model,tile_cache,bindings,'selection',device,torch,deadline, **evaluation_options)
                    tick_threshold=select_threshold(evaluated)
                    metric=score_metrics(evaluated,tick_threshold)
                    curve.append(dict(update=progress['applied_updates'],selection=metric,
                        checkpoint=dict(progress['last_checkpoint']),threshold=tick_threshold,
                        exposure_sha256=exposure.hexdigest(),unique_train_h=len(sampled),
                        known_pair_exposure=progress['known_pair_exposure'],
                        block_train_loss_mean=math.fsum(losses[-1000:])/len(losses[-1000:]),
                        wall_seconds=time.monotonic()-deadline.started))
                    if _select_curve_point(curve) is curve[-1]:
                        # Retain the exact winning draw; never rerun stochastic
                        # CUDA reductions solely to reconstruct its evidence.
                        selected_rows = evaluated
                    _atomic_json(output/'selection-curve.json',dict(points=curve,dev_metrics_computed=False))
                    progress.update(phase='training',selection_curve_updates=[p['update'] for p in curve])
                    _atomic_json(output/'progress.json',progress)
                    model.train()
                    curve_eval_seconds+=time.monotonic()-curve_started
                if (step + 1) % 10 == 0 or step + 1 == steps:
                    progress.update(wall_seconds=time.monotonic() - deadline.started, last_loss=losses[-1])
                    _atomic_json(output / 'progress.json', progress)
            training_seconds = time.monotonic() - training_started - curve_eval_seconds
            progress['phase'] = 'saving_final_checkpoint'
            _save_checkpoint(output, model, torch, progress, config, manifest['_binding'],
                             exposure.hexdigest(), reason='final_before_evaluation')
            _atomic_json(output / 'progress.json', progress)
            deadline.check()
            progress['phase'] = 'final_selection'
            _atomic_json(output / 'progress.json', progress)
            evaluation_started = time.monotonic()
            if selection_updates:
                selected=_select_curve_point(curve)
                _restore_selected(output,selected,model,torch,progress,manifest)
                selected_restored=True
                progress.update(selected_update=selected['update'],selected_checkpoint=selected['checkpoint'])
                _atomic_json(output/'progress.json',progress)
                if selected_rows is None:
                    raise RuntimeError('selected curve rows were not retained')
                selection=selected_rows
                selection_metric=selected['selection']
                threshold=selected['threshold']
            else:
                selection = _evaluate(model, tile_cache, bindings, 'selection', device, torch, deadline, **evaluation_options)
                selection_metric=None
                threshold = select_threshold(selection)
            selection_evaluation_seconds = time.monotonic() - evaluation_started
            progress['phase'] = 'final_dev'
            _atomic_json(output / 'progress.json', progress)
            evaluation_started = time.monotonic()
            dev = _evaluate(model, tile_cache, bindings, 'dev', device, torch, deadline, **evaluation_options)
            # Prespecified no-learning controls, never chosen by dev quality.
            duration_selection = [dict(r, scores=r['duration_scores']) for r in initial_rows]
            duration_dev = [dict(r, scores=r['duration_scores']) for r in dev]
            duration_threshold = choose_threshold(duration_selection)
            controls = dict(
                all_member=score_metrics([dict(r, scores=[1.0]*len(r['scores'])) for r in dev]),
                all_nonmember=score_metrics([dict(r, scores=[0.0]*len(r['scores'])) for r in dev]),
                duration_only=score_metrics(duration_dev, duration_threshold),
                duration_threshold=duration_threshold,
                duration_rule='raw note duration/(1+duration); inner selection threshold; no learning')
            dev_evaluation_seconds = time.monotonic() - evaluation_started
            result = dict(progress, status='completed', phase='completed',
                controls=controls, initial=initial, final=dict(selection=selection_metric if selection_updates else score_metrics(selection, threshold),
                    dev=score_metrics(dev, threshold)), threshold=threshold,
                threshold_source='inner_fit_selection_only', cache_manifest=manifest['_binding'],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                timing=dict(cache_validation_seconds=cache_seconds, training_seconds=training_seconds,
                    curve_selection_seconds=curve_eval_seconds,
                    initial_selection_seconds=initial_evaluation_seconds,
                    final_selection_seconds=selection_evaluation_seconds, final_dev_seconds=dev_evaluation_seconds),
                evaluation_queries=dict(initial_selection=len(initial_rows), final_selection=0 if selection_updates else len(selection), final_dev=len(dev)),
                split_known_targets=dict(split_known),
                train_loss_mean=math.fsum(losses) / len(losses), train_loss_last=losses[-1],
                exposure_sha256=exposure.hexdigest(), unique_train_h=len(sampled),
                cpu_cache_peak_bytes=tile_cache.peak_bytes,
                peak_vram_bytes=torch.cuda.max_memory_allocated(cuda_statistics.logical_device_index) if str(device).startswith('cuda') else 0,
                config=config)
            if selection_updates:
                result.update(selection_curve=curve,selected_update=selected['update'],
                    selected_checkpoint=selected['checkpoint'],
                    selection_policy=config['selection_policy'],
                    final_selection_reused_from_curve=True,
                    selected_selection_rows_reused_from_curve=True,
                    selected_predictions_checkpoint=dict(selected['checkpoint']))
                result['evaluation_queries']['curve_selection']=len(initial_rows)*len(curve)
            temp = output / 'predictions.jsonl.tmp'
            with temp.open('w') as handle:
                for split, rows in (('selection', selection or []), ('dev', dev)):
                    for row in rows:
                        deadline.check()
                        handle.write(json.dumps(dict(row, split=split), separators=(',', ':'), allow_nan=False) + '\n')
            temp.replace(output / 'predictions.jsonl')
            deadline.check()
            result['wall_seconds'] = time.monotonic() - deadline.started
            _atomic_json(output / 'result.json', result)
            _atomic_json(output / 'progress.json', {key: value for key, value in result.items()
                         if key not in {'initial', 'final'}})
            return result
    except BaseException as exc:
        if progress['applied_updates'] > 0 and model is not None and manifest is not None and not selected_restored:
            try:
                # The main deadline has been disarmed by __exit__. The external
                # supervisor's run-cap + cleanup grace remains authoritative.
                with _Deadline(5.0):
                    _save_checkpoint(output, model, torch, progress, config, manifest['_binding'],
                                     exposure.hexdigest(), reason='partial_after_' + type(exc).__name__)
            except BaseException as checkpoint_error:
                progress['checkpoint_error'] = dict(type=type(checkpoint_error).__name__, reason=str(checkpoint_error))
        progress.update(status='aborted' if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else 'failed',
                        error_type=type(exc).__name__, error=str(exc),
                        wall_seconds=time.monotonic() - deadline.started)
        _atomic_json(output / 'progress.json', progress)
        _atomic_json(output / 'result.json', progress)
        raise
