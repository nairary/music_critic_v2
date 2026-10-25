"""Bounded old/new critic comparison on one copied TRAIN pair.

Only the exact project-owned source004 module is executable as a reference.
No checkpoint is loaded here, and neither branch owns the caller's model,
optimizer, feature tensors, or training RNG state.
"""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import math
from pathlib import Path
import stat
import time
from types import ModuleType
from typing import Mapping

import torch

from .contracts import ASPECTS
from .training import _BankMutationGuard, _rng_capture, _rng_restore


REFERENCE_MODEL_SHA256 = 'cecfd06e673f2b0a5b9256c79bcbbed68906bfd3d8937f793e350ad519d1d9fe'
OUTPUT_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-5}
STATE_TOLERANCE = {'rtol': 1e-5, 'atol': 1e-6}
GAUGE_PARAMETERS = frozenset(f'readouts.{aspect}.level_gate.bias' for aspect in ASPECTS)
POLICIES = ('strict', 'softmax_shift_gauge_v1')


class RuntimeEquivalenceError(ValueError):
    def __init__(self, message, report):
        super().__init__('runtime equivalence: ' + message)
        self.report = report


def validate_reference_model_config(value):
    if (not isinstance(value, dict) or set(value) not in ({'path', 'sha256'}, {'path', 'sha256', 'comparison_policy'})
            or value['sha256'] != REFERENCE_MODEL_SHA256 or not isinstance(value['path'], str)):
        raise ValueError('reference_model requires the allowlisted source004 path/SHA256 binding')
    if value.get('comparison_policy', 'strict') not in POLICIES:
        raise ValueError('unknown reference_model comparison_policy')
    path = Path(value['path'])
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('reference_model requires an absolute nonsymlink path')
    return dict(value)


def _read_reference_source(config):
    config = validate_reference_model_config(config)
    path = Path(config['path'])
    if not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > 1024 * 1024:
        raise ValueError('reference_model must be a bounded regular Python source file')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != config['sha256']:
        raise ValueError('reference_model source integrity mismatch')
    return path, raw


def _load_reference_model(config, *, feature_dim, width):
    path, raw = _read_reference_source(config)
    name = 'music_critic.critic._source004_runtime_reference'
    module = ModuleType(name)
    module.__package__ = 'music_critic.critic'
    module.__file__ = str(path)
    # The byte allowlist is checked before compilation or module execution.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    cls = module.__dict__.get('MultiscaleAspectCritic')
    if (not isinstance(cls, type) or not issubclass(cls, torch.nn.Module)
            or cls.__module__ != name):
        raise ValueError('reference_model must define its own allowlisted critic class')
    return cls(feature_dim=feature_dim, width=width)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return replace(value, **{field.name: _snapshot(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, Mapping):
        return {key: _snapshot(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(child) for child in value)
    return value


def _compare(reference, candidate, tolerance, summary, path):
    def mismatch(message):
        summary['mismatches'].append({'path': path, 'reason': message})
        summary['passed'] = False
    if isinstance(reference, torch.Tensor):
        if (not isinstance(candidate, torch.Tensor) or reference.shape != candidate.shape
                or reference.dtype != candidate.dtype):
            mismatch('tensor shape/dtype mismatch')
            return
        summary['tensor_pairs'] += 1
        if not reference.is_floating_point():
            if not torch.equal(reference, candidate):
                mismatch('exact tensor mismatch')
            return
        if not bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
            mismatch('nonfinite tensor')
            return
        if reference.numel():
            delta = (reference.double() - candidate.double()).abs()
            maximum = float(delta.max())
            scaled = float((delta / (tolerance['atol'] + tolerance['rtol'] * reference.double().abs())).max())
            summary['max_absolute_error'] = max(summary['max_absolute_error'], maximum)
            summary['max_tolerance_ratio'] = max(summary['max_tolerance_ratio'], scaled)
            if scaled > 1:
                mismatch(f'numerical mismatch; max_absolute_error={maximum}; max_tolerance_ratio={scaled}')
        return
    if is_dataclass(reference):
        if type(candidate) is not type(reference):
            mismatch('output type mismatch')
            return
        for field in fields(reference):
            _compare(getattr(reference, field.name), getattr(candidate, field.name), tolerance,
                     summary, path + '.' + field.name)
        return
    if isinstance(reference, Mapping):
        if not isinstance(candidate, Mapping) or reference.keys() != candidate.keys():
            mismatch('mapping keys mismatch')
            return
        for key in reference:
            _compare(reference[key], candidate[key], tolerance, summary, path + '.' + str(key))
        return
    if isinstance(reference, (tuple, list)):
        if type(candidate) is not type(reference) or len(candidate) != len(reference):
            mismatch('sequence mismatch')
            return
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _compare(left, right, tolerance, summary, path + '.' + str(index))
        return
    if reference is None and candidate is None:
        summary['none_pairs'] += 1
    elif type(reference) is not type(candidate) or reference != candidate:
        mismatch('exact metadata or None/zero mismatch')


def _comparison(left, right, tolerance, path):
    summary = {**tolerance, 'tensor_pairs': 0, 'none_pairs': 0, 'passed': True,
               'max_absolute_error': 0., 'max_tolerance_ratio': 0., 'mismatches': []}
    _compare(left, right, tolerance, summary, path)
    return summary


def _safe_values(tensor):
    """Only the four authorized gauge coordinates appear as values in reports."""
    if tensor is None:
        return None
    return [value if math.isfinite(value) else None for value in tensor.reshape(-1).tolist()]


def _gauge_report(before, after):
    rows = {}
    for key in sorted(GAUGE_PARAMETERS):
        left, right = before['gauge'][key], after['gauge'][key]
        values = (left['weight'], right['weight'])
        gradients = (left['gradient'], right['gradient'])
        # None is retained as unavailable, never silently converted to zero.
        valid = all(value.shape == (1,) and bool(torch.isfinite(value).all()) for value in values)
        valid = valid and all(value is None or (value.shape == (1,)
            and bool(torch.isfinite(value).all()) and float(value.abs().max()) <= 1e-6)
            for value in gradients)
        valid = valid and ((gradients[0] is None) == (gradients[1] is None))
        maximum = float((values[0].double() - values[1].double()).abs().max()) if valid else None
        bound = 2 * left['learning_rate'] + 1e-6
        valid = valid and left['learning_rate'] == right['learning_rate'] and maximum <= bound
        if gradients[0] is None:
            valid = valid and torch.equal(values[0], values[1])
        def public(row):
            return {name: ({k: _safe_values(v) for k, v in value.items()} if name == 'optimizer'
                    else value if name == 'learning_rate' else _safe_values(value))
                    for name, value in row.items()}
        rows[key] = {'admissible': bool(valid), 'max_absolute_weight_difference': maximum,
                     'weight_difference_bound': bound, 'gradient_absolute_bound': 1e-6,
                     'reference': public(left), 'candidate': public(right)}
    return rows


def compare_reference_runtime(trainer, reference_model, *, clock=time.monotonic,
                              check_budget=lambda: None, progress_callback=None):
    """Compare one copied TRAIN pair; strict by default, explicit gauge opt-in."""
    from .campaign import _emit_probe_progress
    trainer._check_bank()
    if not isinstance(trainer.optimizer, torch.optim.AdamW):
        raise ValueError('runtime equivalence requires the current AdamW optimizer')
    policy = reference_model.get('comparison_policy', 'strict')
    if policy not in POLICIES:
        raise ValueError('unknown reference_model comparison_policy')
    started = clock()
    report = {'schema': 'CriticRuntimeEquivalence@1.0.0', 'passed': False,
        'reference_sha256': reference_model['sha256'], 'comparison_policy': policy, 'pair_count': 1,
        'forward_views': 0, 'paired_draws': 0, 'optimizer_applications': 0,
        'inputs_detached': True, 'timings': {}, 'comparisons': {}, 'raw_state_parity': None,
        'functional_closeness_modulo_softmax_shifts': False,
        'claim': 'One first TRAIN pair only; not full-corpus or worst-case equivalence.'}
    def progress_summary():
        return {key: report[key] for key in ('forward_views', 'paired_draws', 'optimizer_applications',
            'timings', 'comparisons', 'raw_state_parity', 'functional_closeness_modulo_softmax_shifts')}
    def emit(stage, state, *, stage_started=None, details=None):
        def callback(row):
            progress_callback({**row, **({'comparison_summary': copy.deepcopy(details)}
                                         if details is not None else {})})
        return _emit_probe_progress(callback if progress_callback is not None else None,
            stage, state, trainer, started=started, clock=clock, stage_started=stage_started)
    def synchronize():
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize(trainer.device)
    def timed(operation, timings, key):
        check_budget()
        synchronize()
        tick = clock()
        try:
            return operation()
        finally:
            synchronize()
            timings[key] = clock() - tick

    original_guard = _BankMutationGuard(trainer.bank[0])
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    caller_rng = _rng_capture()
    comparison_rng = copy.deepcopy(trainer.rng_state)
    try:
        with torch.random.fork_rng(devices=devices):
            check_budget()
            candidate = copy.deepcopy(trainer.model)
            reference = _load_reference_model(reference_model, feature_dim=trainer.model.feature_dim,
                                              width=trainer.model.width)
            parameter = next(candidate.parameters())
            reference.to(device=trainer.device, dtype=parameter.dtype)
            reference.load_state_dict(copy.deepcopy(trainer.model.state_dict()), strict=True)
            trainable = {name: value.requires_grad for name, value in trainer.model.named_parameters()}
            if set(trainable) != dict(reference.named_parameters()).keys():
                raise ValueError('parameter namespace mismatch')
            for name, value in reference.named_parameters():
                value.requires_grad_(trainable[name])

            def branch(model, name):
                _rng_restore(comparison_rng)
                stage = 'runtime_equivalence_' + name
                tick = emit(stage, 'started')
                timings = report['timings'][name] = {}
                model.train()
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=trainer.config.learning_rate,
                                              weight_decay=trainer.config.weight_decay)
                optimizer.load_state_dict(copy.deepcopy(trainer.optimizer.state_dict()))
                optimizer.zero_grad(set_to_none=True)
                pair = copy.deepcopy(trainer.bank[0])
                report['paired_draws'] += 1
                views = (pair.left.to(trainer.device), pair.right.to(trainer.device))
                inputs = [tensor for view in views for sample in view.samples for tensor in sample.embeddings.values()]
                if any(t.requires_grad or t.grad_fn is not None for t in inputs):
                    report['inputs_detached'] = False
                    raise ValueError('inputs must stay detached')
                guard = _BankMutationGuard(views)
                def forward_views():
                    outputs = []
                    for view in views:
                        outputs.append(model(view))
                        report['forward_views'] += 1
                    return tuple(outputs)
                outputs = timed(forward_views, timings, 'forward_seconds')
                loss = timed(lambda: copy.deepcopy(trainer.loss)(outputs, (pair.targets,)), timings, 'loss_seconds')
                if loss.skip_update or loss.total is None or not bool(torch.isfinite(loss.total)):
                    raise ValueError('requires a finite supported pair update')
                snapshots = {'outputs': _snapshot(outputs), 'loss': _snapshot(loss)}
                timed(loss.total.backward, timings, 'backward_seconds')
                snapshots['gradients'] = _snapshot({key: p.grad for key, p in model.named_parameters()})
                norm = timed(lambda: torch.nn.utils.clip_grad_norm_(
                    parameters, trainer.config.clip_grad_norm, error_if_nonfinite=True), timings, 'clip_seconds')
                snapshots['clipped_gradients'] = _snapshot({key: p.grad for key, p in model.named_parameters()})
                snapshots['gradient_norm'] = _snapshot(norm)
                timed(optimizer.step, timings, 'optimizer_seconds')
                report['optimizer_applications'] += 1
                snapshots['model_state'] = _snapshot(model.state_dict())
                snapshots['optimizer_state'] = _snapshot(optimizer.state_dict())
                # Mandatory functional check of the actual, uncorrected post-step states.
                with torch.no_grad():
                    snapshots['post_step_outputs'] = _snapshot(timed(forward_views, timings, 'post_forward_seconds'))
                named = dict(model.named_parameters())
                learning_rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
                snapshots['gauge'] = {key: {
                    'initial_weight': _snapshot(trainer.model.state_dict()[key]),
                    'weight': snapshots['model_state'][key], 'gradient': snapshots['gradients'][key],
                    'clipped_gradient': snapshots['clipped_gradients'][key],
                    'optimizer': _snapshot(optimizer.state.get(named[key], {})),
                    'learning_rate': learning_rates[id(named[key])]} for key in GAUGE_PARAMETERS}
                if not guard.matches(views) or any(t.grad is not None for t in inputs):
                    raise ValueError('mutated feature inputs')
                emit(stage, 'completed', stage_started=tick, details={'timings': timings,
                     'optimizer_applications': 1, 'paired_draws': 1, 'forward_views': 4})
                return snapshots

            before, after = branch(reference, 'reference'), branch(candidate, 'candidate')
            check_budget()
            tick = emit('runtime_equivalence_comparison', 'started')
            comparisons = report['comparisons']
            for key in before:
                if key == 'gauge':
                    continue
                tolerance = OUTPUT_TOLERANCE if key in {'outputs', 'loss', 'post_step_outputs'} else STATE_TOLERANCE
                comparisons[key] = _comparison(before[key], after[key], tolerance, key)
            report['raw_state_parity'] = comparisons['model_state']['passed']
            report['gauge_parameters'] = _gauge_report(before, after)
            nongauge = lambda branch: {key: value for key, value in branch['model_state'].items()
                                      if key not in GAUGE_PARAMETERS}
            comparisons['model_state_without_gauge'] = _comparison(
                nongauge(before), nongauge(after), STATE_TOLERANCE, 'model_state_without_gauge')
            strict_other = all(row['passed'] for key, row in comparisons.items()
                               if key not in {'model_state', 'model_state_without_gauge'})
            modulo = (strict_other and comparisons['model_state_without_gauge']['passed']
                      and all(row['admissible'] for row in report['gauge_parameters'].values()))
            report['functional_closeness_modulo_softmax_shifts'] = modulo
            report['passed'] = strict_other and (report['raw_state_parity'] if policy == 'strict' else modulo)
            if not report['passed']:
                raise RuntimeEquivalenceError('comparison failed; inspect per-component error maxima', report)
            if not original_guard.matches(trainer.bank[0]):
                raise ValueError('mutated original bank')
            trainer._check_bank()
            emit('runtime_equivalence_comparison', 'completed', stage_started=tick, details=progress_summary())
    except Exception as exc:
        report['wall_seconds'] = clock() - started
        report['passed'] = False
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        try:
            emit('runtime_equivalence_comparison', 'failed', details=progress_summary())
        except Exception as callback_error:
            exc.add_note('failure progress callback also failed: ' + type(callback_error).__name__)
        if isinstance(exc, RuntimeEquivalenceError):
            raise
        raise RuntimeEquivalenceError(str(exc), report) from exc
    finally:
        _rng_restore(caller_rng)
    report['wall_seconds'] = clock() - started
    return report
