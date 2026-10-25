"""Bounded FP32 reference training and immutable update-boundary resume."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import math
import os
from pathlib import Path
import random
import tempfile

import torch

from music_critic.device import resolve_cuda_device_index
from .model import HCM1

CHECKPOINT_SCHEMA = 'HCM1TrainingCheckpoint@1.0.0'


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = .01
    grad_clip: float = 1.
    accumulation: int = 1
    total_updates: int = 1000
    warmup_updates: int = 50
    unfreeze_update: int | None = None

    def __post_init__(self):
        for name in ('lr', 'backbone_lr', 'grad_clip'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('pll.invalid_' + name)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('pll.invalid_weight_decay')
        for name in ('accumulation', 'total_updates'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('pll.invalid_' + name)
        if type(self.warmup_updates) is not int or not 0 <= self.warmup_updates < self.total_updates:
            raise ValueError('pll.invalid_warmup_updates')
        if self.unfreeze_update is not None and (type(self.unfreeze_update) is not int
                or not 0 <= self.unfreeze_update < self.total_updates):
            raise ValueError('pll.invalid_unfreeze_update')


def implementation_fingerprint() -> str:
    """Bind local Python source bytes, including inherited C3 collaborators."""
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    return copy.deepcopy(value)


class PLLTrainer:
    def __init__(self, model: HCM1, config: TrainingConfig = TrainingConfig(), *,
                 data_id: str, protocol_id: str, source_id: str | None = None):
        if not data_id or not protocol_id:
            raise ValueError('pll.data_and_protocol_binding_required')
        if getattr(model.encoder, 'lock_local', False) and config.unfreeze_update is not None:
            raise ValueError('pll.whole_frozen_local_cannot_unfreeze')
        self.model, self.config = model, config
        self.model_contract = model.contract()
        self.initial_freeze_policy = model.encoder.freeze_policy
        self.binding = dict(model=self.model_contract, training=asdict(config), data_id=data_id,
                            protocol_id=protocol_id, source_id=source_id or implementation_fingerprint(),
                            dtype='float32', execution='serial-microbatch-v1', torch_version=str(torch.__version__))
        self.device = next(model.parameters()).device
        if self.device.type not in ('cpu', 'cuda') or any(p.dtype != torch.float32 or p.device != self.device for p in model.parameters()):
            raise ValueError('pll.training_requires_single_device_fp32')
        self.binding['device_type'] = self.device.type
        old, new = [], []
        for name, parameter in model.named_parameters():
            if name.startswith('encoder.backbone.base.task_heads.'):
                continue
            (old if name.startswith('encoder.backbone.') else new).append(parameter)
        # Keep inherited weights in the optimizer from the start so a declared
        # freeze transition does not discard existing optimizer state.
        self.optimizer = torch.optim.AdamW([
            {'params': new, 'lr': config.lr, 'base_lr': config.lr},
            {'params': old, 'lr': config.backbone_lr, 'base_lr': config.backbone_lr},
        ], weight_decay=config.weight_decay)
        self.updates = self.draws = 0
        self._in_step = False

    def _freeze_policy(self, updates=None):
        updates = self.updates if updates is None else updates
        if self.config.unfreeze_update is not None and updates >= self.config.unfreeze_update:
            return 'full' if self.initial_freeze_policy == 'full' else 'upper'
        return self.initial_freeze_policy

    def _lr_multiplier(self, updates=None):
        c = self.config
        updates = self.updates if updates is None else updates
        if updates < c.warmup_updates:
            return (updates + 1) / c.warmup_updates
        progress = (updates - c.warmup_updates) / max(1, c.total_updates - c.warmup_updates)
        return .5 * (1. + math.cos(math.pi * progress))

    def step(self, examples: tuple) -> dict:
        if self.updates >= self.config.total_updates:
            raise ValueError('pll.training_update_budget_exhausted')
        if not isinstance(examples, tuple) or len(examples) != self.config.accumulation:
            raise ValueError('pll.accumulation_example_count_mismatch')
        if self._in_step:
            raise ValueError('pll.reentrant_training_step')
        self.model.encoder.set_freeze_policy(self._freeze_policy())
        self.model.train()
        for group in self.optimizer.param_groups:
            group['lr'] = group['base_lr'] * self._lr_multiplier()
        self.optimizer.zero_grad(set_to_none=True)
        self._in_step = True
        nll = 0.; count = tokens = 0
        context_diagnostics = []
        try:
            for example in examples:
                result = self.model(example)
                if getattr(self.model, 'last_diagnostics', None) is not None:
                    context_diagnostics.append(copy.deepcopy(self.model.last_diagnostics))
                if not torch.isfinite(result.nll):
                    raise FloatingPointError('pll.nonfinite_loss')
                (result.nll / len(examples)).backward()
                nll += float(result.nll.detach())
                count += result.event_count
                tokens += result.token_count
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            if any(not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('pll.nonfinite_gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip, error_if_nonfinite=True)
            self.optimizer.step()
            self.updates += 1
            self.draws += len(examples)
            metrics = dict(update=self.updates, draws=self.draws, block_nll=nll / len(examples),
                        nats_per_event=nll / count if count else None,
                        target_events=count, target_tokens=tokens, grad_norm=float(norm),
                        freeze_policy=self.model.encoder.freeze_policy)
            if context_diagnostics:
                metrics['context_diagnostics'] = context_diagnostics
            return metrics
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self._in_step = False

    def _rng_state(self):
        result = {'python': random.getstate(), 'torch_cpu': torch.get_rng_state(), 'torch_cuda': None}
        if self.device.type == 'cuda':
            result['torch_cuda'] = torch.cuda.get_rng_state(resolve_cuda_device_index(self.device))
        return result

    def _restore_rng(self, state):
        random.setstate(state['python'])
        torch.set_rng_state(state['torch_cpu'])
        if self.device.type == 'cuda':
            if state['torch_cuda'] is None:
                raise ValueError('pll.missing_cuda_rng')
            torch.cuda.set_rng_state(state['torch_cuda'], resolve_cuda_device_index(self.device))

    def save_checkpoint(self, path, *, sampler_state: dict) -> str:
        if self._in_step or not isinstance(sampler_state, dict):
            raise ValueError('pll.checkpoint_requires_update_boundary_and_sampler_state')
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        payload = dict(schema=CHECKPOINT_SCHEMA, binding=copy.deepcopy(self.binding),
                       model=_cpu_tree(self.model.state_dict()), optimizer=_cpu_tree(self.optimizer.state_dict()),
                       updates=self.updates, draws=self.draws, rng=self._rng_state(),
                       sampler_state=copy.deepcopy(sampler_state))
        payload['optimizer_state_ids'] = sorted(payload['optimizer']['state'])
        descriptor, temporary = tempfile.mkstemp(prefix='.hcm1-', dir=destination.parent)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                torch.save(payload, handle)
                handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return sha256(destination.read_bytes()).hexdigest()

    def load_checkpoint(self, path, *, expected_sha256: str) -> dict:
        if self._in_step:
            raise ValueError('pll.resume_requires_update_boundary')
        path = Path(path)
        checkpoint_bytes = path.read_bytes()
        if sha256(checkpoint_bytes).hexdigest() != expected_sha256:
            raise ValueError('pll.checkpoint_hash_mismatch')
        payload = torch.load(BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('binding') != self.binding:
            raise ValueError('pll.checkpoint_binding_mismatch')
        updates, draws = payload['updates'], payload['draws']
        if (type(updates) is not int or not 0 <= updates <= self.config.total_updates
                or type(draws) is not int or draws != updates * self.config.accumulation):
            raise ValueError('pll.checkpoint_progress_invalid')
        expected = self.model.state_dict()
        if set(payload['model']) != set(expected) or any(
                not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                or v.dtype != expected[k].dtype or not torch.isfinite(v).all()
                for k, v in payload['model'].items()):
            raise ValueError('pll.checkpoint_weights_invalid')
        if not isinstance(payload.get('sampler_state'), dict):
            raise ValueError('pll.checkpoint_sampler_invalid')
        self._validate_optimizer(payload.get('optimizer'), updates, payload.get('optimizer_state_ids'))
        rng = payload.get('rng')
        if not isinstance(rng, dict) or set(rng) != {'python', 'torch_cpu', 'torch_cuda'}:
            raise ValueError('pll.checkpoint_rng_invalid')
        try:
            random.Random().setstate(rng['python'])
            torch.Generator(device='cpu').set_state(rng['torch_cpu'])
            if self.device.type == 'cuda':
                current = torch.cuda.get_rng_state(resolve_cuda_device_index(self.device))
                if not isinstance(rng['torch_cuda'], torch.Tensor) or rng['torch_cuda'].dtype != current.dtype or rng['torch_cuda'].shape != current.shape:
                    raise ValueError('CUDA RNG shape mismatch')
            elif rng['torch_cuda'] is not None:
                raise ValueError('unexpected CUDA RNG')
        except Exception as exc:
            raise ValueError('pll.checkpoint_rng_invalid') from exc
        before = (_cpu_tree(expected), copy.deepcopy(self.optimizer.state_dict()), self._rng_state(),
                  self.updates, self.draws, self.model.encoder.freeze_policy)
        try:
            self.model.load_state_dict(payload['model'], strict=True)
            self.optimizer.load_state_dict(payload['optimizer'])
            self.updates, self.draws = updates, draws
            self.model.encoder.set_freeze_policy(self._freeze_policy())
            self._restore_rng(payload['rng'])
        except Exception:
            self.model.load_state_dict(before[0], strict=True)
            self.optimizer.load_state_dict(before[1])
            self._restore_rng(before[2])
            self.updates, self.draws = before[3:5]
            self.model.encoder.set_freeze_policy(before[5])
            raise
        return copy.deepcopy(payload['sampler_state'])

    def _validate_optimizer(self, state, updates, recorded_ids):
        """AdamW's loader accepts malformed tensors; validate before mutation."""
        fail = 'pll.checkpoint_optimizer_invalid'
        expected = self.optimizer.state_dict()
        if not isinstance(state, dict) or set(state) != {'state', 'param_groups'}:
            raise ValueError(fail)
        groups = state['param_groups']
        if not isinstance(groups, list) or len(groups) != len(expected['param_groups']) or not isinstance(state['state'], dict):
            raise ValueError(fail)
        parameters = {}
        for given, wanted, live in zip(groups, expected['param_groups'], self.optimizer.param_groups):
            if not isinstance(given, dict) or set(given) != set(wanted):
                raise ValueError(fail)
            wanted = dict(wanted)
            wanted['lr'] = wanted['base_lr'] * (self._lr_multiplier(updates - 1) if updates else 1.)
            if given != wanted:
                raise ValueError(fail)
            parameters.update(zip(wanted['params'], live['params']))
        if any(type(key) is not int or key not in parameters for key in state['state']):
            raise ValueError(fail)
        if recorded_ids != sorted(state['state']) or (updates == 0 and state['state']):
            raise ValueError(fail)
        always_used = {id(self.model.decoder.output.weight), id(self.model.decoder.output.bias)}
        required_ids = {key for key, parameter in parameters.items() if id(parameter) in always_used}
        if updates and not required_ids <= set(state['state']):
            raise ValueError(fail)
        for key, value in state['state'].items():
            if not isinstance(value, dict) or set(value) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError(fail)
            step = value['step']
            if (not isinstance(step, torch.Tensor) or step.ndim != 0 or not torch.isfinite(step)
                    or step.dtype != torch.float32 or float(step) != int(step)
                    or not 1 <= int(step) <= updates or (key in required_ids and int(step) != updates)):
                raise ValueError(fail)
            parameter = parameters[key]
            for field in ('exp_avg', 'exp_avg_sq'):
                moment = value[field]
                if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype or not torch.isfinite(moment).all()):
                    raise ValueError(fail)
            if (value['exp_avg_sq'] < 0).any():
                raise ValueError(fail)
