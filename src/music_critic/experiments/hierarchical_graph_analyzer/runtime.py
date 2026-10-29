"""Small opt-in H training/evaluation runtime with explicit source bindings.

This module reads no corpus. Callers supply authenticated, indexable raw/target
samples, an optimizer, and the external run authorization. Synthetic tests do
not admit a real training launch. Sampling is uniform work -> tile -> view.
"""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random
import tempfile

import torch

from .losses import compute_loss
from .matching import as_list, canonical_time
from .metrics import aggregate_metrics, evaluate_piece


def _json_default(value):
    if isinstance(value, Fraction):
        return {'num': value.numerator, 'den': value.denominator}
    raise TypeError(f'noncanonical identity value: {type(value).__name__}')


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, default=_json_default).encode()


def tensorize_targets(target, *, device, dtype=torch.float32):
    """Copy tensor-bearing fields while retaining exact target-only metadata."""
    result = dict(target)
    result['spans'] = torch.as_tensor(target['spans'], device=device, dtype=dtype).reshape(-1, 2)
    result['fields'] = {}
    for name, field in target.get('fields', {}).items():
        values, mask = as_list(field['values']), as_list(field['mask'])
        if any(type(value) is not bool for value in mask):
            raise ValueError('target masks must be boolean before tensor conversion')
        if any(type(value) is not int for value in values):
            raise ValueError('target class IDs must be integer before tensor conversion')
        if len(values) != len(mask) or len(mask) != len(result['spans']):
            raise ValueError('target fields must align with spans')
        result['fields'][name] = {'values': torch.tensor(values, device=device, dtype=torch.long),
                                  'mask': torch.tensor(mask, device=device, dtype=torch.bool)}
    if 'event_mask' in target:
        values = as_list(target['event_mask'])
        if any(type(value) is not bool for value in values):
            raise ValueError('event mask must contain booleans')
        result['event_mask'] = torch.tensor(values, device=device, dtype=torch.bool)
    return result


def _tile_times(target):
    core = tuple(canonical_time(value) for value in target['core_span'])
    halo = tuple(canonical_time(value) for value in target['halo_span'])
    if not halo[0] <= core[0] < core[1] <= halo[1]:
        raise ValueError('positive nested core and halo spans required')
    return core, halo


def filter_core_predictions(pred, target):
    """Decode unique start ownership; training deliberately retains halo queries."""
    core, halo = _tile_times(target)
    left, right = [float((time - halo[0]) / (halo[1] - halo[0])) for time in core]
    spans = pred['spans']
    keep = (spans[:, 0].detach() >= left) & (spans[:, 0].detach() < right)
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    result = dict(pred)
    for name in ('spans', 'object_logits', 'states'):
        if name in result:
            result[name] = result[name][indices]
    result['field_logits'] = {name: value[indices] for name, value in pred.get('field_logits', {}).items()}
    for name in ('ids', 'view_ids', 'scope_ids'):
        if name in result:
            result[name] = [result[name][index] for index in indices.cpu().tolist()]
    for name in ('view_id', 'scope_id'):
        result[name] = target.get(name)
    return result


def initialize_shared_parameters(model, seed=17):
    """Initialize each named parameter from a stable independent RNG stream.

    Common names/shapes therefore get identical values across E1--E5 even if
    constructor modules are registered in a different order. Global RNG is
    preserved. LayerNorm scales are one, biases zero, matrix weights Xavier;
    learned query vectors use std .02, relation embeddings std .05.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            stream_seed = int.from_bytes(sha256(f'{seed}:{name}'.encode()).digest()[:8], 'big') % (2**63 - 1)
            generator = torch.Generator(device=parameter.device).manual_seed(stream_seed)
            if name.endswith('bias'):
                parameter.zero_()
            elif parameter.ndim == 1:
                parameter.fill_(1.)
            elif 'event_queries' in name:
                torch.nn.init.normal_(parameter, std=.02, generator=generator)
            elif '.edge_gate.' in name or '.edge_shift.' in name:
                torch.nn.init.normal_(parameter, std=.05, generator=generator)
            else:
                torch.nn.init.xavier_uniform_(parameter, generator=generator)
    return model


def _inventory(samples, sample_metadata=None):
    metadata = sample_metadata if sample_metadata is not None else getattr(samples, 'metadata', None)
    if metadata is None:
        metadata = [samples[index][1] for index in range(len(samples))]
    if len(metadata) != len(samples) or not metadata:
        raise ValueError('nonempty aligned sample metadata required')
    groups, identities, seen = defaultdict(lambda: defaultdict(list)), {}, set()
    for index, target in enumerate(metadata):
        if target.get('split') != 'fit':
            raise ValueError('training schedule admits fit samples only')
        work = target.get('work_id')
        if not isinstance(work, str) or not work:
            raise ValueError('stable work identity required')
        core, halo = _tile_times(target)
        tile = _encoded([core, halo]).decode()
        identity = _sample_identity(target)
        if identity in seen:
            raise ValueError('duplicate work/tile/view sample identity')
        seen.add(identity)
        identities[index] = identity
        groups[work][tile].append(index)
    ordered = {work: {tile: sorted(indices, key=identities.__getitem__)
                      for tile, indices in sorted(tiles.items())}
               for work, tiles in sorted(groups.items())}
    fingerprint = sha256(_encoded(sorted(identities.values()))).hexdigest()
    return ordered, identities, fingerprint


def _sample_identity(target):
    core, halo = _tile_times(target)
    return _encoded([target.get('work_id'), core, halo,
                     target.get('view_id'), target.get('scope_id')]).decode()


def _draw(groups, seed, cursor):
    rng = random.Random(int.from_bytes(sha256(f'{seed}:{cursor}'.encode()).digest(), 'big'))
    work = rng.choice(list(groups))
    tile = rng.choice(list(groups[work]))
    return rng.choice(groups[work][tile])


def sample_schedule(samples, *, count, seed=17, start=0, sample_metadata=None):
    """Deterministic indices; semantic exposure is unchanged by sample ordering."""
    if any(type(value) is not int or value < 0 for value in (count, seed, start)):
        raise ValueError('count, seed and start must be nonnegative integers')
    groups, _, _ = _inventory(samples, sample_metadata)
    return [_draw(groups, seed, cursor) for cursor in range(start, start + count)]


def train_steps(model, samples, optimizer, *, steps, seed=17, state=None,
                weights=None, sample_metadata=None, clip_grad_norm=None, on_step=None):
    """Run a bounded number of attempted draws, recording applied updates.

    All-unknown draws are exposure but never optimizer steps (including Adam
    momentum/weight decay). Resume uses the same seed and inventory. Full model,
    optimizer and RNG restoration is supplied separately by load_checkpoint.
    on_step receives independent copies of state/history after each completed
    draw, so the caller can retain partial accounting when a later draw fails.
    """
    if any(type(value) is not int or value < 0 for value in (steps, seed)):
        raise ValueError('steps and seed must be nonnegative integers')
    groups, identities, fingerprint = _inventory(samples, sample_metadata)
    current = deepcopy(state) if state is not None else {
        'schema': 'HiergraphTrainingState@1', 'seed': seed, 'schedule_fingerprint': fingerprint,
        'attempted_steps': 0, 'applied_updates': 0, 'skipped_unknown': 0,
        'exposure_sha256': sha256(b'').hexdigest()}
    if (current.get('schema') != 'HiergraphTrainingState@1' or current.get('seed') != seed
            or current.get('schedule_fingerprint') != fingerprint):
        raise ValueError('resume schedule/seed mismatch')
    parameter = next(model.parameters())
    history = []
    if state is None:
        random.seed(seed)
        torch.manual_seed(seed)
    model.train()
    for _ in range(steps):
        index = _draw(groups, seed, current['attempted_steps'])
        raw, target = samples[index]
        if target.get('split') != 'fit':
            raise ValueError('non-fit target returned by sample reader')
        if _sample_identity(target) != identities[index]:
            raise ValueError('sample reader identity differs from scheduled metadata')
        raw = {name: tensor.to(parameter.device) for name, tensor in raw.items()}
        target = tensorize_targets(target, device=parameter.device, dtype=parameter.dtype)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(raw)
        result = compute_loss(prediction, target, weights)
        if not bool(torch.isfinite(result['loss'])):
            raise ValueError('non-finite loss; no optimizer update applied')
        supported = bool(result['support']['matched_events'] or result['support']['object_negative']
                         or result['support']['boundary_positive'])
        if supported:
            result['loss'].backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError('non-finite gradient; no optimizer update applied')
            if clip_grad_norm is not None:
                if not isinstance(clip_grad_norm, (int, float)) or clip_grad_norm <= 0:
                    raise ValueError('clip_grad_norm must be positive')
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            current['applied_updates'] += 1
        else:
            current['skipped_unknown'] += 1
        current['attempted_steps'] += 1
        current['exposure_sha256'] = sha256(bytes.fromhex(current['exposure_sha256'])
                                            + identities[index].encode()).hexdigest()
        history.append({'sample_index': index, 'sample_identity': identities[index],
                        'loss': float(result['loss'].detach()), 'applied': supported,
                        'support': result['support']})
        if on_step is not None:
            on_step(deepcopy(current), deepcopy(history[-1]))
    return {'state': current, 'history': history}


def evaluate(model, samples, *, split='selection', allow_dev=False, oracle=False,
             candidates_per_quarter=4, max_predictions=None, iou_threshold=.5):
    """Selection is the default; final dev requires explicit allow_dev=True.

    Oracle-H is an explicitly separate interpretation diagnostic and receives no
    primary discovery score. Every submodule train/eval flag is restored.
    """
    if split not in ('selection', 'dev') or (split == 'dev' and allow_dev is not True):
        raise ValueError('selection evaluation only; final dev requires explicit allow_dev')
    parameter = next(model.parameters())
    flags = [(module, module.training) for module in model.modules()]
    reports = []
    try:
        model.eval()
        with torch.no_grad():
            for raw, target in samples:
                if target.get('split') != split:
                    raise ValueError('evaluation sample split differs from declared role')
                core, halo = _tile_times(target)
                raw = {name: tensor.to(parameter.device) for name, tensor in raw.items()}
                typed = tensorize_targets(target, device=parameter.device, dtype=parameter.dtype)
                if oracle:
                    prediction = model.oracle_readout(raw, typed['spans'])
                    # Known geometry is diagnostic only; no outside-capacity
                    # events or raw discovery-budget interpretation is implied.
                    oracle_target = dict(target, coverage={})
                    report = evaluate_piece(prediction, oracle_target,
                        duration_quarters=core[1] - core[0], span_scale_quarters=halo[1] - halo[0],
                        candidates_per_quarter=max(candidates_per_quarter,
                                                   len(target['ids']) / float(core[1] - core[0]) + 1),
                        iou_threshold=iou_threshold)
                    report['oracle_h'] = True
                else:
                    prediction = filter_core_predictions(model(raw), target)
                    raw_candidates = None
                    if 'time_positions' in raw and 'bank_indices' in raw:
                        bank_spans = raw['time_positions'][raw['bank_indices']]
                        raw_candidates = filter_core_predictions(
                            {'spans': bank_spans, 'object_logits': bank_spans.new_zeros(len(bank_spans)),
                             'field_logits': {}}, target)
                    report = evaluate_piece(prediction, target,
                        duration_quarters=core[1] - core[0], span_scale_quarters=halo[1] - halo[0],
                        candidates_per_quarter=candidates_per_quarter, max_predictions=max_predictions,
                        iou_threshold=iou_threshold, raw_candidates=raw_candidates)
                    report['raw_candidate_reference'] = ('shared_raw_span_bank'
                                                          if raw_candidates is not None else None)
                reports.append(report)
    finally:
        for module, training in flags:
            module.training = training
    aggregate = aggregate_metrics(reports)
    return {'split': split, 'regime': 'oracle_h_diagnostic' if oracle else 'predicted_h',
            'primary': None if oracle else aggregate['family_macro_known_h_recovery'],
            'aggregate': aggregate, 'reports': reports}


def _bindings(bindings):
    required = {'source_sha256', 'config_sha256', 'data_sha256'}
    if not isinstance(bindings, dict) or not required <= set(bindings):
        raise ValueError('source/config/data SHA256 bindings are required')
    if any(not isinstance(value, str) or len(value) != 64
           or any(character not in '0123456789abcdef' for character in value)
           for value in bindings.values()):
        raise ValueError('bindings must contain lowercase SHA256 digests')
    return dict(bindings)


def _model_signature(model):
    return {'kind': getattr(model, 'kind', type(model).__name__),
            'fields': list(getattr(model, 'field_names', [])),
            'geometry': {name: getattr(model, name) for name in
                         ('max_events', 'input_width', 'hidden', 'node_types', 'edge_types')
                         if hasattr(model, name)},
            'state': {name: [list(value.shape), str(value.dtype)]
                      for name, value in model.state_dict().items()}}


def save_checkpoint(path, model, optimizer, state, *, bindings):
    """Write one immutable checkpoint with exact bindings and full replay state."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('checkpoint paths are immutable; choose a fresh path')
    payload = {'schema': 'HiergraphCheckpoint@1', 'bindings': _bindings(bindings),
               'model_signature': _model_signature(model), 'model': model.state_dict(),
               'optimizer_class': type(optimizer).__module__ + '.' + type(optimizer).__name__,
               'optimizer': optimizer.state_dict(), 'state': deepcopy(state),
               'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'training': model.training}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        # A hard link publishes without overwriting a concurrently created path.
        path.hardlink_to(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def load_checkpoint(path, model, optimizer, *, bindings, expected_sha256=None):
    """Validate before mutation; restore model, optimizer, CPU/CUDA/Python RNG."""
    path = Path(path)
    expected = _bindings(bindings)
    if expected_sha256 is not None and sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('checkpoint SHA256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'HiergraphCheckpoint@1' or payload.get('bindings') != expected:
        raise ValueError('checkpoint schema or source/config/data bindings mismatch')
    if payload.get('model_signature') != _model_signature(model):
        raise ValueError('checkpoint model contract mismatch')
    if payload.get('optimizer_class') != type(optimizer).__module__ + '.' + type(optimizer).__name__:
        raise ValueError('checkpoint optimizer class mismatch')
    if payload['cuda_rng'] and (not torch.cuda.is_available()
                                or len(payload['cuda_rng']) != torch.cuda.device_count()):
        raise ValueError('checkpoint CUDA RNG topology unavailable')
    old_model, old_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    old_python, old_torch = random.getstate(), torch.get_rng_state()
    old_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    old_flags = [(module, module.training) for module in model.modules()]
    try:
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        random.setstate(payload['python_rng'])
        torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        model.train(payload['training'])
    except Exception:
        model.load_state_dict(old_model, strict=True)
        optimizer.load_state_dict(old_optimizer)
        random.setstate(old_python)
        torch.set_rng_state(old_torch)
        if old_cuda:
            torch.cuda.set_rng_state_all(old_cuda)
        for module, training in old_flags:
            module.training = training
        raise
    return deepcopy(payload['state'])
