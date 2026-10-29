"""Matched H localization and independent masked categorical auxiliary losses."""
import math

from .matching import event_mask, known_negative_indices, match_events


def compute_loss(pred, targets, weights=None, *, matched_events=None):
    """Compute loss without treating absent or unmatched annotations as negatives.

    Components are flat scalar tensors: localization, object_positive,
    object_negative, boundary_positive, and field:<name>. Available field means are averaged across
    heads. ``weights`` optionally overrides those component weights (default 1).
    The return also includes actual per-component support and detached assignment.
    Positive-only objectness learns confidence on known H, not calibrated absence.
    """
    import torch
    import torch.nn.functional as functional

    spans, objects = pred['spans'], pred['object_logits']
    fields = pred.get('field_logits', {})
    if spans.ndim != 2 or spans.shape[1] != 2 or objects.shape != (len(spans),):
        raise ValueError('prediction spans/logits shape mismatch')
    if not spans.is_floating_point() or not objects.is_floating_point():
        raise ValueError('prediction tensors must be floating point')
    tensors = [spans, objects, *fields.values()]
    if 'boundary_logits' in pred:
        tensors.append(pred['boundary_logits'])
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError('non-finite prediction tensor')
    for name, logits in fields.items():
        if logits.ndim != 2 or len(logits) != len(spans) or logits.shape[1] < 1:
            raise ValueError(f'field logits shape mismatch: {name}')
    components = {'localization': None, 'object_positive': None, 'object_negative': None,
                  'boundary_positive': None}
    components.update({f'field:{name}': None for name in fields})
    weights = dict(weights or {})
    if set(weights) - set(components):
        raise ValueError('unknown loss component weight')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('loss weights must be finite and nonnegative')
    zero = sum(value.sum() * 0 for value in tensors)
    components = {name: zero for name in components}
    matches = match_events(pred, targets, mode='training') if matched_events is None else list(matched_events)
    if (len({p for p,t in matches}) != len(matches) or len({t for p,t in matches}) != len(matches)
        or any(type(p) is not int or type(t) is not int or not 0<=p<len(spans) or not 0<=t<len(targets['spans']) or not event_mask(targets)[t] for p,t in matches)):
        raise ValueError('precomputed event assignment must be valid and one-to-one')
    positive = [p for p, _ in matches]
    target_indices = [t for _, t in matches]
    # Low-overlap localization may assign a currently misplaced query to H.
    # Its positive assignment takes precedence over empty-region geometry.
    negative = [index for index in known_negative_indices(pred, targets) if index not in positive]
    support = {'known_events': sum(event_mask(targets)), 'matched_events': len(matches),
               'object_positive': len(positive), 'object_negative': len(negative),
               'boundary_positive': 0,
               'fields': {name: 0 for name in fields},
               'unmodeled_fields': sorted(set(targets.get('fields', {})) - set(fields))}
    if 'boundary_logits' in pred:
        boundary = pred['boundary_logits']
        if boundary.ndim != 1:
            raise ValueError('boundary_logits must be an interior-grid vector')
        # Only the exact raw adapter can assert canonical endpoint membership.
        # Full-grid indices 0 and T-1 are fixed outer boundaries; the E2 head
        # predicts indices 1..T-2. There is no float-nearest fallback or implicit
        # negative at unknown grid positions.
        full_indices = as_boundary_indices(targets, len(boundary) + 2)
        interior = sorted({index - 1 for index in full_indices if 0 < index < len(boundary) + 1})
        support['boundary_positive'] = len(interior)
        support['boundary_targets_missing'] = 'known_boundary_indices' not in targets
        support['boundary_endpoint_coverage'] = dict(targets.get('boundary_endpoint_coverage', {}))
        if interior:
            components['boundary_positive'] = functional.binary_cross_entropy_with_logits(
                boundary[interior], torch.ones_like(boundary[interior]))
    if matches:
        gold = torch.as_tensor(targets['spans'], dtype=spans.dtype, device=spans.device)
        components['localization'] = functional.smooth_l1_loss(
            spans[positive], gold[target_indices], reduction='none').sum(dim=-1).mean()
        components['object_positive'] = functional.binary_cross_entropy_with_logits(
            objects[positive], torch.ones_like(objects[positive]))
    if negative:
        components['object_negative'] = functional.binary_cross_entropy_with_logits(
            objects[negative], torch.zeros_like(objects[negative]))
    active_fields = []
    for name, logits in fields.items():
        if name not in targets.get('fields', {}):
            continue
        field = targets['fields'][name]
        mask = torch.as_tensor(field['mask'], device=logits.device)
        values = torch.as_tensor(field['values'], device=logits.device)
        if mask.numel() == values.numel() == len(targets['spans']) == 0:
            mask, values = mask.bool(), values.long()
        if mask.dtype != torch.bool or mask.shape != (len(targets['spans']),):
            raise ValueError(f'field mask must be boolean and aligned: {name}')
        if values.shape != mask.shape or values.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'field values must be integer and aligned: {name}')
        selected = [(p, t) for p, t in matches if bool(mask[t])]
        support['fields'][name] = len(selected)
        if selected:
            p_indices, t_indices = zip(*selected)
            labels = values[list(t_indices)].long()
            if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
                raise ValueError(f'known field target outside frozen vocabulary: {name}')
            key = f'field:{name}'
            components[key] = functional.cross_entropy(logits[list(p_indices)], labels)
            active_fields.append(key)
    loss = zero
    for key in ('localization', 'object_positive', 'object_negative', 'boundary_positive'):
        loss = loss + weights.get(key, 1.0) * components[key]
    for key in active_fields:
        loss = loss + weights.get(key, 1.0) * components[key] / len(active_fields)
    return {'loss': loss, 'components': components, 'support': support, 'matches': matches}


def as_boundary_indices(targets, grid_size):
    from .matching import as_list
    values = as_list(targets.get('known_boundary_indices', []))
    if any(type(value) is not int or not 0 <= value < grid_size for value in values):
        raise ValueError('known boundary indices must be exact full-grid integer indices')
    if values and not any(event_mask(targets)):
        raise ValueError('known boundaries require eligible H supervision')
    return values
