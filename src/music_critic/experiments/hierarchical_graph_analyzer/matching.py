"""Exact interval arithmetic and deterministic, dependency-free H assignment.

Target tensor coordinates may be floating point, but canonical metadata uses
``canonical_time`` and never float equality. Matching is detached supervision;
the loss consumes the original differentiable prediction tensors.
"""
from fractions import Fraction
import math
from numbers import Integral, Real


def canonical_time(value):
    """Read an exact canonical time; floats and malformed rationals are rejected."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return Fraction(int(value))
    if isinstance(value, dict) and set(value) == {'num', 'den'}:
        numerator, denominator = value['num'], value['den']
        if (type(numerator) is int and type(denominator) is int
                and denominator > 0):
            result = Fraction(numerator, denominator)
            if (result.numerator, result.denominator) == (numerator, denominator):
                return result
    raise ValueError('canonical time requires an integer, Fraction, or reduced num/den')


def _number(value):
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
        if not math.isfinite(value):
            raise ValueError('non-finite interval coordinate')
        # A tensor boundary is already approximate; preserve its represented
        # value rather than snapping it to an invented source grid.
        return Fraction(float(value))
    return canonical_time(value)


def as_list(values):
    """Detach tensors without importing torch in the pure matching/metric path."""
    if hasattr(values, 'detach'):
        return values.detach().cpu().tolist()
    return list(values)


def intervals(values):
    result = []
    for row in as_list(values):
        if len(row) != 2:
            raise ValueError('spans must have two coordinates')
        start, end = (_number(value) for value in row)
        if start >= end:
            raise ValueError('H spans must have positive duration')
        result.append((start, end))
    return result


def interval_iou(first, second):
    """IoU of positive-duration half-open intervals, returned as a Fraction."""
    (start, end), (other_start, other_end) = intervals([first, second])
    intersection = max(Fraction(0), min(end, other_end) - max(start, other_start))
    return intersection / (max(end, other_end) - min(start, other_start))


def _metadata(data, key, index, default=None):
    plural = key + 's'
    if plural in data:
        return data[plural][index]
    return data.get(key, default)


def compatible(pred, targets, prediction_index, target_index):
    """Match only within a domain attached outside target-blind forward.

    A scalar target domain selects the current evaluation domain for raw
    predictions. Per-row multi-view targets require explicit prediction domains.
    """
    for key in ('view_id', 'scope_id'):
        target = _metadata(targets, key, target_index)
        predicted = _metadata(pred, key, prediction_index, targets.get(key))
        if predicted != target:
            return False
    return True


def event_mask(targets):
    count = len(targets['spans'])
    values = as_list(targets.get('event_mask', [True] * count))
    if len(values) != count or any(type(value) is not bool for value in values):
        raise ValueError('event_mask must contain one boolean per target')
    return values


def prediction_tie_keys(pred):
    """Target-independent canonical tie keys, stable under slot permutation."""
    count = len(pred['spans'])
    objects = as_list(pred.get('object_logits', [0.] * count))
    fields = {name: as_list(value) for name, value in pred.get('field_logits', {}).items()}
    ids = pred.get('ids', [''] * count)
    if len(objects) != count or len(ids) != count or any(len(rows) != count for rows in fields.values()):
        raise ValueError('prediction metadata/logits must be aligned')
    return [(-float(objects[i]), tuple((name, tuple(float(v) for v in fields[name][i]))
                                      for name in sorted(fields)), str(ids[i]))
            for i in range(count)]


def _assignment(cost):
    """Rectangular Hungarian shortest augmenting paths, n rows <= m columns.

    Exact Fraction costs and stable row/column order make ties reproducible.
    Dummy columns supplied by the caller permit unmatched rows.
    """
    n, m = len(cost), len(cost[0]) if cost else 0
    u, v = [Fraction(0)] * (n + 1), [Fraction(0)] * (m + 1)
    owner, previous = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        owner[0] = row
        column = 0
        best, seen = [None] * (m + 1), [False] * (m + 1)
        while True:
            seen[column] = True
            current, delta, next_column = owner[column], None, 0
            for candidate in range(1, m + 1):
                if seen[candidate]:
                    continue
                reduced = cost[current - 1][candidate - 1] - u[current] - v[candidate]
                if best[candidate] is None or reduced < best[candidate]:
                    best[candidate], previous[candidate] = reduced, column
                if delta is None or best[candidate] < delta:
                    delta, next_column = best[candidate], candidate
            for candidate in range(m + 1):
                if seen[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                elif best[candidate] is not None:
                    best[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            predecessor = previous[column]
            owner[column] = owner[predecessor]
            column = predecessor
    return [(owner[column] - 1, column - 1)
            for column in range(1, m + 1) if owner[column]]


def match_events(pred, targets, *, mode='training', iou_threshold=0.5):
    """Return (prediction, target) indices with maximum eligible cardinality.

    The secondary objective minimizes normalized boundary distance plus 1-IoU.
    Training has no overlap cutoff: bad initial boundaries still learn. Metric
    matching admits only IoU >= the declared positive threshold. Field labels
    and objectness never determine assignment or metric eligibility.
    """
    if mode not in ('training', 'metric'):
        raise ValueError('mode must be training or metric')
    threshold = _number(iou_threshold)
    if not 0 < threshold <= 1:
        raise ValueError('iou_threshold must be in (0, 1]')
    predicted, gold = intervals(pred['spans']), intervals(targets['spans'])
    mask = event_mask(targets)
    ids = targets.get('ids', [str(i) for i in range(len(gold))])
    if len(ids) != len(gold) or len(set(ids)) != len(ids):
        raise ValueError('target ids must be unique and aligned')
    tie_keys = prediction_tie_keys(pred)
    p_order = sorted(range(len(predicted)), key=lambda i: (predicted[i], tie_keys[i]))
    t_order = sorted((i for i in range(len(gold)) if mask[i]),
                     key=lambda i: (gold[i], str(ids[i])))
    if not p_order or not t_order:
        return []
    costs, eligible = [], []
    for target_index in t_order:
        row, allowed = [], []
        start, end = gold[target_index]
        for prediction_index in p_order:
            left, right = predicted[prediction_index]
            iou = interval_iou((left, right), (start, end))
            allowed.append(compatible(pred, targets, prediction_index, target_index)
                           and (mode == 'training' or iou >= threshold))
            row.append((abs(left - start) + abs(right - end)) / (end - start) + 1 - iou)
        costs.append(row)
        eligible.append(allowed)
    # One extra eligible edge dominates every possible secondary-cost change.
    unmatched = (max(max(row) for row in costs) + 1) * (len(t_order) + 1)
    matrix = [[cost if ok else 2 * unmatched for cost, ok in zip(row, allow)]
              + [unmatched] * len(t_order) for row, allow in zip(costs, eligible)]
    return sorted((p_order[column], t_order[row]) for row, column in _assignment(matrix)
                  if column < len(p_order) and eligible[row][column])


def known_negative_indices(pred, targets):
    """License negatives only inside explicit complete compatible empty spans.

    Bare span arrays are deliberately insufficient evidence. All target rows,
    including unknown event rows, and ambiguous intervals block overlapping
    negatives; unmatched candidates otherwise remain unlabeled.
    """
    pred = dict(pred)
    for key in ('view_id', 'scope_id'):
        pred.setdefault(key, targets.get(key))
    predicted, gold = intervals(pred['spans']), intervals(targets['spans'])
    regions = targets.get('known_negative_spans', [])
    if not regions:
        return []
    ambiguous = targets.get('ambiguous_spans', [])
    accepted = []
    for prediction_index, (start, end) in enumerate(predicted):
        blocked = any(compatible(pred, targets, prediction_index, target_index)
                      and interval_iou((start, end), span) > 0
                      for target_index, span in enumerate(gold))
        if blocked:
            continue
        for region in regions:
            if not isinstance(region, dict) or region.get('complete') is not True:
                continue
            domain = {key: region.get(key, targets.get(key)) for key in ('view_id', 'scope_id')}
            if not compatible(pred, domain, prediction_index, 0):
                continue
            left, right = intervals([region['span']])[0]
            if not left <= start < end <= right:
                continue
            unknown_overlap = False
            for unknown in ambiguous:
                span = unknown['span'] if isinstance(unknown, dict) else unknown
                unknown_domain = ({key: unknown.get(key, targets.get(key))
                                   for key in ('view_id', 'scope_id')}
                                  if isinstance(unknown, dict) else targets)
                if (compatible(pred, unknown_domain, prediction_index, 0)
                        and interval_iou((start, end), span) > 0):
                    unknown_overlap = True
                    break
            if not unknown_overlap:
                accepted.append(prediction_index)
                break
    return accepted
