"""Coverage-aware metrics for partially annotated harmonic-event discovery."""
from collections import defaultdict
from fractions import Fraction
import math

from .matching import (as_list, compatible, event_mask, interval_iou, intervals,
                       match_events, prediction_tie_keys)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _macro_f1(pairs):
    """Macro over the union of observed true and predicted labels, no unknowns."""
    classes = sorted({value for pair in pairs for value in pair})
    if not classes:
        return None
    values = []
    for label in classes:
        tp = sum(truth == label and predicted == label for truth, predicted in pairs)
        fp = sum(truth != label and predicted == label for truth, predicted in pairs)
        fn = sum(truth == label and predicted != label for truth, predicted in pairs)
        values.append(2 * tp / (2 * tp + fp + fn))
    return sum(values) / len(values)


def _subset(data, selected):
    result = dict(data)
    for name in ('spans', 'object_logits', 'ids', 'view_ids', 'scope_ids', 'event_mask'):
        if name in result:
            rows = as_list(result[name])
            result[name] = [rows[i] for i in selected]
    if 'field_logits' in result:
        result['field_logits'] = {name: [rows[i] for i in selected]
                                  for name, value in result['field_logits'].items()
                                  for rows in [as_list(value)]}
    if 'fields' in result:
        result['fields'] = {name: {key: [as_list(rows)[i] for i in selected]
                                 for key, rows in field.items() if key in ('values', 'mask')}
                            for name, field in result['fields'].items()}
    return result


def rank_predictions(pred, duration_quarters, *, candidates_per_quarter=4,
                     max_predictions=None):
    """Select a raw-duration budget; no target count or label enters ranking."""
    duration, rate = float(duration_quarters), float(candidates_per_quarter)
    if not math.isfinite(duration) or duration < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError('duration must be finite/nonnegative and rate finite/positive')
    if max_predictions is not None and (type(max_predictions) is not int or max_predictions < 0):
        raise ValueError('max_predictions must be a nonnegative integer')
    spans = intervals(pred['spans'])
    scores = as_list(pred['object_logits'])
    if len(scores) != len(spans) or any(not math.isfinite(value) for value in scores):
        raise ValueError('object logits must be finite and aligned')
    budget = math.ceil(Fraction(str(duration_quarters)) * Fraction(str(candidates_per_quarter)))
    if max_predictions is not None:
        budget = min(budget, max_predictions)
    tie_keys = prediction_tie_keys(pred)
    order = sorted(range(len(spans)), key=lambda i: (-scores[i], spans[i], tie_keys[i]))
    return order[:budget], budget


def _complete_precision(pred, targets, threshold):
    # A blanket coverage boolean is insufficient. Complete scope intervals are
    # explicit metadata authenticated by the loader, not inferred from counts.
    complete = targets.get('complete_scopes', [])
    if not complete:
        return {'precision': None, 'recall_complete': None, 'f1': None,
                'complete_predictions': 0, 'complete_known_h': 0, 'complete_matches': 0}
    scoped_pred = dict(pred)
    for key in ('view_id', 'scope_id'):
        scoped_pred.setdefault(key, targets.get(key))
    p_spans, t_spans = intervals(pred['spans']), intervals(targets['spans'])
    mask = event_mask(targets)
    valid_regions = []
    for region in complete:
        if not isinstance(region, dict) or region.get('complete') is not True:
            continue
        span = intervals([region['span']])[0]
        domain = {key: region.get(key, targets.get(key)) for key in ('view_id', 'scope_id')}
        # Unknown events inside an asserted scope invalidate its completeness.
        if any(not mask[i] and compatible(domain, targets, 0, i)
               and interval_iou(span, item) > 0 for i, item in enumerate(t_spans)):
            continue
        blocked = False
        for item in targets.get('ambiguous_spans', []):
            other = item['span'] if isinstance(item, dict) else item
            other_domain = ({key: item.get(key, targets.get(key)) for key in ('view_id', 'scope_id')}
                            if isinstance(item, dict) else targets)
            if compatible(domain, other_domain, 0, 0) and interval_iou(span, other) > 0:
                blocked = True
                break
        if not blocked:
            valid_regions.append((span, domain))
    p_selected = [i for i, (start, end) in enumerate(p_spans)
                  if any(left <= start < end <= right and compatible(scoped_pred, domain, i, 0)
                         for (left, right), domain in valid_regions)]
    t_selected = [i for i, (start, end) in enumerate(t_spans) if mask[i]
                  and any(left <= start < end <= right and compatible(domain, targets, 0, i)
                          for (left, right), domain in valid_regions)]
    matched = len(match_events(_subset(scoped_pred, p_selected), _subset(targets, t_selected),
                               mode='metric', iou_threshold=threshold))
    precision, recall = _ratio(matched, len(p_selected)), _ratio(matched, len(t_selected))
    return {'precision': precision, 'recall_complete': recall,
            'f1': _ratio(2 * matched, len(p_selected) + len(t_selected)),
            'complete_predictions': len(p_selected), 'complete_known_h': len(t_selected),
            'complete_matches': matched}


def evaluate_piece(pred, targets, *, duration_quarters, candidates_per_quarter=4,
                   max_predictions=None, iou_threshold=0.5, raw_candidates=None,
                   function_field='H.function', span_scale_quarters=None, matcher=None):
    """Evaluate one selected domain; all spans share the same coordinate system.

    The budget uses core raw duration; span_scale_quarters can separately name
    the halo duration represented by normalized coordinates. Raw
    candidate ceiling is optional, independent of ranking, and does not claim a
    discovery ceiling for continuous endpoint constructors. Missing support is
    represented by None. Complete precision requires explicit complete_scopes.
    """
    selected, budget = rank_predictions(pred, duration_quarters,
                                        candidates_per_quarter=candidates_per_quarter,
                                        max_predictions=max_predictions)
    ranked = _subset(pred, selected)
    matches = (matcher or match_events)(ranked, targets, mode='metric', iou_threshold=iou_threshold)
    mask = event_mask(targets)
    admitted = sum(mask)
    outside = targets.get('coverage', {}).get('outside_halo', 0)
    outside_fields = targets.get('coverage', {}).get('outside_halo_known_fields', {})
    if type(outside) is not int or outside < 0 or any(type(v) is not int or not 0 <= v <= outside
                                                   for v in outside_fields.values()):
        raise ValueError('outside-halo coverage counts must be nonnegative and consistent')
    known = admitted + outside
    fields, f_pairs, f_support, joint_correct = {}, [], 0, 0
    for name, field in targets.get('fields', {}).items():
        values, available = as_list(field['values']), as_list(field['mask'])
        if len(values) != len(mask) or len(available) != len(mask):
            raise ValueError(f'field target alignment mismatch: {name}')
        if any(type(value) is not bool for value in available):
            raise ValueError(f'field target mask must be boolean: {name}')
        support = sum(known_event and available[i] for i, known_event in enumerate(mask)) + outside_fields.get(name, 0)
        logits = ranked.get('field_logits', {}).get(name)
        pairs = []
        if logits is not None:
            for p, t in matches:
                if available[t]:
                    row = logits[p]
                    if not row or any(not math.isfinite(value) for value in row):
                        raise ValueError(f'field logits must be nonempty and finite: {name}')
                    if type(values[t]) is not int or not 0 <= values[t] < len(row):
                        raise ValueError(f'known field target outside frozen vocabulary: {name}')
                    pairs.append((values[t], max(range(len(row)), key=row.__getitem__)))
        correct = sum(truth == prediction for truth, prediction in pairs)
        fields[name] = {'known_support': support, 'matched_support': len(pairs),
                        'joint_correct': correct, 'joint_recovery': _ratio(correct, support),
                        'conditional_macro_f1': _macro_f1(pairs), 'pairs': pairs}
        if name == function_field:
            f_pairs, f_support, joint_correct = pairs, support, correct
    p_spans, t_spans = intervals(ranked['spans']), intervals(targets['spans'])
    errors = [float((abs(p_spans[p][0] - t_spans[t][0]) +
                     abs(p_spans[p][1] - t_spans[t][1])) / 2) for p, t in matches]
    # Duplicates here mean extra predictions eligible for known H but not used
    # by one-to-one matching. They are not automatically false positives.
    used = {p for p, _ in matches}
    duplicate_count = sum(p not in used and any(mask[t] and compatible(ranked, targets, p, t)
                            and interval_iou(span, target_span) >= Fraction(str(iou_threshold))
                            for t, target_span in enumerate(t_spans))
                          for p, span in enumerate(p_spans))
    ceiling = (len(match_events(raw_candidates, targets, mode='metric', iou_threshold=iou_threshold))
               if raw_candidates is not None else None)
    scale = float(duration_quarters if span_scale_quarters is None else span_scale_quarters)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('span coordinate scale must be finite and nonnegative')
    result = {'family_id': targets.get('family_id'), 'work_id': targets.get('work_id'),
              'view_id': targets.get('view_id'), 'scope_id': targets.get('scope_id'),
              'budget': budget, 'emitted_candidates': len(pred['spans']),
              'selected_predictions': len(selected), 'selected_indices': selected,
              'known_h': known, 'known_h_admitted': admitted, 'outside_halo_h': outside,
              'admitted_known_h_recovery': _ratio(len(matches), admitted),
              'excluded_h': len(mask) - admitted,
              'recovered_h': len(matches), 'known_h_recovery': _ratio(len(matches), known),
              'known_f': f_support, 'joint_h_f_correct': joint_correct,
              'joint_h_f_recovery': _ratio(joint_correct, f_support),
              'conditional_f_macro_f1': _macro_f1(f_pairs), 'conditional_f_pairs': f_pairs,
              'boundary_error_normalized': _ratio(sum(errors), len(errors)),
              'boundary_error_quarters': _ratio(sum(errors) * scale, len(errors)),
              'boundary_error_sum_normalized': sum(errors), 'boundary_error_support': len(errors),
              'raw_candidate_recovered_h': ceiling,
              'raw_candidate_ceiling': _ratio(ceiling, known) if ceiling is not None else None,
              'duplicate_known_h_candidates': duplicate_count,
              'duplicate_rate': _ratio(duplicate_count, duplicate_count + len(matches)),
              'fields': fields, 'matches': [(selected[p], t) for p, t in matches],
              'coverage': dict(targets.get('coverage', {})),
              'coverage_exclusions': targets.get('coverage_exclusions', {})}
    result.update(_complete_precision(ranked, targets, iou_threshold))
    return result


def aggregate_metrics(reports):
    """Primary macro across independent families, pooling their piece counts."""
    groups = defaultdict(list)
    for report in reports:
        if not isinstance(report.get('family_id'), str) or not report['family_id']:
            raise ValueError('family_id is required for family macro aggregation')
        groups[report['family_id']].append(report)
    families = {}
    for family, rows in sorted(groups.items()):
        known, recovered = sum(row['known_h'] for row in rows), sum(row['recovered_h'] for row in rows)
        known_f, joint = sum(row['known_f'] for row in rows), sum(row['joint_h_f_correct'] for row in rows)
        pairs = [pair for row in rows for pair in row['conditional_f_pairs']]
        families[family] = {'known_h': known, 'recovered_h': recovered,
                            'known_h_recovery': _ratio(recovered, known),
                            'known_f': known_f, 'joint_h_f_correct': joint,
                            'joint_h_f_recovery': _ratio(joint, known_f),
                            'conditional_f_macro_f1': _macro_f1(pairs)}
    summary = {'families': families, 'family_count': len(families)}
    for key in ('known_h_recovery', 'joint_h_f_recovery', 'conditional_f_macro_f1'):
        values = [row[key] for row in families.values() if row[key] is not None]
        summary['family_macro_' + key] = _ratio(sum(values), len(values))
        summary['supported_families_' + key] = len(values)
    return summary
