"""Exact-scope membership targets, kept separate from observable music."""
from collections import Counter
from fractions import Fraction

from music_critic.relation_probe.data import exact as _exact, pair, view


def exact(value):
    return value if isinstance(value, Fraction) else _exact(value)


def extract_memberships(bundle):
    if bundle.get('current_split') != 'train':
        raise ValueError('TRAIN bundle required')
    rows = {}
    counts = Counter()
    for h in bundle['harmonies']:
        if h['id'] in rows:
            raise ValueError('duplicate H identity')
        a, b = exact(h['start_qn']), exact(h['end_qn'])
        if a >= b:
            counts['empty_h'] += 1
            continue
        rows[h['id']] = dict(id=h['id'], start=pair(a), end=pair(b),
            view_id=view(h), labels=[], participant_scope=dict(kind='all_observed_tracks',
                provenance='candidate_domain_assumption; source_H_participant_scope_not_established'))
    for c in bundle['note_contexts']:
        h = rows.get(c.get('harmony_id'))
        if h is None:
            counts['unbound_context'] += 1
            continue
        if view(c) != h['view_id']:
            counts['cross_view'] += 1
            continue
        a, b = exact(c['start_qn']), exact(c['end_qn'])
        if not exact(h['start']) <= a <= b <= exact(h['end']):
            raise ValueError('context outside H')
        if a == b:
            counts['empty_context'] += 1
            continue
        f = c.get('fields', {}).get('analysis.chord_membership', {})
        known = (f.get('available') is True and type(f.get('value')) is bool
            and f.get('annotation_state', 'known') == 'known')
        # Unknown overlaps also mask a candidate, even if another annotation is known.
        value = f['value'] if known else None
        h['labels'].append(dict(note_id=c['note_id'], start=pair(a), end=pair(b),
            value=value, origin=f.get('origin'), confidence=f.get('confidence')))
        counts['positive' if value is True else 'negative' if value is False else 'unknown'] += 1
    counts['h'] = len(rows)
    counts['mixed_h_annotation'] = sum({x['value'] for x in h['labels'] if x['value'] is not None} == {True, False} for h in rows.values())
    return list(rows.values()), dict(counts)


def candidate_labels(h, candidates):
    """A candidate is known only under complete, consistent interval coverage."""
    by_note = {}
    for label in h['labels']:
        by_note.setdefault(label['note_id'], []).append(label)
    ys, masks = [], []
    for note_id, left, right in candidates:
        a, b = exact(left), exact(right)
        if not exact(h['start']) <= a < b <= exact(h['end']):
            raise ValueError('candidate outside H or empty')
        overlaps = [x for x in by_note.get(note_id, [])
            if exact(x['start']) < b and a < exact(x['end'])]
        values = {x['value'] for x in overlaps}
        known = len(values) == 1 and None not in values
        covered = a
        for x in sorted(overlaps, key=lambda x: exact(x['start'])):
            l, r = max(a, exact(x['start'])), min(b, exact(x['end']))
            if l > covered:
                break
            covered = max(covered, r)
        known = known and covered == b
        ys.append(int(next(iter(values))) if known else 0)
        masks.append(known)
    return ys, masks
