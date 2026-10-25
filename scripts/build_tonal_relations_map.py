"""Convert a reviewed source sidecar to explicit objects without inferring theory.

Only the supplied events are visible. Simultaneity is mechanical; a sonority
slice is not automatically a complete chord or a harmonic-function label.
An optional case-keyed plan supplies explicitly sourced larger objects.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from scripts.tonal_relations_review import validate as validate_review
from music_critic.analysis.alignment import select_overlapping_events


LEVELS = ('note', 'sonority', 'sequence', 'phrase', 'section', 'part', 'piece')


def rational(value: Fraction) -> str:
    return f'{value.numerator}/{value.denominator}'


def build_map(source: dict, plan: dict) -> dict:
    """Preserve source claims verbatim; add mechanical groups and explicit plan.

    Plan = {case_id: {objects: [...], coverage_reasons: {kind: text}}}.
    No part, phrase, section or piece is guessed from track/voice metadata.
    The caller validates the result using tonal_relations_map.validate.
    """
    validate_review(source)
    if not isinstance(plan, dict) or set(plan) - {c['id'] for c in source['cases']}:
        raise ValueError('plan references unknown cases')
    result = copy.deepcopy(source)
    result['schema'] = 'tonal-relations-map@1'
    for case in result['cases']:
        events = case['events']
        by_id = {e['id']: e for e in events}
        objects, relations = [], []
        boundaries = sorted({t for e in events if Fraction(e['duration']) > 0
                             for t in (Fraction(e['start']), Fraction(e['start']) + Fraction(e['duration']))})
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            refs = select_overlapping_events(events, rational(start), rational(end))
            if len(refs) >= 2:
                pitches = sorted(by_id[ref]['pitch'] for ref in refs)
                objects.append({'id': f'obj:slice:{i}', 'kind': 'sonority',
                                'label': 'Одновременные ноты MIDI ' + ', '.join(map(str, pitches)),
                                'start': rational(start), 'end': rational(end), 'event_ids': refs,
                                'basis': 'Механический срез только доступных событий по границам атак и окончаний. '
                                         'Полнота аккорда и инструментов не установлена; бас — минимальная доступная высота.'})
        if case['membership'] != 'excluded':
            for claim in case['claims']:
                participants = []
                for i, participant in enumerate(claim['participants']):
                    refs = participant['event_ids']
                    selected = [by_id[ref] for ref in refs]
                    starts = {Fraction(e['start']) for e in selected}
                    if len(refs) > 1 and len(starts) > 1 and all(Fraction(e['duration']) > 0 for e in selected):
                        object_id = f'obj:claim:{claim["id"]}:{i}'
                        objects.append({'id': object_id, 'kind': 'sequence', 'label': participant['role'],
                                        'start': rational(min(starts)),
                                        'end': rational(max(Fraction(e['start']) + Fraction(e['duration']) for e in selected)),
                                        'event_ids': refs[:], 'basis': 'Группа участников исходного утверждения; '
                                        'не независимая сегментация и не утверждение о непрерывности линии.'})
                        participants.append({'role': participant['role'], 'event_ids': [], 'object_ids': [object_id]})
                    else:
                        participants.append({'role': participant['role'], 'event_ids': refs[:], 'object_ids': []})
                context_fields = [claim[field] for field in ('context_event_ids', 'source_window_event_ids')
                                  if field in claim]
                context = list(dict.fromkeys(ref for refs in context_fields for ref in refs)) if context_fields else [e['id'] for e in events]
                relations.append({'id': 'rel:' + claim['id'], 'kind': claim['relation'],
                                  'level': 'cross_level', 'claim_id': claim['id'], 'participants': participants,
                                  'context_event_ids': context,
                                  'alignment': 'exact_event_ids' if participants else 'unavailable',
                                  'confidence': None, 'target_available': False})
        case_plan = plan.get(case['id'], {})
        if not isinstance(case_plan, dict) or set(case_plan) - {'objects', 'coverage_reasons'}:
            raise ValueError('invalid case plan')
        extras = case_plan.get('objects', [])
        if not isinstance(extras, list) or case['membership'] == 'excluded' and extras:
            raise ValueError('invalid or excluded extra objects')
        objects.extend(copy.deepcopy(extras))
        reasons = case_plan.get('coverage_reasons', {})
        if not isinstance(reasons, dict) or set(reasons) - set(LEVELS):
            raise ValueError('unknown coverage level')
        case['objects'], case['relations'] = objects, relations
        case['coverage'] = {}
        for level in LEVELS:
            present = bool(events) if level == 'note' else any(o['kind'] == level for o in objects)
            case['coverage'][level] = {
                'status': ('observed' if level == 'note' else 'partial') if present else 'unavailable',
                'reason': reasons.get(level, 'Показаны только явно сохранённые события/группы выбранных окон.'
                                     if present else 'Для этого уровня нет привязанной разметки в выбранных источниках.')}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    payload = args.source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.source_sha256:
        raise ValueError('source SHA-256 mismatch; source was not decoded')
    result = build_map(json.loads(payload), json.loads(args.plan.read_bytes()))
    from scripts.tonal_relations_map import validate
    summary = validate(result)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
