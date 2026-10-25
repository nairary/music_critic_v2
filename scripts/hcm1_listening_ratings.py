"""Validate human exports and describe agreement without inventing tie labels."""
from collections import Counter
from datetime import datetime
import math


def review_ratings(export, payload):
    if (export.get('schema')!='HCM1HumanListeningRatings@1' or
            export.get('source_receipt_sha256')!=payload['source_receipt_sha256']):
        raise ValueError('ratings source receipt mismatch')
    ratings=export.get('ratings')
    if not isinstance(ratings,dict):raise ValueError('ratings mapping required')
    cases={c['id']:c for c in payload['cases']}
    if len(cases)!=len(payload['cases']):raise ValueError('duplicate case ID')
    rows=[]
    for identifier,entry in ratings.items():
        if identifier not in cases:raise ValueError('unknown rating ID')
        case=cases[identifier];label=entry.get('rating')
        allowed=('better','worse','same','unsure') if case['kind']=='pair' else ('plausible','implausible','unsure')
        if label not in allowed or type(entry.get('score_revealed')) is not bool:
            raise ValueError('rating label or reveal provenance invalid')
        try:
            stamp=datetime.fromisoformat(entry['at'].replace('Z','+00:00'))
            if stamp.tzinfo is None:raise ValueError('timezone required')
        except (KeyError,TypeError,AttributeError,ValueError) as exc:raise ValueError('invalid rating timestamp') from exc
        score=case.get('score',{});raw=score.get('delta_nll') if score.get('supported') else None
        normalized=score.get('delta_nats_per_event') if score.get('supported') else None
        def agrees(delta):
            if case['kind']!='pair' or label not in ('better','worse') or delta is None:return None
            if not math.isfinite(delta):raise ValueError('nonfinite model score')
            return delta>0 if label=='worse' else delta<0
        rows.append(dict(id=identifier,work_id=case['work_id'],kind=case['kind'],rating=label,
            at=entry['at'],score_revealed=entry['score_revealed'],delta_nll=raw,
            delta_nats_per_event=normalized,raw_agrees=agrees(raw),per_event_agrees=agrees(normalized)))
    def counts(key):
        values=[r[key] for r in rows if r[key] is not None]
        return dict(agree=sum(values),total=len(values))
    return dict(source_receipt_sha256=export['source_receipt_sha256'],rows=rows,
        label_counts=dict(Counter(r['rating'] for r in rows)),works=len({r['work_id'] for r in rows}),
        score_hidden_at_rating=sum(not r['score_revealed'] for r in rows),
        directional=dict(raw_nll=counts('raw_agrees'),per_event=counts('per_event_agrees')),
        interpretation='One listener, selected development examples; same/unsure do not define a likelihood threshold')
