"""Fixed-query conditional scores with coverage, never implicit quality labels."""
from __future__ import annotations

import torch

from .contracts import MaskQuery
from .masking import split_masked_piece
from .observed_view import UnsupportedContextError


def _query_json(query):
    return dict(start=[query.start.num, query.start.den], end=[query.end.num, query.end.den],
                track_ids=list(query.track_ids))


@torch.no_grad()
def score_piece(model, piece, queries: tuple[MaskQuery, ...], *, allow_validation=False):
    """Evaluate the exact supplied schedule; unsupported blocks keep their slot.

    Library calls on real data still require an experiment record at the caller.
    Raw unlabeled inference (split=None) is supported. TEST is fail-closed.
    """
    split = (piece.split or '').upper()
    if split == 'TEST':
        raise ValueError('pll.TEST_access_closed')
    if split in ('VALIDATION', 'VAL', 'DEV') and not allow_validation:
        raise ValueError('pll.validation_access_not_declared')
    if split not in ('', 'TRAIN', 'VALIDATION', 'VAL', 'DEV'):
        raise ValueError('pll.unknown_split')
    if model.training:
        raise ValueError('pll.scoring_requires_eval_mode')
    if not isinstance(queries, tuple) or not queries:
        raise ValueError('pll.fixed_nonempty_schedule_required')
    blocks = []
    for query in queries:
        row = dict(query=_query_json(query), supported=False, nll=None,
                   nats_per_event=None, nats_per_qn=None, target_events=None, reason=None)
        try:
            example = (model.split(piece, query) if callable(getattr(model, 'split', None))
                       else split_masked_piece(piece, query, model.context_profile))
            try:
                model.codec.encode(example.targets, span=example.span, tracks=example.tracks)
            except ValueError as exc:
                if str(exc).startswith('unsupported'):
                    raise UnsupportedContextError(str(exc)) from exc
                raise
            value = model(example)
            nll = float(value.nll)
            row.update(supported=True, nll=nll, target_events=value.event_count,
                       nats_per_event=nll / value.event_count if value.event_count else None,
                       nats_per_qn=nll / float(query.span.to_fraction()))
        except UnsupportedContextError as exc:
            row['reason'] = str(exc)
        blocks.append(row)
    return dict(schema='HCM1BlockScores@1.0.0', requested=len(queries),
                supported=sum(b['supported'] for b in blocks), blocks=blocks,
                interpretation='conditional_symbolic_likelihood_not_aesthetic_quality',
                aggregation='none; overlapping block probabilities are not piece likelihood')
