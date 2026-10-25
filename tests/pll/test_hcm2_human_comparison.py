from dataclasses import replace
from pathlib import Path

import pytest

from scripts.hcm2_human_comparison import (
    aggregate_comparison, context_identity, expanded_query, join_comparison,
    preflight_cases, read_bound,
)
from music_critic.data import load_piece
from music_critic.pll.contracts import MaskQuery
from music_critic.pll.masking import split_masked_piece


def test_empty_targets_and_nondirectional_labels_do_not_become_errors():
    ratings = [dict(id=k, work_id='w', rating=v) for k, v in
               [('a', 'worse'), ('b', 'same'), ('c', 'unsure')]]
    scores = [dict(id=k, supported=True, delta_nll=-3., delta_nats_per_event=None)
              for k in ('a', 'b', 'c')]
    rows = join_comparison(ratings, {'new': scores})
    assert rows[0]['models']['new']['raw_agrees'] is False
    assert rows[0]['models']['new']['per_event_agrees'] is None
    assert rows[1]['models']['new']['raw_agrees'] is None
    summary = aggregate_comparison(rows)
    assert summary['models']['new']['raw_agrees'] == dict(correct=0, denominator=1)
    assert summary['models']['new']['per_event_agrees'] == dict(correct=0, denominator=0)


def test_comparison_requires_exact_ids_and_separates_expanded_support():
    ratings = [dict(id=k, work_id='w', rating='worse') for k in ('a', 'b', 'c')]
    old = [dict(id=k, supported=True, delta_nll=1., delta_nats_per_event=1.) for k in ('a', 'b')]
    new = [dict(id=k, supported=True, common_query=k != 'b', delta_nll=1.,
                delta_nats_per_event=1.) for k in ('a', 'b', 'c')]
    rows = join_comparison(ratings, {'old': old, 'new': new})
    assert aggregate_comparison(rows)['pairs'] == 1
    assert aggregate_comparison(rows, common_only=False)['models']['new']['supported'] == 3
    with pytest.raises(ValueError, match='duplicate_rating'):
        join_comparison(ratings * 2, {'old': old})
    with pytest.raises(ValueError, match='duplicate_model_score'):
        join_comparison(ratings, {'old': old * 2})


def test_scope_and_hash_fail_closed(tmp_path):
    with pytest.raises(ValueError, match='forbidden_case_scope'):
        preflight_cases([dict(id='a', kind='pair', split='test', source_scope='train')])
    with pytest.raises(ValueError, match='unsafe_path'):
        read_bound(tmp_path, dict(path='../outside'))
    (tmp_path / 'piece.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='input_integrity'):
        read_bound(tmp_path, dict(path='piece.json', sha256='0' * 64, size_bytes=2))


def test_changed_onset_mask_keeps_identical_predictor_context():
    piece = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    chosen = piece.notes[0]
    edited = replace(piece, notes=tuple(replace(n, pitch=n.pitch + 1)
                                       if n.note_id == chosen.note_id else n for n in piece.notes))
    query = expanded_query(piece, edited)
    assert query.start <= chosen.onset_qn < query.end
    a, b = (split_masked_piece(p, query) for p in (piece, edited))
    assert context_identity(a.context) == context_identity(b.context)
    assert a.targets != b.targets
    with pytest.raises(ValueError, match='no_change'):
        expanded_query(piece, piece)
