from types import SimpleNamespace

from music_critic.data import RationalTime as Q
from scripts.hcm2_human_comparison import expanded_query


def test_expanded_query_encloses_both_moved_onsets_beyond_four_bars():
    note = SimpleNamespace(note_id='n', track_id='t', pitch=60, onset_qn=Q(1),
                           duration_qn=Q(1), velocity=80, channel=0, program=0,
                           is_percussion=False, is_grace=False)
    moved = SimpleNamespace(**{**vars(note), 'onset_qn': Q(25)})
    geometry = dict(bars=tuple(SimpleNamespace(start_qn=Q(4*i), duration_qn=Q(4))
                               for i in range(8)),
                    tracks=(SimpleNamespace(track_id='t'),), tempo_events=())
    original = SimpleNamespace(**geometry, notes=(note,))
    variant = SimpleNamespace(**geometry, notes=(moved,))
    query = expanded_query(original, variant)
    assert query.start == Q(0) and query.end == Q(28)
    assert query.track_ids == ('t',)
