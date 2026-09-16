from fractions import Fraction

import pytest

from scripts.analyze_raw_time_fallback import route_uncovered, with_subdivisions


def row(label, onset, slot='b', valid=True):
    return {'label': label, 'onset': onset, 'slot': slot, 'valid': valid,
            'mapped': valid, 'source_available': True, 'known': True, 'masked': False}


def test_fixed_subdivisions_are_raw_only_and_keep_outside_notes():
    raw = {'beat_boundaries': {Fraction(1, 5), Fraction(11, 5)},
           'note_onsets': {Fraction(-1)}, 'note_offsets': {Fraction(3)}}
    assert with_subdivisions(raw, 2) == {Fraction(-1), Fraction(1, 5), Fraction(6, 5), Fraction(11, 5), Fraction(3)}


def test_fallback_retains_old_targets_but_cannot_remove_all_conflicts():
    c = route_uncovered([row('C', '1/3'), row('D', '2/3')], set(), allow_fallback=True)
    assert c['fallback_rows'] == 2 and c['legacy_beat_unavoidable_errors'] == 1
    c = route_uncovered([row('C', '1/3'), row('D', '2/3')], {Fraction(1, 3)}, allow_fallback=True)
    assert c['additional_fine_rows'] == c['fallback_rows'] == 1
    assert c['legacy_beat_unavoidable_errors'] == c['time_unavoidable_errors'] == 0


def test_cadence_unmapped_is_only_recovered_by_exact_time():
    r = row('HC', '1/3', valid=False)
    assert route_uncovered([r], set(), allow_fallback=False)['unmatched_rows'] == 1
    assert route_uncovered([r], {Fraction(1, 3)}, allow_fallback=False)['additional_fine_rows'] == 1


def test_same_time_disagreement_remains_fine_error():
    c = route_uncovered([row('C', '1/3'), row('D', '1/3')], {Fraction(1, 3)}, allow_fallback=True)
    assert c['time_unavoidable_errors'] == 1


def test_masks_and_undeclared_policy_fail_closed():
    r = row('C', '0'); r['masked'] = True
    with pytest.raises(ValueError): route_uncovered([r], {Fraction(0)}, allow_fallback=True)
    with pytest.raises(ValueError): with_subdivisions({}, 5)


def test_error_decomposition_matches_direct_complete_routing():
    from collections import Counter, defaultdict
    old = [row('C', '0'), row('D', '0')]
    misses = [row('E', '1/3'), row('F', '1/3'), row('G', '1/2'), row('A', '2/3')]
    c = route_uncovered(misses, {Fraction(0), Fraction(1, 3)}, allow_fallback=True)
    routed = defaultdict(Counter)
    for r in old + misses:
        key = ('time', r['onset']) if Fraction(r['onset']) in {Fraction(0), Fraction(1, 3)} else ('beat', r['slot'])
        routed[key][r['label']] += 1
    direct = sum(sum(x.values()) - max(x.values()) for x in routed.values())
    assert direct == 1 + c['time_unavoidable_errors'] + c['legacy_beat_unavoidable_errors'] == 3


def test_old_invalid_harmonic_row_cannot_gain_beat_fallback():
    assert route_uncovered([row('C', '1/3', valid=False)], set(), allow_fallback=True)['unmatched_rows'] == 1
