from fractions import Fraction

import pytest

from scripts import audit_raw_time_coverage as audit


def row(label='C', onset='0', slot='b', valid=True, available=True, known=True):
    return {'entity': 'e', 'label': label, 'onset': onset, 'slot': slot,
            'valid': valid, 'source_available': available, 'known': known,
            'masked': not available, 'mapped': valid}


def grids(*values):
    return {p: frozenset(Fraction(x) for x in values) for p in audit.POLICIES}


def test_unmapped_available_cadence_is_in_denominator_and_recovered():
    c, variants, _, missing, recovered, _ = audit.coverage_rows(
        [row('HC', '1/2', valid=False)], grids('1/2'))
    assert c['eligible_rows'] == 1 and c['historical_valid_rows'] == 0
    assert variants[audit.POLICIES[0]]['newly_covered_rows'] == 1
    assert recovered[0]['covered_by'] == list(audit.POLICIES)
    assert missing == []


def test_source_mask_and_unknown_class_do_not_become_labels():
    c, v, classes, *_ = audit.coverage_rows([
        row('HC', valid=False, available=False), row('unknown', valid=False, known=False)], grids('0'))
    assert c['eligible_rows'] == 0 and c['unknown_available_rows'] == 1
    assert v[audit.POLICIES[0]].get('covered_rows', 0) == 0
    assert classes == {}


def test_exact_membership_preserves_simultaneous_disagreement():
    rows = [row('C'), row('D', '1/3'), row('E', '1/3')]
    c, v, _, missing, _, conflicts = audit.coverage_rows(rows, grids('0', '1/3'))
    assert c['ideal_exact_errors'] == 1
    assert len(conflicts) == 1 and missing == []
    m = v[audit.POLICIES[0]]
    assert m['historical_kept_old_slot_errors'] == 2
    assert m['historical_kept_exact_errors'] == 1
    assert m['temporal_errors_removed_on_kept_rows'] == 1


def test_dropping_historical_rows_is_reported_not_claimed_as_resolution():
    c, v, _, missing, *_ = audit.coverage_rows([row(), row('D', '1/3')], grids('0'))
    m = v[audit.POLICIES[0]]
    assert c['eligible_rows'] == 2
    assert m['covered_rows'] == m['historical_lost_rows'] == 1
    assert m['temporal_errors_removed_on_kept_rows'] == 0
    assert missing[0]['onset'] == '1/3'


def test_policy_specific_membership_counts_classes_and_times():
    candidates = grids('0')
    candidates[audit.POLICIES[-1]] = frozenset([Fraction(0), Fraction(1, 2)])
    c, v, classes, missing, *_ = audit.coverage_rows([row(), row(), row('D', '1/2')], candidates)
    assert c['eligible_rows'] == 3 and c['eligible_times'] == 2
    assert v[audit.POLICIES[0]]['covered_rows'] == 2
    assert v[audit.POLICIES[-1]]['covered_rows'] == 3
    assert classes['D'][audit.POLICIES[-1] + '_covered_rows'] == 1
    assert audit.POLICIES[0] + '_covered_rows' not in classes['D']
    assert not missing


def test_joint_does_not_drop_old_unmapped_component_and_keeps_masks():
    projected = {h: [row(label=h)] for h in audit.previous.JOINT}
    projected[audit.previous.JOINT[1]][0]['valid'] = False
    projected[audit.previous.JOINT[1]][0]['mapped'] = False
    joint = audit.eligible_joint(projected)
    assert len(joint) == 1 and not joint[0]['valid']
    projected[audit.previous.JOINT[1]][0]['source_available'] = False
    assert not audit.eligible_joint(projected)


def test_joint_refuses_mismatched_exact_times():
    projected = {h: [row(label=h)] for h in audit.previous.JOINT}
    projected[audit.previous.JOINT[1]][0]['onset'] = '1/2'
    with pytest.raises(ValueError, match='joint identity/time'):
        audit.eligible_joint(projected)


@pytest.mark.parametrize('time', ['0.5', '2/4', 0.5])
def test_target_time_must_be_canonical(time):
    with pytest.raises(ValueError, match='noncanonical target time'):
        audit.coverage_rows([row(onset=time)], grids('1/2'))


def test_note_heads_are_not_temporal_coverage_heads():
    assert set(audit.NOTE_HEADS) == {'metrical_strength', 'note_degree', 'chord_tone', 'is_root', 'is_bass'}
    assert len(audit.TIME_HEADS) == 14  # 12 harmonic + cadence + joint diagnostic


def test_baseline_gate_detects_disappearing_nonzero_counter():
    with pytest.raises(ValueError, match='historical projection changed'):
        audit.verify_baseline_counts({'valid_rows': 2}, {'valid_rows': 2, 'conflict_slots': 1})
    audit.verify_baseline_counts({'valid_rows': 2}, {'valid_rows': 2, 'conflict_slots': 0, 'pieces': 1})


@pytest.mark.parametrize('exclude,expected', [((), 0), ((0,), 1), ((0, 1, 2, 3), None)])
def test_policy_selection(exclude, expected):
    variants = {p: {h: {} for h in audit.TIME_HEADS} for p in audit.POLICIES}
    sizes = {p: i + 1 for i, p in enumerate(audit.POLICIES)}
    for i in exclude:
        variants[audit.POLICIES[i]]['cadence']['uncovered_rows'] = 1
    _, selected = audit.select_policy(variants, sizes)
    assert selected == (audit.POLICIES[expected] if expected is not None else None)


def test_policy_cannot_drop_old_labels_to_be_cheaper():
    variants = {p: {h: {} for h in audit.TIME_HEADS} for p in audit.POLICIES}
    variants[audit.POLICIES[0]]['root']['historical_lost_rows'] = 1
    _, selected = audit.select_policy(variants, {p: i for i, p in enumerate(audit.POLICIES)})
    assert selected == audit.POLICIES[1]
