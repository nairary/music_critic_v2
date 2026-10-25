import pytest
from scripts.prepare_hcm1_diagnostic import selected_authored_rows, generation_ids


def test_explicit_variant_selection_keeps_reserve_closed():
    row=dict(pair_id='a',split='head_dev',variant_id='H2')
    plan=dict(pair_ids=['a'],variant_ids=['H2'])
    assert selected_authored_rows({'pairs':[row]},plan)==[row]
    with pytest.raises(ValueError):selected_authored_rows({'pairs':[dict(row,split='final')]},plan)
    with pytest.raises(ValueError):selected_authored_rows({'pairs':[row]},dict(plan,pair_ids=['a','a']))
    with pytest.raises(ValueError):selected_authored_rows({'pairs':[row]},dict(plan,variant_ids=['H1']))


def test_fixed_generation_ids_preserve_unsupported_slots_and_reject_train():
    cases=[dict(id='a',kind='restoration',split='validation',supported=False),
           dict(id='b',kind='restoration',split='train',supported=True)]
    assert generation_ids({'generation_ids':['a']},cases)==['a']
    for ids in (['b'],['missing'],['a','a']):
        with pytest.raises(ValueError):generation_ids({'generation_ids':ids},cases)
