import pytest
from scripts.hcm1_listening_ratings import review_ratings


def packet():
    source='a'*64
    rows=[dict(id='p',kind='pair',work_id='work',score=dict(supported=True,
        delta_nll=-12.,delta_nats_per_event=2.,original={'target_events':8},variant={'target_events':2}))]
    data=dict(source_receipt_sha256=source,cases=rows)
    ratings=dict(schema='HCM1HumanListeningRatings@1',source_receipt_sha256=source,
        ratings={'p':dict(rating='worse',at='2026-09-28T06:23:21.600Z',score_revealed=False)})
    return ratings,data


def test_human_label_and_two_normalizations_stay_separate():
    r,p=packet();out=review_ratings(r,p)
    assert out['label_counts']=={'worse':1}
    assert out['directional']['raw_nll']==dict(agree=0,total=1)
    assert out['directional']['per_event']==dict(agree=1,total=1)
    assert out['rows'][0]['rating']=='worse'
    r['ratings']['p']['rating']='same';out=review_ratings(r,p)
    assert out['directional']['raw_nll']['total']==0
    assert out['rows'][0]['raw_agrees'] is None


@pytest.mark.parametrize('change', ['source','id','label','revealed','timestamp'])
def test_invalid_or_unbound_ratings_rejected(change):
    r,p=packet()
    if change=='source':r['source_receipt_sha256']='b'*64
    elif change=='id':r['ratings']['unknown']=r['ratings'].pop('p')
    elif change=='label':r['ratings']['p']['rating']='good'
    elif change=='revealed':r['ratings']['p']['score_revealed']='false'
    else:r['ratings']['p']['at']='yesterday'
    with pytest.raises(ValueError):review_ratings(r,p)


def test_missing_score_never_becomes_disagreement():
    r,p=packet();p['cases'][0]['score']={'supported':False}
    out=review_ratings(r,p)
    assert out['directional']['raw_nll']['total']==0
    assert out['rows'][0]['raw_agrees'] is None
