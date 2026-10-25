from dataclasses import asdict,replace
import json
from pathlib import Path
from scripts.build_hcm1_player import script_json,page,payload,generation_audio
from scripts.hcm1_diagnostic_data import query_from_json
from scripts.prepare_hcm1_diagnostic import prepare
from tests.pll.test_diagnostic_data import diagnostic_packet,piece
from music_critic.pll.masking import split_masked_piece
from scripts.hcm1_diagnostic_data import fixed_queries,audio_events


def player_record(tmp_path,cases_path,receipt_path,receipt):
    from hashlib import sha256
    from tests.pll.test_cli import prepare as cli_prepare
    from music_critic.experiments.registry import load_record,seal_record,canonical_json_bytes,DATA_ACCESS_FIELDS
    record_dir=tmp_path/'record-fixture';record_dir.mkdir()
    _,_,rp,_=cli_prepare(record_dir)
    binding=dict(cases_sha256=sha256(cases_path.read_bytes()).hexdigest())
    receipt['binding']=binding;receipt['results']['parent_update']=10000
    receipt_path.write_text(json.dumps(receipt))
    record=load_record(rp);record.update(status='completed',hcm_diagnostic=binding,results=receipt['results'])
    record['data_access'].update({k:False for k in DATA_ACCESS_FIELDS})
    record['data_access']['access_evidence_status']='verified_primary_evidence'
    raw=receipt_path.read_bytes()
    record['artifacts']=[dict(role='receipt',sha256=sha256(raw).hexdigest(),size_bytes=len(raw),availability='local',locator='synthetic-receipt.json')]
    rp.write_bytes(canonical_json_bytes(seal_record(record))+b'\n')
    return rp


def test_script_payload_cannot_close_script_or_inject_markup():
    malicious='</script><script>alert(1)</script>\u2028 __SYNTH__'
    encoded=script_json({'description':malicious})
    assert '<' not in encoded and json.loads(encoded)['description']==malicious
    html=page(dict(cases=[],all_scores=[],source_receipt_sha256='a',description=malicious))
    assert malicious not in html and 'application/json' in html


def test_fixture_preparation_to_listening_payload(tmp_path):
    plan_path,_,_,_=diagnostic_packet(tmp_path)
    cases=prepare(plan_path,tmp_path/'prepared');case=cases['cases'][0]
    receipt=dict(status='completed',results=dict(scores=[dict(id=case['id'],supported=False)],generations=[]))
    rp=tmp_path/'receipt.json';record=player_record(tmp_path,tmp_path/'prepared/cases.json',rp,receipt)
    result=payload(tmp_path/'prepared/cases.json',rp,record);row=result['cases'][0]
    assert row['audio']['original']!=row['audio']['variant']
    assert row['gain']==.012 and 0<=row['mask_seconds'][0]<row['mask_seconds'][1]<=row['duration_seconds']
    assert 'Послушать' in page(result)
    import pytest
    rp.write_text(rp.read_text()+' ')
    with pytest.raises(ValueError,match='terminal evaluation'):payload(tmp_path/'prepared/cases.json',rp,record)


def test_generated_audio_removes_masked_attacks_and_keeps_context():
    p=piece();q=fixed_queries(p)[1];ex=split_masked_piece(p,q)
    events,duration=generation_audio(p,q,[],p.bars[0].start_qn,p.duration_qn)
    observed=replace(p,notes=tuple(n for n in p.notes if not(n.track_id in q.track_ids and q.start<=n.onset_qn<q.end)))
    assert (events,duration)==audio_events(observed,p.bars[0].start_qn,p.duration_qn)
    reconstructed,_=generation_audio(p,q,[asdict(e) for e in ex.targets],p.bars[0].start_qn,p.duration_qn)
    assert len(reconstructed)==len(events)+len(ex.targets)


def test_historical_ratings_do_not_become_new_ratings():
    from scripts.build_hcm1_player import attach_review_context
    from tests.pll.test_listening_ratings import packet
    ratings,old=packet();old['cases'][0].update(original={'sha256':'same'},variant={'sha256':'same2'},query={'mask':1})
    new=json.loads(json.dumps(old));new['source_receipt_sha256']='b'*64;new['parent_update']=20000
    old['cases_sha256']=new['cases_sha256']='c'*64
    attach_review_context(new,old,ratings,old)
    assert new['prior_ratings_source_receipt_sha256']=='a'*64
    assert new['cases'][0]['prior_rating']['rating']=='worse'
    assert new['source_receipt_sha256']=='b'*64 and 'ratings' not in new
    assert '20 000' in page(new)
    new['cases'][0]['query']={'mask':2}
    import pytest
    with pytest.raises(ValueError,match='prior audio'):attach_review_context(new,old,ratings,old)
