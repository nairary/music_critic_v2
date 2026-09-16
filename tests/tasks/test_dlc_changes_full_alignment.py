import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from scripts.prepare_dlc_changes_alignment import event_groups, summarize_alignment, validate_inputs
from scripts.audit_dlc_changes_alignment import digest


def write_groups(path, ids):
    path.write_text(''.join(json.dumps({'record_id': rid})+'\n' for rid in ids))


def test_stream_requires_complete_sorted_unique_groups(tmp_path):
    p = tmp_path/'events.jsonl'
    write_groups(p, ['a','a','b'])
    assert [(rid,len(rows)) for rid,rows in event_groups(p,['b','a'])] == [('a',2),('b',1)]
    for ids in (['a'], ['a','c'], ['a','b','a'], ['b','a'], []):
        write_groups(p, ids)
        with pytest.raises(ValueError): list(event_groups(p,['a','b']))


def test_nontrain_rejected_before_targets_or_source_open(tmp_path, monkeypatch):
    m = tmp_path/'manifest.json';m.write_text(json.dumps({'records':[{'split':'test'}]}))
    parent = tmp_path/'parent.json';parent.write_text(json.dumps({'status':'completed','artifacts':[
        {'locator':'manifest.json','sha256':digest(m),'size_bytes':m.stat().st_size}]}))
    original=Path.open
    def guarded(path,*a,**kw):
        assert path in (m,parent)
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    with pytest.raises(ValueError,match='TRAIN'): validate_inputs(tmp_path,parent)


def fixture():
    events=[{'source_annotation_id':str(i),'target_mask':True,'positive_tokens':tokens,'source_row_numbers':[i+2]}
            for i,tokens in enumerate([['4'],['4'],['6'],[]])]
    targets=[{'target_mask':True,'source_event_ids':['0','1'],'positive_tokens':['4'],'values':[1,0],
              'annotation_weight':'3/2','mask_reasons':[]},
             {'target_mask':True,'source_event_ids':['0'],'positive_tokens':['4'],'values':[1,0],
              'annotation_weight':'1/2','mask_reasons':[]},
             {'target_mask':False,'source_event_ids':['2'],'positive_tokens':[],'values':[0,0],
              'annotation_weight':'0','mask_reasons':['unexplained_source_notation']},
             {'target_mask':True,'source_event_ids':['3'],'positive_tokens':[],'values':[0,0],
              'annotation_weight':'1','mask_reasons':[]}]
    alignment={'targets':targets,'coverage':{},'beat_diagnostic':[],
               'notation_provenance':{str(i):{'available':i!=2,'reason':'mismatch' if i==2 else 'match'} for i in range(4)}}
    return alignment,events


def test_annotation_and_piece_support_not_inflated_by_onsets():
    a,e=fixture();s=summarize_alignment(a,e)
    assert s['before_event_support']=={'4':2,'6':1}
    assert s['usable_event_support']=={'4':2}
    assert s['usable_onset_support']=={'4':2}
    assert s['annotation_weight_sum']=='3'
    assert s['counts']['usable_empty_events']==1
    assert s['counts']['notation_rejected_previously_unmasked_events']==1


@pytest.mark.parametrize('mutation',['weight','target','masked_value'])
def test_invalid_weight_or_target_fails_verification(mutation):
    a,e=fixture()
    if mutation=='weight':a['targets'][0]['annotation_weight']='1'
    if mutation=='target':a['targets'][0]['positive_tokens']=['6']
    if mutation=='masked_value':a['targets'][2]['values']=[1,0]
    with pytest.raises(AssertionError):summarize_alignment(a,e)
