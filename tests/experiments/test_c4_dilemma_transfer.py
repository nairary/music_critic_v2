"""Synthetic reverse-transfer mechanics, without corpus access."""
from types import SimpleNamespace
import torch
import pytest
from music_critic.experiments.c4_pop909.transfer import aligned_events, summarize, validate_request
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary


def state(value, available=True):
    return {'available':available,'masked':not available,'canonical_value':value}


def test_raw_beat_route_preserves_events_and_independent_masks():
    labels=get_vocabulary(TASK_BY_ID['root'].vocabulary_id).labels
    output=SimpleNamespace(candidate_keys={'root':((0,'slice','s'),(0,'beat','b')),'quality':((0,'beat','b'),)},
        logits={'root':torch.zeros(2,len(labels)), 'quality':torch.zeros(1,17)})
    output.logits['root'][1,labels.index('C')]=20
    sc={'entities':[{'entity_type':'harmonic_event','canonical_entity_id':'e1','targets':{'root':state('C'),'quality':state('major triad',False)}},
       {'entity_type':'harmonic_event','canonical_entity_id':'e2','targets':{'root':state('B#'),'quality':state('major triad')}}],
       'relations':[{'relation':'harmonic_event_to_beat','source_entity_id':e,'target_entity_id':'b'} for e in ('e1','e2')]}
    rows=aligned_events(output,sc)
    assert len(rows)==2
    assert rows[0]['targets']['root']['gold']==0 and rows[0]['targets']['root']['predicted']==0
    assert rows[1]['targets']['root']['gold']==0
    assert rows[0]['targets']['quality']['gold'] is None


def test_common_quality_keeps_outside_predictions_as_errors():
    m=torch.zeros(17,17,dtype=torch.long);m[0,0]=2;m[0,16]=1;m[16,16]=10
    r=summarize(m,allowed_gold=(0,))
    assert r['eligible_events']==3
    assert r['accuracy']==pytest.approx(2/3)
    assert r['observed_true_class_macro_f1']==pytest.approx(4/5)
    assert r['excluded_gold_events']==10


def test_empty_metric_is_unavailable_not_zero():
    r=summarize(torch.zeros(12,12,dtype=torch.long))
    assert r['accuracy'] is None and r['observed_true_class_macro_f1'] is None


def test_request_requires_validation_and_complete_training():
    with pytest.raises(ValueError,match='validation_only'):
        validate_request({'split':'test'}, {'status':'completed'})
    with pytest.raises(ValueError,match='training_not_completed'):
        validate_request({'split':'validation'}, {'status':'running'})


def test_ambiguous_source_beat_relation_is_rejected():
    output=SimpleNamespace(logits={},candidate_keys={})
    sidecar={'entities':[],'relations':[{'relation':'harmonic_event_to_beat','source_entity_id':'e','target_entity_id':b} for b in ('b1','b2')]}
    with pytest.raises(ValueError,match='ambiguous_beat_relation'):
        aligned_events(output,sidecar)


def test_unaligned_available_source_target_is_not_scored():
    output=SimpleNamespace(logits={'root':torch.zeros(0,37)},candidate_keys={'root':()})
    sidecar={'entities':[{'entity_type':'harmonic_event','canonical_entity_id':'e','targets':{'root':state('C')}}],'relations':[]}
    cell=aligned_events(output,sidecar)[0]['targets']['root']
    assert cell['gold'] is None and cell['predicted'] is None and cell['reason']=='unaligned'
    assert cell['source_available'] and cell['canonical_known'] and not cell['candidate_aligned']


def test_registered_policy_is_checked_before_any_training_or_corpus_read(monkeypatch):
    from music_critic.experiments.c4_pop909 import transfer
    from music_critic.experiments.analysisgnn.contracts import fingerprint
    request={'split':'validation'}
    record={'status':'planned','kind':'evaluation','protocol':{'request_fingerprint':fingerprint(request)},
        'data_access':{'declared_policy':{'validation_allowed':False}}}
    monkeypatch.setattr(transfer,'load_record',lambda _:record)
    monkeypatch.setattr(transfer,'read_bound_bytes',lambda _:pytest.fail('payload read before policy'))
    with pytest.raises(ValueError,match='access_policy'):
        transfer.evaluate(request,experiment_record='synthetic',output='unused')


def test_checkpoint_must_be_selected_on_pop_before_deserialization(monkeypatch):
    from music_critic.experiments.c4_pop909 import transfer
    parent={'status':'completed','results':{'producer_report':{'best_checkpoint':{'sha256':'1'*64,'size_bytes':100,'update':60000}}}}
    monkeypatch.setattr(transfer,'load_record',lambda _:parent)
    monkeypatch.setattr(transfer,'read_bound_bytes',lambda _:b'parent metadata')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('deserialization before selection guard'))
    with pytest.raises(ValueError,match='selected_checkpoint_binding'):
        transfer.load_frozen_c4({'sha256':'2'*64,'size_bytes':100},{'path':'synthetic'},'cpu')


def test_source_binding_is_checked_before_data(monkeypatch):
    from music_critic.experiments.c4_pop909 import transfer
    from music_critic.experiments.analysisgnn.contracts import fingerprint
    request={'split':'validation','source_fingerprint':'wrong'}
    record={'status':'planned','kind':'evaluation','protocol':{'request_fingerprint':fingerprint(request)},
        'data_access':{'declared_policy':dict(validation_allowed=True,test_inputs_allowed=False,
        test_targets_allowed=False,test_metrics_allowed=False,test_selection_allowed=False)}}
    monkeypatch.setattr(transfer,'load_record',lambda _:record)
    monkeypatch.setattr(transfer,'implementation_fingerprint',lambda:'actual')
    monkeypatch.setattr(transfer,'read_bound_bytes',lambda _:pytest.fail('read before source check'))
    with pytest.raises(ValueError,match='implementation_binding'):
        transfer.evaluate(request,experiment_record='synthetic',output='unused')
