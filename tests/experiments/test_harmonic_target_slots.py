from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import audit_harmonic_target_slots as audit
from music_critic.experiments.analysisgnn import corrected_training as production


def row(label, onset, slot='b', valid=True):
    return {'label':label,'onset':onset,'slot':slot,'valid':valid,
            'source_available':valid,'mapped':True,'known':True,'masked':not valid}


def test_temporal_and_same_time_irreducible_errors_are_separated():
    rows=[row('C','0'),row('C','0'),row('D','1/2'),
          row('C','1','b2'),row('D','1','b2'),row('E','1','b2',False)]
    counts,classes,examples=audit.summarize_rows(rows)
    assert counts['valid_rows']==5 and counts['masked_rows']==1
    assert counts['unavoidable_errors']==2
    assert counts['same_time_unavoidable_errors']==1
    assert counts['temporally_recoverable_errors']==1
    assert counts['different_times_only_conflict_slots']==1
    assert counts['same_time_present_conflict_slots']==1
    assert classes['C']['valid_rows']==3
    assert 'E' not in classes
    assert set(examples)=={'same_time_present','different_times_only'}


def test_repeated_equal_labels_are_not_conflicts():
    counts,_,_=audit.summarize_rows([row('C','0'),row('C','1/2')])
    assert counts['repeated_slots']==1
    assert counts.get('conflict_slots',0)==0


@pytest.mark.parametrize('split',['test','validation','unknown'])
def test_split_gate_precedes_any_source_access(tmp_path,split):
    # No record ID/path is supplied: if routing starts before the gate it fails differently.
    with pytest.raises(ValueError,match='TRAIN only'):
        audit.selected_source(tmp_path,{'split':split})


def test_time_must_be_exact_and_canonical():
    with pytest.raises(ValueError): audit.rational({'num':1,'den':2.0})
    with pytest.raises(ValueError): audit.rational({'num':2,'den':4})
    assert str(audit.rational({'num':1,'den':2}))=='1/2'


def test_an_native_folder_name_is_not_the_frozen_split(tmp_path):
    _,relative,_,_=audit.selected_source(tmp_path,{'record_id':'an:test:abc-op135-2',
        'dialect':'an_joint','split':'train'})
    assert relative=='pitch_arrays/AN/test/abc-op135-2_joint.tsv'
    _,relative,_,_=audit.selected_source(tmp_path,{'record_id':'dlc:ABC:piece',
        'dialect':'dlc','split':'train'})
    assert relative=='pitch_arrays/DLC/ABC/piece.tsv'


def test_census_projection_matches_actual_production_alignment(monkeypatch):
    import music_critic.tasks as tasks
    captured=[]
    prepare=tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        captured.append(piece)
        return prepare(piece, *args, **kwargs)
    monkeypatch.setattr(tasks,'prepare_multisource_sample',capture)
    batch,sidecar=production.build_source_free_fixture()
    raw=batch.raw_graph_batch
    piece=captured[0]
    sidecar=deepcopy(sidecar)
    original=next(e for e in sidecar['entities'] if e['entity_type']=='harmonic_event')
    extra=deepcopy(original); extra['canonical_entity_id']='harmonic:extra'
    extra['targets']['root']['canonical_value']='D'
    sidecar['entities'].append(extra)
    relation=next(r for r in sidecar['relations'] if r['relation']=='harmonic_event_to_beat'
                  and r['source_entity_id']==original['canonical_entity_id'])
    sidecar['relations'].append({**relation,'source_entity_id':extra['canonical_entity_id']})
    # Unknown vocabulary and target masks must match the actual runtime behavior.
    extra['targets']['quality']['canonical_value']='not-a-class'
    extra['targets']['inversion']['masked']=True
    predictions=[]
    for h in audit.HEADS:
        level=audit.TASK_BY_ID[h].prediction_level
        node='beat' if level=='harmonic_event' else level
        count=raw[node].num_nodes
        predictions.append(SimpleNamespace(task_id=h,allowed_node_types=(node,),
            sample_indices=torch.zeros(count,dtype=torch.long),
            global_entity_indices=torch.arange(count,dtype=torch.long)))
    output=SimpleNamespace(predictions=predictions)
    actual=production.align_target_sidecars_after_prediction(output,raw,(sidecar,)).heads
    projected=audit.project_targets(piece,sidecar)
    for h in audit.HEADS:
        assert [r['valid'] for r in projected[h]]==actual[h].valid_mask.tolist(),h
        node='beat' if audit.TASK_BY_ID[h].prediction_level=='harmonic_event' else audit.TASK_BY_ID[h].prediction_level
        for i,r in enumerate(projected[h]):
            if r['valid']:
                assert r['label']==audit.VOCABS[h][actual[h].values[i]],h
                assert r['slot']==raw[node].entity_id[0][actual[h].candidate_indices[i]],h
    assert extra['canonical_entity_id'] not in {r['entity'] for r in projected['joint']}
