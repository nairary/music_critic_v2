from dataclasses import replace
from pathlib import Path

import pytest

from music_critic.data import load_piece, RationalTime as Q
from scripts.hcm1_diagnostic_data import (select_validation,read_validation,paired_examples,
    query_from_json,query_json,fixed_queries,qn_seconds,audio_events)


def piece():
    return load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json')


def test_global_component_test_membership_wins_before_selection():
    assignments=[dict(dataset_id='a',piece_id='good',split='validation',component_fingerprint='safe'),
                 dict(dataset_id='a',piece_id='bad',split='validation',component_fingerprint='shared'),
                 dict(dataset_id='b',piece_id='test',split='test',component_fingerprint='shared')]
    indices={'a':[dict(piece_id=p,source_group_id=p,suggested_split='val',canonical_relative_path=p,canonical_sha256='0'*64) for p in ('bad','good')]}
    assert select_validation(assignments,indices,per_source=1)[0]['piece_id']=='good'


def test_native_test_alias_excludes_whole_component():
    assignments=[dict(dataset_id='a',piece_id=p,split='validation',component_fingerprint='same') for p in ('a','b')]
    rows=[dict(piece_id=p,source_group_id=p,suggested_split=s,canonical_relative_path=p,canonical_sha256='0'*64) for p,s in (('a','val'),('b','test'))]
    with pytest.raises(ValueError,match='insufficient'):
        select_validation(assignments,{'a':rows},per_source=1)


def test_preload_guard_does_not_read(tmp_path):
    accesses=[]
    with pytest.raises(ValueError,match='preload'):
        read_validation(tmp_path,dict(global_split='test',native_split=None),accesses)
    assert accesses==[]


def test_pair_context_equality_is_required_not_assumed():
    p=piece();queries=fixed_queries(p);q=queries[0]
    inside=[n for n in p.notes if q.start<=n.onset_qn<q.end]
    if not inside:
        q=queries[1];inside=[n for n in p.notes if q.start<=n.onset_qn<q.end]
    note=inside[0]
    variant=replace(p,notes=tuple(replace(n,pitch=(n.pitch+1)%128) if n.note_id==note.note_id else n for n in p.notes))
    a,b,_=paired_examples(p,variant,q)
    assert a.targets!=b.targets
    assert query_from_json(query_json(q))==q
    outside=next(n for n in p.notes if not q.start<=n.onset_qn<q.end)
    wrong=replace(variant,notes=tuple(replace(n,velocity=(n.velocity or 70)+1) if n.note_id==outside.note_id else n for n in variant.notes))
    with pytest.raises(ValueError,match='contexts differ'):
        paired_examples(p,wrong,q)


def test_tempo_schedule_integrates_changes():
    p=piece();base=p.tempo_events[0]
    p=replace(p,tempo_events=(replace(base,onset_qn=Q(0),microseconds_per_quarter=500000),
                              replace(base,tempo_event_id='tempo:2',onset_qn=Q(2),microseconds_per_quarter=1000000)))
    assert qn_seconds(p,Q(3))==2.0
    events,duration=audio_events(p,Q(1),Q(3))
    assert duration==1.5
    assert all(e['onset_seconds']>=0 and e['duration_seconds']>=0 for e in events)


def diagnostic_packet(tmp_path):
    import json
    from tests.critic.test_authored_data import packet
    root=tmp_path/'inputs';root.mkdir()
    manifest,path,_,write=packet(root)
    manifest['pairs'][0]['split']='head_dev'
    binding=write('manifest.json',manifest)
    plan=dict(validation=[],train=[],cache_root=str(root),train_root=str(root),authored_root=str(root),
              authored_manifest=binding,pair_ids=['pair:fixture'],training_identities=[])
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    return plan_path,plan,manifest,write


def test_complete_synthetic_preparation_path(tmp_path):
    from scripts.prepare_hcm1_diagnostic import prepare
    plan_path,_,_,_=diagnostic_packet(tmp_path)
    result=prepare(plan_path,tmp_path/'prepared')
    assert result['status']=='completed'
    case,=result['cases']
    assert case['supported'] and case['context_sha256']
    assert case['description']['id']=='S1'
    assert case['hcm_training_overlap']['known_overlap'] is False


def test_authored_forbidden_scope_before_payload_read(tmp_path,monkeypatch):
    import json
    import scripts.prepare_hcm1_diagnostic as module
    plan_path,plan,manifest,write=diagnostic_packet(tmp_path)
    manifest['pairs'][0]['source_scope']='test'
    plan['authored_manifest']=write('manifest.json',manifest)
    plan_path.write_text(json.dumps(plan))
    calls=[];original=module.read_bound
    def observed(root,binding):
        calls.append(binding['path']);return original(root,binding)
    monkeypatch.setattr(module,'read_bound',observed)
    with pytest.raises(ValueError,match='source scope'):
        module.prepare(plan_path,tmp_path/'prepared')
    assert calls==['manifest.json']


def test_output_guard_and_known_overlap(tmp_path):
    from scripts.prepare_hcm1_diagnostic import prepare,training_overlap
    plan_path,_,manifest,_=diagnostic_packet(tmp_path)
    with pytest.raises(ValueError,match='full_artifacts'):
        prepare(plan_path,Path(__file__).parents[2]/'docs/forbidden-diagnostic-output')
    row=manifest['pairs'][0]
    result=training_overlap(piece(),row,[dict(source='fixture',work_id=row['work_id'])])
    assert result['known_overlap'] and result['matches'][0]['reasons']==[row['work_id']]


def test_validation_corpus_cache_layout(tmp_path,monkeypatch):
    import json
    import scripts.prepare_hcm1_diagnostic as module
    plan_path,plan,_,_=diagnostic_packet(tmp_path)
    plan['validation']=[dict(source='hooktheory')]
    plan_path.write_text(json.dumps(plan))
    def check(root,row,accesses):
        assert root==Path(plan['cache_root'])/'hooktheory/multisource-cache-v1.0.0'
        raise RuntimeError('path checked')
    monkeypatch.setattr(module,'read_validation',check)
    with pytest.raises(RuntimeError,match='path checked'):module.prepare(plan_path,tmp_path/'prepared')


def test_window_clipping_cannot_change_masked_answers():
    from types import SimpleNamespace
    from scripts.prepare_hcm1_diagnostic import targets_preserved
    p=piece();window=SimpleNamespace(piece=p,global_offset_qn=Q(0))
    assert targets_preserved(p,window)
    q=fixed_queries(p)[1]
    target=next(n for n in p.notes if q.start<=n.onset_qn<q.end)
    window.piece=replace(p,notes=tuple(replace(n,duration_qn=Q(1,16)) if n.note_id==target.note_id else n for n in p.notes))
    assert not targets_preserved(p,window)
