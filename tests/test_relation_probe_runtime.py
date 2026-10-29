"""Synthetic raw-candidate/mask integration and metric contracts."""
from pathlib import Path
import json
import torch
import pytest
from music_critic.relation_probe.data import sanitize
from music_critic.relation_probe.runtime import binding,build_cache,TileStore,confusion_metrics,exact_slot,retrieval_at_k
from fractions import Fraction


@pytest.mark.parametrize('divergent',[False,True])
def test_cache_keeps_unlabelled_tails_unknown(tmp_path,divergent):
    from music_critic.data import load_piece,dumps_piece
    p=sanitize(load_piece(Path(__file__).parent/'fixtures/data/canonical_piece_v2.json'))
    source=p.notes[0];target=next(n for n in p.notes if n.onset_qn>source.onset_qn)
    none=next(n for n in p.notes if n.note_id not in (source.note_id,target.note_id))
    dataset=tmp_path/'dataset';dataset.mkdir();raw=dataset/'raw.json';raw.write_text(dumps_piece(p))
    t=lambda q:[q.num,q.den]
    targets=[dict(context_id='p',note_id=source.note_id,view_id='v',time=t(source.onset_qn),
                  end=t(source.onset_qn+source.duration_qn),label=1,
                  targets=[dict(note_id=target.note_id,time=t(target.onset_qn))]),
             dict(context_id='n',note_id=none.note_id,view_id='v',time=t(none.onset_qn),
                  end=t(none.onset_qn+none.duration_qn),label=0,targets=[])]
    if divergent:
        targets.append({**targets[0],'view_id':'another','context_id':'alternative',
                        'targets':[dict(note_id=none.note_id,time=t(none.onset_qn))]})
    labels=dataset/'labels.json';labels.write_text(json.dumps(targets))
    (dataset/'manifest.json').write_text(json.dumps(dict(works=[dict(id='w',family='f',split='fit',
        raw=binding(raw,'raw.json'),targets=binding(labels,'labels.json'))])))
    manifest=build_cache(dataset,tmp_path/'cache')
    item=TileStore(tmp_path/'cache').get(manifest['tiles'][0])
    positive_source=item['exist_source'][item['exist_y']==1][0]
    unknown=(item['source']==positive_source)&(item['labels']==0)
    assert unknown.any() and not item['mask'][unknown].any()
    if divergent:
        assert item['labels'][item['mask']].sum()==0
        assert not item['queries']
        assert manifest['counts']['alternative_endpoint_sets_masked']==1
    else:assert item['labels'][item['mask']].sum()>=1
    assert item['graph'].base.raw_only
    for source_id in item['source'].unique():
        mask=(item['source']==source_id)&item['mask']
        if mask.any():assert torch.isclose(item['weights'][mask].sum(),torch.tensor(1.))


def test_metrics_include_both_classes():
    m=confusion_metrics([1,1,1,1])
    assert m['macro_f1']==.5 and m['precision']==.5 and m['recall']==.5
    assert confusion_metrics([3,0,0,0])['macro_f1']==1


def test_held_contexts_cannot_alias_by_flooring_time():
    times=[Fraction(0),Fraction(1),Fraction(2)]
    assert exact_slot(times,Fraction(1,4)) is None
    assert exact_slot(times,Fraction(3,4)) is None
    assert exact_slot(times,Fraction(1))==1
    assert exact_slot(times,Fraction(2))==2


def test_multi_answer_recall_is_distinct_from_hit():
    assert retrieval_at_k([0,1,2],{0,2},2,1)==(1,.5)
    assert retrieval_at_k([0,1,2],{0,2},2,5)==(1,1.)
    assert retrieval_at_k([],{0,2},2,5)==(0,0.)
