"""Synthetic whole-work scheduling/replay tests; no musical-quality claims."""
from copy import deepcopy
import pytest
import torch

from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import (
    train_full_steps, full_sample_schedule, save_full_checkpoint,
    load_full_checkpoint, warmstart_e3,
)

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(1,1);self.drop=torch.nn.Dropout(.2)
    def forward(self,raw):
        assert set(raw)=={'x'}
        return self.layer(self.drop(raw['x']))

class Samples:
    def __init__(self,known=True):
        self.metadata=[dict(work_id='w',family_id='family',view_id='v',split='fit')]
        self.target=dict(self.metadata[0],known=known)
    def __len__(self):return 1
    def __getitem__(self,index):return {'x':torch.ones(1,1)},deepcopy(self.target)

def loss(pred,target,weights=None):
    return {'loss':(pred-2).square().mean() if target['known'] else pred.sum()*0,
            'supported':target['known'],'support':{'known':int(target['known'])},'components':{}}

def test_unknown_draws_do_not_apply_adam_and_accounting_is_exact():
    model=Tiny();opt=torch.optim.Adam(model.parameters(),lr=.01);before=deepcopy(model.state_dict())
    run=train_full_steps(model,Samples(False),opt,steps=4,seed=17,loss_fn=loss)
    assert run['state']['attempted_steps']==4
    assert run['state']['applied_updates']==0 and run['state']['skipped_unknown']==4
    assert not opt.state
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    run2=train_full_steps(model,Samples(),opt,steps=3,seed=17,state=run['state'],loss_fn=loss)
    assert run2['state']['attempted_steps']==7 and run2['state']['applied_updates']==3

def test_whole_sampler_is_work_balanced_and_order_independent():
    metadata=[dict(work_id='a',view_id=str(i),split='fit') for i in range(8)]+[dict(work_id='b',view_id='0',split='fit')]
    samples=[({},x) for x in metadata]
    ids=[metadata[i]['work_id'] for i in full_sample_schedule(samples,count=2000,seed=3,sample_metadata=metadata)]
    assert 900<ids.count('b')<1100
    rev=list(reversed(metadata));indices=full_sample_schedule(list(reversed(samples)),count=2000,seed=3,sample_metadata=rev)
    assert ids==[rev[i]['work_id'] for i in indices]

def test_nonfit_rejected_before_forward():
    samples=Samples();samples.metadata[0]['split']='dev'
    with pytest.raises(ValueError,match='fit'):
        train_full_steps(Tiny(),samples,None,steps=1,loss_fn=loss)

def test_checkpoint_resume_restores_optimizer_rng_and_rejects_wrong_binding(tmp_path):
    torch.manual_seed(4);model=Tiny();opt=torch.optim.Adam(model.parameters(),lr=.01)
    first=train_full_steps(model,Samples(),opt,steps=3,seed=17,loss_fn=loss)
    bindings={k:'a'*64 for k in ['source_sha256','config_sha256','data_sha256','cache_sha256']}
    p=tmp_path/'state.pt';receipt=save_full_checkpoint(p,model,opt,first['state'],bindings=bindings)
    expected=train_full_steps(model,Samples(),opt,steps=4,seed=17,state=first['state'],loss_fn=loss)
    weights=deepcopy(model.state_dict())
    restored=Tiny();other=torch.optim.Adam(restored.parameters(),lr=.7)
    state=load_full_checkpoint(p,restored,other,bindings=bindings,expected_sha256=receipt['sha256'])
    actual=train_full_steps(restored,Samples(),other,steps=4,seed=17,state=state,loss_fn=loss)
    assert actual['state']==expected['state']
    assert actual['history']==expected['history']
    assert all(torch.equal(weights[k],v) for k,v in restored.state_dict().items())
    bad=dict(bindings,data_sha256='b'*64);before=deepcopy(restored.state_dict())
    with pytest.raises(ValueError,match='bindings'):
        load_full_checkpoint(p,restored,other,bindings=bad,expected_sha256=receipt['sha256'])
    assert all(torch.equal(before[k],v) for k,v in restored.state_dict().items())

def test_warmstart_e3_only_copies_verified_constructor_weights(tmp_path):
    from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer
    from music_critic.experiments.hierarchical_graph_analyzer.runtime import save_checkpoint
    torch.manual_seed(1);a=HAnalyzer('E3',{'H.function':2},hidden=8);opt=torch.optim.Adam(a.parameters())
    bindings={k:'a'*64 for k in ['source_sha256','config_sha256','data_sha256']}
    p=tmp_path/'e3.pt';receipt=save_checkpoint(p,a,opt,{'applied_updates':40000},bindings=bindings)
    b=HAnalyzer('E3',{'H.function':2},hidden=8);rng=torch.get_rng_state().clone()
    r=warmstart_e3(b,p,expected_sha256=receipt['sha256'],expected_size_bytes=receipt['size_bytes'])
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state()) and r['optimizer_restored'] is False
    with pytest.raises(ValueError,match='SHA'):
        warmstart_e3(b,p,expected_sha256='b'*64,expected_size_bytes=receipt['size_bytes'])

def test_structured_prediction_preserves_addressed_note_and_global_time():
    from fractions import Fraction
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import structured_prediction
    raw={'note_ids':('physical-note',),'origin':Fraction(4),'duration':Fraction(8)}
    event={'spans':torch.tensor([[.25,.75]]),'object_logits':torch.tensor([1.]),'field_logits':{}}
    pred={'H':event,'P':event,'h':{'spans':torch.tensor([[.25,.5]]),'note_index':torch.tensor([0]),'H_index':torch.tensor([0]),'field_logits':{'h.chord_membership':torch.tensor([[0.,2.]])}},
          'participants':{'P_index':torch.tensor([0]),'endpoint_level':('h',),'endpoint_index':torch.tensor([0]),'existence_logits':torch.tensor([2.]),'field_logits':{}},
          'relations':{'sources':(({ 'kind':'h','index':0},),),'targets':(({ 'kind':'H','index':0},),),'logits':{'resolves_to':torch.tensor([3.])}}}
    doc=structured_prediction(pred,raw,work_id='work')
    assert doc['H'][0]['span_qn']==[[6,1],[10,1]]
    assert doc['h'][0]['note_id']=='physical-note'
    assert doc['h'][0]['H_id']==doc['H'][0]['id']
    assert doc['participants'][0]['endpoint']==doc['h'][0]['id']
    assert doc['relations'][0]['source']==[doc['h'][0]['id']]
    assert 'targets' not in doc and 'ground_truth' not in str(doc)

def test_raw_unlabelled_canonical_build_and_full_inference_share_forward():
    from pathlib import Path
    from music_critic.data import load_piece
    from music_critic.experiments.hierarchical_graph_analyzer.cache import BASE_NODES
    from music_critic.experiments.hierarchical_graph_analyzer.full_inference import encode_raw_work
    from music_critic.experiments.hierarchical_graph_analyzer.full_model import FullGraphAnalyzer,FullGraphConfig
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import infer_full
    class Encoder(torch.nn.Module):
        def _encode_graph(self,graph,flags):
            return {k:torch.full((graph.slice_features.shape[0] if k=='slice' else graph.base[k].num_nodes,128),i/10.) for i,k in enumerate(BASE_NODES)}
    piece=load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json')
    raw=encode_raw_work(piece,Encoder(),device='cpu',wall_seconds=30)
    assert raw['note_ids']==tuple(n.note_id for n in piece.notes)
    assert 'targets' not in raw and len(raw['tiles'])>0
    m=FullGraphAnalyzer({'H.function':2},{'h.chord_membership':2},{'P.process':2},config=FullGraphConfig(hidden=16,attention_heads=2,constructor_rounds=1,rounds=1,h_per_tile=2,max_h=16,max_p=32))
    doc=infer_full(m,raw,work_id='novel-work')
    assert doc['H'] and doc['h'] and doc['P']
    assert all(h['note_id'] in raw['note_ids'] for h in doc['h'])
    assert doc['coverage']['raw']['canonical_notes']==len(piece.notes)

def test_full_launch_requires_prepared_exact_registered_budget(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from hashlib import sha256
    path=Path(__file__).parents[2]/'scripts/run_hiergraph_full.py'
    spec=importlib.util.spec_from_file_location('full_cli',path);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    config=dict(schema='HiergraphFullRun@1',prepared=True,test_locked=True,augmentation='none',freeze_policy='frozen_C3',wall_seconds=10,
        source_manifest_sha256='a'*64,target_manifest_sha256='b'*64,cache_manifest_sha256='c'*64)
    raw=json.dumps(config).encode();output=tmp_path/'campaign'/'fresh'
    launch=dict(mode='gate',seed=17,config_sha256=sha256(raw).hexdigest(),source_manifest_sha256='a'*64,target_manifest_sha256='b'*64,cache_manifest_sha256='c'*64,output_name='fresh',output_path='campaign/fresh',wall_seconds=10)
    record=dict(status='planned',protocol={'launch':launch,'evidence_root':'campaign','output_root':'fresh'},seeds=[17],budget={'planned':{'total_wall_seconds':10}})
    assert cli.admit_launch(raw,config,record,'gate',output,17,evidence_base=tmp_path)==launch
    from music_critic.experiments.registry import seal_record,validate_record
    complete=dict(schema_version='1.0.0',experiment_id='fixture-full-launch',kind='engineering_gate',phase='T08',title='fixture',question='fixture',hypothesis='fixture',decision_use='fixture',repository={'git_commit':'a'*40,'dirty':True},data_access=dict(access_evidence_status='not_applicable_planning_record',validation_accessed=None,test_inputs_accessed=None,test_targets_accessed=None,test_metrics_computed=None,test_used_for_selection=None,declared_policy=dict(source='fixture',validation_allowed=False,test_inputs_allowed=False,test_targets_allowed=False,test_metrics_allowed=False,test_selection_allowed=False)),results={},validity={'valid':None},claims={'supported':[],'not_supported':[]},artifacts=[],import_provenance={'mode':'fixture'},**record)
    validate_record(seal_record(complete))
    with pytest.raises(ValueError,match='output path'):cli.admit_launch(raw,config,record,'gate',tmp_path/'other'/'fresh',17,evidence_base=tmp_path)
    with pytest.raises(ValueError,match='exact launch'):cli.admit_launch(raw,config,record,'gate',output,19,evidence_base=tmp_path)
    record['budget']['planned']['total_wall_seconds']=9
    with pytest.raises(ValueError,match='budget'):cli.admit_launch(raw,config,record,'gate',output,17,evidence_base=tmp_path)
    config['prepared']=False
    with pytest.raises(ValueError,match='prepared'):cli.admit_launch(raw,config,record,'gate',output,17,evidence_base=tmp_path)

def test_full_model_fit_selection_and_verified_inference_weights(tmp_path):
    import importlib.util
    from pathlib import Path
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import evaluate_full
    from music_critic.experiments.hierarchical_graph_analyzer.full_inference import load_full_weights
    path=Path(__file__).with_name('test_hiergraph_full_model.py')
    spec=importlib.util.spec_from_file_location('native_model_fixture',path);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    model=fixture.model();raw=fixture.raw_work()
    target={'H':{'ids':['H0'],'spans':[[0.,.5]],'fields':{'H.function':{'values':[0],'mask':[True]}}},
        'h':{'ids':[],'spans':[],'note_index':[],'H_index':[],'fields':{}},
        'P':{'ids':['P0'],'spans':[[0.,1.]],'fields':{'P.process':{'values':[0],'mask':[True]}}},'participants':[],'relations':[],
        'work_id':'w','family_id':'f','view_id':'v','split':'fit'}
    samples=[(raw,target)];opt=torch.optim.Adam(model.parameters(),lr=.001)
    run=train_full_steps(model,samples,opt,steps=2)
    assert run['state']['applied_updates']==2
    assert run['history'][0]['support']['H']['known_events']==1
    binding={k:'a'*64 for k in ['source_sha256','config_sha256','data_sha256','cache_sha256']}
    p=tmp_path/'full.pt';receipt=save_full_checkpoint(p,model,opt,run['state'],bindings=binding)
    restored=fixture.model();rng=torch.get_rng_state().clone()
    load_full_weights(restored,p,expected_sha256=receipt['sha256'],expected_size_bytes=receipt['size_bytes'],bindings=binding)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    selected=deepcopy(target);selected['split']='selection'
    report=evaluate_full(model,[(raw,selected)])
    assert report['geometry']=='predicted' and 'known_h_recovery' in report['aggregate']['H']
    oracle=evaluate_full(model,[(raw,selected)],geometry='oracle')
    assert oracle['geometry']=='oracle' and oracle['aggregate']['H']['known_h_recovery']==1.
    assert oracle['aggregate']['P']['known_p_recovery']==1.
    with pytest.raises(ValueError,match='final dev'):evaluate_full(model,[(raw,target)],split='dev')

def test_inference_rejects_wrong_frozen_c3_binding():
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import validate_full_config
    config=dict(schema='HiergraphFullRun@1',prepared=True,test_locked=True,augmentation='none',freeze_policy='frozen_C3',wall_seconds=10,
        source_manifest_sha256='a'*64,c3_checkpoint={'sha256':'b'*64},model_checkpoint={'bindings':{'c3_sha256':'c'*64}})
    with pytest.raises(ValueError,match='C3'):validate_full_config(config,'infer')
    config['model_checkpoint']['bindings']['c3_sha256']='b'*64
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import full_vocabulary_sha256
    config.update(head_sizes={'H':{},'h':{},'P':{}},field_values={})
    config['model_checkpoint']['bindings']['output_vocabulary_sha256']=full_vocabulary_sha256(config['head_sizes'],config['field_values'])
    assert validate_full_config(config,'infer') is config


def test_warmstart_rejects_semantic_heads_with_equal_shapes(tmp_path):
    from music_critic.experiments.hierarchical_graph_analyzer.model import HAnalyzer
    from music_critic.experiments.hierarchical_graph_analyzer.runtime import save_checkpoint
    model=HAnalyzer('E3',{'H.function':2},hidden=8);opt=torch.optim.Adam(model.parameters())
    p=tmp_path/'semantic.pt';receipt=save_checkpoint(p,model,opt,{},bindings={k:'a'*64 for k in ['source_sha256','config_sha256','data_sha256']})
    incompatible=HAnalyzer('E3',{'H.chord_quality':2},hidden=8)
    with pytest.raises(ValueError,match='semantic'):warmstart_e3(incompatible,p,expected_sha256=receipt['sha256'],expected_size_bytes=receipt['size_bytes'])

@pytest.mark.parametrize('change',['head','relations','config'])
def test_full_checkpoint_rejects_same_shape_semantic_or_config_change_before_mutation(tmp_path,change):
    from dataclasses import replace
    from music_critic.experiments.hierarchical_graph_analyzer.full_model import FullGraphAnalyzer,FullGraphConfig
    from music_critic.experiments.hierarchical_graph_analyzer.full_inference import load_full_weights
    cfg=FullGraphConfig(hidden=8,attention_heads=2,constructor_rounds=1,rounds=1)
    heads={'H.function':2};relations=('prepares','resolves_to')
    model=FullGraphAnalyzer(heads,{}, {},relation_types=relations,config=cfg);opt=torch.optim.Adam(model.parameters())
    state={'schema':'HiergraphFullTrainingState@1','seed':17,'schedule_fingerprint':'b'*64,'attempted_steps':0,'applied_updates':0,'skipped_unknown':0,'exposure_sha256':'c'*64}
    bindings={k:'a'*64 for k in ['source_sha256','config_sha256','data_sha256','cache_sha256']}
    path=tmp_path/'full.pt';receipt=save_full_checkpoint(path,model,opt,state,bindings=bindings)
    if change=='head':heads={'H.quality':2}
    elif change=='relations':relations=tuple(reversed(relations))
    else:cfg=replace(cfg,soft_feedback=False)
    other=FullGraphAnalyzer(heads,{}, {},relation_types=relations,config=cfg);other_opt=torch.optim.Adam(other.parameters());before=deepcopy(other.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='binding'):load_full_weights(other,path,expected_sha256=receipt['sha256'],expected_size_bytes=receipt['size_bytes'],bindings=bindings)
    with pytest.raises(ValueError,match='binding'):load_full_checkpoint(path,other,other_opt,expected_sha256=receipt['sha256'],bindings=bindings)
    assert torch.equal(rng,torch.get_rng_state()) and not other_opt.state
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in before.items())

def test_verified_unlabelled_midi_import_path(tmp_path):
    import mido
    from hashlib import sha256
    from music_critic.experiments.hierarchical_graph_analyzer.full_inference import load_raw_input
    midi=mido.MidiFile();track=mido.MidiTrack();midi.tracks.append(track)
    track.append(mido.Message('note_on',note=60,velocity=80,time=0))
    track.append(mido.Message('note_off',note=60,velocity=0,time=480))
    path=tmp_path/'unlabelled.mid';midi.save(path);blob=path.read_bytes()
    source=dict(format='midi',path=str(path),sha256=sha256(blob).hexdigest(),size_bytes=len(blob))
    piece=load_raw_input(source)
    assert len(piece.notes)==1 and piece.notes[0].pitch==60 and not piece.targets
    source['sha256']='b'*64
    with pytest.raises(ValueError,match='SHA'):load_raw_input(source)

def test_midi_reopen_is_checked_against_parsed_provenance(tmp_path,monkeypatch):
    import mido
    from hashlib import sha256
    from music_critic.adapters import midi as adapter
    from music_critic.experiments.hierarchical_graph_analyzer.full_inference import load_raw_input
    path=tmp_path/'changed.mid'
    def write(pitch):
        m=mido.MidiFile();track=mido.MidiTrack();m.tracks.append(track)
        track.extend([mido.Message('note_on',note=pitch,velocity=80,time=0),mido.Message('note_off',note=pitch,velocity=0,time=480)])
        m.save(path)
    write(60);blob=path.read_bytes();source=dict(format='midi',path=str(path),sha256=sha256(blob).hexdigest(),size_bytes=len(blob))
    original=adapter.load_midi_piece
    def reopen(*a,**k):write(64);return original(*a,**k)
    monkeypatch.setattr(adapter,'load_midi_piece',reopen)
    with pytest.raises(ValueError,match='parsed MIDI'):load_raw_input(source)

def test_inference_rejects_permuted_output_vocabulary():
    from music_critic.experiments.hierarchical_graph_analyzer.full_runtime import validate_full_config,full_vocabulary_sha256
    heads={'H':{'H.function':2},'h':{},'P':{}};labels={'H.function':['tonic','dominant']}
    config=dict(schema='HiergraphFullRun@1',prepared=True,test_locked=True,augmentation='none',freeze_policy='frozen_C3',wall_seconds=10,source_manifest_sha256='a'*64,
        c3_checkpoint={'sha256':'b'*64},model_checkpoint={'bindings':{'c3_sha256':'b'*64,'output_vocabulary_sha256':full_vocabulary_sha256(heads,labels)}},head_sizes=heads,field_values=labels)
    validate_full_config(config,'infer')
    config['field_values']={'H.function':['dominant','tonic']}
    with pytest.raises(ValueError,match='vocabulary binding'):validate_full_config(config,'infer')
    config['field_values']={}
    with pytest.raises(ValueError,match='complete active'):validate_full_config(config,'infer')
