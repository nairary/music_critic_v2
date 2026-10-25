"""Runner tests use only a tiny checked-in synthetic piece and temporary records."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from music_critic.data import load_piece
from music_critic.data.serialization import dumps_piece
from music_critic.experiments.registry import DATA_ACCESS_FIELDS, load_record, seal_record, write_record_once
from music_critic.pll.cli import load_run_config, main, run_bindings, train
from music_critic.pll.data import DataManifest, seal_manifest


@pytest.fixture(autouse=True)
def one_thread():
    before=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def prepare(tmp_path, *, attempt='PLL-SYNTHETIC-001', total=1, resume=None):
    config=dict(schema='PLLRunConfig@1',seed=17,protocol_id='synthetic-cli-v1',
        model=dict(width=32,heads=4,event_layers=1,global_layers=1,ff_dim=64,
                   field_width=16,field_heads=4,field_ff_dim=32,relation_dim=8,relation_components=2,dropout=0.),
        codec=dict(max_events=16,max_denominator=12,max_numerator=1024),
        training=dict(total_updates=total,warmup_updates=0),attempt_updates=1,
        resume=resume,sampling=dict(length_weights={'1':1.},single_track_probability=1.))
    cfg=tmp_path/(attempt+'-config.json'); cfg.write_text(json.dumps(config))
    piece=replace(load_piece(Path(__file__).parents[1]/'fixtures/data/canonical_piece_v2.json'),split='train')
    raw=dumps_piece(piece).encode(); (tmp_path/'piece.json').write_bytes(raw)
    row=dict(record_id='one',source=piece.dataset_name,work_id='component1',
        source_group_id=piece.source_group_id,piece_id=piece.piece_id,split='train',path='piece.json',
        sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(seal_manifest(dict(schema='PLLDataManifest@1',records=[row],
        source_weights={'synthetic':1.},group_before_window=True,cross_source_components_declared=True,
        split_provenance='synthetic fixture only'))))
    registry=tmp_path/'records'; registry.mkdir(exist_ok=True)
    record=registry/(attempt+'.json')
    payload=dict(schema_version='1.0.0',experiment_id=attempt,kind='engineering_gate',phase='HCM-1',
        title='Synthetic runner test',status='planned',question='Does the fixture workflow execute?',
        hypothesis='Mechanics only',decision_use='No scientific inference',repository={'git_commit':'a'*40},
        seeds=[17],budget={'planned_updates':1},results={},validity={},claims={},artifacts=[],import_provenance={},
        parents=[] if resume is None else [resume['parent_experiment_id']],
        pll=run_bindings(load_run_config(cfg),DataManifest.load(manifest),'cpu'),
        data_access={**{key:None for key in DATA_ACCESS_FIELDS},'access_evidence_status':'not_applicable_planning_record',
            'declared_policy':dict(validation_allowed=False,test_inputs_allowed=False,test_targets_allowed=False,
                                  test_metrics_allowed=False,test_selection_allowed=False,source='synthetic fixture')})
    write_record_once(record,seal_record(payload))
    return cfg,manifest,record,tmp_path/(attempt+'-output')


def test_inspect_reads_only_configuration(tmp_path,capsys):
    cfg,_,_,_=prepare(tmp_path)
    assert main(['inspect','--config',str(cfg)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['data_accessed'] is False
    assert len(result['config_sha256'])==64


def test_inspect_rejects_incomplete_field_capacity_before_execution(tmp_path, capsys):
    cfg, _, _, _ = prepare(tmp_path)
    payload = json.loads(cfg.read_text())
    payload['model']['max_fields'] = 12
    cfg.write_text(json.dumps(payload))
    assert main(['inspect', '--config', str(cfg)]) == 2
    assert 'field_capacity' in capsys.readouterr().err


def test_one_step_creates_immutable_evidence_and_finalizes_record(tmp_path):
    cfg,manifest,record,output=prepare(tmp_path)
    receipt=train(cfg,manifest,record,output)
    assert receipt['status']=='completed' and receipt['applied_updates']==1
    final=load_record(record)
    assert final['status']=='completed'
    assert final['data_access']['access_evidence_status']=='verified_primary_evidence'
    assert all(final['data_access'][key] is False for key in DATA_ACCESS_FIELDS)
    assert (tmp_path/'EXPERIMENT_LEDGER.md').exists()
    assert (output/receipt['checkpoint']).exists()
    before=record.read_bytes()
    with pytest.raises(ValueError,match='planned_record'):
        train(cfg,manifest,record,output)
    assert record.read_bytes()==before


def test_canonical_partition_mismatch_finalizes_failure_with_unknown_access(tmp_path):
    cfg,manifest,record,output=prepare(tmp_path)
    payload=json.loads(manifest.read_text())
    raw=(tmp_path/'piece.json').read_text().replace('"train"','"validation"').encode()
    (tmp_path/'piece.json').write_bytes(raw)
    payload['records'][0].update(sha256=sha256(raw).hexdigest(),size_bytes=len(raw))
    manifest.write_text(json.dumps(seal_manifest(payload)))
    planned=load_record(record)
    planned['pll']=run_bindings(load_run_config(cfg),DataManifest.load(manifest),'cpu')
    from music_critic.experiments.registry import canonical_json_bytes
    record.write_bytes(canonical_json_bytes(seal_record(planned))+b'\n')
    with pytest.raises(ValueError,match='split_mismatch'):
        train(cfg,manifest,record,output)
    failed=load_record(record)
    assert failed['status']=='failed'
    assert failed['data_access']['access_evidence_status']=='pending_primary_evidence'
    assert all(failed['data_access'][key] is None for key in DATA_ACCESS_FIELDS)
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['data_read_evidence']['unverified_reads'][0]['observed_split_claim']=='validation'
    assert not Path(str(record)+'.pll.lock').exists()


def test_resume_uses_new_attempt_and_exact_sampler_checkpoint(tmp_path):
    cfg,manifest,record,output=prepare(tmp_path,total=2)
    first=train(cfg,manifest,record,output)
    checkpoint=output/first['checkpoint']
    resume=dict(checkpoint=str(checkpoint),sha256=sha256(checkpoint.read_bytes()).hexdigest(),
                update=1,parent_experiment_id='PLL-SYNTHETIC-001')
    cfg2,manifest2,record2,output2=prepare(tmp_path,attempt='PLL-SYNTHETIC-002',total=2,resume=resume)
    second=train(cfg2,manifest2,record2,output2)
    assert second['applied_updates']==1 and second['total_updates']==2
    assert second['total_draws']==2
    assert load_record(record)['status']==load_record(record2)['status']=='completed'


def test_bad_plan_and_tracked_output_are_rejected_before_execution(tmp_path):
    cfg,manifest,record,output=prepare(tmp_path)
    cfg.write_text(cfg.read_text().replace('"seed": 17','"seed": 18'))
    with pytest.raises(ValueError,match='binding'):
        train(cfg,manifest,record,output)
    assert not output.exists() and load_record(record)['status']=='planned'
    root=Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError,match='ignored_or_external'):
        train(cfg,manifest,record,root/'docs'/'forbidden-pll-output')
    assert not (root/'docs'/'forbidden-pll-output').exists()


@pytest.mark.parametrize('bad', [dict(path='weights.pt',sha256='bad',size_bytes=1),
                                 dict(path='weights.pt',sha256='0'*64,size_bytes=True),
                                 dict(path='weights.pt',sha256='0'*64,size_bytes=1,typo=True)])
def test_inspect_rejects_unbound_pretrained_configuration(tmp_path,bad):
    cfg,_,_,_=prepare(tmp_path)
    value=json.loads(cfg.read_text()); value['pretrained']=bad
    cfg.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_run_config(cfg)
