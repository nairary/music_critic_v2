"""Cloud runner tests use tiny subprocesses, no network or real models."""
import json
import sys
from pathlib import Path
import pytest
from scripts.run_critic_cloud import execute, validate_config


def config(tmp_path, command):
    return {'schema':'CriticCloudRun@1.0.0','instance_id':'fixture-instance',
            'command':command,'cwd':str(tmp_path),'output_root':str(tmp_path/'evidence'),
            'wall_seconds':2,'export_timeout_seconds':30,'bucket':'fixture-bucket',
            'prefix':'experiments/fixture','bound_files':{}}


def test_success_exports_before_stop_and_retains_exit(tmp_path):
    c=config(tmp_path,[sys.executable,'-c','print("trained")'])
    calls=[]
    r=execute(c,exporter=lambda root,c: calls.append(('export',json.loads((root/'process.json').read_text())['returncode'])),stopper=lambda _:calls.append(('stop',None)),identity=lambda:'fixture-instance')
    assert calls==[('export',0),('stop',None)]
    assert r['returncode']==0
    assert (Path(c['output_root'])/'stdout.log').read_text().strip()=='trained'


def test_export_failure_still_stops_vm(tmp_path):
    c=config(tmp_path,[sys.executable,'-c','raise SystemExit(3)'])
    calls=[]
    def fail(*_):raise OSError('offline')
    r=execute(c,exporter=fail,stopper=lambda _:calls.append('stop'),identity=lambda:'fixture-instance')
    assert calls==['stop'] and r['returncode']==3 and r['export_status']=='failed'
    assert (Path(c['output_root'])/'cloud-terminal.json').exists()


def test_refuses_identity_mismatch_before_command_and_stop(tmp_path):
    c=config(tmp_path,[sys.executable,'-c','print("unsafe")'])
    calls=[]
    with pytest.raises(ValueError,match='instance'):
        execute(c,exporter=lambda *_:None,stopper=lambda _:calls.append('stop'),identity=lambda:'other-instance')
    assert not calls and not Path(c['output_root']).exists()


def test_deadline_kills_process_and_exports_partial(tmp_path):
    c=config(tmp_path,[sys.executable,'-c','import time;time.sleep(10)']);c['wall_seconds']=.1
    calls=[]
    r=execute(c,exporter=lambda *_:calls.append('export'),stopper=lambda _:calls.append('stop'),identity=lambda:'fixture-instance')
    assert r['deadline_hit'] and r['returncode']!=0 and calls==['export','stop']


def test_source_mismatch_and_path_guards(tmp_path):
    p=tmp_path/'source.py';p.write_text('changed')
    c=config(tmp_path,[sys.executable,'-c','pass']);c['bound_files']={str(p):'0'*64}
    with pytest.raises(ValueError,match='hash'):validate_config(c)
    c['bound_files']={};c['prefix']='../secret'
    with pytest.raises(ValueError,match='prefix'):validate_config(c)


@pytest.mark.parametrize('failure', ['existing_root', 'finalization_write', 'source_hash'])
def test_preflight_or_finalization_failure_still_stops_owned_vm(tmp_path, monkeypatch, failure):
    import scripts.run_critic_cloud as runner
    c=config(tmp_path,[sys.executable,'-c','pass']);calls=[]
    if failure=='existing_root':Path(c['output_root']).mkdir()
    elif failure=='source_hash':c['bound_files']={str(tmp_path/'missing'):'0'*64}
    else:
        original=runner.write_once
        def write(path,value):
            if path.name=='process.json':raise OSError('disk full')
            return original(path,value)
        monkeypatch.setattr(runner,'write_once',write)
    with pytest.raises((OSError,ValueError)):
        execute(c,exporter=lambda *_:None,stopper=lambda _:calls.append('stop'),identity=lambda:'fixture-instance')
    assert calls==['stop']
