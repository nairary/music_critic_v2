"""Bound a critic process, retain/export its evidence, then stop its own VM.

An independently armed systemd timer is required in deployment as a second
budget guard. This runner creates no resources and never restarts a process.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from scripts.upload_cloud_bundle import upload


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def validate_config(c):
    if c.get('schema')!='CriticCloudRun@1.0.0':raise ValueError('cloud.schema')
    if not re.fullmatch(r'[a-z0-9-]+',c['instance_id']):raise ValueError('cloud.instance')
    if not isinstance(c['command'],list) or not c['command'] or not all(isinstance(x,str) for x in c['command']):raise ValueError('cloud.command')
    if not 0<float(c['wall_seconds'])<=24*3600:raise ValueError('cloud.wall_seconds')
    if not 1<=float(c['export_timeout_seconds'])<=1800:raise ValueError('cloud.export_timeout')
    if not re.fullmatch(r'[a-zA-Z0-9_/-]+',c['prefix']) or '..' in c['prefix'] or c['prefix'].startswith('/'):raise ValueError('cloud.prefix')
    for key in ('cwd','output_root'):
        p=Path(c[key])
        if not p.is_absolute() or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('cloud.path')
    for path,expected in c['bound_files'].items():
        p=Path(path)
        if not p.is_file() or p.is_symlink() or digest(p)!=expected:raise ValueError('cloud.bound_file_hash')


def metadata(path):
    request=urllib.request.Request('http://169.254.169.254/computeMetadata/v1/'+path,headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=10) as r:return r.read().decode().strip()


def own_instance():return metadata('instance/id')


def stop_vm(instance):
    if own_instance()!=instance:raise ValueError('cloud.instance_changed')
    token=json.loads(metadata('instance/service-accounts/default/token'))['access_token']
    req=urllib.request.Request('https://compute.api.cloud.yandex.net/compute/v1/instances/'+instance+':stop',data=b'{}',headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as r:result=json.load(r)
    if not result.get('id') or result.get('error'):raise RuntimeError('cloud.stop_failed')
    return result['id']


def export_evidence(root,c):
    """Only the dedicated run subtree is archived; credentials live outside it."""
    members=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('cloud.export_symlink')
        if p.is_file():members.append(p)
    manifest={'files':[{'path':str(p.relative_to(root)),'sha256':digest(p),'size_bytes':p.stat().st_size} for p in members]}
    write_once(root/'artifact-manifest.json',manifest)
    archive=root.parent/(root.name+'-evidence.tar.gz')
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f,mode='w:gz') as tar:
            for p in members+[root/'artifact-manifest.json']:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    token=json.loads(metadata('instance/service-accounts/default/token'))['access_token']
    with tempfile.TemporaryDirectory(prefix='critic-export-auth-') as temporary:
        token_path=Path(temporary)/'token';token_path.write_text(token);token_path.chmod(0o600)
        receipt=upload(archive,bucket=c['bucket'],key=c['prefix']+'/'+archive.name,token_file=token_path)
    write_once(root.parent/(root.name+'-export.json'),receipt)
    return receipt


def execute(c,*,exporter=export_evidence,stopper=stop_vm,identity=own_instance):
    if identity()!=c.get('instance_id'):raise ValueError('cloud.instance_mismatch')
    handlers={n:signal.getsignal(n) for n in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    try:
        return _execute_owned(c,exporter=exporter)
    finally:
        try:stopper(c['instance_id'])
        finally:
            signal.alarm(0)
            for number,handler in handlers.items():signal.signal(number,handler)


def _execute_owned(c,*,exporter):
    validate_config(c)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    write_once(root/'config.json',c)
    started=time.monotonic();process=None;deadline=False
    result={'started_at':datetime.now(timezone.utc).isoformat(),'returncode':None,'deadline_hit':False}
    interrupted=[]
    old_handlers={}
    def on_signal(number,_frame):
        interrupted.append(number)
        if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    try:
        for number in (signal.SIGTERM,signal.SIGINT):
            old_handlers[number]=signal.signal(number,on_signal)
        with (root/'stdout.log').open('x') as out,(root/'stderr.log').open('x') as err:
            process=subprocess.Popen(c['command'],cwd=c['cwd'],stdout=out,stderr=err,start_new_session=True)
            try:process.wait(timeout=c['wall_seconds'])
            except subprocess.TimeoutExpired:
                deadline=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=120)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        result.update(returncode=process.returncode,deadline_hit=deadline,signals=interrupted,wall_seconds=time.monotonic()-started)
    except Exception as exc:
        result['process_error']=type(exc).__name__
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        write_once(root/'process.json',result)
        write_once(root/'pre-export-terminal.json',{**result,'export_status':'pending'})
        try:
            # Alarm bounds network retries/archive export while the independent
            # systemd deadline remains the final guard if this process dies.
            def timeout(_signum,_frame):raise TimeoutError('export deadline')
            previous=signal.signal(signal.SIGALRM,timeout)
            signal.alarm(int(c['export_timeout_seconds']))
            try:result['export_receipt']=exporter(root,c);result['export_status']='completed'
            finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        except Exception as exc:result.update(export_status='failed',export_error=type(exc).__name__)
        write_once(root/'cloud-terminal.json',result)
        for number,handler in old_handlers.items():signal.signal(number,handler)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    c=json.loads(Path(parser.parse_args().config).read_text())
    result=execute(c);print(json.dumps(result,sort_keys=True));return 0 if result['returncode']==0 and result['export_status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
