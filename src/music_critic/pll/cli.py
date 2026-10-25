"""Bounded reference runner. All real runs require a pre-bound planned record.

This command is an FP32, serial-microbatch implementation, not GPU admission or
an automatic dataset preparation/TEST workflow. Each attempt has a fresh output
directory and record; a continuation restores a byte-bound update checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import sys
import subprocess
import tempfile
import time

from music_critic.data.serialization import canonical_json_sha256
from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json_bytes, load_record, load_registry,
    render_markdown_ledger, seal_record, validate_record,
)
from .contracts import CodecProfile, ContextProfile, ModelConfig
from .data import DataManifest, MaskSampler, strict_json
from .runtime import TrainingConfig


def _require_sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('pll.sha256_required')


def load_run_config(path: str | Path):
    value = strict_json(Path(path))
    whole = isinstance(value,dict) and value.get('schema') == 'PLLWholeRunConfig@1'
    allowed = {'schema','model','context','codec','training','freeze_policy','seed',
               'protocol_id','sampling','pretrained','resume','attempt_updates','checkpoint_every'}
    if whole:
        allowed |= {'whole_piece','warm_start'}
    if not isinstance(value,dict) or value.get('schema') not in ('PLLRunConfig@1','PLLWholeRunConfig@1') or set(value)-allowed:
        raise ValueError('pll.run_config_schema_or_fields')
    if type(value.get('seed')) is not int or not isinstance(value.get('protocol_id'),str) or not value['protocol_id'].strip():
        raise ValueError('pll.seed_and_protocol_required')
    if value.get('freeze_policy','frozen') not in ('frozen','upper','full'):
        raise ValueError('pll.freeze_policy')
    resolved=dict(schema=value['schema'],model=asdict(ModelConfig(**value.get('model',{}))),
        context=asdict(ContextProfile(**value.get('context',{}))),
        codec=asdict(CodecProfile(**value.get('codec',{}))),
        training=asdict(TrainingConfig(**value.get('training',{}))),
        freeze_policy=value.get('freeze_policy','frozen'),seed=value['seed'],protocol_id=value['protocol_id'],
        sampling=value.get('sampling',{}),pretrained=value.get('pretrained'),resume=value.get('resume'),
        checkpoint_every=value.get('checkpoint_every',100))
    if whole:
        from .whole_contracts import WholePieceConfig, whole_context_profile
        profile = WholePieceConfig(**value.get('whole_piece',{}))
        resolved['whole_piece'] = {**asdict(profile), 'motif_scales':list(profile.motif_scales),
                                   'motif_beat_scales':list(profile.motif_beat_scales)}
        resolved['context'] = asdict(whole_context_profile(ContextProfile(**resolved['context']), profile))
        resolved['warm_start'] = value.get('warm_start')
        if profile.freeze_local and (resolved['freeze_policy'] != 'frozen' or
                                     resolved['training']['unfreeze_update'] is not None):
            raise ValueError('pll.whole_frozen_local_cannot_unfreeze')
        warm = resolved['warm_start']
        if warm is not None:
            if (not isinstance(warm,dict) or set(warm) != {'path','sha256','size_bytes','update'} or
                    not isinstance(warm['path'],str) or not warm['path'].strip() or
                    type(warm['size_bytes']) is not int or warm['size_bytes'] < 1 or
                    type(warm['update']) is not int or warm['update'] < 0):
                raise ValueError('pll.whole_warm_start_binding_required')
            _require_sha(warm['sha256'])
            if resolved['pretrained'] is not None:
                raise ValueError('pll.whole_initialization_is_ambiguous')
    ModelConfig(**resolved['model']).validate_codec(CodecProfile(**resolved['codec']))
    pretrained=resolved['pretrained']
    if pretrained is not None:
        if (not isinstance(pretrained,dict) or not {'path','sha256','size_bytes'}<=set(pretrained) or
                set(pretrained)-{'path','sha256','size_bytes','update'} or
                not isinstance(pretrained['path'],str) or not pretrained['path'] or
                type(pretrained['size_bytes']) is not int or pretrained['size_bytes']<=0 or
                type(pretrained.get('update',80000)) is not int or pretrained.get('update',80000)<0):
            raise ValueError('pll.pretrained_binding_required')
        _require_sha(pretrained['sha256'])
    resume=resolved['resume']
    if resume is not None and (not isinstance(resume,dict) or set(resume)!={'checkpoint','sha256','update','parent_experiment_id'} or
                              type(resume['update']) is not int or resume['update']<0):
        raise ValueError('pll.resume_binding_required')
    if resume is not None:
        _require_sha(resume['sha256'])
        if any(not isinstance(resume[k],str) or not resume[k].strip() for k in ('checkpoint','parent_experiment_id')):
            raise ValueError('pll.resume_binding_required')
    initial=0 if resume is None else resume['update']
    remaining=resolved['training']['total_updates']-initial
    resolved['attempt_updates']=value.get('attempt_updates',remaining)
    if (type(resolved['attempt_updates']) is not int or not 1<=resolved['attempt_updates']<=remaining or
            type(resolved['checkpoint_every']) is not int or resolved['checkpoint_every']<1):
        raise ValueError('pll.attempt_budget')
    if set(resolved['sampling'])-{'length_weights','single_track_probability'}:
        raise ValueError('pll.sampling_config_fields')
    return resolved


def run_bindings(config, manifest, device):
    from .runtime import implementation_fingerprint
    return dict(implementation_sha256=implementation_fingerprint(),config_sha256=canonical_json_sha256(config),manifest_fingerprint=manifest.fingerprint,
                protocol_id=config['protocol_id'],updates=config['attempt_updates'],
                device=device,seed=config['seed'])


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_new(path, value):
    with Path(path).open('xb') as handle:
        handle.write(canonical_json_bytes(value)+b'\n')
        handle.flush(); os.fsync(handle.fileno())


def _atomic(path, payload):
    path=Path(path)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _record_lock(path):
    # A stale lock after SIGKILL needs explicit operator inspection; it must
    # never silently authorize a competing execution of the same record.
    lock=Path(str(path)+'.pll.lock')
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(descriptor,'w') as handle:
            handle.write(str(os.getpid())); handle.flush(); os.fsync(handle.fileno())
        yield
    finally:
        lock.unlink(missing_ok=True)


def _transition(path, before, after, ledger):
    current=load_record(path)
    if current['record_fingerprint']!=before['record_fingerprint'] or current['status'] not in ('planned','running'):
        raise ValueError('pll.record_changed_or_terminal')
    if not ((current['status']=='planned' and after['status']=='running') or
            (current['status']=='running' and after['status'] in ('completed','failed','aborted','invalid'))):
        raise ValueError('pll.invalid_record_transition')
    sealed=validate_record(seal_record(after),filename=path.name)
    _atomic(path,canonical_json_bytes(sealed)+b'\n')
    _atomic(ledger,render_markdown_ledger(load_registry(path.parent)).encode())
    return sealed


def _artifact(path, record_path, role):
    raw=path.read_bytes()
    return dict(role=role,sha256=sha256(raw).hexdigest(),size_bytes=len(raw),
                availability='local',locator=os.path.relpath(path,record_path.parent))


def _admit_output(output):
    root=Path(__file__).resolve().parents[3]
    resolved=Path(output).resolve()
    if not resolved.is_relative_to(root):
        return  # --output explicitly configures an external evidence directory.
    relative=resolved.relative_to(root)
    if not (relative.parts and (relative.parts[0]=='outputs' or
            relative.parts[:2]==('artifacts','experiments'))):
        raise ValueError('pll.full_artifacts_require_ignored_or_external_output')
    result=subprocess.run(['git','check-ignore','--quiet','--',str(relative)],cwd=root,
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    if result.returncode!=0:
        raise ValueError('pll.full_artifacts_output_is_not_ignored')


def train(config_path, manifest_path, record_path, output, *, device='cpu', ledger=None,
          stop_requested=None):
    import torch
    from .encoder import PLLContextEncoder
    from .model import HCM1
    from .runtime import PLLTrainer

    config_path,record_path,output=Path(config_path),Path(record_path),Path(output)
    _admit_output(output)
    config=load_run_config(config_path)
    whole = config['schema'] == 'PLLWholeRunConfig@1'
    if whole:
        from .whole_contracts import WholePieceConfig
        from .whole_data import WholeDataManifest, WholeMaskSampler
        from .whole_model import WholePieceEncoder, WholePieceHCM
        manifest = WholeDataManifest.load(manifest_path)
        if any(row['coverage'] != 'full' for row in manifest.coverage.values()):
            raise ValueError('pll.whole_training_requires_full_coverage')
    else:
        manifest=DataManifest.load(manifest_path)
    torch_device=torch.device(device)
    if torch_device.type not in ('cpu','cuda'):
        raise ValueError('pll.runner_device_cpu_or_cuda')
    if torch_device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('pll.cuda_unavailable')
    ledger=Path(ledger) if ledger else (
        record_path.parents[2]/'EXPERIMENT_LEDGER.md' if record_path.parent.name=='records' and record_path.parent.parent.name=='experiments'
        else (record_path.parent.parent if record_path.parent.name=='records' else record_path.parent)/'EXPERIMENT_LEDGER.md')
    if ledger.name!='EXPERIMENT_LEDGER.md':
        raise ValueError('pll.ledger_filename_required')
    ledger.parent.mkdir(parents=True,exist_ok=True)
    with _record_lock(record_path):
        record=load_record(record_path)
        if record['status']!='planned' or record.get('pll')!=run_bindings(config,manifest,device):
            raise ValueError('pll.planned_record_binding_required')
        if record['budget'].get('planned_updates')!=config['attempt_updates'] or config['seed'] not in record['seeds']:
            raise ValueError('pll.record_budget_or_seed_mismatch')
        if any(record['data_access']['declared_policy'][key] for key in (
                'test_inputs_allowed','test_targets_allowed','test_metrics_allowed','test_selection_allowed')):
            raise ValueError('pll.TEST_must_be_closed')
        if config['resume'] and (not isinstance(record.get('parents'),list) or
                config['resume']['parent_experiment_id'] not in record['parents']):
            raise ValueError('pll.resume_parent_record_required')
        # Complete all checks above before claiming execution or reading a piece.
        output.mkdir(parents=True,exist_ok=False)
        _write_new(output/'planned-record.json',record)
        _write_new(output/'resolved-config.json',config)
        _write_new(output/'manifest.json',strict_json(Path(manifest_path)))
        started=_now(); start_clock=time.monotonic()
        running={**record,'status':'running','execution':dict(started_at=started,device=device,
            python=platform.python_version(),torch=torch.__version__,platform=platform.platform())}
        record=_transition(record_path,record,running,ledger)
        trainer=None; sampler=None; metrics=None; failure=None; checkpoint=None
        status='failed'; stopped=False
        initial=0 if config['resume'] is None else config['resume']['update']
        try:
            random.seed(config['seed']); torch.manual_seed(config['seed'])
            model_class = WholePieceHCM if whole else HCM1
            whole_options = {'whole_config':WholePieceConfig(**config['whole_piece'])} if whole else {}
            model=model_class(ModelConfig(**config['model']),context_profile=ContextProfile(**config['context']),
                       codec_profile=CodecProfile(**config['codec']),freeze_policy=config['freeze_policy'],**whole_options)
            if config['pretrained'] is not None:
                bound=config['pretrained']
                encoder_class = WholePieceEncoder if whole else PLLContextEncoder
                encoder_options = {'freeze_local':model.whole_config.freeze_local} if whole else {}
                model.encoder=encoder_class.from_pretrained(config_path.parent/bound['path'],
                    expected_sha256=bound['sha256'],expected_size_bytes=bound['size_bytes'],
                    expected_update=bound.get('update',80000),profile=model.context_profile,
                    freeze_policy=config['freeze_policy'],**encoder_options)
            if whole and config['warm_start'] is not None:
                from .whole_warmstart import warm_start_hcm1
                bound = config['warm_start']
                warm_start_hcm1(model, config_path.parent/bound['path'],
                    expected_sha256=bound['sha256'],expected_size_bytes=bound['size_bytes'],
                    expected_update=bound['update'])
            model.to(torch_device)
            trainer=PLLTrainer(model,TrainingConfig(**config['training']),data_id=manifest.fingerprint,
                               protocol_id=config['protocol_id'])
            sampler_class = WholeMaskSampler if whole else MaskSampler
            sampler=sampler_class(manifest,seed=config['seed'],context_profile=model.context_profile,
                                  **whole_options,**config['sampling'])
            if config['resume'] is not None:
                resume=config['resume']
                state=trainer.load_checkpoint(config_path.parent/resume['checkpoint'],expected_sha256=resume['sha256'])
                sampler.load_state_dict(state)
                if trainer.updates!=initial or trainer.draws!=sampler.draws:
                    raise ValueError('pll.resume_progress_mismatch')
            _write_new(output/'training-binding.json',trainer.binding)
            with (output/'metrics.jsonl').open('x') as log, (output/'draws.jsonl').open('x') as draws:
                for _ in range(config['attempt_updates']):
                    # A cloud SIGTERM only sets a flag. Never interrupt Adam or
                    # save a checkpoint whose sampler has advanced past weights.
                    if stop_requested is not None and stop_requested():
                        stopped=True
                        break
                    sampled=tuple(sampler.draw() for _ in range(trainer.config.accumulation))
                    for sample in sampled:
                        draws.write(json.dumps(sample.receipt(),sort_keys=True)+'\n')
                    draws.flush(); os.fsync(draws.fileno())
                    metrics=trainer.step(tuple(sample.example for sample in sampled))
                    log.write(json.dumps(dict(metrics=metrics,elapsed_seconds=time.monotonic()-start_clock),sort_keys=True,allow_nan=False)+'\n')
                    log.flush(); os.fsync(log.fileno())
                    if trainer.updates%config['checkpoint_every']==0:
                        checkpoint=output/f'checkpoint-{trainer.updates:08d}.pt'
                        trainer.save_checkpoint(checkpoint,sampler_state=sampler.state_dict())
                stopped=stopped or (stop_requested is not None and stop_requested())
                if trainer.updates>initial:
                    final=output/f'checkpoint-{trainer.updates:08d}.pt'
                    if checkpoint!=final:
                        trainer.save_checkpoint(final,sampler_state=sampler.state_dict())
                    checkpoint=final
            status='aborted' if stopped else 'completed'
        except KeyboardInterrupt as exc:
            status='aborted'; failure=exc
        except Exception as exc:
            status='failed'; failure=exc
        finally:
            updates=0 if trainer is None else max(0,trainer.updates-initial)
            access = {**record['data_access'],
                'access_evidence_status':'pending_primary_evidence' if manifest.unverified_reads else 'verified_primary_evidence',
                **{key:None if manifest.unverified_reads else False for key in DATA_ACCESS_FIELDS},
                'evidence_scope':'Unverified payload read; see attempt receipt' if manifest.unverified_reads else
                    'Only hash-bound, identity-matched TRAIN canonical records were loaded'}
            if whole:
                # A parsed contrary split is known evidence even when identity
                # checks fail. Unknown target access must stay unknown.
                observed = {row.get('observed_split_claim') for row in manifest.unverified_reads}
                violations = []
                if 'test' in observed:
                    access['test_inputs_accessed'] = True
                    violations.append('test_inputs_accessed')
                if 'validation' in observed:
                    access['validation_accessed'] = True
                    if not access['declared_policy']['validation_allowed']:
                        violations.append('validation_accessed')
                if observed & {'test', 'validation'}:
                    access['access_evidence_status'] = 'partial_primary_evidence'
                    access['evidence_scope'] = 'Known contrary split in parsed payload; other unverified access remains unknown'
                access['policy_violations'] = violations
                if violations:
                    status = 'invalid'
            receipt=dict(schema='PLLAttemptReceipt@1',experiment_id=record['experiment_id'],status=status,
                started_at=started,ended_at=_now(),elapsed_seconds=time.monotonic()-start_clock,
                applied_updates=updates,total_updates=None if trainer is None else trainer.updates,
                total_draws=None if sampler is None else sampler.draws,last_metrics=metrics,
                checkpoint=None if checkpoint is None else checkpoint.name,
                error=None if failure is None else dict(type=type(failure).__name__,message=str(failure)),
                bindings=run_bindings(config,manifest,device),
                data_read_evidence=dict(verified_train_reads=manifest.verified_train_reads,
                    unverified_reads=manifest.unverified_reads,
                    scope='Manifest and canonical identity agreement; cross-source split provenance remains caller-declared'),
                claim='bounded TRAIN execution; no quality or native admission claim')
            _write_new(output/'receipt.json',receipt)
            after=dict(record,status=status,results=dict(applied_updates=updates,last_metrics=metrics),
                budget={**record['budget'],'actual_applied_updates':updates,'actual_wall_seconds':receipt['elapsed_seconds']},
                execution={**record['execution'],'ended_at':receipt['ended_at']},
                data_access=access,
                validity={**record['validity'],'reason':('Graceful stop requested at an update boundary' if stopped else
                    None if failure is None else str(failure) or type(failure).__name__)},
                artifacts=[*record['artifacts'],*[_artifact(path,record_path,path.name) for path in sorted(output.iterdir()) if path.is_file()]])
            _transition(record_path,record,after,ledger)
        if failure is not None:
            raise failure
        return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    inspect=commands.add_parser('inspect',help='validate config without reading music or loading weights')
    inspect.add_argument('--config',type=Path,required=True)
    run=commands.add_parser('train',help='execute one pre-bound registered TRAIN attempt')
    for name in ('config','manifest','experiment-record','output'):
        run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--device',default='cpu')
    run.add_argument('--ledger',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='inspect':
            config=load_run_config(args.config)
            print(json.dumps(dict(config=config,config_sha256=canonical_json_sha256(config),data_accessed=False),indent=2,sort_keys=True))
        else:
            print(json.dumps(train(args.config,args.manifest,args.experiment_record,args.output,
                                   device=args.device,ledger=args.ledger),indent=2,sort_keys=True))
    except (ValueError,OSError,TypeError,KeyError) as exc:
        print(str(exc),file=sys.stderr); return 2
    return 0
