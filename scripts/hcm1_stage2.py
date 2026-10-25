"""Explicit full-state second training segment; ordinary PLL resume stays strict.

The parent is loaded under its exact original binding before a new LR segment
is constructed. This script's digest is part of the child's own binding.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from music_critic.pll.runtime import PLLTrainer, TrainingConfig, _cpu_tree


class SegmentTrainer(PLLTrainer):
    def __init__(self, model, parent_config, *, start_update, additional_updates,
                 data_id, protocol_id, parent_binding, parent_sha256, implementation_sha256):
        if start_update != parent_config.total_updates or additional_updates <= parent_config.warmup_updates:
            raise ValueError('stage2 requires completed parent and positive segment')
        self.segment_start=start_update
        self.parent_config=parent_config
        config=replace(parent_config,total_updates=start_update+additional_updates)
        super().__init__(model,config,data_id=data_id,protocol_id=protocol_id)
        self.binding['continuation']=dict(schema='HCM1NewScheduleSegment@1',
            parent_binding=copy.deepcopy(parent_binding),parent_checkpoint_sha256=parent_sha256,
            start_update=start_update,additional_updates=additional_updates,
            preserved='model Adam moments/steps Python CPU CUDA RNG sampler',
            learning_rate='new warmup and cosine segment; parent schedule unchanged',
            implementation_sha256=implementation_sha256)

    def _lr_multiplier(self, updates=None):
        u=self.updates if updates is None else updates
        if u<self.segment_start:
            # Validate the parent's last Adam group LR without retroactively
            # extending the completed first cosine schedule.
            c=self.parent_config
            if u<c.warmup_updates:return (u+1)/c.warmup_updates
            import math
            return .5*(1+math.cos(math.pi*(u-c.warmup_updates)/(c.total_updates-c.warmup_updates)))
        local=u-self.segment_start;c=self.config
        if local<c.warmup_updates:return (local+1)/c.warmup_updates
        import math
        length=c.total_updates-self.segment_start
        return .5*(1+math.cos(math.pi*(local-c.warmup_updates)/(length-c.warmup_updates)))


def continue_full_state(parent, *, additional_updates, protocol_id, parent_sha256,
                        implementation_sha256):
    if parent._in_step or parent.updates!=parent.config.total_updates:
        raise ValueError('completed update-boundary parent required')
    if parent.model.encoder.freeze_policy!='frozen' or parent.config.unfreeze_update is not None:
        raise ValueError('this continuation supports frozen parent only')
    rng=parent._rng_state()
    child=SegmentTrainer(copy.deepcopy(parent.model),parent.config,start_update=parent.updates,
        additional_updates=additional_updates,data_id=parent.binding['data_id'],protocol_id=protocol_id,
        parent_binding=parent.binding,parent_sha256=parent_sha256,implementation_sha256=implementation_sha256)
    state=_cpu_tree(parent.optimizer.state_dict())
    child._validate_optimizer(state,parent.updates,sorted(state['state']))
    child.optimizer.load_state_dict(state)
    child.updates,child.draws=parent.updates,parent.draws
    child._restore_rng(rng)
    return child


def load_parent(config_path,manifest_path,checkpoint,checkpoint_sha,binding_path,record_path,*,device):
    """Read authenticated completed parent; no weight-only or version bypass."""
    import torch
    from music_critic.experiments.registry import load_record
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.cloud import _model
    from music_critic.pll.data import DataManifest,MaskSampler,strict_json
    config_path=Path(config_path);config=load_run_config(config_path)
    record=load_record(record_path)
    if record['status']!='completed' or record['results']['applied_updates']!=config['training']['total_updates']:
        raise ValueError('parent did not complete declared horizon')
    raw=Path(checkpoint).read_bytes()
    if sha256(raw).hexdigest()!=checkpoint_sha or not any(a['sha256']==checkpoint_sha and a['size_bytes']==len(raw) for a in record['artifacts']):
        raise ValueError('parent checkpoint not bound to terminal record')
    binding=strict_json(Path(binding_path))
    if not any(a['sha256']==sha256(Path(binding_path).read_bytes()).hexdigest() for a in record['artifacts']):
        raise ValueError('parent training binding not bound to terminal record')
    whole = config['schema'] == 'PLLWholeRunConfig@1'
    if whole:
        from music_critic.pll.whole_data import WholeDataManifest, WholeMaskSampler
        manifest = WholeDataManifest.load(manifest_path)
        if any(row['coverage'] != 'full' for row in manifest.coverage.values()):
            raise ValueError('whole continuation requires full sources')
    else:
        manifest=DataManifest.load(manifest_path)
    model=_model(config,config_path.parent).to(torch.device(device))
    parent=PLLTrainer(model,TrainingConfig(**config['training']),data_id=manifest.fingerprint,protocol_id=config['protocol_id'])
    # Checkpoint contracts keep tuples; JSON evidence represents them as lists.
    if json.loads(json.dumps(parent.binding))!=binding:
        raise ValueError('parent runtime or configuration differs')
    if whole:
        sampler=WholeMaskSampler(manifest,seed=config['seed'],context_profile=model.context_profile,
                                 whole_config=model.whole_config,**config['sampling'])
    else:
        sampler=MaskSampler(manifest,seed=config['seed'],context_profile=model.context_profile,**config['sampling'])
    state=parent.load_checkpoint(checkpoint,expected_sha256=checkpoint_sha)
    sampler.load_state_dict(state)
    if parent.updates!=config['training']['total_updates'] or parent.draws!=sampler.draws:
        raise ValueError('parent progress or sampler mismatch')
    return parent,sampler


def run(args):
    import torch
    from music_critic.experiments.registry import load_record,DATA_ACCESS_FIELDS
    from music_critic.pll.cli import _transition,_record_lock,_write_new,_artifact,_now,_admit_output
    from music_critic.pll.cloud import stop_signals
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    output=Path(args.output);record_path=Path(args.record);ledger=record_path.parents[2]/'EXPERIMENT_LEDGER.md'
    _admit_output(output)
    if args.wall_seconds<=0:raise ValueError('positive wall budget required')
    code_sha=sha256(Path(__file__).read_bytes()).hexdigest()
    expected=dict(parent_checkpoint_sha256=args.parent_sha256,additional_updates=args.additional_updates,
                  implementation_sha256=code_sha,protocol_id=args.protocol_id,wall_seconds=args.wall_seconds)
    with _record_lock(record_path):
        record=load_record(record_path)
        if record['status']!='planned' or record.get('hcm_stage2')!=expected:
            raise ValueError('stage2 planned binding mismatch')
        if record['data_access']['declared_policy']['validation_allowed'] or any(
                record['data_access']['declared_policy'][name] for name in ('test_inputs_allowed','test_targets_allowed','test_metrics_allowed','test_selection_allowed')):
            raise ValueError('stage2 TRAIN-only policy required')
        output.mkdir(parents=True,exist_ok=False);_write_new(output/'planned-record.json',record)
        record=_transition(record_path,record,dict(record,status='running',execution={'started_at':_now(),'device':args.device,'torch':str(torch.__version__)}),ledger)
        started=time.monotonic();child=None;sampler=None;error=None;status='failed';checkpoint=None
        try:
            parent,sampler=load_parent(args.parent_config,args.manifest,args.parent_checkpoint,args.parent_sha256,
                                      args.parent_binding,args.parent_record,device=args.device)
            child=continue_full_state(parent,additional_updates=args.additional_updates,protocol_id=args.protocol_id,
                                      parent_sha256=args.parent_sha256,implementation_sha256=code_sha)
            del parent
            _write_new(output/'training-binding.json',child.binding)
            checkpoint=output/f'checkpoint-{child.updates:08d}.pt'
            child.save_checkpoint(checkpoint,sampler_state=sampler.state_dict())
            with stop_signals() as stop, (output/'metrics.jsonl').open('x') as log, (output/'draws.jsonl').open('x') as draws:
                for _ in range(args.additional_updates):
                    if stop() or time.monotonic()-started>=args.wall_seconds:break
                    sampled=tuple(sampler.draw() for _ in range(child.config.accumulation))
                    for sample in sampled:draws.write(json.dumps(sample.receipt(),sort_keys=True)+'\n')
                    draws.flush();os.fsync(draws.fileno())
                    metrics=child.step(tuple(x.example for x in sampled))
                    log.write(json.dumps(dict(elapsed_seconds=time.monotonic()-started,metrics=metrics),sort_keys=True,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
                    if child.updates%100==0:
                        checkpoint=output/f'checkpoint-{child.updates:08d}.pt';child.save_checkpoint(checkpoint,sampler_state=sampler.state_dict())
                status='completed' if child.updates==child.config.total_updates else 'aborted'
                final=output/f'checkpoint-{child.updates:08d}.pt'
                if final!=checkpoint:child.save_checkpoint(final,sampler_state=sampler.state_dict())
                checkpoint=final
        except Exception as exc:
            error={'type':type(exc).__name__,'message':str(exc)}
            raise
        finally:
            receipt=dict(status=status,error=error,global_updates=None if child is None else child.updates,
                added_updates=None if child is None else child.updates-child.segment_start,
                draws=None if child is None else child.draws,elapsed_seconds=time.monotonic()-started,
                checkpoint=None if checkpoint is None else checkpoint.name)
            _write_new(output/'receipt.json',receipt)
            after=dict(record,status=status,results=receipt,validity={'reason':None if status=='completed' else str(error or 'operator stop')},
                data_access={**record['data_access'],'access_evidence_status':'verified_primary_evidence' if sampler is not None and not sampler.manifest.unverified_reads else 'pending_primary_evidence',
                  **{key:False if sampler is not None and not sampler.manifest.unverified_reads else None for key in DATA_ACCESS_FIELDS}},
                artifacts=record['artifacts']+[_artifact(p,record_path,p.name) for p in output.iterdir() if p.is_file()])
            _transition(record_path,record,after,ledger)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent-config','manifest','parent-checkpoint','parent-sha256','parent-binding','parent-record','record','output','protocol-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--additional-updates',type=int,default=10000)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--wall-seconds',type=int,default=86400)
    run(parser.parse_args())
