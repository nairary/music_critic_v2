"""Full-state extension of the separate accelerated fusion campaign.

Checkpoint bytes remain immutable. A fresh queue replays only the unconsumed
orbit suffix; cache contents and speculative jobs are not training state.
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import math
import torch
from .contracts import fingerprint
from .corrected_model import model_state_fingerprint
from .full_orbit_training import FullOrbitSampler
from .harmonic_logit_fusion import fusion_model_contract
from .harmonic_fusion_resume import restore_safe_rng, capture_safe_rng, _validate_safe_rng
from .performance_prefetch import PrefetchJob
from .performance_replay import recursive_state_fingerprint


def require(value, reason):
    if not value:
        raise ValueError('fusion_campaign_resume.'+reason)


def continuation_jobs(table, start_update, stop_update):
    require(type(start_update) is int and type(stop_update) is int
            and 0 <= start_update < stop_update <= 80000, 'budget')
    sampler = FullOrbitSampler(table, seed=17, position=2*start_update)
    result = []
    for position, update in enumerate(range(start_update, stop_update)):
        pair = [sampler.peek(k) for k in range(2)]
        result.append(PrefetchJob(position, (2*update, 2*update+1),
            tuple(x.record_id for x in pair), tuple(x.shift_pc for x in pair),
            tuple(x.shift_pc for x in pair), 477102080))
        sampler.advance_after_applied_update(2)
    return tuple(result)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_resume_payload(pin, *, expected_update, expected_data_sha256, expected_model_kind='linear'):
    require(expected_model_kind in ('linear', 'attention', 'control', 'mmoe'), 'model_kind')
    path = Path(pin['path'])
    require(path.is_file() and not path.is_symlink(), 'checkpoint_regular_file')
    content = path.read_bytes()
    require(len(content)==pin['size_bytes'] and sha256(content).hexdigest()==pin['sha256'], 'checkpoint_hash')
    # Exact user-authorized, producer-bound checkpoint; no permissive pickle fallback.
    payload = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
    require(type(payload) is dict and set(payload)=={'schema_version','update','config','model',
        'optimizer','scheduler','scaler','sampler','prefetch','rng'}, 'checkpoint_fields')
    schema = ('FusionAcceleratedCampaignState@1.0.0' if expected_model_kind == 'linear'
              else 'TaskAttentionCampaignState@1.0.0')
    if expected_model_kind == 'mmoe':
        from .grouped_mmoe import STATE_SCHEMA
        schema = STATE_SCHEMA
    require(payload['schema_version']==schema
            and payload['update']==expected_update, 'checkpoint_update')
    require(payload['config'].get('model_kind', 'linear')==expected_model_kind, 'checkpoint_model_kind')
    require(payload['config']['data_manifest_sha256']==expected_data_sha256, 'data_binding')
    require(finite_tree(payload), 'nonfinite_checkpoint')
    return payload


def verify_resume_environment(observed, expected):
    for key in ('python','torch','numpy','packages','torch_cuda_build','cudnn_version',
                'gpu','cuda_device_count','cublas_workspace_config','cudnn_allow_tf32','matmul_allow_tf32'):
        require(observed.get(key)==expected.get(key), 'runtime_'+key)


def restore_campaign_state(payload, model, optimizer, scheduler, scaler, sampler, *, device):
    update = payload['update']
    require(type(update) is int and update > 0, 'update')
    saved = payload['model'];state=saved['model_state']
    from .task_attention_fusion import TaskAttentionModel, model_contract
    from .grouped_mmoe import GroupedMMoEAttentionModel, model_contract as mmoe_contract
    if type(model) is GroupedMMoEAttentionModel:
        contract = mmoe_contract(model)
    else:
        contract = model_contract(model) if type(model) is TaskAttentionModel else fusion_model_contract(model)
    require(saved['model_contract']==contract, 'model_contract')
    require(saved['model_state_fingerprint']==model_state_fingerprint(state), 'model_fingerprint')
    require(set(state)==set(model.state_dict()), 'model_keys')
    for name, expected in model.state_dict().items():
        require(state[name].shape==expected.shape and state[name].dtype==expected.dtype, 'model_tensor_'+name)
    for key in ('sampler','prefetch'):
        body={k:v for k,v in payload[key].items() if k!='fingerprint'}
        require(payload[key].get('fingerprint')==fingerprint(body), key+'_fingerprint')
    require(payload['sampler']['position']==2*update, 'sampler_position')
    queue=payload['prefetch']
    require(queue['committed_draw_cursor']==2*update
            and queue['start_draw_offset']+2*queue['committed_batch_cursor']==2*update, 'queue_position')
    require(payload['scheduler']['last_epoch']==update, 'scheduler_position')
    require(finite_tree(payload['optimizer']), 'nonfinite_optimizer')
    _validate_safe_rng(payload['rng'], device_type=device)
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    sampler.load_state_dict(payload['sampler'])
    for key,actual in (('optimizer',optimizer.state_dict()),('scheduler',scheduler.state_dict()),
                       ('scaler',scaler.state_dict()),('sampler',sampler.state_dict())):
        require(recursive_state_fingerprint(actual)==recursive_state_fingerprint(payload[key]), key+'_restore')
    require(model_state_fingerprint(model)==saved['model_state_fingerprint'], 'model_restore')
    restore_safe_rng(payload['rng'], device_type=device)
    require(recursive_state_fingerprint(capture_safe_rng())==recursive_state_fingerprint(payload['rng']), 'rng_restore')
    require(optimizer.param_groups[0]['lr']==scheduler.get_last_lr()[0], 'scheduler_lr')
    return {'restored_update':update,'next_update':update+1,'model_state_fingerprint':model_state_fingerprint(model),
        'optimizer_state_fingerprint':recursive_state_fingerprint(optimizer.state_dict()),
        'rng_state_fingerprint':recursive_state_fingerprint(payload['rng']),
        'next_lr':optimizer.param_groups[0]['lr'],'sampler_position':sampler.position,
        'queue_strategy':'fresh bounded queue over exact unconsumed orbit suffix; no discarded or repeated applied draws',
        'state_restoration_verified':True}
