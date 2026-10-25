"""Strict evaluation-only loading of a completed explicit HCM schedule segment."""
from hashlib import sha256
import json
from pathlib import Path
from scripts.hcm1_stage2 import continue_full_state


def restore_segment(parent,checkpoint,binding_path,record,*,parent_sha256):
    if record['status']!='completed':raise ValueError('completed segment required')
    spec=record['hcm_stage2'];code_sha=sha256(Path(__file__).with_name('hcm1_stage2.py').read_bytes()).hexdigest()
    if spec['parent_checkpoint_sha256']!=parent_sha256 or spec['implementation_sha256']!=code_sha:
        raise ValueError('segment parent or implementation mismatch')
    horizon=parent.updates+spec['additional_updates']
    if record['results'].get('global_updates')!=horizon:raise ValueError('segment terminal progress mismatch')
    validated={}
    for path in (Path(checkpoint),Path(binding_path)):
        raw=path.read_bytes();digest=sha256(raw).hexdigest()
        if not any(a['sha256']==digest and a['size_bytes']==len(raw) for a in record['artifacts']):
            raise ValueError('segment artifact not bound to terminal record')
        validated[path]=(digest,raw if path==Path(binding_path) else None)
    checkpoint_sha=validated[Path(checkpoint)][0]
    child=continue_full_state(parent,additional_updates=spec['additional_updates'],protocol_id=spec['protocol_id'],
        parent_sha256=parent_sha256,implementation_sha256=code_sha)
    if child.binding!=json.loads(validated[Path(binding_path)][1]):raise ValueError('segment runtime binding mismatch')
    sampler_state=child.load_checkpoint(checkpoint,expected_sha256=checkpoint_sha)
    if child.updates!=horizon:raise ValueError('segment checkpoint is not final')
    return child,sampler_state
