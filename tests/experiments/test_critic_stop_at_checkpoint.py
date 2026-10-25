import importlib.util
from hashlib import sha256
from pathlib import Path
import json

import pytest

spec = importlib.util.spec_from_file_location('critic_stop', Path(__file__).parents[2] / 'scripts/stop_critic_at_checkpoint.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('update,target', [(1661,1722),(1599,1599),(1600,1722),(1476,1476)])
def test_strict_half_epoch_decision(update,target):
    assert module.stop_target(update) == target


def test_incomplete_event_is_ignored(tmp_path):
    p=tmp_path/'events.jsonl';p.write_bytes(b'{"kind":"step"}\n{"partial":')
    assert module.complete_events(p)==[{'kind':'step'}]


def test_committed_checkpoint_hash_required(tmp_path):
    p=tmp_path/'state.pt';p.write_bytes(b'state')
    row={'kind':'checkpoint','payload':{'update':1722,'path':'state.pt','size_bytes':5,'sha256':sha256(b'state').hexdigest()}}
    assert module.checkpoint_ready([row],1723,tmp_path) is None
    assert module.checkpoint_ready([row],1722,tmp_path)==row['payload']
    p.write_bytes(b'other')
    with pytest.raises(ValueError,match='integrity'):
        module.checkpoint_ready([row],1722,tmp_path)


def test_process_reuse_is_rejected():
    command=['python','/own/scripts/run_critic_campaign.py']
    config={'command':command,'cwd':'/own','start_ticks':123}
    module.validate_identity(command,'/own',123,config)
    with pytest.raises(ValueError,match='identity'):
        module.validate_identity(command,'/own',124,config)
