"""Synthetic terminal-to-new-run continuation; no real campaign data."""
from dataclasses import replace
import json
import pytest
from music_critic.critic.campaign import CampaignConfig, run_campaign, _file_digest
from .test_curriculum_training import make_trainer, make_bank


@pytest.mark.parametrize('stop_at', [0, 3, 4])
def test_new_run_continuation_matches_uninterrupted_training_and_evaluation_schedule(tmp_path, stop_at):
    from music_critic.critic.campaign import load_terminal_continuation
    dev = tuple(p for p in make_bank(split='head_dev') if p.edit_count == 1)
    parent, full, child = make_trainer(), make_trainer(), make_trainer()
    config = CampaignConfig('parent', 100, 2, 2, selection_start_draw=14)
    run_campaign(full, dev, config, tmp_path/'full')
    root = tmp_path/'parent'
    result = run_campaign(parent, dev, config, root, stop_requested=lambda: parent.applied_updates >= stop_at)
    assert result['status'] == 'aborted'
    original = {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    options = {'summary_path':str(root/'summary.json'), **_file_digest(root/'summary.json'), 'experiment_id':'parent'}
    child_config = replace(config, experiment_id='child', wall_seconds=90)
    receipt = load_terminal_continuation(child, dev, child_config, options)
    assert child.state_dict()['fingerprint'] == parent.state_dict()['fingerprint']
    run_campaign(child, dev, child_config, tmp_path/'child', continuation=receipt)
    assert child.state_dict()['fingerprint'] == full.state_dict()['fingerprint']
    def evals(path):
        return [e['payload']['draw'] for e in map(json.loads,(path/'events.jsonl').read_text().splitlines()) if e['kind']=='evaluation']
    assert evals(root)+evals(tmp_path/'child') == evals(tmp_path/'full')
    assert original == {str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect', ['summary_hash', 'checkpoint_bytes', 'journal', 'budget',
                                   'schedule', 'dev_bank', 'same_id', 'selected', 'training'])
def test_transfer_rejects_changed_evidence_or_protocol(tmp_path, defect):
    from music_critic.critic.campaign import load_terminal_continuation
    from .test_curriculum_training import config as training_config
    parent = make_trainer()
    dev = tuple(p for p in make_bank(split='head_dev') if p.edit_count == 1)
    root = tmp_path/'parent'
    cfg = CampaignConfig('parent', 100, 2, 2, selection_start_draw=14)
    result = run_campaign(parent, dev, cfg, root, stop_requested=lambda: parent.applied_updates >= (7 if defect=='selected' else 3))
    options = {'summary_path':str(root/'summary.json'), **_file_digest(root/'summary.json'), 'experiment_id':'parent'}
    child = make_trainer(cfg=training_config(learning_rate=.002)) if defect=='training' else make_trainer()
    cfg = replace(cfg, experiment_id='child', wall_seconds=90)
    if defect=='summary_hash': options['sha256']='0'*64
    elif defect=='checkpoint_bytes':
        path=root/result['latest']['path'];path.write_bytes(path.read_bytes()+b'corrupt')
    elif defect=='journal':
        p=root/'events.jsonl';p.write_text(p.read_text().replace('checkpoint','tampered',1))
    elif defect=='budget': cfg=replace(cfg,wall_seconds=101)
    elif defect=='schedule': cfg=replace(cfg,evaluate_every_updates=3)
    elif defect=='dev_bank': dev=dev[:-1]
    elif defect=='same_id': cfg=replace(cfg,experiment_id='parent')
    with pytest.raises(ValueError): load_terminal_continuation(child,dev,cfg,options)


def test_cli_continuation_rejects_nontraining_mode_before_loading(tmp_path):
    from scripts import run_critic_campaign as cli
    from .test_campaign_cli import authored_config
    config=authored_config(tmp_path)
    config['continuation']={'summary_path':str(tmp_path/'summary.json'),'sha256':'0'*64,
                            'size_bytes':1,'experiment_id':'parent'}
    with pytest.raises(ValueError,match='training-only'): cli.execute(config,mode='probe')


@pytest.mark.parametrize('interruption', ['clean', 'unclean'])
def test_early_child_restart_preserves_inherited_evaluation(tmp_path, interruption):
    from music_critic.critic.campaign import load_terminal_continuation
    dev=tuple(p for p in make_bank(split='head_dev') if p.edit_count==1)
    parent, child, full=make_trainer(),make_trainer(),make_trainer()
    cfg=CampaignConfig('parent',100,2,2,selection_start_draw=14)
    run_campaign(full,dev,cfg,tmp_path/'full')
    root=tmp_path/'parent'
    run_campaign(parent,dev,cfg,root,stop_requested=lambda:parent.applied_updates>=3)
    opts={'summary_path':str(root/'summary.json'),**_file_digest(root/'summary.json'),'experiment_id':'parent'}
    cfg=replace(cfg,experiment_id='child',wall_seconds=90)
    receipt=load_terminal_continuation(child,dev,cfg,opts)
    child_root=tmp_path/'child'
    if interruption=='clean':
        run_campaign(child,dev,cfg,child_root,continuation=receipt,stop_requested=lambda:True)
        opts={'summary_path':str(child_root/'summary.json'),**_file_digest(child_root/'summary.json'),'experiment_id':'child'}
        cfg=replace(cfg,experiment_id='grandchild',wall_seconds=80)
        receipt=load_terminal_continuation(child,dev,cfg,opts)
        run_campaign(child,dev,cfg,tmp_path/'grandchild',continuation=receipt)
    else:
        step=child.step
        def interrupted(): raise KeyboardInterrupt('synthetic interruption')
        child.step=interrupted
        with pytest.raises(KeyboardInterrupt): run_campaign(child,dev,cfg,child_root,continuation=receipt)
        child.step=step
        run_campaign(child,dev,cfg,child_root,continuation=receipt,resume=True)
    assert child.state_dict()['fingerprint']==full.state_dict()['fingerprint']
    def evals(root):
        p=root/'events.jsonl'
        return [e['payload']['draw'] for e in map(json.loads,p.read_text().splitlines()) if e['kind']=='evaluation'] if p.exists() else []
    assert evals(tmp_path/'parent')+evals(child_root)+evals(tmp_path/'grandchild')==evals(tmp_path/'full')
