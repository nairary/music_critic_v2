"""Synthetic supervisor/process and update-boundary tests; no cloud is contacted."""
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from music_critic.experiments.registry import load_record
from music_critic.pll.cli import train
from tests.pll.test_cli import prepare


@pytest.fixture(autouse=True)
def one_thread():
    import torch
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_stop_after_update_saves_consistent_checkpoint_and_aborted_record(tmp_path):
    config, manifest, record, output = prepare(tmp_path, total=2)
    calls = []

    def stop():
        calls.append(True)
        return len(calls) > 1

    receipt = train(config, manifest, record, output, stop_requested=stop)
    assert receipt['status'] == 'aborted'
    assert receipt['applied_updates'] == 1
    assert (output / receipt['checkpoint']).is_file()
    assert load_record(record)['status'] == 'aborted'
    assert not Path(str(record) + '.pll.lock').exists()


def test_stop_before_first_update_does_not_read_music(tmp_path):
    config, manifest, record, output = prepare(tmp_path)
    receipt = train(config, manifest, record, output, stop_requested=lambda: True)
    assert receipt['status'] == 'aborted'
    assert receipt['applied_updates'] == 0
    assert receipt['data_read_evidence']['verified_train_reads'] == 0
    assert receipt['checkpoint'] is None


def test_absolute_guard_and_expired_resume_cannot_refresh_budget(tmp_path):
    from music_critic.pll.cloud import guard_units, supervise
    deadline = 1800000000
    service, timer = guard_units('hcm-budget', deadline)
    assert 'Persistent=true' in timer and 'OnCalendar=' in timer
    assert 'OnBootSec' not in timer and 'Restart=' not in service
    config = dict(schema='PLLCloudRun@1', instance_id='synthetic', command=[sys.executable, '-c', 'pass'],
        cwd=str(tmp_path), output_root=str(tmp_path/'out'), deadline_unix=deadline,
        grace_seconds=1, guard_name='hcm-budget', bound_files={})
    with pytest.raises(ValueError, match='expired'):
        supervise(config, now=lambda: deadline + 1, guard_checker=lambda c: None,
                  stopper=lambda: None, identity=lambda: 'synthetic')
    assert not (tmp_path/'out').exists()


def test_supervisor_terminates_process_group_and_stops_vm(tmp_path):
    from music_critic.pll.cloud import supervise
    stopped = []
    config = dict(schema='PLLCloudRun@1', instance_id='synthetic', command=[sys.executable, '-c',
        'import signal,time; signal.signal(signal.SIGTERM, lambda *a: exit(0)); time.sleep(60)'],
        cwd=str(tmp_path), output_root=str(tmp_path/'out'), deadline_unix=time.time()+.4,
        grace_seconds=1, guard_name='hcm-budget', bound_files={})
    result = supervise(config, guard_checker=lambda c: None, stopper=lambda: stopped.append(True),
                       identity=lambda: 'synthetic')
    assert result['deadline_hit'] and result['returncode'] == 0
    assert stopped == [True]
    assert json.loads((tmp_path/'out'/'process.json').read_text())['deadline_hit']
    assert not (tmp_path/'out'/'resume.json').exists()


def test_gate_cpu_fixture_measures_resume_without_admitting_gpu(tmp_path):
    from music_critic.pll.cloud import measure_native
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.data import DataManifest
    config, manifest, _, _ = prepare(tmp_path, total=3)
    output = tmp_path/'gate'; output.mkdir()
    result = measure_native(load_run_config(config), config.parent,
        DataManifest.load(manifest), output, device='cpu')
    assert result['resume_verified'] is True
    assert result['native_cuda_admitted'] is False
    assert result['applied_updates'] == 3
    assert result['max_parameter_resume_difference'] == 0
    assert result['peak_cuda_reserved_bytes'] is None


def test_whole_random_full_encoder_gate_updates_backbone_and_restores_backend(tmp_path):
    import torch
    from music_critic.pll.cloud import measure_native, _model
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.whole_data import WholeDataManifest
    from tests.pll.test_whole_runtime import whole_run
    cfg, manifest, _, _ = whole_run(tmp_path, total=3)
    config = json.loads(cfg.read_text())
    config['freeze_policy'] = 'full'
    config['whole_piece']['freeze_local'] = False
    cfg.write_text(json.dumps(config))
    config = load_run_config(cfg)
    torch.manual_seed(config['seed'])
    initial = _model(config, cfg.parent)
    before = {k:v.clone() for k,v in initial.state_dict().items()}
    assert initial.encoder.weights_origin == 'random_initialization'
    output = tmp_path/'full-gate'; output.mkdir()
    backend = torch.backends.cudnn.enabled
    result = measure_native(config, cfg.parent, WholeDataManifest.load(manifest), output, device='cpu')
    assert torch.backends.cudnn.enabled == backend
    assert result['resume_verified'] and result['parity_cases'][0]['freeze_policy'] == 'full'
    checkpoint = torch.load(output/'gate-update-00000001.pt', weights_only=True)
    after = checkpoint['model']
    assert any(not torch.equal(before[k], after[k]) for k in before
               if k.startswith('encoder.backbone.base.local_baseline.'))
    assert all(torch.equal(before[k], after[k]) for k in before
               if k.startswith('encoder.backbone.base.task_heads.'))


def test_bound_registered_gate_finalizes_and_rejects_reuse(tmp_path):
    from music_critic.experiments.registry import canonical_json_bytes, seal_record
    from music_critic.pll.cloud import gate, gate_bindings
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.data import DataManifest
    config, manifest, record, output = prepare(tmp_path, total=3)
    cases = dict(schema='PLLNativeGateCases@1', cases=[dict(record_id='one', start_bar=0,
                                                           bars=1, role='dense')])
    cases_path = tmp_path/'cases.json'; cases_path.write_text(json.dumps(cases))
    planned = load_record(record)
    planned['pll_gate'] = gate_bindings(load_run_config(config), DataManifest.load(manifest), 'cpu', cases)
    planned['budget']['planned_updates'] = 3
    record.write_bytes(canonical_json_bytes(seal_record(planned))+b'\n')
    result = gate(config, manifest, record, output, cases_path=cases_path, device='cpu')
    assert result['status'] == 'completed'
    final = load_record(record)
    assert final['status'] == 'completed'
    assert final['budget']['actual_applied_updates'] == 3
    assert final['results']['parity_cases'][0]['role'] == 'dense'
    assert (tmp_path/'EXPERIMENT_LEDGER.md').exists()
    before = record.read_bytes()
    with pytest.raises(ValueError, match='planned_binding'):
        gate(config, manifest, record, output, cases_path=cases_path, device='cpu')
    assert record.read_bytes() == before


def test_gate_case_plan_requires_dense_and_all_sources(tmp_path):
    from music_critic.pll.cloud import validate_cases
    from music_critic.pll.data import DataManifest
    _, manifest, _, _ = prepare(tmp_path)
    cases = dict(schema='PLLNativeGateCases@1', cases=[dict(record_id='one', start_bar=0,
                                                           bars=1, role='representative')])
    with pytest.raises(ValueError, match='coverage'):
        validate_cases(cases, DataManifest.load(manifest))


def test_changed_bound_source_stops_vm_without_starting_child(tmp_path):
    from music_critic.pll.cloud import supervise
    source = tmp_path/'source.py'; source.write_text('pass')
    stopped = []
    config = dict(schema='PLLCloudRun@1', instance_id='synthetic', command=[sys.executable, str(source)],
        cwd=str(tmp_path), output_root=str(tmp_path/'out'), deadline_unix=time.time()+10,
        grace_seconds=1, guard_name='hcm-budget', bound_files={str(source):'0'*64})
    with pytest.raises(ValueError, match='bound_file_hash'):
        supervise(config, guard_checker=lambda c: None, stopper=lambda: stopped.append(True),
                  identity=lambda: 'synthetic')
    assert stopped == [True]
    assert not (tmp_path/'out').exists()


def test_signal_handler_defers_interrupt_until_requested_boundary():
    from music_critic.pll.cloud import stop_signals
    before = signal.getsignal(signal.SIGTERM)
    with stop_signals() as requested:
        assert not requested()
        os.kill(os.getpid(), signal.SIGTERM)
        assert requested()
    assert signal.getsignal(signal.SIGTERM) == before


def test_supervisor_never_stops_a_different_instance(tmp_path):
    from music_critic.pll.cloud import supervise
    stopped = []
    with pytest.raises(ValueError, match='instance_mismatch'):
        supervise(dict(instance_id='expected', output_root=str(tmp_path/'out')),
                  identity=lambda: 'another', stopper=lambda: stopped.append(True))
    assert stopped == []
    assert not (tmp_path/'out').exists()


def test_native_gate_initializes_shared_indexed_cuda_lifecycle_before_model(tmp_path, monkeypatch):
    import torch
    from music_critic import cuda_memory
    from music_critic.pll.cloud import measure_native
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.data import DataManifest
    config, manifest, _, _ = prepare(tmp_path, total=3)
    observed = []

    def boundary(device):
        observed.append(device)
        raise RuntimeError('synthetic initialization boundary')

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(cuda_memory, 'initialize_cuda_memory_statistics', boundary)
    with pytest.raises(RuntimeError, match='synthetic initialization boundary'):
        measure_native(load_run_config(config), config.parent, DataManifest.load(manifest),
                       tmp_path, device='cuda:1')
    assert observed == [torch.device('cuda:1')]
