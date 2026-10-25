"""Synthetic POSIX subprocess supervision; no network or GPU access."""
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def config(tmp_path, code="print('synthetic child', flush=True)"):
    return {'schema': 'CriticServerRun@1.0.0', 'cwd': str(tmp_path),
            'output_root': str(tmp_path / 'run'), 'command': [sys.executable, '-c', code],
            'wall_seconds': 5, 'termination_grace_seconds': .1,
            'env': {'PYTHONUNBUFFERED': '1'}}


def test_success_saves_immutable_bound_receipts_and_logs(tmp_path):
    from scripts import run_critic_server as runner
    c = config(tmp_path)
    bound = tmp_path / 'bound.json'
    bound.write_text('{"synthetic": true}\n')
    c['bound_files'] = {str(bound): sha256(bound.read_bytes()).hexdigest()}
    result = runner.execute(c)
    root = Path(c['output_root'])
    assert result['status'] == 'completed' and result['returncode'] == 0
    assert not result['deadline_hit'] and result['signals'] == []
    assert json.loads((root / 'config.json').read_text()) == c
    assert json.loads((root / 'start.json').read_text())['pid'] == result['pid']
    assert json.loads((root / 'terminal.json').read_text()) == result
    assert (root / 'stdout.log').read_text() == 'synthetic child\n'
    assert result['logs']['stdout']['sha256'] == sha256((root / 'stdout.log').read_bytes()).hexdigest()
    saved = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(FileExistsError):
        runner.execute(c)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == saved


@pytest.mark.parametrize('command, expected', [
    (['/nonexistent/synthetic-executable'], None),
    ([sys.executable, '-c', "import sys; print('failure', file=sys.stderr); sys.exit(7)"], 7),
])
def test_launch_and_child_failures_have_terminal_receipts(tmp_path, command, expected):
    from scripts import run_critic_server as runner
    c = config(tmp_path)
    c['command'] = command
    result = runner.execute(c)
    assert result['status'] == 'failed' and result['returncode'] == expected
    assert (Path(c['output_root']) / 'start.json').is_file()
    assert (Path(c['output_root']) / 'terminal.json').is_file()
    assert result['reason'] == ('launch_error' if expected is None else 'child_exit_nonzero')


@pytest.mark.parametrize('cooperative', [True, False])
def test_deadline_terminates_only_owned_group_with_bounded_grace(tmp_path, cooperative):
    from scripts import run_critic_server as runner
    handler = "lambda *_: (print('saved', flush=True), sys.exit(0))" if cooperative else 'signal.SIG_IGN'
    c = config(tmp_path, f"import signal,sys,time; signal.signal(signal.SIGTERM, {handler}); time.sleep(30)")
    c.update(wall_seconds=.3, termination_grace_seconds=.1)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 start_new_session=True)
    try:
        result = runner.execute(c)
        assert result['status'] == 'aborted' and result['reason'] == 'wall_cap'
        assert result['deadline_hit'] and result['wall_seconds'] < 3
        assert signal.SIGTERM in result['group_signals_sent']
        assert (signal.SIGKILL in result['group_signals_sent']) is (not cooperative)
        assert result['returncode'] == (0 if cooperative else -signal.SIGKILL)
        assert unrelated.poll() is None
        if cooperative:
            assert 'saved' in (Path(c['output_root']) / 'stdout.log').read_text()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_normal_leader_exit_does_not_leave_its_child_running(tmp_path):
    from scripts import run_critic_server as runner
    root = tmp_path / 'run'
    marker = root / 'descendant-stopped'
    child = ("import signal,sys,time; from pathlib import Path; "
             f"signal.signal(signal.SIGTERM, lambda *_: (Path({str(marker)!r}).write_text('stopped'), sys.exit(0))); "
             "print('ready',flush=True); time.sleep(30)")
    parent = ("import subprocess,sys; "
              f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True); "
              "assert p.stdout.readline().strip()=='ready'; print(p.pid,flush=True)")
    result = runner.execute(config(tmp_path, parent))
    assert result['status'] == 'completed'
    assert marker.read_text() == 'stopped'
    assert signal.SIGTERM in result['group_signals_sent']


@pytest.mark.parametrize('stop_signal', [signal.SIGTERM, signal.SIGINT])
def test_wrapper_signal_is_forwarded_and_cannot_wait_until_full_wall_cap(tmp_path, stop_signal):
    runner_path = Path(__file__).resolve().parents[2] / 'scripts' / 'run_critic_server.py'
    c = config(tmp_path, "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(30)")
    c['wall_seconds'] = 30
    config_path = tmp_path / 'wrapper.json'
    config_path.write_text(json.dumps(c))
    wrapper = subprocess.Popen([sys.executable, str(runner_path), '--config', str(config_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        log = tmp_path / 'run' / 'stdout.log'
        deadline = time.monotonic() + 5
        while not log.exists() or 'ready' not in log.read_text():
            assert wrapper.poll() is None and time.monotonic() < deadline
            time.sleep(.01)
        wrapper.send_signal(stop_signal)
        stdout, stderr = wrapper.communicate(timeout=3)
        assert wrapper.returncode == 2, stderr
        result = json.loads(stdout)
        assert result['status'] == 'aborted' and result['reason'] == 'signal'
        assert result['signals'] == [stop_signal]
        assert result['returncode'] == -signal.SIGKILL
        assert result['group_signals_sent'] == [signal.SIGTERM, signal.SIGKILL]
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=5)


@pytest.mark.parametrize('changes', [
    {'wall_seconds': 0}, {'wall_seconds': float('nan')}, {'wall_seconds': True},
    {'wall_seconds': 72 * 3600 + 1}, {'termination_grace_seconds': 301},
    {'env': {'PATH': '/unexpected'}}, {'command': 'echo unsafe-shell'},
    {'output_root': 'relative'},
])
def test_invalid_configs_fail_before_output_creation(tmp_path, changes):
    from scripts import run_critic_server as runner
    c = {**config(tmp_path), **changes}
    with pytest.raises(ValueError):
        runner.execute(c)
    assert not (tmp_path / 'run').exists()


def test_extended_wall_budget_requires_explicit_opt_in_and_is_saved(tmp_path):
    from scripts import run_critic_server as runner
    c = config(tmp_path)
    c['wall_seconds'] = 132 * 3600
    with pytest.raises(ValueError, match='wall_seconds_invalid'):
        runner.execute(c)
    assert not Path(c['output_root']).exists()
    c['wall_limit_seconds'] = 7 * 24 * 3600
    result = runner.execute(c)  # The synthetic child exits immediately.
    assert result['status'] == 'completed'
    assert json.loads((Path(c['output_root']) / 'config.json').read_text()) == c
    assert result['config_sha256'] == sha256(runner._bytes(c)).hexdigest()


@pytest.mark.parametrize('limit', [True, False, 604800., '604800', None, 0,
                                  float('nan'), float('inf'), 604801, 4 * 24 * 3600])
def test_wall_limit_requires_one_of_two_explicit_integer_limits(tmp_path, limit):
    from scripts import run_critic_server as runner
    c = {**config(tmp_path), 'wall_limit_seconds': limit}
    with pytest.raises(ValueError, match='wall_limit_seconds_invalid'):
        runner.execute(c)
    assert not Path(c['output_root']).exists()


@pytest.mark.parametrize('grace', [None, .1, 300])
def test_extended_wall_budget_includes_grace_and_final_reap(tmp_path, grace):
    from scripts import run_critic_server as runner
    c = config(tmp_path)
    c['wall_limit_seconds'] = 7 * 24 * 3600
    if grace is None:
        del c['termination_grace_seconds']
    else:
        c['termination_grace_seconds'] = grace
    effective_grace = 120 if grace is None else grace
    c['wall_seconds'] = 604800 - effective_grace - 10
    assert runner.validate_config(c) == c
    c['wall_seconds'] += .5
    with pytest.raises(ValueError, match='total_wall_budget_exceeds_limit'):
        runner.execute(c)
    assert not Path(c['output_root']).exists()
    c['wall_seconds'] = 604801
    with pytest.raises(ValueError, match='wall_seconds_invalid'):
        runner.execute(c)


@pytest.mark.parametrize('explicit', [False, True])
def test_default_wall_budget_boundary_retains_existing_semantics(tmp_path, explicit):
    from scripts import run_critic_server as runner
    c = config(tmp_path)
    c.update(wall_seconds=72 * 3600, termination_grace_seconds=300)
    if explicit:
        c['wall_limit_seconds'] = 72 * 3600
    assert runner.validate_config(c) == c


def test_bound_hash_and_symlink_paths_fail_before_output_creation(tmp_path):
    from scripts import run_critic_server as runner
    bound = tmp_path / 'source.py'
    bound.write_text('synthetic')
    c = config(tmp_path)
    c['bound_files'] = {str(bound): '0' * 64}
    with pytest.raises(ValueError, match='bound_file_hash'):
        runner.execute(c)
    c['bound_files'] = {}
    link = tmp_path / 'alias'
    link.symlink_to(tmp_path, target_is_directory=True)
    c['output_root'] = str(link / 'run')
    with pytest.raises(ValueError, match='nonsymlink'):
        runner.execute(c)
    assert not (tmp_path / 'run').exists()


def test_allowed_environment_override_is_child_only(tmp_path):
    from scripts import run_critic_server as runner
    original = os.environ.get('OMP_NUM_THREADS')
    c = config(tmp_path, "import os; print(os.environ['OMP_NUM_THREADS'])")
    c['env'] = {'OMP_NUM_THREADS': '3'}
    assert runner.execute(c)['status'] == 'completed'
    assert (tmp_path / 'run' / 'stdout.log').read_text() == '3\n'
    assert os.environ.get('OMP_NUM_THREADS') == original


def test_cleanup_error_still_kills_and_reaps_own_child(tmp_path, monkeypatch):
    from scripts import run_critic_server as runner
    def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError('synthetic cleanup failure')
    monkeypatch.setattr(runner, '_finish_owned_group', fail_cleanup)
    c = config(tmp_path, 'import time; time.sleep(30)')
    c['wall_seconds'] = .1
    result = runner.execute(c)
    try:
        assert result['reason'] == 'cleanup_error'
        assert result['returncode'] == -signal.SIGKILL
        assert result['status'] == 'failed'
    finally:
        # Keep even the red regression run from leaving a synthetic process.
        if result['returncode'] is None:
            os.killpg(result['pid'], signal.SIGKILL)
            os.waitpid(result['pid'], 0)
