"""Run one bounded process group on a shared POSIX server.

This supervisor never stops the server or contacts a cloud API. It signals only
the new session/process group created for its child. Children must remain in
that group; deliberately detached daemons and SIGKILL of the supervisor are
outside this cooperative process-lifecycle contract. No retries are performed.
The default wall limit is 72 hours; an explicit seven-day limit includes the
termination grace and final child-reap allowance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ENV_KEYS = frozenset({
    'PYTHONPATH', 'PYTHONUNBUFFERED', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF',
})
DEFAULT_WALL_LIMIT_SECONDS = 72 * 3600
EXTENDED_WALL_LIMIT_SECONDS = 7 * 24 * 3600


def _digest(path):
    checksum = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return {'sha256': checksum.hexdigest(), 'size_bytes': Path(path).stat().st_size}


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _write_once(path, value):
    with Path(path).open('xb') as stream:
        stream.write(_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _absolute_path(value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('server.absolute_nonsymlink_path_required')
    path = Path(value)
    if (not path.is_absolute() or '..' in path.parts
            or any(parent.is_symlink() for parent in (path, *path.parents))):
        raise ValueError('server.absolute_nonsymlink_path_required')
    return path


def validate_config(config):
    required = {'schema', 'cwd', 'output_root', 'command', 'wall_seconds'}
    allowed = required | {'termination_grace_seconds', 'wall_limit_seconds', 'env', 'bound_files'}
    if (not isinstance(config, dict) or not required <= config.keys()
            or config.keys() - allowed or config['schema'] != 'CriticServerRun@1.0.0'):
        raise ValueError('server.config_fields_or_schema_invalid')
    if os.name != 'posix':
        raise ValueError('server.posix_required')
    cwd, root = (_absolute_path(config[name]) for name in ('cwd', 'output_root'))
    if not cwd.is_dir() or not root.parent.is_dir():
        raise ValueError('server.cwd_and_output_parent_must_exist')
    command = config['command']
    if (not isinstance(command, list) or not command
            or any(not isinstance(part, str) or '\x00' in part for part in command)
            or not command[0]):
        raise ValueError('server.command_must_be_argument_list')
    wall_limit = config.get('wall_limit_seconds', DEFAULT_WALL_LIMIT_SECONDS)
    if (type(wall_limit) is not int
            or wall_limit not in (DEFAULT_WALL_LIMIT_SECONDS, EXTENDED_WALL_LIMIT_SECONDS)):
        raise ValueError('server.wall_limit_seconds_invalid')
    grace = config.get('termination_grace_seconds', 120)
    for name, value, maximum in (
            ('wall_seconds', config['wall_seconds'], wall_limit),
            ('termination_grace_seconds', grace, 300)):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 < value <= maximum):
            raise ValueError('server.' + name + '_invalid')
    if (wall_limit == EXTENDED_WALL_LIMIT_SECONDS
            and config['wall_seconds'] + grace + 10 > wall_limit):
        raise ValueError('server.total_wall_budget_exceeds_limit')
    env = config.get('env', {})
    if (not isinstance(env, dict) or env.keys() - ENV_KEYS
            or any(not isinstance(value, str) or '\x00' in value for value in env.values())):
        raise ValueError('server.environment_override_invalid')
    bindings = config.get('bound_files', {})
    if not isinstance(bindings, dict):
        raise ValueError('server.bound_files_invalid')
    for name, checksum in bindings.items():
        path = _absolute_path(name)
        if (not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum)
                or not path.is_file() or _digest(path)['sha256'] != checksum):
            raise ValueError('server.bound_file_hash_mismatch')
    return config


def _group_exists(process):
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin can return EPERM for a group whose last member is an unreaped
        # child. Reap that child and retry the group probe; never treat a real
        # permission failure as proof that descendants have disappeared.
        if process.poll() is None:
            raise
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False


def _send_group(process, number, sent):
    try:
        os.killpg(process.pid, number)
        sent.append(int(number))
    except ProcessLookupError:
        pass
    except PermissionError:
        if _group_exists(process):
            raise


def _finish_owned_group(process, *, grace, sent, term_started=None):
    """Drain or terminate our group even when its original leader has exited."""
    process.poll()
    if _group_exists(process):
        if term_started is None:
            term_started = time.monotonic()
            _send_group(process, signal.SIGTERM, sent)
        deadline = term_started + grace
        while _group_exists(process) and time.monotonic() < deadline:
            process.poll()
            time.sleep(min(.02, max(0, deadline - time.monotonic())))
        if _group_exists(process):
            _send_group(process, signal.SIGKILL, sent)
    process.wait(timeout=10)


def execute(config):
    validate_config(config)
    root = Path(config['output_root'])
    # An existing directory is never a continuation, including a partial run.
    root.mkdir(exist_ok=False)
    _write_once(root / 'config.json', config)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    process = None
    interrupted, sent = [], []
    term_started = None
    previous = {}
    result = {'schema': 'CriticServerTerminal@1.0.0',
              'config_sha256': sha256(_bytes(config)).hexdigest(), 'started_at': started_at,
              'pid': None, 'pgid': None, 'returncode': None, 'deadline_hit': False,
              'status': 'failed', 'reason': 'launch_error', 'signals': interrupted,
              'group_signals_sent': sent}

    def on_signal(number, _frame):
        nonlocal term_started
        interrupted.append(int(number))
        if process is not None and term_started is None:
            term_started = time.monotonic()
            # SIGINT also requests the CLI's cooperative SIGTERM checkpoint path.
            _send_group(process, signal.SIGTERM, sent)

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, on_signal)
        with (root / 'stdout.log').open('xb') as stdout, (root / 'stderr.log').open('xb') as stderr:
            try:
                if not interrupted:
                    process = subprocess.Popen(config['command'], cwd=config['cwd'],
                        env={**os.environ, **config.get('env', {})}, stdin=subprocess.DEVNULL,
                        stdout=stdout, stderr=stderr, start_new_session=True, shell=False)
                    result.update(pid=process.pid, pgid=process.pid)
                _write_once(root / 'start.json', {
                    'schema': 'CriticServerStart@1.0.0', 'started_at': started_at,
                    'config_sha256': result['config_sha256'], 'pid': result['pid'], 'pgid': result['pgid'],
                    'spawned': process is not None})
                while process is not None:
                    returncode = process.poll()
                    if interrupted:
                        result.update(status='aborted', reason='signal')
                        break
                    if returncode is not None:
                        result.update(status='completed' if returncode == 0 else 'failed',
                                      reason='child_exit' if returncode == 0 else 'child_exit_nonzero')
                        break
                    remaining = config['wall_seconds'] - (time.monotonic() - started)
                    if remaining <= 0:
                        result.update(status='aborted', reason='wall_cap', deadline_hit=True)
                        break
                    time.sleep(min(.05, remaining))
                if process is None and interrupted:
                    result.update(status='aborted', reason='signal')
            except BaseException as exc:
                result.update(status='failed', reason='launch_error' if process is None else 'supervisor_error',
                              error_type=type(exc).__name__, error=str(exc))
            finally:
                if process is not None:
                    try:
                        _finish_owned_group(process, grace=config.get('termination_grace_seconds', 120),
                                            sent=sent, term_started=term_started)
                    except BaseException as exc:
                        result.update(status='failed', reason='cleanup_error',
                                      cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
                        try:
                            _send_group(process, signal.SIGKILL, sent)
                            process.wait(timeout=10)
                        except BaseException as final_exc:
                            result.update(final_cleanup_error_type=type(final_exc).__name__,
                                          final_cleanup_error=str(final_exc))
                    result['returncode'] = process.poll()
        if not (root / 'start.json').exists():
            _write_once(root / 'start.json', {
                'schema': 'CriticServerStart@1.0.0', 'started_at': started_at,
                'config_sha256': result['config_sha256'], 'pid': result['pid'], 'pgid': result['pgid'],
                'spawned': process is not None})
        result.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.monotonic() - started,
                      logs={'stdout': _digest(root / 'stdout.log'), 'stderr': _digest(root / 'stderr.log')})
        _write_once(root / 'terminal.json', result)
        return result
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = execute(json.loads(args.config.read_text()))
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0 if result['status'] == 'completed' else 2
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}),
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
