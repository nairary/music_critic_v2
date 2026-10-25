"""Bounded operator control: signal only a byte-bound critic child via pidfd.

Wait for a committed checkpoint when the user-selected next boundary is less
than half an exposure epoch away. This never rewrites training source or state.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import time


def stop_target(update, *, checkpoint_every=246, epoch_updates=246):
    if type(update) is not int or update < 0:
        raise ValueError('invalid update')
    if any(type(x) is not int or x <= 0 for x in (checkpoint_every, epoch_updates)):
        raise ValueError('invalid interval')
    following = (update // checkpoint_every + 1) * checkpoint_every
    return following if 2 * (following - update) < epoch_updates else update


def complete_events(path):
    data = Path(path).read_bytes()
    return [json.loads(line) for line in data.split(b'\n')[:-1] if line]


def checkpoint_ready(rows, target, root):
    for row in reversed(rows):
        if row['kind'] != 'checkpoint':
            continue
        item = row['payload']
        if item.get('update', -1) < target:
            continue
        path = Path(root) / item['path']
        if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():
            raise ValueError('checkpoint path')
        raw = path.read_bytes()
        if len(raw) != item['size_bytes'] or sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('checkpoint integrity')
        return item
    return None


def validate_identity(command, cwd, ticks, config):
    if command != config['command'] or str(cwd) != config['cwd'] or ticks != config['start_ticks']:
        raise ValueError('process identity changed')
    if not any(x.endswith('/scripts/run_critic_campaign.py') for x in command):
        raise ValueError('critic trainer required')


def write_new(path, payload):
    with Path(path).open('x') as f:
        json.dump(payload, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def run(config):
    root = Path(config['training_root']).resolve(strict=True)
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pid = config['pid']
    fd = None
    try:
        if type(pid) is not int or pid <= 1:
            raise ValueError('invalid pid')
        fd = os.pidfd_open(pid)
        proc = Path('/proc') / str(pid)
        command = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        ticks = int(proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19])
        validate_identity(command, os.readlink(proc/'cwd'), ticks, config)
        rows = complete_events(root/'events.jsonl')
        update = max(r['payload']['applied_update'] for r in rows if r['kind'] == 'step')
        target = config['target_update']
        if target != stop_target(config['decision_update'], checkpoint_every=config['checkpoint_every'], epoch_updates=config['epoch_updates']):
            raise ValueError('stop decision mismatch')
        if update < config['decision_update']:
            raise ValueError('progress regressed')
        write_new(output/'armed.json', dict(at=time.time(), pid=pid, observed_update=update,
                  target_update=target, config=config, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
        print('ARMED', target, flush=True)
        deadline = time.monotonic() + config['max_wait_seconds']
        checkpoint = None
        if target > config['decision_update']:
            while checkpoint is None:
                checkpoint = checkpoint_ready(complete_events(root/'events.jsonl'), target, root)
                if checkpoint is not None:
                    break
                signal.pidfd_send_signal(fd, 0)
                if time.monotonic() >= deadline:
                    raise TimeoutError('checkpoint wait expired; no signal sent')
                time.sleep(.25)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        write_new(output/'signal-sent.json', dict(at=time.time(), pid=pid,
                  signal='SIGTERM', waited_for_checkpoint=checkpoint))
        print('SIGTERM_SENT', flush=True)
        deadline = time.monotonic() + 1800
        while not (root/'summary.json').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('terminal summary wait expired')
            time.sleep(.5)
        summary = json.loads((root/'summary.json').read_text())
        if summary.get('status') != 'aborted' or summary.get('reason') != 'stop_requested':
            raise ValueError('unexpected terminal status')
        write_new(output/'terminal-observed.json', dict(at=time.time(), summary=summary,
                  summary_sha256=sha256((root/'summary.json').read_bytes()).hexdigest()))
        print('TERMINAL_OBSERVED', flush=True)
    except Exception as exc:
        write_new(output/'failure.json', dict(at=time.time(), error=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
