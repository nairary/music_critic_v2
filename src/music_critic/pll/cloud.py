"""FP32 admission and an absolute-deadline, one-attempt cloud supervisor.

No resources are provisioned here. Deployment must arm the independent root
timer before invoking the supervisor. A stopped VM never resumes automatically;
interrupted records/locks require evidence inspection and a new attempt ID.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time
import urllib.request

from .cli import (_admit_output, _artifact, _now, _record_lock, _transition,
                  _write_new, load_run_config, run_bindings, train)
from .data import DataManifest, MaskSampler, strict_json
from music_critic.experiments.registry import DATA_ACCESS_FIELDS, load_record


@contextmanager
def stop_signals():
    """Signal handlers only set a flag; Adam cannot be interrupted mid-update."""
    stopped = []
    previous = {}
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, lambda n, _: stopped.append(n))
        yield lambda: bool(stopped)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def guard_units(name: str, deadline_unix: float) -> tuple[str, str]:
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,62}', name):
        raise ValueError('pll.cloud.guard_name')
    if not math.isfinite(deadline_unix) or deadline_unix <= 0:
        raise ValueError('pll.cloud.deadline')
    date = datetime.fromtimestamp(deadline_unix, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    service = ('[Unit]\nDescription=Absolute HCM budget shutdown\n'
               '[Service]\nType=oneshot\nExecStart=/usr/bin/systemctl poweroff\n')
    timer = ('[Unit]\nDescription=Absolute HCM budget deadline\n'
             f'[Timer]\nOnCalendar={date}\nPersistent=true\nAccuracySec=1s\n'
             f'Unit={name}.service\n[Install]\nWantedBy=timers.target\n')
    return service, timer


def verify_guard(config):
    """Require root-owned immutable units AND the loaded timer's exact deadline."""
    name, deadline = config['guard_name'], config['deadline_unix']
    for suffix, content in zip(('service', 'timer'), guard_units(name, deadline)):
        path = Path('/etc/systemd/system') / (name + '.' + suffix)
        if path.is_symlink() or not path.is_file():
            raise ValueError('pll.cloud.guard_file')
        stat = path.stat()
        if stat.st_uid != 0 or stat.st_mode & 0o022 or path.read_text() != content:
            raise ValueError('pll.cloud.guard_ownership_or_content')
    env = dict(os.environ, TZ='UTC', LC_ALL='C')
    command = ['systemctl', 'show', name + '.timer', '--property=ActiveState',
               '--property=NextElapseUSecRealtime', '--property=UnitFileState']
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    properties = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if properties.get('ActiveState') != 'active' or properties.get('UnitFileState') != 'enabled':
        raise ValueError('pll.cloud.guard_not_armed')
    try:
        next_time = datetime.strptime(properties['NextElapseUSecRealtime'],
                                      '%a %Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc).timestamp()
    except (KeyError, ValueError) as exc:
        raise ValueError('pll.cloud.guard_deadline_unverified') from exc
    if abs(next_time - deadline) >= 1:
        raise ValueError('pll.cloud.guard_deadline_mismatch')


def _validate_supervisor(config, now):
    if config.get('schema') != 'PLLCloudRun@1':
        raise ValueError('pll.cloud.schema')
    deadline = config['deadline_unix']
    if isinstance(deadline, bool) or not isinstance(deadline, (float, int)) or not math.isfinite(deadline):
        raise ValueError('pll.cloud.deadline')
    if deadline <= now():
        raise ValueError('pll.cloud.deadline_expired')
    segment = config.get('segment_deadline_unix', deadline)
    if (isinstance(segment, bool) or not isinstance(segment, (float, int)) or
            not math.isfinite(segment) or not now() < segment <= deadline or
            not 0 < config['grace_seconds'] <= 120 or segment-now() > 23*3600 or
            deadline-now() > 48*3600):
        raise ValueError('pll.cloud.wall_budget')
    guard_units(config['guard_name'], deadline)
    command = config['command']
    if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError('pll.cloud.command')
    for name in ('cwd', 'output_root'):
        path = Path(config[name])
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('pll.cloud.path')
    if not isinstance(config['bound_files'], dict):
        raise ValueError('pll.cloud.bound_files')
    for name, expected in config['bound_files'].items():
        path = Path(name)
        if (not path.is_absolute() or path.is_symlink() or not path.is_file() or
                sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('pll.cloud.bound_file_hash')
    if config.get('implementation_sha256') is not None:
        from .runtime import implementation_fingerprint
        if implementation_fingerprint() != config['implementation_sha256']:
            raise ValueError('pll.cloud.source_changed')


def poweroff():
    # Deployment runs this supervisor as root. Training may be wrapped with
    # runuser in its bound list-form command. No sudo credentials are stored.
    subprocess.run(['/usr/bin/systemctl', 'poweroff'], check=True, timeout=15)


def own_instance():
    request = urllib.request.Request('http://169.254.169.254/computeMetadata/v1/instance/id',
                                     headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode().strip()


def load_supervisor_config(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_uid != 0 or path.stat().st_mode & 0o022:
        raise ValueError('pll.cloud.config_requires_root_ownership')
    return strict_json(path)


def supervise(config, *, now=time.time, guard_checker=verify_guard, stopper=poweroff,
              identity=own_instance):
    """Run exactly one command, preserve logs, then stop this VM on every exit.

    The supervisor starts graceful termination before the independent hard
    deadline. ACPI preemption may still give less time than a long update;
    previous immutable checkpoints are the recovery boundary in that case.
    """
    # A budget guard is authorized for this dedicated VM only. A copied config
    # must never shut down a different machine, even on admission failure.
    expected_instance = config.get('instance_id')
    if not isinstance(expected_instance, str) or not re.fullmatch(r'[a-z0-9-]+', expected_instance):
        raise ValueError('pll.cloud.instance_id')
    if identity() != expected_instance:
        raise ValueError('pll.cloud.instance_mismatch')
    process = None
    root = Path(config['output_root'])
    try:
        _validate_supervisor(config, now)
        guard_checker(config)
        root.mkdir(parents=True, exist_ok=False)
        _write_new(root/'supervisor-config.json', config)
        started = now()
        result = dict(started_at=_now(), returncode=None, deadline_hit=False, signal_stop=False)
        with stop_signals() as requested:
            with (root/'stdout.log').open('x') as stdout, (root/'stderr.log').open('x') as stderr:
                process = subprocess.Popen(config['command'], cwd=config['cwd'], stdout=stdout,
                                           stderr=stderr, start_new_session=True)
                # Leave time for a normal boundary checkpoint before hard stop.
                segment = config.get('segment_deadline_unix', config['deadline_unix'])
                soft_deadline = segment - min(config['grace_seconds'], max(0, segment-started)/4)
                while process.poll() is None:
                    if requested() or now() >= soft_deadline:
                        result.update(deadline_hit=now() >= soft_deadline, signal_stop=requested())
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=min(config['grace_seconds'], max(.01, segment-now())))
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                            result['forced_kill'] = True
                        break
                    time.sleep(min(.1, max(.001, soft_deadline - now())))
            result.update(returncode=process.returncode, ended_at=_now(), elapsed_seconds=now()-started)
            _write_new(root/'process.json', result)
        return result
    finally:
        try:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        finally:
            stopper()


def _model(config, directory):
    from .contracts import CodecProfile, ContextProfile, ModelConfig
    from .encoder import PLLContextEncoder
    from .model import HCM1
    if config['schema'] == 'PLLWholeRunConfig@1':
        from .whole_contracts import WholePieceConfig
        from .whole_model import WholePieceHCM, WholePieceEncoder
        from .whole_warmstart import warm_start_hcm1
        model = WholePieceHCM(ModelConfig(**config['model']),
            whole_config=WholePieceConfig(**config['whole_piece']),
            context_profile=ContextProfile(**config['context']),
            codec_profile=CodecProfile(**config['codec']), freeze_policy=config['freeze_policy'])
        if config['pretrained'] is not None:
            bound = config['pretrained']
            model.encoder = WholePieceEncoder.from_pretrained(directory/bound['path'],
                expected_sha256=bound['sha256'], expected_size_bytes=bound['size_bytes'],
                expected_update=bound.get('update',80000), profile=model.context_profile,
                freeze_policy=config['freeze_policy'], freeze_local=model.whole_config.freeze_local)
        if config['warm_start'] is not None:
            bound = config['warm_start']
            warm_start_hcm1(model, directory/bound['path'], expected_sha256=bound['sha256'],
                expected_size_bytes=bound['size_bytes'], expected_update=bound['update'])
        return model
    model = HCM1(ModelConfig(**config['model']), context_profile=ContextProfile(**config['context']),
                 codec_profile=CodecProfile(**config['codec']), freeze_policy=config['freeze_policy'])
    if config['pretrained'] is not None:
        bound = config['pretrained']
        model.encoder = PLLContextEncoder.from_pretrained(directory/bound['path'],
            expected_sha256=bound['sha256'], expected_size_bytes=bound['size_bytes'],
            expected_update=bound.get('update', 80000), profile=model.context_profile,
            freeze_policy=config['freeze_policy'])
    return model


def _compare_tensors(left, right, *, atol, rtol, label):
    import torch
    if set(left) != set(right):
        raise ValueError('pll.gate.' + label + '_keys')
    maximum = 0.
    for key in left:
        a, b = left[key].detach().cpu(), right[key].detach().cpu()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('pll.gate.' + label + '_nonfinite')
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol,
                                   msg=lambda m: 'pll.gate.' + label + ':' + key + '\n' + m)
        if a.numel():
            maximum = max(maximum, float((a-b).abs().max()))
    return maximum


def gate_bindings(config, manifest, device, cases):
    from music_critic.data.serialization import canonical_json_sha256
    return dict(run_bindings(config, manifest, device), gate_schema='PLLNativeGate@1',
                physical_updates=3, logical_updates=2, cases_sha256=canonical_json_sha256(cases))


def validate_cases(cases, manifest):
    if not isinstance(cases, dict) or set(cases) != {'schema', 'cases'} or cases['schema'] != 'PLLNativeGateCases@1':
        raise ValueError('pll.gate.cases_schema')
    rows = cases['cases']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError('pll.gate.cases_count')
    covered = set()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'record_id', 'start_bar', 'bars', 'role'} or
                row['record_id'] not in manifest.by_id or type(row['start_bar']) is not int or row['start_bar'] < 0 or
                type(row['bars']) is not int or row['bars'] not in (1, 2, 4) or
                row['role'] not in ('representative', 'dense')):
            raise ValueError('pll.gate.case')
        covered.add(manifest.by_id[row['record_id']].source)
    if covered != set(manifest.source_weights) or not any(row['role']=='dense' for row in rows):
        raise ValueError('pll.gate.source_or_dense_coverage')


def _case_examples(cases, manifest, profile, *, whole_config=None):
    from .contracts import MaskQuery
    from .masking import split_masked_piece
    for row in cases['cases']:
        piece = manifest.load_piece(row['record_id'])
        start, end = row['start_bar'], row['start_bar'] + row['bars']
        if end > len(piece.bars):
            raise ValueError('pll.gate.case_bar_range')
        first, last = piece.bars[start], piece.bars[end-1]
        query = MaskQuery(first.start_qn, last.start_qn + last.duration_qn,
                          tuple(t.track_id for t in piece.tracks))
        if whole_config is None:
            example = split_masked_piece(piece, query, profile)
        else:
            from .whole_contracts import split_whole_piece
            example = split_whole_piece(piece, query, whole_config=whole_config, context_profile=profile)
        yield dict(row, source=manifest.by_id[row['record_id']].source), example


def measure_native(config, directory, manifest, output, *, device='cuda:0', stop_requested=lambda: False,
                   cases=None):
    """Bounded gate; caller must register real-data/hardware execution first.

    Declared cases are checked CPU/device in eval, including gradients.
    Three optimizer applications establish timing and same-device resume with
    sampler/dropout/Adam restored. Gate weights are never promoted to training.
    """
    import torch
    from music_critic.cuda_memory import initialize_cuda_memory_statistics
    from .runtime import PLLTrainer, TrainingConfig
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or config['resume'] is not None:
        raise ValueError('pll.gate.device_or_resume')
    if config['training']['total_updates'] < 2:
        raise ValueError('pll.gate.needs_two_logical_updates')
    lifecycle = None
    logical_device_index = None
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('pll.gate.cuda_unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        lifecycle = initialize_cuda_memory_statistics(device)
        logical_device_index = lifecycle.logical_device_index
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    model = _model(config, directory)
    whole_config = getattr(model, 'whole_config', None)
    sampler_class, extra = MaskSampler, {}
    if whole_config is not None:
        from .whole_data import WholeMaskSampler
        sampler_class, extra = WholeMaskSampler, {'whole_config':whole_config}
    sampler = sampler_class(manifest, seed=config['seed'], context_profile=model.context_profile,
                            **extra, **config['sampling'])
    first = tuple(sampler.draw() for _ in range(config['training']['accumulation']))
    _write_new(output/'first-draws.json', [s.receipt() for s in first])
    if cases is not None:
        validate_cases(cases, manifest)
    examples = _case_examples(cases, manifest, model.context_profile, whole_config=whole_config) if cases is not None else [
        (dict(role='synthetic_or_unstratified', record_id=first[0].record_id), first[0].example)]
    parity_results = []
    # Check the most demanding declared training policy before admitting a
    # schedule which later unfreezes C3. Its CPU reference uses identical state.
    admission_policy = ('upper' if config['freeze_policy']=='frozen' and
        config['training']['unfreeze_update'] is not None else config['freeze_policy'])
    model.encoder.set_freeze_policy(admission_policy)
    for description, example in examples:
        if stop_requested():
            raise KeyboardInterrupt('Gate stopped before case forward')
        model.to('cpu').eval()
        baseline = model(example).nll
        baseline.backward()
        baseline_value = float(baseline.detach())
        baseline_grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(logical_device_index)
        case_started = time.monotonic()
        # cuDNN RNN evaluation omits the reserve buffer needed for backward.
        # Use the differentiable native backend for eval-gradient parity only;
        # restore cuDNN before the actual training/resume steps below.
        with torch.backends.cudnn.flags(enabled=False):
            native = model(example).nll
            native.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize(logical_device_index)
        elapsed = time.monotonic()-case_started
        native_value = float(native.detach())
        torch.testing.assert_close(torch.tensor(native_value), torch.tensor(baseline_value), atol=1e-3, rtol=1e-4)
        grad_difference = _compare_tensors(baseline_grads,
            {k: p.grad for k, p in model.named_parameters() if p.grad is not None},
            atol=5e-3, rtol=1e-2, label='cpu_device_gradients')
        parity_results.append(dict(description, cpu_nll=baseline_value, native_nll=native_value,
            target_events=len(example.targets), max_gradient_difference=grad_difference,
            forward_backward_seconds=elapsed, freeze_policy=admission_policy,
            eval_gradient_backend='aten_no_cudnn'))
        del baseline, native, baseline_grads
        model.zero_grad(set_to_none=True)
        _write_new(output/f'parity-{len(parity_results):02d}.json', parity_results[-1])
    model.encoder.set_freeze_policy(config['freeze_policy'])
    if stop_requested():
        raise KeyboardInterrupt('Gate stopped after parity')
    trainer = PLLTrainer(model, TrainingConfig(**config['training']), data_id=manifest.fingerprint,
                         protocol_id=config['protocol_id'])
    times = []

    def step(examples):
        if stop_requested():
            raise KeyboardInterrupt('Gate stopped at update boundary')
        if device.type == 'cuda':
            torch.cuda.synchronize(logical_device_index)
        started = time.monotonic()
        metrics = trainer.step(tuple(s.example for s in examples))
        if device.type == 'cuda':
            torch.cuda.synchronize(logical_device_index)
        times.append(time.monotonic()-started)
        _write_new(output/f'gate-step-{len(times):02d}.json',
                   dict(physical_update=len(times), metrics=metrics, seconds=times[-1],
                        draws=[s.receipt() for s in examples]))
        return metrics

    step(first)
    checkpoint = output/'gate-update-00000001.pt'
    digest = trainer.save_checkpoint(checkpoint, sampler_state=sampler.state_dict())
    second = tuple(sampler.draw() for _ in range(trainer.config.accumulation))
    expected_metrics = step(second)
    expected = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
    sampler.load_state_dict(trainer.load_checkpoint(checkpoint, expected_sha256=digest))
    replay = tuple(sampler.draw() for _ in range(trainer.config.accumulation))
    if [s.receipt() for s in replay] != [s.receipt() for s in second]:
        raise ValueError('pll.gate.sampler_resume')
    resumed_metrics = step(replay)
    difference = _compare_tensors(expected, model.state_dict(), atol=1e-6, rtol=1e-5,
                                  label='resume_parameters')
    if not math.isclose(expected_metrics['block_nll'], resumed_metrics['block_nll'], rel_tol=1e-5, abs_tol=1e-4):
        raise ValueError('pll.gate.resume_loss')
    total_memory = torch.cuda.get_device_properties(logical_device_index).total_memory if device.type == 'cuda' else None
    peak = torch.cuda.max_memory_reserved(logical_device_index) if device.type == 'cuda' else None
    if peak is not None and peak > .9 * total_memory:
        raise ValueError('pll.gate.memory_headroom_under_10_percent')
    return dict(parity_cases=parity_results, resume_verified=True,
        max_parameter_resume_difference=difference, update_seconds=times,
        peak_cuda_reserved_bytes=peak, cuda_total_bytes=total_memory,
        device_name=torch.cuda.get_device_name(logical_device_index) if device.type=='cuda' else 'cpu',
        cuda_memory_lifecycle=None if lifecycle is None else lifecycle.to_dict(),
        torch_threads=torch.get_num_threads(),
        native_cuda_admitted=device.type=='cuda', dtype='float32', applied_updates=3,
        logical_updates=2, checkpoint_sha256=digest,
        claim='Sampled profile engineering gate only; no corpus-wide fit or musical-quality claim')


def gate(config_path, manifest_path, record_path, output, *, cases_path, device='cuda:0', ledger=None,
         stop_requested=lambda: False):
    """Registered wrapper; even failed native attempts retain terminal evidence."""
    from .runtime import implementation_fingerprint
    config_path, record_path, output = map(Path, (config_path, record_path, output))
    _admit_output(output)
    config = load_run_config(config_path)
    manifest_class = DataManifest
    if config['schema'] == 'PLLWholeRunConfig@1':
        from .whole_data import WholeDataManifest
        manifest_class = WholeDataManifest
    manifest = manifest_class.load(manifest_path)
    if config['schema'] == 'PLLWholeRunConfig@1' and any(
            row['coverage'] != 'full' for row in manifest.coverage.values()):
        raise ValueError('pll.whole_training_requires_full_coverage')
    cases = strict_json(Path(cases_path))
    validate_cases(cases, manifest)
    ledger = Path(ledger) if ledger else (
        record_path.parents[2]/'EXPERIMENT_LEDGER.md' if record_path.parent.name=='records' and record_path.parent.parent.name=='experiments'
        else (record_path.parent.parent if record_path.parent.name=='records' else record_path.parent)/'EXPERIMENT_LEDGER.md')
    if ledger.name != 'EXPERIMENT_LEDGER.md':
        raise ValueError('pll.gate.ledger_filename')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    binding = gate_bindings(config, manifest, device, cases)
    with _record_lock(record_path):
        record = load_record(record_path)
        if record['status'] != 'planned' or record.get('pll_gate') != binding:
            raise ValueError('pll.gate.planned_binding_required')
        if (record['kind'] != 'engineering_gate' or record['budget'].get('planned_updates') != 3 or
                config['seed'] not in record['seeds']):
            raise ValueError('pll.gate.record_budget')
        if any(record['data_access']['declared_policy'][k] for k in
               ('validation_allowed', 'test_inputs_allowed', 'test_targets_allowed',
                'test_metrics_allowed', 'test_selection_allowed')):
            raise ValueError('pll.gate.TRAIN_only')
        output.mkdir(parents=True, exist_ok=False)
        _write_new(output/'planned-record.json', record)
        _write_new(output/'resolved-config.json', config)
        _write_new(output/'cases.json', cases)
        started = time.monotonic()
        record = _transition(record_path, record, dict(record, status='running',
            execution=dict(started_at=_now(), device=device, implementation_sha256=implementation_fingerprint())), ledger)
        status, failure, results = 'failed', None, {}
        try:
            results = measure_native(config, config_path.parent, manifest, output, device=device,
                                     stop_requested=stop_requested, cases=cases)
            status = 'completed'
        except KeyboardInterrupt as exc:
            status, failure = 'aborted', exc
        except Exception as exc:
            failure = exc
        finally:
            receipt = dict(status=status, results=results, elapsed_seconds=time.monotonic()-started,
                error=None if failure is None else dict(type=type(failure).__name__, message=str(failure)),
                binding=binding, verified_train_reads=manifest.verified_train_reads,
                unverified_reads=manifest.unverified_reads)
            _write_new(output/'gate-receipt.json', receipt)
            final = dict(record, status=status, results=results,
                budget={**record['budget'], 'actual_wall_seconds':receipt['elapsed_seconds'],
                        'actual_applied_updates':results.get('applied_updates')},
                execution={**record['execution'], 'ended_at':_now()},
                validity={**record['validity'], 'reason':None if failure is None else str(failure) or type(failure).__name__},
                data_access={**record['data_access'],
                    'access_evidence_status':'pending_primary_evidence' if manifest.unverified_reads else 'verified_primary_evidence',
                    **{k:None if manifest.unverified_reads else False for k in DATA_ACCESS_FIELDS},
                    'evidence_scope':'Native gate loader receipts; source grouping remains caller-declared'},
                artifacts=[*record['artifacts'], *[_artifact(p, record_path, p.name) for p in sorted(output.iterdir()) if p.is_file()]])
            _transition(record_path, record, final, ledger)
        if failure is not None:
            raise failure
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    supervise_parser = sub.add_parser('supervise')
    supervise_parser.add_argument('--config', type=Path, required=True)
    for name in ('gate', 'train'):
        command = sub.add_parser(name)
        for flag in ('config', 'manifest', 'experiment-record', 'output'):
            command.add_argument('--'+flag, type=Path, required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--ledger', type=Path)
        command.add_argument('--torch-threads', type=int, default=4)
        if name == 'gate':
            command.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'supervise':
        result = supervise(load_supervisor_config(args.config))
        print(json.dumps(result, sort_keys=True))
        return 0 if result['returncode']==0 else 1
    import torch
    if not 1 <= args.torch_threads <= 16:
        parser.error('--torch-threads must be between 1 and 16')
    torch.set_num_threads(args.torch_threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with stop_signals() as requested:
        action = gate if args.command=='gate' else train
        options = dict(cases_path=args.cases) if args.command=='gate' else {}
        result = action(args.config, args.manifest, args.experiment_record, args.output,
                        device=args.device, ledger=args.ledger, stop_requested=requested, **options)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status']=='completed' else 130
