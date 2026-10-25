"""Durable campaign orchestration for the separately authorized critic run.

This module performs no work on import. The CLI checks a planned/running record
before data processing. The injectable Python runner is also usable in fixtures.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Sequence

import torch

from music_critic.cuda_memory import initialize_cuda_memory_statistics

from .training import CachedPair, CriticTrainer, _fingerprint, _json_value


CAMPAIGN_SCHEMA = 'CriticCampaign@1.0.0'


class PreparationInterrupted(RuntimeError):
    """Completed immutable cache receipts remain reusable on the next attempt."""


@dataclass(frozen=True)
class CampaignConfig:
    experiment_id: str
    wall_seconds: float
    evaluate_every_updates: int
    checkpoint_every_attempts: int
    require_all_declared_aspects: bool = False
    selection_start_draw: int = 0

    def __post_init__(self):
        if not self.experiment_id or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError('campaign identity and positive wall cap required')
        for value in (self.evaluate_every_updates, self.checkpoint_every_attempts):
            if type(value) is not int or value <= 0:
                raise ValueError('positive campaign update intervals required')
        if type(self.require_all_declared_aspects) is not bool:
            raise ValueError('explicit declared aspect support policy required')
        if type(self.selection_start_draw) is not int or self.selection_start_draw < 0:
            raise ValueError('selection_start_draw must be a nonnegative integer')


def _json_bytes(value):
    return (json.dumps(_json_value(value), sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def _file_digest(path):
    digest, size = sha256(), 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'sha256': digest.hexdigest(), 'size_bytes': size}


def _publish_bytes(path, content, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic immutable publication; refuses an existing file.
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_run(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.campaign.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('campaign already has a writer') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class _Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.events = []
        self.last = '0' * 64
        if self.path.exists():
            with self.path.open('rb') as stream:
                for line in stream:
                    if not line.endswith(b'\n'):
                        raise ValueError('incomplete campaign journal; preserve it for separate recovery')
                    row = json.loads(line)
                    fingerprint = row.pop('fingerprint', None)
                    if (row.get('sequence') != len(self.events) + 1 or row.get('previous') != self.last
                            or fingerprint != _fingerprint(row)):
                        raise ValueError('campaign journal integrity mismatch')
                    row['fingerprint'] = fingerprint
                    self.events.append(row)
                    self.last = fingerprint

    def append(self, kind, payload, elapsed):
        row = {'schema': CAMPAIGN_SCHEMA, 'sequence': len(self.events) + 1, 'previous': self.last,
               'kind': kind, 'elapsed_seconds': elapsed,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'payload': _json_value(payload)}
        row['fingerprint'] = _fingerprint(row)
        with self.path.open('ab') as stream:
            stream.write(_json_bytes(row))
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(row)
        self.last = row['fingerprint']


def _save_checkpoint(root, trainer, journal, run_fingerprint, elapsed, *, best=False):
    relative = f'checkpoints/update-{trainer.applied_updates:08d}-attempt-{trainer.attempted_updates:08d}-event-{len(journal.events):08d}.pt'
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema': CAMPAIGN_SCHEMA, 'run_fingerprint': run_fingerprint,
               'trainer': trainer.state_dict(), 'committed_event': len(journal.events),
               'committed_fingerprint': journal.last, 'elapsed_seconds': elapsed}
    descriptor, name = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    pointer = {'schema': CAMPAIGN_SCHEMA, 'path': relative, **_file_digest(path),
               'update': trainer.applied_updates, 'attempt': trainer.attempted_updates,
               'committed_event': len(journal.events), 'committed_fingerprint': journal.last}
    _publish_bytes(root / 'latest.json', _json_bytes(pointer), replace=True)
    if best:
        _publish_bytes(root / 'best.json', _json_bytes(pointer), replace=True)
    journal.append('checkpoint', {**pointer, 'selected_best': best}, elapsed)
    return pointer


def _restore_checkpoint(root, trainer, journal, run_fingerprint):
    pointer = json.loads((root / 'latest.json').read_text())
    candidate = root / pointer['path']
    path = candidate.resolve()
    if (not path.is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (candidate, *candidate.parents))):
        raise ValueError('checkpoint pointer escapes campaign root')
    if _file_digest(path) != {key: pointer[key] for key in ('sha256', 'size_bytes')}:
        raise ValueError('campaign checkpoint integrity mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != CAMPAIGN_SCHEMA or payload.get('run_fingerprint') != run_fingerprint:
        raise ValueError('campaign checkpoint binding mismatch')
    committed = payload['committed_event']
    if (type(committed) is not int or not 1 <= committed <= len(journal.events)
            or journal.events[committed - 1]['fingerprint'] != payload['committed_fingerprint']
            or pointer['committed_event'] != committed
            or pointer['committed_fingerprint'] != payload['committed_fingerprint']):
        raise ValueError('campaign checkpoint journal binding mismatch')
    trainer.load_state_dict(payload['trainer'])
    return pointer, committed


def validate_terminal_continuation_config(options):
    """Only a hash-bound terminal campaign summary can authorize a transfer."""
    import re
    if (not isinstance(options, dict)
            or set(options) != {'summary_path', 'sha256', 'size_bytes', 'experiment_id'}
            or not isinstance(options['summary_path'], str)
            or not isinstance(options['experiment_id'], str) or not options['experiment_id']
            or not isinstance(options['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', options['sha256'])
            or type(options['size_bytes']) is not int or options['size_bytes'] <= 0):
        raise ValueError('invalid terminal continuation binding')
    path = Path(options['summary_path'])
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('absolute nonsymlink continuation summary required')
    return path


def load_terminal_continuation(trainer, dev_bank, config, options):
    """Transfer a clean pre-selection stop to a new run, preserving all state.

    Parent files remain immutable. Selected-best artifact migration is not
    implemented; transfers after selection begins are rejected explicitly.
    """
    path = validate_terminal_continuation_config(options)
    if _file_digest(path) != {k: options[k] for k in ('sha256', 'size_bytes')}:
        raise ValueError('continuation summary integrity mismatch')
    summary = json.loads(path.read_text())
    root = path.parent
    metadata = json.loads((root / 'run.json').read_text())
    journal = _Journal(root / 'events.jsonl')
    if (summary.get('status') != 'aborted' or summary.get('reason') != 'stop_requested'
            or summary.get('final_checkpoint_error') is not None
            or summary.get('experiment_id') != options['experiment_id']
            or config.experiment_id == options['experiment_id']
            or not journal.events or journal.events[-1]['kind'] != 'terminal'
            or journal.last != summary.get('journal_fingerprint')
            or journal.events[-1]['payload'] != {k: v for k, v in summary.items() if k != 'journal_fingerprint'}):
        raise ValueError('continuation requires a bound clean terminal stop')
    prior_config = metadata['campaign']
    if (metadata['trainer'] != _json_value(trainer.metadata)
            or metadata['dev_bank'] != _fingerprint(tuple(dev_bank))
            or metadata['campaign']['experiment_id'] != options['experiment_id']
            or _fingerprint(metadata) != summary['run_fingerprint']
            or any(asdict(config)[k] != v for k, v in prior_config.items()
                   if k not in {'experiment_id', 'wall_seconds'})
            or config.wall_seconds > prior_config['wall_seconds'] - summary['wall_seconds']):
        raise ValueError('continuation protocol or remaining budget mismatch')
    pointer = json.loads((root / 'latest.json').read_text())
    if (pointer != summary['latest'] or summary.get('best') is not None
            or summary['draws'] >= config.selection_start_draw
            or pointer['update'] != summary['applied_updates']
            or pointer['attempt'] != summary['attempted_updates']):
        raise ValueError('continuation requires final pre-selection checkpoint')
    pointer, committed = _restore_checkpoint(root, trainer, journal, summary['run_fingerprint'])
    if (trainer.draws != summary['draws'] or trainer.selection.best_update is not None
            or any(e['kind'] == 'step' for e in journal.events[committed:])):
        raise ValueError('continuation checkpoint progress mismatch')
    evaluations = [e['payload'] for e in journal.events[:committed] if e['kind'] == 'evaluation']
    return {'parent_experiment_id': options['experiment_id'],
            'parent_summary_sha256': options['sha256'], 'parent_checkpoint': pointer,
            'loaded_state_fingerprint': trainer.state_dict()['fingerprint'],
            'initial_update': trainer.applied_updates, 'initial_draw': trainer.draws,
            'last_evaluation_update': max([summary['last_evaluation_update'],
                                            *(e['update'] for e in evaluations)]),
            'last_evaluation_draw': max([summary['last_evaluation_draw'],
                                          *(e['draw'] for e in evaluations)])}


def prepare_feature_bank(dataset, adapter, output_root, *, wall_seconds: float,
                         clock: Callable[[], float] = time.monotonic,
                         stop_requested: Callable[[], bool] = lambda: False):
    """Cache each unique raw view once, authenticating every reused entry.

    No target sidecar is passed to the adapter. A feature receipt is published
    only after its cache entry exists and passes the ordinary reader checks.
    Interrupted preparation retains receipts; a complete manifest is immutable.
    """
    from .cache import FrozenFeatureCache
    from .contracts import CriticFeaturePyramid
    from .features import raw_piece_hash
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('positive feature preparation wall cap required')
    root = Path(output_root)
    with _exclusive_run(root):
        cache = FrozenFeatureCache(root / 'features')
        manifest_path = root / 'feature-manifest.json'
        identity = {'schema': 'CriticCampaignFeatureManifest@1.0.0',
                    'dataset_fingerprint': dataset.fingerprint, 'binding': asdict(adapter.binding)}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if any(previous.get(k) != _json_value(v) for k, v in identity.items()):
                raise ValueError('feature manifest binding mismatch')
        started = clock()
        samples, by_id, entries = {}, {}, {}
        extracted = reused = 0
        for pair in dataset.pairs:
            for piece in (pair.original, pair.variant):
                if stop_requested() or clock() - started >= wall_seconds:
                    raise PreparationInterrupted('feature preparation stopped; completed receipts retained')
                if piece.piece_id in by_id:
                    previous_piece, raw_hash = by_id[piece.piece_id]
                    if piece != previous_piece:
                        raise ValueError('one raw sample identity has conflicting contents')
                    continue
                raw_hash = raw_piece_hash(piece)
                by_id[piece.piece_id] = (piece, raw_hash)
                receipt_id = _fingerprint({'binding': asdict(adapter.binding), 'raw_hash': raw_hash})
                receipt_path = root / 'feature-receipts' / (receipt_id + '.json')
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if (receipt.get('raw_hash') != raw_hash or receipt.get('binding') != asdict(adapter.binding)
                            or receipt.get('sample_id') != piece.piece_id):
                        raise ValueError('feature receipt binding mismatch')
                    sample = cache.get(receipt['cache_key'], adapter.binding, raw_hash)
                    reused += 1
                else:
                    pyramid = adapter.extract((piece,))
                    if (pyramid.binding != adapter.binding or len(pyramid.samples) != 1
                            or pyramid.samples[0].raw_hash != raw_hash
                            or pyramid.samples[0].sample_id != piece.piece_id):
                        raise ValueError('feature extraction binding mismatch')
                    key, = cache.put(pyramid)
                    sample = cache.get(key, adapter.binding, raw_hash)
                    receipt = {'sample_id': piece.piece_id, 'raw_hash': raw_hash, 'cache_key': key,
                               'binding': asdict(adapter.binding),
                               'tensor_bytes': sum(t.numel() * t.element_size() for t in sample.embeddings.values())}
                    _publish_bytes(receipt_path, _json_bytes(receipt))
                    extracted += 1
                samples[piece.piece_id] = sample
                entries[piece.piece_id] = receipt
        pairs = tuple(CachedPair(
            p.pair_id, p.work_id, CriticFeaturePyramid((samples[p.original.piece_id],), adapter.binding),
            CriticFeaturePyramid((samples[p.variant.piece_id],), adapter.binding), p.targets,
            split=p.split, component_id=p.component_id,
            edit_count=getattr(p, 'edit_count', None), genre=getattr(p, 'genre', None),
        ) for p in dataset.pairs)
        manifest = {**identity, 'entries': entries, 'pair_count': len(pairs),
                    'unique_views': len(entries), 'tensor_bytes': sum(r['tensor_bytes'] for r in entries.values())}
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != _json_value(manifest):
                raise ValueError('complete feature manifest changed')
        else:
            _publish_bytes(manifest_path, _json_bytes(manifest))
        report = {'schema': 'CriticFeaturePreparationReceipt@1.0.0', 'dataset_fingerprint': dataset.fingerprint,
                  'extracted_views': extracted, 'reused_views': reused, 'unique_views': len(entries),
                  'pair_count': len(pairs), 'tensor_bytes': manifest['tensor_bytes'],
                  'wall_seconds': clock() - started, 'manifest': _file_digest(manifest_path)}
        return pairs, report


def _validate_encoder_batch_tolerance(value):
    if value is None:
        return {'rtol': 1e-5, 'atol': 1e-5}
    if (not isinstance(value, dict) or set(value) != {'rtol', 'atol'}
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1e-3
                   for v in value.values())):
        raise ValueError('encoder_batch_tolerance requires finite positive rtol/atol <=1e-3')
    return dict(value)


def _emit_probe_progress(callback, stage, state, trainer, *, started, clock,
                         stage_started=None, duration_seconds=None):
    """Report counters only; callback failures propagate to the invocation owner."""
    if callback is None:
        return None
    now = clock()
    if state == 'completed' and duration_seconds is None:
        duration_seconds = max(0., now - stage_started)
    callback({'schema': 'CriticNativeRuntimeProgress@1.0.0', 'stage': stage, 'state': state,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': max(0., now - started), 'duration_seconds': duration_seconds,
              'attempted_updates': getattr(trainer, 'attempted_updates', 0),
              'applied_updates': getattr(trainer, 'applied_updates', 0),
              'skipped_updates': getattr(trainer, 'skipped_updates', 0),
              'draws': getattr(trainer, 'draws', 0)})
    return now


def probe_runtime(trainer, trainer_factory, dev_bank, *, updates: int,
                  adapter=None, raw_batch=None, wall_seconds=3600, encoder_batch_tolerance=None,
                  clock=time.monotonic, stop_requested=lambda: False, progress_callback=None,
                  reference_model=None):
    """Native mechanics/throughput gate on the complete configured banks.

    One continuation step is deliberately replayed in an independently restored
    trainer. That extra optimizer application is included in compute accounting.
    No checkpoint selection or musical-quality conclusion is made by this gate.
    """
    from .contracts import CriticFeaturePyramid
    if type(updates) is not int or not 2 <= updates <= trainer.config.max_updates:
        raise ValueError('probe requires at least two updates within trainer budget')
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('probe progress callback must be callable')
    encoder_tolerance = _validate_encoder_batch_tolerance(encoder_batch_tolerance)
    if reference_model is not None:
        from .runtime_equivalence import _read_reference_source
        _read_reference_source(reference_model)
    started = clock()
    def progress(stage, state, *, subject=trainer, stage_started=None, duration_seconds=None):
        return _emit_probe_progress(progress_callback, stage, state, subject, started=started,
                                    clock=clock, stage_started=stage_started,
                                    duration_seconds=duration_seconds)
    def check_budget():
        if stop_requested() or clock() - started >= wall_seconds:
            raise PreparationInterrupted('native probe wall cap or stop requested')
    def synchronize():
        if trainer.device.type == 'cuda':
            torch.cuda.synchronize(trainer.device)
    if trainer.device.type == 'cuda':
        initialize_cuda_memory_statistics(trainer.device)
    check_budget()
    # FP32 batched GRU/graph reductions may choose a different reduction order.
    # Encoder batching has its own predeclared bound; changing it cannot relax
    # critic batching or same-batch checkpoint continuation.
    batch_tolerance = {'rtol': 1e-5, 'atol': 1e-5}
    tolerance = {'rtol': 1e-5, 'atol': 1e-6}
    encoder_gate = {'performed': adapter is not None, 'extra_forward_views': 0}
    if adapter is not None:
        encoder_started = progress('encoder_parity', 'started')
        if not isinstance(raw_batch, tuple) or len(raw_batch) != 2:
            raise ValueError('encoder native probe requires exactly two raw views')
        before = _fingerprint(adapter.model.state_dict())
        singles = [adapter.extract((piece,)).samples[0] for piece in raw_batch]
        batched = adapter.extract(raw_batch)
        max_error = 0.0
        for expected, actual in zip(singles, batched.samples):
            if (expected.geometry != actual.geometry or expected.raw_hash != actual.raw_hash
                    or expected.stages != actual.stages):
                raise ValueError('encoder batch identity mismatch')
            for level, tensor in expected.embeddings.items():
                torch.testing.assert_close(tensor, actual.embeddings[level], **encoder_tolerance)
                if tensor.numel():
                    max_error = max(max_error, float((tensor - actual.embeddings[level]).abs().max()))
        if before != _fingerprint(adapter.model.state_dict()):
            raise ValueError('encoder state changed in native probe')
        encoder_gate.update(passed=True, max_absolute_error=max_error, extra_forward_views=4,
                            state_unchanged=True, **encoder_tolerance)
        progress('encoder_parity', 'completed', stage_started=encoder_started)
    check_budget()
    critic_started = progress('critic_parity', 'started')
    trainer.model.eval()
    source = trainer.bank[0]
    pyramid = CriticFeaturePyramid((source.left.samples[0], source.right.samples[0]), source.left.binding)
    with torch.no_grad():
        singles = [trainer.model(view.to(trainer.device)).samples[0] for view in (source.left, source.right)]
        batched = trainer.model(pyramid.to(trainer.device)).samples
    maximum = 0.0
    for expected, actual in zip(singles, batched):
        if (expected.sample_id != actual.sample_id or expected.horizon != actual.horizon
                or expected.support_policy != actual.support_policy
                or not torch.equal(expected.availability, actual.availability)):
            raise ValueError('critic batch identity mismatch')
        torch.testing.assert_close(expected.raw_scores, actual.raw_scores, **batch_tolerance)
        maximum = max(maximum, float((expected.raw_scores - actual.raw_scores).abs().max()))
        if expected.local.keys() != actual.local.keys():
            raise ValueError('critic batch localization namespace mismatch')
        for namespace, local in expected.local.items():
            other = actual.local[namespace]
            if local.candidates != other.candidates or not torch.equal(local.availability, other.availability):
                raise ValueError('critic batch localization identity mismatch')
            torch.testing.assert_close(local.logits, other.logits, **batch_tolerance)
    progress('critic_parity', 'completed', stage_started=critic_started)
    synchronize()
    guard_started = clock()
    trainer._check_bank()
    guard_seconds = clock() - guard_started
    equivalence = None
    if reference_model is not None:
        from .runtime_equivalence import compare_reference_runtime
        equivalence = compare_reference_runtime(trainer, reference_model, clock=clock,
            check_budget=check_budget, progress_callback=progress_callback)
    step_seconds, steps = [], []
    initial = trainer.applied_updates
    while trainer.applied_updates - initial < updates - 1:
        check_budget()
        progress('train_step', 'started')
        synchronize()
        tick = clock()
        step = trainer.step()
        synchronize()
        step_seconds.append(clock() - tick)
        steps.append(asdict(step))
        progress('train_step', 'completed', duration_seconds=step_seconds[-1])
    restore_started = progress('resume_restore', 'started')
    checkpoint = trainer.state_dict()
    restored = trainer_factory()
    restored.load_state_dict(checkpoint)
    progress('resume_restore', 'completed', subject=restored, stage_started=restore_started)
    check_budget()
    progress('train_step', 'started')
    synchronize()
    tick = clock()
    expected_step = trainer.step()
    synchronize()
    step_seconds.append(clock() - tick)
    steps.append(asdict(expected_step))
    progress('train_step', 'completed', duration_seconds=step_seconds[-1])
    replay_started = progress('resume_replay', 'started', subject=restored)
    actual_step = restored.step()
    synchronize()
    progress('resume_replay', 'completed', subject=restored, stage_started=replay_started)
    verification_started = progress('resume_verification', 'started')
    if not expected_step.applied:
        raise ValueError('native continuation gate requires a supported optimizer application')
    if (expected_step.pair_ids != actual_step.pair_ids or expected_step.swapped != actual_step.swapped
            or expected_step.applied != actual_step.applied):
        raise ValueError('native resume sampled draw mismatch')
    for field in ('loss', 'pair_loss', 'local_loss', 'grad_norm'):
        expected, actual = getattr(expected_step, field), getattr(actual_step, field)
        if (expected is None) != (actual is None) or (expected is not None
                and not math.isclose(expected, actual, rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError('native resume scalar mismatch: ' + field)
    reference, candidate = trainer.state_dict(), restored.state_dict()
    exact = reference['fingerprint'] == candidate['fingerprint']
    def compare_trees(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, **tolerance)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError('native resume state fields mismatch')
            for key in left:
                compare_trees(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            if type(left) != type(right) or len(left) != len(right):
                raise ValueError('native resume state sequence mismatch')
            for a, b in zip(left, right):
                compare_trees(a, b)
        elif left != right:
            raise ValueError('native resume state value mismatch')
    compare_trees({k: v for k, v in reference.items() if k != 'fingerprint'},
                  {k: v for k, v in candidate.items() if k != 'fingerprint'})
    progress('resume_verification', 'completed', stage_started=verification_started)
    check_budget()
    progress('dev', 'started')
    synchronize()
    tick = clock()
    evaluation = trainer.evaluate(dev_bank)
    synchronize()
    evaluation_seconds = clock() - tick
    progress('dev', 'completed', duration_seconds=evaluation_seconds)
    if set(evaluation.supported_aspects) != set(trainer.config.selection_aspects):
        raise ValueError('native probe head-dev lacks declared aspect support')
    backbone_no_grad = all(not p.requires_grad and p.grad is None
                          for backbone in trainer.frozen_backbones for p in backbone.parameters())
    if adapter is not None:
        adapter._check_frozen()
        backbone_no_grad = backbone_no_grad and all(not p.requires_grad and p.grad is None
                                                    for p in adapter.model.parameters())
    if not backbone_no_grad:
        raise ValueError('native probe encoder gradient detected')
    report = {'schema': 'CriticNativeRuntimeProbe@1.0.0', 'status': 'completed',
            'device': str(trainer.device), 'train_bank_pairs': len(trainer.bank),
            'head_dev_pairs': len(dev_bank), 'full_bank_guard_seconds': guard_seconds,
            'step_seconds': step_seconds, 'mean_step_seconds': sum(step_seconds) / len(step_seconds),
            'head_dev_seconds': evaluation_seconds, 'head_dev_metrics': asdict(evaluation),
            'applied_updates': trainer.applied_updates - initial,
            'applied_updates_including_resume_replay': trainer.applied_updates - initial + int(actual_step.applied),
            'steps': steps, 'resume_replay': asdict(actual_step),
            'resume': {'passed': True, 'same_sample_draw': True, 'exact_state': exact, **tolerance},
            'encoder_single_vs_batch': encoder_gate,
            'critic_single_vs_batch': {'passed': True, 'max_absolute_error': maximum, **batch_tolerance},
            'encoder_no_grad': backbone_no_grad,
            'cuda_peak_allocated_bytes': (torch.cuda.max_memory_allocated(trainer.device)
                                           if trainer.device.type == 'cuda' else None),
            'wall_seconds': clock() - started, 'test_accessed': False,
            'claim': 'native runtime mechanics and throughput only; no quality comparison'}
    if equivalence is not None:
        report['runtime_equivalence'] = equivalence
        report['total_optimizer_applications'] = (
            report['applied_updates_including_resume_replay'] + equivalence['optimizer_applications'])
        report['total_optimizer_pair_draws'] = (
            sum(len(step['pair_ids']) for step in steps) + len(actual_step.pair_ids)
            + equivalence['paired_draws'])
    return report


def run_campaign(
    trainer: CriticTrainer, dev_bank: Sequence[CachedPair], config: CampaignConfig, output_root,
    *, resume=False, continuation=None, clock: Callable[[], float] = time.monotonic,
    stop_requested: Callable[[], bool] = lambda: False,
) -> dict:
    """Run under a cooperative wall cap; an external supervisor owns hard limits.

    A clean terminal summary is immutable. Resume repairs only an interrupted
    nonterminal execution. Uncheckpointed journal events remain as historical
    work, and the new segment explicitly records their replay boundary.
    """
    fixed_exposure = bool(trainer.config.curriculum_stages)
    draw_budget = (len(trainer.exposure_plan) if fixed_exposure
                   else trainer.config.max_updates * trainer.config.batch_size)
    if config.selection_start_draw > draw_budget:
        raise ValueError('selection_start_draw exceeds guaranteed draw budget')
    root = Path(output_root)
    with _exclusive_run(root):
        if (root / 'summary.json').exists():
            raise ValueError('terminal campaign cannot be resumed or overwritten')
        metadata = _json_value({'schema': CAMPAIGN_SCHEMA, 'campaign': asdict(config),
                                'trainer': trainer.metadata, 'dev_bank': _fingerprint(tuple(dev_bank))})
        if continuation is not None:
            if trainer.state_dict()['fingerprint'] != continuation['loaded_state_fingerprint']:
                raise ValueError('continuation state changed before campaign')
            metadata['continuation'] = continuation
        run_fingerprint = _fingerprint(metadata)
        manifest = root / 'run.json'
        if resume:
            if not manifest.exists() or json.loads(manifest.read_text()) != metadata:
                raise ValueError('campaign resume binding mismatch')
        else:
            if manifest.exists() or (root / 'events.jsonl').exists():
                raise ValueError('campaign already exists; explicit resume required')
            _publish_bytes(manifest, _json_bytes(metadata))
        journal = _Journal(root / 'events.jsonl')
        previous_wall = max((e['elapsed_seconds'] for e in journal.events), default=0.0)
        started = clock()
        elapsed = lambda: previous_wall + max(0.0, clock() - started)
        if resume:
            pointer, committed = _restore_checkpoint(root, trainer, journal, run_fingerprint)
            tail = journal.events[committed:]
            journal.append('resumed', {'parent_checkpoint': pointer,
                                      'retained_tail_events': len(tail),
                                      'replayed_logged_applied_updates': sum(
                                          e['kind'] == 'step' and e['payload']['applied'] for e in tail),
                                      'unlogged_work_possible': True}, elapsed())
            evaluation_updates = [e['payload']['update'] for e in journal.events[:committed]
                                  if e['kind'] == 'evaluation']
            last_evaluation = max(evaluation_updates, default=-1)
            last_evaluation_draw = max((e['payload'].get('draw', -1)
                                       for e in journal.events[:committed]
                                       if e['kind'] == 'evaluation'), default=-1)
            if continuation:
                last_evaluation = max(last_evaluation, continuation['last_evaluation_update'])
                last_evaluation_draw = max(last_evaluation_draw, continuation['last_evaluation_draw'])
        else:
            journal.append('started', {'experiment_id': config.experiment_id,
                                      'initial_applied_updates': trainer.applied_updates}, elapsed())
            _save_checkpoint(root, trainer, journal, run_fingerprint, elapsed())
            last_evaluation = continuation['last_evaluation_update'] if continuation else -1
            last_evaluation_draw = continuation['last_evaluation_draw'] if continuation else -1

        def evaluate():
            nonlocal last_evaluation, last_evaluation_draw
            result = trainer.evaluate(dev_bank)
            if (config.require_all_declared_aspects
                    and set(result.supported_aspects) != set(trainer.config.selection_aspects)):
                raise ValueError('campaign head-dev lacks declared aspect support')
            eligible = trainer.cursor >= config.selection_start_draw
            selected = (trainer.selection.consider(result, update=trainer.applied_updates)
                        if eligible else False)
            last_evaluation = trainer.applied_updates
            last_evaluation_draw = trainer.cursor
            journal.append('evaluation', {'update': trainer.applied_updates,
                                           'draw': trainer.cursor, 'selection_eligible': eligible,
                                           'selected_best': selected, 'metrics': asdict(result)}, elapsed())
            _save_checkpoint(root, trainer, journal, run_fingerprint, elapsed(), best=selected)

        status, reason = 'completed', 'exposure_cap' if fixed_exposure else 'update_cap'
        error = None
        try:
            pending_evaluation = (continuation is not None and trainer.applied_updates > 0
                                  and trainer.applied_updates % config.evaluate_every_updates == 0
                                  and last_evaluation != trainer.applied_updates)
            if (last_evaluation < 0 or pending_evaluation) and elapsed() < config.wall_seconds and not stop_requested():
                evaluate()
            while (trainer.cursor < draw_budget if fixed_exposure
                   else trainer.applied_updates < trainer.config.max_updates):
                if stop_requested():
                    status, reason = 'aborted', 'stop_requested'
                    break
                if elapsed() >= config.wall_seconds:
                    status, reason = 'aborted', 'wall_cap'
                    break
                if trainer.attempted_updates >= trainer.config.max_attempts:
                    status, reason = 'aborted', 'attempt_cap'
                    break
                result = trainer.step()
                journal.append('step', asdict(result), elapsed())
                if (result.applied and trainer.applied_updates % config.evaluate_every_updates == 0
                        and elapsed() < config.wall_seconds and not stop_requested()):
                    evaluate()
                elif trainer.attempted_updates % config.checkpoint_every_attempts == 0:
                    _save_checkpoint(root, trainer, journal, run_fingerprint, elapsed())
            if (status == 'completed' and last_evaluation_draw != trainer.cursor
                    and elapsed() < config.wall_seconds and not stop_requested()):
                evaluate()
            if status == 'completed' and last_evaluation_draw != trainer.cursor:
                status, reason = 'aborted', 'final_evaluation_not_completed'
        except Exception as exc:
            status, reason, error = 'failed', type(exc).__name__, str(exc)
            journal.append('failure', {'type': reason, 'message': error,
                                       'unlogged_work_possible': True}, elapsed())
        final_checkpoint_error = None
        try:
            latest = _save_checkpoint(root, trainer, journal, run_fingerprint, elapsed())
        except Exception as exc:
            final_checkpoint_error = type(exc).__name__ + ': ' + str(exc)
            status = 'failed'
            if error is None:
                reason, error = 'final_checkpoint_failed', final_checkpoint_error
            journal.append('checkpoint_failure', {'error': final_checkpoint_error}, elapsed())
            latest = json.loads((root / 'latest.json').read_text()) if (root / 'latest.json').exists() else None
        first_update = next(e['payload']['initial_applied_updates'] for e in journal.events if e['kind'] == 'started')
        logged = sum(e['kind'] == 'step' and e['payload']['applied'] for e in journal.events)
        report = {'schema': CAMPAIGN_SCHEMA, 'experiment_id': config.experiment_id,
                  'status': status, 'reason': reason, 'error': error,
                  'final_checkpoint_error': final_checkpoint_error,
                  'applied_updates': trainer.applied_updates, 'attempted_updates': trainer.attempted_updates,
                  'skipped_updates': trainer.skipped_updates, 'draws': trainer.draws,
                  'best_update': trainer.selection.best_update, 'selection': asdict(trainer.selection),
                  'last_evaluation_update': last_evaluation,
                  'last_evaluation_draw': last_evaluation_draw,
                  'logged_applied_updates_including_replay': logged,
                  'replayed_logged_updates': max(0, logged - (trainer.applied_updates - first_update)),
                  'unlogged_work_possible': any(e['kind'] in {'resumed', 'failure'} for e in journal.events),
                  'wall_seconds': elapsed(), 'wall_cap_seconds': config.wall_seconds,
                  'wall_cap_kind': 'cooperative_between_steps_and_evaluations',
                  'run_fingerprint': run_fingerprint, 'latest': latest,
                  'best': json.loads((root / 'best.json').read_text()) if (root / 'best.json').exists() else None,
                  'test_accessed': False,
                  'claim': 'weak intervention ranking and localization; human musical quality unvalidated'}
        journal.append('terminal', report, elapsed())
        report['journal_fingerprint'] = journal.last
        _publish_bytes(root / 'summary.json', _json_bytes(report))
        return report
