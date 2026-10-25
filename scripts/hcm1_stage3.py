"""Explicit full-state third training segment chained from a completed segment.

VERIFIED against real torch; not yet used for a run.

Test counts, timings and the verified digest of THIS file are deliberately not
recorded here: editing this docstring to state them would change the digest they
describe. The current verified digest, the exact pytest results, the environment
and the not-covered list live in
`outputs/hcm_stage3_verify_20260929/verification-manifest.json`, which is outside
the fingerprinted package and outside this file.

What is stable and belongs here: verification runs under python 3.12.11,
torch 2.8.0+cu126 and pytest from the RTX 3090 venv
`hcm1_handoff_20260928/cloud-runtime001/venv`, executed from a scratch copy of
the `hcm_whole_20260928_001/source002` tree whose `src/music_critic` fingerprint
is `049e6098bd78b5804e4b073edf55d187b6fc374de5c3383bfb4be48b1b062aea` - the exact
`source_id` recorded in `EXP-HCM2-3090-CONTINUE-001`, i.e. the identity a real
gen-3 resume must match. `source002` itself is never written to. The regression
set is `test_stage2.py`, `test_completed_segment.py`, `test_whole_runtime.py` and
`test_runtime.py`; it must stay green so that `hcm1_stage2` and its strict-resume
invariant remain unweakened by this additive script.

Established by those tests: `_validate_optimizer` accepts the chained segment, so
the ancestral LR reproduction is bit-exact against a real trainer holding real
AdamW state; bitwise replay of the next step survives checkpoint save/load;
model, optimizer and Python/CPU RNG are preserved and the parent is not mutated;
`schedule_ancestry` derives the chain from the binding for horizons other than
the production one, so no number is hardcoded; and a plain `PLLTrainer` still
refuses a gen-3 checkpoint.

Established independently, without torch, by extracting this file's own pure
functions through `ast` and testing them exhaustively: bit-exact against
`PLLTrainer._lr_multiplier` over all 1000 updates of the gen-1 domain and against
`SegmentTrainer._lr_multiplier` over all 9000 updates of the gen-2 domain; at the
boundary `u=9999` this returns exactly the stored 3.08030470574927051075e-08
where stage2's naive plain-cosine parent branch returns
2.49226138615554759781e-08, a 19.0904% relative error that would abort resume
with `pll.checkpoint_optimizer_invalid`.

NOT COVERED. `load_segment_parent` (the `load_parent` + `restore_segment` chain
over authenticated records) and the whole `run()` loop - record lifecycle,
supervisor stop handling, per-line fsync of metrics/draws, checkpointing every N
updates, receipt and terminal transition - have never been executed. Neither has
any run against the real 208 MB checkpoints or on CUDA; the tests are CPU-only
and synthetic. A first production launch therefore still needs the usual staged
observation of its opening updates rather than being treated as proven.

`hcm1_stage2.SegmentTrainer` reconstructs the learning-rate multiplier of the
update preceding its own segment start as a plain cosine over
``[0, parent_config.total_updates)``. That reconstruction is exact for a
first-generation parent, whose schedule really was a plain cosine from update
zero. It is wrong for a parent that is itself a segment, because a segment
schedules warmup-then-cosine over segment-local steps of length
``total_updates - segment_start``.

`music_critic.pll.runtime.PLLTrainer._validate_optimizer` compares the stored
AdamW group learning rate with exact float equality, so the reconstruction has
to be bit-identical rather than close. On the real chain the plain-cosine
reconstruction yields 2.49226138615554759781e-08 where the completed segment
stored 3.08030470574927051075e-08: a 19% relative error that aborts resume with
``pll.checkpoint_optimizer_invalid``.

This script is additive. It generalises the ancestral reconstruction to an
explicit segment ancestry derived from the parent's own recorded binding, so it
holds for any generation, and it reproduces each generation with that
generation's own arithmetic. `PLLTrainer` evaluates ``math.pi * progress`` while
`SegmentTrainer` evaluates ``math.pi * (local - warmup) / (length - warmup)``;
the two orders can differ in the last ulp, so they are not interchangeable under
an exact comparison.

`hcm1_stage2` and its tests are unchanged, and the strict rule that a plain
`PLLTrainer` refuses a segment checkpoint is preserved. This script's digest is
part of the child's own binding.
"""
from __future__ import annotations

import argparse
import copy
import math
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from music_critic.pll.runtime import PLLTrainer, _cpu_tree
from scripts.hcm1_stage2 import SegmentTrainer, continue_full_state, load_parent

ANCESTRY_SCHEMA = 'HCMScheduleAncestry@1'


def plain_multiplier(total_updates: int, warmup_updates: int, updates: int) -> float:
    """Reproduce `PLLTrainer._lr_multiplier` bit-for-bit (generation zero)."""
    if updates < warmup_updates:
        return (updates + 1) / warmup_updates
    progress = (updates - warmup_updates) / max(1, total_updates - warmup_updates)
    return .5 * (1. + math.cos(math.pi * progress))


def segment_multiplier(segment_start: int, total_updates: int, warmup_updates: int,
                       updates: int) -> float:
    """Reproduce `SegmentTrainer._lr_multiplier` bit-for-bit (generation >= 1)."""
    local = updates - segment_start
    if local < warmup_updates:
        return (local + 1) / warmup_updates
    length = total_updates - segment_start
    return .5 * (1 + math.cos(math.pi * (local - warmup_updates) / (length - warmup_updates)))


def schedule_ancestry(binding) -> list[dict]:
    """Derive ordered ancestral schedule segments from a recorded binding.

    Generation zero first. Each entry carries exactly what its own trainer used
    to compute a multiplier, so no horizon or warmup is hardcoded here.
    """
    if not isinstance(binding, dict) or 'training' not in binding:
        raise ValueError('stage3 ancestry requires a runtime binding')
    entries = []
    current = binding
    generation = 0
    while True:
        if not isinstance(current, dict) or not isinstance(current.get('training'), dict):
            raise ValueError('stage3 ancestry binding is malformed')
        training = current['training']
        for field in ('total_updates', 'warmup_updates'):
            if type(training.get(field)) is not int or training[field] <= 0:
                raise ValueError('stage3 ancestry schedule field invalid')
        continuation = current.get('continuation')
        if continuation is None:
            start = 0
        else:
            start = continuation.get('start_update')
            if type(start) is not int or start < 0:
                raise ValueError('stage3 ancestry start_update invalid')
            current = continuation.get('parent_binding')
        entries.append(dict(generation=generation, segment_start=start,
                            total_updates=training['total_updates'],
                            warmup_updates=training['warmup_updates']))
        if continuation is None:
            break
        generation += 1
    entries.reverse()
    for index, entry in enumerate(entries):
        entry['generation'] = index
    for earlier, later in zip(entries, entries[1:]):
        if later['segment_start'] != earlier['total_updates']:
            raise ValueError('stage3 ancestry segments are not contiguous')
    return entries


def ancestry_multiplier(ancestry: list[dict], updates: int) -> float:
    """Evaluate the ancestral schedule that contains `updates`."""
    for entry in reversed(ancestry):
        if updates >= entry['segment_start']:
            if entry['generation'] == 0:
                return plain_multiplier(entry['total_updates'], entry['warmup_updates'], updates)
            return segment_multiplier(entry['segment_start'], entry['total_updates'],
                                      entry['warmup_updates'], updates)
    raise ValueError('stage3 update precedes recorded ancestry')


class Segment3Trainer(SegmentTrainer):
    """A new warmup+cosine segment whose ancestors may themselves be segments."""

    def __init__(self, model, parent_config, *, start_update, additional_updates,
                 data_id, protocol_id, parent_binding, parent_sha256, implementation_sha256):
        ancestry = schedule_ancestry(parent_binding)
        if len(ancestry) < 2:
            raise ValueError('stage3 requires a parent that is itself a completed segment')
        if ancestry[-1]['total_updates'] != start_update:
            raise ValueError('stage3 ancestry must end at the parent horizon')
        super().__init__(model, parent_config, start_update=start_update,
                         additional_updates=additional_updates, data_id=data_id,
                         protocol_id=protocol_id, parent_binding=parent_binding,
                         parent_sha256=parent_sha256, implementation_sha256=implementation_sha256)
        self.ancestry = ancestry
        continuation = self.binding['continuation']
        continuation['ancestry'] = dict(schema=ANCESTRY_SCHEMA,
                                        segments=copy.deepcopy(ancestry))
        continuation['segment_generation'] = len(ancestry) + 1
        continuation['learning_rate'] = (
            'new warmup and cosine segment; ancestral segment schedules reproduced exactly')

    def _lr_multiplier(self, updates=None):
        u = self.updates if updates is None else updates
        if u < self.segment_start:
            # Validate the parent's last Adam group LR against the schedule that
            # actually produced it, without retroactively extending any segment.
            return ancestry_multiplier(self.ancestry, u)
        local = u - self.segment_start
        c = self.config
        if local < c.warmup_updates:
            return (local + 1) / c.warmup_updates
        length = c.total_updates - self.segment_start
        return .5 * (1 + math.cos(math.pi * (local - c.warmup_updates) / (length - c.warmup_updates)))


def continue_full_state_gen3(parent, *, additional_updates, protocol_id, parent_sha256,
                            implementation_sha256):
    """Chain a third segment; the parent must already be a completed segment."""
    if parent._in_step or parent.updates != parent.config.total_updates:
        raise ValueError('completed update-boundary parent required')
    if parent.model.encoder.freeze_policy != 'frozen' or parent.config.unfreeze_update is not None:
        raise ValueError('this continuation supports frozen parent only')
    if 'continuation' not in parent.binding:
        raise ValueError('stage3 requires a segment parent; use hcm1_stage2 for a first-generation parent')
    rng = parent._rng_state()
    child = Segment3Trainer(copy.deepcopy(parent.model), parent.config, start_update=parent.updates,
                            additional_updates=additional_updates, data_id=parent.binding['data_id'],
                            protocol_id=protocol_id, parent_binding=parent.binding,
                            parent_sha256=parent_sha256, implementation_sha256=implementation_sha256)
    state = _cpu_tree(parent.optimizer.state_dict())
    child._validate_optimizer(state, parent.updates, sorted(state['state']))
    child.optimizer.load_state_dict(state)
    child.updates, child.draws = parent.updates, parent.draws
    child._restore_rng(rng)
    return child


def load_segment_parent(parent_config, manifest, parent_checkpoint, parent_sha256, parent_binding,
                        parent_record, *, segment_checkpoint, segment_sha256, segment_binding,
                        segment_record, device):
    """Reconstruct a completed second segment from its authenticated first generation.

    Composition only: `load_parent` authenticates generation one and
    `restore_segment` authenticates generation two. Neither is modified.
    """
    from music_critic.experiments.registry import load_record
    from scripts.hcm1_completed_segment import restore_segment
    digest = sha256()
    with Path(segment_checkpoint).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != segment_sha256:
        raise ValueError('stage3 segment checkpoint SHA-256 mismatch')
    first, sampler = load_parent(parent_config, manifest, parent_checkpoint, parent_sha256,
                                 parent_binding, parent_record, device=device)
    record = load_record(segment_record)
    second, sampler_state = restore_segment(first, segment_checkpoint, segment_binding, record,
                                           parent_sha256=parent_sha256)
    sampler.load_state_dict(sampler_state)
    if second.updates != second.config.total_updates or second.draws != sampler.draws:
        raise ValueError('segment progress or sampler mismatch')
    return second, sampler


def run(args):
    import torch
    from music_critic.experiments.registry import load_record, DATA_ACCESS_FIELDS
    from music_critic.pll.cli import _transition, _record_lock, _write_new, _artifact, _now, _admit_output
    from music_critic.pll.cloud import stop_signals
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    output = Path(args.output)
    record_path = Path(args.record)
    ledger = record_path.parents[2] / 'EXPERIMENT_LEDGER.md'
    _admit_output(output)
    if args.wall_seconds <= 0 or args.checkpoint_every <= 0 or args.additional_updates <= 0:
        raise ValueError('positive wall budget, checkpoint cadence and update count required')
    code_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    expected = dict(parent_checkpoint_sha256=args.parent_sha256,
                    segment_checkpoint_sha256=args.segment_sha256,
                    additional_updates=args.additional_updates,
                    checkpoint_every=args.checkpoint_every,
                    implementation_sha256=code_sha,
                    protocol_id=args.protocol_id,
                    wall_seconds=args.wall_seconds)
    with _record_lock(record_path):
        record = load_record(record_path)
        if record['status'] != 'planned' or record.get('hcm_stage3') != expected:
            raise ValueError('stage3 planned binding mismatch')
        policy = record['data_access']['declared_policy']
        if policy['validation_allowed'] or any(policy[name] for name in (
                'test_inputs_allowed', 'test_targets_allowed', 'test_metrics_allowed',
                'test_selection_allowed')):
            raise ValueError('stage3 TRAIN-only policy required')
        output.mkdir(parents=True, exist_ok=False)
        _write_new(output / 'planned-record.json', record)
        record = _transition(record_path, record, dict(
            record, status='running',
            execution={'started_at': _now(), 'device': args.device, 'torch': str(torch.__version__)}),
            ledger)
        started = time.monotonic()
        child = None
        sampler = None
        error = None
        status = 'failed'
        checkpoint = None
        try:
            parent, sampler = load_segment_parent(
                args.parent_config, args.manifest, args.parent_checkpoint, args.parent_sha256,
                args.parent_binding, args.parent_record,
                segment_checkpoint=args.segment_checkpoint, segment_sha256=args.segment_sha256,
                segment_binding=args.segment_binding, segment_record=args.segment_record,
                device=args.device)
            child = continue_full_state_gen3(
                parent, additional_updates=args.additional_updates, protocol_id=args.protocol_id,
                parent_sha256=args.segment_sha256, implementation_sha256=code_sha)
            del parent
            _write_new(output / 'training-binding.json', child.binding)
            checkpoint = output / f'checkpoint-{child.updates:08d}.pt'
            child.save_checkpoint(checkpoint, sampler_state=sampler.state_dict())
            with stop_signals() as stop, (output / 'metrics.jsonl').open('x') as log, \
                    (output / 'draws.jsonl').open('x') as draws:
                for _ in range(args.additional_updates):
                    if stop() or time.monotonic() - started >= args.wall_seconds:
                        break
                    sampled = tuple(sampler.draw() for _ in range(child.config.accumulation))
                    for sample in sampled:
                        draws.write(json.dumps(sample.receipt(), sort_keys=True) + '\n')
                    draws.flush()
                    os.fsync(draws.fileno())
                    metrics = child.step(tuple(x.example for x in sampled))
                    log.write(json.dumps(dict(elapsed_seconds=time.monotonic() - started,
                                              metrics=metrics), sort_keys=True,
                                         allow_nan=False) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
                    if child.updates % args.checkpoint_every == 0:
                        checkpoint = output / f'checkpoint-{child.updates:08d}.pt'
                        child.save_checkpoint(checkpoint, sampler_state=sampler.state_dict())
                status = 'completed' if child.updates == child.config.total_updates else 'aborted'
                final = output / f'checkpoint-{child.updates:08d}.pt'
                if final != checkpoint:
                    child.save_checkpoint(final, sampler_state=sampler.state_dict())
                checkpoint = final
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            receipt = dict(
                status=status, error=error,
                global_updates=None if child is None else child.updates,
                added_updates=None if child is None else child.updates - child.segment_start,
                draws=None if child is None else child.draws,
                elapsed_seconds=time.monotonic() - started,
                checkpoint=None if checkpoint is None else checkpoint.name)
            _write_new(output / 'receipt.json', receipt)
            verified = sampler is not None and not sampler.manifest.unverified_reads
            after = dict(
                record, status=status, results=receipt,
                validity={'reason': None if status == 'completed' else str(error or 'operator stop')},
                data_access={
                    **record['data_access'],
                    'access_evidence_status': 'verified_primary_evidence' if verified else 'pending_primary_evidence',
                    **{key: False if verified else None for key in DATA_ACCESS_FIELDS}},
                artifacts=record['artifacts'] + [
                    _artifact(p, record_path, p.name) for p in output.iterdir() if p.is_file()])
            _transition(record_path, record, after, ledger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('parent-config', 'manifest', 'parent-checkpoint', 'parent-sha256',
                 'parent-binding', 'parent-record', 'segment-checkpoint', 'segment-sha256',
                 'segment-binding', 'segment-record', 'record', 'output', 'protocol-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--additional-updates', type=int, default=10000)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wall-seconds', type=int, default=86400)
    run(parser.parse_args())
