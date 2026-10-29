"""Canonical forward-only experiment records for this isolated campaign."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from music_critic.experiments.registry import (
    canonical_json_bytes, load_record, seal_record, write_record_once, validate_record,
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path, locator=None):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(locator or path.name), sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def plan(path, experiment_id, kind, protocol, repository, parents=(), budget=None, seeds=()):
    record = dict(schema_version='1.0.0', experiment_id=experiment_id,
        kind=kind, phase='T08-H-member', title=experiment_id,
        status='planned', question='Does query-conditioned propagation improve oracle-H membership?',
        hypothesis='Within-H known member/nonmember supervision permits a controlled comparison.',
        decision_use='Bounded weak-label diagnostic; no default architecture replacement.',
        repository=repository, parent_ids=list(parents), seeds=list(seeds),
        protocol=protocol, budget=dict(planned=budget or {}),
        data_access=dict(access_evidence_status='pending_primary_evidence',
            validation_accessed=None, test_inputs_accessed=None, test_targets_accessed=None,
            test_metrics_computed=None, test_used_for_selection=None,
            declared_policy=dict(validation_allowed=False, test_inputs_allowed=False,
                test_targets_allowed=False, test_metrics_allowed=False, test_selection_allowed=False,
                source='User authorizes isolated H/member implementation, tests and bounded RTX3090 launch on 2026-10-25; all downstream splits inside corpus TRAIN.')),
        results={}, validity={}, claims=dict(supported=[], not_supported=[
            'human accuracy', 'raw H discovery', 'aesthetics', 'unseen raw pretraining']),
        artifacts=[], import_provenance=dict(mode='own authorized execution', planned_at=stamp()))
    write_record_once(path, seal_record(record))


def transition(path, status, *, results=None, reason=None, artifacts=None, actual=None,
               observed_train_only=False):
    path = Path(path)
    record = load_record(path)
    if record['status'] not in ('planned', 'running'):
        raise ValueError('terminal record is immutable')
    if status not in ('running', 'completed', 'failed', 'aborted', 'invalid'):
        raise ValueError('invalid forward transition')
    if record['status'] == 'running' and status == 'running':
        raise ValueError('already running')
    record['status'] = status
    if results is not None:
        record['results'] = results
    if artifacts is not None:
        record['artifacts'] = artifacts
    if actual is not None:
        record['budget']['actual'] = actual
    record['validity'] = dict(valid=status == 'completed', reason=reason)
    record['import_provenance'][status + '_at'] = stamp()
    if observed_train_only:
        access = record['data_access']
        access.update(access_evidence_status='verified_primary_evidence',
            validation_accessed=False, test_inputs_accessed=False, test_targets_accessed=False,
            test_metrics_computed=False, test_used_for_selection=False,
            observation_scope='Bound authenticated corpus TRAIN inputs only; downstream fit/selection/dev are internal TRAIN partitions.')
    updated = seal_record(record)
    validate_record(updated, filename=path.name)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(canonical_json_bytes(updated) + b'\n')
    tmp.replace(path)


def artifact(path, root, role):
    b = binding(path, Path(path).relative_to(root))
    return dict(role=role, locator=b['path'], sha256=b['sha256'],
                size_bytes=b['size_bytes'], availability='local')
