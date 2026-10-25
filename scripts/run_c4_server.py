"""Record lifecycle around the existing bounded shared-server supervisor."""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
import os
from pathlib import Path

from music_critic.experiments.registry import (
    DATA_ACCESS_FIELDS, canonical_json, load_record, seal_record, validate_record,
)
from scripts.run_critic_server import execute


def _digest(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _portable(value, root, prefix):
    """Compact records use locators; original operational JSON stays in artifacts."""
    if isinstance(value, dict):
        return {key: _portable(child, root, prefix) for key, child in value.items()}
    if isinstance(value, list):
        return [_portable(child, root, prefix) for child in value]
    if isinstance(value, str) and value.startswith('/'):
        path = Path(value)
        try:
            return prefix.rstrip('/') + '/' + path.relative_to(root.resolve()).as_posix()
        except ValueError:
            return 'external://c4-executor' + value
    return value


def finalize_record(record, *, terminal, report, artifact_root, locator_prefix):
    """Exit zero alone never establishes completed scientific execution."""
    if record['status'] not in ('planned', 'running'):
        raise ValueError('c4_server.terminal_record_is_immutable')
    result = copy.deepcopy(record)
    report_bound = (isinstance(report, dict)
                    and report.get('experiment_id') == record['experiment_id'])
    for field in ('manifest_fingerprint', 'source_fingerprint'):
        expected = record.get('protocol', {}).get(field)
        if expected is not None and (not report_bound or report.get(field) != expected):
            report_bound = False
    complete = (terminal.get('status') == 'completed' and terminal.get('returncode') == 0
                and report_bound and report.get('status') == 'completed')
    status = 'completed' if complete else ('aborted' if terminal.get('status') == 'aborted' else 'failed')
    result['status'] = status
    result['validity'] = {'valid': True if complete else None,
        'reason': 'completed_producer_report_and_zero_exit' if complete else
            terminal.get('reason', 'missing_or_incomplete_producer_report')}
    if not complete and terminal.get('status') == 'completed':
        result['validity']['reason'] = 'missing_unbound_or_incomplete_producer_report'
    result['results'] = {'terminal_supervision': _portable(terminal, Path(artifact_root), locator_prefix),
                         'producer_report': _portable(report, Path(artifact_root), locator_prefix)}
    result['budget']['actual_wall_seconds'] = terminal.get('wall_seconds')
    if report_bound:
        result['budget']['actual'] = {key: report[key] for key in
            ('applied_updates', 'attempted_updates', 'skipped_updates', 'draws') if key in report}
        access = report.get('access', {})
        for key in DATA_ACCESS_FIELDS:
            if type(access.get(key)) is bool:
                result['data_access'][key] = access[key]
    observations = [result['data_access'][key] for key in DATA_ACCESS_FIELDS]
    result['data_access']['access_evidence_status'] = (
        'verified_primary_evidence' if all(type(x) is bool for x in observations) else
        'partial_primary_evidence' if any(type(x) is bool for x in observations) else
        'pending_primary_evidence')
    result['data_access']['observation_scope'] = 'Producer access report; no OS file-access tracing.'
    root = Path(artifact_root)
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('c4_server.symlink_in_evidence')
        if path.is_file():
            result['artifacts'].append({'role': 'c4_execution_evidence',
                'sha256': _digest(path), 'size_bytes': path.stat().st_size,
                'availability': 'external',
                'locator': locator_prefix.rstrip('/') + '/' + path.relative_to(root).as_posix()})
    result['conclusion'] = ('Requested execution completed; quality claims require the retained validation metrics.'
        if complete else 'Execution did not complete; retain this attempt and use a new ID for any retry.')
    return validate_record(seal_record(result))


def _transition(path, record):
    current = load_record(path)
    if (current['experiment_id'] != record['experiment_id']
            or current['status'] not in ('planned', 'running')):
        raise ValueError('c4_server.invalid_lifecycle_transition')
    temporary = path.with_suffix('.partial')
    with temporary.open('x') as stream:
        stream.write(canonical_json(validate_record(seal_record(record), filename=path)) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--locator-prefix', required=True)
    args = parser.parse_args(argv)
    record = load_record(args.record)
    if record['status'] != 'planned':
        raise ValueError('c4_server.fresh_planned_record_required')
    config = json.loads(args.config.read_text())
    # Child receives a separate immutable planned record; only this live record advances.
    running = copy.deepcopy(record)
    running['status'] = 'running'
    running['data_access']['access_evidence_status'] = 'pending_primary_evidence'
    _transition(args.record, running)
    try:
        terminal = execute(config)
    except BaseException as exc:
        terminal = {'status': 'failed', 'returncode': None, 'wall_seconds': None,
                    'reason': 'supervisor_exception', 'error': str(exc), 'error_type': type(exc).__name__}
    report = None
    if args.report.exists():
        try:
            candidate = json.loads(args.report.read_text())
            if isinstance(candidate, dict):
                report = candidate
        except (OSError, ValueError):
            pass
    try:
        final = finalize_record(running, terminal=terminal, report=report,
                               artifact_root=args.artifact_root, locator_prefix=args.locator_prefix)
    except Exception as exc:
        final = copy.deepcopy(running)
        final['status'] = 'failed'
        final['validity'] = {'valid': None, 'reason': 'evidence_finalization_error: ' + str(exc)}
        final['results'] = {'terminal_supervision': terminal,
                            'finalization_error_type': type(exc).__name__}
        final['budget']['actual_wall_seconds'] = terminal.get('wall_seconds')
        final = seal_record(final)
    _transition(args.record, final)
    print(json.dumps({'experiment_id': final['experiment_id'], 'status': final['status'],
                      'reason': final['validity']['reason']}), flush=True)
    return 0 if final['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
