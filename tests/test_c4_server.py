"""Synthetic lifecycle checks; these never launch training or use CUDA."""
import pytest

from scripts.run_c4_server import finalize_record, _transition
from music_critic.experiments.registry import load_record, seal_record, write_record_once


def planned():
    return dict(schema_version='1.0.0', experiment_id='SYNTHETIC-C4',
        kind='engineering_gate', phase='synthetic', title='Synthetic C4 lifecycle',
        status='running', question='Does terminal evidence control completion?',
        hypothesis='Exit zero requires a completed report.', decision_use='Unit test only',
        repository={}, seeds=[17], budget={}, results={}, validity={}, claims={},
        artifacts=[], import_provenance={},
        data_access=dict(validation_accessed=None, test_inputs_accessed=None,
            test_targets_accessed=None, test_metrics_computed=None, test_used_for_selection=None,
            access_evidence_status='pending_primary_evidence',
            declared_policy=dict(validation_allowed=True, test_inputs_allowed=False,
                test_targets_allowed=False, test_metrics_allowed=False,
                test_selection_allowed=False, source='Synthetic fixture')))


def test_exit_zero_without_completed_report_is_not_completed(tmp_path):
    record = finalize_record(planned(), terminal={'status': 'completed', 'returncode': 0,
        'wall_seconds': 1}, report=None, artifact_root=tmp_path, locator_prefix='c4/synthetic')
    assert record['status'] == 'failed'
    assert 'report' in record['validity']['reason']
    assert record['data_access']['validation_accessed'] is None


def test_terminal_record_cannot_be_rewritten(tmp_path):
    record = planned()
    record['status'] = 'failed'
    with pytest.raises(ValueError, match='terminal'):
        finalize_record(record, terminal={}, report=None,
                        artifact_root=tmp_path, locator_prefix='c4/synthetic')


def test_report_and_hashes_are_preserved_with_observed_access(tmp_path):
    (tmp_path / 'weights.pt').write_bytes(b'synthetic checkpoint')
    report = {'status': 'completed', 'experiment_id': 'SYNTHETIC-C4',
        'applied_updates': 32, 'draws': 128,
        'access': {'validation_accessed': False, 'test_inputs_accessed': False,
            'test_targets_accessed': False, 'test_metrics_computed': False,
            'test_used_for_selection': False}}
    record = finalize_record(planned(), terminal={'status': 'completed', 'returncode': 0,
        'wall_seconds': 2}, report=report, artifact_root=tmp_path, locator_prefix='c4/synthetic')
    assert record['status'] == 'completed'
    assert record['results']['producer_report'] == report
    assert record['budget']['actual']['applied_updates'] == 32
    assert record['data_access']['access_evidence_status'] == 'verified_primary_evidence'
    assert record['artifacts'][-1]['locator'] == 'c4/synthetic/weights.pt'
    assert record['artifacts'][-1]['size_bytes'] == 20


def test_deadline_remains_aborted_even_if_report_claims_completed(tmp_path):
    record = finalize_record(planned(), terminal={'status': 'aborted', 'returncode': -15,
        'reason': 'wall_cap', 'wall_seconds': 3}, report={'status': 'completed'},
        artifact_root=tmp_path, locator_prefix='c4/synthetic')
    assert record['status'] == 'aborted'
    assert record['validity']['reason'] == 'wall_cap'


def test_other_attempt_report_cannot_establish_completion_or_access(tmp_path):
    report = {'status': 'completed', 'experiment_id': 'OTHER', 'applied_updates': 80000,
              'access': {'validation_accessed': True}}
    record = finalize_record(planned(), terminal={'status': 'completed', 'returncode': 0,
        'wall_seconds': 1}, report=report, artifact_root=tmp_path, locator_prefix='c4/synthetic')
    assert record['status'] == 'failed'
    assert record['data_access']['validation_accessed'] is None
    assert 'actual' not in record['budget']


def test_report_must_match_bound_manifest(tmp_path):
    record = planned()
    record['protocol'] = {'manifest_fingerprint': 'expected'}
    result = finalize_record(record, terminal={'status': 'completed', 'returncode': 0,
        'wall_seconds': 1}, report={'status': 'completed', 'experiment_id': 'SYNTHETIC-C4',
            'manifest_fingerprint': 'wrong'}, artifact_root=tmp_path, locator_prefix='c4/synthetic')
    assert result['status'] == 'failed'


def test_transition_roundtrips_canonical_record(tmp_path):
    record = planned()
    record['status'] = 'planned'
    path = tmp_path / 'SYNTHETIC-C4.json'
    write_record_once(path, seal_record(record))
    record['status'] = 'running'
    _transition(path, record)
    assert load_record(path)['status'] == 'running'
    assert path.read_bytes().endswith(b'\n')


def test_operational_checkpoint_paths_become_portable_compact_locators(tmp_path):
    result = finalize_record(planned(), terminal={'status': 'completed', 'returncode': 0,
        'wall_seconds': 1}, report={'status': 'completed', 'experiment_id': 'SYNTHETIC-C4',
            'checkpoint': {'path': str(tmp_path / 'checkpoint.pt')}},
        artifact_root=tmp_path, locator_prefix='c4/synthetic')
    assert result['status'] == 'completed'
    assert result['results']['producer_report']['checkpoint']['path'] == 'c4/synthetic/checkpoint.pt'
