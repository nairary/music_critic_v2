"""Synthetic authored CLI routing; no campaign data, GPU or external access."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from scripts import run_critic_campaign as cli
from music_critic.critic.authored_data import TARGET_VERSION
from music_critic.critic.campaign import _file_digest
from music_critic.critic.training import CurriculumStage
from .test_authored_data import packet
from .test_campaign import write_fixture_record
from .test_curriculum_training import TinyCritic, make_bank


def authored_config(tmp_path):
    _manifest, manifest_path, checksum, _write = packet(tmp_path)
    checkpoint = tmp_path / 'backbone.pt'
    checkpoint.write_bytes(b'synthetic checkpoint binding; never deserialized')
    return {
        'schema': 'CriticCampaignCLI@1.0.0', 'experiment_id': 'EXP-CLI-SYNTHETIC',
        'experiment_record': str(tmp_path / 'record.json'), 'output_root': str(tmp_path / 'out'),
        'dataset': {'kind': 'authored', 'manifest_path': str(manifest_path), 'manifest_sha256': checksum},
        'backbone': {'checkpoint_path': str(checkpoint), **_file_digest(checkpoint)},
        'training': {'seed': 17, 'learning_rate': .02, 'batch_size': 2, 'max_updates': 10,
                     'max_attempts': 10, 'clip_grad_norm': 1.,
                     'curriculum_stages': [{'end_draw': 6, 'edit_counts': [3, 4]},
                                           {'end_draw': 12, 'edit_counts': [2, 1]},
                                           {'end_draw': 20, 'edit_counts': [1]}]},
        'campaign': {'wall_seconds': 60, 'evaluate_every_updates': 2,
                     'checkpoint_every_attempts': 2, 'selection_start_draw': 14},
        'preparation_wall_seconds': 60, 'device': 'cpu', 'cpu_threads': 1,
        'model': {'feature_dim': 128, 'width': 128}, 'lambda_local': 0.,
        'probe': {'updates': 2},
    }


def test_authored_readiness_decodes_curriculum_without_opening_payloads(tmp_path, monkeypatch):
    config = authored_config(tmp_path)
    write_fixture_record(config, 'prepare')
    # Manifest inspection must succeed even when all referenced payloads are absent.
    for name in ('original.json', 'variant.json', 'plan.json', 'review.json'):
        (tmp_path / name).unlink()
    monkeypatch.setattr(cli, 'load_campaign_dataset', lambda *_: pytest.fail('generated route used'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_kw: pytest.fail('model loaded'))
    before = deepcopy(config)
    result = cli.execute(config, mode='readiness')
    assert config == before
    assert result['source_manifest']['schema'] == 'AuthoredPairManifest@1.0.0'
    assert result['source_manifest']['pairs'] == 1
    assert result['source_manifest']['piece_payloads_accessed'] is False
    assert result['raw_piece_payloads_accessed'] is False
    assert not (tmp_path / 'out').exists()
    parsed = cli._training_config(config)
    assert parsed.curriculum_stages == (CurriculumStage(6, (3, 4)), CurriculumStage(12, (2, 1)),
                                         CurriculumStage(20, (1,)))


@pytest.mark.parametrize('stages', [None, {}, [1], [{'end_draw': 20}],
                                  [{'end_draw': 20, 'edit_counts': '1'}],
                                  [{'end_draw': 20, 'edit_counts': [1], 'unknown': 1}]])
def test_invalid_json_stage_shapes_fail_during_configuration(tmp_path, stages):
    config = authored_config(tmp_path)
    config['training']['curriculum_stages'] = stages
    with pytest.raises(ValueError, match='JSON curriculum'):
        cli.validate_config(config)
    assert not (tmp_path / 'out').exists()


def test_authored_global_only_and_dataset_kind_are_explicit(tmp_path):
    config = authored_config(tmp_path)
    config['lambda_local'] = .25
    with pytest.raises(ValueError, match='lambda_local=0'):
        cli.validate_config(config)
    config['lambda_local'] = 0
    config['dataset']['kind'] = 'misspelled-authored'
    with pytest.raises(ValueError, match='unknown campaign dataset kind'):
        cli.validate_config(config)


@pytest.mark.parametrize('defect', ['manifest_hash', 'final_holdout', 'source_scope',
                                   'dataset_fingerprint', 'backbone_hash'])
def test_authored_admission_fails_before_model_or_output_access(tmp_path, monkeypatch, defect):
    config = authored_config(tmp_path)
    if defect in {'final_holdout', 'source_scope'}:
        manifest = json.loads((tmp_path / 'manifest.json').read_text())
        key, value = ('split', 'final_holdout') if defect == 'final_holdout' else ('source_scope', 'test')
        manifest['pairs'][0][key] = value
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
        config['dataset']['manifest_sha256'] = _file_digest(tmp_path / 'manifest.json')['sha256']
    elif defect == 'manifest_hash':
        config['dataset']['manifest_sha256'] = '0' * 64
    elif defect == 'backbone_hash':
        config['backbone']['sha256'] = '0' * 64
    else:
        config['dataset']['expected_dataset_fingerprint'] = '0' * 64
    write_fixture_record(config, 'prepare')
    monkeypatch.setattr(cli.authored_data, 'load_authored_dataset',
                        lambda *_: pytest.fail('raw payloads accessed'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_kw: pytest.fail('model loaded'))
    with pytest.raises(ValueError, match='(integrity|forbidden|fingerprint)'):
        cli.execute(config, mode='prepare')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('deterministic', [None, False, True])
def test_authored_prepare_replays_saved_pair_and_keeps_weak_target_binding(tmp_path, monkeypatch, deterministic):
    import torch
    config = authored_config(tmp_path)
    if deterministic is not None:
        config['deterministic_algorithms'] = deterministic
    data = cli._load_dataset(config)
    config['dataset']['expected_dataset_fingerprint'] = data.fingerprint
    write_fixture_record(config, 'prepare')
    from music_critic.critic.contracts import FeatureBinding
    adapter = SimpleNamespace(binding=FeatureBinding('synthetic', 'weights', 'v1'))
    adapter.to = lambda _device: adapter
    calls = []
    def load_backbone(path, **bindings):
        calls.append(('backbone', path, bindings))
        return adapter
    def prepare(dataset, passed_adapter, feature_root, **options):
        assert torch.are_deterministic_algorithms_enabled() is bool(deterministic)
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
        assert passed_adapter is adapter
        assert dataset.fingerprint == data.fingerprint
        assert dataset.pairs[0].targets.provenance['target_version'] == TARGET_VERSION
        assert dataset.pairs[0].targets.local_targets == ({}, {})
        calls.append(('features', dataset.pairs[0].edit_count, dataset.pairs[0].genre))
        return (), {'tensor_bytes': 0, 'status': 'completed'}
    monkeypatch.setattr(cli, 'load_campaign_dataset', lambda *_: pytest.fail('generated route used'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', load_backbone)
    monkeypatch.setattr(cli, 'prepare_feature_bank', prepare)
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(not bool(deterministic), warn_only=True)
        result = cli.execute(config, mode='prepare')
        assert torch.are_deterministic_algorithms_enabled() is not bool(deterministic)
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == tf32
    assert result['status'] == 'completed' and result['dataset_fingerprint'] == data.fingerprint
    assert result['environment']['numeric_runtime']['deterministic_algorithms'] is bool(deterministic)
    assert result['environment']['numeric_runtime']['deterministic_warn_only'] is False
    assert calls[0][2]['expected_sha256'] == config['backbone']['sha256']
    assert calls[0][2]['expected_size_bytes'] == config['backbone']['size_bytes']
    assert calls[1] == ('features', 1, 'classical')
    assert (tmp_path / 'out' / 'terminal-prepare.json').is_file()


def test_authored_trainer_is_fresh_frozen_and_binds_curriculum_targets(tmp_path, monkeypatch):
    import torch
    config = authored_config(tmp_path)
    dataset = cli._load_dataset(config)
    adapter = SimpleNamespace(binding=SimpleNamespace(source_id='synthetic'),
                              model=torch.nn.Linear(1, 1).requires_grad_(False))
    monkeypatch.setattr(cli, 'MultiscaleAspectCritic', lambda **_options: TinyCritic())
    bank = (*make_bank(), *(pair for pair in make_bank(split='head_dev') if pair.edit_count == 1))
    first = cli._new_trainer(config, bank, adapter, dataset)
    second = cli._new_trainer(config, bank, adapter, dataset)
    assert first.model is not second.model
    assert first.bindings['target_version'] == TARGET_VERSION
    assert first.bindings['split_id'] == dataset.fingerprint
    assert first.config.curriculum_stages[-1] == CurriculumStage(20, (1,))
    assert len(first.exposure_plan) == 20
    assert all(pair.split == 'train' for pair in first.bank)
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert not {id(p) for p in first.parameters} & {id(p) for p in adapter.model.parameters()}
    step = first.step()
    assert step.applied and set(step.edit_counts) <= {3, 4}


def test_authored_config_mode_binding_cannot_be_changed_after_registration(tmp_path):
    config = authored_config(tmp_path)
    write_fixture_record(config, 'prepare')
    with pytest.raises(ValueError, match='config or mode binding'):
        cli.execute(config, mode='train')
    config['training']['curriculum_stages'][0]['edit_counts'] = [3]
    with pytest.raises(ValueError, match='config or mode binding'):
        cli.execute(config, mode='readiness')


def test_authored_unknown_pairs_remain_in_source_bank_but_not_training_schedule(tmp_path, monkeypatch):
    import torch
    config = authored_config(tmp_path)
    dataset = cli._load_dataset(config)
    adapter = SimpleNamespace(binding=SimpleNamespace(source_id='synthetic'),
                              model=torch.nn.Linear(1, 1).requires_grad_(False))
    monkeypatch.setattr(cli, 'MultiscaleAspectCritic', lambda **_options: TinyCritic())
    eligible = make_bank()
    unknown = replace(eligible[0], pair_id='unknown-direction',
                      targets=replace(eligible[0].targets, pair_mask=(False,) * 4,
                                      preference=(0,) * 4))
    bank = (*eligible, unknown)
    trainer = cli._new_trainer(config, bank, adapter, dataset)
    assert len(bank) == len(eligible) + 1 and bank[-1] is unknown
    assert trainer.bank == eligible
    assert all(draw.pair_id != unknown.pair_id for draw in trainer.exposure_plan)
    # Masking the sole example in a required cell must still block admission.
    with pytest.raises(ValueError, match='curriculum_coverage_missing'):
        cli._new_trainer(config, (*eligible[1:], unknown), adapter, dataset)
    missing_work = tuple(replace(p, targets=replace(p.targets, pair_mask=(False,) * 4,
                                                    preference=(0,) * 4))
                         if p.work_id == eligible[0].work_id else p for p in eligible)
    with pytest.raises(ValueError, match='work without available targets'):
        cli._new_trainer(config, missing_work, adapter, dataset)


@pytest.mark.parametrize('curriculum', [True, False])
def test_selection_gate_beyond_guaranteed_draws_fails_before_payload_access(tmp_path, monkeypatch, curriculum):
    config = authored_config(tmp_path)
    if not curriculum:
        config['training'].pop('curriculum_stages')
        # A larger attempt cap cannot guarantee more than max_updates batches.
        config['training']['max_attempts'] = 15
    config['campaign']['selection_start_draw'] = 21
    write_fixture_record(config, 'train')
    monkeypatch.setattr(cli, '_load_dataset', lambda *_: pytest.fail('raw payloads accessed'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_kw: pytest.fail('model loaded'))
    with pytest.raises(ValueError, match='selection_start_draw'):
        cli.execute(config, mode='train')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('workspace', [None, '', ':bad:8'])
def test_deterministic_cuda_requires_workspace_before_payload_or_model(tmp_path, monkeypatch, workspace):
    config = authored_config(tmp_path)
    config.update(device='cuda:0', deterministic_algorithms=True)
    write_fixture_record(config, 'prepare')
    if workspace is None:
        monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    else:
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', workspace)
    monkeypatch.setattr(cli, '_load_dataset', lambda *_: pytest.fail('payload accessed'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_k: pytest.fail('model loaded'))
    with pytest.raises(ValueError, match='CUBLAS_WORKSPACE_CONFIG'):
        cli.execute(config, mode='prepare')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('workspace', [':4096:8', ':16:8'])
def test_deterministic_cuda_readiness_accepts_valid_workspace_without_loading_model(tmp_path, monkeypatch, workspace):
    config = authored_config(tmp_path)
    config.update(device='cuda:0', deterministic_algorithms=True)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', workspace)
    write_fixture_record(config, 'probe')
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_k: pytest.fail('model loaded'))
    assert cli.execute(config, mode='readiness')['model_loaded'] is False


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_deterministic_config_is_boolean(tmp_path, value):
    config = authored_config(tmp_path)
    config['deterministic_algorithms'] = value
    with pytest.raises(ValueError, match='deterministic_algorithms must be boolean'):
        cli.validate_config(config)


@pytest.mark.parametrize('tolerance', [None, {}, {'rtol': 1e-4}, {'rtol': 1e-4, 'atol': 1e-4, 'extra': 1},
    {'rtol': 0, 'atol': 1e-4}, {'rtol': 1e-2, 'atol': 1e-4},
    {'rtol': float('nan'), 'atol': 1e-4}, {'rtol': 1e-4, 'atol': float('inf')},
    {'rtol': True, 'atol': 1e-4}, {'rtol': 1e-4, 'atol': -1e-4}])
def test_encoder_tolerance_is_explicit_finite_positive_and_bounded(tmp_path, tolerance):
    config = authored_config(tmp_path)
    config['probe']['encoder_batch_tolerance'] = tolerance
    with pytest.raises(ValueError, match='encoder_batch_tolerance'):
        cli.validate_config(config)
    assert not (tmp_path / 'out').exists()


def test_deterministic_flag_restored_and_actual_state_recorded_on_failure(tmp_path, monkeypatch):
    import torch
    config = authored_config(tmp_path)
    config['deterministic_algorithms'] = True
    write_fixture_record(config, 'prepare')
    def fail_load(_config):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise RuntimeError('synthetic payload failure')
    monkeypatch.setattr(cli, '_load_dataset', fail_load)
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        with pytest.raises(RuntimeError, match='synthetic payload failure'):
            cli.execute(config, mode='prepare')
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    report = json.loads((tmp_path / 'out' / 'terminal-prepare.json').read_text())
    assert report['numeric_runtime']['deterministic_algorithms'] is True
    assert report['numeric_runtime']['deterministic_warn_only'] is False
