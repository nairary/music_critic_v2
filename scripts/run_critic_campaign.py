"""Hash-bound entry point for separately registered frozen-C3 critic campaigns.

Run from the repository with PYTHONPATH=src. Readiness only reads metadata and
hashes artifact bytes; prepare/probe/train require a planned or running record.
The cloud supervisor owns the hard wall deadline, evidence export and VM stop.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import sys
import time

import torch

from music_critic.critic import authored_data
from music_critic.critic.c3_backbone import C3_SOURCE_ID, load_c3_relationgate_adapter
from music_critic.critic.campaign import (
    CampaignConfig, _file_digest, _fingerprint, _json_bytes, _publish_bytes,
    _emit_probe_progress, _validate_encoder_batch_tolerance,
    prepare_feature_bank, probe_runtime, run_campaign,
    load_terminal_continuation, validate_terminal_continuation_config,
)
from music_critic.critic.campaign_data import CampaignDataConfig, inspect_manifest, load_campaign_dataset
from music_critic.critic.loss import CriticLoss
from music_critic.critic.model import MultiscaleAspectCritic
from music_critic.critic.runtime_equivalence import (
    RuntimeEquivalenceError, _read_reference_source, validate_reference_model_config,
)
from music_critic.critic.training import CriticTrainConfig, CriticTrainer, CurriculumStage


def _dataset_options(config):
    options = dict(config['dataset'])
    kind = options.pop('kind', 'generated')
    if kind == 'generated':
        CampaignDataConfig(**options)
    elif kind == 'authored':
        required = {'manifest_path', 'manifest_sha256'}
        if not required <= set(options) or set(options) - required - {'expected_dataset_fingerprint'}:
            raise ValueError('authored dataset config fields invalid')
        path = Path(options['manifest_path'])
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('absolute nonsymlink authored manifest path required')
        for field in ('manifest_sha256', *(['expected_dataset_fingerprint']
                                          if 'expected_dataset_fingerprint' in options else [])):
            if not isinstance(options[field], str) or not re.fullmatch('[0-9a-f]{64}', options[field]):
                raise ValueError('authored dataset SHA256 binding invalid')
    else:
        raise ValueError('unknown campaign dataset kind')
    return kind, options


def _training_config(config):
    options = dict(config['training'])
    if 'curriculum_stages' in options:
        stages = options['curriculum_stages']
        if (not isinstance(stages, (list, tuple))
                or any(not isinstance(stage, dict) or set(stage) != {'end_draw', 'edit_counts'}
                       or not isinstance(stage['edit_counts'], (list, tuple)) for stage in stages)):
            raise ValueError('invalid JSON curriculum stages')
        options['curriculum_stages'] = tuple(CurriculumStage(**stage) for stage in stages)
    return CriticTrainConfig(**options)


def _check_dataset_fingerprint(options, fingerprint):
    expected = options.get('expected_dataset_fingerprint')
    if expected is not None and expected != fingerprint:
        raise ValueError('authored dataset fingerprint mismatch')


def _inspect_dataset(config):
    kind, options = _dataset_options(config)
    if kind == 'generated':
        return inspect_manifest(options)
    raw = Path(options['manifest_path']).read_bytes()
    if sha256(raw).hexdigest() != options['manifest_sha256']:
        raise ValueError('authored manifest integrity mismatch')
    manifest = authored_data._json(raw)
    # Use the adapter's authoritative metadata preflight without opening any
    # referenced piece, edit plan or direction-review payload.
    rows = authored_data._preflight(manifest)
    fingerprint = authored_data._digest({'manifest_sha256': options['manifest_sha256'],
                                        'target_version': authored_data.TARGET_VERSION})
    _check_dataset_fingerprint(options, fingerprint)
    return {'schema': authored_data.MANIFEST_SCHEMA, 'manifest_sha256': options['manifest_sha256'],
            'source_id': manifest['source_id'], 'split_id': manifest['split_id'],
            'target_version': authored_data.TARGET_VERSION, 'dataset_fingerprint': fingerprint,
            'pairs': len(rows), 'works': len({row['work_id'] for row in rows}),
            'splits': dict(Counter(row['split'] for row in rows)),
            'source_scopes': dict(Counter(row['source_scope'] for row in rows)),
            'piece_payloads_accessed': False}


def _load_dataset(config):
    kind, options = _dataset_options(config)
    if kind == 'generated':
        return load_campaign_dataset(options)
    dataset = authored_data.load_authored_dataset(options['manifest_path'], options['manifest_sha256'])
    _check_dataset_fingerprint(options, dataset.fingerprint)
    return dataset


def validate_config(config):
    if config.get('schema') != 'CriticCampaignCLI@1.0.0':
        raise ValueError('critic campaign CLI schema mismatch')
    if not isinstance(config.get('experiment_id'), str) or not config['experiment_id'].strip():
        raise ValueError('experiment identity required')
    for field in ('experiment_record', 'output_root', *(['feature_cache_root'] if 'feature_cache_root' in config else [])):
        path = Path(config[field])
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('absolute nonsymlink campaign paths required')
    dataset_kind, _ = _dataset_options(config)
    training = _training_config(config)
    campaign = CampaignConfig(config['experiment_id'], **_campaign_options(config))
    guaranteed_draws = (training.curriculum_stages[-1].end_draw if training.curriculum_stages
                        else training.max_updates * training.batch_size)
    if campaign.selection_start_draw > guaranteed_draws:
        raise ValueError('selection_start_draw exceeds guaranteed campaign draw budget')
    if not math.isfinite(config['preparation_wall_seconds']) or config['preparation_wall_seconds'] <= 0:
        raise ValueError('positive preparation wall cap required')
    if type(config['cpu_threads']) is not int or config['cpu_threads'] < 1:
        raise ValueError('positive CPU thread count required')
    if config['model'] != {'feature_dim': 128, 'width': 128}:
        raise ValueError('this campaign admits only the agreed 128-dimensional model')
    if not math.isfinite(config['lambda_local']) or config['lambda_local'] < 0:
        raise ValueError('invalid localization weight')
    if dataset_kind == 'authored' and config['lambda_local'] != 0:
        raise ValueError('authored global pairs require lambda_local=0')
    if training.curriculum_stages and dataset_kind != 'authored':
        raise ValueError('authored dataset required for edit-count curriculum')
    if str(config['device']) not in {'cpu', 'cuda', 'cuda:0'}:
        raise ValueError('explicit CPU or single CUDA device required')
    deterministic = config.get('deterministic_algorithms', False)
    if type(deterministic) is not bool:
        raise ValueError('deterministic_algorithms must be boolean')
    if (deterministic and str(config['device']).startswith('cuda')
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8', ':16:8'}):
        raise ValueError('deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before execution')
    _probe_encoder_tolerance(config)
    _probe_reference_model(config)
    if 'continuation' in config:
        validate_terminal_continuation_config(config['continuation'])
    backbone = config['backbone']
    if (backbone.get('source_id', C3_SOURCE_ID) != C3_SOURCE_ID
            or len(backbone['sha256']) != 64 or type(backbone['size_bytes']) is not int
            or backbone['size_bytes'] < 1):
        raise ValueError('selected C3 artifact binding required')
    return config


def _probe_encoder_tolerance(config):
    probe = config.get('probe', {})
    if not isinstance(probe, dict) or ('encoder_batch_tolerance' in probe
                                     and not isinstance(probe['encoder_batch_tolerance'], dict)):
        raise ValueError('encoder_batch_tolerance requires an rtol/atol object')
    return _validate_encoder_batch_tolerance(probe.get('encoder_batch_tolerance'))


def _numeric_runtime():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def _probe_reference_model(config):
    probe = config.get('probe', {})
    return validate_reference_model_config(probe['reference_model']) if 'reference_model' in probe else None


def _campaign_options(config):
    result = dict(config['campaign'])
    result.setdefault('require_all_declared_aspects', True)
    return result


def _registered_record(config, mode):
    from music_critic.experiments.registry import validate_record
    record = json.loads(Path(config['experiment_record']).read_text())
    if (record.get('experiment_id') != config['experiment_id']
            or record.get('status') not in {'planned', 'running'}):
        raise ValueError('matching planned or running experiment record required')
    record = validate_record(record)
    protocol = record.get('protocol', {})
    if (protocol.get('cli_config_fingerprint') != _fingerprint(config)
            or protocol.get('cli_mode') not in {'prepare', 'probe', 'train'}
            or (mode != 'readiness' and protocol['cli_mode'] != mode)):
        raise ValueError('registered campaign config or mode binding mismatch')
    return _file_digest(config['experiment_record'])


def _new_trainer(config, bank, adapter, dataset):
    train_bank = tuple(p for p in bank if p.split == 'train')
    authored = _dataset_options(config)[0] == 'authored'
    if authored:
        # Unknown directions stay in the immutable dataset and its omissions
        # receipt; only observed global targets enter the optimizer schedule.
        eligible = tuple(p for p in train_bank if any(p.targets.pair_mask))
        if {p.work_id for p in eligible} != {p.work_id for p in train_bank}:
            raise ValueError('authored training work without available targets')
        train_bank = eligible
    devices = [torch.device(config['device']).index or 0] if str(config['device']).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config['training']['seed'])
        model = MultiscaleAspectCritic(**config['model']).to(config['device'])
    return CriticTrainer(model, CriticLoss(lambda_local=config['lambda_local']),
                         train_bank,
                         _training_config(config),
                         bindings={'source_id': adapter.binding.source_id,
                                   'split_id': dataset.fingerprint,
                                   'target_version': (authored_data.TARGET_VERSION
                                       if authored
                                       else 'CriticCampaignSources@1.0.0')},
                         frozen_backbones=(adapter.model,))


def execute(config, *, mode, resume=False, stop_requested=lambda: False):
    validate_config(config)
    if mode not in {'readiness', 'prepare', 'probe', 'train'} or (resume and mode != 'train'):
        raise ValueError('invalid campaign mode or resume mode')
    if 'continuation' in config and mode not in {'readiness', 'train'}:
        raise ValueError('terminal continuation is training-only')
    record = _registered_record(config, mode)
    reference_model = _probe_reference_model(config)
    if reference_model is not None:
        _read_reference_source(reference_model)
    manifest = _inspect_dataset(config)
    expected = {key: config['backbone'][key] for key in ('sha256', 'size_bytes')}
    if _file_digest(config['backbone']['checkpoint_path']) != expected:
        raise ValueError('backbone artifact integrity mismatch')
    if mode == 'readiness':
        return {'schema': 'CriticCampaignReadiness@1.0.0', 'experiment_id': config['experiment_id'],
                'config_fingerprint': _fingerprint(config), 'record': record, 'source_manifest': manifest,
                'checkpoint': expected, 'raw_piece_payloads_accessed': False, 'model_loaded': False,
                'native_admission': False, 'test_accessed': False}
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    invocation = root / ('invocation-' + mode + '.json')
    identity = {'config': config, 'mode': mode, 'record_at_start': record}
    if invocation.exists():
        previous = json.loads(invocation.read_text())
        if previous['config'] != config or previous['mode'] != mode or (mode == 'train' and not resume):
            raise ValueError('existing invocation requires matching explicit continuation')
    else:
        _publish_bytes(invocation, _json_bytes(identity))
    terminal_path = root / ('terminal-' + mode + '.json')
    if terminal_path.exists():
        raise ValueError('terminal invocation cannot be overwritten')
    prior_threads = torch.get_num_threads()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    started = time.monotonic()
    try:
        torch.set_num_threads(config['cpu_threads'])
        torch.use_deterministic_algorithms(config.get('deterministic_algorithms', False), warn_only=False)
        dataset = _load_dataset(config)
        dataset_receipt = _json_bytes({
            'fingerprint': dataset.fingerprint, 'summary': dataset.source_summary,
            'split_assignments': dataset.split_assignments, 'omissions': dataset.omissions})
        dataset_path = root / ('dataset-' + mode + '.json')
        if dataset_path.exists():
            if dataset_path.read_bytes() != dataset_receipt:
                raise ValueError('dataset receipt binding mismatch')
        else:
            _publish_bytes(dataset_path, dataset_receipt)
        adapter = load_c3_relationgate_adapter(config['backbone']['checkpoint_path'],
            expected_sha256=expected['sha256'], expected_size_bytes=expected['size_bytes'],
            source_id=config['backbone'].get('source_id', C3_SOURCE_ID)).to(config['device'])
        feature_root = Path(config.get('feature_cache_root', root / 'prepared'))
        bank, preparation = prepare_feature_bank(dataset, adapter, feature_root,
            wall_seconds=config['preparation_wall_seconds'], stop_requested=stop_requested)
        memory = {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                  * (1 if sys.platform == 'darwin' else 1024),
                  'feature_tensor_bytes': preparation['tensor_bytes'],
                  'cuda_preparation_peak_allocated_bytes': (torch.cuda.max_memory_allocated(config['device'])
                     if str(config['device']).startswith('cuda') else None),
                  'cuda_free_total_bytes_after_cache': (list(torch.cuda.mem_get_info(config['device']))
                     if str(config['device']).startswith('cuda') else None)}
        if mode == 'prepare':
            result = preparation
        else:
            dev = tuple(p for p in bank if p.split == 'head_dev')
            if not dev or not any(p.split == 'train' for p in bank):
                raise ValueError('both TRAIN and fixed head-dev banks required')
            factory = lambda: _new_trainer(config, bank, adapter, dataset)
            if mode == 'probe':
                with (root / 'progress-probe.jsonl').open('ab') as progress_stream:
                    def record_progress(event):
                        # The CLI clock includes initial trainer construction;
                        # standalone probe callbacks use their own probe clock.
                        row = {**event, 'elapsed_seconds': max(0., time.monotonic() - started)}
                        progress_stream.write(_json_bytes(row))
                        progress_stream.flush()
                    initialization_started = _emit_probe_progress(
                        record_progress, 'trainer_initialization', 'started', None,
                        started=started, clock=time.monotonic)
                    trainer = factory()
                    _emit_probe_progress(record_progress, 'trainer_initialization', 'completed', trainer,
                                         started=started, clock=time.monotonic,
                                         stage_started=initialization_started)
                    result = probe_runtime(trainer, factory, dev, updates=config['probe']['updates'],
                        adapter=adapter, raw_batch=(dataset.pairs[0].original, dataset.pairs[0].variant),
                        encoder_batch_tolerance=_probe_encoder_tolerance(config),
                        wall_seconds=config['campaign']['wall_seconds'], stop_requested=stop_requested,
                        progress_callback=record_progress, reference_model=reference_model)
                trainer.save(root / 'probe-checkpoint.pt')
                result['checkpoint'] = _file_digest(root / 'probe-checkpoint.pt')
            else:
                trainer = factory()
                campaign_config = CampaignConfig(config['experiment_id'], **_campaign_options(config))
                continuation = (load_terminal_continuation(trainer, dev, campaign_config, config['continuation'])
                                if 'continuation' in config else None)
                if continuation is not None and not (root / 'continuation.json').exists():
                    _publish_bytes(root / 'continuation.json', _json_bytes(continuation))
                result = run_campaign(trainer, dev, campaign_config,
                                      root / 'training', resume=resume, continuation=continuation,
                                      stop_requested=stop_requested)
        report = {'schema': 'CriticCampaignInvocation@1.0.0', 'experiment_id': config['experiment_id'],
                  'mode': mode, 'status': result.get('status', 'completed'), 'result': result,
                  'preparation': preparation, 'dataset_fingerprint': dataset.fingerprint,
                  'full_bank_memory': memory,
                  'feature_binding': asdict(adapter.binding), 'wall_seconds': time.monotonic() - started,
                  'feature_cache_root': str(feature_root),
                  'process_final_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      * (1 if sys.platform == 'darwin' else 1024),
                  'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                                  'platform': platform.platform(), 'device': config['device'],
                                  'cuda': torch.version.cuda, 'cpu_threads': config['cpu_threads'],
                                  'numeric_runtime': _numeric_runtime()},
                  'test_accessed': False}
        _publish_bytes(terminal_path, _json_bytes(report))
        return report
    except Exception as exc:
        if not terminal_path.exists():
            _publish_bytes(terminal_path, _json_bytes({'schema': 'CriticCampaignInvocation@1.0.0',
                'experiment_id': config['experiment_id'], 'mode': mode, 'status': 'failed',
                'error_type': type(exc).__name__, 'error': str(exc),
                'wall_seconds': time.monotonic() - started, 'test_accessed': False,
                'numeric_runtime': _numeric_runtime(),
                **({'runtime_equivalence': exc.report} if isinstance(exc, RuntimeEquivalenceError) else {})}))
        raise
    finally:
        try:
            torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)
        finally:
            torch.set_num_threads(prior_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=('readiness', 'prepare', 'probe', 'train'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    stopped = []
    def stop(number, _frame):
        stopped.append(number)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    try:
        report = execute(json.loads(args.config.read_text()), mode=args.mode, resume=args.resume,
                         stop_requested=lambda: bool(stopped))
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        return 0 if report.get('status', 'completed') == 'completed' else 2
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}),
              file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
