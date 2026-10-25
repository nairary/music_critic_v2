"""Synthetic reference-comparison mechanics; no GPU or retained weights."""
import copy
from dataclasses import replace
from hashlib import sha256
import inspect

import pytest
import torch

from music_critic.critic import runtime_equivalence as runtime
from music_critic.critic.loss import CriticLoss
from music_critic.critic.model import MultiscaleAspectCritic
from music_critic.critic.targets import PairTargets
from music_critic.critic.training import CachedPair, CriticTrainConfig, CriticTrainer, _fingerprint, _rng_capture
from .test_model import fixture_sample, pyramid


def subject(*, identical=False, learning_rate=.003, pair_mask=(True,) * 4):
    torch.manual_seed(17)
    original = fixture_sample('synthetic-original')
    variant = fixture_sample('synthetic-variant')
    if not identical:
        variant = replace(variant, embeddings={key: value * .8 + .25
                                               for key, value in variant.embeddings.items()})
    views = []
    for sample in (original, variant):
        views.append(pyramid(replace(sample, embeddings={
            key: value.detach().clone() for key, value in sample.embeddings.items()})))
    target = PairTargets('synthetic-work', ('synthetic-original', 'synthetic-variant'),
                         pair_mask, tuple(int(value) for value in pair_mask), ({}, {}), {})
    pair = CachedPair('synthetic-pair', target.work_id, *views, target)
    return CriticTrainer(MultiscaleAspectCritic(8, 8), CriticLoss(0), (pair,),
        CriticTrainConfig(17, learning_rate, 1, 4, 4, 1., weight_decay=0.),
        bindings={'source_id': 'synthetic', 'split_id': 'synthetic', 'target_version': 'synthetic'})


def test_reference_comparison_preserves_trainer_bank_and_global_rng(monkeypatch):
    trainer = subject()
    reference = copy.deepcopy(trainer.model)
    monkeypatch.setattr(runtime, '_load_reference_model', lambda *_a, **_kw: reference)
    before, bank = trainer.state_dict()['fingerprint'], _fingerprint(trainer.bank)
    rng = _fingerprint(_rng_capture())
    events = []
    report = runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic'},
        clock=lambda: 0., progress_callback=events.append)
    assert report['passed'] and report['pair_count'] == 1 and report['forward_views'] == 8
    assert report['optimizer_applications'] == report['paired_draws'] == 2
    assert report['reference_sha256'] == 'synthetic'
    assert all(row['max_absolute_error'] == 0 for row in report['comparisons'].values())
    assert report['comparisons']['gradients']['none_pairs'] > 0
    assert trainer.state_dict()['fingerprint'] == before and _fingerprint(trainer.bank) == bank
    assert _fingerprint(_rng_capture()) == rng
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert any(event['stage'] == 'runtime_equivalence_reference' for event in events)
    assert any(event['stage'] == 'runtime_equivalence_candidate' for event in events)


def test_reference_loader_never_executes_unbound_source(tmp_path):
    path = tmp_path / 'reference.py'
    path.write_text("raise AssertionError('untrusted source executed')\n")
    with pytest.raises(ValueError, match='allowlisted'):
        runtime._load_reference_model({'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()},
                                      feature_dim=8, width=8)
    with pytest.raises(ValueError, match='integrity'):
        runtime._load_reference_model({'path': str(path), 'sha256': runtime.REFERENCE_MODEL_SHA256},
                                      feature_dim=8, width=8)


def test_reference_loader_executes_only_the_verified_module_class(tmp_path, monkeypatch):
    path = tmp_path / 'reference.py'
    source = ("from music_critic.critic.model import MultiscaleAspectCritic as Base\n"
              "class MultiscaleAspectCritic(Base):\n    pass\n")
    path.write_text(source)
    checksum = sha256(source.encode()).hexdigest()
    monkeypatch.setattr(runtime, 'REFERENCE_MODEL_SHA256', checksum)
    reference = runtime._load_reference_model({'path': str(path), 'sha256': checksum}, feature_dim=8, width=8)
    assert reference.feature_dim == reference.width == 8
    assert type(reference).__module__ == 'music_critic.critic._source004_runtime_reference'


@pytest.mark.parametrize('defect', ['availability', 'context', 'local', 'none_gradient', 'optimizer'])
def test_comparator_rejects_output_gradient_and_step_changes_without_mutating_trainer(monkeypatch, defect):
    trainer = subject()
    reference = copy.deepcopy(trainer.model)
    if defect in {'availability', 'context', 'local'}:
        forward = reference.forward
        def changed(pyramid):
            output = forward(pyramid)
            samples = []
            for sample in output.samples:
                if defect == 'availability':
                    mask = sample.availability.clone()
                    mask[0] = ~mask[0]
                    sample = replace(sample, availability=mask)
                elif defect == 'context':
                    context = dict(sample.contextual)
                    context['note'] = context['note'] + .01
                    sample = replace(sample, contextual=context)
                else:
                    local = dict(sample.local)
                    key = next(iter(local))
                    local[key] = replace(local[key], logits=local[key].logits + .01)
                    sample = replace(sample, local=local)
                samples.append(sample)
            return replace(output, samples=tuple(samples))
        monkeypatch.setattr(reference, 'forward', changed)
    elif defect == 'none_gradient':
        forward = reference.forward
        parameter = reference.local_heads['harmony'][0].weight
        def changed(pyramid):
            output = forward(pyramid)
            return replace(output, samples=tuple(replace(sample, raw_scores=sample.raw_scores + parameter.sum() * 0)
                                                  for sample in output.samples))
        monkeypatch.setattr(reference, 'forward', changed)
    else:
        # A post-backward hook changes the copied optimizer state after the
        # branch step while leaving the source trainer completely independent.
        optimizer_step = torch.optim.AdamW.step
        calls = []
        def changed_step(optimizer, *args, **kwargs):
            result = optimizer_step(optimizer, *args, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                next(iter(optimizer.state.values()))['exp_avg'].add_(.01)
            return result
        monkeypatch.setattr(torch.optim.AdamW, 'step', changed_step)
    monkeypatch.setattr(runtime, '_load_reference_model', lambda *_a, **_kw: reference)
    before, rng = trainer.state_dict()['fingerprint'], _fingerprint(_rng_capture())
    with pytest.raises(ValueError, match='runtime equivalence'):
        runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic'})
    assert trainer.state_dict()['fingerprint'] == before and _fingerprint(_rng_capture()) == rng


def shifted_bias_reference(monkeypatch, trainer, *, delta=.001, defect=None):
    reference = copy.deepcopy(trainer.model)
    monkeypatch.setattr(runtime, '_load_reference_model', lambda *_a, **_kw: reference)
    original = torch.optim.AdamW.step
    calls = []
    def step(optimizer, *args, **kwargs):
        value = original(optimizer, *args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            with torch.no_grad():
                for readout in reference.readouts.values():
                    readout.level_gate.bias.add_(delta)
                if defect == 'other_weight':
                    reference.note_mix.bias.add_(.01)
                elif defect == 'optimizer':
                    bias = reference.readouts['melody'].level_gate.bias
                    optimizer.state[bias]['exp_avg'].add_(.01)
        return value
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    return reference


def test_explicit_gauge_policy_preserves_raw_failure_and_checks_poststep_outputs(monkeypatch):
    trainer = subject()
    shifted_bias_reference(monkeypatch, trainer)
    report = runtime.compare_reference_runtime(trainer,
        {'path': 'synthetic', 'sha256': 'synthetic', 'comparison_policy': 'softmax_shift_gauge_v1'})
    assert report['passed'] and report['raw_state_parity'] is False
    assert report['functional_closeness_modulo_softmax_shifts'] is True
    assert report['forward_views'] == 8
    assert len(report['gauge_parameters']) == 4
    assert report['comparisons']['model_state']['max_absolute_error'] > .0009
    assert report['comparisons']['post_step_outputs']['passed']
    assert report['comparisons']['optimizer_state']['passed']
    for row in report['gauge_parameters'].values():
        assert row['admissible']
        assert row['reference']['gradient'] is not None
        assert 'exp_avg' in row['reference']['optimizer']


def test_default_strict_policy_keeps_bias_shift_failure_with_partial_report(monkeypatch):
    trainer = subject()
    shifted_bias_reference(monkeypatch, trainer)
    events = []
    with pytest.raises(runtime.RuntimeEquivalenceError) as failure:
        runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic'},
                                          progress_callback=events.append)
    report = failure.value.report
    assert not report['passed'] and not report['raw_state_parity']
    assert report['optimizer_applications'] == 2 and report['forward_views'] == 8
    assert report['comparisons']['post_step_outputs']['passed']
    assert set(report['timings']) == {'reference', 'candidate'}
    assert events[-1]['state'] == 'failed'
    assert 'gauge_parameters' not in events[-1]['comparison_summary']


@pytest.mark.parametrize('defect', ['other_weight', 'optimizer', 'oversized_shift', 'post_output'])
def test_gauge_policy_cannot_hide_other_state_or_output_changes(monkeypatch, defect):
    trainer = subject()
    reference = shifted_bias_reference(monkeypatch, trainer,
        delta=.1 if defect == 'oversized_shift' else .001, defect=defect)
    if defect == 'post_output':
        original = reference.forward
        calls = []
        def forward(view):
            output = original(view)
            calls.append(True)
            if len(calls) > 2:
                return replace(output, samples=tuple(replace(sample, raw_scores=sample.raw_scores + .01)
                                                     for sample in output.samples))
            return output
        monkeypatch.setattr(reference, 'forward', forward)
    with pytest.raises(runtime.RuntimeEquivalenceError) as failure:
        runtime.compare_reference_runtime(trainer,
            {'path': 'synthetic', 'sha256': 'synthetic', 'comparison_policy': 'softmax_shift_gauge_v1'})
    assert not failure.value.report['passed']
    assert failure.value.report['forward_views'] == 8


def test_malformed_reference_policy_rejected_before_source_access(tmp_path):
    path = str(tmp_path / 'missing.py')
    for value in ('unknown', None, 1, True):
        with pytest.raises(ValueError, match='comparison_policy'):
            runtime.validate_reference_model_config({'path': path, 'sha256': runtime.REFERENCE_MODEL_SHA256,
                                                     'comparison_policy': value})


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_real_source004_fixture_comparison_on_synthetic_pair(monkeypatch, dtype):
    from .row_reference import Source004Critic
    trainer = subject(learning_rate=3e-4)
    trainer.model.to(dtype=dtype)
    # Bind the fixture to the independently frozen source004 sample method.
    assert sha256(inspect.getsource(Source004Critic._sample).encode()).hexdigest() == (
        'b4111e3da82bd0c3dadbcee56ca472189251ceb3560306fc6ce74e6ad39f76f8')
    monkeypatch.setattr(runtime, '_load_reference_model',
                        lambda *_a, **kw: Source004Critic(**kw))
    report = runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic',
                                                        'comparison_policy': 'softmax_shift_gauge_v1'})
    assert report['passed']
    assert report['comparisons']['outputs']['passed']
    assert report['comparisons']['post_step_outputs']['passed']
    assert report['comparisons']['optimizer_state']['passed']


def test_failure_callback_cannot_replace_compute_failure_or_change_caller_rng(monkeypatch):
    trainer = subject()
    failure = ValueError('synthetic reference construction failure')
    def fail(*_args, **_kwargs):
        raise failure
    monkeypatch.setattr(runtime, '_load_reference_model', fail)
    def callback(row):
        if row['state'] == 'failed':
            torch.rand(1)
            raise OSError('synthetic callback failure')
    rng = _fingerprint(_rng_capture())
    with pytest.raises(runtime.RuntimeEquivalenceError) as caught:
        runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic'},
                                          progress_callback=callback)
    assert caught.value.__cause__ is failure
    assert caught.value.report['error'] == str(failure)
    assert _fingerprint(_rng_capture()) == rng


@pytest.mark.parametrize('defect', ['policy', 'sha', 'bytes', 'symlink'])
def test_cli_reference_admission_fails_before_payload_or_model(tmp_path, monkeypatch, defect):
    from scripts import run_critic_campaign as cli
    from .test_campaign_cli import authored_config
    from .test_campaign import write_fixture_record
    config = authored_config(tmp_path)
    source = tmp_path / 'reference.py'
    source.write_text("raise AssertionError('must not execute')\n")
    ref = {'path': str(source), 'sha256': runtime.REFERENCE_MODEL_SHA256}
    if defect == 'policy':
        ref['comparison_policy'] = 'unknown'
    elif defect == 'sha':
        ref['sha256'] = '0' * 64
    elif defect == 'symlink':
        linked = tmp_path / 'linked.py'
        linked.symlink_to(source)
        ref['path'] = str(linked)
    config['probe']['reference_model'] = ref
    write_fixture_record(config, 'probe')
    monkeypatch.setattr(cli, '_inspect_dataset', lambda *_a: pytest.fail('dataset inspection before reference admission'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_kw: pytest.fail('model access'))
    with pytest.raises(ValueError):
        cli.execute(config, mode='probe')
    assert not (tmp_path / 'out').exists()


def test_probe_comparison_precedes_updates_and_adds_counts_without_changing_old_field(monkeypatch):
    from music_critic.critic.campaign import probe_runtime
    from .test_campaign import trainer, pair
    monkeypatch.setattr(runtime, '_read_reference_source', lambda *_: None)
    calls = []
    def comparison(subject, config, **options):
        assert subject.attempted_updates == subject.applied_updates == subject.draws == 0
        assert config['sha256'] == runtime.REFERENCE_MODEL_SHA256
        calls.append(True)
        return {'passed': True, 'forward_views': 8, 'paired_draws': 2, 'optimizer_applications': 2}
    monkeypatch.setattr(runtime, 'compare_reference_runtime', comparison)
    report = probe_runtime(trainer(), trainer, (pair('dev', 'head_dev'),), updates=2,
        reference_model={'path': '/synthetic/reference.py', 'sha256': runtime.REFERENCE_MODEL_SHA256})
    assert len(calls) == 1
    assert report['applied_updates_including_resume_replay'] == 3
    assert report['total_optimizer_applications'] == 5
    assert report['total_optimizer_pair_draws'] == 5
    assert report['runtime_equivalence']['forward_views'] == 8


def test_cli_retains_comparison_failure_report_and_stops_before_training(tmp_path, monkeypatch):
    import json
    from .test_probe_progress import synthetic_cli
    from .test_campaign import write_fixture_record
    cli, config = synthetic_cli(tmp_path, monkeypatch)
    config['probe']['reference_model'] = {'path': str(tmp_path / 'reference.py'),
                                        'sha256': runtime.REFERENCE_MODEL_SHA256,
                                        'comparison_policy': 'softmax_shift_gauge_v1'}
    write_fixture_record(config, 'probe')
    monkeypatch.setattr(cli, '_read_reference_source', lambda *_: None)
    monkeypatch.setattr(runtime, '_read_reference_source', lambda *_: None)
    comparison_report = {'passed': False, 'raw_state_parity': False, 'comparisons': {'synthetic': .01}}
    def failure(subject, reference, **_options):
        assert subject.attempted_updates == 0
        assert reference == config['probe']['reference_model']
        raise runtime.RuntimeEquivalenceError('synthetic native comparison failure', comparison_report)
    monkeypatch.setattr(runtime, 'compare_reference_runtime', failure)
    with pytest.raises(runtime.RuntimeEquivalenceError):
        cli.execute(config, mode='probe')
    terminal = json.loads((tmp_path / 'out' / 'terminal-probe.json').read_text())
    assert terminal['status'] == 'failed' and terminal['runtime_equivalence'] == comparison_report
    assert not (tmp_path / 'out' / 'probe-checkpoint.pt').exists()


def test_identical_pair_cancellation_is_not_exempt_outside_four_gauge_coordinates(monkeypatch):
    trainer = subject(identical=True, learning_rate=3e-4)
    # Keep the rejection guard independent of the rejected row-batching runtime:
    # an explicit non-gauge perturbation must fail even for an identical pair.
    shifted_bias_reference(monkeypatch, trainer, delta=0., defect='other_weight')
    with pytest.raises(runtime.RuntimeEquivalenceError) as caught:
        runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic',
                                                   'comparison_policy': 'softmax_shift_gauge_v1'})
    assert not caught.value.report['comparisons']['model_state_without_gauge']['passed']
    assert not caught.value.report['comparisons']['post_step_outputs']['passed']


def test_gauge_gradient_bound_is_checked_independently_of_pairwise_gradient_tolerance(monkeypatch):
    trainer = subject()
    for key, parameter in trainer.model.named_parameters():
        if key in runtime.GAUGE_PARAMETERS:
            parameter.register_hook(lambda grad: grad + 1e-4)
    reference = copy.deepcopy(trainer.model)
    # Module deepcopy intentionally does not preserve parameter backward hooks.
    for key, parameter in reference.named_parameters():
        if key in runtime.GAUGE_PARAMETERS:
            parameter.register_hook(lambda grad: grad + 1e-4)
    monkeypatch.setattr(runtime, '_load_reference_model', lambda *_a, **_kw: reference)
    with pytest.raises(runtime.RuntimeEquivalenceError) as caught:
        runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic',
                                                   'comparison_policy': 'softmax_shift_gauge_v1'})
    assert any(not row['admissible'] for row in caught.value.report['gauge_parameters'].values())


def test_frozen_encoder_existing_optimizer_state_and_single_aspect_targets_stay_unchanged(monkeypatch):
    trainer = subject(learning_rate=3e-4, pair_mask=(True, False, False, False))
    trainer.step()
    encoder = torch.nn.Linear(3, 2).requires_grad_(False).eval()
    trainer.frozen_backbones = (encoder,)
    reference = copy.deepcopy(trainer.model)
    monkeypatch.setattr(runtime, '_load_reference_model', lambda *_a, **_kw: reference)
    before, encoder_before, rng = (trainer.state_dict()['fingerprint'], _fingerprint(encoder.state_dict()),
                                   _fingerprint(_rng_capture()))
    report = runtime.compare_reference_runtime(trainer, {'path': 'synthetic', 'sha256': 'synthetic'})
    assert report['passed'] and report['comparisons']['optimizer_state']['max_absolute_error'] == 0
    assert trainer.state_dict()['fingerprint'] == before and _fingerprint(_rng_capture()) == rng
    assert _fingerprint(encoder.state_dict()) == encoder_before
    assert not encoder.training and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
