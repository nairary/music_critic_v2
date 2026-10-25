"""Synthetic native-probe observability; no GPU, corpus or real checkpoint."""
from datetime import datetime
from dataclasses import replace
import json

import pytest
import torch

from music_critic.critic.campaign import probe_runtime
from music_critic.critic.contracts import CriticFeaturePyramid
from .test_campaign import pair, trainer, write_fixture_record


class FixtureAdapter:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1).requires_grad_(False)
        source = pair('encoder')
        self.views = {view.samples[0].sample_id: view for view in (source.left, source.right)}
        self.binding = source.left.binding

    def extract(self, pieces):
        views = [self.views[getattr(piece, 'piece_id', piece)] for piece in pieces]
        return CriticFeaturePyramid(tuple(view.samples[0] for view in views), views[0].binding)

    def to(self, _device):
        return self

    def _check_frozen(self):
        assert all(not parameter.requires_grad for parameter in self.model.parameters())


def test_probe_progress_tracks_stages_and_counters_without_changing_computation():
    subject, reference, adapter = trainer(), trainer(), FixtureAdapter()
    options = dict(updates=2, adapter=adapter, raw_batch=tuple(adapter.views), clock=lambda: 0.)
    events = []
    actual = probe_runtime(subject, trainer, (pair('dev', 'head_dev'),),
                           progress_callback=events.append, **options)
    expected = probe_runtime(reference, trainer, (pair('dev', 'head_dev'),), **options)
    assert actual == expected
    assert subject.state_dict()['fingerprint'] == reference.state_dict()['fingerprint']
    assert [(event['stage'], event['state']) for event in events] == [
        (stage, state)
        for stage in ('encoder_parity', 'critic_parity', 'train_step', 'resume_restore',
                      'train_step', 'resume_replay', 'resume_verification', 'dev')
        for state in ('started', 'completed')
    ]
    completed_steps = [event for event in events
                       if event['stage'] == 'train_step' and event['state'] == 'completed']
    assert [(event['attempted_updates'], event['applied_updates'], event['draws'])
            for event in completed_steps] == [(1, 1, 1), (2, 2, 2)]
    assert set(events[0]) == {
        'schema', 'stage', 'state', 'timestamp_utc', 'elapsed_seconds', 'duration_seconds',
        'attempted_updates', 'applied_updates', 'skipped_updates', 'draws',
    }
    for event in events:
        assert datetime.fromisoformat(event['timestamp_utc']).utcoffset().total_seconds() == 0
        assert event['elapsed_seconds'] == 0
        assert event['duration_seconds'] == (0 if event['state'] == 'completed' else None)


@pytest.mark.parametrize('fail_state, updates', [('started', 0), ('completed', 1)])
def test_callback_failure_propagates_and_stops_later_probe_work(fail_state, updates):
    subject, failure = trainer(), OSError('synthetic journal failure')
    def callback(event):
        if event['stage'] == 'train_step' and event['state'] == fail_state:
            raise failure
    with pytest.raises(OSError) as caught:
        probe_runtime(subject, trainer, (pair('dev', 'head_dev'),), updates=2,
                      progress_callback=callback)
    assert caught.value is failure
    assert subject.applied_updates == subject.attempted_updates == updates


def test_probe_failure_is_not_replaced_by_a_completion_callback(monkeypatch):
    subject, failure, events = trainer(), ValueError('synthetic compute failure'), []
    def fail_step():
        raise failure
    def callback(event):
        events.append(event)
        if event['stage'] == 'train_step' and event['state'] == 'completed':
            raise OSError('must not mask the compute error')
    monkeypatch.setattr(subject, 'step', fail_step)
    with pytest.raises(ValueError) as caught:
        probe_runtime(subject, trainer, (pair('dev', 'head_dev'),), updates=2,
                      progress_callback=callback)
    assert caught.value is failure
    assert (events[-1]['stage'], events[-1]['state']) == ('train_step', 'started')


def test_progress_counts_skipped_draws_separately_from_applied_updates(monkeypatch):
    subject, events = trainer(), []
    forward = subject.model.forward
    def unsupported_first_training_attempt(pyramid):
        output = forward(pyramid)
        if subject.model.training and subject.attempted_updates == 1:
            return replace(output, samples=tuple(replace(
                sample, availability=torch.zeros_like(sample.availability)) for sample in output.samples))
        return output
    monkeypatch.setattr(subject.model, 'forward', unsupported_first_training_attempt)
    probe_runtime(subject, trainer, (pair('dev', 'head_dev'),), updates=2,
                  progress_callback=events.append)
    steps = [event for event in events if event['stage'] == 'train_step' and event['state'] == 'completed']
    assert [(event['attempted_updates'], event['applied_updates'], event['skipped_updates'], event['draws'])
            for event in steps] == [(1, 0, 1, 1), (2, 1, 1, 2), (3, 2, 1, 3)]


def synthetic_cli(tmp_path, monkeypatch):
    from scripts import run_critic_campaign as cli
    from .test_campaign_cli import authored_config
    config = authored_config(tmp_path)
    write_fixture_record(config, 'probe')
    dataset = cli._load_dataset(config)
    adapter = FixtureAdapter()
    adapter.views = dict(zip((dataset.pairs[0].original.piece_id, dataset.pairs[0].variant.piece_id),
                             adapter.views.values()))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_kw: adapter)
    monkeypatch.setattr(cli, 'prepare_feature_bank', lambda *_a, **_kw: (
        (pair('train'), pair('dev', 'head_dev')), {'tensor_bytes': 32}))
    monkeypatch.setattr(cli, '_new_trainer', lambda *_a: trainer())
    return cli, config


def test_cli_appends_visible_progress_including_initial_trainer_before_factory(tmp_path, monkeypatch):
    cli, config = synthetic_cli(tmp_path, monkeypatch)
    root = tmp_path / 'out'
    root.mkdir()
    progress = root / 'progress-probe.jsonl'
    retained = b'{"earlier_interrupted_invocation":true}\n'
    progress.write_bytes(retained)
    calls = []
    def factory(*_args):
        rows = [json.loads(line) for line in progress.read_text().splitlines()]
        expected = 'trainer_initialization' if not calls else 'resume_restore'
        assert (rows[-1]['stage'], rows[-1]['state']) == (expected, 'started')
        calls.append(True)
        return trainer()
    monkeypatch.setattr(cli, '_new_trainer', factory)
    report = cli.execute(config, mode='probe')
    assert report['status'] == 'completed' and len(calls) == 2
    assert progress.read_bytes().startswith(retained)
    rows = [json.loads(line) for line in progress.read_text().splitlines()[1:]]
    assert [(row['stage'], row['state']) for row in rows[:2]] == [
        ('trainer_initialization', 'started'), ('trainer_initialization', 'completed')]
    assert [(row['stage'], row['state']) for row in rows[-2:]] == [('dev', 'started'), ('dev', 'completed')]
    assert [row['elapsed_seconds'] for row in rows] == sorted(row['elapsed_seconds'] for row in rows)


def test_cli_reports_progress_writer_failure_without_starting_trainer(tmp_path, monkeypatch):
    cli, config = synthetic_cli(tmp_path, monkeypatch)
    original_json_bytes = cli._json_bytes
    def fail_progress(value):
        if value.get('schema') == 'CriticNativeRuntimeProgress@1.0.0':
            raise OSError('synthetic progress write failure')
        return original_json_bytes(value)
    monkeypatch.setattr(cli, '_json_bytes', fail_progress)
    monkeypatch.setattr(cli, '_new_trainer', lambda *_a: pytest.fail('trainer must not start'))
    with pytest.raises(OSError, match='synthetic progress write failure'):
        cli.execute(config, mode='probe')
    terminal = json.loads((tmp_path / 'out' / 'terminal-probe.json').read_text())
    assert terminal['status'] == 'failed' and terminal['error_type'] == 'OSError'
    assert terminal['error'] == 'synthetic progress write failure'
