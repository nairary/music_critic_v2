"""Campaign durability on tiny cached tensors; no corpus or cloud access."""
from dataclasses import replace
from fractions import Fraction
import json

import pytest
import torch

from music_critic.critic.campaign import CampaignConfig, run_campaign
from music_critic.critic.contracts import (
    CriticFeaturePyramid, CriticOutput, FeatureBinding, FeatureSample,
    SampleGeometry, SamplePrediction,
)
from music_critic.critic.loss import CriticLoss
from music_critic.critic.targets import PairTargets
from music_critic.critic.training import CachedPair, CriticTrainConfig, CriticTrainer


def write_fixture_record(config, mode='probe', status='planned'):
    from pathlib import Path
    from music_critic.experiments.registry import seal_record, DATA_ACCESS_FIELDS, ACCESS_POLICY_FIELDS
    from music_critic.critic.campaign import _fingerprint
    record = {'schema_version': '1.0.0', 'experiment_id': config['experiment_id'], 'status': status,
        **dict.fromkeys(('kind', 'phase', 'title', 'question', 'hypothesis', 'decision_use'), 'synthetic fixture'),
        'repository': {}, 'seeds': [17], 'budget': {}, 'results': {}, 'validity': {},
        'claims': {}, 'import_provenance': {}, 'artifacts': [],
        'data_access': {**dict.fromkeys(DATA_ACCESS_FIELDS),
            'access_evidence_status': 'not_applicable_planning_record',
            'declared_policy': {**dict.fromkeys(ACCESS_POLICY_FIELDS, False), 'source': 'synthetic fixture'}},
        'protocol': {'cli_mode': mode, 'cli_config_fingerprint': _fingerprint(config)}}
    Path(config['experiment_record']).write_text(json.dumps(seal_record(record)))


class FixtureCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, pyramid):
        return CriticOutput(tuple(SamplePrediction(
            s.sample_id, self.weight * s.embeddings['song'][0],
            torch.ones(4, dtype=torch.bool), {}, s.geometry.duration,
        ) for s in pyramid.samples))


def pair(prefix, split='train'):
    def features(name, value):
        sample = FeatureSample(SampleGeometry(name, (), (), {}, Fraction(4)),
                               {'song': torch.full((1, 4), float(value))},
                               name + '-raw', {'song': 'synthetic'})
        return CriticFeaturePyramid((sample,), FeatureBinding('fixture', 'fixture', 'fixture'))
    left, right = prefix + '-a', prefix + '-b'
    targets = PairTargets(prefix, (left, right), (True,)*4, (1,)*4, ({}, {}), {})
    return CachedPair(prefix, prefix, features(left, 2), features(right, -1), targets, split=split)


def trainer():
    return CriticTrainer(FixtureCritic(), CriticLoss(), (pair('train'),),
                         CriticTrainConfig(17, .02, 1, 4, 6, 1.0),
                         bindings={'source_id': 'fixture', 'split_id': 'fixture', 'target_version': 'v1'})


def test_campaign_selects_dev_and_retains_immutable_checkpoints(tmp_path):
    model_trainer = trainer()
    report = run_campaign(model_trainer, (pair('dev', 'head_dev'),),
                          CampaignConfig('EXP-SYNTHETIC', 60, 2, 2), tmp_path / 'run')
    assert report['status'] == 'completed'
    assert report['applied_updates'] == 4
    assert report['best_update'] == 4
    root = tmp_path / 'run'
    events = [json.loads(row) for row in (root / 'events.jsonl').read_text().splitlines()]
    assert [e['payload']['update'] for e in events if e['kind'] == 'evaluation'] == [0, 2, 4]
    latest = json.loads((root / 'latest.json').read_text())
    assert (root / latest['path']).is_file()
    assert len(list((root / 'checkpoints').glob('*.pt'))) >= 3
    with pytest.raises(ValueError, match='terminal'):
        run_campaign(trainer(), (pair('dev', 'head_dev'),),
                     CampaignConfig('EXP-SYNTHETIC', 60, 2, 2), root, resume=True)


def test_campaign_observes_dev_before_selection_gate_without_selecting(tmp_path):
    model_trainer = trainer()
    root = tmp_path / 'run'
    report = run_campaign(model_trainer, (pair('dev', 'head_dev'),),
                          CampaignConfig('EXP-SYNTHETIC-GATED', 60, 1, 1,
                                         selection_start_draw=3), root)
    events = [json.loads(row) for row in (root / 'events.jsonl').read_text().splitlines()]
    evaluations = [row['payload'] for row in events if row['kind'] == 'evaluation']
    assert [row['update'] for row in evaluations] == [0, 1, 2, 3, 4]
    assert all(not row['selection_eligible'] and not row['selected_best']
               for row in evaluations if row['draw'] < 3)
    assert all(row['selection_eligible'] for row in evaluations if row['draw'] >= 3)
    assert report['best_update'] >= 3


@pytest.mark.parametrize('start', [-1, True, 1.5])
def test_campaign_selection_draw_requires_nonnegative_integer(start):
    with pytest.raises(ValueError, match='selection_start_draw'):
        CampaignConfig('EXP-SYNTHETIC-GATED', 60, 1, 1, selection_start_draw=start)


def test_curriculum_completes_exposure_budget_and_evaluates_after_final_skip(tmp_path):
    from .test_curriculum_training import make_bank, make_trainer
    model_trainer = make_trainer()
    original_step = model_trainer.step
    def step():
        model_trainer.model.supported = model_trainer.attempted_updates != 9
        try:
            return original_step()
        finally:
            model_trainer.model.supported = True
    model_trainer.step = step
    dev = tuple(p for p in make_bank(split='head_dev') if p.edit_count == 1)
    root = tmp_path / 'run'
    report = run_campaign(model_trainer, dev,
        CampaignConfig('EXP-SYNTHETIC-CURRICULUM', 60, 1, 1, selection_start_draw=14), root)
    assert report['status'] == 'completed' and report['reason'] == 'exposure_cap'
    assert report['draws'] == 20 and report['attempted_updates'] == 10
    assert report['applied_updates'] == 9 and report['skipped_updates'] == 1
    assert report['last_evaluation_draw'] == 20
    events = [json.loads(row) for row in (root / 'events.jsonl').read_text().splitlines()]
    evaluations = [e['payload'] for e in events if e['kind'] == 'evaluation']
    assert [(e['draw'], e['update']) for e in evaluations[-2:]] == [(18, 9), (20, 9)]


@pytest.mark.parametrize('curriculum', [False, True])
def test_unreachable_selection_threshold_rejected_before_run_created(tmp_path, curriculum):
    from .test_curriculum_training import make_bank, make_trainer
    model_trainer = make_trainer() if curriculum else trainer()
    dev = (tuple(p for p in make_bank(split='head_dev') if p.edit_count == 1)
           if curriculum else (pair('dev', 'head_dev'),))
    root = tmp_path / 'run'
    with pytest.raises(ValueError, match='selection_start_draw exceeds'):
        run_campaign(model_trainer, dev,
                     CampaignConfig('EXP-SYNTHETIC-GATED', 60, 1, 1,
                                    selection_start_draw=21 if curriculum else 5), root)
    assert not root.exists()


def test_resume_preserves_uncommitted_history_and_exact_training(tmp_path):
    class PowerLoss(BaseException):
        pass
    interrupted = trainer()
    original_step = interrupted.step
    def step():
        if interrupted.attempted_updates == 2:
            raise PowerLoss()
        return original_step()
    interrupted.step = step
    config = CampaignConfig('EXP-SYNTHETIC', 60, 10, 10)
    root = tmp_path / 'interrupted'
    with pytest.raises(PowerLoss):
        run_campaign(interrupted, (pair('dev', 'head_dev'),), config, root)
    retained = (root / 'events.jsonl').read_bytes()
    resumed = trainer()
    report = run_campaign(resumed, (pair('dev', 'head_dev'),), config, root, resume=True)
    assert (root / 'events.jsonl').read_bytes().startswith(retained)
    assert report['replayed_logged_updates'] == 2
    assert report['unlogged_work_possible']
    control = trainer()
    for _ in range(4):
        control.step()
    torch.testing.assert_close(resumed.model.weight, control.model.weight, rtol=0, atol=0)


def test_cooperative_wall_cap_writes_terminal_abort(tmp_path):
    import itertools
    ticks = itertools.count()
    report = run_campaign(trainer(), (pair('dev', 'head_dev'),),
                          CampaignConfig('EXP-SYNTHETIC', 2, 2, 2), tmp_path / 'run',
                          clock=lambda: float(next(ticks)))
    assert report['status'] == 'aborted' and report['reason'] == 'wall_cap'
    assert report['applied_updates'] == 0
    assert json.loads((tmp_path / 'run/summary.json').read_text())['status'] == 'aborted'


def test_runtime_probe_checks_continuation_and_full_bank_scope():
    from music_critic.critic.campaign import probe_runtime
    report = probe_runtime(trainer(), trainer, (pair('dev', 'head_dev'),), updates=2)
    assert report['resume']['exact_state']
    assert report['resume']['same_sample_draw']
    assert report['critic_single_vs_batch']['passed']
    assert report['train_bank_pairs'] == 1 and report['head_dev_pairs'] == 1
    assert report['applied_updates_including_resume_replay'] == 3
    assert report['full_bank_guard_seconds'] >= 0


def test_encoder_probe_tolerance_is_separate_from_critic_and_resume(monkeypatch):
    from music_critic.critic.campaign import probe_runtime
    source = pair('encoder')
    samples = {p.samples[0].sample_id: p.samples[0] for p in (source.left, source.right)}
    class Adapter:
        model = torch.nn.Linear(1, 1).requires_grad_(False)
        def extract(self, pieces):
            selected = tuple(samples[p] for p in pieces)
            if len(pieces) == 2:
                selected = tuple(replace(s, embeddings={k: v + 5e-5 for k, v in s.embeddings.items()})
                                 for s in selected)
            return CriticFeaturePyramid(selected, source.left.binding)
        def _check_frozen(self):
            assert all(not p.requires_grad for p in self.model.parameters())
    options = dict(updates=2, adapter=Adapter(), raw_batch=tuple(samples))
    with pytest.raises(AssertionError):
        probe_runtime(trainer(), trainer, (pair('dev', 'head_dev'),), **options)
    tolerance = {'rtol': 1e-4, 'atol': 1e-4}
    report = probe_runtime(trainer(), trainer, (pair('dev', 'head_dev'),),
                           encoder_batch_tolerance=tolerance, **options)
    assert report['encoder_single_vs_batch']['rtol'] == 1e-4
    assert report['encoder_single_vs_batch']['atol'] == 1e-4
    assert report['critic_single_vs_batch']['rtol'] == 1e-5
    assert report['critic_single_vs_batch']['atol'] == 1e-5
    assert report['resume']['rtol'] == 1e-5 and report['resume']['atol'] == 1e-6
    broken = trainer()
    original_forward = broken.model.forward
    def batch_shift(pyramid):
        output = original_forward(pyramid)
        return (replace(output, samples=tuple(replace(s, raw_scores=s.raw_scores + 5e-5)
                                               for s in output.samples))
                if len(pyramid.samples) == 2 else output)
    monkeypatch.setattr(broken.model, 'forward', batch_shift)
    with pytest.raises(AssertionError):
        probe_runtime(broken, trainer, (pair('dev', 'head_dev'),),
                      encoder_batch_tolerance=tolerance, **options)


def test_campaign_preserves_last_good_checkpoint_when_failure_blocks_new_save(tmp_path):
    broken = trainer()
    def corrupt():
        broken.bank[0].left.samples[0].embeddings['song'].add_(1)
        raise ValueError('bad immutable bank')
    broken.step = corrupt
    report = run_campaign(broken, (pair('dev', 'head_dev'),),
                          CampaignConfig('EXP-SYNTHETIC', 60, 2, 2), tmp_path / 'run')
    assert report['status'] == 'failed'
    assert report['latest']['update'] == 0
    assert 'bank_mutated' in report['final_checkpoint_error']
    assert (tmp_path / 'run/summary.json').exists()


def test_campaign_declared_aspects_must_all_be_observed_when_required(tmp_path):
    dev = pair('dev', 'head_dev')
    dev = replace(dev, targets=replace(dev.targets, pair_mask=(True, False, False, False),
                                      preference=(1, 0, 0, 0)))
    report = run_campaign(trainer(), (dev,),
        CampaignConfig('EXP-SYNTHETIC', 60, 2, 2, require_all_declared_aspects=True), tmp_path / 'run')
    assert report['status'] == 'failed' and 'aspect support' in report['error']
    assert report['applied_updates'] == 0 and report['best'] is None


def test_cli_readiness_never_loads_raw_or_checkpoint(tmp_path, monkeypatch):
    from scripts import run_critic_campaign as cli
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'schema': 'CriticCampaignSources@1.0.0', 'sources': [{
        'source_id': 'fixture', 'source_split': 'train',
        'piece': {'path': str(tmp_path / 'must-not-exist.json')}}]}))
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'not a checkpoint; readiness only hashes it')
    from music_critic.critic.campaign import _file_digest
    config = {'schema': 'CriticCampaignCLI@1.0.0', 'experiment_id': 'EXP-FIXTURE',
        'experiment_record': str(tmp_path / 'record.json'), 'output_root': str(tmp_path / 'out'),
        'dataset': {'sources_manifest': str(manifest), 'sources_manifest_sha256': _file_digest(manifest)['sha256']},
        'backbone': {'checkpoint_path': str(checkpoint), **_file_digest(checkpoint)},
        'training': dict(seed=17,learning_rate=.02,batch_size=1,max_updates=2,max_attempts=3,clip_grad_norm=1.),
        'campaign': dict(wall_seconds=60,evaluate_every_updates=1,checkpoint_every_attempts=1),
        'preparation_wall_seconds': 60, 'device': 'cpu', 'cpu_threads': 1,
        'model': {'feature_dim': 128, 'width': 128}, 'lambda_local': .25}
    write_fixture_record(config, 'prepare')
    monkeypatch.setattr(cli, 'load_campaign_dataset', lambda *_: pytest.fail('raw accessed'))
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_k: pytest.fail('model loaded'))
    report = cli.execute(config, mode='readiness')
    assert report['raw_piece_payloads_accessed'] is False
    assert not (tmp_path / 'out').exists()
    with pytest.raises(ValueError, match='config or mode binding'):
        cli.execute({**config, 'training': {**config['training'], 'max_updates': 3}}, mode='prepare')
    with pytest.raises(ValueError, match='config or mode binding'):
        cli.execute(config, mode='probe')
    (tmp_path / 'record.json').write_text(json.dumps({'experiment_id': 'EXP-FIXTURE', 'status': 'completed'}))
    with pytest.raises(ValueError, match='planned or running'):
        cli.execute(config, mode='prepare')


def test_feature_preparation_deduplicates_original_and_reuses_verified_receipts(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from music_critic.data import load_piece
    from music_critic.critic.campaign import prepare_feature_bank
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
    from music_critic.critic.targets import InterventionTargetBuilder, VerifiedPatch
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    from music_critic.experiments.music_interventions import Intervention, NotePatch, apply_interventions
    original = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    raw_pairs = []
    for i, pitch in enumerate((68, 69)):
        patch = NotePatch('note:melody-000', pitch=pitch)
        variant = apply_interventions(original, (Intervention(f'fixture-{i}', (patch,)),)).piece
        target = InterventionTargetBuilder().build(original, variant, VerifiedPatch(
            original.piece_id, variant.piece_id, 'work', (patch,), ('harmony',), 'pitch'))
        raw_pairs.append(SimpleNamespace(pair_id=f'pair-{i}', work_id='work', original=original,
                                         variant=variant, targets=target, split='train', component_id='component',
                                         edit_count=i + 1, genre='synthetic-genre'))
    dataset = SimpleNamespace(pairs=raw_pairs, fingerprint='synthetic-dataset')
    with torch.random.fork_rng():
        torch.manual_seed(17)
        adapter = BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
                                         FeatureBinding('fixture', 'fixture', C2_ADAPTER_VERSION))
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        bank, first = prepare_feature_bank(dataset, adapter, tmp_path / 'prepared', wall_seconds=60)
        repeated, second = prepare_feature_bank(dataset, adapter, tmp_path / 'prepared', wall_seconds=60)
    finally:
        torch.set_num_threads(prior_threads)
    assert first['extracted_views'] == 3 and second['extracted_views'] == 0
    assert second['reused_views'] == 3
    assert bank[0].left.samples[0] is bank[1].left.samples[0]
    assert [(p.edit_count, p.genre) for p in bank] == [(1, 'synthetic-genre'), (2, 'synthetic-genre')]
    assert [(p.edit_count, p.genre) for p in repeated] == [(1, 'synthetic-genre'), (2, 'synthetic-genre')]
    torch.testing.assert_close(bank[0].left.samples[0].embeddings['note'],
                               repeated[0].left.samples[0].embeddings['note'])


@pytest.mark.parametrize('encoder_tolerance', [None, {'rtol': 1e-4, 'atol': 1e-4}])
def test_cli_full_fixture_probe_covers_cache_encoder_and_critic(tmp_path, monkeypatch, encoder_tolerance):
    from pathlib import Path
    from types import SimpleNamespace
    from scripts import run_critic_campaign as cli
    from music_critic.critic.campaign import _file_digest
    from music_critic.data import load_piece
    from music_critic.critic.features import BackboneFeatureAdapter, C2_ADAPTER_VERSION
    from music_critic.critic.targets import InterventionTargetBuilder, VerifiedPatch
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    from music_critic.experiments.music_interventions import Intervention, NotePatch, apply_interventions
    original = load_piece(Path(__file__).parents[1] / 'fixtures/data/canonical_piece_v2.json')
    rows = []
    for i, split in enumerate(('train', 'head_dev')):
        piece = apply_interventions(original, (Intervention('fixture-source-' + split, tuple(
            NotePatch(note.note_id, pitch=note.pitch + 2 * (i + 1)) for note in original.notes)),)).piece
        patch = NotePatch('note:melody-000', pitch=piece.notes[0].pitch + 5)
        variant = apply_interventions(piece, (Intervention('fixture-pitch', (patch,)),)).piece
        target = InterventionTargetBuilder().build(piece, variant, VerifiedPatch(
            piece.piece_id, variant.piece_id, split, (patch,), ('harmony',), 'pitch'))
        rows.append(SimpleNamespace(pair_id=split, work_id=split, original=piece, variant=variant,
                                    targets=target, split=split, component_id=split))
    dataset = SimpleNamespace(pairs=tuple(rows), fingerprint='fixture-bank', source_summary={},
                              split_assignments={'train': 'train', 'head_dev': 'head_dev'}, omissions=())
    with torch.random.fork_rng():
        torch.manual_seed(17)
        adapter = BackboneFeatureAdapter(CorrectedAnalysisGNNModel(),
                                         FeatureBinding('fixture', 'fixture', C2_ADAPTER_VERSION))
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture injected adapter')
    record = tmp_path / 'record.json'
    config = {'schema': 'CriticCampaignCLI@1.0.0', 'experiment_id': 'EXP-FIXTURE',
        'experiment_record': str(record), 'output_root': str(tmp_path / 'out'),
        'feature_cache_root': str(tmp_path / 'shared-cache'),
        'dataset': {}, 'backbone': {'checkpoint_path': str(checkpoint), **_file_digest(checkpoint)},
        'training': dict(seed=17,learning_rate=.001,batch_size=1,max_updates=2,max_attempts=4,
                         clip_grad_norm=1., selection_aspects=['harmony']),
        'campaign': dict(wall_seconds=60,evaluate_every_updates=1,checkpoint_every_attempts=1),
        'preparation_wall_seconds': 60, 'device': 'cpu', 'cpu_threads': 1, 'probe': {'updates': 2},
        'model': {'feature_dim': 128, 'width': 128}, 'lambda_local': .25}
    if encoder_tolerance is not None:
        config['probe']['encoder_batch_tolerance'] = encoder_tolerance
        config['deterministic_algorithms'] = True
    write_fixture_record(config)
    monkeypatch.setattr(cli, 'inspect_manifest', lambda *_: {'fixture': True})
    monkeypatch.setattr(cli, 'load_campaign_dataset', lambda *_: dataset)
    monkeypatch.setattr(cli, 'load_c3_relationgate_adapter', lambda *_a, **_k: adapter)
    report = cli.execute(config, mode='probe')
    assert report['status'] == 'completed'
    assert report['preparation']['unique_views'] == 4
    assert report['result']['encoder_single_vs_batch']['passed']
    expected_tolerance = encoder_tolerance or {'rtol': 1e-5, 'atol': 1e-5}
    assert all(report['result']['encoder_single_vs_batch'][key] == value
               for key, value in expected_tolerance.items())
    assert report['result']['critic_single_vs_batch']['atol'] == 1e-5
    assert report['result']['resume']['atol'] == 1e-6
    assert report['result']['resume']['passed'] and report['result']['encoder_no_grad']
    assert report['full_bank_memory']['feature_tensor_bytes'] > 0
    assert (tmp_path / 'out/probe-checkpoint.pt').exists()
    with pytest.raises(ValueError, match='terminal'):
        cli.execute(config, mode='probe')
    monkeypatch.setattr(adapter, 'extract', lambda *_: pytest.fail('shared cache did a new encoder forward'))
    other_config = {**config, 'output_root': str(tmp_path / 'other-attempt'),
                    'experiment_record': str(tmp_path / 'other-record.json')}
    write_fixture_record(other_config, 'prepare')
    other = cli.execute(other_config, mode='prepare')
    assert other['preparation']['extracted_views'] == 0 and other['preparation']['reused_views'] == 4
