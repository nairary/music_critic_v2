"""Synthetic whole-song launch/resume and fixed-query integration."""
from dataclasses import asdict, replace
from hashlib import sha256
import json

import pytest
import torch

from music_critic.data import dumps_piece
from music_critic.experiments.registry import canonical_json_bytes, load_record, seal_record
from music_critic.pll.cli import load_run_config, run_bindings, train
from music_critic.pll.contracts import MaskQuery
from music_critic.pll.runtime import PLLTrainer, TrainingConfig
from music_critic.pll.scoring import score_piece
from music_critic.pll.whole_data import WholeDataManifest, WholeSourceCandidate, prepare_whole_sources
from music_critic.pll.whole_model import WholePieceHCM
from tests.pll.test_cli import prepare
from tests.pll.whole_fixtures import full_piece, small_codec, small_config, small_whole, whole_example


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def whole_run(tmp_path, *, attempt='PLL-SYNTHETIC-001', total=2, resume=None, updates=1):
    cfg, _, record, output = prepare(tmp_path, attempt=attempt, total=total, resume=resume)
    manifest = tmp_path / 'whole-data/manifest.json'
    if not manifest.exists():
        p = full_piece()
        raw = dumps_piece(p).encode()
        (tmp_path / 'complete.json').write_bytes(raw)
        candidate = WholeSourceCandidate(record_id='whole', source='synthetic', work_id='fixture',
            piece_id=p.piece_id, source_group_id=p.source_group_id, split='train', native_split='train',
            path='complete.json', sha256=sha256(raw).hexdigest(), size_bytes=len(raw),
            coverage='full', coverage_evidence='Synthetic complete geometry')
        prepare_whole_sources([candidate], manifest.parent, source_root=tmp_path,
            source_weights={'synthetic':1.}, split_provenance='Synthetic only')
    config = json.loads(cfg.read_text())
    config.update(schema='PLLWholeRunConfig@1', whole_piece=asdict(small_whole()),
                  model=asdict(small_config(.1)), codec=asdict(small_codec()), attempt_updates=updates)
    cfg.write_text(json.dumps(config))
    planned = load_record(record)
    planned['budget']['planned_updates'] = updates
    planned['pll'] = run_bindings(load_run_config(cfg), WholeDataManifest.load(manifest), 'cpu')
    record.write_bytes(canonical_json_bytes(seal_record(planned)) + b'\n')
    return cfg, manifest, record, output


def test_whole_cli_continuation_matches_uninterrupted_update_exactly(tmp_path):
    continuous = tmp_path / 'continuous'; continuous.mkdir()
    cfg, manifest, record, output = whole_run(continuous, updates=2)
    ref = train(cfg, manifest, record, output)
    ref_state = torch.load(output / ref['checkpoint'], weights_only=True)
    split = tmp_path / 'split'; split.mkdir()
    cfg, manifest, record, output = whole_run(split)
    first = train(cfg, manifest, record, output)
    cp = output / first['checkpoint']
    resume = dict(checkpoint=str(cp), sha256=sha256(cp.read_bytes()).hexdigest(),
                  update=1, parent_experiment_id=record.stem)
    cfg, manifest, record, output = whole_run(split, attempt='PLL-SYNTHETIC-002', resume=resume)
    last = train(cfg, manifest, record, output)
    got = torch.load(output / last['checkpoint'], weights_only=True)
    assert last['total_updates'] == last['total_draws'] == 2
    assert got['sampler_state'] == ref_state['sampler_state']
    assert all(torch.equal(got['model'][key], value) for key, value in ref_state['model'].items())
    metric = json.loads((output / 'metrics.jsonl').read_text())['metrics']
    assert metric['context_diagnostics'][0]['source_bars'] == 40
    assert 'bar:32' in metric['context_diagnostics'][0]['motifs']['scale_counts']
    assert load_record(record)['status'] == 'completed'


def test_whole_checkpoint_rejects_changed_bank_before_state_mutation(tmp_path):
    model = WholePieceHCM(small_config(), whole_config=small_whole(), codec_profile=small_codec())
    trainer = PLLTrainer(model, TrainingConfig(total_updates=2, warmup_updates=0), data_id='synthetic', protocol_id='test')
    trainer.step((whole_example(),))
    digest = trainer.save_checkpoint(tmp_path / 'state.pt', sampler_state={})
    other = WholePieceHCM(small_config(), whole_config=small_whole(motif_scales=(1,2,4,8)), codec_profile=small_codec())
    resumed = PLLTrainer(other, trainer.config, data_id='synthetic', protocol_id='test')
    before = {k:v.clone() for k,v in other.state_dict().items()}
    with pytest.raises(ValueError, match='binding_mismatch'):
        resumed.load_checkpoint(tmp_path / 'state.pt', expected_sha256=digest)
    assert all(torch.equal(v, other.state_dict()[k]) for k,v in before.items())


def test_locked_local_profile_rejects_unfreeze_schedule():
    model = WholePieceHCM(small_config(), whole_config=small_whole(), codec_profile=small_codec())
    with pytest.raises(ValueError, match='frozen_local_cannot_unfreeze'):
        PLLTrainer(model, TrainingConfig(total_updates=2, warmup_updates=0, unfreeze_update=1),
                   data_id='synthetic', protocol_id='test')


def test_whole_scoring_uses_complete_observation_and_preserves_unsupported_slot():
    model = WholePieceHCM(small_config(), whole_config=small_whole(), codec_profile=small_codec()).eval()
    p = full_piece()
    masked_query = whole_example().context.query
    query = MaskQuery(masked_query.start, masked_query.end, (p.tracks[0].track_id,))
    result = score_piece(model, full_piece(), (query,))
    assert result['supported'] == 1 and model.last_diagnostics['source_bars'] == 40
    model.whole_config = replace(model.whole_config, max_piece_bars=32)
    result = score_piece(model, full_piece(), (query,))
    assert result['supported'] == 0 and 'bar_limit' in result['blocks'][0]['reason']


@pytest.mark.parametrize('change', [dict(freeze_policy='upper'), dict(training=dict(total_updates=2,warmup_updates=0,unfreeze_update=1))])
def test_whole_config_rejects_locked_unfreeze_before_launch(tmp_path, change):
    cfg,_,_,_ = whole_run(tmp_path)
    payload = json.loads(cfg.read_text()); payload.update(change)
    cfg.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='frozen_local_cannot_unfreeze'):
        load_run_config(cfg)


def rebind(cfg, manifest, record):
    planned = load_record(record)
    planned['pll'] = run_bindings(load_run_config(cfg), WholeDataManifest.load(manifest), 'cpu')
    record.write_bytes(canonical_json_bytes(seal_record(planned)) + b'\n')


def test_registered_whole_warm_start_trains_with_fresh_progress(tmp_path):
    from music_critic.pll.model import HCM1
    cfg, manifest, record, output = whole_run(tmp_path)
    old = HCM1(small_config(.1), codec_profile=small_codec())
    trainer = PLLTrainer(old, TrainingConfig(total_updates=2, warmup_updates=0),
                         data_id='old-synthetic', protocol_id='old-synthetic')
    source = tmp_path / 'old.pt'
    digest = trainer.save_checkpoint(source, sampler_state={})
    payload = json.loads(cfg.read_text())
    payload['warm_start'] = dict(path=str(source), sha256=digest, size_bytes=source.stat().st_size, update=0)
    cfg.write_text(json.dumps(payload)); rebind(cfg, manifest, record)
    receipt = train(cfg, manifest, record, output)
    state = torch.load(output / receipt['checkpoint'], weights_only=True)
    assert state['updates'] == state['draws'] == 1
    assert state['binding']['model']['warm_start']['source_sha256'] == digest
    assert state['binding']['model']['warm_start']['optimizer_restored'] is False
    # Fully frozen inherited backbone remains bit-identical after the new update.
    assert torch.equal(state['model']['encoder.boundary_projection.weight'], old.state_dict()['encoder.boundary_projection.weight'])


@pytest.mark.parametrize('split,field', [('test','test_inputs_accessed'), ('validation','validation_accessed')])
def test_whole_runner_retains_known_forbidden_input_observation(tmp_path, split, field):
    from music_critic.pll.data import seal_manifest
    cfg, manifest, record, output = whole_run(tmp_path)
    payload = json.loads(manifest.read_text())
    row = payload['records'][0]
    file = manifest.parent / row['path']
    raw = file.read_text().replace('"train"', '"'+split+'"').encode()
    file.write_bytes(raw)
    row.update(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    manifest.write_text(json.dumps(seal_manifest(payload)))
    rebind(cfg, manifest, record)
    with pytest.raises(ValueError, match='split_mismatch'):
        train(cfg, manifest, record, output)
    final = load_record(record)
    assert final['status'] == 'invalid'
    assert final['data_access'][field] is True
    assert field in final['data_access']['policy_violations']
    assert final['data_access']['access_evidence_status'] == 'partial_primary_evidence'
    assert json.loads((output/'receipt.json').read_text())['status'] == 'invalid'
