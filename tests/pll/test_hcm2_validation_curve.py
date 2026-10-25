import json

import pytest
import torch

from scripts.hcm2_validation_curve import (aggregate, encoder_fingerprint, list_checkpoints,
                                           merge, read_rows, shard)
from tests.pll.test_whole_model import small_model
from tests.pll.whole_fixtures import whole_example


def test_list_checkpoints_orders_dedupes_and_filters(tmp_path):
    parent, cont = tmp_path/'parent', tmp_path/'cont'
    parent.mkdir(); cont.mkdir()
    for update in (100, 200, 1000):
        (parent/f'checkpoint-{update:08d}.pt').write_bytes(b'p')
    for update in (1000, 1100, 2000):
        (cont/f'checkpoint-{update:08d}.pt').write_bytes(b'c')
    groups = [(parent, 100, 1000, parent/'binding.json'),
              (cont, 1000, 10000, cont/'binding.json')]
    entries = list_checkpoints(groups)
    assert [e['update'] for e in entries] == [100, 200, 1000, 1100, 2000]
    # An update present in both groups is taken from the earliest group.
    assert entries[2]['path'].startswith(str(parent))
    assert entries[3]['binding'].startswith(str(cont))


def test_aggregate_is_event_weighted_and_split_specific():
    rows = [
        dict(update=1000, id='a:1', split='validation', nll=20., events=2,
             null_weight=1., eos_first_prob=.5),
        dict(update=1000, id='a:2', split='validation', nll=12., events=6,
             null_weight=.5, eos_first_prob=.25),
        dict(update=1000, id='a:1', split='train', nll=4., events=0,
             null_weight=0., eos_first_prob=1.),
    ]
    points = aggregate(rows)
    assert len(points) == 2
    validation = next(p for p in points if p['split'] == 'validation')
    assert validation['nats_per_event'] == 4.
    assert validation['events'] == 8 and validation['masks'] == 2
    assert validation['works'] == 1
    assert validation['mean_null_weight'] == .75
    train = next(p for p in points if p['split'] == 'train')
    assert train['nats_per_event'] is None and train['events'] == 0


def test_immediate_eos_probability_is_the_first_token_softmax():
    from music_critic.pll.codec import EOS
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        model = small_model().eval()
        example = whole_example(8)
        with torch.no_grad():
            memory = model.encode(example.context)
            log_probs = model.decoder.next_token_log_probs(
                (), memory, span=example.span, tracks=example.tracks)
            probability = float(log_probs[EOS].exp())
            assert 0. <= probability <= 1.
            assert abs(float(log_probs.exp().sum()) - 1.) < 1e-5
    finally:
        torch.set_num_threads(old)


def test_prepare_plan_is_canonical_and_pins_hashes(tmp_path):
    from hashlib import sha256
    from scripts.hcm2_validation_curve import prepare, digest
    cases = tmp_path/'cases.json'; cases.write_text('{"status":"completed","cases":[]}')
    config = tmp_path/'train.json'; config.write_text('{"schema":"PLLWholeRunConfig@1"}')
    ckpt = tmp_path/'checkpoint-00000100.pt'; ckpt.write_bytes(b'weights')
    groups = tmp_path/'groups.json'
    groups.write_text(json.dumps([[str(tmp_path), 100, 100, str(tmp_path/'binding.json')]]))
    output = tmp_path/'prepared'
    prepare(type('A', (), dict(groups=str(groups), output=str(output), cases=str(cases),
                               config=str(config), seed=20260929))())
    plan = json.loads((output/'plan.json').read_text())
    assert plan['status'] == 'prepared'
    assert plan['cases_sha256'] == digest(cases)
    assert plan['config_file_sha256'] == digest(config)
    assert plan['checkpoints'][0]['sha256'] == sha256(b'weights').hexdigest()
    assert plan['checkpoints'][0]['size_bytes'] == 7


def test_shard_partitions_the_pinned_plan_without_gaps_or_overlap():
    plan = [dict(update=update) for update in range(100, 10100, 100)]
    shards = [shard(plan, index, 4) for index in range(4)]
    assert sum(len(part) for part in shards) == len(plan)
    merged = sorted(entry['update'] for part in shards for entry in part)
    assert merged == [entry['update'] for entry in plan]
    assert len(set(merged)) == len(plan)
    assert shard(plan, 0, 1) == plan
    with pytest.raises(ValueError, match='invalid_shard'):
        shard(plan, 4, 4)
    with pytest.raises(ValueError, match='invalid_shard'):
        shard(plan, 0, 0)


def test_encoder_fingerprint_ignores_trained_scopes_and_detects_encoder_changes():
    state = {'encoder.a': torch.zeros(3), 'encoder.b': torch.ones(2, 2),
             'decoder.c': torch.zeros(1), 'memory.d': torch.zeros(1)}
    first = encoder_fingerprint(state)
    assert first.startswith('2:')
    assert encoder_fingerprint(dict(state)) == first
    assert encoder_fingerprint(dict(state, **{'decoder.c': torch.ones(1),
                                              'memory.d': torch.ones(1)})) == first
    assert encoder_fingerprint(dict(state, **{'encoder.b': torch.ones(2, 2) + 1e-7})) != first
    with pytest.raises(ValueError, match='encoder_scope_missing'):
        encoder_fingerprint({'decoder.c': torch.zeros(1)})


def _rows():
    rows = []
    for update in (100, 200):
        for split in ('train', 'validation'):
            for index in range(2):
                rows.append(dict(update=update, id=f'{split}piece{index}:{index+1}', split=split,
                                 nll=10.0 + index, events=2 + index, null_weight=0.5,
                                 eos_first_prob=0.25))
    return rows


def _jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))


def test_merge_creates_its_output_directory_and_matches_a_single_pass(tmp_path):
    rows = _rows()
    _jsonl(tmp_path/'a.jsonl', rows[:4])
    _jsonl(tmp_path/'b.jsonl', rows[4:])
    output = tmp_path/'nested'/'merged'
    merge(type('A', (), dict(inputs=[str(tmp_path/'a.jsonl'), str(tmp_path/'b.jsonl')],
                             output=str(output)))())
    assert output.is_dir()
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['scores'] == len(rows)
    assert summary['checkpoints'] == [100, 200]
    assert summary['points'] == aggregate(rows)
    with pytest.raises(ValueError, match='duplicate_score'):
        read_rows([str(tmp_path/'a.jsonl'), str(tmp_path/'a.jsonl')])
