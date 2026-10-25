"""Synthetic C4 mechanics; no corpus access or model-quality claims."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from music_critic.experiments.c4_pop909 import training as c4
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler
from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_pc_marginal_loss_trains_all_enharmonic_spellings_without_guessing_one():
    logits = torch.zeros((1, 37), requires_grad=True)
    log_probs = c4.pc_log_probabilities(logits, "root")
    loss = -log_probs[0, 0]
    loss.backward()
    mapping = c4.pc_class_mapping("root")
    assert log_probs.exp().sum().item() == pytest.approx(1.)
    assert all(logits.grad[0, i] < 0 for i, pc in enumerate(mapping) if pc == 0)
    assert all(logits.grad[0, i] > 0 for i, pc in enumerate(mapping) if pc != 0)


def test_masked_and_unaligned_targets_never_become_negative_supervision():
    logits = {"root": torch.zeros((1, 37), requires_grad=True),
              "quality": torch.zeros((1, 17), requires_grad=True)}
    output = SimpleNamespace(logits=logits, candidate_keys={
        h: ((0, "slice", "s0"),) for h in logits})
    rows = [[dict(slice_id="s0", beat_id=None, targets={
        "root": dict(available=True, value=0),
        "quality": dict(available=False, value=999)}),
        dict(slice_id="missing", beat_id=None, targets={
            "quality": dict(available=True, value=2)})]]
    report = c4.supervised_loss(output, rows)
    report.total.backward()
    assert report.counts == {"root": 1, "bass": 0, "quality": 0,
                             "inversion": 0, "pitch_class_set": 0}
    assert logits["quality"].grad is None
    assert torch.isfinite(logits["root"].grad).all()


def test_optimizer_preserves_historical_warmup_and_120k_horizon():
    model = torch.nn.Linear(2, 2)
    optimizer, scheduler = c4.build_optimizer(model)
    assert optimizer.defaults["weight_decay"] == .0005
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.005 / 6000)
    assert scheduler.lr_lambdas[0](5999) == 1.
    assert scheduler.lr_lambdas[0](6000) == 1.
    assert scheduler.lr_lambdas[0](120000) == 0.
    assert scheduler.lr_lambdas[0](80000) > 0.


def test_full_state_resume_matches_next_dropout_optimizer_step(tmp_path):
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.1), torch.nn.Linear(4, 2))
    optimizer, scheduler = c4.build_optimizer(model)
    sampler = FullOrbitSampler((FullOrbitPair("a", 0, "g"), FullOrbitPair("b", 1, "h")))
    config = c4.TrainingConfig(total_updates=2, batch_size=1, workers=0)
    def step(m, o, s, sample):
        o.zero_grad(set_to_none=True)
        m(torch.ones(2, 3)).square().mean().backward()
        o.step(); s.step(); sample.advance_after_applied_update(1)
    step(model, optimizer, scheduler, sampler)
    path = tmp_path / "step1.pt"
    c4.save_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
        sampler=sampler, config=config, bindings={"source": "synthetic"},
        progress={"applied_updates": 1, "best_score": None})
    step(model, optimizer, scheduler, sampler)
    expected = recursive_state_fingerprint((model.state_dict(), optimizer.state_dict(),
                                           scheduler.state_dict(), sampler.state_dict()))
    restored = deepcopy(model)
    opt, sched = c4.build_optimizer(restored)
    sample = FullOrbitSampler(sampler.table)
    c4.load_checkpoint(path, model=restored, optimizer=opt, scheduler=sched,
        sampler=sample, config=config, bindings={"source": "synthetic"})
    step(restored, opt, sched, sample)
    assert recursive_state_fingerprint((restored.state_dict(), opt.state_dict(),
        sched.state_dict(), sample.state_dict())) == expected
    before = recursive_state_fingerprint(restored.state_dict())
    with pytest.raises(ValueError, match="binding"):
        c4.load_checkpoint(path, model=restored, optimizer=opt, scheduler=sched,
            sampler=sample, config=config, bindings={"source": "different"})
    assert recursive_state_fingerprint(restored.state_dict()) == before


def test_training_config_rejects_unbounded_or_test_settings():
    with pytest.raises(ValueError):
        c4.TrainingConfig(total_updates=0)
    with pytest.raises(ValueError):
        c4.TrainingConfig(max_wall_seconds=float("inf"))
    with pytest.raises(ValueError):
        c4.TrainingConfig(workers=8, max_pending=0)


def test_cpu_validated_path_preserves_full_c3_dropout_gradients_and_optimizer_state():
    from pathlib import Path
    from music_critic.data import load_piece
    from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
    from music_critic.experiments.analysisgnn.training_rng import capture_safe_rng, restore_safe_rng
    piece = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    graph = build_temporal_batch((piece,))
    row = dict(slice_id=graph.keys["slice"][0][1], beat_id=None,
               targets={"root": dict(available=True, value=0),
                        "quality": dict(available=True, value=2)})
    a, b = c4.components(), c4.components()
    rng = capture_safe_rng()
    normal = c4.attempt_update(*a[:3], graph, ((row,),), validated=False)
    expected = c4.state_digest(a)
    restore_safe_rng(rng, device_type="cpu")
    fast = c4.attempt_update(*b[:3], graph, ((row,),), validated=True)
    assert normal == fast
    assert c4.state_digest(b) == expected
    assert len(a[0].base.task_specs) == 18


def test_ordered_pool_bounds_pending_and_authenticates_private_payload():
    from concurrent.futures import Future
    from dataclasses import asdict
    from pathlib import Path
    from music_critic.data import load_piece
    from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
    from music_critic.experiments.c4_pop909.transport import pack_private_cpu_payload
    piece = load_piece(Path(__file__).parents[1] / "fixtures/data/canonical_piece_v2.json")
    graph = build_temporal_batch((piece,))
    jobs = [c4.BatchJob(i, i, ("synthetic",), (0,)) for i in range(4)]
    class Executor:
        def __init__(self):
            self.submitted = 0
        def submit(self, fn, job):
            self.submitted += 1
            future = Future(); future.set_result(fn(job)); return future
        def shutdown(self, **kwargs):
            pass
    def prepare(job):
        return pack_private_cpu_payload(dict(job=asdict(job), manifest="fixture",
            graph=c4.encode_temporal_graph(graph), rows=((),), cache={}, pid=1), cap=c4.PAYLOAD_CAP)
    executor = Executor()
    pool = c4.OrderedPool(jobs, {"fingerprint": "fixture"}, Path("."),
        c4.TrainingConfig(batch_size=1, workers=1, max_pending=2), executor=executor, prepare_fn=prepare)
    assert executor.submitted == 2
    first = pool.peek()
    assert pool.peek() is first
    with pytest.raises(ValueError, match="pool_commit"):
        pool.commit(jobs[1])
    pool.commit(first.job)
    assert executor.submitted == 3 and len(pool.pending) == 2
    pool.close()
    corrupt = c4.OrderedPool(jobs, {"fingerprint": "different"}, Path("."),
        c4.TrainingConfig(batch_size=1, workers=1), executor=Executor(), prepare_fn=prepare)
    with pytest.raises(ValueError, match="worker_payload_binding"):
        corrupt.peek()
    assert corrupt.committed == 0
    corrupt.close()


def test_cli_failure_cannot_write_into_existing_attempt(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "events.jsonl").write_text("owned by another running attempt\n")
    def fail(**kwargs):
        raise FileExistsError("existing output")
    monkeypatch.setattr(c4, "prepare", fail)
    with pytest.raises(FileExistsError):
        c4.main(["prepare", "--index", "absent", "--split", "absent", "--source-root", "absent",
                 "--output", str(output), "--experiment-record", "absent"])
    assert not (output / "report.json").exists()
    assert (output / "events.jsonl").read_text() == "owned by another running attempt\n"
