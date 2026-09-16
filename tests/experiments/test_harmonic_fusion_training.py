"""Bounded synthetic CPU tests; no corpus or checkpoint selection evidence."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    attempt_applied_update, build_source_free_fixture, load_frozen_class_weights,
    combine_single_record_raw_batches, transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair
from music_critic.experiments.analysisgnn.harmonic_fusion_training import (
    FusionTrainingConfig, FusionTrainingSession, evaluate_validation, paired_sessions, schedule_factor,
    paired_sessions_from_checkpoint,
)
from music_critic.experiments.analysisgnn.performance_replay import _scaler


@pytest.fixture
def fixture():
    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    table = tuple(FullOrbitPair(sidecar["record_id"], shift, sidecar["source_component_id"])
                  for shift in range(12))
    reads = []
    def reader(record_id, *, split):
        assert record_id == sidecar["record_id"] and split in {"train", "validation"}
        reads.append(split)
        return batch.raw_graph_batch, deepcopy(sidecar)
    return batch, sidecar, weights, table, reader, reads


def _config(**changes):
    return FusionTrainingConfig(**{"total_updates": 6, "warmup_updates": 2,
                                 "validation_interval": 3, "checkpoint_interval": 2,
                                 "validation_mode": "synthetic_disabled", **changes})


def _session(fixture, *, enabled=True, validation_mode="synthetic_disabled"):
    return FusionTrainingSession(base=CorrectedAnalysisGNNModel(), enabled=enabled,
                                 config=_config(validation_mode=validation_mode),
                                 bindings={"origin": "synthetic", "class_weight_fingerprint": fixture[2].payload_fingerprint},
                                 table=fixture[3])


@pytest.mark.parametrize("changes", [{"total_updates": True}, {"total_updates": 5000},
    {"batch_size": 1}, {"seed": 23}, {"warmup_updates": 6}, {"peak_lr": float("nan")},
    {"peak_lr": 0.01}, {"weight_decay": 0.1}, {"validation_interval": 0}])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        _config(**changes)


def test_schedule_is_fresh_bounded_and_ends_at_zero():
    config = _config()
    assert [schedule_factor(config, i) for i in range(3)] == [0.5, 1.0, 1.0]
    assert schedule_factor(config, 6) == 0
    assert schedule_factor(config, 4) == 0.5


def test_equal_initial_base_optimizer_parameter_inventory_and_rng(fixture):
    base = CorrectedAnalysisGNNModel()
    before = torch.get_rng_state().clone()
    control, fusion = paired_sessions(base, config=_config(),
        bindings={"class_weight_fingerprint": fixture[2].payload_fingerprint}, table=fixture[3])
    assert torch.equal(before, torch.get_rng_state())
    assert control.source_model_state == fusion.source_model_state == model_state_fingerprint(base)
    assert not control.optimizer.state and not fusion.optimizer.state
    assert control.optimizer.param_groups[0]["lr"] == fusion.optimizer.param_groups[0]["lr"]
    for session in (control, fusion):
        params = session.optimizer.param_groups[0]["params"]
        assert len(params) == len(set(map(id, params))) == len(tuple(session.model.parameters()))
    assert sum(p.numel() for p in fusion.model.parameters()) - sum(p.numel() for p in control.model.parameters()) == 216886


def test_control_step_exactly_reuses_old_reference_and_historical_transposition(fixture):
    from music_critic.experiments.analysisgnn.harmonic_fusion_resume import preserve_caller_rng, restore_safe_rng
    session = _session(fixture, enabled=False)
    old_model = deepcopy(session.model.base)
    opt = torch.optim.AdamW(old_model.parameters(), lr=session.config.peak_lr, weight_decay=0.0005)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: schedule_factor(session.config, n))
    scaler, _ = _scaler()
    shifts = tuple(session.sampler.peek(i).shift_pc for i in range(2))
    raw = combine_single_record_raw_batches([fixture[0].raw_graph_batch] * 2)
    transformed = transpose_raw_graph_batch(raw, shifts)
    with preserve_caller_rng():
        restore_safe_rng(session.rng_state)
        reference = attempt_applied_update(model=old_model, raw_graph_batch=transformed,
            sidecars=[fixture[1]] * 2, shifts=shifts, class_weights=fixture[2],
            optimizer=opt, scheduler=sch, scaler=scaler)
    observed = session.step(fixture[4], fixture[2])
    assert observed["loss"] == float(reference.loss.total.detach())
    assert model_state_fingerprint(old_model) == model_state_fingerprint(session.model.base)
    assert observed["draws"] == [asdict(session.sampler.__class__(fixture[3]).peek(i)) for i in range(2)]
    assert session.sampler.position == 2
    assert fixture[5] == ["train", "train"]


def test_zero_init_first_loss_then_nonzero_learned_fusion_and_equal_draws(fixture):
    base = CorrectedAnalysisGNNModel()
    control, fusion = paired_sessions(base, config=_config(),
        bindings={"class_weight_fingerprint": fixture[2].payload_fingerprint}, table=fixture[3])
    first = control.step(fixture[4], fixture[2])
    second = fusion.step(fixture[4], fixture[2])
    assert first == second
    # The first forward/loss is identical. Global clipping now includes
    # fusion gradients, so post-update base tensor identity is not promised.
    assert any(torch.count_nonzero(p) for p in fusion.model.fusion.parameters())
    control.run_until(6, fixture[4], fixture[2])
    fusion.run_until(6, fixture[4], fixture[2])
    assert [x["draws"] for x in control.history] == [x["draws"] for x in fusion.history]
    assert control.applied_updates == fusion.applied_updates == 6
    assert control.sampler.position == fusion.sampler.position == 12
    with pytest.raises(ValueError, match="budget_exhausted"):
        control.step(fixture[4], fixture[2])


def test_validation_confusions_joint_consistency_cadence_and_best_tie(fixture):
    session = _session(fixture, validation_mode="required")
    def validate(model, *, split):
        report = evaluate_validation(model, records=(fixture[1]["record_id"],), reader=fixture[4],
                                     class_weights=fixture[2], split=split)
        assert len(report["per_head"]) == len(report["confusions"]) == 18
        assert {"cadence", "metrical_strength"} <= set(report["per_head"])
        assert report["prediction_consistency"]["total"] == 2
        assert report["corrected_joint_support"] >= 0
        assert report["component_predictions"]
        assert {row["component_id"] for row in report["component_predictions"]} == {fixture[1]["source_component_id"]}
        return report
    saved = []
    session.run_until(6, fixture[4], fixture[2], validate=validate,
                      checkpoint=lambda current: saved.append(current.applied_updates))
    assert [row["update"] for row in session.validations] == [0, 3, 6]
    assert saved == [2, 4, 6]
    assert session.best_update in (0, 3, 6)
    session.validations = [{"update": i, "metrics": {"corrected_primary_macro_score": .4}}
                           for i in (0, 3, 6)]
    assert session.best_update == 0


def test_test_denied_before_reader_and_weight_or_identity_mismatch(fixture):
    session = _session(fixture)
    with pytest.raises(ValueError, match="test_lock"):
        evaluate_validation(session.model, records=("anything",), reader=fixture[4],
                            class_weights=fixture[2], split="test")
    assert not fixture[5]
    session.bindings["class_weight_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="class_weight_binding"):
        session.step(fixture[4], fixture[2])
    assert not fixture[5]
    session.bindings["class_weight_fingerprint"] = fixture[2].payload_fingerprint
    def wrong_reader(*args, **kwargs):
        return fixture[0].raw_graph_batch, {**fixture[1], "record_id": "wrong"}
    with pytest.raises(ValueError, match="record_binding"):
        session.step(wrong_reader, fixture[2])
    assert session.applied_updates == 0


def test_reader_rng_does_not_advance_dropout_stream(fixture):
    import random
    import numpy as np
    base = CorrectedAnalysisGNNModel()
    options = dict(base=base, enabled=True, config=_config(), table=fixture[3],
                   bindings={"class_weight_fingerprint": fixture[2].payload_fingerprint})
    first, second = FusionTrainingSession(**options), FusionTrainingSession(**options)
    def noisy_reader(*args, **kwargs):
        random.random()
        np.random.rand(7)
        torch.rand(100)
        return fixture[4](*args, **kwargs)
    assert first.step(fixture[4], fixture[2]) == second.step(noisy_reader, fixture[2])
    assert model_state_fingerprint(first.model) == model_state_fingerprint(second.model)


def test_missing_initial_validation_rejected_before_callback(fixture):
    session = _session(fixture)
    session.run_until(1, fixture[4], fixture[2])
    calls = []
    with pytest.raises(ValueError, match="validation_disabled"):
        session.run_until(1, fixture[4], fixture[2], validate=lambda *a, **k: calls.append(1))
    assert not calls and not session.validations


def test_verified_source_checkpoint_supplies_same_fresh_model_to_both_arms(fixture, tmp_path):
    from hashlib import sha256
    from music_critic.experiments.pop909_cl_frozen.checkpoint import create_model_only_payload
    base = CorrectedAnalysisGNNModel()
    path = tmp_path / "synthetic-model-only.pt"
    torch.save(create_model_only_payload(base), path)
    data = path.read_bytes()
    options = dict(expected_sha256=sha256(data).hexdigest(), expected_size_bytes=len(data),
                   config=_config(), bindings={"class_weight_fingerprint": fixture[2].payload_fingerprint},
                   table=fixture[3])
    control, fusion = paired_sessions_from_checkpoint(path, **options)
    assert control.source_model_state == fusion.source_model_state == model_state_fingerprint(base)
    assert control.bindings == fusion.bindings
    assert control.bindings["source_checkpoint_sha256"] == options["expected_sha256"]
    assert not control.optimizer.state and not fusion.optimizer.state
    options["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256_mismatch"):
        paired_sessions_from_checkpoint(path, **options)
