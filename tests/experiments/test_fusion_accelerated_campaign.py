"""Focused launch contracts; no corpus/GPU work."""
from dataclasses import replace, asdict
import pickle
from unittest.mock import patch
import pytest
from music_critic.experiments.analysisgnn import performance_prefetch as pp
from music_critic.experiments.analysisgnn import fusion_accelerated_campaign as campaign
from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitPair, FullOrbitSampler


def preparer(**kwargs):
    return pp.WarmTrainBatchPreparer(ProductionArtifactPaths(),
        {'record_bindings': {'a': 'seal', 'b': 'seal'}}, identity_only=False, **kwargs)


def test_runtime_cap_roundtrip_binding_and_guard():
    p = preparer(runtime_payload_cap_bytes=10000)
    q = pickle.loads(pickle.dumps(p))
    assert p.source_fingerprint == q.source_fingerprint
    assert q.runtime_payload_cap_bytes == 10000
    job = pp.PrefetchJob(0, (0, 1), ('a', 'b'), (1, 6), (1, 6), 10000)
    with patch.object(q, '_new_loader') as check:
        q.validate_jobs([job]); check.assert_called_once()
        with pytest.raises(ValueError, match='reservation_mismatch'):
            q.validate_jobs([replace(job, reserved_bytes=9999)])
    assert p.source_fingerprint != preparer(runtime_payload_cap_bytes=10001).source_fingerprint


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_bad_runtime_cap(value):
    with pytest.raises(ValueError, match='runtime_payload_cap'):
        preparer(runtime_payload_cap_bytes=value)


def test_old_measured_gate_preserved():
    p = preparer()
    assert pickle.loads(pickle.dumps(p)).source_fingerprint == p.source_fingerprint
    with pytest.raises(ValueError, match='reservation_manifest_required'):
        p.validate_jobs([])


def test_cap_cannot_be_misrepresented_as_measurement():
    with pytest.raises(ValueError, match='runtime_payload_cap'):
        preparer(runtime_payload_cap_bytes=10, reservation_manifest={})


def test_paths_rebase_without_touching_original():
    p = campaign.paths_from_root('/tmp/campaign-test')
    assert str(p.cache_root).startswith('/tmp/campaign-test/outputs/')


def test_budget_rejected():
    for n in (0, 15001, True):
        with pytest.raises(ValueError, match='invalid_budget'):
            campaign.training_jobs([], n)


def test_original_scheduler_horizon():
    import torch
    from music_critic.experiments.analysisgnn.full_orbit_training import build_full_orbit_optimizer_scheduler
    model = torch.nn.Linear(1, 1)
    opt, sched = build_full_orbit_optimizer_scheduler(model)
    f = sched.lr_lambdas[0]
    assert opt.param_groups[0]['initial_lr'] == 0.005
    assert f(0) == 1/6000
    assert f(15000) > .98
    assert f(120000) == 0


def test_training_schedule_is_original_orbit_prefix():
    table = tuple(FullOrbitPair(rid, shift, rid) for rid in ('a', 'b') for shift in range(12))
    jobs = campaign.training_jobs(table, 20)
    sampler = FullOrbitSampler(table, seed=17)
    for job in jobs:
        draws = [sampler.peek(k) for k in range(2)]
        assert job.record_ids == tuple(d.record_id for d in draws)
        assert job.applied_shifts == tuple(d.shift_pc for d in draws)
        sampler.advance_after_applied_update(2)
    assert len(campaign.gate_jobs(table)) == 12


def test_fusion_alignment_projection_does_not_rebuild_model_output():
    import torch
    from music_critic.experiments.analysisgnn import corrected_training as ref
    from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
    from music_critic.experiments.analysisgnn.harmonic_logit_fusion import HarmonicFusionModel, HarmonicFusionOutput
    from music_critic.experiments.analysisgnn.performance_training import align_performance_target_sidecars
    from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
    raw, sidecar = ref.build_source_free_fixture()
    model = HarmonicFusionModel(CorrectedAnalysisGNNModel(), enabled=True).eval()
    with torch.no_grad():
        output = model(raw.raw_graph_batch)
    expected = ref.align_target_sidecars_after_prediction(output, raw.raw_graph_batch, (sidecar,), shifts=(0,))
    with patch.object(HarmonicFusionOutput, '__post_init__', side_effect=AssertionError('not a model output')):
        actual = align_performance_target_sidecars(output, raw.raw_graph_batch, (sidecar,), shifts=(0,))
    assert recursive_state_fingerprint(asdict(expected)) == recursive_state_fingerprint(asdict(actual))
