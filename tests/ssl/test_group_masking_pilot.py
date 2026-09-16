"""Bounded CPU mechanics only; no trained model, corpus or TEST access."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.pop909_cl_frozen.checkpoint import create_model_only_payload
from music_critic.ssl.group_masking_fixture import build_group_pilot_fixture
from music_critic.ssl.group_masking_pilot import (
    POLICIES, GroupPilotConfig, GroupMaskingPilot, PilotBatch, bind_train_batch,
    paired_mask_bindings, tree_fingerprint,
)
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartSSLModel, export_theory_warmstart_state, load_theory_warmstart,
    load_model_only_theory_warmstart,
)


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def export():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        return export_theory_warmstart_state(CorrectedAnalysisGNNModel())


@pytest.fixture(scope="module")
def batch():
    return build_group_pilot_fixture()


def make_pilot(export, batch, policy=POLICIES[0], config=None):
    config = config or GroupPilotConfig(updates=3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        warm = load_theory_warmstart(TheoryWarmstartSSLModel(config.ssl_config()), export)
    return GroupMaskingPilot(warm, config, policy=policy,
        schedule=(bind_train_batch(batch, config),) * config.updates,
        source_binding={"kind": "synthetic", "state_fingerprint": export["state_fingerprint"]})


@pytest.mark.parametrize("policy", POLICIES)
def test_interrupted_resume_is_exact_and_artifacts_immutable(export, batch, policy, tmp_path):
    full = make_pilot(export, batch, policy)
    interrupted = make_pilot(export, batch, policy)
    assert full.initial_fingerprint == interrupted.initial_fingerprint
    before_rng = torch.get_rng_state().clone()
    full_rows = [full.step_from_factory(lambda: batch) for _ in range(3)]
    assert torch.equal(before_rng, torch.get_rng_state())
    interrupted.step_from_factory(lambda: batch)
    path = tmp_path / "update1.pt"
    evidence = interrupted.save(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        interrupted.save(path)
    assert path.read_bytes() == original
    resumed = make_pilot(export, batch, policy)
    resumed.resume(path, expected_sha256=evidence["sha256"], expected_size_bytes=evidence["size_bytes"])
    resumed.step_from_factory(lambda: batch)
    resumed.step_from_factory(lambda: batch)
    assert resumed.rows == full_rows
    assert tree_fingerprint(resumed.harness.state_dict()) == tree_fingerprint(full.harness.state_dict())
    assert tree_fingerprint(resumed.optimizer.state_dict()) == tree_fingerprint(full.optimizer.state_dict())
    assert all(row["encoder_forwards"] == 3 for row in resumed.rows)
    assert all(row["counts"]["theory_active_heads"] == 18 for row in resumed.rows)
    with pytest.raises(ValueError, match="budget_exhausted"):
        resumed.step_from_factory(lambda: batch)


@pytest.mark.parametrize("ssl,theory,forwards", [(0.7, 0.3, 3), (1., 0., 2), (0., 1., 1)])
def test_paths_update_only_owned_groups(export, batch, ssl, theory, forwards):
    config = GroupPilotConfig(updates=1, lambda_ssl=ssl, lambda_theory=theory)
    item = PilotBatch(batch.ssl if ssl else None, batch.theory if theory else None,
                      batch.sidecars if theory else (), batch.class_weights if theory else None)
    pilot = make_pilot(export, item, config=config)
    initial = {name: value.clone() for name, value in pilot.harness.named_parameters()}
    row = pilot.step_from_factory(lambda: item)
    assert row["encoder_forwards"] == forwards
    parameters = pilot.harness.optimizer_parameters()
    assert len(parameters) == len({id(p) for p in parameters}) == 605
    changed = {name for name, p in pilot.harness.named_parameters() if not torch.equal(p, initial[name])}
    inventory = pilot.harness.parameter_inventory()
    for role, active in [("shared", True), ("ssl", ssl > 0), ("theory", theory > 0), ("inactive", False)]:
        names = set(inventory[role]["names"])
        assert bool(names & changed) is active


def test_paired_start_masks_density_schedule_and_blindness(export, batch):
    left, right = (make_pilot(export, batch, policy) for policy in POLICIES)
    assert left.initial_fingerprint == right.initial_fingerprint
    assert left.schedule == right.schedule
    assert set(map(id, left.harness.parameters())).isdisjoint(map(id, right.harness.parameters()))
    a, b = left.step_from_factory(lambda: batch), right.step_from_factory(lambda: batch)
    assert a["paired_masks"] == b["paired_masks"]
    assert a["batch_binding"] == b["batch_binding"]
    assert a["encoder_forwards"] == b["encoder_forwards"]
    for policy in POLICIES:
        rows = a["paired_masks"][policy]
        assert len(rows) == 2
        assert all(0 < r["hidden_notes"] < r["eligible_notes"] for r in rows)
        assert all(r["unit"] == ("note" if policy == POLICIES[0] else "beat") for r in rows)
    config = left.config
    bad = deepcopy(batch)
    bad.class_weights.values[next(iter(bad.class_weights.values))][0] += 1
    assert bind_train_batch(bad, config) != bind_train_batch(batch, config)
    assert paired_mask_bindings(bad.ssl, config, update=1)[POLICIES[1]].fingerprint == paired_mask_bindings(batch.ssl, config, update=1)[POLICIES[1]].fingerprint


def test_test_split_denied_before_factory_and_bad_schedule_before_mutation(export, batch):
    pilot = make_pilot(export, batch)
    before = model_state_fingerprint(pilot.harness)
    def forbidden():
        pytest.fail("reader must not run")
    with pytest.raises(ValueError, match="train_only"):
        pilot.step_from_factory(forbidden, split="test")
    changed = deepcopy(batch)
    changed.sidecars[0]["record_id"] = "different"
    with pytest.raises(ValueError, match="actual_schedule"):
        pilot.step_from_factory(lambda: changed)
    assert model_state_fingerprint(pilot.harness) == before
    assert not pilot.optimizer.state


@pytest.mark.parametrize("policy", POLICIES)
def test_ws2_actual_policy_mask_closure_and_stop_gradient(export, batch, policy):
    # Mutate every selected and collateral input slot including availability;
    # the full targets change while masked-online outputs stay bit-identical.
    from test_model import _mutate_masked_slots
    pilot = make_pilot(export, batch, policy)
    model = pilot.harness.ssl_model.eval()
    binding = paired_mask_bindings(batch.ssl, pilot.config, update=0)[policy]
    forward = model.forward if policy == POLICIES[0] else model.forward_hierarchy
    original = forward(batch.ssl, prepared_mask_binding=binding)
    changed_batch = _mutate_masked_slots(batch.ssl, binding.mask_plans)
    changed_binding = paired_mask_bindings(changed_batch, pilot.config, update=0)[policy]
    changed = forward(changed_batch, prepared_mask_binding=changed_binding)
    assert original.mask_plans == changed.mask_plans
    for level, value in original.online_encoder.fused.embeddings.items():
        assert torch.equal(value, changed.online_encoder.fused.embeddings[level])
    assert not torch.equal(original.targets.note, changed.targets.note)
    assert original.note_loss.mean != changed.note_loss.mean
    for target in [original.targets.note, original.targets.bar, original.targets.song,
                   original.bar_latent.target, original.song_latent.target]:
        assert not target.requires_grad and target.grad_fn is None
    original.objective.total_loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_verified_model_only_bridge_provenance_and_rejections(export, tmp_path):
    source = CorrectedAnalysisGNNModel()
    source.load_state_dict(export["corrected_state"], strict=True)
    path = tmp_path / "synthetic-model-only.pt"
    torch.save(create_model_only_payload(source), path)
    raw = path.read_bytes()
    args = dict(expected_sha256=sha256(raw).hexdigest(), expected_size_bytes=len(raw),
                expected_state_fingerprint=export["state_fingerprint"])
    destination = TheoryWarmstartSSLModel()
    result, evidence = load_model_only_theory_warmstart(destination, path, **args)
    assert result.report.provenance == "verified_model_only_artifact"
    assert result.artifact_evidence == evidence
    assert evidence["optimizer_restored"] is False
    assert model_state_fingerprint(result.theory_model) == export["state_fingerprint"]
    assert len(result.theory_model.task_specs) == 18
    assert model_state_fingerprint(source) == export["state_fingerprint"]
    before = model_state_fingerprint(destination)
    with pytest.raises(ValueError, match="state_binding"):
        load_model_only_theory_warmstart(destination, path, **{**args, "expected_state_fingerprint": "0"*64})
    with pytest.raises(ValueError, match="sha256_mismatch"):
        load_model_only_theory_warmstart(destination, path, **{**args, "expected_sha256": "0"*64})
    assert model_state_fingerprint(destination) == before
    forged = {**export, "provenance": "verified_model_only_artifact"}
    with pytest.raises(ValueError, match="contract_incompatible"):
        load_theory_warmstart(TheoryWarmstartSSLModel(), forged)


def test_resume_rejects_policy_schedule_artifact_and_alias_drift(export, batch, tmp_path):
    pilot = make_pilot(export, batch)
    pilot.step_from_factory(lambda: batch)
    evidence = pilot.save(tmp_path / "checkpoint.pt")
    other = make_pilot(export, batch, POLICIES[1])
    with pytest.raises(ValueError, match="resume_binding"):
        other.resume(evidence["path"], expected_sha256=evidence["sha256"], expected_size_bytes=evidence["size_bytes"])
    fresh = make_pilot(export, batch)
    with pytest.raises(ValueError, match="resume_artifact"):
        fresh.resume(evidence["path"], expected_sha256="0"*64, expected_size_bytes=evidence["size_bytes"])
    payload = pilot.checkpoint_payload()
    key = next(k for k in payload["model"] if k.startswith("theory_model.local_baseline.encoder."))
    payload["model"][key].add_(1)
    payload["payload_fingerprint"] = tree_fingerprint({k: v for k,v in payload.items() if k != "payload_fingerprint"})
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    raw = forged.read_bytes()
    with pytest.raises(ValueError, match="shared_alias_conflict"):
        fresh.resume(forged, expected_sha256=sha256(raw).hexdigest(), expected_size_bytes=len(raw))
    assert not fresh.optimizer.state


def test_failure_requires_recreation_from_completed_checkpoint(export, batch, monkeypatch):
    pilot = make_pilot(export, batch)
    def interrupted(*a, **kw):
        raise KeyboardInterrupt("synthetic interruption before optimizer step")
    monkeypatch.setattr(pilot.optimizer, "step", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pilot.step_from_factory(lambda: batch)
    assert pilot.completed_updates == 0 and pilot.failed
    with pytest.raises(ValueError, match="failed_instance"):
        pilot.step_from_factory(lambda: batch)
    with pytest.raises(ValueError, match="cannot_checkpoint_failed"):
        pilot.checkpoint_payload()


@pytest.mark.parametrize("damage,reason", [("empty_optimizer", "state_missing"), ("missing_slot", "state_missing"),
    ("negative_variance", "negative_variance"), ("impossible_forwards", "forward_count"),
    ("invalid_loss", "loss_invalid"), ("zero_step_weights", "initial_state")])
def test_resume_rejects_semantically_forged_state(export, batch, tmp_path, damage, reason):
    pilot = make_pilot(export, batch)
    if damage != "zero_step_weights":
        pilot.step_from_factory(lambda: batch)
    payload = pilot.checkpoint_payload()
    if damage == "empty_optimizer":
        payload["optimizer"]["state"].clear()
    elif damage == "missing_slot":
        payload["optimizer"]["state"].pop(next(iter(payload["optimizer"]["state"])))
    elif damage == "negative_variance":
        next(iter(payload["optimizer"]["state"].values()))["exp_avg_sq"].fill_(-1)
    elif damage == "impossible_forwards":
        payload["rows"][0]["encoder_forwards"] = -100
    elif damage == "invalid_loss":
        payload["rows"][0]["theory_loss"] = None
    else:
        key = next(k for k in payload["model"] if k.startswith("theory_model.task_heads."))
        payload["model"][key].add_(1)
    payload["payload_fingerprint"] = tree_fingerprint({k: v for k, v in payload.items() if k != "payload_fingerprint"})
    path = tmp_path / "forged.pt"
    torch.save(payload, path)
    raw = path.read_bytes()
    fresh = make_pilot(export, batch)
    with pytest.raises(ValueError, match=reason):
        fresh.resume(path, expected_sha256=sha256(raw).hexdigest(), expected_size_bytes=len(raw))
    assert not fresh.optimizer.state
    assert model_state_fingerprint(fresh.harness) == fresh.initial_fingerprint


@pytest.mark.parametrize("policy", POLICIES)
def test_coherent_pitch_rebuild_under_actual_ws2_policy(export, policy):
    from music_critic.ssl.bounded_fixture import build_phase7a_bounded_fixture
    from music_critic.ssl.bounded_fixture import mutate_piece_pitch_group
    from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
    from music_critic.graph import graph_fingerprint
    fixture = build_phase7a_bounded_fixture()
    raw = collate_ssl_samples(fixture.raw_samples("train")[:1])
    config = GroupPilotConfig(updates=1, lambda_theory=0., lambda_ssl=1.)
    pilot = make_pilot(export, PilotBatch(ssl=raw), policy, config)
    model = pilot.harness.ssl_model.eval()
    binding = paired_mask_bindings(raw, config, update=0)[policy]
    forward = model.forward if policy == POLICIES[0] else model.forward_hierarchy
    original = forward(raw, prepared_mask_binding=binding)
    mutation = mutate_piece_pitch_group(fixture.train_pieces[0], binding.mask_plans[0].selected_local_node_indices)
    changed = collate_ssl_samples((SSLRawSample(mutation.mutated_raw_graph,
        graph_fingerprint(mutation.mutated_raw_graph), raw.dataset_ids[0], raw.piece_ids[0]),))
    changed_binding = paired_mask_bindings(changed, config, update=0)[policy]
    updated = forward(changed, prepared_mask_binding=changed_binding)
    assert binding.mask_plans == changed_binding.mask_plans
    assert all(torch.equal(value, updated.online_encoder.fused.embeddings[level])
               for level, value in original.online_encoder.fused.embeddings.items())
    assert not torch.equal(original.targets.note, updated.targets.note)


def test_unavailable_beat_policy_blocks_both_arms_without_fallback(export):
    from music_critic.ssl.bounded_fixture import build_phase7a_bounded_fixture
    from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
    from music_critic.graph import build_raw_graph, graph_fingerprint
    from music_critic.data.timing import RationalTime
    piece = build_phase7a_bounded_fixture().train_pieces[0]
    piece = replace(piece, notes=tuple(replace(n, onset_qn=RationalTime(0), duration_qn=RationalTime(1)) for n in piece.notes[:2]))
    graph = build_raw_graph(piece)
    raw = collate_ssl_samples((SSLRawSample(graph, graph_fingerprint(graph), piece.dataset_name, piece.piece_id),))
    item = PilotBatch(ssl=raw)
    config = GroupPilotConfig(updates=1, lambda_ssl=1., lambda_theory=0.)
    pilot = make_pilot(export, item, config=config)
    with pytest.raises(ValueError):
        pilot.step_from_factory(lambda: item)
    assert pilot.completed_updates == 0 and not pilot.optimizer.state
    assert model_state_fingerprint(pilot.harness) == pilot.initial_fingerprint


@pytest.mark.parametrize("corruption", ["finite_gradient_overflow", "model_after_step"])
def test_post_step_nonfinite_state_is_not_a_success(export, batch, monkeypatch, corruption):
    pilot = make_pilot(export, batch)
    original = pilot.optimizer.step
    if corruption == "finite_gradient_overflow":
        # Finite FP32 gradients overflow Adam's squared accumulator.
        handle = next(p for p in pilot.harness.optimizer_parameters() if p.requires_grad).register_hook(
            lambda gradient: torch.full_like(gradient, 1e30))
    else:
        handle = None
        def corrupt_model(*args, **kwargs):
            result = original(*args, **kwargs)
            with torch.no_grad():
                next(pilot.harness.parameters()).fill_(float("inf"))
            return result
        monkeypatch.setattr(pilot.optimizer, "step", corrupt_model)
    receipts = []
    try:
        with pytest.raises(ValueError, match="nonfinite_(optimizer|model)_state"):
            pilot.step_from_factory(lambda: batch, on_attempt=receipts.append)
    finally:
        if handle is not None:
            handle.remove()
    assert pilot.failed and pilot.completed_updates == 0 and pilot.rows == []
    assert receipts[0]["status"] == "started"
    assert receipts[-1]["stage"] == "post_step_validation"
    assert receipts[-1]["observed_encoder_forwards"] == 3
    assert receipts[-1]["optimizer_outcome"] == "applied"
    assert receipts[-1]["status"] == "failed"
    with pytest.raises(ValueError, match="cannot_checkpoint_failed"):
        pilot.checkpoint_payload()


@pytest.mark.parametrize("operation", ["step", "checkpoint", "resume"])
def test_runtime_drift_after_constructor_rejected(export, batch, tmp_path, operation):
    pilot = make_pilot(export, batch)
    evidence = pilot.save(tmp_path / "fresh.pt")
    previous = torch.get_num_threads()
    readers = []
    try:
        torch.set_num_threads(previous + 1)
        with pytest.raises(ValueError, match="runtime_drift"):
            if operation == "step":
                pilot.step_from_factory(lambda: readers.append(True))
            elif operation == "checkpoint":
                pilot.checkpoint_payload()
            else:
                pilot.resume(tmp_path / "does-not-exist.pt", expected_sha256=evidence["sha256"],
                             expected_size_bytes=evidence["size_bytes"])
    finally:
        torch.set_num_threads(previous)
    assert not readers and not pilot.optimizer.state and pilot.completed_updates == 0
    if operation == "step":
        assert pilot.failed and pilot.last_attempt["optimizer_outcome"] == "not_applied"


def test_runtime_drift_inside_update_rejected_before_adam(export, batch):
    pilot = make_pilot(export, batch)
    previous = torch.get_num_threads()
    def mutate_runtime():
        torch.set_num_threads(previous + 1)
        return batch
    try:
        with pytest.raises(ValueError, match="runtime_drift"):
            pilot.step_from_factory(mutate_runtime)
    finally:
        torch.set_num_threads(previous)
    assert not pilot.optimizer.state and pilot.failed
    assert pilot.last_attempt["observed_encoder_forwards"] == 3
    assert pilot.last_attempt["optimizer_outcome"] == "not_applied"


def test_cuda_runtime_binding_captures_cublas_without_initializing_cuda(monkeypatch):
    from music_critic.ssl.group_masking_pilot import _runtime_binding
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    first = _runtime_binding(torch.device("cuda:0"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    second = _runtime_binding(torch.device("cuda:0"))
    assert first["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert second["CUBLAS_WORKSPACE_CONFIG"] == ":16:8" and first != second
    assert _runtime_binding(torch.device("cpu"))["CUBLAS_WORKSPACE_CONFIG"] is None


def test_optimizer_raises_after_applying_is_unknown_not_rolled_back(export, batch, monkeypatch):
    pilot = make_pilot(export, batch)
    original = pilot.optimizer.step
    def applied_then_interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("synthetic interruption after Adam applied")
    monkeypatch.setattr(pilot.optimizer, "step", applied_then_interrupted)
    receipts = []
    with pytest.raises(KeyboardInterrupt):
        pilot.step_from_factory(lambda: batch, on_attempt=receipts.append)
    assert pilot.optimizer.state and model_state_fingerprint(pilot.harness) != pilot.initial_fingerprint
    assert pilot.failed and pilot.completed_updates == 0 and pilot.rows == []
    assert receipts[-1]["optimizer_outcome"] == "unknown"
    assert receipts[-1]["stage"] == "optimizer_step"
    assert receipts[-1]["observed_encoder_forwards"] == 3
    assert receipts[-1]["confirmed_updates"] == 0


def test_failure_before_factory_has_durable_started_receipt(export, batch):
    pilot = make_pilot(export, batch)
    receipts = []
    def broken_factory():
        assert receipts[0]["status"] == "started"
        raise RuntimeError("synthetic reader failure")
    with pytest.raises(RuntimeError, match="reader failure"):
        pilot.step_from_factory(broken_factory, on_attempt=receipts.append)
    assert pilot.failed and receipts[-1]["stage"] == "factory"
    assert receipts[-1]["observed_encoder_forwards"] == 0
    assert receipts[-1]["optimizer_outcome"] == "not_applied"


def test_checkpoint_revalidates_current_state(export, batch):
    pilot = make_pilot(export, batch)
    with torch.no_grad():
        next(pilot.harness.parameters()).fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite_model_state"):
        pilot.checkpoint_payload()


def test_resume_schema_upgrade_preserves_original_dropout_seed_domain():
    from music_critic.experiments.analysisgnn.contracts import fingerprint
    from music_critic.ssl.group_masking_pilot import _step_rng
    before = torch.get_rng_state().clone()
    seed = int(fingerprint({"domain": "ws2-group-masking-pilot@1.0.0:update",
                            "seed": 17, "update": 2})[:15], 16)
    generator = torch.Generator().manual_seed(seed)
    expected = torch.rand(7, generator=generator)
    with _step_rng(17, 2, torch.device("cpu")):
        actual = torch.rand(7)
    assert torch.equal(expected, actual) and torch.equal(before, torch.get_rng_state())


def test_runtime_binding_distinguishes_high_and_medium_matmul():
    from music_critic.ssl.group_masking_pilot import _runtime_binding
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        first = _runtime_binding(torch.device("cuda:0"))
        torch.set_float32_matmul_precision("medium")
        second = _runtime_binding(torch.device("cuda:0"))
    finally:
        torch.set_float32_matmul_precision(original)
    assert first["matmul_allow_tf32"] == second["matmul_allow_tf32"]
    assert first["float32_matmul_precision"] == "high"
    assert second["float32_matmul_precision"] == "medium" and first != second


def test_entering_autocast_after_constructor_is_runtime_drift(export, batch):
    pilot = make_pilot(export, batch)
    called = []
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="runtime_drift"):
            pilot.step_from_factory(lambda: called.append(True))
    assert not called and not pilot.optimizer.state
