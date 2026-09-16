"""Tiny generated CPU fixtures exercise the actual scientific workflow path."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.ssl.group_masking_fixture import build_group_pilot_fixture
from music_critic.ssl.bounded_fixture import build_phase7a_bounded_fixture
from music_critic.ssl.data import collate_ssl_samples
from music_critic.ssl.group_masking_pilot import (
    POLICIES, GroupPilotConfig, PilotBatch, bind_train_batch, tree_fingerprint,
)
from music_critic.ssl.group_masking_scientific_eval import validation_batch_binding
from music_critic.ssl.group_masking_scientific_runner import (
    run_scientific_workflow, make_theory_recovery_pilot, _Workflow, _read_bound,
)
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartSSLModel, export_theory_warmstart_state, load_theory_warmstart,
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


class FixtureData:
    """A fake provider, never a declaration of real TRAIN/VALIDATION membership."""
    def __init__(self, pilot, recovery):
        self.train = build_group_pilot_fixture()
        self.recovery = PilotBatch(theory=self.train.theory, sidecars=self.train.sidecars,
                                   class_weights=self.train.class_weights)
        self.configs = {"pilot": pilot, "recovery": recovery}
        raw = collate_ssl_samples((build_phase7a_bounded_fixture().raw_samples("train")[0],))
        self.validation = PilotBatch(raw, raw, self.train.sidecars[:1], self.train.class_weights)
        self.validation_membership = tuple(zip(raw.dataset_ids, raw.piece_ids))
        self.validation_bindings = (validation_batch_binding(self.validation, pilot),)
        self.train_seen_tuples = None
        self.binding = {"kind": "generated_fixture_only", "train": bind_train_batch(self.train, pilot),
            "recovery": bind_train_batch(self.recovery, recovery), "validation": list(self.validation_bindings)}

    def schedule_bindings(self, stage):
        batch = self.train if stage == "pilot" else self.recovery
        return (bind_train_batch(batch, self.configs[stage]),) * self.configs[stage].updates

    def load_update(self, stage, index):
        assert 0 <= index < self.configs[stage].updates
        return self.train if stage == "pilot" else self.recovery

    def validation_batches(self):
        return iter((self.validation,))


def arguments(export, root, *, recovery_updates=1, interval=50):
    pilot = GroupPilotConfig(updates=2)
    recovery = GroupPilotConfig(updates=recovery_updates, lambda_ssl=0, lambda_theory=1)
    def factory():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            return load_theory_warmstart(TheoryWarmstartSSLModel(pilot.ssl_config()), export)
    return dict(warmstart_factory=factory,
        source_binding={"kind": "synthetic", "state_fingerprint": export["state_fingerprint"]},
        data=FixtureData(pilot, recovery), pilot_config=pilot, recovery_config=recovery,
        output_root=root, contract_binding={"kind": "unit_synthetic", "implementation": "fixture"},
        checkpoint_interval=interval, fixture_mode=True)


@pytest.fixture(scope="module")
def full(export, tmp_path_factory):
    root = tmp_path_factory.mktemp("workflow") / "full"
    args = arguments(export, root)
    result = run_scientific_workflow(**args)
    return root, result


def manifest_arguments(root, ref):
    return dict(resume_manifest=root / ref["path"], expected_resume_sha256=ref["sha256"],
                expected_resume_size_bytes=ref["size_bytes"])


def read_ref(root, ref):
    return json.loads((root / ref["path"]).read_text())


def checkpoint(root, ref):
    return torch.load(root / ref["path"], weights_only=True, map_location="cpu")


def assert_endpoints_equal(left_root, left, right_root, right):
    assert left["forgetting_recovery"] == right["forgetting_recovery"]
    assert left["beat_minus_random_control"] == right["beat_minus_random_control"]
    for key in left["evaluations"]:
        assert read_ref(left_root, left["evaluations"][key]) == read_ref(right_root, right["evaluations"][key])
    for key in left["checkpoints"]:
        old = checkpoint(left_root, left["checkpoints"][key])
        new = checkpoint(right_root, right["checkpoints"][key])
        for field in ("model", "optimizer", "rows", "completed_updates"):
            assert tree_fingerprint(old[field]) == tree_fingerprint(new[field])
        # Restricted torch serialization may preserve different storage aliases
        # after resume; the scientific states are exact but artifact SHAs need
        # not coincide. Recovery deliberately binds its actual parent bytes.
        old_binding, new_binding = deepcopy(old["binding"]), deepcopy(new["binding"])
        for binding in (old_binding, new_binding):
            if "stage_transition" in binding:
                binding["stage_transition"].pop("parent_checkpoint_sha256")
                binding["stage_transition"].pop("parent_checkpoint_size_bytes")
        assert old_binding == new_binding


def reseal(payload, path):
    payload.pop("payload_fingerprint", None)
    payload["payload_fingerprint"] = tree_fingerprint(payload)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return dict(resume_manifest=path, expected_resume_sha256=sha256(raw).hexdigest(), expected_resume_size_bytes=len(raw))


def test_complete_workflow_has_fixed_endpoints_recovery_and_separate_compute(full):
    root, result = full
    assert result["status"] == "completed"
    assert result["compute"]["confirmed_updates"] == 6
    assert result["compute"]["replayed_confirmed_updates"] == 0
    assert result["compute"]["observed_encoder_forwards"] == 39  # 14 TRAIN + 5 * 5 VALIDATION.
    assert result["test_access"] == "closed"
    assert result["scientific_superiority_claim"] is False
    for policy in POLICIES:
        assert len(result["forgetting_recovery"][policy]["per_head"]) == 18
        for stage, updates, forwards in (("pilot", 2, 6), ("recovery", 1, 1)):
            counts = result["compute"]["stages"][policy + "." + stage]
            assert counts["confirmed_updates"] == updates
            assert counts["observed_encoder_forwards"] == forwards
            assert counts["optimizer_outcomes"] == {"applied": updates, "unknown": 0, "not_applied": 0}
            assert counts["ssl_draws_from_confirmed_rows"] == (4 if stage == "pilot" else 0)
    assert len((root / "execution.jsonl").read_text().splitlines()) == 6
    events = [json.loads(line) for line in (root / "receipts.jsonl").read_text().splitlines()]
    first = {}
    for event in events:
        first.setdefault(event["operation_id"], event)
    assert all(event["receipt"]["status"] == "started" for event in first.values())
    assert len(first) == 11  # Six updates, five evaluations.
    assert not (root / "failure.json").exists()


def test_recovery_retains_endpoint_ssl_state_and_resets_optimizer(full):
    root, result = full
    for policy in POLICIES:
        end = checkpoint(root, result["checkpoints"][policy + ".pilot"])
        start = torch.load(root / ("checkpoints/" + policy + ".recovery-000000.pt"), weights_only=True)
        final = checkpoint(root, result["checkpoints"][policy + ".recovery"])
        assert start["optimizer"]["state"] == {}
        assert tree_fingerprint(start["model"]) == tree_fingerprint(end["model"])
        transition = start["binding"]["stage_transition"]
        assert transition["parent_checkpoint_sha256"] == result["checkpoints"][policy + ".pilot"]["sha256"]
        assert transition["parent_full_state_fingerprint"] == model_state_fingerprint(end["model"])
        assert start["binding"]["initial_state"] == model_state_fingerprint(end["model"])
        names = set(start["binding"]["optimizer_inventory"]["ssl"]["names"])
        assert names
        assert all(torch.equal(start["model"][name], final["model"][name]) for name in names)
        assert final["rows"][0]["ssl_loss"] is None
        assert final["rows"][0]["encoder_forwards"] == 1


def test_real_adam_then_interrupt_resumes_midpilot_and_counts_replay(export, full, tmp_path, monkeypatch):
    root = tmp_path / "failed"
    original = torch.optim.Adam.step
    calls = 0
    def interrupt(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("synthetic after actual Adam")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_scientific_workflow(**arguments(export, root))
    failure = json.loads((root / "failure.json").read_text())
    counts = failure["compute"]["stages"][POLICIES[0] + ".pilot"]
    assert counts["confirmed_updates"] == 1
    assert counts["observed_encoder_forwards"] == 6
    assert counts["optimizer_outcomes"] == {"applied": 1, "not_applied": 0, "unknown": 1}
    assert not (root / "result.json").exists()
    sealed = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    resumed_root = tmp_path / "resumed"
    resumed = run_scientific_workflow(**arguments(export, resumed_root),
        **manifest_arguments(root, failure["resume_manifest"]))
    assert all(path.read_bytes() == raw for path, raw in sealed.items())
    assert_endpoints_equal(*full, resumed_root, resumed)
    assert resumed["compute"]["confirmed_updates"] == 7
    assert resumed["compute"]["replayed_confirmed_updates"] == 1
    assert resumed["compute"]["observed_encoder_forwards"] == 50
    assert resumed["current_attempt_compute"]["confirmed_updates"] == 6
    assert resumed["compute"]["stages"]["baseline.validation_replay"]["observed_encoder_forwards"] == 5
    assert resumed["compute"]["stages"][POLICIES[0] + ".pilot"]["replayed_after_unknown_attempts"] == 1


def test_midrecovery_checkpoint_restores_optimizer_and_phase(export, tmp_path):
    full_root = tmp_path / "full_recovery"
    full_result = run_scientific_workflow(**arguments(export, full_root, recovery_updates=2, interval=1))
    failed_root = tmp_path / "failed_recovery"
    def stop(event):
        if (event["key"] == POLICIES[0] + ".recovery" and event["kind"] == "training"
                and event["receipt"]["update"] == 2 and event["receipt"]["stage"] == "started"):
            raise KeyboardInterrupt("synthetic recovery boundary")
    with pytest.raises(KeyboardInterrupt):
        run_scientific_workflow(**arguments(export, failed_root, recovery_updates=2, interval=1), event_sink=stop)
    failure = json.loads((failed_root / "failure.json").read_text())
    manifest = read_ref(failed_root, failure["resume_manifest"])
    active = checkpoint(failed_root, manifest["checkpoints"][POLICIES[0] + ".recovery"])
    assert active["completed_updates"] == 1
    assert active["optimizer"]["state"]
    resumed_root = tmp_path / "resumed_recovery"
    result = run_scientific_workflow(**arguments(export, resumed_root, recovery_updates=2, interval=1),
        **manifest_arguments(failed_root, failure["resume_manifest"]))
    assert_endpoints_equal(full_root, full_result, resumed_root, result)
    assert result["compute"]["replayed_confirmed_updates"] == 0
    assert result["current_attempt_compute"]["confirmed_updates"] == 5
    assert result["compute"]["stages"][POLICIES[0] + ".pilot.validation_replay"]["observed_encoder_forwards"] == 5


def test_completed_manifest_revalidates_all_metrics_without_training(export, full, tmp_path):
    old_root, old = full
    root = tmp_path / "complete_resume"
    result = run_scientific_workflow(**arguments(export, root), **manifest_arguments(old_root, old["resume_manifest"]))
    assert_endpoints_equal(old_root, old, root, result)
    assert result["current_attempt_compute"]["confirmed_updates"] == 0
    assert result["current_attempt_compute"]["observed_encoder_forwards"] == 25
    assert result["compute"]["confirmed_updates"] == 6


@pytest.mark.parametrize("change,error", [
    (lambda p: p.update(phase=True), "phase_invalid"),
    (lambda p: p["checkpoints"]["baseline"].update(path="../outside.pt"), "artifact_path_invalid"),
    (lambda p: p["binding"]["source"].update(state_fingerprint="0" * 64), "resume_binding_mismatch"),
    (lambda p: p["operations"][1]["row"]["counts"].update(ssl_samples=-999), "count"),
    (lambda p: p["operations"][1].update(replay=True), "replay_accounting_invalid"),
])
def test_rehashed_false_manifest_is_rejected(export, full, tmp_path, change, error):
    old_root, old = full
    payload = read_ref(old_root, old["resume_manifest"])
    change(payload)
    # New manifest alongside original artifact paths, leaving original bytes intact.
    forged = old_root / ("forged-" + tmp_path.name + ".json")
    kwargs = reseal(payload, forged)
    with pytest.raises((ValueError, TypeError), match=error):
        run_scientific_workflow(**arguments(export, tmp_path / "rejected"), **kwargs)
    assert not (tmp_path / "rejected/result.json").exists()


def test_rehashed_false_evaluation_recomputed_and_rejected(export, full, tmp_path):
    old_root, old = full
    imported = tmp_path / "imported"
    imported.mkdir()
    payload = read_ref(old_root, old["resume_manifest"])
    for group in ("checkpoints", "evaluations"):
        for ref in payload[group].values():
            path = imported / ref["path"]
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((old_root / ref["path"]).read_bytes())
    ref = payload["evaluations"]["baseline"]
    path = imported / ref["path"]
    report = json.loads(path.read_text())
    task = next(iter(report["theory"]["per_head"]))
    report["theory"]["per_head"][task]["accuracy"] = 0.123456
    raw = json.dumps(report).encode()
    path.write_bytes(raw)
    ref.update(sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    kwargs = reseal(payload, imported / "false-metrics.json")
    with pytest.raises(ValueError, match="resume_evaluation_replay_mismatch"):
        run_scientific_workflow(**arguments(export, tmp_path / "rejected"), **kwargs)
    failure = json.loads((tmp_path / "rejected/failure.json").read_text())
    assert failure["current_attempt_compute"]["observed_encoder_forwards"] == 5
    assert not (tmp_path / "rejected/result.json").exists()


def test_symlink_artifact_is_rejected_before_checkpoint_load(export, full, tmp_path):
    old_root, old = full
    imported = tmp_path / "imported"
    (imported / "checkpoints").mkdir(parents=True)
    payload = read_ref(old_root, old["resume_manifest"])
    ref = payload["checkpoints"]["baseline"]
    (imported / ref["path"]).symlink_to(old_root / ref["path"])
    kwargs = reseal(payload, imported / "symlink.json")
    with pytest.raises(ValueError, match="symlink_artifact_denied"):
        run_scientific_workflow(**arguments(export, tmp_path / "rejected"), **kwargs)


def test_existing_attempt_directory_not_modified(export, full):
    root, _ = full
    before = (root / "result.json").read_bytes()
    with pytest.raises(FileExistsError, match="fresh_attempt"):
        run_scientific_workflow(**arguments(export, root))
    assert (root / "result.json").read_bytes() == before


def test_scientific_budget_and_source_guard_before_factory(export, tmp_path):
    args = arguments(export, tmp_path / "bad")
    args["fixture_mode"] = False
    args["warmstart_factory"] = lambda: pytest.fail("must reject before checkpoint reader")
    with pytest.raises(ValueError, match="fixed_scientific_protocol_required"):
        run_scientific_workflow(**args)
    assert not (tmp_path / "bad").exists()


def test_runtime_drift_fails_closed_and_no_success_result(export, tmp_path):
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    def drift(event):
        if event["kind"] == "training" and event["receipt"]["stage"] == "post_step_validation":
            torch.use_deterministic_algorithms(not before)
    try:
        with pytest.raises(ValueError, match="runtime_drift"):
            run_scientific_workflow(**arguments(export, tmp_path / "drift"), event_sink=drift)
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn)
    failure = json.loads((tmp_path / "drift/failure.json").read_text())
    counts = failure["compute"]["stages"][POLICIES[0] + ".pilot"]
    assert counts["confirmed_updates"] == 0 and counts["observed_encoder_forwards"] == 3
    assert counts["optimizer_outcomes"]["applied"] == 1
    assert not (tmp_path / "drift/result.json").exists()


def test_bootstrap_failure_has_fsynced_start_and_unverified_source(export, tmp_path):
    root = tmp_path / "bootstrap"
    args = arguments(export, root)
    def fail():
        receipts = [json.loads(line) for line in (root / "workflow-receipts.jsonl").read_text().splitlines()]
        assert receipts[0]["status"] == "started"
        assert receipts[0]["source_validation"] == "pending"
        raise ValueError("synthetic source tensor invalid")
    args["warmstart_factory"] = fail
    with pytest.raises(ValueError, match="source tensor invalid"):
        run_scientific_workflow(**args)
    failure = json.loads((root / "failure.json").read_text())
    assert failure["bootstrap_stage"] == "warmstart_and_initial_harness"
    assert failure["source_validation"] == "pending_or_failed"
    assert failure["binding_fingerprint"] is None
    assert failure["compute"]["confirmed_updates"] == failure["compute"]["observed_encoder_forwards"] == 0
    assert failure["resume_manifest"] is None
    assert not (root / "result.json").exists()


def test_callback_after_confirmation_retains_row_and_replays_from_prior_checkpoint(export, full, tmp_path):
    root = tmp_path / "callback"
    def fail(event):
        if event["kind"] == "training" and event["receipt"]["stage"] == "completed":
            raise KeyboardInterrupt("synthetic journal consumer interruption")
    with pytest.raises(KeyboardInterrupt):
        run_scientific_workflow(**arguments(export, root), event_sink=fail)
    failure = json.loads((root / "failure.json").read_text())
    op = failure["last_operation"]
    assert op["row"] is not None and op["receipt"]["confirmed_updates"] == 1
    assert op["receipt"]["optimizer_outcome"] == "applied"
    assert failure["compute"]["confirmed_updates"] == 1
    manifest = read_ref(root, failure["resume_manifest"])
    assert checkpoint(root, manifest["checkpoints"][POLICIES[0] + ".pilot"])["completed_updates"] == 0
    resumed_root = tmp_path / "callback_resumed"
    result = run_scientific_workflow(**arguments(export, resumed_root), **manifest_arguments(root, failure["resume_manifest"]))
    assert_endpoints_equal(*full, resumed_root, result)
    assert result["compute"]["replayed_confirmed_updates"] == 1


def test_nonfinite_adam_slot_is_applied_but_unconfirmed(export, tmp_path, monkeypatch):
    original = torch.optim.Adam.step
    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        next(iter(self.state.values()))["exp_avg_sq"].fill_(float("inf"))
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", corrupt)
    root = tmp_path / "nonfinite"
    with pytest.raises(ValueError, match="nonfinite_optimizer_state"):
        run_scientific_workflow(**arguments(export, root))
    failure = json.loads((root / "failure.json").read_text())
    op = failure["last_operation"]
    assert op["receipt"]["optimizer_outcome"] == "applied"
    assert op["receipt"]["observed_encoder_forwards"] == 3
    assert op["row"] is None and failure["compute"]["confirmed_updates"] == 0
    assert not (root / "result.json").exists()
    assert not (root / ("checkpoints/" + POLICIES[0] + ".pilot-000001.pt")).exists()


@pytest.mark.parametrize("field,value,error", [
    ("observed_encoder_forwards", 0, "optimizer_receipt_stage_invalid"),
    ("optimizer_outcome", "unknown", "optimizer_receipt_stage_invalid"),
    ("stage", "invented", "training_receipt_invalid"),
])
def test_rehashed_false_attempt_receipts_are_rejected(export, full, tmp_path, field, value, error):
    root, result = full
    payload = read_ref(root, result["resume_manifest"])
    payload["operations"][1]["receipt"][field] = value
    kwargs = reseal(payload, root / ("false-receipt-" + tmp_path.name + ".json"))
    with pytest.raises(ValueError, match=error):
        run_scientific_workflow(**arguments(export, tmp_path / "rejected"), **kwargs)


def test_duplicate_json_key_rejected_before_resume_artifact_reader(export, full, tmp_path):
    root, result = full
    payload = read_ref(root, result["resume_manifest"])
    raw = json.dumps(payload).encode()
    raw = raw[:-1] + b',"phase":9}'
    path = tmp_path / "duplicate.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate_json_key"):
        run_scientific_workflow(**arguments(export, tmp_path / "rejected"), resume_manifest=path,
            expected_resume_sha256=sha256(raw).hexdigest(), expected_resume_size_bytes=len(raw))


@pytest.mark.parametrize("forwards,batches,error", [(0, 0, "validation_receipt_invalid"),
                                                      (1, 1, "evaluation_receipt_missing_or_inconsistent")])
def test_rehashed_completed_validation_compute_cannot_be_zeroed(export, full, tmp_path, forwards, batches, error):
    root, result = full
    payload = read_ref(root, result["resume_manifest"])
    payload["operations"][0]["receipt"].update(observed_encoder_forwards=forwards, confirmed_batches=batches)
    kwargs = reseal(payload, root / ("false-eval-count-" + tmp_path.name + ".json"))
    with pytest.raises(ValueError, match=error):
        run_scientific_workflow(**arguments(export, tmp_path / "rejected"), **kwargs)
