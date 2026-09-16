"""Actual synthetic pilot rows and malformed accounting/resume regressions."""
from copy import deepcopy
from hashlib import sha256

import pytest
import torch

from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.training_policy import PRIMARY_HEADS
from music_critic.ssl.group_masking_accounting import validate_successful_row
from music_critic.ssl.group_masking_fixture import build_group_pilot_fixture
from music_critic.ssl.group_masking_pilot import (
    POLICIES, GroupMaskingPilot, GroupPilotConfig, PilotBatch,
    bind_train_batch, tree_fingerprint,
)
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartSSLModel, export_theory_warmstart_state, load_theory_warmstart,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        return export_theory_warmstart_state(CorrectedAnalysisGNNModel()), build_group_pilot_fixture()


def _pilot(inputs, config=None, policy=POLICIES[0], batch=None):
    export, original_batch = inputs
    config, batch = config or GroupPilotConfig(updates=1), batch or original_batch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        warmstart = load_theory_warmstart(TheoryWarmstartSSLModel(config.ssl_config()), export)
    return GroupMaskingPilot(warmstart, config, policy=policy,
        schedule=(bind_train_batch(batch, config),) * config.updates,
        source_binding={"kind": "synthetic", "state_fingerprint": export["state_fingerprint"]})


@pytest.fixture(scope="module")
def successful(inputs):
    pilot = _pilot(inputs)
    row = pilot.step_from_factory(lambda: inputs[1])
    return pilot, row


def _validate(row, pilot, *, binding=None):
    validate_successful_row(row, config=pilot.config, policy=pilot.policy,
        scheduled_binding=pilot.schedule[0] if binding is None else binding, update_index=0)


def _set(row, path, value):
    destination = row
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("coefficients", [(0.7, 0.3), (1., 0.), (0., 1.)])
def test_real_rows_validate_for_all_active_paths(inputs, policy, coefficients):
    config = GroupPilotConfig(updates=1, lambda_ssl=coefficients[0], lambda_theory=coefficients[1])
    complete = inputs[1]
    batch = PilotBatch(complete.ssl if config.lambda_ssl else None,
        complete.theory if config.lambda_theory else None,
        complete.sidecars if config.lambda_theory else (),
        complete.class_weights if config.lambda_theory else None)
    pilot = _pilot(inputs, config, policy, batch)
    row = pilot.step_from_factory(lambda: batch)
    _validate(row, pilot)
    assert tree_fingerprint(row["batch_evidence"]) == row["batch_binding"] == pilot.schedule[0]


@pytest.mark.parametrize("coefficients", [(1., 0.), (0., 1.)])
def test_supplied_inactive_inputs_do_not_imply_execution(inputs, coefficients):
    config = GroupPilotConfig(updates=1, lambda_ssl=coefficients[0], lambda_theory=coefficients[1])
    pilot = _pilot(inputs, config)
    row = pilot.step_from_factory(lambda: inputs[1])
    _validate(row, pilot)
    assert all(evidence is not None for evidence in row["batch_evidence"]["accounting"].values())
    inactive = "theory" if config.lambda_ssl else "ssl"
    assert row["counts"][inactive + "_samples"] == 0


@pytest.mark.parametrize("masked_tasks", [("root",), PRIMARY_HEADS])
def test_real_missing_and_unsupported_head_rows_remain_distinct(inputs, masked_tasks):
    batch = deepcopy(inputs[1])
    for sidecar in batch.sidecars:
        for entity in sidecar["entities"]:
            for task, target in entity["targets"].items():
                if task in masked_tasks:
                    target.update(available=False, masked=True, canonical_value=None,
                                  missing_reason="synthetic_missing")
        sidecar["fingerprint"] = fingerprint({k: v for k, v in sidecar.items() if k != "fingerprint"})
    batch.class_weights.supported["note_degree"].fill_(False)
    config = GroupPilotConfig(updates=1, lambda_ssl=0., lambda_theory=1.)
    pilot = _pilot(inputs, config=config, batch=batch)
    row = pilot.step_from_factory(lambda: batch)
    _validate(row, pilot)
    for task in masked_tasks:
        assert row["theory_heads"][task] == {"weighted_ce": None, "valid": 0,
                                             "masked": 2, "unsupported": 0}
    assert row["theory_heads"]["note_degree"] == {
        "weighted_ce": None, "valid": 0, "masked": 0, "unsupported": 2}
    assert row["counts"]["theory_active_heads"] == 18 - len(masked_tasks) - 1


def test_repeated_many_to_one_targets_use_population_not_raw_node_cap(inputs):
    batch = deepcopy(inputs[1])
    repeats = batch.theory.node_count + 1
    for sidecar in batch.sidecars:
        harmonic = next(entity for entity in sidecar["entities"] if entity["entity_type"] == "harmonic_event")
        sidecar["entities"].extend(deepcopy(harmonic) for _ in range(repeats))
        sidecar["fingerprint"] = fingerprint({k: v for k, v in sidecar.items() if k != "fingerprint"})
    config = GroupPilotConfig(updates=1, lambda_ssl=0., lambda_theory=1.)
    pilot = _pilot(inputs, config=config, batch=batch)
    row = pilot.step_from_factory(lambda: batch)
    _validate(row, pilot)
    population = row["batch_evidence"]["accounting"]["theory_target_rows"]["root"]
    assert population == row["theory_heads"]["root"]["valid"] == 2 * (repeats + 1)
    assert population > batch.theory.node_count


@pytest.mark.parametrize("path,value,reason", [
    (("update",), True, "schedule_prefix"),
    (("update",), 2, "schedule_prefix"),
    (("encoder_forwards",), True, "forward_count"),
    (("encoder_forwards",), 1, "forward_count"),
    (("counts", "ssl_samples"), -999, "count_range"),
    (("counts", "ssl_samples"), True, "count_range"),
    (("counts", "ssl_samples"), 1, "batch_sample_count"),
    (("counts", "theory_samples"), 1, "batch_sample_count"),
    (("counts", "ssl_selected_notes"), 0, "selected_note_count"),
    (("counts", "theory_active_heads"), 19, "active_head_count"),
    (("counts", "theory_valid_rows"), 999, "head_accounting"),
    (("counts", "theory_masked_rows"), 1, "head_accounting"),
    (("counts", "theory_unsupported_rows"), 1, "head_accounting"),
    (("counts",), {}, "count_schema"),
    (("paired_masks",), {}, "paired_masks"),
    (("paired_masks", POLICIES[0]), [], "mask_sample_count"),
    (("paired_masks", POLICIES[1], 0, "piece"), ["elsewhere", "wrong"], "mask_identity"),
    (("paired_masks", POLICIES[0], 0, "plan_fingerprint"), "x" * 64, "mask_identity"),
    (("paired_masks", POLICIES[0], 0, "eligible_notes"), True, "mask_note_count"),
    (("paired_masks", POLICIES[0], 0, "hidden_notes"), 999, "mask_note_count"),
    (("paired_masks", POLICIES[0], 0, "hidden_fraction"), 0.5, "mask_density"),
    (("paired_masks", POLICIES[1], 0, "requested_fraction"), 0.6, "mask_density"),
    (("paired_masks", POLICIES[1], 0, "unit"), "bar", "mask_unit"),
    (("paired_masks", POLICIES[0], 0, "hidden_units"), 1, "independent_mask_count"),
    (("paired_masks", POLICIES[1], 0, "candidate_units"), 999, "mask_unit_count"),
    (("paired_masks", POLICIES[0], 0, "collateral_rows", "note"), 999, "collateral_count"),
    (("paired_masks", POLICIES[0], 0, "collateral_rows", "track"), 0, "collateral_count"),
    (("theory_heads",), {}, "head_schema"),
    (("theory_heads", "root", "valid"), True, "head_count"),
    (("theory_heads", "root", "masked"), -1, "head_count"),
    (("theory_heads", "root", "unsupported"), 1.5, "head_count"),
    (("theory_heads", "root", "weighted_ce"), None, "head_loss"),
    (("theory_heads", "root", "weighted_ce"), float("inf"), "head_loss"),
    (("theory_heads", "root", "weighted_ce"), -1., "head_loss"),
    (("theory_heads", "root", "weighted_ce"), 100., "head_loss_aggregate"),
    (("theory_loss",), None, "loss_invalid"),
    (("ssl_loss",), True, "loss_invalid"),
    (("ssl_loss",), 10 ** 400, "loss_invalid"),
    (("total_loss",), float("nan"), "total_loss"),
    (("collapse",), None, "collapse_schema"),
    (("collapse", "note", "row_count"), 1, "collapse_note_count"),
    (("collapse", "song", "row_count"), 1, "collapse_song_count"),
    (("collapse", "bar", "row_count"), 999999, "collapse_dimensions"),
    (("collapse", "song", "target_zero_norm_count"), True, "collapse_zero_count"),
    (("collapse", "bar", "prediction_embedding_variance"), -1., "collapse_metric"),
    (("collapse", "bar", "prediction_mean_norm"), float("inf"), "collapse_metric"),
    (("collapse", "song", "target_mean_off_diagonal_cosine"), 2., "collapse_pairwise"),
    (("collapse", "bar", "source_dtype"), "float16", "collapse_contract"),
    (("optimizer_gradient_parameter_indices",), [True], "gradient_inventory"),
    (("optimizer_gradient_parameter_indices",), [], "gradient_inventory"),
    (("optimizer_gradient_parameter_indices",), [1, 1], "gradient_inventory"),
])
def test_inconsistent_real_row_rejected(successful, path, value, reason):
    pilot, original = successful
    row = deepcopy(original)
    _set(row, path, value)
    with pytest.raises(ValueError, match=reason):
        _validate(row, pilot)


@pytest.mark.parametrize("path,value,reason", [
    (("content_fingerprint",), "bad", "content_digest"),
    (("accounting", "ssl", "note_counts"), [True, 18], "note_count"),
    (("accounting", "ssl", "note_counts"), [18], "note_count"),
    (("accounting", "theory", "identities"), [], "sample_count"),
    (("accounting", "ssl", "identities"), [["a"], ["b"]], "identity"),
    (("accounting", "ssl", "node_count"), 1, "resource_count"),
    (("accounting", "ssl", "edge_count"), -1, "resource_count"),
    (("accounting", "ssl", "edge_count"), True, "resource_count"),
    (("accounting", "theory_target_rows"), None, "target_population_schema"),
    (("accounting", "theory_target_rows"), {}, "target_population_schema"),
    (("accounting", "theory_target_rows", "root"), True, "target_population_count"),
    (("accounting", "theory_target_rows", "root"), -1, "target_population_count"),
])
def test_rebound_malformed_batch_accounting_rejected(successful, path, value, reason):
    pilot, original = successful
    row = deepcopy(original)
    _set(row["batch_evidence"], path, value)
    row["batch_binding"] = tree_fingerprint(row["batch_evidence"])
    with pytest.raises(ValueError, match=reason):
        _validate(row, pilot, binding=row["batch_binding"])


def test_batch_accounting_and_content_digest_are_schedule_bound(successful):
    pilot, original = successful
    for path, value in [(("content_fingerprint",), "0" * 64),
                        (("accounting", "ssl", "edge_count"), 0),
                        (("accounting", "theory_target_rows", "root"), 10 ** 9)]:
        row = deepcopy(original)
        _set(row["batch_evidence"], path, value)
        with pytest.raises(ValueError, match="batch_evidence_binding"):
            _validate(row, pilot)


def test_coherent_head_and_aggregate_population_inflation_rejected(successful):
    pilot, original = successful
    row = deepcopy(original)
    row["theory_heads"]["root"]["valid"] += 10 ** 9
    row["counts"]["theory_valid_rows"] += 10 ** 9
    with pytest.raises(ValueError, match="theory_head_population_mismatch"):
        _validate(row, pilot)


@pytest.mark.parametrize("supplied", [False, True])
def test_inactive_theory_population_schema_follows_supplied_batch(inputs, supplied):
    complete = inputs[1]
    batch = complete if supplied else PilotBatch(ssl=complete.ssl)
    config = GroupPilotConfig(updates=1, lambda_ssl=1., lambda_theory=0.)
    pilot = _pilot(inputs, config=config, batch=batch)
    row = pilot.step_from_factory(lambda: batch)
    _validate(row, pilot)
    population = row["batch_evidence"]["accounting"]["theory_target_rows"]
    assert (population is not None) is supplied
    row["batch_evidence"]["accounting"]["theory_target_rows"] = None if supplied else {}
    with pytest.raises(ValueError, match="target_population"):
        _validate(row, pilot)


@pytest.mark.parametrize("key", ["unknown", "missing"])
def test_successful_row_envelope_is_strict(successful, key):
    pilot, original = successful
    row = deepcopy(original)
    if key == "unknown":
        row["extra"] = None
    else:
        del row["counts"]
    with pytest.raises(ValueError, match="accounting_schema"):
        _validate(row, pilot)


@pytest.mark.parametrize("damage", ["negative_samples", "empty_masks", "wrong_head_count",
                                     "changed_batch_count", "coherent_population_inflation"])
def test_resume_rejects_semantically_bad_accounting_after_outer_rehash(successful, inputs, tmp_path, damage):
    pilot, _ = successful
    payload = pilot.checkpoint_payload()
    row = payload["rows"][0]
    if damage == "negative_samples":
        row["counts"]["ssl_samples"] = -999
    elif damage == "empty_masks":
        row["paired_masks"] = {}
    elif damage == "wrong_head_count":
        row["theory_heads"]["root"]["valid"] += 1
    elif damage == "coherent_population_inflation":
        row["theory_heads"]["root"]["valid"] += 10 ** 9
        row["counts"]["theory_valid_rows"] += 10 ** 9
    else:
        row["batch_evidence"]["accounting"]["ssl"]["edge_count"] = 0
    payload["payload_fingerprint"] = tree_fingerprint({k: v for k, v in payload.items() if k != "payload_fingerprint"})
    path = tmp_path / "invalid-accounting.pt"
    torch.save(payload, path)
    content = path.read_bytes()
    fresh = _pilot(inputs)
    before = tree_fingerprint(fresh.harness.state_dict())
    with pytest.raises(ValueError, match="group_pilot.resume_"):
        fresh.resume(path, expected_sha256=sha256(content).hexdigest(), expected_size_bytes=len(content))
    assert tree_fingerprint(fresh.harness.state_dict()) == before
    assert fresh.completed_updates == 0 and not fresh.optimizer.state
