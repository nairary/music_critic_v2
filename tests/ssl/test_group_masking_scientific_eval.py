"""Synthetic CPU evidence for fixed-membership scientific evaluation mechanics."""
from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS, CorrectedValidationAccumulator, align_target_sidecars_after_prediction,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    PAPER_DEFINED_JOINT_COMPONENTS, TASK_BY_ID, get_vocabulary,
)
from music_critic.ssl.bounded_fixture import build_phase7a_bounded_fixture
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.group_masking_fixture import build_group_pilot_fixture
from music_critic.ssl.group_masking_pilot import GroupPilotConfig, POLICIES, PilotBatch, tree_fingerprint
from music_critic.ssl.group_masking_scientific_eval import (
    _metric_alignment, compare_theory_endpoints, evaluate_scientific_validation,
    validation_batch_binding, validation_policy_bindings, ValidationResourceLimits,
)
from music_critic.ssl.objective import anti_collapse_diagnostics
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness
from music_critic.ssl.theory_warmstart import TheoryWarmstartSSLModel, export_theory_warmstart_state, load_theory_warmstart


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def export():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        return export_theory_warmstart_state(CorrectedAnalysisGNNModel())


@pytest.fixture
def harness(export):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return TheoryRetentionHarness(load_theory_warmstart(TheoryWarmstartSSLModel(), export))


@pytest.fixture(scope="module")
def batches():
    source = build_group_pilot_fixture()
    # The tiny generated fixture is not a corpus TRAIN/VALIDATION observation.
    raws = [collate_ssl_samples((sample,)) for sample in build_phase7a_bounded_fixture().raw_samples("train")[:2]]
    return tuple(PilotBatch(raw, raw, (source.sidecars[index],), source.class_weights)
                 for index, raw in enumerate(raws))


def run(harness, batches, **kwargs):
    config = GroupPilotConfig()
    arguments = {"batch_factory": lambda: iter(batches), "config": config,
        "expected_membership": tuple((dataset, piece) for batch in batches
            for dataset, piece in zip(batch.ssl.dataset_ids, batch.ssl.piece_ids)),
        "expected_batch_bindings": tuple(validation_batch_binding(batch, config) for batch in batches)}
    arguments.update(kwargs)
    return evaluate_scientific_validation(harness, **arguments)


def reseal(sidecar):
    sidecar.pop("fingerprint", None)
    sidecar["fingerprint"] = fingerprint(sidecar)
    return sidecar


def set_mask(sidecar, task):
    for entity in sidecar["entities"]:
        if task in entity["targets"]:
            entity["targets"][task].update(available=False, masked=True, canonical_value=None,
                source_value=None, missing_reason="synthetic_missing")
    return reseal(sidecar)


def task_module(harness, task):
    heads = harness.theory_model.task_heads
    index = next(index for index, spec in enumerate(heads.specs) if spec.task_id == task)
    return tuple(heads.heads.values())[index]


def test_full_membership_matches_reference_and_exact_streaming_aggregates(harness, batches, monkeypatch):
    captured = {policy: [] for policy in POLICIES}
    for policy, name in zip(POLICIES, ("forward", "forward_hierarchy")):
        original = getattr(harness.ssl_model, name)
        def wrapped(*args, _original=original, _policy=policy, **kwargs):
            output = _original(*args, **kwargs)
            captured[_policy].append(output)
            return output
        monkeypatch.setattr(harness.ssl_model, name, wrapped)
    receipts = []
    result = run(harness, batches, on_receipt=receipts.append)
    assert result["observed_samples"] == result["confirmed_batches"] == 2
    assert result["observed_encoder_forwards"] == 10
    assert result["optimizer_updates"] == 0
    assert receipts[0]["status"] == "started" and receipts[-1]["stage"] == "completed"
    assert all(receipt["optimizer_outcome"] == "not_applied" for receipt in receipts)
    assert all(len(result["ssl"][policy]["masks"]) == 2 for policy in POLICIES)
    accumulator = CorrectedValidationAccumulator(batches[0].class_weights)
    harness.eval()
    with torch.no_grad():
        for batch in batches:
            raw = batch.theory.raw_graph_batch
            output = harness.theory_model(raw)
            alignment = align_target_sidecars_after_prediction(output, raw, batch.sidecars)
            accumulator.update(output, alignment, sidecars=batch.sidecars)
    expected = accumulator.finalize()
    assert tuple(result["theory"]["per_head"]) == ACTIVE_HEADS
    for task in ACTIVE_HEADS:
        measured = dict(result["theory"]["per_head"][task])
        assert measured.pop("coverage") == {"source_rows": 2, "valid_metric_rows": 2,
            "supported_loss_rows": 2, "masked_rows": 0, "unsupported_train_rows": 0,
            "alignment_failure_rows": 0, "coverage": 1.0}
        assert measured == expected["per_head"][task]
    for policy, outputs in captured.items():
        aggregate = result["ssl"][policy]
        assert len(outputs) == 2
        for level in ("note", "bar", "song"):
            losses = [out.note_loss if level == "note" else getattr(out, level + "_latent").loss for out in outputs]
            numerator = sum(float(loss.numerator) for loss in losses)
            denominator = sum(loss.denominator for loss in losses)
            assert aggregate["components"][level]["numerator"] == numerator
            assert aggregate["components"][level]["denominator"] == denominator
            assert aggregate["components"][level]["mean"] == numerator / denominator
            if level == "note":
                target = torch.cat([out.targets.note.index_select(0, out.selected_global_note_indices) for out in outputs])
                prediction = torch.cat([torch.stack(out.decoder_predictions).mean(0) for out in outputs])
            else:
                target = torch.cat([getattr(out, level + "_latent").target for out in outputs])
                prediction = torch.cat([getattr(out, level + "_latent").prediction for out in outputs])
            direct = anti_collapse_diagnostics(target, prediction).to_dict()
            actual = aggregate["collapse"][level]
            assert actual["aggregation_scope"] == "streaming_aggregate"
            assert actual["row_count"] == target.shape[0]
            for key in ("target_embedding_variance", "prediction_embedding_variance", "target_mean_norm",
                        "prediction_mean_norm", "target_mean_off_diagonal_cosine", "prediction_mean_off_diagonal_cosine"):
                assert actual[key] == pytest.approx(direct[key], rel=1e-5, abs=1e-7)
        assert aggregate["weighted_total"] == sum(aggregate["weights"][level] * aggregate["components"][level]["mean"]
                                                   for level in ("note", "bar", "song"))


@pytest.mark.parametrize("inventory_case", ["absent", "empty", "populated"])
def test_train_tuple_inventory_availability_preserves_reference_metrics(harness, batches, inventory_case):
    # Two distinct gold joint tuples make partial seen/unseen membership observable.
    sidecar = deepcopy(batches[1].sidecars[0])
    quality = get_vocabulary(TASK_BY_ID["quality"].vocabulary_id).labels[2]
    for entity in sidecar["entities"]:
        if "quality" in entity["targets"]:
            entity["targets"]["quality"].update(canonical_value=quality, source_value=quality)
    changed = (batches[0], replace(batches[1], sidecars=(reseal(sidecar),)))
    first_tuple = tuple(get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[1]
                        for task in PAPER_DEFINED_JOINT_COMPONENTS)
    inventory = {"absent": None, "empty": set(), "populated": {first_tuple}}[inventory_case]
    result = run(harness, changed, train_seen_tuples=inventory)
    theory = result["theory"]
    assert theory["train_seen_tuple_inventory"] == {
        "available": inventory is not None,
        "tuple_count": None if inventory is None else len(inventory),
        "unavailable_reason": "train_seen_tuple_inventory_unavailable" if inventory is None else None,
    }
    assert result["train_seen_tuple_binding"] == (None if inventory is None else tree_fingerprint(sorted(inventory)))
    reference = CorrectedValidationAccumulator(changed[0].class_weights, train_seen_tuples=inventory)
    harness.eval()
    with torch.no_grad():
        for batch in changed:
            raw = batch.theory.raw_graph_batch
            output = harness.theory_model(raw)
            alignment = align_target_sidecars_after_prediction(output, raw, batch.sidecars)
            reference.update(output, alignment, sidecars=batch.sidecars)
    partition_fields = {"seen_tuple_joint_accuracy", "seen_tuple_support", "unseen_tuple_joint_accuracy",
                        "unseen_tuple_support", "unseen_tuple_count"}
    expected = reference.finalize()
    assert theory["corrected_joint_support"] == 2
    assert tuple(theory["per_head"]) == ACTIVE_HEADS
    for field, value in expected.items():
        if field == "per_head":
            for task in ACTIVE_HEADS:
                measured = dict(theory[field][task])
                measured.pop("coverage")
                assert measured == value[task]
        elif field in partition_fields and inventory is None:
            assert theory[field] is None
        else:
            assert theory[field] == value
    if inventory_case == "empty":
        assert (theory["seen_tuple_support"], theory["unseen_tuple_support"], theory["unseen_tuple_count"]) == (0, 2, 2)
        assert theory["seen_tuple_joint_accuracy"] is None
        assert theory["unseen_tuple_joint_accuracy"] is not None
    elif inventory_case == "populated":
        assert (theory["seen_tuple_support"], theory["unseen_tuple_support"], theory["unseen_tuple_count"]) == (1, 1, 1)
        assert theory["seen_tuple_joint_accuracy"] is not None
        assert theory["unseen_tuple_joint_accuracy"] is not None


def test_train_tuple_inventory_is_snapshotted_before_reader_callbacks(harness, batches):
    first_tuple = tuple(get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[1]
                        for task in PAPER_DEFINED_JOINT_COMPONENTS)
    inventory = {first_tuple}
    expected_binding = tree_fingerprint(sorted(inventory))
    def factory():
        inventory.clear()
        yield from batches
    result = run(harness, batches, train_seen_tuples=inventory, batch_factory=factory)
    assert inventory == set()
    assert result["train_seen_tuple_binding"] == expected_binding
    assert result["theory"]["train_seen_tuple_inventory"]["tuple_count"] == 1
    assert result["theory"]["seen_tuple_support"] == 2
    assert result["theory"]["unseen_tuple_support"] == 0


def test_comparison_rejects_absent_versus_explicit_empty_train_inventory(harness, batches):
    baseline = run(harness, batches[:1])
    endpoint = run(harness, batches[:1], train_seen_tuples=set())
    assert baseline["theory"]["per_head"] == endpoint["theory"]["per_head"]
    with pytest.raises(ValueError, match="comparison_binding_mismatch"):
        compare_theory_endpoints(baseline, endpoint, endpoint)


def test_missing_and_unsupported_masks_keep_all_heads_and_validation_only_classes(harness, batches):
    weights = deepcopy(batches[0].class_weights)
    weights.supported["root"][1] = False
    changed = []
    for index, batch in enumerate(batches):
        sidecar = set_mask(deepcopy(batch.sidecars[0]), "quality")
        if index == 0:
            sidecar = set_mask(sidecar, "root")
        changed.append(replace(batch, sidecars=(sidecar,), class_weights=weights))
    result = run(harness, changed)
    root = result["theory"]["per_head"]["root"]
    assert root["support"] == 1 and root["validation_only_class_count"] == 1
    assert root["macro_f1_observed_validation_classes"] is not None
    assert root["coverage"] == {"source_rows": 2, "valid_metric_rows": 1, "supported_loss_rows": 0,
        "masked_rows": 1, "unsupported_train_rows": 1, "alignment_failure_rows": 0, "coverage": .5}
    quality = result["theory"]["per_head"]["quality"]
    assert quality["support"] == 0 and quality["accuracy"] is None
    assert quality["macro_f1_observed_validation_classes"] is None
    assert quality["per_class"] and all(row["support"] == 0 for row in quality["per_class"])
    assert quality["coverage"]["masked_rows"] == 2


def test_all_theory_targets_missing_remains_explicitly_unavailable(harness, batches):
    sidecar = deepcopy(batches[0].sidecars[0])
    for task in ACTIVE_HEADS:
        set_mask(sidecar, task)
    result = run(harness, (replace(batches[0], sidecars=(sidecar,)),))
    assert result["observed_encoder_forwards"] == 5
    assert result["theory"]["corrected_primary_macro_score"] is None
    for row in result["theory"]["per_head"].values():
        assert row["support"] == 0 and row["accuracy"] is None
        assert row["masked_cross_entropy"] is None
        assert row["coverage"]["masked_rows"] == 1


def test_nonfinite_ssl_predictions_fail_with_observed_forward_receipts(harness, batches, monkeypatch):
    original = harness.ssl_model.forward
    def corrupted(*args, **kwargs):
        output = original(*args, **kwargs)
        return replace(output, decoder_predictions=tuple(value * float("nan") for value in output.decoder_predictions))
    monkeypatch.setattr(harness.ssl_model, "forward", corrupted)
    receipts = []
    with pytest.raises(ValueError, match="nonfinite_output"):
        run(harness, batches, on_receipt=receipts.append)
    assert receipts[-1]["status"] == "failed"
    assert receipts[-1]["stage"] == "ssl_output_validation"
    assert receipts[-1]["observed_encoder_forwards"] == 3
    assert receipts[-1]["confirmed_batches"] == 0


def test_external_model_mutation_cannot_publish_a_successful_result(harness, batches):
    receipts = []
    def mutate(*_):
        next(harness.parameters()).add_(.01)
    handle = harness.theory_model.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match="model_state_changed"):
            run(harness, batches[:1], on_receipt=receipts.append)
    finally:
        handle.remove()
    assert receipts[-1]["status"] == "failed"
    assert receipts[-1]["observed_encoder_forwards"] == 5


@pytest.mark.parametrize("split", ["train", "test", "TEST"])
def test_split_denied_before_reader(harness, batches, split):
    def forbidden():
        pytest.fail("reader must remain unopened")
    with pytest.raises(ValueError, match="validation_only"):
        run(harness, batches, split=split, batch_factory=forbidden)


@pytest.mark.parametrize("failure", ["hash", "order", "missing", "extra"])
def test_membership_and_batch_binding_fail_closed(harness, batches, failure):
    receipts = []
    kwargs = {"on_receipt": receipts.append}
    if failure == "hash":
        kwargs["expected_batch_bindings"] = ("0" * 64,) * 2
    elif failure == "order":
        kwargs["expected_membership"] = tuple((batch.ssl.dataset_ids[0], batch.ssl.piece_ids[0]) for batch in reversed(batches))
    elif failure == "missing":
        kwargs["batch_factory"] = lambda: iter(batches[:1])
    else:
        kwargs["batch_factory"] = lambda: iter((*batches, batches[0]))
    with pytest.raises(ValueError, match="batch_binding_mismatch|membership"):
        run(harness, batches, **kwargs)
    assert receipts[-1]["status"] == "failed"
    assert receipts[-1]["observed_encoder_forwards"] == {"hash": 0, "order": 0, "missing": 5, "extra": 10}[failure]


def test_nonfinite_logits_on_masked_head_fail_before_metric_success(harness, batches):
    changed = [replace(batch, sidecars=(set_mask(deepcopy(batch.sidecars[0]), "quality"),)) for batch in batches]
    receipts = []
    handle = task_module(harness, "quality").register_forward_hook(lambda _module, _args, output: output * float("inf"))
    try:
        with pytest.raises(ValueError, match="nonfinite_output"):
            run(harness, changed, on_receipt=receipts.append)
    finally:
        handle.remove()
    assert receipts[-1]["stage"] == "theory_output_validation"
    assert receipts[-1]["observed_encoder_forwards"] == 1
    assert receipts[-1]["confirmed_batches"] == 0


def test_runtime_drift_after_forward_is_observed_and_no_success_published(harness, batches):
    before = torch.backends.cudnn.benchmark
    receipts = []
    handle = harness.theory_model.register_forward_hook(lambda *_: setattr(torch.backends.cudnn, "benchmark", not before))
    try:
        with pytest.raises(ValueError, match="runtime_drift"):
            run(harness, batches, on_receipt=receipts.append)
    finally:
        handle.remove()
        torch.backends.cudnn.benchmark = before
    assert receipts[-1]["status"] == "failed"
    assert receipts[-1]["observed_encoder_forwards"] == 1


@pytest.mark.parametrize("fail", [False, True])
def test_modes_rng_state_and_existing_gradients_are_preserved(harness, batches, fail):
    harness.train()
    task_module(harness, "root").eval()
    modes = [module.training for module in harness.modules()]
    parameter = next(harness.parameters())
    parameter.grad = torch.ones_like(parameter)
    gradient = parameter.grad.clone()
    torch_rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    state = model_state_fingerprint(harness)
    def factory():
        random.random()
        np.random.rand()
        torch.rand(1)
        if fail:
            raise KeyboardInterrupt("synthetic interruption")
        yield from batches
    receipts = []
    if fail:
        with pytest.raises(KeyboardInterrupt):
            run(harness, batches, batch_factory=factory, on_receipt=receipts.append)
        assert receipts[-1]["status"] == "failed" and receipts[-1]["observed_encoder_forwards"] == 0
    else:
        run(harness, batches, batch_factory=factory, on_receipt=receipts.append)
    assert modes == [module.training for module in harness.modules()]
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    current_numpy_rng = np.random.get_state()
    assert numpy_rng[0] == current_numpy_rng[0]
    assert np.array_equal(numpy_rng[1], current_numpy_rng[1])
    assert numpy_rng[2:] == current_numpy_rng[2:]
    assert state == model_state_fingerprint(harness)
    assert torch.equal(gradient, parameter.grad)


def test_external_runtime_validator_runs_before_reader(harness, batches):
    receipts = []
    def reject():
        raise ValueError("constructor_runtime_drift")
    def forbidden():
        pytest.fail("reader opened before external runtime validation")
    with pytest.raises(ValueError, match="constructor_runtime_drift"):
        run(harness, batches, batch_factory=forbidden, runtime_validator=reject, on_receipt=receipts.append)
    assert receipts[-1]["observed_encoder_forwards"] == 0


def test_comparison_absolute_deltas_and_unavailable_support(harness, batches):
    baseline = run(harness, batches[:1])
    endpoint, recovered = deepcopy(baseline), deepcopy(baseline)
    for task in ACTIVE_HEADS:
        for report, accuracy, ce in ((baseline, .8, .2), (endpoint, .5, .7), (recovered, .7, .3)):
            head = report["theory"]["per_head"][task]
            head.update(accuracy=accuracy, balanced_accuracy=accuracy,
                macro_f1_observed_validation_classes=accuracy, masked_cross_entropy=ce)
    for report in (baseline, endpoint, recovered):
        report["theory"]["per_head"]["quality"].update(support=0, accuracy=None,
            balanced_accuracy=None, macro_f1_observed_validation_classes=None, masked_cross_entropy=None)
    comparison = compare_theory_endpoints(baseline, endpoint, recovered)
    root = comparison["per_head"]["root"]["metrics"]
    assert root["accuracy"]["forgetting"] == pytest.approx(.3)
    assert root["accuracy"]["recovery_gain"] == pytest.approx(.2)
    assert root["accuracy"]["remaining_gap"] == pytest.approx(.1)
    assert root["masked_cross_entropy"]["forgetting"] == pytest.approx(.5)
    assert root["masked_cross_entropy"]["recovery_gain"] == pytest.approx(.4)
    assert comparison["per_head"]["quality"]["metrics"]["accuracy"]["forgetting"] is None
    endpoint["batch_bindings"] = ["0" * 64]
    with pytest.raises(ValueError, match="comparison_binding_mismatch"):
        compare_theory_endpoints(baseline, endpoint, recovered)


def test_metric_accumulator_empty_candidate_guard_preserves_missing_evidence(harness, batches):
    with torch.no_grad():
        harness.eval()
        output = harness.theory_model(batches[0].theory.raw_graph_batch)
    alignment = align_target_sidecars_after_prediction(output, batches[0].theory.raw_graph_batch, batches[0].sidecars)
    rows = alignment.heads["root"]
    heads = dict(alignment.heads)
    heads["root"] = replace(rows, valid_mask=torch.zeros_like(rows.valid_mask), masked_row_count=1)
    missing = replace(alignment, heads=heads)
    predictions = tuple(replace(prediction, logits=prediction.logits[:0],
        candidate_node_type_codes=prediction.candidate_node_type_codes[:0],
        global_entity_indices=prediction.global_entity_indices[:0], sample_indices=prediction.sample_indices[:0],
        candidate_counts_by_node_type=torch.zeros_like(prediction.candidate_counts_by_node_type))
        if prediction.task_id == "root" else prediction for prediction in output.predictions)
    empty_output = replace(output, predictions=predictions)
    safe = _metric_alignment(empty_output, missing)
    assert safe.heads["root"].values.numel() == 0
    assert missing.heads["root"].values.numel() == 1 and missing.heads["root"].masked_row_count == 1
    accumulator = CorrectedValidationAccumulator(batches[0].class_weights)
    accumulator.update(empty_output, safe, sidecars=batches[0].sidecars)
    assert accumulator.finalize()["per_head"]["root"]["support"] == 0
    with pytest.raises(ValueError, match="valid_target_without_raw_candidates"):
        _metric_alignment(empty_output, alignment)


def piece_with_note_count(batch, count):
    piece = build_phase7a_bounded_fixture().train_pieces[0]
    piece = replace(piece, notes=piece.notes[:count])
    graph = build_raw_graph(piece)
    raw = collate_ssl_samples((SSLRawSample(graph, graph_fingerprint(graph), piece.dataset_name, piece.piece_id),))
    return replace(batch, ssl=raw, theory=raw)


def test_unavailable_beat_policy_keeps_theory_and_random_full_membership(harness, batches):
    single_note = piece_with_note_count(batches[0], 1)
    masks, availability = validation_policy_bindings(single_note.ssl, GroupPilotConfig())
    assert masks[POLICIES[0]] is not None and masks[POLICIES[1]] is None
    assert availability[POLICIES[1]]["reason"]["code"] == "fewer_than_two_pitched_notes"
    report = run(harness, (single_note, batches[1]))
    assert report["observed_samples"] == 2 and report["observed_encoder_forwards"] == 8
    assert report["theory"]["per_head"]["root"]["support"] == 2
    assert report["ssl"][POLICIES[0]]["available_pieces"] == 2
    beat = report["ssl"][POLICIES[1]]
    assert beat["membership_pieces"] == 2 and beat["available_pieces"] == beat["unavailable_pieces"] == 1
    assert beat["unavailable_reason_counts"] == {"fewer_than_two_pitched_notes": 1}
    assert beat["availability"][0]["piece"] == report["membership"][0]
    assert len(beat["masks"]) == 1
    altered = deepcopy(report)
    altered["ssl_coverage_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="comparison_binding_mismatch"):
        compare_theory_endpoints(report, report, altered)


def test_entire_ssl_family_unavailable_still_evaluates_theory(harness, batches):
    empty = piece_with_note_count(batches[0], 0)
    report = run(harness, (empty,))
    assert report["observed_samples"] == 1 and report["observed_encoder_forwards"] == 1
    assert report["theory"]["per_head"]["root"]["support"] == 1
    for policy in POLICIES:
        ssl = report["ssl"][policy]
        assert ssl["membership_pieces"] == ssl["unavailable_pieces"] == 1
        assert ssl["available_pieces"] == 0 and ssl["weighted_total"] is None
        assert ssl["unavailable_reason"] == "no_available_policy_pieces"
        assert all(value is None for value in ssl["collapse"].values())
        assert all(value["mean"] is None and value["denominator"] == 0 for value in ssl["components"].values())


def test_validation_resources_are_explicit_and_independent_of_training_limits(harness, batches):
    # Tiny TRAIN limit cannot remove a larger fixed VALIDATION record when
    # independently bound validation resources explicitly allow that record.
    config = GroupPilotConfig(max_notes_per_batch=1, max_nodes_per_batch=1, max_edges_per_batch=1)
    limits = ValidationResourceLimits(8192, 20000, 200000)
    binding = validation_batch_binding(batches[0], config, resource_limits=limits)
    report = run(harness, batches[:1], config=config, resource_limits=limits, expected_batch_bindings=(binding,))
    assert report["observed_samples"] == 1 and report["resource_limits"]["max_notes_per_piece"] == 8192
    with pytest.raises(ValueError, match="resource_bound_exceeded_no_member_exclusion"):
        validation_batch_binding(batches[0], config, resource_limits=ValidationResourceLimits(1, 1, 1))
    assert binding != validation_batch_binding(batches[0], config, resource_limits=ValidationResourceLimits(9000, 20000, 200000))


def test_untyped_mask_failure_is_not_converted_to_unavailable(harness, batches, monkeypatch):
    def broken(*_args, **_kwargs):
        raise ValueError("corrupt raw graph")
    monkeypatch.setattr("music_critic.ssl.group_masking_scientific_eval.prepare_hierarchy_mask_binding", broken)
    receipts = []
    with pytest.raises(ValueError, match="corrupt raw graph"):
        run(harness, batches, on_receipt=receipts.append)
    assert receipts[-1]["status"] == "failed" and receipts[-1]["observed_encoder_forwards"] == 0
