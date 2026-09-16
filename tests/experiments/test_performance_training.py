"""Exact same-runtime update equivalence on generated, source-free targets.

The only input fixture is explicitly synthetic. No corpus, historical class
weights, trained checkpoint or split loader is accessed. CUDA checks are
skipped unless a CUDA runtime is available; CPU success is not GPU evidence.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Batch

from music_critic.data import load_piece
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    CorrectedComponentSampler,
    CorrectedRuntimeConfig,
    CorrectedTrainingError,
    FrozenClassWeights,
    align_target_sidecars_after_prediction,
    attempt_applied_update,
    build_optimizer_scheduler,
    capture_rng_state,
    checkpoint_payload,
    gradient_norm,
    load_checkpoint,
    move_raw_graph_batch,
    restore_rng_state,
    save_checkpoint,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.analysisgnn.performance_training import (
    align_performance_target_sidecars,
    attempt_performance_update,
    performance_gradient_diagnostics,
    prepare_performance_class_weights,
)
from music_critic.experiments.analysisgnn.training_policy import CORRECTED_NO_TRANSPOSITION_PROFILE_ID
from music_critic.graph import build_raw_graph, validate_raw_graph_batch


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA runtime unavailable"
))]


def _fixture():
    source = load_piece(
        Path(__file__).resolve().parents[1] / "fixtures/data/canonical_piece_v2.json"
    )
    assert source.dataset_name == "synthetic"
    samples = []
    for index, shift in enumerate((-12, 12)):
        piece = replace(
            source, piece_id=f"piece:performance-synthetic-{index}",
            targets=(), annotations=(), quality_flags=(),
            notes=tuple(note if note.is_percussion else replace(note, pitch=note.pitch + shift)
                        for note in source.notes),
        )
        graph = build_raw_graph(piece)
        samples.append(graph)
    graph = Batch.from_data_list(samples)
    validate_raw_graph_batch(graph, sample_count=len(samples))
    sidecars = []
    for index in range(2):
        onset_id = graph["onset"].entity_id[index][0]
        numerator, denominator = map(int, onset_id.removeprefix("onset:").split("_"))
        entity_ids = {
            "harmonic_event": f"synthetic:harmony:{index}",
            "onset": f"synthetic:onset:{index}",
            "note": graph["note"].entity_id[index][0],
        }
        entities = []
        for level, entity_id in entity_ids.items():
            targets = {}
            for task in ACTIVE_HEADS:
                if TASK_BY_ID[task].prediction_level == level:
                    label = get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels[index]
                    targets[task] = {
                        "available": True, "masked": False,
                        "canonical_value": label, "source_value": label,
                        "missing_reason": None,
                        "provenance": {"origin": "synthetic_performance_fixture"},
                    }
            entities.append({
                "entity_type": level, "canonical_entity_id": entity_id,
                "onset_qn": {"num": numerator, "den": denominator}, "targets": targets,
            })
        sidecar = {
            "schema_version": "analysisgnn-source-native-target-sidecar-v1",
            "record_id": f"synthetic:record:{index}",
            "source_component_id": f"synthetic:component:{index}",
            "dialect": "dlc", "entities": entities,
            "relations": [{
                "relation": "harmonic_event_to_beat",
                "source_entity_id": entity_ids["harmonic_event"],
                "target_entity_id": graph["beat"].entity_id[index][0],
            }],
        }
        sidecar["fingerprint"] = fingerprint(sidecar)
        sidecars.append(sidecar)
    sizes = {task: TASK_BY_ID[task].class_count for task in ACTIVE_HEADS}
    weights = FrozenClassWeights(
        values={task: torch.linspace(0.5, 2.0, size) for task, size in sizes.items()},
        supported={task: torch.ones(size, dtype=torch.bool) for task, size in sizes.items()},
        payload_fingerprint=fingerprint({"origin": "synthetic_nonuniform", "sizes": sizes}),
    )
    weights.supported[ACTIVE_HEADS[-1]][0] = False
    return graph, tuple(sidecars), weights


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _equal(left, right)
    else:
        assert actual == expected


def _scaler():
    # The archived CUDA runtime predates torch.amp.GradScaler.
    factory = getattr(torch.amp, "GradScaler", None)
    return factory("cuda", enabled=False) if factory else torch.cuda.amp.GradScaler(enabled=False)


def _runtime(model, *, total_updates=8):
    optimizer, scheduler = build_optimizer_scheduler(model, total_updates=total_updates)
    return {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": _scaler()}


def _snapshot(runtime):
    return deepcopy({key: value.state_dict() for key, value in runtime.items()})


@pytest.mark.parametrize("device", _DEVICES)
def test_full_model_three_train_updates_have_exact_logits_loss_gradients_state_and_rng(device):
    graph, sidecars, weights = _fixture()
    graph = move_raw_graph_batch(graph, device)
    original_graph = deepcopy(graph.to_dict())
    original_sidecars = deepcopy(sidecars)
    resident = prepare_performance_class_weights(weights, device)
    with torch.random.fork_rng(devices=[] if device == "cpu" else [torch.cuda.current_device()]):
        torch.manual_seed(1729)
        reference = _runtime(CorrectedAnalysisGNNModel().to(device).train())
        performance = _runtime(deepcopy(reference["model"]))
        outputs = {}
        gradients = {"reference": {}, "performance": {}}
        handles = []
        for name, runtime in (("reference", reference), ("performance", performance)):
            handles.append(runtime["model"].register_forward_hook(
                lambda module, args, result, key=name: outputs.__setitem__(key, result)
            ))
            for parameter_name, parameter in runtime["model"].named_parameters():
                def capture(gradient, key=name, parameter_key=parameter_name):
                    gradients[key][parameter_key] = gradient.detach().clone()
                handles.append(parameter.register_hook(capture))
        stages = []

        @contextmanager
        def instrument(name):
            stages.append((name, "enter"))
            try:
                yield
            finally:
                stages.append((name, "exit"))

        try:
            for step in range(3):
                gradients["reference"].clear()
                gradients["performance"].clear()
                step_sidecars = deepcopy(sidecars)
                if step == 1:
                    for sidecar in step_sidecars:
                        sidecar["entities"][0]["targets"][ACTIVE_HEADS[0]]["masked"] = True
                if step == 2:
                    # Missing alignment and an unknown label remain masks with diagnostics.
                    step_sidecars[0]["relations"][0]["target_entity_id"] = "missing:synthetic"
                    onset_targets = step_sidecars[1]["entities"][1]["targets"]
                    onset_targets[next(iter(onset_targets))]["canonical_value"] = "unknown"
                before = capture_rng_state()
                expected = attempt_applied_update(
                    **reference, raw_graph_batch=graph, sidecars=step_sidecars,
                    shifts=(0, 0), class_weights=weights,
                )
                expected_rng = capture_rng_state()
                restore_rng_state(before)
                actual = attempt_performance_update(
                    **performance, raw_graph_batch=graph, sidecars=step_sidecars,
                    shifts=(0, 0), class_weights=resident, instrumentation=instrument,
                )
                _equal(capture_rng_state(), expected_rng)
                _equal(actual, expected)
                _equal(outputs["performance"].logits, outputs["reference"].logits)
                _equal(gradients["performance"], gradients["reference"])
                assert gradients["reference"]
                _equal({name: parameter.grad for name, parameter in performance["model"].named_parameters()},
                       {name: parameter.grad for name, parameter in reference["model"].named_parameters()})
                _equal(_snapshot(performance), _snapshot(reference))
                assert actual.applied and actual.finite_logits and actual.finite_gradients
                assert actual.loss.total is not None and actual.loss.total.dtype == torch.float32
                if step == 0:
                    assert not actual.loss.zero_valid_heads
        finally:
            for handle in handles:
                handle.remove()
        expected_stages = ["forward", "alignment", "loss", "diagnostics", "backward", "diagnostics", "optimizer"]
        assert stages == [(name, event) for _ in range(3) for name in expected_stages
                          for event in ("enter", "exit")]
        assert performance["scheduler"].last_epoch == 3
    _equal(graph.to_dict(), original_graph)
    assert sidecars == original_sidecars


@pytest.mark.parametrize("device", _DEVICES)
def test_alignment_metadata_views_preserve_exact_join_failures_and_output_identity(device):
    graph, sidecars, _ = _fixture()
    graph = move_raw_graph_batch(graph, device)
    with torch.random.fork_rng(devices=[] if device == "cpu" else [torch.cuda.current_device()]):
        torch.manual_seed(17)
        model = CorrectedAnalysisGNNModel().to(device).eval()
        with torch.no_grad():
            output = model(graph)
        original_predictions = output.predictions
        original_graph = deepcopy(graph.to_dict())
        for shifts in ((0, 0), (1, 11)):
            _equal(align_performance_target_sidecars(output, graph, sidecars, shifts=shifts),
                   align_target_sidecars_after_prediction(output, graph, sidecars, shifts=shifts))
        assert output.predictions is original_predictions
        assert all(row.logits.device.type == device for row in output.predictions)
        for broken_sidecars, shifts in (((), ()), (sidecars, (0,))):
            with pytest.raises(CorrectedTrainingError) as reference_error:
                align_target_sidecars_after_prediction(output, graph, broken_sidecars, shifts=shifts)
            with pytest.raises(CorrectedTrainingError) as performance_error:
                align_performance_target_sidecars(output, graph, broken_sidecars, shifts=shifts)
            assert str(performance_error.value) == str(reference_error.value)
        first = output.predictions[0]
        broken = replace(output, predictions=(replace(
            first, global_entity_indices=torch.full_like(first.global_entity_indices, 1000000)
        ), *output.predictions[1:]))
        for align in (align_target_sidecars_after_prediction, align_performance_target_sidecars):
            with pytest.raises(CorrectedTrainingError, match="candidate_metadata_mismatch"):
                align(broken, graph, sidecars)
        _equal(graph.to_dict(), original_graph)


@pytest.mark.parametrize("device", _DEVICES)
def test_resident_weights_are_owned_exact_constant_snapshots(device):
    _, _, weights = _fixture()
    source = deepcopy(weights)
    prepared = prepare_performance_class_weights(weights, device)
    assert prepared.payload_fingerprint == weights.payload_fingerprint
    for field in ("values", "supported"):
        for task in ACTIVE_HEADS:
            original = getattr(weights, field)[task]
            resident = getattr(prepared, field)[task]
            assert resident.device.type == device
            assert resident.dtype == original.dtype
            assert torch.equal(resident.cpu(), original)
            assert resident.to(device) is resident
            if device == "cpu":
                assert resident.data_ptr() != original.data_ptr()
    prepared.values[ACTIVE_HEADS[0]][0] = 17.0
    prepared.supported[ACTIVE_HEADS[0]][0] = False
    _equal(weights, source)


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_diagnostics_preserve_per_parameter_fp32_and_python_sum_order(device):
    model = nn.ParameterList([nn.Parameter(torch.zeros(7, device=device)) for _ in range(40)])
    assert performance_gradient_diagnostics(model) == (True, 0.0)
    for index, parameter in enumerate(model.parameters()):
        if index % 7:
            parameter.grad = torch.arange(7, device=device, dtype=torch.float32) * (10.0 ** (index % 12 - 6))
    finite, actual = performance_gradient_diagnostics(model)
    assert finite and actual == gradient_norm(model)
    next(model.parameters()).grad = torch.full((7,), float("inf"), device=device)
    assert performance_gradient_diagnostics(model) == (False, float("inf"))


@pytest.mark.parametrize("failure", ["forward", "gradient", "no_supervision"])
def test_failed_guards_match_reference_without_optimizer_or_scheduler_changes(failure):
    graph, sidecars, weights = _fixture()
    if failure == "no_supervision":
        for sidecar in sidecars:
            for entity in sidecar["entities"]:
                for state in entity["targets"].values():
                    state["masked"] = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        original = CorrectedAnalysisGNNModel().train()
        before_rng = capture_rng_state()
        errors = []
        end_rng = []
        for updater in (attempt_applied_update, attempt_performance_update):
            runtime = _runtime(deepcopy(original))
            before = _snapshot(runtime)
            handle = None
            if failure == "forward":
                handle = runtime["model"].task_heads.heads["task_00"].register_forward_hook(
                    lambda module, args, output: output * float("nan")
                )
            elif failure == "gradient":
                handle = next(runtime["model"].task_heads.heads["task_00"].parameters()).register_hook(
                    lambda gradient: torch.full_like(gradient, float("nan"))
                )
            restore_rng_state(before_rng)
            try:
                with pytest.raises(CorrectedTrainingError) as error:
                    updater(**runtime, raw_graph_batch=graph, sidecars=sidecars,
                            shifts=(0, 0), class_weights=weights)
            finally:
                if handle is not None:
                    handle.remove()
            errors.append(str(error.value))
            end_rng.append(capture_rng_state())
            _equal(_snapshot(runtime), before)
        assert errors[0] == errors[1]
        category = {"forward": "non_finite_forward", "gradient": "non_finite_gradient",
                    "no_supervision": "no_finite_supervision"}[failure]
        assert category in errors[0]
        _equal(end_rng[0], end_rng[1])


def test_frozen_checkpoint_resume_of_performance_path_matches_reference_uninterrupted(tmp_path):
    graph, sidecars, weights = _fixture()
    resident = prepare_performance_class_weights(weights, "cpu")
    config = CorrectedRuntimeConfig(
        profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
        batch_size=2, applied_update_budget=3,
    )
    components = {sidecar["source_component_id"]: (sidecar["record_id"],) for sidecar in sidecars}
    shifts = {sidecar["record_id"]: (0,) for sidecar in sidecars}

    def sampler():
        return CorrectedComponentSampler(components, shifts, profile_id=config.profile_id)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference = _runtime(CorrectedAnalysisGNNModel().train(), total_updates=3)
        interrupted = _runtime(deepcopy(reference["model"]), total_updates=3)
        reference_sampler, interrupted_sampler = sampler(), sampler()
        initial_rng = capture_rng_state()
        reference_losses = []
        reference_after_first = None
        for step in range(3):
            result = attempt_applied_update(
                **reference, raw_graph_batch=graph, sidecars=sidecars,
                shifts=(0, 0), class_weights=weights,
            )
            assert result.applied
            reference_sampler.advance_after_applied_update(draws=2)
            reference_losses.append(result)
            if step == 0:
                reference_after_first = (_snapshot(reference), capture_rng_state())
        final_rng = capture_rng_state()
        restore_rng_state(initial_rng)
        result = attempt_performance_update(
            **interrupted, raw_graph_batch=graph, sidecars=sidecars,
            shifts=(0, 0), class_weights=resident,
        )
        assert result.applied
        interrupted_sampler.advance_after_applied_update(draws=2)
        _equal(result, reference_losses[0])
        _equal((_snapshot(interrupted), capture_rng_state()), reference_after_first)
        checkpoint = tmp_path / "synthetic-performance-resume.ckpt"
        save_checkpoint(checkpoint, checkpoint_payload(
            **interrupted, sampler=interrupted_sampler, config=config,
            applied_update=1, best_primary_score=None, best_update=None,
            record_history=tuple(sidecar["record_id"] for sidecar in sidecars),
            shift_history=(0, 0),
        ))
        resumed = _runtime(CorrectedAnalysisGNNModel().train(), total_updates=3)
        resumed_sampler = sampler()
        # The frozen loader reads only bytes this test just generated.
        restored = load_checkpoint(
            checkpoint, **resumed, sampler=resumed_sampler, config=config,
        )
        assert restored["applied_update"] == 1 and resumed_sampler.position == 2
        _equal((_snapshot(resumed), capture_rng_state()), reference_after_first)
        for step in range(1, 3):
            result = attempt_performance_update(
                **resumed, raw_graph_batch=graph, sidecars=sidecars,
                shifts=(0, 0), class_weights=resident,
            )
            assert result.applied
            resumed_sampler.advance_after_applied_update(draws=2)
            _equal(result, reference_losses[step])
        _equal(_snapshot(resumed), _snapshot(reference))
        _equal(capture_rng_state(), final_rng)
        _equal(resumed_sampler.state_dict(), reference_sampler.state_dict())
        _equal({name: parameter.grad for name, parameter in resumed["model"].named_parameters()},
               {name: parameter.grad for name, parameter in reference["model"].named_parameters()})
        assert resumed_sampler.position == 6 and resumed["scheduler"].last_epoch == 3
