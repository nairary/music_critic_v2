"""Tiny synthetic learning checks; not corpus or musical-quality evidence.

Two generated input patterns deliberately carry opposing artificial labels for
all 18 heads. A constant class prior cannot solve this check. The architecture,
forward, target alignment, loss and applied-update path are production code;
the uniform weights and short update budget are fixture choices.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import torch

from music_critic.data import load_piece
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    FrozenClassWeights,
    align_target_sidecars_after_prediction,
    attempt_applied_update,
    build_optimizer_scheduler,
    corrected_supervised_loss,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.analysisgnn.performance_replay import _scaler
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples


def _fixture():
    source = load_piece(
        Path(__file__).resolve().parents[1] / "fixtures/data/canonical_piece_v2.json"
    )
    assert source.dataset_name == "synthetic"
    samples = []
    for index, shift in enumerate((-12, 12)):
        piece = replace(
            source,
            piece_id=f"piece:learning-audit-{index}",
            targets=(), annotations=(), quality_flags=(),
            notes=tuple(
                note if note.is_percussion else replace(note, pitch=note.pitch + shift)
                for note in source.notes
            ),
        )
        graph = build_raw_graph(piece)
        samples.append(SSLRawSample(
            graph, graph_fingerprint(graph), piece.dataset_name, piece.piece_id,
        ))
    batch = collate_ssl_samples(samples)
    sidecars = []
    for index in range(2):
        graph = batch.raw_graph_batch
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
                if TASK_BY_ID[task].prediction_level != level:
                    continue
                labels = get_vocabulary(TASK_BY_ID[task].vocabulary_id).labels
                assert len(labels) >= 2
                targets[task] = {
                    "available": True, "masked": False,
                    "canonical_value": labels[index], "source_value": labels[index],
                    "missing_reason": None,
                    "provenance": {"origin": "synthetic_learning_check"},
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
        values={task: torch.ones(size) for task, size in sizes.items()},
        supported={task: torch.ones(size, dtype=torch.bool) for task, size in sizes.items()},
        payload_fingerprint=fingerprint({"origin": "synthetic_uniform", "sizes": sizes}),
    )
    return samples, batch, tuple(sidecars), weights


def _evaluate(model, batch, sidecars, weights):
    model.eval()
    with torch.no_grad():
        output = model(batch.raw_graph_batch)
        alignment = align_target_sidecars_after_prediction(output, batch.raw_graph_batch, sidecars)
        report = corrected_supervised_loss(output, alignment, weights)
        assert not alignment.diagnostics and not report.zero_valid_heads
        selected = {}
        correct = {}
        for task in ACTIVE_HEADS:
            rows = alignment.heads[task]
            assert rows.valid_mask.tolist() == [True, True]
            assert rows.values.tolist() in ([0, 1], [1, 0])
            selected[task] = output.logits[task][rows.candidate_indices].clone()
            correct[task] = int((selected[task].argmax(-1) == rows.values).sum())
        return report, selected, correct


def test_all_18_heads_learn_opposing_raw_patterns_through_actual_update_path():
    samples, batch, sidecars, weights = _fixture()
    original_sidecars = deepcopy(sidecars)
    raw_fingerprints = [graph_fingerprint(sample.raw_graph) for sample in samples]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = CorrectedAnalysisGNNModel()
        initial, _, initial_correct = _evaluate(model, batch, sidecars, weights)
        groups = {
            "local": model.local_baseline.encoder,
            "pooling": model.context_encoder.pooling,
            "transformer": model.context_encoder.transformer,
            "fusion": model.context_encoder.fusion,
            "bigru": model.sequence_decoder,
            **{task: model.task_heads.heads[f"task_{index:02d}"]
               for index, task in enumerate(ACTIVE_HEADS)},
        }
        initial_parameters = {
            name: [p.detach().clone() for p in module.parameters()]
            for name, module in groups.items()
        }
        updates = 80
        optimizer, scheduler = build_optimizer_scheduler(model, total_updates=updates)
        scaler, _ = _scaler()
        model.train()
        for _ in range(updates):
            result = attempt_applied_update(
                model=model, raw_graph_batch=batch.raw_graph_batch,
                sidecars=sidecars, shifts=(0, 0), class_weights=weights,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            )
            assert result.applied and result.finite_logits and result.finite_gradients
            assert result.gradient_norm > 0
            assert all(result.loss.heads[task].valid_row_count == 2 for task in ACTIVE_HEADS)
        assert scheduler.last_epoch == updates
        for name, module in groups.items():
            assert any(
                not torch.equal(parameter, old)
                for parameter, old in zip(module.parameters(), initial_parameters[name], strict=True)
            ), name
        final, selected, correct = _evaluate(model, batch, sidecars, weights)
        print(json.dumps({
            "check": "synthetic_18_head_opposing_patterns", "updates": updates,
            "initial_loss": float(initial.total), "final_loss": float(final.total),
            "initial_correct": sum(initial_correct.values()), "final_correct": sum(correct.values()),
            "rows": 36, "per_head_correct": correct,
        }, sort_keys=True))
        assert float(final.total) < 0.15 * float(initial.total)
        assert all(value == 2 for value in correct.values()), correct
        assert all(
            float(final.heads[task].weighted_ce) < float(initial.heads[task].weighted_ce)
            for task in ACTIVE_HEADS
        )

        # Swapping batch order must preserve each input's learned answer.
        reversed_batch = collate_ssl_samples(tuple(reversed(samples)))
        _, reversed_logits, reversed_correct = _evaluate(
            model, reversed_batch, tuple(reversed(sidecars)), weights,
        )
        assert reversed_correct == correct
        for task in ACTIVE_HEADS:
            torch.testing.assert_close(selected[task], reversed_logits[task].flip(0), rtol=1e-5, atol=1e-5)

        # Identical raw inputs with contradictory labels cannot both be right.
        # This rules out targets or the batch slot providing the learned answer.
        identical_batch = collate_ssl_samples((samples[0], samples[0]))
        _, identical_logits, identical_correct = _evaluate(
            model, identical_batch, sidecars, weights,
        )
        for task in ACTIVE_HEADS:
            torch.testing.assert_close(
                identical_logits[task][0], identical_logits[task][1], rtol=1e-5, atol=1e-5,
            )
            assert identical_correct[task] <= 1

    assert sidecars == original_sidecars
    assert [graph_fingerprint(sample.raw_graph) for sample in samples] == raw_fingerprints


def test_gradient_inventory_exposes_only_structurally_unused_final_branches():
    """Final song/track/bar outputs have no downstream head in this model.

    Keep the unused inventory explicit so a future accidental detach elsewhere
    fails this check. This does not classify the coarse pooling as unused.
    """
    _, batch, sidecars, weights = _fixture()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = CorrectedAnalysisGNNModel().train()
        output = model(batch.raw_graph_batch)
        alignment = align_target_sidecars_after_prediction(output, batch.raw_graph_batch, sidecars)
        report = corrected_supervised_loss(output, alignment, weights)
        assert report.total is not None and not report.zero_valid_heads
        report.total.backward()
        expected_prefixes = (
            "context_encoder.fusion.fusions.song.",
            "context_encoder.fusion.fusions.track.",
            "context_encoder.fusion.fusions.bar.",
            "sequence_decoder.bar_",
        )
        expected_unused = {
            name for name, parameter in model.named_parameters()
            if name.startswith(expected_prefixes)
        }
        actual_unused = {
            name for name, parameter in model.named_parameters() if parameter.grad is None
        }
        assert actual_unused == expected_unused
        assert len(actual_unused) == 25
        assert sum(p.numel() for n, p in model.named_parameters() if n in actual_unused) == 231_680
        assert all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters() if parameter.grad is not None
        )
