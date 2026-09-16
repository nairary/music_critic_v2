from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from music_critic.data import RationalTime
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn import corrected_training as old
from music_critic.experiments.analysisgnn.performance_replay import _scaler
from music_critic.experiments.analysisgnn.temporal_slices import build_temporal_batch, TemporalSliceModel
from music_critic.experiments.analysisgnn.temporal_slices.graph import EDGES, FEATURES, owner
from music_critic.experiments.analysisgnn.temporal_slices.alignment import (
    align_temporal_targets, temporal_supervised_loss, exact_target_time,
)
from music_critic.experiments.analysisgnn.temporal_slices.training import attempt_temporal_update


@pytest.fixture
def fixture(monkeypatch):
    import music_critic.tasks as tasks
    captured = []
    prepare = tasks.prepare_multisource_sample
    def capture(piece, *args, **kwargs):
        captured.append(piece)
        return prepare(piece, *args, **kwargs)
    monkeypatch.setattr(tasks, "prepare_multisource_sample", capture)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    _, sidecar = old.build_source_free_fixture()
    yield captured[0], sidecar
    torch.set_num_threads(previous_threads)


def temporal_piece(piece):
    # Sustained first note, second note at 1/3; every role is exercised.
    return replace(piece, notes=(
        replace(piece.notes[0], duration_qn=RationalTime(2)),
        replace(piece.notes[1], onset_qn=RationalTime(1, 3), duration_qn=RationalTime(1, 3)),
    ))


def extra_target(sidecar, level, time, name="extra"):
    original = next(e for e in sidecar["entities"] if e["entity_type"] == level)
    row = deepcopy(original)
    row["canonical_entity_id"] = name
    row["onset_qn"] = {"num": time.numerator, "den": time.denominator}
    sidecar["entities"].append(row)
    if level == "harmonic_event":
        relation = next(r for r in sidecar["relations"] if r["relation"] == "harmonic_event_to_beat"
                        and r["source_entity_id"] == original["canonical_entity_id"])
        sidecar["relations"].append({**relation, "source_entity_id": name})
    return row


def test_exact_topology_and_held_notes(fixture):
    piece, _ = fixture
    graph = build_temporal_batch([temporal_piece(piece)])
    times = [t for t, _ in graph.slots[0].time_slots]
    assert times == [Fraction(0), Fraction(1, 3), Fraction(2, 3), Fraction(1), Fraction(2)]
    assert graph.edges["note", "holds", "slice"].tolist() == [[0, 0, 0], [1, 2, 3]]
    assert graph.edges["note", "ends", "slice"].tolist() == [[0, 1], [4, 2]]
    assert graph.edges["beat", "contains", "slice"].tolist() == [[0, 0, 0, 1, 1], [0, 1, 2, 3, 4]]
    assert graph.slice_features[1, FEATURES.index("phase_in_beat")].item() == pytest.approx(1/3)
    assert graph.slice_features[1, FEATURES.index("meter_numerator")].item() == 4
    for s, r, d in EDGES[::2]:
        assert torch.equal(graph.edges[s, r, d].flip(0), graph.edges[d, "rev_"+r, s])
    assert graph.fingerprint() == build_temporal_batch([temporal_piece(piece)]).fingerprint()
    assert graph.to("cpu").fingerprint() == graph.fingerprint()


def test_grace_note_has_attack_and_release_but_no_hold(fixture):
    piece, _ = fixture
    note = replace(piece.notes[0], duration_qn=RationalTime(0), is_grace=True)
    g = build_temporal_batch([replace(piece, notes=(note, piece.notes[1]))])
    assert 0 in g.edges["note", "starts", "slice"][0]
    assert 0 in g.edges["note", "ends", "slice"][0]
    assert 0 not in g.edges["note", "holds", "slice"][0]


def test_boundary_ownership_and_gaps():
    assert owner(Fraction(1), [Fraction(0), Fraction(1)], [Fraction(1), Fraction(2)]) == 1
    assert owner(Fraction(2), [Fraction(0), Fraction(1)], [Fraction(1), Fraction(2)]) == 1
    assert owner(Fraction(2), [Fraction(0), Fraction(3)], [Fraction(1), Fraction(4)]) is None
    assert owner(Fraction(-1), [Fraction(0)], [Fraction(1)]) is None
    assert owner(Fraction(0), [], []) is None


def test_reference_ids_batch_isolation_and_schema_rejection(fixture):
    piece, _ = fixture
    graph = build_temporal_batch([piece, temporal_piece(piece)])
    graph.validate()
    for node in ("note", "onset", "beat", "bar"):
        assert graph.pointers[node] == tuple(graph.base[node].ptr.tolist())
    with pytest.raises(ValueError, match="schema"):
        replace(graph, schema="old-cache").validate()
    edges = {e: v.clone() for e, v in graph.edges.items()}
    edges["note", "starts", "slice"][1, 0] = graph.pointers["slice"][1]
    with pytest.raises(ValueError, match="cross-sample"):
        replace(graph, edges=edges).validate()


def test_raw_only_candidate_generation_ignores_sidecar(fixture):
    piece, sidecar = fixture
    before = build_temporal_batch([piece]).fingerprint()
    extra_target(sidecar, "harmonic_event", Fraction(7, 13))
    assert build_temporal_batch([piece]).fingerprint() == before


def test_eighteen_outputs_and_initial_base_weights_unchanged(fixture):
    piece, _ = fixture
    torch.manual_seed(17)
    baseline = CorrectedAnalysisGNNModel()
    torch.manual_seed(17)
    model = TemporalSliceModel().eval()
    assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in baseline.state_dict().items())
    graph = build_temporal_batch([temporal_piece(piece)])
    output = model(graph)
    assert tuple(output.logits) == old.ACTIVE_HEADS
    for spec in model.base.task_specs:
        logits = output.logits[spec.task_id]
        count = 7 if spec.node_types == ("beat",) else 5 if spec.node_types == ("onset",) else 2
        assert logits.shape == (count, spec.output_dim)
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    assert not torch.equal(output.logits["root"][0], output.logits["root"][1])
    with pytest.raises(RuntimeError):
        model.load_state_dict(baseline.state_dict())


def test_batch_does_not_mix_pieces(fixture):
    piece, _ = fixture
    model = TemporalSliceModel().eval()
    with torch.no_grad():
        alone = model(build_temporal_batch([piece]))
        together = model(build_temporal_batch([piece, temporal_piece(piece)]))
    for head in old.ACTIVE_HEADS:
        indices = [i for i, k in enumerate(together.candidate_keys[head]) if k[0] == 0]
        torch.testing.assert_close(alone.logits[head], together.logits[head][indices], atol=2e-6, rtol=2e-5)


def test_alignment_new_times_fallback_masks_conflicts_and_common_support(fixture):
    piece, sidecar = fixture
    sidecar = deepcopy(sidecar)
    row = extra_target(sidecar, "harmonic_event", Fraction(1, 3), "exact")
    row["targets"]["root"]["canonical_value"] = "D"
    extra_target(sidecar, "harmonic_event", Fraction(1, 3), "conflict")
    extra_target(sidecar, "harmonic_event", Fraction(7, 13), "fallback")
    extra_target(sidecar, "onset", Fraction(2, 3), "new_cadence")
    bad = extra_target(sidecar, "harmonic_event", Fraction(1), "masked")
    bad["targets"]["root"]["masked"] = True
    bad["targets"]["quality"]["canonical_value"] = "unknown-class"
    output = TemporalSliceModel()(build_temporal_batch([temporal_piece(piece)]))
    alignment = align_temporal_targets(output, [sidecar])
    root = alignment.heads["root"]
    idx = {name: i for i, name in enumerate(root.entity_ids)}
    assert root.candidate_indices[idx["exact"]] == root.candidate_indices[idx["conflict"]]
    assert root.values[idx["exact"]] != root.values[idx["conflict"]]
    assert alignment.routes["root"][idx["fallback"]] == "legacy_beat"
    assert not root.valid_mask[idx["masked"]]
    assert not alignment.heads["quality"].valid_mask[-1]
    assert alignment.heads["cadence"].valid_mask[-1]
    assert not alignment.common_support_masks["cadence"][-1]
    assert len(root.values) == 5  # retain repeated/conflicting native rows


def test_historical_support_and_loss_are_retained_on_fixture(fixture):
    piece, sidecar = fixture
    g = build_temporal_batch([piece])
    baseline = CorrectedAnalysisGNNModel()
    previous = old.align_target_sidecars_after_prediction(baseline(g.base), g.base, [sidecar])
    model = TemporalSliceModel()
    output = model(g)
    aligned = align_temporal_targets(output, [sidecar])
    for h in old.ACTIVE_HEADS:
        assert torch.equal(previous.heads[h].valid_mask, aligned.common_support_masks[h])
        assert torch.equal(previous.heads[h].values, aligned.heads[h].values)
    weights = old.load_frozen_class_weights()
    loss = temporal_supervised_loss(output, aligned, weights)
    reference = old.corrected_supervised_loss(output, aligned, weights)
    torch.testing.assert_close(loss.total, reference.total, rtol=0, atol=0)
    loss.total.backward()
    assert model.slice_projection.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.local_baseline.encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("value", [{"num": 1, "den": 2.0}, {"num": 2, "den": 4},
                                  {"num": True, "den": 1}, {"num": 1, "den": 0}, .5])
def test_nonexact_target_times_rejected(value):
    with pytest.raises(ValueError):
        exact_target_time(value)


def test_optimizer_update_and_state_restore(fixture):
    piece, sidecar = fixture
    g = build_temporal_batch([temporal_piece(piece)])
    model = TemporalSliceModel()
    optimizer, scheduler = old.build_optimizer_scheduler(model, total_updates=120000)
    weights = old.load_frozen_class_weights()
    kwargs = dict(graph=g, sidecars=[sidecar], shifts=[0], class_weights=weights)
    def update(m, opt, sch):
        return attempt_temporal_update(model=m, optimizer=opt, scheduler=sch,
                                       scaler=_scaler()[0], **kwargs)
    first = update(model, optimizer, scheduler)
    assert first.applied and first.finite_gradients and first.finite_logits and scheduler.last_epoch == 1
    state, optstate, schstate = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), deepcopy(scheduler.state_dict())
    rng = torch.get_rng_state()
    second = update(model, optimizer, scheduler)
    restored = TemporalSliceModel()
    opt2, sch2 = old.build_optimizer_scheduler(restored, total_updates=120000)
    restored.load_state_dict(state); opt2.load_state_dict(optstate); sch2.load_state_dict(schstate)
    torch.set_rng_state(rng)
    repeat = update(restored, opt2, sch2)
    assert second.loss.total.item() == repeat.loss.total.item()
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_slice_information_returns_to_notes_and_parents(fixture):
    piece, _ = fixture
    g = build_temporal_batch([temporal_piece(piece)])
    model = TemporalSliceModel().eval()
    output = model(g)
    for node in ("note", "beat", "bar"):
        gradient, = torch.autograd.grad(output.embeddings[node][0, 0], model.slice_projection.weight,
                                       retain_graph=True)
        assert gradient.abs().sum() > 0, node
    # Root at held-only beat time 1 receives note -> holds -> slice messages.
    j = next(i for i, key in enumerate(output.candidate_keys["root"]) if key[2] == "rawtime:1_1")
    gradient, = torch.autograd.grad(output.logits["root"][j, 0], model.temporal_layers[0].messages[2].weight)
    assert gradient.abs().sum() > 0


def test_relative_metric_features_preserve_origin_and_pickup(fixture):
    piece, _ = fixture
    piece = temporal_piece(piece)
    offset = RationalTime(10)
    shifted = replace(piece,
        notes=tuple(replace(n, onset_qn=n.onset_qn+offset) for n in piece.notes),
        beats=tuple(replace(b, start_qn=b.start_qn+offset) for b in piece.beats),
        bars=tuple(replace(b, start_qn=b.start_qn+offset) for b in piece.bars),
        meter_events=tuple(replace(m, onset_qn=m.onset_qn+offset) for m in piece.meter_events),
        tempo_events=tuple(replace(m, onset_qn=m.onset_qn+offset) for m in piece.tempo_events))
    # Canonical pieces require origin zero; test extension features using raw
    # shifted spans independently of the unchanged production builder.
    from music_critic.experiments.analysisgnn.temporal_slices.graph import exact
    for t in [Fraction(1, 3), Fraction(2, 3)]:
        left = t-exact(piece.bars[0].start_qn)+exact(piece.bars[0].metric_offset_qn)
        right = t+10-exact(shifted.bars[0].start_qn)+exact(shifted.bars[0].metric_offset_qn)
        assert left == right
    g = build_temporal_batch([piece])
    assert g.slice_features[1, FEATURES.index("position_in_metric_bar_qn")].item() == pytest.approx(
        float(Fraction(1, 3)+exact(piece.bars[0].metric_offset_qn)))


def test_transposition_preserves_graph_timing_and_transforms_targets(fixture):
    from music_critic.experiments.analysisgnn.temporal_slices.training import transpose_temporal_batch
    from music_critic.experiments.analysisgnn.transposition import DirectedTransposition
    piece, sidecar = fixture
    g = build_temporal_batch([piece])
    before = g.fingerprint()
    shifted = transpose_temporal_batch(g, [DirectedTransposition(2, 2)])
    assert g.fingerprint() == before
    assert shifted.fingerprint() != before
    assert shifted.slots == g.slots and shifted.keys == g.keys
    assert torch.equal(shifted.slice_features, g.slice_features)
    assert all(torch.equal(shifted.edges[e], g.edges[e]) for e in EDGES)
    model = TemporalSliceModel().eval()
    previous = old.align_target_sidecars_after_prediction(model.base(shifted.base), shifted.base, [sidecar], shifts=[2])
    current = align_temporal_targets(model(shifted), [sidecar], shifts=[2])
    for h in old.ACTIVE_HEADS:
        assert torch.equal(previous.heads[h].values, current.heads[h].values)
        assert torch.equal(previous.heads[h].valid_mask, current.common_support_masks[h])


def test_empty_candidate_loss_remains_masked(fixture):
    piece, sidecar = fixture
    output = TemporalSliceModel()(build_temporal_batch([piece]))
    aligned = align_temporal_targets(output, [sidecar])
    h = "root"
    heads = dict(aligned.heads)
    heads[h] = replace(heads[h], valid_mask=torch.zeros_like(heads[h].valid_mask))
    modified = replace(output, logits={**output.logits, h: output.logits[h][:0]})
    loss = temporal_supervised_loss(modified, replace(aligned, heads=heads), old.load_frozen_class_weights())
    assert loss.heads[h].weighted_ce is None
    assert loss.total is not None and torch.isfinite(loss.total)


def test_no_cadence_fallback_and_ambiguous_relation_rejected(fixture):
    piece, sidecar = fixture
    sidecar = deepcopy(sidecar)
    extra_target(sidecar, "onset", Fraction(7, 13), "unmatched_cadence")
    output = TemporalSliceModel()(build_temporal_batch([piece]))
    alignment = align_temporal_targets(output, [sidecar])
    assert not alignment.heads["cadence"].valid_mask[-1]
    assert alignment.routes["cadence"][-1] == "unmatched"
    relation = next(r for r in sidecar["relations"] if r["relation"] == "harmonic_event_to_beat")
    sidecar["relations"].append({**relation, "target_entity_id": "another-beat"})
    with pytest.raises(ValueError, match="ambiguous"):
        align_temporal_targets(output, [sidecar])
