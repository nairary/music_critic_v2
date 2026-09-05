"""Cross-workstream CPU regressions using newly generated sources and weights.

These tests establish API composition, never music-quality or corpus results.
"""

from dataclasses import replace
from hashlib import sha256
import json

import torch

from music_critic.adapters.pdmx import PDMX_SCHEMA_PROFILE_ID
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel, model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.transposition import DirectedTransposition
from music_critic.experiments.analysisgnn.transposition_inputs import (
    transpose_complete_raw_graph_view,
)
from music_critic.experiments.pop909_cl_frozen.checkpoint import (
    create_model_only_payload, load_model_checkpoint,
)
from music_critic.experiments.pop909_cl_frozen.evaluation import evaluate_frozen
from music_critic.experiments.pop909_cl_frozen.run import synthetic_input
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.pdmx_pipeline import (
    PdmxWindowDataset, PipelineConfig, SelectionConfig, SourceManifest,
    SourceRecord, WindowConfig, preprocess_pdmx,
)
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.masking import prepare_mask_binding
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness
from music_critic.ssl.theory_warmstart import (
    BACKBONE_GROUPS, RETAINED_GROUPS, TheoryWarmstartSSLModel,
    export_theory_warmstart_state, load_theory_warmstart,
)
from music_critic.tasks.corpus import load_cached_piece


def _warmstart():
    torch.manual_seed(905)
    source = CorrectedAnalysisGNNModel()
    result = load_theory_warmstart(
        TheoryWarmstartSSLModel(), export_theory_warmstart_state(source)
    )
    return source, result


def test_pdmx_windows_complete_transposition_and_shared_ssl_step(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = json.dumps({
        "resolution": 3,
        "barlines": [{"time": tick} for tick in (0, 12, 24)],
        "time_signatures": [{"time": 0, "numerator": 4, "denominator": 4}],
        "tempos": [{"time": 0, "qpm": 120}],
        "tracks": [{"program": 0, "is_drum": False, "notes": [
            {"time": index * 3, "duration": 2, "pitch": 60 + index}
            for index in range(8)
        ]}],
    }).encode()
    (source_root / "generated.json").write_bytes(payload)
    manifest = SourceManifest((SourceRecord(
        "generated", "generated.json", sha256(payload).hexdigest(),
        "synthetic", "generated-by-integration-test", synthetic=True,
    ),), PDMX_SCHEMA_PROFILE_ID, "fixture")
    prepared = preprocess_pdmx(
        manifest, {"generated": "train"}, source_root=source_root,
        output_root=tmp_path / "prepared",
        config=PipelineConfig(SelectionConfig(("synthetic",)), window=WindowConfig(1, 1)),
    )
    dataset = PdmxWindowDataset.from_output(prepared.root, split="train")
    assert prepared.report["accepted_records"] == 1
    assert prepared.report["windows"] == len(dataset) == 2
    originals, transformed = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        before = graph_fingerprint(sample.raw_graph)
        piece = load_cached_piece(dataset.records[index], dataset.cache_config)
        assert piece.targets == piece.annotations == ()
        view = transpose_complete_raw_graph_view(
            sample.raw_graph, transform=DirectedTransposition(2, 2)
        )
        rebuilt = build_raw_graph(replace(piece, notes=tuple(
            note if note.is_percussion else replace(note, pitch=note.pitch + 2)
            for note in piece.notes
        )))
        assert graph_fingerprint(view) == graph_fingerprint(rebuilt)
        assert graph_fingerprint(view) != before
        originals.append((sample.raw_graph, before))
        transformed.append(SSLRawSample(
            view, graph_fingerprint(view), sample.dataset_id, sample.piece_id
        ))
    batch = collate_ssl_samples(transformed)
    assert batch.sample_count == 2
    source, warmstart = _warmstart()
    source_before = model_state_fingerprint(source)
    harness = TheoryRetentionHarness(warmstart).train()
    binding = prepare_mask_binding(
        batch, global_seed=906, epoch=0, stage="train",
        requested_mask_rate=harness.ssl_model.ssl_config.mask_rate,
    )
    parameters = dict(harness.theory_model.named_parameters())
    shared = {name: p.detach().clone() for name, p in parameters.items()
              if name.startswith(tuple(g + "." for g in BACKBONE_GROUPS))}
    retained = {name: p.detach().clone() for name, p in parameters.items()
                if name.startswith(tuple(g + "." for g in RETAINED_GROUPS))}
    optimizer = torch.optim.SGD(harness.optimizer_parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    output = harness(
        lambda_ssl=1.0, lambda_theory=0.0,
        ssl_batch=batch, prepared_mask_binding=binding,
    )
    assert output.total is not None and bool(torch.isfinite(output.total))
    assert output.counts["ssl_selected_notes"] > 0
    assert output.theory_loss is output.theory_output is None
    output.total.backward()
    gradients = [parameters[name].grad for name in shared if parameters[name].grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    assert any(bool(g.abs().sum() > 0) for g in gradients)
    assert all(parameters[name].grad is None for name in retained)
    optimizer.step()
    assert any(not torch.equal(parameters[name], old) for name, old in shared.items())
    assert all(torch.equal(parameters[name], old) for name, old in retained.items())
    assert model_state_fingerprint(source) == source_before
    assert all(graph_fingerprint(graph) == before for graph, before in originals)


def test_retained_theory_state_roundtrips_through_frozen_pop909_evaluation(tmp_path):
    source, warmstart = _warmstart()
    retained = warmstart.theory_model.eval()
    expected = model_state_fingerprint(source)
    assert model_state_fingerprint(retained) == expected
    path = tmp_path / "new-synthetic-model.pt"
    torch.save(create_model_only_payload(retained), path)
    content = path.read_bytes()
    caller_rng = torch.get_rng_state().clone()
    loaded = load_model_checkpoint(
        path, expected_sha256=sha256(content).hexdigest(), expected_size_bytes=len(content)
    )
    assert torch.equal(torch.get_rng_state(), caller_rng)
    assert model_state_fingerprint(loaded.model) == expected
    assert not {id(p) for p in retained.parameters()} & {id(p) for p in loaded.model.parameters()}
    assert all(not module.training for module in loaded.model.modules())
    assert all(not p.requires_grad for p in loaded.model.parameters())
    assert loaded.evidence["optimizer_restored"] is False
    assert loaded.evidence["checkpoint_rng_applied"] is False
    item = synthetic_input(tmp_path)
    original_report = evaluate_frozen(retained, (item,), split="synthetic")
    loaded_report = evaluate_frozen(
        loaded.model, (item,), split="synthetic", checkpoint_evidence=loaded.evidence
    )
    assert set(loaded_report["views"]) == {"native", "metadata_removed"}
    assert loaded_report["views"] == original_report["views"]
    assert loaded_report["pieces"] == original_report["pieces"]
    assert model_state_fingerprint(retained) == model_state_fingerprint(source) == expected
