from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from music_critic.adapters import dilemmadata as dilemmadata_adapter
from music_critic.adapters import validate_dilemmadata_record_binding
from music_critic.experiments.analysisgnn import corrected_training
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    CorrectedComponentSampler,
    CorrectedRuntimeConfig,
    CorrectedTrainingError,
    CorrectedValidationAccumulator,
    ProductionArtifactPaths,
    align_target_sidecars_after_prediction,
    build_optimizer_scheduler,
    build_source_free_fixture,
    checkpoint_payload,
    corrected_primary_macro_score,
    corrected_supervised_loss,
    initialize_paired_models,
    joint_metric_contract_evidence,
    load_checkpoint,
    load_frozen_class_weights,
    load_production_record,
    per_head_validation_metrics,
    record_schedule_fingerprint,
    require_non_test_split,
    save_checkpoint,
    select_best_validation_checkpoint,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    TASK_BY_ID,
    get_vocabulary,
)
from music_critic.experiments.analysisgnn.training_policy import (
    AUXILIARY_HEADS,
    CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
    CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID,
    PRIMARY_HEADS,
    aggregate_corrected_losses,
)
from music_critic.tasks import collate_multisource_samples
from tests.adapters.test_dilemmadata import CORPUS, _record
from tests.adapters.test_dilemmadata_targets import _sample, _target


def _batch():
    return collate_multisource_samples(
        (_sample(_target(CORPUS, "an:training:same")),)
    )


def _supported_labels() -> dict[str, str]:
    weights = load_frozen_class_weights()
    return {
        task_id: get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels[
            int(torch.nonzero(weights.supported[task_id], as_tuple=False)[0])
        ]
        for task_id in ACTIVE_HEADS
    }


def _state(task_id: str, entity_id: str, *, available: bool = True):
    return {
        "available": available,
        "masked": not available,
        "missing_reason": None if available else "fixture_missing",
        "source_value": _supported_labels()[task_id] if available else None,
        "canonical_value": _supported_labels()[task_id] if available else None,
        "source_entity_id": f"source:{entity_id}",
        "canonical_entity_id": entity_id,
        "provenance": {"fixture": True},
    }


def _sidecar(batch=None):
    batch = _batch() if batch is None else batch
    beat_id = batch.raw_graph_batch["beat"].entity_id[0][0]
    note_id = batch.raw_graph_batch["note"].entity_id[0][0]
    harmonic_id = "harmonic-event:fixture"
    onset_id = "onset-sidecar:fixture"
    harmonic_tasks = [
        task for task in ACTIVE_HEADS
        if TASK_BY_ID[task].prediction_level == "harmonic_event"
    ]
    onset_tasks = [
        task for task in ACTIVE_HEADS
        if TASK_BY_ID[task].prediction_level == "onset"
    ]
    note_tasks = [
        task for task in ACTIVE_HEADS
        if TASK_BY_ID[task].prediction_level == "note"
    ]
    return {
        "schema_version": "analysisgnn-source-native-target-sidecar-v1",
        "record_id": "dlc:fixture:one",
        "dialect": "dlc",
        "source_component_id": "component:fixture",
        "entities": [
            {
                "canonical_entity_id": harmonic_id,
                "entity_type": "harmonic_event",
                "onset_qn": {"num": 0, "den": 1},
                "targets": {task: _state(task, harmonic_id) for task in harmonic_tasks},
            },
            {
                "canonical_entity_id": onset_id,
                "entity_type": "onset",
                "onset_qn": {"num": 0, "den": 1},
                "targets": {task: _state(task, onset_id) for task in onset_tasks},
            },
            {
                "canonical_entity_id": note_id,
                "entity_type": "note",
                "onset_qn": {"num": 0, "den": 1},
                "targets": {task: _state(task, note_id) for task in note_tasks},
            },
        ],
        "relations": [
            {
                "relation": "harmonic_event_to_beat",
                "source_entity_id": harmonic_id,
                "target_entity_id": beat_id,
            }
        ],
        "fingerprint": "fixture-sidecar-v1",
    }


def _forward_and_align(model, batch, sidecar=None):
    output = model(batch.raw_graph_batch)
    alignment = align_target_sidecars_after_prediction(
        output, batch.raw_graph_batch, (_sidecar(batch) if sidecar is None else sidecar,)
    )
    return output, alignment


def test_targets_are_post_prediction_and_mutation_cannot_change_logits() -> None:
    model = CorrectedAnalysisGNNModel().eval()
    batch = _batch()
    original = _sidecar(batch)
    mutated = copy.deepcopy(original)
    mutated["entities"][0]["targets"]["quality"]["canonical_value"] = (
        get_vocabulary(TASK_BY_ID["quality"].vocabulary_id).labels[-1]
    )
    with torch.no_grad():
        first = model(batch.raw_graph_batch)
        first_alignment = align_target_sidecars_after_prediction(
            first, batch.raw_graph_batch, (original,)
        )
        second = model(batch.raw_graph_batch)
        second_alignment = align_target_sidecars_after_prediction(
            second, batch.raw_graph_batch, (mutated,)
        )
    assert all(
        torch.equal(first.logits[task], second.logits[task]) for task in ACTIVE_HEADS
    )
    assert not torch.equal(
        first_alignment.heads["quality"].values,
        second_alignment.heads["quality"].values,
    )


def test_exact_alignment_failure_masks_and_diagnoses_without_heuristic() -> None:
    model = CorrectedAnalysisGNNModel().eval()
    batch = _batch()
    sidecar = _sidecar(batch)
    sidecar["relations"][0]["target_entity_id"] = "beat:does-not-exist"
    with torch.no_grad():
        output, alignment = _forward_and_align(model, batch, sidecar)
    harmonic_tasks = [
        task for task in ACTIVE_HEADS
        if TASK_BY_ID[task].prediction_level == "harmonic_event"
    ]
    assert all(not alignment.heads[task].valid_mask.any() for task in harmonic_tasks)
    assert {
        row.category for row in alignment.diagnostics
        if row.task_id in harmonic_tasks
    } == {"exact_alignment_failed"}


def test_loss_is_b5b_formula_masks_missing_and_excludes_unsupported_classes() -> None:
    model = CorrectedAnalysisGNNModel().eval()
    batch = _batch()
    weights = load_frozen_class_weights()
    with torch.no_grad():
        output, alignment = _forward_and_align(model, batch)
        report = corrected_supervised_loss(output, alignment, weights)
    assert report.fp32_boundary
    assert report.total is not None and report.primary is not None and report.auxiliary is not None
    assert torch.equal(report.total, report.primary + 0.25 * report.auxiliary)
    assert not report.zero_valid_heads
    reference = aggregate_corrected_losses(
        {task: report.heads[task].weighted_ce for task in ACTIVE_HEADS}
    )
    assert torch.equal(report.total, reference.total)

    missing = _sidecar(batch)
    missing["entities"][0]["targets"]["quality"] = _state(
        "quality", "harmonic-event:fixture", available=False
    )
    with torch.no_grad():
        _, missing_alignment = _forward_and_align(model, batch, missing)
        missing_report = corrected_supervised_loss(output, missing_alignment, weights)
    assert missing_report.heads["quality"].weighted_ce is None
    assert "quality" in missing_report.zero_valid_heads
    expected_primary = torch.stack([
        missing_report.heads[task].weighted_ce
        for task in PRIMARY_HEADS
        if missing_report.heads[task].weighted_ce is not None
    ]).mean()
    assert torch.equal(missing_report.primary, expected_primary)

    unsupported = _sidecar(batch)
    unsupported_id = int(torch.nonzero(~weights.supported["quality"], as_tuple=False)[0])
    unsupported_label = get_vocabulary(TASK_BY_ID["quality"].vocabulary_id).labels[unsupported_id]
    unsupported["entities"][0]["targets"]["quality"]["canonical_value"] = unsupported_label
    with torch.no_grad():
        _, unsupported_alignment = _forward_and_align(model, batch, unsupported)
        unsupported_report = corrected_supervised_loss(output, unsupported_alignment, weights)
    assert unsupported_report.heads["quality"].unsupported_row_count == 1
    assert unsupported_report.heads["quality"].valid_row_count == 0
    assert unsupported_report.heads["quality"].weighted_ce is None


def test_all_active_heads_and_shared_encoder_receive_gradients() -> None:
    model = CorrectedAnalysisGNNModel().train()
    batch = _batch()
    output, alignment = _forward_and_align(model, batch)
    report = corrected_supervised_loss(output, alignment, load_frozen_class_weights())
    assert report.total is not None
    report.total.backward()
    for index, task_id in enumerate(ACTIVE_HEADS):
        assert report.heads[task_id].weighted_ce is not None
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in model.task_heads.heads[f"task_{index:02d}"].parameters()
        )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.local_baseline.parameters()
    )

    for group in (PRIMARY_HEADS, AUXILIARY_HEADS):
        model.zero_grad(set_to_none=True)
        output, alignment = _forward_and_align(model, batch)
        report = corrected_supervised_loss(output, alignment, load_frozen_class_weights())
        group_loss = torch.stack([
            report.heads[task].weighted_ce for task in group
            if report.heads[task].weighted_ce is not None
        ]).mean()
        group_loss.backward()
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in model.local_baseline.parameters()
        )


def test_fp32_head_and_loss_boundary_under_cpu_autocast() -> None:
    model = CorrectedAnalysisGNNModel().train()
    batch = _batch()
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        output, alignment = _forward_and_align(model, batch)
        report = corrected_supervised_loss(output, alignment, load_frozen_class_weights())
    assert all(value.dtype == torch.float32 for value in output.logits.values())
    assert report.fp32_boundary
    assert report.total is not None and report.total.dtype == torch.float32


def test_c0_c1_initialization_record_schedule_and_shift_domain_separation() -> None:
    c0_model, c1_model = initialize_paired_models()
    assert model_state_fingerprint(c0_model) == model_state_fingerprint(c1_model)
    components = {"component:a": ("record:a",), "component:b": ("record:b",)}
    shifts = {"record:a": (0, 1), "record:b": (0, 2)}
    c0 = CorrectedComponentSampler(
        components, shifts, profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID
    )
    c1 = CorrectedComponentSampler(
        components, shifts, profile_id=CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID
    )
    c0_draws, c1_draws = [], []
    for _ in range(20):
        c0_draws.append(c0.peek())
        c1_draws.append(c1.peek())
        c0.advance_after_applied_update()
        c1.advance_after_applied_update()
    assert [(row.component_id, row.record_id) for row in c0_draws] == [
        (row.component_id, row.record_id) for row in c1_draws
    ]
    assert all(row.shift_pc == 0 for row in c0_draws)
    assert any(row.shift_pc != 0 for row in c1_draws)
    assert record_schedule_fingerprint(components, seed=17, draw_count=20) == (
        record_schedule_fingerprint(components, seed=17, draw_count=20)
    )
    validation = CorrectedRuntimeConfig(
        profile_id=CORRECTED_SAFE_TRANSPOSITION_PROFILE_ID
    ).to_dict()
    assert validation["validation_transposition"] == "identity"


def test_test_lock_is_fail_closed_before_loader_or_target_read() -> None:
    calls = {"loader": False, "targets": False, "metrics": False}
    with pytest.raises(CorrectedTrainingError) as caught:
        require_non_test_split("test")
        calls["loader"] = True
    assert caught.value.category == "analysisgnn.corrected.test_lock"
    assert calls == {"loader": False, "targets": False, "metrics": False}
    with pytest.raises(CorrectedTrainingError):
        CorrectedRuntimeConfig(
            profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
            test_enabled=True,
        )


def _one_update(model, optimizer, scheduler, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output, alignment = _forward_and_align(model, batch)
    report = corrected_supervised_loss(output, alignment, load_frozen_class_weights())
    assert report.total is not None
    report.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()


def test_checkpoint_contains_complete_state_and_resume_matches_uninterrupted(tmp_path: Path) -> None:
    batch = _batch()
    uninterrupted, interrupted = initialize_paired_models()
    config = CorrectedRuntimeConfig(
        profile_id=CORRECTED_NO_TRANSPOSITION_PROFILE_ID,
        applied_update_budget=2,
    )
    components = {"component:fixture": ("record:fixture",)}
    shifts = {"record:fixture": (0,)}

    optimizer_u, scheduler_u = build_optimizer_scheduler(uninterrupted, total_updates=2)
    torch.manual_seed(177)
    for _ in range(2):
        _one_update(uninterrupted, optimizer_u, scheduler_u, batch)

    optimizer_i, scheduler_i = build_optimizer_scheduler(interrupted, total_updates=2)
    sampler_i = CorrectedComponentSampler(
        components, shifts, profile_id=config.profile_id
    )
    scaler_i = torch.amp.GradScaler("cpu", enabled=False)
    torch.manual_seed(177)
    _one_update(interrupted, optimizer_i, scheduler_i, batch)
    sampler_i.advance_after_applied_update()
    payload = checkpoint_payload(
        model=interrupted,
        optimizer=optimizer_i,
        scheduler=scheduler_i,
        scaler=scaler_i,
        sampler=sampler_i,
        config=config,
        applied_update=1,
        best_primary_score=0.5,
        best_update=1,
        record_history=("record:fixture",),
        shift_history=(0,),
    )
    required = {
        "model_state", "optimizer_state", "scheduler_state", "amp_scaler_state",
        "sampler_state", "rng_state", "applied_update", "epoch", "draw_position",
        "best_primary_score", "record_history", "shift_history",
    }
    assert required <= set(payload)
    checkpoint = tmp_path / "last.ckpt"
    save_checkpoint(checkpoint, payload)

    resumed = CorrectedAnalysisGNNModel()
    optimizer_r, scheduler_r = build_optimizer_scheduler(resumed, total_updates=2)
    sampler_r = CorrectedComponentSampler(
        components, shifts, profile_id=config.profile_id
    )
    scaler_r = torch.amp.GradScaler("cpu", enabled=False)
    restored = load_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=optimizer_r,
        scheduler=scheduler_r,
        scaler=scaler_r,
        sampler=sampler_r,
        config=config,
    )
    assert restored["applied_update"] == 1 and sampler_r.position == 1
    _one_update(resumed, optimizer_r, scheduler_r, batch)
    sampler_r.advance_after_applied_update()
    assert model_state_fingerprint(resumed) == model_state_fingerprint(uninterrupted)
    assert sampler_r.position == 2


def test_validation_selection_and_joint_metric_names_are_separate() -> None:
    model = CorrectedAnalysisGNNModel().eval()
    batch = _batch()
    with torch.no_grad():
        output, alignment = _forward_and_align(model, batch)
        metrics = per_head_validation_metrics(
            output, alignment, load_frozen_class_weights()
        )
    score = corrected_primary_macro_score(metrics)
    assert score is not None
    assert select_best_validation_checkpoint(current_score=score, best_score=None)
    assert not select_best_validation_checkpoint(current_score=score, best_score=score)
    evidence = joint_metric_contract_evidence()
    assert evidence["corrected"]["unit"] == "harmonic_event"
    assert evidence["corrected"]["quality_classes"] == 17
    assert evidence["paper"]["unit"] == "note"
    assert evidence["paper"]["quality_classes"] == 15
    assert evidence["direct_roman_numeral"]["classes"] == 184
    assert evidence["direct_roman_numeral"]["derived_joint_correctness_separate"]


def test_joint_accumulator_keeps_event_note_and_direct_roman_metrics_separate() -> None:
    model = CorrectedAnalysisGNNModel().eval()
    batch = _batch()
    sidecar = _sidecar(batch)
    harmonic_id = sidecar["entities"][0]["canonical_entity_id"]
    note_id = sidecar["entities"][2]["canonical_entity_id"]
    sidecar["relations"].append(
        {
            "relation": "note_to_harmonic_event",
            "source_entity_id": note_id,
            "target_entity_id": harmonic_id,
        }
    )
    with torch.no_grad():
        output, alignment = _forward_and_align(model, batch, sidecar)
    accumulator = CorrectedValidationAccumulator(load_frozen_class_weights())
    accumulator.update(output, alignment, sidecars=(sidecar,))
    metrics = accumulator.finalize()
    assert metrics["corrected_joint_support"] == 1
    assert metrics["paper_note_joint_support"] == 1
    assert "v2_corrected_harmonic_event_joint_accuracy" in metrics
    assert "analysisgnn_paper_text_note_joint_accuracy" in metrics
    assert metrics["direct_roman_numeral_accuracy"] == metrics["per_head"]["roman_numeral"]["accuracy"]
    assert metrics["derived_joint_correctness_separate"] is True


def test_source_free_fixture_reproduces_and_test_record_cannot_reach_loader() -> None:
    first_batch, first_sidecar = build_source_free_fixture()
    second_batch, second_sidecar = build_source_free_fixture()
    assert first_sidecar == second_sidecar
    for node_type in first_batch.raw_graph_batch.node_types:
        assert first_batch.raw_graph_batch[node_type].entity_id == second_batch.raw_graph_batch[node_type].entity_id
        assert torch.equal(first_batch.raw_graph_batch[node_type].x_cat, second_batch.raw_graph_batch[node_type].x_cat)
    with pytest.raises(CorrectedTrainingError) as caught:
        load_production_record("an:test:abc-op127-2", split="test")
    assert caught.value.category == "analysisgnn.corrected.test_lock"


def test_production_binding_accepts_an_attested_portable_corpus_root(
    tmp_path: Path,
) -> None:
    historical = _record(CORPUS, "dlc:demo:same")
    portable_root = tmp_path / "portable-corpus"
    portable = replace(
        historical,
        path=portable_root / historical.relative_path,
        record_binding_sha256="",
    )
    portable = dilemmadata_adapter._bind_record(portable)
    assert portable.record_binding_sha256 != historical.record_binding_sha256

    audit_root = tmp_path / "b2-audit"
    audit_root.mkdir()
    inventory = audit_root / "source_inventory.jsonl"
    inventory.write_text("", encoding="utf-8")
    (audit_root / "audit_summary.json").write_text(
        json.dumps(
            {
                "snapshot": {
                    "actual_path": str(CORPUS),
                    "content_fingerprint": historical.corpus_identity.content_fingerprint,
                    "file_count": historical.corpus_identity.installation_file_count,
                }
            }
        ),
        encoding="utf-8",
    )
    paths = replace(
        ProductionArtifactPaths(),
        b2_source_inventory=inventory,
        corpus_root=portable_root,
    )
    rebound = corrected_training._bind_portable_production_record(
        portable,
        expected_binding_sha256=historical.record_binding_sha256,
        paths=paths,
        adapter=dilemmadata_adapter,
    )
    assert rebound.path == portable_root / historical.relative_path
    assert validate_dilemmadata_record_binding(rebound)


def test_production_binding_rejects_nonmatching_historical_seal(
    tmp_path: Path,
) -> None:
    historical = _record(CORPUS, "dlc:demo:same")
    portable_root = tmp_path / "portable-corpus"
    portable = replace(
        historical,
        path=portable_root / historical.relative_path,
        record_binding_sha256="",
    )
    audit_root = tmp_path / "b2-audit"
    audit_root.mkdir()
    inventory = audit_root / "source_inventory.jsonl"
    inventory.write_text("", encoding="utf-8")
    (audit_root / "audit_summary.json").write_text(
        json.dumps(
            {
                "snapshot": {
                    "actual_path": str(tmp_path / "wrong-historical-root"),
                    "content_fingerprint": historical.corpus_identity.content_fingerprint,
                    "file_count": historical.corpus_identity.installation_file_count,
                }
            }
        ),
        encoding="utf-8",
    )
    paths = replace(
        ProductionArtifactPaths(),
        b2_source_inventory=inventory,
        corpus_root=portable_root,
    )
    with pytest.raises(CorrectedTrainingError) as caught:
        corrected_training._bind_portable_production_record(
            portable,
            expected_binding_sha256=historical.record_binding_sha256,
            paths=paths,
            adapter=dilemmadata_adapter,
        )
    assert caught.value.category == "analysisgnn.corrected.b2_record_binding_mismatch"


def test_production_binding_replays_historical_parser_categories_only_for_seal(
    tmp_path: Path,
) -> None:
    discovered = _record(CORPUS, "dlc:demo:same")
    historical = replace(
        discovered,
        raw_issue_categories=("dilemmadata.grace_conflict",),
        record_binding_sha256="",
    )
    historical = dilemmadata_adapter._bind_record(historical)
    portable_root = tmp_path / "portable-corpus"
    portable = replace(
        discovered,
        path=portable_root / discovered.relative_path,
        raw_issue_categories=(),
        record_binding_sha256="",
    )

    audit_root = tmp_path / "b2-audit"
    audit_root.mkdir()
    inventory = audit_root / "source_inventory.jsonl"
    inventory.write_text("", encoding="utf-8")
    quarantine = audit_root / "quarantine_records.jsonl"
    quarantine.write_text(
        json.dumps(
            {
                "record_id": discovered.record_id,
                "pipeline_stage": "raw_parse",
                "evidence": {
                    "native_categories": ["dilemmadata.grace_conflict"]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (audit_root / "audit_summary.json").write_text(
        json.dumps(
            {
                "snapshot": {
                    "actual_path": str(CORPUS),
                    "content_fingerprint": discovered.corpus_identity.content_fingerprint,
                    "file_count": discovered.corpus_identity.installation_file_count,
                }
            }
        ),
        encoding="utf-8",
    )
    paths = replace(
        ProductionArtifactPaths(),
        b2_source_inventory=inventory,
        corpus_root=portable_root,
    )
    corrected_training._b2_historical_raw_issue_categories.cache_clear()
    categories = corrected_training._b2_historical_raw_issue_categories(quarantine)
    rebound = corrected_training._bind_portable_production_record(
        portable,
        expected_binding_sha256=historical.record_binding_sha256,
        historical_raw_issue_categories=categories[discovered.record_id],
        paths=paths,
        adapter=dilemmadata_adapter,
    )
    assert rebound.raw_issue_categories == ()
    assert rebound.path == portable_root / discovered.relative_path
    assert validate_dilemmadata_record_binding(rebound)
