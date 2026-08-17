from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import shutil

import pytest
import torch

from music_critic.adapters import (
    DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION,
    DILEMMADATA_TARGET_ADAPTER_VERSION,
    DILEMMADATA_TARGET_AUDIT_MANIFEST_VERSION,
    DILEMMADATA_TARGET_AUDIT_REPORT_VERSION,
    DILEMMADATA_TARGET_METADATA_VERSION,
    DILEMMADATA_TARGET_SIDECAR_VERSION,
    DilemmadataAccepted,
    DilemmadataQuarantine,
    DilemmadataTargetAccepted,
    DilemmadataTargetAdapterConfig,
    DilemmadataTargetAdapterError,
    DilemmadataTargetQuarantine,
    build_dilemmadata_target_sidecar,
    convert_dilemmadata_target_sidecar,
    convert_dilemmadata_record,
    reconstruct_dilemmadata_alignment_evidence,
)
from music_critic.adapters import dilemmadata as raw_adapter_module
from music_critic.adapters import dilemmadata_targets as target_adapter_module
from music_critic.data import RationalTime, dumps_piece
from music_critic.graph import graph_fingerprint, model_input_fingerprint
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
)
from music_critic.tasks import (
    DILEMMADATA_DEFERRED_MAPPINGS,
    DILEMMADATA_SOURCE_FAMILIES,
    DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
    DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION,
    DILEMMADATA_TARGET_ENCODINGS,
    DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION,
    DILEMMADATA_TARGET_FAMILIES,
    DILEMMADATA_TARGET_REGISTRY_ID,
    DILEMMADATA_TASK_IDS_BY_DIALECT,
    TARGET_BUNDLE_CONTRACT_VERSION,
    CorpusCacheConfig,
    IndexedMultiSourceDataset,
    TargetBundle,
    align_sample_targets,
    attach_target_bundle,
    build_dilemmadata_corpus_cache,
    collate_multisource_samples,
    dilemmadata_family_registry_fingerprint,
    dumps_target_bundle,
    prepare_multisource_sample,
    target_bundle_fingerprint,
)
from tests.adapters.test_dilemmadata import (
    CORPUS,
    _accepted,
    _fixture_identity,
    _raw_evidence,
    _record,
    _set_cell,
)


def _target(root: Path, record_id: str) -> DilemmadataTargetAccepted:
    outcome = build_dilemmadata_target_sidecar(_accepted(root, record_id))
    assert isinstance(outcome, DilemmadataTargetAccepted)
    return outcome


def _family(outcome: DilemmadataTargetAccepted, task_id: str):
    return next(
        row
        for row in outcome.statistics.family_statistics
        if row.task_id == task_id
    )


def _sample(outcome: DilemmadataTargetAccepted):
    raw = convert_dilemmadata_record(outcome.record)
    assert isinstance(raw, DilemmadataAccepted)
    return prepare_multisource_sample(
        raw.piece,
        target_sidecar=outcome.target_bundle,
    )


def _model() -> HierarchicalHeterogeneousBaseline:
    return HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    return rows[0], rows[1:]


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_public_versions_registry_inventory_and_frozen_encodings() -> None:
    assert DILEMMADATA_TARGET_ADAPTER_VERSION == "1.1.0"
    assert DILEMMADATA_TARGET_SIDECAR_VERSION == "1.0.0"
    assert DILEMMADATA_TARGET_AUDIT_REPORT_VERSION == "1.1.0"
    assert DILEMMADATA_TARGET_AUDIT_MANIFEST_VERSION == "1.1.0"
    assert DILEMMADATA_TARGET_METADATA_VERSION == "1.0.0"
    assert DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION == "1.0.0"
    assert DILEMMADATA_TARGET_ENCODING_REGISTRY_VERSION == "1.0.0"
    assert DILEMMADATA_TARGET_ALIGNMENT_RULES_VERSION == "1.0.0"
    assert DILEMMADATA_RAW_TARGET_ALIGNMENT_EVIDENCE_VERSION == "1.1.0"
    assert TARGET_BUNDLE_CONTRACT_VERSION == "1.0.0"
    assert DILEMMADATA_TARGET_REGISTRY_ID == "music_critic.dilemmadata@1.0.0"

    assert len(DILEMMADATA_SOURCE_FAMILIES) == 22
    assert len(DILEMMADATA_TARGET_FAMILIES) == 22
    assert len(DILEMMADATA_TARGET_ENCODINGS) == 22
    assert len(DILEMMADATA_TASK_IDS_BY_DIALECT["an_joint"]) == 9
    assert len(DILEMMADATA_TASK_IDS_BY_DIALECT["dlc"]) == 13
    task_ids = tuple(row.task_id for row in DILEMMADATA_SOURCE_FAMILIES)
    assert task_ids == tuple(sorted(task_ids))
    assert all(task.startswith("dilemmadata.an.") for task in DILEMMADATA_TASK_IDS_BY_DIALECT["an_joint"])
    assert all(task.startswith("dilemmadata.dlc.") for task in DILEMMADATA_TASK_IDS_BY_DIALECT["dlc"])
    assert all("borrowed" not in task and "voice" not in task for task in task_ids)
    assert len(next(row.vocabulary for row in DILEMMADATA_SOURCE_FAMILIES if row.task_id == "dilemmadata.an.chord.quality")) == 64
    assert {row.encoding_kind for row in DILEMMADATA_TARGET_ENCODINGS} == {
        "closed_categorical_index",
        "open_string_cpu",
    }
    assert all(
        row.vocabulary is None and not row.model_ready
        for row in DILEMMADATA_TARGET_ENCODINGS
        if row.encoding_kind == "open_string_cpu"
    )
    assert DILEMMADATA_DEFERRED_MAPPINGS == (
        "borrowed_harmony_unavailable",
        "staff_voice_to_semantic_role_incompatible",
        "tonal_region_alias_deferred",
        "an_dlc_crosswalk_deferred",
        "hooktheory_pop909_crosswalk_deferred",
        "root_crosswalk_deferred",
        "bass_crosswalk_deferred",
        "chord_quality_crosswalk_deferred",
    )
    assert len(dilemmadata_family_registry_fingerprint()) == 64


@pytest.mark.parametrize(
    ("field", "implemented"),
    [
        ("target_column_policy", "phase9a_evidenced_target_columns_only_v1"),
        ("span_policy", "exact_source_identity_next_boundary_no_terminal_inference_v1"),
        ("point_policy", "exact_rational_onset_no_snap_v1"),
        ("tie_policy", "all_merged_source_rows_must_agree_v1"),
        ("duplicate_policy", "merge_equal_mask_conflicts_v1"),
    ],
)
def test_target_config_is_closed_to_implemented_policies(
    field: str,
    implemented: str,
) -> None:
    assert getattr(DilemmadataTargetAdapterConfig(**{field: implemented}), field) == implemented
    for invalid in (True, False, None, 1, "", "unknown", (), []):
        with pytest.raises(DilemmadataTargetAdapterError) as raised:
            DilemmadataTargetAdapterConfig(**{field: invalid})
        assert raised.value.category == "dilemmadata.target.config_invalid"


@pytest.mark.parametrize(
    ("record_id", "dialect", "target_count"),
    [
        ("an:training:same", "an_joint", 9),
        ("dlc:demo:same", "dlc", 13),
    ],
)
def test_sidecar_is_deterministic_raw_external_and_state_explicit(
    record_id: str,
    dialect: str,
    target_count: int,
) -> None:
    raw = _accepted(CORPUS, record_id)
    first = build_dilemmadata_target_sidecar(raw)
    second = build_dilemmadata_target_sidecar(raw)
    assert isinstance(first, DilemmadataTargetAccepted)
    assert isinstance(second, DilemmadataTargetAccepted)

    assert raw.piece.annotations == raw.piece.targets == ()
    assert first.record.dialect == dialect
    assert len(first.target_bundle.targets) == target_count
    assert first.target_bundle.registry_extension_ids == (
        DILEMMADATA_TARGET_REGISTRY_ID,
    )
    assert first.target_bundle.analysis_view_id.startswith(f"dilemmadata.{dialect}.")
    assert first.target_bundle == second.target_bundle
    assert dumps_target_bundle(first.target_bundle) == dumps_target_bundle(second.target_bundle)
    assert first.sidecar_fingerprint == target_bundle_fingerprint(first.target_bundle)
    assert first.sidecar_fingerprint == second.sidecar_fingerprint
    assert all(target.annotation_view_id is None for target in first.target_bundle.targets)
    assert all(
        row.provenance_ids[index] is None
        for row in first.target_bundle.targets
        for index, available in enumerate(row.availability_mask)
        if not available
    )
    diagnostic_codes = {row.code for row in first.target_bundle.diagnostics}
    assert diagnostic_codes & {
        "dilemmadata.target.state.masked",
        "dilemmadata.target.state.missing",
        "dilemmadata.target.state.unsupported",
    }
    assert "dilemmadata.target.state.deferred_open_vocabulary" in diagnostic_codes
    assert "dilemmadata.target.analyst_metadata_present" in diagnostic_codes
    assert first.statistics.analyst_metadata_field_count > 0
    assert len(first.statistics.analyst_metadata_fingerprint) == 64


def test_analysis_views_remain_separate_inside_one_atomic_group() -> None:
    first_raw = _accepted(CORPUS, "an:training:same")
    second_raw = _accepted(CORPUS, "an:validation:same-alt")
    first = _target(CORPUS, "an:training:same")
    second = _target(CORPUS, "an:validation:same-alt")

    assert first_raw.record.source_group_id == second_raw.record.source_group_id
    assert first_raw.record.lineage_group_id == second_raw.record.lineage_group_id
    assert first.target_bundle.analysis_view_id != second.target_bundle.analysis_view_id
    assert first.target_bundle.piece_id != second.target_bundle.piece_id
    assert first.target_bundle.targets is not second.target_bundle.targets


def test_theory_only_mutation_changes_sidecar_not_raw_cache_graph_or_group(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    cache = CorpusCacheConfig(tmp_path / "cache")
    before_raw = _accepted(copied, "an:training:same")
    before_target = build_dilemmadata_target_sidecar(before_raw)
    assert isinstance(before_target, DilemmadataTargetAccepted)
    before_index, before_report = build_dilemmadata_corpus_cache(
        copied,
        cache_config=cache,
        identity=_fixture_identity(copied),
    )
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 0, "a_romanNumeral", "target-only-II")

    after_raw = _accepted(copied, "an:training:same")
    after_target = build_dilemmadata_target_sidecar(after_raw)
    assert isinstance(after_target, DilemmadataTargetAccepted)
    after_index, after_report = build_dilemmadata_corpus_cache(
        copied,
        cache_config=cache,
        identity=_fixture_identity(copied),
    )

    assert before_target.sidecar_fingerprint != after_target.sidecar_fingerprint
    assert before_raw.record.raw_projection_sha256 == after_raw.record.raw_projection_sha256
    assert before_raw.record.grouping_fingerprint == after_raw.record.grouping_fingerprint
    assert before_raw.record.source_group_id == after_raw.record.source_group_id
    assert _raw_evidence(before_raw) == _raw_evidence(after_raw)
    assert dumps_piece(before_raw.piece) == dumps_piece(after_raw.piece)
    before_record = next(row for row in before_index.records if row.piece_id == before_raw.piece.piece_id)
    after_record = next(row for row in after_index.records if row.piece_id == after_raw.piece.piece_id)
    assert before_record.cache_key == after_record.cache_key
    assert before_record.canonical_sha256 == after_record.canonical_sha256
    assert before_report.cache_miss_count == 3
    assert after_report.cache_hit_count == 3


def test_analyst_metadata_mutation_changes_only_target_sidecar(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    before_raw = _accepted(copied, "an:training:same")
    before_target = build_dilemmadata_target_sidecar(before_raw)
    assert isinstance(before_target, DilemmadataTargetAccepted)
    summary = copied / "pitch_arrays" / "AN" / "dataset_summary.tsv"
    payload = summary.read_text(encoding="utf-8")
    summary.write_text(
        payload.replace("Analyst A", "Analyst Z", 1),
        encoding="utf-8",
    )

    after_raw = _accepted(copied, "an:training:same")
    after_target = build_dilemmadata_target_sidecar(after_raw)
    assert isinstance(after_target, DilemmadataTargetAccepted)
    assert _raw_evidence(before_raw) == _raw_evidence(after_raw)
    assert before_raw.record.raw_projection_sha256 == after_raw.record.raw_projection_sha256
    assert before_raw.record.grouping_fingerprint == after_raw.record.grouping_fingerprint
    assert before_target.sidecar_fingerprint != after_target.sidecar_fingerprint
    assert (
        before_target.statistics.analyst_metadata_fingerprint
        != after_target.statistics.analyst_metadata_fingerprint
    )
    source_provenance = after_target.target_bundle.provenance[0]
    assert ("metadata.analyst", "Analyst Z") in source_provenance.details


def test_post_acceptance_raw_mutation_cannot_be_hidden_by_target_adapter(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    accepted = _accepted(copied, "an:training:same")
    _set_cell(accepted.record.path, 0, "s_midi", "61")

    target = build_dilemmadata_target_sidecar(accepted)
    assert isinstance(target, DilemmadataTargetQuarantine)
    assert "dilemmadata.target.source_changed_after_raw_acceptance" in target.categories
    rediscovered = convert_dilemmadata_record(_record(copied, "an:training:same"))
    assert isinstance(rediscovered, DilemmadataAccepted)
    assert rediscovered.record.raw_projection_sha256 != accepted.record.raw_projection_sha256


def test_forged_raw_to_target_alignment_evidence_fails_closed() -> None:
    accepted = _accepted(CORPUS, "an:training:same")
    first_row = accepted.alignment_evidence.rows[0]
    forged_evidence = replace(
        accepted.alignment_evidence,
        rows=(
            replace(first_row, onset_qn=RationalTime(1, 2)),
            *accepted.alignment_evidence.rows[1:],
        ),
    )
    outcome = build_dilemmadata_target_sidecar(
        replace(accepted, alignment_evidence=forged_evidence)
    )
    assert isinstance(outcome, DilemmadataTargetQuarantine)
    assert "dilemmadata.target.alignment_binding_mismatch" in outcome.categories


def _reseal_alignment_rows(accepted: DilemmadataAccepted, rows):
    draft = replace(
        accepted.alignment_evidence,
        rows=tuple(rows),
        fingerprint="",
    )
    return replace(
        draft,
        fingerprint=raw_adapter_module._alignment_evidence_fingerprint(draft),
    )


@pytest.mark.parametrize(
    "mutation",
    ("onset", "canonical_note_id", "tie_continuation", "row_semantics_order"),
)
def test_resealed_alignment_evidence_forgery_fails_closed(
    mutation: str,
) -> None:
    accepted = _accepted(CORPUS, "an:training:same")
    rows = list(accepted.alignment_evidence.rows)
    first, second = rows[:2]
    if mutation == "onset":
        rows[0] = replace(first, onset_qn=RationalTime(1, 2))
    elif mutation == "canonical_note_id":
        rows[0] = replace(first, canonical_note_id=second.canonical_note_id)
    elif mutation == "tie_continuation":
        rows[0] = replace(first, tie_continuation=not first.tie_continuation)
    else:
        rows[0] = replace(
            first,
            onset_qn=second.onset_qn,
            canonical_note_id=second.canonical_note_id,
            tie_continuation=second.tie_continuation,
        )
        rows[1] = replace(
            second,
            onset_qn=first.onset_qn,
            canonical_note_id=first.canonical_note_id,
            tie_continuation=first.tie_continuation,
        )
    forged = _reseal_alignment_rows(accepted, rows)
    assert forged.fingerprint == raw_adapter_module._alignment_evidence_fingerprint(
        forged
    )

    outcome = build_dilemmadata_target_sidecar(
        replace(accepted, alignment_evidence=forged)
    )
    assert isinstance(outcome, DilemmadataTargetQuarantine)
    assert outcome.categories == ("dilemmadata.target.alignment_binding_mismatch",)


def test_independent_alignment_oracle_reads_no_target_or_metadata_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = _accepted(CORPUS, "an:training:same")
    target_fields = set(target_adapter_module._selected_fields("an_joint"))
    assert set(accepted.alignment_evidence.raw_source_fields).isdisjoint(
        target_fields
    )

    def unexpected_target_access(*_args, **_kwargs):
        raise AssertionError("raw alignment oracle accessed target-only data")

    monkeypatch.setattr(
        target_adapter_module,
        "load_dilemmadata_target_metadata_index",
        unexpected_target_access,
    )
    monkeypatch.setattr(
        target_adapter_module,
        "_read_target_rows",
        unexpected_target_access,
    )
    oracle = reconstruct_dilemmadata_alignment_evidence(
        accepted.record,
        accepted.piece,
    )
    assert oracle == accepted.alignment_evidence

    first = accepted.alignment_evidence.rows[0]
    forged = _reseal_alignment_rows(
        accepted,
        (replace(first, onset_qn=RationalTime(1, 2)), *accepted.alignment_evidence.rows[1:]),
    )
    outcome = build_dilemmadata_target_sidecar(
        replace(accepted, alignment_evidence=forged)
    )
    assert isinstance(outcome, DilemmadataTargetQuarantine)
    assert outcome.categories == ("dilemmadata.target.alignment_binding_mismatch",)


def test_forged_record_binding_is_rejected_before_target_metadata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = _accepted(CORPUS, "an:training:same")
    forged_record = replace(
        accepted.record,
        path=tmp_path / "untrusted" / accepted.record.path.name,
    )

    def unexpected_metadata_read(*_args, **_kwargs):
        raise AssertionError("metadata must not be read before record validation")

    monkeypatch.setattr(
        target_adapter_module,
        "load_dilemmadata_target_metadata_index",
        unexpected_metadata_read,
    )
    outcome = convert_dilemmadata_target_sidecar(
        forged_record,
        accepted.piece,
        accepted.alignment_evidence,
    )
    assert isinstance(outcome, DilemmadataTargetQuarantine)
    assert outcome.categories == ("dilemmadata.target.record_binding_mismatch",)


def test_merged_tie_requires_all_source_target_rows_to_agree(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    header, rows = _read_tsv(source)
    columns = {name: header.index(name) for name in header}
    continuation = rows[0].copy()
    continuation[columns["onset_div"]] = "480"
    continuation[columns["s_offset_frac"]] = "1"
    continuation[columns["s_isOnset"]] = "False"
    _write_tsv(source, header, [rows[0], continuation, rows[1]])

    agreed_raw = _accepted(copied, "an:training:same")
    oracle = reconstruct_dilemmadata_alignment_evidence(
        agreed_raw.record,
        agreed_raw.piece,
    )
    assert oracle == agreed_raw.alignment_evidence
    assert any(row.tie_continuation for row in oracle.rows)
    assert len({row.canonical_note_id for row in oracle.rows}) < len(oracle.rows)
    agreed = build_dilemmadata_target_sidecar(agreed_raw)
    assert isinstance(agreed, DilemmadataTargetAccepted)
    agreed_stats = _family(agreed, "dilemmadata.an.note.scale_degree")
    assert agreed_stats.merged_tie_agreement_count == 1
    assert agreed_stats.merged_tie_conflict_count == 0
    scale_degree = next(
        target
        for target in agreed.target_bundle.targets
        if target.task_id == "dilemmadata.an.note.scale_degree"
    )
    assert scale_degree.availability_mask == (True, True)

    _set_cell(source, 1, "note_degree", "2")
    conflicted_raw = _accepted(copied, "an:training:same")
    conflicted = build_dilemmadata_target_sidecar(conflicted_raw)
    assert isinstance(conflicted, DilemmadataTargetAccepted)
    assert _raw_evidence(agreed_raw) == _raw_evidence(conflicted_raw)
    conflict_stats = _family(conflicted, "dilemmadata.an.note.scale_degree")
    assert conflict_stats.merged_tie_agreement_count == 0
    assert conflict_stats.merged_tie_conflict_count == 1
    scale_degree = next(
        target
        for target in conflicted.target_bundle.targets
        if target.task_id == "dilemmadata.an.note.scale_degree"
    )
    assert scale_degree.availability_mask == (False, True)
    assert scale_degree.values == (None, "3")
    assert "dilemmadata.target.alignment_conflict" in {
        row.code for row in conflicted.target_bundle.diagnostics
    }


def test_point_duplicate_conflict_is_masked_without_synthetic_negative(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    source = copied / "pitch_arrays" / "DLC" / "demo" / "same.tsv"
    header, rows = _read_tsv(source)
    columns = {name: header.index(name) for name in header}
    duplicate = rows[1].copy()
    duplicate[columns["onset_div"]] = "480"
    duplicate[columns["quarterbeats_playthrough"]] = "1"
    duplicate[columns["cadence_type"]] = "HC"
    duplicate[columns["pitch"]] = "67"
    duplicate[columns["step"]] = "G"
    _write_tsv(source, header, [rows[0], duplicate, rows[1]])

    outcome = _target(copied, "dlc:demo:same")
    cadence = next(
        target
        for target in outcome.target_bundle.targets
        if target.task_id == "dilemmadata.dlc.cadence"
    )
    stats = _family(outcome, "dilemmadata.dlc.cadence")
    assert stats.conflict_count == 1
    assert cadence.availability_mask == (False,)
    assert cadence.values == (None,)


def test_exact_alignment_retains_unaligned_event_and_half_open_span() -> None:
    outcome = _target(CORPUS, "dlc:demo:same")
    sample = _sample(outcome)
    aligned = {
        family.task_id: family
        for family in align_sample_targets(
            sample.canonical_piece,
            sample.raw_graph,
            sample,
        )
    }
    roman = aligned["dilemmadata.dlc.harmony.roman_numeral"]
    assert len(roman.rows) == 4
    assert sum(row.local_entity_index == -1 for row in roman.rows) == 1

    cadence = next(
        target
        for target in outcome.target_bundle.targets
        if target.task_id == "dilemmadata.dlc.cadence"
    )
    cadence_span_id = cadence.entity_ids[0]
    moved_spans = tuple(
        replace(
            span,
            start_qn=RationalTime(1, 2),
            end_qn=RationalTime(1, 2),
        )
        if span.annotation_id == cadence_span_id
        else span
        for span in outcome.target_bundle.alignment_spans
    )
    moved = replace(outcome.target_bundle, alignment_spans=tuple(sorted(
        moved_spans,
        key=lambda span: (span.start_qn, span.end_qn, span.annotation_id),
    )))
    moved_sample = prepare_multisource_sample(
        sample.canonical_piece,
        target_sidecar=moved,
    )
    moved_cadence = next(
        family
        for family in align_sample_targets(
            moved_sample.canonical_piece,
            moved_sample.raw_graph,
            moved_sample,
        )
        if family.task_id == "dilemmadata.dlc.cadence"
    )
    assert len(moved_cadence.rows) == 1
    assert moved_cadence.rows[0].availability
    assert moved_cadence.rows[0].local_entity_index == -1
    assert moved_cadence.rows[0].entity_node_type is None


def test_raw_cache_dataset_sidecar_alignment_tensorization_and_logits(
    tmp_path: Path,
) -> None:
    cache = CorpusCacheConfig(tmp_path / "cache")
    index, report = build_dilemmadata_corpus_cache(
        CORPUS,
        cache_config=cache,
        identity=_fixture_identity(CORPUS),
    )
    assert report.accepted_count == 3
    dataset = IndexedMultiSourceDataset(index, cache_config=cache)
    outcomes = {
        outcome.piece_id: outcome
        for record_id in (
            "an:training:same",
            "an:validation:same-alt",
            "dlc:demo:same",
        )
        for outcome in (_target(CORPUS, record_id),)
    }
    raw_samples = tuple(dataset[index] for index in range(len(dataset)))
    samples = tuple(
        attach_target_bundle(sample, outcomes[sample.piece_id].target_bundle)
        for sample in raw_samples
    )
    assert all(
        sample.raw_graph_fingerprint == raw.raw_graph_fingerprint
        and graph_fingerprint(sample.raw_graph) == graph_fingerprint(raw.raw_graph)
        and model_input_fingerprint(sample.raw_graph)
        == model_input_fingerprint(raw.raw_graph)
        for raw, sample in zip(raw_samples, samples, strict=True)
    )
    assert all(len(sample.target_availability) == 40 for sample in samples)
    batch = collate_multisource_samples(samples)
    assert batch.raw_graph_batch.num_graphs == 3
    assert len(batch.target_batches) == 40
    assert any(
        target.task_id.startswith("dilemmadata.")
        and target.encoding_kind == "open_string_cpu"
        and isinstance(target.values, tuple)
        and not target.model_ready
        for target in batch.target_batches
    )
    assert any(
        target.task_id.startswith("dilemmadata.")
        and target.encoding_kind == "closed_categorical_index"
        and isinstance(target.values, torch.Tensor)
        and target.model_ready
        for target in batch.target_batches
    )
    assert any(
        target.task_id.startswith("dilemmadata.")
        and bool(torch.any(~target.entity_index_mask))
        for target in batch.target_batches
    )
    assert all(not value.is_cuda for target in batch.target_batches for value in (
        target.availability_mask,
        target.entity_indices,
        target.entity_index_mask,
        target.entity_node_type_codes,
        target.sample_indices,
    ))

    raw_batch = collate_multisource_samples(raw_samples)
    torch.manual_seed(919)
    model = _model().eval()
    with torch.no_grad():
        raw_predictions = model.predict(raw_batch.raw_graph_batch)[1]
        target_predictions = model.predict(batch.raw_graph_batch)[1]
    assert all(
        left.task_id == right.task_id
        and torch.equal(left.logits, right.logits)
        and torch.equal(left.global_entity_indices, right.global_entity_indices)
        and torch.equal(left.candidate_node_type_codes, right.candidate_node_type_codes)
        for left, right in zip(raw_predictions, target_predictions, strict=True)
    )


def test_target_replacement_and_masking_change_only_supervision() -> None:
    outcome = _target(CORPUS, "dlc:demo:same")
    raw = _accepted(CORPUS, "dlc:demo:same")
    cadence = next(
        target
        for target in outcome.target_bundle.targets
        if target.task_id == "dilemmadata.dlc.cadence"
    )
    replaced_cadence = replace(cadence, values=("HC",))
    masked_cadence = replace(
        cadence,
        values=(None,),
        availability_mask=(False,),
        source=(None,),
        provenance_ids=(None,),
    )

    def variant(replacement) -> TargetBundle:
        return replace(
            outcome.target_bundle,
            targets=tuple(
                replacement if target.task_id == cadence.task_id else target
                for target in outcome.target_bundle.targets
            ),
        )

    bundles = (outcome.target_bundle, variant(replaced_cadence), variant(masked_cadence))
    assert len({target_bundle_fingerprint(bundle) for bundle in bundles}) == 3
    batches = tuple(
        collate_multisource_samples((
            prepare_multisource_sample(raw.piece, target_sidecar=bundle),
        ))
        for bundle in bundles
    )
    torch.manual_seed(923)
    model = _model().eval()
    with torch.no_grad():
        predictions = tuple(model.predict(batch.raw_graph_batch)[1] for batch in batches)
    reference = predictions[0]
    for candidate in predictions[1:]:
        assert all(
            torch.equal(left.logits, right.logits)
            and torch.equal(left.global_entity_indices, right.global_entity_indices)
            for left, right in zip(reference, candidate, strict=True)
        )
    cadence_batches = tuple(
        next(
            target
            for target in batch.target_batches
            if target.task_id == cadence.task_id
        )
        for batch in batches
    )
    assert not torch.equal(cadence_batches[0].values, cadence_batches[1].values)
    assert not torch.equal(
        cadence_batches[0].availability_mask,
        cadence_batches[2].availability_mask,
    )
