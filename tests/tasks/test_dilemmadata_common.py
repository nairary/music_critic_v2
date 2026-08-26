from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
from pathlib import Path
import shutil

import pytest

from music_critic.adapters import (
    DilemmadataTargetAccepted,
    build_dilemmadata_target_sidecar,
)
from music_critic.graph import build_raw_graph, graph_fingerprint, model_input_fingerprint
from music_critic.tasks import (
    ANALYSISGNN_REFERENCE,
    ANALYSISGNN_REFERENCE_MAPPING_VERSION,
    DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION,
    DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION,
    DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION,
    DILEMMADATA_COMMON_HARMONIC_REGISTRY,
    DILEMMADATA_COMMON_HARMONIC_REGISTRY_VERSION,
    DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION,
    DILEMMADATA_SOURCE_FAMILY_BY_TASK,
    TargetBundle,
    build_dilemmadata_common_harmonic_projection,
    common_projection_fingerprint,
    dilemmadata_common_registry_fingerprint,
    dumps_dilemmadata_common_projection,
    dumps_dilemmadata_common_registry,
    dumps_target_bundle,
    loads_dilemmadata_common_projection,
    map_dilemmadata_common_inversion,
    map_dilemmadata_common_pitch_class,
    map_dilemmadata_common_quality,
    project_dilemmadata_common_harmony,
)
from music_critic.tasks.dilemmadata_common import (
    COMMON_INVERSION_TASK,
    COMMON_LOCAL_KEY_TASK,
    COMMON_PITCH_CLASS_SET_TASK,
    COMMON_QUALITY_TASK,
    DilemmadataCommonLocalKeyValue,
    DilemmadataCommonProjectionError,
    DilemmadataCommonTargetEntry,
)
from music_critic.tasks import dilemmadata_common as common_module
from tests.adapters.test_dilemmadata import (
    CORPUS,
    _accepted,
    _raw_evidence,
    _set_cell,
)


def _target(root: Path, record_id: str) -> DilemmadataTargetAccepted:
    result = build_dilemmadata_target_sidecar(_accepted(root, record_id))
    assert isinstance(result, DilemmadataTargetAccepted)
    return result


def _common_target(projection, task_id: str):
    return next(target for target in projection.targets if target.task_id == task_id)


def _replace_source_target(
    bundle: TargetBundle,
    task_id: str,
    values: tuple[str, ...],
) -> TargetBundle:
    original = next(target for target in bundle.targets if target.task_id == task_id)
    assert len(values) == len(original.entity_ids)
    replacement = replace(
        original,
        values=values,
        availability_mask=tuple(True for _ in values),
        confidence=tuple(None for _ in values),
        source=tuple("dataset" for _ in values),
        provenance_ids=tuple(
            "prov:dilemmadata-target-annotation" for _ in values
        ),
    )
    return replace(
        bundle,
        targets=tuple(
            sorted(
                (
                    replacement if target.task_id == task_id else target
                    for target in bundle.targets
                ),
                key=lambda target: target.task_id,
            )
        ),
    )


def test_versions_registry_reference_and_frozen_contracts() -> None:
    assert DILEMMADATA_COMMON_HARMONIC_PROJECTION_VERSION == "1.0.0"
    assert DILEMMADATA_COMMON_HARMONIC_REGISTRY_VERSION == "1.0.0"
    assert DILEMMADATA_COMMON_MAPPING_EVIDENCE_VERSION == "1.0.0"
    assert DILEMMADATA_COMMON_HARMONIC_AUDIT_REPORT_VERSION == "1.0.0"
    assert DILEMMADATA_COMMON_HARMONIC_AUDIT_MANIFEST_VERSION == "1.0.0"
    assert ANALYSISGNN_REFERENCE_MAPPING_VERSION == "1.0.0"
    assert len(DILEMMADATA_COMMON_HARMONIC_REGISTRY.families) == 6
    assert len(DILEMMADATA_COMMON_HARMONIC_REGISTRY.quality_mapping_rows) == 79
    assert len(DILEMMADATA_COMMON_HARMONIC_REGISTRY.inversion_mapping_rows) == 10
    assert ANALYSISGNN_REFERENCE.commit_sha == (
        "e115182fb29b74bdcb6bf3547ed427d967580947"
    )
    assert ANALYSISGNN_REFERENCE.license_spdx == "MIT"
    assert len(dilemmadata_common_registry_fingerprint()) == 64
    assert dumps_dilemmadata_common_registry() == dumps_dilemmadata_common_registry()
    with pytest.raises(FrozenInstanceError):
        DILEMMADATA_COMMON_HARMONIC_REGISTRY.fingerprint = "0" * 64


def test_every_quality_and_inversion_vocabulary_row_is_explicit() -> None:
    for task_id in (
        "dilemmadata.an.chord.quality",
        "dilemmadata.dlc.chord.quality",
    ):
        vocabulary = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id].vocabulary
        assert vocabulary is not None
        rows = [map_dilemmadata_common_quality(task_id, value) for value in vocabulary]
        assert len(rows) == len(vocabulary)
        assert all(row.state in {"exact", "coarsened"} for row in rows)
        assert all(row.common_value is not None for row in rows)
    for task_id in (
        "dilemmadata.an.chord.inversion",
        "dilemmadata.dlc.chord.inversion",
    ):
        vocabulary = DILEMMADATA_SOURCE_FAMILY_BY_TASK[task_id].vocabulary
        assert vocabulary is not None
        rows = [map_dilemmadata_common_inversion(task_id, value) for value in vocabulary]
        assert all(row.state == "exact" for row in rows)
    assert map_dilemmadata_common_quality(
        "dilemmadata.an.chord.quality",
        "enharmonic equivalent to major triad",
    ).state == "coarsened"
    assert map_dilemmadata_common_quality(
        "dilemmadata.dlc.chord.quality", "+7"
    ).analysisgnn_agreement == "diverge"
    assert map_dilemmadata_common_quality(
        "dilemmadata.dlc.chord.quality", "+M7"
    ).analysisgnn_agreement == "diverge"


def test_mapping_states_do_not_turn_missing_or_bad_values_into_classes() -> None:
    invalid = map_dilemmadata_common_quality(
        "dilemmadata.an.chord.quality", None
    )
    unsupported = map_dilemmadata_common_quality(
        "dilemmadata.an.chord.quality", "not-a-quality"
    )
    assert (invalid.state, invalid.common_value) == ("invalid", None)
    assert (unsupported.state, unsupported.common_value) == ("unsupported", None)
    target = _target(CORPUS, "an:training:same")
    projection = project_dilemmadata_common_harmony(target.target_bundle)
    quality = _common_target(projection, COMMON_QUALITY_TASK)
    assert quality.entries
    assert {entry.state for entry in quality.entries} == {"masked"}
    assert all(entry.common_value is None for entry in quality.entries)
    missing = DilemmadataCommonTargetEntry(
        entity_id="span:synthetic-missing",
        source_task_ids=("dilemmadata.an.chord.root",),
        source_values=("C",),
        state="missing",
        common_value=None,
        field_availability=(("pitch_classes", False),),
        information_loss=(),
        diagnostic_code=None,
        mapping_evidence_ids=(),
        source_provenance_ids=(),
        dependency_entity_ids=(),
    )
    assert missing.state == "missing" and missing.common_value is None


def test_pitch_class_mapping_is_exact_but_declares_enharmonic_loss() -> None:
    sharp = map_dilemmadata_common_pitch_class(
        "dilemmadata.an.chord.root", "C#"
    )
    flat = map_dilemmadata_common_pitch_class(
        "dilemmadata.an.chord.root", "D-"
    )
    tpc = map_dilemmadata_common_pitch_class(
        "dilemmadata.dlc.chord.root", "-1", source_spelling="F"
    )
    conflict = map_dilemmadata_common_pitch_class(
        "dilemmadata.dlc.chord.root", "-1", source_spelling="F#"
    )
    assert sharp.common_value == flat.common_value == 1
    assert sharp.information_loss == ("enharmonic_spelling_removed",)
    assert tpc.common_value == 5
    assert conflict.state == "invalid" and conflict.common_value is None


def test_inversion_cardinality_conflict_is_ambiguous_and_masked() -> None:
    target = _target(CORPUS, "an:training:same")
    bundle = _replace_source_target(
        target.target_bundle,
        "dilemmadata.an.chord.quality",
        ("major triad", "major triad"),
    )
    bundle = _replace_source_target(
        bundle,
        "dilemmadata.an.chord.inversion",
        ("3", "3"),
    )
    projection = project_dilemmadata_common_harmony(bundle)
    inversion = _common_target(projection, COMMON_INVERSION_TASK)
    assert {entry.state for entry in inversion.entries} == {"ambiguous"}
    assert all(entry.common_value is None for entry in inversion.entries)
    assert {
        entry.diagnostic_code for entry in inversion.entries
    } == {"dilemmadata.common.inversion_cardinality_inconsistent"}


def test_local_key_unknown_mode_is_retained_with_an_independent_mask() -> None:
    target = _target(CORPUS, "dlc:demo:same")
    bundle = _replace_source_target(
        target.target_bundle,
        "dilemmadata.dlc.key.local",
        ("I",),
    )
    source_target = next(
        row for row in bundle.targets if row.task_id == "dilemmadata.dlc.key.local"
    )
    evidence = {
        (source_target.task_id, source_target.entity_ids[0]): {
            "localkey_tpc": "0"
        }
    }
    projection = project_dilemmadata_common_harmony(
        bundle,
        supplemental_source_evidence=evidence,
    )
    entry = _common_target(projection, COMMON_LOCAL_KEY_TASK).entries[0]
    assert entry.state == "exact"
    assert entry.common_value == DilemmadataCommonLocalKeyValue(0, "unknown")
    assert entry.field_availability == (("mode", False), ("tonic_pc", True))


@pytest.mark.parametrize(
    "quality",
    (
        "incomplete dominant-seventh chord",
        "dominant-ninth",
        "French augmented sixth chord",
    ),
)
def test_incomplete_extended_and_augmented_sixth_quality_has_no_fabricated_pcset(
    quality: str,
) -> None:
    mapping = map_dilemmadata_common_quality(
        "dilemmadata.an.chord.quality", quality
    )
    assert mapping.state == "exact"
    assert mapping.common_value == quality
    assert quality not in {
        template.quality
        for template in DILEMMADATA_COMMON_HARMONIC_REGISTRY.quality_templates
    }
    target = _target(CORPUS, "an:training:same")
    bundle = _replace_source_target(
        target.target_bundle,
        "dilemmadata.an.chord.quality",
        (quality, quality),
    )
    pcset = _common_target(
        project_dilemmadata_common_harmony(bundle),
        COMMON_PITCH_CLASS_SET_TASK,
    )
    assert {entry.state for entry in pcset.entries} == {"unsupported"}
    assert all(entry.common_value is None for entry in pcset.entries)


def test_projection_is_deterministic_source_bound_and_source_bundle_immutable() -> None:
    raw = _accepted(CORPUS, "dlc:demo:same")
    target = _target(CORPUS, "dlc:demo:same")
    source_bytes = dumps_target_bundle(target.target_bundle)
    first = build_dilemmadata_common_harmonic_projection(raw, target)
    second = build_dilemmadata_common_harmonic_projection(raw, target)
    payload = dumps_dilemmadata_common_projection(first)
    assert payload == dumps_dilemmadata_common_projection(second)
    assert dumps_target_bundle(target.target_bundle) == source_bytes
    assert common_projection_fingerprint(first) == first.projection_fingerprint
    assert loads_dilemmadata_common_projection(payload) == first
    assert str(CORPUS.resolve()) not in payload


def test_alternative_views_are_not_voted_or_collapsed() -> None:
    first_target = _target(CORPUS, "an:training:same")
    second_target = _target(CORPUS, "an:validation:same-alt")
    first = project_dilemmadata_common_harmony(first_target.target_bundle)
    second = project_dilemmadata_common_harmony(second_target.target_bundle)
    assert first.analysis_view_id != second.analysis_view_id
    assert first.piece_id != second.piece_id
    assert first.projection_fingerprint != second.projection_fingerprint


def test_theory_mutation_changes_common_sidecar_only(tmp_path: Path) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    before_raw = _accepted(copied, "an:training:same")
    before_target = _target(copied, "an:training:same")
    before = build_dilemmadata_common_harmonic_projection(
        before_raw, before_target
    )
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 0, "a_quality", "minor triad")
    after_raw = _accepted(copied, "an:training:same")
    after_target = _target(copied, "an:training:same")
    after = build_dilemmadata_common_harmonic_projection(after_raw, after_target)
    assert _raw_evidence(before_raw) == _raw_evidence(after_raw)
    assert before_target.sidecar_fingerprint != after_target.sidecar_fingerprint
    assert before.projection_fingerprint != after.projection_fingerprint


def test_raw_mutation_cannot_be_hidden_by_resealed_target_or_common_fingerprint(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "corpus"
    shutil.copytree(CORPUS, copied)
    raw = _accepted(copied, "an:training:same")
    target = _target(copied, "an:training:same")
    source = copied / "pitch_arrays" / "AN" / "training" / "same_joint.tsv"
    _set_cell(source, 0, "s_midi", "61")
    with pytest.raises(
        DilemmadataCommonProjectionError,
        match="raw_target_binding_mismatch|source_replay_mismatch",
    ):
        build_dilemmadata_common_harmonic_projection(raw, target)


def test_target_deletion_replacement_and_reordering_leave_raw_model_input_unchanged() -> None:
    raw = _accepted(CORPUS, "dlc:demo:same")
    target = _target(CORPUS, "dlc:demo:same")
    graph = build_raw_graph(raw.piece)
    before = (graph_fingerprint(graph), model_input_fingerprint(graph))
    deleted = replace(
        target.target_bundle,
        targets=tuple(
            row
            for row in target.target_bundle.targets
            if row.task_id != "dilemmadata.dlc.chord.quality"
        ),
    )
    with pytest.raises(
        DilemmadataCommonProjectionError,
        match="source_family_missing",
    ):
        project_dilemmadata_common_harmony(deleted)
    with pytest.raises(ValueError):
        replace(
            target.target_bundle,
            targets=tuple(reversed(target.target_bundle.targets)),
        )
    replacement = _replace_source_target(
        target.target_bundle,
        "dilemmadata.dlc.chord.quality",
        ("m", "m"),
    )
    assert project_dilemmadata_common_harmony(
        replacement
    ).projection_fingerprint != project_dilemmadata_common_harmony(
        target.target_bundle
    ).projection_fingerprint
    after = (graph_fingerprint(graph), model_input_fingerprint(graph))
    assert before == after


def test_no_python_hash_runtime_vocabulary_or_majority_label_path() -> None:
    before = dumps_dilemmadata_common_registry()
    map_dilemmadata_common_quality(
        "dilemmadata.an.chord.quality", "unregistered-runtime-class"
    )
    assert dumps_dilemmadata_common_registry() == before
    source = inspect.getsource(common_module)
    assert "hash(" not in source
    assert "majority" not in source.lower()
    assert "Counter(" not in source
