from __future__ import annotations

from music_critic.eda import (
    APPROVED_PROJECTION_REGISTRIES,
    DILEMMADATA_COMMON_TASK_IDS,
    CorpusId,
    ProjectionEvidence,
    SourceValueIdentity,
    SourceValueKind,
)
from music_critic.tasks.dilemmadata_common import (
    COMMON_TASK_IDS,
    DILEMMADATA_COMMON_HARMONIC_REGISTRY,
)


def test_eda_projection_contract_is_bound_to_existing_registry_rows() -> None:
    registry = next(iter(APPROVED_PROJECTION_REGISTRIES.values()))
    existing = DILEMMADATA_COMMON_HARMONIC_REGISTRY
    assert registry.fingerprint == existing.fingerprint
    assert registry.version == existing.contract_version
    assert DILEMMADATA_COMMON_TASK_IDS == COMMON_TASK_IDS

    rows = (*existing.quality_mapping_rows, *existing.inversion_mapping_rows)
    for row in rows:
        projection = ProjectionEvidence(
            source_value=SourceValueIdentity(
                corpus=CorpusId.DILEMMADATA,
                source_task_id=row.source_task_id,
                dialect=row.dialect,
                source_value=row.source_value,
                value_kind=SourceValueKind.SCALAR,
            ),
            mapping_registry=registry,
            common_task_identity=row.common_task_id,
            native_state="available",
            mapping_state=row.state,
            projected_value=row.common_value,
            provenance=(row.evidence_id,),
        )
        assert projection.projected_value == row.common_value
