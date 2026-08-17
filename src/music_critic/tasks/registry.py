"""Resolution of the immutable core ontology and explicit sidecar extensions."""

from __future__ import annotations

from types import MappingProxyType

from music_critic.tasks.dilemmadata_registry import (
    DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION,
    DILEMMADATA_TARGET_FAMILIES,
    DILEMMADATA_TARGET_FAMILY_BY_ID,
)
from music_critic.tasks.ontology import (
    TARGET_FAMILIES,
    TARGET_FAMILY_BY_ID,
    TARGET_ONTOLOGY_VERSION,
    TargetFamilySpec,
)


CORE_TARGET_REGISTRY_ID = f"music_critic.core@{TARGET_ONTOLOGY_VERSION}"
DILEMMADATA_TARGET_REGISTRY_ID = (
    "music_critic.dilemmadata@"
    f"{DILEMMADATA_SOURCE_NATIVE_FAMILY_REGISTRY_VERSION}"
)

TARGET_REGISTRY_EXTENSIONS = MappingProxyType(
    {
        DILEMMADATA_TARGET_REGISTRY_ID: DILEMMADATA_TARGET_FAMILIES,
    }
)


def target_family_spec(task_id: str) -> TargetFamilySpec:
    """Resolve one core or explicitly registered extension family."""

    spec = TARGET_FAMILY_BY_ID.get(task_id)
    if spec is None:
        spec = DILEMMADATA_TARGET_FAMILY_BY_ID.get(task_id)
    if spec is None:
        raise KeyError(task_id)
    return spec


def target_families_for_registries(
    extension_registry_ids: tuple[str, ...] = (),
) -> tuple[TargetFamilySpec, ...]:
    """Return core plus explicitly requested extension tasks in stable order."""

    if tuple(sorted(extension_registry_ids)) != extension_registry_ids:
        raise ValueError("target registry extension IDs must be unique and sorted")
    if len(extension_registry_ids) != len(set(extension_registry_ids)):
        raise ValueError("target registry extension IDs must be unique and sorted")
    extensions: list[TargetFamilySpec] = []
    for registry_id in extension_registry_ids:
        try:
            extensions.extend(TARGET_REGISTRY_EXTENSIONS[registry_id])
        except KeyError as exc:
            raise ValueError(f"unknown target registry extension {registry_id!r}") from exc
    combined = tuple(sorted((*TARGET_FAMILIES, *extensions), key=lambda item: item.task_id))
    if len(combined) != len({item.task_id for item in combined}):
        raise ValueError("target registry extensions contain duplicate task IDs")
    return combined


def registry_extensions_for_task_ids(
    task_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Infer complete extension registries and reject partial task inventories."""

    supplied = set(task_ids)
    extension_ids: list[str] = []
    for registry_id, families in TARGET_REGISTRY_EXTENSIONS.items():
        registry_tasks = {family.task_id for family in families}
        overlap = supplied & registry_tasks
        if overlap and overlap != registry_tasks:
            raise ValueError(
                f"target task inventory contains a partial extension {registry_id!r}"
            )
        if overlap:
            extension_ids.append(registry_id)
    expected = tuple(
        family.task_id
        for family in target_families_for_registries(tuple(sorted(extension_ids)))
    )
    if task_ids != expected:
        raise ValueError(
            "target task inventory must contain core plus complete extensions in order"
        )
    return tuple(sorted(extension_ids))


__all__ = [
    "CORE_TARGET_REGISTRY_ID",
    "DILEMMADATA_TARGET_REGISTRY_ID",
    "TARGET_REGISTRY_EXTENSIONS",
    "target_families_for_registries",
    "target_family_spec",
    "registry_extensions_for_task_ids",
]
