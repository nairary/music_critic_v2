from __future__ import annotations

import pytest

from music_critic.graph import (
    FEATURE_REGISTRY_VERSION,
    MANDATORY_NODE_TYPES,
    RAW_FEATURE_REGISTRY,
    FeatureRegistry,
    FeatureSpec,
)


def test_registry_is_versioned_complete_and_raw_safe() -> None:
    assert RAW_FEATURE_REGISTRY.version == FEATURE_REGISTRY_VERSION == "1.0.0"
    assert {spec.node_type for spec in RAW_FEATURE_REGISTRY.specs} == set(
        MANDATORY_NODE_TYPES
    )
    assert all(spec.raw_inference_safe for spec in RAW_FEATURE_REGISTRY.specs)
    identities = [(spec.node_type, spec.name) for spec in RAW_FEATURE_REGISTRY.specs]
    assert len(identities) == len(set(identities))


def test_registry_contains_no_supervision_or_provenance_names() -> None:
    names = " ".join(spec.name for spec in RAW_FEATURE_REGISTRY.specs).lower()
    for forbidden in (
        "target",
        "theory",
        "chord",
        "roman",
        "scale_degree",
        "role",
        "section",
        "phrase",
        "split",
        "dataset",
        "source_group",
        "provenance",
        "confidence",
    ):
        assert forbidden not in names


def test_registry_rejects_unsafe_and_duplicate_features() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        FeatureSpec(
            name="gold_chord",
            node_type="beat",
            kind="categorical",
            vocabulary_size=2,
            raw_inference_safe=False,
        )
    spec = FeatureSpec(
        name="pitch",
        node_type="note",
        kind="categorical",
        vocabulary_size=128,
    )
    with pytest.raises(ValueError, match="unique"):
        FeatureRegistry("bad", (spec, spec))
