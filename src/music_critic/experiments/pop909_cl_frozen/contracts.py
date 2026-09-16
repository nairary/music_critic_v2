"""Versioned, local evaluation policy; shared training contracts stay frozen."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

REPORT_SCHEMA = "Pop909ClFrozenEvaluationReport@1.0.0"
PROTOCOL_SCHEMA = "Pop909ClFrozenEvaluationProtocol@1.0.0"
VIEW_IDS = ("native", "metadata_removed")


def fingerprint(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def guard_execution(split: str) -> None:
    """Call before any source, cache, target or checkpoint reader callback."""
    if split.lower() == "test":
        raise ValueError("pop909_frozen.test_access_forbidden")
    if split != "synthetic":
        raise ValueError("pop909_frozen.real_data_requires_future_registered_protocol")


def evaluation_protocol() -> dict[str, Any]:
    from .projection import projection_contract
    from .views import view_contract

    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "task_scope": ["root_pitch_class", "quality_subset9_full17_prediction"],
        "projection": projection_contract(),
        "views": view_contract(),
        "view_order": list(VIEW_IDS),
        "primary_view": "native",
        "secondary_view": "metadata_removed",
        "target_join": "after_raw_predictions_for_both_predeclared_views",
        "alignment": "source_event_start_in_unique_raw_beat_half_open_exact_rational",
        "event_counting": "one_source_chord_block_one_observation_including_same_beat_events",
        "resolution_limit": "events_in_one_raw_beat_share_one_prediction",
        "ambiguous_or_missing_containing_beat": "unavailable_no_nearest_or_span_average",
        "task_masks": "independent_root_quality_intersection_only_for_joint_accuracy",
        "macro_f1": "unweighted_mean_over_observed_true_classes_only",
        "quality_outside_subset_prediction": "retain_full17_confusion_false_positive_and_true_class_false_negative",
        "coverage": "separate_event_piece_group_denominators_and_task_exclusion_reasons",
        "no_chord_and_trailing_spans": "count_separately_not_chord_block_events",
        "empty_denominator": "null_with_reason",
        "model_selection": False,
        "music_quality_score": False,
        "execution_policy": "synthetic_only_test_denied_before_readers",
    }
    return {**payload, "fingerprint": fingerprint(payload)}
