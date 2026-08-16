from __future__ import annotations

from copy import deepcopy

import pytest

from music_critic.experiments.phase8b2.config import Phase8B2Config
from music_critic.experiments.phase8b2.contracts import (
    Phase8B2ContractError,
    fingerprint,
)
from music_critic.experiments.phase8b2.runner import build_experiment_plan
from music_critic.experiments.phase8b2.schedule import (
    SeedDomains,
    build_variant_schedule,
    validate_paired_schedules,
)


def test_bounded_plan_has_required_cells_and_exact_matched_compute() -> None:
    plan = build_experiment_plan(Phase8B2Config())

    assert plan["protocol"]["protocol_contract"] == (
        "Phase8B2ComparisonProtocol@1.2.0"
    )
    assert len(plan["ssl_cells"]) == 8
    assert len(plan["downstream_cells"]) == 18
    assert {row["variant_id"] for row in plan["ssl_cells"]} == {
        "phase7a_control",
        "phase8a_mask_only",
        "onset_latent",
        "multilevel_equal",
    }
    for row in plan["ssl_cells"]:
        assert row["schedule"]["encoder_forward_count"] == 24
        assert row["schedule"]["raw_sample_exposures"] == 4
        assert row["paired_schedule_evidence"]["primary_compute_matched"]
    assert not plan["training_performed"]
    assert plan["test_membership_metadata_resolved"] is True
    assert plan["test_inference_performed"] is False
    assert plan["test_targets_accessed"] is False
    assert plan["test_metrics_accessed"] is False
    assert not plan["claims"]["pdmx_evidence"]


def test_launch_permutation_does_not_change_plan_or_artifacts() -> None:
    left = Phase8B2Config()
    right = deepcopy(left)
    right.comparison.variants = list(reversed(right.comparison.variants))
    right.comparison.seeds = list(reversed(right.comparison.seeds))

    first = build_experiment_plan(left)
    second = build_experiment_plan(right)

    assert first == second
    assert first["fingerprint"] == second["fingerprint"]


def test_every_binding_field_changes_protocol_fingerprint() -> None:
    base = Phase8B2Config()
    original = build_experiment_plan(base)["protocol"]["fingerprint"]
    mutations = []
    for mutate in (
        lambda value: setattr(value.comparison, "ssl_optimizer_steps", 3),
        lambda value: setattr(value.data, "batch_size", 3),
        lambda value: setattr(value.model, "hidden_dim", 16),
        lambda value: setattr(value.ssl, "mask_rate", 0.4),
        lambda value: setattr(value.optimizer, "learning_rate", 0.01),
        lambda value: setattr(value.scheduler, "name", "cosine"),
        lambda value: setattr(value.device, "non_blocking", True),
        lambda value: setattr(value.comparison, "validation_samples", 2),
    ):
        candidate = deepcopy(base)
        mutate(candidate)
        mutations.append(
            build_experiment_plan(candidate)["protocol"]["fingerprint"]
        )
    assert all(value != original for value in mutations)
    assert len(mutations) == len(set(mutations))


def test_stale_caller_supplied_data_fingerprint_fails_closed() -> None:
    config = Phase8B2Config()
    config.data.validation_membership_fingerprint = "b" * 64
    with pytest.raises(
        Phase8B2ContractError,
        match="validation_membership_fingerprint_mismatch",
    ):
        build_experiment_plan(config)


@pytest.mark.parametrize(
    ("mutation", "category"),
    (
        (
            lambda value: setattr(value.optimizer, "name", "sgd"),
            "optimizer_unsupported",
        ),
        (
            lambda value: setattr(value.device, "amp_dtype", "float32"),
            "amp_dtype_unsupported",
        ),
        (
            lambda value: setattr(value.device, "amp", True),
            "amp_requires_cuda",
        ),
        (
            lambda value: setattr(value.ssl, "epsilon", 1e-6),
            "ssl_epsilon_unsupported",
        ),
        (
            lambda value: setattr(
                value,
                "downstream_task_ids",
                ["theory.local_key.tonic_pc"],
            ),
            "task_subset_primary_datasets_incomplete",
        ),
    ),
)
def test_unsupported_protocol_runtime_fields_fail_structured(
    mutation, category: str
) -> None:
    config = Phase8B2Config()
    mutation(config)
    with pytest.raises(Phase8B2ContractError, match=category):
        build_experiment_plan(config)


def test_natural_schedule_is_explicitly_not_compute_matched() -> None:
    config = Phase8B2Config()
    config.comparison.name = "natural_schedule_diagnostic"
    config.comparison.comparison_mode = "natural_schedule"
    config.comparison.seeds = [17, 29, 43]
    plan = build_experiment_plan(config)
    counts = {
        row["variant_id"]: row["schedule"]["encoder_forward_count"]
        for row in plan["ssl_cells"]
        if row["seed"] == 17
    }
    assert counts["phase7a_control"] == 4
    assert counts["onset_latent"] == 6
    assert counts["phase8a_mask_only"] == 16
    assert counts["multilevel_equal"] == 24
    assert not next(iter(plan["ssl_cells"]))[
        "paired_schedule_evidence"
    ]["primary_compute_matched"]


def test_unmatchable_view_budget_fails_closed() -> None:
    identities = tuple(("hooktheory", str(index)) for index in range(4))
    with pytest.raises(
        Phase8B2ContractError, match="compute_budget_unmatchable"
    ):
        build_variant_schedule(
            "multilevel_equal",
            comparison_mode="encoder_forward_matched",
            logical_updates=2,
            batch_size=2,
            matched_encoder_forwards_per_update=9,
            sample_identity_schedule=identities,
            mask_seed=SeedDomains.create(1).ssl_mask_planning,
        )


def test_paired_schedule_rejects_sample_order_mismatch() -> None:
    identities = tuple(("hooktheory", str(index)) for index in range(4))
    common = dict(
        comparison_mode="encoder_forward_matched",
        logical_updates=2,
        batch_size=2,
        matched_encoder_forwards_per_update=12,
        mask_seed=3,
    )
    left = build_variant_schedule(
        "phase7a_control",
        sample_identity_schedule=identities,
        **common,
    )
    right = build_variant_schedule(
        "onset_latent",
        sample_identity_schedule=tuple(reversed(identities)),
        **common,
    )
    with pytest.raises(
        Phase8B2ContractError, match="paired_binding_mismatch"
    ):
        validate_paired_schedules((left, right))


def test_seed_domains_are_independent_and_deterministic() -> None:
    first = SeedDomains.create(99)
    second = SeedDomains.create(99)
    values = set(first.to_dict().values()) - {"1.1.0", 99}
    assert first == second
    assert len(values) == 6
    assert fingerprint(first.to_dict()) == fingerprint(second.to_dict())
