from __future__ import annotations

from music_critic.ssl.phase8b_acceptance import (
    PHASE8B_BOUNDED_COMPARISON_CONTRACT_VERSION,
    phase8b_cross_policy_manual_oracle,
    run_phase8b_bounded_comparison,
)


def test_bounded_comparison_overfits_each_family_and_combined_mode() -> None:
    report = run_phase8b_bounded_comparison(steps=2)
    assert report["contract_version"] == (
        PHASE8B_BOUNDED_COMPARISON_CONTRACT_VERSION
    )
    assert report["shared_base_initialization"]
    assert report["shared_new_head_initialization"]
    assert report["all_variant_train_overfit_checks_passed"]
    assert report["all_reports_retain_zero_cuda_predictions"]
    assert report["cross_policy_manual_oracle"] == (
        phase8b_cross_policy_manual_oracle()
    )
    assert report["protocol"][
        "family_global_numerator_denominator_aggregation"
    ]
    assert report["protocol"]["one_weight_application_per_available_family"]
    assert not report["protocol"]["availability_renormalization"]
    assert tuple(row["variant"] for row in report["variants"]) == (
        "phase7a_control",
        "phase8a_masks_old_objectives",
        "onset_only",
        "beat_only",
        "bar_only",
        "track_only",
        "multilevel_equal_weight",
    )
    for variant in report["variants"]:
        assert "scheduled_pass_divisor" not in variant
        assert variant["gradient_coverage"]["nonzero_gradient_tensor_count"] > 0
        for boundary in ("initial", "final"):
            for stage in ("train", "held_out"):
                evidence = variant[boundary][stage]
                assert evidence["families"]
                for family in evidence["families"]:
                    assert family["eligible_denominator"] > 0
                    assert family["mean_loss"] is not None
                    assert family["anti_collapse"] is not None
                    assert family["anti_collapse"]["row_count"] > 0
