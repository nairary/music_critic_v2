"""Optional exact-device CUDA+AMP acceptance for Phase 8A mechanics."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import warnings

import pytest
import torch

from scripts.accept_phase8a_cuda_amp import (
    PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION,
    _guard_host_materialization,
    _mutate_unmasked_velocity,
    _pitch_mutation_evidence,
    _portable_contract_bindings,
    _portable_policy_projection,
    _validate_portable_cpu_report,
    build_phase8a_cuda_amp_hardware_report,
)
from scripts.accept_phase8a_hierarchical_masking import (
    build_phase8a_bounded_acceptance_report,
)
from music_critic.ssl.contracts import canonical_sha256
from music_critic.ssl.data import collate_ssl_samples
from music_critic.ssl.hierarchical_masking import (
    HIERARCHY_MASK_POLICIES,
    INDEPENDENT_NOTE_PITCH,
)
from music_critic.ssl.hierarchy_fixture import (
    build_phase8a_hierarchy_fixture,
)


def test_phase8a_cuda_acceptance_rejects_abstract_cuda() -> None:
    with pytest.raises(
        ValueError,
        match="requires cuda:0",
    ):
        build_phase8a_cuda_amp_hardware_report(device="cuda")


def test_documented_cuda_acceptance_module_cli_is_importable() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "scripts.accept_phase8a_cuda_amp",
            "--help",
        ),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--device DEVICE" in completed.stdout
    assert "--expected-device-name" in completed.stdout
    assert "--portable-report PORTABLE_REPORT" in completed.stdout


def test_exact_final_path_requires_rtx_name_and_portable_report() -> None:
    with pytest.raises(
        ValueError,
        match="NVIDIA GeForce RTX 3090",
    ):
        build_phase8a_cuda_amp_hardware_report(
            device="cuda:0",
            expected_head="0" * 40,
            require_clean=True,
        )
    with pytest.raises(
        ValueError,
        match="portable CPU report",
    ):
        build_phase8a_cuda_amp_hardware_report(
            device="cuda:0",
            expected_head="0" * 40,
            expected_device_name="NVIDIA GeForce RTX 3090",
            require_clean=True,
        )


def test_cuda_acceptance_rejects_invalid_cublas_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(
        RuntimeError,
        match="cublas_workspace_config_invalid",
    ):
        build_phase8a_cuda_amp_hardware_report(device="cuda:0")


def test_host_guard_tracks_graph_storage_aliases() -> None:
    fixture = build_phase8a_hierarchy_fixture()
    batch = collate_ssl_samples(fixture.raw_samples("train"))
    graph = batch.raw_graph_batch
    alias = graph["note"].x_cat[:, 0]

    with _guard_host_materialization(graph):
        with pytest.raises(
            AssertionError,
            match="bulk tensor tolist",
        ):
            alias.tolist()


def test_source_isolation_mutates_only_available_velocity_value() -> None:
    fixture = build_phase8a_hierarchy_fixture()
    batch = collate_ssl_samples(fixture.raw_samples("train"))
    before_available = (
        batch.raw_graph_batch["note"].x_cont_available.detach().clone()
    )
    before_values = batch.raw_graph_batch["note"].x_cont.detach().clone()

    changed = _mutate_unmasked_velocity(batch)

    assert torch.equal(
        changed.raw_graph_batch["note"].x_cont_available,
        before_available,
    )
    changed_positions = torch.nonzero(
        changed.raw_graph_batch["note"].x_cont != before_values,
        as_tuple=False,
    )
    assert tuple(changed_positions.shape) == (1, 2)
    row, column = (int(value) for value in changed_positions[0])
    assert bool(before_available[row, column])
    assert (
        int(batch.raw_graph_batch["note"].ptr[0])
        <= row
        < int(batch.raw_graph_batch["note"].ptr[1])
    )


def test_portable_cpu_report_binding_is_semantic_and_failure_closed() -> None:
    report = build_phase8a_bounded_acceptance_report()
    policy_projection = _portable_policy_projection(report)
    evidence = _validate_portable_cpu_report(
        report,
        portable_report_sha256="a" * 64,
        contracts=_portable_contract_bindings(),
        policy_contract_fingerprint=report[
            "hierarchy_mask_policy_contract_fingerprint"
        ],
        fixture_fingerprints=report["fixture"]["fingerprints"],
        model_metadata_fingerprint=report[
            "model_contract_metadata_fingerprint"
        ],
        portable_policies=policy_projection,
    )
    assert evidence["validated"] is True
    assert evidence["contracts_exact"] is True
    assert evidence["policy_fingerprints_exact"] is True

    forged = deepcopy(report)
    forged["contracts"]["ssl_training_report"] = "forged"
    with pytest.raises(
        RuntimeError,
        match="portable_cpu_report_mismatch:contracts_exact",
    ):
        _validate_portable_cpu_report(
            forged,
            portable_report_sha256="a" * 64,
            contracts=_portable_contract_bindings(),
            policy_contract_fingerprint=report[
                "hierarchy_mask_policy_contract_fingerprint"
            ],
            fixture_fingerprints=report["fixture"]["fingerprints"],
            model_metadata_fingerprint=report[
                "model_contract_metadata_fingerprint"
            ],
            portable_policies=policy_projection,
        )


def test_negative_preference_margin_is_non_gating_cpu_semantics() -> None:
    fixture = build_phase8a_hierarchy_fixture()
    sample = fixture.raw_samples("train")[0]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CUDA is not available.*",
            category=UserWarning,
        )
        no_leakage, pitch_sensitive = _pitch_mutation_evidence(
            fixture.train_pieces[0],
            sample,
            policy=INDEPENDENT_NOTE_PITCH,
            device=torch.device("cpu"),
        )

    assert no_leakage["passed"] is True
    assert pitch_sensitive["correct_minus_mutated_margin"] < 0
    assert pitch_sensitive["preference_status"] == "not_observed"
    assert pitch_sensitive["preference_is_acceptance_criterion"] is False
    assert pitch_sensitive["passed"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 8A CUDA+AMP acceptance requires actual CUDA",
)
def test_all_phase8a_policies_and_mixture_on_explicit_cuda_zero() -> None:
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cpu_rng = torch.get_rng_state().clone()
    previous_cuda_rng = [
        value.clone() for value in torch.cuda.get_rng_state_all()
    ]
    previous_workspace_present = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    report = build_phase8a_cuda_amp_hardware_report(
        device="cuda:0",
        require_clean=False,
    )
    assert (
        torch.are_deterministic_algorithms_enabled()
        is previous_algorithms
    )
    assert (
        torch.is_deterministic_algorithms_warn_only_enabled()
        is previous_warn_only
    )
    assert torch.backends.cudnn.benchmark is previous_benchmark
    assert (
        torch.backends.cudnn.deterministic
        is previous_cudnn_deterministic
    )
    assert torch.equal(torch.get_rng_state(), previous_cpu_rng)
    assert all(
        torch.equal(current, previous)
        for current, previous in zip(
            torch.cuda.get_rng_state_all(),
            previous_cuda_rng,
            strict=True,
        )
    )
    assert ("CUBLAS_WORKSPACE_CONFIG" in os.environ) is (
        previous_workspace_present
    )
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == previous_workspace

    assert report["hardware_evidence_contract_version"] == (
        PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION
    ) == "1.0.0"
    assert report["portable"] is False
    assert report["runtime"]["requested_device"] == "cuda:0"
    assert report["runtime"]["resolved_device"] == "cuda:0"
    assert report["runtime"]["amp_enabled"] is True
    assert report["runtime"]["amp_dtype"] == "torch.float16"
    assert report["runtime"]["deterministic_algorithms"] is True
    assert report["runtime"]["cudnn_benchmark"] is False
    assert report["runtime"]["cudnn_deterministic"] is True
    assert report["runtime"]["cublas_workspace_config"] in {
        ":4096:8",
        ":16:8",
    }
    assert isinstance(report["runtime"]["gpu_name"], str)
    assert report["runtime"]["gpu_name"]
    assert report["runtime"]["driver_version"] is None or (
        isinstance(report["runtime"]["driver_version"], str)
        and report["runtime"]["driver_version"]
    )
    assert report["all_five_policies_exercised"] is True
    assert len(report["policies"]) == len(HIERARCHY_MASK_POLICIES) == 5
    assert tuple(report["policies"]) == HIERARCHY_MASK_POLICIES
    assert report["global_peak_allocated_bytes"] > 0
    assert report["global_peak_reserved_bytes"] > 0
    assert report["quality_claim"] is None
    assert report["performance_thresholds"] is None
    assert report["loss_decrease_is_acceptance_criterion"] is False
    assert (
        report["correct_target_preference_is_acceptance_criterion"]
        is False
    )
    assert report["gpu_values_are_portable_fingerprint_inputs"] is False
    assert report["portable_binding"][
        "portable_cpu_report_validation"
    ]["provided"] is False
    assert report["contracts"]["ssl_training_report"] == "1.2.2"
    assert report["contracts"]["prepared_binding"] == "1.1.0"
    assert report["contracts"]["hierarchy_prepared_binding"] == "1.1.0"
    assert report["contracts"]["hierarchy_unavailable_reason"] == "1.0.0"
    fingerprint = report["hardware_evidence_fingerprint"]
    fingerprint_payload = dict(report)
    del fingerprint_payload["hardware_evidence_fingerprint"]
    assert canonical_sha256(fingerprint_payload) == fingerprint

    for policy, evidence in report["policies"].items():
        assert evidence["policy"] == policy
        assert evidence["requested_device"] == "cuda:0"
        assert evidence["resolved_device"] == "cuda:0"
        assert evidence["amp_enabled"] is True
        assert evidence["amp_dtype"] == "torch.float16"
        assert evidence["prepared_binding_validated_on_cuda"] is True
        assert evidence["all_model_facing_tensors_on_cuda_0"] is True
        assert evidence["deterministic_repeat_bit_exact"] is True
        assert evidence["forward_tensors"][
            "all_tensors_on_cuda_0"
        ] is True
        assert evidence["forward_tensors"][
            "all_floating_tensors_finite"
        ] is True
        assert evidence["forward_tensors"]["tensor_count"] > 0
        assert evidence["losses"][
            "all_required_objectives_float32"
        ] is True
        assert evidence["losses"][
            "all_required_objectives_finite"
        ] is True
        assert evidence["losses"][
            "all_required_objectives_on_cuda_0"
        ] is True
        assert set(evidence["losses"]["dtypes"].values()) == {
            "torch.float32"
        }
        assert evidence["gradients"][
            "all_expected_paths_finite_nonzero"
        ] is True
        assert evidence["gradients"][
            "all_present_gradients_finite"
        ] is True
        assert evidence["gradients"][
            "all_present_gradients_on_cuda_0"
        ] is True
        assert evidence["gradients"][
            "present_gradient_tensor_count"
        ] > 0
        assert evidence["raw_cpu_source_unchanged"] is True
        assert evidence["raw_cuda_graph_unchanged"] is True
        assert evidence["prepared_binding_unchanged"] is True
        assert evidence[
            "model_parameters_unchanged_without_optimizer"
        ] is True
        assert evidence[
            "no_graph_sized_accelerator_to_host_materialization"
        ] is True
        assert evidence[
            "prepared_mutation_rejected_before_encoder"
        ] is True
        assert evidence[
            "masked_pitch_and_collateral_track_slots_closed"
        ] is True
        assert evidence["source_sample_isolation"] is True
        assert evidence[
            "target_provenance_diagnostic_blindness"
        ] is True
        no_leakage = evidence["no_leakage_mutation_evidence"]
        pitch_sensitive = evidence[
            "pitch_sensitive_reconstruction_evidence"
        ]
        assert no_leakage is not pitch_sensitive
        assert no_leakage["evidence_kind"] == "no_leakage_mutation"
        assert no_leakage["contract_version"] == "1.0.0"
        assert no_leakage["passed"] is True
        assert no_leakage[
            "online_embeddings_bit_exact_after_masked_mutation"
        ] is True
        assert no_leakage[
            "online_predictions_bit_exact_after_masked_mutation"
        ] is True
        assert pitch_sensitive["evidence_kind"] == (
            "pitch_sensitive_reconstruction"
        )
        assert pitch_sensitive["contract_version"] == "1.0.0"
        assert pitch_sensitive["full_view_target_changed"] is True
        assert pitch_sensitive["reconstruction_loss_changed"] is True
        assert pitch_sensitive[
            "preference_is_acceptance_criterion"
        ] is False
        assert pitch_sensitive["diagnostic_compute_dtype"] == "float32"
        assert math.isfinite(
            pitch_sensitive["correct_minus_mutated_margin"]
        )
        assert pitch_sensitive["margin_floor"] == pytest.approx(
            8.0 * torch.finfo(torch.float32).eps
        )
        assert pitch_sensitive["preference_status"] == (
            "observed"
            if pitch_sensitive["correct_target_preference_observed"]
            else "not_observed"
        )
        assert pitch_sensitive["passed"] is True
        assert evidence["elapsed_seconds"] > 0
        assert evidence["peak_allocated_bytes"] > 0
        assert evidence["peak_reserved_bytes"] > 0

    mixture = report["mixture"]
    assert mixture["repeat_bit_exact"] is True
    assert mixture["resolved_only_from_eligible_set"] is True
    assert mixture[
        "resolution_bound_to_config_fingerprint"
    ] is True
    assert mixture["finite_cuda_amp_forward"] is True
    assert mixture["forward_tensors"]["all_tensors_on_cuda_0"] is True
    assert mixture["forward_tensors"][
        "all_floating_tensors_finite"
    ] is True
    assert mixture[
        "unavailable_policy_excluded_before_encoder"
    ] is True
    assert mixture[
        "unavailable_only_configuration_rejected"
    ] is True
    assert mixture["unavailable_policy_silent_fallback"] is False
    control = report["independent_control"]
    assert control["portable_binding_bit_exact"] is True
    assert control["cuda_amp_model_facing_output_bit_exact"] is True
    assert control["phase7a_forward_tensors"][
        "all_tensors_on_cuda_0"
    ] is True
    assert control["phase8a_forward_tensors"][
        "all_tensors_on_cuda_0"
    ] is True
