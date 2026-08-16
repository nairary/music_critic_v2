"""Optional exact-device CUDA+AMP acceptance for Phase 8A mechanics."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import warnings

import pytest
import torch

import music_critic.ssl.phase8a_cuda_acceptance as cuda_acceptance_module
from music_critic.ssl.phase8a_cuda_acceptance import (
    PHASE8A_CUDA_AMP_HARDWARE_EVIDENCE_CONTRACT_VERSION,
    PHASE8A_CPU_CUDA_NUMERICAL_PARITY_ATOL,
    PHASE8A_CPU_CUDA_NUMERICAL_PARITY_RTOL,
    _atomic_write,
    _graphs_cross_device_bit_exact,
    _guard_host_materialization,
    _mutate_unmasked_velocity,
    _pitch_mutation_evidence,
    _portable_contract_bindings,
    _portable_policy_projection,
    _tensor_numerical_parity_evidence,
    _validate_exact_final_source,
    _validate_portable_cpu_report,
    build_phase8a_cuda_amp_hardware_report,
)
from music_critic.ssl.phase8a_acceptance import (
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


def _equivalent_cpu_raw_graphs() -> tuple[object, object]:
    fixture = build_phase8a_hierarchy_fixture()
    graph = collate_ssl_samples(
        fixture.raw_samples("train")
    ).raw_graph_batch
    return graph, deepcopy(graph)


def _graph_surface(graph: object) -> tuple[object, ...]:
    return (
        tuple(graph.node_types),
        tuple(graph.edge_types),
        tuple(str(key) for key in graph._global_store.keys()),
        tuple(
            (
                node_type,
                tuple(str(key) for key in store.keys()),
            )
            for node_type, store in graph.node_items()
        ),
        tuple(
            (
                edge_type,
                tuple(str(key) for key in store.keys()),
            )
            for edge_type, store in graph.edge_items()
        ),
    )


def test_cpu_graph_cross_device_path_accepts_equivalent_batches_without_mutation(
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    left_surface = _graph_surface(left)
    right_surface = _graph_surface(right)

    assert _graphs_cross_device_bit_exact(left, right) is True
    assert _graph_surface(left) == left_surface
    assert _graph_surface(right) == right_surface


@pytest.mark.parametrize("mutation", ("value", "dtype", "shape"))
def test_cpu_graph_cross_device_path_rejects_tensor_changes(
    mutation: str,
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    if mutation == "value":
        right["note"].x_cat[0, 0].add_(1)
    elif mutation == "dtype":
        right["note"].x_cont = right["note"].x_cont.to(torch.float64)
    else:
        right["note"].x_cont = right["note"].x_cont[:, :-1].clone()

    assert _graphs_cross_device_bit_exact(left, right) is False


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_cpu_graph_cross_device_path_rejects_node_attribute_delta(
    mutation: str,
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    if mutation == "missing":
        del right["note"]["x_cont"]
    else:
        right["note"]["phase8a_test_extra"] = torch.zeros(
            right["note"].num_nodes,
            dtype=torch.int64,
        )

    assert _graphs_cross_device_bit_exact(left, right) is False


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_cpu_graph_cross_device_path_rejects_edge_attribute_delta(
    mutation: str,
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    edge_type = right.edge_types[0]
    if mutation == "missing":
        del right[edge_type]["edge_index"]
    else:
        right[edge_type]["phase8a_test_extra"] = torch.zeros(
            right[edge_type].edge_index.shape[1],
            dtype=torch.float32,
        )

    assert _graphs_cross_device_bit_exact(left, right) is False


def test_cpu_graph_cross_device_path_rejects_global_attribute_change(
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    right.raw_only.logical_not_()

    assert _graphs_cross_device_bit_exact(left, right) is False


@pytest.mark.parametrize("surface", ("attribute", "node_store", "edge_store"))
def test_cpu_graph_cross_device_path_rejects_reordered_surface(
    surface: str,
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    if surface == "attribute":
        value = right["note"]["x_cat"]
        del right["note"]["x_cat"]
        right["note"]["x_cat"] = value
    elif surface == "node_store":
        node_type = right.node_types[0]
        store = right._node_store_dict.pop(node_type)
        right._node_store_dict[node_type] = store
    else:
        edge_type = right.edge_types[0]
        store = right._edge_store_dict.pop(edge_type)
        right._edge_store_dict[edge_type] = store

    assert _graphs_cross_device_bit_exact(left, right) is False


@pytest.mark.parametrize(
    ("store_kind", "mutation"),
    (
        ("node", "missing"),
        ("node", "extra"),
        ("edge", "missing"),
        ("edge", "extra"),
    ),
)
def test_cpu_graph_cross_device_path_rejects_store_delta(
    store_kind: str,
    mutation: str,
) -> None:
    left, right = _equivalent_cpu_raw_graphs()
    if store_kind == "node" and mutation == "missing":
        right._node_store_dict.pop(right.node_types[0])
    elif store_kind == "node":
        right["phase8a_test_extra_node"].num_nodes = 0
    elif mutation == "missing":
        right._edge_store_dict.pop(right.edge_types[0])
    else:
        right[
            ("song", "phase8a_test_extra_edge", "song")
        ].edge_index = torch.empty((2, 0), dtype=torch.int64)

    assert _graphs_cross_device_bit_exact(left, right) is False


def test_cpu_graph_cross_device_path_rejects_extra_target_without_access(
) -> None:
    class ForbiddenTarget:
        def __repr__(self) -> str:
            raise AssertionError("target/provenance value must not be read")

    left, right = _equivalent_cpu_raw_graphs()
    right["note"]["target"] = ForbiddenTarget()

    assert _graphs_cross_device_bit_exact(left, right) is False


def test_phase8a_cuda_acceptance_rejects_abstract_cuda() -> None:
    with pytest.raises(
        ValueError,
        match="requires cuda:0",
    ):
        build_phase8a_cuda_amp_hardware_report(device="cuda")


def test_documented_portable_cpu_cli_creates_versioned_report(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "phase8a-cpu.json"
    completed = subprocess.run(
        (
            sys.executable,
            "scripts/accept_phase8a_hierarchical_masking.py",
            "--output",
            str(output),
        ),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["acceptance_contract_version"] == "1.2.0"
    assert report["contracts"]["hierarchical_mask_plan"] == "1.2.0"
    assert len(report["hierarchy_mask_policy_contract_fingerprint"]) == 64
    assert report["model_contract_metadata_fingerprint"]
    assert json.loads(completed.stdout) == report


def _cuda_cli_command(
    *,
    expected_head: str,
    portable_report: Path,
    output: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "scripts/accept_phase8a_cuda_amp.py",
        "--device",
        "cuda:0",
        "--amp",
        "--amp-dtype",
        "float16",
        "--expected-head",
        expected_head,
        "--expected-device-name",
        "NVIDIA GeForce RTX 3090",
        "--portable-report",
        str(portable_report),
        "--output",
        str(output),
    )


def _head(repository_root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_documented_cuda_cli_rejects_wrong_expected_sha_before_cuda(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    portable_report = tmp_path / "portable.json"
    portable_report.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "hardware.json"

    completed = subprocess.run(
        _cuda_cli_command(
            expected_head="0" * 40,
            portable_report=portable_report,
            output=output,
        ),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "phase8a.cuda.expected_head_mismatch" in completed.stderr
    assert not output.exists()


def test_documented_cuda_cli_rejects_missing_portable_report(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    output = tmp_path / "hardware.json"

    completed = subprocess.run(
        _cuda_cli_command(
            expected_head=_head(repository_root),
            portable_report=tmp_path / "missing-portable.json",
            output=output,
        ),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        "phase8a.cuda.portable_cpu_report_unreadable"
        in completed.stderr
    )
    assert not output.exists()


def test_documented_cuda_cli_rejects_dirty_worktree_before_cuda(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    portable_report = tmp_path / "portable.json"
    portable_report.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "hardware.json"
    marker = repository_root / (
        f".phase8a-cli-dirty-test-{tmp_path.name}"
    )
    marker.write_text("dirty\n", encoding="utf-8")
    try:
        completed = subprocess.run(
            _cuda_cli_command(
                expected_head=_head(repository_root),
                portable_report=portable_report,
                output=output,
            ),
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        marker.unlink(missing_ok=True)

    assert completed.returncode != 0
    assert "phase8a.cuda.source_tree_dirty" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("existing", (False, True))
def test_cuda_cli_build_failure_preserves_artifact_atomically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: bool,
) -> None:
    portable_report = tmp_path / "portable.json"
    portable_report.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "hardware.json"
    original = b'{"complete":true}\n'
    if existing:
        output.write_bytes(original)

    def fail_build(**_: object) -> dict[str, object]:
        raise RuntimeError("phase8a.cuda.synthetic_build_failure")

    monkeypatch.setattr(
        cuda_acceptance_module,
        "build_phase8a_cuda_amp_hardware_report",
        fail_build,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accept_phase8a_cuda_amp.py",
            "--device",
            "cuda:0",
            "--amp",
            "--amp-dtype",
            "float16",
            "--expected-head",
            "a" * 40,
            "--expected-device-name",
            "NVIDIA GeForce RTX 3090",
            "--portable-report",
            str(portable_report),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="phase8a.cuda.synthetic_build_failure",
    ):
        cuda_acceptance_module.main()

    if existing:
        assert output.read_bytes() == original
    else:
        assert not output.exists()
    assert tuple(tmp_path.glob(".hardware.json.*.tmp")) == ()


def test_atomic_write_failure_preserves_existing_artifact_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "hardware.json"
    original = b'{"complete":true}\n'
    output.write_bytes(original)

    def fail_replace(self: Path, target: Path) -> Path:
        del self, target
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        _atomic_write(output, '{"replacement":true}')

    assert output.read_bytes() == original
    assert tuple(tmp_path.glob(".hardware.json.*.tmp")) == ()


def test_exact_final_source_checks_dirty_before_shallow_ancestry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_head = "a" * 40
    calls: list[tuple[str, ...]] = []

    def fake_git(*arguments: str) -> str:
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return expected_head
        if arguments == ("status", "--porcelain=v1"):
            return "?? dirty"
        raise AssertionError("ancestry must not run for a dirty tree")

    monkeypatch.setattr(cuda_acceptance_module, "_git", fake_git)

    with pytest.raises(
        RuntimeError,
        match="phase8a.cuda.source_tree_dirty",
    ):
        _validate_exact_final_source(expected_head=expected_head)
    assert calls == [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain=v1"),
    ]


def test_exact_final_source_structures_shallow_ancestry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_head = "a" * 40

    def fake_git(*arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return expected_head
        if arguments == ("status", "--porcelain=v1"):
            return ""
        raise subprocess.CalledProcessError(128, ("git", *arguments))

    monkeypatch.setattr(cuda_acceptance_module, "_git", fake_git)

    with pytest.raises(
        RuntimeError,
        match=(
            "phase8a.cuda.hotfix_ancestor_missing_or_unavailable"
        ),
    ):
        _validate_exact_final_source(expected_head=expected_head)


def test_cross_backend_tolerance_contract_has_fixed_boundary() -> None:
    reference = torch.tensor([1.0], dtype=torch.float32)
    allowance = (
        PHASE8A_CPU_CUDA_NUMERICAL_PARITY_ATOL
        + PHASE8A_CPU_CUDA_NUMERICAL_PARITY_RTOL
    )
    inside = torch.tensor(
        [1.0 + 0.99 * allowance],
        dtype=torch.float32,
    )
    outside = torch.tensor(
        [1.0 + 1.01 * allowance],
        dtype=torch.float32,
    )

    accepted = _tensor_numerical_parity_evidence(
        {"inside": (reference, inside)}
    )
    rejected = _tensor_numerical_parity_evidence(
        {"outside": (reference, outside)}
    )
    cosine_rejected = _tensor_numerical_parity_evidence(
        {
            "rotation": (
                torch.tensor([1.0, 0.0], dtype=torch.float32),
                torch.tensor([0.0, 1.0], dtype=torch.float32),
            )
        },
        rtol=0.0,
        atol=2.0,
    )

    assert accepted["cross_backend_parity_passed"] is True
    assert rejected["cross_backend_parity_passed"] is False
    assert cosine_rejected["cross_backend_parity_passed"] is False
    assert cosine_rejected["minimum_cosine_similarity"] == 0.0
    assert accepted["configured_rtol"] == 1.0e-3
    assert accepted["configured_atol"] == 5.0e-5


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
    reason="documented Phase 8A CUDA CLI requires actual CUDA",
)
def test_documented_cuda_cli_creates_bound_hardware_report(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    portable_report = tmp_path / "portable.json"
    hardware_report = tmp_path / "hardware.json"
    subprocess.run(
        (
            sys.executable,
            "scripts/accept_phase8a_hierarchical_masking.py",
            "--output",
            str(portable_report),
        ),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    completed = subprocess.run(
        _cuda_cli_command(
            expected_head=_head(repository_root),
            portable_report=portable_report,
            output=hardware_report,
        ),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(hardware_report.read_text(encoding="utf-8"))
    assert report["hardware_evidence_contract_version"] == "1.2.2"
    assert report["source"]["expected_head_match"] is True
    assert report["source"]["source_tree_clean"] is True
    assert report["portable_binding"][
        "portable_cpu_report_validation"
    ]["validated"] is True
    assert len(report["hardware_evidence_fingerprint"]) == 64
    assert json.loads(completed.stdout) == report


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
    ) == "1.2.2"
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
    assert report["contracts"]["ssl_training_report"] == "1.2.4"
    assert report["contracts"][
        "cuda_memory_statistics_lifecycle"
    ] == "1.0.0"
    assert report["contracts"]["prepared_binding"] == "1.1.0"
    assert report["contracts"]["hierarchy_prepared_binding"] == "1.2.0"
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
        assert evidence["cuda_memory_statistics_lifecycle"][
            "contract_version"
        ] == "1.0.0"
        assert evidence["cuda_memory_statistics_lifecycle"][
            "logical_device_index"
        ] == 0
        assert evidence["cuda_memory_statistics_lifecycle"][
            "initialized_after"
        ] is True
        assert evidence["prepared_binding_validated_on_cuda"] is True
        assert evidence["all_model_facing_tensors_on_cuda_0"] is True
        assert evidence["deterministic_repeat_bit_exact"] is True
        assert evidence[
            "same_device_cuda_fp32_repeat_bit_exact"
        ] is True
        assert all(
            evidence["cross_backend_exact_invariants"].values()
        )
        parity = evidence["cross_backend_numerical_parity"]
        assert parity["configured_rtol"] == 1.0e-3
        assert parity["configured_atol"] == 5.0e-5
        assert parity["configured_min_cosine_similarity"] == 0.999
        assert parity["compared_tensor_count"] > 0
        assert parity["max_abs_cpu_cuda_difference"] >= 0.0
        assert parity["max_rel_cpu_cuda_difference"] >= 0.0
        assert parity["finite_status_exact"] is True
        assert parity["shapes_exact"] is True
        assert parity["dtypes_exact"] is True
        assert parity["cross_backend_parity_passed"] is True
        assert parity["objective_difference"] >= 0.0
        assert parity["per_node_type"]
        for node_evidence in parity["per_node_type"].values():
            assert node_evidence["configured_rtol"] == 1.0e-3
            assert node_evidence["configured_atol"] == 5.0e-5
            assert (
                node_evidence["configured_min_cosine_similarity"]
                == 0.999
            )
            assert node_evidence["compared_tensor_count"] == 1
            assert (
                node_evidence["max_abs_cpu_cuda_difference"] >= 0.0
            )
            assert (
                node_evidence["max_rel_cpu_cuda_difference"] >= 0.0
            )
            assert (
                node_evidence["cross_backend_parity_passed"] is True
            )
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
