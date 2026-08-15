from __future__ import annotations

import copy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from hydra import compose, initialize
import pytest
import torch

from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.engine import _plain_config, run_ssl_training
from music_critic.ssl.multilevel import PHASE8B_NEW_OBJECTIVE_FAMILIES
from music_critic.ssl.phase8b_engine import (
    Phase8BEngineError,
    _evidence_parameter_groups,
    _optimization_step_evidence,
    _parameter_snapshots,
    _prepare,
    _stage,
)
from music_critic.training.checkpoint import capture_rng_state


_ROOT = Path(__file__).resolve().parents[2]
_FAMILY = {
    "onset_only": "onset_latent",
    "beat_only": "beat_latent",
    "bar_only": "hierarchy_bar_latent",
    "track_only": "track_latent",
}
_POLICY = {
    "onset_only": "onset_pitch_descendants",
    "beat_only": "beat_pitch_descendants",
    "bar_only": "contiguous_bar_pitch_span",
    "track_only": "track_bar_pitch_span",
}
_HIERARCHY_POLICIES = [
    "onset_pitch_descendants",
    "beat_pitch_descendants",
    "contiguous_bar_pitch_span",
    "track_bar_pitch_span",
]


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _common_cli(output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "music_critic.ssl.run",
        "experiment=one_batch",
        "experiment.steps=1",
        "model=hierarchical",
        "model.hidden_dim=8",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=2",
        "model.ffn_multiplier=2",
        "model.dropout=0",
        "data=bounded",
        "data.batch_size=3",
        "device=cpu",
        "optimizer.learning_rate=0.02",
        "optimizer.weight_decay=0",
        "ssl.mask_rate=0.5",
        "ssl.decoder_views=1",
        "ssl.decoder_remask_prob=0",
        "ssl.projector_hidden_dim=8",
        "ssl.decoder_hidden_dim=8",
        f"output_dir={output}",
    ]


def _run_cli(
    output: Path,
    *overrides: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        [*_common_cli(output), *overrides],
        cwd=_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _report(output: Path) -> dict[str, object]:
    return json.loads(
        (output / "one_batch_report.json").read_text(encoding="utf-8")
    )


def _pretrain_config(output: Path, mode: str = "onset_only"):
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name="ssl_training",
            overrides=[
                f"+phase8b_objective={mode}",
                f"+phase8b_masking={mode}",
                "experiment=pretrain",
                "experiment.epochs=2",
                "experiment.collect_gradient_evidence=false",
                "model=hierarchical",
                "model.hidden_dim=8",
                "model.local_gnn_layers=1",
                "model.transformer_layers=1",
                "model.attention_heads=2",
                "model.ffn_multiplier=2",
                "model.dropout=0",
                "data=bounded",
                "data.batch_size=3",
                "data.epoch_size=3",
                "data.validation_epoch_size=2",
                "device=cpu",
                "optimizer.learning_rate=0.02",
                "optimizer.weight_decay=0",
                "ssl.mask_rate=0.5",
                "ssl.decoder_views=1",
                "ssl.decoder_remask_prob=0",
                "ssl.projector_hidden_dim=8",
                "ssl.decoder_hidden_dim=8",
                f"output_dir={output}",
            ],
        )


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _assert_family_global_total(stage: dict[str, object]) -> None:
    objective = stage["objective"]
    expected = sum(
        float(row["configured_weight"])
        * float(row["numerator"])
        / int(row["eligible_denominator"])
        for row in objective["families"]
        if row["active"] and row["available"]
    )
    assert stage["total_ssl_loss"] == pytest.approx(expected)
    assert objective["total_loss"] == pytest.approx(expected)
    assert objective["optimizer_total_loss"] == pytest.approx(expected)
    assert objective["reported_total_loss"] == pytest.approx(expected)
    assert objective["optimizer_reported_total_consistency"]["consistent"]


class _SkipOncePublicScalerOracle:
    """Public GradScaler-surface oracle: first step skips, second applies."""

    def __init__(self) -> None:
        self._scale = 16.0
        self._step_index = 0

    def is_enabled(self) -> bool:
        return True

    def get_scale(self) -> float:
        return self._scale

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self._step_index > 0:
            optimizer.step()

    def update(self) -> None:
        if self._step_index == 0:
            self._scale /= 2.0
        self._step_index += 1


def test_optimizer_attempt_applied_skipped_accounting_uses_public_scaler_state(
    tmp_path: Path,
) -> None:
    config = _pretrain_config(tmp_path / "skip-oracle")
    config.experiment.name = "one_batch"
    config.experiment.steps = 2
    (
        _output,
        device,
        runtime,
        model,
        optimizer,
        _scheduler,
        _scaler,
        objective,
        masking,
        execution_mode,
        resolved,
        _cuda_memory_lifecycle,
    ) = _prepare(_plain_config(config))
    scaler = _SkipOncePublicScalerOracle()
    metric, gradient = _stage(
        model,
        (runtime.first_train_batch, runtime.first_train_batch),
        objective=objective,
        masking=masking,
        execution_mode=execution_mode,
        config=resolved,
        device=device,
        epoch=0,
        stage="train",
        optimizer=optimizer,
        scaler=scaler,
        collect_gradient_evidence=True,
    )
    accounting = metric["accounting"]
    assert accounting["optimizer_step_attempt_count"] == 2
    assert accounting["optimizer_step_applied_count"] == 1
    assert accounting["optimizer_step_skipped_count"] == 1
    assert accounting["optimizer_step_count"] == 1
    assert metric["amp_scaler_evidence"]["scale_decrease_skip_count"] == 1
    assert metric["amp_scaler_evidence"][
        "scale_non_decrease_applied_count"
    ] == 1
    assert len(metric["optimizer_step_evidence"]) == 2
    assert not metric["optimizer_step_evidence"][0][
        "optimizer_step_applied"
    ]
    assert metric["optimizer_step_evidence"][1][
        "optimizer_step_applied"
    ]
    assert gradient is not None
    assert gradient["acceptance"]["passed"]


def test_zero_gradient_and_zero_update_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    config = _pretrain_config(tmp_path / "zero-gradient")
    (
        _output,
        _device,
        _runtime,
        model,
        _optimizer,
        _scheduler,
        _scaler,
        objective,
        _masking,
        _execution_mode,
        _resolved,
        _cuda_memory_lifecycle,
    ) = _prepare(_plain_config(config))
    groups = _evidence_parameter_groups(model)
    snapshots = _parameter_snapshots(groups)
    for _name, parameter in groups["online_encoder"]:
        parameter.grad = torch.zeros_like(parameter)
    for _name, parameter in groups["onset_latent"]:
        parameter.grad = torch.zeros_like(parameter)
    evidence = _optimization_step_evidence(
        model,
        objective=objective,
        groups=groups,
        snapshots=snapshots,
        scaler_enabled=True,
        scale_before=16384.0,
        scale_after=16384.0,
        optimizer_step_applied=True,
    )
    assert not evidence["acceptance"]["passed"]
    assert "online_encoder_finite_nonzero_update_missing" in evidence[
        "acceptance"
    ]["failures"]
    assert "active_head_invalid:onset_latent" in evidence["acceptance"][
        "failures"
    ]


def test_real_cli_without_phase8_config_uses_exact_phase7a_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "phase7a"
    _run_cli(output)
    report = _report(output)
    assert report["phase8_started"] is False
    assert report["evidence_kind"] == "bounded_phase7a_ssl_plumbing"
    assert "phase8b_masking" not in json.loads(
        (output / "resolved_config.json").read_text(encoding="utf-8")
    )
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(config_name="ssl_training")
    assert "phase8b_masking" not in _plain_config(config)


@pytest.mark.parametrize("mode", tuple(_FAMILY))
def test_real_cli_single_level_routes_only_its_family_and_policy(
    tmp_path: Path,
    mode: str,
) -> None:
    output = tmp_path / mode
    _run_cli(
        output,
        f"+phase8b_objective={mode}",
        f"+phase8b_masking={mode}",
    )
    report = _report(output)
    assert report["model_class"] == "Phase8BMultilevelSSLModel"
    assert report["active_objective_families"] == [_FAMILY[mode]]
    assert report["resolved_mask_policies"] == [_POLICY[mode]]
    assert report["accounting"]["optimizer_step_count"] == 1
    assert report["accounting"]["optimizer_step_attempt_count"] == 1
    assert report["accounting"]["optimizer_step_applied_count"] == 1
    assert report["accounting"]["optimizer_step_skipped_count"] == 0
    assert report["accounting"]["forward_pass_count"] == 3
    assert report["accounting"]["scheduled_policy_pass_count"] == 3
    assert report["accounting"]["objective_evaluation_count"] == 3
    assert report["initial"]["retained_cuda_tensor_count"] == 0
    assert report["initial"]["retained_prediction_tensor_count"] == 0
    assert report["mechanics_acceptance"]["passed"]
    evidence = report["gradient_coverage"]["groups"]
    active = evidence[_FAMILY[mode]]
    assert active["finite_gradient_count"] == active["with_gradient_count"]
    assert active["nonzero_gradient_count"] > 0
    assert active["changed_parameter_count"] > 0
    for family in set(PHASE8B_NEW_OBJECTIVE_FAMILIES) - {_FAMILY[mode]}:
        assert evidence[family]["with_gradient_count"] == 0
        assert evidence[family]["changed_parameter_count"] == 0


def test_real_cli_equal_and_mask_only_exercise_all_hierarchy_policies(
    tmp_path: Path,
) -> None:
    equal = tmp_path / "equal"
    _run_cli(
        equal,
        "+phase8b_objective=multilevel_equal_weight",
        "+phase8b_masking=multilevel_equal_weight",
    )
    equal_report = _report(equal)
    assert equal_report["active_objective_families"] == [
        "onset_latent",
        "beat_latent",
        "hierarchy_bar_latent",
        "track_latent",
    ]
    assert equal_report["resolved_mask_policies"] == _HIERARCHY_POLICIES
    assert equal_report["accounting"]["forward_pass_count"] == 12
    assert equal_report["accounting"]["scheduled_policy_pass_count"] == 12
    assert equal_report["accounting"]["family_view_pass_count"] == 15
    assert equal_report["cross_policy_manual_oracle"][
        "family_global_total"
    ] == 6.875
    assert equal_report["cross_policy_manual_oracle"][
        "family_weight_application_counts"
    ]["hierarchy_bar_latent"] == 1
    assert equal_report["initial"]["objective"][
        "family_view_pass_counts"
    ]["hierarchy_bar_latent"] == 2
    assert equal_report["initial"]["objective"][
        "applied_family_weight_count"
    ]["hierarchy_bar_latent"] == 1
    assert equal_report["initial"]["metrics_transfer"] == {
        "packed_device_to_host_transfer_count": 0,
        "packed_host_materialization_count": 1,
        "maximum_packed_d2h_transfers_per_cpu_batch": 0,
        "retained_cuda_tensor_count": 0,
        "retained_prediction_tensor_count": 0,
    }
    _assert_family_global_total(equal_report["initial"])
    assert all(
        row["eligible_denominator"] > 0
        for row in equal_report["initial"]["objective"]["families"]
        if row["active"]
    )

    mask_only = tmp_path / "mask-only"
    _run_cli(
        mask_only,
        "+phase8b_objective=phase7a_control",
        "+phase8b_masking=phase8a_mask_only",
    )
    control = _report(mask_only)
    assert control["execution_mode"] == "phase8a_mask_only"
    assert control["model_class"] == "MaskedGraphSSLModel"
    assert control["active_objective_families"] == [
        "phase7a_note_reconstruction",
        "phase7a_bar_latent",
        "phase7a_song_latent",
    ]
    assert control["resolved_mask_policies"] == _HIERARCHY_POLICIES


def test_real_cli_weight_override_changes_fingerprint_and_fixed_total(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full"
    half = tmp_path / "half"
    common = (
        "+phase8b_objective=onset_only",
        "+phase8b_masking=onset_only",
    )
    _run_cli(full, *common)
    _run_cli(half, *common, "phase8b_objective.onset_latent=0.5")
    full_report = _report(full)
    half_report = _report(half)
    assert full_report["objective_config_fingerprint"] != (
        half_report["objective_config_fingerprint"]
    )
    assert half_report["initial"]["total_ssl_loss"] == pytest.approx(
        float(full_report["initial"]["total_ssl_loss"]) * 0.5
    )
    assert full_report["masking_config_fingerprint"] == (
        half_report["masking_config_fingerprint"]
    )


def test_incompatible_cli_pair_fails_before_output_or_optimizer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "incompatible"
    result = _run_cli(
        output,
        "+phase8b_objective=onset_only",
        "+phase8b_masking=beat_only",
        check=False,
    )
    assert result.returncode != 0
    assert "phase8b.engine.objective_masking_mode_incompatible" in (
        result.stderr
    )
    assert not output.exists()


def test_one_batch_loss_decreases_and_artifacts_bind_every_phase8b_surface(
    tmp_path: Path,
) -> None:
    output = tmp_path / "overfit"
    _run_cli(
        output,
        "+phase8b_objective=onset_only",
        "+phase8b_masking=onset_only",
        "experiment.steps=2",
    )
    report = _report(output)
    assert report["loss_decreased"]
    assert report["checkpoint_reload"] == {
        "bit_exact": True,
        "next_epoch": 0,
    }
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase8b_bindings"]["model_class"] == (
        "Phase8BMultilevelSSLModel"
    )
    checkpoint = torch.load(
        output / "one_batch.pt", map_location="cpu", weights_only=True
    )
    binding = checkpoint["metadata"]["phase8b_binding"]
    assert binding["objective_registry_fingerprint"] == (
        report["objective_registry_fingerprint"]
    )
    assert binding["objective_config_fingerprint"] == (
        report["objective_config_fingerprint"]
    )
    assert binding["masking_config_fingerprint"] == (
        report["masking_config_fingerprint"]
    )
    assert binding["mask_policy_mixture_fingerprint"] == (
        report["mask_policy_mixture_fingerprint"]
    )
    assert binding["scheduled_view_aggregation"] == (
        report["scheduled_view_aggregation"]
    )


def test_validation_and_best_checkpoint_use_family_global_formula(
    tmp_path: Path,
) -> None:
    output = tmp_path / "equal-validation"
    config = _pretrain_config(output, "multilevel_equal_weight")
    config.experiment.epochs = 1
    config.experiment.overwrite_output = True
    report = run_ssl_training(config)
    row = json.loads(
        (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    _assert_family_global_total(report["initial_validation"])
    _assert_family_global_total(row["validation"])
    assert row["validation"]["objective"]["family_view_pass_counts"][
        "hierarchy_bar_latent"
    ] == 2 * row["validation"]["batch_count"]
    assert report["best_checkpoint_selection"] == (
        "minimum_family_global_validation_total_ssl_loss"
    )
    assert report["best_validation_loss"] == pytest.approx(
        row["validation"]["total_ssl_loss"]
    )
    best = torch.load(
        output / "best.pt", map_location="cpu", weights_only=True
    )
    assert best["best_validation_loss"] == pytest.approx(
        row["validation"]["total_ssl_loss"]
    )


def test_two_epoch_stop_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resume"
    uninterrupted_config = _pretrain_config(output)
    uninterrupted_config.experiment.overwrite_output = True
    uninterrupted = run_ssl_training(uninterrupted_config)
    uninterrupted_checkpoint = copy.deepcopy(
        torch.load(
            output / "last.pt", map_location="cpu", weights_only=True
        )
    )

    interrupted_config = _pretrain_config(output)
    interrupted_config.experiment.overwrite_output = True
    first = run_ssl_training(interrupted_config, stop_after_epoch=1)
    assert first["completed_epochs"] == 1
    interrupted_config.experiment.overwrite_output = False
    interrupted_config.experiment.resume_from = str(output / "last.pt")
    resumed = run_ssl_training(interrupted_config)
    resumed_checkpoint = torch.load(
        output / "last.pt", map_location="cpu", weights_only=True
    )
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "next_epoch",
        "best_validation_loss",
        "epoch_journal",
    ):
        _assert_tree_equal(
            uninterrupted_checkpoint[key], resumed_checkpoint[key]
        )
    assert uninterrupted["accounting"] == resumed["accounting"]
    assert resumed["start_epoch"] == 1
    assert resumed["completed_epochs"] == 2


def test_changed_policy_resume_rejects_before_rng_or_artifact_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "policy-mismatch"
    config = _pretrain_config(output)
    config.experiment.overwrite_output = True
    run_ssl_training(config, stop_after_epoch=1)
    before_artifact = sha256((output / "last.pt").read_bytes()).hexdigest()
    random.seed(707)
    torch.manual_seed(707)
    before_rng = capture_rng_state()
    config.experiment.overwrite_output = False
    config.experiment.resume_from = str(output / "last.pt")
    config.phase8b_masking.max_span_bars = 3
    with pytest.raises(
        Phase8BEngineError,
        match="run_manifest_checkpoint_binding_mismatch",
    ):
        run_ssl_training(config)
    _assert_tree_equal(before_rng, capture_rng_state())
    assert sha256((output / "last.pt").read_bytes()).hexdigest() == (
        before_artifact
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "mode",
    (
        "onset_only",
        "beat_only",
        "bar_only",
        "track_only",
        "multilevel_equal_weight",
        "phase8a_mask_only",
    ),
)
def test_official_engine_cuda_amp_smoke(tmp_path: Path, mode: str) -> None:
    output = tmp_path / f"cuda-{mode}"
    command = _common_cli(output)
    command[command.index("device=cpu")] = "device=cuda"
    command[command.index("experiment.steps=1")] = "experiment.steps=8"
    overrides = (
        (
            "+phase8b_objective=phase7a_control",
            "+phase8b_masking=phase8a_mask_only",
        )
        if mode == "phase8a_mask_only"
        else (
            f"+phase8b_objective={mode}",
            f"+phase8b_masking={mode}",
        )
    )
    subprocess.run(
        [
            *command,
            *overrides,
            "device.amp=true",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=360,
    )
    report = _report(output)
    assert report["device"]["resolved_device"].startswith("cuda")
    assert report["amp_enabled"]
    assert report["scaler_enabled"]
    assert report["mechanics_acceptance"]["passed"]
    assert report["accounting"]["optimizer_step_attempt_count"] == 8
    assert report["accounting"]["optimizer_step_applied_count"] > 0
    assert report["accounting"]["optimizer_step_attempt_count"] == (
        report["accounting"]["optimizer_step_applied_count"]
        + report["accounting"]["optimizer_step_skipped_count"]
    )
    assert report["model_state_fingerprints"]["changed"]
    assert report["initial"]["input_batch_fingerprints"] == report["final"][
        "input_batch_fingerprints"
    ]
    assert report["loss_decreased"]
    assert math.isfinite(report["final"]["total_ssl_loss"])
    assert report["cuda_peak_memory"]["peak_allocated_bytes"] > 0
    assert report["cuda_peak_memory"]["peak_reserved_bytes"] > 0
    gradient = report["gradient_coverage"]
    assert gradient["acceptance"]["passed"]
    encoder = gradient["groups"]["online_encoder"]
    assert encoder["finite_gradient_count"] == encoder["with_gradient_count"]
    assert encoder["nonzero_gradient_count"] > 0
    assert encoder["changed_parameter_count"] > 0
    active_new = (
        set(PHASE8B_NEW_OBJECTIVE_FAMILIES)
        if mode == "multilevel_equal_weight"
        else ({_FAMILY[mode]} if mode in _FAMILY else set())
    )
    for family in PHASE8B_NEW_OBJECTIVE_FAMILIES:
        row = gradient["groups"][family]
        if family in active_new:
            assert row["finite_gradient_count"] == row["with_gradient_count"]
            assert row["nonzero_gradient_count"] > 0
            assert row["changed_parameter_count"] > 0
        else:
            assert row["with_gradient_count"] == 0
            assert row["changed_parameter_count"] == 0
    if mode == "phase8a_mask_only":
        for group in (
            "online_local_encoder",
            "hierarchy_pooling",
            "transformer",
            "fusion",
            "decoder",
            "phase7a_bar_projector",
            "phase7a_bar_predictor",
            "phase7a_song_projector",
            "phase7a_song_predictor",
        ):
            row = gradient["groups"][group]
            assert row["finite_gradient_count"] == row["with_gradient_count"]
            assert row["nonzero_gradient_count"] > 0
            assert row["changed_parameter_count"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ("onset_only", "multilevel_equal_weight"))
def test_official_engine_cuda_fp32_amp_bounded_parity(
    tmp_path: Path, mode: str
) -> None:
    reports = {}
    for precision, amp in (("fp32", False), ("amp", True)):
        output = tmp_path / f"parity-{mode}-{precision}"
        command = _common_cli(output)
        command[command.index("device=cpu")] = "device=cuda"
        command[command.index("experiment.steps=1")] = "experiment.steps=8"
        subprocess.run(
            [
                *command,
                f"+phase8b_objective={mode}",
                f"+phase8b_masking={mode}",
                f"device.amp={'true' if amp else 'false'}",
            ],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=360,
        )
        reports[precision] = _report(output)
    fp32, amp = reports["fp32"], reports["amp"]
    assert fp32["input_fixture_fingerprint"] == amp["input_fixture_fingerprint"]
    assert fp32["model_state_fingerprints"]["initial"] == (
        amp["model_state_fingerprints"]["initial"]
    )
    assert fp32["resolved_mask_policies"] == amp["resolved_mask_policies"]
    assert fp32["initial"]["prepared_binding_fingerprints"] == (
        amp["initial"]["prepared_binding_fingerprints"]
    )
    assert fp32["initial"]["prepared_objective_binding_fingerprints"] == (
        amp["initial"]["prepared_objective_binding_fingerprints"]
    )
    assert fp32["initial"]["objective"]["family_denominators"] == (
        amp["initial"]["objective"]["family_denominators"]
    )
    assert fp32["initial"]["objective"]["family_view_pass_counts"] == (
        amp["initial"]["objective"]["family_view_pass_counts"]
    )
    assert fp32["final"]["objective"]["family_denominators"] == (
        amp["final"]["objective"]["family_denominators"]
    )
    assert fp32["final"]["objective"]["family_view_pass_counts"] == (
        amp["final"]["objective"]["family_view_pass_counts"]
    )
    assert fp32["initial"]["total_ssl_loss"] == pytest.approx(
        amp["initial"]["total_ssl_loss"], rel=0.02, abs=0.02
    )
    assert fp32["mechanics_acceptance"]["passed"]
    assert amp["mechanics_acceptance"]["passed"]
    assert math.isfinite(fp32["final"]["total_ssl_loss"])
    assert math.isfinite(amp["final"]["total_ssl_loss"])
    assert fp32["final"]["total_ssl_loss"] == pytest.approx(
        amp["final"]["total_ssl_loss"], rel=0.02, abs=0.02
    )
