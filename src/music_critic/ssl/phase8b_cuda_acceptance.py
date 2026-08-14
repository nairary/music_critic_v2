"""Independent RTX acceptance for real Phase 8B.1 optimizer updates.

This module orchestrates the official CLI only.  It does not provide a second
training implementation and makes no CUDA-success claim unless every FP32 and
AMP subprocess passes the fail-closed mechanics contract.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Callable

import torch

from music_critic.ssl.multilevel import PHASE8B_NEW_OBJECTIVE_FAMILIES


PHASE8B_CUDA_TRAINING_ACCEPTANCE_CONTRACT_VERSION = "1.0.0"
PHASE8B_CUDA_PARITY_RELATIVE_TOLERANCE = 0.02
PHASE8B_CUDA_PARITY_ABSOLUTE_TOLERANCE = 0.02

_MODE_CONFIGS = {
    "onset_only": ("onset_only", "onset_only", ("onset_latent",)),
    "beat_only": ("beat_only", "beat_only", ("beat_latent",)),
    "bar_only": ("bar_only", "bar_only", ("hierarchy_bar_latent",)),
    "track_only": ("track_only", "track_only", ("track_latent",)),
    "multilevel_equal_weight": (
        "multilevel_equal_weight",
        "multilevel_equal_weight",
        PHASE8B_NEW_OBJECTIVE_FAMILIES,
    ),
    "phase8a_mask_only": ("phase7a_control", "phase8a_mask_only", ()),
}


class Phase8BCudaAcceptanceError(ValueError):
    """Fail-closed CUDA acceptance or artifact error."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _subprocess_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _require_exact_head(*, expected_head: str, actual_head: object) -> None:
    if (
        len(expected_head) != 40
        or any(
            character not in "0123456789abcdef"
            for character in expected_head
        )
    ):
        raise Phase8BCudaAcceptanceError(
            "phase8b.cuda.expected_head_invalid"
        )
    if actual_head != expected_head:
        raise Phase8BCudaAcceptanceError(
            "phase8b.cuda.exact_head_mismatch:"
            f"expected={expected_head}:actual={actual_head}"
        )


def _official_command(
    output: Path,
    *,
    mode: str,
    amp: bool,
    seed: int,
    steps: int,
) -> list[str]:
    if mode not in _MODE_CONFIGS:
        raise Phase8BCudaAcceptanceError("phase8b.cuda.mode_unknown")
    objective_mode, masking_mode, _active = _MODE_CONFIGS[mode]
    return [
        sys.executable,
        "-m",
        "music_critic.ssl.run",
        f"+phase8b_objective={objective_mode}",
        f"+phase8b_masking={masking_mode}",
        "experiment=one_batch",
        f"experiment.steps={steps}",
        "model=hierarchical",
        "model.hidden_dim=8",
        "model.local_gnn_layers=1",
        "model.transformer_layers=1",
        "model.attention_heads=2",
        "model.ffn_multiplier=2",
        "model.dropout=0",
        "data=bounded",
        "data.batch_size=3",
        "device=cuda",
        f"device.amp={'true' if amp else 'false'}",
        "optimizer.learning_rate=0.02",
        "optimizer.weight_decay=0",
        "ssl.mask_rate=0.5",
        "ssl.decoder_views=1",
        "ssl.decoder_remask_prob=0",
        "ssl.projector_hidden_dim=8",
        "ssl.decoder_hidden_dim=8",
        f"seed={seed}",
        f"output_dir={output}",
        "experiment.overwrite_output=true",
    ]


def _validate_training_report(
    report: dict[str, object], *, mode: str, expected_steps: int
) -> list[str]:
    failures: list[str] = []
    mechanics = report.get("mechanics_acceptance")
    if not isinstance(mechanics, dict) or not mechanics.get("passed", False):
        failures.append("mechanics_acceptance_failed")
    accounting = report.get("accounting")
    if not isinstance(accounting, dict):
        return [*failures, "optimizer_accounting_missing"]
    attempts = accounting.get("optimizer_step_attempt_count")
    applied = accounting.get("optimizer_step_applied_count")
    skipped = accounting.get("optimizer_step_skipped_count")
    if attempts != expected_steps:
        failures.append("optimizer_attempt_count_mismatch")
    if not isinstance(applied, int) or applied <= 0:
        failures.append("no_optimizer_step_applied")
    if (
        not isinstance(attempts, int)
        or not isinstance(skipped, int)
        or attempts != applied + skipped
    ):
        failures.append("optimizer_step_accounting_inconsistent")
    if accounting.get("optimizer_step_count") != applied:
        failures.append("optimizer_step_alias_not_applied_count")
    if not report.get("loss_decreased", False):
        failures.append("bounded_loss_did_not_decrease")
    initial = report.get("initial")
    final = report.get("final")
    if not isinstance(initial, dict) or not isinstance(final, dict):
        return [*failures, "initial_or_final_stage_missing"]
    for name, stage in (("initial", initial), ("final", final)):
        loss = stage.get("total_ssl_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            failures.append(f"{name}_loss_nonfinite")
    if initial.get("input_batch_fingerprints") != final.get(
        "input_batch_fingerprints"
    ):
        failures.append("initial_final_fixture_mismatch")
    model_fingerprints = report.get("model_state_fingerprints")
    if not isinstance(model_fingerprints, dict):
        model_fingerprints = {}
    if not model_fingerprints.get("changed", False):
        failures.append("model_parameters_unchanged")
    coverage = report.get("optimizer_parameter_coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    if not coverage.get("all_trainable_parameters_present_exactly_once", False):
        failures.append("optimizer_parameter_coverage_invalid")
    gradient = report.get("gradient_coverage")
    if not isinstance(gradient, dict) or not gradient.get("acceptance", {}).get(
        "passed", False
    ):
        failures.append("gradient_acceptance_failed")
        return failures
    groups = gradient.get("groups")
    if not isinstance(groups, dict):
        failures.append("gradient_groups_missing")
        return failures
    encoder = groups.get("online_encoder", {})
    if (
        encoder.get("finite_gradient_count")
        != encoder.get("with_gradient_count")
        or not isinstance(encoder.get("nonzero_gradient_count"), int)
        or encoder["nonzero_gradient_count"] <= 0
        or not isinstance(encoder.get("changed_parameter_count"), int)
        or encoder["changed_parameter_count"] <= 0
    ):
        failures.append("online_encoder_gradient_or_update_invalid")
    active = set(_MODE_CONFIGS[mode][2])
    for family in PHASE8B_NEW_OBJECTIVE_FAMILIES:
        row = groups.get(family, {})
        if family in active:
            if (
                row.get("finite_gradient_count")
                != row.get("with_gradient_count")
                or not isinstance(row.get("nonzero_gradient_count"), int)
                or row["nonzero_gradient_count"] <= 0
                or not isinstance(row.get("changed_parameter_count"), int)
                or row["changed_parameter_count"] <= 0
            ):
                failures.append(f"active_head_invalid:{family}")
        elif row.get("with_gradient_count") != 0 or row.get(
            "changed_parameter_count"
        ) != 0:
            failures.append(f"inactive_head_changed:{family}")
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
            row = groups.get(group, {})
            if (
                row.get("finite_gradient_count")
                != row.get("with_gradient_count")
                or not isinstance(row.get("nonzero_gradient_count"), int)
                or row["nonzero_gradient_count"] <= 0
                or not isinstance(row.get("changed_parameter_count"), int)
                or row["changed_parameter_count"] <= 0
            ):
                failures.append(f"mask_only_path_invalid:{group}")
    peak = report.get("cuda_peak_memory")
    if not isinstance(peak, dict):
        peak = {}
    if (
        not peak.get("available", False)
        or not isinstance(peak.get("peak_allocated_bytes"), int)
        or peak["peak_allocated_bytes"] <= 0
        or not isinstance(peak.get("peak_reserved_bytes"), int)
        or peak["peak_reserved_bytes"] <= 0
    ):
        failures.append("cuda_peak_memory_missing")
    return failures


def _compare_precision_reports(
    fp32: dict[str, object],
    amp: dict[str, object],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, object]:
    checks = {
        "input_fixture_fingerprint_equal": (
            fp32.get("input_fixture_fingerprint")
            == amp.get("input_fixture_fingerprint")
        ),
        "initial_model_state_fingerprint_equal": (
            fp32.get("model_state_fingerprints", {}).get("initial")
            == amp.get("model_state_fingerprints", {}).get("initial")
        ),
        "resolved_mask_policies_equal": (
            fp32.get("resolved_mask_policies")
            == amp.get("resolved_mask_policies")
        ),
        "initial_binding_fingerprints_equal": (
            fp32.get("initial", {}).get("prepared_binding_fingerprints")
            == amp.get("initial", {}).get("prepared_binding_fingerprints")
        ),
        "initial_objective_binding_fingerprints_equal": (
            fp32.get("initial", {}).get(
                "prepared_objective_binding_fingerprints"
            )
            == amp.get("initial", {}).get(
                "prepared_objective_binding_fingerprints"
            )
        ),
        "final_binding_fingerprints_equal": (
            fp32.get("final", {}).get("prepared_binding_fingerprints")
            == amp.get("final", {}).get("prepared_binding_fingerprints")
        ),
        "final_objective_binding_fingerprints_equal": (
            fp32.get("final", {}).get(
                "prepared_objective_binding_fingerprints"
            )
            == amp.get("final", {}).get(
                "prepared_objective_binding_fingerprints"
            )
        ),
        "family_denominators_equal": (
            fp32.get("initial", {}).get("objective", {}).get(
                "family_denominators"
            )
            == amp.get("initial", {}).get("objective", {}).get(
                "family_denominators"
            )
        ),
        "family_view_pass_counts_equal": (
            fp32.get("initial", {}).get("objective", {}).get(
                "family_view_pass_counts"
            )
            == amp.get("initial", {}).get("objective", {}).get(
                "family_view_pass_counts"
            )
        ),
        "final_family_denominators_equal": (
            fp32.get("final", {}).get("objective", {}).get(
                "family_denominators"
            )
            == amp.get("final", {}).get("objective", {}).get(
                "family_denominators"
            )
        ),
        "final_family_view_pass_counts_equal": (
            fp32.get("final", {}).get("objective", {}).get(
                "family_view_pass_counts"
            )
            == amp.get("final", {}).get("objective", {}).get(
                "family_view_pass_counts"
            )
        ),
    }
    fp32_loss = fp32.get("initial", {}).get("total_ssl_loss")
    amp_loss = amp.get("initial", {}).get("total_ssl_loss")
    initial_loss_close = (
        isinstance(fp32_loss, (int, float))
        and isinstance(amp_loss, (int, float))
        and math.isclose(
            float(fp32_loss),
            float(amp_loss),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    )
    checks["initial_loss_within_documented_tolerance"] = initial_loss_close
    fp32_final_loss = fp32.get("final", {}).get("total_ssl_loss")
    amp_final_loss = amp.get("final", {}).get("total_ssl_loss")
    checks["final_loss_within_documented_tolerance"] = (
        isinstance(fp32_final_loss, (int, float))
        and isinstance(amp_final_loss, (int, float))
        and math.isclose(
            float(fp32_final_loss),
            float(amp_final_loss),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    )
    return {
        "bit_exact_required": False,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _environment_report(repo_root: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise Phase8BCudaAcceptanceError("phase8b.cuda.source_tree_dirty")
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "exact_head": head,
        "source_tree_clean": True,
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver_versions": driver,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def run_phase8b_cuda_training_acceptance(
    output_dir: Path,
    *,
    expected_head: str,
    seed: int = 42,
    steps: int = 12,
    relative_tolerance: float = PHASE8B_CUDA_PARITY_RELATIVE_TOLERANCE,
    absolute_tolerance: float = PHASE8B_CUDA_PARITY_ABSOLUTE_TOLERANCE,
    emit: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run all official FP32/AMP modes and archive fail-closed evidence."""

    if not torch.cuda.is_available():
        raise Phase8BCudaAcceptanceError("phase8b.cuda.cuda_unavailable")
    if steps < 8:
        raise Phase8BCudaAcceptanceError("phase8b.cuda.steps_below_eight")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    logs = output_dir / "logs"
    runs = output_dir / "runs"
    logs.mkdir()
    runs.mkdir()
    log_path = output_dir / "acceptance.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        emit(message)

    repo_root = Path(__file__).resolve().parents[3]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    environment = _environment_report(repo_root)
    _require_exact_head(
        expected_head=expected_head,
        actual_head=environment["exact_head"],
    )
    log(f"exact_head={environment['exact_head']}")
    reports: dict[str, dict[str, object]] = {}
    run_rows = []
    failures: list[str] = []
    for mode in _MODE_CONFIGS:
        for precision, amp in (("fp32", False), ("amp_float16", True)):
            run_output = runs / f"{mode}-{precision}"
            command = _official_command(
                run_output,
                mode=mode,
                amp=amp,
                seed=seed,
                steps=steps,
            )
            log(f"run={mode}/{precision} command={json.dumps(command)}")
            command_log = logs / f"{mode}-{precision}.log"
            return_code: int | None = None
            stdout = ""
            stderr = ""
            launch_failure: str | None = None
            try:
                process = subprocess.run(
                    command,
                    cwd=repo_root,
                    env={
                        **os.environ,
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                return_code = process.returncode
                stdout = process.stdout
                stderr = process.stderr
            except subprocess.TimeoutExpired as exc:
                stdout = _subprocess_output_text(exc.stdout)
                stderr = _subprocess_output_text(exc.stderr)
                launch_failure = "timeout_seconds:900"
            except OSError as exc:
                launch_failure = f"launch_error:{type(exc).__name__}:{exc}"
            command_log.write_text(
                "COMMAND\n"
                + json.dumps(command, ensure_ascii=False)
                + "\nSTDOUT\n"
                + stdout
                + "\nSTDERR\n"
                + stderr
                + "\nLAUNCH_FAILURE\n"
                + (launch_failure or "none"),
                encoding="utf-8",
            )
            report_path = run_output / "one_batch_report.json"
            run_failures = []
            report = None
            if launch_failure is not None:
                run_failures.append(launch_failure)
            elif return_code != 0:
                run_failures.append(f"exit_code:{return_code}")
            if not report_path.is_file():
                run_failures.append("report_missing")
            else:
                try:
                    loaded_report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    run_failures.append(
                        f"report_unreadable:{type(exc).__name__}"
                    )
                else:
                    if not isinstance(loaded_report, dict):
                        run_failures.append("report_root_not_object")
                    else:
                        report = loaded_report
                        run_failures.extend(
                            _validate_training_report(
                                report, mode=mode, expected_steps=steps
                            )
                        )
                        reports[f"{mode}:{precision}"] = report
            row = {
                "mode": mode,
                "precision": precision,
                "amp": amp,
                "exit_code": return_code,
                "command": command,
                "command_log": str(command_log),
                "command_log_sha256": _sha256_file(command_log),
                "report": str(report_path),
                "report_sha256": (
                    _sha256_file(report_path) if report_path.is_file() else None
                ),
                "failures": run_failures,
                "passed": not run_failures,
            }
            run_rows.append(row)
            failures.extend(
                f"{mode}/{precision}:{failure}"
                for failure in run_failures
            )
            log(
                f"result={mode}/{precision} exit={return_code} "
                f"passed={not run_failures}"
            )
    parity = {}
    for mode in _MODE_CONFIGS:
        fp32 = reports.get(f"{mode}:fp32")
        amp = reports.get(f"{mode}:amp_float16")
        if fp32 is None or amp is None:
            parity[mode] = {"passed": False, "reason": "report_missing"}
        else:
            parity[mode] = _compare_precision_reports(
                fp32,
                amp,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        if not parity[mode]["passed"]:
            failures.append(f"{mode}:fp32_amp_parity_failed")
    report = {
        "contract_version": PHASE8B_CUDA_TRAINING_ACCEPTANCE_CONTRACT_VERSION,
        "evidence_kind": "independent_phase8b1_cuda_amp_real_update_acceptance",
        "environment": environment,
        "seed": seed,
        "steps": steps,
        "modes": list(_MODE_CONFIGS),
        "precisions": ["fp32", "amp_float16"],
        "runs": run_rows,
        "fp32_amp_parity": parity,
        "all_runs_passed": all(row["passed"] for row in run_rows),
        "all_parity_checks_passed": all(
            row["passed"] for row in parity.values()
        ),
        "failures": failures,
        "passed": not failures,
        "claim_boundary": (
            "bounded_cuda_training_mechanics_only_not_effectiveness"
        ),
    }
    _write_json(output_dir / "acceptance_report.json", report)
    log(f"acceptance_passed={report['passed']}")
    archive = output_dir.with_name(output_dir.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output_dir, arcname=output_dir.name)
    archive_sha = _sha256_file(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{archive_sha}  {archive.name}\n", encoding="utf-8"
    )
    log(f"archive={archive} sha256={archive_sha}")
    return report


__all__ = [
    "PHASE8B_CUDA_PARITY_ABSOLUTE_TOLERANCE",
    "PHASE8B_CUDA_PARITY_RELATIVE_TOLERANCE",
    "PHASE8B_CUDA_TRAINING_ACCEPTANCE_CONTRACT_VERSION",
    "Phase8BCudaAcceptanceError",
    "run_phase8b_cuda_training_acceptance",
]
