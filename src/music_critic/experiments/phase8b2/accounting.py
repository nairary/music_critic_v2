"""Exact comparison-cell compute accounting and budget validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

from music_critic.experiments.phase8b2.contracts import (
    PHASE8B2_COMPUTE_ACCOUNTING_VERSION,
    Phase8B2ContractError,
    fingerprint,
)


@dataclass(frozen=True, slots=True)
class ComputeAccounting:
    logical_updates: int
    policy_views: int
    encoder_forwards: int
    raw_samples_seen: int
    nodes_seen: int
    edges_seen: int
    eligible_objective_rows: int
    optimizer_updates_applied: int
    optimizer_updates_skipped: int
    wall_seconds: float
    peak_allocated_vram_bytes: int | None
    peak_reserved_vram_bytes: int | None

    def __post_init__(self) -> None:
        integers = (
            self.logical_updates,
            self.policy_views,
            self.encoder_forwards,
            self.raw_samples_seen,
            self.nodes_seen,
            self.edges_seen,
            self.eligible_objective_rows,
            self.optimizer_updates_applied,
            self.optimizer_updates_skipped,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in integers
        ) or (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds < 0
        ):
            raise Phase8B2ContractError(
                "phase8b2.accounting.value_invalid"
            )
        for value in (
            self.peak_allocated_vram_bytes,
            self.peak_reserved_vram_bytes,
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise Phase8B2ContractError(
                    "phase8b2.accounting.vram_invalid"
                )
        if (self.peak_allocated_vram_bytes is None) != (
            self.peak_reserved_vram_bytes is None
        ):
            raise Phase8B2ContractError(
                "phase8b2.accounting.partial_vram_evidence"
            )
        if self.optimizer_updates_applied + self.optimizer_updates_skipped != (
            self.logical_updates
        ):
            raise Phase8B2ContractError(
                "phase8b2.accounting.optimizer_update_mismatch"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "compute_accounting_version": (
                PHASE8B2_COMPUTE_ACCOUNTING_VERSION
            ),
            **asdict(self),
            "cuda_vram_evidence_available": (
                self.peak_allocated_vram_bytes is not None
            ),
        }


def compute_accounting_from_ssl_report(
    report: Mapping[str, object],
) -> ComputeAccounting:
    """Translate one official Phase 8B.2A report without inferred counters."""

    if report.get("phase8b2_started") is not True:
        raise Phase8B2ContractError(
            "phase8b2.accounting.comparison_report_required"
        )
    if report.get("run_scope") != "epoch_pretraining":
        raise Phase8B2ContractError(
            "phase8b2.accounting.epoch_report_required"
        )
    accounting = report.get("accounting")
    schedule = report.get("phase8b2_schedule")
    cuda = report.get("cuda_peak_memory")
    duration = report.get("duration_seconds")
    if (
        not isinstance(accounting, Mapping)
        or not isinstance(schedule, Mapping)
        or not isinstance(cuda, Mapping)
    ):
        raise Phase8B2ContractError(
            "phase8b2.accounting.official_report_incomplete"
        )
    try:
        policy_views = int(accounting["scheduled_policy_pass_count"])
        result = ComputeAccounting(
            logical_updates=int(accounting["cpu_batch_count"]),
            policy_views=policy_views,
            encoder_forwards=int(accounting["encoder_forward_count"]),
            raw_samples_seen=int(accounting["sample_count"]),
            nodes_seen=int(accounting["node_count"]),
            edges_seen=int(accounting["edge_count"]),
            eligible_objective_rows=int(
                accounting["eligible_prediction_row_count"]
            ),
            optimizer_updates_applied=int(
                accounting["optimizer_step_applied_count"]
            ),
            optimizer_updates_skipped=int(
                accounting["optimizer_step_skipped_count"]
            ),
            wall_seconds=float(duration),
            peak_allocated_vram_bytes=(
                int(cuda["peak_allocated_bytes"])
                if cuda.get("available") is True
                else None
            ),
            peak_reserved_vram_bytes=(
                int(cuda["peak_reserved_bytes"])
                if cuda.get("available") is True
                else None
            ),
        )
        expected_forwards = (
            policy_views
            * int(schedule["encoder_forwards_per_policy_view"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase8B2ContractError(
            "phase8b2.accounting.official_report_incomplete"
        ) from exc
    if result.encoder_forwards != expected_forwards:
        raise Phase8B2ContractError(
            "phase8b2.accounting.encoder_forward_evidence_mismatch"
        )
    return result


def validate_compute_matrix(
    cells: Iterable[tuple[str, ComputeAccounting]],
    *,
    comparison_mode: str,
) -> dict[str, object]:
    rows = tuple(cells)
    if not rows or len({variant for variant, _ in rows}) != len(rows):
        raise Phase8B2ContractError(
            "phase8b2.accounting.matrix_invalid"
        )
    required_equal = (
        "logical_updates",
        "raw_samples_seen",
        "optimizer_updates_applied",
        "optimizer_updates_skipped",
    )
    if comparison_mode == "encoder_forward_matched":
        required_equal += ("encoder_forwards",)
    elif comparison_mode != "natural_schedule":
        raise Phase8B2ContractError(
            "phase8b2.accounting.comparison_mode_invalid"
        )
    mismatches = [
        field
        for field in required_equal
        if len({getattr(row, field) for _, row in rows}) != 1
    ]
    if mismatches:
        raise Phase8B2ContractError(
            "phase8b2.accounting.budget_mismatch:"
            + ",".join(mismatches)
        )
    payload = {
        "compute_accounting_version": PHASE8B2_COMPUTE_ACCOUNTING_VERSION,
        "comparison_mode": comparison_mode,
        "matched_fields": list(required_equal),
        "cells": {
            variant: accounting.to_dict()
            for variant, accounting in sorted(rows)
        },
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


__all__ = [
    "ComputeAccounting",
    "compute_accounting_from_ssl_report",
    "validate_compute_matrix",
]
