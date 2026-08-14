"""Bounded scalar diagnostics for Phase 8A exact replay differences."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from music_critic.ssl.model import Phase8AHierarchySSLForwardOutput


PHASE8A_OUTPUT_DIFFERENCE_DIAGNOSTIC_CONTRACT_VERSION = "1.0.0"
PHASE8A_OUTPUT_DIFFERENCE_MAX_RETAINED = 64

DifferenceGroup = Literal[
    "embeddings",
    "predictions",
    "targets",
    "loss_tensors",
    "other",
]


def _json_float(value: Tensor) -> float | None:
    result = float(value.detach().cpu().item())
    return result if math.isfinite(result) else None


def _bounded_summary(value: object) -> str:
    if isinstance(value, Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)!r},dtype={value.dtype},"
            f"device={value.device})"
        )
    if isinstance(value, Mapping):
        return f"{type(value).__name__}(keys={tuple(value)!r})"[:240]
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}(len={len(value)})"
    rendered = repr(value)
    return rendered if len(rendered) <= 240 else rendered[:237] + "..."


def _difference_group(path: str) -> DifferenceGroup:
    if ".note_loss" in path or ".objective" in path or ".loss" in path:
        return "loss_tensors"
    if ".online_encoder" in path:
        return "embeddings"
    if (
        ".decoder_predictions" in path
        or ".bar_latent.prediction" in path
        or ".song_latent.prediction" in path
    ):
        return "predictions"
    if (
        ".targets" in path
        or ".bar_latent.target" in path
        or ".song_latent.target" in path
    ):
        return "targets"
    return "other"


def _ordered_float_bits(value: Tensor) -> Tensor:
    if value.dtype == torch.float16:
        raw = value.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        return torch.where(
            (raw & 0x8000) != 0,
            0xFFFF - raw,
            raw + 0x8000,
        )
    if value.dtype == torch.float32:
        raw = value.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        return torch.where(
            (raw & 0x80000000) != 0,
            0xFFFFFFFF - raw,
            raw + 0x80000000,
        )
    raise TypeError("ULP evidence supports only FP16 and FP32")


@dataclass(frozen=True, slots=True)
class Phase8AOutputDifference:
    path: str
    group: DifferenceGroup
    kind: Literal["tensor", "structure", "value"]
    reason: str
    left_summary: str
    right_summary: str
    left_shape: tuple[int, ...] | None = None
    right_shape: tuple[int, ...] | None = None
    left_dtype: str | None = None
    right_dtype: str | None = None
    left_device: str | None = None
    right_device: str | None = None
    total_element_count: int | None = None
    different_element_count: int | None = None
    max_absolute_difference: float | None = None
    max_relative_difference: float | None = None
    max_ulp_difference: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name) for field in fields(self)
        }


@dataclass(frozen=True, slots=True)
class Phase8AOutputDifferenceDiagnostic:
    contract_version: str
    bit_exact: bool
    first_difference_path: str | None
    total_difference_count: int
    retained_difference_count: int
    retained_limit: int
    truncated: bool
    difference_count_by_group: tuple[tuple[str, int], ...]
    retained_path_by_group: tuple[tuple[str, tuple[str, ...]], ...]
    differences: tuple[Phase8AOutputDifference, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "bit_exact": self.bit_exact,
            "first_difference_path": self.first_difference_path,
            "total_difference_count": self.total_difference_count,
            "retained_difference_count": self.retained_difference_count,
            "retained_limit": self.retained_limit,
            "truncated": self.truncated,
            "difference_count_by_group": dict(
                self.difference_count_by_group
            ),
            "retained_path_by_group": {
                key: list(paths)
                for key, paths in self.retained_path_by_group
            },
            "differences": [value.to_dict() for value in self.differences],
        }


def _tensor_difference(
    left: Tensor,
    right: Tensor,
    *,
    path: str,
) -> Phase8AOutputDifference | None:
    common = {
        "path": path,
        "group": _difference_group(path),
        "kind": "tensor",
        "left_summary": _bounded_summary(left),
        "right_summary": _bounded_summary(right),
        "left_shape": tuple(left.shape),
        "right_shape": tuple(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "left_device": str(left.device),
        "right_device": str(right.device),
        "total_element_count": left.numel(),
    }
    if left.shape != right.shape:
        return Phase8AOutputDifference(
            **common,
            reason="shape_mismatch",
        )
    if left.dtype != right.dtype:
        return Phase8AOutputDifference(
            **common,
            reason="dtype_mismatch",
        )
    if left.device != right.device:
        return Phase8AOutputDifference(
            **common,
            reason="device_mismatch",
        )
    if torch.equal(left, right):
        return None

    different = left != right
    different_count = int(
        torch.count_nonzero(different).detach().cpu().item()
    )
    max_absolute: float | None = None
    max_relative: float | None = None
    max_ulp: int | None = None
    if left.numel() and (left.dtype == torch.bool or not left.is_complex()):
        if left.dtype == torch.bool:
            absolute = torch.logical_xor(left, right).to(torch.float64)
            left_magnitude = left.to(torch.float64)
            right_magnitude = right.to(torch.float64)
        else:
            left64 = left.to(torch.float64)
            right64 = right.to(torch.float64)
            absolute = torch.abs(left64 - right64)
            left_magnitude = torch.abs(left64)
            right_magnitude = torch.abs(right64)
        selected_absolute = absolute[different]
        if selected_absolute.numel():
            max_absolute = _json_float(selected_absolute.max())
            denominator = torch.maximum(
                left_magnitude[different],
                right_magnitude[different],
            ).clamp_min(torch.finfo(torch.float64).tiny)
            max_relative = _json_float(
                (selected_absolute / denominator).max()
            )
    if left.dtype in {torch.float16, torch.float32} and left.numel():
        finite = torch.isfinite(left) & torch.isfinite(right) & different
        if bool(torch.any(finite).detach().cpu().item()):
            left_bits = _ordered_float_bits(left)
            right_bits = _ordered_float_bits(right)
            max_ulp = int(
                torch.abs(left_bits[finite] - right_bits[finite])
                .max()
                .detach()
                .cpu()
                .item()
            )
    return Phase8AOutputDifference(
        **common,
        reason="tensor_values_differ",
        different_element_count=different_count,
        max_absolute_difference=max_absolute,
        max_relative_difference=max_relative,
        max_ulp_difference=max_ulp,
    )


def compare_phase8a_hierarchy_outputs(
    left: Phase8AHierarchySSLForwardOutput,
    right: Phase8AHierarchySSLForwardOutput,
    *,
    retained_limit: int = PHASE8A_OUTPUT_DIFFERENCE_MAX_RETAINED,
) -> Phase8AOutputDifferenceDiagnostic:
    """Compare two outputs without retaining any tensor in the result."""

    if not isinstance(
        left,
        Phase8AHierarchySSLForwardOutput,
    ) or not isinstance(right, Phase8AHierarchySSLForwardOutput):
        raise TypeError("Phase 8A hierarchy outputs are required")
    if (
        isinstance(retained_limit, bool)
        or not isinstance(retained_limit, int)
        or not 1 <= retained_limit <= PHASE8A_OUTPUT_DIFFERENCE_MAX_RETAINED
    ):
        raise ValueError("retained_limit is outside the bounded contract")

    retained: list[Phase8AOutputDifference] = []
    total = 0
    group_counts = {
        "embeddings": 0,
        "predictions": 0,
        "targets": 0,
        "loss_tensors": 0,
        "other": 0,
    }

    def record(value: Phase8AOutputDifference) -> None:
        nonlocal total
        total += 1
        group_counts[value.group] += 1
        if len(retained) < retained_limit:
            retained.append(value)

    def structure(
        path: str,
        reason: str,
        left_value: object,
        right_value: object,
    ) -> None:
        record(
            Phase8AOutputDifference(
                path=path,
                group=_difference_group(path),
                kind="structure",
                reason=reason,
                left_summary=_bounded_summary(left_value),
                right_summary=_bounded_summary(right_value),
            )
        )

    def visit(left_value: object, right_value: object, path: str) -> None:
        if isinstance(left_value, Tensor) or isinstance(right_value, Tensor):
            if not isinstance(left_value, Tensor) or not isinstance(
                right_value,
                Tensor,
            ):
                structure(
                    path,
                    "tensor_type_mismatch",
                    left_value,
                    right_value,
                )
                return
            difference = _tensor_difference(
                left_value,
                right_value,
                path=path,
            )
            if difference is not None:
                record(difference)
            return
        if is_dataclass(left_value) or is_dataclass(right_value):
            if (
                not is_dataclass(left_value)
                or not is_dataclass(right_value)
                or type(left_value) is not type(right_value)
            ):
                structure(
                    path,
                    "dataclass_type_mismatch",
                    left_value,
                    right_value,
                )
                return
            left_fields = fields(left_value)
            right_fields = fields(right_value)
            if tuple(field.name for field in left_fields) != tuple(
                field.name for field in right_fields
            ):
                structure(
                    path,
                    "dataclass_field_mismatch",
                    left_value,
                    right_value,
                )
                return
            for field in left_fields:
                visit(
                    getattr(left_value, field.name),
                    getattr(right_value, field.name),
                    f"{path}.{field.name}",
                )
            return
        if isinstance(left_value, Mapping) or isinstance(right_value, Mapping):
            if not isinstance(left_value, Mapping) or not isinstance(
                right_value,
                Mapping,
            ):
                structure(
                    path,
                    "mapping_type_mismatch",
                    left_value,
                    right_value,
                )
                return
            left_keys = tuple(left_value)
            right_keys = tuple(right_value)
            if left_keys != right_keys:
                structure(
                    path,
                    "mapping_key_mismatch",
                    left_value,
                    right_value,
                )
            for key in left_keys:
                if key in right_value:
                    visit(
                        left_value[key],
                        right_value[key],
                        f"{path}[{key!r}]",
                    )
            return
        if isinstance(left_value, (tuple, list)) or isinstance(
            right_value,
            (tuple, list),
        ):
            if type(left_value) is not type(right_value):
                structure(
                    path,
                    "sequence_type_mismatch",
                    left_value,
                    right_value,
                )
                return
            assert isinstance(left_value, (tuple, list))
            assert isinstance(right_value, (tuple, list))
            if len(left_value) != len(right_value):
                structure(
                    path,
                    "sequence_length_mismatch",
                    left_value,
                    right_value,
                )
            for index, (left_child, right_child) in enumerate(
                zip(left_value, right_value)
            ):
                visit(left_child, right_child, f"{path}[{index}]")
            return
        if (
            type(left_value) is not type(right_value)
            or left_value != right_value
        ):
            record(
                Phase8AOutputDifference(
                    path=path,
                    group=_difference_group(path),
                    kind="value",
                    reason="value_mismatch",
                    left_summary=_bounded_summary(left_value),
                    right_summary=_bounded_summary(right_value),
                )
            )

    visit(left, right, "output")
    group_order = (
        "embeddings",
        "predictions",
        "targets",
        "loss_tensors",
        "other",
    )
    retained_paths = {
        group: tuple(
            value.path for value in retained if value.group == group
        )
        for group in group_order
    }
    return Phase8AOutputDifferenceDiagnostic(
        contract_version=(
            PHASE8A_OUTPUT_DIFFERENCE_DIAGNOSTIC_CONTRACT_VERSION
        ),
        bit_exact=total == 0,
        first_difference_path=(None if not retained else retained[0].path),
        total_difference_count=total,
        retained_difference_count=len(retained),
        retained_limit=retained_limit,
        truncated=total > len(retained),
        difference_count_by_group=tuple(
            (group, group_counts[group]) for group in group_order
        ),
        retained_path_by_group=tuple(
            (group, retained_paths[group]) for group in group_order
        ),
        differences=tuple(retained),
    )


__all__ = [
    "PHASE8A_OUTPUT_DIFFERENCE_DIAGNOSTIC_CONTRACT_VERSION",
    "PHASE8A_OUTPUT_DIFFERENCE_MAX_RETAINED",
    "Phase8AOutputDifference",
    "Phase8AOutputDifferenceDiagnostic",
    "compare_phase8a_hierarchy_outputs",
]
