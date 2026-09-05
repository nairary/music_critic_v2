"""One-batch objective composition for the synthetic WS2 engineering harness.

No corpus factory, scheduler, epoch loop, optimizer, or checkpoint manager is
created here. Callers prepare raw batches/mask bindings and explicitly choose
coefficients. The historical corrected loss is reused without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from torch import Tensor, nn

from music_critic.experiments.analysisgnn.corrected_model import CorrectedModelOutput
from music_critic.experiments.analysisgnn.corrected_training import (
    CorrectedLossReport,
    CorrectedTargetAlignment,
    FrozenClassWeights,
    align_target_sidecars_after_prediction,
    corrected_supervised_loss,
)
from music_critic.ssl.data import SSLBatch
from music_critic.ssl.masking import PreparedMaskBinding
from music_critic.ssl.model import SSLForwardOutput, Phase8AHierarchySSLForwardOutput
from music_critic.ssl.theory_warmstart import (
    BACKBONE_GROUPS, FRESH_GROUPS, INACTIVE_GROUPS, RETAINED_GROUPS,
    TheoryWarmstartResult,
)


@dataclass(frozen=True)
class TheoryRetentionOutput:
    total: Tensor | None
    ssl_loss: Tensor | None
    theory_loss: Tensor | None
    ssl_output: SSLForwardOutput | Phase8AHierarchySSLForwardOutput | None
    theory_output: CorrectedModelOutput | None
    theory_report: CorrectedLossReport | None
    alignment: CorrectedTargetAlignment | None
    counts: Mapping[str, int]
    diagnostics: tuple[object, ...]
    parameter_inventory: Mapping[str, object]


def _coefficient(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
        not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"ws2.{name}.finite_nonnegative_required")
    return float(value)


def _loss_with_empty_candidate_heads(
    output: CorrectedModelOutput,
    alignment: CorrectedTargetAlignment,
    class_weights: FrozenClassWeights,
) -> CorrectedLossReport:
    """Omit unaddressable rows only from the temporary loss input.

    Raw graphs may have beats but no notes/onsets. Exact alignment retains
    unmatched rows as invalid evidence with a placeholder candidate index;
    that index must never be gathered from an empty prediction tensor. Pass
    genuinely empty rows to the unchanged corrected loss, then restore the
    original invalid-row counts in its report. The returned public alignment,
    diagnostics, and predictions remain untouched; no logits are fabricated.
    """
    loss_heads = dict(alignment.heads)
    omitted_counts = {}
    for task_id, rows in alignment.heads.items():
        if output.logits[task_id].shape[0] != 0 or rows.valid_mask.numel() == 0:
            continue
        if bool(rows.valid_mask.any()):
            raise ValueError(f"ws2.valid_target_without_raw_candidates:{task_id}")
        omitted_counts[task_id] = rows.valid_mask.numel()
        loss_heads[task_id] = replace(
            rows,
            candidate_indices=rows.candidate_indices[:0],
            values=rows.values[:0],
            valid_mask=rows.valid_mask[:0],
            sample_indices=rows.sample_indices[:0],
            entity_ids=(), record_ids=(), component_ids=(),
            masked_row_count=0, alignment_failure_count=0,
        )
    loss_alignment = replace(alignment, heads=loss_heads) if omitted_counts else alignment
    report = corrected_supervised_loss(output, loss_alignment, class_weights)
    if not omitted_counts:
        return report
    return replace(report, heads={
        task_id: replace(head, masked_row_count=omitted_counts[task_id])
        if task_id in omitted_counts else head
        for task_id, head in report.heads.items()
    })


class TheoryRetentionHarness(nn.Module):
    """Two objective branches over identical backbone Parameter objects.

    Branch modes follow this owning module. If changing individual submodules'
    modes, do so after ``harness.train()/eval()`` because the backbone is shared.
    Before each backward call, clear gradients with ``set_to_none=True`` so
    previously active branches cannot retain stale gradients in an optimizer.
    """

    def __init__(self, warmstart: TheoryWarmstartResult):
        super().__init__()
        self.ssl_model = warmstart.ssl_model
        self.theory_model = warmstart.theory_model
        self.transfer_report = warmstart.report
        self._parameter_groups()  # Verify sharing and complete partition now.

    def _parameter_groups(self):
        groups: dict[str, list[tuple[str, nn.Parameter]]] = {
            "shared": [], "ssl": [], "theory": [], "inactive": [],
        }
        for group in BACKBONE_GROUPS:
            if self.theory_model.get_submodule(group) is not self.ssl_model.get_submodule(
                "encoder." + group
            ):
                raise ValueError(f"ws2.shared_module_identity_mismatch:{group}")
        for name, parameter in self.ssl_model.named_parameters():
            if name.startswith(tuple("encoder." + group + "." for group in BACKBONE_GROUPS)):
                role = "shared"
            elif any(name == group or name.startswith(group + ".") for group in FRESH_GROUPS):
                role = "ssl"
            elif name.startswith(tuple(group + "." for group in INACTIVE_GROUPS)):
                role = "inactive"
                if parameter.requires_grad:
                    raise ValueError("ws2.inactive_parameter_requires_grad")
            else:
                raise ValueError(f"ws2.unaccounted_ssl_parameter:{name}")
            groups[role].append(("ssl_model." + name, parameter))
        for name, parameter in self.theory_model.named_parameters():
            if name.startswith(tuple(group + "." for group in RETAINED_GROUPS)):
                groups["theory"].append(("theory_model." + name, parameter))
            elif not name.startswith(tuple(group + "." for group in BACKBONE_GROUPS)):
                raise ValueError(f"ws2.unaccounted_theory_parameter:{name}")
        parameters = [parameter for rows in groups.values() for _, parameter in rows]
        if len(parameters) != len({id(parameter) for parameter in parameters}):
            raise ValueError("ws2.parameter_inventory_duplicate")
        if {id(parameter) for parameter in parameters} != {id(p) for p in self.parameters()}:
            raise ValueError("ws2.parameter_inventory_incomplete")
        return groups

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        """Each active trainable parameter exactly once; legacy heads excluded."""
        groups = self._parameter_groups()
        return tuple(parameter for role in ("shared", "ssl", "theory")
                     for _, parameter in groups[role] if parameter.requires_grad)

    def parameter_inventory(self) -> dict[str, object]:
        return {
            role: {
                "names": tuple(name for name, _ in rows),
                "parameter_count": len(rows),
                "numel": sum(parameter.numel() for _, parameter in rows),
                "trainable_numel": sum(p.numel() for _, p in rows if p.requires_grad),
                "optimizer_active": role != "inactive",
            }
            for role, rows in self._parameter_groups().items()
        }

    def forward(
        self, *, lambda_ssl: float, lambda_theory: float,
        ssl_batch: SSLBatch | None = None,
        prepared_mask_binding: PreparedMaskBinding | None = None,
        theory_raw_batch: object | None = None,
        theory_sidecars: Sequence[Mapping[str, object]] | None = None,
        class_weights: FrozenClassWeights | None = None,
    ) -> TheoryRetentionOutput:
        lambda_ssl = _coefficient(lambda_ssl, "lambda_ssl")
        lambda_theory = _coefficient(lambda_theory, "lambda_theory")
        inventory = self.parameter_inventory()
        if (ssl_batch is None) != (prepared_mask_binding is None):
            raise ValueError("ws2.ssl_batch_and_binding_required_together")
        if theory_sidecars and theory_raw_batch is None:
            raise ValueError("ws2.theory_sidecars_require_raw_batch")
        if lambda_theory > 0 and theory_raw_batch is not None and theory_sidecars:
            if class_weights is None:
                raise ValueError("ws2.theory_class_weights_required")
            if len(theory_sidecars) != int(theory_raw_batch.num_graphs):
                raise ValueError("ws2.theory_sidecar_batch_count_mismatch")
        diagnostics: list[object] = []
        ssl_output = theory_output = theory_report = alignment = None
        ssl_loss = theory_loss = None
        counts = {
            "ssl_samples": 0, "ssl_selected_notes": 0,
            "theory_samples": 0, "theory_valid_rows": 0,
            "theory_masked_rows": 0, "theory_unsupported_rows": 0,
            "theory_active_heads": 0,
        }
        if lambda_ssl > 0 and ssl_batch is not None:
            assert prepared_mask_binding is not None
            forward = (self.ssl_model.forward_hierarchy
                       if prepared_mask_binding.hierarchy_profile_version is not None
                       else self.ssl_model.forward)
            ssl_output = forward(ssl_batch, prepared_mask_binding=prepared_mask_binding)
            ssl_loss = ssl_output.objective.total_loss
            counts["ssl_samples"] = ssl_batch.sample_count
            counts["ssl_selected_notes"] = ssl_output.selected_global_note_indices.numel()
            if ssl_loss is None:
                diagnostics.append(("ssl_unavailable", ssl_output.objective.unavailable_reason))
        elif lambda_ssl > 0:
            diagnostics.append("ssl_batch_absent")
        if lambda_theory > 0 and theory_raw_batch is not None:
            # No labels, masks, weights, confidence, or semantic provenance can
            # enter this call. All sidecar processing follows raw prediction.
            theory_output = self.theory_model(theory_raw_batch)
            counts["theory_samples"] = int(theory_raw_batch.num_graphs)
            if theory_sidecars:
                assert class_weights is not None
                alignment = align_target_sidecars_after_prediction(
                    theory_output, theory_raw_batch, theory_sidecars
                )
                theory_report = _loss_with_empty_candidate_heads(
                    theory_output, alignment, class_weights
                )
                theory_loss = theory_report.total
                reports = tuple(theory_report.heads.values())
                counts["theory_valid_rows"] = sum(row.valid_row_count for row in reports)
                counts["theory_masked_rows"] = sum(row.masked_row_count for row in reports)
                counts["theory_unsupported_rows"] = sum(row.unsupported_row_count for row in reports)
                counts["theory_active_heads"] = sum(row.weighted_ce is not None for row in reports)
                diagnostics.extend(alignment.diagnostics)
                if theory_loss is None:
                    diagnostics.append("theory_no_valid_targets")
            else:
                diagnostics.append("theory_sidecars_absent")
        elif lambda_theory > 0:
            diagnostics.append("theory_batch_absent")
        terms = [coefficient * loss for coefficient, loss in (
            (lambda_ssl, ssl_loss), (lambda_theory, theory_loss)
        ) if loss is not None]
        total = sum(terms[1:], terms[0]) if terms else None
        return TheoryRetentionOutput(
            total, ssl_loss, theory_loss, ssl_output, theory_output,
            theory_report, alignment, counts, tuple(diagnostics), inventory,
        )
