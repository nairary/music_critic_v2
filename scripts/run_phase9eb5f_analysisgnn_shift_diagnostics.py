#!/usr/bin/env python3
"""Evaluate sealed B5D C0/C1 checkpoints on 12 VALIDATION shift views."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import (
    CorrectedAnalysisGNNModel,
    corrected_model_contract,
    model_state_fingerprint,
)
from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS,
    CorrectedTrainingError,
    CorrectedValidationAccumulator,
    ProductionArtifactPaths,
    align_target_sidecars_after_prediction,
    build_source_free_fixture,
    frozen_split_assignments,
    load_frozen_class_weights,
    load_production_record,
    move_raw_graph_batch,
    per_head_validation_metrics,
    train_seen_joint_tuples,
    transpose_raw_graph_batch,
)
from music_critic.experiments.analysisgnn.multitask_contract import (
    TASK_BY_ID,
    get_vocabulary,
)
from music_critic.experiments.analysisgnn.transposition import (
    SHIFT_PCS,
    transform_semantic_value,
)
from music_critic.experiments.analysisgnn.transposition_diagnostics import (
    B5F_CHECKPOINT_SCHEMA,
    INVARIANT_HEADS,
    audit_graph_transform,
    audit_sidecar_targets,
    validate_checkpoint_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
B5E_FIXTURE = ROOT / "tests/fixtures/analysisgnn/phase9eb5e_full_training_results.json"
SHIFT0_ATOL = 1e-7


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_b5e() -> dict[str, object]:
    return json.loads(B5E_FIXTURE.read_text(encoding="utf-8"))


def _inverse_class_map(
    task_id: str, *, shift_pc: int, dialect: str
) -> tuple[dict[int, int], tuple[int, ...]]:
    labels = get_vocabulary(TASK_BY_ID[task_id].vocabulary_id).labels
    if task_id in INVARIANT_HEADS:
        return {index: index for index in range(len(labels))}, ()
    mapped: dict[int, int] = {}
    excluded: list[int] = []
    destinations: set[int] = set()
    for shifted_id, label in enumerate(labels):
        try:
            canonical = transform_semantic_value(
                task_id,
                label,
                shift_pc=(-shift_pc) % 12,
                dialect=dialect,
                profile="corrected_v2",
            )
            canonical_id = labels.index(canonical)
        except (ValueError, CorrectedTrainingError):
            excluded.append(shifted_id)
            continue
        except Exception:
            excluded.append(shifted_id)
            continue
        if canonical_id in destinations:
            excluded.append(shifted_id)
            continue
        destinations.add(canonical_id)
        mapped[shifted_id] = canonical_id
    return mapped, tuple(excluded)


class _EquivarianceAccumulator:
    def __init__(self) -> None:
        self.support = Counter()
        self.consistent = Counter()
        self.distance_sum = Counter()
        self.excluded_rows = Counter()
        self.excluded_classes: defaultdict[str, set[int]] = defaultdict(set)

    def update(
        self,
        canonical: Any,
        shifted: Any,
        *,
        shift_pc: int,
        dialect: str,
    ) -> None:
        for task_id in ACTIVE_HEADS:
            left = canonical.logits[task_id].detach().float().softmax(dim=1).cpu()
            right = shifted.logits[task_id].detach().float().softmax(dim=1).cpu()
            if left.shape != right.shape:
                self.excluded_rows[task_id] += max(left.shape[0], right.shape[0])
                continue
            mapping, excluded = _inverse_class_map(
                task_id, shift_pc=shift_pc, dialect=dialect
            )
            self.excluded_classes[task_id].update(excluded)
            for row_index in range(left.shape[0]):
                shifted_argmax = int(right[row_index].argmax())
                mapped_argmax = mapping.get(shifted_argmax)
                if mapped_argmax is None:
                    self.excluded_rows[task_id] += 1
                    continue
                mapped = torch.zeros_like(left[row_index])
                for source_id, destination_id in mapping.items():
                    mapped[destination_id] += right[row_index, source_id]
                mass = float(mapped.sum())
                if mass <= 0:
                    self.excluded_rows[task_id] += 1
                    continue
                mapped /= mass
                self.support[task_id] += 1
                self.consistent[task_id] += int(
                    mapped_argmax == int(left[row_index].argmax())
                )
                self.distance_sum[task_id] += float(
                    torch.abs(mapped - left[row_index]).sum() / 2
                )

    def finalize(self) -> dict[str, object]:
        return {
            task_id: {
                "argmax_consistency": (
                    None
                    if self.support[task_id] == 0
                    else self.consistent[task_id] / self.support[task_id]
                ),
                "mean_total_variation_distance": (
                    None
                    if self.support[task_id] == 0
                    else self.distance_sum[task_id] / self.support[task_id]
                ),
                "consistency_support": self.support[task_id],
                "excluded_rows": self.excluded_rows[task_id],
                "excluded_oov_or_non_bijective_class_ids": sorted(
                    self.excluded_classes[task_id]
                ),
            }
            for task_id in ACTIVE_HEADS
        }


def _checkpoint_payload(
    path: Path,
    *,
    profile: str,
    b5e: Mapping[str, object],
    device: str,
) -> tuple[CorrectedAnalysisGNNModel, dict[str, object], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = str(b5e["run_summaries"][profile]["final_model_state_fingerprint"])  # type: ignore[index]
    architecture = corrected_model_contract(CorrectedAnalysisGNNModel())["fingerprint"]
    schedule = str(b5e["run_summaries"][profile]["record_schedule_fingerprint"])  # type: ignore[index]
    metadata = validate_checkpoint_metadata(
        payload,
        profile=profile,
        expected_model_fingerprint=expected,
        expected_model_contract_fingerprint=str(architecture),
        expected_record_schedule_fingerprint=schedule,
    )
    if metadata["valid"] is not True:
        raise CorrectedTrainingError(
            "analysisgnn.b5f.checkpoint_metadata_invalid",
            f"{profile}:{metadata['checks']}",
        )
    model = CorrectedAnalysisGNNModel()
    model.load_state_dict(payload["model_state"])
    if model_state_fingerprint(model) != expected:
        raise CorrectedTrainingError(
            "analysisgnn.b5f.checkpoint_state_invalid", profile
        )
    model.to(device).eval()
    return model, dict(payload), metadata


def _evaluate_profile(
    model: CorrectedAnalysisGNNModel,
    *,
    profile: str,
    device: str,
    paths: ProductionArtifactPaths,
) -> dict[str, object]:
    assignments = frozen_split_assignments(paths)
    records = tuple(
        sorted(
            record_id
            for record_id, row in assignments.items()
            if row["split"] == "validation"
        )
    )
    if len(records) != 162:
        raise CorrectedTrainingError(
            "analysisgnn.b5f.validation_count_mismatch", str(len(records))
        )
    weights = load_frozen_class_weights()
    seen = train_seen_joint_tuples(paths)
    accumulators = {
        shift: CorrectedValidationAccumulator(weights, train_seen_tuples=seen)
        for shift in SHIFT_PCS
    }
    equivariance = {shift: _EquivarianceAccumulator() for shift in SHIFT_PCS}
    eligible_records = Counter()
    eligible_components: defaultdict[int, set[str]] = defaultdict(set)
    excluded: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    with torch.no_grad():
        for record_index, record_id in enumerate(records, 1):
            batch, sidecar = load_production_record(
                record_id, split="validation", paths=paths
            )
            canonical_raw = move_raw_graph_batch(batch.raw_graph_batch, device)
            canonical_output = model(canonical_raw)
            source_graph = batch.raw_graph_batch.to_data_list()[0]
            for shift_pc in SHIFT_PCS:
                graph_audit = audit_graph_transform(source_graph, shift_pc=shift_pc)
                target_audit = audit_sidecar_targets(sidecar, shift_pc=shift_pc)
                reasons = list(graph_audit["invalid_reasons"]) + list(
                    target_audit["invalid_reasons"]
                )
                eligible = bool(graph_audit["midi_range_valid"]) and bool(
                    target_audit["target_vocabulary_closed"]
                )
                if not eligible:
                    excluded[shift_pc].append(
                        {"record_id": record_id, "reasons": sorted(set(reasons))}
                    )
                    continue
                shifted_cpu = transpose_raw_graph_batch(
                    batch.raw_graph_batch, (shift_pc,)
                )
                shifted = move_raw_graph_batch(shifted_cpu, device)
                output = model(shifted)
                alignment = align_target_sidecars_after_prediction(
                    output, shifted, (sidecar,), shifts=(shift_pc,)
                )
                accumulators[shift_pc].update(
                    output, alignment, sidecars=(sidecar,)
                )
                equivariance[shift_pc].update(
                    canonical_output,
                    output,
                    shift_pc=shift_pc,
                    dialect=str(sidecar["dialect"]),
                )
                eligible_records[shift_pc] += 1
                eligible_components[shift_pc].add(str(sidecar["source_component_id"]))
            if record_index % 20 == 0 or record_index == len(records):
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_validation_progress",
                            "profile": profile,
                            "record_index": record_index,
                            "record_count": len(records),
                            "record_id": record_id,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    per_shift = {}
    for shift_pc in SHIFT_PCS:
        prediction_support = {
            task: accumulators[shift_pc].confusions[task].sum(0).tolist()
            for task in ACTIVE_HEADS
        }
        metrics = accumulators[shift_pc].finalize()
        per_head = metrics["per_head"]
        for task_id in ACTIVE_HEADS:
            per_head[task_id]["prediction_support"] = prediction_support[task_id]
            per_head[task_id]["valid_target_rows"] = per_head[task_id]["support"]
        per_shift[str(shift_pc)] = {
            "shift_pc": shift_pc,
            "eligible_records": eligible_records[shift_pc],
            "eligible_components": len(eligible_components[shift_pc]),
            "excluded_record_count": len(excluded[shift_pc]),
            "excluded_records": excluded[shift_pc],
            "metrics": metrics,
            "equivariance": equivariance[shift_pc].finalize(),
            "diagnostic_views_are_independent_examples": False,
            "canonical_validation_record_count": 162,
        }
    body: dict[str, object] = {
        "profile": profile,
        "per_shift": per_shift,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def _shift0_reproduction(
    results: Mapping[str, Mapping[str, object]], b5e: Mapping[str, object]
) -> dict[str, object]:
    metric_keys = (
        "corrected_primary_macro_score",
        "corrected_harmonic_event_joint_accuracy",
        "paper_text_note_joint_accuracy",
        "seen_tuple_joint_accuracy",
        "unseen_tuple_joint_accuracy",
        "direct_roman_numeral_accuracy",
        "direct_roman_numeral_macro_f1",
    )
    comparisons = {}
    passed = True
    for profile in ("C0", "C1"):
        observed = results[profile]["per_shift"]["0"]["metrics"]  # type: ignore[index]
        expected = b5e["final_validation_metrics"][profile]  # type: ignore[index]
        rows = {}
        for key in metric_keys:
            left = observed.get(key)
            right = expected.get(key)
            equal = left is not None and right is not None and abs(
                float(left) - float(right)
            ) <= SHIFT0_ATOL
            rows[key] = {"observed": left, "expected": right, "passed": equal}
            passed &= equal
        comparisons[profile] = rows
    body = {
        "absolute_tolerance": SHIFT0_ATOL,
        "passed": passed,
        "comparisons": comparisons,
    }
    body["fingerprint"] = fingerprint(body)
    return body


def run_smoke(*, output_dir: Path | None = None) -> dict[str, object]:
    """Exercise the runner on an in-memory checkpoint-free fixture."""

    batch, sidecar = build_source_free_fixture()
    weights = load_frozen_class_weights()
    torch.manual_seed(17)
    first = CorrectedAnalysisGNNModel().eval()
    second = CorrectedAnalysisGNNModel().eval()
    second.load_state_dict(first.state_dict())
    profiles = {}
    with torch.no_grad():
        for profile, model in (("C0", first), ("C1", second)):
            rows = {}
            canonical = model(batch.raw_graph_batch)
            for shift_pc in SHIFT_PCS:
                shifted = transpose_raw_graph_batch(
                    batch.raw_graph_batch, (shift_pc,)
                )
                output = model(shifted)
                alignment = align_target_sidecars_after_prediction(
                    output, shifted, (sidecar,), shifts=(shift_pc,)
                )
                metrics = per_head_validation_metrics(output, alignment, weights)
                equivariance = _EquivarianceAccumulator()
                equivariance.update(
                    canonical,
                    output,
                    shift_pc=shift_pc,
                    dialect=str(sidecar["dialect"]),
                )
                rows[str(shift_pc)] = {
                    "finite_logits": all(
                        bool(torch.isfinite(value).all())
                        for value in output.logits.values()
                    ),
                    "head_count": len(metrics),
                    "valid_target_rows": {
                        task: metrics[task]["support"] for task in ACTIVE_HEADS
                    },
                    "equivariance": equivariance.finalize(),
                }
            profiles[profile] = {
                "model_state_fingerprint": model_state_fingerprint(model),
                "per_shift": rows,
            }
    body: dict[str, object] = {
        "schema": "Phase9EB5FCheckpointShiftDiagnosticsSmoke@1.0.0",
        "phase": "9E-B5F",
        "valid": all(
            row["finite_logits"] and row["head_count"] == len(ACTIVE_HEADS)
            for profile in profiles.values()
            for row in profile["per_shift"].values()
        ),
        "profiles": profiles,
        "smoke_checkpoint_diagnostics_run": True,
        "checkpoint_diagnostics_run": False,
        "shift0_metrics_reproduced": False,
        "training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    if output_dir is not None:
        _write_json(output_dir / "smoke_checkpoint_shift_diagnostics.json", body)
    return body


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.smoke:
        return run_smoke(output_dir=args.output_dir)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise CorrectedTrainingError(
            "analysisgnn.b5f.cuda_unavailable", "checkpoint diagnostics require CUDA"
        )
    if args.device == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    b5e = _load_b5e()
    if args.c0_checkpoint is None or args.c1_checkpoint is None:
        raise CorrectedTrainingError(
            "analysisgnn.b5f.checkpoint_paths_required",
            "--c0-checkpoint and --c1-checkpoint are required outside --smoke",
        )
    checkpoint_paths = {"C0": args.c0_checkpoint, "C1": args.c1_checkpoint}
    models = {}
    metadata = {}
    for profile in ("C0", "C1"):
        model, _payload, metadata[profile] = _checkpoint_payload(
            checkpoint_paths[profile], profile=profile, b5e=b5e, device=args.device
        )
        models[profile] = model
    paths = ProductionArtifactPaths()
    results = {
        profile: _evaluate_profile(
            models[profile], profile=profile, device=args.device, paths=paths
        )
        for profile in ("C0", "C1")
    }
    shift0 = _shift0_reproduction(results, b5e)
    body: dict[str, object] = {
        "schema": B5F_CHECKPOINT_SCHEMA,
        "phase": "9E-B5F",
        "valid": bool(shift0["passed"]),
        "checkpoint_metadata": metadata,
        "results": results,
        "shift0_reproduction": shift0,
        "checkpoint_diagnostics_run": True,
        "shift0_metrics_reproduced": shift0["passed"],
        "model_selection_performed": False,
        "training_run": False,
        "test_loader_created": False,
        "test_targets_read": False,
        "test_metrics_computed": False,
    }
    body["fingerprint"] = fingerprint(body)
    _write_json(args.output_dir / "checkpoint_shift_diagnostics.json", body)
    return body


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-checkpoint", type=Path)
    parser.add_argument("--c1-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = run(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
