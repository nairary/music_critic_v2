"""Deterministic candidate-first supervised checkpoint evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch

from music_critic.device import RuntimeDeviceError, resolve_runtime_device
from music_critic.evaluation.checkpoint import (
    load_evaluation_checkpoint,
)
from music_critic.evaluation.contracts import (
    EVALUATION_ARTIFACT_VERSION,
    EVALUATION_CONTRACT_VERSION,
    MACRO_SUMMARY_CONTRACT_VERSION,
    EvaluationContractError,
    canonical_fingerprint,
    metric_value,
    write_json_atomic,
)
from music_critic.evaluation.data import build_evaluation_data_runtime
from music_critic.evaluation.metrics import make_metric_accumulator
from music_critic.evaluation.priors import (
    TrainPriorBuilder,
    TrivialBaselineAccumulator,
    validate_train_priors,
)
from music_critic.models import ACTIVE_TASK_IDS
from music_critic.models.heads import join_task_supervision
from music_critic.tasks import (
    TARGET_ENCODING_BY_TASK,
    MultiSourceBatch,
)
from music_critic.training.device import move_multisource_batch


_MANAGED_ARTIFACTS = {
    "resolved_evaluation_config.json",
    "checkpoint_evidence.json",
    "train_priors.json",
    "metrics.json",
    "evaluation_report.json",
}

_DATASET_ADAPTER_FRAGMENT = {
    "hooktheory": "music_critic.adapters.hooktheory",
    "pop909_cl": "music_critic.adapters.pop909_cl",
}

_MACRO_METRICS_BY_KIND = {
    "closed_categorical_index": (
        "top1_accuracy",
        "top3_accuracy",
        "balanced_accuracy",
        "macro_f1",
        "micro_f1",
    ),
    "closed_multilabel": (
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ),
}

_OMITTED_MACRO_METRICS_BY_KIND = {
    "closed_categorical_index": {
        "nll": (
            "not aggregated across tasks with distinct categorical "
            "vocabularies and probability spaces"
        ),
    },
    "closed_multilabel": {
        "bce_nll": (
            "not aggregated across tasks with distinct label dimensions "
            "and label semantics"
        ),
        "exact_match_accuracy": (
            "not aggregated because exact-match difficulty depends on the "
            "task-specific label-set dimension"
        ),
    },
}

_ENCODING_KIND_BY_METRIC_KIND = {
    "closed_categorical": "closed_categorical_index",
    "closed_multilabel": "closed_multilabel",
}


@dataclass(slots=True)
class _TaskCounts:
    sample_count: int = 0
    candidate_count: int = 0
    target_row_count: int = 0
    eligible_row_count: int = 0
    masked_row_count: int = 0
    unaligned_row_count: int = 0
    conflict_row_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sample_count": self.sample_count,
            "candidate_count": self.candidate_count,
            "target_row_count": self.target_row_count,
            "eligible_row_count": self.eligible_row_count,
            "masked_row_count": self.masked_row_count,
            "unaligned_row_count": self.unaligned_row_count,
            "conflict_row_count": self.conflict_row_count,
        }


def _plain_config(config: object) -> dict[str, Any]:
    if OmegaConf.is_config(config):
        result = OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, "__dataclass_fields__"):
        result = OmegaConf.to_container(
            OmegaConf.structured(config), resolve=True
        )
    elif isinstance(config, dict):
        result = json.loads(json.dumps(config))
    else:
        raise EvaluationContractError(
            "evaluation.config.type_invalid"
        )
    if not isinstance(result, dict):
        raise EvaluationContractError(
            "evaluation.config.mapping_invalid"
        )
    return result


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("split") not in {"validation", "test"}:
        raise EvaluationContractError(
            "evaluation.config.split_invalid"
        )
    if config["split"] == "test" and not config.get(
        "acknowledge_test_evaluation"
    ):
        raise EvaluationContractError(
            "evaluation.test.acknowledgement_required"
        )
    if (
        not isinstance(config.get("checkpoint"), str)
        or not config["checkpoint"]
    ):
        raise EvaluationContractError(
            "evaluation.config.checkpoint_required"
        )
    if (
        isinstance(config.get("seed"), bool)
        or not isinstance(config.get("seed"), int)
        or config["seed"] < 0
    ):
        raise EvaluationContractError(
            "evaluation.config.seed_invalid"
        )
    data = config.get("data")
    if not isinstance(data, dict):
        raise EvaluationContractError(
            "evaluation.config.data_invalid"
        )
    for name in ("batch_size",):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvaluationContractError(
                f"evaluation.config.{name}_invalid"
            )
    for name in ("max_train_samples", "max_evaluation_samples"):
        value = data.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise EvaluationContractError(
                f"evaluation.config.{name}_invalid"
            )
    workers = data.get("workers")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 0
    ):
        raise EvaluationContractError(
            "evaluation.config.workers_invalid"
        )
    if not isinstance(config.get("overwrite_output"), bool):
        raise EvaluationContractError(
            "evaluation.config.overwrite_invalid"
        )
    if (
        not isinstance(config.get("output_dir"), str)
        or not config["output_dir"]
    ):
        raise EvaluationContractError(
            "evaluation.config.output_dir_invalid"
        )
    device = config.get("device")
    if (
        not isinstance(device, dict)
        or not isinstance(device.get("amp"), bool)
        or not isinstance(device.get("non_blocking"), bool)
    ):
        raise EvaluationContractError(
            "evaluation.config.device_invalid"
        )


def _resolve_device(config: dict[str, Any]) -> torch.device:
    name = config["device"]["name"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}:
        raise EvaluationContractError(
            f"evaluation.device.unknown:{name}"
        )
    try:
        return resolve_runtime_device(name)
    except RuntimeDeviceError as exc:
        if exc.category == "runtime.device.cuda_unavailable":
            raise EvaluationContractError(
                "evaluation.device.cuda_unavailable"
            ) from exc
        raise EvaluationContractError(
            f"evaluation.device.invalid:{exc}"
        ) from exc


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise EvaluationContractError(
            "evaluation.output.not_directory"
        )
    existing = {
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name in _MANAGED_ARTIFACTS
    } if path.exists() else set()
    if existing and not overwrite:
        raise EvaluationContractError(
            "evaluation.output.managed_artifact_collision"
        )
    if overwrite:
        for name in _MANAGED_ARTIFACTS:
            (path / name).unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


def _prior_bindings(bindings: dict[str, object]) -> dict[str, object]:
    names = (
        "kind",
        "index_fingerprints",
        "cache_fingerprints",
        "split_manifest_fingerprint",
        "effective_split_manifest_fingerprint",
        "train_composition_fingerprint",
        "train_membership_fingerprint",
        "ontology_version",
        "ontology_fingerprint",
        "encoding_version",
        "encoding_fingerprint",
        "cache_validation",
    )
    return {name: bindings[name] for name in names}


def _validate_checkpoint_data(
    checkpoint_evidence: dict[str, Any],
    bindings: dict[str, object],
    *,
    split: str,
) -> dict[str, object]:
    saved = checkpoint_evidence.get("training_data_fingerprints")
    if saved is None:
        return {
            "verified": False,
            "reason": "model-only checkpoint has no Phase 6C data binding",
            "matched_fields": [],
        }
    if not isinstance(saved, dict):
        raise EvaluationContractError(
            "evaluation.checkpoint.data_binding_invalid"
        )
    if bindings["kind"] == "bounded":
        fields = ["kind", "validation_membership_fingerprint"]
    else:
        fields = [
            "kind",
            "index_fingerprints",
            "split_manifest_fingerprint",
            "train_composition_fingerprint",
        ]
        if split == "validation":
            fields.extend(
                [
                    "validation_composition_fingerprint",
                    "validation_membership_fingerprint",
                ]
            )
    current = dict(bindings)
    current["validation_composition_fingerprint"] = bindings.get(
        "evaluation_composition_fingerprint"
    )
    current["validation_membership_fingerprint"] = bindings.get(
        "evaluation_membership_fingerprint"
    )
    mismatched = [
        name for name in fields if saved.get(name) != current.get(name)
    ]
    if mismatched:
        raise EvaluationContractError(
            "evaluation.checkpoint.data_binding_mismatch:"
            + ",".join(mismatched)
        )
    return {
        "verified": True,
        "matched_fields": fields,
        "model_contract_verified_against_current_code": True,
        "ontology_and_encoding_verified_via_model_contract": True,
        "cache_artifacts_verified_on_read": True,
        "test_not_used_for_checkpoint_selection": split == "test",
    }


def _load_or_build_priors(
    config: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    bindings = _prior_bindings(runtime.bindings)
    path = config.get("train_priors_path")
    if path:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_train_priors(
            artifact, expected_bindings=bindings
        )
        return artifact
    builder = TrainPriorBuilder(bindings=bindings)
    for batch in runtime.train_loader():
        builder.add_batch(batch)
    artifact = builder.finalize()
    validate_train_priors(artifact, expected_bindings=bindings)
    return artifact


def _conflict_flags(target: Any) -> tuple[bool, ...]:
    return tuple(
        any(
            diagnostic.code == "multisource.alignment_conflict"
            for diagnostic in diagnostics
        )
        for diagnostics in target.diagnostics_cpu
    )


def _delta(
    model_metric: dict[str, Any],
    baseline_metric: dict[str, Any],
) -> dict[str, Any]:
    left = model_metric["value"]
    right = baseline_metric["value"]
    if left is None or right is None:
        return metric_value(
            None,
            category="comparison_metric_undefined",
            reason="model or baseline metric is undefined",
        )
    return metric_value(float(left) - float(right))


def _task_macro_metric(
    task_metrics: dict[str, dict[str, Any]],
    metric_name: str,
) -> dict[str, Any]:
    included = []
    undefined = []
    values = []
    for task_id, metrics in sorted(task_metrics.items()):
        metric = metrics.get(metric_name)
        if (
            not isinstance(metric, dict)
            or "value" not in metric
            or metric["value"] is None
        ):
            undefined.append(task_id)
            continue
        included.append(task_id)
        values.append(float(metric["value"]))
    value = (
        metric_value(
            None,
            category="no_defined_task_metrics",
            reason=(
                f"no task in this dataset/encoding group defines "
                f"{metric_name}"
            ),
        )
        if not values
        else metric_value(math.fsum(values) / len(values))
    )
    return {
        **value,
        "included_task_ids": included,
        "undefined_task_ids": undefined,
        "defined_task_count": len(included),
        "undefined_task_count": len(undefined),
        "aggregation_rule": (
            "unweighted arithmetic mean over defined task-level metric "
            "values; undefined task metrics are excluded and counted"
        ),
    }


def build_macro_summaries(
    datasets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate only comparable normalized metrics within source and kind."""

    groups = []
    for dataset_id, tasks in sorted(datasets.items()):
        by_kind: dict[str, dict[str, dict[str, Any]]] = {}
        for task_id, evidence in sorted(tasks.items()):
            model_metrics = evidence["model"]
            metric_kind = model_metrics["kind"]
            kind = _ENCODING_KIND_BY_METRIC_KIND.get(metric_kind)
            if kind is None:
                raise EvaluationContractError(
                    f"evaluation.summary.encoding_unknown:{metric_kind}"
                )
            by_kind.setdefault(kind, {})[task_id] = model_metrics
        for kind, task_metrics in sorted(by_kind.items()):
            groups.append(
                {
                    "dataset_id": dataset_id,
                    "encoding_kind": kind,
                    "candidate_task_ids": sorted(task_metrics),
                    "metrics": {
                        metric_name: _task_macro_metric(
                            task_metrics, metric_name
                        )
                        for metric_name in _MACRO_METRICS_BY_KIND[kind]
                    },
                    "omitted_metrics": {
                        metric_name: {
                            "category": "scientifically_incomparable",
                            "reason": reason,
                        }
                        for metric_name, reason in (
                            _OMITTED_MACRO_METRICS_BY_KIND[kind].items()
                        )
                    },
                }
            )
    return {
        "macro_summary_contract_version": (
            MACRO_SUMMARY_CONTRACT_VERSION
        ),
        "grouping_keys": ["dataset_id", "encoding_kind"],
        "cross_dataset_aggregation": False,
        "cross_encoding_aggregation": False,
        "groups": groups,
    }


def _evaluate(
    model: Any,
    batches: Any,
    *,
    priors: dict[str, Any],
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[tuple[str, str], Any] = {}
    baselines: dict[tuple[str, str], TrivialBaselineAccumulator] = {}
    counts: dict[tuple[str, str], _TaskCounts] = {}
    dataset_samples: Counter[str] = Counter()
    aggregate = Counter()
    batch_count = 0
    model.eval()
    with torch.no_grad():
        for cpu_batch in batches:
            if not isinstance(cpu_batch, MultiSourceBatch):
                raise EvaluationContractError(
                    "evaluation.batch.type_invalid"
                )
            batch_count += 1
            aggregate["sample_count"] += cpu_batch.statistics.sample_count
            dataset_samples.update(cpu_batch.dataset_ids)
            batch = move_multisource_batch(
                cpu_batch,
                device,
                non_blocking=bool(config["device"]["non_blocking"]),
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=bool(config["device"]["amp"]),
            ):
                # Candidate-first boundary: no target sidecar is passed here.
                _, predictions = model.predict(batch.raw_graph_batch)
            prediction_by_task = {
                prediction.task_id: prediction
                for prediction in predictions
            }
            # Only after raw logits exist may held-out targets be joined.
            supervisions = join_task_supervision(
                predictions, batch.target_batches
            )
            supervision_by_task = {
                item.task_id: item for item in supervisions
            }
            targets_by_task = {
                item.task_id: item for item in batch.target_batches
            }
            for task_id in ACTIVE_TASK_IDS:
                prediction = prediction_by_task[task_id]
                target = targets_by_task[task_id]
                conflicts = _conflict_flags(cpu_batch.target_batches[
                    tuple(
                        item.task_id
                        for item in cpu_batch.target_batches
                    ).index(task_id)
                ])
                encoding = TARGET_ENCODING_BY_TASK[task_id]
                labels = tuple(encoding.vocabulary or ())
                supervision = supervision_by_task.get(task_id)
                for sample_index, dataset_id in enumerate(
                    batch.dataset_ids
                ):
                    adapter_fragment = _DATASET_ADAPTER_FRAGMENT.get(
                        dataset_id
                    )
                    if adapter_fragment is None:
                        raise EvaluationContractError(
                            "evaluation.dataset.adapter_unknown:"
                            f"{dataset_id}"
                        )
                    if not prediction.source_adapter.startswith(
                        adapter_fragment
                    ):
                        # A source-native head is not a task for this
                        # dataset. It must not create an empty cross-source
                        # metric bucket or enter a macro average.
                        continue
                    key = (dataset_id, task_id)
                    state = counts.setdefault(key, _TaskCounts())
                    state.sample_count += 1
                    candidate_rows = (
                        prediction.sample_indices == sample_index
                    )
                    candidate_count = int(candidate_rows.sum())
                    state.candidate_count += candidate_count
                    aggregate["candidate_count"] += candidate_count
                    target_rows = (
                        target.sample_indices == sample_index
                    )
                    target_count = int(target_rows.sum())
                    state.target_row_count += target_count
                    aggregate["target_row_count"] += target_count
                    for row_index in torch.nonzero(
                        target_rows, as_tuple=False
                    ).flatten().to("cpu").tolist():
                        if conflicts[row_index]:
                            state.conflict_row_count += 1
                            aggregate["conflict_row_count"] += 1
                        elif not bool(
                            target.availability_mask[row_index]
                        ):
                            state.masked_row_count += 1
                            aggregate["masked_row_count"] += 1
                        elif not bool(
                            target.entity_index_mask[row_index]
                        ):
                            state.unaligned_row_count += 1
                            aggregate["unaligned_row_count"] += 1
                    metrics.setdefault(
                        key,
                        make_metric_accumulator(
                            encoding.encoding_kind, labels
                        ),
                    )
                    prior = (
                        priors.get("datasets", {})
                        .get(dataset_id, {})
                        .get(task_id)
                    )
                    if prior is not None:
                        baselines.setdefault(
                            key, TrivialBaselineAccumulator(prior)
                        )
                    if supervision is None:
                        continue
                    selected = torch.nonzero(
                        supervision.sample_indices == sample_index,
                        as_tuple=False,
                    ).flatten()
                    if selected.numel() == 0:
                        continue
                    candidate_indices = (
                        supervision.candidate_indices.index_select(
                            0, selected
                        )
                    )
                    target_indices = (
                        supervision.target_row_indices.index_select(
                            0, selected
                        )
                    )
                    logits = prediction.logits.index_select(
                        0, candidate_indices
                    )
                    values = target.values.index_select(
                        0, target_indices
                    )
                    metrics[key].update(logits, values)
                    if key in baselines:
                        baselines[key].update(values)
                    eligible = int(selected.numel())
                    state.eligible_row_count += eligible
                    aggregate["eligible_row_count"] += eligible
    datasets: dict[str, dict[str, Any]] = {}
    for (dataset_id, task_id), accumulator in sorted(metrics.items()):
        model_metrics = accumulator.finalize()
        baseline_metrics = (
            baselines[(dataset_id, task_id)].finalize()
            if (dataset_id, task_id) in baselines
            else {
                "unavailable": {
                    "category": "no_train_prior_rows",
                    "reason": (
                        "the train split has no eligible rows for this "
                        "dataset/task"
                    ),
                }
            }
        )
        comparison: dict[str, Any] = {}
        if "unavailable" not in baseline_metrics:
            for metric_name in (
                "top1_accuracy",
                "top3_accuracy",
                "balanced_accuracy",
                "macro_f1",
                "micro_f1",
                "macro_precision",
                "macro_recall",
                "micro_precision",
                "micro_recall",
                "exact_match_accuracy",
                "nll",
                "bce_nll",
            ):
                if (
                    metric_name in model_metrics
                    and metric_name in baseline_metrics
                ):
                    comparison[f"{metric_name}_model_minus_baseline"] = (
                        _delta(
                            model_metrics[metric_name],
                            baseline_metrics[metric_name],
                        )
                    )
        datasets.setdefault(dataset_id, {})[task_id] = {
            "counts": counts[(dataset_id, task_id)].as_dict(),
            "model": model_metrics,
            "train_only_baseline": baseline_metrics,
            "comparison": comparison,
        }
    return {
        "batch_count": batch_count,
        "dataset_sample_counts": dict(sorted(dataset_samples.items())),
        "counts": dict(sorted(aggregate.items())),
        "datasets": datasets,
        "macro_summaries": build_macro_summaries(datasets),
        "retained_prediction_tensor_count": 0,
        "retained_prediction_element_count": 0,
    }


def run_evaluation(config: object) -> dict[str, Any]:
    """Evaluate one existing checkpoint on fixed validation or explicit test."""

    resolved = _plain_config(config)
    _validate_config(resolved)
    output_dir = Path(resolved["output_dir"]).resolve()
    _prepare_output(
        output_dir, overwrite=bool(resolved["overwrite_output"])
    )
    device = _resolve_device(resolved)
    runtime = build_evaluation_data_runtime(
        OmegaConf.create(resolved["data"]),
        split=resolved["split"],
        seed=resolved["seed"],
    )
    model, checkpoint_evidence = load_evaluation_checkpoint(
        resolved["checkpoint"], device=device
    )
    data_verification = _validate_checkpoint_data(
        checkpoint_evidence,
        runtime.bindings,
        split=resolved["split"],
    )
    checkpoint_evidence["evaluation_data_verification"] = (
        data_verification
    )
    priors = _load_or_build_priors(resolved, runtime)
    evaluated = _evaluate(
        model,
        runtime.evaluation_loader(),
        priors=priors,
        device=device,
        config=resolved,
    )
    bindings = {
        **runtime.bindings,
        "checkpoint_sha256": checkpoint_evidence["checkpoint_sha256"],
        "train_prior_fingerprint": priors[
            "train_prior_fingerprint"
        ],
    }
    metrics = {
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_artifact_version": EVALUATION_ARTIFACT_VERSION,
        "split": resolved["split"],
        "test_evaluation_acknowledged": bool(
            resolved["acknowledge_test_evaluation"]
        ),
        "candidate_first_inference": True,
        "targets_joined_after_raw_logits": True,
        "checkpoint_selection_uses_test": False,
        "bindings": bindings,
        **evaluated,
    }
    metrics["metrics_fingerprint"] = canonical_fingerprint(metrics)
    report = {
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "status": "completed",
        "split": resolved["split"],
        "checkpoint_kind": checkpoint_evidence["checkpoint_kind"],
        "metrics_fingerprint": metrics["metrics_fingerprint"],
        "train_prior_fingerprint": priors["train_prior_fingerprint"],
        "data_verification": data_verification,
        "sample_count": evaluated["counts"].get("sample_count", 0),
        "eligible_row_count": evaluated["counts"].get(
            "eligible_row_count", 0
        ),
        "artifacts": sorted(_MANAGED_ARTIFACTS),
        "scientific_scope": "supervised_auxiliary_task_evaluation",
        "quality_score_claimed": False,
    }
    write_json_atomic(
        output_dir / "resolved_evaluation_config.json", resolved
    )
    write_json_atomic(
        output_dir / "checkpoint_evidence.json", checkpoint_evidence
    )
    write_json_atomic(output_dir / "train_priors.json", priors)
    write_json_atomic(output_dir / "metrics.json", metrics)
    write_json_atomic(output_dir / "evaluation_report.json", report)
    return report


__all__ = ["build_macro_summaries", "run_evaluation"]
