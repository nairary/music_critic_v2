"""Train-split-only trivial baselines with provenance-bearing artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any

import torch
from torch import Tensor

from music_critic.evaluation.contracts import (
    TRAIN_PRIOR_CONTRACT_VERSION,
    EvaluationContractError,
    canonical_fingerprint,
    metric_value,
)
from music_critic.evaluation.metrics import make_metric_accumulator
from music_critic.models import ACTIVE_TASK_IDS
from music_critic.tasks import TARGET_ENCODING_BY_TASK, MultiSourceBatch


@dataclass(slots=True)
class _CategoricalCounts:
    class_labels: tuple[str, ...]
    counts: list[int] = field(init=False)
    row_count: int = 0

    def __post_init__(self) -> None:
        self.counts = [0] * len(self.class_labels)

    def add(self, values: Tensor) -> None:
        counts = torch.bincount(
            values.detach().to(dtype=torch.long, device="cpu"),
            minlength=len(self.class_labels),
        )
        for index, count in enumerate(counts.tolist()):
            self.counts[index] += int(count)
        self.row_count += int(values.shape[0])

    def artifact(self) -> dict[str, Any]:
        probabilities = (
            [count / self.row_count for count in self.counts]
            if self.row_count
            else [None] * len(self.counts)
        )
        majority = (
            None
            if not self.row_count
            else max(
                range(len(self.counts)),
                key=lambda index: (self.counts[index], -index),
            )
        )
        return {
            "kind": "closed_categorical",
            "class_labels": list(self.class_labels),
            "train_eligible_row_count": self.row_count,
            "class_counts": list(self.counts),
            "empirical_probabilities": probabilities,
            "majority_class_index": majority,
            "majority_class_label": (
                None if majority is None else self.class_labels[majority]
            ),
            "tie_break": "lowest_ontology_index",
        }


@dataclass(slots=True)
class _MultilabelCounts:
    class_labels: tuple[str, ...]
    positive_counts: list[int] = field(init=False)
    row_count: int = 0

    def __post_init__(self) -> None:
        self.positive_counts = [0] * len(self.class_labels)

    def add(self, values: Tensor) -> None:
        positives = values.detach().to(
            dtype=torch.long, device="cpu"
        ).sum(dim=0)
        for index, count in enumerate(positives.tolist()):
            self.positive_counts[index] += int(count)
        self.row_count += int(values.shape[0])

    def artifact(self) -> dict[str, Any]:
        prevalence = (
            [count / self.row_count for count in self.positive_counts]
            if self.row_count
            else [None] * len(self.positive_counts)
        )
        return {
            "kind": "closed_multilabel",
            "class_labels": list(self.class_labels),
            "train_eligible_row_count": self.row_count,
            "positive_counts": list(self.positive_counts),
            "negative_counts": [
                self.row_count - count for count in self.positive_counts
            ],
            "prevalence": prevalence,
            "majority_threshold": 0.5,
            "majority_prediction": (
                [value >= 0.5 for value in prevalence]
                if self.row_count
                else [None] * len(prevalence)
            ),
        }


class TrainPriorBuilder:
    """Stream only train rows into dataset/task-specific priors."""

    def __init__(
        self,
        *,
        bindings: dict[str, object],
        split: str = "train",
    ) -> None:
        if split != "train":
            raise EvaluationContractError(
                "evaluation.priors.non_train_split_forbidden"
            )
        self.bindings = bindings
        self.split = split
        self._counts: dict[
            tuple[str, str], _CategoricalCounts | _MultilabelCounts
        ] = {}
        self.sample_count = 0
        self.batch_count = 0

    @property
    def retained_target_tensor_count(self) -> int:
        return 0

    def add_batch(self, batch: MultiSourceBatch) -> None:
        if not isinstance(batch, MultiSourceBatch):
            raise EvaluationContractError(
                "evaluation.priors.batch_type_invalid"
            )
        self.sample_count += len(batch.dataset_ids)
        self.batch_count += 1
        for target in batch.target_batches:
            if (
                target.task_id not in ACTIVE_TASK_IDS
                or target.supervision_regime != "fully_supervised"
                or not target.model_ready
            ):
                continue
            eligibility = (
                target.availability_mask & target.entity_index_mask
            )
            for sample_index, dataset_id in enumerate(batch.dataset_ids):
                rows = torch.nonzero(
                    eligibility
                    & (target.sample_indices == sample_index),
                    as_tuple=False,
                ).flatten()
                if rows.numel() == 0:
                    continue
                values = target.values.index_select(0, rows)
                self.add_rows(
                    dataset_id=dataset_id,
                    task_id=target.task_id,
                    values=values,
                )

    def add_rows(
        self,
        *,
        dataset_id: str,
        task_id: str,
        values: Tensor,
    ) -> None:
        if task_id not in ACTIVE_TASK_IDS:
            raise EvaluationContractError(
                f"evaluation.priors.task_inactive:{task_id}"
            )
        encoding = TARGET_ENCODING_BY_TASK[task_id]
        labels = tuple(encoding.vocabulary or ())
        key = (dataset_id, task_id)
        counts = self._counts.get(key)
        if counts is None:
            counts = (
                _CategoricalCounts(labels)
                if encoding.encoding_kind == "closed_categorical_index"
                else _MultilabelCounts(labels)
            )
            self._counts[key] = counts
        if encoding.encoding_kind == "closed_categorical_index":
            if not isinstance(counts, _CategoricalCounts):
                raise EvaluationContractError(
                    "evaluation.priors.encoding_changed"
                )
            counts.add(values)
        elif encoding.encoding_kind == "closed_multilabel":
            if not isinstance(counts, _MultilabelCounts):
                raise EvaluationContractError(
                    "evaluation.priors.encoding_changed"
                )
            counts.add(values)
        else:
            raise EvaluationContractError(
                "evaluation.priors.open_vocabulary_forbidden"
            )

    def finalize(self) -> dict[str, Any]:
        tasks: dict[str, dict[str, Any]] = {}
        for (dataset_id, task_id), counts in sorted(self._counts.items()):
            tasks.setdefault(dataset_id, {})[task_id] = counts.artifact()
        payload: dict[str, Any] = {
            "train_prior_contract_version": TRAIN_PRIOR_CONTRACT_VERSION,
            "source_split": self.split,
            "bindings": self.bindings,
            "train_sample_count": self.sample_count,
            "train_batch_count": self.batch_count,
            "datasets": tasks,
            "retained_target_tensor_count": 0,
        }
        payload["train_prior_fingerprint"] = canonical_fingerprint(payload)
        return payload


def validate_train_priors(
    artifact: dict[str, Any],
    *,
    expected_bindings: dict[str, object] | None = None,
) -> None:
    if artifact.get("train_prior_contract_version") != (
        TRAIN_PRIOR_CONTRACT_VERSION
    ):
        raise EvaluationContractError(
            "evaluation.priors.version_mismatch"
        )
    if artifact.get("source_split") != "train":
        raise EvaluationContractError(
            "evaluation.priors.source_split_invalid"
        )
    if expected_bindings is not None and artifact.get(
        "bindings"
    ) != expected_bindings:
        raise EvaluationContractError(
            "evaluation.priors.binding_mismatch"
        )
    supplied = artifact.get("train_prior_fingerprint")
    payload = dict(artifact)
    payload.pop("train_prior_fingerprint", None)
    if supplied != canonical_fingerprint(payload):
        raise EvaluationContractError(
            "evaluation.priors.fingerprint_mismatch"
        )
    for dataset_id, tasks in artifact.get("datasets", {}).items():
        if not isinstance(dataset_id, str) or not isinstance(tasks, dict):
            raise EvaluationContractError(
                "evaluation.priors.dataset_payload_invalid"
            )
        for task_id, prior in tasks.items():
            if task_id not in ACTIVE_TASK_IDS or not isinstance(prior, dict):
                raise EvaluationContractError(
                    "evaluation.priors.task_payload_invalid"
                )
            rows = prior.get("train_eligible_row_count")
            if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
                raise EvaluationContractError(
                    "evaluation.priors.empty_task_invalid"
                )
            encoding = TARGET_ENCODING_BY_TASK[task_id]
            expected_labels = list(encoding.vocabulary or ())
            expected_kind = (
                "closed_categorical"
                if encoding.encoding_kind
                == "closed_categorical_index"
                else "closed_multilabel"
            )
            if (
                prior.get("kind") != expected_kind
                or prior.get("class_labels") != expected_labels
            ):
                raise EvaluationContractError(
                    "evaluation.priors.task_contract_mismatch"
                )
            probabilities = (
                prior.get("empirical_probabilities")
                if prior.get("kind") == "closed_categorical"
                else prior.get("prevalence")
            )
            if (
                not isinstance(probabilities, list)
                or not probabilities
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    or value > 1
                    for value in probabilities
                )
            ):
                raise EvaluationContractError(
                    "evaluation.priors.probabilities_invalid"
                )
            if len(probabilities) != len(expected_labels):
                raise EvaluationContractError(
                    "evaluation.priors.probability_width_invalid"
                )
            if expected_kind == "closed_categorical":
                counts = prior.get("class_counts")
                if (
                    not isinstance(counts, list)
                    or len(counts) != len(expected_labels)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in counts
                    )
                ):
                    raise EvaluationContractError(
                        "evaluation.priors.categorical_counts_invalid"
                    )
                expected_majority = max(
                    range(len(expected_labels)),
                    key=lambda index: (counts[index], -index),
                )
                if (
                    sum(counts) != rows
                    or probabilities
                    != [value / rows for value in counts]
                    or prior.get("majority_class_index")
                    != expected_majority
                ):
                    raise EvaluationContractError(
                        "evaluation.priors.categorical_counts_invalid"
                    )
            else:
                positives = prior.get("positive_counts")
                negatives = prior.get("negative_counts")
                if (
                    not isinstance(positives, list)
                    or not isinstance(negatives, list)
                    or len(positives) != len(expected_labels)
                    or len(negatives) != len(expected_labels)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        or value > rows
                        for value in positives
                    )
                    or negatives
                    != [rows - value for value in positives]
                    or probabilities
                    != [value / rows for value in positives]
                    or prior.get("majority_threshold") != 0.5
                    or prior.get("majority_prediction")
                    != [value >= 0.5 for value in probabilities]
                ):
                    raise EvaluationContractError(
                        "evaluation.priors.multilabel_counts_invalid"
                    )


class TrivialBaselineAccumulator:
    """Evaluate fixed train priors without consulting held-out labels early."""

    def __init__(self, prior: dict[str, Any]) -> None:
        self.prior = prior
        self.kind = str(prior["kind"])
        self.class_labels = tuple(prior["class_labels"])
        encoding = (
            "closed_categorical_index"
            if self.kind == "closed_categorical"
            else "closed_multilabel"
        )
        self.classification = make_metric_accumulator(
            encoding, self.class_labels
        )
        self.empirical_nll_sum = Fraction()
        self.empirical_nll_denominator = 0
        self.impossible_observation_count = 0

    @property
    def retained_prediction_tensor_count(self) -> int:
        return 0

    def update(self, targets: Tensor) -> None:
        if self.kind == "closed_categorical":
            probabilities = tuple(
                float(value)
                for value in self.prior["empirical_probabilities"]
            )
            scores = torch.tensor(
                probabilities, dtype=torch.float64
            ).repeat(targets.shape[0], 1)
            self.classification.update(scores, targets.to("cpu"))
            for target in targets.detach().to("cpu").tolist():
                probability = probabilities[int(target)]
                if probability == 0:
                    self.impossible_observation_count += 1
                else:
                    self.empirical_nll_sum += Fraction.from_float(
                        -math.log(probability)
                    )
                self.empirical_nll_denominator += 1
            return
        prevalence = tuple(float(value) for value in self.prior["prevalence"])
        majority = torch.tensor(
            self.prior["majority_prediction"], dtype=torch.bool
        )
        # Sign alone determines the fixed threshold prediction. Magnitude is
        # irrelevant because BCE is replaced with the exact prevalence NLL.
        logits = torch.where(
            majority,
            torch.ones_like(majority, dtype=torch.float64),
            -torch.ones_like(majority, dtype=torch.float64),
        ).repeat(targets.shape[0], 1)
        cpu_targets = targets.detach().to("cpu")
        self.classification.update(logits, cpu_targets)
        for row in cpu_targets.tolist():
            for truth, probability in zip(row, prevalence, strict=True):
                if (truth and probability == 0) or (
                    not truth and probability == 1
                ):
                    self.impossible_observation_count += 1
                else:
                    loss = (
                        -math.log(probability)
                        if truth
                        else -math.log1p(-probability)
                    )
                    self.empirical_nll_sum += Fraction.from_float(loss)
                self.empirical_nll_denominator += 1

    def finalize(self) -> dict[str, Any]:
        result = self.classification.finalize()
        likelihood = (
            metric_value(
                None,
                category="zero_train_probability",
                reason=(
                    f"{self.impossible_observation_count} held-out "
                    "observations have zero empirical train probability"
                ),
            )
            if self.impossible_observation_count
            else (
                metric_value(
                    float(self.empirical_nll_sum)
                    / self.empirical_nll_denominator
                )
                if self.empirical_nll_denominator
                else metric_value(
                    None,
                    category="no_eligible_rows",
                    reason="no eligible held-out rows were observed",
                )
            )
        )
        if self.kind == "closed_categorical":
            result["nll"] = likelihood
            result["baseline_predictor"] = "train_majority_class"
            result["likelihood_predictor"] = "train_empirical_prior"
        else:
            result["bce_nll"] = likelihood
            result["baseline_predictor"] = (
                "train_per_class_majority_at_0.5"
            )
            result["likelihood_predictor"] = (
                "train_per_class_prevalence"
            )
        result["impossible_observation_count"] = (
            self.impossible_observation_count
        )
        result["train_prior_fingerprint_scope"] = (
            "dataset_task_train_rows_only"
        )
        return result


__all__ = [
    "TrainPriorBuilder",
    "TrivialBaselineAccumulator",
    "validate_train_priors",
]
