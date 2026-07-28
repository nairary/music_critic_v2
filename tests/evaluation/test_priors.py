from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from music_critic.models import (
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    join_task_supervision,
)
from music_critic.evaluation.priors import (
    TrainPriorBuilder,
    validate_train_priors,
)
from music_critic.tasks import (
    collate_multisource_samples,
    prepare_multisource_sample,
)
from music_critic.training.data import _hook_piece


def _build_prior(
    validation_or_test_labels: torch.Tensor,
) -> dict[str, object]:
    # The argument intentionally never enters the builder. It represents
    # arbitrary held-out-label state available to an evaluation caller.
    assert validation_or_test_labels.ndim == 1
    builder = TrainPriorBuilder(
        bindings={
            "index_fingerprints": [["fixture", "index-v1"]],
            "split_manifest_fingerprint": "split-v1",
            "ontology_fingerprint": "ontology-v1",
            "encoding_fingerprint": "encoding-v1",
            "train_membership_fingerprint": "train-v1",
        }
    )
    builder.add_rows(
        dataset_id="hooktheory",
        task_id="theory.chord.extent",
        values=torch.tensor([0, 0, 2, 0], dtype=torch.long),
    )
    builder.add_rows(
        dataset_id="hooktheory",
        task_id="theory.chord.adds",
        values=torch.tensor(
            [
                [True, False, False],
                [False, False, False],
            ],
            dtype=torch.bool,
        ),
    )
    return builder.finalize()


def test_validation_and_test_label_mutation_does_not_change_train_priors() -> None:
    original = _build_prior(torch.tensor([0, 1, 2]))
    mutated = _build_prior(torch.tensor([9, 9, 9]))
    assert original == mutated
    assert (
        original["train_prior_fingerprint"]
        == mutated["train_prior_fingerprint"]
    )


def test_categorical_majority_and_multilabel_prevalence_are_train_only() -> None:
    artifact = _build_prior(torch.tensor([0]))
    extent = artifact["datasets"]["hooktheory"]["theory.chord.extent"]
    adds = artifact["datasets"]["hooktheory"]["theory.chord.adds"]

    assert extent["class_counts"][0] == 3
    assert extent["class_counts"][2] == 1
    assert extent["majority_class_index"] == 0
    assert extent["empirical_probabilities"][0] == 0.75
    assert adds["prevalence"][0] == 0.5
    assert adds["majority_prediction"][0] is True
    assert all(value is False for value in adds["majority_prediction"][1:])
    validate_train_priors(artifact)


def test_prior_fingerprint_rejects_mutation() -> None:
    artifact = _build_prior(torch.tensor([0]))
    mutated = deepcopy(artifact)
    mutated["datasets"]["hooktheory"]["theory.chord.extent"][
        "class_counts"
    ][0] += 1
    try:
        validate_train_priors(mutated)
    except ValueError as exc:
        assert "fingerprint_mismatch" in str(exc)
    else:
        raise AssertionError("mutated train priors must fail validation")


def test_masked_unavailable_rows_enter_neither_join_nor_priors() -> None:
    task_id = "theory.chord.extent"
    piece = _hook_piece("phase6d-masked", 1)
    piece = replace(
        piece,
        targets=tuple(
            replace(
                target,
                values=(None,),
                mask=(False,),
                confidence=(None,),
                source=(None,),
                provenance=(None,),
            )
            if target.task == task_id
            else target
            for target in piece.targets
        ),
    )
    batch = collate_multisource_samples(
        (prepare_multisource_sample(piece),)
    )
    target = next(
        item for item in batch.target_batches if item.task_id == task_id
    )
    assert target.entry_count == 1
    assert target.availability_mask.tolist() == [False]

    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(
            variant="feature_only",
            hidden_dim=8,
            gnn_layers=0,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        predictions = model.predict(batch.raw_graph_batch)[1]
        supervisions = join_task_supervision(
            predictions, batch.target_batches
        )
    assert task_id not in {item.task_id for item in supervisions}

    builder = TrainPriorBuilder(bindings={"fixture": "masked"})
    builder.add_batch(batch)
    artifact = builder.finalize()
    assert task_id not in artifact["datasets"]["hooktheory"]
