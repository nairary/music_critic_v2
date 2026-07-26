from __future__ import annotations

from types import SimpleNamespace

import torch

from music_critic.tasks import collate_multisource_samples
from music_critic.training.config import DataConfig
from music_critic.training.data import _bounded_samples, build_data_runtime
from music_critic.training.metrics import EpochMetricAccumulator


def _identities(loader) -> tuple[tuple[str, str], ...]:
    return tuple(
        (dataset_id, piece_id)
        for batch in loader
        for dataset_id, piece_id in zip(
            batch.dataset_ids,
            batch.piece_ids,
            strict=True,
        )
    )


def test_default_validation_visits_full_view_once_with_fixed_membership(
) -> None:
    runtime = build_data_runtime(
        DataConfig(batch_size=2, validation_epoch_size=0),
        seed=73,
    )
    first = _identities(runtime.validation_loader())
    second = _identities(runtime.validation_loader())
    evidence = runtime.validation_membership
    assert first == second == evidence.identities
    assert len(first) == len(set(first)) == evidence.full_view_count
    assert evidence.selected_count == evidence.full_view_count == 3
    assert evidence.subset_limit == 0
    assert evidence.dataset_counts == {
        "hooktheory": 2,
        "pop909_cl": 1,
    }


def test_bounded_validation_subset_is_selected_once_without_replacement(
) -> None:
    first = build_data_runtime(
        DataConfig(batch_size=1, validation_epoch_size=2),
        seed=73,
    )
    second = build_data_runtime(
        DataConfig(batch_size=3, validation_epoch_size=2),
        seed=73,
    )
    left = _identities(first.validation_loader())
    right = _identities(second.validation_loader())
    assert left == right
    assert len(left) == len(set(left)) == 2
    assert (
        first.validation_membership.membership_fingerprint
        == second.validation_membership.membership_fingerprint
    )
    assert (
        first.fingerprints["validation_membership_fingerprint"]
        == second.fingerprints["validation_membership_fingerprint"]
    )


def _row_values(dataset_id: str, piece_id: str):
    if dataset_id == "pop909_cl":
        return (1.0,), 0.25
    if piece_id.endswith("bounded-validation-hook"):
        return (0.25, 0.5), 0.125
    return (), 0.5


def _synthetic_output(batch):
    task_losses = []
    sample_indices = []
    reconstruction_by_sample = []
    for sample_index, (dataset_id, piece_id) in enumerate(
        zip(batch.dataset_ids, batch.piece_ids, strict=True)
    ):
        rows, reconstruction = _row_values(dataset_id, piece_id)
        task_losses.extend(rows)
        sample_indices.extend([sample_index] * len(rows))
        reconstruction_by_sample.append(reconstruction)
    supervision = SimpleNamespace(
        task_id="theory.chord.extent",
        per_row_loss=torch.tensor(task_losses, dtype=torch.float32),
        sample_indices=torch.tensor(sample_indices, dtype=torch.long),
    )
    song_batch = batch.raw_graph_batch["song"].batch
    per_node = torch.tensor(
        [
            reconstruction_by_sample[int(sample_index)]
            for sample_index in song_batch
        ],
        dtype=torch.float32,
    )
    reconstruction = SimpleNamespace(
        node_type="song",
        feature_name="duration_qn",
        per_node_loss=per_node,
        availability_mask=torch.ones_like(per_node, dtype=torch.bool),
    )
    return SimpleNamespace(
        supervisions=(supervision,),
        reconstruction=(reconstruction,),
    )


def _aggregate(groups):
    accumulator = EpochMetricAccumulator(
        harmonic_weight=2.0,
        reconstruction_weight=0.5,
        task_weights={"theory.chord.extent": 3.0},
    )
    for samples in groups:
        batch = collate_multisource_samples(samples)
        accumulator.add(_synthetic_output(batch), batch)
    return accumulator.finalize()


def test_epoch_metrics_and_best_scalar_ignore_batch_size_and_order() -> None:
    _, validation = _bounded_samples()
    together = _aggregate((validation,))
    split = _aggregate(
        ((validation[0],), (validation[1], validation[2]))
    )
    reversed_order = _aggregate(
        ((validation[2],), (validation[1],), (validation[0],))
    )
    for result in (split, reversed_order):
        assert result["tasks"] == together["tasks"]
        assert result["reconstruction"] == together["reconstruction"]
        assert result["harmonic_loss"] == together["harmonic_loss"]
        assert result["reconstruction_loss"] == together[
            "reconstruction_loss"
        ]
        assert result["objective_loss"] == together["objective_loss"]
        assert result["per_dataset"] == together["per_dataset"]
    task = together["tasks"]["theory.chord.extent"]
    assert task == {
        "loss_numerator": 1.75,
        "eligible_row_count": 3,
        "mean_loss": 1.75 / 3,
    }
    assert together["dataset_counts"] == {
        "hooktheory": 2,
        "pop909_cl": 1,
    }
    assert set(together["per_dataset"]) == {
        "hooktheory",
        "pop909_cl",
    }
    assert together["hot_path_instrumentation"] == {
        "gradient_evidence_scans": 0,
        "per_parameter_host_syncs": 0,
        "per_task_host_syncs": 0,
        "per_family_host_syncs": 0,
        "epoch_finalize_device_to_host_syncs": 1,
    }
