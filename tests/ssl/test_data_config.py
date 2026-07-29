from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path

from hydra import compose, initialize
import pytest
import torch
from torch import Tensor

from music_critic.graph import graph_fingerprint, model_input_fingerprint
from music_critic.models import HierarchicalBaselineConfig
from music_critic.ssl.config import register_ssl_configs
from music_critic.ssl.data import (
    IndexedSSLRawDataset,
    SSLBatch,
    SSLRawSample,
    build_ssl_data_runtime,
    collate_ssl_samples,
    move_ssl_batch,
    strip_multisource_batch,
    validate_ssl_batch,
)
from music_critic.ssl.masking import build_mask_plans_for_batch
from music_critic.ssl.model import (
    MaskedGraphSSLConfig,
    MaskedGraphSSLModel,
)
from music_critic.ssl.views import build_feature_mask_overlay
from music_critic.tasks import (
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
)
from music_critic.training.config import DataConfig
from music_critic.training.data import build_data_runtime
from tests.training.test_split_and_corpus import _build_index


@pytest.fixture(scope="module")
def phase6_bounded_batch():
    return build_data_runtime(DataConfig(), seed=42).first_train_batch


def _source_graphs(batch: SSLBatch):
    graphs = batch.raw_graph_batch.to_data_list()
    for graph in graphs:
        if isinstance(graph.raw_only, Tensor):
            graph.raw_only = bool(graph.raw_only.item())
    return tuple(graphs)


def _source_graph_fingerprints(batch: SSLBatch) -> tuple[str, ...]:
    return tuple(
        graph_fingerprint(graph)
        for graph in _source_graphs(batch)
    )


def _plans(batch: SSLBatch, *, stage: str, epoch: int):
    return build_mask_plans_for_batch(
        batch,
        global_seed=42,
        epoch=epoch,
        encoder_view_index=0,
        requested_mask_rate=0.30,
        stage=stage,
    )


def test_ssl_batch_is_raw_only_and_target_sidecar_mutation_is_inert(
    phase6_bounded_batch,
) -> None:
    phase6_batch = deepcopy(phase6_bounded_batch)
    before = strip_multisource_batch(phase6_batch)
    plans_before = _plans(before, stage="train", epoch=3)
    overlay_before = build_feature_mask_overlay(
        before.raw_graph_batch,
        plans_before,
    )
    graph_before = _source_graph_fingerprints(before)

    for target in phase6_batch.target_batches:
        for name in (
            "values",
            "availability_mask",
            "entity_indices",
            "entity_index_mask",
            "entity_node_type_codes",
            "sample_indices",
            "confidence",
            "confidence_mask",
        ):
            value = getattr(target, name)
            if not isinstance(value, Tensor) or value.numel() == 0:
                continue
            if value.dtype == torch.bool:
                value.logical_not_()
            elif value.is_floating_point():
                value.add_(17.0)
            else:
                value.add_(17)
        object.__setattr__(target, "provenance_cpu", ())
    object.__setattr__(phase6_batch, "diagnostics_cpu", ())

    after = strip_multisource_batch(phase6_batch)
    plans_after = _plans(after, stage="train", epoch=3)
    overlay_after = build_feature_mask_overlay(
        after.raw_graph_batch,
        plans_after,
    )

    assert tuple(item.name for item in fields(SSLBatch)) == (
        "raw_graph_batch",
        "dataset_ids",
        "piece_ids",
        "sample_count",
        "node_count",
        "edge_count",
    )
    for forbidden in (
        "target_batches",
        "targets",
        "annotations",
        "provenance",
        "diagnostics_cpu",
        "source_group_ids",
        "lineage_group_ids",
        "statistics",
    ):
        assert not hasattr(after, forbidden)
    assert after.raw_graph_batch is before.raw_graph_batch
    assert after.dataset_ids == before.dataset_ids
    assert after.piece_ids == before.piece_ids
    assert _source_graph_fingerprints(after) == graph_before
    assert plans_after == plans_before
    assert overlay_after.fingerprint == overlay_before.fingerprint


def test_bounded_runtime_preserves_group_safe_fixed_validation() -> None:
    runtime = build_ssl_data_runtime(
        DataConfig(
            name="bounded",
            batch_size=2,
            epoch_size=6,
            validation_epoch_size=2,
        ),
        seed=42,
    )
    train = tuple(runtime.train_loader(0))
    validation_first = tuple(runtime.validation_loader())
    validation_second = tuple(runtime.validation_loader())
    train_identities = {
        identity
        for batch in train
        for identity in zip(batch.dataset_ids, batch.piece_ids, strict=True)
    }
    validation_identities = tuple(
        identity
        for batch in validation_first
        for identity in zip(batch.dataset_ids, batch.piece_ids, strict=True)
    )
    repeated_identities = tuple(
        identity
        for batch in validation_second
        for identity in zip(batch.dataset_ids, batch.piece_ids, strict=True)
    )

    assert train_identities.isdisjoint(validation_identities)
    assert validation_identities == runtime.validation_membership.identities
    assert repeated_identities == validation_identities
    assert (
        runtime.fingerprints["validation_membership_fingerprint"]
        == runtime.validation_membership.membership_fingerprint
    )
    first_plans = tuple(
        plan
        for batch in validation_first
        for plan in _plans(batch, stage="validation", epoch=0)
    )
    later_plans = tuple(
        plan
        for batch in validation_second
        for plan in _plans(batch, stage="validation", epoch=99)
    )
    assert later_plans == first_plans


def test_ssl_config_refreshes_process_global_hydra_groups() -> None:
    from music_critic.evaluation.config import register_evaluation_configs

    register_evaluation_configs()
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(
            config_name="ssl_training",
            overrides=[
                "experiment=one_batch",
                "model=hierarchical",
                "data=bounded",
                "device=cpu",
            ],
        )

    assert config.data.name == "bounded"
    assert dict(config.data.mixture_weights) == {
        "hooktheory": 1.0,
        "pop909_cl": 1.0,
    }
    assert config.device.name == "cpu"


def test_ssl_runtime_never_invokes_target_alignment_or_tensorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("SSL raw-only collation accessed target logic")

    monkeypatch.setattr(
        "music_critic.tasks.collator.align_sample_targets",
        forbidden,
    )
    monkeypatch.setattr(
        "music_critic.tasks.collator.tensorize_aligned_targets",
        forbidden,
    )
    runtime = build_ssl_data_runtime(
        DataConfig(
            name="bounded",
            batch_size=2,
            epoch_size=3,
            validation_epoch_size=2,
        ),
        seed=42,
    )

    assert isinstance(runtime.first_train_batch, SSLBatch)
    assert tuple(runtime.train_loader(0))
    assert tuple(runtime.validation_loader())


def test_indexed_ssl_dataset_never_projects_supervised_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _, cache = _build_index(
        tmp_path / "raw-only-cache",
        "raw-only",
        1,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("SSL dataset projected supervised targets")

    monkeypatch.setattr(
        "music_critic.tasks.multisource.project_multisource_targets",
        forbidden,
    )
    dataset = IndexedSSLRawDataset(index, cache_config=cache)
    sample = dataset[0]
    batch = collate_ssl_samples((sample,))

    assert isinstance(sample, SSLRawSample)
    assert not hasattr(sample, "target_bundle")
    assert not hasattr(sample, "annotations")
    assert batch.dataset_ids == ("raw-only",)
    assert batch.piece_ids == (sample.piece_id,)
    assert batch.sample_count == 1


@pytest.mark.parametrize(
    ("experiment", "data", "expected_epochs"),
    (
        ("one_batch", "bounded", 1),
        ("pretrain", "mixed", 20),
    ),
)
def test_exact_phase7a_hydra_composition(
    experiment: str,
    data: str,
    expected_epochs: int,
) -> None:
    register_ssl_configs()
    with initialize(version_base="1.3", config_path=None):
        config = compose(
            config_name="ssl_training",
            overrides=[
                f"experiment={experiment}",
                "model=hierarchical",
                f"data={data}",
                "device=cpu",
            ],
        )

    assert config.experiment.name == experiment
    assert config.experiment.epochs == expected_epochs
    assert config.model.name == "hierarchical"
    assert config.data.name == data
    assert config.device.name == "cpu"
    assert config.ssl.mask_rate == 0.30
    assert config.ssl.decoder_views == 3
    assert config.ssl.decoder_remask_prob == 0.20
    assert config.ssl.note_weight == 1.0
    assert config.ssl.bar_weight == 1.0
    assert config.ssl.song_weight == 1.0
    assert config.ssl.epsilon == 1e-8
    assert config.ssl.projector_hidden_dim == 128
    assert config.ssl.decoder_hidden_dim == 128


def test_ssl_device_transfer_is_non_mutating_and_tensor_only(
    phase6_bounded_batch,
) -> None:
    source = strip_multisource_batch(phase6_bounded_batch)
    validate_ssl_batch(source)
    source_fingerprints = _source_graph_fingerprints(source)
    source_tensors = {
        (store_index, key): value
        for store_index, store in enumerate(source.raw_graph_batch.stores)
        for key, value in store.items()
        if isinstance(value, Tensor)
    }

    moved = move_ssl_batch(source, "cpu")
    validate_ssl_batch(moved)

    assert moved is not source
    assert moved.raw_graph_batch is not source.raw_graph_batch
    assert moved.dataset_ids is source.dataset_ids
    assert moved.piece_ids is source.piece_ids
    assert _source_graph_fingerprints(source) == source_fingerprints
    assert _source_graph_fingerprints(moved) == source_fingerprints
    assert (
        moved.raw_graph_batch["song"].entity_id
        == source.raw_graph_batch["song"].entity_id
    )
    for store_index, store in enumerate(moved.raw_graph_batch.stores):
        for key, value in store.items():
            if not isinstance(value, Tensor):
                continue
            original = source_tensors[(store_index, key)]
            assert value.device.type == "cpu"
            assert value.shape == original.shape
            assert value.dtype == original.dtype
            assert torch.equal(value, original)
            if value.numel():
                assert value.data_ptr() != original.data_ptr()


def _worker_parity_config(tmp_path: Path) -> DataConfig:
    alpha, _, alpha_cache = _build_index(
        tmp_path / "alpha-cache",
        "alpha",
        4,
    )
    beta, _, beta_cache = _build_index(
        tmp_path / "beta-cache",
        "beta",
        4,
    )
    alpha_path = tmp_path / "alpha.index.json"
    beta_path = tmp_path / "beta.index.json"
    split_path = tmp_path / "global.split.json"
    dump_corpus_index(alpha, alpha_path)
    dump_corpus_index(beta, beta_path)
    assignments = {
        (record.dataset_id, record.piece_id): (
            "train" if index < 2 else "validation"
        )
        for corpus in (alpha, beta)
        for index, record in enumerate(corpus.records)
    }
    manifest = create_split_manifest(
        (alpha, beta),
        assignments,
        seed=42,
        policy="phase7a_worker_parity_test",
        policy_config={"purpose": "worker_parity"},
    )
    dump_split_manifest(manifest, split_path)
    return DataConfig(
        name="mixed",
        index_paths=[str(alpha_path), str(beta_path)],
        cache_roots=[
            str(alpha_cache.root),
            str(beta_cache.root),
        ],
        split_manifest=str(split_path),
        batch_size=2,
        workers=0,
        epoch_size=4,
        validation_epoch_size=4,
        mixture_weights={"alpha": 1.0, "beta": 1.0},
    )


def _loader_evidence(
    batches,
    *,
    model: MaskedGraphSSLModel,
    stage: str,
    epoch: int,
) -> tuple[tuple[tuple[str, str], ...], dict[tuple[str, str], object]]:
    order: list[tuple[str, str]] = []
    evidence: dict[tuple[str, str], object] = {}
    for batch in batches:
        with torch.no_grad():
            output = model(
                batch,
                global_seed=42,
                epoch=epoch,
                validation=stage == "validation",
            )
        plans = output.mask_plans
        graphs = _source_graphs(batch)
        cursor = 0
        for sample_index, (
            dataset_id,
            piece_id,
            graph,
            plan,
        ) in enumerate(zip(
            batch.dataset_ids,
            batch.piece_ids,
            graphs,
            plans,
            strict=True,
        )):
            identity = (dataset_id, piece_id)
            assert identity not in evidence
            order.append(identity)
            row_count = len(plan.selected_local_node_indices)
            compact = slice(cursor, cursor + row_count)
            global_indices = output.selected_global_note_indices[
                compact
            ]
            online_rows = output.online_encoder.fused.embeddings[
                "note"
            ].index_select(0, global_indices)
            target_rows = output.targets.note.index_select(
                0, global_indices
            )
            evidence[identity] = (
                graph_fingerprint(graph),
                model_input_fingerprint(graph),
                plan.fingerprint,
                tuple(
                    (
                        view[sample_index].stable_seed,
                        view[sample_index].fingerprint,
                        view[sample_index].remasked_positions,
                    )
                    for view in output.decoder_remask_plans
                ),
                online_rows.detach().cpu().tolist(),
                target_rows.detach().cpu().tolist(),
                tuple(
                    prediction[compact].detach().cpu().tolist()
                    for prediction in output.decoder_predictions
                ),
            )
            cursor += row_count
        assert cursor == int(
            output.selected_global_note_indices.shape[0]
        )
    return tuple(order), evidence


def test_workers_zero_and_two_preserve_per_identity_inputs_and_plans(
    tmp_path: Path,
) -> None:
    base = _worker_parity_config(tmp_path)
    runtime_zero = build_ssl_data_runtime(
        replace(base, workers=0),
        seed=42,
    )
    runtime_two = build_ssl_data_runtime(
        replace(base, workers=2),
        seed=42,
    )
    torch.manual_seed(99)
    model = MaskedGraphSSLModel(
        HierarchicalBaselineConfig(
            hidden_dim=8,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=2,
            ffn_multiplier=2,
            dropout=0.0,
        ),
        MaskedGraphSSLConfig(
            decoder_hidden_dim=8,
            projector_hidden_dim=8,
        ),
    ).eval()

    train_zero = _loader_evidence(
        runtime_zero.train_loader(3),
        model=model,
        stage="train",
        epoch=3,
    )
    train_two = _loader_evidence(
        runtime_two.train_loader(3),
        model=model,
        stage="train",
        epoch=3,
    )
    validation_zero = _loader_evidence(
        runtime_zero.validation_loader(),
        model=model,
        stage="validation",
        epoch=0,
    )
    validation_two = _loader_evidence(
        runtime_two.validation_loader(),
        model=model,
        stage="validation",
        epoch=57,
    )

    assert train_two == train_zero
    assert validation_two == validation_zero
    assert (
        runtime_two.validation_membership
        == runtime_zero.validation_membership
    )
    assert runtime_two.fingerprints == runtime_zero.fingerprints
