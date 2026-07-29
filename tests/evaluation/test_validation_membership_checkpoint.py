from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from music_critic.evaluation.config import (
    EvaluationConfig,
    EvaluationDataConfig,
    EvaluationDeviceConfig,
)
from music_critic.evaluation.contracts import EvaluationContractError
from music_critic.evaluation.engine import run_evaluation
from music_critic.models import LocalBaselineConfig, LocalHeterogeneousBaseline
from music_critic.tasks import (
    CanonicalCorpusInput,
    CorpusCacheConfig,
    cache_canonical_corpus,
    create_split_manifest,
    dump_corpus_index,
    dump_split_manifest,
)
from music_critic.training.checkpoint import save_training_checkpoint
from music_critic.training.config import DataConfig
from music_critic.training.data import _hook_piece, build_data_runtime


@dataclass(frozen=True, slots=True)
class _Fixture:
    indices: tuple[object, ...]
    index_paths: tuple[Path, ...]
    cache_roots: tuple[Path, ...]
    manifest_path: Path
    assignments: dict[tuple[str, str], str]


def _index(root: Path, dataset_id: str):
    cache = CorpusCacheConfig(root / f"{dataset_id}-cache")
    inputs = []
    for ordinal in range(5):
        piece = replace(
            _hook_piece(f"{dataset_id}-{ordinal}", ordinal + 1),
            dataset_name=dataset_id,
        )
        inputs.append(
            CanonicalCorpusInput(
                piece=piece,
                lineage_group_id=piece.source_group_id,
                source_identity=piece.piece_id,
                source_relative_path=f"{piece.piece_id}.json",
                source_sha256=sha256(
                    f"{dataset_id}:{ordinal}".encode()
                ).hexdigest(),
                suggested_split=None,
            )
        )
    index, _report = cache_canonical_corpus(
        inputs,
        cache_config=cache,
        dataset_id=dataset_id,
        adapter_name="phase6d_membership_test",
        adapter_version="1.0.0",
        adapter_config={},
        source_identity=f"{dataset_id}-fixture",
        source_fingerprint=sha256(dataset_id.encode()).hexdigest(),
        creation_policy="phase6d_membership_test",
    )
    path = root / f"{dataset_id}.index.json"
    dump_corpus_index(index, path)
    return index, path, cache.root


def _fixture(root: Path) -> _Fixture:
    hook, hook_path, hook_cache = _index(root, "hooktheory")
    pop, pop_path, pop_cache = _index(root, "pop909_cl")
    assignments = {
        (record.dataset_id, record.piece_id): (
            "train"
            if ordinal == 0
            else "validation"
            if ordinal < 4
            else "test"
        )
        for index in (hook, pop)
        for ordinal, record in enumerate(index.records)
    }
    manifest = create_split_manifest(
        (hook, pop),
        assignments,
        seed=42,
        policy="phase6d_membership_fixture",
    )
    manifest_path = root / "global.split.json"
    dump_split_manifest(manifest, manifest_path)
    return _Fixture(
        indices=(hook, pop),
        index_paths=(hook_path, pop_path),
        cache_roots=(hook_cache, pop_cache),
        manifest_path=manifest_path,
        assignments=assignments,
    )


def _training_checkpoint(root: Path, fixture: _Fixture) -> Path:
    runtime = build_data_runtime(
        DataConfig(
            name="mixed",
            index_paths=[str(path) for path in fixture.index_paths],
            cache_roots=[str(path) for path in fixture.cache_roots],
            split_manifest=str(fixture.manifest_path),
            batch_size=1,
            workers=0,
            epoch_size=2,
            validation_epoch_size=2,
            mixture_weights={"hooktheory": 1.0, "pop909_cl": 1.0},
        ),
        seed=42,
    )
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(
            variant="feature_only",
            hidden_dim=8,
            gnn_layers=0,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = root / "phase6c-training.pt"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scheduler=None,
        scaler=scaler,
        next_epoch=0,
        best_validation_loss=None,
        committed_metric_rows=0,
        resolved_config={"fixture": "phase6c_membership"},
        data_fingerprints=runtime.fingerprints,
    )
    return path


def _evaluation_config(
    fixture: _Fixture,
    checkpoint: Path,
    output: Path,
    *,
    seed: int = 42,
    max_evaluation_samples: int = 2,
    name: str = "mixed",
    index_paths: tuple[Path, ...] | None = None,
    cache_roots: tuple[Path, ...] | None = None,
    manifest_path: Path | None = None,
) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint=str(checkpoint),
        output_dir=str(output),
        seed=seed,
        data=EvaluationDataConfig(
            name=name,
            index_paths=[
                str(path)
                for path in (index_paths or fixture.index_paths)
            ],
            cache_roots=[
                str(path)
                for path in (cache_roots or fixture.cache_roots)
            ],
            split_manifest=str(manifest_path or fixture.manifest_path),
            batch_size=1,
            workers=0,
            max_train_samples=1,
            max_evaluation_samples=max_evaluation_samples,
        ),
        device=EvaluationDeviceConfig(),
    )


def test_phase6c_checkpoint_accepts_identical_phase6d_membership(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = _training_checkpoint(tmp_path, fixture)
    checkpoint_before = checkpoint.read_bytes()
    first = tmp_path / "evaluation-first"
    second = tmp_path / "evaluation-second"

    report = run_evaluation(
        _evaluation_config(fixture, checkpoint, first)
    )
    run_evaluation(_evaluation_config(fixture, checkpoint, second))

    assert report["data_verification"]["verified"] is True
    assert "validation_membership_fingerprint" in report[
        "data_verification"
    ]["matched_fields"]
    assert checkpoint.read_bytes() == checkpoint_before
    for name in (
        "checkpoint_evidence.json",
        "train_priors.json",
        "metrics.json",
        "evaluation_report.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize(
    ("seed", "limit"),
    [(43, 2), (42, 3)],
)
def test_phase6c_checkpoint_rejects_membership_seed_or_limit_change(
    tmp_path: Path,
    seed: int,
    limit: int,
) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = _training_checkpoint(tmp_path, fixture)

    with pytest.raises(
        EvaluationContractError,
        match=(
            "evaluation.checkpoint.data_binding_mismatch:"
            "validation_membership_fingerprint"
        ),
    ):
        run_evaluation(
            _evaluation_config(
                fixture,
                checkpoint,
                tmp_path / f"negative-{seed}-{limit}",
                seed=seed,
                max_evaluation_samples=limit,
            )
        )


def test_phase6c_checkpoint_still_rejects_split_binding_change(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = _training_checkpoint(tmp_path, fixture)
    changed = dict(fixture.assignments)
    moved = next(
        key for key, split in sorted(changed.items()) if split == "validation"
    )
    changed[moved] = "test"
    manifest = create_split_manifest(
        fixture.indices,
        changed,
        seed=42,
        policy="phase6d_membership_changed_fixture",
    )
    path = tmp_path / "changed.split.json"
    dump_split_manifest(manifest, path)

    with pytest.raises(
        EvaluationContractError,
        match="evaluation.checkpoint.data_binding_mismatch:",
    ):
        run_evaluation(
            _evaluation_config(
                fixture,
                checkpoint,
                tmp_path / "changed-split",
                manifest_path=path,
            )
        )


def test_phase6c_checkpoint_still_rejects_index_and_composition_change(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    checkpoint = _training_checkpoint(tmp_path, fixture)

    with pytest.raises(
        EvaluationContractError,
        match="evaluation.checkpoint.data_binding_mismatch:",
    ):
        run_evaluation(
            _evaluation_config(
                fixture,
                checkpoint,
                tmp_path / "single-dataset",
                name="hooktheory",
                index_paths=(fixture.index_paths[0],),
                cache_roots=(fixture.cache_roots[0],),
            )
        )
