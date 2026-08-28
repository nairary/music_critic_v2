"""From-scratch common-arm runner with applied-update accounting."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Iterable

import torch

from music_critic.experiments.analysisgnn.contracts import (
    ANALYSISGNN_COMMIT,
    COMMON_BENCHMARK_CONFIG,
    DILEMMADATA_COMMIT,
    GRAPHMUSE_COMMIT,
    Phase9EB1Config,
    TRANSPOSITIONS,
    canonical_json,
    fingerprint,
)
from music_critic.experiments.analysisgnn.dataset import (
    CommonDatasetManifest,
    CommonDatasetRecord,
    load_common_record,
)
from music_critic.experiments.analysisgnn.graph import (
    GraphEntry,
    build_analysisgnn_graph,
    graph_fingerprint,
)
from music_critic.experiments.analysisgnn.metrics import (
    EntryPrediction,
    aggregate_entry_predictions,
    benchmark_metrics,
    grouped_bootstrap,
)
from music_critic.experiments.analysisgnn.model import AnalysisGNNCommonModel
from music_critic.experiments.analysisgnn.optimization import (
    TwoTaskUncertaintyLoss,
    apply_update_learning_rate,
    configure_optimizer,
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _append(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _graph(
    cache_root: Path,
    row: CommonDatasetRecord,
    transposition: str,
    device: torch.device,
) -> tuple[object, tuple[GraphEntry, ...], str]:
    piece, targets, projection = load_common_record(cache_root, row)
    graph, entries = build_analysisgnn_graph(
        piece, targets, projection, transposition=transposition
    )
    digest = graph_fingerprint(graph)
    return graph.to(device), entries, digest


@torch.no_grad()
def _evaluate_source_split(
    model: AnalysisGNNCommonModel,
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    *,
    split: str,
    device: torch.device,
) -> tuple[dict[str, object], tuple[EntryPrediction, ...], tuple[dict[str, str], ...]]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation may only open validation or test")
    model.eval()
    predictions: list[EntryPrediction] = []
    graph_rows: list[dict[str, str]] = []
    for row in (item for item in manifest.records if item.split == split):
        graph, entries, graph_digest = _graph(Path(cache_root), row, "P1", device)
        logits = model(graph)  # type: ignore[arg-type]
        graph_rows.append(
            {"piece_id": row.piece_id, "transposition": "P1", "sha256": graph_digest}
        )
        for task in ("quality", "inversion"):
            task_entries = tuple(entry for entry in entries if entry.task == task)
            predictions.extend(
                aggregate_entry_predictions(
                    record_id=row.record_id,
                    piece_id=row.piece_id,
                    split=split,
                    task=task,
                    note_logits=logits[task],
                    note_targets=graph["note"][task],  # type: ignore[index]
                    note_entry_index=graph["note"][f"{task}_entry_index"],  # type: ignore[index]
                    entity_ids=tuple(entry.entity_id for entry in task_entries),
                    entry_masks=tuple(entry.mask for entry in task_entries),
                )
            )
    return benchmark_metrics(predictions), tuple(predictions), tuple(graph_rows)


def evaluate_validation(
    model: AnalysisGNNCommonModel,
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    *,
    device: torch.device,
) -> tuple[dict[str, object], tuple[EntryPrediction, ...], tuple[dict[str, str], ...]]:
    """Expose validation scoring without a public test-split switch."""

    return _evaluate_source_split(
        model, manifest, cache_root, split="validation", device=device
    )


def validation_objective(metrics: dict[str, object]) -> float:
    quality = float(metrics["quality"]["nll"]) / math.log(50)  # type: ignore[index]
    inversion = float(metrics["inversion"]["nll"]) / math.log(4)  # type: ignore[index]
    return 0.5 * (quality + inversion)


def _checkpoint(
    path: Path,
    *,
    model: AnalysisGNNCommonModel,
    objective: TwoTaskUncertaintyLoss,
    optimizer: torch.optim.Optimizer,
    seed: int,
    applied_update: int,
    validation_score: float,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "applied_update": applied_update,
            "config": asdict(model.config),
            "config_fingerprint": model.config.config_fingerprint,
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": seed,
            "validation_score": validation_score,
        },
        temporary,
    )
    temporary.replace(path)
    return _sha256(path)


def _prediction_payload(row: EntryPrediction) -> dict[str, object]:
    value = asdict(row)
    value["logits"] = [number if math.isfinite(number) else None for number in row.logits]
    return value


def train_seed(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "cuda",
    dependency_lock: str | Path,
    runtime_environment: dict[str, object],
    config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG,
) -> dict[str, object]:
    """Train one preregistered seed and leave locked test unopened."""

    if seed not in config.seeds:
        raise ValueError("seed is outside the frozen three-seed protocol")
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 9E-B1 training requires the remote CUDA gate")
    _seed(seed)
    output = Path(output_root) / f"seed-{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = Path(dependency_lock)
    if not lock_path.is_file():
        raise FileNotFoundError(f"resolved dependency lock is absent: {lock_path}")
    _write(output / "config.json", canonical_json(asdict(config), indent=2) + "\n")
    _write(
        output / "provenance.json",
        canonical_json(
            {
                "analysisgnn_commit": ANALYSISGNN_COMMIT,
                "dependency_lock": {
                    "bytes": lock_path.stat().st_size,
                    "name": lock_path.name,
                    "sha256": _sha256(lock_path),
                },
                "dilemmadata_commit": DILEMMADATA_COMMIT,
                "graphmuse_commit": GRAPHMUSE_COMMIT,
                "runtime_environment": runtime_environment,
            },
            indent=2,
        )
        + "\n",
    )
    _write(
        output / "data_binding.json",
        canonical_json(
            {
                "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
                "source_split_fingerprint": manifest.source_split_fingerprint,
                "train_transpositions": list(TRANSPOSITIONS),
            },
            indent=2,
        )
        + "\n",
    )
    model = AnalysisGNNCommonModel(config).to(runtime_device)
    objective = TwoTaskUncertaintyLoss(config).to(runtime_device)
    optimizer = configure_optimizer(model, objective, config)
    _write(
        output / "architecture.json",
        canonical_json(model.architecture_manifest(), indent=2) + "\n",
    )
    train_views = [
        (row, transposition)
        for row in manifest.records
        if row.split == "train"
        for transposition in TRANSPOSITIONS
    ]
    if len(train_views) != 577 * 12:
        raise RuntimeError("training view count differs from 577 x 12")
    schedule_rng = random.Random(seed)
    applied_update = 0
    candidate_index = 0
    skipped_update = 0
    best_score = float("inf")
    best_digest = ""
    best_path = output / "checkpoints" / "best.pt"
    graph_digests: dict[tuple[str, str], str] = {}
    while applied_update < config.applied_update_budget:
        if candidate_index % len(train_views) == 0:
            schedule_rng.shuffle(train_views)
        row, transposition = train_views[candidate_index % len(train_views)]
        candidate_index += 1
        graph, _entries, graph_digest = _graph(
            Path(cache_root), row, transposition, runtime_device
        )
        graph_digests[(row.piece_id, transposition)] = graph_digest
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph)  # type: ignore[arg-type]
        total, task_losses = objective(
            logits,
            {task: graph["note"][task] for task in ("quality", "inversion")},  # type: ignore[index]
        )
        if total is None:
            skipped_update += 1
            _append(
                output / "batch_schedule.jsonl",
                {
                    "applied_update_after": applied_update,
                    "candidate_index": candidate_index,
                    "graph_sha256": graph_digest,
                    "outcome": "skipped_no_supervised_target",
                    "piece_id": row.piece_id,
                    "record_id": row.record_id,
                    "source_group_id": row.source_group_id,
                    "transposition": transposition,
                },
            )
            continue
        next_update = applied_update + 1
        learning_rate = apply_update_learning_rate(optimizer, next_update, config)
        total.backward()
        optimizer.step()
        applied_update = next_update
        _append(
            output / "batch_schedule.jsonl",
            {
                "applied_update_after": applied_update,
                "candidate_index": candidate_index,
                "graph_sha256": graph_digest,
                "outcome": "applied",
                "piece_id": row.piece_id,
                "record_id": row.record_id,
                "source_group_id": row.source_group_id,
                "transposition": transposition,
            },
        )
        _append(
            output / "training.jsonl",
            {
                "applied_update": applied_update,
                "candidate_index": candidate_index,
                "graph_sha256": graph_digest,
                "learning_rate": learning_rate,
                "loss": float(total.detach().cpu()),
                "piece_id": row.piece_id,
                "task_losses": {
                    task: float(loss.detach().cpu()) for task, loss in task_losses.items()
                },
                "transposition": transposition,
                "uncertainty_scales": [
                    float(value) for value in objective.scales.detach().cpu().tolist()
                ],
            },
        )
        if applied_update % config.validation_every_applied_updates == 0:
            validation, _rows, validation_graphs = evaluate_validation(
                model, manifest, cache_root, device=runtime_device
            )
            score = validation_objective(validation)
            _append(
                output / "validation.jsonl",
                {
                    "applied_update": applied_update,
                    "metrics": validation,
                    "selection_score": score,
                },
            )
            for graph_row in validation_graphs:
                graph_digests[(graph_row["piece_id"], "P1")] = graph_row["sha256"]
            if score < best_score:
                best_score = score
                best_digest = _checkpoint(
                    best_path,
                    model=model,
                    objective=objective,
                    optimizer=optimizer,
                    seed=seed,
                    applied_update=applied_update,
                    validation_score=score,
                )
    graph_rows = [
        {"piece_id": piece_id, "transposition": transposition, "sha256": digest}
        for (piece_id, transposition), digest in sorted(graph_digests.items())
    ]
    _write(
        output / "graph_fingerprints.json",
        canonical_json(
            {"fingerprint": fingerprint(graph_rows), "graphs": graph_rows}, indent=2
        )
        + "\n",
    )
    result: dict[str, object] = {
        "applied_update_budget": applied_update,
        "candidate_update_count": candidate_index,
        "best_checkpoint": {"path": str(best_path), "sha256": best_digest},
        "best_validation_score": best_score,
        "claim": "analysisgnn_common_subset_reconstruction_not_exact_official_reproduction",
        "config_fingerprint": config.config_fingerprint,
        "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
        "seed": seed,
        "skipped_update_count": skipped_update,
        "status": "validation_selected_test_locked",
        "test": None,
    }
    _write(output / "result.json", canonical_json(result, indent=2) + "\n")
    files = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write(
        output / "artifact_manifest.json",
        canonical_json(
            {"files": files, "files_fingerprint": fingerprint(files)}, indent=2
        )
        + "\n",
    )
    return result


def create_test_unlock(
    manifest: CommonDatasetManifest,
    runs_root: str | Path,
    output: str | Path,
    *,
    authorized_by: str,
) -> dict[str, object]:
    """Bind all three selected checkpoints before any test graph is opened."""

    if not authorized_by.strip():
        raise ValueError("locked-test authorization requires an operator identity")
    runs_root = Path(runs_root)
    checkpoints: dict[str, dict[str, object]] = {}
    for seed in COMMON_BENCHMARK_CONFIG.seeds:
        result = json.loads(
            (runs_root / f"seed-{seed}" / "result.json").read_text(encoding="utf-8")
        )
        if (
            result.get("status") != "validation_selected_test_locked"
            or result.get("test") is not None
            or result.get("config_fingerprint") != COMMON_BENCHMARK_CONFIG.config_fingerprint
            or result.get("dataset_manifest_fingerprint") != manifest.manifest_fingerprint
        ):
            raise ValueError(f"seed {seed} is not a locked validation-selected run")
        checkpoint = Path(result["best_checkpoint"]["path"])
        digest = _sha256(checkpoint)
        if digest != result["best_checkpoint"]["sha256"]:
            raise ValueError(f"seed {seed} checkpoint digest changed")
        checkpoints[str(seed)] = {
            "applied_update": int(result["applied_update_budget"]),
            "path": str(checkpoint),
            "sha256": digest,
            "validation_score": float(result["best_validation_score"]),
        }
    payload: dict[str, object] = {
        "authorized": True,
        "authorized_by": authorized_by.strip(),
        "checkpoints": checkpoints,
        "config_fingerprint": COMMON_BENCHMARK_CONFIG.config_fingerprint,
        "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
        "protocol": "phase9eb1-locked-test-unlock-v1",
        "seeds": list(COMMON_BENCHMARK_CONFIG.seeds),
    }
    payload["unlock_fingerprint"] = fingerprint(payload)
    _write(Path(output), canonical_json(payload, indent=2) + "\n")
    return payload


def evaluate_seed_test(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    runs_root: str | Path,
    unlock_path: str | Path,
    *,
    seed: int,
    device: str = "cuda",
    config: Phase9EB1Config = COMMON_BENCHMARK_CONFIG,
) -> dict[str, object]:
    """Open locked test only after a hash-bound three-checkpoint unlock."""

    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("locked-test evaluation requires CUDA")
    unlock = json.loads(Path(unlock_path).read_text(encoding="utf-8"))
    unlock_fingerprint = unlock.pop("unlock_fingerprint", None)
    if (
        unlock_fingerprint != fingerprint(unlock)
        or unlock.get("authorized") is not True
        or unlock.get("protocol") != "phase9eb1-locked-test-unlock-v1"
        or unlock.get("seeds") != list(config.seeds)
        or unlock.get("config_fingerprint") != config.config_fingerprint
        or unlock.get("dataset_manifest_fingerprint") != manifest.manifest_fingerprint
    ):
        raise ValueError("locked-test unlock is absent, malformed, or misbound")
    if seed not in config.seeds:
        raise ValueError("seed is outside the frozen protocol")
    output = Path(runs_root) / f"seed-{seed}"
    result_path = output / "result.json"
    train_result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(unlock["checkpoints"][str(seed)]["path"])
    checkpoint_digest = _sha256(checkpoint_path)
    if (
        checkpoint_digest != unlock["checkpoints"][str(seed)]["sha256"]
        or checkpoint_digest != train_result["best_checkpoint"]["sha256"]
    ):
        raise ValueError("selected checkpoint differs from locked-test binding")
    if (output / "test_entry_predictions.jsonl").exists():
        raise FileExistsError("refusing to overwrite locked-test predictions")
    _seed(seed)
    model = AnalysisGNNCommonModel(config).to(runtime_device)
    checkpoint = torch.load(checkpoint_path, map_location=runtime_device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics, prediction_rows, test_graphs = _evaluate_source_split(
        model, manifest, cache_root, split="test", device=runtime_device
    )
    bootstrap = grouped_bootstrap(
        prediction_rows, samples=config.bootstrap_samples, seed=seed
    )
    for row in prediction_rows:
        _append(output / "test_entry_predictions.jsonl", _prediction_payload(row))
    test_result: dict[str, object] = {
        "bootstrap_95_percent": bootstrap,
        "checkpoint_sha256": checkpoint_digest,
        "claim": "locked_test_evaluation_of_reconstructed_common_arm",
        "config_fingerprint": config.config_fingerprint,
        "dataset_manifest_fingerprint": manifest.manifest_fingerprint,
        "graph_fingerprint": fingerprint(test_graphs),
        "graphs": list(test_graphs),
        "metrics": test_metrics,
        "seed": seed,
        "unlock_fingerprint": unlock_fingerprint,
    }
    _write(output / "test_result.json", canonical_json(test_result, indent=2) + "\n")
    train_result["status"] = "locked_test_evaluated"
    train_result["test"] = test_metrics
    train_result["test_result_sha256"] = _sha256(output / "test_result.json")
    train_result["unlock_fingerprint"] = unlock_fingerprint
    _write(result_path, canonical_json(train_result, indent=2) + "\n")
    files = {
        str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write(
        output / "artifact_manifest.json",
        canonical_json({"files": files, "files_fingerprint": fingerprint(files)}, indent=2)
        + "\n",
    )
    return test_result


__all__ = [
    "create_test_unlock",
    "evaluate_seed_test",
    "evaluate_validation",
    "train_seed",
    "validation_objective",
]
