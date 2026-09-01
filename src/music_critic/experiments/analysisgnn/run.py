"""Command line entrypoint for Phase 9E-B1 evidence and benchmark runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import types
from typing import Any

import torch

from music_critic.experiments.analysisgnn.attestation import (
    attest_historical_directory,
)
from music_critic.experiments.analysisgnn.contracts import (
    COMMON_BENCHMARK_CONFIG,
    EDGE_TYPES,
    NODE_TYPES,
    canonical_json,
    graph_schema_fingerprint,
)
from music_critic.experiments.analysisgnn.dataset import (
    CommonDatasetManifest,
    CommonDatasetRecord,
    load_common_manifest,
    load_common_record,
    prepare_common_dataset,
)
from music_critic.experiments.analysisgnn.graph import (
    build_analysisgnn_graph,
    graph_fingerprint,
)
from music_critic.experiments.analysisgnn.forensics import (
    compact_forensic_evidence,
    label_binding_forensic_audit,
)
from music_critic.experiments.analysisgnn.metrics import summarize_seeds
from music_critic.experiments.analysisgnn.model import AnalysisGNNCommonModel
from music_critic.experiments.analysisgnn.optimization import (
    TwoTaskUncertaintyLoss,
)
from music_critic.experiments.analysisgnn.preflight import (
    label_binding_preflight,
    require_zero_conflicting_overlaps,
    source_row_bindings_from_report,
    validate_label_binding_preflight,
)
from music_critic.experiments.analysisgnn.training import (
    create_test_unlock,
    evaluate_seed_test,
    require_deterministic_cuda_environment,
    train_seed,
)


def _dependency(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_report() -> dict[str, object]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__, "gpus": []}
    rows = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            columns = [value.strip() for value in line.split(",")]
            if len(columns) == 6:
                rows.append(
                    dict(
                        zip(
                            (
                                "index",
                                "uuid",
                                "name",
                                "driver_version",
                                "memory_mib",
                                "compute_capability",
                            ),
                            columns,
                            strict=True,
                        )
                    )
                )
    return {
        "available": completed.returncode == 0,
        "error": completed.stderr.strip() or None,
        "gpus": rows,
    }


def environment_report() -> dict[str, object]:
    return {
        "claim": "runtime_environment_observation",
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn": torch.backends.cudnn.version(),
        "dependencies": {
            name: _dependency(name)
            for name in (
                "graphmuse",
                "music21",
                "numpy",
                "partitura",
                "pytorch-lightning",
                "torch",
                "torch-geometric",
            )
        },
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi": _nvidia_smi_report(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _synthetic_graph(device: torch.device) -> object:
    from torch_geometric.data import HeteroData

    graph = HeteroData()
    counts = {"note": 8, "measure": 2, "beat": 4}
    generator = torch.Generator().manual_seed(9_001)
    for node_type, count in counts.items():
        graph[node_type].x = torch.rand((count, 25), generator=generator)
    graph["note"].pitch_spelling = torch.arange(8) % 35
    graph["note"].key_signature = torch.arange(8) % 15
    note_index = torch.arange(8)
    graph["note", "onset", "note"].edge_index = torch.stack((note_index, note_index))
    for name in ("consecutive", "during", "rest"):
        graph["note", name, "note"].edge_index = torch.stack((note_index[:-1], note_index[1:]))
        graph["note", f"{name}_rev", "note"].edge_index = torch.stack((note_index[1:], note_index[:-1]))
    measure = torch.arange(8) // 4
    beat = torch.arange(8) // 2
    graph["note", "connects", "measure"].edge_index = torch.stack((note_index, measure))
    graph["measure", "connects", "note"].edge_index = torch.stack((measure, note_index))
    graph["measure", "next", "measure"].edge_index = torch.tensor([[0], [1]])
    graph["note", "connects", "beat"].edge_index = torch.stack((note_index, beat))
    graph["beat", "connects", "note"].edge_index = torch.stack((beat, note_index))
    graph["beat", "next", "beat"].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    graph["note"].quality = torch.tensor([0, 1, 2, 3, -1, 4, 5, 6])
    graph["note"].inversion = torch.tensor([0, 1, 2, 3, -1, 0, 1, 2])
    if set(graph.edge_types) != set(EDGE_TYPES) or set(graph.node_types) != set(NODE_TYPES):
        raise AssertionError("synthetic smoke graph schema is incomplete")
    return graph.to(device)


def smoke(device_name: str, *, allow_model_only_stub: bool = False) -> dict[str, object]:
    device = torch.device(device_name)
    require_deterministic_cuda_environment(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    substitution = None
    if allow_model_only_stub:
        # This permits model-only CPU evidence on build-tool-free hosts. It does
        # not replace the compiled sampler for graph preparation or real runs.
        sys.modules.setdefault(
            "graphmuse.samplers.csamplers", types.ModuleType("graphmuse.samplers.csamplers")
        )
        partitura = types.ModuleType("partitura")
        partitura.__path__ = []  # type: ignore[attr-defined]
        partitura_score = types.ModuleType("partitura.score")
        partitura_score.Measure = type("Measure", (), {})  # type: ignore[attr-defined]
        partitura.score = partitura_score  # type: ignore[attr-defined]
        sys.modules.setdefault("partitura", partitura)
        sys.modules.setdefault("partitura.score", partitura_score)
        graph_utils = types.ModuleType("graphmuse.utils.graph_utils")
        graph_utils.edges_from_note_array = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        graph_utils.create_random_music_graph = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        graph_utils.trim_to_layer = lambda **kwargs: (  # type: ignore[attr-defined]
            kwargs["x"], kwargs["edge_index"], kwargs.get("edge_attr")
        )
        sys.modules.setdefault("graphmuse.utils.graph_utils", graph_utils)
        torch_scatter = types.ModuleType("torch_scatter")
        torch_scatter.scatter_add = lambda source, index, dim=0, **_kwargs: (  # type: ignore[attr-defined]
            torch.zeros(
                (int(index.max()) + 1, *source.shape[1:]),
                dtype=source.dtype,
                device=source.device,
            ).index_add_(dim, index, source)
        )
        sys.modules.setdefault("torch_scatter", torch_scatter)
        substitution = "model_only_empty_csamplers_import_stub"
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = AnalysisGNNCommonModel().to(device)
    objective = TwoTaskUncertaintyLoss().to(device)
    graph = _synthetic_graph(device)
    model.train()
    logits = model(graph)  # type: ignore[arg-type]
    loss, task_losses = objective(
        logits,
        {task: graph["note"][task] for task in ("quality", "inversion")},  # type: ignore[index]
    )
    if loss is None:
        raise AssertionError("smoke graph unexpectedly has no labels")
    loss.backward()
    return {
        "claim": "synthetic_model_only_diagnostic_not_graph_acceptance",
        "architecture": model.architecture_manifest(),
        "device": str(device),
        "environment": environment_report(),
        "finite_gradients": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in (*model.parameters(), *objective.parameters())
        ),
        "loss": float(loss.detach().cpu()),
        "logit_shapes": {task: list(value.shape) for task, value in logits.items()},
        "schema_fingerprint": graph_schema_fingerprint(),
        "substitution": substitution,
        "task_losses": {
            task: float(value.detach().cpu()) for task, value in task_losses.items()
        },
    }


def _select_real_smoke_record(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
) -> tuple[CommonDatasetRecord, Any, Any, Any]:
    """Select the largest TRAIN piece with deterministic identity tie-breaking."""

    selected: tuple[CommonDatasetRecord, Any, Any, Any] | None = None
    selected_key: tuple[int, str, str] | None = None
    train_rows = sorted(
        (row for row in manifest.records if row.split == "train"),
        key=lambda row: (row.record_id, row.piece_id),
    )
    if not train_rows:
        raise ValueError("real graph smoke requires at least one TRAIN record")
    for row in train_rows:
        piece, targets, projection = load_common_record(cache_root, row)
        key = (-len(piece.notes), row.record_id, row.piece_id)
        if selected_key is None or key < selected_key:
            selected_key = key
            selected = (row, piece, targets, projection)
    if selected is None or selected[0].split != "train":
        raise AssertionError("real graph smoke selection escaped the TRAIN split")
    return selected


def real_graph_smoke(
    manifest: CommonDatasetManifest,
    cache_root: str | Path,
    *,
    device_name: str,
    label_binding_preflight_path: str | Path,
) -> dict[str, object]:
    """Run one real GraphMuse TRAIN-graph forward/backward without an optimizer."""

    device = torch.device(device_name)
    require_deterministic_cuda_environment(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA real graph smoke requested but CUDA is unavailable")
    preflight_report = validate_label_binding_preflight(
        label_binding_preflight_path, manifest
    )
    source_row_bindings = source_row_bindings_from_report(preflight_report)
    row, piece, targets, projection = _select_real_smoke_record(manifest, cache_root)
    if row.split != "train":
        raise AssertionError("validation/test records are forbidden in real graph smoke")
    graph, entries = build_analysisgnn_graph(
        piece,
        targets,
        projection,
        transposition="P1",
        record_id=row.record_id,
        dialect=row.dialect,
        source_row_binding=source_row_bindings.get(row.record_id),
    )
    digest = graph_fingerprint(graph)
    node_counts = {
        node_type: int(graph[node_type].num_nodes) for node_type in graph.node_types
    }
    edge_counts = {
        "|".join(edge_type): int(graph[edge_type].edge_index.shape[1])
        for edge_type in graph.edge_types
    }
    note_count = node_counts.get("note", 0)
    expected_shapes = {"quality": [note_count, 50], "inversion": [note_count, 4]}

    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    model = AnalysisGNNCommonModel().to(device)
    objective = TwoTaskUncertaintyLoss().to(device)
    graph = graph.to(device)
    model.train()
    logits = model(graph)  # type: ignore[arg-type]
    logit_shapes = {task: list(value.shape) for task, value in logits.items()}
    if logit_shapes != expected_shapes:
        raise AssertionError(
            f"real graph smoke logits differ: {logit_shapes} != {expected_shapes}"
        )
    if not all(bool(torch.isfinite(value).all()) for value in logits.values()):
        raise FloatingPointError("real graph smoke produced non-finite logits")
    loss, task_losses = objective(
        logits,
        {task: graph["note"][task] for task in ("quality", "inversion")},
    )
    if loss is None or not bool(torch.isfinite(loss)):
        raise FloatingPointError("real graph smoke has no finite supervised loss")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in (*model.parameters(), *objective.parameters())
        if parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    if not finite_gradients:
        raise FloatingPointError("real graph smoke produced absent or non-finite gradients")
    return {
        "acceptance": True,
        "architecture": model.architecture_manifest(),
        "claim": "real_pinned_graphmuse_train_graph_smoke_without_optimizer_step",
        "device": str(device),
        "edge_counts": edge_counts,
        "environment": environment_report(),
        "finite_gradients": finite_gradients,
        "graph_entry_count": len(entries),
        "graph_sha256": digest,
        "logit_shapes": logit_shapes,
        "loss": float(loss.detach().cpu()),
        "node_counts": node_counts,
        "optimizer_step": False,
        "piece_id": row.piece_id,
        "record_id": row.record_id,
        "selected_note_count": len(piece.notes),
        "selection": "maximum_note_count_then_record_id_then_piece_id_over_train_only",
        "source_group_id": row.source_group_id,
        "split": row.split,
        "task_losses": {
            task: float(value.detach().cpu()) for task, value in task_losses.items()
        },
        "transposition": "P1",
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    attest = commands.add_parser("attest")
    attest.add_argument("--evidence-root", type=Path, required=True)
    attest.add_argument("--output", type=Path)
    environment = commands.add_parser("environment")
    environment.add_argument("--output", type=Path)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--corpus-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    preflight = commands.add_parser("label-binding-preflight")
    preflight.add_argument("--cache-root", type=Path, required=True)
    preflight.add_argument("--corpus-root", type=Path, required=True)
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    forensics = commands.add_parser("label-binding-forensics")
    forensics.add_argument("--cache-root", type=Path, required=True)
    forensics.add_argument("--corpus-root", type=Path, required=True)
    forensics.add_argument("--manifest", type=Path, required=True)
    forensics.add_argument("--preflight", type=Path, required=True)
    forensics.add_argument("--output", type=Path, required=True)
    forensics.add_argument("--compact-output", type=Path)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    smoke_parser.add_argument("--allow-model-only-stub", action="store_true")
    smoke_parser.add_argument("--output", type=Path)
    real_smoke = commands.add_parser("real-graph-smoke")
    real_smoke.add_argument("--cache-root", type=Path, required=True)
    real_smoke.add_argument("--manifest", type=Path, required=True)
    real_smoke.add_argument("--device", choices=("cpu", "cuda"), required=True)
    real_smoke.add_argument("--label-binding-preflight", type=Path, required=True)
    real_smoke.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=(17, 23, 42), required=True)
    train.add_argument("--device", choices=("cuda",), default="cuda")
    train.add_argument("--dependency-lock", type=Path, required=True)
    train.add_argument("--label-binding-preflight", type=Path, required=True)
    unlock = commands.add_parser("unlock-test")
    unlock.add_argument("--manifest", type=Path, required=True)
    unlock.add_argument("--runs-root", type=Path, required=True)
    unlock.add_argument("--output", type=Path, required=True)
    unlock.add_argument("--authorized-by", required=True)
    unlock.add_argument("--authorize-locked-test", action="store_true", required=True)
    evaluate = commands.add_parser("evaluate-test")
    evaluate.add_argument("--cache-root", type=Path, required=True)
    evaluate.add_argument("--corpus-root", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--runs-root", type=Path, required=True)
    evaluate.add_argument("--unlock", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, choices=(17, 23, 42), required=True)
    evaluate.add_argument("--device", choices=("cuda",), default="cuda")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--runs-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "attest":
        value = attest_historical_directory(args.evidence_root)
        result = {**asdict(value), "attestation_fingerprint": value.fingerprint}
        if args.output:
            _write(args.output, result)
    elif args.command == "environment":
        result = environment_report()
        if args.output:
            _write(args.output, result)
    elif args.command == "prepare-data":
        manifest = prepare_common_dataset(args.corpus_root, args.output_root)
        result = {
            key: value
            for key, value in asdict(manifest).items()
            if key != "records"
        }
    elif args.command == "label-binding-preflight":
        manifest = load_common_manifest(args.manifest)
        result = label_binding_preflight(
            manifest, args.cache_root, corpus_root=args.corpus_root
        )
        _write(args.output, result)
        require_zero_conflicting_overlaps(result)
    elif args.command == "label-binding-forensics":
        manifest = load_common_manifest(args.manifest)
        result = label_binding_forensic_audit(
            manifest,
            args.cache_root,
            args.preflight,
            corpus_root=args.corpus_root,
        )
        _write(args.output, result)
        if args.compact_output:
            _write(args.compact_output, compact_forensic_evidence(result))
    elif args.command == "smoke":
        result = smoke(args.device, allow_model_only_stub=args.allow_model_only_stub)
        if args.output:
            _write(args.output, result)
    elif args.command == "real-graph-smoke":
        manifest = load_common_manifest(args.manifest)
        result = real_graph_smoke(
            manifest,
            args.cache_root,
            device_name=args.device,
            label_binding_preflight_path=args.label_binding_preflight,
        )
        _write(args.output, result)
    elif args.command == "train":
        require_deterministic_cuda_environment(args.device)
        manifest = load_common_manifest(args.manifest)
        result = train_seed(
            manifest,
            args.cache_root,
            args.output_root,
            seed=args.seed,
            device=args.device,
            dependency_lock=args.dependency_lock,
            label_binding_preflight=args.label_binding_preflight,
            runtime_environment=environment_report(),
        )
    elif args.command == "unlock-test":
        manifest = load_common_manifest(args.manifest)
        result = create_test_unlock(
            manifest,
            args.runs_root,
            args.output,
            authorized_by=args.authorized_by,
        )
    elif args.command == "evaluate-test":
        manifest = load_common_manifest(args.manifest)
        result = evaluate_seed_test(
            manifest,
            args.cache_root,
            args.runs_root,
            args.unlock,
            args.corpus_root,
            seed=args.seed,
            device=args.device,
        )
    else:
        per_seed = {
            seed: json.loads(
                (args.runs_root / f"seed-{seed}" / "result.json").read_text(
                    encoding="utf-8"
                )
            )["test"]
            for seed in COMMON_BENCHMARK_CONFIG.seeds
        }
        result = {
            "claim": "three_seed_common_subset_summary",
            "config_fingerprint": COMMON_BENCHMARK_CONFIG.config_fingerprint,
            "per_seed": per_seed,
            "summary": summarize_seeds(per_seed),
        }
        _write(args.output, result)
    print(canonical_json(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
