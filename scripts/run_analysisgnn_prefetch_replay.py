#!/usr/bin/env python3
"""Registered ordered CPU prefetch comparison; guarded spawn-safe entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path


def main(argv=None):
    # Imports define classes only. No GPU/model is initialized at module import,
    # including when multiprocessing spawn imports this script as __mp_main__.
    from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
    from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchConfig
    from music_critic.experiments.analysisgnn.performance_prefetch_replay import (
        build_prefetch_schedule, load_production_prefetch_inputs, protocol_bindings, run_prefetch_replay,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--parent-experiment-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "production"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--measured-draws", type=int, default=800)
    parser.add_argument("--warmup-draws", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--preparation-strategy", choices=("frozen", "compact_cache"), default="frozen")
    parser.add_argument("--compact-cache-global-max-bytes", type=int,
                        help="One aggregate retained-entry budget, divided between active preparing processes")
    parser.add_argument("--include-original-baseline", action="store_true",
                        help="Append one original preparation/reference-update arm, sharing the same initial state and draws")
    parser.add_argument("--worker-config", type=Path,
                        help="JSON containing every PrefetchConfig field, workers=0; mandatory production")
    parser.add_argument("--schedule-descriptor", type=Path)
    parser.add_argument("--source-bindings", type=Path)
    parser.add_argument("--reservation-manifest", type=Path)
    parser.add_argument("--records-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/experiments/records")
    parser.add_argument("--fixture-reserved-bytes", type=int, default=2_000_000)
    for field in fields(ProductionArtifactPaths):
        parser.add_argument("--" + field.name.replace("_", "-"), type=Path)
    args = parser.parse_args(argv)
    paths = production_input_paths = None
    binding_manifest = reservation_manifest = None
    if args.worker_config is not None:
        if args.worker_config.is_symlink():
            parser.error("worker config must be a regular file")
        config_payload = json.loads(args.worker_config.read_bytes())
        if set(config_payload) != set(asdict(PrefetchConfig())):
            parser.error("worker config must explicitly contain every PrefetchConfig field")
        config = PrefetchConfig(**config_payload)
    else:
        config = PrefetchConfig()
    if args.mode == "production":
        if args.device != "cuda" or args.worker_config is None:
            parser.error("production requires CUDA and an explicit worker config")
        values = {field.name: getattr(args, field.name) for field in fields(ProductionArtifactPaths)}
        if any(value is None for value in values.values()):
            parser.error("production requires every explicit ProductionArtifactPaths path")
        if any(value is None for value in (args.schedule_descriptor, args.source_bindings, args.reservation_manifest)):
            parser.error("production requires completed sizing artifacts: schedule, source bindings, reservation manifest")
        paths = ProductionArtifactPaths(**values)
        production_input_paths = {"schedule_path": args.schedule_descriptor, "binding_path": args.source_bindings,
            "reservation_path": args.reservation_manifest, "records_root": args.records_root}
        inputs = load_production_prefetch_inputs(**production_input_paths)
        descriptor, binding_manifest, reservation_manifest = (inputs[name] for name in
            ("descriptor", "binding_manifest", "reservation_manifest"))
    else:
        if any(value is not None for value in (args.schedule_descriptor, args.source_bindings, args.reservation_manifest)):
            parser.error("production source artifacts cannot be supplied in fixture mode")
        _, descriptor = build_prefetch_schedule(mode="fixture", measured_draws=args.measured_draws,
            warmup_draws=args.warmup_draws, reserved_bytes=args.fixture_reserved_bytes)
    bindings = protocol_bindings(mode=args.mode, descriptor=descriptor, reservation_manifest=reservation_manifest,
                                config=config, worker_counts=args.worker_counts,
                                preparation_strategy=args.preparation_strategy,
                                compact_cache_global_max_bytes=args.compact_cache_global_max_bytes,
                                include_original_baseline=args.include_original_baseline)
    protocol = {**bindings, "mode": args.mode, "device": args.device, "repeats": args.repeats,
        "batch_size": 2, "batch_sizes": [2], "measured_draws": args.measured_draws,
        "warmup_draws": args.warmup_draws, "quality_claims": False, "update_strategy": "optimized",
        "question": "Does ordered CPU batch preparation improve B2 throughput while preserving exact optimized-update training?",
        "hypothesis": "Overlap of original warm TRAIN preparation with parent updates offsets multiprocessing transport costs.",
        "decision_use": "Compare worker counts with direct worker-zero control; no model-quality or OOM-safety selection."}
    if args.preparation_strategy == "compact_cache":
        protocol.update(question="Does ordered prefetch improve throughput with compact preparation and optimized updates in both arms?",
            hypothesis="Worker overlap can offset IPC and partitioned process-local cache costs.",
            decision_use="One marginal worker-topology comparison with direct elapsed arm times; no total-gain or multi-repeat acceptance claim.")
    if args.include_original_baseline:
        protocol.update(update_strategy="mixed_reference_and_optimized",
            question="Do compact preparation, optimized updates and optional workers reduce elapsed B2 training time against the original implementation?",
            decision_use="One three-arm comparison of total optimization gain and marginal worker topology; no multi-repeat or model-quality claim.")
    result = run_registered_replay(experiment_id=args.experiment_id, parent_experiment_id=args.parent_experiment_id,
        output_root=args.output_root, protocol=protocol,
        execute=lambda: run_prefetch_replay(args.output_root, experiment_id=args.experiment_id,
            mode=args.mode, device=args.device, measured_draws=args.measured_draws, warmup_draws=args.warmup_draws,
            repeats=args.repeats, worker_counts=tuple(args.worker_counts), config=config, paths=paths,
            descriptor=descriptor, binding_manifest=binding_manifest, reservation_manifest=reservation_manifest,
            production_input_paths=production_input_paths, fixture_reserved_bytes=args.fixture_reserved_bytes,
            preparation_strategy=args.preparation_strategy,
            compact_cache_global_max_bytes=args.compact_cache_global_max_bytes,
            include_original_baseline=args.include_original_baseline))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
