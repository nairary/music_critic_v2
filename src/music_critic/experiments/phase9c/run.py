"""Official CLI for Phase 9C-A plan/profile/run/resume/aggregate/select/verify."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import read_json, verify_bundle, write_json_once
from .contracts import ACTIONS
from .planner import (
    DEFAULT_MIXTURE,
    build_experiment_plan,
    materialize_ssl_split_manifest,
)
from .runner import execute_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preset", choices=(
        "bounded_acceptance",
        "rtx_profile",
        "one_seed_primary_pilot",
        "one_seed_full_ablation",
    ))
    parser.add_argument("--ssl-updates", type=int)
    parser.add_argument("--downstream-epochs", type=int)
    parser.add_argument("--downstream-steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--profile-batch-candidates", type=int, nargs="+")
    parser.add_argument("--profile-report-path")
    parser.add_argument("--ssl-index-paths", nargs="+")
    parser.add_argument("--ssl-cache-roots", nargs="+")
    parser.add_argument("--ssl-split-manifest")
    parser.add_argument("--ssl-source-split-manifests", nargs=2)
    parser.add_argument("--downstream-raw-index")
    parser.add_argument("--downstream-raw-cache-root")
    parser.add_argument("--target-cache-index")
    parser.add_argument("--target-cache-root")
    parser.add_argument("--downstream-split-manifest")
    parser.add_argument("--fail-after-cell", type=int, default=0)
    return parser


def _config(arguments: argparse.Namespace) -> dict[str, Any]:
    value: dict[str, Any] = {}
    if arguments.config is not None:
        loaded = read_json(arguments.config)
        if not isinstance(loaded, dict):
            raise ValueError("phase9c.cli.config_mapping_required")
        value.update(loaded)
    for name in (
        "preset",
        "ssl_updates",
        "downstream_epochs",
        "downstream_steps_per_epoch",
        "batch_size",
        "bootstrap_replicates",
        "profile_batch_candidates",
        "profile_report_path",
    ):
        observed = getattr(arguments, name)
        if observed is not None:
            value[name] = observed
    data = dict(value.get("data", {}))
    for name in (
        "ssl_index_paths",
        "ssl_cache_roots",
        "ssl_split_manifest",
        "ssl_source_split_manifests",
        "downstream_raw_index",
        "downstream_raw_cache_root",
        "target_cache_index",
        "target_cache_root",
        "downstream_split_manifest",
    ):
        observed = getattr(arguments, name)
        if observed is not None:
            data[name] = observed
    if data:
        value["data"] = data
    value.setdefault("preset", "bounded_acceptance")
    value.setdefault("mixture_weights", DEFAULT_MIXTURE)
    return value


def main() -> int:
    arguments = _parser().parse_args()
    root = arguments.output_root.resolve()
    if arguments.action == "verify":
        result = verify_bundle(root)
    else:
        config = materialize_ssl_split_manifest(_config(arguments))
        plan = build_experiment_plan(config)
        if arguments.action == "plan":
            root.mkdir(parents=True, exist_ok=True)
            write_json_once(root / "experiment_plan.json", plan)
            write_json_once(root / "protocol.json", plan["protocol"])
            write_json_once(root / "data_semantic_projection.json", plan["data_semantic_projection"])
            result = {
                "status": "planned",
                "plan_fingerprint": plan["fingerprint"],
                "production_started": False,
                "test_access": False,
            }
        else:
            result = execute_experiment(
                root,
                plan,
                action=arguments.action,
                fail_after_cell=arguments.fail_after_cell,
            )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
