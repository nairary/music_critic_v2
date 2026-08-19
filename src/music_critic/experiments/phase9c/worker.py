"""Isolated bounded-fixture and profile workers for Phase 9C-A."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from .contracts import TASK_IDS, fingerprint
from .artifacts import write_json_once


_VARIANT_OFFSET = {
    "scratch": 0.22,
    "phase7a_control": 0.16,
    "phase8a_mask_only": 0.13,
    "multilevel_equal": 0.10,
    "onset_latent": 0.15,
    "beat_latent": 0.14,
    "hierarchy_bar_latent": 0.12,
    "track_latent": 0.17,
}


def _task_report(variant: str, mode: str) -> dict[str, object]:
    mode_offset = 0.04 if "frozen" in mode else 0.0
    tasks = {}
    entries = []
    for task_index, task_id in enumerate(TASK_IDS):
        class_count = 8 if task_id.endswith("quality") else 4
        nll = 0.72 + _VARIANT_OFFSET[variant] + mode_offset + task_index * 0.01
        macro = 0.55 - _VARIANT_OFFSET[variant] / 2.0 - mode_offset
        tasks[task_id] = {
            "available": True,
            "undefined_reason": None,
            "source_entry_count": 4,
            "expanded_row_count": 4,
            "class_count": class_count,
            "nll": nll,
            "top1_accuracy": max(0.0, 0.65 - _VARIANT_OFFSET[variant]),
            "top3_accuracy": 0.8 if task_id.endswith("quality") else None,
            "top3_undefined_reason": None if task_id.endswith("quality") else "not_applicable_non_quality_task",
            "balanced_accuracy": max(0.0, 0.6 - _VARIANT_OFFSET[variant]),
            "macro_f1": macro,
            "weighted_f1": macro + 0.02,
            "confusion_matrix": [[1 if i == j else 0 for j in range(class_count)] for i in range(class_count)],
            "record_metrics": {f"record-{index}": {"available": True, "nll": nll} for index in range(4)},
            "component_metrics": {f"component-{index}": {"available": True, "nll": nll} for index in range(4)},
            "train_only_baselines": {
                "source": "train_only",
                "majority_top1_accuracy": 0.25,
                "empirical_prior_nll": math.log(class_count),
                "empirical_prior_nll_undefined_reason": None,
                "zero_train_probability_count": 0,
            },
        }
        for index in range(4):
            entries.append(
                {
                    "task_id": task_id,
                    "dataset_id": "dilemmadata",
                    "piece_id": f"validation-{index}",
                    "component_fingerprint": sha256(f"component-{index}".encode()).hexdigest(),
                    "source_entry_index": task_index,
                    "expanded_row_count": 1,
                    "label": 0,
                    "log_probabilities": [-nll] + [-nll - 1.0] * (class_count - 1),
                }
            )
    payload = {
        "contract_version": "bounded-dilemmadata-evaluation@1.0.0",
        "split": "validation",
        "membership_fingerprint": fingerprint({"fixture": "validation"}),
        "tasks": tasks,
        "counts": {"record_count": 4, "component_count": 4},
        "entry_predictions": entries,
        "test_inference": False,
        "test_targets_accessed": False,
        "test_metrics_computed": False,
        "test_unlock_used": False,
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def _run_cell(spec: dict[str, object], output: Path) -> None:
    cell_id = str(spec["cell_id"])
    kind = cell_id.split("/", 1)[0]
    variant = str(spec.get("variant_id", "scratch"))
    write_json_once(output / "resolved_config.json", spec)
    if kind == "ssl":
        initial = str(spec["initial_encoder_fingerprint"])
        report = {
            "status": "complete",
            "variant_id": variant,
            "initial_encoder_fingerprint": initial,
            "final_encoder_fingerprint": fingerprint({"initial": initial, "variant": variant, "updated": True}),
            "attempted_optimizer_updates": 1,
            "applied_optimizer_updates": 1,
            "skipped_optimizer_updates": 0,
            "raw_sample_count": 2,
            "policy_view_count": len(spec["schedule"]["policy_views"]),
            "actual_encoder_forward_count": 12,
            "declared_encoder_forward_count": 12,
            "sample_schedule_fingerprint": spec["schedule"]["sample_schedule_fingerprint"],
            "amp": True,
            "scaler_initial_scale": 16384,
            "finite_loss": True,
            "applied_gradients": True,
            "peak_vram_bytes": 1024,
            "retained_prediction_tensor_count": 0,
            "lifecycle_allocated_growth_bytes": 0,
            "loss_curve": [1.0, 0.8],
        }
        write_json_once(output / "training_report.json", report)
        (output / "ssl_metrics.jsonl").write_text(
            json.dumps({"logical_update": 0, "loss": 0.8}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_json_once(output / "compute_accounting.json", report)
        write_json_once(output / "curves.json", {"loss": report["loss_curve"]})
        (output / "last.pt").write_bytes(fingerprint(report).encode("ascii"))
        (output / "fixed_budget.pt").write_bytes(fingerprint({"fixed": report}).encode("ascii"))
        write_json_once(output / "checkpoint_manifest.json", {"last": "last.pt", "selected": "fixed_budget.pt"})
    elif kind == "encoder_export":
        report = {
            "status": "complete",
            "variant_id": variant,
            "loaded_encoder_tensor_count": 470,
            "excluded_prefixes": ["ssl_decoder", "masking_heads", "phase8b_latent_heads", "task_heads"],
            "optimizer_scheduler_scaler_transferred": False,
            "failure_atomic": True,
        }
        (output / "encoder.pt").write_bytes(fingerprint(report).encode("ascii"))
        write_json_once(output / "encoder_transfer_report.json", report)
    elif kind == "downstream":
        mode = str(spec["transfer_mode"])
        frozen = "frozen" in mode
        report = {
            "status": "complete",
            "variant_id": variant,
            "transfer_mode": mode,
            "loaded_encoder_tensor_count": 0 if variant == "scratch" else 470,
            "source_encoder_fingerprint": None if variant == "scratch" else fingerprint({"variant": variant, "source": True}),
            "loaded_encoder_fingerprint": None if variant == "scratch" else fingerprint({"variant": variant, "source": True}),
            "fresh_head_fingerprint": spec["fresh_head_fingerprint"],
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "fresh_scaler": True,
            "encoder_frozen": frozen,
            "frozen_encoder_bit_exact": frozen,
            "full_finetune_finite_encoder_gradients": not frozen,
            "full_finetune_encoder_changed": not frozen,
            "head_logits_dtype": "float32",
            "ce_dtype": "float32",
            "total_loss_dtype": "float32",
            "attempted_optimizer_updates": 1,
            "applied_optimizer_updates": 1,
            "skipped_optimizer_updates": 0,
            "loss_curve": [1.2, 0.9],
            "primary_curve": [1.1, 0.95],
        }
        write_json_once(output / "training_report.json", report)
        (output / "downstream_metrics.jsonl").write_text(
            json.dumps({"epoch": 0, "loss": 0.9}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "epoch_performance.jsonl").write_text(
            json.dumps({"epoch": 0, "samples_per_second": 1.0}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "last.pt").write_bytes(fingerprint(report).encode("ascii"))
        write_json_once(output / "checkpoint_manifest.json", {"last": "last.pt"})
        write_json_once(output / "encoder_transfer_report.json", report)
        write_json_once(output / "curves.json", {"loss": report["loss_curve"], "primary": report["primary_curve"]})
    elif kind == "train_priors":
        tasks = {}
        for task_id in TASK_IDS:
            class_count = 8 if task_id.endswith("quality") else 4
            tasks[task_id] = {
                "class_counts": [1 for _ in range(class_count)],
                "class_probabilities": [1.0 / class_count for _ in range(class_count)],
                "majority_class_id": 0,
                "source_entry_count": class_count,
            }
        payload = {
            "contract_version": "1.0.0",
            "source_split": "train_only",
            "train_membership_fingerprint": fingerprint({"fixture": "train"}),
            "tasks": tasks,
        }
        write_json_once(output / "train_priors.json", {**payload, "fingerprint": fingerprint(payload)})
    elif kind == "validation":
        write_json_once(output / "validation_report.json", _task_report(variant, str(spec["transfer_mode"])))
    else:
        raise ValueError(f"unsupported bounded cell: {cell_id}")


def _profile(candidate: int, output: Path) -> None:
    if candidate > 8:
        write_json_once(output, {"status": "oom", "batch_size": candidate, "state_cleaned_by_subprocess_exit": True})
        raise SystemExit(42)
    report = {
        "status": "passed",
        "batch_size": candidate,
        "warmup_steps": 1,
        "measured_steps": 3,
        "peak_allocated_vram_bytes": candidate * 1000,
        "peak_reserved_vram_bytes": candidate * 1200,
        "samples_per_second": candidate * 2.0,
        "encoder_forwards_per_second": candidate * 24.0,
        "seconds_per_downstream_epoch": 10.0 / candidate,
        "validation_seconds": 2.0,
        "subprocess_isolation": True,
    }
    write_json_once(output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    cell = sub.add_parser("bounded-cell")
    cell.add_argument("spec", type=Path)
    cell.add_argument("output", type=Path)
    profile = sub.add_parser("profile-candidate")
    profile.add_argument("batch_size", type=int)
    profile.add_argument("output", type=Path)
    initial = sub.add_parser("export-initial-encoder")
    initial.add_argument("output", type=Path)
    production_profile = sub.add_parser("profile-production-candidate")
    production_profile.add_argument("config", type=Path)
    production_profile.add_argument("batch_size", type=int)
    production_profile.add_argument("root", type=Path)
    production_profile.add_argument("output", type=Path)
    priors = sub.add_parser("build-train-priors")
    priors.add_argument("--raw-index", type=Path, required=True)
    priors.add_argument("--raw-cache-root", type=Path, required=True)
    priors.add_argument("--target-index", type=Path, required=True)
    priors.add_argument("--target-cache-root", type=Path, required=True)
    priors.add_argument("--split-manifest", type=Path, required=True)
    priors.add_argument("--batch-size", type=int, required=True)
    priors.add_argument("--output", type=Path, required=True)
    class_weights = sub.add_parser("build-class-weights")
    class_weights.add_argument("--train-priors", type=Path, required=True)
    class_weights.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "bounded-cell":
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        args.output.mkdir(parents=True, exist_ok=True)
        _run_cell(spec, args.output)
    elif args.operation == "profile-candidate":
        _profile(args.batch_size, args.output)
    elif args.operation == "export-initial-encoder":
        import torch

        from music_critic.models import (
            DilemmadataHierarchicalConfig,
            DilemmadataHierarchicalModel,
        )

        torch.manual_seed(17)
        model = DilemmadataHierarchicalModel(DilemmadataHierarchicalConfig())
        prefixes = (
            "local_baseline.encoder.",
            "context_encoder.pooling.",
            "context_encoder.transformer.",
            "context_encoder.fusion.",
        )
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if name.startswith(prefixes)
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".partial", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save({"encoder_state": state}, temporary)
            os.replace(temporary, args.output)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(json.dumps({"status": "complete", "tensor_count": len(state)}, sort_keys=True))
    elif args.operation == "profile-production-candidate":
        from music_critic.experiments.phase9c.planner import build_experiment_plan
        from music_critic.experiments.phase9c.runner import run_production_profile_candidate

        config = json.loads(args.config.read_text(encoding="utf-8"))
        config["batch_size"] = args.batch_size
        plan = build_experiment_plan(config)
        report = run_production_profile_candidate(
            plan, args.root, batch_size=args.batch_size
        )
        write_json_once(args.output, report)
    elif args.operation == "build-class-weights":
        from music_critic.models import (
            DILEMMADATA_ACTIVE_TASK_IDS,
            class_weight_artifact,
        )
        from music_critic.evaluation import validate_dilemmadata_train_priors

        try:
            priors = json.loads(args.train_priors.read_text(encoding="utf-8"))
            validate_dilemmadata_train_priors(priors)
            tasks = priors["tasks"]
            counts = {
                task_id: tuple(tasks[task_id]["class_counts"])
                for task_id in DILEMMADATA_ACTIVE_TASK_IDS
            }
            membership = str(priors["train_membership_fingerprint"])
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise SystemExit("phase9c.class_weights.train_priors_invalid") from exc
        write_json_once(
            args.output,
            class_weight_artifact(
                counts,
                policy="inverse_sqrt_frequency_supported",
                train_membership_fingerprint=membership,
            ),
        )
    else:
        from torch.utils.data import DataLoader

        from music_critic.evaluation import build_dilemmadata_train_priors
        from music_critic.models import DILEMMADATA_ACTIVE_TASK_IDS
        from music_critic.tasks import (
            CorpusCacheConfig,
            DilemmadataTargetCacheConfig,
            IndexedMultiSourceDataset,
            MultiCorpusDataset,
            collate_multisource_samples,
            load_corpus_index,
            load_dilemmadata_target_cache_index,
            load_split_manifest,
        )
        from music_critic.tasks import DILEMMADATA_TARGET_ENCODING_BY_TASK

        raw_index = load_corpus_index(args.raw_index)
        target_index = load_dilemmadata_target_cache_index(args.target_index)
        dataset = IndexedMultiSourceDataset(
            raw_index,
            cache_config=CorpusCacheConfig(args.raw_cache_root),
            target_cache_index=target_index,
            target_cache_config=DilemmadataTargetCacheConfig(args.target_cache_root),
            require_target_sidecars=True,
        )
        manifest = load_split_manifest(args.split_manifest)
        view = MultiCorpusDataset((dataset,), manifest, split="train")
        identities = tuple(view.record_identity(index) for index in range(len(view)))
        membership = fingerprint(
            {
                "split": "train",
                "split_manifest_fingerprint": manifest.manifest_fingerprint,
                "identities": identities,
            }
        )
        loader = DataLoader(
            view,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_multisource_samples,
        )
        rows = []
        for batch in loader:
            for target in batch.target_batches:
                if target.task_id not in DILEMMADATA_ACTIVE_TASK_IDS:
                    continue
                mask = target.availability_mask & target.entity_index_mask
                positions = mask.nonzero(as_tuple=False).flatten().tolist()
                labels = target.values.tolist()
                samples = target.sample_indices.tolist()
                sources = target.source_entry_indices.tolist()
                grouped: dict[tuple[int, int], int] = {}
                for position in positions:
                    key = (int(samples[position]), int(sources[position]))
                    label = int(labels[position])
                    previous = grouped.setdefault(key, label)
                    if previous != label:
                        raise ValueError("phase9c.priors.source_entry_label_conflict")
                class_count = len(DILEMMADATA_TARGET_ENCODING_BY_TASK[target.task_id].vocabulary)
                for (sample_index, source_index), label in sorted(grouped.items()):
                    rows.append(
                        {
                            "task_id": target.task_id,
                            "dataset_id": batch.dataset_ids[sample_index],
                            "piece_id": batch.piece_ids[sample_index],
                            "source_entry_index": source_index,
                            "label": label,
                            "log_probabilities": [0.0 for _ in range(class_count)],
                        }
                    )
        artifact = build_dilemmadata_train_priors(
            rows, train_membership_fingerprint=membership
        )
        write_json_once(args.output, artifact)


if __name__ == "__main__":
    main()
