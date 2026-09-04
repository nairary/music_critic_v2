"""Official validation-first CLI for Phase 9B.2B Dilemmadata checkpoints."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from music_critic.evaluation.dilemmadata import evaluate_dilemmadata_model
from music_critic.models import (
    DilemmadataHierarchicalModel,
    dilemmadata_config_from_model_contract,
)
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
from music_critic.training.device import move_multisource_batch


def _fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path | None):
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _model(path: Path, device: torch.device) -> DilemmadataHierarchicalModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = payload["metadata"]["model_contract"]
    model = DilemmadataHierarchicalModel(
        dilemmadata_config_from_model_contract(
            contract, payload["model_state"]
        )
    )
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="dilemmadata_evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-index", type=Path, required=True)
    parser.add_argument("--raw-cache-root", type=Path, required=True)
    parser.add_argument("--target-index", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-priors", type=Path)
    parser.add_argument("--test-unlock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.preset != "dilemmadata_evaluation":
        parser.error("unknown evaluation preset")
    device = torch.device(arguments.device)
    raw_index = load_corpus_index(arguments.raw_index)
    target_index = load_dilemmadata_target_cache_index(arguments.target_index)
    dataset = IndexedMultiSourceDataset(
        raw_index,
        cache_config=CorpusCacheConfig(arguments.raw_cache_root),
        target_cache_index=target_index,
        target_cache_config=DilemmadataTargetCacheConfig(
            arguments.target_cache_root
        ),
        require_target_sidecars=True,
    )
    manifest = load_split_manifest(arguments.split_manifest)
    view = MultiCorpusDataset((dataset,), manifest, split=arguments.split)
    component_by_identity = {
        (row.dataset_id, row.piece_id): row.component_fingerprint
        for row in manifest.assignments
        if row.split == arguments.split
    }
    identities = tuple(
        view.record_identity(index) for index in range(len(view))
    )
    membership_fingerprint = _fingerprint(
        {
            "split": arguments.split,
            "split_manifest_fingerprint": manifest.manifest_fingerprint,
            "identities": identities,
        }
    )
    loader = DataLoader(
        view,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        collate_fn=collate_multisource_samples,
        multiprocessing_context=("spawn" if arguments.workers else None),
    )
    model = _model(arguments.checkpoint, device)
    batches = (
        move_multisource_batch(batch, device) for batch in loader
    )
    report = evaluate_dilemmadata_model(
        model,
        batches,
        component_by_identity=component_by_identity,
        split=arguments.split,
        membership_fingerprint=membership_fingerprint,
        train_priors=_load_json(arguments.train_priors),
        test_unlock=_load_json(arguments.test_unlock),
    )
    _write(arguments.output, report)
    print(report["fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
