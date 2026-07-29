"""Read-only Phase 6A/6B/6C checkpoint loading for evaluation."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import random
from typing import Any

import torch

from music_critic.evaluation.contracts import (
    EvaluationContractError,
    canonical_fingerprint,
)
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
)
from music_critic.models.checkpoint import _validate_model_state
from music_critic.training.models import (
    BaselineModel,
    model_contract_metadata,
)


def _tuple_task_weights(value: object) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, list) or len(item) != 2 for item in value
    ):
        raise EvaluationContractError(
            "evaluation.checkpoint.task_weights_invalid"
        )
    return tuple((str(item[0]), float(item[1])) for item in value)


def _build_model(contract: dict[str, Any]) -> BaselineModel:
    config = contract.get("model_config")
    if not isinstance(config, dict):
        raise EvaluationContractError(
            "evaluation.checkpoint.model_config_missing"
        )
    values = dict(config)
    values["task_weights"] = _tuple_task_weights(
        values.get("task_weights")
    )
    try:
        if "hierarchical_model_contract_version" in contract:
            model: BaselineModel = HierarchicalHeterogeneousBaseline(
                HierarchicalBaselineConfig(**values)
            )
        elif "model_contract_version" in contract:
            model = LocalHeterogeneousBaseline(
                LocalBaselineConfig(**values)
            )
        else:
            raise EvaluationContractError(
                "evaluation.checkpoint.model_kind_unknown"
            )
    except EvaluationContractError:
        raise
    except Exception as exc:
        raise EvaluationContractError(
            f"evaluation.checkpoint.model_config_invalid:{exc}"
        ) from exc
    if model_contract_metadata(model) != contract:
        raise EvaluationContractError(
            "evaluation.checkpoint.model_contract_mismatch"
        )
    return model


def _capture_rng() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda = state["torch_cuda"]
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[BaselineModel, dict[str, Any]]:
    """Load model weights only while preserving every caller RNG state."""

    checkpoint_path = Path(path).resolve()
    before = _capture_rng()
    try:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise EvaluationContractError(
                f"evaluation.checkpoint.unreadable:{exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise EvaluationContractError(
                "evaluation.checkpoint.payload_invalid"
            )
        metadata = payload.get("metadata")
        state = payload.get("model_state")
        if not isinstance(metadata, dict):
            raise EvaluationContractError(
                "evaluation.checkpoint.metadata_invalid"
            )
        if metadata.get("training_checkpoint_version") is not None:
            checkpoint_kind = "phase6c_training"
            model_contract = metadata.get("model_contract")
            data_fingerprints = metadata.get("data_fingerprints")
            if not isinstance(model_contract, dict) or not isinstance(
                data_fingerprints, dict
            ):
                raise EvaluationContractError(
                    "evaluation.checkpoint.training_metadata_invalid"
                )
        else:
            checkpoint_kind = (
                "phase6b_hierarchical"
                if "hierarchical_model_contract_version" in metadata
                else "phase6a_local"
            )
            model_contract = metadata
            data_fingerprints = None
        model = _build_model(model_contract)
        try:
            validated_state = _validate_model_state(state, model)
        except Exception as exc:
            raise EvaluationContractError(
                f"evaluation.checkpoint.model_state_invalid:{exc}"
            ) from exc
        model.load_state_dict(validated_state, strict=True)
        model.to(torch.device(device))
        model.eval()
        evidence = {
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "model_contract": model_contract,
            "model_contract_fingerprint": canonical_fingerprint(
                model_contract
            ),
            "training_checkpoint_version": metadata.get(
                "training_checkpoint_version"
            ),
            "training_data_fingerprints": data_fingerprints,
            "optimizer_state_loaded": False,
            "checkpoint_rng_state_loaded": False,
            "model_state_mutation_scope": "new_evaluation_model_only",
        }
        return model, evidence
    finally:
        _restore_rng(before)


__all__ = ["load_evaluation_checkpoint"]
