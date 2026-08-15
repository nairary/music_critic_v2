from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from music_critic.experiments.phase8b2.contracts import Phase8B2ContractError
from music_critic.experiments.phase8b2.config import Phase8B2Config
from music_critic.experiments.phase8b2.runner import (
    build_experiment_plan,
    official_test_evaluation_overrides,
)
from music_critic.experiments.phase8b2.selection import (
    TestUnlockRequest,
    authorize_test_evaluation,
    consume_test_authorization,
    select_validation_checkpoint,
)
from music_critic.experiments.phase8b2.statistics import (
    PieceMetric,
    aggregate_paired_piece_metrics,
)


def _candidate(variant: str, hook: float, pop: float, nll: float, compute: int):
    return {
        "variant_id": variant,
        "checkpoint": f"/{variant}.pt",
        "protocol_fingerprint": "protocol",
        "split": "validation",
        "dataset_endpoints": {"hooktheory": hook, "pop909_cl": pop},
        "validation_nll": nll,
        "encoder_forward_count": compute,
    }


def test_validation_selection_uses_mean_rank_and_declared_ties() -> None:
    artifact = select_validation_checkpoint(
        (
            _candidate("z", 0.9, 0.5, 0.2, 10),
            _candidate("a", 0.5, 0.9, 0.2, 10),
            _candidate("middle", 0.6, 0.6, 0.1, 20),
        ),
        protocol_fingerprint="protocol",
    )
    assert artifact["source_split"] == "validation"
    assert not artifact["test_used"]
    assert artifact["selected_variant_id"] == "middle"
    assert artifact["selected_count"] == 1


def test_selection_rejects_test_candidate() -> None:
    candidate = _candidate("a", 1.0, 1.0, 0.1, 2)
    candidate["split"] = "test"
    with pytest.raises(Phase8B2ContractError, match="validation_only"):
        select_validation_checkpoint(
            (candidate,), protocol_fingerprint="protocol"
        )


def _write_selection(path: Path) -> None:
    artifact = select_validation_checkpoint(
        (_candidate("a", 1.0, 1.0, 0.1, 2),),
        protocol_fingerprint="protocol",
    )
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_test_lock_requires_all_conditions_and_is_single_use(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.json"
    _write_selection(selection)
    request = TestUnlockRequest(
        protocol_fingerprint="protocol",
        experiment_identity="experiment",
        selection_artifact=str(selection),
        output_directory=str(tmp_path / "new-test-output"),
        test_membership_fingerprint="test-membership",
        acknowledge=True,
    )
    authorization = authorize_test_evaluation(request)
    assert authorization["authorization_stage"] == "pre_inference"
    assert not Path(request.output_directory).exists()
    consume_test_authorization(authorization)
    assert Path(request.output_directory).is_dir()
    plan = build_experiment_plan(Phase8B2Config())
    plan["protocol"]["fingerprint"] = "protocol"
    plan["protocol"]["data"]["test_membership_fingerprint"] = (
        "test-membership"
    )
    overrides = official_test_evaluation_overrides(plan, authorization)
    assert "split=test" in overrides
    assert "acknowledge_test_evaluation=true" in overrides
    with pytest.raises(Phase8B2ContractError, match="already_used"):
        consume_test_authorization(authorization)


def test_test_lock_rejects_stale_selection_artifact(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    _write_selection(selection)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["selected_checkpoint"] = "/changed.pt"
    selection.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase8B2ContractError, match="selection_artifact_stale"):
        authorize_test_evaluation(
            TestUnlockRequest(
                protocol_fingerprint="protocol",
                experiment_identity="experiment",
                selection_artifact=str(selection),
                output_directory=str(tmp_path / "output"),
                test_membership_fingerprint="test-membership",
                acknowledge=True,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    (
        ("acknowledge", False, "acknowledgement_required"),
        ("protocol_fingerprint", "wrong", "protocol_fingerprint_mismatch"),
        ("test_membership_fingerprint", "", "test_membership_missing"),
    ),
)
def test_test_lock_negative_paths(
    tmp_path: Path, field: str, value: object, category: str
) -> None:
    selection = tmp_path / "selection.json"
    _write_selection(selection)
    values = {
        "protocol_fingerprint": "protocol",
        "experiment_identity": "experiment",
        "selection_artifact": str(selection),
        "output_directory": str(tmp_path / "output"),
        "test_membership_fingerprint": "test-membership",
        "acknowledge": True,
    }
    values[field] = value
    with pytest.raises(Phase8B2ContractError, match=category):
        authorize_test_evaluation(TestUnlockRequest(**values))


def _records() -> list[PieceMetric]:
    rows = []
    for seed in (1, 2, 3):
        for piece in ("p1", "p2", "p3"):
            rows.extend(
                (
                    PieceMetric(
                        "hooktheory",
                        piece,
                        seed,
                        "supervised_scratch",
                        "supervised_scratch",
                        "macro_f1",
                        0.4 + 0.01 * seed,
                    ),
                    PieceMetric(
                        "hooktheory",
                        piece,
                        seed,
                        "phase7a_control",
                        "full_finetune",
                        "macro_f1",
                        0.5 + 0.01 * seed,
                    ),
                    PieceMetric(
                        "hooktheory",
                        piece,
                        seed,
                        "onset_latent",
                        "full_finetune",
                        "macro_f1",
                        0.6 + 0.01 * seed,
                    ),
                )
            )
    return rows


def test_piece_level_bootstrap_is_paired_and_deterministic() -> None:
    first = aggregate_paired_piece_metrics(
        _records(), bootstrap_seed=91, bootstrap_replicates=100
    )
    second = aggregate_paired_piece_metrics(
        reversed(_records()), bootstrap_seed=91, bootstrap_replicates=100
    )
    assert first == second
    assert first["statistical_unit"] == "independent_piece"
    assert not first["bounded_results_are_scientific_evidence"]
    references = {
        row["reference_variant_id"] for row in first["paired_comparisons"]
    }
    assert {"supervised_scratch", "phase7a_control"} <= references
    assert all(
        row["bootstrap"]["retained_bootstrap_draws"] == 0
        for row in first["paired_comparisons"]
        if row["bootstrap"]["available"]
    )


def test_statistics_reject_duplicate_piece_rows() -> None:
    rows = _records()
    with pytest.raises(Phase8B2ContractError, match="duplicate_piece_metric"):
        aggregate_paired_piece_metrics(
            [*rows, deepcopy(rows[0])],
            bootstrap_seed=1,
            bootstrap_replicates=10,
        )
