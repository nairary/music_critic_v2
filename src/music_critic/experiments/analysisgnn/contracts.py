"""Frozen scientific contracts for Phase 9E-B1.

The module contains no network, filesystem, CUDA, or model imports.  It is the
single machine-checkable boundary between the preregistered protocol and the
runtime implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Literal


PHASE9EB1_CONTRACT_VERSION = "1.0.0"
ANALYSISGNN_REPOSITORY = "https://github.com/manoskary/analysisgnn"
ANALYSISGNN_COMMIT = "e115182fb29b74bdcb6bf3547ed427d967580947"
ANALYSISGNN_RUN_PATH = "melkisedeath/AnalysisGNN/rhsjiz03"
ANALYSISGNN_RUN_SOURCE_COMMIT = "7738a282abe5090d44627759786dfa31b71e1a43"
HISTORICAL_ARTIFACT = "melkisedeath/AnalysisGNN/model-rhsjiz03:v0"
HISTORICAL_CHECKPOINT_SHA256 = (
    "a557d0046e2c03c19514e1351a3cd0f2b49c31b991c370307345a7f1c6a65f31"
)
HISTORICAL_CHECKPOINT_BYTES = 289_662_455

GRAPHMUSE_REPOSITORY = "https://github.com/manoskary/graphmuse"
# The exact paper/run GraphMuse revision was not published.  This is the
# closest pre-run public revision whose HybridGNN API and state layout match
# the public checkpoint; the choice is an explicit reconstruction substitution.
GRAPHMUSE_COMMIT = "c36eedba811a24c0addf96bdd3d1df449cf753c1"

DILEMMADATA_COMMIT = "d60ee75b4a9495e932a4a7be39381578be17e222"
EXPECTED_RECORD_COUNT = 719
EXPECTED_DIALECT_COUNTS = {"an_joint": 108, "dlc": 611}
EXPECTED_SPLIT_COUNTS = {"train": 577, "validation": 71, "test": 71}
EXPECTED_SPLIT_FINGERPRINT = (
    "58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e"
)
EXPECTED_RAW_INDEX_FINGERPRINT = (
    "c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e"
)
EXPECTED_COMMON_REGISTRY_FINGERPRINT = (
    "bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e"
)

QUALITY_TASK = "quality"
INVERSION_TASK = "inversion"
TASK_CLASS_COUNTS = {QUALITY_TASK: 50, INVERSION_TASK: 4}
INVERSION_VOCABULARY = ("root", "first", "second", "third")
TRANSPOSITIONS = (
    "P1",
    "m2",
    "M2",
    "m3",
    "M3",
    "P4",
    "A4",
    "P5",
    "m6",
    "M6",
    "m7",
    "M7",
)
SEMITONES_BY_TRANSPOSITION = {
    "P1": 0,
    "m2": 1,
    "M2": 2,
    "m3": 3,
    "M3": 4,
    "P4": 5,
    "A4": 6,
    "P5": 7,
    "m6": 8,
    "M6": 9,
    "m7": 10,
    "M7": 11,
}

NODE_TYPES = ("note", "measure", "beat")
EDGE_TYPES = (
    ("note", "onset", "note"),
    ("note", "consecutive", "note"),
    ("note", "during", "note"),
    ("note", "rest", "note"),
    ("note", "consecutive_rev", "note"),
    ("note", "during_rev", "note"),
    ("note", "rest_rev", "note"),
    ("note", "connects", "measure"),
    ("measure", "connects", "note"),
    ("measure", "next", "measure"),
    ("note", "connects", "beat"),
    ("beat", "connects", "note"),
    ("beat", "next", "beat"),
)
BASE_FEATURE_NAMES = (
    "bar_exp_duration",
    "onset_bar_norm",
    "is_down_beat",
    *(f"pc_{index:02d}" for index in range(12)),
    *(f"octave_{index:02d}" for index in range(10)),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Phase9EB1ContractError(ValueError):
    """Raised when a run attempts to leave the preregistered protocol."""


def canonical_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def fingerprint(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase9EB1Config:
    contract_version: str = PHASE9EB1_CONTRACT_VERSION
    arm: Literal["analysisgnn-common-subset"] = "analysisgnn-common-subset"
    seeds: tuple[int, ...] = (17, 23, 42)
    applied_update_budget: int = 10_000
    warmup_applied_updates: int = 500
    validation_every_applied_updates: int = 500
    graphs_per_candidate_update: int = 1
    training_schedule: Literal[
        "seeded_shuffled_source_transposition_views"
    ] = "seeded_shuffled_source_transposition_views"
    learning_rate: float = 0.005
    weight_decay: float = 0.0005
    label_smoothing: float = 0.1
    ignore_index: int = -1
    class_weights: None = None
    uncertainty_weighting: Literal[
        "half_inverse_square_plus_log1p_square"
    ] = "half_inverse_square_plus_log1p_square"
    hidden_channels: int = 256
    output_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    use_jk: bool = True
    use_beat_hierarchy: bool = True
    use_measure_hierarchy: bool = True
    bigru_layers: int = 2
    bigru_bidirectional: bool = True
    logit_fusion: bool = True
    quality_classes: int = 50
    inversion_classes: int = 4
    head_layers: int = 2
    train_transpositions: tuple[str, ...] = TRANSPOSITIONS
    validation_transpositions: tuple[str, ...] = ("P1",)
    test_transpositions: tuple[str, ...] = ("P1",)
    split_before_augmentation: bool = True
    source_grouped_split: bool = True
    entry_aggregation: Literal[
        "mean_note_log_probability"
    ] = "mean_note_log_probability"
    macro_f1_rule: Literal[
        "mean_f1_over_supported_true_classes"
    ] = "mean_f1_over_supported_true_classes"
    balanced_accuracy_rule: Literal[
        "mean_recall_over_supported_true_classes"
    ] = "mean_recall_over_supported_true_classes"
    validation_selection: Literal[
        "mean_normalized_quality_inversion_nll"
    ] = "mean_normalized_quality_inversion_nll"
    bootstrap_samples: int = 2_000

    def __post_init__(self) -> None:
        expected = Phase9EB1Config.__dataclass_fields__
        if self.contract_version != PHASE9EB1_CONTRACT_VERSION:
            raise Phase9EB1ContractError("contract version differs from Phase 9E-B1")
        if self.seeds != (17, 23, 42):
            raise Phase9EB1ContractError("seeds must be exactly 17, 23, and 42")
        if self.applied_update_budget <= self.warmup_applied_updates:
            raise Phase9EB1ContractError("update budget must exceed the 500-update warmup")
        if (
            self.validation_every_applied_updates != 500
            or self.applied_update_budget % self.validation_every_applied_updates
        ):
            raise Phase9EB1ContractError("validation cadence must divide the update budget")
        if (
            self.graphs_per_candidate_update != 1
            or self.training_schedule
            != "seeded_shuffled_source_transposition_views"
        ):
            raise Phase9EB1ContractError("candidate-update schedule differs from protocol")
        if (
            self.warmup_applied_updates != 500
            or self.learning_rate != 0.005
            or self.weight_decay != 0.0005
            or self.label_smoothing != 0.1
            or self.ignore_index != -1
            or self.class_weights is not None
        ):
            raise Phase9EB1ContractError("optimisation differs from preregistration")
        if (
            self.hidden_channels,
            self.output_channels,
            self.num_layers,
            self.use_jk,
            self.use_beat_hierarchy,
            self.use_measure_hierarchy,
            self.bigru_layers,
            self.bigru_bidirectional,
        ) != (256, 128, 3, True, True, True, 2, True):
            raise Phase9EB1ContractError("encoder differs from checkpoint-attested arm")
        if (
            self.quality_classes,
            self.inversion_classes,
            self.head_layers,
            self.logit_fusion,
        ) != (50, 4, 2, True):
            raise Phase9EB1ContractError("output surface must be the two fused common heads")
        if (
            self.train_transpositions != TRANSPOSITIONS
            or self.validation_transpositions != ("P1",)
            or self.test_transpositions != ("P1",)
            or not self.split_before_augmentation
            or not self.source_grouped_split
        ):
            raise Phase9EB1ContractError("augmentation or leakage control differs from protocol")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise Phase9EB1ContractError("dropout must be finite in [0, 1)")
        if (
            self.macro_f1_rule != "mean_f1_over_supported_true_classes"
            or self.balanced_accuracy_rule
            != "mean_recall_over_supported_true_classes"
        ):
            raise Phase9EB1ContractError("evaluation averaging rule differs from protocol")
        if self.bootstrap_samples <= 0 or not expected:
            raise Phase9EB1ContractError("bootstrap sample count must be positive")

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(asdict(self))


COMMON_BENCHMARK_CONFIG = Phase9EB1Config()


@dataclass(frozen=True, slots=True)
class HistoricalAttestation:
    source_commit: str
    run_path: str
    run_source_commit: str
    artifact_path: str
    artifact_version: int
    original_filename: str
    epoch: int
    global_step: int
    checkpoint_bytes: int
    checkpoint_sha256: str
    config_sha256: str
    requirements_sha256: str
    summary_sha256: str
    output_log_sha256: str
    claim: Literal["historical_checkpoint_and_result_attestation"] = (
        "historical_checkpoint_and_result_attestation"
    )

    def __post_init__(self) -> None:
        if (
            self.source_commit != ANALYSISGNN_COMMIT
            or self.run_path != ANALYSISGNN_RUN_PATH
            or self.run_source_commit != ANALYSISGNN_RUN_SOURCE_COMMIT
            or self.artifact_path != HISTORICAL_ARTIFACT
            or self.artifact_version != 0
            or self.original_filename != "epoch=98-step=8910.ckpt"
            or self.epoch != 98
            or self.global_step != 8910
            or self.checkpoint_bytes != HISTORICAL_CHECKPOINT_BYTES
            or self.checkpoint_sha256 != HISTORICAL_CHECKPOINT_SHA256
        ):
            raise Phase9EB1ContractError("historical identity differs from public evidence")
        for value in (
            self.checkpoint_sha256,
            self.config_sha256,
            self.requirements_sha256,
            self.summary_sha256,
            self.output_log_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise Phase9EB1ContractError("attestation digests must be SHA-256")

    @property
    def fingerprint(self) -> str:
        return fingerprint(asdict(self))


def graph_schema_fingerprint() -> str:
    return fingerprint(
        {
            "base_feature_names": BASE_FEATURE_NAMES,
            "edge_types": EDGE_TYPES,
            "graphmuse_commit": GRAPHMUSE_COMMIT,
            "node_types": NODE_TYPES,
            "schema": "analysisgnn-native-note-beat-measure-v1",
        }
    )


__all__ = [
    "ANALYSISGNN_COMMIT",
    "ANALYSISGNN_REPOSITORY",
    "ANALYSISGNN_RUN_PATH",
    "ANALYSISGNN_RUN_SOURCE_COMMIT",
    "BASE_FEATURE_NAMES",
    "COMMON_BENCHMARK_CONFIG",
    "DILEMMADATA_COMMIT",
    "EDGE_TYPES",
    "EXPECTED_COMMON_REGISTRY_FINGERPRINT",
    "EXPECTED_DIALECT_COUNTS",
    "EXPECTED_RAW_INDEX_FINGERPRINT",
    "EXPECTED_RECORD_COUNT",
    "EXPECTED_SPLIT_COUNTS",
    "EXPECTED_SPLIT_FINGERPRINT",
    "GRAPHMUSE_COMMIT",
    "GRAPHMUSE_REPOSITORY",
    "HISTORICAL_ARTIFACT",
    "HISTORICAL_CHECKPOINT_BYTES",
    "HISTORICAL_CHECKPOINT_SHA256",
    "HistoricalAttestation",
    "INVERSION_TASK",
    "INVERSION_VOCABULARY",
    "NODE_TYPES",
    "PHASE9EB1_CONTRACT_VERSION",
    "Phase9EB1Config",
    "Phase9EB1ContractError",
    "QUALITY_TASK",
    "SEMITONES_BY_TRANSPOSITION",
    "TASK_CLASS_COUNTS",
    "TRANSPOSITIONS",
    "canonical_json",
    "fingerprint",
    "graph_schema_fingerprint",
]
