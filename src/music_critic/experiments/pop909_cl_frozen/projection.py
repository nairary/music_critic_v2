"""Frozen POP909-CL root/quality crosswalk; no training or target inputs.

Root probabilities marginalize all spelled classes before prediction. Quality
predictions retain the full corrected vocabulary; only gold coverage is limited
to the explicitly supported nine interval templates.
"""

from __future__ import annotations

from dataclasses import asdict
import re

import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import (
    TASK_BY_ID,
    get_vocabulary,
)


PROJECTION_ID = "pop909-cl-corrected-root-pc-quality-subset9-v1"
PROJECTION_SCHEMA_VERSION = "1.0.0"
# Binds ordered labels, aliases, task routing and source provenance. Changing
# this lock requires a new reviewed projection protocol, never silent remapping.
EXPECTED_REGISTRY_FINGERPRINT = (
    "c76437a04c6b2b985831d4c249b894dd49eb4a0e97045612521bf76e4184db52"
)
QUALITY_SUBSET9 = ("M", "m", "o", "+", "D7", "M7", "m7", "/o7", "o7")
UNSUPPORTED_GOLD_QUALITIES = ("sus2", "sus4", "mM7", "+7")

# Explicit root-relative semitone templates. The source spellings below match
# POP909-CL's existing exact-triad/seventh normalizer; no quality-15 map is used.
_POP_TEMPLATES = {
    "M": (0, 4, 7),
    "m": (0, 3, 7),
    "o": (0, 3, 6),
    "+": (0, 4, 8),
    "D7": (0, 4, 7, 10),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "/o7": (0, 3, 6, 10),
    "o7": (0, 3, 6, 9),
}
_CORRECTED_TEMPLATES = {
    "major triad": (0, 4, 7),
    "minor triad": (0, 3, 7),
    "diminished triad": (0, 3, 6),
    "augmented triad": (0, 4, 8),
    "dominant seventh chord": (0, 4, 7, 10),
    "major seventh chord": (0, 4, 7, 11),
    "minor seventh chord": (0, 3, 7, 10),
    "half-diminished seventh chord": (0, 3, 6, 10),
    "diminished seventh chord": (0, 3, 6, 9),
}
_UNSUPPORTED_TEMPLATES = {
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "mM7": (0, 3, 7, 11),
    "+7": (0, 4, 8, 10),
}
_NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SPELLING = re.compile(r"([A-G])(#*|b*)\Z")


class ProjectionContractError(ValueError):
    """Logits or the live registry violate the frozen projection contract."""


def _registry_binding() -> dict[str, object]:
    payload = {
        task_id: {
            "task": asdict(TASK_BY_ID[task_id]),
            "vocabulary": asdict(get_vocabulary(TASK_BY_ID[task_id].vocabulary_id)),
        }
        for task_id in ("root", "quality")
    }
    if fingerprint(payload) != EXPECTED_REGISTRY_FINGERPRINT:
        raise ProjectionContractError("root/quality registry drift: frozen crosswalk must be reviewed")
    return payload


def root_pc_from_label(label: str) -> int:
    """Resolve a spelled root without an enharmonic preference or class-order guess."""
    if not isinstance(label, str) or (match := _SPELLING.fullmatch(label)) is None:
        raise ProjectionContractError(f"unsupported spelled root: {label!r}")
    letter, accidentals = match.groups()
    return (_NATURAL_PC[letter] + accidentals.count("#") - accidentals.count("b")) % 12


def root_class_labels() -> tuple[str, ...]:
    _registry_binding()
    return get_vocabulary(TASK_BY_ID["root"].vocabulary_id).labels


def quality_class_labels() -> tuple[str, ...]:
    _registry_binding()
    return get_vocabulary(TASK_BY_ID["quality"].vocabulary_id).labels


def _quality_rows() -> list[dict[str, object]]:
    labels = quality_class_labels()
    by_intervals = {intervals: label for label, intervals in _CORRECTED_TEMPLATES.items()}
    return [
        {
            "pop909_label": pop_label,
            "intervals": list(_POP_TEMPLATES[pop_label]),
            "corrected_label": by_intervals[_POP_TEMPLATES[pop_label]],
            "corrected_class_id": labels.index(by_intervals[_POP_TEMPLATES[pop_label]]),
        }
        for pop_label in QUALITY_SUBSET9
    ]


def quality_subset_ids() -> tuple[int, ...]:
    """Corrected vocabulary IDs in the declared POP subset9 order."""
    return tuple(row["corrected_class_id"] for row in _quality_rows())


def quality_gold_class_id(label: str) -> int | None:
    """Return corrected ID for supported gold, None for the four excluded qualities.

    Missing/ambiguous labels must be handled by the caller's independent target
    mask. Unknown source labels raise instead of becoming silent exclusions.
    """
    rows = _quality_rows()
    if label in UNSUPPORTED_GOLD_QUALITIES:
        return None
    for row in rows:
        if row["pop909_label"] == label:
            return row["corrected_class_id"]
    raise ProjectionContractError(f"unknown POP909-CL quality: {label!r}")


def projection_contract() -> dict[str, object]:
    """Return a fresh deterministic, self-fingerprinted versioned crosswalk."""
    registry = _registry_binding()
    root_labels = root_class_labels()
    payload = {
        "projection_id": PROJECTION_ID,
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "registry_fingerprint": EXPECTED_REGISTRY_FINGERPRINT,
        "registry_binding": registry,
        "root": {
            "class_labels": list(root_labels),
            "class_to_pc": [root_pc_from_label(label) for label in root_labels],
            "output_class_ids": list(range(12)),
            "prediction_policy": "softmax_all37_then_sum_enharmonic_probabilities_then_argmax12",
            "tie_policy": "lowest_pitch_class_id",
        },
        "quality": {
            "class_labels": list(quality_class_labels()),
            "subset9": _quality_rows(),
            "unsupported_gold": [
                {"pop909_label": label, "intervals": list(_UNSUPPORTED_TEMPLATES[label]),
                 "reason": "outside_frozen_subset9"}
                for label in UNSUPPORTED_GOLD_QUALITIES
            ],
            "prediction_policy": "argmax_all17_before_gold_subset9_comparison",
            "outside_subset_prediction_policy": "retain_corrected_class_id_and_count_as_error",
            "tie_policy": "lowest_corrected_class_id",
            "renormalize_subset9": False,
        },
    }
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _validate_logits(logits: torch.Tensor, class_count: int) -> None:
    if not isinstance(logits, torch.Tensor):
        raise ProjectionContractError("logits must be a torch Tensor")
    if logits.ndim < 1 or logits.shape[-1] != class_count:
        raise ProjectionContractError(f"logits require final dimension {class_count}")
    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
        raise ProjectionContractError("logits must have finite floating values")


def project_root_logits(logits: torch.Tensor) -> torch.Tensor:
    """Marginalize spelled-root softmax probabilities, preserving leading shape.

    Class IDs follow the verified live registry, not alphabetical or PC order.
    Every input class contributes to exactly one output PC. No labels are read.
    """
    labels = root_class_labels()
    _validate_logits(logits, len(labels))
    probabilities = logits.softmax(dim=-1)
    mapping = tuple(root_pc_from_label(label) for label in labels)
    return torch.stack(
        [probabilities[..., [index for index, mapped in enumerate(mapping) if mapped == pc]].sum(-1)
         for pc in range(12)],
        dim=-1,
    )


def predict_root_ids(logits: torch.Tensor) -> torch.Tensor:
    """Return PC IDs; exact ties select the lowest PC ID."""
    return project_root_logits(logits).argmax(dim=-1)


def predict_quality_ids(logits: torch.Tensor) -> torch.Tensor:
    """Return full corrected17 winners, including every outside-subset class."""
    labels = quality_class_labels()
    _validate_logits(logits, len(labels))
    return logits.argmax(dim=-1)
