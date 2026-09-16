"""Post-prediction B3 target routing, with historical common-support masks."""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Mapping, Sequence
from types import SimpleNamespace

import torch

from music_critic.experiments.analysisgnn.corrected_training import (
    ACTIVE_HEADS, AlignedHeadTargets, corrected_supervised_loss,
    transform_semantic_value,
)
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from .model import TemporalOutput, MODEL_SCHEMA

ALIGNMENT_SCHEMA = "TemporalSliceTargets@1.0.0"


def exact_target_time(value):
    if not isinstance(value, Mapping):
        raise ValueError("target time must be a rational mapping")
    num, den = value.get("num"), value.get("den")
    if type(num) is not int or type(den) is not int or den <= 0 or gcd(num, den) != 1:
        raise ValueError("target time must be canonical exact rational")
    return Fraction(num, den)


@dataclass(frozen=True)
class TemporalAlignment:
    heads: Mapping[str, AlignedHeadTargets]
    common_support_masks: Mapping[str, torch.Tensor]
    routes: Mapping[str, tuple[str, ...]]
    diagnostics: tuple[dict, ...]
    target_sidecar_fingerprints: tuple[str, ...]
    schema: str = ALIGNMENT_SCHEMA


def align_temporal_targets(output: TemporalOutput, sidecars: Sequence[Mapping], *, shifts=None):
    """Keep native row multiplicity. Exact slice first, pre-existing beat second.

    `common_support_masks` intersects new validity with historical raw-entity
    alignment; use this separately when comparing historical validation scores.
    Additional cadence support is reported, never silently made comparable.
    """
    if output.schema != MODEL_SCHEMA or len(sidecars) != len(output.graph.slots):
        raise ValueError("temporal output/sidecar batch mismatch")
    shifts = (0,) * len(sidecars) if shifts is None else tuple(shifts)
    if len(shifts) != len(sidecars) or any(type(s) is not int or not 0 <= s < 12 for s in shifts):
        raise ValueError("one pitch-class shift in 0..11 per sample is required")
    lookups = {h: {key: i for i, key in enumerate(output.candidate_keys[h])} for h in ACTIVE_HEADS}
    old_ids = {n: set(output.graph.keys[n]) for n in ("note", "onset", "beat")}
    rows = {h: defaultdict(list) for h in ACTIVE_HEADS}
    diagnostics = []
    for sample, (sidecar, shift) in enumerate(zip(sidecars, shifts)):
        if sidecar.get("schema_version") != "analysisgnn-source-native-target-sidecar-v1":
            raise ValueError("unsupported target sidecar schema")
        beats = {}
        for relation in sidecar["relations"]:
            if relation.get("relation") == "harmonic_event_to_beat":
                source, target = relation["source_entity_id"], relation["target_entity_id"]
                if source in beats and beats[source] != target:
                    raise ValueError("ambiguous harmonic-event beat relation")
                beats[source] = target
        seen = set()
        for entity in sidecar["entities"]:
            level, entity_id = entity["entity_type"], entity["canonical_entity_id"]
            if (level, entity_id) in seen:
                raise ValueError("duplicate target entity identity")
            seen.add((level, entity_id))
            if level not in ("note", "onset", "harmonic_event"):
                continue
            if level != "note":
                time = exact_target_time(entity["onset_qn"])
                slice_id = f"rawtime:{time.numerator}_{time.denominator}"
                onset_id = f"onset:{time.numerator}_{time.denominator}"
            for head, state in entity.get("targets", {}).items():
                if head not in rows or TASK_BY_ID[head].prediction_level != level:
                    continue
                candidate, route = None, "unmatched"
                if level == "note":
                    candidate = lookups[head].get((sample, "note", entity_id))
                    route = "note" if candidate is not None else route
                    historical = (sample, entity_id) in old_ids["note"]
                else:
                    candidate = lookups[head].get((sample, "slice", slice_id))
                    if candidate is not None:
                        route = "exact_time"
                    if level == "harmonic_event":
                        historical = (sample, beats.get(entity_id)) in old_ids["beat"]
                        if candidate is None:
                            candidate = lookups[head].get((sample, "beat", beats.get(entity_id)))
                            if candidate is not None:
                                route = "legacy_beat"
                    else:
                        historical = (sample, onset_id) in old_ids["onset"]
                available = state.get("available") is True and state.get("masked") is False
                label, value, known = state.get("canonical_value"), 0, False
                if available and isinstance(label, str):
                    if shift:
                        label = transform_semantic_value(head, label, shift_pc=shift,
                                                         dialect=sidecar["dialect"], profile="corrected_v2")
                    vocabulary = get_vocabulary(TASK_BY_ID[head].vocabulary_id).labels
                    if label in vocabulary:
                        value, known = vocabulary.index(label), True
                valid = available and known and candidate is not None
                reason = ("source_unavailable" if not available else "unknown_target" if not known
                          else "alignment_failed" if candidate is None else route)
                diagnostics.append(dict(sample=sample, task=head, entity=entity_id, reason=reason))
                row = rows[head]
                for key, item in dict(candidate=0 if candidate is None else candidate,
                                      value=value, valid=valid, common=valid and historical,
                                      sample=sample, entity=entity_id, record=sidecar["record_id"],
                                      component=sidecar["source_component_id"], route=route,
                                      masked=state.get("masked") is not False,
                                      failed=candidate is None).items():
                    row[key].append(item)
    heads, common, routes = {}, {}, {}
    for h, r in rows.items():
        heads[h] = AlignedHeadTargets(
            h, torch.tensor(r["candidate"], dtype=torch.long), torch.tensor(r["value"], dtype=torch.long),
            torch.tensor(r["valid"], dtype=torch.bool), torch.tensor(r["sample"], dtype=torch.long),
            tuple(r["entity"]), tuple(r["record"]), tuple(r["component"]), sum(r["masked"]), sum(r["failed"]),
        )
        common[h] = torch.tensor(r["common"], dtype=torch.bool)
        routes[h] = tuple(r["route"])
    return TemporalAlignment(heads, common, routes, tuple(diagnostics),
                             tuple(s["fingerprint"] for s in sidecars))


def temporal_supervised_loss(output, alignment, class_weights):
    """The frozen C2 loss, including class weights and primary/auxiliary scale.

    For an empty candidate bank, masked rows use a loss-only zero sentinel.
    This never adds an inference candidate or changes target validity.
    """
    if alignment.schema != ALIGNMENT_SCHEMA or output.schema != MODEL_SCHEMA:
        raise ValueError("temporal loss schema mismatch")
    logits = dict(output.logits)
    for head, values in logits.items():
        if values.shape[0] == 0:
            if alignment.heads[head].valid_mask.any():
                raise ValueError("supervision without temporal candidates")
            logits[head] = values.new_zeros((1, values.shape[1]))
    return corrected_supervised_loss(SimpleNamespace(logits=logits), alignment, class_weights)
