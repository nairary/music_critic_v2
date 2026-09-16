"""Complete bound identity VALIDATION for the isolated paired fusion pilot.

The dispatcher authenticates source/index bytes and registers access before
constructing this callback. This module has no file discovery or corpus reader.
It derives the entire population from that supplied frozen assignment mapping;
callers cannot pass a sampled validation list.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import re

import torch

from music_critic.graph import validate_raw_graph_batch

from .contracts import fingerprint
from .corrected_model import model_state_fingerprint
from .corrected_training import ACTIVE_HEADS, FrozenClassWeights, require_non_test_split
from .harmonic_fusion_data import FROZEN_JOINT_TUPLE_SHA256
from .harmonic_fusion_training import evaluate_validation
from .harmonic_fusion_resume import preserve_caller_rng
from .multitask_contract import PAPER_DEFINED_JOINT_COMPONENTS, TASK_BY_ID


_RECORD_COUNTS = {"train": 1295, "validation": 162, "test": 162}
_COMPONENT_COUNTS = {"train": 1209, "validation": 147, "test": 151}


def _fail(reason):
    raise ValueError("fusion_pilot_validation." + reason)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _population(assignments, *, synthetic):
    if not isinstance(assignments, Mapping) or not assignments:
        _fail("assignments_required")
    rows = []
    components = defaultdict(set)
    for record_id, assignment in assignments.items():
        if (type(record_id) is not str or not record_id or not isinstance(assignment, Mapping)
                or assignment.get("record_id") != record_id
                or assignment.get("split") not in _RECORD_COUNTS
                or type(assignment.get("source_component_id")) is not str
                or not assignment["source_component_id"]):
            _fail("assignment_identity_invalid")
        split, component = assignment["split"], assignment["source_component_id"]
        components[component].add(split)
        rows.append({"record_id": record_id, "split": split, "source_component_id": component})
    if any(len(splits) != 1 for splits in components.values()):
        _fail("component_crosses_splits")
    rows.sort(key=lambda row: row["record_id"])
    record_counts = {split: sum(row["split"] == split for row in rows) for split in _RECORD_COUNTS}
    component_counts = {split: sum(split in splits for splits in components.values()) for split in _RECORD_COUNTS}
    if not synthetic and (record_counts != _RECORD_COUNTS or component_counts != _COMPONENT_COUNTS):
        _fail("complete_frozen_population_required")
    selected = tuple((row["record_id"], row["source_component_id"]) for row in rows if row["split"] == "validation")
    if not selected:
        _fail("empty_validation_population")
    return selected, {"assignment_projection_fingerprint": fingerprint(rows),
                      "assignment_record_counts": record_counts, "assignment_component_counts": component_counts}


def _copy_class_weights(weights):
    if (not isinstance(weights, FrozenClassWeights) or not _sha(weights.payload_fingerprint)
            or set(weights.values) != set(ACTIVE_HEADS) or set(weights.supported) != set(ACTIVE_HEADS)):
        _fail("class_weight_contract_invalid")
    values, supported, tensors = {}, {}, {}
    for task in ACTIVE_HEADS:
        value, mask = weights.values[task], weights.supported[task]
        count = TASK_BY_ID[task].class_count
        if (type(value) is not torch.Tensor or value.dtype != torch.float32 or value.shape != (count,)
                or value.layout != torch.strided or not bool(torch.isfinite(value).all())
                or bool((value <= 0).any()) or type(mask) is not torch.Tensor or mask.dtype != torch.bool
                or mask.shape != (count,) or mask.layout != torch.strided):
            _fail("class_weight_tensor_invalid")
        values[task], supported[task] = value.detach().cpu().clone(), mask.detach().cpu().clone()
        tensors[f"values.{task}"], tensors[f"supported.{task}"] = values[task], supported[task]
    return FrozenClassWeights(values, supported, weights.payload_fingerprint), model_state_fingerprint(tensors)


class FullValidationCallback:
    """Detached population/tuple/weight bindings with the existing metric loop."""

    def __init__(self, *, assignments, reader, class_weights, train_seen_tuples,
                 binding_fingerprint, native_train_tuple_source_sha256=None,
                 synthetic=False, progress=None):
        if type(synthetic) is not bool or not _sha(binding_fingerprint):
            _fail("binding_or_mode_invalid")
        if not callable(reader) or (progress is not None and not callable(progress)):
            _fail("reader_or_progress_invalid")
        if (not isinstance(train_seen_tuples, (set, frozenset)) or not train_seen_tuples
                or any(type(row) is not tuple or len(row) != len(PAPER_DEFINED_JOINT_COMPONENTS)
                       or any(type(value) is not str or not value for value in row) for row in train_seen_tuples)):
            _fail("bound_native_train_tuple_inventory_required")
        # The production dispatcher has verified these bytes before extracting
        # TRAIN/corrected_harmonic_event tuples. An arbitrary nonempty set and
        # opaque binding fingerprint alone cannot claim that source provenance.
        if ((not synthetic and native_train_tuple_source_sha256 != FROZEN_JOINT_TUPLE_SHA256)
                or (native_train_tuple_source_sha256 is not None
                    and not _sha(native_train_tuple_source_sha256))):
            _fail("verified_native_train_tuple_source_required")
        self._seen = frozenset(tuple(row) for row in train_seen_tuples)
        self._selected, metadata = _population(assignments, synthetic=synthetic)
        self._components = dict(self._selected)
        self._reader, self._progress = reader, progress
        self._weights, weight_tensor_fingerprint = _copy_class_weights(class_weights)
        payload = {
            "schema_version": "HarmonicFusionFullValidationContract@1.0.0",
            "data_binding_fingerprint": binding_fingerprint,
            "mode": "synthetic_fixture" if synthetic else "frozen_dilemmadata_validation",
            "records": [record_id for record_id, _ in self._selected],
            "record_component_fingerprint": fingerprint(self._selected),
            "record_count": len(self._selected), "component_count": len(set(self._components.values())),
            "class_weight_fingerprint": self._weights.payload_fingerprint,
            "class_weight_tensor_fingerprint": weight_tensor_fingerprint,
            "native_train_tuple_count": len(self._seen),
            "native_train_tuple_fingerprint": fingerprint(sorted(self._seen)),
            "native_train_tuple_source_sha256": native_train_tuple_source_sha256,
            "tuple_scope": ("synthetic_fixture_not_verified_train" if synthetic
                            else "verified_native_train_not_augmented_exposure"),
            "selection_view": "identity_shift_zero",
            "population_policy": "entire_validation_once_each_record_no_sampling_or_windows",
            "source_authentication": ("caller_supplied_synthetic_fixture" if synthetic
                                      else "dispatcher_verified_bound_source_and_metadata_bytes"),
            "metric_policy": "unchanged_18_head_arithmetic_masks_support_components_joint_and_consistency",
            "test_payload_access": False, **metadata,
        }
        self._contract = {**payload, "fingerprint": fingerprint(payload)}

    @property
    def contract(self):
        return deepcopy(self._contract)

    def __call__(self, model, *, split="validation"):
        require_non_test_split(split)
        if split != "validation":
            _fail("validation_split_required")
        with preserve_caller_rng():
            return self._evaluate(model)

    def _evaluate(self, model):
        records = tuple(self._components)
        requested = []
        completed = []

        def emit(record_id, status):
            if self._progress is not None:
                self._progress({"stage": "validation_record", "status": status, "split": "validation",
                                "record_id": record_id, "component_id": self._components[record_id],
                                "record_index": records.index(record_id) + 1, "record_count": len(records)})

        def complete_previous():
            if len(requested) > len(completed):
                completed.append(requested[-1])
                emit(requested[-1], "completed")

        def checked_reader(record_id, *, split):
            require_non_test_split(split)
            if split != "validation" or len(requested) >= len(records) or record_id != records[len(requested)]:
                _fail("complete_ordered_membership_required")
            # The original loop requests the next record only after the
            # previous raw forward, target alignment and metric update return.
            complete_previous()
            emit(record_id, "started")
            raw, sidecar = self._reader(record_id, split="validation")
            if (not isinstance(sidecar, Mapping) or sidecar.get("record_id") != record_id
                    or sidecar.get("source_component_id") != self._components[record_id]):
                _fail("loaded_record_component_mismatch")
            validate_raw_graph_batch(raw, sample_count=1)
            if int(raw.num_graphs) != 1:
                _fail("single_full_record_required")
            requested.append(record_id)
            return raw, sidecar

        result = evaluate_validation(model, records=records, reader=checked_reader,
                                     class_weights=self._weights, split="validation",
                                     train_seen_tuples=set(self._seen))
        complete_previous()
        if tuple(completed) != records:
            _fail("validation_incomplete")
        result["full_validation_population"] = {
            "contract": self.contract, "evaluated_record_ids": list(completed),
            "evaluated_record_count": len(completed),
            "evaluated_component_count": len(set(self._components.values())),
            "complete": True, "per_record_draws": 1, "shift_pc": 0,
            "reader_calls": len(requested), "test_inputs_accessed": False,
            "test_targets_accessed": False, "test_metrics_computed": False,
        }
        return result


def make_full_validation_callback(**kwargs):
    return FullValidationCallback(**kwargs)


__all__ = ["FullValidationCallback", "make_full_validation_callback"]
