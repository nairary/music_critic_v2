"""Sequential WS2 input from an explicitly hash-bound historical Dilemma cache.

This preserves the corrected reference canonical/sidecar bytes. It does not
repeat their original raw conversion, authenticate that conversion's history,
or use the unsealed cache-hit shortcut in the production trainer. Preparation
scans TRAIN raw pieces one at a time, freezes a shared eligibility pool and
ordered schedules, and retains only compact evidence between records.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from music_critic.data import loads_piece
from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.training_policy import component_balanced_record_draw
from music_critic.graph import build_raw_graph, graph_fingerprint
from music_critic.ssl.data import SSLRawSample, collate_ssl_samples
from music_critic.ssl.group_masking_pilot import PilotBatch, POLICIES, bind_train_batch, paired_mask_bindings
from music_critic.ssl.group_masking_scientific_data import (
    BoundFile, MAX_REQUEST_BYTES, _bound, _fail, _identity, _indexed, _json,
    _positive, _rows, _sha, _source_path, _stat, guard_split,
)

REQUEST_SCHEMA = "WS2GroupMaskingScientificCachedDataRequest@1.0.0"
BINDING_SCHEMA = "WS2GroupMaskingScientificCachedDataBinding@1.0.0"
MAX_CACHE_FILE_BYTES = 128 * 1024 * 1024
COMPILER_POLICY = {
    "version": "WS2ConservativePairedEligibilitySchedule@1.0.0",
    "train_scan": "full_frozen_train_raw_only_sequential",
    "record_budget": "floor_each_training_batch_graph_budget_divided_by_fixed_batch_size",
    "eligibility": "both_pitch_policies_available_at_epoch0_under_conservative_record_budget",
    "filter_scope": "conservative_shared_pool_not_maximal_eligibility_or_representative_corpus_claim",
    "schedule": "existing_component_balanced_record_draw_with_replacement",
    "epoch_domains": {"ssl": 0, "theory_replay": 1, "theory_recovery": 2},
    "draw_index": "zero_based_update_times_batch_size_plus_slot",
    "augmentation": None, "arm_specific_replacement": False,
    "validation": "full_frozen_validation_singletons_theory_even_if_ssl_policy_unavailable",
}


@dataclass(frozen=True)
class CachedScientificDataRequest:
    payload: dict

    def to_dict(self):
        return deepcopy(self.payload)


ScientificDataRequest = CachedScientificDataRequest


@dataclass(frozen=True)
class CacheFile(BoundFile):
    def __post_init__(self):
        if (not isinstance(self.path, Path) or not self.path.is_absolute()
                or not _sha(self.sha256) or not _positive(self.size_bytes, MAX_CACHE_FILE_BYTES)):
            _fail("invalid_cache_artifact_binding")


def _cache_file(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}
            or not isinstance(value["path"], str)):
        _fail("invalid_cache_artifact_binding")
    return CacheFile(Path(value["path"]), value["sha256"], value["size_bytes"])


def load_data_request(path, *, split="train"):
    guard_split(split)
    path = Path(path)
    if _stat(path)[2] > MAX_REQUEST_BYTES:
        _fail("request_too_large")
    with path.open("rb") as stream:
        request = CachedScientificDataRequest(_json(stream.read(MAX_REQUEST_BYTES + 1)))
    _request(request)
    return request


def _request(request, *, sizing=False):
    if not isinstance(request, CachedScientificDataRequest):
        _fail("cached_request_required")
    value = request.to_dict()
    if (not isinstance(value, dict) or set(value) != {
            "schema", "cache_root", "cache_index", "split_assignments", "source_components",
            "class_weights", "validation_limits"} or value["schema"] != REQUEST_SCHEMA
            or not isinstance(value["cache_root"], str) or not Path(value["cache_root"]).is_absolute()):
        _fail("cached_request_schema_invalid")
    files = {name: _bound(value[name]) for name in (
        "cache_index", "split_assignments", "source_components", "class_weights")
        if not (sizing and name == "class_weights")}
    from music_critic.ssl.group_masking_scientific_eval import ValidationResourceLimits
    limits = value["validation_limits"]
    if sizing and limits is None:
        return value, files, None
    if not isinstance(limits, dict) or set(limits) != {
            "max_notes_per_piece", "max_nodes_per_piece", "max_edges_per_piece"}:
        _fail("validation_limits_required")
    return value, files, ValidationResourceLimits(**limits)


def preflight_data(request, pilot_config, recovery_config, *, split="train", sizing=False):
    guard_split(split)
    value, files, limits = _request(request, sizing=sizing)
    if (pilot_config.lambda_ssl <= 0 or pilot_config.lambda_theory <= 0
            or recovery_config.lambda_ssl != 0 or recovery_config.lambda_theory <= 0
            or pilot_config.seed != recovery_config.seed
            or pilot_config.max_samples_per_batch != recovery_config.max_samples_per_batch):
        _fail("stage_config_or_budget_mismatch")
    payloads = {name: item.read() for name, item in files.items()}
    if files["split_assignments"].sha256 != reference.CORRECTED_SPLIT_ASSIGNMENTS_SHA256:
        _fail("frozen_split_sha_mismatch")
    assignments = _indexed(_rows(payloads["split_assignments"]))
    cache = _indexed(_rows(payloads["cache_index"]))
    components = _rows(payloads["source_components"])
    component_records, component_splits, record_component = {}, {}, {}
    for row in components:
        component, ids = row.get("source_component_id"), row.get("record_ids")
        if (not _identity(component) or component in component_records or not isinstance(ids, list)
                or not ids or len(ids) != len(set(ids)) or not all(_identity(record) for record in ids)):
            _fail("component_metadata_invalid")
        component_records[component] = ids
        for record in ids:
            if record in record_component:
                _fail("duplicate_component_record")
            record_component[record] = component
    for record, row in assignments.items():
        name, component = row.get("split"), row.get("source_component_id")
        if name not in {"train", "validation", "test"} or record_component.get(record) != component:
            _fail("complete_split_component_closure_mismatch")
        if component in component_splits and component_splits[component] != name:
            _fail("component_cross_split_leakage")
        component_splits[component] = name
    expected = {record for record, row in assignments.items() if row["split"] != "test"}
    if set(cache) != expected:
        _fail("complete_train_validation_cache_required")
    root = Path(value["cache_root"])
    for record, row in cache.items():
        if (set(row) != {"record_id", "split", "source_component_id", "raw", "sidecar", "metadata"}
                or row["split"] != assignments[record]["split"]
                or row["source_component_id"] != assignments[record]["source_component_id"]):
            _fail("cache_split_identity_mismatch")
        from hashlib import sha256
        token = sha256(record.encode("utf-8")).hexdigest()[:24]
        for role, filename in (("raw", "piece.json"), ("sidecar", "target_sidecar.json"),
                               ("metadata", "metadata.json")):
            item = _cache_file(row[role])
            path = _source_path(root, token + "/" + filename)
            if item.path != path:
                _fail("cache_path_identity_or_confinement_mismatch")
    train = sorted(record for record, row in assignments.items() if row["split"] == "train")
    validation = sorted(record for record, row in assignments.items() if row["split"] == "validation")
    if not train or not validation:
        _fail("train_validation_membership_empty")
    payload = {"schema": "WS2GroupMaskingScientificCachedPreflight@1.0.0",
               "request_fingerprint": fingerprint(value), "compiler_policy": deepcopy(COMPILER_POLICY),
               "pilot_config": pilot_config.contract(), "recovery_config": recovery_config.contract(),
               "validation_limits": None if limits is None else asdict(limits),
               "sizing_only": sizing,
               "metadata": {name: {"sha256": item.sha256, "size_bytes": item.size_bytes}
                            for name, item in files.items()},
               "split_record_counts": {name: sum(row["split"] == name for row in assignments.values())
                                       for name in ("train", "validation", "test")},
               "split_component_counts": {name: sum(item == name for item in component_splits.values())
                                          for name in ("train", "validation", "test")},
               "train_record_ids": train, "validation_record_ids": validation,
               "cache_origin_limit": "historical_canonical_and_expanded_sidecar_bytes_original_conversion_not_repeated",
               "source_bytes_verified": False, "test_source_access": False,
               "cache_file_limit_bytes": MAX_CACHE_FILE_BYTES}
    return {**payload, "fingerprint": fingerprint(payload)}


class BoundScientificData:
    def __init__(self, request, pilot_config, recovery_config, *, binding=None, on_receipt=None, sizing=False):
        self.request = CachedScientificDataRequest(request.to_dict())
        self.configs = {"pilot": pilot_config, "recovery": recovery_config}
        self.preflight = preflight_data(self.request, pilot_config, recovery_config, sizing=sizing)
        self._value, self._files, self.validation_resource_limits = _request(self.request, sizing=sizing)
        self._stats = {name: _stat(item.path) for name, item in self._files.items()}
        self._cache = _indexed(_rows(self._files["cache_index"].read()))
        self._weights = None if sizing else reference.load_frozen_class_weights(self._files["class_weights"].path)
        self._verify_metadata()
        self._binding, self._records, self._schedules = None, {}, None
        self._sizing = sizing
        self.on_receipt = on_receipt
        self.access_observations = {"raw_files_verified": 0, "raw_bytes_verified": 0,
            "sidecar_files_verified": 0, "sidecar_bytes_verified": 0, "metadata_files_verified": 0,
            "raw_graphs_built": 0, "test_source_reads": 0, "source_conversion_calls": 0}
        if binding is not None:
            if sizing:
                _fail("sizing_cannot_load_training_binding")
            self._accept_binding(binding)

    @property
    def binding(self):
        if self._binding is None:
            _fail("prepared_binding_required")
        return deepcopy(self._binding)

    @property
    def validation_membership(self):
        return tuple(tuple(row) for row in self.binding["validation_membership"])

    @property
    def validation_bindings(self):
        return tuple(self.binding["validation_batch_bindings"])

    @property
    def train_seen_tuples(self):
        return None

    def schedule_bindings(self, stage):
        if stage not in self.configs:
            _fail("stage_invalid")
        return tuple(self.binding["schedule_bindings"][stage])

    def _verify_metadata(self):
        for name, item in self._files.items():
            if _stat(item.path) != self._stats[name]:
                item.read()
                self._stats[name] = _stat(item.path)

    def _receipt(self, value):
        if self.on_receipt:
            self.on_receipt({"schema": "WS2GroupMaskingCachedDataReceipt@1.0.0", **value,
                             "access_observations": deepcopy(self.access_observations)})

    def _read(self, row, role):
        item = _cache_file(row[role])
        # Recheck confinement if an ancestor is replaced after metadata preflight.
        path = item.path.relative_to(Path(self._value["cache_root"]))
        _source_path(Path(self._value["cache_root"]), path.as_posix())
        payload = item.read()
        self.access_observations[role + "_files_verified"] += 1
        if role != "metadata":
            self.access_observations[role + "_bytes_verified"] += item.size_bytes
        return payload

    def _record(self, record_id, *, split, targets, limits=None):
        guard_split(split)
        if self._sizing and targets:
            _fail("sizing_targets_forbidden")
        if record_id not in self._cache or self._cache[record_id]["split"] != split:
            _fail("record_not_in_bound_membership")
        self._receipt({"status": "started", "record_id": record_id, "split": split, "targets": targets})
        try:
            self._verify_metadata()
            row = self._cache[record_id]
            metadata = _json(self._read(row, "metadata"))
            if metadata != {"record_id": record_id, "split": split,
                            "piece_sha256": row["raw"]["sha256"],
                            "sidecar_sha256": row["sidecar"]["sha256"]}:
                _fail("cache_metadata_content_mismatch")
            piece = loads_piece(self._read(row, "raw"))
            if (piece.dataset_name != "dilemmadata" or piece.source_group_id != row["source_component_id"]
                    or piece.targets or piece.annotations):
                _fail("cached_raw_identity_or_target_leakage")
            from music_critic.adapters.dilemmadata import _piece_id
            dialect = "an_joint" if record_id.startswith("an:") else "dlc"
            if piece.piece_id != _piece_id(record_id, dialect):
                _fail("cached_raw_record_identity_mismatch")
            if limits is not None:
                # Do not allocate a graph for a plainly oversized raw piece.
                if len(piece.notes) > limits["notes"]:
                    _fail("record_resource_bound_notes")
                if 1 + len(piece.notes) + len(piece.tracks) + len(piece.bars) + len(piece.beats) > limits["nodes"]:
                    _fail("record_resource_bound_nodes")
            graph = build_raw_graph(piece)
            self.access_observations["raw_graphs_built"] += 1
            counts = {"notes": len(piece.notes), "nodes": sum(graph[node].num_nodes for node in graph.node_types),
                      "edges": sum(graph[edge].edge_index.shape[1] for edge in graph.edge_types)}
            if limits is not None and any(counts[key] > limit for key, limit in limits.items()):
                _fail("record_resource_bound_graph")
            sample = SSLRawSample(graph, graph_fingerprint(graph), piece.dataset_name, piece.piece_id)
            sidecar = _json(self._read(row, "sidecar")) if targets else None
            if targets and (not isinstance(sidecar, dict) or sidecar.get("record_id") != record_id
                    or sidecar.get("source_component_id") != row["source_component_id"]
                    or sidecar.get("schema_version") != "analysisgnn-source-native-target-sidecar-v1"
                    or sidecar.get("fingerprint") != fingerprint({key: val for key, val in sidecar.items() if key != "fingerprint"})):
                _fail("cached_sidecar_identity_or_fingerprint_mismatch")
            evidence = {"identity": [sample.dataset_id, sample.piece_id],
                "raw_graph_fingerprint": sample.raw_graph_fingerprint, "counts": counts,
                "sidecar_fingerprint": None if sidecar is None else fingerprint(sidecar)}
            previous = self._records.get(record_id)
            if previous is not None and (any(previous.get(key) != evidence[key] for key in ("identity", "raw_graph_fingerprint", "counts"))
                    or (targets and previous.get("sidecar_fingerprint") is not None
                        and previous["sidecar_fingerprint"] != evidence["sidecar_fingerprint"])):
                _fail("record_content_binding_mismatch")
            if self._binding is not None and (previous is None or (targets and previous["sidecar_fingerprint"] is None)):
                _fail("record_content_binding_missing")
            if previous is None or targets:
                self._records[record_id] = evidence
            self._verify_metadata()
            self._receipt({"status": "completed", "record_id": record_id, "split": split, "targets": targets})
            return sample, sidecar
        except BaseException as exc:
            try:
                self._receipt({"status": "failed", "record_id": record_id, "split": split,
                    "targets": targets, "exception_type": type(exc).__name__, "reason": str(exc)})
            except BaseException:
                pass
            raise

    def _batch(self, row, config):
        limits = {"notes": config.max_notes_per_batch, "nodes": config.max_nodes_per_batch,
                  "edges": config.max_edges_per_batch}
        ssl = [self._record(record, split="train", targets=False, limits=limits)[0] for record in row.get("ssl", ())]
        theory = [self._record(record, split="train", targets=True, limits=limits) for record in row["theory"]]
        return PilotBatch(collate_ssl_samples(ssl) if ssl else None, collate_ssl_samples([item[0] for item in theory]),
                          tuple(item[1] for item in theory), self._weights)

    def load_update(self, stage, index, *, split="train"):
        guard_split(split)
        if split != "train":
            _fail("updates_train_only")
        if stage not in self.configs or self._schedules is None:
            _fail("prepared_schedule_required")
        if type(index) is not int or not 0 <= index < len(self._schedules[stage]):
            _fail("update_index_invalid")
        batch = self._batch(self._schedules[stage][index], self.configs[stage])
        actual = bind_train_batch(batch, self.configs[stage])
        if stage == "pilot":
            paired_mask_bindings(batch.ssl, self.configs[stage], update=index)
        if self._binding is not None and actual != self.schedule_bindings(stage)[index]:
            _fail("update_content_binding_mismatch")
        return batch

    def validation_batches(self, *, split="validation"):
        guard_split(split)
        if split != "validation":
            _fail("validation_only")
        from music_critic.ssl.group_masking_scientific_eval import validation_batch_binding, validation_policy_bindings
        limits = {"notes": self.validation_resource_limits.max_notes_per_piece,
                  "nodes": self.validation_resource_limits.max_nodes_per_piece,
                  "edges": self.validation_resource_limits.max_edges_per_piece}
        def iterate():
            for index, record in enumerate(self.preflight["validation_record_ids"]):
                sample, sidecar = self._record(record, split="validation", targets=True, limits=limits)
                raw = collate_ssl_samples((sample,))
                batch = PilotBatch(raw, raw, (sidecar,), self._weights)
                actual = validation_batch_binding(batch, self.configs["pilot"], resource_limits=self.validation_resource_limits)
                if self._binding is not None and actual != self.validation_bindings[index]:
                    _fail("validation_content_binding_mismatch")
                _, availability = validation_policy_bindings(raw, self.configs["pilot"])
                if self._binding is not None and availability != self._binding["validation_policy_availability"][index]:
                    _fail("validation_policy_availability_mismatch")
                yield batch
        return iterate()

    def _accept_binding(self, binding):
        value = deepcopy(binding)
        if (not isinstance(value, dict) or set(value) != {"schema", "preflight", "records", "eligibility", "schedules",
                "schedule_bindings", "validation_membership", "validation_batch_bindings", "validation_policy_availability", "fingerprint"}
                or value["schema"] != BINDING_SCHEMA or value["preflight"] != self.preflight
                or value["fingerprint"] != fingerprint({key: item for key, item in value.items() if key != "fingerprint"})):
            _fail("prepared_cached_binding_invalid")
        eligibility = value["eligibility"]
        train = self.preflight["train_record_ids"]
        if not isinstance(eligibility, dict) or set(eligibility) != set(train):
            _fail("full_train_eligibility_required")
        eligible = []
        for record, row in eligibility.items():
            if (not isinstance(row, dict) or set(row) != {"eligible", "reason"}
                    or type(row["eligible"]) is not bool
                    or (row["eligible"] and row["reason"] is not None)
                    or (not row["eligible"] and not _identity(row["reason"]))):
                _fail("eligibility_schema_invalid")
            if row["eligible"]:
                eligible.append(record)
        schedules = _schedules(self, eligible)
        if value["schedules"] != schedules or not isinstance(value["schedule_bindings"], dict) or set(value["schedule_bindings"]) != set(self.configs):
            _fail("prepared_deterministic_schedule_mismatch")
        for stage, config in self.configs.items():
            hashes = value["schedule_bindings"][stage]
            if not isinstance(hashes, list) or len(hashes) != config.updates or not all(_sha(item) for item in hashes):
                _fail("prepared_schedule_hashes_invalid")
        validation = self.preflight["validation_record_ids"]
        records = value["records"]
        selected = {record for rows in schedules.values() for row in rows for ids in row.values() for record in ids} | set(validation)
        if not isinstance(records, dict) or not selected <= set(records) or not set(records) <= set(self._cache):
            _fail("prepared_record_inventory_invalid")
        for record, row in records.items():
            if (not isinstance(row, dict) or set(row) != {"identity", "raw_graph_fingerprint", "counts", "sidecar_fingerprint"}
                    or not isinstance(row["identity"], list) or len(row["identity"]) != 2 or row["identity"][0] != "dilemmadata"
                    or not _identity(row["identity"][1]) or not _sha(row["raw_graph_fingerprint"])
                    or (row["sidecar_fingerprint"] is not None and not _sha(row["sidecar_fingerprint"]))
                    or not isinstance(row["counts"], dict) or set(row["counts"]) != {"notes", "nodes", "edges"}
                    or any(type(count) is not int or count < 0 for count in row["counts"].values())):
                _fail("prepared_cached_record_invalid")
        membership = [records[record]["identity"] for record in validation]
        if (value["validation_membership"] != membership or len({tuple(row) for row in membership}) != len(membership)
                or not isinstance(value["validation_batch_bindings"], list)
                or len(value["validation_batch_bindings"]) != len(validation)
                or not all(_sha(item) for item in value["validation_batch_bindings"])
                or not isinstance(value["validation_policy_availability"], list)
                or len(value["validation_policy_availability"]) != len(validation)):
            _fail("prepared_validation_membership_invalid")
        for identity, available in zip(membership, value["validation_policy_availability"], strict=True):
            if not isinstance(available, dict) or set(available) != set(POLICIES):
                _fail("prepared_validation_availability_invalid")
            for row in available.values():
                if (not isinstance(row, dict) or set(row) != {"piece", "available", "reason", "plan_fingerprint", "resolution_fingerprint"}
                        or row["piece"] != identity or type(row["available"]) is not bool
                        or (row["available"] and (row["reason"] is not None or not _sha(row["plan_fingerprint"])))
                        or (not row["available"] and (not isinstance(row["reason"], dict) or not _identity(row["reason"].get("code"))))
                        or any(row[key] is not None and not _sha(row[key]) for key in ("plan_fingerprint", "resolution_fingerprint"))):
                    _fail("prepared_validation_availability_invalid")
        self._records, self._schedules, self._binding = deepcopy(records), schedules, value


def _schedules(data, eligible):
    if not eligible:
        _fail("no_jointly_eligible_train_records")
    groups = {}
    for record in sorted(eligible):
        groups.setdefault(data._cache[record]["source_component_id"], []).append(record)
    def draw(config, epoch, index):
        return component_balanced_record_draw(groups, seed=config.seed, epoch=epoch, draw_index=index)[1]
    result = {}
    for stage, config in data.configs.items():
        rows = []
        for update in range(config.updates):
            slots = range(update * config.max_samples_per_batch, (update + 1) * config.max_samples_per_batch)
            row = {"theory": [draw(config, 1 if stage == "pilot" else 2, index) for index in slots]}
            if stage == "pilot":
                row["ssl"] = [draw(config, 0, index) for index in slots]
            rows.append(row)
        result[stage] = rows
    return result


def prepare_scientific_data(request, pilot_config, recovery_config, *, on_receipt=None):
    from music_critic.ssl.group_masking_scientific_eval import validation_batch_binding, validation_policy_bindings
    from music_critic.ssl.hierarchical_masking import HierarchyMaskUnavailableError
    data = BoundScientificData(request, pilot_config, recovery_config, on_receipt=on_receipt)
    divisor = pilot_config.max_samples_per_batch
    limits = {"notes": min(pilot_config.max_notes_per_batch, recovery_config.max_notes_per_batch) // divisor,
              "nodes": min(pilot_config.max_nodes_per_batch, recovery_config.max_nodes_per_batch) // divisor,
              "edges": min(pilot_config.max_edges_per_batch, recovery_config.max_edges_per_batch) // divisor}
    eligibility = {}
    for record in data.preflight["train_record_ids"]:
        try:
            sample, _ = data._record(record, split="train", targets=False, limits=limits)
            raw = collate_ssl_samples((sample,))
            # Structural eligibility cannot be inferred from target availability.
            if raw.raw_graph_batch["note"].num_nodes < 2:
                eligibility[record] = {"eligible": False, "reason": "fewer_than_two_raw_notes"}
                sample = raw = None
                continue
            paired_mask_bindings(raw, pilot_config, update=0)
        except HierarchyMaskUnavailableError as exc:
            eligibility[record] = {"eligible": False, "reason": "hierarchy_policy_unavailable:" + str(exc)}
        except ValueError as exc:
            if str(exc).startswith("group_scientific_data.record_resource_bound"):
                eligibility[record] = {"eligible": False, "reason": str(exc)}
            else:
                raise
        else:
            eligibility[record] = {"eligible": True, "reason": None}
        # No raw graph/sidecar retained while another record is loaded.
        sample = raw = None
    eligible = [record for record, row in eligibility.items() if row["eligible"]]
    data._schedules = _schedules(data, eligible)
    schedules = {stage: [bind_train_batch(data.load_update(stage, index), config)
                         for index in range(config.updates)] for stage, config in data.configs.items()}
    validation_hashes, membership, availability = [], [], []
    for batch in data.validation_batches():
        validation_hashes.append(validation_batch_binding(batch, pilot_config, resource_limits=data.validation_resource_limits))
        membership.extend([list(item) for item in zip(batch.ssl.dataset_ids, batch.ssl.piece_ids, strict=True)])
        _, available = validation_policy_bindings(batch.ssl, pilot_config)
        availability.append(available)
    payload = {"schema": BINDING_SCHEMA, "preflight": data.preflight, "records": deepcopy(data._records),
               "eligibility": eligibility, "schedules": data._schedules, "schedule_bindings": schedules,
               "validation_membership": membership, "validation_batch_bindings": validation_hashes,
               "validation_policy_availability": availability}
    data._accept_binding({**payload, "fingerprint": fingerprint(payload)})
    return data


def load_bound_scientific_data(request, binding_payload, pilot_config, recovery_config, *, on_receipt=None):
    return BoundScientificData(request, pilot_config, recovery_config, binding=binding_payload, on_receipt=on_receipt)


def size_scientific_data(request, pilot_config, recovery_config, *, on_receipt=None):
    """Raw-only full TRAIN/VALIDATION sizing before choosing validation limits.

    ``validation_limits`` and ``class_weights`` may be null for this operation.
    It never loads model weights, class weights or corpus target sidecars. The
    existing graph builder runs sequentially without inferred shape cutoffs;
    its single-piece allocation requires the coordinator's external RAM/time
    guard. Declared cache-file lengths remain capped at 128 MiB. Nothing here
    permits a training run with null resource limits or absent class weights.
    """
    data = BoundScientificData(request, pilot_config, recovery_config, on_receipt=on_receipt, sizing=True)
    for split in ("train", "validation"):
        for record in data.preflight[split + "_record_ids"]:
            sample, _ = data._record(record, split=split, targets=False)
            del sample
    rows = [{"record_id": record, "split": data._cache[record]["split"],
             "source_component_id": data._cache[record]["source_component_id"], **evidence,
             "raw_size_bytes": data._cache[record]["raw"]["size_bytes"],
             "sidecar_declared_size_bytes_not_read": data._cache[record]["sidecar"]["size_bytes"]}
            for record, evidence in sorted(data._records.items())]
    payload = {"schema": "WS2GroupMaskingScientificCachedSizing@1.0.0", "preflight": data.preflight,
               "records": rows, "access_observations": deepcopy(data.access_observations),
               "max_counts_by_split": {split: {key: max(row["counts"][key] for row in rows if row["split"] == split)
                                               for key in ("notes", "nodes", "edges")}
                                       for split in ("train", "validation")},
               "limits_not_selected_automatically": True,
               "resource_boundary": "one_piece_graph_at_a_time_external_hard_ram_and_time_guard_required"}
    return {**payload, "fingerprint": fingerprint(payload)}
