"""Opt-in historical C2 TRAIN metadata schedules; no graph, pool or update I/O.

A registered caller must authorize the input metadata hashes before invoking
the production builder. Descriptor validation checks internal consistency;
``verify_c2_train_schedule`` additionally re-reads the bound production metadata.
Neither operation establishes raw-payload integrity or measured reservations.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from hashlib import sha256
import json
from pathlib import Path
import re


SCHEMA = "HistoricalC2TrainSchedule@1.0.0"
CASES = ("shift_coverage", "orbit_prefix")
SIGNED = (0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1)
METADATA_KEYS = {"split_assignments", "shift_eligibility"}
IMPLEMENTATION_FILES = (
    "performance_c2_train_schedule.py", "corrected_training.py", "full_orbit_training.py",
    "transposition.py", "contracts.py", "performance_prefetch.py", "performance_data.py",
)
LIMITS = [
    "TRAIN membership and B5A eligibility metadata only; no raw graphs, sidecars or corpus payloads read.",
    "Metadata files can include excluded split assignments; excluded payloads are never opened.",
    "Coverage selection is lexical metadata selection, not measured speed or model quality selection.",
    "The coverage case is not a historical orbit stream; the independent orbit case starts at draw zero.",
    "Reservation bytes are provisional policy, not measured serialized sizes or a RAM bound.",
    "Head support, dialect coverage, historical stale-feature behavior and update equivalence remain untested.",
    "A self-consistent descriptor is not independent source authentication; production verification re-reads pinned metadata.",
]


def _require(value, reason):
    if not value:
        raise ValueError("c2_train_schedule." + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _fp(value):
    return sha256(_canonical(value)).hexdigest()


def _seal(body):
    return {**body, "fingerprint": _fp(body)}


def _string(value, reason):
    _require(type(value) is str and bool(value) and len(value) <= 1024, reason)
    return value


def _metadata_bindings(value):
    _require(type(value) is dict and set(value) == METADATA_KEYS, "metadata_binding_fields")
    for row in value.values():
        _require(type(row) is dict and set(row) == {"sha256", "size_bytes"}, "metadata_file_binding_fields")
        _require(type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "metadata_sha256")
        _require(type(row["size_bytes"]) is int and row["size_bytes"] > 0, "metadata_size")
    return json.loads(_canonical(value))


def _train_records(split_rows, eligibility_rows):
    assignments = {}
    for row in split_rows:
        _require(type(row) is dict, "split_row_object")
        rid = _string(row.get("record_id"), "split_record_id")
        _require(rid not in assignments, "duplicate_split_record")
        _require(row.get("split") in ("train", "validation", "test"), "split_value")
        _string(row.get("source_component_id"), "split_component_id")
        assignments[rid] = row
    train = {rid: row for rid, row in assignments.items() if row["split"] == "train"}
    _require(bool(train), "train_empty")
    pairs = {rid: {} for rid in train}
    for row in eligibility_rows:
        _require(type(row) is dict, "eligibility_row_object")
        rid = _string(row.get("record_id"), "eligibility_record_id")
        _require(rid in assignments, "eligibility_record_outside_split")
        if rid not in train:
            continue  # Shared audit metadata is allowed; excluded payloads are never requested.
        shift = row.get("shift_pc")
        _require(type(shift) is int and shift in range(12), "eligibility_shift")
        _require(shift not in pairs[rid], "duplicate_eligibility_pair")
        _require(type(row.get("signed_semitones")) is int and row["signed_semitones"] == SIGNED[shift], "signed_shift")
        _require(row.get("source_component_id") == train[rid]["source_component_id"], "eligibility_component")
        _string(row.get("dialect"), "eligibility_dialect")
        for key in ("corrected_valid", "graph_range_valid", "all_targets_closed", "round_trip_valid"):
            _require(type(row.get(key)) is bool, "eligibility_boolean:" + key)
        if row["corrected_valid"]:
            _require(all(row[key] for key in ("graph_range_valid", "all_targets_closed", "round_trip_valid"))
                     and row.get("corrected_invalid_reasons") == [], "contradictory_valid_eligibility")
        pairs[rid][shift] = row
    records = []
    for rid in sorted(train):
        rows = pairs[rid]
        _require(set(rows) == set(range(12)), "complete_train_eligibility_required")
        dialects = {row["dialect"] for row in rows.values()}
        _require(len(dialects) == 1, "record_dialect_changed")
        valid = [shift for shift in range(12) if rows[shift]["corrected_valid"]]
        _require(0 in valid, "identity_shift_required")
        records.append({"record_id": rid, "component_id": train[rid]["source_component_id"],
            "dialect": next(iter(dialects)), "valid_shifts": valid,
            "eligibility_fingerprints": {str(shift): _fp(rows[shift]) for shift in range(12)}})
    return records


def _validate_records(records):
    _require(type(records) is list and bool(records), "train_records_required")
    ids = []
    for row in records:
        _require(type(row) is dict and set(row) == {"record_id", "component_id", "dialect", "valid_shifts", "eligibility_fingerprints"}, "train_record_fields")
        ids.append(_string(row["record_id"], "train_record_id"))
        for key in ("component_id", "dialect"):
            _string(row[key], "train_" + key)
        shifts = row["valid_shifts"]
        _require(type(shifts) is list and all(type(s) is int and s in range(12) for s in shifts)
                 and shifts == sorted(set(shifts)) and 0 in shifts, "valid_shift_set")
        hashes = row["eligibility_fingerprints"]
        _require(type(hashes) is dict and set(hashes) == {str(s) for s in range(12)}
                 and all(type(h) is str and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()), "eligibility_fingerprints")
    _require(ids == sorted(set(ids)), "unique_sorted_train_records")


def _assemble(records, *, mode, metadata_bindings, production_paths, implementation, reserved_bytes):
    from music_critic.experiments.analysisgnn.full_orbit_training import (
        FULL_ORBIT_PROFILE_ID, FullOrbitSampler, build_full_orbit_table, full_orbit_table_contract,
    )
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchJob
    _require(mode in ("production", "synthetic_fixture"), "mode")
    _require(type(reserved_bytes) is int and reserved_bytes > 0, "explicit_reservation_required")
    _validate_records(records)
    _require(type(implementation) is dict and set(implementation) == {"files", "fingerprint"}
             and type(implementation["files"]) is dict and set(implementation["files"]) == set(IMPLEMENTATION_FILES)
             and all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in implementation["files"].values())
             and implementation["fingerprint"] == _fp({"files": implementation["files"]}), "implementation_binding")
    if mode == "synthetic_fixture":
        _require(production_paths is None, "fixture_has_no_production_paths")
    else:
        from music_critic.experiments.analysisgnn.corrected_training import ProductionArtifactPaths
        _require(type(production_paths) is dict and set(production_paths) == {f.name for f in fields(ProductionArtifactPaths)}
                 and all(type(path) is str and Path(path).is_absolute() for path in production_paths.values()), "production_path_bindings")
        _require(len(records) == 1295 and len({row["component_id"] for row in records}) == 1209, "production_train_population")
    records = json.loads(_canonical(records))
    by_id = {row["record_id"]: row for row in records}
    components = {}
    for row in records:
        components.setdefault(row["component_id"], []).append(row["record_id"])
    table = build_full_orbit_table(components, {rid: row["valid_shifts"] for rid, row in by_id.items()})
    all12 = [rid for rid, row in by_id.items() if row["valid_shifts"] == list(range(12))]
    shared = all12[:2] if len(all12) >= 2 else None
    coverage = []
    for shift in range(12):
        other = (shift + 5) % 12
        eligible_a = [rid for rid, row in by_id.items() if shift in row["valid_shifts"]]
        eligible_b = [rid for rid, row in by_id.items() if other in row["valid_shifts"]]
        _require(bool(eligible_a) and bool(eligible_b), "coverage_shift_unavailable")
        first = shared[0] if shared else eligible_a[0]
        second = shared[1] if shared else next((rid for rid in eligible_b if rid != first), eligible_b[0])
        coverage.extend({"record_id": rid, "component_id": by_id[rid]["component_id"], "shift_pc": s}
                        for rid, s in ((first, shift), (second, other)))
    sampler = FullOrbitSampler(table, seed=17, position=0)
    orbit = [asdict(sampler.peek(i)) for i in range(200)]
    _require(sampler.position == 0, "declaration_advanced_sampler")
    cases = []
    for name, draws, resume in ((CASES[0], coverage, 5), (CASES[1], orbit, 50)):
        jobs = []
        for position in range(len(draws) // 2):
            pair = draws[2*position:2*position+2]
            ids = tuple(row["record_id"] for row in pair)
            shifts = tuple(row["shift_pc"] for row in pair)
            jobs.append(asdict(PrefetchJob(position, (2*position, 2*position+1), ids, shifts, shifts, reserved_bytes)))
        cases.append(_seal({"case_id": name, "scope": "metadata_shift_coverage_not_historical_stream" if name == CASES[0] else "historical_full_orbit_seed17_prefix",
            "draws": draws, "jobs": json.loads(_canonical(jobs)), "planned_updates": len(jobs), "planned_draws": len(draws),
            "resume_after_updates": resume, "start_draw_offset": 0, "start_committed_batch_cursor": 0,
            "resume_committed_batch_cursor": resume, "final_committed_batch_cursor": len(jobs),
            "draw_position_rule": "start_draw_offset+2*committed_batch_cursor",
            "sampler_states": ({key: FullOrbitSampler(table, seed=17, position=position).state_dict()
                for key, position in (("initial", 0), ("after_applied_resume_prefix", 100), ("after_all_applied", 200))}
                if name == CASES[1] else None)}))
    scheduled = sorted({rid for case in cases for job in case["jobs"] for rid in job["record_ids"]})
    return _seal({"schema_version": SCHEMA, "mode": mode, "split": "train", "batch_size": 2, "seed": 17,
        "historical_profile_id": FULL_ORBIT_PROFILE_ID, "signed_semitones_by_shift_pc": list(SIGNED),
        "metadata_bindings": _metadata_bindings(metadata_bindings), "production_paths": production_paths,
        "implementation": implementation, "train_records": records, "train_metadata_fingerprint": _fp(records),
        "full_orbit_table": full_orbit_table_contract(table),
        "selection": {"policy": "two_lexical_all12_else_lexical_per_slot_prefer_distinct", "all12_eligible_count": len(all12),
            "shared_all12_record_ids": shared, "selected_coverage_record_ids": sorted({row["record_id"] for row in coverage})},
        "cases": cases, "scheduled_train_record_ids": scheduled, "reserved_bytes": reserved_bytes,
        "planned_sizing_jobs": 112, "planned_sizing_record_draws": 224, "optimizer_updates_executed": 0,
        "actual_payload_bindings_verified": False, "serialized_reservations_measured": False,
        "limits": LIMITS})


def _implementation():
    directory = Path(__file__).parent
    return _seal({"files": {name: sha256((directory / name).read_bytes()).hexdigest() for name in IMPLEMENTATION_FILES}})


def _read_metadata(path, expected):
    from music_critic.experiments.analysisgnn.performance_data import _read_stable
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "regular_absolute_metadata_path")
    payload, binding = _read_stable(path)
    _require({"sha256": binding.digest, "size_bytes": len(payload)} == expected, "metadata_binding_mismatch")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    return rows


def build_c2_train_schedule(paths, *, metadata_bindings, reserved_bytes, split="train"):
    """Read only SHA/size-authorized B3/B5A metadata; use frozen reference APIs.

    This function neither registers nor launches an experiment. A future caller
    must preregister real metadata selection and separately prebind actual warm
    TRAIN payloads before measuring or executing these jobs.
    """
    _require(split == "train", "train_only")
    _require(type(reserved_bytes) is int and reserved_bytes > 0, "explicit_reservation_required")
    expected = _metadata_bindings(metadata_bindings)
    from music_critic.experiments.analysisgnn import corrected_training as reference
    _require(type(paths) is reference.ProductionArtifactPaths, "concrete_production_paths_required")
    metadata_paths = {"split_assignments": paths.b3_root / "split_assignments.jsonl",
                      "shift_eligibility": paths.b5a_shift_eligibility}
    captured = {key: _read_metadata(path, expected[key]) for key, path in metadata_paths.items()}
    records = _train_records(captured["split_assignments"], captured["shift_eligibility"])
    authority = reference.frozen_split_assignments(paths)
    _require({row["record_id"]: row for row in captured["split_assignments"]} == authority, "frozen_split_authority_mismatch")
    components = reference.production_component_records(paths)
    grouped = {}
    for row in records:
        grouped.setdefault(row["component_id"], []).append(row["record_id"])
    _require({key: tuple(value) for key, value in grouped.items()} == components, "component_authority_mismatch")
    _require({row["record_id"]: tuple(row["valid_shifts"]) for row in records} == reference.production_valid_shifts(paths), "eligibility_authority_mismatch")
    for key, path in metadata_paths.items():
        _read_metadata(path, expected[key])
    return _assemble(records, mode="production", metadata_bindings=expected,
        production_paths={field.name: str(getattr(paths, field.name)) for field in fields(paths)},
        implementation=_implementation(), reserved_bytes=reserved_bytes)


def make_fixture_c2_train_schedule(split_rows, eligibility_rows, *, reserved_bytes, synthetic):
    """Explicit tiny in-memory metadata fixture; never emits production evidence."""
    _require(synthetic is True, "explicit_synthetic_fixture_required")
    bindings = {key: {"sha256": sha256(_canonical(value)).hexdigest(), "size_bytes": len(_canonical(value))}
        for key, value in (("split_assignments", split_rows), ("shift_eligibility", eligibility_rows))}
    return _assemble(_train_records(split_rows, eligibility_rows), mode="synthetic_fixture", metadata_bindings=bindings,
        production_paths=None, implementation=_implementation(), reserved_bytes=reserved_bytes)


def validate_c2_train_schedule(descriptor, *, expected_metadata_bindings=None):
    """Regenerate both cases; callers must externally bind source/descriptor SHA."""
    _require(type(descriptor) is dict, "descriptor_object")
    if expected_metadata_bindings is not None:
        _require(descriptor.get("metadata_bindings") == _metadata_bindings(expected_metadata_bindings), "expected_metadata_mismatch")
    expected = _assemble(descriptor["train_records"], mode=descriptor["mode"],
        metadata_bindings=descriptor["metadata_bindings"], production_paths=descriptor["production_paths"],
        implementation=descriptor["implementation"], reserved_bytes=descriptor["reserved_bytes"])
    _require(_canonical(descriptor) == _canonical(expected), "descriptor_or_schedule_mismatch")
    return json.loads(_canonical(expected))


def verify_c2_train_schedule(paths, descriptor):
    """Re-authenticate actual pinned production metadata; no graph payload reads."""
    validate_c2_train_schedule(descriptor)
    _require(descriptor["mode"] == "production", "production_descriptor_required")
    actual = build_c2_train_schedule(paths, metadata_bindings=descriptor["metadata_bindings"],
                                   reserved_bytes=descriptor["reserved_bytes"])
    _require(_canonical(actual) == _canonical(descriptor), "production_source_changed")
    return actual


def jobs_for_case(descriptor, case_id):
    """Return immutable existing PrefetchJob values; these still require sizing."""
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchJob
    _require(case_id in CASES, "unknown_case")
    validated = validate_c2_train_schedule(descriptor)
    case = next(row for row in validated["cases"] if row["case_id"] == case_id)
    return tuple(PrefetchJob(row["position"], tuple(row["draw_ids"]), tuple(row["record_ids"]),
        tuple(row["schedule_shifts"]), tuple(row["applied_shifts"]), row["reserved_bytes"]) for row in case["jobs"])
