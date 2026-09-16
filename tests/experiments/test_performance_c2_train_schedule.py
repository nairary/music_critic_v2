"""Metadata-only fixtures; never loads real TRAIN/VALIDATION/TEST payloads."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from music_critic.experiments.analysisgnn import performance_c2_train_schedule as schedule
from music_critic.experiments.analysisgnn import corrected_training as reference
from music_critic.experiments.analysisgnn.full_orbit_training import FullOrbitSampler, build_full_orbit_table


def metadata(valid=None):
    valid = valid or {"an:test:a": range(12), "b": range(12), "c": range(9)}
    splits = [{"record_id": rid, "source_component_id": "component:" + rid, "split": "train"} for rid in valid]
    splits += [{"record_id": "held:" + split, "source_component_id": "held:" + split, "split": split}
               for split in ("validation", "test")]
    rows = []
    for rid, shifts in valid.items():
        for shift in range(12):
            accepted = shift in shifts
            rows.append({"record_id": rid, "source_component_id": "component:" + rid, "dialect": "an_joint",
                "shift_pc": shift, "signed_semitones": schedule.SIGNED[shift], "corrected_valid": accepted,
                "corrected_invalid_reasons": [] if accepted else ["synthetic_ineligible"],
                "graph_range_valid": True, "all_targets_closed": True, "round_trip_valid": True})
    # Only record IDs in excluded metadata are inspected, never their payload paths.
    rows += [{"record_id": "held:" + split, "payload_path": "/forbidden/" + split} for split in ("validation", "test")]
    return splits, rows


def fixture(valid=None):
    return schedule.make_fixture_c2_train_schedule(*metadata(valid), reserved_bytes=477102080, synthetic=True)


def test_shared_all12_selection_exact_counts_signed_shift_and_metadata_namespace():
    plan = fixture()
    assert plan["mode"] == "synthetic_fixture" and plan["production_paths"] is None
    assert plan["selection"]["shared_all12_record_ids"] == ["an:test:a", "b"]
    assert plan["planned_sizing_jobs"] == 112 and plan["planned_sizing_record_draws"] == 224
    assert plan["optimizer_updates_executed"] == 0 and not plan["serialized_reservations_measured"]
    jobs = schedule.jobs_for_case(plan, "shift_coverage")
    assert len(jobs) == 12 and all(job.record_ids == ("an:test:a", "b") for job in jobs)
    for s, job in enumerate(jobs):
        assert job.position == s and job.draw_ids == (2*s, 2*s+1)
        assert job.schedule_shifts == job.applied_shifts == (s, (s+5) % 12)
    assert plan["signed_semitones_by_shift_pc"][6] == 6
    assert not any(rid.startswith("held:") for rid in plan["scheduled_train_record_ids"])


def test_fallback_uses_only_eligible_rows_and_is_order_independent():
    valid = {"a": [0, 1, 2, 3, 4, 5], "b": [0, 6, 7, 8, 9, 10, 11]}
    splits, rows = metadata(valid)
    first = schedule.make_fixture_c2_train_schedule(splits, rows, reserved_bytes=123, synthetic=True)
    second = schedule.make_fixture_c2_train_schedule(list(reversed(splits)), list(reversed(rows)), reserved_bytes=123, synthetic=True)
    # The source-file fingerprint legitimately changes with input byte order;
    # selection, table and actual stream must remain exactly identical.
    assert first["selection"] == second["selection"] and first["cases"] == second["cases"]
    assert first["selection"]["shared_all12_record_ids"] is None
    for job in schedule.jobs_for_case(first, "shift_coverage"):
        assert all(s in valid[rid] for rid, s in zip(job.record_ids, job.applied_shifts))


def test_real_sampler_prefix_preserves_repeated_draws_odd_epoch_boundary_and_resume():
    plan = fixture()
    rows = plan["train_records"]
    table = build_full_orbit_table({r["component_id"]: [r["record_id"]] for r in rows},
                                  {r["record_id"]: r["valid_shifts"] for r in rows})
    assert len(table) == 33
    actual = plan["cases"][1]
    sampler = FullOrbitSampler(table, seed=17)
    assert actual["draws"] == [asdict(sampler.peek(i)) for i in range(200)]
    assert sampler.position == 0 and actual["draws"][32]["orbit_epoch"] == 0 and actual["draws"][33]["orbit_epoch"] == 1
    assert len({(r["record_id"], r["shift_pc"]) for r in actual["draws"]}) < 200
    restored = FullOrbitSampler(table, seed=17)
    restored.load_state_dict(actual["sampler_states"]["after_applied_resume_prefix"])
    assert restored.position == 100 and [asdict(restored.peek(i)) for i in range(100)] == actual["draws"][100:]
    assert [case["start_draw_offset"] for case in plan["cases"]] == [0, 0]
    assert [case["resume_after_updates"] for case in plan["cases"]] == [5, 50]


@pytest.mark.parametrize("defect", ["duplicate_split", "duplicate_pair", "missing_pair", "invalid_shift", "signed6", "bool_shift", "identity_missing", "component", "nonbool", "contradiction", "outside_split", "no_coverage"])
def test_malformed_or_incomplete_metadata_fails(defect):
    splits, rows = metadata()
    if defect == "duplicate_split": splits.append(deepcopy(splits[0]))
    elif defect == "duplicate_pair": rows.append(deepcopy(rows[0]))
    elif defect == "missing_pair": rows.pop(3)
    elif defect == "invalid_shift": rows[0]["shift_pc"] = 12
    elif defect == "signed6": rows[6]["signed_semitones"] = -6
    elif defect == "bool_shift": rows[0]["shift_pc"] = False
    elif defect == "identity_missing": rows[0]["corrected_valid"] = False
    elif defect == "component": rows[0]["source_component_id"] = "wrong"
    elif defect == "nonbool": rows[0]["corrected_valid"] = 1
    elif defect == "contradiction": rows[0]["graph_range_valid"] = False
    elif defect == "outside_split": rows[0]["record_id"] = "unknown"
    else:
        for row in rows:
            if row.get("shift_pc") == 11: row["corrected_valid"] = False
    with pytest.raises(ValueError):
        schedule.make_fixture_c2_train_schedule(splits, rows, reserved_bytes=123, synthetic=True)


@pytest.mark.parametrize("defect", ["applied_identity", "coverage_selection", "draw_order", "cursor", "reservation", "foreign_record", "promotion"])
def test_coherently_resealed_descriptor_is_regenerated(defect):
    plan = fixture()
    if defect == "applied_identity": plan["cases"][0]["jobs"][1]["applied_shifts"] = [0, 0]
    elif defect == "coverage_selection": plan["selection"]["shared_all12_record_ids"].reverse()
    elif defect == "draw_order": plan["cases"][1]["draws"][0:2] = list(reversed(plan["cases"][1]["draws"][0:2]))
    elif defect == "cursor": plan["cases"][1]["sampler_states"]["after_applied_resume_prefix"]["position"] = 50
    elif defect == "reservation": plan["cases"][0]["jobs"][0]["reserved_bytes"] += 1
    elif defect == "foreign_record": plan["cases"][0]["jobs"][0]["record_ids"][0] = "held:test"
    else: plan["mode"] = "production"
    for case in plan["cases"]:
        case["fingerprint"] = schedule._fp({k: v for k, v in case.items() if k != "fingerprint"})
    plan["fingerprint"] = schedule._fp({k: v for k, v in plan.items() if k != "fingerprint"})
    with pytest.raises(ValueError): schedule.validate_c2_train_schedule(plan)


@pytest.mark.parametrize("split", ["validation", "test", "TRAIN", None])
def test_nontrain_gate_precedes_all_io(tmp_path, monkeypatch, split):
    def forbidden(*args, **kwargs): pytest.fail("forbidden metadata/payload read")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="train_only"):
        schedule.build_c2_train_schedule(None, metadata_bindings=None, reserved_bytes=123, split=split)


@pytest.fixture
def production_metadata(tmp_path, monkeypatch):
    valid = {f"record:{i:04}": range(12) for i in range(1295)}
    splits, rows = metadata(valid)
    # Actual frozen component population:1123 singletons+86 pairs=1209 components.
    mapping = {f"record:{i:04}": f"group:{i if i < 1123 else 1123+(i-1123)//2:04}" for i in range(1295)}
    for row in splits + rows:
        if row["record_id"] in mapping: row["source_component_id"] = mapping[row["record_id"]]
    b3 = tmp_path / "b3"; b3.mkdir()
    files = {"split_assignments": b3 / "split_assignments.jsonl", "shift_eligibility": tmp_path / "eligibility.jsonl"}
    for key, values in (("split_assignments", splits), ("shift_eligibility", rows)):
        files[key].write_bytes(b"".join(schedule._canonical(row) + b"\n" for row in values))
    bindings = {key: {"sha256": sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size} for key, path in files.items()}
    grouped = {}
    for rid, component in mapping.items(): grouped.setdefault(component, []).append(rid)
    monkeypatch.setattr(reference, "frozen_split_assignments", lambda _: {r["record_id"]: r for r in splits})
    monkeypatch.setattr(reference, "production_component_records", lambda _: {key: tuple(value) for key, value in grouped.items()})
    monkeypatch.setattr(reference, "production_valid_shifts", lambda _: {rid: tuple(range(12)) for rid in mapping})
    def forbidden(*args, **kwargs): pytest.fail("payload/model/CUDA access")
    monkeypatch.setattr(reference, "load_production_record", forbidden)
    monkeypatch.setattr(reference, "_resolve_selected_corpus_record", forbidden)
    paths = replace(reference.ProductionArtifactPaths(), b3_root=b3, b5a_shift_eligibility=files["shift_eligibility"],
        corpus_root=tmp_path / "forbidden-corpus", cache_root=tmp_path / "forbidden-cache")
    return paths, bindings, files


def test_production_metadata_reads_only_two_authorized_files_and_verifies(production_metadata, monkeypatch):
    paths, bindings, files = production_metadata
    original = Path.read_bytes; reads = []
    def observe(path):
        reads.append(path)
        assert path in files.values() or path.parent == Path(schedule.__file__).parent
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", observe)
    plan = schedule.build_c2_train_schedule(paths, metadata_bindings=bindings, reserved_bytes=123)
    assert plan["mode"] == "production" and len(plan["train_records"]) == 1295
    assert schedule.verify_c2_train_schedule(paths, plan) == plan
    assert all(reads.count(path) >= 4 for path in files.values())
    changed = deepcopy(bindings); changed["shift_eligibility"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="expected_metadata_mismatch"):
        schedule.validate_c2_train_schedule(plan, expected_metadata_bindings=changed)


@pytest.mark.parametrize("defect", ["hash", "size", "symlink", "reference_drift", "metadata_drift"])
def test_production_hash_paths_and_reference_changes_fail(production_metadata, monkeypatch, defect):
    paths, bindings, files = production_metadata
    if defect == "hash": bindings["shift_eligibility"]["sha256"] = "0" * 64
    elif defect == "size": bindings["shift_eligibility"]["size_bytes"] += 1
    elif defect == "symlink":
        link = files["shift_eligibility"].with_name("link.jsonl"); link.symlink_to(files["shift_eligibility"])
        paths = replace(paths, b5a_shift_eligibility=link)
    elif defect == "reference_drift": monkeypatch.setattr(reference, "production_valid_shifts", lambda _: {})
    else:
        original = reference.production_valid_shifts
        def drift(p):
            result = original(p)
            with files["shift_eligibility"].open("ab") as stream: stream.write(b"\n")
            return result
        monkeypatch.setattr(reference, "production_valid_shifts", drift)
    with pytest.raises(ValueError): schedule.build_c2_train_schedule(paths, metadata_bindings=bindings, reserved_bytes=123)


def test_explicit_fixture_and_reservation_are_required():
    with pytest.raises(ValueError, match="explicit_synthetic"):
        schedule.make_fixture_c2_train_schedule(*metadata(), reserved_bytes=123, synthetic=False)
    with pytest.raises(ValueError, match="explicit_reservation"):
        schedule.make_fixture_c2_train_schedule(*metadata(), reserved_bytes=True, synthetic=True)
