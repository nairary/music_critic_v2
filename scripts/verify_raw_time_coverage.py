"""Verify a completed raw-time census from retained artifacts, without TSV reads.

This independent verifier does not import the census runner or candidate
generator. It reconstructs fixed candidate unions from saved raw time sources.
Full per-target rows are not retained: class counts and cross-head distinct
time counts therefore receive aggregate/bound checks, not a new label census.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
import time

POLICIES = ("note_onsets", "note_onsets_offsets", "note_onsets_beats",
            "note_onsets_offsets_beats")
TIME_HEADS = ("local_key", "tonicized_key", "root", "bass", "primary_degree",
              "secondary_degree", "quality", "inversion", "roman_numeral",
              "pitch_class_set", "harmonic_rhythm", "cadence", "pedal", "joint")
NOTE_HEADS = ("metrical_strength", "note_degree", "chord_tone", "is_root", "is_bass")
SPLIT_SHA = "056de0942918976981401db5017f6e567bc8949bae91ae395f0bd669adbfd51a"
FILES = ("summary.json", "pieces.jsonl", "sources.jsonl", "raw_candidates.jsonl.gz",
         "uncovered.jsonl", "eligible_old_unmapped.jsonl", "simultaneous_conflicts.jsonl")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return {"sha256": result.hexdigest(), "size_bytes": Path(path).stat().st_size}


def json_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream, 1):
            require(bool(line.strip()), f"blank JSONL row: {path}:{ordinal}")
            row = json.loads(line)
            require(isinstance(row, dict), f"non-object JSONL row: {path}:{ordinal}")
            yield row


def unique_index(rows, name):
    result = {}
    for row in rows:
        rid = row["record_id"]
        require(rid not in result, f"duplicate record in {name}: {rid}")
        result[rid] = row
    return result


def exact(value):
    require(isinstance(value, str), "time must be a canonical rational string")
    result = Fraction(value)
    require(str(result) == value, "time must be a canonical rational string")
    return result


def reconstruct_candidates(sources):
    require(set(sources) == {"note_onsets", "note_offsets", "beat_boundaries"},
            "raw time source keys differ")
    converted = {}
    for name, values in sources.items():
        times = [exact(value) for value in values]
        require(times == sorted(set(times)), f"raw times not sorted and unique: {name}")
        converted[name] = set(times)
    onset, offset, beat = (converted[k] for k in ("note_onsets", "note_offsets", "beat_boundaries"))
    return {POLICIES[0]: onset, POLICIES[1]: onset | offset,
            POLICIES[2]: onset | beat, POLICIES[3]: onset | offset | beat}


def counts(values, name):
    require(isinstance(values, dict), f"counter is not an object: {name}")
    require(all(type(v) is int and v >= 0 for v in values.values()),
            f"counter contains non-integer/negative values: {name}")
    return Counter(values)


def same_counts(left, right, name, ignored=()):
    a, b = counts(left, name), counts(right, name)
    for key in (set(a) | set(b)) - set(ignored):
        require(a[key] == b[key], f"counter mismatch {name}/{key}: {a[key]} != {b[key]}")


def check_coverage(head, variant, name):
    h, v = counts(head, name), counts(variant, name)
    s, historical, t = h["eligible_rows"], h["historical_valid_rows"], h["eligible_times"]
    require(s == h["source_available_rows"] - h["unknown_available_rows"], name + ": eligibility")
    require(0 <= historical <= s <= h["target_rows"], name + ": denominator bounds")
    require(0 <= t <= s and (t == 0) == (s == 0), name + ": time support")
    require(h["ideal_exact_errors"] <= s - t, name + ": ideal error bounds")
    require(v["covered_rows"] + v["uncovered_rows"] == s, name + ": covered + uncovered")
    require(v["covered_times"] + v["uncovered_times"] == t, name + ": time coverage")
    require(v["historical_kept_rows"] + v["historical_lost_rows"] == historical,
            name + ": historical retention")
    require(v["covered_rows"] == v["historical_kept_rows"] + v["newly_covered_rows"],
            name + ": newly covered difference")
    require(v["newly_covered_rows"] <= s - historical, name + ": newly covered bounds")
    require(v["historical_lost_rows"] <= v["uncovered_rows"], name + ": historical loss bounds")
    require(v["covered_times"] <= min(t, v["candidate_positions"], v["covered_rows"]),
            name + ": covered time bounds")
    require(v["uncovered_times"] <= v["uncovered_rows"], name + ": uncovered time bounds")
    require(v["covered_exact_errors"] <= min(h["ideal_exact_errors"],
            v["covered_rows"] - v["covered_times"]), name + ": covered conflict bounds")
    require(v["conflict_times"] <= v["covered_times"], name + ": conflict time bounds")
    require(v["historical_kept_exact_errors"] <= min(v["covered_exact_errors"],
            v["historical_kept_rows"]), name + ": historical conflict bounds")
    require(v["temporal_errors_removed_on_kept_rows"] ==
            v["historical_kept_old_slot_errors"] - v["historical_kept_exact_errors"],
            name + ": removed temporal errors")


def expected_selection(variants, inventory):
    qualifying = []
    for policy in POLICIES:
        if (all(variants[policy][h].get("historical_lost_rows", 0) == 0 for h in TIME_HEADS)
                and variants[policy]["cadence"].get("uncovered_rows", 0) == 0):
            qualifying.append(policy)
    selected = min(qualifying, key=lambda p: (inventory[p], POLICIES.index(p))) if qualifying else None
    return qualifying, selected


def _label(value):
    return tuple(value) if isinstance(value, list) else value


def _eligible(row, where):
    require(row["source_available"] is True and row["known"] is True and row["masked"] is False,
            where + ": diagnostic row not source eligible")
    exact(row["onset"])


def verify(run_root, bindings, parent_record, repo_root=Path.cwd()):
    started = time.perf_counter()
    summary = json.loads((run_root / "summary.json").read_text())
    require(summary.get("schema") == "RawTimeCoverage@1.0.0" and summary.get("status") == "completed",
            "only a completed RawTimeCoverage run may be verified")
    require(summary.get("completed_records") == summary.get("expected_records") == 1295
            and summary.get("failures") == [], "run is not complete for 1295 records")
    require(summary.get("split_sha256") == SPLIT_SHA, "run split hash mismatch")
    require(summary.get("validation_payload_accessed") is False and summary.get("test_payload_accessed") is False,
            "unexpected declared data access")
    inputs = {name: run_root / name for name in FILES}
    split_path = bindings / "metadata/split_assignments.jsonl"
    require(digest(split_path)["sha256"] == SPLIT_SHA, "frozen split bytes mismatch")
    inputs["frozen_split"] = split_path
    inputs["parent_record"] = parent_record
    parent = json.loads(parent_record.read_text())
    from music_critic.experiments.registry import validate_record
    validate_record(parent)
    require(parent["experiment_id"] == "EXP-TARGET-SLOT-CENSUS-002" and parent["status"] == "completed",
            "unexpected parent record")
    refs = {a["locator"]: a for a in parent["artifacts"]}
    require(len(refs) == len(parent["artifacts"]), "duplicate parent artifact locator")
    for name in ("sources.jsonl", "pieces.jsonl", "summary.json"):
        locator = "outputs/target_slot_census_002/run/" + name
        path = repo_root / locator
        observed = digest(path)
        require(all(observed[k] == refs[locator][k] for k in ("sha256", "size_bytes")),
                "parent artifact hash/size mismatch: " + name)
        inputs["parent_" + name] = path
    before = {name: digest(path) for name, path in inputs.items()}
    split = unique_index(json_rows(split_path), "frozen split")
    component_splits = {}
    for row in split.values():
        component = row["source_component_id"]
        require(component_splits.setdefault(component, row["split"]) == row["split"], "cross-split component")
    train = {rid: r for rid, r in split.items() if r["split"] == "train"}
    require(len(train) == 1295 and len({r["source_component_id"] for r in train.values()}) == 1209,
            "frozen TRAIN inventory mismatch")
    require(Counter(r["dialect"] for r in train.values()) == {"an_joint": 284, "dlc": 1011},
            "frozen TRAIN dialect mismatch")
    pieces = unique_index(json_rows(inputs["pieces.jsonl"]), "pieces")
    sources = unique_index(json_rows(inputs["sources.jsonl"]), "sources")
    parent_sources = unique_index(json_rows(inputs["parent_sources.jsonl"]), "parent sources")
    baseline = unique_index(json_rows(inputs["parent_pieces.jsonl"]), "parent pieces")
    for name, index in (("pieces", pieces), ("sources", sources), ("parent sources", parent_sources), ("baseline", baseline)):
        require(set(index) == set(train), name + ": TRAIN IDs mismatch")
    totals = {h: Counter() for h in TIME_HEADS}
    variants = {p: {h: Counter() for h in TIME_HEADS} for p in POLICIES}
    dialects = {d: {p: {h: Counter() for h in TIME_HEADS} for p in POLICIES} for d in ("an_joint", "dlc")}
    notes = {h: Counter() for h in NOTE_HEADS}
    inventory = Counter()
    for rid, piece in pieces.items():
        require(sources[rid] == parent_sources[rid], rid + ": parent source binding mismatch")
        require(sources[rid]["split"] == "train" and sources[rid]["component_id"] == train[rid]["source_component_id"],
                rid + ": source membership mismatch")
        require(piece["dialect"] == train[rid]["dialect"] == sources[rid]["dialect"], rid + ": dialect mismatch")
        require(set(piece["heads"]) == set(TIME_HEADS) and set(piece["variants"]) == set(POLICIES)
                and set(piece["unchanged_note_heads"]) == set(NOTE_HEADS), rid + ": head/policy set mismatch")
        inventory.update(counts(piece["inventory"], rid + "/inventory"))
        for h in NOTE_HEADS:
            same_counts(piece["unchanged_note_heads"][h], baseline[rid]["heads"][h], rid + "/" + h,
                        ignored=("pieces", "pieces_with_conflicts"))
            notes[h].update(piece["unchanged_note_heads"][h])
        for h in TIME_HEADS:
            c = counts(piece["heads"][h], rid + "/" + h)
            old = Counter(baseline[rid]["heads"][h])
            require(c["historical_valid_rows"] == old["valid_rows"], rid + "/" + h + ": historical support changed")
            if h != "joint":
                for new, previous in (("target_rows", "target_rows"), ("source_available_rows", "source_available_rows"),
                                      ("unknown_available_rows", "unknown_available_rows")):
                    require(c[new] == old[previous], rid + "/" + h + ": source support changed")
            else:
                require(c["target_rows"] == c["eligible_rows"], rid + ": joint not source intersection")
            totals[h].update(c)
            for p in POLICIES:
                require(set(piece["variants"][p]) == set(TIME_HEADS), rid + ": variant head set mismatch")
                v = counts(piece["variants"][p][h], rid + "/" + p + "/" + h)
                check_coverage(c, v, rid + "/" + p + "/" + h)
                require(v["candidate_positions"] == piece["inventory"][p], rid + ": candidate inventory mismatch")
                require(v["pieces_with_uncovered_rows"] == int(v["uncovered_rows"] > 0)
                        and v["pieces_with_historical_loss"] == int(v["historical_lost_rows"] > 0), rid + ": piece flags")
                require(v["historical_kept_old_slot_errors"] <= old["unavoidable_errors"], rid + ": old error bound")
                if not v["historical_lost_rows"]:
                    require(v["historical_kept_old_slot_errors"] == old["unavoidable_errors"], rid + ": retained old errors")
                variants[p][h].update(v)
                dialects[piece["dialect"]][p][h].update(v)
    same_counts(dict(inventory), summary["inventory"], "inventory sum")
    require(set(summary["heads"]) == set(TIME_HEADS) and set(summary["class_counts"]) == set(TIME_HEADS)
            and set(summary["variants"]) == set(POLICIES) and set(summary["by_dialect"]) == set(dialects)
            and set(summary["unchanged_note_heads"]) == set(NOTE_HEADS), "summary group set mismatch")
    for h in TIME_HEADS:
        same_counts(dict(totals[h]), summary["heads"][h], "head sum/" + h)
        class_sum = Counter()
        for label, value in summary["class_counts"][h].items():
            c = counts(value, h + "/class/" + label)
            require(set(c) <= {"eligible_rows", *(p + "_covered_rows" for p in POLICIES)}, "unexpected class counter")
            require(all(c[p + "_covered_rows"] <= c["eligible_rows"] for p in POLICIES), "class coverage bound")
            class_sum.update(c)
        require(class_sum["eligible_rows"] == totals[h]["eligible_rows"], h + ": class support sum")
        for p in POLICIES:
            same_counts(dict(variants[p][h]), summary["variants"][p][h], "variant sum/" + p + "/" + h)
            require(class_sum[p + "_covered_rows"] == variants[p][h]["covered_rows"], h + ": class covered sum")
            for d in dialects:
                same_counts(dict(dialects[d][p][h]), summary["by_dialect"][d][p][h], "dialect sum/" + d + "/" + p + "/" + h)
    for h in NOTE_HEADS:
        same_counts(dict(notes[h]), summary["unchanged_note_heads"][h], "note sum/" + h)
    diagnostics = {name: defaultdict(list) for name in ("uncovered.jsonl", "eligible_old_unmapped.jsonl", "simultaneous_conflicts.jsonl")}
    diagnostic_counts = Counter()
    for name, by_piece in diagnostics.items():
        unique = set()
        for row in json_rows(inputs[name]):
            rid, h = row["record_id"], row["head"]
            require(rid in train and h in TIME_HEADS, name + ": foreign row")
            identity = (rid, h, row["onset"] if name == "simultaneous_conflicts.jsonl" else row["entity"])
            require(identity not in unique, name + ": duplicate diagnostic row")
            unique.add(identity)
            by_piece[rid].append(row)
            diagnostic_counts[name] += 1
    seen = set()
    uncovered_class_counts = Counter()
    for raw in json_rows(inputs["raw_candidates.jsonl.gz"]):
        rid = raw["record_id"]
        require(rid in train and rid not in seen, "raw candidate record missing/duplicate/foreign")
        seen.add(rid)
        require(raw["schema"] == "RawTimeSources@1.0.0" and raw["piece_sha256"] == sources[rid]["piece_sha256"],
                rid + ": raw candidate source binding mismatch")
        candidates = reconstruct_candidates(raw["sources"])
        piece = pieces[rid]
        for p in POLICIES:
            require(len(candidates[p]) == piece["inventory"][p], rid + ": reconstructed inventory mismatch")
        require(len(raw["sources"]["note_onsets"]) <= piece["inventory"]["raw_note_count"]
                and len(raw["sources"]["note_offsets"]) <= piece["inventory"]["raw_note_count"]
                and len(raw["sources"]["beat_boundaries"]) <= 2 * piece["inventory"]["historical_beat_count"],
                rid + ": raw unique timing bounds")
        distinct = piece["distinct_eligible_times"]
        head_times = [piece["heads"][h]["eligible_times"] for h in TIME_HEADS]
        require(type(distinct) is int and max(head_times) <= distinct <= sum(head_times), rid + ": distinct eligible bounds")
        for p in POLICIES:
            covered = piece["covered_distinct_eligible_times"][p]
            hs = [piece["variants"][p][h].get("covered_times", 0) for h in TIME_HEADS]
            require(type(covered) is int and max(hs) <= covered <= min(sum(hs), distinct, len(candidates[p])),
                    rid + ": distinct covered bounds")
        missing, lost, missed_times = Counter(), Counter(), defaultdict(set)
        for row in diagnostics["uncovered.jsonl"][rid]:
            _eligible(row, rid)
            require(exact(row["onset"]) not in candidates[POLICIES[-1]], rid + ": false union miss")
            h = row["head"]
            missing[h] += 1; lost[h] += int(row["valid"])
            missed_times[h].add(row["onset"])
            value = row["label"]
            key = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
            uncovered_class_counts[(h, key)] += 1
        eligible_unmapped, newly = Counter(), Counter()
        for row in diagnostics["eligible_old_unmapped.jsonl"][rid]:
            _eligible(row, rid)
            require(row["valid"] is False, rid + ": old-unmapped diagnostic is historically valid")
            expected = [p for p in POLICIES if exact(row["onset"]) in candidates[p]]
            require(row["covered_by"] == expected, rid + ": recovered membership mismatch")
            eligible_unmapped[row["head"]] += 1
            for p in expected:
                newly[(p, row["head"])] += 1
        conflict_errors, conflict_covered, conflict_n, historical_exact = Counter(), Counter(), Counter(), Counter()
        for group in diagnostics["simultaneous_conflicts.jsonl"][rid]:
            h, t = group["head"], exact(group["onset"])
            rows = group["rows"]
            require(len({r["entity"] for r in rows}) == len(rows), rid + ": duplicate conflict entity")
            for row in rows:
                _eligible(row, rid)
                require(exact(row["onset"]) == t, rid + ": conflict mixes times")
            labels = Counter(_label(r["label"]) for r in rows)
            error = len(rows) - max(labels.values(), default=0)
            require(error > 0 and error == group["unavoidable_errors"], rid + ": conflict errors mismatch")
            historical_labels = Counter(_label(r["label"]) for r in rows if r["valid"])
            hist_error = sum(historical_labels.values()) - max(historical_labels.values(), default=0)
            conflict_errors[h] += error
            for p in POLICIES:
                if t in candidates[p]:
                    conflict_covered[(p, h)] += error; conflict_n[(p, h)] += 1
                    historical_exact[(p, h)] += hist_error
        for h in TIME_HEADS:
            c = Counter(piece["heads"][h]); full = Counter(piece["variants"][POLICIES[-1]][h])
            require(missing[h] == full["uncovered_rows"] and lost[h] == full["historical_lost_rows"]
                    and len(missed_times[h]) == full["uncovered_times"], rid + "/" + h + ": union miss diagnostic count")
            require(eligible_unmapped[h] == c["eligible_rows"] - c["historical_valid_rows"], rid + "/" + h + ": unmapped diagnostic count")
            require(conflict_errors[h] == c["ideal_exact_errors"], rid + "/" + h + ": full conflict diagnostic count")
            for p in POLICIES:
                v = Counter(piece["variants"][p][h])
                require(newly[(p, h)] == v["newly_covered_rows"], rid + ": newly covered diagnostic count")
                require(conflict_covered[(p, h)] == v["covered_exact_errors"]
                        and conflict_n[(p, h)] == v["conflict_times"]
                        and historical_exact[(p, h)] == v["historical_kept_exact_errors"], rid + ": retained conflict mismatch")
    require(seen == set(train), "raw candidate TRAIN coverage mismatch")
    require(all(h in summary["class_counts"] and label in summary["class_counts"][h]
                for h, label in uncovered_class_counts), "unknown uncovered diagnostic class")
    for h, labels in summary["class_counts"].items():
        for label, c in labels.items():
            require(uncovered_class_counts[(h, label)] == c.get("eligible_rows", 0) -
                    c.get(POLICIES[-1] + "_covered_rows", 0), h + ": uncovered class diagnostic mismatch")
    qualifying, selected = expected_selection(summary["variants"], summary["inventory"])
    require(summary["qualifying_policies"] == qualifying and summary["selected_policy"] == selected,
            "selected policy does not match independent qualification/minimum")
    expected_status = "candidate_only_not_a_trained_readout" if selected else "no_lossless_policy_selected"
    require(summary["selection_status"] == expected_status, "selection status mismatch")
    require(summary["historical_hash_matches"] == {"piece_equal": 1295, "sidecar_equal": 1295}, "historical hash match counts")
    for name, path in inputs.items():
        require(digest(path) == before[name], "artifact changed during verification: " + name)
    return {"schema": "RawTimeCoverageVerification@1.0.0", "status": "passed",
            "verified_records": len(seen), "split_sha256": SPLIT_SHA,
            "selected_policy": selected, "qualifying_policies": qualifying,
            "diagnostic_rows": dict(diagnostic_counts), "elapsed_seconds": time.perf_counter() - started,
            "input_artifacts": {name: {"path": str(inputs[name]), **value} for name, value in before.items()},
            "checks": ["TRAIN identity and parent metadata hashes", "per-piece/head/policy/dialect/note aggregate sums",
                       "source and historical coverage equations", "class aggregate sums and bounds",
                       "raw candidate union reconstruction and inventories", "distinct target time bounds",
                       "unmapped/missing/conflict rows and exact memberships", "minimum qualifying fixed policy",
                       "input artifact immutability during verification"],
            "limits": ["No raw TSV, canonical payload or target sidecar reread; source hashes are checked against retained parent metadata only.",
                       "Full eligible target rows are not retained: per-class counts and cross-head distinct times are checked through sums/bounds, not independently recounted from all targets.",
                       "Historical kept-slot error counts are bound to parent counts and algebraic invariants; partial-support slot membership cannot be rebuilt without full target rows.",
                       "Candidate generation from original raw notes is established by the runner; this verifier reconstructs unions from its retained raw timing manifest.",
                       "No model quality, VALIDATION/TEST coverage or inference correctness is established."],
            "raw_tsv_read": False, "checkpoint_or_model_execution": False, "validation_payload_read": False,
            "test_payload_read": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("docs/experiments/records/EXP-TARGET-SLOT-CENSUS-002.json"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "verification output already exists")
    result = verify(args.run, args.bindings, args.parent, args.repo_root)
    # Exclusive creation; never replace a previous verification or source artifact.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "verified_records", "selected_policy", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
