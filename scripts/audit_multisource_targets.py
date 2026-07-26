"""Build deterministic Phase 5A target-inventory and crosswalk evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from music_critic.adapters import (
    POP909_CL_ADAPTER_VERSION,
    HookTheoryAdapterConfig,
    convert_hooktheory_record,
)
from music_critic.data import SCHEMA_VERSION
from music_critic.graph import GRAPH_BUILDER_VERSION, GRAPH_SCHEMA_VERSION
from music_critic.tasks import (
    TARGET_FAMILIES,
    TARGET_ENCODING_REGISTRY_VERSION,
    TARGET_ONTOLOGY_VERSION,
    ontology_contract_dict,
    ontology_contract_fingerprint,
    project_multisource_targets,
    target_encoding_contract_dict,
    target_encoding_contract_fingerprint,
)


AUDIT_SCHEMA_VERSION = "1.1.0"
_HOOK_FIXTURE_MANIFEST = Path("tests/fixtures/hooktheory/golden_manifest.json")
_HOOK_CASE_ROOT = Path("tests/fixtures/hooktheory/cases")
_POP_MANIFEST = Path("tests/fixtures/pop909_cl/production_manifest.json")
_CONTRACT_SOURCES = (
    Path("src/music_critic/adapters/hooktheory.py"),
    Path("src/music_critic/adapters/pop909_cl.py"),
    Path("src/music_critic/data/schema.py"),
    Path("src/music_critic/graph/builder.py"),
    Path("src/music_critic/graph/validation.py"),
    Path("src/music_critic/tasks/ontology.py"),
    Path("src/music_critic/tasks/encoding.py"),
    Path("src/music_critic/tasks/alignment.py"),
    Path("src/music_critic/tasks/collator.py"),
    Path("src/music_critic/tasks/multisource.py"),
    Path("scripts/benchmark_multisource_collator.py"),
    Path("scripts/audit_multisource_targets.py"),
    Path("docs/HARMONIC_SUPERVISION.md"),
    Path("docs/MULTISOURCE_COLLATOR.md"),
    Path("docs/MULTISOURCE_TARGET_CONTRACT.md"),
    Path("docs/POP909_CL_ADAPTER_CONTRACT.md"),
    _HOOK_FIXTURE_MANIFEST,
    _POP_MANIFEST,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _number(value: object) -> Fraction | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None


def _bounded_end_beat(raw_excerpt: Mapping[str, Any]) -> int | float:
    end = Fraction(5)
    for collection in ("notes", "chords"):
        values = raw_excerpt.get(collection, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for indexed in values:
            if not isinstance(indexed, Mapping):
                continue
            value = indexed.get("value")
            if not isinstance(value, Mapping):
                continue
            beat = _number(value.get("beat"))
            duration = _number(value.get("duration"))
            if beat is not None:
                end = max(end, beat + (duration or Fraction(0)))
    return end.numerator if end.denominator == 1 else float(end)


def _hook_record(case: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    reference = case["source_reference"]
    clip_id = reference["clip_id"]
    raw_excerpt = case["raw_excerpt"]
    regions = raw_excerpt.get("regions", {})
    payload = {
        "endBeat": _bounded_end_beat(raw_excerpt),
        "keys": list(regions.get("keys", ())),
        "tempos": list(regions.get("tempos", ())),
        "meters": list(regions.get("meters", ())),
        "notes": [
            item["value"]
            for item in raw_excerpt.get("notes", ())
            if isinstance(item, Mapping) and isinstance(item.get("value"), Mapping)
        ],
        "chords": [
            item["value"]
            for item in raw_excerpt.get("chords", ())
            if isinstance(item, Mapping) and isinstance(item.get("value"), Mapping)
        ],
    }
    record = {"hash": clip_id, "split": reference["split"], "json": payload}
    structure = case.get("structure_excerpt")
    structure_row = None
    if isinstance(structure, Mapping):
        structure_row = {
            "audio_path": f"audio/{clip_id}.mp3",
            "ori_uid": structure.get("ori_uid"),
            "segment_start": structure.get("segment_start"),
            "segment_end": structure.get("segment_end"),
        }
    return clip_id, record, structure_row


def _empty_stats(task_id: str) -> dict[str, Any]:
    spec = next(item for item in TARGET_FAMILIES if item.task_id == task_id)
    return {
        "task_id": task_id,
        "canonical_dtype": spec.canonical_dtype,
        "value_type": spec.value_type,
        "vocabulary": list(spec.vocabulary) if spec.vocabulary is not None else None,
        "open_vocabulary": spec.open_vocabulary,
        "time_unit": spec.time_unit,
        "interval_semantics": spec.interval_semantics,
        "available": 0,
        "masked": 0,
        "observed_values": [],
        "provenance_required": spec.provenance_required,
    }


def _hooktheory_inventory(repo_root: Path) -> dict[str, Any]:
    manifest = _load_json(repo_root / _HOOK_FIXTURE_MANIFEST)
    task_stats = {
        spec.task_id: _empty_stats(spec.task_id)
        for spec in TARGET_FAMILIES
        if spec.source_adapter == "music_critic.adapters.hooktheory"
    }
    values: dict[str, set[str]] = {task_id: set() for task_id in task_stats}
    diagnostics: Counter[str] = Counter()
    converted = 0
    skipped: list[str] = []
    for case_id in manifest["cases"]:
        case = _load_json(repo_root / _HOOK_CASE_ROOT / f"{case_id}.json")
        if case["raw_excerpt"].get("json_present") is False:
            skipped.append(case_id)
            continue
        clip_id, record, structure_row = _hook_record(case)
        piece = convert_hooktheory_record(
            clip_id,
            record,
            config=HookTheoryAdapterConfig(dataset_name="hooktheory"),
            structure_row=structure_row,
            source_path="4_merged.json",
        )
        sample = project_multisource_targets(piece)
        converted += 1
        diagnostics.update(flag.code for flag in sample.diagnostics)
        for target in sample.target_bundle:
            stats = task_stats[target.task_id]
            stats["available"] += sum(target.availability_mask)
            stats["masked"] += len(target.availability_mask) - sum(
                target.availability_mask
            )
            for value, available in zip(target.values, target.availability_mask):
                if not available:
                    continue
                if isinstance(value, tuple):
                    values[target.task_id].update(str(item) for item in value)
                else:
                    values[target.task_id].add(str(value))
    for task_id, stats in task_stats.items():
        stats["observed_values"] = sorted(values[task_id])
    return {
        "evidence_scope": "19 bounded real-source golden excerpts; no corpus scan",
        "fixture_schema_version": manifest["fixture_schema_version"],
        "fixture_cases": len(manifest["cases"]),
        "converted_cases": converted,
        "skipped_unusable_cases": skipped,
        "target_inventory": [task_stats[key] for key in sorted(task_stats)],
        "diagnostic_counts": dict(sorted(diagnostics.items())),
    }


def _pop909_cl_inventory(repo_root: Path) -> dict[str, Any]:
    manifest = _load_json(repo_root / _POP_MANIFEST)
    expected = manifest["expected"]
    block_total = expected["chord_blocks"] + expected["accepted_missing_targets"]
    counts = {
        "pop909_cl.chord.bass": (
            expected["bass_available"],
            block_total - expected["bass_available"],
        ),
        "pop909_cl.chord.boundary": (
            expected["boundary_available"],
            block_total - expected["boundary_available"],
        ),
        "pop909_cl.chord.inversion": (
            expected["inversion_available"],
            block_total - expected["inversion_available"],
        ),
        "pop909_cl.chord.quality": (
            expected["quality_available"],
            block_total - expected["quality_available"],
        ),
        "pop909_cl.chord.root": (
            expected["root_available"],
            block_total - expected["root_available"],
        ),
        "pop909_cl.chord.no_chord": (
            expected["derived_n_spans"],
            expected["trailing_masked_spans"]
            + expected["accepted_missing_targets"],
        ),
    }
    inventory = []
    for task_id in sorted(counts):
        stats = _empty_stats(task_id)
        stats["available"], stats["masked"] = counts[task_id]
        stats["observed_values"] = None
        inventory.append(stats)
    return {
        "evidence_scope": (
            "accepted production manifest aggregates from the prior streaming "
            "909-file acceptance; no Phase 5A corpus rescan"
        ),
        "manifest_version": manifest["acceptance_schema_version"],
        "adapter_version": manifest["adapter_version"],
        "logical_files": expected["logical_files"],
        "accepted": expected["accepted"],
        "quarantined": expected["quarantined"],
        "ambiguous_blocks": expected["ambiguous_blocks"],
        "unsupported_blocks": expected["unsupported_blocks"],
        "target_inventory": inventory,
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    """Build the deterministic bounded evidence report."""

    root = repo_root.resolve()
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "versions": {
            "canonical_schema": SCHEMA_VERSION,
            "graph_schema": GRAPH_SCHEMA_VERSION,
            "graph_builder": GRAPH_BUILDER_VERSION,
            "target_ontology": TARGET_ONTOLOGY_VERSION,
            "target_encoding_registry": TARGET_ENCODING_REGISTRY_VERSION,
            "hooktheory_adapter": None,
            "hooktheory_contract_phase": "2B.1",
            "pop909_cl_adapter": POP909_CL_ADAPTER_VERSION,
        },
        "ontology_fingerprint": ontology_contract_fingerprint(),
        "ontology": ontology_contract_dict(),
        "target_encoding_fingerprint": target_encoding_contract_fingerprint(),
        "target_encoding": target_encoding_contract_dict(),
        "source_inventories": {
            "hooktheory": _hooktheory_inventory(root),
            "pop909_cl": _pop909_cl_inventory(root),
        },
        "contract_source_sha256": {
            path.as_posix(): _file_sha256(root / path) for path in _CONTRACT_SOURCES
        },
        "scan_policy": {
            "manual_corpus_file_reads": False,
            "hooktheory_full_corpus_scan": False,
            "pop909_cl_full_acceptance_rerun": False,
        },
    }


def dumps_report(report: Mapping[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def report_fingerprint(report: Mapping[str, Any]) -> str:
    return sha256(dumps_report(report).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    report = build_report(args.repo_root)
    payload = dumps_report(report, indent=2) + "\n"
    if args.check is not None:
        if args.check.read_text(encoding="utf-8") != payload:
            raise SystemExit(f"artifact differs from deterministic audit: {args.check}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    elif args.check is None:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "build_report",
    "dumps_report",
    "main",
    "report_fingerprint",
]
