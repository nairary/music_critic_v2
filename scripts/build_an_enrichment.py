"""Export pre-admitted native/current TRAIN AN sources without corpus discovery."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from fractions import Fraction
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import time
import traceback

from music_critic.adapters import dilemmadata as adapter
from music_critic.analysis.an_enrichment import build_an_enrichment
from music_critic.data import dumps_piece
from music_critic.tasks.event_theory_enrichment import build_vocabulary, encode_enrichment


def binding(path, root):
    path = Path(path); raw = path.read_bytes()
    return dict(path=str(path.resolve().relative_to(root.resolve())),
                sha256=sha256(raw).hexdigest(), size_bytes=len(raw))


def write_new(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_bytes(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected["sha256"] or len(raw) != expected["size_bytes"]:
        raise ValueError("bound file hash/size mismatch: " + str(path))
    return raw


def validate_admission(row, current, assignments, components, source_root):
    """Repeat metadata guards before any selected source musical read."""
    rid = row["record_id"]
    if (row["status"] != "admitted" or row["current_split"] != "train"
            or row["native_split"] != "training" or not rid.startswith("an:training:")):
        raise ValueError("native/current TRAIN required")
    expected = current[rid]
    if expected["split"] != "train" or assignments[rid]["split"] != "train":
        raise ValueError("current assignment is heldout")
    component = row["source_component_id"]
    if component != expected["source_component_id"] or component != assignments[rid]["source_component_id"]:
        raise ValueError("source component mismatch")
    members = components[component]["record_ids"]
    if members != row["component_record_ids"]:
        raise ValueError("component membership mismatch")
    if any(assignments.get(i, {}).get("split") != "train" for i in members):
        raise ValueError("component has current heldout")
    if any(i.startswith("an:") and not i.startswith("an:training:") for i in members):
        raise ValueError("component has native AN heldout")
    if row["canonical_expected"] != expected["piece"]:
        raise ValueError("canonical manifest binding mismatch")
    name = rid.split(":", 2)[2]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("unsafe source identity")
    source = source_root / "pitch_arrays/AN/training" / (name + "_joint.tsv")
    if Path(row["source_path"]).resolve() != source.resolve():
        raise ValueError("source locator mismatch")
    return source


def reconstruct(row, source):
    """Require historical raw projection and full canonical bytes, not an approximate match."""
    raw = source.read_bytes(); scan = adapter._parse_raw_file(source, "an_joint")
    if scan.raw_projection_sha256 != row["raw_projection_sha256"]:
        raise ValueError("historical raw projection mismatch")
    rid = row["record_id"]; name = rid.split(":", 2)[2]
    record = adapter.DilemmadataCorpusRecord(
        record_id=rid, piece_id=adapter._piece_id(rid, "an_joint"), dialect="an_joint",
        path=source.resolve(), relative_path="pitch_arrays/AN/training/" + source.name,
        collection=name.split("-", 1)[0], piece_name=name, suggested_split="training",
        physical_source_sha256=sha256(raw).hexdigest(), raw_projection_sha256=scan.raw_projection_sha256,
        raw_equivalence_id="dilemmadata-raw:" + scan.raw_projection_sha256,
        grouping_fingerprint=scan.grouping_fingerprint,
        source_group_id=row["source_component_id"],
        lineage_group_id=row["source_component_id"].replace("dilemmadata-component:", "dilemmadata-lineage:"),
        source_resolution=scan.source_resolution,
        score_path=None, score_relative_path=None, score_sha256=None,
        raw_issue_categories=scan.categories, note_row_count=scan.note_row_count,
        tie_continuation_row_count=scan.tie_continuation_row_count,
        zero_duration_row_count=scan.zero_duration_row_count,
        corpus_identity=adapter.DilemmadataCorpusIdentity(),
        record_binding_version=adapter.DILEMMADATA_RECORD_BINDING_VERSION, record_binding_sha256="")
    accepted = adapter.convert_dilemmadata_record(adapter._bind_record(record))
    if not isinstance(accepted, adapter.DilemmadataAccepted):
        raise ValueError("selected raw conversion quarantined: " + repr(accepted))
    piece_bytes = dumps_piece(accepted.piece).encode()
    expected = row["canonical_expected"]
    if sha256(piece_bytes).hexdigest() != expected["sha256"] or len(piece_bytes) != expected["size_bytes"]:
        raise ValueError("canonical bytes differ from current manifest")
    if source.read_bytes() != raw:
        raise ValueError("selected source changed during conversion")
    return accepted, piece_bytes, raw


def run(output, root):
    root = root.resolve(); output = output.resolve(); started = time.monotonic()
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("ignored output root required")
    record_path = root / "docs/experiments/records/EXP-MULTIGENRE-THEORY-ENRICHMENT-001.json"
    record = json.loads(record_path.read_text())
    if record["status"] not in ("planned", "running"):
        raise ValueError("active registered T07 campaign required")
    receipt = json.loads((output / "admission.json").read_text())
    if receipt["schema"] != "RemainingANMetadataAdmission@1":
        raise ValueError("metadata admission required")
    metadata = {Path(b["path"]).name: checked_bytes(root / b["path"], b)
                for b in receipt["metadata_bindings"]}
    current = {r["record_id"]: r for r in json.loads(metadata["dilemma-data-manifest.json"])["records"]}
    assignments = {r["record_id"]: r for r in map(json.loads, metadata["split_assignments.jsonl"].splitlines())}
    components = {r["source_component_id"]: r for r in map(json.loads, metadata["source_components.jsonl"].splitlines())}
    paper = {r["record_id"]: r for r in map(json.loads, metadata["paper_candidate_records.jsonl"].splitlines())}
    source_root = root / "data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75"
    rows = [r for r in receipt["rows"] if r["status"] == "admitted"]
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate admitted record")
    selected = []
    for row in rows:
        if row["raw_projection_sha256"] != paper[row["record_id"]]["raw_projection_sha256"]:
            raise ValueError("raw metadata binding mismatch")
        selected.append((row, validate_admission(row, current, assignments, components, source_root)))
    # Exclusive paths protect all earlier partial/completed outputs.
    (output / "pieces").mkdir(); (output / "source").mkdir()
    snapshot = []
    for name in ("scripts/build_an_enrichment.py", "src/music_critic/analysis/an_enrichment.py",
                 "src/music_critic/adapters/dilemmadata.py", "src/music_critic/tasks/event_theory_enrichment.py"):
        path = root / name; dest = output / "source" / path.name
        with dest.open("xb") as stream: stream.write(path.read_bytes())
        snapshot.append(binding(dest, root))
    write_new(output / "bindings.json", dict(metadata=receipt["metadata_bindings"],
        admission=binding(output / "admission.json", root), source_snapshot=snapshot,
        experiment_at_execution=binding(record_path, root),
        source_policy="Selected source snapshot hashes plus historical raw/canonical equality; no whole corpus identity re-audit.",
        historical_target_sidecars_reused=False))
    totals, fields, reasons = Counter(), Counter(), Counter()
    with (output / "index.jsonl").open("x") as index, (output / "numeric-verification.jsonl").open("x") as checks:
        for row, source in selected:
            rid = row["record_id"]; token = sha256(rid.encode()).hexdigest()[:24]
            entry = dict(record_id=rid, source_identity=rid, current_split="train", native_split="training",
                         source_component_id=row["source_component_id"], component_record_ids=row["component_record_ids"])
            try:
                accepted, piece_bytes, source_bytes = reconstruct(row, source)
                piece = accepted.piece
                canonical_path = output / "pieces" / (token + ".piece.json")
                with canonical_path.open("xb") as stream: stream.write(piece_bytes)
                alignment_path = output / "pieces" / (token + ".alignment.json")
                write_new(alignment_path, asdict(accepted.alignment_evidence))
                canonical_binding = binding(canonical_path, root)
                source_bindings = [dict(path=str(source.relative_to(root)), sha256=sha256(source_bytes).hexdigest(), size_bytes=len(source_bytes)), binding(alignment_path, root)]
                source_rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")), delimiter="\t"))
                provenance = next(p for p in piece.provenance if p.provenance_id == "prov:dilemmadata-conversion")
                offset = Fraction(dict(provenance.details).get("source_to_canonical_offset_qn", "0/1"))
                bundle = build_an_enrichment(piece, accepted.alignment_evidence, source_rows, row,
                    canonical_binding=canonical_binding, source_bindings=source_bindings, offset=offset)
                numeric = encode_enrichment(piece, bundle, build_vocabulary([bundle]))
                if numeric["oov_counts"]: raise ValueError("same-bundle vocabulary OOV")
                payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                path = output / "pieces" / (token + ".json.gz")
                with path.open("xb") as stream: stream.write(gzip.compress(payload, compresslevel=3, mtime=0))
                entry.update(status="completed", dataset_id=piece.dataset_name, piece_id=piece.piece_id,
                    canonical_piece=canonical_binding, enrichment=binding(path, root), coverage=bundle["coverage"],
                    source_bindings=source_bindings, historical_canonical_bytes_exact=True)
                check = dict(record_id=rid, piece_id=piece.piece_id, oov_counts=numeric["oov_counts"],
                    context_rows=len(numeric["note_contexts"]["ids"]), canonical_notes=len(piece.notes),
                    available_numeric_fields=sum(sum(v) for v in numeric["note_contexts"]["masks"].values()),
                    symbolic_available_fields=sum(bundle["coverage"]["available_by_field"].values()))
                if check["available_numeric_fields"] != check["symbolic_available_fields"]:
                    raise ValueError("numeric availability differs from symbolic fields")
                checks.write(json.dumps(check, separators=(",", ":")) + "\n"); checks.flush()
                totals["completed_records"] += 1
                totals.update({k:v for k,v in bundle["coverage"].items() if type(v) is int})
                fields.update(bundle["coverage"]["available_by_field"])
                reasons.update(bundle["coverage"]["unknown_by_reason"])
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc())
                totals["failed_records"] += 1
            index.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"); index.flush()
            if (totals["completed_records"] + totals["failed_records"]) % 10 == 0:
                print(json.dumps(dict(totals, elapsed_seconds=time.monotonic()-started)), flush=True)
    summary = dict(schema="RemainingANEnrichmentSummary@1", status="completed" if not totals["failed_records"] else "completed_with_piece_failures",
        counts=dict(totals), admitted_records=len(selected), excluded_records=receipt["counts"]["excluded"],
        excluded_reasons=receipt["reasons"], available_by_field=dict(fields), unknown_by_reason=dict(reasons),
        elapsed_seconds=time.monotonic()-started, observed_native_heldout_payload_reads=0,
        observed_current_heldout_payload_reads=0, shared_source_components_retained=True,
        native_note_contexts_are_not_exact_harmony_intervals=True,
        historical_target_sidecar_equivalence="not_asserted; current selected TSV source bytes separately hashed")
    write_new(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = run(args.output, args.root)
    print(json.dumps(result), flush=True)
    raise SystemExit(bool(result["counts"].get("failed_records", 0)))
