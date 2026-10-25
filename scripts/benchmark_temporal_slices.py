"""Four frozen TRAIN records, two batches of two, CPU forward/backward only.

Requires a planned registry record and a source manifest created before launch.
Never scans corpus directories or opens VALIDATION/TEST payloads. No optimizer
step, checkpoint loading, campaign dispatch or network access.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import resource
import time

import torch
from torch_geometric.data import Batch

from scripts import audit_harmonic_target_slots as prior
from music_critic.experiments.registry import validate_record
from music_critic.experiments.analysisgnn import corrected_training as old
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel
from music_critic.experiments.analysisgnn.temporal_slices import TemporalSliceModel, build_temporal_batch
from music_critic.experiments.analysisgnn.temporal_slices.alignment import align_temporal_targets, temporal_supervised_loss
from music_critic.graph import build_raw_graph

EXPERIMENT_ID = "EXP-TEMPORAL-SLICE-CPU-001"


def verify_sources(root):
    entries = json.loads((root / "source-manifest.json").read_text())
    for entry in entries:
        if prior.digest(entry["path"]) != entry["sha256"]:
            raise ValueError("implementation changed after source freeze: " + entry["path"])
    return len(entries)


def verified_artifact(record_path, locator):
    record = json.loads(Path(record_path).read_text())
    validate_record(record)
    if record["status"] != "completed":
        raise ValueError("requires completed parent")
    artifact = next(a for a in record["artifacts"] if a["locator"] == locator)
    path = Path(locator)
    if prior.digest(path) != artifact["sha256"] or path.stat().st_size != artifact["size_bytes"]:
        raise ValueError("parent artifact changed")
    return path


def tensor_bytes(graph):
    stores = graph.base.stores if hasattr(graph, "base") else graph.stores
    total = sum(v.numel()*v.element_size() for store in stores for v in store.values() if isinstance(v, torch.Tensor))
    if hasattr(graph, "slice_features"):
        total += graph.slice_features.numel()*graph.slice_features.element_size()
        total += sum(v.numel()*v.element_size() for v in graph.edges.values())
    return total


def run(root):
    record = json.loads(Path(f"docs/experiments/records/{EXPERIMENT_ID}.json").read_text())
    validate_record(record)
    if record["status"] != "planned" or record["protocol"]["records"] != 4:
        raise ValueError("missing planned four-record CPU gate")
    verify_sources(root)
    inventory_path = verified_artifact("docs/experiments/records/EXP-RAW-TIME-COVERAGE-001.json",
                                      "outputs/raw_time_coverage_001/run/pieces.jsonl")
    bindings_path = verified_artifact("docs/experiments/records/EXP-TARGET-SLOT-CENSUS-002.json",
                                     "outputs/target_slot_census_002/run/sources.jsonl")
    inventory = sorted((json.loads(l) for l in inventory_path.read_text().splitlines()),
                       key=lambda r: (r["inventory"]["raw_note_count"], r["record_id"]))
    selected = [inventory[int(q*(len(inventory)-1))] for q in (.25, .5, .75, .9)]
    frozen = {r["record_id"]: r for r in map(json.loads, bindings_path.read_text().splitlines())}
    assignments = {r["record_id"]: r for r in prior.train_assignments(
        Path("outputs/research_resume_20260913/EXP-RESEARCH-DILEMMA-BINDINGS-001/metadata/split_assignments.jsonl"))}
    corpus = Path("data/dilemmadata/dilemmadata-v1.0/johentsch-dilemmadata-d60ee75")
    torch.set_num_threads(1)
    weights = old.load_frozen_class_weights()
    results, sources = [], []
    started = time.perf_counter()
    for batch_index in range(2):
        pieces, sidecars = [], []
        for item in selected[batch_index*2:batch_index*2+2]:
            rid = item["record_id"]
            a = assignments[rid]
            path, _, _, _ = prior.selected_source(corpus, a)  # split gate precedes payload access
            bound = frozen[rid]
            if prior.digest(path) != bound["source_sha256"]:
                raise ValueError("physical source mismatch")
            accepted = prior.adapter.convert_dilemmadata_record(prior.bind_source(corpus, a))
            if not isinstance(accepted, prior.adapter.DilemmadataAccepted):
                raise ValueError("conversion rejected")
            piece, sidecar = accepted.piece, prior.materialize_target_sidecar(accepted, target_sidecar_version=prior.LEGACY_TARGET_SIDECAR_VERSION)
            piece_hash = sha256(prior.dumps_piece(piece).encode()).hexdigest()
            target_hash = sha256(json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if piece_hash != bound["piece_sha256"] or target_hash != bound["sidecar_sha256"]:
                raise ValueError("historical piece/sidecar mismatch")
            pieces.append(piece); sidecars.append(sidecar)
            sources.append({**bound, "verified": True, "notes": len(piece.notes)})
        historical = None
        for variant in ("c2", "temporal"):
            gc.collect()
            torch.manual_seed(17)
            model = CorrectedAnalysisGNNModel() if variant == "c2" else TemporalSliceModel()
            t = time.perf_counter()
            graph = Batch.from_data_list([build_raw_graph(p) for p in pieces]) if variant == "c2" else build_temporal_batch(pieces)
            build_seconds = time.perf_counter()-t
            t = time.perf_counter()
            output = model(graph)
            forward_seconds = time.perf_counter()-t
            t = time.perf_counter()
            aligned = (old.align_target_sidecars_after_prediction(output, graph, sidecars) if variant == "c2"
                       else align_temporal_targets(output, sidecars))
            loss = (old.corrected_supervised_loss if variant == "c2" else temporal_supervised_loss)(output, aligned, weights)
            alignment_loss_seconds = time.perf_counter()-t
            if loss.total is None or not torch.isfinite(loss.total):
                raise ValueError("nonfinite/absent loss")
            if not all(torch.isfinite(v).all() for v in output.logits.values()):
                raise ValueError("nonfinite logits")
            t = time.perf_counter()
            loss.total.backward()
            backward_seconds = time.perf_counter()-t
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("nonfinite gradients")
            if variant == "c2":
                historical = {h: r.valid_mask.clone() for h, r in aligned.heads.items()}
            else:
                for h in old.ACTIVE_HEADS:
                    if not torch.equal(historical[h], aligned.common_support_masks[h]):
                        raise ValueError("historical support changed: " + h)
            row = dict(batch=batch_index, variant=variant, build_seconds=build_seconds,
                       forward_seconds=forward_seconds, alignment_loss_seconds=alignment_loss_seconds,
                       backward_seconds=backward_seconds, graph_tensor_bytes=tensor_bytes(graph),
                       parameters=sum(p.numel() for p in model.parameters()),
                       candidate_rows={h: v.shape[0] for h, v in output.logits.items()},
                       valid_rows={h: int(r.valid_mask.sum()) for h, r in aligned.heads.items()},
                       finite_logits=True, finite_gradients=True)
            if variant == "temporal":
                row.update(slice_nodes=len(graph.keys["slice"]),
                           extension_edges=sum(v.shape[1] for v in graph.edges.values()),
                           graph_fingerprint=graph.fingerprint(),
                           fallback_rows={h: sum(v == "legacy_beat" and bool(aligned.heads[h].valid_mask[i])
                                                for i, v in enumerate(routes)) for h, routes in aligned.routes.items()})
            results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del loss, aligned, output, graph, model
        del pieces, sidecars
    count = verify_sources(root)
    result = dict(schema="TemporalSliceCPUSmoke@1.0.0", status="completed", records=4, batch_size=2,
                  seed=17, threads=1, optimizer_updates=0, source_entries_verified=count,
                  elapsed_seconds=time.perf_counter()-started, sources=sources, measurements=results,
                  process_maxrss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  historical_support_retained=True, gpu_access=False, validation_access=False, test_access=False,
                  limitation="Single un-warmed CPU pass per variant and batch; no quality or GPU throughput conclusion")
    with (root / "summary.json").open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
