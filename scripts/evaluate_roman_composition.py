#!/usr/bin/env python3
"""CPU-only paired decoder evaluation of an existing corrected C2 checkpoint.

A TRAIN-frozen oracle audit must finish first. The same aligned VALIDATION
events are used for every decoder. No training, calibration, threshold search,
optimizer, saved RNG restore, cloud mutation, or TEST input is involved.
"""
import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

import torch

from music_critic.data import loads_piece
from music_critic.tasks import prepare_multisource_sample, collate_multisource_samples
from music_critic.experiments.analysisgnn.corrected_training import align_target_sidecars_after_prediction
from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.analysisgnn.roman_composition import classification_metrics, lookup_predict
from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
from scripts.audit_roman_composition import checked_bytes, checked_member


TASKS = ("primary_degree", "quality", "roman_numeral")


def read_validation(package, manifest_sha256, receipt_sha256):
    manifest = json.loads(checked_bytes(package / "data-manifest.json", manifest_sha256))
    receipt = json.loads(checked_bytes(package / "receipt.json", receipt_sha256))
    if receipt["relocated_manifest_sha256"] != manifest_sha256 or receipt["test_payload_packaged"] is not False:
        raise ValueError("untrusted package")
    if any(r["split"] not in ("train", "validation") for r in manifest["records"]):
        raise ValueError("TEST is forbidden")
    records = [r for r in manifest["records"] if r["split"] == "validation"]
    if len(records) != 162 or len({r["record_id"] for r in records}) != 162:
        raise ValueError("full frozen VALIDATION required")
    expected = {}
    root = PurePosixPath(receipt["destination"])
    for record in records:
        for kind in ("piece", "sidecar"):
            name = str(PurePosixPath(record[kind]["path"]).relative_to(root))
            checked_member(name)
            if name in expected:
                raise ValueError("duplicate binding")
            expected[name] = (record, kind)
    pending, seen, complete = {}, set(), set()
    for part in receipt["parts"]:
        checked_member(part["path"])
        path = package / part["path"]
        checked_bytes(path, part["sha256"], part["size_bytes"])
        with tarfile.open(path, mode="r|gz") as archive:
            for member in archive:
                checked_member(member.name)
                if member.name in seen or not member.isfile():
                    raise ValueError("duplicate or nonregular archive member")
                seen.add(member.name)
                if member.name not in expected:
                    continue
                record, kind = expected[member.name]
                raw = archive.extractfile(member).read()
                if sha256(raw).hexdigest() != record[kind]["sha256"] or len(raw) != record[kind]["size_bytes"]:
                    raise ValueError("payload hash/size mismatch")
                identity = record["record_id"]
                pending.setdefault(identity, {})[kind] = raw
                if len(pending[identity]) == 2:
                    values = pending.pop(identity)
                    sidecar = json.loads(values["sidecar"])
                    if (sidecar["record_id"] != identity or
                            sidecar["source_component_id"] != record["source_component_id"]):
                        raise ValueError("sidecar identity mismatch")
                    piece = loads_piece(values["piece"].decode())
                    if piece.split not in (None, "validation") or piece.source_group_id != record["source_component_id"]:
                        raise ValueError("piece identity/split mismatch")
                    if piece.annotations or piece.targets:
                        raise ValueError("raw-only piece required")
                    complete.add(identity)
                    yield record, piece, sidecar
    if expected.keys() - seen or pending or len(complete) != 162:
        raise ValueError("incomplete validation")


def fit_from_rows(path, expected_sha256):
    # Stream twice: authenticate bytes first, then fit only explicit TRAIN rows.
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("harmonic rows hash mismatch")
    table = defaultdict(Counter)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] not in ("train", "validation"):
                raise ValueError("TEST is forbidden")
            if row["split"] != "train":
                continue
            values = [row["labels"].get(t) for t in TASKS]
            if all(isinstance(value, str) for value in values):
                table[tuple(values[:2])][values[2]] += 1
    return dict(table)


def compatibility_tensor(table, vocabularies):
    values = torch.zeros(*(len(vocabularies[t]) for t in TASKS), dtype=torch.float64)
    for (degree, quality), counts in table.items():
        for numeral, count in counts.items():
            values[vocabularies[TASKS[0]].index(degree), vocabularies[TASKS[1]].index(quality),
                   vocabularies[TASKS[2]].index(numeral)] = count
    return values / values.sum((0, 1)).clamp_min(1)


def decode_probabilities(probabilities, table, vocabularies, compatibility):
    p_degree, p_quality, p_roman = [probabilities[t] for t in TASKS]
    scores = torch.einsum("nd,nq,dqr->nr", p_degree.double(), p_quality.double(), compatibility)
    soft_ids = scores.argmax(1).tolist()
    hybrid_ids = (scores * p_roman.double()).argmax(1).tolist()
    supported = scores.sum(1).gt(0).tolist()
    labels = vocabularies["roman_numeral"]
    return dict(
        direct=[labels[i] for i in p_roman.argmax(1).tolist()],
        hard_lookup=[lookup_predict(table, (vocabularies["primary_degree"][d], vocabularies["quality"][q]))
                     for d, q in zip(p_degree.argmax(1).tolist(), p_quality.argmax(1).tolist(), strict=True)],
        soft_components=[labels[i] if valid else None for i, valid in zip(soft_ids, supported, strict=True)],
        direct_times_components=[labels[i] if valid else None for i, valid in zip(hybrid_ids, supported, strict=True)],
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("package", "rows", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("manifest-sha256", "receipt-sha256", "rows-sha256", "checkpoint-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint-size", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    table = fit_from_rows(args.rows, args.rows_sha256)
    vocabs = {t: get_vocabulary(TASK_BY_ID[t].vocabulary_id).labels for t in TASKS}
    compat = compatibility_tensor(table, vocabs)
    loaded = load_model_checkpoint(args.checkpoint, expected_sha256=args.checkpoint_sha256,
                                   expected_size_bytes=args.checkpoint_size)
    model = loaded.model
    before = model_state_fingerprint(model)
    all_targets, all_predictions, groups, record_metrics = [], defaultdict(list), [], []
    masked = Counter()
    with gzip.open(args.output / "paired-predictions.jsonl.gz", "wt") as sink:
        for number, (record, piece, sidecar) in enumerate(read_validation(
                args.package, args.manifest_sha256, args.receipt_sha256), start=1):
            graph = collate_multisource_samples((prepare_multisource_sample(piece),)).raw_graph_batch
            with torch.inference_mode():
                output = model.predict(graph)
            alignment = align_target_sidecars_after_prediction(output, graph, (sidecar,))
            aligned = {t: alignment.heads[t] for t in TASKS}
            ids = aligned["roman_numeral"].entity_ids
            if any(aligned[t].entity_ids != ids for t in TASKS):
                raise ValueError("head identities differ")
            valid = aligned["roman_numeral"].valid_mask
            # Prediction alignment, not availability of gold component labels,
            # determines decoder eligibility. Harmonic heads share beat routing.
            for task in TASKS:
                if not torch.equal(aligned[task].candidate_indices, aligned["roman_numeral"].candidate_indices):
                    raise ValueError("harmonic candidate routing differs")
            masked["roman_invalid_or_unaligned"] += int((~valid).sum())
            probabilities = {t: output.logits[t][aligned[t].candidate_indices[valid]].float().softmax(-1).cpu()
                             for t in TASKS}
            predictions = decode_probabilities(probabilities, table, vocabs, compat)
            targets = [vocabs["roman_numeral"][i] for i in aligned["roman_numeral"].values[valid].tolist()]
            selected_ids = [identity for identity, keep in zip(ids, valid.tolist(), strict=True) if keep]
            for i, identity in enumerate(selected_ids):
                sink.write(json.dumps(dict(record_id=record["record_id"], component_id=record["source_component_id"],
                    entity_id=identity, target=targets[i], predictions={k: v[i] for k, v in predictions.items()},
                    probabilities={t: p[i].tolist() for t, p in probabilities.items()}), separators=(",", ":")) + "\n")
            all_targets.extend(targets)
            groups.extend([record["source_component_id"]] * len(targets))
            for mode, values in predictions.items():
                all_predictions[mode].extend(values)
            record_metrics.append(dict(record_id=record["record_id"], events=len(targets),
                elapsed_seconds=time.monotonic() - started, predictions={mode: classification_metrics(
                    targets, values, vocabs["roman_numeral"]) for mode, values in predictions.items()} if targets else {}))
            print(json.dumps(dict(records=number, valid_events=len(all_targets), elapsed_seconds=time.monotonic()-started)), flush=True)
    if before != model_state_fingerprint(model):
        raise ValueError("model state changed")
    result = dict(checkpoint=loaded.evidence, elapsed_seconds=time.monotonic()-started,
                  masked=dict(masked), validation_records=len(record_metrics), test_accessed=False,
                  metrics={mode: classification_metrics(all_targets, values, vocabs["roman_numeral"])
                           for mode, values in all_predictions.items()}, record_metrics=record_metrics)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
