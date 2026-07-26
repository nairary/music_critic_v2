#!/usr/bin/env python3
"""Bounded CPU plumbing evidence for Phase 6A; never corpus feasibility."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

import mido
import torch

try:
    import resource
except ImportError:  # pragma: no cover - available on supported CI/Linux.
    resource = None

from music_critic.adapters import (
    HookTheoryAdapterConfig,
    Pop909ClCorpusRecord,
    convert_hooktheory_record,
    convert_pop909_cl_file,
)
from music_critic.graph import MANDATORY_EDGE_TYPES, MANDATORY_NODE_TYPES
from music_critic.models import (
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    perturb_canonical_note_pitch,
    single_note_sensitivity,
)
from music_critic.tasks import (
    collate_multisource_samples,
    prepare_multisource_sample,
)


def _hook_piece(index: int, *, include_targets: bool = True):
    clip_id = f"phase6a-hook-{index:03d}"
    return convert_hooktheory_record(
        clip_id,
        {
            "hash": clip_id,
            "split": "train",
            "json": {
                "endBeat": 5,
                "keys": [{"beat": 1, "tonic": "C", "scale": "major"}],
                "tempos": [{"beat": 1, "bpm": 120}],
                "meters": [{"beat": 1, "numBeats": 4, "beatUnit": 1}],
                "notes": [
                    {
                        "beat": 1,
                        "duration": 1,
                        "sd": "1",
                        "octave": 0,
                        "isRest": False,
                    }
                ],
                "chords": [
                    {
                        "beat": 1,
                        "duration": 2,
                        "root": 1,
                        "type": 5,
                        "inversion": 0,
                        "adds": [],
                        "omits": [],
                        "alterations": [],
                        "suspensions": [],
                        "borrowed": None,
                        "isRest": False,
                        "applied": 0,
                        "alternate": "",
                        "pedal": None,
                    }
                ],
            },
        },
        config=HookTheoryAdapterConfig(
            dataset_name="hooktheory", include_targets=include_targets
        ),
        structure_row={
            "audio_path": f"audio/{clip_id}.mp3",
            "ori_uid": f"phase6a-lineage-hook-{index:03d}",
        },
        source_path="4_merged.json",
    )


def _pop_piece(root: Path, index: int):
    root.mkdir(parents=True, exist_ok=True)
    song_id = f"{index + 1:03d}"
    path = root / f"{song_id}.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=4, denominator=4, time=0
                ),
                mido.MetaMessage("end_of_track", time=1_920),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", channel=0, program=0, time=0),
                mido.Message(
                    "note_on", channel=0, note=60 + index, velocity=80, time=0
                ),
                mido.Message(
                    "note_off",
                    channel=0,
                    note=60 + index,
                    velocity=0,
                    time=1_920,
                ),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    chord = mido.MidiTrack(
        [mido.Message("program_change", channel=1, program=0, time=0)]
    )
    for pitch in (60, 64, 67):
        chord.append(
            mido.Message("note_on", channel=1, note=pitch, velocity=70, time=0)
        )
    for ordinal, pitch in enumerate((60, 64, 67)):
        chord.append(
            mido.Message(
                "note_off",
                channel=1,
                note=pitch,
                velocity=0,
                time=1_920 if ordinal == 0 else 0,
            )
        )
    chord.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(chord)
    midi.save(path)
    result = convert_pop909_cl_file(
        Pop909ClCorpusRecord(
            song_id=song_id,
            path=path,
            relative_path=f"POP909_processed/{song_id}.mid",
            corpus_relative_path=f"{song_id}.mid",
            sha256=sha256(path.read_bytes()).hexdigest(),
            source_group_id=f"pop909-cl:{song_id}",
            lineage_group_id=f"pop909-lineage:{song_id}",
        )
    )
    if result.status != "accepted":
        raise RuntimeError("bounded POP fixture was not accepted")
    return result.piece


def make_bounded_batch(root: Path, repeats: int):
    pieces = []
    for index in range(repeats):
        hook = _hook_piece(index)
        pieces.extend((hook, _pop_piece(root, index)))
    raw = _hook_piece(repeats, include_targets=False)
    pieces.append(replace(raw, annotations=(), targets=()))
    return collate_multisource_samples(
        tuple(prepare_multisource_sample(piece) for piece in pieces)
    )


def _counts(batch) -> dict[str, object]:
    return {
        "graphs": int(batch.raw_graph_batch.num_graphs),
        "samples": len(batch.piece_ids),
        "nodes": sum(
            int(batch.raw_graph_batch[node_type].num_nodes)
            for node_type in MANDATORY_NODE_TYPES
        ),
        "edges": sum(
            int(batch.raw_graph_batch[edge_type].edge_index.shape[1])
            for edge_type in MANDATORY_EDGE_TYPES
        ),
    }


def benchmark_variant(
    batch, variant: str, *, gnn_layers: int = 2
) -> dict[str, object]:
    torch.manual_seed(101)
    config = LocalBaselineConfig(
        variant=variant,
        hidden_dim=32,
        gnn_layers=gnn_layers,
        dropout=0.0,
    )
    model = LocalHeterogeneousBaseline(config).cpu().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    start = perf_counter()
    output = model(batch)
    forward_seconds = perf_counter() - start
    if output.harmonic_loss.total_loss is None or output.reconstruction_loss is None:
        raise RuntimeError("bounded benchmark unexpectedly has no active loss")
    loss = output.harmonic_loss.total_loss + output.reconstruction_loss
    start = perf_counter()
    loss.backward()
    backward_seconds = perf_counter() - start
    start = perf_counter()
    optimizer.step()
    step_seconds = perf_counter() - start
    rss_peak_bytes = None
    if resource is not None:
        rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_peak_bytes = int(
            rss_raw if platform.system() == "Darwin" else rss_raw * 1024
        )
    return {
        "variant": variant,
        "config": config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **_counts(batch),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_step_seconds": step_seconds,
        "total_step_seconds": forward_seconds + backward_seconds + step_seconds,
        "active_task_count": len(output.harmonic_loss.task_losses),
        "prediction_task_count": len(output.predictions),
        "candidate_logit_rows": sum(
            int(task.logits.shape[0]) for task in output.predictions
        ),
        "supervision_rows": sum(
            int(item.per_row_loss.shape[0]) for item in output.supervisions
        ),
        "routing_operations": asdict(output.routing_operations),
        "rss_peak_bytes": rss_peak_bytes,
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
    }


def overfit_evidence(batch, steps: int) -> dict[str, object]:
    torch.manual_seed(103)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    first = model(batch)
    if (
        first.harmonic_loss.total_loss is None
        or first.reconstruction_loss is None
    ):
        raise RuntimeError("bounded overfit batch has no active loss")
    initial_harmonic = float(first.harmonic_loss.total_loss.detach())
    initial_reconstruction = float(first.reconstruction_loss.detach())
    initial_task_losses = {
        item.task_id: float(item.mean_loss.detach())
        for item in first.harmonic_loss.task_losses
    }
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        assert output.harmonic_loss.total_loss is not None
        assert output.reconstruction_loss is not None
        loss = output.harmonic_loss.total_loss + output.reconstruction_loss
        loss.backward()
        optimizer.step()
    final = model(batch)
    assert final.harmonic_loss.total_loss is not None
    assert final.reconstruction_loss is not None
    node_gradient_coverage = {
        node_type: any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.encoder.feature_encoder.node_encoders[
                node_type
            ].parameters()
        )
        for node_type in MANDATORY_NODE_TYPES
    }
    head_gradient_coverage = {
        spec.task_id: any(
            parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.task_heads.heads[
                f"task_{index:02d}"
            ].parameters()
        )
        for index, spec in enumerate(model.task_specs)
    }
    return {
        "steps": steps,
        "initial_harmonic_loss": initial_harmonic,
        "final_harmonic_loss": float(final.harmonic_loss.total_loss.detach()),
        "initial_reconstruction_loss": initial_reconstruction,
        "final_reconstruction_loss": float(final.reconstruction_loss.detach()),
        "initial_task_losses": initial_task_losses,
        "final_task_losses": {
            item.task_id: float(item.mean_loss.detach())
            for item in final.harmonic_loss.task_losses
        },
        "node_encoder_gradient_coverage": node_gradient_coverage,
        "task_head_gradient_coverage": head_gradient_coverage,
        "interpretation": "bounded plumbing/overfit evidence; not a scientific result",
    }


def candidate_evidence(batch, model) -> dict[str, object]:
    """Report raw candidates separately from the target supervision join."""

    output = model(batch, include_reconstruction=False)
    raw_only_sample = len(batch.piece_ids) - 1
    supervision_by_task = {
        item.task_id: item for item in output.supervisions
    }
    tasks = {}
    for prediction in output.predictions:
        tasks[prediction.task_id] = {
            "candidate_rows": int(prediction.logits.shape[0]),
            "candidate_rows_by_node_type": {
                node_type: int(
                    prediction.candidate_counts_by_node_type[index]
                )
                for index, node_type in enumerate(MANDATORY_NODE_TYPES)
                if int(prediction.candidate_counts_by_node_type[index]) > 0
            },
            "supervised_rows": (
                int(
                    supervision_by_task[
                        prediction.task_id
                    ].per_row_loss.shape[0]
                )
                if prediction.task_id in supervision_by_task
                else 0
            ),
            "raw_only_candidate_rows": int(
                (prediction.sample_indices == raw_only_sample).sum()
            ),
        }
    raw_piece = replace(
        _hook_piece(999, include_targets=False),
        annotations=(),
        targets=(),
    )
    raw_batch = collate_multisource_samples(
        (prepare_multisource_sample(raw_piece),)
    )
    raw_output = model(raw_batch, include_reconstruction=False)
    return {
        "tasks": tasks,
        "raw_only_prediction_task_count": len(raw_output.predictions),
        "raw_only_candidate_rows": sum(
            int(item.logits.shape[0]) for item in raw_output.predictions
        ),
        "raw_only_supervision_rows": len(raw_output.supervisions),
        "raw_only_harmonic_loss": (
            None
            if raw_output.harmonic_loss.total_loss is None
            else float(raw_output.harmonic_loss.total_loss.detach())
        ),
    }


def diagnostic_evidence() -> dict[str, object]:
    """Run the canonical perturbation through two production graph builds."""

    torch.manual_seed(109)
    model = LocalHeterogeneousBaseline(
        LocalBaselineConfig(hidden_dim=16, gnn_layers=2, dropout=0.0)
    ).cpu().eval()
    original = replace(_hook_piece(998), annotations=(), targets=())
    note_id = original.notes[0].note_id
    perturbed = perturb_canonical_note_pitch(original, note_id)
    return asdict(
        single_note_sensitivity(
            model,
            original,
            perturbed,
            note_id=note_id,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--larger-repeats", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.larger_repeats < 2 or args.overfit_steps <= 0:
        parser.error("larger-repeats must be >=2 and overfit-steps must be positive")
    with TemporaryDirectory(prefix="music-critic-phase6a-") as directory:
        root = Path(directory)
        tiny = make_bounded_batch(root / "tiny", 1)
        larger = make_bounded_batch(root / "larger", args.larger_repeats)
        torch.manual_seed(107)
        evidence_model = LocalHeterogeneousBaseline(
            LocalBaselineConfig(hidden_dim=16, gnn_layers=1, dropout=0.0)
        ).cpu().eval()
        report = {
            "scope": (
                "bounded CPU evidence only; no full-corpus training feasibility "
                "or quality claim"
            ),
            "benchmarks": {
                size: [
                    benchmark_variant(batch, variant)
                    for variant in ("feature_only", "local_gnn")
                ]
                for size, batch in (("tiny_mixed", tiny), ("larger_synthetic", larger))
            },
            "candidate_evidence": candidate_evidence(tiny, evidence_model),
            "canonical_single_note_diagnostic": diagnostic_evidence(),
            "overfit": overfit_evidence(tiny, args.overfit_steps),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
