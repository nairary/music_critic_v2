"""Synthetic CPU smoke CLI. Without --smoke, no checkpoint or source is read."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import tempfile

import torch

from music_critic.adapters.pop909_cl import (
    Pop909ClAdapterConfig, Pop909ClCorpusRecord, convert_pop909_cl_file, inspect_pop909_cl_instruments,
    project_pop909_cl_score_bytes, pop909_cl_raw_input_group_id, pop909_cl_piece_id,
)
from music_critic.adapters.midi import MidiAdapterConfig, load_midi_piece
from music_critic.adapters.pop909_cl_synthetic import build_pop909_cl_smoke_midi
from music_critic.experiments.analysisgnn.corrected_model import CorrectedAnalysisGNNModel

from .checkpoint import create_model_only_payload, load_model_checkpoint
from .contracts import canonical_json, guard_execution
from .evaluation import EvaluationPiece, evaluate_from_readers, targets_from_chord_evidence


def synthetic_input(directory: Path) -> EvaluationPiece:
    """Generate a small channel-0 polyphonic score and target-only channel 1."""
    midi = build_pop909_cl_smoke_midi()
    path = directory / "001.mid"
    midi.save(path)
    resolution = inspect_pop909_cl_instruments(midi)
    score_bytes = project_pop909_cl_score_bytes(midi, resolution)
    score_sha = sha256(score_bytes).hexdigest()
    record = Pop909ClCorpusRecord(
        song_id="001", path=path, relative_path="POP909_processed/001.mid", corpus_relative_path="001.mid",
        sha256=sha256(path.read_bytes()).hexdigest(), source_group_id=pop909_cl_raw_input_group_id(score_sha),
        lineage_group_id="pop909-lineage:001",
    )
    score_path = directory / "score-only.mid"
    score_path.write_bytes(score_bytes)
    piece = load_midi_piece(score_path, config=MidiAdapterConfig(
        dataset_name="pop909_cl", piece_id=pop909_cl_piece_id(record.song_id),
        source_group_id=record.source_group_id,
    ))

    def read_targets():
        # Source normalization and target extraction happen after raw inference.
        accepted = convert_pop909_cl_file(record, config=Pop909ClAdapterConfig(include_targets=False))
        return targets_from_chord_evidence(accepted.chord_evidence)

    return EvaluationPiece(piece, record.source_group_id, read_targets)


def run_smoke(*, split: str = "synthetic") -> dict:
    guard_execution(split)
    with tempfile.TemporaryDirectory(prefix="pop909-frozen-smoke-") as temporary:
        directory = Path(temporary)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = CorrectedAnalysisGNNModel().eval()
        path = directory / "synthetic-model.pt"
        torch.save(create_model_only_payload(model), path)
        payload = path.read_bytes()
        report = evaluate_from_readers(
            split=split,
            checkpoint_reader=lambda: load_model_checkpoint(
                path, expected_sha256=sha256(payload).hexdigest(), expected_size_bytes=len(payload)),
            source_reader=lambda: (synthetic_input(directory),),
        )
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="generate and evaluate only temporary synthetic data/weights")
    parser.add_argument("--split", default="synthetic", help="synthetic only; TEST is always denied before reads")
    parser.add_argument("--output", type=Path, help="new JSON report file; default stdout")
    args = parser.parse_args(argv)
    guard_execution(args.split)
    if not args.smoke:
        parser.print_help()
        return 0
    report = run_smoke(split=args.split)
    serialized = canonical_json(report)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
