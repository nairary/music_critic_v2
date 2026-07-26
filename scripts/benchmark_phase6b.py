#!/usr/bin/env python3
"""Bounded CPU evidence for Phase 6B; never corpus feasibility or quality."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

try:
    from scripts.benchmark_phase6a import (
        _counts,
        _hook_piece,
        benchmark_variant,
        make_bounded_batch,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from benchmark_phase6a import (  # type: ignore[no-redef]
        _counts,
        _hook_piece,
        benchmark_variant,
        make_bounded_batch,
    )
from music_critic.graph import MANDATORY_NODE_TYPES
from music_critic.models import (
    HierarchicalBaselineConfig,
    HierarchicalHeterogeneousBaseline,
    LocalBaselineConfig,
    LocalHeterogeneousBaseline,
    extract_hierarchy_ownership,
    hierarchical_single_note_sensitivity,
    perturb_canonical_note_pitch,
)


def _hierarchical_config(hidden_dim: int = 32):
    return HierarchicalBaselineConfig(
        hidden_dim=hidden_dim,
        local_gnn_layers=1,
        transformer_layers=1,
        attention_heads=4,
        ffn_multiplier=4,
        dropout=0.0,
    )


def benchmark_hierarchical_variant(batch) -> dict[str, object]:
    """Time inspectable hierarchy stages and one complete CPU training step."""

    torch.manual_seed(907)
    model = HierarchicalHeterogeneousBaseline(
        _hierarchical_config()
    ).cpu().train()
    graph = batch.raw_graph_batch
    start = perf_counter()
    local = model.local_baseline.encode(graph)
    local_seconds = perf_counter() - start
    start = perf_counter()
    ownership = extract_hierarchy_ownership(graph, local.final_output)
    pooling = model.context_encoder.pooling(local.final_output, ownership)
    pooling_seconds = perf_counter() - start
    start = perf_counter()
    coarse = model.context_encoder.transformer(
        local.final_output, pooling
    )
    transformer_seconds = perf_counter() - start
    start = perf_counter()
    fused = model.context_encoder.fusion(
        local.final_output, coarse, ownership
    )
    fusion_seconds = perf_counter() - start
    del fused

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    start = perf_counter()
    output = model(batch)
    forward_seconds = perf_counter() - start
    assert output.harmonic_loss.total_loss is not None
    assert output.reconstruction_loss is not None
    loss = output.harmonic_loss.total_loss + output.reconstruction_loss
    start = perf_counter()
    loss.backward()
    backward_seconds = perf_counter() - start
    sequence_lengths = coarse.sequence.sequence_lengths.tolist()
    bar_counts_by_sample = torch.bincount(
        ownership.batch_membership["bar"],
        minlength=ownership.sample_count,
    ).tolist()
    track_counts_by_sample = torch.bincount(
        ownership.batch_membership["track"],
        minlength=ownership.sample_count,
    ).tolist()
    return {
        "variant": "local_gnn_hierarchy_transformer_fusion",
        "config": model.config.to_dict(),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        **_counts(batch),
        "bars": int(graph["bar"].num_nodes),
        "tracks": int(graph["track"].num_nodes),
        "bar_counts_by_sample": bar_counts_by_sample,
        "track_counts_by_sample": track_counts_by_sample,
        "coarse_sequence_lengths": sequence_lengths,
        "maximum_coarse_sequence_length": max(sequence_lengths),
        "local_encoder_seconds": local_seconds,
        "pooling_seconds": pooling_seconds,
        "transformer_seconds": transformer_seconds,
        "fusion_seconds": fusion_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "complete_forward_backward_seconds": (
            forward_seconds + backward_seconds
        ),
        "active_task_count": len(output.harmonic_loss.task_losses),
        "prediction_task_count": len(output.predictions),
        "candidate_logit_rows": sum(
            int(item.logits.shape[0]) for item in output.predictions
        ),
        "supervision_rows": sum(
            int(item.per_row_loss.shape[0])
            for item in output.supervisions
        ),
        "peak_tensor_shapes": {
            "padded_coarse_tokens": list(coarse.sequence.tokens.shape),
            "contextual_song": list(coarse.song_embeddings.shape),
            "contextual_bars": list(coarse.bar_embeddings.shape),
            "contextual_tracks": list(coarse.track_embeddings.shape),
            **{
                f"fused_{node_type}": list(
                    output.encoder.fused.embeddings[node_type].shape
                )
                for node_type in MANDATORY_NODE_TYPES
            },
        },
        "device": "cpu",
        "dtype": str(next(model.parameters()).dtype),
    }


def benchmark_controlled_ablation(batch) -> list[dict[str, object]]:
    """Compare the three requested variants on one immutable batch."""

    local_reports = [
        benchmark_variant(batch, variant, gnn_layers=1)
        for variant in ("feature_only", "local_gnn")
    ]
    local_reports[0]["variant"] = "phase6a_feature_only"
    local_reports[1]["variant"] = "phase6a_local_gnn"
    return [*local_reports, benchmark_hierarchical_variant(batch)]


@torch.no_grad()
def benchmark_uneven_sequence_packing(
    batch, *, repeats: int = 8
) -> dict[str, object]:
    """Measure bounded uneven packing and complete eval forward plumbing."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    torch.manual_seed(909)
    model = HierarchicalHeterogeneousBaseline(
        _hierarchical_config()
    ).cpu().eval()
    graph = batch.raw_graph_batch
    ownership = extract_hierarchy_ownership(graph)
    local = model.local_baseline.encode(graph)
    pooling = model.context_encoder.pooling(
        local.final_output, ownership
    )
    sequence = model.context_encoder.transformer.build_sequence(
        local.final_output, pooling
    )
    lengths = sequence.sequence_lengths.tolist()
    if len(set(lengths)) < 2:
        raise ValueError(
            "uneven benchmark requires at least two sequence lengths"
        )
    start = perf_counter()
    for _ in range(repeats):
        sequence = model.context_encoder.transformer.build_sequence(
            local.final_output, pooling
        )
    packing_total = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        output = model(batch, include_reconstruction=False)
    forward_total = perf_counter() - start
    return {
        "scope": (
            "bounded uneven-sequence plumbing only; no throughput or "
            "speed acceptance threshold"
        ),
        "repeats": repeats,
        "sequence_lengths": lengths,
        "padded_shape": list(sequence.tokens.shape),
        "sequence_construction_total_seconds": packing_total,
        "sequence_construction_mean_seconds": packing_total / repeats,
        "hierarchical_forward_total_seconds": forward_total,
        "hierarchical_forward_mean_seconds": forward_total / repeats,
        "candidate_logit_rows": sum(
            int(item.logits.shape[0]) for item in output.predictions
        ),
    }


def hierarchical_overfit_evidence(batch, steps: int) -> dict[str, object]:
    torch.manual_seed(911)
    model = HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    first = model(batch)
    assert first.harmonic_loss.total_loss is not None
    assert first.reconstruction_loss is not None
    initial_harmonic = float(first.harmonic_loss.total_loss.detach())
    initial_reconstruction = float(first.reconstruction_loss.detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        assert output.harmonic_loss.total_loss is not None
        assert output.reconstruction_loss is not None
        (
            output.harmonic_loss.total_loss + output.reconstruction_loss
        ).backward()
        optimizer.step()
    final = model(batch)
    assert final.harmonic_loss.total_loss is not None
    assert final.reconstruction_loss is not None

    def covered(module) -> bool:
        return any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )

    layer = model.context_encoder.transformer.encoder.layers[0]
    return {
        "steps": steps,
        "initial_harmonic_loss": initial_harmonic,
        "final_harmonic_loss": float(
            final.harmonic_loss.total_loss.detach()
        ),
        "initial_reconstruction_loss": initial_reconstruction,
        "final_reconstruction_loss": float(
            final.reconstruction_loss.detach()
        ),
        "pooling_gradient_coverage": {
            name: covered(module)
            for name, module in (
                ("bar_beat", model.context_encoder.pooling.bar_beat),
                ("bar_onset", model.context_encoder.pooling.bar_onset),
                ("bar_note", model.context_encoder.pooling.bar_note),
                ("track_note", model.context_encoder.pooling.track_note),
                ("bar_builder", model.context_encoder.pooling.bar_builder),
                ("track_builder", model.context_encoder.pooling.track_builder),
            )
        },
        "transformer_gradient_coverage": {
            "attention": covered(layer.self_attn),
            "ffn_in": covered(layer.linear1),
            "ffn_out": covered(layer.linear2),
        },
        "fusion_gradient_coverage": {
            node_type: covered(
                model.context_encoder.fusion.fusions[node_type]
            )
            for node_type in MANDATORY_NODE_TYPES
        },
        "local_node_gradient_coverage": {
            node_type: covered(
                model.local_baseline.encoder.feature_encoder.node_encoders[
                    node_type
                ]
            )
            for node_type in MANDATORY_NODE_TYPES
        },
        "task_head_gradient_coverage": {
            spec.task_id: covered(
                model.local_baseline.task_heads.heads[
                    f"task_{index:02d}"
                ]
            )
            for index, spec in enumerate(model.task_specs)
        },
        "interpretation": (
            "bounded trainability plumbing only; no quality or corpus claim"
        ),
    }


def diagnostic_evidence() -> dict[str, object]:
    torch.manual_seed(919)
    model = HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig(
            hidden_dim=16,
            local_gnn_layers=1,
            transformer_layers=1,
            attention_heads=4,
            ffn_multiplier=2,
            dropout=0.0,
        )
    ).eval()
    original = replace(_hook_piece(998), annotations=(), targets=())
    note_id = original.notes[0].note_id
    perturbed = perturb_canonical_note_pitch(original, note_id)
    return asdict(
        hierarchical_single_note_sensitivity(
            model,
            original,
            perturbed,
            note_id=note_id,
        )
    )


def reference_parameter_counts() -> dict[str, int]:
    local = {
        variant: sum(
            parameter.numel()
            for parameter in LocalHeterogeneousBaseline(
                LocalBaselineConfig(
                    variant=variant,
                    hidden_dim=128,
                    gnn_layers=3,
                    dropout=0.1,
                )
            ).parameters()
        )
        for variant in ("feature_only", "local_gnn")
    }
    hierarchical = HierarchicalHeterogeneousBaseline(
        HierarchicalBaselineConfig()
    )
    return {
        "phase6a_feature_only": local["feature_only"],
        "phase6a_local_gnn": local["local_gnn"],
        "phase6b_hierarchy": sum(
            parameter.numel() for parameter in hierarchical.parameters()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--larger-repeats", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=30)
    parser.add_argument("--packing-repeats", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.larger_repeats < 2
        or args.overfit_steps <= 0
        or args.packing_repeats <= 0
    ):
        parser.error(
            "larger-repeats must be >=2 and overfit/packing repeats "
            "must be positive"
        )
    with TemporaryDirectory(prefix="music-critic-phase6b-") as directory:
        root = Path(directory)
        tiny = make_bounded_batch(root / "tiny", 1)
        larger = make_bounded_batch(root / "larger", args.larger_repeats)
        report = {
            "scope": (
                "bounded CPU evidence only; timings have no threshold and "
                "establish neither corpus feasibility nor model superiority"
            ),
            "reference_parameter_counts": reference_parameter_counts(),
            "benchmarks": {
                "tiny_mixed": benchmark_controlled_ablation(tiny),
                "larger_synthetic": benchmark_controlled_ablation(larger),
            },
            "uneven_sequence_packing": (
                benchmark_uneven_sequence_packing(
                    tiny, repeats=args.packing_repeats
                )
            ),
            "overfit": hierarchical_overfit_evidence(
                tiny, args.overfit_steps
            ),
            "one_note_diagnostic": diagnostic_evidence(),
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
