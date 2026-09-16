"""Isolated bounded fresh C2 continuation; old training defaults are unchanged.

The session consumes explicit TRAIN records through a caller-owned reader and
the frozen full-orbit sampler. It applies the historical transposition itself.
An interrupted session is restored through harmonic_fusion_resume, never the
model-only loader. Synthetic checks establish mechanics, not model quality.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .contracts import fingerprint
from .corrected_model import CorrectedAnalysisGNNModel, model_state_fingerprint
from .corrected_training import (
    CorrectedValidationAccumulator, FrozenClassWeights,
    align_target_sidecars_after_prediction, attempt_applied_update,
    combine_single_record_raw_batches, move_raw_graph_batch,
    require_non_test_split, transpose_raw_graph_batch,
)
from .full_orbit_training import FullOrbitPair, FullOrbitSampler
from .harmonic_logit_fusion import HarmonicFusionModel, fusion_model_contract
from .multitask_contract import TASK_BY_ID, get_vocabulary
from music_critic.experiments.pop909_cl_frozen.projection import root_pc_from_label


@dataclass(frozen=True, slots=True)
class FusionTrainingConfig:
    total_updates: int = 512
    warmup_updates: int = 64
    peak_lr: float = 0.0003
    validation_interval: int = 256
    checkpoint_interval: int = 64
    weight_decay: float = 0.0005
    seed: int = 17
    batch_size: int = 2
    validation_mode: str = "required"

    def __post_init__(self):
        integers = (self.total_updates, self.warmup_updates, self.validation_interval,
                    self.checkpoint_interval, self.seed, self.batch_size)
        if (any(type(x) is not int or x <= 0 for x in integers)
                or self.total_updates > 4096 or self.warmup_updates >= self.total_updates
                or self.seed != 17 or self.batch_size != 2
                or self.validation_interval > self.total_updates
                or self.checkpoint_interval > self.total_updates
                or not math.isfinite(self.peak_lr) or not 0 < self.peak_lr <= 0.005
                or self.weight_decay != 0.0005
                or self.validation_mode not in {"required", "synthetic_disabled"}):
            raise ValueError("fusion_training.config_invalid")

    def to_dict(self):
        return {"schema_version": "HarmonicFusionBoundedTraining@1.1.0", **asdict(self),
                "continuation": "fresh_optimizer_not_historical_resume",
                "transposition": "historical_C2_stale_track_pitch_summaries",
                "precision": "fp32", "gradient_clip_norm": 1.0,
                "scheduler": "linear_warmup_then_cosine_to_zero",
                "loss": "unchanged_18_heads_primary_plus_0.25_auxiliary",
                "validation_view": "identity", "test_enabled": False,
                "early_stopping": False, "loader_workers": 0}


def schedule_factor(config: FusionTrainingConfig, step: int) -> float:
    if step < config.warmup_updates:
        return (step + 1) / config.warmup_updates
    progress = (step - config.warmup_updates) / (config.total_updates - config.warmup_updates)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def pending_validation_update(config, applied_updates, validations):
    """Require the complete expected prefix, allowing only the current boundary pending.

    A failed boundary evaluation can be retried against the same model before
    any more training. Earlier missing evaluations cannot be recreated from a
    later model and are therefore never silently accepted or backfilled.
    """
    observed = [row["update"] for row in validations]
    if config.validation_mode == "synthetic_disabled":
        if observed:
            raise ValueError("fusion_training.validation_disabled")
        return None
    expected = [0, *range(config.validation_interval, applied_updates + 1, config.validation_interval)]
    if applied_updates == config.total_updates and expected[-1] != applied_updates:
        expected.append(applied_updates)
    if observed == expected:
        return None
    if expected[-1] == applied_updates and observed == expected[:-1]:
        return applied_updates
    raise ValueError("fusion_training.validation_schedule_incomplete")


class FusionTrainingSession:
    """One arm with independent model, optimizer, sampler and stochastic state."""

    def __init__(self, *, base: CorrectedAnalysisGNNModel, enabled: bool,
                 config: FusionTrainingConfig, bindings: dict, table: Sequence[FullOrbitPair],
                 device: str = "cpu"):
        from .harmonic_fusion_resume import capture_safe_rng, preserve_caller_rng
        if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("fusion_training.device_unavailable")
        if not bindings or not isinstance(bindings, dict):
            raise ValueError("fusion_training.bindings_required")
        fingerprint(bindings)  # Require finite, serializable immutable evidence.
        self.config = config
        self.bindings = deepcopy(bindings)
        self.device = device
        self.source_model_state = model_state_fingerprint(base)
        with preserve_caller_rng():
            self.model = HarmonicFusionModel(deepcopy(base), enabled=enabled).to(device)
            self.model.requires_grad_(True).train()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.peak_lr,
                                              weight_decay=config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: schedule_factor(config, step))
            # Same disabled FP32 scaler, including the accepted Torch 2.2 API.
            from .performance_replay import _scaler
            self.scaler, _ = _scaler()
            self.sampler = FullOrbitSampler(table, seed=config.seed)
            random.seed(config.seed * 1000 + 1)
            np.random.seed(config.seed * 1000 + 1)
            torch.manual_seed(config.seed * 1000 + 1)
            if device == "cuda":
                torch.cuda.manual_seed_all(config.seed * 1000 + 1)
            self.rng_state = capture_safe_rng()
        self.applied_updates = 0
        self.optimizer_steps = {name: 0 for name, _ in self.model.named_parameters()}
        self.history: list[dict] = []
        self.validations: list[dict] = []

    def step(self, reader: Callable, class_weights: FrozenClassWeights):
        """Apply one B2 update. Reader receives only an explicit TRAIN request.

        Reader returns (single-record raw batch, sidecar); it must return the
        requested identity. A failure is terminal for this attempt: no skip or
        replacement draw can quietly change the matched pair.
        """
        from .harmonic_fusion_resume import capture_safe_rng, preserve_caller_rng, restore_safe_rng
        if self.pending_validation is not None:
            raise ValueError("fusion_training.validation_pending")
        if self.applied_updates >= self.config.total_updates:
            raise ValueError("fusion_training.budget_exhausted")
        if self.bindings.get("class_weight_fingerprint") != class_weights.payload_fingerprint:
            raise ValueError("fusion_training.class_weight_binding_mismatch")
        draws = [self.sampler.peek(i) for i in range(self.config.batch_size)]
        with preserve_caller_rng():
            loaded = [reader(row.record_id, split="train") for row in draws]
            for draw, (_, sidecar) in zip(draws, loaded, strict=True):
                if (sidecar.get("record_id") != draw.record_id
                        or sidecar.get("source_component_id") != draw.component_id):
                    raise ValueError("fusion_training.record_binding_mismatch")
            raw = combine_single_record_raw_batches([item[0] for item in loaded])
            shifts = tuple(row.shift_pc for row in draws)
            raw = move_raw_graph_batch(transpose_raw_graph_batch(raw, shifts), self.device)
            # Readers/preparation cannot consume the independently owned
            # dropout stream. Only the update below advances that domain.
            restore_safe_rng(self.rng_state, device_type=self.device)
            self.model.train()
            lr_used = self.optimizer.param_groups[0]["lr"]
            result = attempt_applied_update(
                model=self.model, raw_graph_batch=raw, sidecars=[item[1] for item in loaded],
                shifts=shifts, class_weights=class_weights, optimizer=self.optimizer,
                scheduler=self.scheduler, scaler=self.scaler, gradient_clip_norm=1.0)
            if not result.applied:
                raise ValueError("fusion_training.skipped_update_pair_invalid")
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    self.optimizer_steps[name] += 1
            self.rng_state = capture_safe_rng()
        self.sampler.advance_after_applied_update(self.config.batch_size)
        self.applied_updates += 1
        row = {"update": self.applied_updates, "draws": [asdict(x) for x in draws],
               "loss": float(result.loss.total.detach().cpu()), "lr_used": lr_used,
               "lr_next": self.optimizer.param_groups[0]["lr"]}
        self.history.append(row)
        return deepcopy(row)

    def run_until(self, stop: int, reader: Callable, class_weights: FrozenClassWeights,
                  *, validate: Callable | None = None, checkpoint: Callable | None = None,
                  progress: Callable | None = None):
        if type(stop) is not int or not self.applied_updates <= stop <= self.config.total_updates:
            raise ValueError("fusion_training.stop_invalid")
        if self.config.validation_mode == "required" and validate is None:
            raise ValueError("fusion_training.validation_callback_required")
        if self.config.validation_mode == "synthetic_disabled" and validate is not None:
            raise ValueError("fusion_training.validation_disabled")

        def emit(stage, status, **values):
            if progress is not None:
                progress({"stage": stage, "status": status,
                          "applied_updates": self.applied_updates, **values})

        def finish_pending():
            if self.pending_validation is not None:
                emit("validation", "started")
                self._validate(validate)
                emit("validation", "completed")

        # Includes initial validation and recovery from a checkpoint saved
        # after a failed boundary callback. No update precedes this recovery.
        finish_pending()
        while self.applied_updates < stop:
            draws = [asdict(self.sampler.peek(i)) for i in range(self.config.batch_size)]
            emit("update", "started", draws=draws)
            self.step(reader, class_weights)
            emit("update", "completed", draws=draws)
            update = self.applied_updates
            finish_pending()
            if checkpoint is not None and (update % self.config.checkpoint_interval == 0
                                           or update == stop):
                checkpoint(self)
        return deepcopy(self.history)

    def _validate(self, validate: Callable):
        if self.config.validation_mode != "required" or self.pending_validation != self.applied_updates:
            raise ValueError("fusion_training.validation_not_due")
        from .harmonic_fusion_resume import preserve_caller_rng
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            with preserve_caller_rng(), torch.no_grad():
                self.model.eval()
                metrics = validate(self.model, split="validation")
            fingerprint(metrics)
            score = metrics["corrected_primary_macro_score"]
            if score is not None and (type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("fusion_training.validation_score_invalid")
            self.validations.append({"update": self.applied_updates, "metrics": deepcopy(metrics)})
        finally:
            for module, mode in modes:
                module.training = mode

    @property
    def pending_validation(self):
        return pending_validation_update(self.config, self.applied_updates, self.validations)

    @property
    def best_update(self):
        if self.pending_validation is not None:
            raise ValueError("fusion_training.validation_pending_best_unavailable")
        available = [row for row in self.validations
                     if row["metrics"]["corrected_primary_macro_score"] is not None]
        if not available:
            return None
        return max(available, key=lambda row: row["metrics"]["corrected_primary_macro_score"])["update"]


def paired_sessions(base, *, config, bindings, table, device="cpu"):
    """Both arms own all common parameters and receive all additional training."""
    sessions = tuple(FusionTrainingSession(base=base, enabled=enabled, config=config,
                                         bindings=bindings, table=table, device=device)
                     for enabled in (False, True))
    if model_state_fingerprint(sessions[0].model.base) != model_state_fingerprint(sessions[1].model.base):
        raise ValueError("fusion_training.initial_state_mismatch")
    if set(map(id, sessions[0].model.parameters())) & set(map(id, sessions[1].model.parameters())):
        raise ValueError("fusion_training.aliased_parameters")
    return sessions


def paired_sessions_from_checkpoint(path, *, expected_sha256, expected_size_bytes,
                                    config, bindings, table, device="cpu"):
    """Strictly load one byte-verified C2 artifact for fresh equal continuation.

    This reads only model weights through the old restricted loader. No
    historical optimizer, sampler or RNG state is represented as restored.
    Registration and frozen data/gate attestation belong to the dispatcher.
    """
    from music_critic.experiments.pop909_cl_frozen.checkpoint import load_model_checkpoint
    loaded = load_model_checkpoint(path, expected_sha256=expected_sha256,
                                    expected_size_bytes=expected_size_bytes)
    source = {"source_checkpoint_sha256": expected_sha256,
              "source_checkpoint_size_bytes": expected_size_bytes,
              "source_model_state_fingerprint": model_state_fingerprint(loaded.model)}
    if any(key in bindings and bindings[key] != value for key, value in source.items()):
        raise ValueError("fusion_training.source_checkpoint_binding_mismatch")
    return paired_sessions(loaded.model, config=config, bindings={**bindings, **source},
                           table=table, device=device)


class FusionValidationAccumulator(CorrectedValidationAccumulator):
    """Old arithmetic plus confusions and a narrowly defined consistency check.

    Root-in-predicted-PC-set uses aligned raw predictions only. It measures one
    internal constraint, not harmonic correctness or a music-quality score.
    Native TRAIN tuple statistics keep their original eligibility semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_in_set = 0
        self.consistency_rows = 0
        self.component_predictions = []

    def update(self, output, alignment, *, sidecars=()):
        # The enabled wrapper checks all 12 beat identities; enforce this for
        # the equally continued disabled control as well, before metrics.
        from .harmonic_logit_fusion import _validate_predictions
        _validate_predictions(output.predictions)
        super().update(output, alignment, sidecars=sidecars)
        for task, rows in alignment.heads.items():
            predictions = output.logits[task].argmax(1).tolist()
            for index in torch.nonzero(rows.valid_mask, as_tuple=False).flatten().tolist():
                self.component_predictions.append({
                    "task_id": task, "record_id": rows.record_ids[index],
                    "component_id": rows.component_ids[index], "entity_id": rows.entity_ids[index],
                    "target_class_id": int(rows.values[index]),
                    "predicted_class_id": predictions[int(rows.candidate_indices[index])]})
        root_labels = get_vocabulary(TASK_BY_ID["root"].vocabulary_id).labels
        set_labels = get_vocabulary(TASK_BY_ID["pitch_class_set"].vocabulary_id).labels
        roots = output.logits["root"].argmax(1).tolist()
        sets = output.logits["pitch_class_set"].argmax(1).tolist()
        self.root_in_set += sum(root_pc_from_label(root_labels[r]) in
                                {int(pc) for pc in set_labels[s].split(",")}
                                for r, s in zip(roots, sets, strict=True))
        self.consistency_rows += len(roots)

    def finalize(self):
        result = super().finalize()
        result["confusions"] = {task: matrix.tolist() for task, matrix in self.confusions.items()}
        result["component_predictions"] = deepcopy(self.component_predictions)
        result["prediction_consistency"] = {
            "definition": "argmax_spelled_root_pc_in_argmax_absolute_pitch_class_set",
            "population": "all_raw_beat_candidates_independent_of_target_availability",
            "consistent": self.root_in_set, "total": self.consistency_rows,
            "fraction": None if not self.consistency_rows else self.root_in_set / self.consistency_rows,
            "music_quality_claim": False}
        return result


def evaluate_validation(model, *, records, reader, class_weights, split="validation",
                        train_seen_tuples=None):
    require_non_test_split(split)
    if split != "validation":
        raise ValueError("fusion_training.validation_split_required")
    accumulator = FusionValidationAccumulator(class_weights, train_seen_tuples=train_seen_tuples)
    modes = [(module, module.training) for module in model.modules()]
    from .harmonic_fusion_resume import preserve_caller_rng
    try:
        with preserve_caller_rng(), torch.no_grad():
            model.eval()
            for record_id in records:
                raw, sidecar = reader(record_id, split="validation")
                if sidecar.get("record_id") != record_id:
                    raise ValueError("fusion_training.validation_record_binding_mismatch")
                raw = move_raw_graph_batch(raw, next(model.parameters()).device)
                output = model(raw)
                alignment = align_target_sidecars_after_prediction(output, raw, (sidecar,), shifts=(0,))
                accumulator.update(output, alignment, sidecars=(sidecar,))
    finally:
        for module, mode in modes:
            module.training = mode
    return accumulator.finalize()
