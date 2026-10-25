"""Bounded, critic-only optimization and selection on detached feature banks.

Synthetic tests establish CPU continuation mechanics. Real data, native hardware
admission and scientific runs require the separate experiment protocol.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from music_critic.critic.contracts import ASPECTS, CriticFeaturePyramid, SamplePrediction


TRAINING_SCHEMA = "CriticCachedPairTraining@2.0.0"
CURRICULUM_SCHEMA = "CriticFixedExposureCurriculum@1.0.0"


def _selection_aspects(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if not result or len(result) != len(set(result)) or set(result) - set(ASPECTS):
        raise ValueError("critic.training.selection_aspects_invalid")
    return result


@dataclass(frozen=True)
class PairEvaluation:
    pair_accuracy: float | None
    pair_loss: float | None
    per_aspect: Mapping[str, float | None]
    aspect_pair_losses: Mapping[str, float | None]
    work_support: Mapping[str, int]
    pair_support: Mapping[str, int]
    declared_aspects: tuple[str, ...]
    supported_aspects: tuple[str, ...]
    selection_eligible: bool = True
    evaluation_fingerprint: str = ""


def evaluate_pairs(
    rows: Sequence[tuple[SamplePrediction, SamplePrediction, Any]], *,
    selection_aspects: Sequence[str] = ASPECTS, tie_tolerance: float = 1e-6,
) -> PairEvaluation:
    """Average pairs within work, then works within aspect, then aspects."""
    aspects = _selection_aspects(selection_aspects)
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("critic.training.tie_tolerance_invalid")
    values: dict[str, dict[str, list[tuple[float, float]]]] = {
        aspect: defaultdict(list) for aspect in aspects}
    for left, right, target in rows:
        if (left.sample_id, right.sample_id) != target.sample_ids:
            raise ValueError("critic.training.target_identity_mismatch")
        if (left.horizon, left.support_policy) != (right.horizon, right.support_policy):
            raise ValueError("critic.training.pair_scope_mismatch")
        for aspect in aspects:
            index = ASPECTS.index(aspect)
            if not (target.pair_mask[index] and left.availability[index] and right.availability[index]):
                continue
            direction = target.preference[index]
            if direction not in (-1, 1):
                raise ValueError("critic.training.preference_invalid")
            margin = float((left.raw_scores[index] - right.raw_scores[index]).detach().cpu()) * direction
            if not math.isfinite(margin):
                raise ValueError("critic.training.nonfinite_score")
            accuracy = 0.5 if abs(margin) <= tie_tolerance else float(margin > 0)
            loss = float(F.softplus(torch.tensor(-margin, dtype=torch.float64)))
            values[aspect][target.work_id].append((accuracy, loss))
    accuracy, losses, work_support, pair_support = {}, {}, {}, {}
    for aspect, works in values.items():
        work_support[aspect] = len(works)
        pair_support[aspect] = sum(len(pairs) for pairs in works.values())
        accuracy[aspect] = (sum(sum(x[0] for x in pairs) / len(pairs)
                               for pairs in works.values()) / len(works) if works else None)
        losses[aspect] = (sum(sum(x[1] for x in pairs) / len(pairs)
                             for pairs in works.values()) / len(works) if works else None)
    supported = tuple(a for a in aspects if accuracy[a] is not None)
    return PairEvaluation(
        sum(accuracy[a] for a in supported) / len(supported) if supported else None,
        sum(losses[a] for a in supported) / len(supported) if supported else None,
        accuracy, losses, work_support, pair_support, aspects, supported,
        evaluation_fingerprint=_fingerprint([
            {"target": target, "horizon": left.horizon, "support_policy": left.support_policy}
            for left, _right, target in rows]),
    )


@dataclass
class SelectionState:
    declared_aspects: tuple[str, ...]
    supported_aspects: tuple[str, ...] | None = None
    best_update: int | None = None
    best_accuracy: float | None = None
    best_pair_loss: float | None = None
    evaluation_fingerprint: str | None = None

    def __post_init__(self) -> None:
        self.declared_aspects = _selection_aspects(self.declared_aspects)

    def consider(self, result: PairEvaluation, *, update: int) -> bool:
        if not result.selection_eligible:
            raise ValueError("critic.training.diagnostic_not_for_selection")
        if type(update) is not int or update < 0 or result.declared_aspects != self.declared_aspects:
            raise ValueError("critic.training.selection_binding_mismatch")
        if self.supported_aspects is not None and result.supported_aspects != self.supported_aspects:
            raise ValueError("critic.training.selection_support_changed")
        if not result.evaluation_fingerprint:
            raise ValueError("critic.training.selection_cohort_required")
        if self.evaluation_fingerprint is not None and result.evaluation_fingerprint != self.evaluation_fingerprint:
            raise ValueError("critic.training.selection_cohort_changed")
        if result.pair_accuracy is None:
            self.evaluation_fingerprint = result.evaluation_fingerprint
            return False
        if result.pair_loss is None or not all(math.isfinite(v) for v in (result.pair_accuracy, result.pair_loss)):
            raise ValueError("critic.training.selection_nonfinite")
        self.supported_aspects = result.supported_aspects
        self.evaluation_fingerprint = result.evaluation_fingerprint
        candidate = (-result.pair_accuracy, result.pair_loss, update)
        prior = None if self.best_update is None else (-self.best_accuracy, self.best_pair_loss, self.best_update)
        if prior is None or candidate < prior:
            self.best_update, self.best_accuracy, self.best_pair_loss = update, result.pair_accuracy, result.pair_loss
            return True
        return False


@dataclass(frozen=True)
class CachedPair:
    pair_id: str
    work_id: str
    left: CriticFeaturePyramid
    right: CriticFeaturePyramid
    targets: Any
    split: str = "train"
    component_id: str | None = None
    edit_count: int | None = None
    genre: str | None = None

    def __post_init__(self) -> None:
        if not self.pair_id or not self.work_id or self.work_id != self.targets.work_id:
            raise ValueError("critic.training.pair_identity_invalid")
        if self.split not in {"train", "head_dev"} or self.component_id == "":
            raise ValueError("critic.training.pair_split_invalid")
        if self.edit_count is not None and (type(self.edit_count) is not int or self.edit_count <= 0):
            raise ValueError("critic.training.edit_count_invalid")
        if self.genre is not None and (not isinstance(self.genre, str) or not self.genre.strip()):
            raise ValueError("critic.training.genre_invalid")
        if len(self.left.samples) != 1 or len(self.right.samples) != 1:
            raise ValueError("critic.training.one_sample_per_cached_view")
        if (self.left.samples[0].sample_id, self.right.samples[0].sample_id) != self.targets.sample_ids:
            raise ValueError("critic.training.target_identity_mismatch")
        if self.left.binding != self.right.binding:
            raise ValueError("critic.training.feature_binding_mismatch")
        for pyramid in (self.left, self.right):
            for sample in pyramid.samples:
                if not sample.raw_hash or not sample.stages:
                    raise ValueError("critic.training.feature_identity_missing")
                for tensor in sample.embeddings.values():
                    if tensor.requires_grad or tensor.grad_fn is not None:
                        raise ValueError("critic.training.features_not_detached")
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError("critic.training.features_nonfinite")


@dataclass(frozen=True)
class CurriculumStage:
    """Exclusive exposure boundary and equally weighted authored-edit counts."""
    end_draw: int
    edit_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        counts = tuple(self.edit_counts)
        if type(self.end_draw) is not int or self.end_draw <= 0:
            raise ValueError("critic.training.curriculum_boundary_invalid")
        if (not counts or any(type(count) is not int or count <= 0 for count in counts)
                or len(set(counts)) != len(counts)):
            raise ValueError("critic.training.curriculum_counts_invalid")
        object.__setattr__(self, "edit_counts", counts)


@dataclass(frozen=True)
class CurriculumDraw:
    """Target-side journal row; no curriculum metadata enters model inputs."""
    pair_id: str
    swapped: bool
    source_stage: int
    edit_count: int
    aspect: str
    genre: str


@dataclass(frozen=True)
class CriticTrainConfig:
    seed: int
    learning_rate: float
    batch_size: int
    max_updates: int
    max_attempts: int
    clip_grad_norm: float
    weight_decay: float = 0.0
    selection_aspects: tuple[str, ...] = ASPECTS
    tie_tolerance: float = 1e-6
    curriculum_stages: tuple[CurriculumStage, ...] = ()
    sampling_order: str = "curriculum"

    def __post_init__(self) -> None:
        for value in (self.seed, self.batch_size, self.max_updates, self.max_attempts):
            if type(value) is not int or value < 0:
                raise ValueError("critic.training.integer_config_invalid")
        if not self.batch_size or not self.max_updates or self.max_attempts < self.max_updates:
            raise ValueError("critic.training.budget_invalid")
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (self.learning_rate, self.clip_grad_norm)):
            raise ValueError("critic.training.optimizer_config_invalid")
        if any(not math.isfinite(v) or v < 0 for v in (self.weight_decay, self.tie_tolerance)):
            raise ValueError("critic.training.config_nonfinite")
        object.__setattr__(self, "selection_aspects", _selection_aspects(self.selection_aspects))
        stages = tuple(self.curriculum_stages)
        if self.sampling_order not in {"curriculum", "shuffled"} or (
                not stages and self.sampling_order != "curriculum"):
            raise ValueError("critic.training.sampling_order_invalid")
        if any(not isinstance(stage, CurriculumStage) for stage in stages):
            raise ValueError("critic.training.curriculum_stage_invalid")
        if stages:
            boundaries = tuple(stage.end_draw for stage in stages)
            # A fixed exposure experiment has one attempted-update budget. Skips
            # consume draws; successful updates cannot end the plan prematurely.
            if (self.max_updates != self.max_attempts
                    or boundaries[-1] != self.max_attempts * self.batch_size
                    or any(end % self.batch_size for end in boundaries)
                    or any(left >= right for left, right in zip((0, *boundaries), boundaries))):
                raise ValueError("critic.training.curriculum_budget_invalid")
        object.__setattr__(self, "curriculum_stages", stages)


@dataclass(frozen=True)
class StepResult:
    pair_ids: tuple[str, ...]
    work_ids: tuple[str, ...]
    swapped: tuple[bool, ...]
    applied: bool
    reason: str | None
    loss: float | None
    pair_loss: float | None
    local_loss: float | None
    pair_support: int
    local_support: int
    grad_norm: float | None
    applied_update: int
    attempted_update: int
    edit_counts: tuple[int | None, ...] = ()
    genres: tuple[str | None, ...] = ()
    aspects: tuple[str | None, ...] = ()
    source_stages: tuple[int, ...] = ()


def _json_value(value: Any, *, tensor_guard: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        if tensor_guard:
            try:
                return {"object": id(value), "pointer": value.data_ptr(), "version": value._version,
                        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                        "requires_grad": value.requires_grad, "grad_fn": id(value.grad_fn)}
            except RuntimeError:
                # Inference-mode tensors have no version counter: check their bytes.
                pass
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        result = {"tensor_sha256": sha256(raw).hexdigest(), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if tensor_guard:
            result.update(requires_grad=value.requires_grad, grad_fn=id(value.grad_fn), device=str(value.device))
        return result
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name), tensor_guard=tensor_guard) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v, tensor_guard=tensor_guard) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v, tensor_guard=tensor_guard) for v in value]
    return value


def _fingerprint(value: Any) -> str:
    return sha256(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


class _BankMutationGuard:
    """Compile mutable leaves once instead of reserializing every raw note.

    Frozen dataclasses, tuples and scalar values cannot change through normal
    APIs. Mutable descendants still get guards, deduplicated by object identity.
    Tensor version checks retain the existing fallback for inference tensors.
    Bypassing Python immutability or Tensor.data is outside this guard's contract.
    """
    def __init__(self, bank):
        self.root = bank
        self.checks = []
        self._retained = []  # Prevent identity reuse after a mutable parent drops a child.
        visited = set()

        def visit(value):
            if id(value) in visited:
                return
            visited.add(id(value))
            self._retained.append(value)
            if isinstance(value, torch.Tensor):
                getter = lambda value=value: _json_value(value, tensor_guard=True)
            elif isinstance(value, Mapping):
                getter = lambda value=value: tuple((k, id(v)) for k, v in value.items())
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                getter = lambda value=value: tuple(id(v) for v in value)
                for child in value:
                    visit(child)
            elif is_dataclass(value):
                for field in fields(value):
                    visit(getattr(value, field.name))
                if value.__dataclass_params__.frozen:
                    return
                names = tuple(field.name for field in fields(value))
                getter = lambda value=value, names=names: tuple(id(getattr(value, name)) for name in names)
            elif isinstance(value, tuple):
                for child in value:
                    visit(child)
                return
            else:
                return
            self.checks.append((getter, getter()))

        visit(bank)

    def matches(self, bank):
        return bank is self.root and all(getter() == expected for getter, expected in self.checks)


def _rng_capture() -> dict[str, Any]:
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": ([v.clone() for v in torch.cuda.get_rng_state_all()]
                           if torch.cuda.is_initialized() else [])}


def _rng_validate(state: Any) -> None:
    if not isinstance(state, dict) or set(state) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("critic.training.rng_invalid")
    try:
        random.Random().setstate(state["python"])
        torch.Generator().set_state(state["torch_cpu"])
        cuda = state["torch_cuda"]
        if type(cuda) is not list or (cuda and (not torch.cuda.is_initialized()
                                              or len(cuda) != torch.cuda.device_count())):
            raise ValueError("CUDA RNG inventory mismatch")
        for i, tensor in enumerate(cuda):
            torch.Generator(device=f"cuda:{i}").set_state(tensor)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("critic.training.rng_invalid") from exc


def _rng_restore(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


class CriticTrainer:
    """Owns only the new critic's optimizer; cached backbone values are constants.

    Initialization of the model is caller-owned. The configured seed controls
    sample order and Python/Torch stochastic training, isolated from caller RNG.
    No NumPy randomness is used. Device determinism still requires native admission.
    """

    def __init__(self, model: nn.Module, loss: Any, bank: Sequence[CachedPair],
                 config: CriticTrainConfig, *, bindings: Mapping[str, str],
                 frozen_backbones: Sequence[nn.Module] = ()):
        self.model, self.loss, self.config = model, loss, config
        self.bank = tuple(bank)
        if not self.bank or len({x.pair_id for x in self.bank}) != len(self.bank):
            raise ValueError("critic.training.bank_invalid")
        if any(pair.split != "train" for pair in self.bank):
            raise ValueError("critic.training.train_split_required")
        if any(pair.left.binding != self.bank[0].left.binding for pair in self.bank):
            raise ValueError("critic.training.feature_binding_mismatch")
        required = {"source_id", "split_id", "target_version"}
        if not required <= bindings.keys() or any(not isinstance(v, str) or not v.strip()
                                                 or v.lower() in {"unknown", "none", "null"}
                                                 for v in bindings.values()):
            raise ValueError("critic.training.run_bindings_required")
        self.bindings = dict(bindings)
        self.frozen_backbones = tuple(frozen_backbones)
        excluded = {id(p) for backbone in self.frozen_backbones for p in backbone.parameters()}
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters or excluded & {id(p) for p in self.parameters}:
            raise ValueError("critic.training.optimizer_backbone_overlap")
        if any(p.requires_grad for backbone in self.frozen_backbones for p in backbone.parameters()):
            raise ValueError("critic.training.backbone_not_frozen")
        self.device = self.parameters[0].device
        if any(p.device != self.device for p in model.parameters()):
            raise ValueError("critic.training.mixed_devices")
        for backbone in self.frozen_backbones:
            backbone.eval()
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
        self.sampler = torch.Generator(device="cpu").manual_seed(config.seed)
        self.groups: dict[str, dict[str, list[int]]] = {}
        for index, pair in enumerate(self.bank):
            available = {aspect for aspect, mask in zip(ASPECTS, pair.targets.pair_mask) if mask}
            if loss.lambda_local > 0:
                for local in pair.targets.local_targets:
                    available.update(key.split("/")[0] for key, target in local.items() if any(target.mask))
            for aspect in sorted(available or {"unsupported"}):
                self.groups.setdefault(pair.work_id, {}).setdefault(aspect, []).append(index)
        self.exposure_plan = self._build_exposure_plan()
        self._bound_exposure_plan = self.exposure_plan
        self._pair_indices = {pair.pair_id: index for index, pair in enumerate(self.bank)}
        self.cursor = 0
        self.attempted_updates = self.applied_updates = self.skipped_updates = self.draws = 0
        self.selection = SelectionState(config.selection_aspects)
        caller = _rng_capture()
        try:
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            self.rng_state = _rng_capture()
        finally:
            _rng_restore(caller)
        self.metadata = {
            "schema": TRAINING_SCHEMA, "config": asdict(config), "bindings": self.bindings,
            "bank_fingerprint": _fingerprint(self.bank), "model_class": type(model).__qualname__,
            "model_definition": repr(model), "loss_class": type(loss).__qualname__,
            "loss_config": {"lambda_local": loss.lambda_local},
            "device_type": self.device.type,
            "sampling_schema": CURRICULUM_SCHEMA if config.curriculum_stages else None,
            "exposure_plan_fingerprint": _fingerprint(self.exposure_plan) if self.exposure_plan else None,
        }
        self._bank_guard = _BankMutationGuard(self.bank)

    def _build_exposure_plan(self) -> tuple[CurriculumDraw, ...]:
        stages = self.config.curriculum_stages
        if not stages:
            return ()
        if self.loss.lambda_local != 0:
            raise ValueError("critic.training.curriculum_pair_only_required")
        counts = {count for stage in stages for count in stage.edit_counts}
        groups: dict[str, dict[str, dict[str, dict[int, list[str]]]]] = {}
        work_genres = {}
        for pair in self.bank:
            if pair.edit_count is None or pair.genre is None:
                raise ValueError("critic.training.curriculum_metadata_required")
            if pair.edit_count not in counts:
                raise ValueError("critic.training.curriculum_unused_edit_count")
            aspects = tuple(aspect for aspect, observed in zip(ASPECTS, pair.targets.pair_mask) if observed)
            if len(aspects) != 1 or aspects[0] not in self.config.selection_aspects:
                raise ValueError("critic.training.curriculum_single_aspect_required")
            if work_genres.setdefault(pair.work_id, pair.genre) != pair.genre:
                raise ValueError("critic.training.curriculum_work_genre_changed")
            by_count = groups.setdefault(pair.genre, {}).setdefault(pair.work_id, {}).setdefault(aspects[0], {})
            by_count.setdefault(pair.edit_count, []).append(pair.pair_id)
        # Every work has the same declared aspect/count support throughout the
        # course. Missing groups fail closed instead of changing the cohort.
        for genre, works in groups.items():
            for work, aspects in works.items():
                for aspect in self.config.selection_aspects:
                    for count in sorted(counts):
                        if not aspects.get(aspect, {}).get(count):
                            raise ValueError(
                                f"critic.training.curriculum_coverage_missing:{genre}:{work}:{aspect}:{count}")
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        choose = lambda values: values[int(torch.randint(len(values), (1,), generator=generator))]
        genres = sorted(groups)
        rows = []
        for stage_index, stage in enumerate(stages):
            for _ in range(len(rows), stage.end_draw):
                genre = choose(genres)
                work = choose(sorted(groups[genre]))
                aspect = choose(self.config.selection_aspects)
                count = choose(stage.edit_counts)
                pair_id = choose(sorted(groups[genre][work][aspect][count]))
                rows.append(CurriculumDraw(pair_id, bool(choose((False, True))), stage_index,
                                           count, aspect, genre))
        if self.config.sampling_order == "shuffled":
            # Shuffle after all choices: both arms retain exactly the same
            # pair identities and side swaps, not merely expected proportions.
            order = torch.randperm(len(rows), generator=generator).tolist()
            rows = [rows[index] for index in order]
        return tuple(rows)

    def _check_bank(self) -> None:
        if not self._bank_guard.matches(self.bank):
            raise ValueError("critic.training.bank_mutated")
        if self.loss.lambda_local != self.metadata["loss_config"]["lambda_local"]:
            raise ValueError("critic.training.loss_config_mutated")
        if self.exposure_plan is not self._bound_exposure_plan:
            raise ValueError("critic.training.exposure_plan_mutated")

    def _next_pairs(self) -> tuple[tuple[CachedPair, ...], tuple[bool, ...]]:
        result, swaps = [], []
        if self.exposure_plan:
            rows = self.exposure_plan[self.cursor:self.cursor + self.config.batch_size]
            if len(rows) != self.config.batch_size:
                raise RuntimeError("critic.training.budget_exhausted")
            for row in rows:
                pair = self.bank[self._pair_indices[row.pair_id]]
                if row.swapped:
                    pair = replace(pair, left=pair.right, right=pair.left, targets=pair.targets.swapped())
                result.append(pair)
                swaps.append(row.swapped)
            self.cursor += len(rows)
            return tuple(result), tuple(swaps)
        choose = lambda size: int(torch.randint(size, (1,), generator=self.sampler))
        works = sorted(self.groups)
        for _ in range(self.config.batch_size):
            work = works[choose(len(works))]
            aspects = sorted(self.groups[work])
            indices = self.groups[work][aspects[choose(len(aspects))]]
            pair = self.bank[indices[choose(len(indices))]]
            swapped = bool(choose(2))
            if swapped:
                pair = replace(pair, left=pair.right, right=pair.left, targets=pair.targets.swapped())
            result.append(pair)
            swaps.append(swapped)
            self.cursor += 1
        return tuple(result), tuple(swaps)

    def _forward_pairs(self, pairs: Sequence[CachedPair]):
        left = CriticFeaturePyramid(tuple(p.left.samples[0] for p in pairs), pairs[0].left.binding)
        right = CriticFeaturePyramid(tuple(p.right.samples[0] for p in pairs), pairs[0].right.binding)
        return self.model(left.to(self.device)), self.model(right.to(self.device))

    def step(self) -> StepResult:
        self._check_bank()
        if self.applied_updates >= self.config.max_updates or self.attempted_updates >= self.config.max_attempts:
            raise RuntimeError("critic.training.budget_exhausted")
        self.model.train()
        for backbone in self.frozen_backbones:
            backbone.eval()
        pairs, swaps = self._next_pairs()
        self.attempted_updates += 1
        self.draws += len(pairs)
        self.optimizer.zero_grad(set_to_none=True)
        caller = _rng_capture()
        grad_norm = None
        try:
            _rng_restore(self.rng_state)
            outputs = self._forward_pairs(pairs)
            result = self.loss(outputs, tuple(p.targets for p in pairs))
            if result.skip_update:
                self.skipped_updates += 1
                applied, reason = False, "unsupported_targets"
            else:
                if result.total is None or not bool(torch.isfinite(result.total)):
                    raise ValueError("critic.training.loss_nonfinite")
                result.total.backward()
                try:
                    norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm,
                                                         error_if_nonfinite=True)
                except RuntimeError as exc:
                    self.optimizer.zero_grad(set_to_none=True)
                    raise ValueError("critic.training.gradient_nonfinite") from exc
                grad_norm = float(norm.detach().cpu())
                # Roll back the optimizer step if arithmetic becomes nonfinite.
                before_model = copy.deepcopy(self.model.state_dict())
                before_optimizer = copy.deepcopy(self.optimizer.state_dict())
                try:
                    self.optimizer.step()
                    if not _finite_tree(self.model.state_dict()) or not _finite_tree(self.optimizer.state_dict()):
                        raise ValueError("critic.training.optimizer_nonfinite")
                except BaseException:
                    self.model.load_state_dict(before_model)
                    self.optimizer.load_state_dict(before_optimizer)
                    raise
                self.applied_updates += 1
                applied, reason = True, None
        finally:
            self.rng_state = _rng_capture()
            _rng_restore(caller)
        scalar = lambda value: None if value is None else float(value.detach().cpu())
        exposures = self.exposure_plan[self.cursor - len(pairs):self.cursor] if self.exposure_plan else ()
        single_aspect = lambda pair: next((aspect for aspect, observed in zip(ASPECTS, pair.targets.pair_mask)
                                         if observed), None) if sum(pair.targets.pair_mask) == 1 else None
        return StepResult(tuple(p.pair_id for p in pairs), tuple(p.work_id for p in pairs), swaps,
                          applied, reason, scalar(result.total),
                          scalar(result.pair), scalar(result.local), result.pair_support,
                          result.local_support, grad_norm, self.applied_updates, self.attempted_updates,
                          tuple(p.edit_count for p in pairs), tuple(p.genre for p in pairs),
                          tuple(single_aspect(p) for p in pairs), tuple(row.source_stage for row in exposures))

    def evaluate(self, bank: Sequence[CachedPair], *, allow_train_diagnostic: bool = False) -> PairEvaluation:
        self._check_bank()
        bank = tuple(bank)
        if not bank:
            raise ValueError("critic.training.evaluation_bank_empty")
        if self.config.curriculum_stages and not allow_train_diagnostic:
            if any(pair.edit_count != 1 for pair in bank):
                raise ValueError("critic.training.curriculum_single_edit_dev_required")
        if not allow_train_diagnostic:
            train_works = {p.work_id for p in self.bank}
            train_components = {p.component_id for p in self.bank if p.component_id is not None}
            train_samples = {s.sample_id for p in self.bank for view in (p.left, p.right) for s in view.samples}
            train_content = {s.raw_hash for p in self.bank for view in (p.left, p.right) for s in view.samples}
            if any(p.split != "head_dev" or p.work_id in train_works
                   or (p.component_id is not None and p.component_id in train_components)
                   or any(s.sample_id in train_samples or s.raw_hash in train_content
                          for view in (p.left, p.right) for s in view.samples) for p in bank):
                raise ValueError("critic.training.evaluation_split_overlap")
        prior_modes = [(module, module.training) for module in self.model.modules()]
        rows = []
        # Preparation shares a FeatureSample object across repeated originals.
        # Reuse only that exact object, never a sample ID or asserted raw hash:
        # distinct feature contents may carry the same external identities.
        # Strong references prevent identity reuse; the existing mutation guard
        # checks mutable geometry/mappings and tensor versions (bytes for
        # inference tensors). Nothing survives this fixed-weight evaluation.
        predictions = {}

        def predict(view):
            sample = view.samples[0]
            cached = predictions.get(id(sample))
            if cached is not None:
                retained, guard, prediction = cached
                if retained is not sample or not guard.matches(sample):
                    raise ValueError("critic.training.evaluation_bank_mutated")
                return prediction
            guard = _BankMutationGuard(sample)
            prediction = self.model(view.to(self.device)).samples[0]
            if not guard.matches(sample):
                raise ValueError("critic.training.evaluation_bank_mutated")
            predictions[id(sample)] = (sample, guard, prediction)
            return prediction

        try:
            self.model.eval()
            with torch.no_grad():
                for pair in bank:
                    if pair.left.binding != self.bank[0].left.binding:
                        raise ValueError("critic.training.evaluation_feature_binding_mismatch")
                    rows.append((predict(pair.left), predict(pair.right), pair.targets))
                # Also reject a view changed after its final memo lookup.
                if any(not guard.matches(sample) for sample, guard, _ in predictions.values()):
                    raise ValueError("critic.training.evaluation_bank_mutated")
        finally:
            for module, mode in prior_modes:
                module.training = mode
            for backbone in self.frozen_backbones:
                backbone.eval()
        return replace(evaluate_pairs(rows, selection_aspects=self.config.selection_aspects,
                                      tie_tolerance=self.config.tie_tolerance),
                       selection_eligible=not allow_train_diagnostic,
                       evaluation_fingerprint=_fingerprint(bank))

    def state_dict(self) -> dict[str, Any]:
        self._check_bank()
        state = {
            "metadata": copy.deepcopy(self.metadata),
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "sampler": {"state": self.sampler.get_state().clone(), "cursor": self.cursor},
            "rng": copy.deepcopy(self.rng_state),
            "progress": {"attempted": self.attempted_updates, "applied": self.applied_updates,
                         "skipped": self.skipped_updates, "draws": self.draws},
            "selection": asdict(self.selection),
        }
        state["fingerprint"] = _fingerprint(state)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate the complete candidate state before mutating this trainer."""
        self._check_bank()
        expected = {"metadata", "model", "optimizer", "sampler", "rng", "progress", "selection", "fingerprint"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("critic.training.resume_fields_invalid")
        if state["metadata"] != self.metadata:
            raise ValueError("critic.training.resume_binding_mismatch")
        if state["fingerprint"] != _fingerprint({k: v for k, v in state.items() if k != "fingerprint"}):
            raise ValueError("critic.training.resume_fingerprint_mismatch")
        progress = state["progress"]
        if (not isinstance(progress, dict) or set(progress) != {"attempted", "applied", "skipped", "draws"}
                or any(type(v) is not int or v < 0 for v in progress.values())
                or progress["applied"] + progress["skipped"] > progress["attempted"]
                or progress["attempted"] > self.config.max_attempts
                or progress["applied"] > self.config.max_updates
                or progress["draws"] != progress["attempted"] * self.config.batch_size):
            raise ValueError("critic.training.resume_progress_invalid")
        sampler = state["sampler"]
        if (not isinstance(sampler, dict) or set(sampler) != {"state", "cursor"}
                or type(sampler["cursor"]) is not int or sampler["cursor"] != progress["draws"]):
            raise ValueError("critic.training.resume_sampler_invalid")
        candidate_sampler = torch.Generator()
        try:
            candidate_sampler.set_state(sampler["state"])
        except (RuntimeError, TypeError) as exc:
            raise ValueError("critic.training.resume_sampler_invalid") from exc
        _rng_validate(state["rng"])
        selected = state["selection"]
        if (not isinstance(selected, dict) or set(selected) != {f.name for f in fields(SelectionState)}
                or selected["declared_aspects"] != self.config.selection_aspects):
            raise ValueError("critic.training.resume_selection_invalid")
        candidate_selection = SelectionState(**selected)
        if candidate_selection.evaluation_fingerprint is not None:
            value = candidate_selection.evaluation_fingerprint
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("critic.training.resume_selection_invalid")
        if candidate_selection.supported_aspects is not None:
            _selection_aspects(candidate_selection.supported_aspects)
            if not set(candidate_selection.supported_aspects) <= set(self.config.selection_aspects):
                raise ValueError("critic.training.resume_selection_invalid")
        if candidate_selection.best_update is None:
            if any(v is not None for v in (candidate_selection.best_accuracy, candidate_selection.best_pair_loss)):
                raise ValueError("critic.training.resume_selection_invalid")
        elif (type(candidate_selection.best_update) is not int
              or not 0 <= candidate_selection.best_update <= progress["applied"]
              or candidate_selection.supported_aspects is None
              or candidate_selection.evaluation_fingerprint is None
              or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                     for v in (candidate_selection.best_accuracy, candidate_selection.best_pair_loss))
              or not 0 <= candidate_selection.best_accuracy <= 1 or candidate_selection.best_pair_loss < 0):
            raise ValueError("critic.training.resume_selection_invalid")
        current_model = self.model.state_dict()
        if (not isinstance(state["model"], Mapping) or state["model"].keys() != current_model.keys()
                or not _finite_tree(state["model"])):
            raise ValueError("critic.training.resume_model_invalid")
        for key, tensor in state["model"].items():
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != current_model[key].shape
                    or tensor.dtype != current_model[key].dtype):
                raise ValueError("critic.training.resume_model_invalid")
        # Validate optimizer slots with a disposable copy, leaving real weights/RNG untouched.
        candidate_model = copy.deepcopy(self.model)
        candidate_model.load_state_dict(state["model"], strict=True)
        candidate_parameters = [p for p in candidate_model.parameters() if p.requires_grad]
        candidate_optimizer = torch.optim.AdamW(candidate_parameters, lr=self.config.learning_rate,
                                              weight_decay=self.config.weight_decay)
        optimizer_state = state["optimizer"]
        if not isinstance(optimizer_state, dict) or set(optimizer_state) != {"param_groups", "state"}:
            raise ValueError("critic.training.resume_optimizer_invalid")
        own_groups = self.optimizer.state_dict()["param_groups"]
        if optimizer_state["param_groups"] != own_groups or not _finite_tree(optimizer_state):
            raise ValueError("critic.training.resume_optimizer_invalid")
        try:
            candidate_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            for param, slots in candidate_optimizer.state.items():
                if set(slots) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("unexpected optimizer slot")
                if (slots["exp_avg"].shape != param.shape or slots["exp_avg_sq"].shape != param.shape
                        or slots["exp_avg"].dtype != param.dtype or slots["exp_avg_sq"].dtype != param.dtype
                        or slots["step"].numel() != 1
                        or not 0 <= float(slots["step"]) <= progress["applied"]):
                    raise ValueError("optimizer slot mismatch")
            if progress["applied"] and not candidate_optimizer.state:
                raise ValueError("missing optimizer slots")
        except (RuntimeError, TypeError, ValueError, KeyError, AttributeError) as exc:
            raise ValueError("critic.training.resume_optimizer_invalid") from exc
        before_model = copy.deepcopy(self.model.state_dict())
        before_optimizer = copy.deepcopy(self.optimizer.state_dict())
        try:
            self.model.load_state_dict(state["model"], strict=True)
            self.optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        except BaseException:
            self.model.load_state_dict(before_model, strict=True)
            self.optimizer.load_state_dict(before_optimizer)
            raise
        self.sampler = candidate_sampler
        self.cursor = sampler["cursor"]
        self.rng_state = copy.deepcopy(state["rng"])
        self.selection = candidate_selection
        self.attempted_updates, self.applied_updates = progress["attempted"], progress["applied"]
        self.skipped_updates, self.draws = progress["skipped"], progress["draws"]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            torch.save(self.state_dict(), temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path: str | Path) -> None:
        try:
            state = torch.load(Path(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("critic.training.resume_unreadable") from exc
        self.load_state_dict(state)
