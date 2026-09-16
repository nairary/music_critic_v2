"""Bounded, opt-in WS2 masking comparison; no corpus discovery or TEST loader.

Only the Phase 8A policy changes. Existing note/bar/song losses and the complete
corrected theory branch are reused. Callers bind TRAIN batches before execution;
real corpus provenance and held-out evaluation remain coordinator-owned gates.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Callable

import torch

from music_critic.experiments.analysisgnn.contracts import fingerprint
from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.corrected_training import ACTIVE_HEADS, FrozenClassWeights
from music_critic.experiments.analysisgnn.multitask_contract import TASK_BY_ID
from music_critic.ssl.data import SSLBatch, validate_ssl_batch, move_ssl_batch
from music_critic.ssl.autocast_runtime import autocast_runtime_binding
from music_critic.ssl.hierarchical_masking import HierarchyMaskPolicyConfig
from music_critic.ssl.hierarchy_leakage import build_phase8a_pitch_leakage_audit
from music_critic.ssl.masking import (
    prepare_hierarchy_mask_binding, move_ssl_batch_with_prepared_binding,
)
from music_critic.ssl.model import MaskedGraphSSLConfig
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness
from music_critic.ssl.theory_warmstart import TheoryWarmstartResult
from music_critic.ssl.group_masking_accounting import validate_successful_row

VERSION = "ws2-group-masking-pilot@1.1.0"
POLICIES = ("independent_note_pitch", "beat_pitch_descendants")
RESUME_SCHEMA = "WS2GroupMaskingResume@1.1.0"
RNG_DOMAIN = "ws2-group-masking-pilot@1.0.0:update"


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("group_pilot.sha256_required")
    return value


def _plain(value):
    """Typed tensor tree identity, including optimizer parameter IDs."""
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        return {"tensor": sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                "dtype": str(cpu.dtype), "shape": list(cpu.shape)}
    if isinstance(value, dict):
        return {"mapping": sorted([[type(k).__name__, str(k), _plain(v)] for k, v in value.items()])}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_plain(v) for v in value]}
    return value


def tree_fingerprint(value):
    return fingerprint(_plain(value))


@dataclass(frozen=True)
class GroupPilotConfig:
    seed: int = 17
    updates: int = 200
    mask_rate: float = 0.30
    lambda_ssl: float = 0.7
    lambda_theory: float = 0.3
    learning_rate: float = 0.0003
    max_samples_per_batch: int = 2
    max_notes_per_batch: int = 8192
    max_nodes_per_batch: int = 20000
    max_edges_per_batch: int = 200000

    def __post_init__(self):
        for key, maximum in (("seed", 2**31 - 1), ("updates", 1000),
                             ("max_samples_per_batch", 2), ("max_notes_per_batch", 8192),
                             ("max_nodes_per_batch", 20000), ("max_edges_per_batch", 200000)):
            value = getattr(self, key)
            if type(value) is not int or not (0 if key == "seed" else 1) <= value <= maximum:
                raise ValueError(f"group_pilot.invalid_{key}")
        for key in ("mask_rate", "lambda_ssl", "lambda_theory", "learning_rate"):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"group_pilot.invalid_{key}")
        if not 0 < self.mask_rate < 1 or not 0 < self.learning_rate <= 0.001:
            raise ValueError("group_pilot.rate_outside_pilot_bounds")
        if self.lambda_ssl + self.lambda_theory <= 0:
            raise ValueError("group_pilot.no_active_objective")

    def ssl_config(self):
        return MaskedGraphSSLConfig(mask_rate=self.mask_rate)

    def contract(self):
        return {"version": VERSION, "config": asdict(self), "policies": list(POLICIES),
                "ssl_config": asdict(self.ssl_config()),
                "leakage_audit": build_phase8a_pitch_leakage_audit().to_dict(),
                "optimizer": {"kind": "Adam", "weight_decay": 0, "scheduler": None,
                              "amp": False, "grad_clip_norm": None},
                "rng": "per_update_torch_fork_seed_sha256_same_across_arms",
                "rng_domain": RNG_DOMAIN,
                "mask_epoch": "zero_based_applied_update", "test_access": "closed"}


@dataclass(frozen=True)
class PilotBatch:
    ssl: SSLBatch | None = None
    theory: SSLBatch | None = None
    sidecars: tuple[dict, ...] = ()
    class_weights: FrozenClassWeights | None = None


def _batch_evidence(batch, config):
    if batch is None:
        return None
    # Check numerical budgets before materializing graph serialization.
    validate_ssl_batch(batch)
    if (batch.sample_count > config.max_samples_per_batch or batch.node_count > config.max_nodes_per_batch
            or batch.edge_count > config.max_edges_per_batch
            or batch.raw_graph_batch["note"].num_nodes > config.max_notes_per_batch):
        raise ValueError("group_pilot.batch_resource_bound")
    if any(t.device.type != "cpu" for t in batch.raw_graph_batch["note"].values() if isinstance(t, torch.Tensor)):
        raise ValueError("group_pilot.cpu_preparation_required")
    return {"identities": list(zip(batch.dataset_ids, batch.piece_ids)),
            "raw_batch_fingerprint": tree_fingerprint(batch.raw_graph_batch.to_dict())}


def train_batch_evidence(batch: PilotBatch, config: GroupPilotConfig, *, split: str = "train") -> dict:
    """Bind actual raw bytes, target sidecars and weight tensors independently.

    This validates a caller-prepared TRAIN batch, not the authority of its split
    declaration. The real-data factory must attest full split/component closure.
    """
    if split != "train":
        raise ValueError("group_pilot.train_only")
    if config.lambda_ssl > 0 and batch.ssl is None:
        raise ValueError("group_pilot.ssl_batch_required")
    if config.lambda_theory > 0 and (batch.theory is None or not batch.sidecars or batch.class_weights is None):
        raise ValueError("group_pilot.theory_replay_required")
    if batch.theory is None and batch.sidecars:
        raise ValueError("group_pilot.sidecars_require_raw_batch")
    weights = batch.class_weights
    content = {"ssl": _batch_evidence(batch.ssl, config),
        "theory": _batch_evidence(batch.theory, config), "sidecars": batch.sidecars,
        "weights": None if weights is None else {
            "declared_fingerprint": weights.payload_fingerprint,
            "values": dict(weights.values), "supported": dict(weights.supported)}, "split": split}
    def accounting(raw):
        if raw is None:
            return None
        return {"identities": [list(pair) for pair in zip(raw.dataset_ids, raw.piece_ids)],
                "note_counts": torch.bincount(raw.raw_graph_batch["note"].batch,
                                               minlength=raw.sample_count).tolist(),
                "node_count": raw.node_count, "edge_count": raw.edge_count}
    target_rows = None
    if batch.theory is not None:
        # Bind population, not label values or an invented raw-node ceiling:
        # several native targets can legitimately map to one raw candidate.
        # Availability/alignment/loss processing remains after raw prediction.
        target_rows = dict.fromkeys(ACTIVE_HEADS, 0)
        for sidecar in batch.sidecars:
            for entity in sidecar["entities"]:
                level, targets = str(entity["entity_type"]), entity.get("targets", {})
                if level not in {"harmonic_event", "onset", "note"} or not isinstance(targets, Mapping):
                    continue
                for task in targets:
                    if task in target_rows and TASK_BY_ID[task].prediction_level == level:
                        target_rows[task] += 1
    return {"content_fingerprint": tree_fingerprint(content),
            "accounting": {"ssl": accounting(batch.ssl), "theory": accounting(batch.theory),
                           "theory_target_rows": target_rows}}


def bind_train_batch(batch: PilotBatch, config: GroupPilotConfig, *, split: str = "train") -> str:
    return tree_fingerprint(train_batch_evidence(batch, config, split=split))


def paired_mask_bindings(batch, config, *, update):
    """Both policies must be eligible before either arm can update; no fallback."""
    return {policy: prepare_hierarchy_mask_binding(
        batch, policy_config=HierarchyMaskPolicyConfig.create(weights={policy: 1.0}),
        global_seed=config.seed, epoch=update, requested_mask_rate=config.mask_rate, stage="train",
    ) for policy in POLICIES}


def mask_accounting(binding):
    return [{"piece": [plan.dataset_id, plan.piece_id], "plan_fingerprint": plan.fingerprint,
             "hidden_notes": plan.selected_count, "eligible_notes": plan.maskable_node_count,
             "hidden_fraction": plan.selected_count / plan.maskable_node_count,
             "hidden_units": len(getattr(plan, "selected_local_unit_indices", plan.selected_local_node_indices)),
             "unit": getattr(plan, "selected_unit_node_type", "note"),
             "candidate_units": getattr(getattr(plan, "selection", None), "total_valid_candidate_count", plan.maskable_node_count),
             "collateral_rows": {kind: len({i for mask in plan.collateral_feature_masks if mask.node_type == kind
                                              for i in mask.local_node_indices}) for kind in ("note", "track")},
             "requested_fraction": plan.requested_mask_rate}
            for plan in binding.mask_plans]


@contextmanager
def _step_rng(seed, update, device):
    devices = [device.index] if device.type == "cuda" else []
    derived = int(fingerprint({"domain": RNG_DOMAIN, "seed": seed, "update": update})[:15], 16)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(derived)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(derived)
        yield


def _scalar(value):
    return None if value is None else float(value.detach().cpu())


def _runtime_binding(device):
    """Read the actual process runtime without initializing CUDA or changing flags."""
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            **autocast_runtime_binding(device.type),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                                       if device.type == "cuda" else None}


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tensors(v) for v in value)
    return True


class GroupMaskingPilot:
    """One arm, explicit finite schedule, FP32 Adam and resumable update boundary.

    No caller RNG is consumed by a step. Save only after a successful update.
    An interrupted/failed step poisons this instance; recreate and resume the
    last hash-bound completed checkpoint. CUDA exact replay needs the existing
    deterministic CUDA evidence runtime; CPU is the tested reference.
    """

    def __init__(self, warmstart: TheoryWarmstartResult, config: GroupPilotConfig, *,
                 policy: str, schedule: tuple[str, ...], source_binding: dict, device="cpu"):
        if policy not in POLICIES:
            raise ValueError("group_pilot.unsupported_policy")
        if not isinstance(schedule, tuple) or len(schedule) != config.updates:
            raise ValueError("group_pilot.complete_bounded_schedule_required")
        for entry in schedule:
            _sha(entry)
        if warmstart.ssl_model.ssl_config != config.ssl_config():
            raise ValueError("group_pilot.ssl_config_mismatch")
        if source_binding.get("state_fingerprint") != warmstart.report.source_state_fingerprint:
            raise ValueError("group_pilot.source_state_binding_mismatch")
        if model_state_fingerprint(warmstart.theory_model) != warmstart.report.source_state_fingerprint:
            raise ValueError("group_pilot.source_state_mutated_after_transfer")
        if warmstart.report.provenance == "verified_model_only_artifact":
            if source_binding.get("kind") != "C2_model_only" or not source_binding.get("parent_experiment_id"):
                raise ValueError("group_pilot.C2_provenance_binding_required")
            _sha(source_binding.get("sha256"))
            if type(source_binding.get("size_bytes")) is not int or source_binding["size_bytes"] <= 0:
                raise ValueError("group_pilot.source_size_required")
            evidence = warmstart.artifact_evidence or {}
            if any(source_binding[key] != evidence.get(artifact_key) for key, artifact_key in (
                    ("sha256", "artifact_sha256"), ("size_bytes", "artifact_size_bytes"),
                    ("state_fingerprint", "model_state_fingerprint"))):
                raise ValueError("group_pilot.source_artifact_binding_mismatch")
        elif source_binding.get("kind") != "synthetic":
            raise ValueError("group_pilot.synthetic_source_mislabeled")
        self.config, self.policy, self.schedule = config, policy, schedule
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"} or (self.device.type == "cuda" and self.device.index is None):
            raise ValueError("group_pilot.explicit_cpu_or_indexed_cuda_required")
        self.harness = TheoryRetentionHarness(warmstart).to(self.device)
        self.optimizer = torch.optim.Adam(self.harness.optimizer_parameters(), lr=config.learning_rate)
        self.initial_fingerprint = model_state_fingerprint(self.harness)
        self.binding = {"contract": config.contract(), "policy": policy, "schedule": list(schedule),
                        "source": deepcopy(source_binding), "initial_state": self.initial_fingerprint,
                        "transfer_report": asdict(warmstart.report),
                        "optimizer_inventory": self.harness.parameter_inventory(),
                        "device": str(self.device),
                        "runtime": _runtime_binding(self.device)}
        self.binding_fingerprint = tree_fingerprint(self.binding)
        self.completed_updates = 0
        self.rows: list[dict] = []
        self.failed = False
        self.last_attempt = None

    def _validate_runtime(self):
        if tree_fingerprint(_runtime_binding(self.device)) != tree_fingerprint(self.binding["runtime"]):
            raise ValueError("group_pilot.runtime_drift")

    def _validate_finite_state(self):
        if not _finite_tensors(self.harness.state_dict()):
            raise ValueError("group_pilot.nonfinite_model_state")
        if not _finite_tensors(self.optimizer.state_dict()):
            raise ValueError("group_pilot.nonfinite_optimizer_state")

    def step_from_factory(self, factory: Callable[[], PilotBatch], *, split="train", on_attempt=None):
        # Guard precedes external reads; caller cannot request TEST here.
        if split != "train":
            raise ValueError("group_pilot.train_only")
        if self.failed:
            raise ValueError("group_pilot.failed_instance_requires_resume")
        if self.completed_updates >= self.config.updates:
            raise ValueError("group_pilot.budget_exhausted")
        count, handle, optimizer_entered = [0], None, False
        self.last_attempt = {"schema": "WS2GroupMaskingAttempt@1.0.0",
            "update": self.completed_updates + 1, "completed_updates_before": self.completed_updates,
            "batch_binding": self.schedule[self.completed_updates], "status": "started",
            "stage": "started", "observed_encoder_forwards": 0,
            "optimizer_outcome": "not_applied", "confirmed_updates": self.completed_updates}

        def receipt(stage, *, status="in_progress", **fields):
            self.last_attempt.update(stage=stage, status=status, observed_encoder_forwards=count[0],
                                     confirmed_updates=self.completed_updates, **fields)
            if on_attempt is not None:
                on_attempt(deepcopy(self.last_attempt))

        def observed_forward(*_):
            count[0] += 1
            receipt("forward")

        try:
            receipt("started", status="started")
            receipt("runtime_validation")
            self._validate_runtime()
            self._validate_finite_state()
            receipt("factory")
            batch = factory()
            receipt("batch_validation")
            evidence = train_batch_evidence(batch, self.config, split=split)
            if tree_fingerprint(evidence) != self.schedule[self.completed_updates]:
                raise ValueError("group_pilot.actual_schedule_mismatch")
            ssl, theory, selected = batch.ssl, batch.theory, None
            masks = {}
            receipt("masking")
            if self.config.lambda_ssl:
                paired = paired_mask_bindings(ssl, self.config, update=self.completed_updates)
                masks = {policy: mask_accounting(b) for policy, b in paired.items()}
                selected = paired[self.policy]
            receipt("device_transfer")
            if self.config.lambda_ssl and self.device.type != "cpu":
                ssl, selected = move_ssl_batch_with_prepared_binding(ssl, selected, self.device)
            theory_graph = None if theory is None else (
                theory.raw_graph_batch if self.device.type == "cpu"
                else move_ssl_batch(theory, self.device).raw_graph_batch)
            handle = self.harness.theory_model.context_encoder.transformer.register_forward_hook(observed_forward)
            self.harness.train()
            self.optimizer.zero_grad(set_to_none=True)
            with _step_rng(self.config.seed, self.completed_updates, self.device):
                receipt("forward")
                output = self.harness(
                    lambda_ssl=self.config.lambda_ssl, lambda_theory=self.config.lambda_theory,
                    ssl_batch=ssl if self.config.lambda_ssl else None,
                    prepared_mask_binding=selected,
                    theory_raw_batch=theory_graph if self.config.lambda_theory else None,
                    theory_sidecars=batch.sidecars if self.config.lambda_theory else None,
                    class_weights=batch.class_weights if self.config.lambda_theory else None)
                if output.total is None or not bool(torch.isfinite(output.total)):
                    raise ValueError("group_pilot.unavailable_or_nonfinite_objective")
                if self.config.lambda_ssl and output.ssl_loss is None:
                    raise ValueError("group_pilot.ssl_loss_missing")
                if self.config.lambda_theory and output.theory_loss is None:
                    raise ValueError("group_pilot.theory_loss_missing")
                expected_forwards = 2 * int(self.config.lambda_ssl > 0) + int(self.config.lambda_theory > 0)
                if count[0] != expected_forwards:
                    raise ValueError("group_pilot.encoder_forward_accounting_mismatch")
                receipt("backward")
                output.total.backward()
                if any(not bool(torch.isfinite(p.grad).all()) for p in self.harness.optimizer_parameters() if p.grad is not None):
                    raise ValueError("group_pilot.nonfinite_gradient")
                gradient_indices = [i for i, p in enumerate(self.harness.optimizer_parameters()) if p.grad is not None]
                self._validate_runtime()
                receipt("optimizer_step", optimizer_outcome="unknown")
                optimizer_entered = True
                self.optimizer.step()
                receipt("post_step_validation", optimizer_outcome="applied")
                self._validate_runtime()
                self._validate_finite_state()
            receipt("accounting")
            ssl_output = output.ssl_output
            row = {"update": self.completed_updates + 1, "batch_binding": self.schedule[self.completed_updates],
                   "batch_evidence": evidence, "encoder_forwards": count[0], "paired_masks": masks,
                   "optimizer_gradient_parameter_indices": gradient_indices,
                   "ssl_loss": _scalar(output.ssl_loss), "theory_loss": _scalar(output.theory_loss),
                   "total_loss": _scalar(output.total), "counts": dict(output.counts),
                   "collapse": None if ssl_output is None else {
                       "note": ssl_output.note_diagnostics.to_dict(),
                       "bar": ssl_output.bar_latent.diagnostics.to_dict(),
                       "song": ssl_output.song_latent.diagnostics.to_dict()},
                   "theory_heads": None if output.theory_report is None else {
                       task: {"weighted_ce": _scalar(head.weighted_ce), "valid": head.valid_row_count,
                              "masked": head.masked_row_count, "unsupported": head.unsupported_row_count}
                       for task, head in output.theory_report.heads.items()}}
            validate_successful_row(row, config=self.config, policy=self.policy,
                                    scheduled_binding=self.schedule[self.completed_updates],
                                    update_index=self.completed_updates)
            self.rows.append(row)
            self.completed_updates += 1
            receipt("completed", status="completed")
            return deepcopy(row)
        except BaseException as exc:
            self.failed = True
            if not optimizer_entered:
                self.last_attempt["optimizer_outcome"] = "not_applied"
            # A raised optimizer call may have mutated all, some, or no state.
            # Do not claim rollback or infer application from an exception.
            try:
                receipt(self.last_attempt["stage"], status="failed", error_type=type(exc).__name__, reason=str(exc))
            except BaseException:
                pass  # Keep the original failure; last_attempt remains in memory.
            raise
        finally:
            if handle is not None:
                handle.remove()
            self.optimizer.zero_grad(set_to_none=True)

    def checkpoint_payload(self):
        if self.failed:
            raise ValueError("group_pilot.cannot_checkpoint_failed_step")
        self._validate_runtime()
        self._validate_finite_state()
        payload = {"schema": RESUME_SCHEMA, "binding_fingerprint": self.binding_fingerprint,
                   "binding": deepcopy(self.binding), "completed_updates": self.completed_updates,
                   "rows": deepcopy(self.rows), "model": {k: v.detach().cpu().clone() for k, v in self.harness.state_dict().items()},
                   "optimizer": deepcopy(self.optimizer.state_dict())}
        payload["payload_fingerprint"] = tree_fingerprint(payload)
        return payload

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_payload()
        # Exclusive final publication never overwrites a terminal artifact.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.link(temporary, path)
            finally:
                temporary.unlink()
        content = path.read_bytes()
        return {"sha256": sha256(content).hexdigest(), "size_bytes": len(content), "path": str(path)}

    def resume(self, path, *, expected_sha256, expected_size_bytes):
        if self.completed_updates or self.failed:
            raise ValueError("group_pilot.resume_requires_fresh_instance")
        self._validate_runtime()
        _sha(expected_sha256)
        if type(expected_size_bytes) is not int or not 0 < expected_size_bytes <= 256 * 1024**2:
            raise ValueError("group_pilot.resume_size_bound")
        with Path(path).open("rb") as stream:
            raw = stream.read(expected_size_bytes + 1)
        if len(raw) != expected_size_bytes or sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("group_pilot.resume_artifact_mismatch")
        payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
        expected_keys = {"schema", "binding_fingerprint", "binding", "completed_updates", "rows", "model", "optimizer", "payload_fingerprint"}
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("group_pilot.resume_envelope_mismatch")
        unhashed = {k: v for k, v in payload.items() if k != "payload_fingerprint"}
        if payload["payload_fingerprint"] != tree_fingerprint(unhashed):
            raise ValueError("group_pilot.resume_payload_mismatch")
        if (payload["schema"] != RESUME_SCHEMA or payload["binding_fingerprint"] != self.binding_fingerprint
                or tree_fingerprint(payload["binding"]) != self.binding_fingerprint):
            raise ValueError("group_pilot.resume_binding_mismatch")
        step = payload["completed_updates"]
        if (type(step) is not int or not 0 <= step <= self.config.updates
                or type(payload["rows"]) is not list or len(payload["rows"]) != step):
            raise ValueError("group_pilot.resume_progress_mismatch")
        for i, row in enumerate(payload["rows"]):
            validate_successful_row(row, config=self.config, policy=self.policy,
                                    scheduled_binding=self.schedule[i], update_index=i)
        state = payload["model"]
        expected = self.harness.state_dict()
        if set(state) != set(expected) or any(type(state[k]) is not torch.Tensor or state[k].shape != v.shape
                or state[k].dtype != v.dtype or not bool(torch.isfinite(state[k]).all()) for k, v in expected.items()):
            raise ValueError("group_pilot.resume_model_invalid")
        if step == 0 and model_state_fingerprint(state) != self.initial_fingerprint:
            raise ValueError("group_pilot.resume_initial_state_mismatch")
        from music_critic.ssl.theory_warmstart import BACKBONE_GROUPS
        for key in state:
            prefix = "theory_model."
            if key.startswith(tuple(prefix + group + "." for group in BACKBONE_GROUPS)):
                if not torch.equal(state[key], state["ssl_model.encoder." + key[len(prefix):]]):
                    raise ValueError("group_pilot.resume_shared_alias_conflict")
        optimizer = payload["optimizer"]
        reference_groups = self.optimizer.state_dict()["param_groups"]
        if optimizer["param_groups"] != reference_groups:
            raise ValueError("group_pilot.resume_optimizer_ownership_mismatch")
        parameters = self.harness.optimizer_parameters()
        counts = {}
        for row in payload["rows"]:
            indices = row["optimizer_gradient_parameter_indices"]
            if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(parameters) for i in indices) or indices != sorted(set(indices)):
                raise ValueError("group_pilot.resume_gradient_inventory_invalid")
            for index in indices:
                counts[index] = counts.get(index, 0) + 1
        if set(counts) != set(optimizer["state"]) or (step > 0 and not counts):
            raise ValueError("group_pilot.resume_optimizer_state_missing_or_unexpected")
        for index, slots in optimizer["state"].items():
            if type(index) is not int or not 0 <= index < len(parameters) or set(slots) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("group_pilot.resume_optimizer_invalid")
            for name, tensor in slots.items():
                shape = torch.Size([]) if name == "step" else parameters[index].shape
                if type(tensor) is not torch.Tensor or tensor.shape != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("group_pilot.resume_optimizer_tensor_invalid")
            if float(slots["step"]) != counts[index]:
                raise ValueError("group_pilot.resume_optimizer_step_invalid")
            if bool((slots["exp_avg_sq"] < 0).any()):
                raise ValueError("group_pilot.resume_optimizer_negative_variance")
        self._validate_runtime()
        original = deepcopy(self.harness.state_dict())
        try:
            self.harness.load_state_dict(state, strict=True)
            self.optimizer.load_state_dict(optimizer)
        except BaseException:
            self.harness.load_state_dict(original, strict=True)
            self.optimizer = torch.optim.Adam(self.harness.optimizer_parameters(), lr=self.config.learning_rate)
            raise
        self.rows, self.completed_updates = deepcopy(payload["rows"]), step
