"""Synthetic-only exact historical C2 update/prefetch/checkpoint gate.

This has its own protocol. It neither relaxes the C0 identity descriptor nor
changes frozen transposition, including its stale track pitch statistics.
Imports are lazy: importing this module starts no Torch runtime or workers.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch


SCHEMA = "HistoricalC2SyntheticEquivalence@1.0.0"
SCHEDULE_SCHEMA = "HistoricalC2EquivalenceSchedule@1.0.0"
WORKER_COUNTS = (0, 1, 2, 4)
SHIFTS = tuple(range(12))
SIGNED = (0, 1, 2, 3, 4, 5, 6, -5, -4, -3, -2, -1)
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class C2EquivalenceError(ValueError):
    pass


def _require(condition, category):
    if not condition:
        raise C2EquivalenceError("c2_equivalence." + category)


def require_thread_environment(device):
    _require(device in ("cpu", "cuda"), "invalid_device")
    observed = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if device == "cuda":
        _require(all(value == "1" for value in observed.values()), "cuda_requires_inherited_thread_limits")
    return observed


def _json_fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _normalized(value):
    """Preserve mapping keys and tuple/list identity, detach Tensor leaves."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _normalized(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_normalized(item) for item in value)
    return deepcopy(value)


def _tensor_fingerprint(value):
    from music_critic.experiments.analysisgnn.performance_replay import recursive_state_fingerprint
    return recursive_state_fingerprint(_normalized(value))


def raw_fingerprint(raw):
    return _tensor_fingerprint({"graph": raw.to_dict(), "num_graphs": raw.num_graphs,
                                "slice": raw._slice_dict, "inc": raw._inc_dict})


def _model_plan(model=None):
    from music_critic.experiments.analysisgnn.corrected_model import (
        CORRECTED_MODEL_ID, CorrectedAnalysisGNNConfig, corrected_task_head_specs,
    )
    return {"model_id": CORRECTED_MODEL_ID,
            "config": asdict(CorrectedAnalysisGNNConfig() if model is None else model.config),
            "task_specs": [asdict(spec) for spec in (corrected_task_head_specs() if model is None else model.task_specs)]}


def _plan(device, *, shifts=SHIFTS, worker_counts=WORKER_COUNTS, resume_after=5):
    """Private reduced plans serve unit tests only; public/CLI plan is fixed."""
    from music_critic.experiments.analysisgnn import corrected_training as reference
    from music_critic.experiments.analysisgnn.full_orbit_training import (
        FULL_ORBIT_PROFILE_ID, FULL_ORBIT_UPDATE_BUDGET,
        FULL_ORBIT_WARMUP_UPDATES, FULL_ORBIT_PEAK_LEARNING_RATE,
    )
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchConfig, PrefetchJob
    from music_critic.experiments.analysisgnn.performance_prefetch_replay import worker_configurations
    from music_critic.experiments.analysisgnn.transposition import directed_transposition_contract
    _require(device in ("cpu", "cuda"), "invalid_device")
    _require(isinstance(shifts, tuple) and len(shifts) >= 2 and len(set(shifts)) == len(shifts)
             and all(type(s) is int and s in SHIFTS for s in shifts), "invalid_test_shift_plan")
    _require(type(resume_after) is int and 0 < resume_after < len(shifts), "invalid_resume_position")
    config = PrefetchConfig()
    configs = worker_configurations(config, worker_counts)
    jobs = tuple(PrefetchJob(index, (2*index, 2*index+1),
        ("synthetic:c2:A", "synthetic:c2:A" if shift == 6 else "synthetic:c2:B"),
        (shift, (shift+5) % 12), (shift, (shift+5) % 12), 2_000_000)
        for index, shift in enumerate(shifts))
    schedule = {"schema": SCHEDULE_SCHEMA, "batch_size": 2, "jobs": [asdict(job) for job in jobs],
                "scope": "synthetic_shift_coverage_not_a_historical_orbit_stream"}
    schedule = json.loads(json.dumps(schedule))
    schedule["fingerprint"] = _json_fingerprint(schedule)
    arms = [{"arm_id": "reference", "kind": "baseline", "worker_count": 0,
             "update_strategy": "original", "resume": False}]
    arms += [{"arm_id": f"optimized-w{count}-{'resumed' if resume else 'uninterrupted'}",
              "kind": "optimized", "worker_count": count, "update_strategy": "optimized", "resume": resume}
             for resume in (False, True) for count in configs]
    weights = reference.load_frozen_class_weights()
    vocabulary = {task: {"vocabulary_id": reference.TASK_BY_ID[task].vocabulary_id,
                         "labels": reference.get_vocabulary(reference.TASK_BY_ID[task].vocabulary_id).labels,
                         "prediction_level": reference.TASK_BY_ID[task].prediction_level}
                  for task in reference.ACTIVE_HEADS}
    return {"schema": SCHEMA, "comparison_kind": "historical_c2_equivalence", "mode": "fixture",
        "device": device, "repeats": 1, "batch_size": 2, "batch_sizes": [2],
        "measured_draws": 2*len(jobs), "warmup_draws": 0, "planned_arm_count": len(arms),
        "planned_updates": len(arms)*len(jobs), "planned_record_draws": 2*len(arms)*len(jobs),
        "worker_counts": list(configs), "worker_configs": {str(n): asdict(c) for n, c in configs.items()},
        "worker_configs_fingerprint": _json_fingerprint({str(n): asdict(c) for n, c in configs.items()}),
        "schedule": schedule, "schedule_descriptor_fingerprint": schedule["fingerprint"],
        "reservation_manifest_fingerprint": _json_fingerprint({"synthetic_only": True, "reserved_bytes": 2_000_000}),
        "arm_manifest": arms, "arm_manifest_fingerprint": _json_fingerprint(arms),
        "resume_after_updates": resume_after,
        "full_protocol_exercised": shifts == SHIFTS and tuple(configs) == WORKER_COUNTS and resume_after == 5,
        "historical_profile_id": FULL_ORBIT_PROFILE_ID,
        "historical_transform_fingerprint": _json_fingerprint(directed_transposition_contract()),
        "model_plan_fingerprint": _tensor_fingerprint(_model_plan()),
        "vocabulary_fingerprint": _tensor_fingerprint(vocabulary),
        "class_weight_fingerprint": weights.payload_fingerprint,
        "class_weight_tensor_fingerprint": _tensor_fingerprint(weights),
        "class_weight_fixture_sha256": sha256(reference._B5B_FIXTURE.read_bytes()).hexdigest(),
        "optimizer_plan": {"frozen_envelope": reference.corrected_optimizer_envelope(),
            "schedule_updates": FULL_ORBIT_UPDATE_BUDGET, "warmup_updates": FULL_ORBIT_WARMUP_UPDATES,
            "peak_lr": FULL_ORBIT_PEAK_LEARNING_RATE, "gradient_clip_norm": 1.0,
            "precision": "fp32_baseline", "scaler_enabled": False},
        "seeds": {"model": 17, "dropout": 17001, "worker_base": 17291},
        "reference_data_strategy": "frozen", "optimized_data_strategy": "frozen",
        "track_statistics_policy": "historical_stale_mean_min_max_preserved",
        "canonical_checkpoint_input_used": False, "quality_claims": False,
        "question": "Does optimized prefetch preserve historical C2 shifted updates and serialized resume exactly?",
        "hypothesis": "All nine synthetic branches preserve inputs, logits, masks, losses, gradients, full AdamW and RNG.",
        "decision_use": "Synthetic runtime gate before separate real TRAIN C2 acceptance; no speed, RAM or quality claim."}


def build_protocol(device="cuda"):
    return _plan(device)


def _jobs(protocol):
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchJob
    descriptor = deepcopy(protocol["schedule"])
    seal = descriptor.pop("fingerprint")
    _require(seal == _json_fingerprint(descriptor) == protocol["schedule_descriptor_fingerprint"], "schedule_seal_mismatch")
    _require(descriptor["schema"] == SCHEDULE_SCHEMA and descriptor["batch_size"] == 2, "schedule_schema_mismatch")
    jobs = tuple(PrefetchJob(row["position"], tuple(row["draw_ids"]), tuple(row["record_ids"]),
                tuple(row["schedule_shifts"]), tuple(row["applied_shifts"]), row["reserved_bytes"])
                for row in descriptor["jobs"])
    for index, job in enumerate(jobs):
        _require(job.position == index and job.draw_ids == (2*index, 2*index+1)
                 and job.schedule_shifts == job.applied_shifts
                 and job.applied_shifts[1] == (job.applied_shifts[0]+5) % 12, "schedule_job_mismatch")
    return jobs


def assert_historical_raw_view(source, actual, shifts):
    """Independent arithmetic oracle; never calls either transposition API."""
    import torch
    from torch_geometric.data import Batch
    from music_critic.graph import validate_raw_graph_batch
    validate_raw_graph_batch(source, sample_count=2)
    validate_raw_graph_batch(actual, sample_count=2)
    originals, observed = source.to_data_list(), actual.to_data_list()
    _require(len(shifts) == len(originals) == len(observed) == 2, "raw_batch_size_mismatch")
    expected, moved_notes = [], 0
    for original, shifted, shift in zip(originals, observed, shifts, strict=True):
        _require(type(shift) is int and shift in SHIFTS, "raw_shift_invalid")
        _require(_tensor_fingerprint(original["track"].to_dict()) == _tensor_fingerprint(shifted["track"].to_dict()),
                 "historical_track_statistics_changed")
        names = original["track"].cont_feature_names
        _require(all(name in names for name in ("mean_pitch", "min_pitch", "max_pitch", "pitch_std")),
                 "track_statistics_columns_missing")
        view = deepcopy(original)
        if shift:
            note = view["note"]
            pitch_i, pc_i, octave_i, drum_i = (note.cat_feature_names.index(name)
                for name in ("pitch", "pitch_class", "octave", "is_percussion"))
            pitches = note.x_cat[:, pitch_i].clone()
            pitched = note.x_cat[:, drum_i].eq(0)
            pitches[pitched] += SIGNED[shift]
            _require(not pitches[pitched].numel() or bool(((pitches[pitched] >= 0) & (pitches[pitched] <= 127)).all()),
                     "oracle_midi_range_violation")
            moved_notes += int(pitched.sum())
            note.x_cat[:, pitch_i] = pitches
            note.x_cat[pitched, pc_i] = pitches[pitched].remainder(12)
            note.x_cat[pitched, octave_i] = torch.div(pitches[pitched], 12, rounding_mode="floor")
            relation = ("track", "contains_note", "note")
            if "track_relative_pitch" in note.cont_feature_names and relation in view.edge_types:
                relative_i = note.cont_feature_names.index("track_relative_pitch")
                note.x_cont[:, relative_i] = 0
                note.x_cont_available[:, relative_i] = False
                edges = view[relation].edge_index
                for owner in edges[0].unique(sorted=True).tolist():
                    indices = edges[1, edges[0].eq(owner)]
                    values = pitches[indices].to(torch.float64)
                    if values.numel():
                        mean = values.mean()
                        std = torch.sqrt(((values-mean)**2).mean())
                        if std.item() > 0:
                            note.x_cont[indices, relative_i] = ((values-mean)/std).to(note.x_cont.dtype)
                            note.x_cont_available[indices, relative_i] = True
        _require(_tensor_fingerprint(view.to_dict()) == _tensor_fingerprint(shifted.to_dict()), "historical_raw_oracle_mismatch")
        expected.append(view)
    _require(raw_fingerprint(Batch.from_data_list(expected)) == raw_fingerprint(actual), "raw_reconstruction_metadata_mismatch")
    return {"historical_track_statistics_preserved": True, "nonzero_pitched_notes_checked": moved_notes}


def _unshifted_fixture(job):
    from music_critic.experiments.analysisgnn import performance_replay as existing
    loaded = [existing._fixture_callback(record) for record in job.record_ids]
    raw = existing.reference.combine_single_record_raw_batches([row[0].raw_graph_batch for row in loaded])
    return raw, tuple(row[1] for row in loaded)


def fixture_metadata_preflight():
    from music_critic.experiments.analysisgnn import corrected_training as reference
    from music_critic.experiments.analysisgnn.transposition import transform_semantic_value
    _, sidecar = reference.build_source_free_fixture()
    weights = reference.load_frozen_class_weights()
    rows = []
    for entity in sidecar["entities"]:
        for task, state in entity["targets"].items():
            vocabulary = reference.get_vocabulary(reference.TASK_BY_ID[task].vocabulary_id).labels
            for shift in SHIFTS:
                value = transform_semantic_value(task, state["canonical_value"], shift_pc=shift,
                                                 dialect=sidecar["dialect"], profile="corrected_v2")
                _require(value in vocabulary and bool(weights.supported[task][vocabulary.index(value)]),
                         "fixture_shift_not_closed_or_supported:" + task)
                rows.append({"task": task, "shift": shift, "class_id": vocabulary.index(value)})
    _require(len(rows) == 18*12 and {row["task"] for row in rows} == set(reference.ACTIVE_HEADS), "fixture_head_coverage_missing")
    return {"task_shift_checks": len(rows), "rows": rows, "fingerprint": _json_fingerprint(rows),
            "dialect": sidecar["dialect"], "source": "existing_source_free_fixture_no_corpus"}


def _rng_fingerprint(device):
    from music_critic.experiments.analysisgnn import performance_replay as existing
    return _tensor_fingerprint(existing._capture_rng(device == "cuda"))


def _close_stream(stream, device):
    before = _rng_fingerprint(device)
    stream.close()
    if stream.pool is not None:
        _require(all(not slot.process.is_alive() and not slot.receiver.is_alive() for slot in stream.pool._slots),
                 "owned_workers_not_cleaned_up")
    _require(before == _rng_fingerprint(device), "cleanup_changed_parent_rng")


def _observed_step(stream, components, *, strategy, device, progress):
    from music_critic.experiments.analysisgnn import performance_replay as existing
    from music_critic.experiments.analysisgnn import performance_training as optimized
    reference = existing.reference
    model, optimizer, scheduler, scaler, resident, _ = components
    before_prepare = _rng_fingerprint(device)
    prepared = stream.peek()
    job = prepared.job
    source, sidecars = _unshifted_fixture(job)
    source_before = raw_fingerprint(source)
    prepared_before = raw_fingerprint(prepared.raw_graph_batch)
    sidecars_before = _tensor_fingerprint(prepared.sidecars)
    _require(_tensor_fingerprint(sidecars) == sidecars_before, "source_sidecars_changed")
    oracle = assert_historical_raw_view(source, prepared.raw_graph_batch, job.applied_shifts)
    raw = reference.move_raw_graph_batch(prepared.raw_graph_batch, device)
    raw_before = raw_fingerprint(raw)
    _require(before_prepare == _rng_fingerprint(device), "preparation_changed_parent_rng")
    observations, gradients, handles = {}, {}, []
    native = reference.align_target_sidecars_after_prediction
    _require(strategy in ("original", "optimized"), "invalid_update_strategy")
    target = reference if strategy == "original" else optimized
    update = reference.attempt_applied_update if strategy == "original" else optimized.attempt_performance_update
    weights = reference.load_frozen_class_weights() if strategy == "original" else resident
    _require(float(scaler.get_scale()) == 1.0 and not scaler.is_enabled(), "disabled_fp32_scaler_required")

    def capture_forward(_model, _args, output):
        observations["forward_calls"] = observations.get("forward_calls", 0) + 1
        _require(tuple(output.logits) == reference.ACTIVE_HEADS
                 and all(t.device.type == device for t in output.logits.values()), "forward_head_or_device_mismatch")
        observations["logits"] = _tensor_fingerprint(output.logits)
        observations["candidates"] = _tensor_fingerprint([
            {field.name: getattr(prediction, field.name) for field in fields(prediction) if field.name != "logits"}
            for prediction in output.predictions])

    def capture_alignment(*args, **kwargs):
        observations["native_alignment_calls"] = observations.get("native_alignment_calls", 0) + 1
        alignment = native(*args, **kwargs)
        observations["alignment"] = _tensor_fingerprint(alignment)
        observations["alignment_rows"] = {task: {
            "count": len(row.entity_ids), "valid": int(row.valid_mask.sum()),
            "sample_indices": row.sample_indices.tolist(), "record_ids": list(row.record_ids),
            "masked": row.masked_row_count, "failures": row.alignment_failure_count}
            for task, row in alignment.heads.items()}
        return alignment

    try:
        handles.append(model.register_forward_hook(capture_forward))
        for name, parameter in model.named_parameters():
            def capture_gradient(value, key=name):
                gradients[key] = value.detach().cpu().clone()
            handles.append(parameter.register_hook(capture_gradient))
        progress["attempted_updates"] += 1
        with patch.object(target, "align_target_sidecars_after_prediction", capture_alignment):
            try:
                outcome = update(model=model, raw_graph_batch=raw, sidecars=prepared.sidecars,
                    shifts=job.applied_shifts, class_weights=weights, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler)
            except BaseException:
                progress["applied_count_exact"] = False
                raise
        if outcome.applied:
            progress["completed_applied_updates"] += 1
        else:
            progress["skipped_updates"] += 1
        _require(outcome.applied, "skipped_update")
        stream.commit(job.position, applied=True)
        _require(observations.get("forward_calls") == observations.get("native_alignment_calls") == 1,
                 "single_native_forward_alignment_required")
        _require(all(row == {"count": 2, "valid": 2, "sample_indices": [0, 1],
                 "record_ids": list(job.record_ids), "masked": 0, "failures": 0}
                 for row in observations["alignment_rows"].values()), "all_18_head_alignment_coverage_required")
        _require(not outcome.loss.zero_valid_heads and len(outcome.loss.heads) == 18
                 and all(row.valid_row_count == 2 and row.masked_row_count == row.unsupported_row_count == 0
                         for row in outcome.loss.heads.values()), "all_18_head_loss_coverage_required")
        _require(gradients and outcome.finite_logits and outcome.finite_gradients, "finite_gradient_coverage_required")
        snapshot = existing._snapshot(model, optimizer, scheduler, scaler)
        result = {"job": json.loads(json.dumps(asdict(job))), **observations, **oracle,
            "raw": prepared_before, "source_raw": source_before, "sidecars": sidecars_before,
            "loss": _tensor_fingerprint(outcome.loss), "outcome": _tensor_fingerprint(outcome),
            "gradients_before_clip": _tensor_fingerprint(gradients),
            "gradients_after_clip": _tensor_fingerprint({name: parameter.grad for name, parameter in model.named_parameters()}),
            "training_state": _tensor_fingerprint(snapshot),
            **{name + "_state": _tensor_fingerprint(snapshot[name]) for name in ("model", "optimizer", "scheduler", "scaler", "rng")},
            "committed_batch_cursor": stream.state_dict()["committed_batch_cursor"]}
        _require(raw_fingerprint(source) == source_before and raw_fingerprint(prepared.raw_graph_batch) == prepared_before
                 and raw_fingerprint(raw) == raw_before and _tensor_fingerprint(prepared.sidecars) == sidecars_before,
                 "consumer_mutated_source_or_inputs")
        return result
    finally:
        for handle in handles:
            handle.remove()


def _validate_result(protocol, result):
    _require(result.get("full_protocol_exercised") is protocol["full_protocol_exercised"], "result_protocol_scope_mismatch")
    for field, expected in protocol.items():
        _require(field in result and _json_fingerprint(result[field]) == _json_fingerprint(expected),
                 "result_binding_mismatch:" + field)
    rows = result.get("arms")
    _require(isinstance(rows, list) and len(rows) == len(protocol["arm_manifest"]), "arm_coverage_mismatch")
    for row, expected in zip(rows, protocol["arm_manifest"], strict=True):
        _require(all(row.get(key) == value and type(row.get(key)) is type(value) for key, value in expected.items()),
                 "arm_identity_mismatch")
        _require(row.get("exact") is True and row["window"]["records"] == protocol["measured_draws"]
                 and row["applied_updates"] == protocol["measured_draws"] // 2, "arm_accounting_mismatch")
        if expected["resume"]:
            _require(row.get("checkpoint_sha256") and row.get("resume_prefix_rows") == protocol["resume_after_updates"],
                     "checkpoint_coverage_missing")
            _require(row.get("fresh_state_differed_before_restore") is True
                     and row.get("restored_training_state_fingerprint") == row.get("checkpoint_training_state_fingerprint")
                     and type(row.get("restored_training_state_fingerprint")) is str,
                     "fresh_runtime_restore_not_proven")
            if expected["worker_count"]:
                _require(type(row.get("pending_at_resume")) is int and row["pending_at_resume"] > 0,
                         "pending_resume_not_exercised")
    _require(result.get("equivalence_passed") is True and result.get("actual_applied_updates") == protocol["planned_updates"]
             and result.get("actual_attempted_updates") == protocol["planned_updates"]
             and result.get("actual_measured_draws") == protocol["planned_record_draws"]
             and result.get("skipped_updates") == 0, "total_accounting_or_equivalence_failed")


def _execute(output_root, *, experiment_id, protocol):
    """Only private tests may pass a reduced synthetic plan here."""
    import platform
    import torch
    from music_critic.experiments.analysisgnn import performance_replay as existing
    from music_critic.experiments.analysisgnn import performance_prefetch_replay as prefetch
    from music_critic.experiments.analysisgnn.performance_prefetch import PrefetchConfig
    output = Path(output_root)
    _require(not output.exists(), "fresh_output_required")
    jobs = _jobs(protocol)
    _require(protocol["arm_manifest_fingerprint"] == _json_fingerprint(protocol["arm_manifest"]), "arm_manifest_seal_mismatch")
    device = protocol["device"]
    output.mkdir(parents=True)
    existing._write(output / "protocol.json", protocol)
    progress = {"attempted_updates": 0, "completed_applied_updates": 0, "skipped_updates": 0, "applied_count_exact": True}
    arms, reference_trace = [], []
    try:
        idle_exemptions = None
        if device == "cuda":
            require_thread_environment(device)
            idle_exemptions = existing._assert_idle_gpu()
            _require(torch.cuda.is_available(), "cuda_unavailable")
        _require(sha256(existing.reference._B5B_FIXTURE.read_bytes()).hexdigest() == protocol["class_weight_fixture_sha256"],
                 "class_weight_source_changed")
        preflight = fixture_metadata_preflight()
        existing._write(output / "fixture-coverage.json", preflight)
        with existing._preserve_runtime(device == "cuda"):
            import torch_geometric
            environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                "pyg": str(torch_geometric.__version__), "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                "inherited_thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT},
                "other_environment": {name: os.environ.get(name) for name in (
                    "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED")},
                "parent_torch_threads": torch.get_num_threads(), "parent_torch_interop_threads": torch.get_num_interop_threads(),
                "worker_torch_threads": 1, "worker_torch_interop_threads": 1,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "idle_gpu_observed_exemptions": idle_exemptions}
            existing._write(output / "environment.json", environment)
            initial_components = prefetch._components(device)
            _require(_tensor_fingerprint(_model_plan(initial_components[0])) == protocol["model_plan_fingerprint"],
                     "model_plan_mismatch")
            actual_model_contract = existing.reference.corrected_model_contract(initial_components[0])
            scaler_api = initial_components[-1]
            initial = existing._snapshot(*initial_components[:4])
            del initial_components
            for arm in protocol["arm_manifest"]:
                directory = output / arm["arm_id"]
                directory.mkdir()
                components = prefetch._components(device)
                _require(_tensor_fingerprint(_model_plan(components[0])) == protocol["model_plan_fingerprint"], "model_plan_mismatch")
                _require(_tensor_fingerprint(components[4]) == protocol["class_weight_tensor_fingerprint"], "resident_weights_mismatch")
                _require(components[-1] == scaler_api, "scaler_api_changed")
                existing._restore(initial, *components[:4])
                config = PrefetchConfig(**protocol["worker_configs"][str(arm["worker_count"])])
                source = _json_fingerprint({"schedule": protocol["schedule_descriptor_fingerprint"],
                    "class_weights": protocol["class_weight_fixture_sha256"], "scope": "synthetic_historical_C2"})
                before_stream = _rng_fingerprint(device)
                stream = prefetch.BatchStream(jobs, prefetch.FixturePreparer(), source_fingerprint=source,
                                              config=config, synthetic=True)
                trace, resume_details = [], {}
                try:
                    _require(before_stream == _rng_fingerprint(device), "worker_startup_changed_parent_rng")
                    for index, job in enumerate(jobs):
                        if arm["resume"] and index == protocol["resume_after_updates"]:
                            pending = stream.diagnostics()["pending_batches"]
                            _require(not arm["worker_count"] or pending > 0, "pending_resume_not_exercised")
                            checkpoint = directory / "own-checkpoint.pt"
                            saved = {"schema": SCHEMA, "protocol_fingerprint": _json_fingerprint(protocol),
                                "training": existing._snapshot(*components[:4]), "cursor": stream.state_dict(), "prefix": trace}
                            torch.save(saved, checkpoint)
                            checkpoint_sha = sha256(checkpoint.read_bytes()).hexdigest()
                            del saved
                            _close_stream(stream, device)
                            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                            _require(sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
                                and state["schema"] == SCHEMA and state["protocol_fingerprint"] == _json_fingerprint(protocol)
                                and state["prefix"] == trace, "own_checkpoint_binding_mismatch")
                            saved_training = _tensor_fingerprint(state["training"])
                            # A restore into already-correct objects would not
                            # test Adam/LR/RNG restoration. Rebuild all runtime
                            # objects, then restore only after stream startup.
                            components = None
                            gc.collect()
                            components = prefetch._components(device)
                            _require(_tensor_fingerprint(_model_plan(components[0])) == protocol["model_plan_fingerprint"]
                                and _tensor_fingerprint(components[4]) == protocol["class_weight_tensor_fingerprint"]
                                and components[-1] == scaler_api, "fresh_runtime_contract_mismatch")
                            before_resume = _rng_fingerprint(device)
                            stream = prefetch.BatchStream(jobs, prefetch.FixturePreparer(), source_fingerprint=source,
                                config=config, synthetic=True, checkpoint=state["cursor"])
                            _require(before_resume == _rng_fingerprint(device), "resume_startup_changed_parent_rng")
                            fresh_training = _tensor_fingerprint(existing._snapshot(*components[:4]))
                            _require(fresh_training != saved_training, "fresh_runtime_not_different")
                            existing._restore(state["training"], *components[:4])
                            restored_training = _tensor_fingerprint(existing._snapshot(*components[:4]))
                            _require(restored_training == saved_training, "restored_full_training_state_mismatch")
                            _require(stream.state_dict() == state["cursor"], "restored_cursor_mismatch")
                            trace = deepcopy(state["prefix"])
                            del state
                            resume_details = {"checkpoint_sha256": checkpoint_sha, "pending_at_resume": pending,
                                "resume_prefix_rows": len(trace), "fresh_state_differed_before_restore": True,
                                "fresh_training_state_fingerprint": fresh_training,
                                "checkpoint_training_state_fingerprint": saved_training,
                                "restored_training_state_fingerprint": restored_training}
                        row = _observed_step(stream, components, strategy=arm["update_strategy"], device=device, progress=progress)
                        _require(row["job"] == json.loads(json.dumps(asdict(job))), "consumed_job_order_mismatch")
                        existing._append(directory / "trace.jsonl", row)
                        if arm["kind"] == "baseline":
                            reference_trace.append(row)
                        else:
                            for key, expected in reference_trace[index].items():
                                _require(row.get(key) == expected, f"exact_mismatch:{arm['arm_id']}:job{index}:{key}")
                        trace.append(row)
                    _require(len(trace) == len(jobs) and stream.state_dict()["committed_batch_cursor"] == len(jobs), "final_cursor_mismatch")
                    arms.append({**arm, **resume_details, "repeat": 0, "batch_size": 2, "exact": True,
                        "window": {"records": 2*len(jobs), "updates": len(jobs), "scope": "diagnostic_counts_no_throughput"},
                        "applied_updates": len(jobs), "trace_fingerprint": _json_fingerprint(trace),
                        "final_training_state_fingerprint": trace[-1]["training_state"],
                        "final_cursor": stream.state_dict()})
                finally:
                    _close_stream(stream, device)
                    del components
                    gc.collect()
            del initial
        _require(sha256(existing.reference._B5B_FIXTURE.read_bytes()).hexdigest() == protocol["class_weight_fixture_sha256"],
                 "class_weight_source_changed")
        result = {**protocol, "experiment_id": experiment_id, "completed": True, "equivalence_passed": True,
            "measured_draws_per_arm": protocol["measured_draws"], "warmup_draws_per_arm": 0,
            "arms": arms, "fixture_coverage": preflight, "actual_model_contract": actual_model_contract,
            "actual_attempted_updates": progress["attempted_updates"],
            "actual_applied_updates": progress["completed_applied_updates"], "skipped_updates": progress["skipped_updates"],
            "actual_measured_draws": 2*progress["completed_applied_updates"],
            "checkpoint_input_used": True, "own_fixture_checkpoint_input_used": True,
            "data_access": {key: None for key in ("validation_accessed", "test_inputs_accessed", "test_targets_accessed", "test_metrics_computed", "test_used_for_selection")},
            "data_access_observation_scope": "synthetic-only callable path; no OS-level read audit",
            "environment": environment, "scaler_api": scaler_api,
            "claim_limits": ["Synthetic DLC fixture only", "No real TRAIN C2 equivalence", "No AN/full-vocabulary coverage",
                "No throughput or RAM claim", "Historical stale track features retained", "No historical checkpoint bytes loaded"]}
        _validate_result(protocol, result)
        existing._write(output / "summary.json", result)
        return result
    except BaseException as exc:
        existing._write(output / "failure.json", {"completed": False, "equivalence_passed": False,
            "error_type": type(exc).__name__, "error": str(exc), "completed_arms": arms,
            "actual_attempted_updates": progress["attempted_updates"],
            "actual_applied_updates": progress["completed_applied_updates"] if progress["applied_count_exact"] else None,
            "known_completed_applied_updates": progress["completed_applied_updates"], "skipped_updates": progress["skipped_updates"]})
        raise


def run_c2_equivalence(output_root, *, experiment_id, device="cpu", expected_protocol=None):
    protocol = build_protocol(device)
    _require(expected_protocol is None or expected_protocol == protocol, "planned_protocol_changed")
    return _execute(output_root, experiment_id=experiment_id, protocol=protocol)


def run_registered_c2_gate(*, experiment_id, parent_experiment_id, output_root, device="cuda", repository_root=None):
    require_thread_environment(device)  # Before every heavy import and model operation.
    _require(bool(parent_experiment_id), "terminal_parent_required")
    from music_critic.experiments.analysisgnn.performance_evidence import run_registered_replay
    protocol = build_protocol(device)
    def execute():
        result = run_c2_equivalence(output_root, experiment_id=experiment_id, device=device, expected_protocol=protocol)
        _validate_result(protocol, result)
        _require(result["full_protocol_exercised"] is True, "registered_full_protocol_required")
        return result
    return run_registered_replay(experiment_id=experiment_id, parent_experiment_id=parent_experiment_id,
        output_root=Path(output_root), protocol=protocol, execute=execute, repository_root=repository_root)
