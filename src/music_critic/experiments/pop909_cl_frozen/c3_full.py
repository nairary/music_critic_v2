"""Registered full POP909-CL transfer for the selected frozen cloud C3.

This separate entrypoint requires explicit POP TEST authorization. Historical
C0/C2 transfer and synthetic entrypoints retain their TEST-denying guards.
Root/quality arithmetic, raw beat alignment and views reuse the frozen protocol.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import platform
import re
from time import perf_counter

import torch

from music_critic.adapters.pop909_cl import (
    Pop909ClAdapterConfig, Pop909ClCorpusRecord, Pop909ClQuarantine,
    convert_pop909_cl_file, pop909_cl_piece_id, pop909_cl_raw_input_group_id,
    pop909_lineage_group_id,
)
from music_critic.adapters.pop909_cl_frozen_source import _read_bound_score
from music_critic.critic.c3_backbone import C3_SOURCE_ID, load_c3_relationgate_adapter
from music_critic.experiments.analysisgnn.corrected_model import model_state_fingerprint
from music_critic.experiments.analysisgnn.temporal_slices.graph import build_temporal_batch
from music_critic.experiments.registry import load_record
from music_critic.tasks import loads_corpus_index, loads_split_manifest, validate_split_manifest

from .contracts import VIEW_IDS, canonical_json, evaluation_protocol, fingerprint
from .evaluation import _summarize, align_event_start, targets_from_chord_evidence
from .projection import predict_quality_ids, predict_root_ids, quality_gold_class_id
from .transfer import BoundFile
from .views import build_view

SCHEMA = 'C3Pop909FullTransfer@1.0.0'
SPLITS = ('train', 'validation', 'test')
PINNED_INDEX_FINGERPRINT = 'b2008221fa59ddd0df31289561b22341db9c2eac527e1a503eac57b74da27daf'


def full_protocol() -> dict:
    original = evaluation_protocol()
    payload = dict(schema_version='C3Pop909FullProtocol@1.0.0',
        based_on_metric_protocol_fingerprint=original['fingerprint'],
        metric_policy={k: v for k, v in original.items()
                       if k not in ('schema_version', 'fingerprint', 'execution_policy')},
        execution_policy='explicit_registered_full_pinned_POP909_CL_CPU_test_authorized',
        checkpoint_selection='Dilemmadata_VALIDATION_best_80000_never_POP_selection',
        population='pinned_908_accepted_records_preserve_train_validation_test',
        candidate_policy='C3_raw_beat_candidates_source_start_half_open_containing_beat')
    return {**payload, 'fingerprint': fingerprint(payload)}


@dataclass(frozen=True)
class FullRequest:
    indices: tuple[BoundFile, ...]
    split_manifest: BoundFile
    checkpoint: BoundFile
    source_root: Path
    test_authorized: bool = False

    def __post_init__(self):
        if (not self.indices or any(not isinstance(x, BoundFile) for x in self.indices)
                or not isinstance(self.split_manifest, BoundFile)
                or not isinstance(self.checkpoint, BoundFile)
                or not isinstance(self.source_root, Path) or not self.source_root.is_absolute()
                or type(self.test_authorized) is not bool):
            raise ValueError('c3_pop_full.invalid_explicit_request')


def preflight(request: FullRequest) -> dict:
    """Validate all constituent indices and complete split closure, metadata only."""
    if request.test_authorized is not True:
        raise ValueError('c3_pop_full.explicit_test_authorization_required')
    indices = tuple(loads_corpus_index(x.read().decode('utf-8')) for x in request.indices)
    split = loads_split_manifest(request.split_manifest.read().decode('utf-8'))
    validate_split_manifest(split, indices)
    populations = [x for x in indices if x.header.dataset_id == 'pop909_cl']
    if len(populations) != 1:
        raise ValueError('c3_pop_full.one_pop_index_required')
    if (populations[0].header.index_fingerprint != PINNED_INDEX_FINGERPRINT
            or len(populations[0].records) != 908):
        raise ValueError('c3_pop_full.complete_pinned_908_index_required')
    assignments = {x.piece_id: x for x in split.assignments if x.dataset_id == 'pop909_cl'}
    selected = []
    for row in sorted(populations[0].records, key=lambda x: x.piece_id):
        assignment = assignments[row.piece_id]
        path = PurePosixPath(row.source_relative_path)
        projection = row.source_group_id.removeprefix('pop909-cl-score:')
        if (path.is_absolute() or '..' in path.parts or '\\' in row.source_relative_path
                or not re.fullmatch('[0-9]{3}', row.source_identity) or row.source_identity == '172'
                or row.piece_id != pop909_cl_piece_id(row.source_identity)
                or row.source_group_id != pop909_cl_raw_input_group_id(projection)
                or row.lineage_group_id != pop909_lineage_group_id(row.source_identity)
                or assignment.split not in SPLITS):
            raise ValueError('c3_pop_full.invalid_source_identity_or_split')
        selected.append(dict(piece_id=row.piece_id, song_id=row.source_identity,
            split=assignment.split, source_relative_path=row.source_relative_path,
            source_sha256=row.source_sha256, score_projection_sha256=projection,
            source_group_id=row.source_group_id, lineage_group_id=row.lineage_group_id,
            component_fingerprint=assignment.component_fingerprint))
    if not selected or set(assignments) != {r['piece_id'] for r in selected}:
        raise ValueError('c3_pop_full.complete_population_required')
    payload = dict(schema_version=SCHEMA, metric_protocol=full_protocol(),
        model='cloud_C3_threepass_relationgate', source_id=C3_SOURCE_ID, update=80000,
        checkpoint=request.checkpoint.identity(),
        indices=sorted((x.identity() for x in request.indices), key=lambda x: x['sha256']),
        split_artifact=request.split_manifest.identity(), split_fingerprint=split.manifest_fingerprint,
        records=selected, split_counts=dict(Counter(r['split'] for r in selected)),
        population='all_accepted_index_records_preserve_original_splits',
        test_scope='POP909-CL_only_user_requested_full_population', model_selection=False,
        candidate_policy='C3_raw_beat_candidates_source_start_half_open_containing_beat',
        targets='read_once_after_both_views', training_updates=0)
    return {**payload, 'binding_fingerprint': fingerprint(payload)}


def registered_record(path: Path, binding: dict) -> dict:
    record = load_record(path)
    if (record['status'] != 'planned' or record.get('protocol', {}).get(
            'c3_pop_full_binding_fingerprint') != binding['binding_fingerprint']):
        raise ValueError('c3_pop_full.matching_planned_record_required')
    policy = record['data_access'].get('declared_policy', {})
    if (any(policy.get(k) is not True for k in ('validation_allowed', 'test_inputs_allowed',
            'test_targets_allowed', 'test_metrics_allowed'))
            or policy.get('test_selection_allowed') is not False):
        raise ValueError('c3_pop_full.registered_access_policy_mismatch')
    return record


def read_score(record: Pop909ClCorpusRecord, row: dict, *, test_authorized: bool):
    if row['split'] == 'test' and test_authorized is not True:
        raise ValueError('c3_pop_full.explicit_test_authorization_required')
    if row['split'] not in SPLITS:
        raise ValueError('c3_pop_full.invalid_split')
    return _read_bound_score(record, split=row['split'],
        expected_projection_sha256=row['score_projection_sha256'])


def beat_predictions(output, beat_ids: tuple[str, ...]) -> tuple[dict, dict]:
    """Ignore slice outputs for parity with the historical raw-beat evaluation."""
    predictions, bindings = {}, {}
    for task, project in (('root', predict_root_ids), ('quality', predict_quality_ids)):
        keys = output.candidate_keys[task]
        indices = [i for i, (sample, level, _) in enumerate(keys) if sample == 0 and level == 'beat']
        ids = [keys[i][2] for i in indices]
        if len(ids) != len(set(ids)) or set(ids) != set(beat_ids):
            raise ValueError('c3_pop_full.beat_candidate_inventory')
        logits = output.logits[task][indices].detach().cpu()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('c3_pop_full.nonfinite_logits')
        predictions[task] = dict(zip(ids, project(logits).tolist(), strict=True))
        bindings[task] = dict(candidate_ids=ids,
                             logits_fingerprint=model_state_fingerprint({task: logits}))
    return predictions, bindings


def summarize_populations(rows: list[dict], pieces: list[dict]) -> dict:
    split_by_piece = {p['piece_id']: p['split'] for p in pieces}
    return dict(all=_summarize(rows, pieces), splits={split: _summarize(
        [r for r in rows if split_by_piece[r['piece_id']] == split],
        [p for p in pieces if p['split'] == split]) for split in SPLITS})


def evaluate(request: FullRequest, *, experiment_record: Path, output_root: Path) -> dict:
    binding = preflight(request)
    record = registered_record(experiment_record, binding)  # Before weights or source payloads.
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / 'preflight.json').write_text(canonical_json(binding), encoding='utf-8')
    started = perf_counter()
    source_root = request.source_root.resolve()
    access = dict(source_reads={s: 0 for s in SPLITS}, target_reads={s: 0 for s in SPLITS},
                  model_forwards=0, applied_updates=0)
    # Persist actual access before callbacks, including informative failed attempts.
    def save_access():
        (output_root / 'access.json').write_text(canonical_json(access), encoding='utf-8')
    save_access()
    adapter = load_c3_relationgate_adapter(request.checkpoint.path,
        expected_sha256=request.checkpoint.sha256, expected_size_bytes=request.checkpoint.size_bytes)
    model = adapter.model
    initial_state = model_state_fingerprint(model)
    per_view = {v: [] for v in VIEW_IDS}
    pieces = []
    with (output_root / 'observations.jsonl').open('x', encoding='utf-8') as events_file, \
            (output_root / 'progress.jsonl').open('x', encoding='utf-8') as progress_file:
        for number, row in enumerate(binding['records'], 1):
            piece_start = perf_counter()
            path = (source_root / row['source_relative_path']).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError('c3_pop_full.source_path_escapes_root')
            source_record = Pop909ClCorpusRecord(row['song_id'], path, row['source_relative_path'],
                path.name, row['source_sha256'], row['source_group_id'], row['lineage_group_id'])
            access['source_reads'][row['split']] += 1
            save_access()
            piece = read_score(source_record, row, test_authorized=request.test_authorized)
            predictions, inputs = {}, {}
            for view_id in VIEW_IDS:
                view = build_view(piece, view_id)
                graph = build_temporal_batch((view.piece,))
                with torch.inference_mode():
                    output = model(graph)
                access['model_forwards'] += 1
                predictions[view_id], logits_binding = beat_predictions(output,
                    tuple(b.beat_id for b in piece.beats))
                inputs[view_id] = dict(raw_graph_fingerprint=view.graph_fingerprint,
                    model_input_fingerprint=view.model_input_fingerprint,
                    temporal_graph_fingerprint=graph.fingerprint(), logits=logits_binding)
                del output, graph, view
            access['target_reads'][row['split']] += 1
            save_access()
            converted = convert_pop909_cl_file(source_record,
                config=Pop909ClAdapterConfig(include_targets=False))
            if (isinstance(converted, Pop909ClQuarantine)
                    or converted.score_projection_sha256 != row['score_projection_sha256']):
                raise ValueError('c3_pop_full.target_source_projection_mismatch')
            targets = targets_from_chord_evidence(converted.chord_evidence)
            target_events = sorted(targets.events, key=lambda e: (e.start_qn, e.event_id))
            piece_row = dict(piece_id=piece.piece_id, group_id=row['component_fingerprint'],
                split=row['split'], source_sha256=row['source_sha256'], inputs=inputs,
                source_status=targets.source_status, event_count=len(target_events),
                no_chord_span_count=targets.no_chord_span_count,
                trailing_unannotated_span_count=targets.trailing_unannotated_span_count,
                target_fingerprint=fingerprint([asdict(e) for e in target_events]))
            pieces.append(piece_row)
            for event in target_events:
                index, reason = align_event_start(event.start_qn, piece.beats)
                beat_id = None if index is None else piece.beats[index].beat_id
                for view_id in VIEW_IDS:
                    observation = dict(piece_id=piece.piece_id, group_id=row['component_fingerprint'],
                        event_id=event.event_id, start_qn=asdict(event.start_qn), beat_id=beat_id,
                        alignment_reason=reason)
                    for task in ('root', 'quality'):
                        gold = event.root_pc if task == 'root' else (
                            quality_gold_class_id(event.quality) if event.quality_available else None)
                        label_reason = getattr(event, f'{task}_reason')
                        if getattr(event, f'{task}_available') and gold is None:
                            label_reason = 'unsupported'
                        observation[task] = dict(gold=gold,
                            prediction=None if beat_id is None else predictions[view_id][task][beat_id],
                            reason=label_reason or reason, label_reason=label_reason,
                            source_label=event.root_pc if task == 'root' else event.quality)
                    per_view[view_id].append(observation)
                    events_file.write(canonical_json(dict(view=view_id, split=row['split'], **observation)))
            progress = dict(completed=number, total=len(binding['records']),
                piece_id=piece.piece_id, split=row['split'], events=len(target_events),
                piece_wall_seconds=perf_counter() - piece_start, elapsed_seconds=perf_counter() - started)
            progress_file.write(canonical_json(progress)); progress_file.flush(); events_file.flush()
            if number == 1 or number % 25 == 0 or number == len(binding['records']):
                print(canonical_json(progress), end='', flush=True)
    if model_state_fingerprint(model) != initial_state:
        raise ValueError('c3_pop_full.model_mutated_during_evaluation')
    report = dict(schema_version=SCHEMA, experiment_id=record['experiment_id'],
        binding_fingerprint=binding['binding_fingerprint'], checkpoint=binding['checkpoint'],
        source_id=C3_SOURCE_ID, checkpoint_update=80000, model_state_fingerprint=initial_state,
        protocol=binding['metric_protocol'], candidate_policy=binding['candidate_policy'],
        pieces=pieces, views={v: summarize_populations(per_view[v], pieces) for v in VIEW_IDS},
        access=access, wall_seconds=perf_counter() - started,
        environment=dict(python=platform.python_version(), torch=torch.__version__,
            platform=platform.platform(), device='cpu', threads=torch.get_num_threads()),
        limitations=['root_quality_only_not_all_18_heads_or_music_quality',
            'no_cross_corpus_duplicate_audit', 'full_population_includes_previously_seen_development_data',
            'beat_resolution_and_subset9_gold_follow_historical_protocol'])
    report['report_fingerprint'] = fingerprint(report)
    with (output_root / 'report.json').open('x', encoding='utf-8') as f:
        f.write(canonical_json(report))
    return report
