"""Fixed human-rated development pairs: whole-source inference and comparison.

The caller must register the experiment. No training, generation, automatic
selection, target truncation, or final TEST access occurs here. Human labels
are joined only when summarizing the completed score stream.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import time


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_bound(root, bound):
    root = Path(root).resolve()
    relative = Path(bound['path'])
    path = (root / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('human_eval.unsafe_path')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != bound['sha256'] or len(raw) != bound['size_bytes']:
        raise ValueError('human_eval.input_integrity')
    return raw


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')


def expanded_query(original, variant):
    """Enclose every changed onset, without the historical four-bar cutoff."""
    from music_critic.pll.contracts import MaskQuery

    if (original.bars != variant.bars or original.tracks != variant.tracks or
            original.tempo_events != variant.tempo_events):
        raise ValueError('human_eval.paired_geometry_changed')
    left = {n.note_id: n for n in original.notes}
    right = {n.note_id: n for n in variant.notes}

    def content(note):
        return None if note is None else (
            note.track_id, note.pitch, note.onset_qn, note.duration_qn,
            note.velocity, note.channel, note.program, note.is_percussion, note.is_grace)

    changed = [key for key in left.keys() | right.keys()
               if content(left.get(key)) != content(right.get(key))]
    notes = [n for key in changed for n in (left.get(key), right.get(key)) if n is not None]
    if not notes:
        raise ValueError('human_eval.no_change')
    start, end = min(n.onset_qn for n in notes), max(n.onset_qn for n in notes)
    bars = [b for b in original.bars
            if b.start_qn <= end and b.start_qn + b.duration_qn > start]
    if not bars:
        raise ValueError('human_eval.change_outside_bars')
    return MaskQuery(bars[0].start_qn, bars[-1].start_qn + bars[-1].duration_qn,
                     tuple(t.track_id for t in original.tracks))


def context_identity(context):
    return sha256(json.dumps(asdict(context), sort_keys=True,
                            separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def preflight_cases(cases):
    identifiers = [c['id'] for c in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('human_eval.duplicate_case')
    for case in cases:
        if (case['kind'] != 'pair' or case['split'] != 'campaign_head_dev' or
                case['source_scope'] not in ('train', 'external_development')):
            raise ValueError('human_eval.forbidden_case_scope')


def evaluate(args):
    import torch
    from music_critic.data import loads_piece
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.cloud import _model
    from music_critic.pll.codec import EOS
    from music_critic.pll.contracts import MaskQuery
    from music_critic.data import RationalTime as Q
    from music_critic.pll.observed_view import UnsupportedContextError
    from music_critic.pll.runtime import implementation_fingerprint

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output, root = Path(args.output), Path(args.cases).parent
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reads, rows = [], []
    try:
        plan = json.loads(Path(args.plan).read_text())
        if implementation_fingerprint() != plan['runtime_fingerprint']:
            raise ValueError('human_eval.runtime_binding')
        for path, key in ((args.cases, 'cases_sha256'), (args.config, 'config_sha256'),
                          (__file__, 'script_sha256')):
            if digest(path) != plan[key]:
                raise ValueError('human_eval.plan_binding:' + key)
        cases = json.loads(Path(args.cases).read_text())['cases']
        preflight_cases(cases)
        config = dict(load_run_config(args.config), pretrained=None, warm_start=None)
        model = _model(config, Path(args.config).parent).to(args.device).eval()
        checkpoint = plan['checkpoint']
        raw = Path(checkpoint['path']).read_bytes()
        if sha256(raw).hexdigest() != checkpoint['sha256'] or len(raw) != checkpoint['size_bytes']:
            raise ValueError('human_eval.checkpoint_integrity')
        payload = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
        bound_raw = Path(checkpoint['binding']).read_bytes()
        if sha256(bound_raw).hexdigest() != checkpoint['binding_sha256']:
            raise ValueError('human_eval.binding_integrity')
        if (payload['updates'] != checkpoint['update'] or
                json.dumps(payload['binding'], sort_keys=True) !=
                json.dumps(json.loads(bound_raw), sort_keys=True)):
            raise ValueError('human_eval.checkpoint_binding')
        model.load_state_dict(payload['model'], strict=True)
        del payload, raw
        with (output / 'scores.jsonl').open('x') as log:
            for case in cases:
                if time.monotonic() - started >= args.wall_seconds:
                    raise TimeoutError('human_eval.wall_budget')
                row = dict(id=case['id'], work_id=case['work_id'], aspect=case['aspect'],
                           common_query=case['common_query'], supported=False)
                pieces = []
                for key in ('original', 'variant'):
                    piece = loads_piece(read_bound(root, case[key]).decode())
                    read = dict(id=case['id'], role=key, split=piece.split, verified=False)
                    reads.append(read)
                    if (piece.split or '').lower() not in ('', 'train', 'validation', 'val', 'dev'):
                        raise ValueError('human_eval.forbidden_embedded_split')
                    read['verified'] = True
                    pieces.append(piece)
                q = case['query']
                query = MaskQuery(Q(*q['start']), Q(*q['end']), tuple(q['track_ids']))
                try:
                    examples = [model.split(piece, query) for piece in pieces]
                    identities = [context_identity(e.context) for e in examples]
                    if identities[0] != identities[1]:
                        raise ValueError('human_eval.paired_observed_context_mismatch')
                    for example in examples:
                        try:
                            model.codec.encode(example.targets, span=example.span, tracks=example.tracks)
                        except ValueError as exc:
                            if str(exc).startswith('unsupported'):
                                raise UnsupportedContextError(str(exc)) from exc
                            raise
                    with torch.no_grad():
                        # The two contexts have identical authenticated contents.
                        # Reuse one memory; no filling enters the predictor.
                        memory = model.encode(examples[0].context)
                        eos = float(model.decoder.next_token_log_probs(
                            (), memory, span=examples[0].span, tracks=examples[0].tracks)[EOS].exp())
                        for key, example in zip(('original', 'variant'), examples):
                            score = model.decoder(example.targets, memory,
                                                  span=example.span, tracks=example.tracks)
                            nll, events = float(score.nll), len(example.targets)
                            row[key] = dict(nll=nll, target_events=events,
                                            nats_per_event=nll / events if events else None,
                                            tokens=score.token_count)
                    row.update(supported=True, context_sha256=identities[0],
                               source_bars=[len(p.bars) for p in pieces],
                               observed_notes=len(examples[0].context.piece.notes),
                               identical_projected_targets=examples[0].targets == examples[1].targets,
                               eos_first_prob=eos,
                               null_weight=memory.diagnostics['motifs']['null_weight'],
                               delta_nll=row['variant']['nll'] - row['original']['nll'],
                               delta_nats_per_event=(row['variant']['nats_per_event'] -
                                   row['original']['nats_per_event'] if all(e.targets for e in examples) else None))
                except UnsupportedContextError as exc:
                    row['reason'] = str(exc)
                rows.append(row)
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                print(json.dumps(dict(done=len(rows), total=len(cases), id=case['id'],
                                      supported=row['supported'])), flush=True)
        summary = dict(status='completed', requested=len(cases), scored=len(rows),
                       supported=sum(r['supported'] for r in rows), reads=reads,
                       checkpoint=checkpoint, weight_updates=0,
                       elapsed_seconds=time.monotonic() - started,
                       peak_cuda_reserved_bytes=(torch.cuda.max_memory_reserved()
                                                  if args.device.startswith('cuda') else None),
                       runtime=dict(torch=str(torch.__version__), device=args.device,
                                    gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None))
        write_json(output / 'receipt.json', summary)
    except BaseException as exc:
        write_json(output / 'failure.json', dict(status='failed', error=repr(exc), reads=reads,
                   scores=len(rows), elapsed_seconds=time.monotonic() - started))
        raise


def join_comparison(ratings, model_rows):
    """Only exact shared IDs; empty targets and undefined deltas stay unavailable."""
    if len({r['id'] for r in ratings}) != len(ratings):
        raise ValueError('human_eval.duplicate_rating')
    indexes = {}
    for name, rows in model_rows.items():
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('human_eval.duplicate_model_score')
        indexes[name] = {r['id']: r for r in rows}
    output = []
    for rating in ratings:
        if rating['rating'] not in ('better', 'worse', 'same', 'unsure'):
            raise ValueError('human_eval.unknown_rating')
        row = dict(rating)
        row['models'] = {}
        for name, index in indexes.items():
            score = index.get(rating['id'])
            result = dict(supported=bool(score and score['supported']),
                          raw_agrees=None, per_event_agrees=None)
            if score and score['supported']:
                for key in ('delta_nll', 'delta_nats_per_event', 'original', 'variant',
                            'common_query', 'identical_projected_targets', 'null_weight'):
                    if key in score:
                        result[key] = score[key]
                if rating['rating'] in ('better', 'worse'):
                    sign = 1 if rating['rating'] == 'worse' else -1
                    for metric, field in (('delta_nll', 'raw_agrees'),
                                          ('delta_nats_per_event', 'per_event_agrees')):
                        delta = score.get(metric)
                        if delta is not None:
                            result[field] = sign * delta > 0
            else:
                result['reason'] = score.get('reason') if score else 'missing_score'
            row['models'][name] = result
        output.append(row)
    return output


def aggregate_comparison(rows, *, common_only=True):
    selected = [r for r in rows if not common_only or
                all(s['supported'] and s.get('common_query', True) for s in r['models'].values())]
    models = {}
    for name in (rows[0]['models'] if rows else ()):
        values = [r['models'][name] for r in selected]
        models[name] = dict(supported=sum(s['supported'] for s in values))
        for field in ('raw_agrees', 'per_event_agrees'):
            valid = [s[field] for s in values if s[field] is not None]
            models[name][field] = dict(correct=sum(valid), denominator=len(valid))
    return dict(pairs=len(selected), works=len({r['work_id'] for r in selected}), models=models)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--cases', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261001)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--wall-seconds', type=float, default=3600)
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
