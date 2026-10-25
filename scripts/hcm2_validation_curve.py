"""HCM-2 validation curve over every retained checkpoint.

Evaluates a fixed, hash-pinned set of TRAIN/VALIDATION masks on each saved
checkpoint of the whole-song HCM-2 run and records NLL/event, motif
``null_weight`` and the immediate-EOS probability. Development evaluation only;
final TEST is never opened. Reuses the bounded input reader of
``scripts/hcm2_pilot_diagnostic`` so the masks stay comparable with the pilot.

Two measured cost levers are built in:

* the frozen local-encoder stage is computed once per case and reused for every
  checkpoint, guarded by a fail-closed encoder-weight fingerprint check;
* the pinned checkpoint list can be split across independent processes with
  ``--shard-index/--shard-count`` and merged deterministically with ``merge``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from scripts.hcm2_pilot_diagnostic import bindings_equal, query_from_json, read_bound


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def list_checkpoints(groups):
    """Ordered unique checkpoints from ``[(root, first, last, binding), ...]``.

    An update present in several groups is taken from the earliest group so the
    global-step curve has one point per update.
    """
    seen = {}
    for root, first, last, binding in groups:
        for path in sorted(Path(root).glob('checkpoint-*.pt')):
            update = int(path.stem.rsplit('-', 1)[-1])
            if first <= update <= last and update not in seen:
                seen[update] = dict(update=update, path=str(path), binding=str(binding))
    return [seen[update] for update in sorted(seen)]


def shard(checkpoints, index, count):
    """Deterministic disjoint partition; the union of all shards is the plan."""
    if count < 1 or not 0 <= index < count:
        raise ValueError('curve.invalid_shard')
    return [entry for position, entry in enumerate(checkpoints) if position % count == index]


def encoder_fingerprint(model_state):
    """Fingerprint of the frozen local-encoder weights inside one checkpoint."""
    keys = sorted(key for key in model_state if key.startswith('encoder.'))
    if not keys:
        raise ValueError('curve.encoder_scope_missing')
    digestor = sha256()
    for key in keys:
        digestor.update(key.encode())
        digestor.update(model_state[key].detach().cpu().numpy().tobytes())
    return f'{len(keys)}:{digestor.hexdigest()}'


def prepare(args):
    groups = [(row[0], int(row[1]), int(row[2]), row[3])
              for row in json.loads(Path(args.groups).read_text())]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    for entry in list_checkpoints(groups):
        raw = Path(entry['path']).read_bytes()
        checkpoints.append(dict(update=entry['update'], path=entry['path'],
                                binding=entry['binding'], size_bytes=len(raw),
                                sha256=sha256(raw).hexdigest()))
    plan = dict(status='prepared', seed=args.seed,
                cases_sha256=digest(Path(args.cases)),
                config_file_sha256=digest(Path(args.config)),
                script_sha256=digest(__file__),
                checkpoints=checkpoints)
    write_json(output/'plan.json', plan)
    print(json.dumps(dict(checkpoints=len(checkpoints),
                          updates=[c['update'] for c in checkpoints]), sort_keys=True))


def build_cache(model, cases, data):
    """Parse, split and run the frozen encoder stage once per case."""
    import torch
    from music_critic.data import loads_piece
    entries = []
    for case in cases['cases']:
        piece = loads_piece(read_bound(data, case['piece']).decode())
        assert piece.split == case['split']
        example = model.split(piece, query_from_json(case['query']))
        with torch.no_grad():
            tiles = model.encode_tiles(example.context)
        entries.append(dict(case=case, example=example, tiles=tiles))
    return entries


def aggregate(rows):
    """Event-weighted NLL/event plus mean null/EOS per update and split."""
    points = {}
    for row in rows:
        key = (row['update'], row['split'])
        point = points.setdefault(key, dict(update=row['update'], split=row['split'],
            masks=0, works=set(), sum_nll=0.0, events=0, null=[], eos=[]))
        point['masks'] += 1
        point['works'].add(row['id'].rsplit(':', 1)[0])
        if row['events'] > 0:
            point['sum_nll'] += row['nll']
            point['events'] += row['events']
        point['null'].append(row['null_weight'])
        point['eos'].append(row['eos_first_prob'])
    result = []
    for key in sorted(points):
        point = points[key]
        result.append(dict(update=point['update'], split=point['split'],
            masks=point['masks'], works=len(point['works']), events=point['events'],
            nats_per_event=(point['sum_nll']/point['events']) if point['events'] else None,
            mean_null_weight=statistics.mean(point['null']),
            mean_eos_first_prob=statistics.mean(point['eos'])))
    return result


def read_rows(paths):
    rows, seen = [], set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row['update'], row['id'])
            if key in seen:
                raise ValueError('curve.duplicate_score')
            seen.add(key)
            rows.append(row)
    return rows


def merge(args):
    rows = read_rows(args.inputs)
    points = aggregate(rows)
    summary = dict(status='completed', scores=len(rows),
                   checkpoints=sorted({row['update'] for row in rows}), points=points)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(scores=len(rows), checkpoints=len(summary['checkpoints']),
                          points=len(points)), sort_keys=True))


def evaluate(args):
    import torch
    from music_critic.pll.cloud import _model
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.codec import EOS

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.plan).read_text())
    assert plan['status'] == 'prepared'
    assert digest(Path(args.cases)) == plan['cases_sha256']
    assert digest(Path(__file__)) == plan['script_sha256']
    assert digest(Path(args.config)) == plan['config_file_sha256']
    cases = json.loads(Path(args.cases).read_text())
    assert cases['status'] == 'completed'
    assert all(case['split'] in ('train', 'validation') for case in cases['cases'])
    selected = shard(plan['checkpoints'], args.shard_index, args.shard_count)
    config = load_run_config(args.config)
    config = dict(config, warm_start=None, pretrained=None)
    model = _model(config, Path(args.config).parent)
    model.to('cuda:0').eval()
    started = time.monotonic()
    rows, cache, fingerprint = [], None, None
    try:
        with (output/'curve.jsonl').open('x') as log:
            for checkpoint in selected:
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError('curve.wall_limit')
                raw = Path(checkpoint['path']).read_bytes()
                assert sha256(raw).hexdigest() == checkpoint['sha256']
                assert len(raw) == checkpoint['size_bytes']
                payload = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
                assert payload['updates'] == checkpoint['update']
                assert bindings_equal(payload['binding'],
                                      json.loads(Path(checkpoint['binding']).read_text()))
                observed = encoder_fingerprint(payload['model'])
                if fingerprint is None:
                    fingerprint = observed
                elif observed != fingerprint:
                    # Reusing cached tiles would silently apply stale weights.
                    raise ValueError('curve.encoder_not_identical_across_checkpoints')
                model.load_state_dict(payload['model'], strict=True)
                if cache is None:
                    cache = build_cache(model, cases, data)
                del payload, raw
                for entry in cache:
                    example = entry['example']
                    with torch.no_grad():
                        memory = model.encode_from(entry['tiles'], example.context)
                        scored = model.decoder(example.targets, memory,
                                               span=example.span, tracks=example.tracks)
                        eos_first = float(model.decoder.next_token_log_probs(
                            (), memory, span=example.span, tracks=example.tracks)[EOS].exp())
                    row = dict(update=checkpoint['update'], id=entry['case']['id'],
                               split=entry['case']['split'], nll=float(scored.nll),
                               events=len(example.targets), tokens=scored.token_count,
                               null_weight=memory.diagnostics['motifs']['null_weight'],
                               eos_first_prob=eos_first)
                    rows.append(row)
                    log.write(json.dumps(row, sort_keys=True)+'\n')
                    log.flush()
                print(json.dumps(dict(update=checkpoint['update'], scores=len(rows))), flush=True)
        summary = dict(status='completed', seed=args.seed, elapsed_seconds=time.monotonic()-started,
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                       shard_index=args.shard_index, shard_count=args.shard_count,
                       checkpoints=[entry['update'] for entry in selected],
                       encoder_fingerprint=fingerprint, cached_encoder_tiles=True,
                       points=aggregate(rows))
    except BaseException as exc:
        write_json(output/'failure.json', dict(status='failed', error=repr(exc),
                                               elapsed_seconds=time.monotonic()-started))
        raise
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(points=len(summary['points'])), sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'evaluate', 'merge'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases')
    parser.add_argument('--config')
    parser.add_argument('--groups')
    parser.add_argument('--plan')
    parser.add_argument('--data')
    parser.add_argument('--inputs', nargs='*', default=[])
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--wall-seconds', type=float, default=10800.)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        merge(args)


if __name__ == '__main__':
    main()
