"""Bounded HCM-2 development diagnostic; never train or open final TEST."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def bindings_equal(left, right):
    # The checkpoint retains tuples; JSON evidence serializes them as lists.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def read_bound(root, binding):
    root = Path(root).resolve()
    relative = Path(binding['path'])
    path = (root/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
        raise ValueError('diagnostic unsafe input path')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != binding['sha256'] or len(raw) != binding['size_bytes']:
        raise ValueError('diagnostic input integrity')
    return raw


def query_from_json(raw):
    from music_critic.data import RationalTime as Q
    from music_critic.pll.contracts import MaskQuery
    return MaskQuery(Q(*raw['start']), Q(*raw['end']), tuple(raw['track_ids']))


def bounded_generate(model, context, *, seed, max_tokens, wall_seconds):
    import torch
    from music_critic.pll.codec import EOS
    started = time.monotonic()
    prefix = ()
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    memory = model.encode(context)
    for _ in range(max_tokens):
        if time.monotonic()-started >= wall_seconds:
            reason = 'wall_budget'
            break
        probs = model.decoder.next_token_log_probs(prefix, memory, span=context.query.span,
                                                   tracks=context.allowed_tracks)
        prefix += (int(torch.multinomial(probs.exp(), 1, generator=generator).item()),)
        if prefix[-1] == EOS:
            events = model.codec.decode(prefix, span=context.query.span, tracks=context.allowed_tracks)
            return dict(status='completed', events=[asdict(e) for e in events], tokens=list(prefix),
                        elapsed_seconds=time.monotonic()-started)
    else:
        reason = 'token_budget'
    return dict(status='incomplete', reason=reason, tokens=list(prefix),
                elapsed_seconds=time.monotonic()-started)


def ablate_motifs(memory, *, null_value=None):
    """Drop ALL motif paths, or substitute a content-independent null memory.

    The null control preserves token count and tests retrieval content separately
    from the effect of removing an entire trained attention branch.
    """
    import torch
    from music_critic.pll.whole_decoder import MultiBranchMemory
    n = len(memory.branches['motif'])
    assert n > 0 and all(t.level.startswith('motif_') for t in memory.metadata[-n:])
    branch = dict(memory.branches)
    if null_value is None:
        branch.pop('motif')
        values, metadata = memory.values[:-n], memory.metadata[:-n]
        families = {k: v[:-n] for k, v in memory.families.items()}
    else:
        replacement = torch.zeros_like(branch['motif'])
        replacement[0] = null_value
        branch['motif'] = replacement
        values, metadata = torch.cat((memory.values[:-n], replacement)), memory.metadata
        families = {k: torch.cat((v[:-n], replacement)) for k, v in memory.families.items()}
    return MultiBranchMemory(values, families, metadata, memory.query,
                             branches=branch, diagnostics=memory.diagnostics)


def variants(events, span):
    """Target-only synthetic edits; identical observed context for every candidate."""
    from music_critic.data import RationalTime as Q
    def ordered(rows):
        return tuple(sorted(rows, key=lambda e: e.key))
    pitch = ordered(replace(e, pitch=e.pitch + (1 if e.pitch < 127 else -1))
                    if i % 2 == 0 and e.kind < 2 else e for i, e in enumerate(events))
    rhythm = ordered(replace(e, onset=e.onset + Q(1, 4))
                     if i % 2 == 0 and e.onset + Q(1, 4) < span else e
                     for i, e in enumerate(events))
    return {'original': events, 'pitch': pitch, 'rhythm': rhythm,
            'delete_half': events[1::2], 'silence': ()}


def log_summary(rows):
    out = []
    for lo, hi in ((1, 100), (101, 300), (301, 600), (601, 1000)):
        selected = [r['metrics'] for r in rows if lo <= r['metrics']['update'] <= hi]
        if not selected:
            continue
        weights = [c['motifs']['null_weight'] for r in selected for c in r['context_diagnostics']]
        count = sum(r['target_events'] for r in selected)
        out.append(dict(first=lo, last=hi, examples=len(selected),
            null_mean=statistics.mean(weights), null_median=statistics.median(weights),
            fraction_null_over_099=sum(w > .99 for w in weights)/len(weights),
            nll_per_event=sum(r['block_nll'] for r in selected)/count if count else None,
            target_events=count))
    return out


def summarize(report):
    """Paired descriptive quantities; masks from one work are not independent."""
    result = dict(status=report['status'], elapsed_seconds=report['elapsed_seconds'],
                  likelihood=[], ablations=[], edits=[], generations=[])
    for split in ('train', 'validation'):
        for update in (100, 1000):
            rows = [r for r in report['scores'] if r['split']==split and r['update']==update]
            if not rows:
                continue
            events = sum(r['scores']['full']['events'] for r in rows)
            total = sum(r['scores']['full']['nll'] for r in rows)
            nulls = [r['diagnostics']['motifs']['null_weight'] for r in rows]
            result['likelihood'].append(dict(split=split, update=update, masks=len(rows),
                works=len({r['id'].rsplit(':', 1)[0] for r in rows}), events=events,
                nats_per_event=total/events if events else None, null_mean=statistics.mean(nulls),
                fraction_null_over_099=sum(n>.99 for n in nulls)/len(nulls)))
            for arm in ('no_motif', 'null_only'):
                deltas = [r['scores'][arm]['nll']-r['scores']['full']['nll'] for r in rows]
                result['ablations'].append(dict(split=split, update=update, arm=arm,
                    delta_nats_per_event=sum(deltas)/events if events else None,
                    max_abs_block_delta=max(abs(d) for d in deltas)))
        final = [r for r in report['scores'] if r['split']==split and r['update']==1000]
        for edit in ('pitch', 'rhythm', 'delete_half', 'silence'):
            pairs = [(r['variants']['original'], r['variants'][edit]) for r in final
                     if r['variants'][edit]['changed']]
            if pairs:
                result['edits'].append(dict(split=split, edit=edit, pairs=len(pairs),
                    lower_nll_for_original=sum(a['nll'] < b['nll']-1e-4 for a,b in pairs),
                    lower_nll_for_edit=sum(b['nll'] < a['nll']-1e-4 for a,b in pairs),
                    mean_delta_nll=statistics.mean(b['nll']-a['nll'] for a,b in pairs),
                    mean_original_events=statistics.mean(a['events'] for a,b in pairs),
                    mean_edited_events=statistics.mean(b['events'] for a,b in pairs)))
    for g in report['generations']:
        result['generations'].append(dict(id=g['id'], status=g['status'], reason=g.get('reason'),
            events=len(g['events']) if 'events' in g else None, target_events=len(g['target_events']),
            tokens=len(g['tokens']), elapsed_seconds=g['elapsed_seconds']))
    return result


def prepare(root, output, seed):
    from scripts.hcm1_diagnostic_data import select_validation, read_validation, save_piece, query_json
    from music_critic.pll.whole_data import WholeDataManifest
    from music_critic.pll.cli import load_run_config
    from music_critic.pll.whole_contracts import WholePieceConfig, split_whole_piece
    from music_critic.pll.contracts import ContextProfile, CodecProfile, MaskQuery
    from music_critic.pll.codec import EventCodec
    output.mkdir(parents=True, exist_ok=False)
    cache = root/'data/cache'
    base = root/'outputs/hcm_whole_launch_20260928'
    cfg = load_run_config(base/'config/train.json')
    profile, whole = ContextProfile(**cfg['context']), WholePieceConfig(**cfg['whole_piece'])
    codec = EventCodec(CodecProfile(**cfg['codec']))
    split_data = json.loads((cache/'global.split.json').read_text())
    assignments = split_data['assignments']
    from music_critic.tasks.corpus import load_corpus_index
    from music_critic.tasks.loading import load_split_manifest
    authenticated_split = load_split_manifest(cache/'global.split.json')
    for source in ('hooktheory', 'pop909_cl'):
        index = load_corpus_index(cache/f'{source}.index.json', require_current=False)
        assert dict(authenticated_split.index_fingerprints)[source] == index.header.index_fingerprint
    pop_header = json.loads((cache/'pop909_cl.index.json').read_text())['header']
    assert pop_header['creation_policy'] == 'offline_full_corpus'
    indexes = {s: json.loads((cache/f'{s}.index.json').read_text())['records']
               for s in ('hooktheory', 'pop909_cl')}
    # Both indexes constrain shared components, even though only POP is used.
    selected_val = [r for r in select_validation(assignments, indexes, per_source=6, seed=seed)
                    if r['source'] == 'pop909_cl']
    manifest = WholeDataManifest.load(base/'train-manifest.json')
    train_groups = {r.work_id for r in manifest.records}
    assert not train_groups.intersection(r['component_id'] for r in selected_val)
    selected_train = sorted(manifest.records, key=lambda r: sha256(f'{seed}:{r.work_id}'.encode()).hexdigest())[:6]
    report = dict(status='running', accesses=[], cases=[], excluded=[])
    write_json(output/'selection.json', dict(validation=selected_val, train=[asdict(r) for r in selected_train]))
    try:
        pieces = []
        for row in selected_train:
            p = manifest.load_piece(row.record_id)
            report['accesses'].append(dict(piece_id=p.piece_id, declared_split='train', verified=True))
            pieces.append((p, row.work_id, 'train'))
        for row in selected_val:
            p = read_validation(cache/'pop909_cl/multisource-cache-v1.0.0', row, report['accesses'])
            pieces.append((p, row['component_id'], 'validation'))
        for p, group, split in pieces:
            bound = save_piece(output, p)
            for length in (1, 2):
                i = (len(p.bars)-length)//2
                first, last = p.bars[i], p.bars[i+length-1]
                q = MaskQuery(first.start_qn, last.start_qn+last.duration_qn,
                    tuple(t.track_id for t in p.tracks if any(n.track_id==t.track_id for n in p.notes)))
                row = dict(id=f'{p.piece_id}:{length}', piece=bound, split=split, work_id=group,
                           query=query_json(q), source_bars=len(p.bars), mask_bars=length)
                try:
                    ex = split_whole_piece(p, q, whole_config=whole, context_profile=profile)
                    for events in variants(ex.targets, ex.span).values():
                        codec.encode(events, span=ex.span, tracks=ex.tracks)
                    row['target_events'] = len(ex.targets)
                    report['cases'].append(row)
                except ValueError as exc:
                    report['excluded'].append(dict(row, reason=str(exc)))
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['train_manifest_reads'] = manifest.verified_train_reads
        report['train_manifest_unverified_reads'] = manifest.unverified_reads
        write_json(output/'cases.json', report)


def evaluate(args):
    import torch
    from io import BytesIO
    from music_critic.data import loads_piece
    from music_critic.pll.cloud import _model
    from music_critic.pll.cli import load_run_config
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    root, data, output = Path(args.run_root), Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(args.plan).read_text())
    assert digest(data/'cases.json') == plan['cases_sha256']
    assert digest(__file__) == plan['script_sha256']
    assert digest(root/'config/train.json') == plan['config_file_sha256']
    cases = json.loads((data/'cases.json').read_text())
    assert cases['status'] == 'completed'
    assert all(c['split'] in ('train', 'validation') for c in cases['cases'])
    cfg = load_run_config(root/'config/train.json')
    started = time.monotonic()
    report = dict(status='running', scores=[], generations=[], reads=[], seed=args.seed)
    def score(model, events, memory, ex):
        result = model.decoder(events, memory, span=ex.span, tracks=ex.tracks)
        nll = float(result.nll)
        return dict(nll=nll, events=len(events), nats_per_event=nll/len(events) if events else None,
                    tokens=result.token_count)
    try:
        for update in (100, 1000):
            raw = (root/f'outputs/train001/checkpoint-{update:08d}.pt').read_bytes()
            assert sha256(raw).hexdigest() == plan['checkpoints'][str(update)]
            payload = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
            assert payload['updates'] == update
            binding = json.loads((root/'outputs/train001/training-binding.json').read_text())
            assert bindings_equal(payload['binding'], binding)
            model = _model(cfg, root/'config')
            model.load_state_dict(payload['model'], strict=True)
            model.to('cuda:0').eval()
            del payload, raw
            with torch.no_grad(), (output/f'scores-{update}.jsonl').open('x') as log:
                for case in cases['cases']:
                    if time.monotonic()-started > 1500:
                        raise TimeoutError('diagnostic wall limit')
                    attempt = dict(id=case['id'], declared_split=case['split'], verified=False)
                    report['reads'].append(attempt)
                    p = loads_piece(read_bound(data, case['piece']).decode())
                    attempt['observed_split'] = p.split
                    assert p.split == case['split']
                    attempt['verified'] = True
                    ex = model.split(p, query_from_json(case['query']))
                    memory = model.encode(ex.context)
                    row = dict(id=case['id'], split=case['split'], update=update,
                               diagnostics=memory.diagnostics, scores={})
                    for name, mem in (('full', memory), ('no_motif', ablate_motifs(memory)),
                                      ('null_only', ablate_motifs(memory, null_value=model.motifs.null_value))):
                        row['scores'][name] = score(model, ex.targets, mem, ex)
                    if update == 1000:
                        row['variants'] = {name: dict(score(model, events, memory, ex), changed=events!=ex.targets)
                                           for name, events in variants(ex.targets, ex.span).items()}
                    report['scores'].append(row)
                    log.write(json.dumps(row, sort_keys=True)+'\n'); log.flush()
                    print(json.dumps(dict(update=update, id=case['id'], nll=row['scores']['full']['nll'])), flush=True)
                if update == 1000:
                    chosen = [c for c in cases['cases'] if c['split']=='validation' and c['mask_bars']==1][:2]
                    for i, case in enumerate(chosen):
                        attempt = dict(id=case['id'], purpose='generation', declared_split=case['split'], verified=False)
                        report['reads'].append(attempt)
                        p = loads_piece(read_bound(data, case['piece']).decode())
                        attempt['observed_split'] = p.split
                        assert p.split == case['split']
                        attempt['verified'] = True
                        ex = model.split(p, query_from_json(case['query']))
                        g = bounded_generate(model, ex.context, seed=args.seed+i, max_tokens=1024, wall_seconds=90)
                        g.update(id=case['id'], target_events=[asdict(e) for e in ex.targets])
                        report['generations'].append(g)
                        write_json(output/f'generation-{i}.json', g)
                        print(json.dumps(dict(generation=case['id'], status=g['status'])), flush=True)
            del model
            torch.cuda.empty_cache()
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write_json(output/'summary.json', report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare','evaluate'))
    p.add_argument('--root', default='.')
    p.add_argument('--output', required=True)
    p.add_argument('--data'); p.add_argument('--run-root'); p.add_argument('--plan')
    p.add_argument('--seed', type=int, default=20260929)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(Path(args.root), Path(args.output), args.seed)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
