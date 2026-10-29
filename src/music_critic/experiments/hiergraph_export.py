"""Bounded, opt-in TRAIN target export. No training, inference or pointer expansion.

The caller owns experiment registration and finalization. Metadata is frozen before
payload access; every worker has its own process and exclusive output directory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
import gzip
import json
import os
import re
import resource
import subprocess
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def fingerprint(value):
    return sha256(canonical({k: v for k, v in value.items() if k != 'fingerprint'})).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(root, name):
    """Resolve only literal flat-admission relative paths, rejecting symlink hops."""
    root = Path(root).resolve()
    _require(isinstance(name, str) and bool(name) and '\\' not in name, 'unsafe relative path')
    path = PurePosixPath(name)
    _require(not path.is_absolute() and not PureWindowsPath(name).drive and
             all(p not in ('', '.', '..') for p in name.split('/')), 'unsafe relative path')
    result = root
    for part in path.parts:
        result = result / part
        _require(not result.is_symlink(), 'symlink in admitted path')
    _require(result.resolve().is_relative_to(root), 'path escaped source root')
    return result


def _binding_shape(binding):
    _require(isinstance(binding, dict), 'binding object required')
    _require(isinstance(binding.get('path'), str), 'binding path required')
    _require(type(binding.get('size_bytes')) is int and binding['size_bytes'] >= 0, 'binding size required')
    _require(isinstance(binding.get('sha256'), str) and
             re.fullmatch('[0-9a-f]{64}', binding['sha256']), 'binding SHA required')


def checked_read(root, binding, max_bytes=None):
    """Authenticate bytes before JSON decode; never follows embedded references."""
    _binding_shape(binding)
    path = safe_relative(root, binding['path'])
    _require(path.is_file(), 'admitted regular file required')
    _require(max_bytes is None or binding['size_bytes'] <= max_bytes, 'input byte cap')
    _require(path.stat().st_size == binding['size_bytes'], 'input size mismatch')
    data = path.read_bytes()
    _require(len(data) == binding['size_bytes'] and sha256(data).hexdigest() == binding['sha256'],
             'input SHA/size mismatch')
    return data


def _json_bytes(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)


def file_binding(path, root=None):
    path = Path(path)
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path.relative_to(root)) if root else path.name,
                sha256=h.hexdigest(), size_bytes=path.stat().st_size)


def write_json(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
    return file_binding(path)


def write_gzip_json(path, value, max_bytes):
    """Exclusive deterministic gzip (empty filename and zero mtime)."""
    path = Path(path)
    # JSONEncoder chunks avoid holding both a complete JSON string and its bytes.
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            count = 0
            for chunk in encoder.iterencode(value):
                data = chunk.encode('utf-8'); count += len(data)
                _require(count <= max_bytes, 'uncompressed output byte cap')
                zipped.write(data)
        _require(raw.tell() <= max_bytes, 'compressed output byte cap')
    return file_binding(path)


def read_gzip_json(root, binding, max_bytes):
    data = checked_read(root, binding, max_bytes)
    # Bounded decompression also protects against an authenticated but oversized cache.
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as stream:
        payload = stream.read(max_bytes + 1)
    _require(len(payload) <= max_bytes, 'decompressed cache byte cap')
    return _json_bytes(payload)


def build_family_split(works, prior_manifests=(), parent_bindings=()):
    """Preserve prior dev/selection; refine a prior fit into selection only."""
    prior, pieces = {}, {}
    for manifest in prior_manifests:
        for row in manifest['works']:
            family, piece = row['family'], row['piece_id']
            role = 'fit' if row['split'] == 'train' else row['split']
            _require(role in ('fit', 'selection', 'dev'), 'unsupported prior split')
            _require(family not in pieces or pieces[family] == piece, 'prior family identity conflict')
            pieces[family] = piece
            if family in prior:
                _require(prior[family] == role or {prior[family], role} == {'fit','selection'},
                         'prior split conflict')
                if prior[family] == 'selection':
                    role = 'selection'
            prior[family] = role
    rows = []
    for work in works:
        family = work['family']
        _require(family not in pieces or pieces[family] == work['piece_id'], 'family piece identity conflict')
        rows.append(dict(work_id=work['id'], piece_id=work['piece_id'], family_id=family,
                         split=prior.get(family, 'fit')))
    result = dict(schema='HiergraphFamilySplit@1', assignments=rows,
                  parent_manifests=list(parent_bindings),
                  policy='Preserve prior dev/selection by family; train renamed fit; new families fit')
    result['fingerprint'] = fingerprint(result)
    return result


DEFAULT_CAPS = dict(max_files_per_role=67, max_input_bytes=3182172659,
    worker_wall_seconds=300, total_wall_seconds=2400, worker_rss_bytes=6 * 1024**3,
    per_work_output_bytes=512 * 1024**2, total_output_bytes=12 * 1024**3,
    workers=1)


def _validated_metadata(admission, split, config, test_mode):
    from music_critic.tasks.hiergraph_projection import ONTOLOGY, VERSION, projection_contract
    works = admission.get('works', [])
    expected = config.get('max_works')
    _require(type(expected) is int and expected > 0 and (test_mode or expected == 67), 'production requires 67 works')
    _require(len(works) == expected, 'admitted work count mismatch')
    _require(split.get('schema') == 'HiergraphFamilySplit@1' and
             split.get('fingerprint') == fingerprint(split), 'split fingerprint mismatch')
    _require(config.get('schema') == 'HiergraphExportConfig@1', 'export config schema required')
    _require(config.get('projection_version') == VERSION, 'stale projection version in export config')
    _require(config.get('contract_fingerprint') == projection_contract()['fingerprint'],
             'stale projection contract in export config')
    _require(config.get('degree_frames') == {}, 'unreviewed degree conventions must remain masked')
    _require(set(config.get('exported_fields', [])) == set(ONTOLOGY), 'all ontology fields must be exported')
    _require(set(config.get('allow_coarsened_fields', [])) <= set(ONTOLOGY), 'unknown named coarsening field')
    caps = dict(DEFAULT_CAPS, **config.get('caps', {}))
    _require(all(type(v) is int and v > 0 for v in caps.values()) and caps['workers'] == 1,
             'positive integer caps and one worker required')
    assignments = split.get('assignments', [])
    _require(len(assignments) == expected, 'split work count mismatch')
    by_id, family_split, names, entries, paths = {}, {}, set(), [], set()
    piece_ids, canonical_shas = set(), set()
    for row in assignments:
        _require(row['work_id'] not in by_id and row['split'] in ('fit','selection','dev'), 'split duplicate/invalid role')
        family = row['family_id']
        _require(isinstance(family, str) and bool(family), 'family identity required')
        _require(family not in family_split or family_split[family] == row['split'], 'family leakage')
        family_split[family] = row['split']; by_id[row['work_id']] = row
    for work in works:
        name = work['id']
        _require(isinstance(name, str) and re.fullmatch('[A-Za-z0-9_-]+', name), 'unsafe work identity')
        _require(name not in names and name in by_id, 'duplicate or missing work assignment'); names.add(name)
        _require(work['piece_id'] not in piece_ids, 'duplicate canonical piece identity')
        piece_ids.add(work['piece_id'])
        canonical_sha = work['bindings']['canonical_piece']['sha256']
        _require(canonical_sha not in canonical_shas, 'duplicate canonical bytes across works')
        canonical_shas.add(canonical_sha)
        row = by_id[name]
        _require(row['piece_id'] == work['piece_id'] and row['family_id'] == work['family'], 'split identity mismatch')
        for role in ('canonical_piece','bundle'):
            b = work['bindings'][role]; _binding_shape(b)
            _require(b.get('role',role) == role and b.get('work_id',name) == name, 'binding role/identity mismatch')
            _require(b['path'] not in paths, 'duplicate admitted payload path'); paths.add(b['path'])
            entries.append(b)
    payload_entries = [e for e in admission.get('entries', [])
                       if e.get('role') in ('canonical_piece','bundle')]
    flat = {(e['work_id'], e['role']):e for e in payload_entries}
    _require(len(payload_entries) == expected * 2 and len(flat) == expected * 2 and
             all(flat[(w['id'],r)] == w['bindings'][r]
             for w in works for r in ('canonical_piece','bundle')), 'flat admission differs from work bindings')
    _require(expected <= caps['max_files_per_role'], 'role file cap')
    _require(sum(e['size_bytes'] for e in entries) <= caps['max_input_bytes'], 'total input byte cap')
    _require(any(r['split'] == 'fit' for r in assignments), 'at least one fit family required')
    return works, by_id, caps


def _load_piece(work, source_root, config, readlog, *, bundle_required):
    from music_critic.data import loads_piece
    binding = work['bindings']['canonical_piece']
    cb = checked_read(source_root, binding)
    readlog.append(dict(binding, verified=True))
    piece = loads_piece(cb)
    _require(piece.piece_id == work['piece_id'] and piece.split in ('train', None), 'non-TRAIN or wrong canonical identity')
    if piece.split is None:
        authority = config.get('null_split_authority', {})
        _require(authority.get('work_id') == work['id'] and authority.get('piece_id') == piece.piece_id and
                 authority.get('canonical_sha256') == binding['sha256'], 'null canonical split lacks explicit TRAIN authority')
    if not bundle_required:
        return piece, None
    bb = checked_read(source_root, work['bindings']['bundle'])
    readlog.append(dict(work['bindings']['bundle'], verified=True))
    bundle = _json_bytes(bb)
    _require(bundle.get('current_split') == 'train' and bundle.get('piece_id') == piece.piece_id and
             bundle.get('dataset_id') == piece.dataset_name, 'non-TRAIN or mismatched bundle')
    _require(bundle.get('canonical_piece', {}).get('sha256') == binding['sha256'], 'bundle canonical SHA mismatch')
    return piece, bundle


def _summary(projection, source_bundle=None):
    """Bounded support/diagnostics; no raw annotation payload is copied."""
    states, support, examples = Counter(), defaultdict(Counter), defaultdict(Counter)
    raw_examples, overflow = defaultdict(Counter), Counter()
    state_support = defaultdict(lambda: defaultdict(Counter))
    rows = {row['id']:row for scope in ('harmonies','note_contexts','objects')
            for row in (source_bundle or {}).get(scope, [])}
    for identity, mappings in projection['mappings'].items():
        for name, mapping in mappings.items():
            states[name + ':' + mapping['state']] += 1
            if mapping.get('available'):
                value = canonical(mapping['value']).decode()
                support[name][value] += 1
                state_support[name][mapping['state']][value] += 1
            elif mapping['state'] in ('unsupported','invalid','ambiguous'):
                # At most five distinct route diagnostics per field, no free raw prose dump.
                label = str(mapping.get('diagnostic') or mapping.get('rule') or mapping['state'])[:160]
                if label in examples[name] or len(examples[name]) < 5:
                    examples[name][label] += 1
                source = rows.get(identity, {}).get('fields', {})
                for path in mapping.get('source_paths', []):
                    field = source.get(path)
                    if field and field.get('available'):
                        raw = canonical(field.get('value')).decode()[:160]
                        key = path + ':' + raw
                        if key in raw_examples[name] or len(raw_examples[name]) < 5:
                            raw_examples[name][key] += 1
                        else:
                            overflow[name] += 1
    return dict(projected_states=dict(states), support={k:dict(v) for k,v in support.items()},
                support_by_state={field:{state:dict(counts) for state,counts in states.items()}
                                  for field,states in state_support.items()},
                unsupported_examples={k:dict(v) for k,v in examples.items()},
                unsupported_raw_examples={k:dict(v) for k,v in raw_examples.items()},
                unsupported_raw_example_overflow=dict(overflow))


def _worker(job):
    from music_critic.tasks.hiergraph_projection import (
        project_hiergraph_targets, encode_projection, VERSION, projection_contract,
    )
    work, config = job['work'], job['config']
    out = Path(job['directory']); out.mkdir(exist_ok=False)
    reads = []
    try:
        _require(config.get('projection_version') == VERSION and
                 config.get('contract_fingerprint') == projection_contract()['fingerprint'],
                 'worker projection config drift')
        piece, bundle = _load_piece(work, job['source_root'], config, reads, bundle_required=job['mode'] == 'project')
        if job['mode'] == 'project':
            source = {role:work['bindings'][role] for role in ('bundle','canonical_piece')}
            p = project_hiergraph_targets(piece, bundle, family_id=work['family'], degree_frames={}, source_binding=source)
            summary = _summary(p, bundle)
            del bundle
            output = write_gzip_json(out/'projection.json.gz', p, job['caps']['per_work_output_bytes'])
        else:
            p = read_gzip_json(job['cache_root'], job['projection'], job['caps']['per_work_output_bytes'])
            vocab = _json_bytes(checked_read(job['vocabulary_root'], job['vocabulary']))
            numeric = encode_projection(piece, p, vocab, split_fingerprint=job['split_fingerprint'])
            # Source is an authenticated reference envelope from the compact consumer.
            summary = dict(coverage=numeric['coverage'],
                           oov_counts=numeric['targets'].get('oov_counts', {}),
                           object_oov_counts=numeric['objects'].get('oov_counts', {}),
                           coarsening_excluded_counts=numeric['coarsening_excluded_counts'])
            output = write_gzip_json(out/'numeric.json.gz', numeric, job['caps']['per_work_output_bytes'])
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = usage.ru_maxrss
        peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
        _require(peak_bytes <= job['caps']['worker_rss_bytes'], 'worker peak RSS cap')
        receipt = dict(status='completed', work_id=work['id'], mode=job['mode'], output=output,
                       reads=reads, peak_rss_bytes=peak_bytes, cpu_seconds=usage.ru_utime+usage.ru_stime, summary=summary)
        write_json(out/'receipt.json', receipt)
    except BaseException as error:
        write_json(out/'failure.json', dict(status='failed', type=type(error).__name__, message=str(error), reads=reads))
        raise


def _tree_size(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def _run_worker(job, output, deadline):
    mode, name = job['mode'], job['work']['id']
    jobpath = output/'jobs'/f'{name}-{mode}.json'; write_json(jobpath, job)
    worker_output = Path(job['directory'])
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    # Keep execution tied to this implementation checkout, not a mutable caller cwd.
    source = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = source
    started = time.monotonic(); peak = 0
    with (output/'jobs'/f'{name}-{mode}.stderr').open('xb') as stderr:
        process = subprocess.Popen([sys.executable,'-m',__name__,'--worker',str(jobpath)],
                                   env=env, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            while process.poll() is None:
                _require(time.monotonic() < deadline and time.monotonic()-started < job['caps']['worker_wall_seconds'],
                         'worker/total wall cap')
                # ps reports resident KiB on both macOS and Linux; no RLIMIT_AS claim.
                result = subprocess.run(['ps','-o','rss=','-p',str(process.pid)], capture_output=True, text=True, check=False)
                if result.stdout.strip():
                    peak = max(peak,int(result.stdout.strip()) * 1024)
                    _require(peak <= job['caps']['worker_rss_bytes'], 'worker RSS cap')
                if worker_output.exists():
                    _require(_tree_size(worker_output) <= job['caps']['per_work_output_bytes'], 'per-work output cap')
                _require(_tree_size(output) <= job['caps']['total_output_bytes'], 'total output cap')
                time.sleep(.1)
            _require(process.returncode == 0, f'worker failed: {name}/{mode}; inspect retained stderr/failure')
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    receipt = _json_bytes((worker_output/'receipt.json').read_bytes())
    receipt['sampled_peak_rss_bytes'] = peak
    _require(_tree_size(worker_output) <= job['caps']['per_work_output_bytes'], 'per-work output cap')
    return receipt


def _heads(vocab, work_summaries, assignments, exported):
    from music_critic.tasks.hiergraph_projection import projection_field_manifest, VERSION
    fields = projection_field_manifest(); families = defaultdict(set); classes = defaultdict(set)
    for name, summary in work_summaries.items():
        if assignments[name]['split'] != 'fit':
            continue
        for field, states in summary['support_by_state'].items():
            fitted = {canonical(row['value']).decode() for row in vocab['fit_support'].get(field, [])
                      if row['active_id'] is not None and row['count'] > 0}
            values = set()
            for state, counts in states.items():
                if state == 'exact' or (state == 'coarsened' and field in vocab['allow_coarsened_fields']):
                    values.update(value for value,count in counts.items() if count > 0 and value in fitted)
            if values:
                families[field].add(assignments[name]['family_id']); classes[field].update(values)
    heads = {}
    for field in exported:
        reasons = []
        if field.startswith('P.'):
            reasons.append('P_evidence_only_initial_slice')
        if fields[field]['supervision_regime'] == 'positive_unlabeled':
            reasons.append('positive_unlabeled_negative_loss_protocol_not_admitted')
        if len(families[field]) < 3:
            reasons.append('fewer_than_three_fit_families')
        if len(classes[field]) < 2:
            reasons.append('fewer_than_two_fit_classes')
        heads[field] = dict(exported=True, active=not reasons, deferred_reasons=reasons,
                            fit_families=len(families[field]), fit_classes=len(classes[field]))
    result = dict(schema='HiergraphTrainableHeads@1', vocabulary_fingerprint=vocab['fingerprint'],
                  projection_version=VERSION, contract_fingerprint=vocab['contract_fingerprint'], fields=heads)
    result['fingerprint'] = fingerprint(result)
    return result


def export_targets(*, admission_path, admission_sha256, source_root, split_path, split_sha256,
                   config_path, config_sha256, output, test_mode=False):
    """Run a frozen CPU preparation; caller must register before invoking this API.

    Metadata paths are explicit. Split parent-manifest receipt paths are relative
    to split_path.parent and must be frozen copied metadata, never payload paths.
    Test mode exists only at the Python API for tiny synthetic fixtures.
    """
    def metadata(path, digest):
        path=Path(path)
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024*1024,
                 'regular bounded metadata file required')
        data=path.read_bytes()
        _require(sha256(data).hexdigest() == digest, 'metadata SHA mismatch')
        return _json_bytes(data)
    admission=metadata(admission_path,admission_sha256)
    split=metadata(split_path,split_sha256); config=metadata(config_path,config_sha256)
    _require(config.get('admission_sha256') == admission_sha256 and config.get('split_sha256') == split_sha256,
             'config input binding mismatch')
    parent_bindings = split.get('parent_manifests', [])
    _require(test_mode or len(parent_bindings) == 2, 'production requires two frozen prior split receipts')
    parents = [_json_bytes(checked_read(Path(split_path).parent,binding,1024*1024))
               for binding in parent_bindings]
    works, assignments, caps = _validated_metadata(admission,split,config,test_mode)
    if parents:
        expected = build_family_split(works,parents,parent_bindings)
        _require(expected['assignments'] == split['assignments'], 'frozen split changed prior assignments')
    out=Path(output)
    _require(not out.exists(), 'fresh output root required')
    # Production artifacts belong to an ignored outputs tree; no arbitrary publication target.
    _require(test_mode or 'outputs' in out.parts, 'production export must use outputs')
    out.mkdir(parents=True,exist_ok=False); (out/'jobs').mkdir(); (out/'project').mkdir(); (out/'encode').mkdir()
    deadline=time.monotonic()+caps['total_wall_seconds']; started=time.monotonic()
    receipts=[]; summaries={}; caches={}
    try:
        from music_critic.tasks.hiergraph_projection import build_projection_vocabulary, VERSION
        for work in works:
            job=dict(mode='project', work=work, config=config, caps=caps,
                     source_root=str(Path(source_root).resolve()), directory=str(out/'project'/work['id']))
            receipt=_run_worker(job,out,deadline); receipts.append(receipt)
            summaries[work['id']]=receipt['summary']
            caches[work['id']]=dict(receipt['output'],path=f"project/{work['id']}/"+receipt['output']['path'])
        def fit_projections():
            for work in works:
                if assignments[work['id']]['split']=='fit':
                    _require(time.monotonic()<deadline,'total wall cap')
                    yield read_gzip_json(out,caches[work['id']],caps['per_work_output_bytes'])
        vocab=build_projection_vocabulary(fit_projections(),
            fit_families=sorted({r['family_id'] for r in assignments.values() if r['split']=='fit'}),
            split_fingerprint=split['fingerprint'], active_fields=config['exported_fields'],
            allow_coarsened_fields=config['allow_coarsened_fields'])
        parent_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        parent_peak_bytes=int(parent_peak if sys.platform == 'darwin' else parent_peak*1024)
        _require(parent_peak_bytes <= caps['worker_rss_bytes'], 'vocabulary process RSS cap')
        _require(time.monotonic() < deadline, 'total wall cap')
        vb=write_json(out/'vocabulary.json',vocab)
        heads=_heads(vocab,summaries,assignments,config['exported_fields']); hb=write_json(out/'heads.json',heads)
        rows=[]
        for work in works:
            job=dict(mode='encode', work=work,config=config,caps=caps,source_root=str(Path(source_root).resolve()),
                directory=str(out/'encode'/work['id']), cache_root=str(out),projection=caches[work['id']],
                vocabulary_root=str(out),vocabulary=vb,split_fingerprint=split['fingerprint'])
            receipt=_run_worker(job,out,deadline);receipts.append(receipt)
            _require(_tree_size(out/'project'/work['id'])+_tree_size(out/'encode'/work['id']) <=
                     caps['per_work_output_bytes'], 'combined per-work output cap')
            rows.append(dict(assignments[work['id']],projection=caches[work['id']],
                numeric=dict(receipt['output'],path=f"encode/{work['id']}/"+receipt['output']['path']),
                source_bindings={r:work['bindings'][r] for r in ('canonical_piece','bundle')},
                coverage=receipt['summary']))
        reads={}
        for receipt in receipts:
            for read in receipt['reads']:
                reads[read['path']]=read
        _require(len(reads)==2*len(works), 'distinct payload read count mismatch')
        _require(_tree_size(out)<=caps['total_output_bytes'] and time.monotonic()<deadline,'total export cap')
        readbinding=write_json(out/'read-receipt.json',dict(schema='HiergraphReadReceipt@1',files=list(reads.values()),
            distinct_files=len(reads),distinct_bytes=sum(r['size_bytes'] for r in reads.values()),
            read_operations=sum(len(r['reads']) for r in receipts)))
        totals=Counter();family_coverage={}
        for work in works:
            summary=summaries[work['id']];totals.update(summary['projected_states'])
            family_coverage[work['family']]=dict(split=assignments[work['id']]['split'],**summary)
        aggregate_oov, aggregate_object_oov, aggregate_coarsening = Counter(),Counter(),Counter()
        for row in rows:
            aggregate_oov.update(row['coverage']['oov_counts'])
            aggregate_object_oov.update(row['coverage']['object_oov_counts'])
            aggregate_coarsening.update(row['coverage']['coarsening_excluded_counts'])
        manifest=dict(schema='HiergraphDataset@1',status='completed',works=rows,vocabulary=vb,heads=hb,
            projection_version=VERSION,contract_fingerprint=vocab['contract_fingerprint'],
            admission_sha256=admission_sha256,split_sha256=split_sha256,split_fingerprint=split['fingerprint'],
            config_sha256=config_sha256,read_receipt=readbinding,coverage_by_field=dict(totals),coverage_by_family=family_coverage,
            oov_counts=dict(aggregate_oov),object_oov_counts=dict(aggregate_object_oov),
            coarsening_excluded_counts=dict(aggregate_coarsening),vocabulary_process_peak_rss_bytes=parent_peak_bytes,
            wall_seconds=time.monotonic()-started,peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in receipts),
            total_worker_cpu_seconds=sum(r['cpu_seconds'] for r in receipts),
            human_gold=False,raw_pretraining_overlap='unknown',test_locked=True)
        manifest['fingerprint']=fingerprint(manifest);write_json(out/'manifest.json',manifest)
        return manifest
    except BaseException as error:
        write_json(out/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),
                   wall_seconds=time.monotonic()-started,completed_worker_receipts=len(receipts)))
        raise


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    for name in ('admission','split','config'):
        parser.add_argument('--'+name,type=Path)
        parser.add_argument('--'+name+'-sha256')
    parser.add_argument('--source-root',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.worker:
        _worker(_json_bytes(args.worker.read_bytes()));return
    for name in ('admission','split','config','admission_sha256','split_sha256','config_sha256','source_root','output'):
        if getattr(args,name) is None:parser.error(name+' required')
    result=export_targets(admission_path=args.admission,admission_sha256=args.admission_sha256,
        split_path=args.split,split_sha256=args.split_sha256,config_path=args.config,config_sha256=args.config_sha256,
        source_root=args.source_root,output=args.output)
    print(json.dumps(dict(status=result['status'],works=len(result['works']),fingerprint=result['fingerprint'])))


if __name__=='__main__':
    main()
