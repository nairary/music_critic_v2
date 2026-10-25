#!/usr/bin/env python3
"""Prepare complete canonical TRAIN songs from an explicit four-source manifest.

The candidates document declares source/component splits and full/excerpt/unknown
coverage with evidence. Default admission is full songs only. This command does
not turn earlier clip caches into full-song sources, infer cross-corpus aliases,
download corpora or access held-out payloads. Every execution needs a pre-bound
planned data_preparation record; ``inspect`` opens candidate metadata only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time

from music_critic.experiments.registry import DATA_ACCESS_FIELDS, load_record
from music_critic.pll.cli import (_admit_output, _artifact, _now, _record_lock,
                                 _transition, _write_new)
from music_critic.pll.runtime import implementation_fingerprint
from music_critic.pll.whole_contracts import WholePieceConfig
from music_critic.pll.whole_data import WholeSourceCandidate, _preflight, prepare_whole_sources


def preparation_bindings(raw: bytes, source_root) -> dict:
    return dict(schema='PLLWholePreparationBinding@1', candidates_sha256=sha256(raw).hexdigest(),
        candidates_size_bytes=len(raw), source_root_sha256=sha256(str(Path(source_root).resolve()).encode()).hexdigest(),
        implementation_sha256=implementation_fingerprint(),
        cli_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def candidate_document(raw: bytes, source_root):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('pll.whole_duplicate_json_key')
            out[key] = value
        return out
    def nonfinite(_):
        raise ValueError('pll.whole_nonfinite_json')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique, parse_constant=nonfinite)
    fields = {'schema', 'records', 'source_weights', 'split_provenance', 'allowed_coverage', 'whole'}
    if (not isinstance(value, dict) or value.get('schema') != 'PLLWholeSourceCandidates@1'
            or set(value) - fields or not isinstance(value.get('records'), list)):
        raise ValueError('pll.whole_candidates_schema')
    candidates = tuple(WholeSourceCandidate(**row) for row in value['records'])
    _preflight(candidates, Path(source_root).resolve(), value['source_weights'], value['split_provenance'])
    allowed = value.get('allowed_coverage', ['full'])
    if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed)
            or not set(allowed) <= {'full', 'excerpt', 'unknown'}):
        raise ValueError('pll.whole_coverage_policy')
    WholePieceConfig(**value.get('whole', {}))
    return value, candidates


def prepare_registered(candidates_path, record_path, output, *, source_root, ledger=None):
    candidates_path, record_path, output = Path(candidates_path), Path(record_path), Path(output)
    _admit_output(output)
    raw = candidates_path.read_bytes()
    document, candidates = candidate_document(raw, source_root)
    bindings = preparation_bindings(raw, source_root)
    ledger = Path(ledger) if ledger else (
        record_path.parents[2] / 'EXPERIMENT_LEDGER.md'
        if record_path.parent.name == 'records' and record_path.parent.parent.name == 'experiments'
        else (record_path.parent.parent if record_path.parent.name == 'records' else record_path.parent)
        / 'EXPERIMENT_LEDGER.md')
    if ledger.name != 'EXPERIMENT_LEDGER.md':
        raise ValueError('pll.ledger_filename_required')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with _record_lock(record_path):
        record = load_record(record_path)
        if (record['status'] != 'planned' or record['kind'] != 'data_preparation'
                or record.get('pll_whole_prepare') != bindings):
            raise ValueError('pll.whole_planned_record_binding_required')
        if any(record['data_access']['declared_policy'][key] for key in (
                'validation_allowed', 'test_inputs_allowed', 'test_targets_allowed',
                'test_metrics_allowed', 'test_selection_allowed')):
            raise ValueError('pll.whole_held_out_must_be_closed')
        output.mkdir(parents=True, exist_ok=False)
        _write_new(output / 'planned-record.json', record)
        # Exactly these retained bytes were parsed and bound above.
        with (output / 'candidates.json').open('xb') as stream:
            stream.write(raw)
        _write_new(output / 'preparation-binding.json', bindings)
        started, clock = _now(), time.monotonic()
        record = _transition(record_path, record, dict(record, status='running',
            execution=dict(started_at=started, device='cpu', python=platform.python_version(),
                           platform=platform.platform())), ledger)
        report, failure, status = None, None, 'failed'
        try:
            report = prepare_whole_sources(candidates, output / 'data', source_root=source_root,
                source_weights=document['source_weights'], split_provenance=document['split_provenance'],
                allowed_coverage=tuple(document.get('allowed_coverage', ['full'])),
                whole_config=WholePieceConfig(**document.get('whole', {})),
                input_bindings=[dict(path=str(candidates_path.resolve()), sha256=sha256(raw).hexdigest(),
                                     size_bytes=len(raw))])
            status = 'completed'
        except KeyboardInterrupt as exc:
            status, failure = 'aborted', exc
        except Exception as exc:
            failure = exc
        finally:
            receipt_path = output / 'data/prepare-receipt.json'
            if report is None and receipt_path.exists():
                report = json.loads(receipt_path.read_text())
            unknown = report is None or bool(report.get('unverified_reads'))
            observations = {key: None if unknown else False for key in DATA_ACCESS_FIELDS}
            # A later unknown/mismatched access cannot erase an already parsed
            # forbidden split. Preserve known incident evidence independently.
            accesses = [] if report is None else report.get('accesses', [])
            if any(a.get('observed_split') == 'validation' for a in accesses):
                observations['validation_accessed'] = True
            if any(a.get('observed_split') == 'test' for a in accesses):
                observations['test_inputs_accessed'] = True
            if any(a.get('observed_split') == 'test' and a.get('observed_target_arrays', 0) > 0
                   for a in accesses):
                observations['test_targets_accessed'] = True
            violations = [key for key, value in observations.items() if value is True]
            if violations:
                status = 'invalid'
            access_status = ('partial_primary_evidence' if unknown and violations else
                             'pending_primary_evidence' if unknown else 'verified_primary_evidence')
            summary = dict(schema='PLLWholePreparationAttempt@1', status=status, started_at=started,
                ended_at=_now(), wall_seconds=time.monotonic() - clock, bindings=bindings,
                preparation_receipt='data/prepare-receipt.json' if receipt_path.exists() else None,
                error=None if failure is None else dict(type=type(failure).__name__, message=str(failure)),
                claim='Preparation mechanics only; full-song coverage is caller-declared, GPU admission unmeasured')
            _write_new(output / 'receipt.json', summary)
            paths = [p for p in output.iterdir() if p.is_file()]
            paths += [p for p in (receipt_path, output / 'data/manifest.json') if p.exists()]
            final = dict(record, status=status,
                execution={**record['execution'], 'ended_at': summary['ended_at']},
                budget={**record['budget'], 'actual_wall_seconds': summary['wall_seconds']},
                results={} if report is None else {k: report[k] for k in
                    ('verified_train_reads', 'records', 'source_counts', 'coverage_counts') if k in report},
                validity={**record['validity'], 'reason': None if failure is None else
                          str(failure) or type(failure).__name__},
                data_access={**record['data_access'],
                    'access_evidence_status': access_status, **observations,
                    'policy_violations': violations,
                    'evidence_scope': ('Forbidden input split observed; see preparation receipt' if violations else
                        'Unverified source access; see preparation receipt' if unknown
                        else 'Only hash-bound identity-matched TRAIN canonical payloads were loaded')},
                artifacts=[*record['artifacts'], *[_artifact(p, record_path, str(p.relative_to(output)))
                            for p in sorted(paths)]])
            _transition(record_path, record, final, ledger)
        if failure is not None:
            raise failure
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect = commands.add_parser('inspect', help='inspect candidate metadata without music payload access')
    prepare = commands.add_parser('prepare', help='execute one registered whole-source preparation')
    for command in (inspect, prepare):
        command.add_argument('--candidates', type=Path, required=True)
        command.add_argument('--source-root', type=Path, required=True)
    prepare.add_argument('--experiment-record', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--ledger', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            raw = args.candidates.read_bytes()
            document, candidates = candidate_document(raw, args.source_root)
            result = dict(bindings=preparation_bindings(raw, args.source_root), candidates=len(candidates),
                          allowed_coverage=document.get('allowed_coverage', ['full']), data_accessed=False)
        else:
            result = prepare_registered(args.candidates, args.experiment_record, args.output,
                                        source_root=args.source_root, ledger=args.ledger)
        print(json.dumps(result, sort_keys=True, indent=2))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
