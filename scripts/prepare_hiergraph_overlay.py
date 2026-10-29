#!/usr/bin/env python3
"""Prepare accepted native overlay metadata and invoke a registered CPU export."""
import argparse
from pathlib import Path
import json
from music_critic.experiments.hierarchical_graph_analyzer.overlay import prepare_overlay_export
from music_critic.experiments.hiergraph_export import export_targets, write_json, file_binding
from music_critic.experiments.registry import validate_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=Path('.'))
    parser.add_argument('--experiment-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--source-fallback',type=Path)
    args=parser.parse_args(); root=args.repository.resolve()
    def record(identity):
        if not identity.replace('-','').isalnum(): raise ValueError('unsafe record identity')
        value=json.loads((root/'docs/experiments/records'/f'{identity}.json').read_bytes())
        validate_record(value);return value
    active=record(args.experiment_id)
    if active['status'] not in ('planned','running'):raise ValueError('registered nonterminal record required')
    kwargs=prepare_overlay_export(root,record('EXP-HIERGRAPH-INTERVAL-EXPORT-001'),
        record('EXP-HIERGRAPH-ANNOTATION-RELEASE-003'),args.metadata,source_fallback=args.source_fallback)
    sources=[root/'src/music_critic/experiments/hiergraph_export.py',
             root/'src/music_critic/experiments/hierarchical_graph_analyzer/overlay.py',Path(__file__).resolve()]
    write_json(args.metadata/'execution.json',dict(experiment_id=args.experiment_id,
        command=__import__('sys').argv, source_code=[file_binding(p,root) for p in sources],
        python=__import__('sys').version, platform=__import__('platform').platform()))
    result=export_targets(**kwargs,output=args.output)
    print(json.dumps(dict(status=result['status'],works=len(result['works']),fingerprint=result['fingerprint'])))

if __name__=='__main__':main()
