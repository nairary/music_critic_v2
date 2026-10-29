#!/usr/bin/env python3
"""Copy only authenticated numeric targets and canonical inputs for raw caching.

The target manifest is preserved byte-for-byte. Projection and source bundle
references remain provenance references and are explicitly not packaged.
"""
from pathlib import Path
import argparse
import json
from music_critic.experiments.hiergraph_export import (
    checked_read, safe_relative, file_binding, write_json, _require,
)
from music_critic.experiments.hierarchical_graph_analyzer.data import load_target_dataset
from music_critic.experiments.registry import validate_record


def package_inputs(*, export_root, manifest_sha256, source_root, output):
    """Caller must register this real-data copy before invoking it."""
    export_root, source_root, output = map(lambda p:Path(p).resolve(),(export_root,source_root,output))
    ds=load_target_dataset(export_root,manifest_sha256=manifest_sha256)
    _require(not output.exists(),'fresh immutable package root required')
    manifest_binding=file_binding(export_root/'manifest.json')
    _require(manifest_binding['sha256']==manifest_sha256,'manifest changed during preparation')
    output.mkdir(parents=True,exist_ok=False)
    target_root, raw_root=output/'targets',output/'source'
    target_root.mkdir();raw_root.mkdir();copied=[];seen={}
    def copy(root,binding,destination_root,kind):
        key=(str(destination_root),binding['path'])
        if key in seen:
            _require(all(seen[key][k]==binding[k] for k in ('sha256','size_bytes')),'conflicting repeated source path')
            return
        data=checked_read(root,binding,1024**3)
        destination=safe_relative(destination_root,binding['path'])
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write(data)
        receipt=dict(path=str(destination.relative_to(output)),sha256=binding['sha256'],
                     size_bytes=binding['size_bytes'],kind=kind)
        copied.append(receipt);seen[key]=dict(binding)
    try:
        for binding in (manifest_binding,ds.manifest['vocabulary'],ds.manifest['heads']):
            copy(export_root,binding,target_root,'target_metadata')
        for work in ds.manifest['works']:
            copy(export_root,work['numeric'],target_root,'numeric_targets')
            copy(source_root,work['source_bindings']['canonical_piece'],raw_root,'canonical_piece')
        receipt=dict(schema='HiergraphPortableInputs@1',status='completed',
            manifest_sha256=manifest_sha256,vocabulary_fingerprint=ds.vocabulary['fingerprint'],
            split_fingerprint=ds.manifest['split_fingerprint'],works=len(ds.works),
            targets_root='targets',source_root='source',files=copied,
            total_bytes=sum(b['size_bytes'] for b in copied),
            excluded=['projection payloads','source bundle payloads'],
            notes=['Original target manifest remains exact; excluded bindings are retained as provenance references',
                   'No raw cache or C3 tensors computed; no model or optimizer executed'])
        write_json(output/'package.json',receipt)
        return receipt
    except BaseException as error:
        write_json(output/'failure.json',dict(status='failed',type=type(error).__name__,message=str(error),copied=copied))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('export-root','source-root','output','repository'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--experiment-id',required=True)
    args=parser.parse_args()
    _require(args.experiment_id.replace('-','').isalnum(),'safe experiment identity required')
    record=json.loads((args.repository/'docs/experiments/records'/f'{args.experiment_id}.json').read_bytes())
    validate_record(record)
    _require(record['status'] in ('planned','running'),'planned/running copy experiment required')
    _require(args.output.resolve().is_relative_to(args.repository.resolve()/'outputs'),'package belongs in ignored outputs')
    result=package_inputs(export_root=args.export_root,manifest_sha256=args.manifest_sha256,
                          source_root=args.source_root,output=args.output)
    print(json.dumps(dict(status=result['status'],works=result['works'],total_bytes=result['total_bytes'])))

if __name__=='__main__':main()
