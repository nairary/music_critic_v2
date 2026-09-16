"""Hash-bound full TRAIN CPU graph sizing before a temporal GPU campaign."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import time

import torch
from music_critic.data import loads_piece
from music_critic.experiments.analysisgnn.harmonic_fusion_data import read_bound_bytes
from music_critic.experiments.analysisgnn.training_policy import CORRECTED_SPLIT_ASSIGNMENTS_SHA256
from music_critic.experiments.analysisgnn.temporal_slices import build_temporal_batch
from music_critic.experiments.registry import load_record


def train_rows(manifest):
    """Reject TEST declarations before opening even split metadata."""
    if any(r.get('split') not in ('train', 'validation') for r in manifest['records']):
        raise ValueError('TRAIN/VALIDATION declaration required; TEST forbidden')
    pin = manifest['split_assignments']
    if pin['sha256'] != CORRECTED_SPLIT_ASSIGNMENTS_SHA256:
        raise ValueError('frozen split required')
    assignments = {r['record_id']: r for r in map(json.loads, read_bound_bytes(pin).splitlines())}
    rows = sorted((r for r in manifest['records'] if r['split'] == 'train'), key=lambda r: r['record_id'])
    if len(rows) != 1295 or len({r['record_id'] for r in rows}) != 1295:
        raise ValueError('full1295TRAIN inventory required')
    for row in rows:
        a = assignments[row['record_id']]
        if a['split'] != 'train' or a['source_component_id'] != row['source_component_id']:
            raise ValueError('TRAIN identity mismatch')
    return rows


def dimensions(graph):
    base_bytes = sum(v.numel()*v.element_size() for store in graph.base.stores
                     for v in store.values() if isinstance(v, torch.Tensor))
    extra_bytes = graph.slice_features.numel()*graph.slice_features.element_size()
    extra_bytes += sum(v.numel()*v.element_size() for v in graph.edges.values())
    return dict(base_nodes={n: int(graph.base[n].num_nodes) for n in graph.base.node_types},
                slice_nodes=len(graph.keys['slice']), base_edges=sum(graph.base[e].edge_index.shape[1] for e in graph.base.edge_types),
                extension_edges=sum(v.shape[1] for v in graph.edges.values()),
                extension_edges_by_relation={'/'.join(e): v.shape[1] for e, v in graph.edges.items()},
                base_tensor_bytes=base_bytes, total_tensor_bytes=base_bytes+extra_bytes)


def run(args):
    record = load_record(args.record)
    if record['status'] != 'planned' or record['budget']['planned'] != 1295:
        raise ValueError('planned1295graph record required')
    if sha256(args.manifest.read_bytes()).hexdigest() != args.manifest_sha256:
        raise ValueError('manifest mismatch')
    for row in json.loads(args.source_manifest.read_text())['files']:
        if sha256((args.source_root/row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('source changed')
    rows = train_rows(json.loads(args.manifest.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.perf_counter()
    inventory = []
    with (args.output/'pieces.jsonl').open('x') as output:
        for index, row in enumerate(rows):
            # No sidecar payload, original TSV, VALIDATION or TEST read.
            piece = loads_piece(read_bound_bytes(row['piece']).decode())
            tick = time.perf_counter()
            graph = build_temporal_batch([piece])
            item = dict(record_id=row['record_id'], component_id=row['source_component_id'],
                        piece_sha256=row['piece']['sha256'], seconds=time.perf_counter()-tick, **dimensions(graph))
            inventory.append(item)
            output.write(json.dumps(item, sort_keys=True)+'\n'); output.flush()
            del graph, piece
            if (index+1)%100 == 0:
                print(json.dumps({'records':index+1,'elapsed_seconds':time.perf_counter()-started}), flush=True)
    result = {'schema':'TemporalFullTrainSizing@1.0.0','completed':True,'records':len(inventory),
              'elapsed_seconds':time.perf_counter()-started,'maxrss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'total_slice_nodes':sum(r['slice_nodes'] for r in inventory),
              'total_extension_edges':sum(r['extension_edges'] for r in inventory),
              'top_by_tensor_bytes':sorted(inventory,key=lambda r:r['total_tensor_bytes'],reverse=True)[:10],
              'top_by_slice_nodes':sorted(inventory,key=lambda r:r['slice_nodes'],reverse=True)[:10],
              'validation_payload_accessed':False,'test_payload_accessed':False,'sidecar_payload_accessed':False,
              'optimizer_updates':0,'model_forwards':0,'limitations':'CPU graph tensor sizes exclude activations, IPC copies, sidecars and optimizer; GPU admission separate'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'completed':True,'records':len(inventory),'seconds':result['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('record','manifest','source-manifest','source-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    run(parser.parse_args())
