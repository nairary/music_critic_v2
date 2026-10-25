"""Run the explicitly registered, full-population C3 POP909-CL evaluation."""
from pathlib import Path
import argparse
import json

import torch

from music_critic.experiments.pop909_cl_frozen.c3_full import FullRequest, evaluate
from music_critic.experiments.pop909_cl_frozen.transfer import BoundFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--experiment-record', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    value = json.loads(args.request.read_text(encoding='utf-8'))
    if set(value) != {'indices', 'split_manifest', 'checkpoint', 'source_root', 'test_authorized'}:
        raise ValueError('exact full request fields required')
    def bound(row):
        return BoundFile(Path(row['path']), row['sha256'], row['size_bytes'])
    request = FullRequest(tuple(bound(r) for r in value['indices']), bound(value['split_manifest']),
        bound(value['checkpoint']), Path(value['source_root']), value['test_authorized'])
    evaluate(request, experiment_record=args.experiment_record, output_root=args.output_root)


if __name__ == '__main__':
    main()
