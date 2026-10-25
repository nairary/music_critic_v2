"""Evaluate the POP-selected C4 on full frozen Dilemmadata VALIDATION."""
import argparse
import json
from pathlib import Path
from music_critic.experiments.c4_pop909.transfer import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--experiment-record',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=evaluate(json.loads(a.request.read_text()),experiment_record=a.experiment_record,output=a.output)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
