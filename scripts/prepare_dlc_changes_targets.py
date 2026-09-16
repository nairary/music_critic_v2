"""Export source-native changes targets from an explicitly TRAIN-bound manifest."""
import argparse
import json
from pathlib import Path

from music_critic.tasks.dlc_changes import export_train_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_train_targets(args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
