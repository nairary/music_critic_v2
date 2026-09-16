#!/usr/bin/env python3
"""Create the target runtime pin, registered reference request, or P512 request."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    from music_critic.experiments.analysisgnn import harmonic_fusion_dispatch as setup
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("runtime")
    runtime.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser("reference")
    for field in ("server-root", "repository-root", "handoff-root", "runtime", "output-root", "records-root"):
        reference.add_argument("--" + field, type=Path, required=True)
    reference.add_argument("--comparison-id", required=True)
    reference.add_argument("--experiment-id", required=True)
    pair = commands.add_parser("pair")
    for field in ("reference-request", "reference-record", "reference-report", "repository-root", "output-root", "records-root"):
        pair.add_argument("--" + field, type=Path, required=True)
    for field in ("attempt-id", "control-id", "fusion-id"):
        pair.add_argument("--" + field, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "runtime":
        result = setup.capture_runtime(**args)
    elif command == "reference":
        args["runtime_pin"] = setup.artifact_pin(args.pop("runtime"))
        result = setup.prepare_reference(**args)
    else:
        for field in ("reference_request", "reference_record", "reference_report"):
            args[field + "_pin"] = setup.artifact_pin(args.pop(field))
        result = setup.prepare_pair(**args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
