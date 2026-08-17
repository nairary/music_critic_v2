"""Independent source-free verifier for Phase 9B.2C RTX evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_critic.experiments.dilemmadata.supervised_smoke import (
    DilemmadataSupervisedSmokeError,
    verify_evidence_bundle,
    verify_evidence_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence-dir", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--expected-head", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.evidence_dir is not None:
            if arguments.sidecar is not None:
                parser.error("--sidecar is valid only with --bundle")
            manifest = verify_evidence_directory(
                arguments.evidence_dir,
                expected_head=arguments.expected_head,
                require_current_hardware=True,
            )
        else:
            if arguments.sidecar is None:
                parser.error("--bundle requires --sidecar")
            manifest = verify_evidence_bundle(
                arguments.bundle,
                arguments.sidecar,
                expected_head=arguments.expected_head,
                require_current_hardware=True,
            )
    except DilemmadataSupervisedSmokeError as exc:
        parser.exit(2, f"{exc}\n")
    print(manifest["fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
