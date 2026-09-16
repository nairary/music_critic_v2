#!/usr/bin/env python3
"""Prepare pinned fusion input metadata without reading warm payloads or running a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    data = subcommands.add_parser("data-manifest", help="Convert the coordinator cache index and validate all frozen metadata")
    for prefix in ("cache-index", "metadata-pins"):
        data.add_argument("--" + prefix, type=Path, required=True)
        data.add_argument("--" + prefix + "-sha256", required=True)
        data.add_argument("--" + prefix + "-size-bytes", type=int, required=True)
    data.add_argument("--output", type=Path, required=True, help="New manifest file; an existing file is never replaced")
    args = parser.parse_args(argv)
    from music_critic.experiments.analysisgnn.harmonic_fusion_prepare import prepare_data_manifest
    def pin(prefix):
        return {"path": str(getattr(args, prefix).absolute()),
                "sha256": getattr(args, prefix + "_sha256"), "size_bytes": getattr(args, prefix + "_size_bytes")}
    try:
        report = prepare_data_manifest(cache_index=pin("cache_index"), metadata_pins=pin("metadata_pins"),
                                       output=args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
