"""Explicit manifest-driven PDMX preprocessing; importing never scans data."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_critic.adapters.pdmx import PdmxAdapterConfig
from music_critic.data.serialization import dumps_canonical_json
from music_critic.pdmx_pipeline import (
    PipelineConfig, SelectionConfig, WindowConfig, load_source_manifest, preprocess_pdmx,
)
from music_critic.pdmx_pipeline.manifest import strict_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = load_source_manifest(args.manifest)
    values = strict_json(args.config.read_text(encoding="utf-8"))
    selection = values.pop("selection")
    selection["allowed_licenses"] = tuple(selection["allowed_licenses"])
    config = PipelineConfig(
        selection=SelectionConfig(**selection),
        window=WindowConfig(**values.pop("window", {})),
        adapter=PdmxAdapterConfig(**values.pop("adapter", {})), **values,
    )
    splits = strict_json(args.splits.read_text(encoding="utf-8"))
    result = preprocess_pdmx(manifest, splits, source_root=args.source_root,
                             output_root=args.output, config=config)
    print(dumps_canonical_json({"publication": str(result.root / "publication.json"),
          "report": result.report, "cache_hit_records": result.cache_hit_records,
          "cache_hit_windows": result.cache_hit_windows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
