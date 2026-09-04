#!/usr/bin/env python3
"""Validate, render, inspect, and store compact experiment evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from music_critic.experiments.registry import (
    ExperimentRegistryError,
    inspect_archive,
    load_registry,
    registry_fingerprint,
    render_markdown_ledger,
    store_archive,
)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage source-free experiment evidence records."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate all registry records")
    check.add_argument("--records", type=Path, required=True)

    render = commands.add_parser("render", help="render the deterministic ledger")
    render.add_argument("--records", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument(
        "--check",
        action="store_true",
        help="fail if output does not already equal the deterministic rendering",
    )

    inspect = commands.add_parser(
        "inspect-archive", help="validate an archive without extracting it"
    )
    inspect.add_argument("path", metavar="PATH", type=Path)

    store = commands.add_parser(
        "store-archive", help="copy a .tar.gz archive to immutable SHA-256 storage"
    )
    store.add_argument("path", metavar="PATH", type=Path)
    store.add_argument("--root", type=Path, required=True)
    return parser


def _json_print(value: object) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "check":
            records = load_registry(arguments.records)
            _json_print(
                {
                    "valid": True,
                    "record_count": len(records),
                    "registry_fingerprint": registry_fingerprint(records),
                }
            )
            return 0
        if arguments.command == "render":
            ledger = render_markdown_ledger(load_registry(arguments.records))
            if arguments.check:
                try:
                    observed = arguments.output.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    observed = None
                if observed != ledger:
                    expected_sha256 = sha256(ledger.encode("utf-8")).hexdigest()
                    observed_sha256 = (
                        None
                        if observed is None
                        else sha256(observed.encode("utf-8")).hexdigest()
                    )
                    _json_print(
                        {
                            "valid": False,
                            "reason": "ledger_out_of_date",
                            "expected_sha256": expected_sha256,
                            "observed_sha256": observed_sha256,
                        }
                    )
                    return 1
            else:
                _write_text_atomic(arguments.output, ledger)
            _json_print(
                {
                    "valid": True,
                    "output": arguments.output.as_posix(),
                    "sha256": sha256(ledger.encode("utf-8")).hexdigest(),
                }
            )
            return 0
        if arguments.command == "inspect-archive":
            _json_print(inspect_archive(arguments.path))
            return 0
        if arguments.command == "store-archive":
            _json_print(store_archive(arguments.path, arguments.root))
            return 0
    except (ExperimentRegistryError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
