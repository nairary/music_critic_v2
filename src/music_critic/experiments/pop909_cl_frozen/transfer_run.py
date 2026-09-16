"""Preflight or run a registered POP909-CL frozen transfer evaluation on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contracts import canonical_json
from .transfer import evaluate_transfer, guard_transfer, load_request, preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read index/split metadata only")
    mode.add_argument("--evaluate", action="store_true", help="read bound weights and selected MIDI sources")
    parser.add_argument("--split", required=True, help="train or validation; TEST is denied before any read")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--experiment-record", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON file; never overwrite evidence")
    args = parser.parse_args(argv)
    guard_transfer(args.split)
    if args.evaluate and args.experiment_record is None:
        parser.error("--evaluate requires --experiment-record")
    if args.output.exists():
        parser.error("output already exists; use a new output path")
    request = load_request(args.request, split=args.split)
    # Reserve the destination before inference, including concurrent invocations.
    # Catchable interruptions/failures retain a receipt at this path.
    with args.output.open("x", encoding="utf-8") as output:
        try:
            report = (preflight(request) if args.preflight else
                      evaluate_transfer(request, experiment_record=args.experiment_record))
        except BaseException as exc:
            output.write(canonical_json({
                "schema_version": "Pop909ClTransferFailure@1.0.0",
                "split": args.split, "stage": "preflight" if args.preflight else "evaluation",
                "exception_type": type(exc).__name__, "error": str(exc),
                "access_observations": "incomplete_do_not_infer_zero_reads_from_failure",
                "next_action": "retain_receipt_finalize_failed_record_before_retry_with_new_id",
            }))
            raise
        output.write(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
