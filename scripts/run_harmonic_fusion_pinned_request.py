#!/usr/bin/env python3
"""Coordinator resource-wrapper child: kind, sealed request path, fresh run root."""
from pathlib import Path
import runpy
import sys


def main(argv=None):
    from music_critic.experiments.analysisgnn.harmonic_fusion_pilot import artifact_pin
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3 or args[0] not in ("reference", "pilot"):
        raise ValueError("usage: run-pinned-request.py reference|pilot REQUEST OUTPUT_ROOT")
    kind, request, output = args
    request, output = Path(request), Path(output)
    if not request.is_absolute() or not output.is_absolute():
        raise ValueError("absolute explicit request/output paths required")
    name = "run_harmonic_fusion_reference_gate.py" if kind == "reference" else "run_harmonic_fusion_real_pilot.py"
    entry = Path.cwd() / "scripts" / name
    pin = artifact_pin(request)
    sys.argv = [str(entry), "--request", pin["path"], "--request-sha256", pin["sha256"],
        "--request-size-bytes", str(pin["size_bytes"]), "--execute", "--output-root", str(output)]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
