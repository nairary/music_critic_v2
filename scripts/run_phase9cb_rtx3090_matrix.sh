#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: $0 {profile|run|resume} EXPECTED_SHA CONFIG_JSON OUTPUT_ROOT" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
action=$1
expected_sha=$2
config_json=$3
output_root=$4
[[ "$action" == "profile" || "$action" == "run" || "$action" == "resume" ]] || usage

repo_root=$(git rev-parse --show-toplevel) || exit 1
cd "$repo_root" || exit 1
actual_sha=$(git rev-parse HEAD) || exit 1
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "phase9cb.rtx.head_mismatch expected=$expected_sha actual=$actual_sha" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "phase9cb.rtx.clean_head_required" >&2
  exit 1
fi
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -n 1)
if [[ "$gpu_name" != "NVIDIA GeForce RTX 3090" ]]; then
  echo "phase9cb.rtx.gpu_mismatch observed=$gpu_name" >&2
  exit 1
fi
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.current_device() == 0; assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"'; then
  echo "phase9cb.rtx.cuda0_unavailable" >&2
  exit 1
fi

mkdir -p "$output_root"
log_path="${output_root%/}.phase9cb_${action}.log"
.venv/bin/python -m music_critic.experiments.phase9cb.run "$action" \
  --config "$config_json" --output-root "$output_root" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
if [[ $status -ne 0 ]]; then
  echo "$output_root" > "$output_root/FAILED_ROOT.txt"
  echo "phase9cb.rtx.failed_root=$output_root" >&2
  exit "$status"
fi

if [[ "$action" == "run" || "$action" == "resume" ]]; then
  .venv/bin/python scripts/verify_phase9cb_rtx3090_matrix.py \
    --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
  archive="${output_root%/}.tar"
  .venv/bin/python -c 'from pathlib import Path; from music_critic.experiments.phase9cb import create_evidence_tar; import sys; print(create_evidence_tar(Path(sys.argv[1]), Path(sys.argv[2])))' "$output_root" "$archive" || exit 1
fi

echo "phase9cb.rtx.$action.complete"
