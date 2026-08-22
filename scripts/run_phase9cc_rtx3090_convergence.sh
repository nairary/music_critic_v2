#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: $0 {run|resume|verify} EXPECTED_SHA CONFIG_JSON OUTPUT_ROOT" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
action=$1
expected_sha=$2
config_json=$3
output_root=$4
[[ "$action" == "run" || "$action" == "resume" || "$action" == "verify" ]] || usage

repo_root=$(git rev-parse --show-toplevel) || exit 1
cd "$repo_root" || exit 1
actual_sha=$(git rev-parse HEAD) || exit 1
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "phase9cc.rtx.head_mismatch expected=$expected_sha actual=$actual_sha" >&2
  exit 1
fi
if ! git merge-base --is-ancestor 786d0dd9320545f2eee50b6d59e609e72d96da49 HEAD; then
  echo "phase9cc.rtx.phase9cb_base_missing" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "phase9cc.rtx.clean_head_required" >&2
  exit 1
fi

if [[ "$action" == "verify" ]]; then
  .venv/bin/python scripts/verify_phase9cc_convergence.py \
    --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
  echo "phase9cc.rtx.verify.complete"
  echo "EVIDENCE_BUNDLE=$output_root"
  exit 0
fi

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -n 1)
if [[ "$gpu_name" != "NVIDIA GeForce RTX 3090" ]]; then
  echo "phase9cc.rtx.gpu_mismatch observed=$gpu_name" >&2
  exit 1
fi
if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.current_device() == 0; assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090"'; then
  echo "phase9cc.rtx.cuda0_unavailable" >&2
  exit 1
fi
if [[ "$action" == "run" && -e "$output_root" ]]; then
  echo "phase9cc.rtx.fresh_output_root_required" >&2
  exit 1
fi
if [[ "$action" == "resume" && ! -d "$output_root" ]]; then
  echo "phase9cc.rtx.resume_root_missing" >&2
  exit 1
fi

mkdir -p "$output_root"
log_path="${output_root%/}.phase9cc_${action}.log"
.venv/bin/python -m music_critic.experiments.phase9cc.run "$action" \
  --config "$config_json" --output-root "$output_root" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
if [[ $status -ne 0 ]]; then
  printf '%s\n' "$output_root" > "$output_root/FAILED_ROOT.txt"
  printf '%s\n' "$log_path" > "$output_root/FAILED_LOG.txt"
  echo "phase9cc.rtx.failed_root=$output_root" >&2
  echo "phase9cc.rtx.failed_log=$log_path" >&2
  exit "$status"
fi

.venv/bin/python scripts/verify_phase9cc_convergence.py \
  --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
echo "phase9cc.rtx.$action.complete"
echo "EVIDENCE_BUNDLE=$output_root"
echo "LOG=$log_path"
