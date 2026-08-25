#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: $0 {run|resume|verify} EXPECTED_SHA CONFIG_JSON OUTPUT_ROOT" >&2
  echo "       $0 continue EXPECTED_SHA PARENT_OUTPUT_ROOT CONFIG_JSON NEW_ROOT --start-update 9000 --target-update 15000 --validation-milestones 9000,12000,15000" >&2
  echo "       $0 continue EXPECTED_SHA BIGRU_PARENT_ROOT CONFIG_JSON NEW_ROOT --cells scratch_onset_bigru,ssl_onset_bigru --start-update 3000 --target-update 15000 --validation-milestones 3000,6000,9000,12000,15000 --mlp-reference-root MLP_ROOT" >&2
  exit 2
}

if [[ ${1:-} == "continue" ]]; then
  [[ $# -eq 11 || $# -eq 15 ]] || usage
  action=$1
  expected_sha=$2
  parent_output_root=$3
  config_json=$4
  output_root=$5
  phase9cd=false
  if [[ $# -eq 15 ]]; then
    phase9cd=true
    [[ $6 == "--cells" && $7 == "scratch_onset_bigru,ssl_onset_bigru" ]] || usage
    [[ $8 == "--start-update" && $9 == "3000" ]] || usage
    [[ ${10} == "--target-update" && ${11} == "15000" ]] || usage
    [[ ${12} == "--validation-milestones" && ${13} == "3000,6000,9000,12000,15000" ]] || usage
    [[ ${14} == "--mlp-reference-root" && -n ${15} ]] || usage
    mlp_reference_root=${15}
  else
    [[ $6 == "--start-update" && $7 == "9000" ]] || usage
    [[ $8 == "--target-update" && $9 == "15000" ]] || usage
    [[ ${10} == "--validation-milestones" && ${11} == "9000,12000,15000" ]] || usage
  fi
else
  [[ $# -eq 4 ]] || usage
fi

action=$1
expected_sha=$2
if [[ "$action" != "continue" ]]; then
  config_json=$3
  output_root=$4
fi
[[ "$action" == "run" || "$action" == "resume" || "$action" == "verify" || "$action" == "continue" ]] || usage

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

if [[ "$action" == "continue" ]]; then
  if [[ ! -d "$parent_output_root" ]]; then
    echo "phase9cc.continuation.parent_root_missing=$parent_output_root" >&2
    exit 1
  fi
  if [[ ! -f "$config_json" ]]; then
    echo "phase9cc.continuation.config_missing=$config_json" >&2
    exit 1
  fi
  if [[ -f "$output_root/manifest.json" ]]; then
    if [[ "$phase9cd" == true ]]; then
      .venv/bin/python scripts/verify_phase9cd_continuation.py \
        --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
    else
      .venv/bin/python scripts/verify_phase9cc_continuation.py \
        --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
    fi
    echo "phase9cc.rtx.continue.complete"
    echo "EVIDENCE_BUNDLE=$output_root"
    echo "LOG=$output_root/execution.log"
    exit 0
  fi
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
if [[ "$action" == "continue" ]]; then
  if [[ "$phase9cd" == true ]]; then
    .venv/bin/python -m music_critic.experiments.phase9cd.run continue \
      --parent-output-root "$parent_output_root" \
      --mlp-reference-root "$mlp_reference_root" \
      --config "$config_json" \
      --output-root "$output_root" \
      --start-update 3000 \
      --target-update 15000 \
      --validation-milestones 3000,6000,9000,12000,15000 2>&1 | tee -a "$log_path"
  else
    .venv/bin/python -m music_critic.experiments.phase9cc_continuation.run continue \
      --parent-output-root "$parent_output_root" \
      --config "$config_json" \
      --output-root "$output_root" \
      --start-update 9000 \
      --target-update 15000 \
      --validation-milestones 9000,12000,15000 2>&1 | tee -a "$log_path"
  fi
  status=${PIPESTATUS[0]}
  mkdir -p "$output_root"
  cp "$log_path" "$output_root/execution.log"
  if [[ $status -ne 0 ]]; then
    printf '%s\n' "$output_root" > "$output_root/FAILED_ROOT.txt"
    printf '%s\n' "$log_path" > "$output_root/FAILED_LOG.txt"
    echo "phase9cc.continuation.failed_root=$output_root" >&2
    echo "phase9cc.continuation.failed_log=$log_path" >&2
    exit "$status"
  fi
  rm -f "$output_root/FAILED_ROOT.txt" "$output_root/FAILED_LOG.txt"
  if [[ "$phase9cd" == true ]]; then
    .venv/bin/python -m music_critic.experiments.phase9cd.run finalize \
      --output-root "$output_root" --expected-sha "$expected_sha" || exit 1
    .venv/bin/python scripts/verify_phase9cd_continuation.py \
      --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
  else
    .venv/bin/python -m music_critic.experiments.phase9cc_continuation.run finalize \
      --output-root "$output_root" --expected-sha "$expected_sha" || exit 1
    .venv/bin/python scripts/verify_phase9cc_continuation.py \
      --bundle "$output_root" --expected-sha "$expected_sha" || exit 1
  fi
  echo "phase9cc.rtx.continue.complete"
  echo "EVIDENCE_BUNDLE=$output_root"
  echo "LOG=$output_root/execution.log"
  exit 0
fi

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
