#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 --expected-head SHA --raw-index PATH --raw-cache-root DIR --target-index PATH --target-cache-root DIR --split-manifest PATH --output-root DIR [--python PATH] [--updates 10..20] [--validation-limit N]" >&2
}

expected_head=""
raw_index=""
raw_cache_root=""
target_index=""
target_cache_root=""
split_manifest=""
output_root=""
python_bin="python"
updates="10"
validation_limit="8"

while (($#)); do
  case "$1" in
    --expected-head) expected_head="${2:-}"; shift 2 ;;
    --raw-index) raw_index="${2:-}"; shift 2 ;;
    --raw-cache-root) raw_cache_root="${2:-}"; shift 2 ;;
    --target-index) target_index="${2:-}"; shift 2 ;;
    --target-cache-root) target_cache_root="${2:-}"; shift 2 ;;
    --split-manifest) split_manifest="${2:-}"; shift 2 ;;
    --output-root) output_root="${2:-}"; shift 2 ;;
    --python) python_bin="${2:-}"; shift 2 ;;
    --updates) updates="${2:-}"; shift 2 ;;
    --validation-limit) validation_limit="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in "$expected_head" "$raw_index" "$raw_cache_root" "$target_index" "$target_cache_root" "$split_manifest" "$output_root"; do
  if [[ -z "$value" ]]; then
    usage
    exit 2
  fi
done
if [[ ! "$expected_head" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected HEAD must be an exact lowercase 40-character SHA" >&2
  exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python executable is unavailable: $python_bin" >&2
  exit 2
fi
if [[ ! "$updates" =~ ^[0-9]+$ ]] || ((updates < 10 || updates > 20)); then
  echo "updates must lie in [10, 20]" >&2
  exit 2
fi
if [[ ! "$validation_limit" =~ ^[0-9]+$ ]] || ((validation_limit < 2)); then
  echo "validation-limit must be at least 2" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
mkdir -p -- "$output_root"
output_root="$(cd "$output_root" && pwd -P)"
run_id="${PHASE9B2C_RUN_ID:-phase9b2c-$(date -u +%Y%m%dT%H%M%S)-${expected_head:0:12}-$$}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run ID contains unsafe characters" >&2
  exit 2
fi
staging="$output_root/.${run_id}.partial"
final="$output_root/$run_id"
failed="$output_root/.${run_id}.failed"
for path in "$staging" "$final" "$failed"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "output collision: $path" >&2
    exit 2
  fi
done
mkdir -- "$staging"
mkdir -- "$staging/evidence"

publish_failure() {
  local status=$?
  if [[ -d "$staging" && ! -e "$failed" ]]; then
    mv -- "$staging" "$failed"
    echo "failed run retained at: $failed" >&2
  fi
  exit "$status"
}
trap publish_failure ERR

set +e
"$python_bin" -m music_critic.experiments.dilemmadata.supervised_smoke run \
  --repo-root "$repo_root" \
  --expected-head "$expected_head" \
  --raw-index "$raw_index" \
  --raw-cache-root "$raw_cache_root" \
  --target-index "$target_index" \
  --target-cache-root "$target_cache_root" \
  --split-manifest "$split_manifest" \
  --output-root "$output_root" \
  --output-dir "$staging/evidence" \
  --updates "$updates" \
  --validation-limit "$validation_limit" 2>&1 | tee "$staging/evidence/execution.log"
run_status=${PIPESTATUS[0]}
set -e
if ((run_status != 0)); then
  mv -- "$staging" "$failed"
  trap - ERR
  echo "failed run retained at: $failed" >&2
  exit "$run_status"
fi

"$python_bin" -m music_critic.experiments.dilemmadata.supervised_smoke seal \
  --evidence-dir "$staging/evidence" \
  --expected-head "$expected_head"
"$python_bin" scripts/verify_phase9b2c_rtx3090_supervised_smoke.py \
  --evidence-dir "$staging/evidence" \
  --expected-head "$expected_head"
"$python_bin" -m music_critic.experiments.dilemmadata.supervised_smoke pack \
  --evidence-dir "$staging/evidence" \
  --tar "$staging/evidence.tar" \
  --sidecar "$staging/evidence.tar.sha256" \
  --expected-head "$expected_head"
"$python_bin" scripts/verify_phase9b2c_rtx3090_supervised_smoke.py \
  --bundle "$staging/evidence.tar" \
  --sidecar "$staging/evidence.tar.sha256" \
  --expected-head "$expected_head"

mv -- "$staging" "$final"
trap - ERR
echo "Phase 9B.2C evidence directory: $final/evidence"
echo "Phase 9B.2C evidence bundle: $final/evidence.tar"
echo "Phase 9B.2C SHA-256 sidecar: $final/evidence.tar.sha256"
