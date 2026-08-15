#!/usr/bin/env bash
# Independent Phase 8B.2A production-format real-corpus bounded smoke.
# The outer subshell prevents strict-mode settings from leaking when sourced.
(
set -euo pipefail

EXPECTED_SHA="${1:-}"
REPOSITORY_ROOT="/home/humtech/Paper/critic"
BRANCH="phase/8b2a-scientific-comparison-protocol"
SOURCE_PLAN="${PHASE8B2_SOURCE_PLAN:-outputs/phase8b2a-real-gpu-smoke-20260815-142857/plan.json}"

if [[ ! "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "usage: $0 EXACT_40_CHARACTER_GIT_SHA" >&2
    exit 64
fi

cd "$REPOSITORY_ROOT"
if [[ ! -f .venv/bin/activate ]]; then
    echo "missing virtual environment: $REPOSITORY_ROOT/.venv" >&2
    exit 65
fi
# shellcheck disable=SC1091
source .venv/bin/activate

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT_ROOT="$REPOSITORY_ROOT/outputs/phase8b2a-real-corpus-bounded-smoke-$RUN_ID"
EVIDENCE_ROOT="$REPOSITORY_ROOT/outputs/phase8b2a-real-corpus-bounded-smoke-evidence-$RUN_ID"
ARCHIVE="$EVIDENCE_ROOT/phase8b2a-rtx3090-bounded-smoke-evidence.tar.gz"
PREFLIGHT_LOG="$EVIDENCE_ROOT/exact-head-preflight.log"
NVIDIA_LOG="$EVIDENCE_ROOT/nvidia-smi.txt"
SMOKE_LOG="$EVIDENCE_ROOT/smoke.log"
VERIFICATION_OUTPUT="$EVIDENCE_ROOT/verification.json"
INVOCATION_CONFIG="$EVIDENCE_ROOT/invocation-config.json"

if [[ -e "$OUTPUT_ROOT" || -e "$EVIDENCE_ROOT" ]]; then
    echo "refusing to reuse an existing smoke or evidence root" >&2
    exit 66
fi
mkdir -p "$EVIDENCE_ROOT"

finish() {
    status=$?
    trap - EXIT
    if (( status == 0 )); then
        echo "gate completed; output root: $OUTPUT_ROOT"
        echo "evidence archive: $ARCHIVE"
        echo "archive checksum: $ARCHIVE.sha256"
    else
        echo "gate failed with status $status; preserved evidence root: $EVIDENCE_ROOT" >&2
        echo "smoke output root, if created: $OUTPUT_ROOT" >&2
    fi
    exit "$status"
}
trap finish EXIT

log() {
    printf '%s\n' "$*" | tee -a "$PREFLIGHT_LOG"
}

run_preflight_command() {
    "$@" 2>&1 | tee -a "$PREFLIGHT_LOG"
}

record_untracked() {
    label="$1"
    destination="$2"
    git ls-files --others --exclude-standard > "$destination"
    log "$label"
    if [[ -s "$destination" ]]; then
        run_preflight_command sed 's/^/  /' "$destination"
    else
        log "  (none)"
    fi
}

log "run label: production-format real-corpus bounded smoke"
log "repository: $REPOSITORY_ROOT"
log "requested exact SHA: $EXPECTED_SHA"
log "UTC run id: $RUN_ID"

if ! git diff --quiet; then
    log "ERROR: tracked unstaged changes are present; nothing was removed or moved"
    run_preflight_command git diff --stat
    exit 67
fi
if ! git diff --cached --quiet; then
    log "ERROR: staged changes are present; nothing was removed or moved"
    run_preflight_command git diff --cached --stat
    exit 68
fi
record_untracked \
    "untracked files before fetch/detach (allowed and preserved):" \
    "$EVIDENCE_ROOT/untracked-before-switch.txt"
if [[ -d outputs ]]; then
    find outputs -mindepth 1 -maxdepth 1 \
        ! -name "$(basename "$EVIDENCE_ROOT")" -printf '%P\n' | sort \
        > "$EVIDENCE_ROOT/preserved-output-roots-before-run.txt"
    log "existing ignored output/evidence roots before this run (allowed and preserved):"
    if [[ -s "$EVIDENCE_ROOT/preserved-output-roots-before-run.txt" ]]; then
        run_preflight_command sed \
            's/^/  /' "$EVIDENCE_ROOT/preserved-output-roots-before-run.txt"
    else
        log "  (none)"
    fi
fi

run_preflight_command git fetch origin "$BRANCH"
FETCHED_SHA="$(git rev-parse FETCH_HEAD)"
if [[ "$FETCHED_SHA" != "$EXPECTED_SHA" ]]; then
    log "ERROR: fetched branch head $FETCHED_SHA does not equal requested $EXPECTED_SHA"
    exit 69
fi
run_preflight_command git switch --detach "$EXPECTED_SHA"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    log "ERROR: detached HEAD $ACTUAL_SHA does not equal requested $EXPECTED_SHA"
    exit 70
fi
if ! git diff --quiet; then
    log "ERROR: tracked unstaged changes appeared after detach"
    exit 71
fi
if ! git diff --cached --quiet; then
    log "ERROR: staged changes appeared after detach"
    exit 72
fi
record_untracked \
    "untracked files after exact-head detach (allowed and preserved):" \
    "$EVIDENCE_ROOT/untracked-after-switch.txt"
run_preflight_command git status --short --branch --untracked-files=all
log "exact detached HEAD verified: $ACTUAL_SHA"

if [[ "$SOURCE_PLAN" != /* ]]; then
    SOURCE_PLAN="$REPOSITORY_ROOT/$SOURCE_PLAN"
fi
if [[ ! -f "$SOURCE_PLAN" ]]; then
    log "ERROR: source plan for production paths is missing: $SOURCE_PLAN"
    exit 73
fi
run_preflight_command python -c \
    'import json, pathlib, sys; p=json.loads(pathlib.Path(sys.argv[1]).read_text()); r=p["runtime_paths"]; assert len(r["index_paths"]) == 2; assert len(r["cache_roots"]) == 2; assert isinstance(r["split_manifest"], str) and r["split_manifest"]' \
    "$SOURCE_PLAN"
mapfile -d '' -t DATA_PATHS < <(
    python -c \
        'import json, pathlib, sys; r=json.loads(pathlib.Path(sys.argv[1]).read_text())["runtime_paths"]; values=[*r["index_paths"], *r["cache_roots"], r["split_manifest"]]; sys.stdout.buffer.write(b"".join(str(value).encode()+b"\0" for value in values))' \
        "$SOURCE_PLAN"
)
if (( ${#DATA_PATHS[@]} != 5 )); then
    log "ERROR: expected exactly two indices, two cache roots, and one split manifest"
    exit 74
fi
INDEX_PATHS=("${DATA_PATHS[0]}" "${DATA_PATHS[1]}")
CACHE_ROOTS=("${DATA_PATHS[2]}" "${DATA_PATHS[3]}")
SPLIT_MANIFEST="${DATA_PATHS[4]}"
for path in "${INDEX_PATHS[@]}"; do
    if [[ ! -f "$path" ]]; then
        log "ERROR: required production index is missing: $path"
        exit 75
    fi
    log "production index present: $path"
done
for path in "${CACHE_ROOTS[@]}"; do
    if [[ ! -d "$path" ]]; then
        log "ERROR: required production cache root is missing: $path"
        exit 76
    fi
    log "production cache root present: $path"
done
if [[ ! -f "$SPLIT_MANIFEST" ]]; then
    log "ERROR: required global split manifest is missing: $SPLIT_MANIFEST"
    exit 77
fi
log "global split manifest present: $SPLIT_MANIFEST"
log "old plan usage is paths-only: $SOURCE_PLAN"

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=src
run_preflight_command nvidia-smi
nvidia-smi > "$NVIDIA_LOG"
run_preflight_command python -c \
    'import json, torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.device_count() == 1, torch.cuda.device_count(); name=torch.cuda.get_device_name(0); assert "RTX 3090" in name, name; print(json.dumps({"cuda_available": True, "visible_device_count": 1, "logical_device_index": 0, "device_name": name, "torch_cuda_runtime": torch.version.cuda}, sort_keys=True))'
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
log "PYTHONPATH=$PYTHONPATH"
log "new output root: $OUTPUT_ROOT"
log "new evidence root: $EVIDENCE_ROOT"

mapfile -t DATA_OVERRIDES < <(
    python -c \
        'import json, sys; print("data.index_paths="+json.dumps(sys.argv[1:3], separators=(",", ":"))); print("data.cache_roots="+json.dumps(sys.argv[3:5], separators=(",", ":"))); print("data.split_manifest="+json.dumps(sys.argv[5]))' \
        "${INDEX_PATHS[@]}" "${CACHE_ROOTS[@]}" "$SPLIT_MANIFEST"
)
if (( ${#DATA_OVERRIDES[@]} != 3 )); then
    log "ERROR: failed to construct safe Hydra data-path overrides"
    exit 78
fi

python -c \
    'import json, pathlib, sys; payload={"run_label":"production-format real-corpus bounded smoke","expected_git_sha":sys.argv[2],"comparison":"bounded_acceptance","variants":["phase7a_control"],"seeds":[17],"ssl_optimizer_steps":1,"downstream_optimizer_steps":1,"optimizer_steps_per_epoch":1,"device":"cuda:0","amp":True,"amp_dtype":"float16","test_split_opened":False,"data_paths_source":"paths_only_from_existing_plan","source_plan":sys.argv[3],"output_root":sys.argv[4],"index_paths":sys.argv[5:7],"cache_roots":sys.argv[7:9],"split_manifest":sys.argv[9]}; pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")' \
    "$INVOCATION_CONFIG" "$EXPECTED_SHA" "$SOURCE_PLAN" "$OUTPUT_ROOT" \
    "${INDEX_PATHS[@]}" "${CACHE_ROOTS[@]}" "$SPLIT_MANIFEST"

GATE_OVERRIDES=(
    comparison=bounded_acceptance
    'comparison.variants=[phase7a_control]'
    'comparison.seeds=[17]'
    comparison.ssl_optimizer_steps=1
    comparison.downstream_optimizer_steps=1
    comparison.optimizer_steps_per_epoch=1
    comparison.bootstrap_replicates=2
    device.name=cuda:0
    device.amp=true
    device.amp_dtype=float16
    "${DATA_OVERRIDES[@]}"
)

python -m music_critic.experiments.phase8b2.run \
    action=plan \
    "output_root=$OUTPUT_ROOT" \
    "hydra.run.dir=$EVIDENCE_ROOT/hydra-plan" \
    "${GATE_OVERRIDES[@]}" \
    > "$EVIDENCE_ROOT/resolved-plan.json" \
    2> >(tee "$EVIDENCE_ROOT/plan-validation.stderr.log" >&2)
python -m json.tool "$EVIDENCE_ROOT/resolved-plan.json" > /dev/null
log "bounded one-seed production-path plan/config validation passed"

set +e
python -m music_critic.experiments.phase8b2.run \
    action=run \
    "output_root=$OUTPUT_ROOT" \
    "hydra.run.dir=$EVIDENCE_ROOT/hydra-run" \
    "${GATE_OVERRIDES[@]}" \
    2>&1 | tee "$SMOKE_LOG"
SMOKE_STATUS=${PIPESTATUS[0]}
set -e
if (( SMOKE_STATUS != 0 )); then
    echo "smoke failed with status $SMOKE_STATUS; full log preserved: $SMOKE_LOG" >&2
    exit "$SMOKE_STATUS"
fi

for artifact in \
    final_comparison_report.json \
    compute_accounting.json \
    run_manifest.json \
    comparison_protocol.json \
    actual_sample_schedule.json; do
    if [[ ! -f "$OUTPUT_ROOT/final_bundle/$artifact" ]]; then
        echo "missing final bundle artifact: $artifact" >&2
        exit 79
    fi
done

set +e
python scripts/verify_phase8b2a_rtx3090_bounded_smoke.py \
    "$OUTPUT_ROOT" \
    --expected-sha "$EXPECTED_SHA" \
    --invocation-config "$INVOCATION_CONFIG" \
    --expected-device-name "RTX 3090" \
    2>&1 | tee "$VERIFICATION_OUTPUT"
VERIFY_STATUS=${PIPESTATUS[0]}
set -e
if (( VERIFY_STATUS != 0 )); then
    echo "verification failed with status $VERIFY_STATUS" >&2
    exit "$VERIFY_STATUS"
fi

BUNDLE_ROOT="$EVIDENCE_ROOT/bundle"
mkdir -p "$BUNDLE_ROOT/preflight" "$BUNDLE_ROOT/gate" "$BUNDLE_ROOT/run"
cp "$PREFLIGHT_LOG" "$NVIDIA_LOG" \
    "$EVIDENCE_ROOT/untracked-before-switch.txt" \
    "$EVIDENCE_ROOT/untracked-after-switch.txt" \
    "$BUNDLE_ROOT/preflight/"
if [[ -f "$EVIDENCE_ROOT/preserved-output-roots-before-run.txt" ]]; then
    cp "$EVIDENCE_ROOT/preserved-output-roots-before-run.txt" \
        "$BUNDLE_ROOT/preflight/"
fi
cp "$INVOCATION_CONFIG" "$EVIDENCE_ROOT/resolved-plan.json" \
    "$EVIDENCE_ROOT/plan-validation.stderr.log" "$SMOKE_LOG" \
    "$VERIFICATION_OUTPUT" "$BUNDLE_ROOT/gate/"
cp scripts/run_phase8b2a_rtx3090_bounded_smoke.sh \
    scripts/verify_phase8b2a_rtx3090_bounded_smoke.py \
    "$BUNDLE_ROOT/gate/"
if [[ -d "$EVIDENCE_ROOT/hydra-plan" ]]; then
    cp -R "$EVIDENCE_ROOT/hydra-plan" "$BUNDLE_ROOT/gate/"
fi
if [[ -d "$EVIDENCE_ROOT/hydra-run" ]]; then
    cp -R "$EVIDENCE_ROOT/hydra-run" "$BUNDLE_ROOT/gate/"
fi
cp "$OUTPUT_ROOT/plan.json" "$OUTPUT_ROOT/actual_sample_schedule.json" \
    "$BUNDLE_ROOT/run/"
cp -R "$OUTPUT_ROOT/final_bundle" "$BUNDLE_ROOT/run/"
while IFS= read -r -d '' path; do
    relative="${path#"$OUTPUT_ROOT"/}"
    destination="$BUNDLE_ROOT/run/$relative"
    mkdir -p "$(dirname "$destination")"
    cp "$path" "$destination"
done < <(
    find "$OUTPUT_ROOT/cells" -type f \
        ! -name '*.pt' ! -name '*.pth' ! -name '*.ckpt' -print0
)

(
    cd "$BUNDLE_ROOT"
    find . -type f ! -name payload.sha256 -print0 | sort -z \
        | xargs -0 sha256sum > payload.sha256
    sha256sum -c payload.sha256
)
tar -czf "$ARCHIVE" -C "$BUNDLE_ROOT" .
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
sha256sum -c "$ARCHIVE.sha256"
)
