#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: classify-ci-change.sh BASE_SHA HEAD_SHA CHANGED_FILES_PATH" >&2
  exit 2
fi

base_sha="$1"
head_sha="$2"
changed_files_path="$3"

emit_fail_open() {
  local reason="$1"
  : > "$changed_files_path"
  printf '%s\n' \
    "ci_relevant=true" \
    "change_detection_succeeded=false" \
    "changed_count=0" \
    "reason=$reason"
}

if ! base_commit=$(
  git rev-parse --verify --end-of-options "${base_sha}^{commit}" 2>/dev/null
); then
  emit_fail_open "The comparison base could not be resolved; running fail-open."
  exit 0
fi
if ! head_commit=$(
  git rev-parse --verify --end-of-options "${head_sha}^{commit}" 2>/dev/null
); then
  emit_fail_open "The comparison head could not be resolved; running fail-open."
  exit 0
fi
if ! git -c core.quotePath=true diff \
  --name-only \
  --no-renames \
  "$base_commit" \
  "$head_commit" \
  -- > "$changed_files_path"
then
  emit_fail_open "Git change detection failed; running fail-open."
  exit 0
fi

ci_relevant=false
changed_count=0
while IFS= read -r path || [[ -n "$path" ]]; do
  [[ -z "$path" ]] && continue
  changed_count=$((changed_count + 1))
  case "$path" in
    docs/* | *.md)
      ;;
    *)
      ci_relevant=true
      ;;
  esac
done < "$changed_files_path"

if [[ "$changed_count" -eq 0 ]]; then
  ci_relevant=true
  reason="Change detection returned no paths; running fail-open."
elif [[ "$ci_relevant" == true ]]; then
  reason="At least one changed path is outside docs/** and *.md."
else
  reason="All changed paths are limited to docs/** or *.md."
fi

printf '%s\n' \
  "ci_relevant=$ci_relevant" \
  "change_detection_succeeded=true" \
  "changed_count=$changed_count" \
  "reason=$reason"
