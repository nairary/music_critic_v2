#!/usr/bin/env bash

set -euo pipefail

git rev-parse --verify HEAD^{commit} >/dev/null

git ls-tree -r -z --full-tree HEAD \
  | while IFS= read -r -d '' entry; do
      path="${entry#*$'\t'}"
      case "$path" in
        docs/* | *.md)
          ;;
        *)
          printf '%s\0' "$entry"
          ;;
      esac
    done \
  | sha256sum \
  | awk '{print $1}'
