#!/usr/bin/env bash
set -euo pipefail

# PostToolUse — autosnapshot + artifact sanity

# 1) Autosnapshot small diffs
git add -A
if ! git diff --cached --quiet; then
  git commit -m "auto: post-tool snapshot $(date -u +%FT%TZ)"
fi

# 2) Metrics sanity (if present, must have accuracy and f1)
if [[ -f data/artifacts/metrics.json ]] && command -v jq >/dev/null; then
  jq -e '.accuracy and .f1' data/artifacts/metrics.json >/dev/null || {
    echo "warn: metrics.json missing keys"; exit 0;
  }
fi

echo "ok"
