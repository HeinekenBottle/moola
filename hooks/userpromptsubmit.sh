#!/usr/bin/env bash
set -euo pipefail

# UserPromptSubmit — block legacy refs and enforce "python3" wording

PROMPT="${CLAUDE_PROMPT:-}"

# 1) No legacy projects
if grep -qiE '\b(pivot|hopsketch|iron)\b' <<<"$PROMPT"; then
  echo "deny: legacy reference in prompt"; exit 1
fi

# 2) Enforce python3 (reject 'python ' usage)
if echo "$PROMPT" | grep -qE '\bpython\b' | grep -qvE '\bpython3\b'; then
  echo "deny: use 'python3', not 'python'"; exit 1
fi

echo "ok"
