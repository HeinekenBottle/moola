#!/usr/bin/env bash
set -euo pipefail

# UserPromptSubmit — block legacy refs, enforce "python3", and check context optimization

PROMPT="${CLAUDE_PROMPT:-}"

# 1) No legacy projects
if grep -qiE '\b(pivot|hopsketch|iron)\b' <<<"$PROMPT"; then
  echo "deny: legacy reference in prompt"; exit 1
fi

# 2) Enforce python3 (reject 'python ' usage)
if echo "$PROMPT" | grep -qE '\bpython\b' | grep -qvE '\bpython3\b'; then
  echo "deny: use 'python3', not 'python'"; exit 1
fi

# 3) Check context optimization (non-blocking)
HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HOOK_DIR/context_reminder.py" ]; then
  # Pass conversation data to context reminder hook (if CLAUDE_CONVERSATION_JSON is available)
  if [ -n "${CLAUDE_CONVERSATION_JSON:-}" ]; then
    REMINDER_OUTPUT=$(echo "$CLAUDE_CONVERSATION_JSON" | python3 "$HOOK_DIR/context_reminder.py" 2>&1 || true)
    if [ -n "$REMINDER_OUTPUT" ]; then
      echo "$REMINDER_OUTPUT"
    fi
  fi
fi

echo "ok"
