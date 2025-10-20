#!/usr/bin/env python3
"""
Context Management Reminder Hook
Reminds user to compact chat when context is getting large
"""
import json
import sys
import os
from pathlib import Path


def should_remind_compact(data: dict) -> tuple[bool, str]:
    """
    Determine if we should remind about compacting based on:
    - Number of messages in conversation
    - Token usage if available
    - Complexity of recent exchanges

    Returns: (should_remind, reason)
    """
    # Get conversation metadata if available
    messages = data.get("messages", [])
    token_usage = data.get("token_usage", {})

    # Count messages
    message_count = len(messages)

    # Get token usage
    total_tokens = token_usage.get("total_tokens", 0)
    remaining_tokens = token_usage.get("remaining_tokens", 0)

    # Thresholds for reminders
    HIGH_MESSAGE_COUNT = 50  # Suggest compact after 50 messages
    MEDIUM_MESSAGE_COUNT = 30  # Light suggestion after 30
    TOKEN_THRESHOLD = 0.7  # Remind if using >70% of context

    # Calculate token usage percentage
    if total_tokens and remaining_tokens:
        used_percentage = 1 - (remaining_tokens / (total_tokens + remaining_tokens))
    else:
        used_percentage = 0

    # Decision logic
    if message_count >= HIGH_MESSAGE_COUNT:
        return True, f"Conversation has {message_count} messages - consider compacting to optimize context"

    if used_percentage >= TOKEN_THRESHOLD:
        percentage = int(used_percentage * 100)
        return True, f"Context usage at {percentage}% - good time to compact for efficiency"

    if message_count >= MEDIUM_MESSAGE_COUNT and used_percentage >= 0.5:
        return True, f"Context building up ({message_count} messages, {int(used_percentage*100)}% tokens) - compact when convenient"

    return False, ""


def main():
    """
    Hook that runs before user prompt submission.
    Checks context state and suggests compacting if needed.
    """
    try:
        # Read hook data from stdin
        data = json.load(sys.stdin)
    except Exception as e:
        # If we can't read data, silently pass (don't block)
        sys.stderr.write(f"Context reminder hook: Could not read data - {e}\n")
        sys.exit(0)

    # Check if we should remind
    should_remind, reason = should_remind_compact(data)

    if should_remind:
        # Output reminder directly to stdout so it appears in chat
        print(f"\n💡 CONTEXT TIP: {reason}\n")

        # Also write to file for manual checking
        reminder_file = Path.home() / ".moola_context_reminder"
        with open(reminder_file, "w") as f:
            f.write(reason)

    # Always exit 0 (don't block the prompt)
    sys.exit(0)


if __name__ == "__main__":
    main()
