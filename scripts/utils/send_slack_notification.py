#!/usr/bin/env python3
"""Send Slack notifications for experiment completion.

This script sends formatted Slack messages with experiment results,
including comparison reports and key metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


def send_slack_message(
    webhook_url: str,
    channel: str,
    title: str,
    message: str,
    report_path: Optional[Path] = None,
    color: str = "good"
) -> bool:
    """Send a Slack message via webhook.

    Args:
        webhook_url: Slack webhook URL
        channel: Slack channel name
        title: Message title
        message: Message body
        report_path: Optional path to markdown report to attach
        color: Message color (good, warning, danger)

    Returns:
        True if successful, False otherwise
    """
    # Build message payload
    attachments = [{
        "color": color,
        "title": title,
        "text": message,
        "footer": "Moola ML Experiments",
        "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
        "ts": int(__import__('time').time())
    }]

    # Add report as code block if provided
    if report_path and report_path.exists():
        with open(report_path) as f:
            report_content = f.read()

        # Truncate if too long (Slack limit: 3000 chars)
        if len(report_content) > 3000:
            report_content = report_content[:2950] + "\n...\n(truncated)"

        attachments.append({
            "color": "#36a64f",
            "title": "Comparison Report",
            "text": f"```\n{report_content}\n```",
            "mrkdwn_in": ["text"]
        })

    payload = {
        "channel": channel,
        "username": "LSTM Experiment Bot",
        "icon_emoji": ":robot_face:",
        "attachments": attachments
    }

    # Send request
    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()

        logger.success(f"Slack notification sent to {channel}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Slack notification: {e}")
        return False


def format_experiment_summary(
    phase: Optional[str] = None,
    duration: Optional[str] = None,
    best_accuracy: Optional[float] = None,
    best_class1_acc: Optional[float] = None
) -> str:
    """Format experiment summary message.

    Args:
        phase: Phase name
        duration: Duration string
        best_accuracy: Best overall accuracy
        best_class1_acc: Best class 1 accuracy

    Returns:
        Formatted message string
    """
    lines = []

    if phase:
        lines.append(f"*Phase*: {phase}")

    if duration:
        lines.append(f"*Duration*: {duration}")

    if best_accuracy is not None:
        lines.append(f"*Best Accuracy*: {best_accuracy:.2%}")

    if best_class1_acc is not None:
        emoji = "✅" if best_class1_acc >= 0.30 else "⚠️"
        lines.append(f"*Class 1 Accuracy*: {best_class1_acc:.2%} {emoji}")

    lines.append("")
    lines.append("View results in MLflow UI: http://mlflow:5000")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Send Slack notification for experiment results"
    )
    parser.add_argument(
        "--webhook-url",
        required=True,
        help="Slack webhook URL",
    )
    parser.add_argument(
        "--channel",
        default="#ml-experiments",
        help="Slack channel name",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Message title",
    )
    parser.add_argument(
        "--message",
        required=True,
        help="Message body",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Path to comparison report markdown file",
    )
    parser.add_argument(
        "--color",
        choices=["good", "warning", "danger"],
        default="good",
        help="Message color",
    )

    args = parser.parse_args()

    logger.info(f"Sending Slack notification to {args.channel}")

    success = send_slack_message(
        webhook_url=args.webhook_url,
        channel=args.channel,
        title=args.title,
        message=args.message,
        report_path=args.report,
        color=args.color
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
