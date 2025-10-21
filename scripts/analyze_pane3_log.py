#!/usr/bin/env python3
"""
Analyze pane3_running_log.txt for issues, errors, and patterns.
Usage: python3 scripts/analyze_pane3_log.py [log_file]
"""
import re
import sys
from collections import Counter
from pathlib import Path


def parse_log(log_file):
    """Parse the timestamped log file."""
    with open(log_file) as f:
        content = f.read()

    # Split by capture timestamps
    captures = re.split(r'──+\nCapture at: ([\d\-: ]+)\n──+\n', content)

    sessions = []
    for i in range(1, len(captures), 2):
        if i + 1 < len(captures):
            timestamp = captures[i]
            text = captures[i + 1]
            sessions.append({'timestamp': timestamp, 'content': text})

    return sessions


def extract_issues(sessions):
    """Extract errors, problems, and patterns."""
    issues = {
        'errors': [],
        'warnings': [],
        'missing_modules': [],
        'missing_files': [],
        'ssh_failures': [],
        'todo_status': [],
        'background_tasks': []
    }

    for session in sessions:
        ts = session['timestamp']
        text = session['content']

        # Errors
        for match in re.finditer(r'(Error|ERROR|❌).*', text, re.IGNORECASE):
            issues['errors'].append({'timestamp': ts, 'message': match.group(0)})

        # Warnings
        for match in re.finditer(r'(Warning|WARNING|⚠).*', text, re.IGNORECASE):
            issues['warnings'].append({'timestamp': ts, 'message': match.group(0)})

        # Missing modules
        for match in re.finditer(r'ModuleNotFoundError.*|No module named.*|Missing.*module', text):
            issues['missing_modules'].append({'timestamp': ts, 'message': match.group(0)})

        # Missing files/directories
        for match in re.finditer(r'No such file or directory.*|FileNotFoundError', text):
            issues['missing_files'].append({'timestamp': ts, 'message': match.group(0)})

        # SSH failures
        for match in re.finditer(r'ssh:.*failed|Connection.*refused|Permission denied', text, re.IGNORECASE):
            issues['ssh_failures'].append({'timestamp': ts, 'message': match.group(0)})

        # Todo status
        if '☐' in text or '☒' in text:
            todo_lines = [line for line in text.split('\n') if '☐' in line or '☒' in line]
            issues['todo_status'].append({'timestamp': ts, 'todos': todo_lines})

        # Background tasks
        for match in re.finditer(r'\d+ background.*task', text, re.IGNORECASE):
            issues['background_tasks'].append({'timestamp': ts, 'message': match.group(0)})

    return issues


def generate_report(issues, sessions):
    """Generate human-readable report."""
    print("=" * 70)
    print("PANE 3 MONITORING REPORT")
    print("=" * 70)
    print(f"\nTotal captures: {len(sessions)}")
    if sessions:
        print(f"First capture: {sessions[0]['timestamp']}")
        print(f"Last capture: {sessions[-1]['timestamp']}")
    print()

    # Errors
    print(f"ERRORS DETECTED: {len(issues['errors'])}")
    if issues['errors']:
        print("-" * 70)
        for i, err in enumerate(issues['errors'][-10:], 1):  # Last 10
            print(f"{i}. [{err['timestamp']}]")
            print(f"   {err['message']}")
        if len(issues['errors']) > 10:
            print(f"\n   ... and {len(issues['errors']) - 10} more errors")
    print()

    # Missing modules
    print(f"MISSING MODULES: {len(issues['missing_modules'])}")
    if issues['missing_modules']:
        print("-" * 70)
        unique_modules = {}
        for item in issues['missing_modules']:
            msg = item['message']
            if msg not in unique_modules:
                unique_modules[msg] = item['timestamp']
        for msg, ts in list(unique_modules.items())[:5]:
            print(f"  [{ts}] {msg}")
    print()

    # Missing files
    print(f"MISSING FILES/DIRS: {len(issues['missing_files'])}")
    if issues['missing_files']:
        print("-" * 70)
        for item in issues['missing_files'][-5:]:
            print(f"  [{item['timestamp']}] {item['message']}")
    print()

    # SSH failures
    print(f"SSH FAILURES: {len(issues['ssh_failures'])}")
    if issues['ssh_failures']:
        print("-" * 70)
        for item in issues['ssh_failures'][-5:]:
            print(f"  [{item['timestamp']}] {item['message']}")
    print()

    # Todo progress
    print("TODO PROGRESS:")
    print("-" * 70)
    if issues['todo_status']:
        latest = issues['todo_status'][-1]
        print(f"Latest status at {latest['timestamp']}:")
        for todo in latest['todos']:
            print(f"  {todo.strip()}")
    print()

    # Background tasks
    print(f"BACKGROUND TASKS: {len(issues['background_tasks'])}")
    if issues['background_tasks']:
        print("-" * 70)
        latest = issues['background_tasks'][-1]
        print(f"  {latest['message']} (at {latest['timestamp']})")
    print()

    print("=" * 70)


def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'pane3_running_log.txt'

    if not Path(log_file).exists():
        print(f"Error: {log_file} not found")
        sys.exit(1)

    print(f"Analyzing {log_file}...\n")

    sessions = parse_log(log_file)
    issues = extract_issues(sessions)
    generate_report(issues, sessions)


if __name__ == '__main__':
    main()
