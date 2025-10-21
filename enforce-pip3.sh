#!/bin/bash
if grep -r " pip " . --include="*.py" --include="*.sh" 2>/dev/null | grep -v "pip3" | grep -v "pre-commit"; then
    echo "ERROR: Use pip3, not pip"
    exit 1
fi