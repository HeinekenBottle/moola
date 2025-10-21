#!/bin/bash
if grep -r "python " . --include="*.py" --include="*.sh" 2>/dev/null | grep -v "python3" | grep -v "pre-commit"; then
    echo "ERROR: Use python3, not python"
    exit 1
fi