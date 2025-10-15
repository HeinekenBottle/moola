#!/bin/bash
# Master script to run complete forensic audit

set -e  # Exit on error

echo "================================================================================"
echo "                      MOOLA FORENSIC AUDIT - COMPLETE RUN"
echo "================================================================================"
echo ""
echo "This will run all 5 phases of the forensic audit:"
echo "  Phase 1: Index-level data flow tracing"
echo "  Phase 2: Feature contamination analysis"
echo "  Phase 3: Averaging/smoothing detection"
echo "  Phase 4: Window region verification"
echo "  Phase 5: Architecture comparison"
echo ""
echo "Output will be saved to reports/forensic_audit_TIMESTAMP.log"
echo ""

# Create reports directory
mkdir -p reports

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="reports/forensic_audit_${TIMESTAMP}.log"

echo "Logging to: $LOGFILE"
echo ""

# Run all phases and capture output
{
    echo "================================================================================"
    echo "FORENSIC AUDIT STARTED: $(date)"
    echo "================================================================================"
    echo ""

    echo ">>> Phase 1: Index-Level Data Flow Tracing"
    python3 scripts/forensic_audit_pt1_trace.py
    echo ""

    echo ">>> Phase 2: Feature Contamination Analysis"
    python3 scripts/forensic_audit_pt2_contamination.py
    echo ""

    echo ">>> Phase 3: Averaging/Smoothing Detection"
    python3 scripts/forensic_audit_pt3_smoothing.py
    echo ""

    echo ">>> Phase 4: Window Region Verification"
    python3 scripts/forensic_audit_pt4_regions.py
    echo ""

    echo ">>> Phase 5: Architecture Comparison"
    python3 scripts/forensic_audit_pt5_architecture.py
    echo ""

    echo "================================================================================"
    echo "FORENSIC AUDIT COMPLETED: $(date)"
    echo "================================================================================"
} 2>&1 | tee "$LOGFILE"

echo ""
echo "================================================================================"
echo "Forensic audit complete!"
echo "Full report saved to: $LOGFILE"
echo ""
echo "Next steps:"
echo "  1. Review the report: cat $LOGFILE"
echo "  2. Implement recommended fixes based on findings"
echo "  3. Re-run training pipeline to validate improvements"
echo "================================================================================"
