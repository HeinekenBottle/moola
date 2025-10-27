#!/usr/bin/env python3
"""Generate training report from metrics."""

import datetime
import json
import sys
from pathlib import Path


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "enhanced_simple_lstm"
    artifacts_dir = Path("artifacts")
    reports_dir = artifacts_dir / "reports"

    metrics_path = artifacts_dir / "models" / model / "metrics.json"

    if not metrics_path.exists():
        print("❌ No metrics found - run make eval first")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    report = """# MOOLA Training Report

## Model Performance
"""
    report += f"- **Model**: {metrics.get('model', 'N/A')}\n"
    report += f"- **Accuracy**: {metrics.get('accuracy', 0):.3f}\n"
    report += f"- **F1 Score**: {metrics.get('f1', 0):.3f}\n"
    report += f"- **Precision**: {metrics.get('precision', 0):.3f}\n"
    report += f"- **Recall**: {metrics.get('recall', 0):.3f}\n"
    report += f"- **CV Folds**: {metrics.get('cv_folds', 'N/A')}\n"

    report += """

## Training Configuration
"""
    report += f"- **Device**: {metrics.get('device', 'N/A')}\n"
    report += f"- **Epochs**: {metrics.get('epochs', 'N/A')}\n"
    report += f"- **Seed**: {metrics.get('seed', 'N/A')}\n"
    report += f"- **Timestamp**: {metrics.get('timestamp', 'N/A')}\n"

    report += """

## Fold Details
"""
    fold_details = metrics.get("fold_details", [])
    for i, fold in enumerate(fold_details, 1):
        report += f"""### Fold {i}
- Accuracy: {fold.get('accuracy', 0):.3f}
- F1: {fold.get('f1', 0):.3f}

"""

    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"training_report_{timestamp}.md"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
