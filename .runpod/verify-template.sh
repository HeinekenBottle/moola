#!/bin/bash
# Template Verification Script
# Run this BEFORE optimized-setup.sh to verify RunPod template has required packages
# This prevents 45+ minute pip compilation disasters

set -e

echo "🔍 RunPod Template Verification"
echo "==============================="
echo ""

python3 << 'VERIFY_EOF'
import sys

# Core packages that MUST be in the template
required_packages = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
    'sklearn': 'scikit-learn'
}

missing = []
found = {}
versions = {}

print("Checking template packages...")
print("")

for module_name, display_name in required_packages.items():
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        versions[display_name] = version
        found[display_name] = True
        print(f"✅ {display_name:15s}: {version}")
    except ImportError as e:
        missing.append(display_name)
        found[display_name] = False
        print(f"❌ {display_name:15s}: NOT FOUND")

print("")

if missing:
    print("=" * 60)
    print("❌ CRITICAL ERROR: Wrong RunPod Template!")
    print("=" * 60)
    print("")
    print(f"Missing packages: {', '.join(missing)}")
    print("")
    print("⚠️  WITHOUT THESE PACKAGES:")
    print("   - Setup will take 45-60 minutes (pip compilation)")
    print("   - You will waste GPU time and money")
    print("   - Training will NOT start")
    print("")
    print("🛑 ACTION REQUIRED:")
    print("   1. TERMINATE this pod immediately")
    print("   2. Select correct template:")
    print("      → runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04")
    print("      → OR any PyTorch template with full scientific stack")
    print("   3. Verify template includes: pandas, scipy, scikit-learn")
    print("   4. Re-run this script before setup")
    print("")
    sys.exit(1)

print("=" * 60)
print("✅ SUCCESS: Template is correct!")
print("=" * 60)
print("")
print("Your template includes all required packages:")
for name, version in versions.items():
    print(f"  • {name}: {version}")
print("")
print("🚀 You can now proceed with setup:")
print("   bash /workspace/scripts/optimized-setup.sh")
print("")
print("Expected setup time: 60-90 seconds")
print("Expected venv size: ~50MB")
print("")
VERIFY_EOF

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ Template verification PASSED"
else
    echo "❌ Template verification FAILED"
    echo ""
    echo "DO NOT PROCEED WITH SETUP - You will waste 45+ minutes"
fi

exit $exit_code
