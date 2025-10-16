"""
Test RunPod orchestrator functionality before using it for training.

This script verifies that all orchestrator functions work correctly:
- SSH connectivity
- File upload/download
- Command execution
- Environment verification
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from moola.runpod import RunPodOrchestrator


def test_connection(orch: RunPodOrchestrator) -> bool:
    """Test basic SSH connectivity."""
    print("\n" + "=" * 60)
    print("TEST 1: SSH Connection")
    print("=" * 60)

    exit_code = orch.execute_command("echo 'Hello from RunPod'", timeout=10)

    if exit_code == 0:
        print("✓ SSH connection successful")
        return True
    else:
        print("✗ SSH connection failed")
        return False


def test_file_upload_download(orch: RunPodOrchestrator) -> bool:
    """Test file upload and download."""
    print("\n" + "=" * 60)
    print("TEST 2: File Upload/Download")
    print("=" * 60)

    # Create test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_file = Path(f.name)
        f.write("This is a test file for RunPod orchestrator\n")
        f.write("If you see this, file transfer works!\n")

    try:
        # Upload test file
        remote_path = f"{orch.workspace}/test_file.txt"
        upload_success = orch.upload_file(test_file, remote_path)

        if not upload_success:
            print("✗ File upload failed")
            return False

        # Verify file exists on remote
        exit_code = orch.execute_command(f"cat {remote_path}", timeout=10, stream_output=True)

        if exit_code != 0:
            print("✗ Uploaded file not found on remote")
            return False

        # Download file back
        download_path = Path(tempfile.gettempdir()) / "downloaded_test_file.txt"
        download_success = orch.download_file(remote_path, download_path)

        if not download_success:
            print("✗ File download failed")
            return False

        # Verify downloaded content matches
        with open(download_path) as f:
            downloaded_content = f.read()

        with open(test_file) as f:
            original_content = f.read()

        if downloaded_content == original_content:
            print("✓ File upload/download successful")

            # Cleanup
            orch.execute_command(f"rm {remote_path}", timeout=10, stream_output=False)
            download_path.unlink()
            return True
        else:
            print("✗ Downloaded content doesn't match original")
            return False

    finally:
        test_file.unlink()


def test_directory_operations(orch: RunPodOrchestrator) -> bool:
    """Test directory upload and download."""
    print("\n" + "=" * 60)
    print("TEST 3: Directory Upload/Download")
    print("=" * 60)

    # Create test directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_dir"
        test_dir.mkdir()

        # Create some test files
        (test_dir / "file1.txt").write_text("File 1 content")
        (test_dir / "file2.txt").write_text("File 2 content")

        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("File 3 content")

        # Upload directory
        remote_dir = f"{orch.workspace}/test_dir"
        upload_success = orch.upload_directory(test_dir, remote_dir)

        if not upload_success:
            print("✗ Directory upload failed")
            return False

        # Verify files exist on remote
        exit_code = orch.execute_command(
            f"ls -R {remote_dir}",
            timeout=10,
            stream_output=True,
        )

        if exit_code != 0:
            print("✗ Uploaded directory not found on remote")
            return False

        # Download directory back
        download_dir = Path(tmpdir) / "downloaded_dir"
        download_success = orch.download_directory(remote_dir, download_dir)

        if not download_success:
            print("✗ Directory download failed")
            return False

        # Verify structure
        expected_files = ["file1.txt", "file2.txt", "subdir/file3.txt"]
        all_exist = all((download_dir / f).exists() for f in expected_files)

        if all_exist:
            print("✓ Directory upload/download successful")

            # Cleanup
            orch.execute_command(f"rm -rf {remote_dir}", timeout=10, stream_output=False)
            return True
        else:
            print("✗ Downloaded directory structure incomplete")
            return False


def test_environment_verification(orch: RunPodOrchestrator) -> bool:
    """Test environment verification."""
    print("\n" + "=" * 60)
    print("TEST 4: Environment Verification")
    print("=" * 60)

    results = orch.verify_environment()

    # Check critical components
    critical_checks = ["SSH Connection", "PyTorch", "CUDA", "Workspace"]

    all_passed = all(results.get(check, False) for check in critical_checks)

    if all_passed:
        print("\n✓ Environment verification successful")
        return True
    else:
        failed = [check for check in critical_checks if not results.get(check, False)]
        print(f"\n✗ Environment verification failed: {', '.join(failed)}")
        return False


def test_python_imports(orch: RunPodOrchestrator) -> bool:
    """Test that moola package imports work."""
    print("\n" + "=" * 60)
    print("TEST 5: Python Imports")
    print("=" * 60)

    import_tests = [
        ("PyTorch", "import torch; print(torch.__version__)"),
        ("Moola", "import moola; print('Moola imported')"),
        ("CNN-Transformer", "from moola.models import CnnTransformerModel; print('Model imported')"),
        ("Config", "from moola.config import training_config; print('Config imported')"),
    ]

    all_passed = True

    for name, import_stmt in import_tests:
        exit_code = orch.execute_command(
            f"cd {orch.workspace} && "
            f"source /tmp/moola-venv/bin/activate && "
            f"python -c '{import_stmt}'",
            timeout=30,
            stream_output=False,
        )

        if exit_code == 0:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            all_passed = False

    if all_passed:
        print("\n✓ All imports successful")
    else:
        print("\n✗ Some imports failed")

    return all_passed


def main():
    """Run all orchestrator tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RunPod orchestrator")
    parser.add_argument("--host", default="213.173.98.6", help="RunPod host IP")
    parser.add_argument("--port", type=int, default=14385, help="SSH port")
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--workspace", default="/workspace/moola", help="Remote workspace")

    args = parser.parse_args()

    print("=" * 60)
    print("RUNPOD ORCHESTRATOR TEST SUITE")
    print("=" * 60)
    print(f"Host: {args.host}:{args.port}")
    print(f"Workspace: {args.workspace}")

    # Initialize orchestrator
    try:
        orch = RunPodOrchestrator(
            host=args.host,
            port=args.port,
            key_path=args.key,
            workspace=args.workspace,
            verbose=False,  # Reduce noise during tests
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize orchestrator: {e}")
        return 1

    # Run tests
    tests = [
        ("Connection", test_connection),
        ("File Upload/Download", test_file_upload_download),
        ("Directory Operations", test_directory_operations),
        ("Environment Verification", test_environment_verification),
        ("Python Imports", test_python_imports),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(orch)
        except Exception as e:
            print(f"\n✗ {test_name} raised exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ ALL TESTS PASSED - Orchestrator ready for use")
        return 0
    else:
        failed_count = sum(1 for v in results.values() if not v)
        print(f"\n❌ {failed_count}/{len(tests)} TESTS FAILED - Fix issues before using")
        return 1


if __name__ == "__main__":
    sys.exit(main())
