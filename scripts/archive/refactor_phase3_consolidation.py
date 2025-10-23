#!/usr/bin/env python3

"""
Phase 3: Consolidation
Merge duplicate code, unify configs
"""

import os
import sys
import shutil
from pathlib import Path
import hashlib

def get_file_hash(filepath):
    """Get MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_duplicates(directory):
    """Find duplicate files in directory."""
    hashes = {}
    duplicates = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.yaml', '.yml')):
                filepath = os.path.join(root, file)
                file_hash = get_file_hash(filepath)
                if file_hash in hashes:
                    duplicates.append((filepath, hashes[file_hash]))
                else:
                    hashes[file_hash] = filepath
    return duplicates

def main():
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("Phase 3: Consolidation - Starting...")

    # Safety check
    if not (project_root / "pyproject.toml").exists():
        print("Error: Not in project root")
        sys.exit(1)

    # Create backup
    import tarfile
    import datetime
    backup_file = f".backup_phase3_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    print(f"Creating backup: {backup_file}")
    with tarfile.open(backup_file, "w:gz") as tar:
        for root, dirs, files in os.walk("."):
            if any(excl in root for excl in ['artifacts', 'data', 'logs', '.git']):
                continue
            for file in files:
                tar.add(os.path.join(root, file))

    # Find and merge duplicate configs
    print("Finding duplicate configs...")
    config_dir = project_root / "configs"
    duplicates = find_duplicates(config_dir)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate config files")
        for dup, orig in duplicates:
            print(f"Duplicate: {dup} -> {orig}")
            # For now, just report, don't remove
    else:
        print("No duplicate configs found")

    # Unify model configs - ensure they inherit from base
    base_config = config_dir / "_base" / "model.yaml"
    if base_config.exists():
        print("Checking model configs inherit from base...")
        # Read base config
        import yaml
        with open(base_config) as f:
            base = yaml.safe_load(f)

        # Check jade config
        jade_config = config_dir / "model" / "jade.yaml"
        if jade_config.exists():
            with open(jade_config) as f:
                jade = yaml.safe_load(f)
            # Ensure base settings are present
            for key, value in base.items():
                if key not in jade:
                    jade[key] = value
                    print(f"Added {key} to jade.yaml")
            with open(jade_config, 'w') as f:
                yaml.dump(jade, f, default_flow_style=False)

    # Merge duplicate code if any
    print("Checking for duplicate code...")
    src_dir = project_root / "src" / "moola"
    duplicates = find_duplicates(src_dir)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate source files")
        for dup, orig in duplicates:
            print(f"Duplicate: {dup} -> {orig}")
            # Archive duplicate
            archive_dir = project_root / "archive" / "duplicates"
            archive_dir.mkdir(exist_ok=True)
            shutil.move(dup, archive_dir / Path(dup).name)
            print(f"Archived duplicate: {dup}")
    else:
        print("No duplicate source files found")

    # Unify requirements files
    print("Consolidating requirements...")
    req_files = ["requirements.txt", "requirements-runpod.txt"]
    consolidated = set()
    for req_file in req_files:
        if os.path.exists(req_file):
            with open(req_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        consolidated.add(line)

    # Write back consolidated requirements
    with open("requirements.txt", 'w') as f:
        f.write("# Consolidated requirements\n")
        for req in sorted(consolidated):
            f.write(f"{req}\n")

    if os.path.exists("requirements-runpod.txt"):
        os.rename("requirements-runpod.txt", "archive/requirements-runpod.txt")
        print("Archived requirements-runpod.txt, consolidated into requirements.txt")

    print(f"Phase 3 completed. Backup: {backup_file}")
    (project_root / ".refactor_phase3_done").touch()

if __name__ == "__main__":
    main()