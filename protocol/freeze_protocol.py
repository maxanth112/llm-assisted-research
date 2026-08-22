#!/usr/bin/env python3
"""
Protocol Freezing and Verification Tool

This script computes SHA-256 hashes of all critical files in the experimental
protocol (prompts, datasets, analysis code, scoring code) and writes them to
PROTOCOL.lock.json. This ensures reproducibility and detects any post-hoc
modifications to the protocol.

Usage:
    # Freeze the protocol (compute and save hashes)
    python protocol/freeze_protocol.py --freeze

    # Verify the protocol (compare current hashes to locked hashes)
    python protocol/freeze_protocol.py --verify

    # Show current hashes without saving
    python protocol/freeze_protocol.py --show

Exit codes:
    0: Success (freeze successful, or verification passed)
    1: Verification failed (hashes do not match locked values)
    2: File not found or other error
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


# Files tracked by the protocol lock, grouped by category.
# Paths are relative to the repository root.
TRACKED_FILES = {
    "protocol": [
        "protocol/protocol.md",
    ],
    "prompts": [
        "harness/prompts/templates.py",
    ],
    "harness": [
        "harness/schemas.py",
        "harness/conditions.py",
        "harness/parsers.py",
        "harness/randomization.py",
        "harness/scoring.py",
        "harness/engine.py",
    ],
    "providers": [
        "harness/providers/base.py",
        "harness/providers/mock_adapter.py",
        "harness/providers/openai_compat.py",
    ],
    "datasets": [
        "datasets/t2_generator/generator.py",
    ],
    "analysis": [
        "analysis/leakage_check.py",
        "analysis/power_simulation.py",
    ],
}


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file to hash

    Returns:
        Hexadecimal SHA-256 hash string

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_all_hashes(repo_root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Compute hashes for all tracked files in the protocol.

    Args:
        repo_root: Root directory of the repository

    Returns:
        Dictionary mapping category -> {relative_path: hash}
    """
    all_hashes: Dict[str, Dict[str, Optional[str]]] = {}

    for category, file_list in TRACKED_FILES.items():
        category_hashes: Dict[str, Optional[str]] = {}
        for rel_path in file_list:
            full_path = repo_root / rel_path
            if full_path.exists():
                category_hashes[rel_path] = compute_file_hash(full_path)
            else:
                print(f"WARNING: File not found: {full_path}", file=sys.stderr)
                category_hashes[rel_path] = None
        all_hashes[category] = category_hashes

    return all_hashes


def flatten_hashes(categorized: Dict[str, Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    """Flatten categorized hashes into a single dict of path -> hash."""
    flat: Dict[str, Optional[str]] = {}
    for _category, file_hashes in categorized.items():
        flat.update(file_hashes)
    return flat


def freeze_protocol(repo_root: Path, lock_file: Path) -> None:
    """
    Freeze the protocol by computing all hashes and writing to lock file.

    Args:
        repo_root: Root directory of the repository
        lock_file: Path to PROTOCOL.lock.json
    """
    print("Computing hashes for all protocol files...")
    categorized_hashes = compute_all_hashes(repo_root)
    flat_hashes = flatten_hashes(categorized_hashes)

    # Collect all tracked file paths
    all_tracked = []
    for file_list in TRACKED_FILES.values():
        all_tracked.extend(file_list)

    # Count warnings
    n_missing = sum(1 for v in flat_hashes.values() if v is None)
    n_found = len(flat_hashes) - n_missing

    lock_data = {
        "protocol_version": "1.0.0",
        "hash_algorithm": "SHA-256",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Hashes computed by protocol/freeze_protocol.py. Changes after freezing require version bump and re-preregistration.",
        "files_tracked": sorted(all_tracked),
        "file_hashes": {k: v for k, v in sorted(flat_hashes.items())},
    }

    # Write lock file
    with open(lock_file, 'w') as f:
        json.dump(lock_data, f, indent=2)
        f.write('\n')

    print(f"Hashed {n_found}/{len(flat_hashes)} files ({n_missing} missing)")
    print(f"Protocol frozen at {lock_data['frozen_at']}")
    print(f"Lock file written to: {lock_file}")


def verify_protocol(repo_root: Path, lock_file: Path) -> bool:
    """
    Verify that current file hashes match locked hashes.

    Args:
        repo_root: Root directory of the repository
        lock_file: Path to PROTOCOL.lock.json

    Returns:
        True if verification passed, False otherwise
    """
    if not lock_file.exists():
        print(f"ERROR: Lock file not found: {lock_file}", file=sys.stderr)
        print("Run with --freeze to create the lock file.", file=sys.stderr)
        return False

    with open(lock_file, 'r') as f:
        lock_data = json.load(f)

    print(f"Verifying against protocol frozen at: {lock_data.get('frozen_at', 'UNKNOWN')}")

    locked_hashes = lock_data.get("file_hashes", {})
    mismatches = []
    missing = []

    for rel_path, expected_hash in sorted(locked_hashes.items()):
        full_path = repo_root / rel_path
        if not full_path.exists():
            missing.append(rel_path)
            continue

        actual_hash = compute_file_hash(full_path)
        if actual_hash != expected_hash:
            mismatches.append({
                "file": rel_path,
                "expected": expected_hash,
                "actual": actual_hash,
            })

    if missing:
        print(f"\nMISSING FILES ({len(missing)}):")
        for f in missing:
            print(f"  {f}")

    if mismatches:
        print(f"\nHASH MISMATCHES ({len(mismatches)}):")
        for mm in mismatches:
            print(f"  File: {mm['file']}")
            print(f"    Expected: {mm['expected']}")
            print(f"    Actual:   {mm['actual']}")
            print()

        print("IMPORTANT: Changes to frozen protocol files after preregistration may invalidate")
        print("the study. If changes are necessary:")
        print("  1. Document the change in protocol/DEVIATIONS.md")
        print("  2. Bump protocol_version in PROTOCOL.lock.json")
        print("  3. Re-freeze with --freeze")
        print("  4. Consider re-preregistering if changes are substantial")
        return False

    if missing:
        print("\nWARNING: Some files are missing but no hash mismatches found.")
        return False

    print("\nVERIFICATION PASSED: All hashes match locked protocol")
    return True


def show_hashes(repo_root: Path) -> None:
    """
    Display current hashes without saving.

    Args:
        repo_root: Root directory of the repository
    """
    print("Current file hashes:\n")
    categorized = compute_all_hashes(repo_root)

    for category, file_hashes in categorized.items():
        print(f"[{category}]")
        for rel_path, hash_val in sorted(file_hashes.items()):
            status = hash_val[:16] + "..." if hash_val else "NOT FOUND"
            print(f"  {rel_path}: {status}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Freeze or verify experimental protocol hashes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true',
                      help='Compute and save hashes to PROTOCOL.lock.json')
    group.add_argument('--verify', action='store_true',
                      help='Verify current hashes match locked hashes')
    group.add_argument('--show', action='store_true',
                      help='Show current hashes without saving')

    parser.add_argument('--repo-root', type=Path, default=None,
                       help='Repository root directory (default: auto-detect)')

    args = parser.parse_args()

    # Auto-detect repo root (assume script is in protocol/ subdirectory)
    if args.repo_root:
        repo_root = args.repo_root
    else:
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent

    lock_file = repo_root / 'protocol' / 'PROTOCOL.lock.json'

    try:
        if args.freeze:
            freeze_protocol(repo_root, lock_file)
            sys.exit(0)
        elif args.verify:
            success = verify_protocol(repo_root, lock_file)
            sys.exit(0 if success else 1)
        elif args.show:
            show_hashes(repo_root)
            sys.exit(0)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
