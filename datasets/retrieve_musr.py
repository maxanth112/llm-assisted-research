#!/usr/bin/env python3
"""
MuSR Dataset Retrieval Script

Downloads the MuSR dataset (murder mystery subset) from HuggingFace,
computes SHA-256 hash, updates manifest, and saves locally.

Usage:
    python datasets/retrieve_musr.py

Output:
    - datasets/data/musr/ (gitignored)
    - Updated datasets/manifests/musr_manifest.json with SHA-256
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install: pip install datasets huggingface_hub")
    sys.exit(1)


def compute_sha256(data: Any) -> str:
    """
    Compute SHA-256 hash of dataset.

    Args:
        data: Dataset object or serializable data

    Returns:
        Hex string of SHA-256 hash
    """
    # Serialize the dataset to JSON string for consistent hashing
    json_str = json.dumps(
        [item for item in data],
        sort_keys=True,
        ensure_ascii=False
    )
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def update_manifest(manifest_path: Path, sha256_hash: str) -> None:
    """
    Update manifest file with computed SHA-256 hash.

    Args:
        manifest_path: Path to manifest JSON file
        sha256_hash: Computed SHA-256 hash to insert
    """
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        manifest['sha256'] = sha256_hash

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"✓ Updated manifest: {manifest_path}")
    except Exception as e:
        print(f"ERROR updating manifest: {e}", file=sys.stderr)
        raise


def retrieve_musr() -> None:
    """
    Main retrieval function for MuSR dataset.
    """
    print("=" * 70)
    print("MuSR Dataset Retrieval")
    print("=" * 70)

    # Setup paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    data_dir = script_dir / "data" / "musr"
    manifest_path = script_dir / "manifests" / "musr_manifest.json"

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Data directory: {data_dir}")

    try:
        # Load manifest
        print(f"\n📋 Loading manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Download dataset
        print(f"\n⬇️  Downloading from: {manifest['source']}")
        print(f"   Subset: {manifest['subset']}")

        dataset = load_dataset(
            "TAUR-Lab/MuSR",
            name="murder_mysteries",
            trust_remote_code=True
        )

        print(f"✓ Download complete")

        # Filter to murder mystery subset if needed
        # (The dataset name already specifies murder_mysteries)

        # Print dataset info
        print(f"\n📊 Dataset Statistics:")
        for split_name, split_data in dataset.items():
            print(f"   {split_name}: {len(split_data)} items")

        # Save dataset locally
        print(f"\n💾 Saving to: {data_dir}")
        dataset.save_to_disk(str(data_dir))
        print(f"✓ Saved successfully")

        # Compute SHA-256
        print(f"\n🔐 Computing SHA-256 hash...")
        # Use the train split for hashing (or combine all splits)
        hash_data = dataset.get('train', dataset[list(dataset.keys())[0]])
        sha256_hash = compute_sha256(hash_data)
        print(f"   SHA-256: {sha256_hash}")

        # Update manifest
        print(f"\n📝 Updating manifest...")
        update_manifest(manifest_path, sha256_hash)

        # Print sample item
        print(f"\n📄 Sample Item:")
        sample_split = list(dataset.keys())[0]
        sample = dataset[sample_split][0]
        print(f"   Split: {sample_split}")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")

        print("\n" + "=" * 70)
        print("✓ MuSR retrieval complete!")
        print("=" * 70)
        print(f"\nDataset location: {data_dir}")
        print(f"Manifest: {manifest_path}")
        print(f"SHA-256: {sha256_hash}")

    except Exception as e:
        print(f"\n❌ ERROR during retrieval: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    retrieve_musr()
