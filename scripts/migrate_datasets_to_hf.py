"""Migrate local config-bundled dataset CSVs to HuggingFace Datasets.

For each dataset this script will:
  1. Create the HuggingFace dataset repo  MLLab-TS/<dataset_name>  (if it does
     not already exist).
  2. Upload the local  configs/<dataset_name>/dataset.csv  file to that repo.
  3. Rewrite  configs/<dataset_name>/config.yaml : replace the ``dataset_path``
     key with ``dataset_url`` pointing at the freshly-uploaded file.
  4. Delete the local CSV.

The oil dataset is used as the canonical reference for the resulting layout; it
is therefore skipped automatically (already migrated).

Usage
-----
Migrate a single dataset (recommended for spot-checking):

    python scripts/migrate_datasets_to_hf.py --dataset abalone

Migrate all remaining datasets one by one (runs sequentially):

    python scripts/migrate_datasets_to_hf.py --all

Options
-------
--dataset NAME     Name of a single dataset directory under configs/  to migrate.
--all              Iterate over every config subdir that still has a local CSV
                   and has not yet been migrated.
--org ORG          HuggingFace organisation name (default: MLLab-TS).
--dry-run          Print what would happen without uploading or modifying files.
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hf_resolve_url(org: str, dataset_name: str, filename: str = "dataset.csv") -> str:
    """Return the direct-download URL for a file in a HuggingFace dataset repo."""
    return f"https://huggingface.co/datasets/{org}/{dataset_name}/resolve/main/{filename}"


def find_pending_datasets() -> list[str]:
    """Return sorted list of dataset dir names that still have a local CSV."""
    pending = []
    for subdir in sorted(CONFIGS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "dataset.csv"
        config_path = subdir / "config.yaml"
        if csv_path.exists() and config_path.exists():
            pending.append(subdir.name)
    return pending


def config_has_dataset_path(config_path: pathlib.Path) -> bool:
    text = config_path.read_text()
    return bool(re.search(r"^\s*dataset_path\s*:", text, re.MULTILINE))


def rewrite_config(config_path: pathlib.Path, dataset_url: str) -> None:
    """Replace the dataset_path line with dataset_url in the YAML file.

    We do a targeted line-level replacement rather than round-tripping through
    a YAML parser so that comments and formatting are preserved.
    """
    text = config_path.read_text()
    new_text = re.sub(
        r"^(\s*)dataset_path\s*:.*$",
        rf"\1dataset_url: {dataset_url}",
        text,
        flags=re.MULTILINE,
    )
    config_path.write_text(new_text)


def migrate_one(dataset_name: str, org: str, dry_run: bool) -> None:
    """Upload CSV to HuggingFace, update config, and delete the local CSV."""
    csv_path = CONFIGS_DIR / dataset_name / "dataset.csv"
    config_path = CONFIGS_DIR / dataset_name / "config.yaml"

    if not csv_path.exists():
        print(f"[SKIP]  {dataset_name}: no local dataset.csv found")
        return

    if not config_path.exists():
        print(f"[SKIP]  {dataset_name}: no config.yaml found")
        return

    if not config_has_dataset_path(config_path):
        print(f"[SKIP]  {dataset_name}: config already uses dataset_url (already migrated)")
        return

    repo_id = f"{org}/{dataset_name}"
    dataset_url = hf_resolve_url(org, dataset_name)

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name}")
    print(f"  Repo    : {repo_id}")
    print(f"  URL     : {dataset_url}")
    print(f"  CSV     : {csv_path.relative_to(REPO_ROOT)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] Would create repo, upload CSV, rewrite config, delete CSV.")
        return

    from huggingface_hub import HfApi

    api = HfApi()

    # 1. Create repo if it doesn't exist
    print(f"  Creating repo {repo_id} (if not exists)…")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,  # no-op if already exists
        private=False,
    )

    # 2. Upload the CSV
    print(f"  Uploading dataset.csv…")
    api.upload_file(
        path_or_fileobj=str(csv_path),
        path_in_repo="dataset.csv",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Upload complete.")

    # 3. Rewrite config.yaml
    print(f"  Rewriting config.yaml…")
    rewrite_config(config_path, dataset_url)

    # 4. Delete local CSV
    print(f"  Deleting local CSV…")
    csv_path.unlink()

    print(f"  [DONE] {dataset_name} migrated successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate dataset CSVs from configs/ to HuggingFace Datasets."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        metavar="NAME",
        help="Name of a single dataset directory under configs/ to migrate.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Migrate all config subdirs that still have a local CSV.",
    )
    parser.add_argument(
        "--org",
        default="MLLab-TS",
        help="HuggingFace organisation name (default: MLLab-TS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes.",
    )
    args = parser.parse_args()

    # Lazy import so the script is importable even without huggingface_hub when
    # only --dry-run or --help is used.
    if not args.dry_run:
        try:
            from huggingface_hub import login
            login()
        except ImportError:
            print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
            sys.exit(1)

    if args.all:
        pending = find_pending_datasets()
        if not pending:
            print("No datasets with local CSVs found — nothing to migrate.")
            return
        print(f"Found {len(pending)} dataset(s) to migrate: {pending}")
        for name in pending:
            migrate_one(name, args.org, args.dry_run)
    else:
        migrate_one(args.dataset, args.org, args.dry_run)


if __name__ == "__main__":
    main()
