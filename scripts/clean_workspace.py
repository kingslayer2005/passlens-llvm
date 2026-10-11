#!/usr/bin/env python3
"""
clean_workspace.py — Free disk space without touching data or results.

Deletes only things that can be rebuilt:
  data/cache/      resume caches of Phases 3-6
  data/models/     model files written by older versions of the pipeline
  tmp/             scratch folder
  __pycache__/     Python bytecode
  *.log            run logs in the project root

It never deletes data/ir, data/features, data/labels, data/raw, the index
files, or anything in results/.

Usage:
  python -m scripts.clean_workspace           # show what would be deleted
  python -m scripts.clean_workspace --yes     # delete it
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import DATA_DIR, PROJECT_ROOT


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="really delete")
    args = parser.parse_args()

    targets = [DATA_DIR / "cache", DATA_DIR / "models", PROJECT_ROOT / "tmp"]
    targets += [p for p in PROJECT_ROOT.rglob("__pycache__") if ".venv" not in p.parts]
    targets += list(PROJECT_ROOT.glob("*.log"))
    targets = [p for p in targets if p.exists()]

    free_before = shutil.disk_usage(str(PROJECT_ROOT)).free / 1e9
    total = 0.0
    for path in targets:
        megabytes = size_mb(path)
        total += megabytes
        print(f"{megabytes:9.1f} MB  {path.relative_to(PROJECT_ROOT)}")
        if args.yes:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    print(f"{total:9.1f} MB  total {'deleted' if args.yes else '(nothing deleted; add --yes)'}")
    if args.yes:
        free_after = shutil.disk_usage(str(PROJECT_ROOT)).free / 1e9
        print(f"Free space: {free_before:.2f} GB -> {free_after:.2f} GB")


if __name__ == "__main__":
    main()
