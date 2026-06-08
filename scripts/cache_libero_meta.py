#!/usr/bin/env python3
"""Pre-download LIBERO metadata so lerobot's version check doesn't crash.

lerobot >= 1.0 requires a _version_ tag on HF Hub datasets.
nvidia/LIBERO_LeRobot_v3 doesn't have one (they never tagged it).
This script downloads metadata files to the exact cache path lerobot
expects, so LeRobotDatasetMetadata.load_metadata() succeeds and
get_safe_version() is never reached.

One-time setup. Run once per machine.
"""
import sys, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "nvidia/LIBERO_LeRobot_v3"
HOME = Path.home()
CACHE_DIR = Path(
    __import__("os").environ.get(
        "LEROBOT_CACHE_DIR",
        str(HOME / ".cache" / "huggingface" / "lerobot"),
    )
)
DEST = CACHE_DIR / "nvidia" / "LIBERO_LeRobot_v3" / "meta"

if (DEST / "info.json").exists() and (DEST / "tasks.parquet").exists():
    print(f"Metadata already cached at {DEST}")
    # Verify it's not corrupted
    import pandas as pd
    try:
        pd.read_parquet(str(DEST / "tasks.parquet"))
        print("Cache OK (parquet readable)")
        sys.exit(0)
    except Exception:
        print("Cache corrupted, re-downloading...")
        shutil.rmtree(DEST)

print(f"Downloading metadata to: {DEST}")
local_dir = snapshot_download(
    REPO_ID, repo_type="dataset", revision="main",
    allow_patterns="*/meta/*",
)

# LIBERO stores meta under sub-task dirs: libero_10/meta/, libero_90/meta/, etc.
# Use the FIRST one found to avoid schema mismatches.
for item in sorted(Path(local_dir).iterdir()):
    meta_src = item / "meta"
    if meta_src.is_dir() and (meta_src / "info.json").exists():
        print(f"  Using: {item.name}/meta/ ({len(list(meta_src.rglob('*')))} files)")
        for f in meta_src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(meta_src)
                dest = DEST / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        break

# Verify
import json
with open(DEST / "info.json") as f:
    info = json.load(f)
print(f"Cached {len(list(DEST.rglob('*')))} files, {info['total_episodes']} episodes")
print("Done. LIBERO metadata cache ready.")
