#!/usr/bin/env python3
"""Merge independent ablation workers' atomic summaries without writer races."""
import argparse
import json
from pathlib import Path
import time

from run_intention_ablation import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    manifest = json.loads((out / "manifest.json").read_text())
    records = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    while True:
        for name in manifest["variants"]:
            summary = out / name / "summary.json"
            if summary.exists():
                worker_records = json.loads(summary.read_text())
                if name in worker_records:
                    records[name] = worker_records[name]
        summarize(out, records, manifest)
        if all(records.get(name, {}).get("completed_epochs", 0) >= manifest["epochs"]
               and (out / name / "COMPLETE").exists() for name in manifest["variants"]):
            (out / "COMPLETE").write_text("All parallel variants completed.\n")
            print("All parallel variants completed", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
