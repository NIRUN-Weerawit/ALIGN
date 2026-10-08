#!/usr/bin/env python3
"""Combine incremental results from the four LIBERO Goal evaluators."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    workers = json.loads((out / "workers.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    while True:
        records = {}
        active = False
        lines = ["# LIBERO Goal: all-task evaluation", "",
                 "One held-out episode per task, seed 42, 300 steps, expert-to-model switch at half the demonstration length.",
                 "EEF error averages only frames with a reference pose and includes the expert-controlled prefix.", "",
                 "| Variant | Completed tasks | Successes |", "|---|---:|---:|"]
        for name, worker in workers.items():
            path = out / name / "episode_results.json"
            episodes = json.loads(path.read_text())["episodes"] if path.exists() else []
            records[name] = episodes
            successes = sum(bool(ep["success"]) for ep in episodes)
            lines.append(f"| {name} | {len(episodes)}/10 | {successes}/{len(episodes)} |")
            state = subprocess.run(["systemctl", "--user", "show", worker["unit"],
                                    "--property=ActiveState", "--value"], capture_output=True, text=True).stdout.strip()
            active |= state in ("active", "activating")
        lines += ["", "| Task | " + " | ".join(workers) + " |",
                  "|---|" + "---|" * len(workers)]
        for task, metadata in protocol["tasks"].items():
            values = []
            for name in workers:
                episode = next((ep for ep in records[name] if ep["episode"] == metadata["episode"]), None)
                values.append(("Success" if episode["success"] else "Failed") if episode else "Pending")
            lines.append("| " + task + " | " + " | ".join(values) + " |")
        for filename, content in (("comparison.md", "\n".join(lines) + "\n"),
                                  ("summary.json", json.dumps(records, indent=2) + "\n")):
            temporary = out / (filename + ".tmp")
            temporary.write_text(content)
            temporary.replace(out / filename)
        if all(len(episodes) == 10 for episodes in records.values()):
            (out / "COMPLETE").write_text("All 40 task rollouts completed.\n")
            return
        if not active:
            (out / "INCOMPLETE").write_text("Evaluation workers exited before all tasks completed; inspect logs.\n")
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
