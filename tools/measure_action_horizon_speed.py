#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure and compare inference speed across --action-horizon settings.

Workflow:
    1. Run eval_libero_v3_trajectory.py for each horizon H of interest, with
       --save-timing results/timing_H.json, on the same model + dataset.
    2. Run this script with all those JSONs as inputs:
         python tools/measure_action_horizon_speed.py \\
             --timing results/timing_1.json \\
             --timing results/timing_2.json \\
             --timing results/timing_5.json \\
             --output results/horizon_speed/

Output:
    - results.json          : per-horizon aggregate metrics
    - horizon_comparison.csv : per-horizon table (easy to drop in paper)
    - horizon_comparison.pdf : bar chart of effective control rate vs horizon
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_timings(path: Path) -> List[Dict]:
    """Load a timing JSON. Expected format: list of {episode, timing: [...]},
    where each timing entry is {step, model_called, step_ms, inference_ms}.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list at top level in {path}; got {type(data)}")
    return data


def aggregate_one(data: List[Dict]) -> Dict:
    """Compute aggregate metrics for one timing file."""
    # Flatten all per-step timing records across episodes.
    step_ms_all: List[float] = []
    inference_ms_all: List[float] = []  # only the steps where model was called
    for entry in data:
        for rec in entry["timing"]:
            step_ms_all.append(rec["step_ms"])
            if rec["model_called"] and rec["inference_ms"] > 0:
                inference_ms_all.append(rec["inference_ms"])

    if not step_ms_all:
        raise ValueError("No timing records found")

    step_ms = np.array(step_ms_all)
    inference_ms = np.array(inference_ms_all) if inference_ms_all else np.array([0.0])

    n_steps = len(step_ms)
    n_model_calls = len(inference_ms)
    model_call_ratio = n_model_calls / n_steps  # should be 1/H for horizon H

    # Effective env-step rate (steps per second, end-to-end).
    mean_step_ms = float(step_ms.mean())
    p95_step_ms = float(np.percentile(step_ms, 95))
    effective_hz = 1000.0 / mean_step_ms if mean_step_ms > 0 else float("inf")

    # Amortized per-step inference cost (model time / env step).
    mean_inference_per_step_ms = (
        float(inference_ms.sum()) / n_steps if n_steps > 0 else 0.0
    )

    # Per-call inference cost (model forward+sample time).
    mean_inference_call_ms = float(inference_ms.mean()) if inference_ms.size else 0.0
    p95_inference_call_ms = (
        float(np.percentile(inference_ms, 95)) if inference_ms.size else 0.0
    )

    # Effective model-call rate.
    model_hz = 1000.0 / mean_inference_call_ms if mean_inference_call_ms > 0 else float("inf")

    return {
        "n_steps": int(n_steps),
        "n_model_calls": int(n_model_calls),
        "model_call_ratio": float(model_call_ratio),
        "effective_env_step_hz": float(effective_hz),
        "mean_env_step_ms": mean_step_ms,
        "p95_env_step_ms": p95_step_ms,
        "amortized_model_cost_per_step_ms": mean_inference_per_step_ms,
        "mean_model_call_ms": mean_inference_call_ms,
        "p95_model_call_ms": p95_inference_call_ms,
        "effective_model_call_hz": float(model_hz),
    }


def render_table(per_horizon: Dict[int, Dict], output_path: Path):
    """Write a CSV summarizing per-horizon metrics."""
    rows = ["horizon,n_steps,model_call_ratio,effective_env_step_hz,mean_env_step_ms,"
            "p95_env_step_ms,amortized_model_cost_per_step_ms,mean_model_call_ms,p95_model_call_ms,"
            "effective_model_call_hz"]
    for h, m in sorted(per_horizon.items()):
        rows.append(
            f"{h},"
            f"{m['n_steps']},"
            f"{m['model_call_ratio']:.4f},"
            f"{m['effective_env_step_hz']:.2f},"
            f"{m['mean_env_step_ms']:.2f},"
            f"{m['p95_env_step_ms']:.2f},"
            f"{m['amortized_model_cost_per_step_ms']:.2f},"
            f"{m['mean_model_call_ms']:.2f},"
            f"{m['p95_model_call_ms']:.2f},"
            f"{m['effective_model_call_hz']:.2f}"
        )
    output_path.write_text("\n".join(rows) + "\n")


def render_plot(per_horizon: Dict[int, Dict], output_path: Path):
    """Bar chart: effective env-step Hz vs horizon."""
    horizons = sorted(per_horizon.keys())
    env_hz = [per_horizon[h]["effective_env_step_hz"] for h in horizons]
    model_hz = [per_horizon[h]["effective_model_call_hz"] for h in horizons]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(horizons))
    width = 0.35
    bars1 = ax.bar(x - width / 2, env_hz, width, label="Env-step rate",
                   color="#0072B2")
    bars2 = ax.bar(x + width / 2, model_hz, width, label="Model-call rate",
                   color="#E69F00")

    ax.set_xlabel("Action horizon H (actions per model call)")
    ax.set_ylabel("Rate (Hz)")
    ax.set_title("Inference speed vs action horizon")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, h in zip(bars1, env_hz):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, h in zip(bars2, model_hz):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timing", action="append", required=True, type=Path,
                        help="Path to a timing JSON. Repeat for multiple horizons. "
                             "The horizon value is read from the filename: "
                             "expects pattern timing_<horizon>.json or "
                             "horizon_<horizon>_*.json.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Extract horizon from filename. Tries several patterns:
    #   timing_5.json       -> 5
    #   horizon_3_v2.json   -> 3
    #   results_horizon_1.json -> 1
    import re
    per_horizon: Dict[int, Dict] = {}
    for path in args.timing:
        m = re.search(r"(?:horizon[_-])(\d+)|timing_(\d+)", path.name)
        if not m:
            print(f"[warn] Could not extract horizon from {path.name}; skipping")
            continue
        horizon = int(m.group(1) or m.group(2))
        print(f"[load] horizon={horizon}  file={path}")
        data = load_timings(path)
        agg = aggregate_one(data)
        per_horizon[horizon] = agg
        print(f"  n_steps={agg['n_steps']}, n_model_calls={agg['n_model_calls']}, "
              f"call_ratio={agg['model_call_ratio']:.3f}, "
              f"effective_env_hz={agg['effective_env_step_hz']:.2f}, "
              f"model_call_ms={agg['mean_model_call_ms']:.2f}")

    if not per_horizon:
        print("[error] No valid timing files loaded.")
        sys.exit(1)

    # Save aggregate JSON
    out_json = args.output / "results.json"
    out_json.write_text(json.dumps(
        {str(h): m for h, m in sorted(per_horizon.items())}, indent=2
    ))
    print(f"\n[save] aggregate results -> {out_json}")

    # Save CSV table
    out_csv = args.output / "horizon_comparison.csv"
    render_table(per_horizon, out_csv)
    print(f"[save] CSV table         -> {out_csv}")

    # Save PDF plot
    out_pdf = args.output / "horizon_comparison.pdf"
    render_plot(per_horizon, out_pdf)
    print(f"[save] PDF plot          -> {out_pdf}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"{'H':<5} {'env_Hz':<10} {'model_Hz':<10} "
          f"{'call_ratio':<12} {'model_ms':<12}")
    for h, m in sorted(per_horizon.items()):
        print(f"{h:<5} {m['effective_env_step_hz']:<10.2f} "
              f"{m['effective_model_call_hz']:<10.2f} "
              f"{m['model_call_ratio']:<12.4f} "
              f"{m['mean_model_call_ms']:<12.2f}")


if __name__ == "__main__":
    main()
