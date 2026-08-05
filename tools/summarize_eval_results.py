#!/usr/bin/env python3
"""Summarize ALIGN V4 mujoco_eval results across runs.

For each (run, suite) directory, gather:
  - config.json (training hyperparameters)
  - every intention_best*.mujoco_eval_*.json file (per-switch-at results)

Output a markdown table with:
  - run | suite | params (differing) | per-switch-at success rate + EEF error

Run from anywhere; resolves paths relative to repo root.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/home/ucluser/ALIGN")


def safe_load_json(p: Path):
    """Load JSON; return None on failure."""
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] failed to load {p}: {e}", file=sys.stderr)
        return None


def label_sort_key(s):
    """Sort key for switch-at labels: numeric first (by float), 'default' last."""
    if s == "default":
        return (1, 0, s)
    if "_bak" in s:
        try:
            return (0, float(s.replace("_bak", "")), s)
        except ValueError:
            return (1, 0, s)
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0, s)


def parse_switch_at(filename: str) -> str:
    """Extract the switch-at value from a result filename.

    Handles all observed patterns:
      - intention_best.mujoco_eval_0.2.json          -> 0.2
      - intention_best_fixed.mujoco_eval_0.5.json   -> 0.5
      - intention_best.mujoco_eval_0.6_bak.json     -> 0.6_bak
      - intention_best.mujoco_eval.json             -> default

    Strategy: split on '_eval_' and take everything after the LAST one
    (then strip .json, .bak, etc.).
    """
    name = filename
    if name.endswith(".json"):
        name = name[:-5]
    # Take the substring after the last '_eval_'
    idx = name.rfind("_eval_")
    if idx < 0:
        return "default"
    suffix = name[idx + len("_eval_"):]
    return suffix


def discover_eval_files(run_dir: Path):
    """Find all eval JSON files in a run dir; return {switch_at_label: path}."""
    found = {}
    for p in sorted(run_dir.glob("intention_best*.mujoco_eval*.json")):
        if p.name.endswith(".bak.json"):
            continue  # skip backup files
        label = parse_switch_at(p.name)
        found[label] = p
    return found


def load_config(run_dir: Path) -> dict:
    """Load training config; return empty dict if missing."""
    cfg_path = run_dir / "config.json"
    data = safe_load_json(cfg_path)
    return data if isinstance(data, dict) else {}


def summarize_eval(eval_data: dict) -> dict:
    """Compute aggregate metrics from an eval JSON."""
    if not isinstance(eval_data, dict):
        return {}
    episodes = eval_data.get("episodes", [])
    n = len(episodes)
    if n == 0:
        return {"n": 0}
    successes = sum(1 for e in episodes if e.get("success"))
    errs_model = [e.get("mean_error_model", 0.0) for e in episodes]
    errs_replay = [e.get("mean_error_replay", 0.0) for e in episodes]
    n_steps = [e.get("n_steps", 0) for e in episodes]
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n,
        "mean_err_model": sum(errs_model) / n if n else 0.0,
        "mean_err_replay": sum(errs_replay) / n if n else 0.0,
        "mean_n_steps": sum(n_steps) / n if n else 0.0,
        "switch_at_from_data": eval_data.get("switch_at"),
        "alpha_from_data": eval_data.get("alpha"),
        "n_episodes_configured": eval_data.get("n_episodes"),
    }


def diff_configs(ref_cfg: dict, other_cfg: dict) -> list:
    """Return list of keys whose values differ between two configs (excluding noise)."""
    keys = set(ref_cfg) | set(other_cfg)
    noise_keys = {
        "output_dir", "run_name", "val_episodes_file", "seed",
        "experiment_name", "notes",
    }
    diffs = []
    for k in sorted(keys):
        if k in noise_keys:
            continue
        if k not in ref_cfg or k not in other_cfg:
            diffs.append((k, ref_cfg.get(k), other_cfg.get(k)))
            continue
        if ref_cfg[k] != other_cfg[k]:
            diffs.append((k, ref_cfg[k], other_cfg[k]))
    return diffs


def shorten_diff_list(diffs: list, max_items: int = 6) -> str:
    """Render the diff between configs as a compact string."""
    if not diffs:
        return "(identical to first run)"
    items = []
    for k, v_ref, v_other in diffs[:max_items]:
        items.append(f"{k}={v_other}")
    s = "; ".join(items)
    if len(diffs) > max_items:
        s += f"; ... (+{len(diffs) - max_items} more)"
    return s


def collect_all_runs():
    """Walk the checkpoints tree and collect all run data."""
    runs = []
    ckpt_root = REPO_ROOT / "checkpoints" / "v4"
    for suite_dir in sorted(ckpt_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        for run_dir in sorted(suite_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            cfg = load_config(run_dir)
            evals = discover_eval_files(run_dir)
            eval_summaries = {}
            for label, p in evals.items():
                data = safe_load_json(p)
                eval_summaries[label] = summarize_eval(data or {})
            runs.append({
                "suite": suite,
                "run": run_dir.name,
                "config": cfg,
                "evals": eval_summaries,
                "eval_paths": evals,
            })
    return runs


def render_table_libero_spatial(runs: list) -> str:
    """Render the libero_spatial table."""
    spatial_runs = [r for r in runs if r["suite"] == "libero_spatial"]
    if not spatial_runs:
        return "_(no libero_spatial runs found)_\n"

    ref_cfg = spatial_runs[0]["config"]

    # Collect all unique switch-at labels across runs
    all_switch_labels = set()
    for r in spatial_runs:
        all_switch_labels.update(r["evals"].keys())
    # Sort numeric switch-at labels by float value; non-numeric last.
    all_switch_labels = sorted(all_switch_labels, key=label_sort_key)

    # Header
    lines = []
    lines.append("## Table 1: libero_spatial results — all switch-at ratios")
    lines.append("")
    lines.append("Each column is one `--switch-at` value. The model is the same"
                 " checkpoint; only the human-guidance ratio varies.")
    lines.append("`switch_at` is the fraction of the episode that runs under expert"
                 " control before switching to the model.")
    lines.append("")
    # Three-line header: run | params-diff | per-switch-at columns
    header = ["run", "params (vs first)"]
    for s in all_switch_labels:
        header.append(f"sw={s} acc")
        header.append(f"sw={s} err")
    # Markdown table
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in spatial_runs:
        diffs = diff_configs(ref_cfg, r["config"])
        diff_str = shorten_diff_list(diffs, max_items=4)
        row = [r["run"], diff_str]
        for s in all_switch_labels:
            es = r["evals"].get(s)
            if es and es.get("n", 0) > 0:
                row.append(f"{es['success_rate']*100:.0f}% ({es['successes']}/{es['n']})")
                row.append(f"{es['mean_err_model']:.3f}")
            else:
                row.append("-")
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_table_libero_goal(runs: list) -> str:
    """Render the libero_goal table."""
    goal_runs = [r for r in runs if r["suite"] == "libero_goal"]
    if not goal_runs:
        return "_(no libero_goal runs found)_\n"

    ref_cfg = goal_runs[0]["config"]

    all_switch_labels = set()
    for r in goal_runs:
        all_switch_labels.update(r["evals"].keys())
    all_switch_labels = sorted(all_switch_labels, key=label_sort_key)

    lines = []
    lines.append("## Table 2: libero_goal results — all switch-at ratios")
    lines.append("")
    lines.append("| " + " | ".join(["run", "params (vs first)"] +
                                     [f"sw={s} acc" for s in all_switch_labels] +
                                     [f"sw={s} err" for s in all_switch_labels]) + " |")
    lines.append("|" + "|".join(["---"] * (2 + 2 * len(all_switch_labels))) + "|")

    for r in goal_runs:
        diffs = diff_configs(ref_cfg, r["config"])
        diff_str = shorten_diff_list(diffs, max_items=4)
        row = [r["run"], diff_str]
        for s in all_switch_labels:
            es = r["evals"].get(s)
            if es and es.get("n", 0) > 0:
                row.append(f"{es['success_rate']*100:.0f}% ({es['successes']}/{es['n']})")
            else:
                row.append("-")
        for s in all_switch_labels:
            es = r["evals"].get(s)
            if es and es.get("n", 0) > 0:
                row.append(f"{es['mean_err_model']:.3f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_long_form_per_run(runs: list) -> str:
    """Render per-run, per-switch-at detail lines (one per eval file)."""
    lines = ["## Detailed per-eval breakdown", ""]
    for r in runs:
        lines.append(f"### {r['suite']}/{r['run']}")
        cfg = r["config"]
        # Compact config summary
        keys_to_show = [
            "use_intent_tokens", "use_memory_bank", "use_history",
            "use_text", "head_type", "history_size", "chunk_size",
            "anchor_weight", "compressed_dim", "num_intent_tokens",
            "memory_bank_len", "segment_min_mult", "segment_max_mult",
            "epochs", "batch_size", "lr",
        ]
        cfg_parts = []
        for k in keys_to_show:
            if k in cfg:
                cfg_parts.append(f"{k}={cfg[k]}")
        lines.append("Config: " + "; ".join(cfg_parts))
        lines.append("")
        # Eval table
        if not r["evals"]:
            lines.append("_(no eval results)_")
            lines.append("")
            continue
        lines.append("| switch_at | n_eps | success | success_rate | mean_err_model | mean_n_steps |")
        lines.append("|---|---|---|---|---|---|")
        for label in sorted(r["evals"].keys(), key=label_sort_key):
            es = r["evals"][label]
            if es.get("n", 0) > 0:
                lines.append(
                    f"| {label} | {es['n']} | {es['successes']}/{es['n']} | "
                    f"{es['success_rate']*100:.1f}% | {es['mean_err_model']:.4f} | "
                    f"{es['mean_n_steps']:.0f} |"
                )
            else:
                lines.append(f"| {label} | - | - | - | - | - |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    runs = collect_all_runs()
    if not runs:
        print("No runs found.")
        return

    print("=" * 80)
    print("ALIGN V4 Eval Summary")
    print("=" * 80)
    print()

    # Summary tables
    print(render_table_libero_spatial(runs))
    print()
    print(render_table_libero_goal(runs))
    print()

    # Detail
    print(render_long_form_per_run(runs))


if __name__ == "__main__":
    main()