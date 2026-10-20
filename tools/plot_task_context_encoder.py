"""Standalone scientific figure from a fixed causal-student assessment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    report_path = Path(report_path)
    r = json.loads(report_path.read_text())
    data = np.load(report_path.with_suffix(".npz"), allow_pickle=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = dict(full="#086A87", current="#D06B2C", short="#549747", teacher_oracle="#6D5DA8", train_mean="#888888")
    labels = dict(full="Full prefix", current="Current only", short="Last 16 frames", teacher_oracle="Whole-episode teacher (privileged)", train_mean="Constant train-mean context")
    x = np.asarray(r["evaluation_fractions"]) * 100
    for ax, metric, title in zip(axes[0], ["terminal", "procedure"], ["Terminal outcome", "Observed visual procedure"]):
        for arm in colors:
            values = data[f"{arm}__{metric}"]
            ax.plot(x, values.mean(0), label=labels[arm], color=colors[arm],
                linestyle="--" if arm in ("teacher_oracle", "train_mean") else "-", marker="o", markersize=3)
        ax.axvspan(25, 70, color="#086A87", alpha=.05)
        ax.set(xlabel="Assessment anchor (% of valid action range)", ylabel="Normalized-feature MSE", title=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    ax = axes[1, 0]
    for i, arm in enumerate(["full", "current", "short", "persistence"]):
        means = []
        for h in (16, 64):
            values = data[f"full__persistence_h{h}"] if arm == "persistence" else data[f"{arm}__future_h{h}"]
            means.append(np.nanmean(values, axis=1).mean())
        ax.bar(np.arange(2) + (i - 1.5) * .2, means, width=.18,
            color=colors.get(arm, "#999999"), label=labels.get(arm, "Persistence"))
    ax.set(xticks=[0, 1], xticklabels=["+16 frames (0.8 s)", "+64 frames (3.2 s)"],
        ylabel="Visual / geometric change MSE", title="Execution token: selected checkpoints")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax = axes[1, 1]
    arms = ["full", "same_task_history", "wrong_task_history", "reset_history"]
    values = [data[f"{a}__action_normalized"][:, r["middle_indices"]].mean(1) for a in arms]
    rng = np.random.default_rng(20261020)
    indices = rng.integers(0, len(values[0]), (10000, len(values[0])))
    means = np.asarray([v.mean() for v in values])
    ci = np.asarray([np.quantile(v[indices].mean(1), [.025, .975]) for v in values])
    ax.bar(np.arange(4), means, color=[colors["full"], "#549747", "#B54A46", "#999999"])
    ax.errorbar(np.arange(4), means, yerr=np.stack([means - ci[:, 0], ci[:, 1] - means]), fmt="none", color="#333333", capsize=4)
    ax.set(xticks=np.arange(4), xticklabels=["Correct", "Same task\ndonor", "Wrong task\ndonor", "Reset"],
        ylabel="Frozen-head normalized action MSE", title="History intervention, fixed current observation")
    ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"Causal task-context student: {r['split']} ({r['episodes']} episodes, one seed)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    plot(a.report, a.output)
