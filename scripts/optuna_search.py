"""
Optuna hyperparameter search for ALIGN decision head.

Runs on Nvidia H100 90GB — uses large batch sizes and longer training
per trial since we have abundant GPU memory. Reports per-category
rankings at the end.

Usage:
    # Quick smoke test (5 trials, 10 epochs each)
    python scripts/optuna_search.py --n-trials 5 --epochs-decision 10

    # Full search (30 trials, 30 epochs each)
    python scripts/optuna_search.py --n-trials 30 --epochs-decision 30

    # Resume a previous study
    python scripts/optuna_search.py --resume optuna_studies/align_h100.db
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Suppress Optuna's noisy logging except warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# Search space definition
# ============================================================

def suggest_params(trial: optuna.Trial) -> dict:
    """Sample a hyperparameter configuration for one trial."""
    config = {
        # ── Architecture choice ──
        "decision_arch": trial.suggest_categorical(
            "decision_arch", ["mlp", "transformer"]
        ),
        # ── Window size ──
        "decision_K": trial.suggest_categorical("decision_K", [5, 10, 20]),

        # ── Training noise ──
        "decision_noise_std": trial.suggest_float(
            "decision_noise_std", 0.005, 0.05, log=True
        ),
        # Noise schedule: 'constant' = fixed sigma; 'random_uniform' = random per batch
        "noise_schedule": trial.suggest_categorical(
            "noise_schedule", ["constant", "random_uniform", "curriculum"]
        ),

        # ── Learning rate ──
        "lr_decision": trial.suggest_float(
            "lr_decision", 1e-5, 1e-2, log=True
        ),

        # ── Batch size (large for H100) ──
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256]
        ),

        # ── MLP-specific params (only used if decision_arch == 'mlp') ──
        "mlp_hidden_dim": trial.suggest_categorical(
            "mlp_hidden_dim", [256, 512, 1024]
        ),
        "mlp_num_layers": trial.suggest_int("mlp_num_layers", 2, 5),

        # ── Transformer-specific params ──
        "transformer_layers": trial.suggest_int(
            "transformer_layers", 1, 4
        ),
        "transformer_d_model": trial.suggest_categorical(
            "transformer_d_model", [128, 256, 384, 512]
        ),
        "transformer_nhead": trial.suggest_categorical(
            "transformer_nhead", [2, 4, 8]
        ),
        "transformer_dropout": trial.suggest_float(
            "transformer_dropout", 0.0, 0.3
        ),
        "transformer_dim_feedforward": trial.suggest_categorical(
            "transformer_dim_feedforward", [512, 1024, 2048]
        ),

        # ── Loss params ──
        "loss_decay": trial.suggest_float(
            "loss_decay", 0.5, 0.95
        ),
        # Warmup epochs for LR schedule
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 5),
    }
    return config


# ============================================================
# Trial execution
# ============================================================

def run_trial(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Train + evaluate one trial. Returns the future-prediction loss."""
    config = suggest_params(trial)
    trial_dir = Path(args.output_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save trial config immediately (in case of crash)
    with open(trial_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Train ──
    train_cmd = [
        sys.executable, "training/train_heads.py",
        "--data", args.data,
        "--encoder-checkpoint", args.encoder_checkpoint,
        "--output-dir", str(trial_dir),
        "--epochs-decision", str(args.epochs_decision),
        "--epochs-assistant", str(args.epochs_assistant),
        "--max-steps-per-epoch", str(args.max_steps_per_epoch),
        "--batch-size", str(config["batch_size"]),
        "--bf16",
        "--decision-arch", config["decision_arch"],
        "--decision-K", str(config["decision_K"]),
        "--decision-noise-std", str(config["decision_noise_std"]),
        "--lr-decision", str(config["lr_decision"]),
    ]

    # MLP-specific
    if config["decision_arch"] == "mlp":
        train_cmd += ["--mlp-hidden", str(config["mlp_hidden_dim"])]
        train_cmd += ["--mlp-layers", str(config["mlp_num_layers"])]

    # Transformer-specific
    if config["decision_arch"] == "transformer":
        train_cmd += ["--transformer-layers", str(config["transformer_layers"])]
        train_cmd += ["--transformer-d-model", str(config["transformer_d_model"])]
        train_cmd += ["--transformer-nhead", str(config["transformer_nhead"])]
        train_cmd += ["--transformer-dropout", str(config["transformer_dropout"])]
        train_cmd += ["--transformer-dim-ff", str(config["transformer_dim_feedforward"])]

    print(f"\n[Optuna trial {trial.number}] Starting training...")
    print(f"  config: {json.dumps(config, indent=2)}")
    train_log = trial_dir / "train.log"

    t0 = time.time()
    with open(train_log, "w") as f:
        result = subprocess.run(
            train_cmd,
            stdout=f, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
    train_time = time.time() - t0

    if result.returncode != 0:
        print(f"  [trial {trial.number}] Training failed with code {result.returncode}")
        raise optuna.exceptions.TrialPruned()

    # ── Pruning check: read training log, report intermediate loss ──
    last_loss = parse_last_decision_loss(train_log)
    if last_loss is not None:
        trial.report(last_loss, args.epochs_decision)
        if trial.should_prune():
            print(f"  [trial {trial.number}] Pruned at epoch {args.epochs_decision}")
            raise optuna.exceptions.TrialPruned()

    # ── Evaluate ──
    heads_ckpt = find_heads_checkpoint(trial_dir)
    if heads_ckpt is None:
        print(f"  [trial {trial.number}] No heads_best.pt found")
        raise optuna.exceptions.TrialPruned()

    eval_cmd = [
        sys.executable, "eval/eval_heads.py",
        "--data", args.data,
        "--checkpoint", str(heads_ckpt),
        "--encoder-checkpoint", args.encoder_checkpoint,
        "--batch-size", str(config["batch_size"]),
    ]
    print(f"  [trial {trial.number}] Evaluating...")
    eval_log = trial_dir / "eval.log"
    eval_result = subprocess.run(
        eval_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    eval_log.write_text(eval_result.stdout)

    if eval_result.returncode != 0:
        print(f"  [trial {trial.number}] Eval failed")
        raise optuna.exceptions.TrialPruned()

    eval_loss = parse_eval_loss(eval_result.stdout)
    if eval_loss is None:
        print(f"  [trial {trial.number}] Could not parse eval loss")
        raise optuna.exceptions.TrialPruned()

    print(f"  [trial {trial.number}] eval_loss={eval_loss:.4f}  "
          f"train={train_time/60:.1f}min  total={(time.time()-t0)/60:.1f}min")
    return eval_loss


# ============================================================
# Parsing helpers
# ============================================================

def parse_last_decision_loss(log_path: Path) -> float | None:
    """Extract the last decision-stage loss from a training log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    # Look for the most recent 'loss=' or 'Decision' line
    losses = []
    for line in text.splitlines():
        if "decision" in line.lower() and "loss=" in line:
            try:
                # Common formats: "loss=0.4123" or "loss: 0.4123"
                val = line.split("loss=")[1].split()[0].rstrip(",")
                losses.append(float(val))
            except (ValueError, IndexError):
                pass
    return losses[-1] if losses else None


def parse_eval_loss(stdout: str) -> float | None:
    """Extract the future-prediction loss from eval_heads output."""
    for line in stdout.splitlines():
        if "future-prediction loss" in line.lower():
            try:
                # "Decision head  future-prediction loss:  0.9888  (cosine, [0, 2])"
                return float(line.split(":")[1].split()[0])
            except (ValueError, IndexError):
                pass
    return None


def find_heads_checkpoint(trial_dir: Path) -> Path | None:
    """Locate heads_best.pt within the trial's output directory."""
    candidates = list(trial_dir.rglob("heads_best.pt"))
    if not candidates:
        # Try the standard location
        std = trial_dir / "heads_best.pt"
        if std.exists():
            return std
    return candidates[0] if candidates else None


# ============================================================
# Per-category ranking report
# ============================================================

def print_category_rankings(study: optuna.Study) -> None:
    """Print best-performing value of each categorical hyperparameter."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials — no rankings to report.")
        return

    print("\n" + "=" * 72)
    print("PER-CATEGORY RANKINGS (lower eval_loss is better)")
    print("=" * 72)

    # Identify the search-space keys from completed trials
    sample_params = completed[0].params
    # Buckets we want to rank
    categorical_keys = [
        ("decision_arch", None),
        ("decision_K", None),
        ("noise_schedule", None),
        ("mlp_hidden_dim", "mlp"),
        ("mlp_num_layers", "mlp"),
        ("transformer_layers", "transformer"),
        ("transformer_d_model", "transformer"),
        ("transformer_nhead", "transformer"),
        ("transformer_dim_feedforward", "transformer"),
        ("batch_size", None),
    ]

    for key, group_arch in categorical_keys:
        # Filter trials by arch if needed
        if group_arch is not None:
            trials = [t for t in completed if t.params.get("decision_arch") == group_arch]
        else:
            trials = completed

        if not trials or key not in trials[0].params:
            continue

        # Aggregate: mean eval_loss per value of this key
        from collections import defaultdict
        buckets = defaultdict(list)
        for t in trials:
            v = t.params[key]
            buckets[v].append(t.value)

        ranked = sorted(buckets.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))

        label = f"{key} (group={group_arch})" if group_arch else key
        print(f"\n  {label}:")
        print(f"    {'rank':<6}{'value':<25}{'mean_loss':<12}{'n_trials':<10}")
        print(f"    {'-'*53}")
        for i, (val, losses) in enumerate(ranked, 1):
            mean_loss = sum(losses) / len(losses)
            print(f"    {i:<6}{str(val):<25}{mean_loss:<12.4f}{len(losses):<10}")

    # Also rank a few continuous params
    print("\n  Top 5 continuous params (for context):")
    for key in ["decision_noise_std", "lr_decision", "transformer_dropout", "loss_decay"]:
        trials_with_key = [t for t in completed if key in t.params]
        if not trials_with_key:
            continue
        trials_sorted = sorted(trials_with_key, key=lambda t: t.value)[:5]
        print(f"\n    {key} (best 5):")
        for t in trials_sorted:
            print(f"      loss={t.value:.4f}  {key}={t.params[key]:.4g}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna search for ALIGN decision head")
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--encoder-checkpoint", required=True, help="Phase 1 backbone")
    parser.add_argument("--output-dir", default="./optuna_trials",
                        help="Where to put per-trial checkpoints")
    parser.add_argument("--study-name", default="align_decision_head",
                        help="Optuna study name (for resume)")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL (e.g. sqlite:///foo.db) — required for resume")
    parser.add_argument("--resume", default=None,
                        help="Resume from existing study by storage path")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs-decision", type=int, default=30)
    parser.add_argument("--epochs-assistant", type=int, default=50)
    parser.add_argument("--max-steps-per-epoch", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel trials (requires multiple GPUs)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Total timeout in seconds")
    args = parser.parse_args()

    # Resolve study
    if args.resume:
        storage = args.resume if args.resume.startswith("sqlite") else f"sqlite:///{args.resume}"
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        print(f"Resumed study '{args.study_name}' from {storage}")
        print(f"  Already completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    else:
        storage = args.storage or f"sqlite:///{args.output_dir}/study.db"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="minimize",
            sampler=TPESampler(seed=42, multivariate=True),
            pruner=SuccessiveHalvingPruner(
                min_resource=5,    # min epochs before pruning
                reduction_factor=3,
                min_early_stopping_rate=0,
            ),
        )

    # Run
    print(f"Starting Optuna search:")
    print(f"  n_trials={args.n_trials}, epochs={args.epochs_decision}")
    print(f"  output_dir={args.output_dir}, storage={storage}")
    print(f"  GPU: 1 H100 (assumed)")

    study.optimize(
        lambda t: run_trial(t, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    # ── Final report ──
    print("\n" + "=" * 72)
    print("OPTUNA STUDY COMPLETE")
    print("=" * 72)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  eval_loss: {study.best_trial.value:.4f}")
    print(f"  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k} = {v}")

    print_category_rankings(study)


if __name__ == "__main__":
    main()
