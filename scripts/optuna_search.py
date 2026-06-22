"""
Optuna hyperparameter search for the full ALIGN pipeline.

Searches across encoders, mixer, decision head, and assistant head.
Designed for Nvidia H100 90GB — large batch sizes, longer training
per trial. Reports per-category rankings at the end.

Stages controlled by --search-encoders, --search-decision, --search-assistant.
When a stage is enabled, its hyperparameters are sampled by Optuna.
When disabled, defaults are used and that stage is fixed.

Usage:
    # Full search (all 3 stages, 30 trials)
    python scripts/optuna_search.py --n-trials 30 --epochs 30

    # Decision head only (encoders + assistant head fixed)
    python scripts/optuna_search.py \\
        --search-decision --skip-encoder-training \\
        --encoder-checkpoint checkpoints/pretrain/run_3/best.pt

    # Encoders + decision (skip assistant head tuning)
    python scripts/optuna_search.py \\
        --search-encoders --search-decision \\
        --no-search-assistant

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
from collections import defaultdict
from pathlib import Path

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

# Suppress Optuna's noisy logging except warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# Search space definition
# ============================================================

def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    """Sample a hyperparameter configuration for one trial."""
    config = {}

    # ── Stage 1: Encoders + Mixer (pretrain) ──
    if args.search_encoders:
        config["mixer_dim"] = trial.suggest_categorical(
            "mixer_dim", [256, 512, 768, 1024]
        )
        config["num_mixer_blocks"] = trial.suggest_int(
            "num_mixer_blocks", 1, 4
        )
        config["mixer_nhead"] = trial.suggest_categorical(
            "mixer_nhead", [4, 8, 16]
        )
        config["temperature"] = trial.suggest_float(
            "temperature", 0.03, 0.3, log=True
        )
        config["epochs_encoder"] = trial.suggest_int(
            "epochs_encoder", 20, 60
        )
        config["epochs_mixer"] = trial.suggest_int(
            "epochs_mixer", 5, 20
        )
        config["lr_encoder"] = trial.suggest_float(
            "lr_encoder", 1e-5, 1e-3, log=True
        )
        config["lr_mixer"] = trial.suggest_float(
            "lr_mixer", 1e-5, 1e-3, log=True
        )

    # ── Stage 2: Decision (Future Prediction) Head ──
    if args.search_decision:
        config["decision_arch"] = trial.suggest_categorical(
            "decision_arch", ["mlp", "transformer"]
        )
        config["decision_K"] = trial.suggest_categorical("decision_K", [5, 10, 20])
        config["decision_noise_std"] = trial.suggest_float(
            "decision_noise_std", 0.005, 0.05, log=True
        )
        config["noise_schedule"] = trial.suggest_categorical(
            "noise_schedule", ["constant", "random_uniform", "curriculum"]
        )
        config["lr_decision"] = trial.suggest_float(
            "lr_decision", 1e-5, 1e-2, log=True
        )
        config["loss_decay"] = trial.suggest_float(
            "loss_decay", 0.5, 0.95
        )
        config["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 5)

        # MLP-specific
        config["mlp_hidden_dim"] = trial.suggest_categorical(
            "mlp_hidden_dim", [256, 512, 1024]
        )
        config["mlp_num_layers"] = trial.suggest_int("mlp_num_layers", 2, 5)

        # Transformer-specific
        config["transformer_layers"] = trial.suggest_int(
            "transformer_layers", 1, 4
        )
        config["transformer_d_model"] = trial.suggest_categorical(
            "transformer_d_model", [128, 256, 384, 512]
        )
        config["transformer_nhead"] = trial.suggest_categorical(
            "transformer_nhead", [2, 4, 8]
        )
        config["transformer_dropout"] = trial.suggest_float(
            "transformer_dropout", 0.0, 0.3
        )
        config["transformer_dim_feedforward"] = trial.suggest_categorical(
            "transformer_dim_feedforward", [512, 1024, 2048]
        )

    # ── Stage 3: Assistant Head ──
    if args.search_assistant:
        config["assistant_hidden_dim"] = trial.suggest_categorical(
            "assistant_hidden_dim", [128, 256, 512, 1024]
        )
        config["assistant_layers"] = trial.suggest_int(
            "assistant_layers", 1, 4
        )
        config["assistant_dropout"] = trial.suggest_float(
            "assistant_dropout", 0.0, 0.3
        )
        config["lr_assistant"] = trial.suggest_float(
            "lr_assistant", 1e-5, 1e-2, log=True
        )

    # ── Training-wide ──
    config["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    config["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )

    return config


# ============================================================
# Trial execution
# ============================================================

def run_trial(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Train + evaluate one trial. Returns the combined eval loss."""
    config = suggest_params(trial, args)
    trial_dir = Path(args.output_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save trial config immediately (in case of crash)
    with open(trial_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[Optuna trial {trial.number}] Starting...")
    print(f"  config: {json.dumps(config, indent=2)}")

    t0 = time.time()
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}

    # ── Stage 1: Encoder pretrain (if searched) ──
    if args.search_encoders and not args.skip_encoder_training:
        encoder_dir = trial_dir / "encoder"
        encoder_dir.mkdir(parents=True, exist_ok=True)
        pretrain_cmd = [
            sys.executable, "training/pretrain.py",
            "--data", args.data,
            "--output-dir", str(encoder_dir),
            "--epochs-encoder", str(config.get("epochs_encoder", 40)),
            "--epochs-mixer", str(config.get("epochs_mixer", 10)),
            "--max-steps-per-epoch", str(args.max_steps_per_epoch),
            "--batch-size", str(config["batch_size"]),
            "--bf16",
            "--temperature", str(config["temperature"]),
            "--mixer-dim", str(config["mixer_dim"]),
            "--num-mixer-blocks", str(config["num_mixer_blocks"]),
            "--mixer-nhead", str(config["mixer_nhead"]),
        ]
        print(f"  [stage 1] Pretraining encoders...")
        log = trial_dir / "pretrain.log"
        with open(log, "w") as f:
            result = subprocess.run(pretrain_cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        if result.returncode != 0:
            print(f"  [trial {trial.number}] Pretrain failed")
            raise optuna.exceptions.TrialPruned()
        # Find the resulting best.pt
        encoder_ckpt = find_pretrain_checkpoint(encoder_dir)
        if encoder_ckpt is None:
            print(f"  [trial {trial.number}] No pretrain checkpoint found")
            raise optuna.exceptions.TrialPruned()
    else:
        # Use the fixed encoder checkpoint
        encoder_ckpt = Path(args.encoder_checkpoint)
        if not encoder_ckpt.exists():
            raise RuntimeError(f"--encoder-checkpoint does not exist: {encoder_ckpt}")

    # ── Stage 2 + 3: Head training ──
    heads_dir = trial_dir / "heads"
    heads_dir.mkdir(parents=True, exist_ok=True)
    head_cmd = [
        sys.executable, "training/train_heads.py",
        "--data", args.data,
        "--pretrained", str(encoder_ckpt),
        "--output-dir", str(heads_dir),
        "--epochs-decision", str(args.epochs_decision),
        "--epochs-assistant", str(args.epochs_assistant),
        "--max-steps-per-epoch", str(args.max_steps_per_epoch),
        "--batch-size", str(config["batch_size"]),
        "--bf16",
        "--weight-decay", str(config["weight_decay"]),
    ]
    if args.search_decision:
        head_cmd += [
            "--decision-arch", config["decision_arch"],
            "--chunk-size", str(config["decision_K"]),  # decision_K = chunk_size
            "--decision-noise-std", str(config["decision_noise_std"]),
            "--lr-decision", str(config["lr_decision"]),
            "--loss-decay", str(config["loss_decay"]),
            "--warmup-epochs", str(config["warmup_epochs"]),
        ]
        if config["decision_arch"] == "mlp":
            head_cmd += ["--mlp-hidden", str(config["mlp_hidden_dim"])]
            head_cmd += ["--mlp-layers", str(config["mlp_num_layers"])]
        else:
            head_cmd += ["--transformer-layers", str(config["transformer_layers"])]
            head_cmd += ["--transformer-d-model", str(config["transformer_d_model"])]
            head_cmd += ["--transformer-nhead", str(config["transformer_nhead"])]
            head_cmd += ["--transformer-dropout", str(config["transformer_dropout"])]
            head_cmd += ["--transformer-dim-ff", str(config["transformer_dim_feedforward"])]
    else:
        # Use defaults from train_heads.py
        head_cmd += ["--decision-arch", "mlp"]

    if args.search_assistant:
        head_cmd += [
            "--lr-assistant", str(config["lr_assistant"]),
            "--assistant-hidden", str(config["assistant_hidden_dim"]),
            "--assistant-layers", str(config["assistant_layers"]),
            "--assistant-dropout", str(config["assistant_dropout"]),
        ]

    print(f"  [stage 2+3] Training heads...")
    log = trial_dir / "heads.log"
    with open(log, "w") as f:
        result = subprocess.run(head_cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        print(f"  [trial {trial.number}] Head training failed")
        raise optuna.exceptions.TrialPruned()

    # ── Pruning check ──
    last_loss = parse_last_decision_loss(log)
    if last_loss is not None:
        trial.report(last_loss, args.epochs_decision)
        if trial.should_prune():
            print(f"  [trial {trial.number}] Pruned")
            raise optuna.exceptions.TrialPruned()

    # ── Evaluate ──
    heads_ckpt = find_heads_checkpoint(heads_dir)
    if heads_ckpt is None:
        raise optuna.exceptions.TrialPruned()

    eval_cmd = [
        sys.executable, "eval/eval_heads.py",
        "--data", args.data,
        "--checkpoint", str(heads_ckpt),
        "--encoder-checkpoint", str(encoder_ckpt),
        "--batch-size", str(config["batch_size"]),
        "--chunk-size", str(config["decision_K"] if args.search_decision else 5),
    ]
    if args.search_decision:
        eval_cmd += [
            "--decision-arch", config["decision_arch"],
            "--mlp-hidden", str(config["mlp_hidden_dim"]),
            "--mlp-layers", str(config["mlp_num_layers"]),
            "--transformer-layers", str(config["transformer_layers"]),
            "--transformer-d-model", str(config["transformer_d_model"]),
            "--transformer-nhead", str(config["transformer_nhead"]),
            "--transformer-dropout", str(config["transformer_dropout"]),
            "--transformer-dim-ff", str(config["transformer_dim_feedforward"]),
        ]
    if args.search_assistant:
        eval_cmd += [
            "--assistant-hidden", str(config["assistant_hidden_dim"]),
            "--assistant-layers", str(config["assistant_layers"]),
            "--assistant-dropout", str(config["assistant_dropout"]),
        ]
    print(f"  [eval] Evaluating...")
    eval_log = trial_dir / "eval.log"
    eval_result = subprocess.run(
        eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    eval_log.write_text(eval_result.stdout)
    if eval_result.returncode != 0:
        print(f"  [trial {trial.number}] Eval failed")
        raise optuna.exceptions.TrialPruned()

    # Combined metric: weighted sum of future-prediction loss and assistant RMSE
    fp_loss = parse_metric(eval_result.stdout, "future-prediction loss")
    asst_rmse = parse_metric(eval_result.stdout, "Δ RMSE")
    if fp_loss is None:
        raise optuna.exceptions.TrialPruned()

    # Minimize a weighted combination (fp_loss is bounded [0,2], asst_rmse in [0,~0.05])
    # Normalize asst_rmse to a similar scale by multiplying by 40
    combined = fp_loss + 40 * (asst_rmse or 0.0)

    elapsed = (time.time() - t0) / 60
    print(f"  [trial {trial.number}] fp_loss={fp_loss:.4f}  "
          f"asst_rmse={asst_rmse or 0:.4f}  combined={combined:.4f}  "
          f"time={elapsed:.1f}min")
    return combined


# ============================================================
# Parsing helpers
# ============================================================

def parse_metric(stdout: str, label: str) -> float | None:
    """Extract a metric value from eval_heads output by label."""
    for line in stdout.splitlines():
        if label.lower() in line.lower():
            try:
                # e.g. "Decision head  future-prediction loss:  0.9888  (...)"
                return float(line.split(":")[1].split()[0])
            except (ValueError, IndexError):
                pass
    return None


def parse_last_decision_loss(log_path: Path) -> float | None:
    """Extract the last decision-stage loss from a training log file."""
    if not log_path.exists():
        return None
    text = log_path.read_text()
    losses = []
    for line in text.splitlines():
        if "decision" in line.lower() and "loss=" in line.lower():
            try:
                val = line.split("loss=")[1].split()[0].rstrip(",")
                losses.append(float(val))
            except (ValueError, IndexError):
                pass
    return losses[-1] if losses else None


def find_pretrain_checkpoint(pretrain_dir: Path) -> Path | None:
    """Locate the encoder best.pt from a pretrain run."""
    candidates = list(pretrain_dir.rglob("best.pt"))
    if candidates:
        return candidates[0]
    candidates = list(pretrain_dir.rglob("encoder_best.pt"))
    if candidates:
        return candidates[0]
    return None


def find_heads_checkpoint(trial_dir: Path) -> Path | None:
    """Locate heads_best.pt within the trial's output directory."""
    candidates = list(trial_dir.rglob("heads_best.pt"))
    if candidates:
        return candidates[0]
    std = trial_dir / "heads_best.pt"
    return std if std.exists() else None


# ============================================================
# Per-category ranking report
# ============================================================

def print_category_rankings(study: optuna.Study) -> None:
    """Print best-performing value of each categorical/continuous hyperparameter."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials — no rankings to report.")
        return

    print("\n" + "=" * 72)
    print("PER-CATEGORY RANKINGS (lower combined_loss is better)")
    print("=" * 72)

    # Categorical hyperparameters (with optional group-by arch)
    categorical_keys = [
        # Decision head
        ("decision_arch", None),
        ("decision_K", None),
        ("noise_schedule", None),
        ("mlp_hidden_dim", "mlp"),
        ("mlp_num_layers", "mlp"),
        ("transformer_layers", "transformer"),
        ("transformer_d_model", "transformer"),
        ("transformer_nhead", "transformer"),
        ("transformer_dim_feedforward", "transformer"),
        # Assistant head
        ("assistant_hidden_dim", None),
        ("assistant_layers", None),
        # Encoders/mixer
        ("mixer_dim", None),
        ("num_mixer_blocks", None),
        ("mixer_nhead", None),
        # Training
        ("batch_size", None),
    ]

    for key, group_arch in categorical_keys:
        if group_arch is not None:
            # Both decision and assistant use a single "arch" key in Optuna
            arch_key = "decision_arch"
            trials = [t for t in completed if t.params.get(arch_key) == group_arch]
        else:
            trials = completed
        if not trials or key not in trials[0].params:
            continue

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

    # Continuous params: best 5 of each
    print("\n  Top 5 values of each continuous parameter (best 5):")
    continuous_keys = [
        "decision_noise_std", "lr_decision", "transformer_dropout",
        "loss_decay", "lr_assistant", "assistant_dropout",
        "temperature", "mixer_dim", "weight_decay",
    ]
    for key in continuous_keys:
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
    parser = argparse.ArgumentParser(
        description="Optuna search for the full ALIGN pipeline"
    )
    # ── Required ──
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s) — pass multiple to train on the concatenation")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Phase 1 backbone (used when --search-encoders is OFF)")

    # ── Search control ──
    parser.add_argument("--search-encoders", action="store_true",
                        help="Search encoder + mixer hyperparameters (slow)")
    parser.add_argument("--search-decision", action="store_true",
                        help="Search Decision head hyperparameters")
    parser.add_argument("--search-assistant", action="store_true",
                        help="Search Assistant head hyperparameters")
    parser.add_argument("--skip-encoder-training", action="store_true",
                        help="When --search-encoders is set, skip actual pretraining "
                             "and just sample encoder params. (For testing only.)")

    # ── Study management ──
    parser.add_argument("--output-dir", default="./optuna_trials")
    parser.add_argument("--study-name", default="align_full")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL (e.g. sqlite:///foo.db)")
    parser.add_argument("--resume", default=None,
                        help="Resume from existing study by storage path")

    # ── Training budget ──
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs-decision", type=int, default=30)
    parser.add_argument("--epochs-assistant", type=int, default=50)
    parser.add_argument("--max-steps-per-epoch", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)

    # ── Sanity check: at least one stage enabled ──
    args = parser.parse_args()
    if not (args.search_encoders or args.search_decision or args.search_assistant):
        parser.error("At least one of --search-encoders, --search-decision, "
                     "--search-assistant must be set.")

    # ── Resolve study ──
    if args.resume:
        storage = args.resume if args.resume.startswith("sqlite") else f"sqlite:///{args.resume}"
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        print(f"Resumed study '{args.study_name}' from {storage}")
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        print(f"  Already completed: {n_complete}, Pruned: {n_pruned}")
    else:
        storage = args.storage or f"sqlite:///{args.output_dir}/study.db"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="minimize",
            sampler=TPESampler(seed=42, multivariate=True),
            pruner=SuccessiveHalvingPruner(
                min_resource=5,
                reduction_factor=3,
                min_early_stopping_rate=0,
            ),
        )

    # ── Banner ──
    print(f"Starting Optuna search:")
    print(f"  n_trials={args.n_trials}")
    print(f"  search: encoders={args.search_encoders}  "
          f"decision={args.search_decision}  assistant={args.search_assistant}")
    print(f"  output_dir={args.output_dir}, storage={storage}")

    # ── Optimize ──
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
    print(f"  combined loss: {study.best_trial.value:.4f}")
    print(f"  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k} = {v}")

    print_category_rankings(study)


if __name__ == "__main__":
    main()
