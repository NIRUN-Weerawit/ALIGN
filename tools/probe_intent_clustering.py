#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe intent_emb structure on held-out episodes.

Implements EXP-D.5 (task-discriminative clustering) and EXP-C
(goal-directionality probe) from the paper draft.

The probe runs entirely on a frozen model checkpoint; no training is
performed. It collects intent_emb at every timestep on held-out
episodes, then runs sklearn classifiers to test whether the embedding
space has interpretable structure.

Usage:
    python tools/probe_intent_clustering.py \\
        --checkpoint checkpoints/v4/libero_spatial/run_27/intention_best.pt \\
        --data data/libero_spatial.h5 \\
        --output results/exp_d5/ \\
        --cameras image wrist_image \\
        --max-episodes 100

Outputs:
    - results.json     : quantitative metrics
    - intent_tsne.pdf  : t-SNE / UMAP visualization
    - confusion_matrix.pdf : classifier confusion matrix
    - intent_emb.npz   : raw embeddings + labels (for downstream analysis)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.align_intention import ALIGNIntentionModel  # noqa: E402
from data.align_dataset import ALIGNDataset  # noqa: E402


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """Load state_dict from a checkpoint, tolerant to minor mismatches."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # Strip any 'module.' prefix from DataParallel checkpoints
    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    return ckpt.get("args", None) if isinstance(ckpt, dict) else None


def _build_model_from_args(args_namespace, ckpt_args: dict) -> ALIGNIntentionModel:
    """Construct the model with the right flags for the checkpoint.

    Falls back to CLI flags / defaults if the checkpoint's args are missing.
    """
    cfg: dict = {
        "use_intent_tokens": True,
        "num_intent_tokens": 2,
        "intent_dim": 512,
        "use_memory_bank": False,
        "head_type": "diffusion",
        "mamba_output_dim": 512,
        "state_dim": 256,
        "chunk_size": 10,
        "history_size": 1,
        "num_cameras": len(args_namespace.cameras),
    }
    if ckpt_args:
        for key in cfg:
            if key in ckpt_args:
                cfg[key] = ckpt_args[key]
    # CLI overrides
    if args_namespace.use_memory_bank:
        cfg["use_memory_bank"] = True
    return ALIGNIntentionModel(**cfg)


def _format_batch_for_probe(sample: dict, device: torch.device) -> dict:
    """Convert a single sample from ALIGNDataset into a (B=1, T, ...) batch."""
    frames = sample["frames"]                  # (T_full, V, H, W, 3) or pre-computed
    state_full = sample.get("robot_state", None)  # 7-D, only last-step
    actions = sample.get("actions", None)      # (T_full, A)

    # The dataset's __getitem__ returns `count + traj_window` frames. We use
    # the whole window so the intent tokens see the full history.
    frames_t = torch.from_numpy(np.asarray(frames)).unsqueeze(0).to(device)

    # State: per-step robot state isn't in the dataset's __getitem__; we use
    # the last-step `robot_state` broadcast across the window as a degenerate
    # but usable approximation. (Real per-step state is only available through
    # poses / grippers; reconstructing it requires PoseReader.)
    # For the probe, the intent tokens depend primarily on vision + Mamba
    # recurrence, so constant state is acceptable for the qualitative probe.
    T_full = frames_t.shape[1]
    if state_full is not None:
        state_window = np.tile(np.asarray(state_full), (T_full, 1))
    else:
        # Fallback: zero state if absent
        state_window = np.zeros((T_full, 7), dtype=np.float32)
    state_t = torch.from_numpy(state_window).unsqueeze(0).to(device)

    return {
        "frames": frames_t,
        "state": state_t,
        "task_text": sample.get("text", ""),
        "actions": actions,
    }


# ---------------------------------------------------------------
# Collection
# ---------------------------------------------------------------
def collect_intent_embeddings(
    model: ALIGNIntentionModel,
    dataset: ALIGNDataset,
    device: torch.device,
    max_episodes: int = None,
    stride: int = 1,
) -> tuple:
    """Run the model on each episode and collect (intent_emb, label) pairs.

    Args:
        model: trained ALIGNIntentionModel.
        dataset: ALIGNDataset (probe mode).
        device: cuda / cpu.
        max_episodes: cap on number of episodes to process.
        stride: subsample every N-th timestep within each window (for memory).

    Returns:
        (embeddings, labels, episode_indices) where
        embeddings: np.ndarray of shape (N_total, intent_dim)
        labels:     list of task_text strings (length N_total)
        episode_indices: np.ndarray of shape (N_total,) mapping each row to its source episode.
    """
    model.eval()
    n = max_episodes or len(dataset)
    all_embeddings, all_labels, all_indices = [], [], []
    skipped_no_intent = 0

    for ep_idx in range(n):
        sample = dataset[ep_idx]
        batch = _format_batch_for_probe(sample, device)
        try:
            outputs = model.forward_with_probe(batch["frames"], batch["state"])
        except Exception as e:
            print(f"  [warn] Episode {ep_idx} failed: {e}")
            continue
        intent_emb = outputs.get("intent_emb", None)
        if intent_emb is None:
            skipped_no_intent += 1
            continue
        # intent_emb shape: (1, N, intent_dim). Mean over N for a single
        # per-window representation.
        emb = intent_emb.mean(dim=1).squeeze(0).detach().cpu().numpy()  # (intent_dim,)
        all_embeddings.append(emb)
        all_labels.append(batch["task_text"])
        all_indices.append(ep_idx)

    if skipped_no_intent:
        print(
            f"  [warn] Skipped {skipped_no_intent} episodes "
            f"(model has no intent tokens)"
        )
    if not all_embeddings:
        raise RuntimeError(
            "No intent embeddings collected. Check that the checkpoint "
            "was trained with --use-intent-tokens."
        )
    return (
        np.stack(all_embeddings, axis=0),
        all_labels,
        np.array(all_indices),
    )


# ---------------------------------------------------------------
# Analysis 1: task clustering (EXP-D.5)
# ---------------------------------------------------------------
def analyze_task_clustering(
    embeddings: np.ndarray,
    labels: list,
    output_dir: Path,
) -> dict:
    """Test whether intent_emb clusters by task label.

    Reports:
        - Silhouette score (higher = better cluster cohesion)
        - Cross-validated classification accuracy (logistic + KNN)
        - Per-class confusion matrix
        - t-SNE / PCA visualization
    """
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(labels)
    n_classes = len(label_enc.classes_)

    # Standardize before clustering / classification
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    # --- Silhouette score ---
    sil = silhouette_score(X, y)

    # --- Cross-validated classifiers ---
    clf_logistic = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)
    cv_logistic = cross_val_score(clf_logistic, X, y, cv=5)
    clf_knn = KNeighborsClassifier(n_neighbors=min(5, max(1, n_classes)))
    cv_knn = cross_val_score(clf_knn, X, y, cv=5)

    # --- Per-class breakdown ---
    preds_logistic = cross_val_predict(
        LogisticRegression(max_iter=2000, multi_class="multinomial"),
        X, y, cv=5,
    )
    cm = confusion_matrix(y, preds_logistic)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # --- Cluster purity ---
    km = KMeans(n_clusters=n_classes, n_init=10, random_state=0).fit(X)
    cluster_labels = km.labels_
    purity = 0.0
    for c in range(n_classes):
        members = [l for cl, l in zip(cluster_labels, labels) if cl == c]
        if not members:
            continue
        most_common = max(set(members), key=members.count)
        purity += members.count(most_common) / len(embeddings)

    # --- Visualization: t-SNE ---
    try:
        tsne = TSNE(
            n_components=2, perplexity=min(30, len(embeddings) - 1),
            random_state=0, init="pca",
        ).fit_transform(X)
    except Exception as e:
        print(f"  [warn] t-SNE failed ({e}); falling back to PCA.")
        tsne = PCA(n_components=2).fit_transform(X)

    # Save t-SNE plot
    _plot_scatter(
        tsne, y, label_enc.classes_,
        title=f"Intent Embeddings (t-SNE) — silhouette={sil:.3f}, "
              f"logistic_acc={cv_logistic.mean():.2f}",
        output_path=output_dir / "intent_tsne.pdf",
    )

    # Save confusion matrix
    _plot_confusion_matrix(cm, label_enc.classes_, output_dir / "confusion_matrix.pdf")

    # Save raw embeddings + labels for downstream analysis
    np.savez(
        output_dir / "intent_embeddings.npz",
        embeddings=embeddings,
        labels=np.array(labels),
        episode_indices=np.array([hash(l) for l in labels]),
        tsne_2d=tsne,
    )

    return {
        "n_samples": len(embeddings),
        "n_classes": n_classes,
        "class_names": label_enc.classes_.tolist(),
        "silhouette_score": float(sil),
        "logistic_cv_accuracy_mean": float(cv_logistic.mean()),
        "logistic_cv_accuracy_std": float(cv_logistic.std()),
        "knn_cv_accuracy_mean": float(cv_knn.mean()),
        "knn_cv_accuracy_std": float(cv_knn.std()),
        "cluster_purity": float(purity),
        "per_class_accuracy_mean": float(per_class_acc.mean()),
        "per_class_accuracy_min": float(per_class_acc.min()),
        "per_class_accuracy_max": float(per_class_acc.max()),
        "chance_accuracy": 1.0 / n_classes,
    }


# ---------------------------------------------------------------
# Analysis 2: temporal stability (intent consistency over time)
# ---------------------------------------------------------------
def analyze_temporal_stability(
    embeddings: np.ndarray,
    labels: list,
    output_dir: Path,
) -> dict:
    """Test whether intent_emb is stable within an episode and distinct across tasks.

    For each pair of (i, j) with same label, compute cosine similarity.
    For each pair with different labels, compute cosine similarity.
    Same-task should be high; cross-task should be low.
    """
    # Normalize for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    X = embeddings / norms

    sim_matrix = X @ X.T
    n = len(embeddings)
    labels_arr = np.array(labels)
    same_mask = labels_arr[:, None] == labels_arr[None, :]
    np.fill_diagonal(same_mask, False)  # exclude self
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)

    sim_same = sim_matrix[same_mask]
    sim_diff = sim_matrix[diff_mask]

    out = {
        "within_task_cosine_mean": float(sim_same.mean()) if sim_same.size else float("nan"),
        "within_task_cosine_std": float(sim_same.std()) if sim_same.size else float("nan"),
        "cross_task_cosine_mean": float(sim_diff.mean()) if sim_diff.size else float("nan"),
        "cross_task_cosine_std": float(sim_diff.std()) if sim_diff.size else float("nan"),
        "separation": float(sim_same.mean() - sim_diff.mean())
                       if (sim_same.size and sim_diff.size) else float("nan"),
    }
    return out


# ---------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------
def _plot_scatter(coords, y, class_names, title, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", len(class_names))
    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(cls_idx)], label=cls_name[:30] + ("..." if len(cls_name) > 30 else ""),
            alpha=0.7, s=18,
        )
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
    plt.title(title, fontsize=10)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(cm, class_names, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    # Truncate class names to keep the matrix readable
    short_names = [n[:18] + ("..." if len(n) > 18 else "") for n in class_names]
    ConfusionMatrixDisplay(cm, display_labels=short_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format="d",
    )
    ax.set_title("Intent embedding task classifier (logistic regression, 5-fold CV)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to a model checkpoint (.pt).")
    parser.add_argument("--data", required=True, type=str,
                        help="Path to the HDF5 dataset (e.g. data/libero_spatial.h5).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory to write results to (created if missing).")
    parser.add_argument("--cameras", nargs="+", default=["image", "wrist_image"],
                        help="Camera names matching the dataset.")
    parser.add_argument("--use-memory-bank", action="store_true",
                        help="Set if the model was trained with --use-memory-bank.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap on number of episodes to process.")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Skip the temporal-stability analysis.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device: {device}")
    print(f"[probe] checkpoint: {args.checkpoint}")
    print(f"[probe] data: {args.data}")

    # ---- Build model ----
    print("[probe] Building model...")
    model = _build_model_from_args(args, ckpt_args=None)
    model = model.to(device)
    print(f"[probe] use_intent_tokens={model.use_intent_tokens}, "
          f"num_intent_tokens={model.num_intent_tokens}, intent_dim={model.intent_dim}")

    # ---- Load checkpoint ----
    print("[probe] Loading checkpoint...")
    ckpt_args = _load_checkpoint(model, args.checkpoint, device)
    if ckpt_args is not None and hasattr(ckpt_args, "use_intent_tokens"):
        # Re-build with the actual checkpoint flags if they differ
        actual_use_intent = bool(getattr(ckpt_args, "use_intent_tokens", True))
        if actual_use_intent != model.use_intent_tokens:
            print(f"[probe] Rebuilding model with use_intent_tokens={actual_use_intent}")
            model = _build_model_from_args(args, ckpt_args=vars(ckpt_args))
            model = model.to(device)
            _load_checkpoint(model, args.checkpoint, device)
    print(f"[probe] use_intent_tokens (final)={model.use_intent_tokens}, "
          f"num_intent_tokens={model.num_intent_tokens}, intent_dim={model.intent_dim}")

    if not model.use_intent_tokens:
        print("[probe] ERROR: checkpoint has no intent tokens. Aborting.")
        sys.exit(1)

    # ---- Load dataset ----
    print(f"[probe] Loading dataset ({args.data})...")
    dataset = ALIGNDataset(
        args.data,
        mode="probe",
        cameras=args.cameras,
    )
    print(f"[probe] dataset has {len(dataset)} episodes")

    # ---- Collect embeddings ----
    print("[probe] Collecting intent embeddings...")
    embeddings, labels, ep_idx = collect_intent_embeddings(
        model, dataset, device, max_episodes=args.max_episodes,
    )
    print(f"[probe] collected {len(embeddings)} embeddings, "
          f"{len(set(labels))} unique tasks")

    # ---- EXP-D.5: task clustering ----
    print("\n[probe] === EXP-D.5: task-discriminative clustering ===")
    clustering_results = analyze_task_clustering(
        embeddings, labels, args.output,
    )
    for k, v in clustering_results.items():
        if k == "class_names":
            print(f"  {k}: ({len(v)} tasks)")
        elif isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    results = {
        "exp_d5_clustering": clustering_results,
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "n_episodes_processed": len(embeddings),
    }

    # ---- Temporal stability ----
    if not args.no_temporal:
        print("\n[probe] === Temporal stability ===")
        stability_results = analyze_temporal_stability(
            embeddings, labels, args.output,
        )
        for k, v in stability_results.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        results["temporal_stability"] = stability_results

    # ---- Save results ----
    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[probe] Results saved to {results_path}")
    print(f"[probe] t-SNE plot:    {args.output / 'intent_tsne.pdf'}")
    print(f"[probe] Confusion:     {args.output / 'confusion_matrix.pdf'}")
    print(f"[probe] Raw data:      {args.output / 'intent_embeddings.npz'}")


if __name__ == "__main__":
    main()
