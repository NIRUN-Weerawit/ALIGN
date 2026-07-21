#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation script for the ALIGN Task Identification Head.

Tests:
  1. Classification accuracy — per-class accuracy, confusion matrix, top-3
  2. OOD detection — ROC, recall, false positive rate
  3. Calibration — confidence vs correctness (ECE)
  4. α quality — distribution of (1-H)*(1-ood) across timesteps within episodes
  5. z_sext quality — cosine similarity of p@E_clip to the correct task embedding

Usage:
    PYTHONNOUSERSITE=1 python eval/eval_task_head.py \
        --data h5_data/libero_spatial.h5 \
        --task-head-checkpoint checkpoints/task_head/.../task_head_best.pt \
        --encoder-checkpoint checkpoints/encoder/.../best.pt \
        --cameras image wrist_image
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.task_head import TaskHead, TaskHeadBundle, create_task_head, task_head_loss
from data.align_dataset import ALIGNDataset, MultiALIGNDataset, head_collate


# ================================================================
# Helpers (mirrors train_task_head.py)
# ================================================================

def _encode_zv_zt(model, frames, state):
    """Encode (frames, state) through frozen encoders + mixer with zero text.

    v2: ``state`` is a one-step robot state (B, 7) — replaces the (B, K, 6)
    trajectory window that older callers passed.
    """
    device = next(model.parameters()).device
    if not torch.is_tensor(frames):
        frames = torch.as_tensor(frames, dtype=torch.uint8, device=device)
    else:
        frames = frames.to(device)
    if not torch.is_tensor(state):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
    else:
        state = state.to(device).float()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        z_v_raw = model.encode_vision(frames)
        z_s_tokens_raw = model.encode_trajectory_tokens(state)
        z_v_mixed, z_s_tokens_mixed, _ = model.cross_attention_mixer(
            z_v_raw, z_s_tokens_raw, torch.zeros_like(z_v_raw),
        )
        z_s = z_s_tokens_mixed.mean(dim=1)
    return z_v_mixed.float(), z_s.float()


def build_task_vocabulary(h5_path: str) -> List[str]:
    """Extract unique task descriptions from HDF5."""
    tasks: set = set()
    with h5py.File(h5_path, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("ep_"):
                continue
            ep = f[key]
            try:
                meta = json.loads(ep["meta"][()])
                if "task_description" in meta:
                    tasks.add(meta["task_description"])
                    continue
            except (KeyError, ValueError):
                pass
            try:
                raw = ep["texts"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                texts = json.loads(raw)
                if isinstance(texts, list) and texts:
                    tasks.add(str(texts[0]))
            except (KeyError, ValueError):
                pass
    return sorted(tasks)


# ================================================================
# Test 1: Classification accuracy + confusion matrix
# ================================================================

def test_1_classification(
    model, task_head, loader, device, known_tasks, use_bf16=True,
) -> Dict:
    """Per-class accuracy, confusion matrix, top-3."""
    print("\n[Test 1] Classification accuracy")
    print("-" * 50)

    task_head.eval()
    K = len(known_tasks)
    confusion = np.zeros((K, K), dtype=np.int64)
    correct = 0
    correct_top3 = 0
    total = 0
    all_confidences = []
    all_correct = []

    for batch in loader:
        frames = batch["frames"]
        # v2: one-step robot state (B, 7) — replaces (B, K, 6) trajectory
        state = batch["robot_state"]
        task_ids = batch["task_id"]
        if not torch.is_tensor(task_ids):
            task_ids = torch.as_tensor(task_ids)
        task_ids = task_ids.to(device)

        known_mask = task_ids != -100
        if not known_mask.any():
            continue

        z_v, z_s = _encode_zv_zt(model, frames, state)
        with torch.no_grad():
            logits = task_head(z_v, z_s)
        logits = logits.float()
        known_logits = logits[:, :-1]

        preds = known_logits.argmax(dim=-1)
        confidences = F.softmax(known_logits, dim=-1).max(dim=-1).values

        for i in range(len(preds)):
            if known_mask[i]:
                gt = task_ids[i].item()
                pr = preds[i].item()
                confusion[gt, pr] += 1
                total += 1
                if pr == gt:
                    correct += 1
                    all_correct.append(1)
                else:
                    all_correct.append(0)
                all_confidences.append(confidences[i].item())

                top3 = known_logits[i].topk(min(3, K)).indices
                if gt in top3.tolist():
                    correct_top3 += 1

    acc = correct / max(total, 1)
    top3_acc = correct_top3 / max(total, 1)

    # Per-class accuracy
    per_class = {}
    for i, task in enumerate(known_tasks):
        row_sum = confusion[i].sum()
        per_class[task] = float(confusion[i, i] / max(row_sum, 1))

    print(f"  N samples:          {total}")
    print(f"  Top-1 accuracy:     {acc:.4f}")
    print(f"  Top-3 accuracy:     {top3_acc:.4f}")
    print(f"  Per-class accuracy:")
    for task, a in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"    {task[:50]:50s}  {a:.3f}")

    # Confusion matrix (compact)
    print(f"  Confusion matrix (rows=GT, cols=Pred):")
    for i in range(min(K, 10)):
        row = confusion[i]
        if row.sum() > 0:
            print(f"    [{i}] {known_tasks[i][:30]:30s}  {row.tolist()}")

    return {
        "n_samples": total,
        "top1_accuracy": acc,
        "top3_accuracy": top3_acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion.tolist(),
    }


# ================================================================
# Test 2: OOD detection
# ================================================================

def test_2_ood_detection(
    model, task_head, loader, device, use_bf16=True,
) -> Dict:
    """ROC-like analysis of OOD detection."""
    print("\n[Test 2] OOD detection")
    print("-" * 50)

    task_head.eval()
    all_p_ood = []
    all_ood_labels = []

    for batch in loader:
        frames = batch["frames"]
        # v2: one-step robot state (B, 7) — replaces (B, K, 6) trajectory
        state = batch["robot_state"]
        ood_labels = batch["ood_label"]
        if not torch.is_tensor(ood_labels):
            ood_labels = torch.as_tensor(ood_labels)
        ood_labels = ood_labels.to(device)

        z_v, z_s = _encode_zv_zt(model, frames, state)
        with torch.no_grad():
            logits = task_head(z_v, z_s)
        logits = logits.float()
        p = F.softmax(logits, dim=-1)
        p_ood = p[:, -1]

        all_p_ood.extend(p_ood.cpu().numpy().tolist())
        all_ood_labels.extend(ood_labels.cpu().numpy().tolist())

    all_p_ood = np.array(all_p_ood)
    all_ood_labels = np.array(all_ood_labels)

    n_ood = int(all_ood_labels.sum())
    n_id = len(all_ood_labels) - n_ood

    # At threshold 0.5
    ood_pred = (all_p_ood > 0.5).astype(float)
    tp = int(((ood_pred == 1) & (all_ood_labels == 1)).sum())
    fp = int(((ood_pred == 1) & (all_ood_labels == 0)).sum())
    tn = int(((ood_pred == 0) & (all_ood_labels == 0)).sum())
    fn = int(((ood_pred == 0) & (all_ood_labels == 1)).sum())

    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)

    # AUROC (simple trapezoidal)
    thresholds = np.linspace(0, 1, 101)
    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        pred = (all_p_ood > thresh).astype(float)
        tp_t = int(((pred == 1) & (all_ood_labels == 1)).sum())
        fp_t = int(((pred == 1) & (all_ood_labels == 0)).sum())
        tpr_list.append(tp_t / max(tp + fn, 1))
        fpr_list.append(fp_t / max(fp + tn, 1))
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    auroc = float(np.trapz(tpr_arr, fpr_arr))

    print(f"  N samples:          {len(all_ood_labels)}  (OOD={n_ood}, ID={n_id})")
    print(f"  At threshold 0.5:")
    print(f"    OOD recall (TPR):  {recall:.4f}")
    print(f"    OOD precision:     {precision:.4f}")
    print(f"    FPR (ID→OOD):      {fpr:.4f}")
    print(f"  AUROC:               {auroc:.4f}")
    print(f"  Mean p_ood (OOD):    {all_p_ood[all_ood_labels==1].mean():.4f}" if n_ood else "  (no OOD samples)")
    print(f"  Mean p_ood (ID):     {all_p_ood[all_ood_labels==0].mean():.4f}" if n_id else "  (no ID samples)")

    return {
        "n_samples": len(all_ood_labels),
        "n_ood": n_ood,
        "n_id": n_id,
        "recall_at_0.5": recall,
        "fpr_at_0.5": fpr,
        "precision_at_0.5": precision,
        "auroc": auroc,
    }


# ================================================================
# Test 3: Calibration (ECE)
# ================================================================

def test_3_calibration(
    model, task_head, loader, device, n_bins=10, use_bf16=True,
) -> Dict:
    """Expected Calibration Error: confidence vs accuracy."""
    print("\n[Test 3] Calibration (ECE)")
    print("-" * 50)

    task_head.eval()
    all_confidences = []
    all_correct = []

    for batch in loader:
        frames = batch["frames"]
        # v2: one-step robot state (B, 7) — replaces (B, K, 6) trajectory
        state = batch["robot_state"]
        task_ids = batch["task_id"]
        if not torch.is_tensor(task_ids):
            task_ids = torch.as_tensor(task_ids)
        task_ids = task_ids.to(device)

        known_mask = task_ids != -100
        if not known_mask.any():
            continue

        z_v, z_s = _encode_zv_zt(model, frames, state)
        with torch.no_grad():
            logits = task_head(z_v, z_s)
        logits = logits.float()
        known_logits = logits[:, :-1]

        p = F.softmax(known_logits, dim=-1)
        conf, pred = p.max(dim=-1)

        for i in range(len(pred)):
            if known_mask[i]:
                all_confidences.append(conf[i].item())
                all_correct.append(int(pred[i].item() == task_ids[i].item()))

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    # ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(all_confidences)
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (all_confidences >= lo) & (all_confidences < hi)
        if i == n_bins - 1:
            mask = (all_confidences >= lo) & (all_confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = all_confidences[mask].mean()
        bin_acc = all_correct[mask].mean()
        bin_frac = mask.sum() / n
        ece += abs(bin_conf - bin_acc) * bin_frac
        bin_data.append({
            "range": f"[{lo:.1f}, {hi:.1f})",
            "n": int(mask.sum()),
            "confidence": float(bin_conf),
            "accuracy": float(bin_acc),
        })

    print(f"  N samples:          {n}")
    print(f"  ECE ({n_bins} bins):       {ece:.4f}")
    print(f"  Mean confidence:    {all_confidences.mean():.4f}")
    print(f"  Mean accuracy:      {all_correct.mean():.4f}")
    print(f"  Bins:")
    for b in bin_data:
        print(f"    {b['range']:12s}  n={b['n']:5d}  conf={b['confidence']:.3f}  acc={b['accuracy']:.3f}")

    return {
        "n_samples": n,
        "ece": ece,
        "mean_confidence": float(all_confidences.mean()),
        "mean_accuracy": float(all_correct.mean()),
        "bins": bin_data,
    }


# ================================================================
# Test 4: α quality across timesteps within episodes
# ================================================================

def test_4_alpha_quality(
    model, task_head, ds, device, n_episodes=10, use_bf16=True,
) -> Dict:
    """Track α = (1-H)*(1-p_ood) across timesteps within episodes."""
    print("\n[Test 4] α quality across timesteps")
    print("-" * 50)

    task_head.eval()
    all_alpha_curves = []
    n_episodes_checked = 0

    for ep_idx in range(min(n_episodes, len(ds))):
        try:
            source = ds[ep_idx]
            frames_i = source["frames"]
            poses_i = source["poses"][..., :6]
            text_i = source["text"]
            if not isinstance(text_i, list):
                text_i = [text_i]

            N = len(frames_i)
            if N < 10:
                continue

            alphas = []
            for t in range(N):
                # v2: one-step robot state at t — (1, 7) = [pos(3), euler(3), gripper(1)]
                # Gripper is unknown here (the per-step `grippers` array isn't on
                # the dataset item in this test path), so use 0.0 — the model
                # falls back to the last-pose behavior for state inputs.
                p_t = np.concatenate(
                    [poses_i[t].astype(np.float32), [0.0]], axis=0
                )[np.newaxis]  # (1, 7)

                f_t = frames_i[t]
                if f_t.ndim == 3:
                    f_t = f_t[np.newaxis]  # (1, H, W, 3)
                else:
                    f_t = f_t[np.newaxis]  # (1, V, H, W, 3) for multi-cam

                z_v, z_s = _encode_zv_zt(model, f_t, p_t)
                with torch.no_grad():
                    logits = task_head(z_v, z_s)
                logits = logits.float()
                p = F.softmax(logits, dim=-1)

                K = p.shape[-1]
                log_p = torch.log(p.clamp_min(1e-12))
                H = -(p * log_p).sum(dim=-1) / np.log(K)
                p_ood = p[:, -1]
                alpha = (1.0 - H) * (1.0 - p_ood)
                alphas.append(alpha.item())

            all_alpha_curves.append(alphas)
            n_episodes_checked += 1

        except Exception as e:
            print(f"  Skipping episode {ep_idx}: {e}")
            continue

    if n_episodes_checked == 0:
        print("  No valid episodes")
        return {"n_episodes": 0}

    # Analyze: first 25% vs last 25%
    first_q_alphas = []
    last_q_alphas = []
    for curve in all_alpha_curves:
        q = len(curve) // 4
        if q > 0:
            first_q_alphas.append(np.mean(curve[:q]))
            last_q_alphas.append(np.mean(curve[-q:]))

    first_q_mean = float(np.mean(first_q_alphas)) if first_q_alphas else 0
    last_q_mean = float(np.mean(last_q_alphas)) if last_q_alphas else 0
    ramp = last_q_mean - first_q_mean

    print(f"  N episodes:         {n_episodes_checked}")
    print(f"  α first quartile:   {first_q_mean:.4f}")
    print(f"  α last quartile:    {last_q_mean:.4f}")
    print(f"  Ramp (last - first): {ramp:+.4f}")
    print(f"  Interpretation: positive ramp = α increases as task becomes clear (good)")

    # Show a few example curves
    for i, curve in enumerate(all_alpha_curves[:3]):
        # Downsample to 10 points for readability
        n = len(curve)
        if n > 10:
            indices = np.linspace(0, n - 1, 10).astype(int)
            sampled = [curve[j] for j in indices]
        else:
            sampled = curve
        print(f"  Episode {i}: " + " ".join(f"{a:.2f}" for a in sampled))

    return {
        "n_episodes": n_episodes_checked,
        "alpha_first_quartile": first_q_mean,
        "alpha_last_quartile": last_q_mean,
        "alpha_ramp": ramp,
        "example_curves": all_alpha_curves[:3],
    }


# ================================================================
# Test 5: z_sext quality
# ================================================================

def test_5_ztext_quality(
    model, task_head, e_clip, loader, device, known_tasks, use_bf16=True,
) -> Dict:
    """Cosine similarity of p@E_clip to the correct task's CLIP embedding."""
    print("\n[Test 5] z_sext quality (p@E_clip vs correct task embedding)")
    print("-" * 50)

    task_head.eval()
    K = len(known_tasks)
    all_cosines = []

    for batch in loader:
        frames = batch["frames"]
        # v2: one-step robot state (B, 7) — replaces (B, K, 6) trajectory
        state = batch["robot_state"]
        task_ids = batch["task_id"]
        if not torch.is_tensor(task_ids):
            task_ids = torch.as_tensor(task_ids)
        task_ids = task_ids.to(device)

        known_mask = task_ids != -100
        if not known_mask.any():
            continue

        z_v, z_s = _encode_zv_zt(model, frames, state)
        with torch.no_grad():
            logits = task_head(z_v, z_s)
        logits = logits.float()
        p = F.softmax(logits, dim=-1)

        z_sext = p @ e_clip  # (B, 256)

        for i in range(len(task_ids)):
            if known_mask[i]:
                gt_id = task_ids[i].item()
                target_embedding = e_clip[gt_id]  # (256,)
                cos = F.cosine_similarity(
                    z_sext[i].unsqueeze(0), target_embedding.unsqueeze(0), dim=-1
                ).item()
                all_cosines.append(cos)

    all_cosines = np.array(all_cosines)
    print(f"  N samples:          {len(all_cosines)}")
    print(f"  Mean cosine:        {all_cosines.mean():.4f}")
    print(f"  Median cosine:      {np.median(all_cosines):.4f}")
    print(f"  Percentiles:        p5={np.percentile(all_cosines, 5):.3f}  "
          f"p25={np.percentile(all_cosines, 25):.3f}  "
          f"p75={np.percentile(all_cosines, 75):.3f}  "
          f"p95={np.percentile(all_cosines, 95):.3f}")

    return {
        "n_samples": len(all_cosines),
        "mean_cosine": float(all_cosines.mean()),
        "median_cosine": float(np.median(all_cosines)),
        "p5": float(np.percentile(all_cosines, 5)),
        "p95": float(np.percentile(all_cosines, 95)),
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ALIGN Task Identification Head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s)")
    parser.add_argument("--task-head-checkpoint", required=True,
                        help="Path to task_head_best.pt")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Path to encoder+mixer checkpoint")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views (e.g. 'image wrist_image')")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of episodes for Test 4 (α quality)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None,
                        help="Optional path to write the full report as JSON")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- Load task head checkpoint ----
    print(f"Loading task head: {args.task_head_checkpoint}")
    th_ckpt = torch.load(args.task_head_checkpoint, map_location=device, weights_only=False)
    th_cfg = th_ckpt["config"]
    K = th_cfg["K"]
    embed_dim = th_cfg.get("embed_dim", 256)
    hidden_dim = th_cfg.get("hidden_dim", 256)
    dropout = th_cfg.get("dropout", 0.0)
    known_tasks = th_cfg["vocab_known"]
    ood_tasks = th_cfg.get("vocab_ood", [])

    task_head = create_task_head(
        num_known_classes=K, embed_dim=embed_dim,
        hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)
    task_head.load_state_dict(th_ckpt["task_head_state_dict"])
    task_head.eval()
    print(f"  K={K} known tasks, {len(ood_tasks)} OOD tasks")
    print(f"  Params: {sum(p.numel() for p in task_head.parameters()):,}")

    E_clip = th_ckpt["E_clip"].to(device)

    # ---- Load encoder+mixer ----
    print(f"Loading encoder: {args.encoder_checkpoint}")
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    enc_cfg = enc_ckpt.get("config", {}) if isinstance(enc_ckpt, dict) else {}
    num_cameras = len(args.cameras) if args.cameras else 1
    align = ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=args.chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=enc_cfg.get("mixer_dim", 512),
        num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
        num_cameras=num_cameras,
    ).to(device)
    if "trainable_state_dict" in enc_ckpt:
        align.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    else:
        align.load_state_dict(enc_ckpt, strict=False)
    align.freeze_backbone()
    align.freeze_all_encoders()
    align.eval()

    # ---- Dataset ----
    task_to_id = {t: i for i, t in enumerate(known_tasks)}
    ood_set = set(ood_tasks)

    if len(args.data) == 1:
        ds = ALIGNDataset(args.data[0], mode="head", traj_window=args.traj_window,
                          cameras=args.cameras)
    else:
        ds = MultiALIGNDataset(args.data, mode="head", traj_window=args.traj_window,
                               cameras=args.cameras)

    n_total = len(ds)
    n_val = max(1, int(n_total * args.val_split))
    indices = list(range(n_total - n_val, n_total))
    print(f"  Dataset: {args.data}  (N={n_total}, val={n_val})")
    print(f"  Device: {device}")

    # Build collate that attaches task_id and ood_label
    import numpy as _np
    _rng = _np.random.default_rng(42)

    def _collate(batch):
        from training.train_task_head import task_collate
        return task_collate(
            batch, chunk_size=args.chunk_size, task_to_id=task_to_id,
            ood_set=ood_set, augment_ood_prob=0.0, rng=_rng,
        )

    loader = DataLoader(
        ds, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices),
        drop_last=False, collate_fn=_collate, num_workers=0,
    )

    # ---- Run tests ----
    report = {}

    # Test 1: Classification
    report["test1_classification"] = test_1_classification(
        align, task_head, loader, device, known_tasks,
    )

    # Test 2: OOD detection
    report["test2_ood_detection"] = test_2_ood_detection(
        align, task_head, loader, device,
    )

    # Test 3: Calibration
    report["test3_calibration"] = test_3_calibration(
        align, task_head, loader, device,
    )

    # Test 4: α quality
    report["test4_alpha_quality"] = test_4_alpha_quality(
        align, task_head, ds, device, n_episodes=args.n_episodes,
    )

    # Test 5: z_sext quality
    report["test5_ztext_quality"] = test_5_ztext_quality(
        align, task_head, E_clip, loader, device, known_tasks,
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  TASK HEAD EVALUATION REPORT")
    print("=" * 60)
    t1 = report["test1_classification"]
    t2 = report["test2_ood_detection"]
    t3 = report["test3_calibration"]
    t4 = report["test4_alpha_quality"]
    t5 = report["test5_ztext_quality"]

    print(f"  Test 1 (Classification):  top1={t1['top1_accuracy']:.3f}  top3={t1['top3_accuracy']:.3f}")
    print(f"  Test 2 (OOD Detection):   recall={t2['recall_at_0.5']:.3f}  AUROC={t2['auroc']:.3f}  FPR={t2['fpr_at_0.5']:.3f}")
    print(f"  Test 3 (Calibration):     ECE={t3['ece']:.3f}")
    print(f"  Test 4 (α Quality):       ramp={t4.get('alpha_ramp', 0):+.3f}  (first={t4.get('alpha_first_quartile', 0):.3f}  last={t4.get('alpha_last_quartile', 0):.3f})")
    print(f"  Test 5 (z_sext Quality):  mean_cos={t5['mean_cosine']:.3f}")
    print("=" * 60)

    if args.output_json:
        with open(args.output_json, "w") as f:
            # Convert non-serializable items
            def _serialize(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            json.dump(report, f, indent=2, default=_serialize)
        print(f"  Report written to {args.output_json}")


if __name__ == "__main__":
    main()