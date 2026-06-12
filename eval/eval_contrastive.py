#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate ALIGN pretrained models — contrastive backbone and full model.

Evaluates three modes:
  1. Raw encoder outputs (Phase 1a) — InfoNCE on pre-mixer embeddings
  2. Mixed encoder outputs (Phase 1b) — InfoNCE on post-mixer embeddings
  3. Full model (Phase 2) — Decision head α predictions + Assistant head Δposes

Usage:
    # Evaluate contrastive backbone (Phase 1a or 1b checkpoint)
    python eval/eval_contrastive.py \
        --checkpoint checkpoints/pretrain/pretrain/best.pt \
        --data-dir /path/to/libero_10 --n-samples 20

    # Compare raw vs mixed embeddings
    python eval/eval_contrastive.py \
        --checkpoint checkpoints/pretrain/pretrain/best.pt \
        --data-dir /path/to/libero_10 --compare-raw-vs-mixed

    # Evaluate full model with heads (Phase 2 checkpoint)
    python eval/eval_contrastive.py \
        --checkpoint checkpoints/heads/joint_best.pt \
        --data-dir /path/to/libero_10 --eval-heads

    # Compare multiple texts against the same trajectory
    python eval/eval_contrastive.py \
        --checkpoint checkpoints/pretrain/pretrain/best.pt \
        --data-dir /path/to/libero_10 --compare-texts \
        --texts "pick up the mug,pick and place,grasp the object,do nothing"

    # Single sample evaluation
    python eval/eval_contrastive.py \
        --checkpoint checkpoints/pretrain/pretrain/best.pt \
        --frame test.png --pose "0.1 0.2 0.3 0.0 0.0 0.0" \
        --text "pick up the red mug"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel


# ================================================================
# Model loading
# ================================================================

def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    use_text: bool = True,
) -> ALIGNModel:
    """Load model from checkpoint. Handles both pretrain and head checkpoints.

    Pretrain checkpoints (Phase 1a/1b) store state under 'trainable_state_dict'.
    Head checkpoints (Phase 2) store full model state under 'model_state_dict'.
    Falls back to loading directly if neither key is found.
    """
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=5,
        use_text=use_text,
        device=device,
    ).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try loading in priority order
    if "trainable_state_dict" in ckpt:
        model.load_trainable_state_dict(ckpt["trainable_state_dict"])
        print(f"[eval] Loaded trainable state from {checkpoint_path}")
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[eval] Loaded full model state from {checkpoint_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"[eval] Loaded raw state dict from {checkpoint_path}")

    return model


# ================================================================
# Embedding extraction
# ================================================================

@torch.no_grad()
def get_raw_embeddings(
    model: ALIGNModel,
    frames: torch.Tensor,
    trajs: torch.Tensor,
    texts: List[str],
) -> dict:
    """Extract raw encoder embeddings (no mixer) — Phase 1a style.

    Args:
        frames: (B, H, W, 3) uint8.
        trajs: (B, K, 6) trajectory windows.
        texts: list of str, length B.

    Returns:
        dict with 'z_v', 'z_t', 'z_text' (B, D) and
        'cos_vt', 'cos_vl', 'cos_tl' (B,) pairwise similarities.
    """
    z_v = model.encode_raw_vision(frames)
    z_t = model.encode_raw_trajectory(trajs)
    z_text = model.encode_raw_text(texts)
    if z_text is None:
        z_text = torch.zeros_like(z_v)

    z_v_n = F.normalize(z_v, dim=-1)
    z_t_n = F.normalize(z_t, dim=-1)
    z_text_n = F.normalize(z_text, dim=-1)

    return {
        "z_v": z_v.detach(),
        "z_t": z_t.detach(),
        "z_text": z_text.detach(),
        "cos_vt": (z_v_n * z_t_n).sum(dim=-1).detach(),
        "cos_vl": (z_v_n * z_text_n).sum(dim=-1).detach(),
        "cos_tl": (z_t_n * z_text_n).sum(dim=-1).detach(),
    }


@torch.no_grad()
def get_mixed_embeddings(
    model: ALIGNModel,
    frames: torch.Tensor,
    trajs: torch.Tensor,
    texts: List[str],
) -> dict:
    """Extract embeddings through cross-attention mixer — Phase 1b style.

    Args:
        frames: (B, H, W, 3) uint8.
        trajs: (B, K, 6) trajectory windows.
        texts: list of str, length B.

    Returns:
        dict with 'z_v', 'z_t', 'z_text' (B, D) and
        'cos_vt', 'cos_vl', 'cos_tl' (B,) pairwise similarities.
    """
    mixed = model.encode_mixed(frames, trajs, texts)
    z_v, z_t, z_text = mixed["z_v"], mixed["z_t"], mixed["z_text"]

    z_v_n = F.normalize(z_v, dim=-1)
    z_t_n = F.normalize(z_t, dim=-1)
    z_text_n = F.normalize(z_text, dim=-1)

    return {
        "z_v": z_v.detach(),
        "z_t": z_t.detach(),
        "z_text": z_text.detach(),
        "cos_vt": (z_v_n * z_t_n).sum(dim=-1).detach(),
        "cos_vl": (z_v_n * z_text_n).sum(dim=-1).detach(),
        "cos_tl": (z_t_n * z_text_n).sum(dim=-1).detach(),
    }


@torch.no_grad()
def get_head_predictions(
    model: ALIGNModel,
    frames: torch.Tensor,
    trajs: torch.Tensor,
    texts: List[str],
) -> dict:
    """Run full forward pass through heads — Phase 2 style.

    Args:
        frames: (B, H, W, 3) uint8.
        trajs: (B, K, 6) trajectory windows.
        texts: list of str, length B.

    Returns:
        dict with 'alpha' (B, 1), 'delta' (B, K, 6), and
        'cos_vt', 'cos_vl', 'cos_tl' (B,) from the mixer output.
    """
    mixed = model.encode_mixed(frames, trajs, texts)
    z_v, z_t, z_text = mixed["z_v"], mixed["z_t"], mixed["z_text"]

    z_v_n = F.normalize(z_v, dim=-1)
    z_t_n = F.normalize(z_t, dim=-1)
    z_text_n = F.normalize(z_text, dim=-1)

    alpha = model.decision_head(z_v, z_t, z_text)
    delta = model.assistant_head(z_v, z_t, z_text, trajs[:, -1])

    return {
        "alpha": alpha.detach(),
        "delta": delta.detach(),
        "cos_vt": (z_v_n * z_t_n).sum(dim=-1).detach(),
        "cos_vl": (z_v_n * z_text_n).sum(dim=-1).detach(),
        "cos_tl": (z_t_n * z_text_n).sum(dim=-1).detach(),
    }


# ================================================================
# Data loading
# ================================================================

def load_libero_samples(
    data_dir: str,
    n_samples: int = 10,
    repo_id: str = "nvidia/LIBERO_LeRobot_v3",
    traj_window: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load samples from local LIBERO data directory.

    Uses LeRobotDataset with pyav backend (no torchcodec dependency).

    Returns:
        (frames, trajs, texts) where:
            frames: (B, H, W, 3) uint8
            trajs:  (B, K, 6) float32
            texts:  list of str
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError("Need lerobot: pip install lerobot")

    ds = LeRobotDataset(repo_id, root=data_dir, video_backend="pyav")
    indices = sorted(np.random.choice(len(ds), min(n_samples, len(ds)), replace=False))

    frames, trajs, texts = [], [], []
    for idx in indices:
        sample = ds[idx]

        # Camera frame — prefer wrist, fall back to front
        img = sample.get("observation.images.wrist_image",
                         sample.get("observation.images.image", None))
        if img is None:
            for k in sample:
                if "images" in str(k):
                    img = sample[k]
                    break
        if img is not None:
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:   # (T, C, H, W) from delta_timestamps
                    img = img[-1]    # most recent frame
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)  # C,H,W → H,W,C
                if img.dtype == torch.float32 or img.dtype == torch.float16:
                    img = img.mul(255).to(torch.uint8)
                else:
                    img = img.to(torch.uint8)
            frames.append(img)

        # State → trajectory window
        state = sample.get("observation.state", sample.get("state", None))
        if state is not None:
            if isinstance(state, torch.Tensor):
                if state.dim() == 2:
                    state = state[-1]  # temporal dim → single frame
                s = state[:6].float() if state.shape[-1] >= 6 else \
                    torch.cat([state.float(), torch.zeros(6 - state.shape[-1])])
                trajs.append(s.unsqueeze(0).repeat(traj_window, 1))  # (K, 6)

        # Text
        task = sample.get("task", sample.get("language_instruction", "pick and place"))
        texts.append(str(task))

    B = len(frames)
    return (
        torch.stack(frames) if frames else torch.zeros(0, 224, 224, 3, dtype=torch.uint8),
        torch.stack(trajs) if trajs else torch.zeros(0, traj_window, 6),
        texts,
    )


# ================================================================
# Reporting
# ================================================================

def print_report(results: dict, texts: List[str], prefix: str = ""):
    """Pretty-print cosine similarity results."""
    cos_vt = results["cos_vt"].cpu().numpy()
    cos_vl = results["cos_vl"].cpu().numpy()
    cos_tl = results["cos_tl"].cpu().numpy()
    alphas = np.minimum(np.minimum(cos_vt, cos_vl), cos_tl)

    print(f"\n{'='*70}")
    print(f"{prefix}COSINE SIMILARITIES REPORT")
    print(f"{'='*70}")
    print(f"  {'#':>3s}  {'cos_vt':>8s}  {'cos_vl':>8s}  {'cos_tl':>8s}  {'α=min':>8s}  {'text':40s}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*40}")

    for i in range(len(texts)):
        print(f"  {i:3d}  {cos_vt[i]:8.4f}  {cos_vl[i]:8.4f}  {cos_tl[i]:8.4f}  {alphas[i]:8.4f}  {texts[i][:40]}")

    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*40}")
    print(f"  {'mean':>3s}  {cos_vt.mean():8.4f}  {cos_vl.mean():8.4f}  {cos_tl.mean():8.4f}  {alphas.mean():8.4f}")
    print(f"  {'std':>3s}  {cos_vt.std():8.4f}  {cos_vl.std():8.4f}  {cos_tl.std():8.4f}  {alphas.std():8.4f}")


def print_head_report(results: dict, texts: List[str]):
    """Pretty-print head prediction results."""
    alpha = results["alpha"].cpu().numpy()
    delta = results["delta"].cpu().numpy()
    cos_vt = results["cos_vt"].cpu().numpy()
    cos_vl = results["cos_vl"].cpu().numpy()
    cos_tl = results["cos_tl"].cpu().numpy()

    print(f"\n{'='*70}")
    print("HEAD PREDICTIONS REPORT")
    print(f"{'='*70}")
    print(f"  {'#':>3s}  {'α':>8s}  {'|Δ|mean':>8s}  {'cos_vt':>8s}  {'cos_vl':>8s}  {'cos_tl':>8s}  {'text':30s}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*30}")

    for i in range(len(texts)):
        delta_norm = np.linalg.norm(delta[i].reshape(-1, 6), axis=1).mean()
        print(f"  {i:3d}  {alpha[i].item():8.4f}  {delta_norm:8.4f}  "
              f"{cos_vt[i]:8.4f}  {cos_vl[i]:8.4f}  {cos_tl[i]:8.4f}  {texts[i][:30]}")

    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*30}")
    print(f"  {'mean':>3s}  {alpha.mean():8.4f}  {delta.mean():8.4f}  "
          f"{cos_vt.mean():8.4f}  {cos_vl.mean():8.4f}  {cos_tl.mean():8.4f}")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ALIGN models")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint (.pt)")
    parser.add_argument("--data-dir", help="Local LIBERO data directory")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--traj-window", type=int, default=20, help="Trajectory window K")
    parser.add_argument("--compare-raw-vs-mixed", action="store_true",
                        help="Compare raw encoder vs mixer outputs")
    parser.add_argument("--compare-texts", action="store_true",
                        help="Compare each sample against multiple texts")
    parser.add_argument("--texts", default="pick up the mug,pick and place,grasp the object",
                        help="Comma-separated text list for comparison")
    parser.add_argument("--eval-heads", action="store_true",
                        help="Evaluate full model with Decision + Assistant heads")
    parser.add_argument("--frame", help="Single image file path")
    parser.add_argument("--pose", help="Single pose as space-separated numbers")
    parser.add_argument("--text", help="Single text query")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)

    if args.data_dir:
        # ── Batch evaluation from LIBERO data ──
        frames, trajs, texts = load_libero_samples(
            args.data_dir, args.n_samples, traj_window=args.traj_window)
        frames = frames.to(args.device)
        trajs = trajs.float().to(args.device)

        if args.eval_heads:
            # Phase 2: full model with heads
            results = get_head_predictions(model, frames, trajs, texts)
            print_head_report(results, texts)
        elif args.compare_raw_vs_mixed:
            # Compare raw vs mixed embeddings
            raw = get_raw_embeddings(model, frames, trajs, texts)
            mixed = get_mixed_embeddings(model, frames, trajs, texts)
            print_report(raw, texts, prefix="[RAW] ")
            print_report(mixed, texts, prefix="[MIXED] ")
            print(f"\n  Improvement (mixed - raw):")
            print(f"    cos_vt: {mixed['cos_vt'].mean().item() - raw['cos_vt'].mean().item():+.4f}")
            print(f"    cos_vl: {mixed['cos_vl'].mean().item() - raw['cos_vl'].mean().item():+.4f}")
            print(f"    cos_tl: {mixed['cos_tl'].mean().item() - raw['cos_tl'].mean().item():+.4f}")
        else:
            # Default: mixed embeddings (Phase 1b style)
            results = get_mixed_embeddings(model, frames, trajs, texts)
            print_report(results, texts)

        # Multi-text comparison
        if args.compare_texts:
            alt_texts = [t.strip() for t in args.texts.split(",")]
            print(f"\n{'='*70}")
            print("MULTI-TEXT COMPARISON: first sample against all texts")
            print(f"{'='*70}")
            for t in alt_texts:
                if args.eval_heads:
                    r = get_head_predictions(model, frames[:1], trajs[:1], [t])
                else:
                    r = get_mixed_embeddings(model, frames[:1], trajs[:1], [t])
                al = min(r['cos_vt'].item(), r['cos_vl'].item(), r['cos_tl'].item())
                print(f"  text='{t:>42s}'  "
                      f"cos_vt={r['cos_vt'].item():.4f}  "
                      f"cos_vl={r['cos_vl'].item():.4f}  "
                      f"cos_tl={r['cos_tl'].item():.4f}  "
                      f"alpha={al:.4f}")

    elif args.frame and args.pose and args.text:
        # ── Single sample evaluation ──
        from PIL import Image
        img = torch.from_numpy(np.array(Image.open(args.frame))).to(args.device)
        if img.dim() == 3 and img.shape[-1] == 3:
            pass  # already HWC
        pose_vals = torch.tensor([float(x) for x in args.pose.split()]).float()
        if len(pose_vals) < 6:
            pose_vals = torch.cat([pose_vals, torch.zeros(6 - len(pose_vals))])
        traj = pose_vals.unsqueeze(0).repeat(args.traj_window, 1).unsqueeze(0).to(args.device)

        if args.eval_heads:
            results = get_head_predictions(model, img.unsqueeze(0), traj, [args.text])
            print_head_report(results, [args.text])
        else:
            results = get_mixed_embeddings(model, img.unsqueeze(0), traj, [args.text])
            print_report(results, [args.text])

    else:
        print("Provide --data-dir for batch eval, or --frame --pose --text for single eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
