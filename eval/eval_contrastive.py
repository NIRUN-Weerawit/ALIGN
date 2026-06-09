#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the pretrained contrastive backbone — inspect cosine similarities.

Usage:
    # Load pretrained backbone and test on LIBERO samples
    python eval/eval_contrastive.py \\
        --checkpoint checkpoints/pretrain/best.pt \\
        --data-dir /path/to/libero_10 --n-samples 20

    # Test with custom frames + poses + text
    python eval/eval_contrastive.py \\
        --checkpoint checkpoints/pretrain/best.pt \\
        --frame test.png --pose "0.1 0.2 0.3 0.0 0.0 0.0" \\
        --text "pick up the red mug"

    # Compare multiple texts against the same trajectory
    python eval/eval_contrastive.py \\
        --checkpoint checkpoints/pretrain/best.pt \\
        --data-dir /path/to/libero_10 --compare-texts \
        --texts "pick up the mug,pick and place,grasp the object,do nothing"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel


@torch.no_grad()
def get_embeddings(
    model: ALIGNModel,
    frames: torch.Tensor,
    trajs: torch.Tensor,
    texts: list[str],
) -> dict:
    """Extract embeddings and compute cosine similarities.

    Args:
        frames: (B, H, W, 3) uint8 or float32.
        trajs: (B, K, 6) trajectory windows.
        texts: list of str, length B.

    Returns:
        dict with 'z_v', 'z_t', 'z_text' (B, D) and
        'cos_vt', 'cos_vl', 'cos_tl' (B,) pairwise similarities.
    """
    z_v = model.encode_vision(frames)
    z_t = model.encode_trajectory(trajs)
    z_text = model.encode_text(texts)

    # L2 normalize for cosine
    z_v_n = F.normalize(z_v, dim=-1)
    z_t_n = F.normalize(z_t, dim=-1)
    z_text_n = F.normalize(z_text, dim=-1)

    cos_vt = (z_v_n * z_t_n).sum(dim=-1)
    cos_vl = (z_v_n * z_text_n).sum(dim=-1)
    cos_tl = (z_t_n * z_text_n).sum(dim=-1)

    return {
        "z_v": z_v.detach(),
        "z_t": z_t.detach(),
        "z_text": z_text.detach(),
        "cos_vt": cos_vt.detach(),
        "cos_vl": cos_vl.detach(),
        "cos_tl": cos_tl.detach(),
    }


def load_backbone(checkpoint_path: str, device: str = "cuda") -> ALIGNModel:
    """Load only the backbone (no heads needed for contrastive eval)."""
    model = ALIGNModel(
        embed_dim=256,
        chunk_size=5,
        use_text=True,
        device=device,
    ).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"[eval] Loaded backbone from {checkpoint_path}")
    return model


def load_streaming_samples(
    data_dir: str,
    n_samples: int = 10,
    repo_id: str = "nvidia/LIBERO_LeRobot_v3",
) -> tuple:
    """Load samples from local LIBERO data directory."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError("Need lerobot for streaming: pip install lerobot")

    ds = LeRobotDataset(repo_id, root=data_dir, split="train")
    indices = sorted(np.random.choice(len(ds), min(n_samples, len(ds)), replace=False))

    frames, trajs, texts = [], [], []
    for idx in indices:
        sample = ds[idx]
        # Camera frame
        for key in sample:
            if "images" in str(key):
                img = sample[key]
                if isinstance(img, torch.Tensor):
                    if img.dim() == 4:   # (T, C, H, W)
                        img = img[-1]
                    if img.dim() == 3 and img.shape[0] in (1, 3):
                        img = img.permute(1, 2, 0)
                frames.append(img.to(torch.uint8) if img.dtype != torch.uint8 else img)
                break

        # State → trajectory window (repeat single frame)
        state = sample.get("observation.state", sample.get("state", None))
        if state is not None:
            if isinstance(state, torch.Tensor) and state.dim() == 2:
                state = state[-1]
            s = state[:6].float() if state.shape[-1] >= 6 else torch.cat([state.float(), torch.zeros(6 - state.shape[-1])])
            trajs.append(s.unsqueeze(0).repeat(10, 1))  # (K, 6)

        # Text
        task = sample.get("task", sample.get("language_instruction", "pick and place"))
        texts.append(str(task))

    B = len(frames)
    return (
        torch.stack(frames) if frames else torch.zeros(0, 224, 224, 3, dtype=torch.uint8),
        torch.stack(trajs) if trajs else torch.zeros(0, 10, 6),
        texts,
    )


def print_report(results: dict, texts: list[str], prefix: str = ""):
    """Pretty-print cosine similarity results."""
    cos_vt = results["cos_vt"].cpu().numpy()
    cos_vl = results["cos_vl"].cpu().numpy()
    cos_tl = results["cos_tl"].cpu().numpy()
    # α = min of all three (the gating signal)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate ALIGN contrastive backbone")
    parser.add_argument("--checkpoint", required=True, help="Pretrained checkpoint (.pt)")
    parser.add_argument("--data-dir", help="Local data directory (LIBERO)")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare-texts", action="store_true",
                        help="Compare each sample against multiple texts")
    parser.add_argument("--texts", default="pick up the mug,pick and place,grasp the object",
                        help="Comma-separated text list for comparison")
    parser.add_argument("--frame", help="Single image file path")
    parser.add_argument("--pose", help="Single pose as space-separated numbers")
    parser.add_argument("--text", help="Single text query")
    args = parser.parse_args()

    model = load_backbone(args.checkpoint, args.device)

    if args.data_dir:
        # ── Batch evaluation from LIBERO data ──
        frames, trajs, texts = load_streaming_samples(args.data_dir, args.n_samples)
        frames = frames.to(args.device)
        trajs = trajs.float().to(args.device)

        results = get_embeddings(model, frames, trajs, texts)
        print_report(results, texts)

        # Multi-text comparison
        if args.compare_texts:
            alt_texts = [t.strip() for t in args.texts.split(",")]
            print(f"\n{'='*70}")
            print("MULTI-TEXT COMPARISON: first sample against all texts")
            print(f"{'='*70}")
            for t in alt_texts:
                r = get_embeddings(model, frames[:1], trajs[:1], [t])
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
        traj = pose_vals.unsqueeze(0).repeat(10, 1).unsqueeze(0).to(args.device)

        results = get_embeddings(model, img.unsqueeze(0), traj, [args.text])
        print_report(results, [args.text])

    else:
        print("Provide --data-dir for batch eval, or --frame --pose --text for single eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()