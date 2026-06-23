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
from tqdm import tqdm

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
    actions: Optional[torch.Tensor] = None,
) -> dict:
    """Run full forward pass through heads — Phase 2 style.

    Args:
        frames: (B, H, W, 3) uint8.
        trajs: (B, K, 6) trajectory windows.
        texts: list of str, length B.
        actions: (B, 6) current actions (delta) for the Assistant head.
                 Optional — falls back to last pose if not provided (for
                 backward-compat with tests that don't have actions).

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
    if actions is None:
        # Backward-compat fallback: use last pose in the trajectory window
        actions = trajs[:, -1]
    delta = model.assistant_head(z_v, z_t, z_text, actions)

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

def load_episodes_from_hdf5(
    h5_path: str,
    n_episodes: int = 5,
    traj_window: int = 20,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """Load full episodes from HDF5 for temporal window sampling.

    Returns:
        (all_frames, all_poses, all_texts) where each is a list of arrays
        per episode. all_frames[i].shape = (T, H, W, 3), all_poses[i].shape = (T, 6).
    """
    import h5py, json
    all_frames, all_poses, all_texts = [], [], []
    with h5py.File(h5_path, "r") as h5:
        ep_keys = sorted([k for k in h5.keys() if k.startswith("ep_")])
        selected = sorted(np.random.choice(len(ep_keys), min(n_episodes, len(ep_keys)), replace=False))
        for idx in selected:
            ep = h5[ep_keys[idx]]
            text = json.loads(ep["texts"][()])[0]
            # Frames
            frames_group = ep["frames"]
            cam = "wrist_image" if "wrist_image" in frames_group else list(frames_group.keys())[0]
            frames = frames_group[cam][:]
            # Poses
            poses = ep["poses"][:, :6]
            all_frames.append(frames)
            all_poses.append(poses)
            all_texts.append(text)
    return all_frames, all_poses, all_texts


def sample_windows(
    all_frames: List[np.ndarray],
    all_poses: List[np.ndarray],
    all_texts: List[str],
    n_samples: int = 20,
    traj_window: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Sample (frame, trajectory_window, text) tuples from episodes.

    Each sample picks a random episode, a random timestep t, and extracts:
      - frame at t
      - trajectory window [t, t+traj_window) of consecutive poses
      - the episode's text

    Returns:
        frames: (B, H, W, 3) uint8
        trajs:  (B, K, 6) float32
        texts:  list of str
    """
    frames_list, trajs_list, texts_list = [], [], []
    for _ in range(n_samples):
        ep_idx = np.random.randint(len(all_frames))
        ep_frames = all_frames[ep_idx]
        ep_poses = all_poses[ep_idx]
        T = len(ep_frames)
        if T < traj_window + 1:
            continue
        t = np.random.randint(0, T - traj_window)
        # Frame at t
        frame = ep_frames[t]
        # Trajectory window [t, t+traj_window)
        traj = ep_poses[t:t + traj_window]
        frames_list.append(frame)
        trajs_list.append(traj)
        texts_list.append(all_texts[ep_idx])
    if not frames_list:
        return torch.zeros(0, 224, 224, 3, dtype=torch.uint8), torch.zeros(0, traj_window, 6), []
    return (
        torch.from_numpy(np.stack(frames_list)),
        torch.from_numpy(np.stack(trajs_list)).float(),
        texts_list,
    )


def build_misalignment_pairs(
    all_frames: List[np.ndarray],
    all_poses: List[np.ndarray],
    all_texts: List[str],
    n_pairs: int = 20,
    traj_window: int = 20,
) -> dict:
    """Build aligned and misaligned (frame, trajectory) pairs for contrastive eval.

    Returns:
        dict with keys:
            "aligned": (frames, trajs, texts) — same episode, same timestep
            "wrong_ep": (frames, trajs, texts) — frame from ep A, traj from ep B
            "wrong_time": (frames, trajs, texts) — frame at t, traj at t+offset
            "wrong_text": (frames, trajs, texts) — frame+traj from ep A, text from ep B
    """
    aligned_f, aligned_t, aligned_txt = sample_windows(all_frames, all_poses, all_texts, n_pairs, traj_window)

    # Wrong episode: frame from ep A, trajectory from ep B
    wep_f, wep_t, wep_txt = [], [], []
    for _ in range(n_pairs):
        a = np.random.randint(len(all_frames))
        b = np.random.randint(len(all_frames))
        while b == a:
            b = np.random.randint(len(all_frames))
        ep_a = all_frames[a]
        ep_b = all_poses[b]
        T_a, T_b = len(ep_a), len(ep_b)
        if T_a < 1 or T_b < traj_window:
            continue
        t_a = np.random.randint(0, T_a)
        t_b = np.random.randint(0, T_b - traj_window)
        wep_f.append(ep_a[t_a])
        wep_t.append(ep_b[t_b:t_b + traj_window])
        wep_txt.append(all_texts[a])
    wep_f = torch.from_numpy(np.stack(wep_f)) if wep_f else torch.zeros(0, 224, 224, 3, dtype=torch.uint8)
    wep_t = torch.from_numpy(np.stack(wep_t)).float() if wep_t else torch.zeros(0, traj_window, 6)

    # Wrong time: frame at t, trajectory at t+offset (same episode)
    wt_f, wt_t, wt_txt = [], [], []
    offset = traj_window  # shift by a full window
    for _ in range(n_pairs):
        ep_idx = np.random.randint(len(all_frames))
        ep_frames = all_frames[ep_idx]
        ep_poses = all_poses[ep_idx]
        T = len(ep_frames)
        if T < traj_window + offset + 1:
            continue
        t = np.random.randint(0, T - traj_window - offset)
        wt_f.append(ep_frames[t])
        wt_t.append(ep_poses[t + offset:t + offset + traj_window])
        wt_txt.append(all_texts[ep_idx])
    wt_f = torch.from_numpy(np.stack(wt_f)) if wt_f else torch.zeros(0, 224, 224, 3, dtype=torch.uint8)
    wt_t = torch.from_numpy(np.stack(wt_t)).float() if wt_t else torch.zeros(0, traj_window, 6)

    # Wrong text: frame+traj from ep A, text from ep B
    wl_f, wl_t, wl_txt = [], [], []
    for _ in range(n_pairs):
        a = np.random.randint(len(all_frames))
        b = np.random.randint(len(all_texts))
        while b == a:
            b = np.random.randint(len(all_texts))
        ep_a = all_frames[a]
        ep_poses = all_poses[a]
        T = len(ep_a)
        if T < traj_window + 1:
            continue
        t = np.random.randint(0, T - traj_window)
        wl_f.append(ep_a[t])
        wl_t.append(ep_poses[t:t + traj_window])
        wl_txt.append(all_texts[b])
    wl_f = torch.from_numpy(np.stack(wl_f)) if wl_f else torch.zeros(0, 224, 224, 3, dtype=torch.uint8)
    wl_t = torch.from_numpy(np.stack(wl_t)).float() if wl_t else torch.zeros(0, traj_window, 6)

    return {
        "aligned": (aligned_f, aligned_t, aligned_txt),
        "wrong_ep": (wep_f, wep_t, wep_txt),
        "wrong_time": (wt_f, wt_t, wt_txt),
        "wrong_text": (wl_f, wl_t, wl_txt),
    }


def load_libero_samples(
    data_dir: str,
    n_samples: int = 10,
    repo_id: str = "nvidia/LIBERO_LeRobot_v3",
    traj_window: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Load samples from local LIBERO data directory (legacy, single-timestep).

    Uses LeRobotDataset with pyav backend (no torchcodec dependency).

    Returns:
        (frames, trajs, texts) where:
            frames: (B, H, W, 3) uint8
            trajs:  (B, K, 6) float32 — each row is the SAME pose repeated K times
            texts:  list of str
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError("Need lerobot: pip install lerobot")

    ds = LeRobotDataset(repo_id, root=data_dir, video_backend="pyav")
    indices = sorted(np.random.choice(len(ds), min(n_samples, len(ds)), replace=False))

    frames, trajs, texts = [], [], []
    print(f"[load] Loading {len(indices)} samples from {data_dir}...")
    for idx in tqdm(indices, desc="Loading samples", unit="sample"):
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
    # alphas = np.minimum(np.minimum(cos_vt, cos_vl), cos_tl)
    alphas = (cos_vt + cos_vl + cos_tl) / 3

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
    parser.add_argument("--data", required=True, help="HDF5 dataset path")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--traj-window", type=int, default=20, help="Trajectory window K")
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to load from HDF5")
    parser.add_argument("--compare-raw-vs-mixed", action="store_true",
                        help="Compare raw encoder vs mixer outputs")
    parser.add_argument("--compare-texts", action="store_true",
                        help="Compare each sample against multiple texts")
    parser.add_argument("--texts", default="pick up the mug,pick and place,grasp the object",
                        help="Comma-separated text list for comparison")
    parser.add_argument("--eval-heads", action="store_true",
                        help="Evaluate full model with Decision + Assistant heads")
    parser.add_argument("--eval-misalignment", action="store_true",
                        help="Evaluate aligned vs misaligned pairs")
    parser.add_argument("--frame", help="Single image file path")
    parser.add_argument("--pose", help="Single pose as space-separated numbers")
    parser.add_argument("--text", help="Single text query")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)

    # ── Load episodes from HDF5 (used by all modes except single-sample) ──
    if args.data and not (args.frame and args.pose and args.text):
        print(f"[load] Loading {args.n_episodes} episodes from {args.data}...")
        all_frames, all_poses, all_texts = load_episodes_from_hdf5(
            args.data, args.n_episodes, args.traj_window)
        print(f"  Loaded {len(all_frames)} episodes")

    if args.eval_misalignment:
        pairs = build_misalignment_pairs(
            all_frames, all_poses, all_texts,
            n_pairs=args.n_samples, traj_window=args.traj_window)

        print(f"\n{'='*70}")
        print("MISALIGNMENT EVALUATION")
        print(f"{'='*70}")
        for name, (f, t, txt) in pairs.items():
            if len(f) == 0:
                print(f"  {name:15s}: no samples")
                continue
            f = f.to(args.device)
            t = t.float().to(args.device)
            results = get_mixed_embeddings(model, f, t, txt)
            cos_vt = results["cos_vt"].mean().item()
            cos_vl = results["cos_vl"].mean().item()
            cos_tl = results["cos_tl"].mean().item()
            print(f"  {name:15s}:  cos_vt={cos_vt:.4f}  cos_vl={cos_vl:.4f}  cos_tl={cos_tl:.4f}  (N={len(f)})")

        # Also run head eval if checkpoint has heads
        try:
            head_results = get_head_predictions(model, pairs["aligned"][0].to(args.device),
                                                pairs["aligned"][1].float().to(args.device),
                                                pairs["aligned"][2])
            print(f"\n  Head predictions (aligned):")
            print(f"    α={head_results['alpha'].mean().item():.4f}  |Δ|={head_results['delta'].norm(dim=-1).mean().item():.4f}")
        except Exception:
            pass

    elif args.data:
        # ── Standard evaluation from HDF5 ──
        frames, trajs, texts = sample_windows(
            all_frames, all_poses, all_texts,
            n_samples=args.n_samples, traj_window=args.traj_window)
        frames = frames.to(args.device)
        trajs = trajs.float().to(args.device)

        if args.eval_heads:
            results = get_head_predictions(model, frames, trajs, texts)
            print_head_report(results, texts)
        elif args.compare_raw_vs_mixed:
            raw = get_raw_embeddings(model, frames, trajs, texts)
            mixed = get_mixed_embeddings(model, frames, trajs, texts)
            print_report(raw, texts, prefix="[RAW] ")
            print_report(mixed, texts, prefix="[MIXED] ")
            print(f"\n  Improvement (mixed - raw):")
            print(f"    cos_vt: {mixed['cos_vt'].mean().item() - raw['cos_vt'].mean().item():+.4f}")
            print(f"    cos_vl: {mixed['cos_vl'].mean().item() - raw['cos_vl'].mean().item():+.4f}")
            print(f"    cos_tl: {mixed['cos_tl'].mean().item() - raw['cos_tl'].mean().item():+.4f}")
        else:
            results = get_mixed_embeddings(model, frames, trajs, texts)
            print_report(results, texts)

        # Multi-text comparison
        if args.compare_texts:
            alt_texts = [t.strip() for t in args.texts.split(",")]
            print(f"\n{'='*70}")
            print("MULTI-TEXT COMPARISON: first sample against all texts")
            print(f"{'='*70}")
            for t in tqdm(alt_texts, desc="Comparing texts", unit="text"):
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
        print("Provide --data for batch eval, or --frame --pose --text for single eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
