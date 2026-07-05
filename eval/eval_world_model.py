#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained WorldModel on three diagnostic tests.

Tests:
    1. In-distribution dynamics accuracy
       For real (s, a, s') transitions, check that the world model
       predicts s' close to the actual next state (cosine similarity).
       Target: > 0.95 for most samples.

    2. OOD sanity (random actions)
       Apply a random action to a real state. The predicted next state
       must have a reasonable norm (not exploding to infinity).
       Target: ||s'|| < 10x mean(real ||s||).

    3. Multi-step rollout
       Start from a real state s_0 and roll out K steps using real
       actions. Compare the rolled-out trajectory to the real one.
       Target: > 0.8 cosine at step 5, > 0.6 at step 10.

Usage:
    python eval/eval_world_model.py \\
        --checkpoint checkpoints/world_model/libero_spatial/run_1/world_model_best.pt \\
        --data /path/to/libero_spatial.h5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.world_model import create_world_model
from data.align_dataset import ALIGNDataset, MultiALIGNDataset

# world_model_collate is preferred when available.
try:
    from data.align_dataset import world_model_collate  # noqa: F401
    _HAS_WM_COLLATE = True
except ImportError:
    _HAS_WM_COLLATE = False


# ================================================================
# Fallback collate (matches training/train_world_model.py)
# ================================================================

def _fallback_world_model_collate(batch: list, chunk_size: int = 1) -> dict:
    """Build (s_t, a_t, s_{t+1}) triples from each batch item.

    Returns frame windows (B, K, H, W, 3) matching the training pipeline.
    """
    all_frames_t, all_traj_t = [], []
    all_frames_next, all_traj_next = [], []
    all_actions, all_texts = [], []
    all_ep_idx = []
    rng = np.random.default_rng()

    for item in batch:
        frames = item["frames"]
        poses = item["poses"][..., :6]
        actions = item.get("actions", None)
        text = item["text"]
        ep_idx = item.get("ep_idx", -1)

        N = len(frames)
        max_t = max(0, N - 2)
        t = int(rng.integers(0, max_t + 1)) if max_t > 0 else 0

        # Frame window ending at t (length chunk_size)
        start_f = max(0, t - chunk_size + 1)
        frame_window = frames[start_f:t + 1]
        if len(frame_window) < chunk_size:
            pad = np.zeros((chunk_size - len(frame_window), *frames.shape[1:]), dtype=frames.dtype)
            frame_window = np.concatenate([pad, frame_window], axis=0)
        all_frames_t.append(frame_window)

        # Trajectory window ending at t
        start_t = max(0, t - chunk_size + 1)
        traj_t = poses[start_t:t + 1]
        if len(traj_t) < chunk_size:
            pad = np.zeros((chunk_size - len(traj_t), 6), dtype=np.float32)
            traj_t = np.concatenate([pad, traj_t], axis=0)
        all_traj_t.append(traj_t.astype(np.float32))

        all_frames_next.append(frames[t + 1])
        start_n = max(0, t + 1 - chunk_size + 1)
        traj_next = poses[start_n:t + 2]
        if len(traj_next) < chunk_size:
            pad = np.zeros((chunk_size - len(traj_next), 6), dtype=np.float32)
            traj_next = np.concatenate([pad, traj_next], axis=0)
        all_traj_next.append(traj_next.astype(np.float32))

        if actions is not None and t < len(actions):
            act = actions[t, :6].astype(np.float32)
        else:
            act = np.zeros(6, dtype=np.float32)
        all_actions.append(act)
        all_texts.append(text)
        all_ep_idx.append(ep_idx)

    return {
        "frames_t": np.stack(all_frames_t, axis=0),       # (B, K, H, W, 3)
        "trajectory_t": np.stack(all_traj_t, axis=0),
        "frames_next": np.stack(all_frames_next, axis=0),
        "trajectory_next": np.stack(all_traj_next, axis=0),
        "action": np.stack(all_actions, axis=0),
        "texts": all_texts,
        "ep_idx": np.array(all_ep_idx, dtype=np.int64),
    }


# ================================================================
# Helpers
# ================================================================

def _percentiles(x: np.ndarray, ps=(5, 25, 50, 75, 95)) -> dict:
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def _joint_norm(z_v: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Concatenated (z_v, z_t) along the embedding dim → (B, 2D)."""
    return torch.cat([z_v, z_t], dim=-1)


# ================================================================
# Evaluation
# ================================================================

def evaluate(
    data_paths: List[str],
    world_model_checkpoint: str,
    encoder_checkpoint: str = None,
    batch_size: int = 64,
    traj_window: int = 20,
    chunk_size: int = 1,
    val_split: float = 0.1,
    device: str = None,
    use_bf16: bool = True,
    n_batches: int = 50,
    rollout_steps: int = 10,
    seed: int = 42,
    cameras: Optional[List[str]] = None,
) -> dict:
    """Run the three diagnostic tests on the world model.

    Args:
        data_paths: HDF5 datasets.
        world_model_checkpoint: Path to a world_model .pt (with
            ``world_model_state`` + ``config``).
        encoder_checkpoint: Path to the Phase 1b encoder+mixer
            checkpoint. Auto-detected if not given.
        batch_size: DataLoader batch size.
        traj_window: Trajectory window length for state encoding.
        chunk_size: Used by the fallback collate.
        val_split: Validation fraction of the dataset.
        device: 'cuda' or 'cpu'. Auto if None.
        use_bf16: BF16 autocast for the frozen encoder.
        n_batches: How many batches to use per test.
        rollout_steps: Number of steps for the multi-step rollout test.
        seed: RNG seed.

    Returns:
        dict with all metrics + per-test pass flags.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- Load world model checkpoint --
    wm_ckpt = torch.load(world_model_checkpoint, map_location=device)
    cfg = wm_ckpt.get("config", {})
    arch = cfg.get("arch", "mlp")
    embed_dim = cfg.get("embed_dim", 256)
    action_dim = cfg.get("action_dim", 6)
    print(f"  Loaded world model: {world_model_checkpoint}")
    print(f"    arch={arch}  embed_dim={embed_dim}  action_dim={action_dim}")
    print(f"    epoch={wm_ckpt.get('epoch', '?')}  loss={wm_ckpt.get('loss', '?'):.4f}")

    # -- Build world model head with the saved config --
    wm_kwargs: dict = {}
    _is_old_arch = False  # global flag for old single-timestep checkpoints
    if arch == "mlp":
        wm_kwargs = {
            "hidden_dim": cfg.get("mlp_hidden", 512),
            "num_layers": cfg.get("mlp_layers", 3),
        }
        # Detect old vs new architecture from state dict shape
        state = wm_ckpt.get("world_model_state", wm_ckpt)
        first_w = state.get("mlp.0.weight", None)
        if first_w is not None and first_w.shape[1] == 774:
            # Old architecture: single timestep (3*256+6 = 774)
            _is_old_arch = True
            wm_kwargs["window_size"] = 0
            print(f"    Detected OLD architecture (single timestep, input_dim=774)")
        else:
            # New architecture: window of K timesteps
            wm_kwargs["window_size"] = cfg.get("window_size", 5)
    elif arch == "rnn":
        # Auto-detect num_rnn_layers from the saved state_dict.
        # The saved config may have the wrong value (e.g., the
        # training script hardcoded 1 layer), so we count the
        # `_lN` suffixes in keys like "gru.weight_ih_lN" to find
        # the actual number of layers.
        max_l = 0
        for k in wm_ckpt.get("world_model_state", {}).keys():
            if k.startswith("gru.weight_ih_l"):
                try:
                    l = int(k.split("_l")[-1])
                    max_l = max(max_l, l + 1)
                except ValueError:
                    pass
        wm_kwargs = {
            "hidden_dim": cfg.get("rnn_hidden_dim",
                                   cfg.get("mlp_hidden", 256)),
            "num_rnn_layers": max_l if max_l > 0 else cfg.get("num_rnn_layers", 1),
        }
    elif arch == "transformer":
        wm_kwargs = {
            "d_model": cfg.get("transformer_d_model", 384),
            "nhead": cfg.get("transformer_nhead", 4),
            "num_layers": cfg.get("transformer_layers", 2),
            "dim_feedforward": cfg.get("transformer_dim_ff", 1024),
            "dropout": cfg.get("transformer_dropout", 0.0),
        }
    # Auto-detect window_size for RNN/Transformer from state_dict
    # if not in config (older training runs may not have saved it).
    if arch in ("rnn", "transformer"):
        if "window_size" not in wm_kwargs:
            wm_kwargs["window_size"] = cfg.get("window_size", 5)
    world_model = create_world_model(
        arch=arch, embed_dim=embed_dim, action_dim=action_dim, **wm_kwargs,
    ).to(device)
    world_model.load_state_dict(wm_ckpt["world_model_state"])
    world_model.eval()

    # -- Auto-detect encoder checkpoint if needed --
    if encoder_checkpoint is None or not Path(encoder_checkpoint).exists():
        # Try the path stored in the world model config
        stored = cfg.get("pretrained_checkpoint")
        if stored and Path(stored).exists():
            encoder_checkpoint = stored
        else:
            # Try ../<name>/run_2/best.pt relative to the world model dir
            wm_dir = Path(world_model_checkpoint).resolve().parent
            for run in sorted(wm_dir.parent.parent.glob("*/run_*")):
                cand = run / "best.pt"
                if cand.exists():
                    encoder_checkpoint = str(cand)
                    break
            if encoder_checkpoint is None:
                raise FileNotFoundError(
                    "Could not find encoder checkpoint. Pass --encoder-checkpoint."
                )
    print(f"  Encoder checkpoint: {encoder_checkpoint}")

    enc_ckpt = torch.load(encoder_checkpoint, map_location=device)
    enc_phase = enc_ckpt.get("phase", "?")
    print(f"    phase={enc_phase}  epoch={enc_ckpt.get('epoch', '?')}")

    # -- ALIGNModel for the frozen encoder+mixer --
    # IMPORTANT: read mixer_dim from the ENCODER checkpoint's config,
    # not from the world model config. The world model was trained
    # against a specific encoder with a specific mixer_dim. If the
    # encoder checkpoint has been retrained with a different mixer_dim
    # (e.g., 1280 instead of 512), using the world model's recorded
    # mixer_dim causes a shape mismatch in the mixer.
    enc_cfg = enc_ckpt.get("config", {}) if isinstance(enc_ckpt, dict) else {}
    align = ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        mixer_dim=enc_cfg.get("mixer_dim", cfg.get("mixer_dim", 512)),
        num_mixer_blocks=enc_cfg.get("num_mixer_blocks", cfg.get("num_mixer_blocks", 2)),
        num_cameras=len(cameras) if cameras else 1,
    ).to(device)
    if "trainable_state_dict" in enc_ckpt:
        align.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    else:
        align.load_state_dict(enc_ckpt, strict=False)
    align.freeze_backbone()
    align.freeze_all_encoders()
    align.eval()

    # -- Pick collate function --
    if _HAS_WM_COLLATE:
        from data.align_dataset import world_model_collate as wm_collate
        collate_fn = lambda b: wm_collate(b, traj_window=traj_window)
    else:
        collate_fn = lambda b: _fallback_world_model_collate(b, chunk_size=chunk_size)

    # -- Dataset (validation tail) --
    if len(data_paths) == 1:
        ds = ALIGNDataset(data_paths[0], mode="head", traj_window=traj_window,
                          cameras=cameras)
    else:
        ds = MultiALIGNDataset(data_paths, mode="head", traj_window=traj_window,
                               cameras=cameras)
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    indices = list(range(n_total - n_val, n_total))
    print(f"  Dataset: {data_paths}  (N={n_total}, val={n_val})")
    print(f"  Device: {device}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(indices),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # =============================================================
    # Accumulator containers
    # =============================================================

    # Test 1: real (s, a, s') — cosine sim to predicted s'
    test1_cos_joint: List[float] = []   # cos on (z_v', z_t') concatenated
    test1_cos_v: List[float] = []
    test1_cos_t: List[float] = []
    test1_pred_norm: List[float] = []   # ||predicted s'||
    test1_real_norm: List[float] = []   # ||real s'||

    # Test 2: random-action predictions — norm only
    test2_random_norm: List[float] = []

    # Test 3: rollout — per-step cosine
    rollout_steps_kept = max(2, min(rollout_steps, 10))
    test3_cos_per_step: List[List[float]] = [[] for _ in range(rollout_steps_kept)]

    # -- Get a representative sample of action vectors for Test 2 --
    real_action_samples: List[np.ndarray] = []

    def _encode_state_window(frames: torch.Tensor,
                             traj: torch.Tensor,
                             texts,
                             window_size: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode state with K-frame window for world model input.

        Returns (z_v_window, z_t_window, z_text) all (B, W, D) except z_text (B, D).
        Handles both single-cam (B, K, H, W, 3) and multi-cam (B, K, V, H, W, 3).
        """
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            # Handle single frame (4D) by wrapping
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            # Use encode_raw_vision_window which handles both 5D and 6D
            z_v = align.encode_raw_vision_window(frames)  # (B, K, D)

            z_t_tokens = align.encode_raw_trajectory_tokens(traj)
            z_text = align.encode_raw_text(texts)
            if z_text is None:
                z_text = torch.zeros_like(z_v[:, 0])

            z_v, z_t_tokens, z_text = align.cross_attention_mixer(
                z_v, z_t_tokens, z_text
            )

            z_v = z_v[:, -window_size:]
            z_t_tokens = z_t_tokens[:, -window_size:]

        return z_v, z_t_tokens, z_text

    def _encode_state_target(frames: torch.Tensor,
                              traj: torch.Tensor,
                              texts) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode target state (single embedding, mean-pooled)."""
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_bf16
        ):
            mixed = align.encode_mixed(frames, traj, texts)
        return mixed["z_v"].float(), mixed["z_t"].float()

    def _batch_to_tensors(batch: dict) -> dict:
        """Normalize batch keys (frame_t vs frames_t) and to-device."""
        if "frame_t" in batch:
            result = {
                "frames_t": torch.from_numpy(batch["frame_t"]).to(device),
                "traj_t": torch.from_numpy(batch["traj_t"]).float().to(device),
                "frames_next": torch.from_numpy(batch["frame_next"]).to(device),
                "traj_next": torch.from_numpy(batch["traj_next"]).float().to(device),
                "action": torch.from_numpy(batch["action"]).float().to(device),
                "texts": batch["text"],
            }
        else:
            result = {
                "frames_t": torch.from_numpy(batch["frames_t"]).to(device),
                "traj_t": torch.from_numpy(batch["trajectory_t"]).float().to(device),
                "frames_next": torch.from_numpy(batch["frames_next"]).to(device),
                "traj_next": torch.from_numpy(batch["trajectory_next"]).float().to(device),
                "action": torch.from_numpy(batch["action"]).float().to(device),
                "texts": batch["texts"],
            }
        if "ep_idx" in batch:
            result["ep_idx"] = batch["ep_idx"]
        return result

    n_seen_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= n_batches:
                break
            n_seen_batches += 1

            t = _batch_to_tensors(batch)
            frames_t, traj_t = t["frames_t"], t["traj_t"]
            frames_next, traj_next = t["frames_next"], t["traj_next"]
            action = t["action"]
            texts = t["texts"]
            B = frames_t.shape[0]

            real_action_samples.append(action.detach().cpu().numpy().copy())

            # -- Encode state_t and state_t+1 --
            if _is_old_arch:
                # Old architecture: single embeddings (B, D)
                z_v, z_t, z_text = _encode_state_target(frames_t, traj_t, texts)
                z_v_next, z_t_next = _encode_state_target(frames_next, traj_next, texts)
            else:
                z_v, z_t, z_text = _encode_state_window(frames_t, traj_t, texts, window_size=cfg.get("window_size", 5))
                z_v_next, z_t_next = _encode_state_target(frames_next, traj_next, texts)

            # ============== TEST 1: real (s, a) -> predicted s' ====
            z_v_pred, z_t_pred = world_model(z_v, z_t, z_text, action)
            pred_joint = _joint_norm(z_v_pred, z_t_pred)
            real_joint = _joint_norm(z_v_next, z_t_next)
            cos_joint = F.cosine_similarity(pred_joint, real_joint, dim=-1)
            cos_v = F.cosine_similarity(z_v_pred, z_v_next, dim=-1)
            cos_t = F.cosine_similarity(z_t_pred, z_t_next, dim=-1)
            test1_cos_joint.extend(cos_joint.cpu().tolist())
            test1_cos_v.extend(cos_v.cpu().tolist())
            test1_cos_t.extend(cos_t.cpu().tolist())
            test1_pred_norm.extend(pred_joint.norm(dim=-1).cpu().tolist())
            test1_real_norm.extend(real_joint.norm(dim=-1).cpu().tolist())

            # ============== TEST 2: random-action norm =============
            # Build a random action from the empirical distribution in this batch.
            action_mean = action.mean(dim=0, keepdim=True).expand_as(action)
            action_std = action.std(dim=0, keepdim=True).clamp_min(1e-3)
            rand_action = action_mean + action_std * torch.randn_like(action)
            z_v_rand, z_t_rand = world_model(z_v, z_t, z_text, rand_action)
            pred_rand = _joint_norm(z_v_rand, z_t_rand)
            test2_random_norm.extend(pred_rand.norm(dim=-1).cpu().tolist())

            # ============== TEST 3: multi-step rollout =============
            # We re-use the dataset directly (NOT the batch's collated items)
            # to get a clean episode window. For each batch item we already
            # have (z_v, z_t, z_text) at the anchor t, and the corresponding
            # `action` for stepping forward. We need:
            #   - the source episode (frames, poses, actions)
            #   - the anchor t chosen by the collate
            # The collate samples t randomly per-item. Recover it by
            # locating the frame in the source episode that matches the
            # frame in the batch. To keep this O(K) and reliable, use
            # the first frame that matches.
            n_rollout = min(B, 16)
            for i in range(n_rollout):
                # Get the source ep_id from the collate
                if "ep_idx" in batch:
                    source_ep_id = int(batch["ep_idx"][i])
                else:
                    continue
                # Read the FULL episode from the HDF5 file directly,
                # not via the chunked __getitem__. This avoids the
                # chunk size limit and gives us the actual anchor t
                # to use for the rollout. Currently only supports
                # single-dataset eval (MultiALIGNDataset skipped).
                if not hasattr(ds, "_h5") or not hasattr(ds, "_episode_keys"):
                    continue
                if hasattr(ds, "datasets"):
                    # MultiALIGNDataset: skip for now (would need
                    # to track which sub-dataset the batch came from)
                    continue
                if source_ep_id >= len(ds._episode_keys):
                    continue
                ep_key = ds._episode_keys[source_ep_id]
                try:
                    # Read full frames from the camera group.
                    # Use the first camera from the dataset's cameras list.
                    camera_name = ds.cameras[0] if ds.cameras else "image"
                    full_frames = ds._h5[f"{ep_key}/frames/{camera_name}"][:]
                    # Use the dataset's offset-aware _read_poses and
                    # _read_actions methods to get the trajectory
                    # window and actions. These handle the cumulative
                    # pose layout correctly.
                    N = len(full_frames)
                    full_poses = ds._read_poses(source_ep_id, 0, N)
                    if getattr(ds, "_has_actions", False):
                        full_actions = ds._read_actions(source_ep_id, 0, N)[:, :6]
                    else:
                        full_actions = np.zeros((N, 6), dtype=np.float32)
                except Exception as e:
                    print(f"  Error reading episode {source_ep_id} (key={ep_key}): {e}", flush=True)
                    continue
                # Now find the anchor t by hash-matching
                # frames_t[i] is (K, H, W, 3) if window, or (H, W, 3) if single frame
                if frames_t.dim() == 5:
                    target_hash = int(frames_t[i, -1].cpu().numpy().sum())
                else:
                    target_hash = int(frames_t[i].cpu().numpy().sum())
                chosen_t = None
                for t in range(len(full_frames)):
                    if int(full_frames[t].sum()) == target_hash:
                        if full_frames[t].shape == (frames_t.shape[-3], frames_t.shape[-2], frames_t.shape[-1]):
                            chosen_t = t
                            break
                if chosen_t is None:
                    # Debug: print first few frame hashes to diagnose
                    if i == 0 and step == 0:
                        print(f"  [debug] target_hash={target_hash}, full_frames range=[0,{len(full_frames)-1}]")
                        for t in range(min(5, len(full_frames))):
                            print(f"  [debug]   full_frames[{t}].sum()={int(full_frames[t].sum())}")
                    continue
                # Check if we have enough frames for K-step rollout
                if chosen_t + rollout_steps_kept >= len(full_frames):
                    # Not enough — use what we have
                    K = min(rollout_steps_kept, len(full_frames) - chosen_t - 1)
                else:
                    K = rollout_steps_kept
                if K < 1:
                    continue
                frames_i = full_frames
                poses_i = full_poses
                actions_i = full_actions

                K = rollout_steps_kept
                # Build input batch of 1 for the rollout, anchored at chosen_t
                if _is_old_arch:
                    # Old: single embedding (1, D) — unsqueeze to (1, 1, D)
                    z_v_cur = z_v[i:i + 1].unsqueeze(1).clone()
                    z_t_cur = z_t[i:i + 1].unsqueeze(1).clone()
                else:
                    z_v_cur = z_v[i:i + 1].clone()       # (1, W, D)
                    z_t_cur = z_t[i:i + 1].clone()       # (1, W, D)
                z_text_cur = z_text[i:i + 1].clone()  # (1, D)
                text_i = [texts[i]] if isinstance(texts, list) else [texts]
                for k in range(K):
                    if chosen_t + k + 1 >= len(frames_i):
                        break
                    # Apply the next real action
                    a_real = torch.from_numpy(
                        actions_i[chosen_t + k, :6].astype(np.float32)
                    ).to(device).unsqueeze(0)
                    z_v_pred, z_t_pred = world_model(
                        z_v_cur, z_t_cur, z_text_cur, a_real
                    )
                    # Slide window: drop oldest, append predicted
                    z_v_cur = torch.cat([z_v_cur[:, 1:], z_v_pred.unsqueeze(1)], dim=1)
                    z_t_cur = torch.cat([z_t_cur[:, 1:], z_t_pred.unsqueeze(1)], dim=1)
                    # Encode the real next state (k+1 steps ahead)
                    f_real = torch.from_numpy(
                        frames_i[chosen_t + k + 1]
                    ).to(device).unsqueeze(0)
                    end_p = min(chosen_t + k + 1 + 1, len(poses_i))
                    start_p = max(0, end_p - traj_window)
                    p_real = torch.from_numpy(
                        poses_i[start_p:end_p].astype(np.float32)
                    ).to(device).unsqueeze(0)
                    # Pad if too short
                    if p_real.shape[1] < traj_window:
                        pad = torch.zeros(
                            1, traj_window - p_real.shape[1], 6,
                            device=device,
                        )
                        p_real = torch.cat([pad, p_real], dim=1)
                    z_v_real_next, z_t_real_next = _encode_state_target(
                        f_real, p_real, text_i
                    )
                    real_joint_k = _joint_norm(z_v_real_next, z_t_real_next)
                    pred_joint_k = _joint_norm(z_v_pred, z_t_pred)
                    cos_k = F.cosine_similarity(
                        pred_joint_k, real_joint_k, dim=-1
                    ).item()
                    test3_cos_per_step[k].append(cos_k)

    elapsed = time.time() - t0

    # =============================================================
    # Build report
    # =============================================================

    def _to_np(x: List[float]) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    report: dict = {}

    # --- Test 1 ---
    cos_joint = _to_np(test1_cos_joint)
    cos_v = _to_np(test1_cos_v)
    cos_t = _to_np(test1_cos_t)
    real_norm = _to_np(test1_real_norm)
    pred_norm = _to_np(test1_pred_norm)

    # Per-sample pass: cosine >= 0.95 on the joint embedding
    t1_pass_rate = float((cos_joint >= 0.95).mean()) if cos_joint.size else 0.0
    # Aggregate pass: mean >= 0.95
    t1_pass = bool((cos_joint.mean() >= 0.95) if cos_joint.size else False)

    report["test1_in_distribution"] = {
        "n_samples": int(cos_joint.size),
        "cos_joint_mean": float(cos_joint.mean()) if cos_joint.size else None,
        "cos_joint_percentiles": _percentiles(cos_joint) if cos_joint.size else {},
        "pass_rate_at_0.95": t1_pass_rate,
        "cos_v_mean": float(cos_v.mean()) if cos_v.size else None,
        "cos_t_mean": float(cos_t.mean()) if cos_t.size else None,
        "pred_norm_mean": float(pred_norm.mean()) if pred_norm.size else None,
        "real_norm_mean": float(real_norm.mean()) if real_norm.size else None,
        "target": "mean cos >= 0.95",
        "passed": t1_pass,
    }

    # --- Test 2 ---
    rand_norm = _to_np(test2_random_norm)
    real_norm_mean = float(real_norm.mean()) if real_norm.size else 1.0
    rand_norm_mean = float(rand_norm.mean()) if rand_norm.size else float("inf")
    t2_pass = bool(rand_norm_mean < 10.0 * real_norm_mean) if rand_norm.size else False

    report["test2_ood_sanity"] = {
        "n_samples": int(rand_norm.size),
        "random_action_norm_mean": rand_norm_mean,
        "real_action_norm_mean": float(pred_norm.mean()) if pred_norm.size else None,
        "real_state_norm_mean": real_norm_mean,
        "ratio_random_over_real": (rand_norm_mean / real_norm_mean)
                                   if real_norm_mean > 0 else None,
        "target": "random ||s'|| < 10x mean(real ||s||)",
        "passed": t2_pass,
    }

    # --- Test 3 ---
    t3_per_step_mean = []
    for k in range(rollout_steps_kept):
        v = test3_cos_per_step[k]
        t3_per_step_mean.append(_mean(v) if v else float("nan"))

    # targets: step 5 > 0.8, step 10 > 0.6 (indexes 4 and 9 if rollout_steps_kept=10)
    def _step(k: int) -> float:
        return t3_per_step_mean[k] if k < len(t3_per_step_mean) else float("nan")

    step_targets = {
        "step_1": (_step(0), 0.95),
        "step_5": (_step(4), 0.80),
        "step_10": (_step(9), 0.60),
    }
    # Aggregate pass: ALL listed targets meet their threshold
    t3_pass = all(
        (np.isnan(v) or v >= thr) for v, thr in step_targets.values()
    )

    report["test3_rollout"] = {
        "n_trajectories_per_step": [
            len(test3_cos_per_step[k]) for k in range(rollout_steps_kept)
        ],
        "cos_per_step_mean": t3_per_step_mean,
        "step_1": {"mean_cos": _step(0), "target": 0.95,
                   "passed": bool(not np.isnan(_step(0)) and _step(0) >= 0.95)},
        "step_5": {"mean_cos": _step(4), "target": 0.80,
                   "passed": bool(not np.isnan(_step(4)) and _step(4) >= 0.80)},
        "step_10": {"mean_cos": _step(9), "target": 0.60,
                    "passed": bool(not np.isnan(_step(9)) and _step(9) >= 0.60)},
        "target": "step 5 > 0.8, step 10 > 0.6",
        "passed": t3_pass,
    }

    # --- Overall ---
    overall = t1_pass and t2_pass and t3_pass
    report["overall"] = {
        "test1_pass": t1_pass,
        "test2_pass": t2_pass,
        "test3_pass": t3_pass,
        "passed": overall,
        "n_batches": n_seen_batches,
        "elapsed_seconds": elapsed,
    }

    _print_report(report)
    return report


# ================================================================
# Pretty-print
# ================================================================

def _print_report(report: dict) -> None:
    print("\n" + "=" * 64)
    print("  WORLD MODEL EVALUATION REPORT")
    print("=" * 64)

    t1 = report["test1_in_distribution"]
    print("\n[Test 1] In-distribution dynamics accuracy")
    print(f"  N samples:                  {t1['n_samples']}")
    print(f"  Cosine (joint) mean:        {t1['cos_joint_mean']:.4f}")
    if t1["cos_joint_percentiles"]:
        p = t1["cos_joint_percentiles"]
        print(f"  Cosine (joint) percentiles: "
              f"p5={p.get('p5', float('nan')):.3f}  "
              f"p25={p.get('p25', float('nan')):.3f}  "
              f"p50={p.get('p50', float('nan')):.3f}  "
              f"p75={p.get('p75', float('nan')):.3f}  "
              f"p95={p.get('p95', float('nan')):.3f}")
    print(f"  Cosine v mean:              {t1['cos_v_mean']:.4f}")
    print(f"  Cosine t mean:              {t1['cos_t_mean']:.4f}")
    print(f"  Pass rate (>=0.95):         {t1['pass_rate_at_0.95']:.1%}")
    print(f"  Target:                     {t1['target']}")
    status = "PASS ✓" if t1["passed"] else "FAIL ✗"
    print(f"  Result:                     {status}")

    t2 = report["test2_ood_sanity"]
    print("\n[Test 2] OOD sanity (random actions)")
    print(f"  N samples:                  {t2['n_samples']}")
    print(f"  Random-action ||s'|| mean:  {t2['random_action_norm_mean']:.3f}")
    print(f"  Real-state ||s|| mean:      {t2['real_state_norm_mean']:.3f}")
    if t2["ratio_random_over_real"] is not None:
        print(f"  Ratio (random/real):        {t2['ratio_random_over_real']:.2f}x")
    print(f"  Target:                     {t2['target']}")
    status = "PASS ✓" if t2["passed"] else "FAIL ✗"
    print(f"  Result:                     {status}")

    t3 = report["test3_rollout"]
    print("\n[Test 3] Multi-step rollout")
    print(f"  Trajectories per step:      {t3['n_trajectories_per_step']}")
    print(f"  Cosine per step (mean):     "
          + "  ".join(f"k={k+1}: {v:.3f}"
                     for k, v in enumerate(t3['cos_per_step_mean'])))
    for label in ("step_1", "step_5", "step_10"):
        sub = t3[label]
        print(f"  {label:8s} mean cos: {sub['mean_cos']:.4f}  "
              f"(target >= {sub['target']:.2f})  "
              f"-> {'PASS' if sub['passed'] else 'FAIL'}")
    print(f"  Target:                     {t3['target']}")
    status = "PASS ✓" if t3["passed"] else "FAIL ✗"
    print(f"  Result:                     {status}")

    print("\n" + "-" * 64)
    overall = report["overall"]
    summary = (
        f"Test 1: {'PASS' if overall['test1_pass'] else 'FAIL'}  |  "
        f"Test 2: {'PASS' if overall['test2_pass'] else 'FAIL'}  |  "
        f"Test 3: {'PASS' if overall['test3_pass'] else 'FAIL'}"
    )
    print(f"  Summary: {summary}")
    print(f"  Overall: {'PASS ✓' if overall['passed'] else 'FAIL ✗'}")
    print(f"  ({overall['n_batches']} batches in {overall['elapsed_seconds']:.1f}s)")
    print("=" * 64)


# ================================================================
# CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ALIGN WorldModel (3 diagnostic tests)"
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to world_model_best.pt")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Path to the Phase 1b encoder+mixer checkpoint "
                             "(auto-detected if not given)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Max number of batches per test")
    parser.add_argument("--rollout-steps", type=int, default=10,
                        help="Number of rollout steps in Test 3 (capped at 10)")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None,
                        help="Optional path to write the full report as JSON.")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views to use (e.g. 'wrist_image image'). "
                             "Must match the cameras used during pretrain.")
    args = parser.parse_args()

    report = evaluate(
        data_paths=args.data,
        world_model_checkpoint=args.checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        batch_size=args.batch_size,
        traj_window=args.traj_window,
        chunk_size=args.chunk_size,
        val_split=args.val_split,
        device=args.device,
        use_bf16=args.bf16,
        n_batches=args.n_batches,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        cameras=args.cameras,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report written to {args.output_json}")


if __name__ == "__main__":
    main()
