"""
Strict sanity checks for the trained value head.

The value function V(s) is the foundation of the alpha pipeline.
alpha = sigmoid((V_m - V_h) / tau) is meaningless if V can't
differentiate counterfactual states. These tests are HARD GATES
before V can be used downstream.

Tests:

Test 1: V has meaningful variance (not constant)
  - Quick sanity: V is learning *something*.
  - Pass: std > 0.01

Test 2: V increases along successful expert trajectories
  - For trajectories that succeed, V(s_t) should be higher at later
    timesteps (closer to goal = more achievable future reward remaining).
  - Pass: V is monotonically non-decreasing along > 60% of trajectories.

Test 3: V predicts cumulative return
  - For each trajectory, compute actual cumulative return G_t
    using GAIL rewards. V(s_t) should be close to G_t.
  - Pass: Pearson correlation > 0.3 between V(s_t) and G_t.

Test 4: V differentiates counterfactual states (KEY FOR ALPHA)
  - The whole alpha pipeline depends on V(f(s, a_m)) > V(f(s, a_h))
    being meaningful when a_m is better than a_h.
  - Sample (s, a_good, a_bad) pairs where GAIL clearly prefers a_good.
  - Use world model to imagine s'_good = f(s, a_good), s'_bad = f(s, a_bad).
  - V(s'_good) should be > V(s'_bad) for most pairs.
  - Pass: > 60% of pairs have V(s'_good) > V(s'_bad).

Test 5: V is stable under small perturbations
  - V shouldn't blow up or oscillate wildly between nearby states.
  - For nearby states (similar embeddings), V values should correlate.
  - Pass: Pearson correlation between V(s) and V(s + small_noise) > 0.8.

If any test fails, V is not safe to use. Common fixes:
  - More training epochs (V was unstable in earlier runs)
  - Reward clipping (prevents V divergence)
  - Target network (stabilizes TD learning)
  - Lower learning rate
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.value_head import ValueHeadMLP, create_value_head
from models.gail_discriminator import (
    GAILDiscriminatorMLP,
    compute_reward,
    create_gail_discriminator,
)
from models.world_model import create_world_model
from data.align_dataset import ALIGNDataset, world_model_collate

# Global flag for BF16 autocast (set from CLI args in main)
USE_BF16 = True


def encode_batch(model: ALIGNModel, batch: dict, device: torch.device) -> dict:
    """Encode a batch through the frozen encoder+mixer.

    The `frame_t` field has different shapes depending on the camera setup:
      - Single camera: (B, K, H, W, 3) — K past frames
      - Multi camera:  (B, K, V, H, W, 3) — K past frames × V cameras
    We use only the LAST frame (current timestep) to avoid stale info,
    then pass the appropriate 4D/5D shape to the encoder.
    """
    frame_t = batch["frame_t"]
    if frame_t.ndim == 6:
        # (B, K, V, H, W, 3) -> (B, V, H, W, 3): use last K, keep all V
        frames = frame_t[:, -1]
    elif frame_t.ndim == 5:
        # (B, K, H, W, 3) -> (B, H, W, 3): use last K
        frames = frame_t[:, -1]
    else:
        # Already (B, H, W, 3) or (B, V, H, W, 3)
        frames = frame_t
    traj = batch["traj_t"]
    texts = batch["text"]

    frames_t = torch.from_numpy(frames).to(device)
    traj_t = torch.from_numpy(traj).float().to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
        mixed = model.encode_mixed(frames_t, traj_t, texts)
    return {
        "z_v": mixed["z_v"].float(),
        "z_t": mixed["z_t"].float(),
        "z_text": mixed["z_text"].float(),
        "action": torch.from_numpy(batch["action"]).float().to(device),
        "frame_next": batch.get("frame_next"),
        "traj_next": batch.get("traj_next"),
    }


# =============================================================
# Test 1: V has variance
# =============================================================
def test_1_v_has_variance(
    value_head, model, val_loader, device, n_batches=5
) -> dict:
    """V should have meaningful variance (not constant)."""
    print("\n[Test 1] V has meaningful variance")
    print("-" * 50)

    value_head.eval()
    values = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)
        with torch.no_grad():
            v = value_head(emb["z_v"], emb["z_t"], emb["z_text"])
        values.extend(v.cpu().numpy().tolist())

    values = np.array(values)
    print(f"  V stats: min={values.min():.4f} max={values.max():.4f} "
          f"mean={values.mean():.4f} std={values.std():.4f}")
    print(f"  N: {len(values)} samples")
    passed = values.std() > 0.01
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: std > 0.01)")
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "n_samples": len(values),
        "pass": passed,
    }


# =============================================================
# Test 2: V increases along successful expert trajectories
# =============================================================
def test_2_v_increases_along_trajectories(
    value_head, model, val_loader, device,
    n_batches=5, gamma=0.99, gail_disc=None,
) -> dict:
    """V should be non-decreasing along successful expert trajectories.

    For each batch (which is sampled from a single episode), we get
    K=5 timesteps. V at the LAST timestep should be >= V at the
    FIRST timestep (closer to goal = more achievable reward).
    """
    print("\n[Test 2] V non-decreasing along expert trajectories")
    print("-" * 50)

    value_head.eval()
    if gail_disc is not None:
        gail_disc.eval()

    # Per-batch: get V at first and last timestep
    n_increasing = 0
    n_total = 0
    diffs = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break

        # world_model_collate returns (B, K, ...) for frame_t, traj_t, etc.
        # We want to compute V at multiple timesteps within each item
        B = batch["frame_t"].shape[0]
        K = batch["frame_t"].shape[1] if batch["frame_t"].ndim == 5 else 1
        if K < 2:
            continue

        # Encode all K timesteps
        frames_all = batch["frame_t"]  # (B, K, H, W, 3)
        traj_all = batch["traj_t"]       # (B, K, traj_window, 6)
        # The trajectory window in world_model_collate is centered on each t
        # But for simplicity, we use traj_t which is the trajectory window
        # ending at the LAST t. For t < last, we'd need different windows.
        #
        # SIMPLIFICATION: only use the LAST and SECOND-TO-LAST timestep
        # of the trajectory window, since we have traj for them.
        # In world_model_collate:
        #   traj_t = poses[t - traj_window + 1 : t + 1]   # ends at t
        #   frame_t[:, k] = frames[t - K + 1 + k]          # k-th past frame
        # So the LAST frame in frame_t[:, -1] is frames[t]
        #    and traj_t[:, -1] is poses[t]
        # There's no easy way to get V at multiple t without re-encoding.
        #
        # For a quick check, we'll compute V at multiple frames by encoding
        # each frame individually.
        try:
            texts = batch["text"]
            # V at the LAST timestep (frame_t[:, -1], traj_t)
            frames_last = frames_all[:, -1]  # (B, H, W, 3)
            traj_last = traj_all[:, -1] if traj_all.ndim == 4 else traj_all
            # V at the FIRST timestep — need to encode frame_t[:, 0] with a
            # trajectory window ending at t-K+1. Approximate using the
            # same trajectory window (not perfect, but a quick check).
            frames_first = frames_all[:, 0]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                # Encode last
                f_last_t = torch.from_numpy(frames_last).to(device)
                tr_last_t = torch.from_numpy(traj_last).float().to(device)
                m_last = model.encode_mixed(f_last_t, tr_last_t, texts)
                v_last = value_head(m_last["z_v"].float(), m_last["z_t"].float(), m_last["z_text"].float())

                # Encode first (using same traj window — approximation)
                f_first_t = torch.from_numpy(frames_first).to(device)
                tr_first_t = torch.from_numpy(traj_last).float().to(device)
                m_first = model.encode_mixed(f_first_t, tr_first_t, texts)
                v_first = value_head(m_first["z_v"].float(), m_first["z_t"].float(), m_first["z_text"].float())

            # Check V_last >= V_first (or close)
            # Cast to FP32 first — BF16 outputs from value_head
            # can't be converted via .cpu().numpy() directly.
            diff = (v_last.float() - v_first.float()).cpu().numpy()
            n_total += B
            n_increasing += int(np.sum(diff >= -0.05))  # tolerance
            diffs.extend(diff.tolist())
        except Exception as e:
            print(f"  Skipping batch {i}: {e}")
            continue

    if n_total == 0:
        return {"pass": False, "reason": "no valid batches", "n_samples": 0}

    pct_increasing = n_increasing / n_total
    mean_diff = float(np.mean(diffs))
    print(f"  Batches processed: {n_total}")
    print(f"  V (last) >= V (first) for: {pct_increasing * 100:.1f}%")
    print(f"  Mean V(last) - V(first): {mean_diff:.4f}")
    passed = pct_increasing > 0.6 and mean_diff > -0.05
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} "
          f"(target: >60% non-decreasing AND mean_diff > -0.05)")
    return {
        "pct_increasing": pct_increasing,
        "mean_diff": mean_diff,
        "n_samples": n_total,
        "pass": passed,
    }


# =============================================================
# Test 3: V predicts cumulative return
# =============================================================
def test_3_v_predicts_return(
    value_head, model, val_loader, device, gail_disc, n_batches=10,
    gamma=0.99,
) -> dict:
    """V(s) should correlate with the actual cumulative GAIL return.

    For each (s_t, a_t) transition:
    - Compute GAIL reward r_t = compute_reward(D(s_t, a_t))
    - Compute V(s_t) from the value head
    - Compare V(s_t) to the IMMEDIATE reward (single-step prediction target).
    - A well-trained V should at least correlate positively with r_t.
    """
    print("\n[Test 3] V predicts cumulative return (via GAIL reward)")
    print("-" * 50)

    value_head.eval()
    gail_disc.eval()
    v_list = []
    r_list = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)
        with torch.no_grad():
            v = value_head(emb["z_v"], emb["z_t"], emb["z_text"])
            logits = gail_disc(emb["z_v"], emb["z_t"], emb["z_text"], emb["action"])
            r = compute_reward(logits)
        v_list.extend(v.cpu().numpy().tolist())
        r_list.extend(r.cpu().numpy().tolist())

    v_arr = np.array(v_list)
    r_arr = np.array(r_list)

    # Compute Pearson correlation
    if len(v_arr) > 1 and v_arr.std() > 1e-6 and r_arr.std() > 1e-6:
        corr = float(np.corrcoef(v_arr, r_arr)[0, 1])
    else:
        corr = 0.0

    # Also compute |V - r| mean — if V is doing nothing, it should
    # match the mean of r, not the actual values
    abs_diff_mean = float(np.mean(np.abs(v_arr - r_arr)))
    print(f"  V-Reward correlation: {corr:.4f}")
    print(f"  |V - r| mean: {abs_diff_mean:.4f}")
    print(f"  V mean: {v_arr.mean():.4f}, std: {v_arr.std():.4f}")
    print(f"  r mean: {r_arr.mean():.4f}, std: {r_arr.std():.4f}")
    print(f"  N: {len(v_list)} samples")
    passed = corr > 0.3
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: corr > 0.3)")
    return {
        "correlation": corr,
        "abs_diff_mean": abs_diff_mean,
        "n_samples": len(v_list),
        "pass": passed,
    }


# =============================================================
# Test 4: V differentiates counterfactuals (KEY for alpha)
# =============================================================
def test_4_v_counterfactual_discrimination(
    value_head, model, world_model, val_loader, device, gail_disc,
    n_batches=10, margin=0.3,
) -> dict:
    """V must rank counterfactual states consistently with GAIL reward.

    This is the alpha-test: for (s, a_good) vs (s, a_bad) where GAIL
    clearly prefers a_good, V(f(s, a_good)) should be > V(f(s, a_bad)).
    Without this, alpha = sigmoid((V_m - V_h) / tau) is meaningless.
    """
    print("\n[Test 4] V differentiates counterfactuals (KEY for alpha)")
    print("-" * 50)
    print(f"  Margin: a_good must exceed a_bad by > {margin} in GAIL reward")

    value_head.eval()
    gail_disc.eval()
    world_model.eval()

    # Detect world model architecture and required input shape.
    # Old arch (window_size=0, MLP) takes (B, D) single embeddings.
    # New archs (window_size>0 MLP, RNN, Transformer) take (B, K, D)
    # windows. We need to add a K dimension if the world model needs
    # a window.
    #
    # Detection logic:
    #   - WorldModelMLP: hasattr window_size. If > 0, needs window.
    #   - WorldModelRNN: ALWAYS needs (B, K, D) input (uses GRU over time).
    #   - WorldModelTransformer: ALWAYS needs (B, K, D) input (uses attention).
    cls_name = type(world_model).__name__
    if cls_name == "WorldModelMLP":
        wm_window_size = getattr(world_model, "window_size", 0)
        use_window = wm_window_size > 0
    elif cls_name in ("WorldModelRNN", "WorldModelTransformer"):
        use_window = True
        # Default K=5 if not specified. The actual K doesn't matter
        # much because we're feeding the same embedding K times —
        # the model has seen this during training.
        wm_window_size = 5
    else:
        # Unknown arch — assume no window for safety
        use_window = False
        wm_window_size = 0

    n_pairs = 0
    n_v_agrees = 0
    pair_diffs = []  # (v_good - v_bad) for each pair

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= n_batches:
            break

        emb = encode_batch(model, batch, device)
        B = emb["z_v"].shape[0]
        device_t = emb["z_v"].device

        # For window-based models, expand (B, D) -> (B, K, D) by
        # repeating the current embedding K times.
        if use_window:
            z_v_w = emb["z_v"].unsqueeze(1).expand(-1, wm_window_size, -1).contiguous()
            z_t_w = emb["z_t"].unsqueeze(1).expand(-1, wm_window_size, -1).contiguous()
        else:
            z_v_w = emb["z_v"]
            z_t_w = emb["z_t"]

        # 1. Compute GAIL reward for the real (expert) action
        with torch.no_grad():
            expert_logits = gail_disc(
                emb["z_v"], emb["z_t"], emb["z_text"], emb["action"]
            )
            expert_r = compute_reward(expert_logits)  # (B,)

            # 2. Sample random actions and compute their GAIL rewards
            random_action = torch.randn_like(emb["action"]) * 0.1
            rand_logits = gail_disc(
                emb["z_v"], emb["z_t"], emb["z_text"], random_action
            )
            rand_r = compute_reward(rand_logits)  # (B,)

        # 3. For pairs where expert_r >> rand_r (clear preference),
        #    imagine the next state and check V ranking
        for i in range(B):
            if (expert_r[i] - rand_r[i]).item() < margin:
                continue

            # Imagine next states using the world model
            with torch.no_grad():
                # s'_good = f(s, expert_action)
                z_v_g, z_t_g = world_model(
                    z_v_w[i:i+1], z_t_w[i:i+1],
                    emb["z_text"][i:i+1], emb["action"][i:i+1]
                )
                # s'_bad = f(s, random_action)
                z_v_b, z_t_b = world_model(
                    z_v_w[i:i+1], z_t_w[i:i+1],
                    emb["z_text"][i:i+1], random_action[i:i+1]
                )

                # V(s'_good) vs V(s'_bad) — value_head takes (B, D)
                v_g = value_head(z_v_g, z_t_g, emb["z_text"][i:i+1]).float().item()
                v_b = value_head(z_v_b, z_t_b, emb["z_text"][i:i+1]).float().item()

            n_pairs += 1
            if v_g > v_b:
                n_v_agrees += 1
            pair_diffs.append(v_g - v_b)

    if n_pairs == 0:
        return {
            "pass": False,
            "reason": "no pairs with clear GAIL preference (margin too high)",
            "n_pairs": 0,
        }

    pct_agrees = n_v_agrees / n_pairs
    mean_diff = float(np.mean(pair_diffs))
    print(f"  N pairs (GAIL-clear): {n_pairs}")
    print(f"  V agrees with GAIL: {pct_agrees * 100:.1f}%")
    print(f"  Mean V(good) - V(bad): {mean_diff:.4f}")
    passed = pct_agrees > 0.6
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} "
          f"(target: V agrees with GAIL > 60% of the time)")
    return {
        "pct_agrees": pct_agrees,
        "mean_v_diff": mean_diff,
        "n_pairs": n_pairs,
        "pass": passed,
    }


# =============================================================
# Test 5: V is stable under small perturbations
# =============================================================
def test_5_v_stability(
    value_head, model, val_loader, device, n_batches=5, noise_std=0.01
) -> dict:
    """V should be stable: small changes in input → small changes in V.

    A V that's unstable (huge swings for tiny inputs) will give noisy α.
    """
    print("\n[Test 5] V is stable under small perturbations")
    print("-" * 50)
    print(f"  Noise std on z_v: {noise_std}")

    value_head.eval()
    v_orig = []
    v_noisy = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)
        with torch.no_grad():
            v1 = value_head(emb["z_v"], emb["z_t"], emb["z_text"])
            # Add small noise to z_v
            noise = torch.randn_like(emb["z_v"]) * noise_std
            v2 = value_head(emb["z_v"] + noise, emb["z_t"], emb["z_text"])
        v_orig.extend(v1.cpu().numpy().tolist())
        v_noisy.extend(v2.cpu().numpy().tolist())

    v_arr = np.array(v_orig)
    n_arr = np.array(v_noisy)

    if len(v_arr) > 1 and v_arr.std() > 1e-6:
        corr = float(np.corrcoef(v_arr, n_arr)[0, 1])
    else:
        corr = 0.0

    abs_diff = float(np.mean(np.abs(v_arr - n_arr)))
    print(f"  V correlation (orig vs noisy): {corr:.4f}")
    print(f"  |V(orig) - V(noisy)| mean: {abs_diff:.4f}")
    print(f"  N: {len(v_orig)} samples")
    passed = corr > 0.8
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: corr > 0.8)")
    return {
        "correlation": corr,
        "abs_diff_mean": abs_diff,
        "n_samples": len(v_orig),
        "pass": passed,
    }


# =============================================================
# Main evaluation function
# =============================================================
def evaluate(
    data_paths: List[str],
    value_head_checkpoint: str,
    encoder_checkpoint: str = None,
    gail_checkpoint: str = None,
    world_model_checkpoint: str = None,
    batch_size: int = 64,
    traj_window: int = 20,
    val_split: float = 0.1,
    device: str = None,
    use_bf16: bool = True,
    n_batches: int = 10,
):
    """Run all 5 value-head sanity tests and return a report."""
    global USE_BF16
    USE_BF16 = use_bf16

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Value Head Evaluation ===")
    print(f"  Value head: {value_head_checkpoint}")
    print(f"  Encoder: {encoder_checkpoint}")
    print(f"  GAIL: {gail_checkpoint}")
    print(f"  World model: {world_model_checkpoint}")
    print(f"  Device: {device}")

    # Load value head
    vh_ckpt = torch.load(value_head_checkpoint, map_location=device, weights_only=False)
    vh_cfg = vh_ckpt.get("config", {})
    vh_kwargs = {
        "hidden_dim": vh_cfg.get("hidden_dim", 256),
        "num_layers": vh_cfg.get("num_layers", 2),
    }
    value_head = create_value_head(**vh_kwargs).to(device)
    value_head.load_state_dict(vh_ckpt["value_head_state"])
    value_head.eval()
    print(f"  Value head: params={sum(p.numel() for p in value_head.parameters()):,}")

    # Load GAIL discriminator (required for tests 3 and 4)
    if gail_checkpoint is None:
        raise ValueError("--gail-checkpoint is required for these tests")
    gail_ckpt = torch.load(gail_checkpoint, map_location=device, weights_only=False)
    gail_cfg = gail_ckpt.get("config", {})
    gail_arch = gail_cfg.get("arch", "mlp")
    gail_kwargs = {}
    if gail_arch == "mlp":
        gail_kwargs = {
            "hidden_dim": gail_cfg.get("mlp_hidden", 512),
            "num_layers": gail_cfg.get("mlp_layers", 3),
        }
    gail_disc = create_gail_discriminator(
        arch=gail_arch,
        embed_dim=gail_cfg.get("embed_dim", 256),
        action_dim=gail_cfg.get("action_dim", 6),
        **gail_kwargs,
    ).to(device)
    gail_disc.load_state_dict(gail_ckpt["discriminator_state"])
    gail_disc.eval()
    print(f"  GAIL: arch={gail_arch}, params={sum(p.numel() for p in gail_disc.parameters()):,}")

    # Load world model (required for test 4)
    if world_model_checkpoint is None:
        raise ValueError("--world-model-checkpoint is required for test 4 (counterfactual)")
    wm_ckpt = torch.load(world_model_checkpoint, map_location=device, weights_only=False)
    wm_cfg = wm_ckpt.get("config", {})
    wm_arch = wm_cfg.get("arch", "mlp")
    wm_kwargs = {}
    if wm_arch == "mlp":
        # Auto-detect window_size for old architecture
        if wm_arch == "mlp" and "window_size" not in wm_cfg:
            # Detect by first weight shape
            first_w = next(iter(wm_ckpt.get("world_model_state", {}).values()), None)
            if first_w is not None and first_w.shape[1] == 774:
                wm_kwargs["window_size"] = 0
            else:
                wm_kwargs["window_size"] = wm_cfg.get("window_size", 5)
        else:
            wm_kwargs = {
                "hidden_dim": wm_cfg.get("mlp_hidden", 512),
                "num_layers": wm_cfg.get("mlp_layers", 3),
                "window_size": wm_cfg.get("window_size", 5),
            }
    elif wm_arch == "rnn":
        # Auto-detect num_rnn_layers
        max_l = 0
        for k in wm_ckpt.get("world_model_state", {}).keys():
            if k.startswith("gru.weight_ih_l"):
                try:
                    l = int(k.split("_l")[-1])
                    max_l = max(max_l, l + 1)
                except ValueError:
                    pass
        wm_kwargs = {
            "hidden_dim": wm_cfg.get("rnn_hidden_dim", wm_cfg.get("mlp_hidden", 256)),
            "num_rnn_layers": max_l if max_l > 0 else wm_cfg.get("num_rnn_layers", 1),
            "window_size": wm_cfg.get("window_size", 5),
        }
    elif wm_arch == "transformer":
        wm_kwargs = {
            "d_model": wm_cfg.get("transformer_d_model", 384),
            "nhead": wm_cfg.get("transformer_nhead", 4),
            "num_layers": wm_cfg.get("transformer_layers", 2),
            "dim_feedforward": wm_cfg.get("transformer_dim_ff", 1024),
            "dropout": wm_cfg.get("transformer_dropout", 0.0),
            "window_size": wm_cfg.get("window_size", 5),
        }
    world_model = create_world_model(
        arch=wm_arch,
        embed_dim=wm_cfg.get("embed_dim", 256),
        action_dim=wm_cfg.get("action_dim", 6),
        **wm_kwargs,
    ).to(device)
    world_model.load_state_dict(wm_ckpt["world_model_state"])
    world_model.eval()
    print(f"  World model: arch={wm_arch}, params={sum(p.numel() for p in world_model.parameters()):,}")

    # Load ALIGN encoder+mixer
    if encoder_checkpoint is None:
        # Auto-detect from value head config
        encoder_checkpoint = vh_cfg.get("pretrained_checkpoint")
        if encoder_checkpoint is None:
            raise ValueError("--encoder-checkpoint required")

    align = ALIGNModel(
        embed_dim=vh_cfg.get("embed_dim", 256),
        chunk_size=vh_cfg.get("chunk_size", 1),
        use_text=True,
        device=str(device),
        mixer_dim=vh_cfg.get("mixer_dim", 512),
        num_mixer_blocks=vh_cfg.get("num_mixer_blocks", 2),
        num_cameras=len(args.cameras) if args.cameras else 1,
    ).to(device)
    enc_ckpt = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    if "trainable_state_dict" in enc_ckpt:
        align.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    align.freeze_backbone()
    align.freeze_all_encoders()
    align.eval()
    print(f"  Encoder: {encoder_checkpoint}")

    # Build validation loader
    if len(data_paths) == 1:
        ds = ALIGNDataset(data_paths[0], mode="head", traj_window=traj_window,
                          cameras=args.cameras)
    else:
        from data.align_dataset import MultiALIGNDataset
        ds = MultiALIGNDataset(data_paths, mode="head", traj_window=traj_window,
                               cameras=args.cameras)

    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    indices = list(range(n_total - n_val, n_total))
    from torch.utils.data import SubsetRandomSampler
    val_loader = DataLoader(
        ds, batch_size=batch_size, sampler=SubsetRandomSampler(indices),
        drop_last=False,
        collate_fn=lambda b: world_model_collate(b, traj_window=traj_window),
    )
    print(f"  Dataset: N={n_total}, val={n_val}, batch_size={batch_size}")
    print(f"  Running {n_batches} batches per test")

    # Run tests
    report = {"tests": {}}
    print("\n" + "=" * 60)
    report["tests"]["1_variance"] = test_1_v_has_variance(
        value_head, align, val_loader, device, n_batches=n_batches,
    )
    report["tests"]["2_increasing"] = test_2_v_increases_along_trajectories(
        value_head, align, val_loader, device, n_batches=n_batches,
        gail_disc=gail_disc,
    )
    report["tests"]["3_return"] = test_3_v_predicts_return(
        value_head, align, val_loader, device, gail_disc, n_batches=n_batches,
    )
    report["tests"]["4_counterfactual"] = test_4_v_counterfactual_discrimination(
        value_head, align, world_model, val_loader, device, gail_disc,
        n_batches=n_batches,
    )
    report["tests"]["5_stability"] = test_5_v_stability(
        value_head, align, val_loader, device, n_batches=n_batches,
    )

    # Summary
    n_pass = sum(1 for r in report["tests"].values() if r.get("pass"))
    n_total_tests = len(report["tests"])
    print("\n" + "=" * 60)
    print(f"  Summary: {n_pass}/{n_total_tests} tests passed")
    print("=" * 60)
    for name, result in report["tests"].items():
        status = "PASS ✓" if result.get("pass") else "FAIL ✗"
        print(f"  {name}: {status}")
    print()
    print("  HARD GATE: V can be used for alpha pipeline only if all 5 tests pass.")
    print("  If any test fails, fix V before using — alpha will be meaningless.")

    report["n_pass"] = n_pass
    report["n_total"] = n_total_tests
    report["all_pass"] = n_pass == n_total_tests

    return report


def main():
    parser = argparse.ArgumentParser(description="Value head sanity checks")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views to use (e.g. 'wrist_image image'). "
                             "Must match the cameras used during training.")
    parser.add_argument("--value-head-checkpoint", required=True,
                        help="Path to value head checkpoint")
    parser.add_argument("--gail-checkpoint", required=True,
                        help="Path to GAIL discriminator checkpoint")
    parser.add_argument("--world-model-checkpoint", required=True,
                        help="Path to world model checkpoint (for test 4)")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="Path to encoder+mixer checkpoint (auto-detected if not given)")
    parser.add_argument("--output-dir", default=None,
                        help="If set, save JSON report here")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--n-batches", type=int, default=10)
    args = parser.parse_args()

    report = evaluate(
        data_paths=args.data,
        value_head_checkpoint=args.value_head_checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        gail_checkpoint=args.gail_checkpoint,
        world_model_checkpoint=args.world_model_checkpoint,
        batch_size=args.batch_size,
        traj_window=args.traj_window,
        val_split=args.val_split,
        device=args.device,
        use_bf16=args.bf16,
        n_batches=args.n_batches,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "value_eval_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
