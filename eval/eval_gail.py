"""
Strict sanity checks for the trained GAIL discriminator.

The reward function r(s, a) is the foundation of the alpha pipeline.
If it's wrong, V(s) trained on it will be useless, and alpha is
meaningless. These tests are HARD GATES before the reward can be
used downstream.

Tests:

Test 1: Reward has meaningful variance
  - Quick sanity: the discriminator is learning *something*.
  - Pass: std > 0.1 (not 0.01 — that's too easy for any noise)

Test 2: Expert reward > Random action reward (margin)
  - The discriminator must clearly distinguish expert actions
    from random actions. A weak distinction means GAIL is no
    better than a random baseline.
  - Pass: mean_diff > 0.3 (expert - random)

Test 3: Reward signals progress along expert trajectories
  - Along a successful expert trajectory, the reward at LATER
    timesteps should be HIGHER (closer to goal = more
    "expert-like" state-action pair).
  - For this test we need to iterate through full episodes from
    the HDF5 file (similar to Test 3 of eval_world_model.py).
  - Pass: reward at last 25% > reward at first 25% on > 60% of
    episodes

Test 4: Reward is sharp (calibrated to action distance)
  - Reward should decrease smoothly as we move the action away
    from expert toward random. If reward jumps wildly or is
    constant for similar actions, the GAIL is miscalibrated.
  - Pass: Pearson correlation between action distance and
    reward < -0.3 (further from expert → lower reward)

Test 5: Reward is consistent (not seed-dependent)
  - For the same (s, a) pair, computing reward twice should
    give the same value. (Determinism check — catches bugs
    where dropout is active in eval mode, etc.)
  - Pass: max |r1 - r2| < 1e-5

If any test fails, GAIL is not safe to use. Common fixes:
  - More training epochs
  - Use world-model rollouts as negative examples (instead of random)
  - Different architecture (transformer vs MLP)
  - Check that the GAIL config matches the encoder's mixer_dim
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

# Global flag for BF16 autocast (set from CLI args in main)
USE_BF16 = True
from torch.utils.data import DataLoader, SubsetRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.gail_discriminator import (
    GAILDiscriminatorMLP,
    compute_reward,
    create_gail_discriminator,
)
from data.align_dataset import ALIGNDataset, world_model_collate


def encode_batch(model: ALIGNModel, batch: dict, device: torch.device) -> dict:
    """Encode a batch through the frozen encoder+mixer.

    Returns dict with z_v (B, D), z_s (B, D), z_sext (B, D), action (B, 6).

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
    traj = batch.get("state", batch["traj_t"])  # v2 (B,7) or legacy (B,K,6)
    texts = batch["text"]

    frames_t = torch.from_numpy(frames).to(device)
    traj_t = torch.from_numpy(traj).float().to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
        mixed = model.encode_mixed(frames_t, traj_t, texts)
    return {
        "z_v": mixed["z_v"].float(),       # (B, D)
        "z_s": mixed["z_s"].float(),       # (B, D) — mean-pooled
        "z_sext": mixed["z_sext"].float(), # (B, D)
        "action": torch.from_numpy(batch["action"]).float().to(device),  # (B, 6)
    }


# =============================================================
# Test 1: Reward has variance
# =============================================================
def test_1_reward_variance(
    model, discriminator, val_loader, device, n_batches=10
) -> dict:
    """Reward should have meaningful variance (not constant)."""
    print("\n[Test 1] Reward has meaningful variance")
    print("-" * 50)

    discriminator.eval()
    rewards = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)
        with torch.no_grad():
            logits = discriminator(emb["z_v"], emb["z_s"], emb["z_sext"], emb["action"])
            r = compute_reward(logits)
        rewards.extend(r.float().cpu().numpy().tolist())

    rewards = np.array(rewards)
    print(f"  Reward stats: min={rewards.min():.4f} max={rewards.max():.4f} "
          f"mean={rewards.mean():.4f} std={rewards.std():.4f}")
    print(f"  N: {len(rewards)} samples")
    # Use std > 0.1 instead of 0.01 — any noise passes 0.01
    passed = rewards.std() > 0.1
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: std > 0.1)")
    return {
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "n_samples": len(rewards),
        "pass": passed,
    }


# =============================================================
# Test 2: Expert reward > Random reward
# =============================================================
def test_2_expert_vs_random(
    model, discriminator, val_loader, device, n_batches=10, margin=0.3
) -> dict:
    """Expert actions should have meaningfully higher reward than random actions.

    A small margin (e.g., < 0.1) means the discriminator has barely
    learned the difference — GAIL would be useless for V training.
    """
    print("\n[Test 2] Expert reward > Random reward (with margin)")
    print("-" * 50)
    print(f"  Target margin: {margin}")

    discriminator.eval()
    expert_rewards = []
    random_rewards = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)
        with torch.no_grad():
            # Expert
            expert_logits = discriminator(emb["z_v"], emb["z_s"], emb["z_sext"], emb["action"])
            expert_r = compute_reward(expert_logits).float()

            # Random: sample from N(0, 0.1) — small but nonzero
            random_action = torch.randn_like(emb["action"]) * 0.1
            random_logits = discriminator(emb["z_v"], emb["z_s"], emb["z_sext"], random_action)
            random_r = compute_reward(random_logits).float()

        expert_rewards.extend(expert_r.cpu().numpy().tolist())
        random_rewards.extend(random_r.cpu().numpy().tolist())

    mean_expert = float(np.mean(expert_rewards))
    mean_random = float(np.mean(random_rewards))
    diff = mean_expert - mean_random
    print(f"  Mean expert reward: {mean_expert:.4f}")
    print(f"  Mean random reward: {mean_random:.4f}")
    print(f"  Diff (expert - random): {diff:.4f}")
    print(f"  N: {len(expert_rewards)} samples")
    passed = diff > margin
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: diff > {margin})")
    return {
        "mean_expert": mean_expert,
        "mean_random": mean_random,
        "diff": diff,
        "n_samples": len(expert_rewards),
        "pass": passed,
    }


# =============================================================
# Test 3: Reward signals progress along expert trajectories
# =============================================================
def test_3_reward_signals_progress(
    model, discriminator, val_loader, device, ds, n_episodes=10
) -> dict:
    """Reward at later timesteps > reward at earlier timesteps along episodes.

    For each episode, read full frames + poses + actions from the HDF5
    directly (not via the chunked __getitem__). Compute GAIL reward at
    each timestep. Check that reward(last_quartile) > reward(first_quartile)
    for the majority of episodes.

    This is the key test: if reward doesn't track progress, V trained on
    it will learn nothing useful for the alpha pipeline.
    """
    print("\n[Test 3] Reward signals progress along expert trajectories")
    print("-" * 50)

    discriminator.eval()

    # Read full episodes from HDF5 directly
    n_episodes_checked = 0
    n_progress_correct = 0
    progress_diffs = []  # (mean_reward_last - mean_reward_first) per episode

    # Iterate over validation episodes
    val_n = len(val_loader.sampler) if hasattr(val_loader, "sampler") else len(ds)
    for ep_idx in range(min(n_episodes, val_n)):
        try:
            source = ds[ep_idx]
            frames_i = source["frames"]
            poses_i = source["poses"][..., :6]
            actions_i = source.get("actions")
            if actions_i is None:
                continue
            actions_i = actions_i[..., :6]
            text_i_str = source["text"]
            if not isinstance(text_i_str, list):
                text_i_str = [text_i_str]

            N = len(frames_i)
            if N < 10:  # too short
                continue

            # Compute reward at each timestep by encoding frames one-by-one
            rewards = []
            for t in range(N - 1):  # need t+1 to be valid
                # Encode state at t
                f_t = torch.from_numpy(frames_i[t]).unsqueeze(0).to(device)
                # Build trajectory window ending at t
                traj_window = 5
                end_p = min(t + 1, len(poses_i))
                start_p = max(0, end_p - traj_window)
                p_t = torch.from_numpy(poses_i[start_p:end_p].astype(np.float32)).unsqueeze(0).to(device)
                if p_t.shape[1] < traj_window:
                    pad = torch.zeros(1, traj_window - p_t.shape[1], 6, device=device)
                    p_t = torch.cat([pad, p_t], dim=1)
                # Use the action at t
                a_t = torch.from_numpy(actions_i[t].astype(np.float32)).unsqueeze(0).to(device)

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                    mixed = model.encode_mixed(f_t, p_t, text_i_str)
                    z_v = mixed["z_v"].float()
                    z_s = mixed["z_s"].float()
                    z_sext = mixed["z_sext"].float()
                    logits = discriminator(z_v, z_s, z_sext, a_t)
                    r = compute_reward(logits).float().item()
                rewards.append(r)

            if len(rewards) < 4:
                continue

            rewards = np.array(rewards)
            # Compare first quartile vs last quartile
            q = len(rewards) // 4
            first_q_mean = float(np.mean(rewards[:q]))
            last_q_mean = float(np.mean(rewards[-q:]))
            diff = last_q_mean - first_q_mean

            progress_diffs.append(diff)
            n_episodes_checked += 1
            if last_q_mean > first_q_mean:
                n_progress_correct += 1

        except Exception as e:
            print(f"  Skipping episode {ep_idx}: {e}")
            continue

    if n_episodes_checked == 0:
        return {"pass": False, "reason": "no valid episodes", "n_episodes": 0}

    pct_correct = n_progress_correct / n_episodes_checked
    mean_diff = float(np.mean(progress_diffs))
    print(f"  N episodes checked: {n_episodes_checked}")
    print(f"  Reward(last quartile) > Reward(first quartile) for: {pct_correct * 100:.1f}%")
    print(f"  Mean reward(last_q) - reward(first_q): {mean_diff:.4f}")
    passed = pct_correct > 0.6 and mean_diff > -0.05
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} "
          f"(target: >60% episodes + mean_diff > -0.05)")
    return {
        "pct_progress_correct": pct_correct,
        "mean_progress_diff": mean_diff,
        "n_episodes": n_episodes_checked,
        "pass": passed,
    }


# =============================================================
# Test 4: Reward is sharp (calibrated to action distance)
# =============================================================
def test_4_reward_calibrated_to_action_distance(
    model, discriminator, val_loader, device, n_batches=5, n_steps=8
) -> dict:
    """Reward should decrease as action moves from expert toward random.

    For each (s, a_expert), generate a chain of perturbed actions:
    a_k = (1 - k/n) * a_expert + (k/n) * a_random, for k = 0, 1, ..., n.
    Compute reward at each. Reward should monotonically decrease.

    Pass: Pearson correlation between action_distance and reward < -0.3.
    """
    print("\n[Test 4] Reward is calibrated to action distance")
    print("-" * 50)
    print(f"  Sampling {n_steps} action perturbations per (s, a)")

    discriminator.eval()
    all_distances = []
    all_rewards = []

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)
        B = emb["z_v"].shape[0]

        with torch.no_grad():
            # Random action to interpolate toward
            random_action = torch.randn_like(emb["action"]) * 0.1

            for k in range(n_steps + 1):
                alpha = k / n_steps  # 0 = expert, 1 = random
                # Linear interpolation: (1-alpha) * expert + alpha * random
                a_interp = (1 - alpha) * emb["action"] + alpha * random_action
                # Action distance from expert (alpha * ||expert - random||)
                action_dist = alpha * torch.norm(random_action - emb["action"], dim=-1)

                logits = discriminator(emb["z_v"], emb["z_s"], emb["z_sext"], a_interp)
                r = compute_reward(logits).float()

                all_distances.extend(action_dist.cpu().numpy().tolist())
                all_rewards.extend(r.cpu().numpy().tolist())

    distances = np.array(all_distances)
    rewards = np.array(all_rewards)

    if len(rewards) > 1 and rewards.std() > 1e-6 and distances.std() > 1e-6:
        corr = float(np.corrcoef(distances, rewards)[0, 1])
    else:
        corr = 0.0

    print(f"  N samples: {len(rewards)}")
    print(f"  Correlation (action_dist vs reward): {corr:.4f}")
    print(f"  Mean reward: {rewards.mean():.4f}, std: {rewards.std():.4f}")
    # Negative correlation: more distance → lower reward
    passed = corr < -0.3
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} "
          f"(target: corr < -0.3, i.e., reward decreases as action moves away from expert)")
    return {
        "correlation": corr,
        "n_samples": len(rewards),
        "pass": passed,
    }


# =============================================================
# Test 5: Reward is deterministic (not seed-dependent)
# =============================================================
def test_5_reward_deterministic(
    model, discriminator, val_loader, device, n_batches=3
) -> dict:
    """Reward should be deterministic: same (s, a) → same reward.

    Catches bugs where dropout is active in eval mode, or where the
    discriminator has non-deterministic ops.
    """
    print("\n[Test 5] Reward is deterministic")
    print("-" * 50)

    discriminator.eval()
    max_diff = 0.0
    n_checked = 0

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        emb = encode_batch(model, batch, device)

        with torch.no_grad():
            logits1 = discriminator(emb["z_v"], emb["z_s"], emb["z_sext"], emb["action"])
            r1 = compute_reward(logits1).float()

            # Same input, second evaluation
            logits2 = discriminator(emb["z_v"], emb["z_s"], emb["z_sext"], emb["action"])
            r2 = compute_reward(logits2).float()

        diff = (r1 - r2).abs().max().item()
        max_diff = max(max_diff, diff)
        n_checked += emb["z_v"].shape[0]

    print(f"  N: {n_checked} samples")
    print(f"  Max |r1 - r2|: {max_diff:.2e}")
    # Tiny numerical error is OK
    passed = max_diff < 1e-5
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: max_diff < 1e-5)")
    return {
        "max_diff": max_diff,
        "n_samples": n_checked,
        "pass": passed,
    }


# =============================================================
# Main evaluation
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Strict sanity checks for trained GAIL")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s)")
    parser.add_argument("--cameras", nargs="+", default=None,
                        help="Camera views to use (e.g. 'wrist_image image'). "
                             "Must match the cameras used during training.")
    parser.add_argument("--gail-checkpoint", required=True,
                        help="Path to gail_best.pt")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Path to Phase 1b encoder+mixer checkpoint")
    parser.add_argument("--output-dir", default="./checkpoints/gail_eval")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of episodes for Test 3 (progress)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")

    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    global USE_BF16
    USE_BF16 = args.bf16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GAIL checkpoint
    print(f"Loading GAIL checkpoint: {args.gail_checkpoint}")
    gail_ckpt = torch.load(args.gail_checkpoint, map_location=device, weights_only=False)
    config = gail_ckpt.get("config", {})

    # Build discriminator
    arch = config.get("arch", "mlp")
    disc_kwargs = {}
    if arch == "mlp":
        disc_kwargs = {
            "hidden_dim": config.get("mlp_hidden_dim", config.get("mlp_hidden", 512)),
            "num_layers": config.get("mlp_layers", 3),
            "dropout": config.get("dropout", 0.0),
        }
    elif arch == "transformer":
        disc_kwargs = {
            "d_model": config.get("transformer_d_model", 384),
            "nhead": config.get("transformer_nhead", 4),
            "num_layers": config.get("transformer_layers", 2),
            "dropout": config.get("transformer_dropout", 0.0),
            "dim_feedforward": config.get("transformer_dim_ff", 1024),
        }
    discriminator = create_gail_discriminator(
        arch=arch,
        embed_dim=config.get("embed_dim", 256),
        action_dim=config.get("action_dim", 6),
        **disc_kwargs,
    ).to(device)
    discriminator.load_state_dict(gail_ckpt["discriminator_state"])
    discriminator.eval()
    print(f"  Discriminator: {arch}, {sum(p.numel() for p in discriminator.parameters()):,} params")

    # Load encoder+mixer (read mixer_dim from encoder config, like eval_world_model.py)
    print(f"Loading encoder: {args.encoder_checkpoint}")
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    enc_cfg = enc_ckpt.get("config", {}) if isinstance(enc_ckpt, dict) else {}
    # num_cameras must match the cameras used during training
    num_cameras = len(args.cameras) if args.cameras else 1
    align = ALIGNModel(
        embed_dim=256,
        chunk_size=5,
        use_text=True,
        device=str(device),
        mixer_dim=enc_cfg.get("mixer_dim", 512),
        num_mixer_blocks=enc_cfg.get("num_mixer_blocks", 2),
        num_cameras=num_cameras,
    ).to(device)
    if "trainable_state_dict" in enc_ckpt:
        align.load_trainable_state_dict(enc_ckpt["trainable_state_dict"])
    align.freeze_backbone()
    align.freeze_all_encoders()
    align.eval()
    print(f"  Encoder: mixer_dim={align.cross_attention_mixer.mixer_dim}, "
          f"num_blocks={align.cross_attention_mixer.num_blocks}, "
          f"num_cameras={align.num_cameras}")

    # Build val dataset
    if len(args.data) == 1:
        ds = ALIGNDataset(args.data[0], mode="head", traj_window=5, cameras=args.cameras)
    else:
        from data.align_dataset import MultiALIGNDataset
        ds = MultiALIGNDataset(args.data, mode="head", traj_window=5, cameras=args.cameras)

    val_split = max(1, int(len(ds) * 0.1))
    val_indices = list(range(len(ds) - val_split, len(ds)))

    val_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_indices),
        collate_fn=lambda b: world_model_collate(b, traj_window=5),
        num_workers=0,
    )
    print(f"  Dataset: N={len(ds)}, val={val_split}, batch_size={args.batch_size}")
    print(f"  Running {args.n_batches} batches per test, {args.n_episodes} episodes for Test 3")

    # Run tests
    results = {}
    print("\n" + "=" * 60)
    results["1_variance"] = test_1_reward_variance(
        align, discriminator, val_loader, device, n_batches=args.n_batches,
    )
    results["2_expert_vs_random"] = test_2_expert_vs_random(
        align, discriminator, val_loader, device, n_batches=args.n_batches,
    )
    results["3_progress"] = test_3_reward_signals_progress(
        align, discriminator, val_loader, device, ds, n_episodes=args.n_episodes,
    )
    results["4_calibrated"] = test_4_reward_calibrated_to_action_distance(
        align, discriminator, val_loader, device, n_batches=args.n_batches,
    )
    results["5_deterministic"] = test_5_reward_deterministic(
        align, discriminator, val_loader, device, n_batches=3,
    )

    # Summary
    n_pass = sum(1 for r in results.values() if r.get("pass"))
    n_total = len(results)
    print("\n" + "=" * 60)
    print(f"  Summary: {n_pass}/{n_total} tests passed")
    print("=" * 60)
    for name, r in results.items():
        status = "PASS ✓" if r.get("pass") else "FAIL ✗"
        print(f"  {name}: {status}")
    print()
    print("  HARD GATE: GAIL can be used as a reward source only if all 5 tests pass.")
    print("  If any test fails, fix GAIL before using — V trained on it will be useless.")

    # Save report
    report_path = out_dir / "sanity_report.json"
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(report_path, "w") as f:
        json.dump(convert({
            "results": results,
            "all_pass": n_pass == n_total,
            "n_pass": n_pass,
            "n_total": n_total,
        }), f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()