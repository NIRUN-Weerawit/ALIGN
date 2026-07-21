"""Diagnose the pose_to_action model: does it have a bounded output range?

Three things to check:
  1. What range does the model output on real LIBERO data?
  2. What range does the model output for synthetic extreme inputs?
  3. Is the model just predicting the mean (a sign of failed learning)?

Usage:
    PYTHONNOUSERSITE=1 python eval/diagnose_pose_to_action.py \\
        --checkpoint checkpoints/pose_to_action/libero_spatial/run_2/pose_to_action_best.pt \\
        --data h5_data/libero_spatial.h5 \\
        --out-prefix reports/pose_to_action_diag
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pose_to_action import PoseDeltaToAction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out-prefix", default="reports/pose_to_action_diag")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    hidden_dim = cfg.get("hidden_dim", 128)
    model = PoseDeltaToAction(pose_dim=6, action_dim=6, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.checkpoint}")
    print(f"  hidden_dim={hidden_dim}, val_mse={cfg.get('val_mse', '?')}")

    # 1. Real LIBERO data
    samples = []
    with h5py.File(args.data, "r") as f:
        for ep_key in sorted(k for k in f.keys() if k.startswith("ep_")):
            ep = f[ep_key]
            poses = ep.get("poses", ep.get("noisy_poses"))[:]
            actions = ep.get("actions")
            if actions is None:
                continue
            N = min(len(poses), len(actions))
            for t in range(N - 1):
                pd = (poses[t + 1, :6] - poses[t, :6]).astype(np.float32)
                a = actions[t, :6].astype(np.float32)
                samples.append((pd, a))
    samples = list(samples)
    pose_deltas = np.stack([s[0] for s in samples]).astype(np.float32)  # (N, 6)
    actions = np.stack([s[1] for s in samples]).astype(np.float32)        # (N, 6)
    print(f"\nReal data: {len(pose_deltas)} samples")
    print(f"  pose_delta  abs mean per dim: {np.abs(pose_deltas).mean(axis=0)}")
    print(f"  pose_delta  range per dim:    min={pose_deltas.min(axis=0)}  max={pose_deltas.max(axis=0)}")
    print(f"  action      abs mean per dim: {np.abs(actions).mean(axis=0)}")
    print(f"  action      range per dim:    min={actions.min(axis=0)}  max={actions.max(axis=0)}")

    # 2. Model output on real data
    with torch.no_grad():
        pred_real = model(torch.from_numpy(pose_deltas).to(device)).cpu().numpy()
    err = pred_real - actions
    print(f"\nModel output on real data:")
    print(f"  pred   abs mean per dim: {np.abs(pred_real).mean(axis=0)}")
    print(f"  pred   range per dim:    min={pred_real.min(axis=0)}  max={pred_real.max(axis=0)}")
    print(f"  error  abs mean per dim: {np.abs(err).mean(axis=0)}")
    print(f"  error  std    per dim:   {err.std(axis=0)}")
    rel = np.abs(err) / (np.abs(actions) + 1e-6)
    print(f"  rel error  median: {np.median(rel):.3f}  p90: {np.percentile(rel, 90):.3f}  p99: {np.percentile(rel, 99):.3f}")

    # 3. Sanity check: is the model just predicting the mean?
    mean_pred = pred_real.mean(axis=0)
    print(f"\n  mean prediction per dim: {mean_pred}")
    print(f"  mean target    per dim:  {actions.mean(axis=0)}")
    print(f"  std  prediction per dim: {pred_real.std(axis=0)}")
    print(f"  std  target    per dim:  {actions.std(axis=0)}")
    print(f"  pred/target std ratio:   {pred_real.std(axis=0) / (actions.std(axis=0) + 1e-9)}")
    print("  (ratio < 0.5 → model is shrinking to the mean, not learning the mapping)")

    # 4. Model output for synthetic extreme inputs
    print(f"\nExtrapolation test — synthetic inputs at 1x, 5x, 10x training range:")
    pd_max = np.abs(pose_deltas).max(axis=0)
    print(f"  max |pose_delta| seen in training: {pd_max}")
    test_scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    for scale in test_scales:
        # Sign-preserving scaled version of a real input
        x = pose_deltas[:1000] * scale
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        print(f"  scale={scale:5.1f}x  → pred abs mean={np.abs(pred).mean():.4f}  range=[{pred.min():.3f}, {pred.max():.3f}]")

    # 5. Model output for zero input
    with torch.no_grad():
        pred_zero = model(torch.zeros(1, 6).to(device)).cpu().numpy()[0]
    print(f"\n  model(zero) = {pred_zero}  (should be near zero if the model learned identity-like)")

    # 6. Save stats
    out = {
        "checkpoint": args.checkpoint,
        "n_samples": int(len(pose_deltas)),
        "val_mse_from_training": cfg.get("val_mse"),
        "real_data": {
            "pose_delta_range": pose_deltas.tolist() if pose_deltas.size < 1000 else "truncated",
            "action_range": actions.tolist() if actions.size < 1000 else "truncated",
            "action_mean": actions.mean(axis=0).tolist(),
            "action_std": actions.std(axis=0).tolist(),
        },
        "model_output_on_real": {
            "pred_mean": pred_real.mean(axis=0).tolist(),
            "pred_std": pred_real.std(axis=0).tolist(),
            "abs_error_mean": np.abs(err).mean(axis=0).tolist(),
            "rel_error_median": float(np.median(rel)),
            "rel_error_p90": float(np.percentile(rel, 90)),
            "rel_error_p99": float(np.percentile(rel, 99)),
            "std_ratio_pred_over_target": (pred_real.std(axis=0) / (actions.std(axis=0) + 1e-9)).tolist(),
        },
        "extrapolation": {},
    }
    for scale in test_scales:
        x = pose_deltas[:1000] * scale
        with torch.no_grad():
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        out["extrapolation"][f"scale_{scale}x"] = {
            "abs_mean": float(np.abs(pred).mean()),
            "min": float(pred.min()),
            "max": float(pred.max()),
        }
    out["model_zero_input"] = pred_zero.tolist()
    out_path = Path(str(args.out_prefix) + "_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
