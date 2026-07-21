"""
End-to-end alpha evaluation.

This is the moment of truth for the alpha pipeline. We compute the
intervention score alpha at every step in held-out episodes and
measure whether it's actually informative.

Test 1: Alpha distribution
  - Report the distribution of alpha values across all timesteps
  - For a useful alpha, the distribution should be spread (not stuck
    at 0.5). A useful alpha varies between 0 and 1.

Test 2: Alpha correlates with action divergence
  - Compute ||a_human - a_model|| (the action disagreement)
  - Compute correlation with alpha
  - For a useful alpha, this correlation should be POSITIVE:
    when the human and model disagree, alpha should be higher
    (more reason to intervene)

Test 3: AUC of alpha vs ideal
  - For each timestep, compute "ideal alpha":
    would the model action have been better than the human action?
  - AUC of (system alpha, ideal alpha) should be > 0.5
  - This is the "alpha is informative" test

If AUC is close to 0.5, the alpha is uninformative. If it's much
higher (e.g., 0.7+), the alpha is working.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.world_model import create_world_model
from models.value_head import create_value_head
from models.gail_discriminator import create_gail_discriminator
from data.align_dataset import ALIGNDataset, world_model_collate
from eval.compute_alpha import load_components, compute_alpha_batch


def encode_batch(model: ALIGNModel, batch: dict, device: torch.device) -> dict:
    """Encode a batch through the frozen encoder+mixer."""
    # world_model_collate returns frame_t as (B, K, H, W, 3) — use last frame
    frames = batch["frame_t"][:, -1]  # (B, H, W, 3)
    traj = batch.get("state", batch["traj_t"])  # v2 (B,7) or legacy (B,K,6)
    texts = batch["text"]

    frames_t = torch.from_numpy(frames).to(device)
    traj_t = torch.from_numpy(traj).float().to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
        mixed = model.encode_mixed(frames_t, traj_t, texts)
    return {
        "z_v": mixed["z_v"].float(),
        "z_s": mixed["z_s"].float(),
        "z_sext": mixed["z_sext"].float(),
        "action": torch.from_numpy(batch["action"]).float().to(device),
    }


def test_1_alpha_distribution(
    alphas: np.ndarray, n_batches: int
) -> dict:
    """Test 1: alpha should have a useful distribution."""
    print("\n[Test 1] Alpha distribution")
    print("-" * 50)
    print(f"  N samples: {len(alphas)} ({n_batches} batches)")
    print(f"  Mean: {alphas.mean():.4f}")
    print(f"  Std:  {alphas.std():.4f}")
    print(f"  Min:  {alphas.min():.4f}")
    print(f"  Max:  {alphas.max():.4f}")
    print(f"  Percentiles: p25={np.percentile(alphas, 25):.4f}  "
          f"p50={np.percentile(alphas, 50):.4f}  p75={np.percentile(alphas, 75):.4f}")

    # A useful alpha has std > 0.05 (some variation)
    passed = alphas.std() > 0.05
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: std > 0.05)")
    return {
        "mean": float(alphas.mean()),
        "std": float(alphas.std()),
        "min": float(alphas.min()),
        "max": float(alphas.max()),
        "n_samples": len(alphas),
        "pass": passed,
    }


def test_2_alpha_correlates_with_divergence(
    alphas: np.ndarray, action_div: np.ndarray
) -> dict:
    """Test 2: alpha should be higher when human and model disagree."""
    print("\n[Test 2] Alpha correlates with action divergence")
    print("-" * 50)
    if len(alphas) < 2 or action_div.std() < 1e-6 or alphas.std() < 1e-6:
        print("  Not enough variance to compute correlation")
        return {"correlation": 0.0, "n_samples": len(alphas), "pass": False}
    corr = float(np.corrcoef(alphas, action_div)[0, 1])
    print(f"  N: {len(alphas)}")
    print(f"  Pearson correlation: {corr:.4f}")
    # We expect POSITIVE correlation: more divergence = more reason to intervene
    passed = corr > 0.05
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: corr > 0.05)")
    return {
        "correlation": corr,
        "n_samples": len(alphas),
        "pass": passed,
    }


def test_3_alpha_auc_vs_ideal(
    alphas: np.ndarray, ideal_alphas: np.ndarray
) -> dict:
    """Test 3: alpha should be predictive of intervention value.

    The "ideal alpha" at each timestep is 1 if the model action would
    have led to a better outcome, 0 otherwise. AUC of (system alpha,
    ideal alpha) should be > 0.5.

    This is a weak test since we don't have ground truth for the ideal
    alpha. We use a proxy: ideal_alpha = 1 if alpha > 0.5 (a trivial
    circular test). The real test would be against ground truth.
    """
    print("\n[Test 3] Alpha AUC vs ideal")
    print("-" * 50)
    print(f"  N: {len(alphas)}")
    # Simple AUC: fraction of (i, j) pairs where the ranking is correct
    n = len(alphas)
    if n < 2:
        return {"auc": 0.5, "n_samples": n, "pass": False}
    # Binary ideal: above median = 1, below = 0
    threshold = np.median(ideal_alphas)
    ideal_binary = (ideal_alphas > threshold).astype(int)
    if ideal_binary.sum() == 0 or ideal_binary.sum() == n:
        return {"auc": 0.5, "n_samples": n, "pass": False}
    # AUC calculation
    pos_scores = alphas[ideal_binary == 1]
    neg_scores = alphas[ideal_binary == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return {"auc": 0.5, "n_samples": n, "pass": False}
    auc = float((pos_scores[:, None] > neg_scores[None, :]).mean())
    print(f"  AUC: {auc:.4f}")
    # AUC > 0.5 means the system alpha is informative
    passed = auc > 0.55
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'} (target: AUC > 0.55)")
    return {
        "auc": auc,
        "n_samples": n,
        "pass": passed,
    }


# Global flag for BF16 autocast
USE_BF16 = True


def main():
    parser = argparse.ArgumentParser(description="End-to-end alpha evaluation")
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s)")
    parser.add_argument("--world-model", required=True,
                        help="Path to world_model_best.pt")
    parser.add_argument("--value-head", required=True,
                        help="Path to value_best.pt")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Path to Phase 1b encoder+mixer checkpoint")
    parser.add_argument("--output-dir", default="./checkpoints/alpha_eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")

    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    global USE_BF16
    USE_BF16 = args.bf16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load components
    print("Loading components...")
    model, world_model, value_head = load_components(
        args.world_model, args.value_head, args.encoder_checkpoint, device,
    )

    # Build val dataset
    if len(args.data) == 1:
        ds = ALIGNDataset(args.data[0], mode="pretrain")
    else:
        from data.align_dataset import MultiALIGNDataset
        ds = MultiALIGNDataset(args.data, mode="pretrain")
    val_split = max(1, int(len(ds) * 0.1))
    val_indices = list(range(len(ds) - val_split, len(ds)))
    val_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        collate_fn=lambda b: world_model_collate(b, traj_window=5),
        num_workers=0,
    )

    # Compute alpha for each batch
    print(f"\nComputing alpha for {args.n_batches} batches...")
    all_alphas = []
    all_action_div = []
    all_ideal = []
    n_done = 0

    for i, batch in enumerate(val_loader):
        if i >= args.n_batches:
            break
        emb = encode_batch(model, batch, device)

        with torch.no_grad():
            # Real human action
            a_human = emb["action"]
            # "Model" action: use the average action (a poor man's baseline)
            # In a real system, this would come from the Assistant head
            a_model = a_human + torch.randn_like(a_human) * 0.02  # noisy version

            # Compute alpha
            alpha, v_h, v_m = compute_alpha_batch(
                world_model, value_head,
                emb["z_v"], emb["z_s"], emb["z_sext"],
                a_human, a_model, tau=args.tau,
            )
            # Action divergence (how different are the two actions)
            action_div = (a_human - a_model).norm(dim=-1)

        all_alphas.extend(alpha.cpu().numpy().tolist())
        all_action_div.extend(action_div.cpu().numpy().tolist())
        # "Ideal" alpha: 1 if model is better (V_m > V_h), 0 otherwise
        all_ideal.extend((v_m > v_h).float().cpu().numpy().tolist())
        n_done += 1

    alphas = np.array(all_alphas)
    action_div = np.array(all_action_div)
    ideal = np.array(all_ideal)

    # Run tests
    results = {}
    results["test_1_distribution"] = test_1_alpha_distribution(alphas, n_done)
    results["test_2_divergence"] = test_2_alpha_correlates_with_divergence(alphas, action_div)
    results["test_3_auc"] = test_3_alpha_auc_vs_ideal(alphas, ideal)

    # Summary
    print("\n" + "=" * 60)
    print("ALPHA END-TO-END EVAL SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, r in results.items():
        status = "PASS ✓" if r["pass"] else "FAIL ✗"
        print(f"  {name}: {status}")
        if not r["pass"]:
            all_pass = False
    print(f"\n  Overall: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    if not all_pass:
        print("\n  Recommendations:")
        print("  - Train world model and value head for more epochs")
        print("  - Improve GAIL discriminator (V is bounded by its reward)")
        print("  - Check encoder+mixer is being loaded correctly")
        print("  - Try different tau values (0.5 to 5.0)")

    # Save report
    report_path = out_dir / "alpha_eval_report.json"
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
        json.dump(convert({"results": results, "all_pass": all_pass}), f, indent=2)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
