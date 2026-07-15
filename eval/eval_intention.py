#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained ALIGNIntentionModel (Mamba) checkpoint.

Loads an `intention_best.pt` checkpoint, builds the corresponding
ALIGNIntentionModel, runs a forward pass over a held-out validation
split of an HDF5 dataset, and reports:

  - Per-dimension MSE / MAE between actions_pred and actions_window
  - Per-step RMSE/MAE for the K future actions
  - Mode-collapse diagnostics (output std across batch)
  - Step-1 alignment check (cosine similarity, magnitude comparison)

The script reads model hyperparameters (chunk_size, mamba dims, head
dims, num_cameras, etc.) from the checkpoint's `config` field when
present, falling back to sensible defaults.

Usage:
    python eval/eval_intention.py \\
        --data /path/to/align.h5 \\
        --checkpoint checkpoints/intention/libero_spatial/run_1/intention_best.pt \\
        --n-batches 20 --batch-size 32
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
# Disable cuDNN — see align_model.py for the full explanation.
torch.backends.cudnn.enabled = False
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_intention import ALIGNIntentionModel
from data.align_dataset import ALIGNDataset, head_collate


# ================================================================
# Defaults & config
# ================================================================

# Used when the checkpoint's `config` field is missing or incomplete.
DEFAULT_CONFIG: Dict[str, object] = dict(
    chunk_size=5,
    vision_dim=256,
    state_dim=256,
    mamba_output_dim=512,
    mamba_d_state=16,
    mamba_d_conv=4,
    mamba_expand=2,
    head_d_model=384,
    head_nhead=4,
    head_num_layers=2,
    head_dim_ff=1024,
    num_cameras=1,
    use_patch_tokens=True,
    head_type="mamba",
    use_text=False,
    text_dim=256,
)


def _merge_config(ckpt_cfg: Optional[Dict]) -> Dict:
    """Merge checkpoint config with defaults (ckpt takes precedence)."""
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(ckpt_cfg, dict):
        for k, v in ckpt_cfg.items():
            if v is not None:
                cfg[k] = v
    return cfg


# ================================================================
# Model loading
# ================================================================

def load_intention_model(
    checkpoint_path: str,
    device: torch.device,
    override_chunk_size: Optional[int] = None,
    override_num_cameras: Optional[int] = None,
):
    """Load an ALIGNIntentionModel from `intention_best.pt`.

    The checkpoint is expected to contain:
      - "model_state_dict": state dict of the full model
      - "config":           dict of hyperparameters

    If the checkpoint's chunk_size or num_cameras doesn't match the
    defaults, the model is rebuilt with the right shape before loading
    weights (so we can correctly load older checkpoints that were
    trained with a different K or number of cameras).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _merge_config(ckpt.get("config", {}))
    if override_chunk_size is not None:
        cfg["chunk_size"] = override_chunk_size
    if override_num_cameras is not None:
        cfg["num_cameras"] = override_num_cameras

    print(f"  Loading:    {checkpoint_path}")
    print(f"  Epoch:      {ckpt.get('epoch', '?')}")
    print(f"  Val loss:   {ckpt.get('val_loss', ckpt.get('loss', '?'))}")
    print(f"  Chunk (K):  {cfg['chunk_size']}")
    print(f"  Cameras:    {cfg['num_cameras']}")
    print(f"  Mamba dim:  {cfg['mamba_output_dim']}")
    print(f"  Head:       {cfg.get('head_type', 'mamba')}")
    if cfg.get('use_text', False):
        print(f"  Text:       enabled (dim={cfg.get('text_dim', 256)})")

    model = ALIGNIntentionModel(
        vision_dim=cfg["vision_dim"],
        state_dim=cfg["state_dim"],
        mamba_output_dim=cfg["mamba_output_dim"],
        action_dim=6,
        chunk_size=cfg["chunk_size"],
        num_cameras=cfg["num_cameras"],
        use_patch_tokens=cfg["use_patch_tokens"],
        mamba_d_state=cfg["mamba_d_state"],
        mamba_d_conv=cfg["mamba_d_conv"],
        mamba_expand=cfg["mamba_expand"],
        head_d_model=cfg["head_d_model"],
        head_nhead=cfg["head_nhead"],
        head_num_layers=cfg["head_num_layers"],
        head_dim_ff=cfg["head_dim_ff"],
        head_type=cfg.get("head_type", "mamba"),
        use_text=cfg.get("use_text", False),
        text_dim=cfg.get("text_dim", 256),
    ).to(device)

    # Load the state dict. We try a strict load first; if that fails
    # (e.g. checkpoint was trained with slightly different head config)
    # we fall back to a non-strict load and report mismatches.
    sd = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(sd, strict=True)
        print(f"  Loaded state_dict strictly ({len(sd)} tensors)")
    except RuntimeError as e:
        print(f"  Strict load failed ({type(e).__name__}); "
              f"falling back to non-strict load.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"    Missing keys:   {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"    Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    model.eval()
    return model, cfg


# ================================================================
# Evaluation
# ================================================================

def evaluate(
    data_paths: List[str],
    checkpoint_path: str,
    batch_size: int = 32,
    traj_window: int = 20,
    val_split: float = 0.1,
    n_batches: int = 20,
    device_str: Optional[str] = None,
    override_chunk_size: Optional[int] = None,
    override_num_cameras: Optional[int] = None,
    task_text: Optional[str] = None,
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n=== ALIGN Intention (Mamba) Evaluation ===")
    print(f"  Device:   {device}")
    print(f"  Data:     {data_paths}")

    # -- Model
    model, cfg = load_intention_model(
        checkpoint_path, device,
        override_chunk_size=override_chunk_size,
        override_num_cameras=override_num_cameras,
    )
    chunk_size = cfg["chunk_size"]
    cameras = ["wrist_image"] if cfg["num_cameras"] == 1 else None  # let dataset auto-detect

    # -- Dataset (held-out split)
    if len(data_paths) == 1:
        ds = ALIGNDataset(
            data_paths[0], mode="head",
            traj_window=traj_window, cameras=cameras,
        )
    else:
        from data.align_dataset import MultiALIGNDataset
        ds = MultiALIGNDataset(
            data_paths, mode="head",
            traj_window=traj_window, cameras=cameras,
        )
    n_total = len(ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)
    _, val_ds = random_split(ds, [n_train, n_val], generator=g)
    print(f"  Dataset:  {n_train} train, {n_val} val  (using {n_val} for eval)")

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size,
                                          vision_window_size=chunk_size),
    )

    # -- Evaluation loop
    n_samples = 0
    sum_se = 0.0   # sum of squared errors (overall)
    sum_ae = 0.0   # sum of absolute errors (overall)
    per_dim_se = np.zeros(6, dtype=np.float64)  # per-output-dim squared errors
    per_dim_ae = np.zeros(6, dtype=np.float64)  # per-output-dim absolute errors
    per_step_mse: List[List[float]] = [[] for _ in range(chunk_size)]
    per_step_cos: List[List[float]] = [[] for _ in range(chunk_size)]
    per_step_mag_pred: List[List[float]] = [[] for _ in range(chunk_size)]
    per_step_mag_target: List[List[float]] = [[] for _ in range(chunk_size)]
    all_pred_stds: List[float] = []
    sample_collected: Optional[Dict] = None

    print(f"\n  Running up to {n_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            frames = torch.from_numpy(batch["frames_window"]).to(device)  # (B, K, H, W, 3) or (B, K, V, H, W, 3)
            state = torch.from_numpy(batch["robot_state_window"]).float().to(device)  # (B, K, 7)
            # Always use 'actions_window' (target) for error computation
            target = torch.from_numpy(batch["actions_window"]).float().to(device)  # (B, K, 6)

            # Optional text encoding (only if model has text encoder)
            z_text = None
            if getattr(model, "text_encoder", None) is not None:
                B_size = frames.shape[0]
                if "texts" in batch and batch["texts"]:
                    texts = batch["texts"]
                elif task_text:
                    texts = [task_text] * B_size
                else:
                    texts = ["default task"] * B_size
                z_text = model.text_encoder(texts)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                out = model(frames, state)
                h_current = out["h_seq"][:, -1]
                if model.head_type == "flow":
                    # Flow head: use sample_actions (ODE integration)
                    actions_pred = model.sample_actions(
                        out["z_v_pooled_seq"], out["z_t_seq"], h_current, z_text=z_text,
                    )
                else:
                    # Direct regression head
                    actions_pred = model.predict_actions(
                        out["z_v_pooled_seq"], out["z_t_seq"], h_current, z_text=z_text,
                    )
                # (B, K, 6)

            actions_pred_f = actions_pred.float()
            target_f = target.float()

            # ---- Aggregate errors ----
            B = actions_pred_f.shape[0]
            n_samples += B
            err = actions_pred_f - target_f        # (B, K, 6)
            sum_se += (err ** 2).sum().item()
            sum_ae += err.abs().sum().item()
            # per-dim (collapsed over B and K)
            per_dim_se += (err ** 2).sum(dim=(0, 1)).cpu().numpy()
            per_dim_ae += err.abs().sum(dim=(0, 1)).cpu().numpy()
            # per-step
            for k in range(chunk_size):
                e_k = err[:, k]  # (B, 6)
                per_step_mse[k].append((e_k ** 2).mean().item())
                cos_k = F.cosine_similarity(
                    actions_pred_f[:, k], target_f[:, k], dim=-1,
                )
                per_step_cos[k].extend(cos_k.cpu().tolist())
                per_step_mag_pred[k].extend(
                    actions_pred_f[:, k].norm(dim=-1).cpu().tolist()
                )
                per_step_mag_target[k].extend(
                    target_f[:, k].norm(dim=-1).cpu().tolist()
                )

            # Mode collapse: per-batch std of predictions
            all_pred_stds.append(actions_pred_f.std(dim=0).mean().item())

            # Capture first batch for inspection
            if i == 0:
                sample_collected = {
                    "current_state_0": state[0, -1, :3].cpu().tolist(),
                    "target_step0": target_f[0, 0, :3].cpu().tolist(),
                    "pred_step0": actions_pred_f[0, 0, :3].cpu().tolist(),
                }

    if n_samples == 0:
        print("  No samples evaluated (n_batches too small or val split empty).")
        return

    # ---- Aggregate metrics ----
    total_elements = n_samples * chunk_size * 6
    overall_mse = sum_se / total_elements
    overall_mae = sum_ae / total_elements
    overall_rmse = float(np.sqrt(overall_mse))
    n_dim_elements = n_samples * chunk_size
    per_dim_mse = per_dim_se / n_dim_elements
    per_dim_mae = per_dim_ae / n_dim_elements
    per_dim_rmse = np.sqrt(per_dim_mse)

    # ---- Print results ----
    print(f"\n{'='*68}")
    print(f"=== Results ({n_samples} samples, K={chunk_size} steps, 6 dims) ===")
    print(f"{'='*68}")
    print(f"\nLoss mode: action (only)")
    print(f"\nOverall metrics (flattened over K and 6 dims):")
    print(f"  MSE:  {overall_mse:.6f}")
    print(f"  RMSE: {overall_rmse:.6f}")
    print(f"  MAE:  {overall_mae:.6f}")

    dim_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    print(f"\nPer-dimension metrics (averaged over K steps):")
    print(f"  {'dim':<6}{'MSE':<14}{'RMSE':<14}{'MAE':<14}")
    for d in range(6):
        print(f"  {dim_names[d]:<6}"
              f"{per_dim_mse[d]:<14.6f}"
              f"{per_dim_rmse[d]:<14.6f}"
              f"{per_dim_mae[d]:<14.6f}")

    print(f"\nPer-step metrics (averaged over 6 dims):")
    print(f"  {'step':<6}{'MSE':<14}{'cos':<10}{'|pred|':<12}{'|target|':<12}")
    for k in range(chunk_size):
        step_mse = float(np.mean(per_step_mse[k]))
        step_cos = float(np.mean(per_step_cos[k]))
        step_pred = float(np.mean(per_step_mag_pred[k]))
        step_tgt = float(np.mean(per_step_mag_target[k]))
        print(f"  k={k:<4}"
              f"{step_mse:<14.6f}"
              f"{step_cos:<10.4f}"
              f"{step_pred:<12.4f}"
              f"{step_tgt:<12.4f}")

    # ---- Mode collapse check ----
    avg_pred_std = float(np.mean(all_pred_stds))
    print(f"\nMode collapse check:")
    print(f"  Avg prediction std across batch: {avg_pred_std:.6f}")
    if avg_pred_std < 0.001:
        print(f"  ⚠️  WARNING: predictions have near-zero variance. Model may "
              f"have mode-collapsed.")
    else:
        print(f"  ✓ predictions have meaningful variance across samples.")

    # ---- Step-1 alignment ----
    step1_cos = float(np.mean(per_step_cos[0])) if per_step_cos[0] else 0.0
    print(f"\nStep-1 alignment (most important for inference):")
    print(f"  Step-1 mean cosine: {step1_cos:.4f}")
    if step1_cos > 0.7:
        print(f"  ✓ Step-1 alignment is good (cos > 0.7)")
    elif step1_cos > 0.3:
        print(f"  ⚠️  Step-1 alignment is moderate (0.3 < cos < 0.7)")
    else:
        print(f"  ⚠️  Step-1 alignment is poor (cos < 0.3)")

    # ---- Sample inspection ----
    if sample_collected is not None:
        print(f"\nSample inspection (first batch, sample 0):")
        print(f"  current_state[0:3]:    {sample_collected['current_state_0']}")
        print(f"  target_action[0:3]:    {sample_collected['target_step0']}")
        print(f"  pred_action[0:3]:      {sample_collected['pred_step0']}")

    # Return a small summary so the script can be called programmatically
    return {
        "mse": overall_mse,
        "rmse": overall_rmse,
        "mae": overall_mae,
        "per_dim_mse": per_dim_mse.tolist(),
        "per_dim_mae": per_dim_mae.tolist(),
        "per_step_mse": [float(np.mean(per_step_mse[k])) for k in range(chunk_size)],
        "step1_cos": step1_cos,
        "n_samples": n_samples,
    }


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ALIGNIntentionModel (Mamba) checkpoint."
    )
    parser.add_argument("--data", required=True, nargs="+",
                        help="Path(s) to HDF5 dataset(s).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to intention_best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--traj-window", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--n-batches", type=int, default=20,
                        help="Max number of batches to evaluate.")
    parser.add_argument("--device", default=None)
    # NOTE: --bf16 / --loss-mode removed; BF16 always on, loss mode always 'action'.
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Override chunk_size (default: read from ckpt).")
    parser.add_argument("--num-cameras", type=int, default=None,
                        help="Override num_cameras (default: read from ckpt).")
    parser.add_argument("--task-text", type=str, default=None,
                        help="Task text for text-conditioned models "
                             "(default: 'default task' if model uses text).")

    args = parser.parse_args()
    summary = evaluate(
        data_paths=args.data,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        traj_window=args.traj_window,
        val_split=args.val_split,
        n_batches=args.n_batches,
        device_str=args.device,
        override_chunk_size=args.chunk_size,
        override_num_cameras=args.num_cameras,
        task_text=args.task_text,
    )
    # Optional: dump a JSON summary next to the checkpoint
    if summary is not None:
        out_json = Path(args.checkpoint).with_suffix(".eval.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary written to: {out_json}")


if __name__ == "__main__":
    main()
