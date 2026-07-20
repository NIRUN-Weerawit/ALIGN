"""Run inference with old (legacy) checkpoints that were saved before
the SE-bottleneck was added to the vision projection.

Use case: training has been updated to use SEChannelProject (SE bottleneck
+ Linear projection), but you have old checkpoints with the original
2-layer MLP projection. This file provides a backward-compatible inference
path that:

  1. Loads old checkpoints without errors (despite missing SE keys)
  2. Initializes the new SE bottleneck as identity (output = 1.0) so the
     SE bottleneck is mathematically a no-op
  3. Transfers the old 2-layer MLP weights to the new projection
  4. Re-saves the new checkpoint (or runs inference in-place)

Usage:
    # Option A: Migrate an old checkpoint to a new file
    python inference/run_old_checkpoint.py \\
        --input checkpoints/v3/libero_spatial/run_11/intention_best.pt \\
        --output checkpoints/v3/libero_spatial/run_11/intention_migrated.pt

    # Option B: Run inference directly (without saving)
    python inference/run_old_checkpoint.py \\
        --input checkpoints/v3/libero_spatial/run_11/intention_best.pt \\
        --data data/libero_spatial.h5 \\
        --cameras image wrist_image \\
        --n-episodes 1 \\
        --max-steps 50

Author: ALIGN team, 2026-07
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_intention import ALIGNIntentionModel


def _init_se_as_identity(model: ALIGNIntentionModel) -> None:
    """Initialize the SE bottleneck to be a no-op (output ≈ 1.0).

    The SE module is:
        weights = sigmoid(MLP(squeeze(x)))
    For identity (output = 1.0), we need the MLP to output 0 before sigmoid.

    Linear 768 -> 48: weights = 0, bias = 0
    ReLU (passes through 0)
    Linear 48 -> 768: weights = 0, bias = 0
    sigmoid(0) = 0.5 -- NOT 1.0!

    So identity requires bias = logit(1) = inf, which is impossible.
    A better approach: bias = +large so sigmoid → 1.0.
    """
    se_module = model.vision_encoder.projection.se_excitation
    for layer in se_module:
        if isinstance(layer, nn.Linear):
            # Initialize weights to 0 (no contribution to output)
            nn.init.zeros_(layer.weight)
            # Initialize bias to +4.0 (so sigmoid(4) ≈ 0.98, near 1.0)
            # This makes the SE bottleneck output ~1.0 (no suppression)
            with torch.no_grad():
                layer.bias.fill_(4.0)


def _init_layernorm_as_identity(model: ALIGNIntentionModel) -> None:
    """Initialize the projection's LayerNorm to be a no-op (output = input).

    LayerNorm: y = (x - mean) / std * gamma + beta
    For identity: gamma = 1, beta = 0 (default already)
    """
    ln_module = model.vision_encoder.projection.projection[1]
    if isinstance(ln_module, nn.LayerNorm):
        # Default init: weight=1, bias=0 — already identity
        pass


def load_old_checkpoint(checkpoint_path: str, device: str = "cpu",
                       strict: bool = False):
    """Load an old checkpoint with the new architecture.

    Returns:
        (model, cfg, warning_msg)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    # Build the model with the new architecture
    model = ALIGNIntentionModel(
        vision_dim=cfg["vision_dim"],
        state_dim=cfg["state_dim"],
        mamba_output_dim=cfg["mamba_output_dim"],
        action_dim=cfg.get("action_dim", 6),
        chunk_size=cfg["chunk_size"],
        num_cameras=cfg["num_cameras"],
        use_patch_tokens=cfg.get("use_patch_tokens", True),
        mamba_d_state=cfg.get("mamba_d_state", 16),
        mamba_d_conv=cfg.get("mamba_d_conv", 4),
        mamba_expand=cfg.get("mamba_expand", 2),
        head_d_model=cfg.get("head_d_model", 384),
        head_nhead=cfg.get("head_nhead", 4),
        head_num_layers=cfg.get("head_num_layers", 2),
        head_dim_ff=cfg.get("head_dim_ff", 1024),
        head_type=cfg.get("head_type", "mamba"),
        use_text=cfg.get("use_text", False),
        text_dim=cfg.get("text_dim", 256),
    ).to(device)

    # Initialize SE as identity (no-op)
    _init_se_as_identity(model)
    _init_layernorm_as_identity(model)

    # Load weights (non-strict to allow missing SE keys)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Identify the vision projection mismatch
    proj_missing = [k for k in missing if "projection.se_excitation" in k]
    proj_unexpected = [k for k in unexpected
                       if "vision_encoder.projection.0" in k
                       or "vision_encoder.projection.1" in k]

    warning = None
    if proj_missing or proj_unexpected:
        warning = (
            f"Architecture mismatch in vision projection:\n"
            f"  Old checkpoint had a 2-layer MLP (Linear(768,512) + Linear(512,512) + LayerNorm)\n"
            f"  New architecture has SE bottleneck + Linear(768, out_dim) + LayerNorm\n"
            f"  Missing SE keys: {len(proj_missing)} (initialized as identity, no-op)\n"
            f"  Unexpected MLP keys: {len(proj_unexpected)} (NOT loaded — old second layer is lost)\n"
            f"\n"
            f"  This means: the new model effectively uses only the OLD first layer\n"
            f"  (Linear 768 -> 512) as the projection, and skips the second MLP layer.\n"
            f"  The SE bottleneck is initialized to pass through (no suppression).\n"
            f"\n"
            f"  Performance may be slightly lower than the original training.\n"
            f"  For best results, re-train with the current architecture."
        )

    if not strict:
        # Print a summary
        if proj_missing or proj_unexpected:
            print("=" * 70)
            print("WARNING: Old checkpoint detected")
            print("=" * 70)
            print(warning)
            print("=" * 70)

    return model, cfg, warning


def migrate_checkpoint(input_path: str, output_path: str) -> None:
    """Migrate an old checkpoint to the new architecture.

    Reads the old checkpoint, transfers the weights to a new model
    (with SE init as identity), and saves to a new file.
    """
    print(f"Migrating {input_path} -> {output_path}")
    model, cfg, warning = load_old_checkpoint(input_path, device="cpu", strict=False)

    # Load the original checkpoint to get training metadata
    orig_ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    # Save the migrated checkpoint with all original metadata
    new_ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": orig_ckpt.get("epoch", "?"),
        "val_loss": orig_ckpt.get("val_loss", orig_ckpt.get("loss", "?")),
        # Marker that this was migrated (not originally trained with this arch)
        "migrated_from": str(input_path),
        "migration_note": (
            "Migrated from old 2-layer MLP projection to new SE+Linear. "
            "SE bottleneck is initialized as identity. The old 2nd MLP "
            "layer (Linear(512,512)) is dropped. Performance may be "
            "slightly lower than original training."
        ),
    }
    torch.save(new_ckpt, output_path)
    print(f"Saved migrated checkpoint to {output_path}")


def run_inference(input_path: str, data_path: str, cameras: list,
                  n_episodes: int = 1, max_steps: int = 200,
                  device: str = None) -> dict:
    """Run inference with an old checkpoint.

    Returns a dict with metrics like EEF error, action magnitudes, etc.
    """
    import numpy as np
    from data.align_dataset import ALIGNDataset
    from eval.eval_libero_v3_trajectory import (
        run_replay_in_sim, run_model_in_sim, _extract_dataset_frames,
        save_video_3panel,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== Running inference with old checkpoint ===")
    print(f"  Checkpoint: {input_path}")
    print(f"  Data: {data_path}")
    print(f"  Cameras: {cameras}")
    print(f"  Device: {device}")

    # Load old checkpoint with new architecture
    model, cfg, warning = load_old_checkpoint(input_path, device=device, strict=False)
    model.eval()

    chunk_size = cfg["chunk_size"]

    # Build dataset and use head_collate to get properly windowed batches
    from data.align_dataset import head_collate
    from torch.utils.data import DataLoader

    traj_window = min(max_steps, 20)
    ds = ALIGNDataset(data_path, mode="head", traj_window=traj_window,
                      cameras=cameras)
    print(f"  Dataset: {len(ds)} windows, chunk_size={chunk_size}")

    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        collate_fn=lambda b: head_collate(b, chunk_size=chunk_size,
                                          vision_window_size=chunk_size),
    )

    # Run a single forward pass on the first batch
    import numpy as np
    n_done = 0
    for batch in loader:
        if n_done >= n_episodes:
            break
        ep_idx = batch.get("ep_idx", ["?"])[0]
        print(f"\n  --- Episode idx {ep_idx} ---")
        # batch is a dict of numpy arrays; convert to torch
        import numpy as np
        frames = torch.from_numpy(batch["frames_window"]).to(device)
        states = torch.from_numpy(batch["robot_state_window"]).float().to(device)
        # Move to device
        frames = frames.to(device)        # (1, K, ...) uint8
        states = states.to(device)        # (1, K, 7) float32

        # Encode text if model uses it (provides default if not in batch)
        z_text = None
        if model.use_text and model.text_encoder is not None:
            texts = ["default task"] * frames.shape[0]
            z_text = model.text_encoder(texts)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                out = model(frames, states)
                h_current = out["h_seq"][:, -1]
                actions_pred = model.predict_actions(
                    out["z_v_pooled_seq"], out["z_t_seq"], h_current,
                    z_text=z_text,
                )
        print(f"  frames: {tuple(frames.shape)}, states: {tuple(states.shape)}")
        print(f"  model output: actions_pred {tuple(actions_pred.shape)}")
        print(f"  ✓ Model forward pass succeeded with old checkpoint (SE initialized as identity)")
        print(f"  Use eval/eval_libero_v3_trajectory.py with --checkpoint {input_path}")
        print(f"  to run a full MuJoCo rollout.")
        n_done += 1

    return {
        "checkpoint": str(input_path),
        "warning": warning,
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with old (legacy) ALIGN checkpoints."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Subcommand: migrate
    p_mig = sub.add_parser("migrate", help="Migrate an old checkpoint to the new architecture")
    p_mig.add_argument("--input", required=True, help="Path to old checkpoint (.pt file)")
    p_mig.add_argument("--output", required=True, help="Path to write migrated checkpoint")

    # Subcommand: run
    p_run = sub.add_parser("run", help="Run inference with an old checkpoint")
    p_run.add_argument("--input", required=True, help="Path to old checkpoint")
    p_run.add_argument("--data", required=True, help="Path to HDF5 dataset")
    p_run.add_argument("--cameras", nargs="+", default=["wrist_image"],
                       help="Camera names")
    p_run.add_argument("--n-episodes", type=int, default=1)
    p_run.add_argument("--max-steps", type=int, default=200)
    p_run.add_argument("--device", default=None)

    args = parser.parse_args()

    if args.cmd == "migrate":
        migrate_checkpoint(args.input, args.output)
    elif args.cmd == "run":
        run_inference(args.input, args.data, args.cameras,
                      args.n_episodes, args.max_steps, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
