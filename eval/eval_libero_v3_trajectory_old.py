"""Evaluate a trained ALIGNIntentionModel checkpoint that was saved with
the OLD vision projection (2-layer MLP) on LIBERO trajectory data.

This is a backward-compat wrapper around eval/eval_libero_v3_trajectory.py
for checkpoints that were trained BEFORE the SE-bottleneck was added
to the vision projection. Examples: run_11, run_12, run_13, run_14.

The new architecture has:
    SEChannelProject(768 → out_dim)
      ├── se_squeeze (no params)
      ├── se_excitation = Linear(768→48) → ReLU → Linear(48→768) → Sigmoid
      └── projection = Linear(768 → out_dim) + LayerNorm

Old checkpoints have a different structure:
    Sequential(Linear(768→512) + LayerNorm + Linear(512→512) + LayerNorm)
    or similar 2-layer MLP

This script:
  1. Uses inference/run_old_checkpoint.py's load_old_checkpoint()
     which inits SE as identity so the projection is mathematically
     equivalent to the old 1st MLP layer
  2. Runs the same MuJoCo sim-based eval as the v3 trajectory eval
  3. Reports per-episode EEF error, action magnitudes, etc.

Usage:
    # Basic usage (works like the new v3 eval)
    python eval/eval_libero_v3_trajectory_old.py \\
        --data /home/ucluser/ALIGN/data/libero_spatial.h5 \\
        --checkpoint checkpoints/v3/libero_spatial/run_11/intention_best.pt \\
        --cameras image wrist_image \\
        --n-episodes 5 \\
        --alpha 1.0

    # Same as v3 eval but works with old checkpoints
    python eval/eval_libero_v3_trajectory_old.py \\
        --data data/libero_spatial.h5 \\
        --checkpoint checkpoints/v3/libero_spatial/run_12/intention_best.pt \\
        --cameras image wrist_image \\
        --n-episodes 1 \\
        --max-steps 100

    # With migration warning suppressed (already seen)
    python eval/eval_libero_v3_trajectory_old.py \\
        --data data/libero_spatial.h5 \\
        --checkpoint path/to/old.pt \\
        --cameras image wrist_image \\
        --quiet-migration
"""

import argparse
import sys
from pathlib import Path

import torch

# Add repo root to path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the legacy checkpoint loader
from inference.run_old_checkpoint import load_old_checkpoint

# Re-use the v3 trajectory eval functions (main, run_replay_in_sim, run_model_in_sim, etc.)
# We import lazily inside main() to avoid loading the LIBERO deps if --help is called
# But since the user is going to actually run the eval, we import eagerly here.
# We avoid importing the new eval_intention's load_intention_model (which has strict loading)
# by importing the v3 trajectory eval directly.

# We need to import the v3 trajectory eval's main() but with our own model loading.
# Strategy: monkey-patch the load_intention_model function that the v3 trajectory
# eval imports, so that it uses our old-checkpoint-aware loader instead.

import eval.eval_libero_v3_trajectory as v3_eval
from eval.eval_libero_v3_trajectory import (
    run_replay_in_sim,
    run_model_in_sim,
    list_episodes,
)


def patched_load_intention_model(checkpoint_path, device):
    """Replacement for eval.eval_intention.load_intention_model that
    supports old (legacy) checkpoints by initializing the SE bottleneck
    as identity. Drops the warning message (3rd return value) so the
    signature matches eval_intention.load_intention_model."""
    model, cfg, _warning = load_old_checkpoint(
        checkpoint_path, device=str(device), strict=False
    )
    return model, cfg


# v3_eval already uses our patched loader (set up at top of file)
# Nothing to do here.


def main():
    # Parse args (same as v3 eval)
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OLD v3 intention model checkpoint on LIBERO "
            "trajectory data. Old checkpoints used a 2-layer MLP vision "
            "projection; this script handles the SE-bottleneck mismatch."
        ),
    )
    parser.add_argument("--data", required=True,
                        help="Path to HDF5 dataset.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to old intention_best.pt (run_11, run_12, etc.)")
    parser.add_argument("--cameras", nargs="+", default=["wrist_image"],
                        help="Camera names (default: wrist_image). "
                             "MUST match the cameras used during training.")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Gaussian noise std for noised actions.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode.")
    parser.add_argument("--task-text", type=str, default=None,
                        help="Task text (default: read from HDF5 if available).")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for plots (default: alongside checkpoint).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Save trajectory plots (default: on).")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    # MuJoCo / LIBERO options
    parser.add_argument("--use-mujoco", action="store_true", default=True,
                        help="Run episodes in MuJoCo sim (default on).")
    parser.add_argument("--no-mujoco", dest="use_mujoco", action="store_false",
                        help="Skip MuJoCo evaluation (offline only).")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save 3-panel side-by-side video (default on).")
    parser.add_argument("--no-video", dest="save_video", action="store_false",
                        help="Skip video saving.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Blend factor: action = (1-alpha) * a_human + alpha * a_model. "
                             "Default 1.0 (use model only).")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Scale factor applied to model actions before applying to sim. "
                             "Default: 1.0 (no scaling).")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step model action values.")
    parser.add_argument("--libero-suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_object",
                                 "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO benchmark suite.")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Frame size for MuJoCo rendering.")
    parser.add_argument("--no-flip-vertical", action="store_true",
                        help="Skip vertical flip on sim and dataset frames "
                             "(default: flip vertical like old eval).")
    parser.add_argument("--no-flip-horizontal", action="store_true",
                        help="Skip horizontal flip on sim and dataset frames "
                             "(default: don't flip horizontal).")

    args = parser.parse_args()

    # Run the v3 eval (which will use our patched loader)
    v3_eval.main()


if __name__ == "__main__":
    main()
