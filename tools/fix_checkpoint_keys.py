#!/usr/bin/env python3
"""Fix checkpoint key remapping when model structure changes.

Copies weights from old key paths to new key paths in a checkpoint's
state_dict. Useful when refactoring module names without changing
the actual computation.

Usage:
    # Fix vision_patch_encoder moved from inside intention_encoder to top level
    python tools/fix_checkpoint_keys.py \\
        --checkpoint checkpoints/v4/libero_spatial/run_15/intention_best.pt \\
        --old-prefix intention_encoder.vision_patch_encoder. \\
        --new-prefix vision_patch_encoder.

    # Dry run to see what would change
    python tools/fix_checkpoint_keys.py \\
        --checkpoint checkpoints/v4/libero_spatial/run_15/intention_best.pt \\
        --old-prefix intention_encoder.vision_patch_encoder. \\
        --new-prefix vision_patch_encoder. \\
        --dry-run
"""
import argparse
import torch
from pathlib import Path


def fix_checkpoint_keys(
    checkpoint_path: str,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool = False,
    output_path: str = None,
) -> int:
    """Copy weights from old key paths to new key paths.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        old_prefix: Old key prefix to match (e.g. 'intention_encoder.vision_patch_encoder.').
        new_prefix: New key prefix to write to (e.g. 'vision_patch_encoder.').
        dry_run: If True, only print what would be copied without saving.
        output_path: Output path. Defaults to checkpoint with '_fixed' suffix.

    Returns:
        Number of keys copied.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    keys_to_copy = [k for k in sd.keys() if k.startswith(old_prefix)]
    if not keys_to_copy:
        print(f"No keys found with prefix '{old_prefix}'")
        return 0

    print(f"Found {len(keys_to_copy)} keys with prefix '{old_prefix}':")
    for k in keys_to_copy[:5]:
        print(f"  {k} → {k.replace(old_prefix, new_prefix)}")
    if len(keys_to_copy) > 5:
        print(f"  ... and {len(keys_to_copy) - 5} more")

    if dry_run:
        print("\nDry run — no changes saved.")
        return len(keys_to_copy)

    for k in keys_to_copy:
        new_k = k.replace(old_prefix, new_prefix)
        sd[new_k] = sd[k]

    ckpt["model_state_dict"] = sd

    if output_path is None:
        p = Path(checkpoint_path)
        output_path = str(p.parent / f"{p.stem}_fixed{p.suffix}")

    torch.save(ckpt, output_path)
    print(f"\nSaved fixed checkpoint to: {output_path}")
    return len(keys_to_copy)


def main():
    parser = argparse.ArgumentParser(
        description="Fix checkpoint key remapping when model structure changes."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--old-prefix", required=True, help="Old key prefix to match")
    parser.add_argument("--new-prefix", required=True, help="New key prefix to write to")
    parser.add_argument("--output", default=None, help="Output path (default: <checkpoint>_fixed.pt)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without saving")
    args = parser.parse_args()

    n = fix_checkpoint_keys(
        args.checkpoint,
        args.old_prefix,
        args.new_prefix,
        dry_run=args.dry_run,
        output_path=args.output,
    )
    if n > 0:
        print(f"\nDone. {n} keys remapped.")


if __name__ == "__main__":
    main()
