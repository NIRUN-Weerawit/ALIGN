#!/usr/bin/env python3
"""Migrate an old checkpoint to the new architecture.

The old code had `vision_patch_encoder.se_compressor.se_excitation.*`
and `vision_patch_encoder.se_compressor.projection.*`. The new code
inlined these into `vision_patch_encoder.se_excitation.*` and
`vision_patch_encoder.projection.*`.

This script renames the keys to match the new structure.

Usage:
    python tools/migrate_checkpoint.py --input old.pt --output new.pt
"""
import argparse
import torch


# Map of old key prefix -> new key prefix
KEY_REMAPS = [
    # The VisionPatchEncoder refactor (commit 4aa5904) inlined
    # SEVisualCompressor into VisionPatchEncoder.
    ("vision_patch_encoder.se_compressor.", "vision_patch_encoder."),
]


def migrate_keys(state_dict: dict) -> tuple[dict, list[str]]:
    """Apply key remappings to the state dict.

    Returns: (migrated_state_dict, list of changes)
    """
    migrated = {}
    changes = []
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in KEY_REMAPS:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                changes.append(f"  {key} -> {new_key}")
        migrated[new_key] = value
    return migrated, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to old checkpoint")
    parser.add_argument("--output", required=True, help="Path to save migrated checkpoint")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())[:5]}")

    # Find the state dict (could be at various keys)
    sd_key = None
    for candidate in ["state_dict", "model", "model_state_dict", "model_state", "weights"]:
        if isinstance(ckpt, dict) and candidate in ckpt:
            sd = ckpt[candidate]
            sd_key = candidate
            break
    if sd_key is None:
        sd = ckpt
        sd_key = None

    print(f"\nMigrating keys (using {len(KEY_REMAPS)} remap rules)...")
    migrated_sd, changes = migrate_keys(sd)
    print(f"  {len(changes)} keys remapped")
    for change in changes[:10]:
        print(change)
    if len(changes) > 10:
        print(f"  ... and {len(changes) - 10} more")

    # Save back
    if sd_key:
        ckpt[sd_key] = migrated_sd
    else:
        ckpt = migrated_sd

    print(f"\nSaving: {args.output}")
    torch.save(ckpt, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
