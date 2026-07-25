"""Dump checkpoint key structure for diagnosis."""
import sys
import torch

if len(sys.argv) < 2:
    print("Usage: python tools/dump_ckpt_keys.py <checkpoint.pt>")
    sys.exit(1)

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"Top-level keys: {list(ckpt.keys())}")

# Find state dict
sd = None
for k in ["state_dict", "model", "model_state_dict", "model_state", "weights"]:
    if k in ckpt:
        sd = ckpt[k]
        print(f"Found state dict at: {k}")
        break
if sd is None:
    sd = ckpt

print(f"\nTotal keys in state dict: {len(sd)}")
print(f"First 5 keys:")
for k in list(sd.keys())[:5]:
    print(f"  {k}: {sd[k].shape if hasattr(sd[k], 'shape') else type(sd[k])}")

# Search for specific patterns
patterns = ["se_compressor", "se_excitation", "vision_patch_encoder", "intention_encoder"]
for pattern in patterns:
    matches = [k for k in sd.keys() if pattern in k]
    print(f"\n{pattern}: {len(matches)} keys")
    for k in matches[:5]:
        print(f"  {k}")
