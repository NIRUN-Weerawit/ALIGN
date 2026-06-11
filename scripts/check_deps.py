#!/usr/bin/env python3
"""Verify ALIGN dependencies are installed and compatible.

Usage:
    python scripts/check_deps.py
    PYTHONNOUSERSITE=1 python scripts/check_deps.py  # ignore ~/.local pip installs
"""
import importlib
import sys
from pathlib import Path

DEPS = [
    ("torch", "2.10.0"),
    ("torchvision", "0.25.0"),
    ("numpy", None),
    ("scipy", None),
    ("h5py", None),
    ("PIL", None, "Pillow"),
    ("open_clip", None, "open-clip-torch"),
    ("lerobot", None),
    ("xformers", None),
    ("transformers", None),
    ("av", None, "PyAV"),
    ("wandb", None),
    ("tqdm", None),
]

OPTIONAL = [
    ("torchcodec", "0.10.0"),
    ("paho", None, "paho-mqtt"),
]

def check(name, min_version=None, pip_name=None):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        status = "✅"
        if min_version:
            from packaging.version import parse
            if parse(str(ver)) < parse(min_version):
                status = "⚠️"
        print(f"  {status} {pip_name or name:20s} {ver}")
        return True
    except ImportError:
        print(f"  ❌ {pip_name or name:20s} not installed")
        return False

def main():
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print()

    print("Core dependencies:")
    all_ok = True
    for dep in DEPS:
        ok = check(*dep)
        all_ok = all_ok and ok

    print()
    print("Optional:")
    for dep in OPTIONAL:
        check(*dep)

    print()
    if all_ok:
        print("All core deps OK.")
    else:
        print("Some core deps missing. Run: pip install -r requirements.txt")
        sys.exit(1)

    # CUDA check
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # torchcodec check
    try:
        from torchcodec.decoders import VideoDecoder
        print("torchcodec: ✅ loads OK")
    except Exception as e:
        print(f"torchcodec: ❌ {e}")

if __name__ == "__main__":
    main()
