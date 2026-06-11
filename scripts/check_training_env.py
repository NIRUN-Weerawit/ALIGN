#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-flight check: verify the training environment is correct.

Catches the "running on CPU because ~/.local shadowed conda torch" bug
that cost 100 minutes of training. Run before any long training session.

Usage:
    python scripts/check_training_env.py
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python():
    print("=" * 60)
    print("1. PYTHON INTERPRETER")
    print("=" * 60)
    print(f"  Executable: {sys.executable}")
    print(f"  Version:    {sys.version.split()[0]}")
    is_conda = "miniconda" in sys.executable or "anaconda" in sys.executable or "conda" in sys.executable
    print(f"  Conda env:  {'YES' if is_conda else 'NO'}")
    if not is_conda:
        print("  ⚠️  WARNING: not running from a conda env")
    return is_conda


def check_torch():
    print()
    print("=" * 60)
    print("2. TORCH + CUDA")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("  ✗ torch not installed")
        return False, None, None

    print(f"  Version:  {torch.__version__}")
    print(f"  Path:     {torch.__file__}")
    has_cuda = "+cu" in torch.__version__ or ".cu" in torch.__version__
    print(f"  CUDA:     {torch.__version__}")
    print(f"           {'✓ CUDA build' if has_cuda else '✗ CPU-only build'}")

    if not has_cuda:
        print("  ⚠️  WARNING: torch is CPU-only. Reinstall with CUDA support:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")

    cuda_available = torch.cuda.is_available()
    print(f"  Available: {'YES' if cuda_available else 'NO'}")

    if cuda_available:
        print(f"  Device:   {torch.cuda.get_device_name()}")
        print(f"  Memory:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return cuda_available, has_cuda, torch.__version__


def check_shadowing():
    print()
    print("=" * 60)
    print("3. ~/.local PACKAGE SHADOWING")
    print("=" * 60)

    local = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
    shadowed = []
    if local.exists():
        for pkg in ["torch", "torchvision", "transformers", "xformers", "open_clip"]:
            local_path = local / pkg
            if local_path.exists():
                shadowed.append(pkg)

    if shadowed:
        print(f"  ⚠️  ~/.local shadows: {', '.join(shadowed)}")
        print(f"     Fix: export PYTHONNOUSERSITE=1 before running")
        return False
    else:
        print("  ✓ No shadowing detected")
        return True


def check_dependencies():
    print()
    print("=" * 60)
    print("4. REQUIRED PACKAGES")
    print("=" * 60)

    required = {
        "torch": None,
        "torchvision": None,
        "transformers": None,
        "open_clip": None,
        "xformers": None,
        "lerobot": None,
    }

    for pkg in required:
        try:
            mod = __import__(pkg)
            required[pkg] = getattr(mod, "__version__", "ok")
        except ImportError:
            required[pkg] = "MISSING"

    for pkg, ver in required.items():
        status = "✓" if ver != "MISSING" else "✗"
        print(f"  {status} {pkg:15s} {ver}")


def check_model():
    print()
    print("=" * 60)
    print("5. MODEL SMOKE TEST")
    print("=" * 60)
    try:
        sys.path.insert(0, str(Path.cwd()))
        import torch
        from models.align_model import ALIGNModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ALIGNModel(embed_dim=256, use_text=True, device=device)
        if torch.cuda.is_available():
            model = model.to("cuda")

        B, K = 2, 10
        frames = torch.randint(0, 255, (B, 224, 224, 3), dtype=torch.uint8)
        frames = frames.to(device)
        traj = torch.randn(B, K, 6, device=device)
        texts = ["test"] * B

        # Warmup
        _ = model.encode_mixed(frames, traj, texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time
        t0 = time.time()
        for _ in range(5):
            mixed = model.encode_mixed(frames, traj, texts)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 5

        print(f"  ✓ Model loaded and ran on {device}")
        print(f"  Forward (B=2, K=10): {dt*1000:.1f}ms")
        print(f"  1200 steps @ B=32: ~{1200 * dt * 16 / 60:.1f} min")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    is_conda = check_python()
    cuda_available, has_cuda, torch_ver = check_torch()
    no_shadow = check_shadowing()
    check_dependencies()
    model_ok = check_model()

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    issues = []
    if not is_conda:
        issues.append("Not running in conda env")
    if not has_cuda:
        issues.append(f"torch is CPU-only ({torch_ver})")
    if not cuda_available:
        issues.append("CUDA not available")
    if not no_shadow:
        issues.append("~/.local packages shadowing torch")
    if not model_ok:
        issues.append("Model smoke test failed")

    if not issues:
        print("  ✓ All checks pass. Training should run at full speed.")
        print(f"    Expected: 1200 steps in ~0.9 min on RTX 4060 (B=32)")
    else:
        print(f"  ✗ {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, 1):
            print(f"     {i}. {issue}")
        print()
        print("  Quick fix (in most cases):")
        print("    export PYTHONNOUSERSITE=1")
        print("    /home/ucluser/miniconda3/envs/align/bin/python training/pretrain_streaming.py --wandb")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
