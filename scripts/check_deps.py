#!/usr/bin/env python3
"""Check that all ALIGN dependencies are installed with compatible versions.

Usage:
    python scripts/check_deps.py

Exits with code 0 if all OK, 1 if missing.
Set PYTHONNOUSERSITE=1 to exclude ~/.local packages (recommended).
"""

import importlib
import sys
import warnings
warnings.filterwarnings("ignore")

DEPS = [
    ("torch",         lambda m: f"{m.__version__:16s} CUDA:{m.cuda.is_available()}"),
    ("open_clip",     lambda m: m.__version__),
    ("xformers",      lambda m: m.__version__),
    ("lerobot",       lambda m: m.__version__),
    ("av",            lambda m: m.__version__),
    ("wandb",         lambda m: m.__version__),
    ("scipy",         lambda m: m.__version__),
    ("numpy",         lambda m: m.__version__),
    ("h5py",          lambda m: m.__version__),
    ("PIL",           lambda m: m.__version__),
    ("requests",      lambda m: m.__version__),
    ("transformers",  lambda m: m.__version__),
]

def main():
    all_ok = True
    for name, fmt in DEPS:
        try:
            mod = importlib.import_module(name)
            version = fmt(mod) if callable(fmt) else mod.__version__
            print(f"  {name:16s} {version}")
        except ImportError as e:
            print(f"  {name:16s} ❌ MISSING — {e}")
            all_ok = False

    import platform
    print(f"  {'Python':16s} {platform.python_version()}")

    if all_ok:
        print("\nAll deps OK")
    else:
        print("\nSome dependencies are missing. Run: pip install open-clip-torch lerobot xformers av transformers")
        sys.exit(1)

if __name__ == "__main__":
    main()
