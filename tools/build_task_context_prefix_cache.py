"""Materialize frame-local frozen teacher summaries and separate training labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.task_context_prefix_dataset import build_prefix_cache

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("targets", "latents", "teacher", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.24)
        torch.backends.cuda.enable_cudnn_sdp(False)
    build_prefix_cache(a.targets, a.latents, a.teacher, a.output, a.device)
