#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure ALIGN inference speed on a trained checkpoint.

Three measurements:
  1. Per-step latency (one observation → one action chunk)
  2. Throughput (steps/sec under batched inference)
  3. Component breakdown (latency of each submodule)

Usage:
    python tools/measure_inference_speed.py \\
        --checkpoint checkpoints/v4/libero_spatial/run_27/intention_best.pt \\
        --data data/libero_spatial.h5 \\
        --output results/inference_speed/

Outputs:
    - results.json : all measured metrics
    - breakdown.png : bar chart of component latencies
    - scaling.png   : batch-size vs throughput curve
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.align_intention import ALIGNIntentionModel  # noqa: E402
from data.align_dataset import ALIGNDataset  # noqa: E402


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def cuda_sync():
    """Synchronize CUDA before reading elapsed time (so we measure GPU time, not launch time)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_model_flags_from_state_dict(keys, raw_sd=None):
    """Infer use_intent_tokens / use_memory_bank / use_history / compressed_dim
    from checkpoint keys (and optionally state_dict values for shape info).

    This is more reliable than reading ckpt['args'] because:
    - args may not be present in older checkpoints
    - args may differ from what was actually trained (e.g., resumed from
      different config)
    """
    has_intent = any("intention_encoder.intent_tokens" in k for k in keys)
    has_memory = any("memory_module" in k for k in keys)
    has_intention_encoder = any(k.startswith("intention_encoder.") for k in keys)
    use_history = has_intention_encoder

    # Infer compressed_dim from the se_compressor weight shape if available
    compressed_dim = 16  # default
    if raw_sd is not None:
        for k, v in raw_sd.items():
            if "se_compressor.projection.1.weight" in k and isinstance(v, torch.Tensor):
                # This is the bottleneck output projection, shape [compressed_dim]
                compressed_dim = int(v.shape[0])
                break

    return {
        "use_intent_tokens": has_intent,
        "use_memory_bank": has_memory,
        "use_history": use_history,
        "compressed_dim": compressed_dim,
    }


def build_model(args, device, inferred_flags=None):
    """Construct model with flags inferred from the checkpoint.

    If inferred_flags is provided (from the state_dict inspection), use those
    as the source of truth. Otherwise fall back to CLI defaults.
    """
    flags = inferred_flags or {
        "use_intent_tokens": True,
        "use_memory_bank": args.use_memory_bank,
        "use_history": True,
        "compressed_dim": 16,
    }
    cfg: dict = {
        "use_intent_tokens": flags["use_intent_tokens"],
        "num_intent_tokens": 2,
        "intent_dim": 512,
        "use_memory_bank": flags["use_memory_bank"],
        "head_type": "diffusion",
        "mamba_output_dim": 512,
        "state_dim": 256,
        "chunk_size": 10,
        "history_size": 1,
        "num_cameras": len(args.cameras),
        "compressed_dim": flags.get("compressed_dim", 16),
    }
    model = ALIGNIntentionModel(**cfg).to(device)
    return model


def make_random_batch(B, T, V, H, W, state_dim, device, dtype=torch.float32):
    """Generate a synthetic batch for benchmarking (avoids dataset I/O cost)."""
    frames = torch.randn(B, T, V, H, W, 3, device=device, dtype=dtype)
    states = torch.randn(B, T, 7, device=device, dtype=dtype)
    return frames, states


def make_random_features(B, T, V, H, W, device, dtype=torch.float32):
    """Pre-computed DINOv2-style features (the post-precompute path)."""
    P = (H // 14) * (W // 14)  # ViT-B/14 patch count
    features = torch.randn(B, T, V * (P + 1), 768, device=device, dtype=dtype)
    return features


# ---------------------------------------------------------------
# Measurement 1: per-step latency
# ---------------------------------------------------------------
def measure_per_step_latency(
    model, frames, states, n_warmup=20, n_runs=100, device="cuda"
):
    """Time one full forward pass (B, T, ...) → actions.

    Returns mean, std, min, max latency in milliseconds.
    """
    model.eval()
    times_ms = []
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(frames, states)
        cuda_sync()

        # Measure
        for _ in range(n_runs):
            cuda_sync()
            t0 = time.perf_counter()
            _ = model(frames, states)
            cuda_sync()
            elapsed = (time.perf_counter() - t0) * 1000.0
            times_ms.append(elapsed)

    times = np.array(times_ms)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------
# Measurement 2: throughput vs batch size
# ---------------------------------------------------------------
def measure_throughput_scaling(
    model,
    T,
    V,
    H,
    W,
    state_dim,
    device,
    batch_sizes=(1, 2, 4, 8, 16, 32, 64),
    n_warmup=5,
    n_runs=20,
):
    """Measure steps-per-second as a function of batch size."""
    results = {}
    for B in batch_sizes:
        if B > 64:
            continue
        try:
            frames, states = make_random_batch(B, T, V, H, W, state_dim, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [warn] OOM at B={B}; skipping.")
            break

        model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(frames, states)
            cuda_sync()

            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = model(frames, states)
            cuda_sync()
            elapsed = time.perf_counter() - t0

        total_steps = B * n_runs
        steps_per_sec = total_steps / elapsed
        results[B] = {
            "steps_per_sec": float(steps_per_sec),
            "mean_latency_ms": float(elapsed / n_runs * 1000),
            "per_sample_latency_ms": float(elapsed / total_steps * 1000),
        }
        # Free memory
        del frames, states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ---------------------------------------------------------------
# Measurement 3: component breakdown
# ---------------------------------------------------------------
def measure_component_breakdown(model, frames, states, n_runs=50, device="cuda"):
    """Time each submodule of the model independently.

    For ALIGN, this measures:
      - vision encoder (DINOv2)
      - state encoder
      - vision patch encoder
      - Mamba encoder (T-loop with intent tokens)
      - intention head (diffusion DDIM sampling)

    Returns dict of {component: {mean_ms, std_ms}}
    """
    model.eval()
    B, T, V = frames.shape[0], frames.shape[1], frames.shape[2]
    state_dim = states.shape[-1]

    # ---- Vision encoder ----
    times = []
    with torch.no_grad():
        for _ in range(5):
            _ = model.vision_encoder(frames.reshape(B * T * V, *frames.shape[3:]))
        cuda_sync()
        for _ in range(n_runs):
            cuda_sync()
            t0 = time.perf_counter()
            _ = model.vision_encoder(frames.reshape(B * T * V, *frames.shape[3:]))
            cuda_sync()
            times.append((time.perf_counter() - t0) * 1000)
    vision_ms = float(np.mean(times))

    # ---- State encoder ----
    times = []
    with torch.no_grad():
        for _ in range(5):
            _ = model.state_encoder(states.reshape(B * T, -1))
        cuda_sync()
        for _ in range(n_runs):
            cuda_sync()
            t0 = time.perf_counter()
            _ = model.state_encoder(states.reshape(B * T, -1))
            cuda_sync()
            times.append((time.perf_counter() - t0) * 1000)
    state_ms = float(np.mean(times))

    # ---- Full forward (sanity check) ----
    times = []
    with torch.no_grad():
        for _ in range(5):
            _ = model(frames, states)
        cuda_sync()
        for _ in range(n_runs):
            cuda_sync()
            t0 = time.perf_counter()
            _ = model(frames, states)
            cuda_sync()
            times.append((time.perf_counter() - t0) * 1000)
    full_ms = float(np.mean(times))

    # ---- Diffusion head alone (post-Mamba) ----
    # Build dummy conditioning for the head
    if model.intention_head is not None and hasattr(model.intention_head, "sample"):
        cond_dim = (
            (model.intention_head.cond_dim
             if hasattr(model.intention_head, "cond_dim") else 4352)
        )
        head_cond = torch.randn(B, 1, cond_dim, device=device)
        head_state = torch.randn(B, 1, state_dim, device=device)
        head_intent = torch.randn(B, model.num_intent_tokens, model.intent_dim, device=device)

        times = []
        with torch.no_grad():
            for _ in range(5):
                if hasattr(model.intention_head, "sample"):
                    _ = model.intention_head(head_cond, head_state, intent_emb=head_intent)
            cuda_sync()
            for _ in range(n_runs):
                cuda_sync()
                t0 = time.perf_counter()
                if hasattr(model.intention_head, "sample"):
                    _ = model.intention_head(head_cond, head_state, intent_emb=head_intent)
                cuda_sync()
                times.append((time.perf_counter() - t0) * 1000)
        head_ms = float(np.mean(times))
    else:
        head_ms = float("nan")

    # ---- Memory bank (if enabled) ----
    memory_ms = float("nan")
    if model.use_memory_bank and model.memory_module is not None:
        bank_dim = (model.pool_out_dim or 4096)
        bank_state_dim = model.state_dim
        bank_cog_dim = (
            model.intent_dim * model.num_intent_tokens
            if model.use_intent_tokens else 0
        )
        bank_perc = torch.randn(B, model.memory_bank_len, bank_dim, device=device)
        bank_state = torch.randn(B, model.memory_bank_len, bank_state_dim, device=device)
        bank_cog = (
            torch.randn(B, model.memory_bank_len, bank_cog_dim, device=device)
            if bank_cog_dim > 0 else None
        )
        query_perc = torch.randn(B, bank_dim, device=device)
        query_state = torch.randn(B, bank_state_dim, device=device)
        query_cog = (
            torch.randn(B, bank_cog_dim, device=device)
            if bank_cog_dim > 0 else None
        )

        times = []
        with torch.no_grad():
            for _ in range(5):
                _ = model.memory_module(
                    bank_perc, bank_state, bank_cog,
                    query_perc, query_state, query_cog,
                )
            cuda_sync()
            for _ in range(n_runs):
                cuda_sync()
                t0 = time.perf_counter()
                _ = model.memory_module(
                    bank_perc, bank_state, bank_cog,
                    query_perc, query_state, query_cog,
                )
                cuda_sync()
                times.append((time.perf_counter() - t0) * 1000)
        memory_ms = float(np.mean(times))

    # Sum of components should be <= full forward (parallelism, no overlap)
    sum_components = vision_ms + state_ms + head_ms
    if not np.isnan(memory_ms):
        sum_components += memory_ms

    return {
        "vision_encoder": {"mean_ms": vision_ms},
        "state_encoder": {"mean_ms": state_ms},
        "diffusion_head": {"mean_ms": head_ms},
        "memory_bank": {"mean_ms": memory_ms},
        "full_forward": {"mean_ms": full_ms},
        "sum_of_components_ms": float(sum_components),
        "overhead_ms": float(full_ms - sum_components),
        "note": "sum_of_components < full_forward if components run sequentially; "
                "overhead_ms = full_forward - sum_of_components",
    }


# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
def plot_breakdown(breakdown, output_path):
    """Bar chart of component latencies."""
    components = [
        ("Vision encoder", breakdown["vision_encoder"]["mean_ms"]),
        ("State encoder", breakdown["state_encoder"]["mean_ms"]),
        ("Diffusion head", breakdown["diffusion_head"]["mean_ms"]),
    ]
    if not np.isnan(breakdown["memory_bank"]["mean_ms"]):
        components.append(("Memory bank", breakdown["memory_bank"]["mean_ms"]))
    components.append(("Full forward", breakdown["full_forward"]["mean_ms"]))

    names = [c[0] for c in components]
    times = [c[1] for c in components]
    colors = ["#0072B2", "#E69F00", "#D55E00", "#009E73", "#000000"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color=colors[: len(names)])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("ALIGN inference component breakdown")
    ax.grid(axis="y", alpha=0.3)
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{t:.1f}",
            ha="center", va="bottom", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_scaling(scaling, output_path):
    """Steps/sec vs batch size."""
    bs = list(scaling.keys())
    sps = [scaling[b]["steps_per_sec"] for b in bs]
    per_sample = [scaling[b]["per_sample_latency_ms"] for b in bs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(bs, sps, "o-", color="#0072B2", linewidth=2, markersize=8)
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Throughput (steps/sec)")
    ax1.set_title("ALIGN throughput vs batch size")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)

    ax2.plot(bs, per_sample, "s-", color="#E69F00", linewidth=2, markersize=8)
    ax2.axhline(y=33.3, color="r", linestyle="--", alpha=0.6, label="30 Hz")
    ax2.axhline(y=10.0, color="orange", linestyle="--", alpha=0.6, label="10 Hz")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Per-sample latency (ms)")
    ax2.set_title("Effective control rate")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=str,
                        help="Path to dataset (used only for shape inference)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras", nargs="+", default=["image", "wrist_image"])
    parser.add_argument("--use-memory-bank", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for the per-step latency measurement.")
    parser.add_argument("--seq-length", type=int, default=30,
                        help="Sequence length T for the measurement batch.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-component", action="store_true",
                        help="Skip the per-component breakdown (faster).")
    parser.add_argument("--no-scaling", action="store_true",
                        help="Skip the batch-size scaling (faster).")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[measure] device: {device}")
    if not torch.cuda.is_available():
        print("[measure] WARNING: CUDA not available; timings will be CPU only.")

    # ---- Load model ----
    print(f"[measure] Loading checkpoint: {args.checkpoint}")
    # One-time checkpoint read: capture the state_dict for both flag inference
    # and weight application. We infer the flags from key inspection, build
    # the right model, then apply the cached state_dict.
    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Try common state_dict key names; fall back to the dict itself.
    if isinstance(raw, dict):
        raw_sd = (
            raw.get("state_dict")
            or raw.get("model_state_dict")
            or raw.get("model")
            or raw
        )
        # If the chosen dict has non-tensor keys (config, epoch, etc.), fall
        # back to picking only tensor entries.
        if isinstance(raw_sd, dict):
            cleaned = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in raw_sd.items()
                if isinstance(v, torch.Tensor)
            }
            # If cleaning yielded too few keys (e.g., we picked the metadata
            # dict), try harder.
            if len(cleaned) < 5 and "model_state_dict" in raw:
                cleaned = {
                    (k[len("module."):] if k.startswith("module.") else k): v
                    for k, v in raw["model_state_dict"].items()
                }
            raw_sd = cleaned
    else:
        raw_sd = raw
    inferred_flags = infer_model_flags_from_state_dict(raw_sd.keys(), raw_sd=raw_sd)
    print(f"[measure] Inferred flags from state_dict: {inferred_flags}")

    # Build the model with the correct flags, then apply the cached state_dict
    model = build_model(args, device, inferred_flags)
    print(f"[measure] use_intent_tokens={model.use_intent_tokens}, "
          f"num_intent_tokens={model.num_intent_tokens}, intent_dim={model.intent_dim}")

    model.load_state_dict(raw_sd, strict=False)
    model.eval()
    print(f"[measure] model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    T = args.seq_length
    V = len(args.cameras)
    H = W = args.image_size
    state_dim = 7
    B = args.batch_size

    # ---- Build measurement batch ----
    print(f"[measure] Building random batch: B={B}, T={T}, V={V}, {H}x{W}")
    frames, states = make_random_batch(B, T, V, H, W, state_dim, device)

    results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_size": B,
        "seq_length": T,
        "n_cameras": V,
        "image_size": H,
        "model_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "trainable_params_M": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "use_intent_tokens": model.use_intent_tokens,
        "use_memory_bank": model.use_memory_bank,
        "head_type": model.head_type,
    }

    # ---- Measurement 1: per-step latency ----
    print("\n[measure] === Per-step latency (B={}, T={}) ===".format(B, T))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lat = measure_per_step_latency(model, frames, states, device=device)
    results["per_step_latency"] = lat
    if torch.cuda.is_available():
        results["peak_vram_MB"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    for k, v in lat.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    effective_hz = 1000.0 / lat["mean_ms"] if lat["mean_ms"] > 0 else float("inf")
    results["effective_control_rate_hz"] = effective_hz
    print(f"  effective control rate: {effective_hz:.1f} Hz")

    # ---- Measurement 2: throughput scaling ----
    if not args.no_scaling and torch.cuda.is_available():
        print("\n[measure] === Throughput scaling ===")
        scaling = measure_throughput_scaling(
            model, T, V, H, W, state_dim, device,
        )
        results["throughput_scaling"] = scaling
        for B_, m in scaling.items():
            print(f"  B={B_}: {m['steps_per_sec']:.1f} steps/sec, "
                  f"{m['per_sample_latency_ms']:.2f} ms/sample")
        plot_scaling(scaling, args.output / "scaling.png")

    # ---- Measurement 3: component breakdown ----
    if not args.no_component:
        print("\n[measure] === Component breakdown ===")
        breakdown = measure_component_breakdown(model, frames, states, device=device)
        results["component_breakdown"] = breakdown
        for k, v in breakdown.items():
            if isinstance(v, dict):
                print(f"  {k}: {v.get('mean_ms', float('nan')):.3f} ms")
            else:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        plot_breakdown(breakdown, args.output / "breakdown.png")

    # ---- Save ----
    with open(args.output / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[measure] Results saved to {args.output / 'results.json'}")
    print(f"[measure] Plots:    {args.output}")


if __name__ == "__main__":
    main()
