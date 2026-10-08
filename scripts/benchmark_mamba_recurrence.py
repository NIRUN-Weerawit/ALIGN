#!/usr/bin/env python3
"""Compare encoder-only repeated-window and causal-segment training on CUDA."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.intention_encoder import IntentionEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    model = IntentionEncoder(state_dim=64, mamba_output_dim=128, num_cameras=2,
                             compressed_dim=4, use_intent_tokens=True,
                             num_intent_tokens=1, intent_dim=128).cuda().train()
    results = []
    for length, history in ((20, 8), (40, 20), (80, 20)):
        end = length - 8 + 1
        visual = torch.randn(2, end, 2, 768, device="cuda")
        state = torch.randn(2, end, 64, device="cuda")
        samples = {"windows": [], "segment": []}
        peaks = {"windows": [], "segment": []}
        for round_idx in range(args.rounds + 2):
            # Alternate order to reduce bias from the other GPU workload.
            order = ("windows", "segment") if round_idx % 2 == 0 else ("segment", "windows")
            for name in order:
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if name == "windows":
                        intent = torch.stack([model(visual[:, t - history + 1:t + 1],
                                                    state[:, t - history + 1:t + 1])[1]
                                              for t in range(history - 1, end)], dim=1)
                    else:
                        intent = model.forward_sequence(visual, state, readout_start=history - 1)
                        intent = intent[:, history - 1:]
                    loss = intent.square().mean()
                loss.backward()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if round_idx >= 2:
                    samples[name].append(elapsed)
                    peaks[name].append(torch.cuda.max_memory_allocated() / 2**20)
                del loss, intent
        medians = {name: statistics.median(values) for name, values in samples.items()}
        record = dict(segment_length=length, history_size=history, action_chunk_size=8,
                      prediction_windows=end - history + 1, median_ms=medians,
                      speedup=medians["windows"] / medians["segment"],
                      peak_allocated_MiB={name: max(values) for name, values in peaks.items()},
                      samples_ms=samples)
        results.append(record)
        print(json.dumps(record), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
        precision="BF16 autocast; FP32 SSM recurrence", batch_size=2, results=results,
        scope="Encoder forward/backward only, shared GPU; state history differs by design"), indent=2) + "\n")


if __name__ == "__main__":
    main()
