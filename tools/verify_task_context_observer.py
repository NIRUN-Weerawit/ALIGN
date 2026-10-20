"""Check the portable observer on real held-out streams against cached prefixes."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from data.task_context_dataset import TaskContextEpisodes, file_hash
from data.task_context_prefix_dataset import TaskContextPrefixes
from models.task_context_observer import load_observer
from training.train_task_context_encoder import load_checkpoint


@torch.no_grad()
def verify(bundle, student, targets, cache, output, split="confirmation", device="cuda"):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.24)
        torch.backends.cuda.matmul.allow_tf32 = False
    observer, metadata = load_observer(bundle, device)
    encoder, _ = load_checkpoint(student, device)
    ds = TaskContextPrefixes(cache, split, verify=True)
    source = TaskContextEpisodes(targets, split, verify=True)
    if metadata["student_sha256"] != file_hash(student) or metadata["prefix_manifest_sha256"] != ds.manifest["manifest_sha256"] or metadata["target_manifest_sha256"] != source.manifest["manifest_sha256"]:
        raise ValueError("Observer/source binding mismatch")
    rows, timings = [], []
    for i, key in enumerate(ds.keys):
        item = ds[i]
        summaries = torch.from_numpy(item["summary"]).to(device)[None]
        anchors = torch.from_numpy(item["anchors"]).to(device)[None]
        expected = encoder(summaries, torch.tensor([summaries.shape[1]], device=device), anchors)
        visual = torch.from_numpy(np.array(np.load(Path(targets) / f"{key}.npy", mmap_mode="r"), copy=True)).to(device)
        raw = dict(np.load(Path(targets) / f"{key}.npz", allow_pickle=False))
        state = torch.from_numpy(raw["states"]).to(device)
        times = torch.from_numpy(raw["times"]).to(device)
        stream = observer.new_stream()
        errors = {name: 0. for name in ("tokens", "logits", "execution_token")}
        for t in range(len(visual)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = observer.observe(visual[t:t + 1], state[t:t + 1], times[t:t + 1], stream)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if t >= 10:
                timings.append((time.perf_counter() - start) * 1000)
            for a in np.flatnonzero(item["anchors"] == t):
                for name in errors:
                    errors[name] = max(errors[name], float((out[name] - expected[name][:, a]).abs().max()))
        rows.append(dict(key=key, frames=len(visual), max_absolute_error=errors))
    maximum = max(max(r["max_absolute_error"].values()) for r in rows)
    report = dict(bundle_sha256=file_hash(bundle), student_sha256=file_hash(student), split=split,
        episodes=len(rows), frames=sum(r["frames"] for r in rows),
        max_absolute_error=maximum, tolerance=.0002, passed=bool(maximum < .0002),
        observer_step_ms=dict(p50=float(np.median(timings)), p95=float(np.quantile(timings, .95))),
        latency_scope="Frozen summary, state normalization, encoder/readout, input validation; preloaded DINO spatial slots. Excludes DINO extraction and action generation; shared GPU.",
        rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
    if not report["passed"]:
        raise AssertionError("Portable observer differs from the causal cached-prefix model")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "student", "targets", "cache", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", choices=["development", "confirmation"], default="confirmation")
    a = p.parse_args()
    verify(a.bundle, a.student, a.targets, a.cache, a.output, a.split, a.device)
