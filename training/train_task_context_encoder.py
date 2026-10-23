"""Distill a frozen demonstration teacher into a causal Mamba prefix belief.

The teacher, summary, and diagnostic action head remain frozen. No deployed
policy is trained here. Selection uses student top-weight modes on development.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.task_context_dataset import file_hash
from data.task_context_prefix_dataset import TaskContextPrefixes, collate_prefixes, EVAL_FRACTIONS
from models.task_context_encoder import TaskContextEncoder, belief_loss, execution_loss, mixture_log_prob, selector_loss
from training.train_task_context_teacher import load_checkpoint as load_teacher, seed_all, to_device

SOURCES = ["models/task_context_encoder.py", "data/task_context_prefix_dataset.py", "training/train_task_context_encoder.py"]


def make_loader(ds, batch_size=8, shuffle=False, seed=42):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=collate_prefixes, generator=torch.Generator().manual_seed(seed))


def normalized_target(model, batch):
    return ((batch["task_tokens"] - model.latent_mean) / model.latent_std)[:, None].expand(-1, batch["anchors"].shape[1], -1, -1)


def assigned_tokens(model, out, batch):
    # Privileged discrete responsibility is training-only. It never selects evaluation modes.
    mode = mixture_log_prob(out, normalized_target(model, batch)).detach().argmax(-1)
    native = out["means"] * model.latent_std + model.latent_mean
    return native.gather(2, mode[..., None, None, None].expand(-1, -1, 1,
        model.config["tokens"], model.config["token_dim"])).squeeze(2)


def outcome_loss(teacher, tokens, batch):
    bi = torch.arange(len(tokens), device=tokens.device)[:, None]
    current = batch["summary"][bi, batch["anchors"]]
    decoded = teacher.decode(tokens, current)
    f = torch.nn.functional.smooth_l1_loss
    return (f(decoded["terminal"].float(), batch["terminal"][:, None].expand_as(decoded["terminal"]))
        + f(decoded["procedure"].float(), batch["procedure"][:, None].expand_as(decoded["procedure"]))
        + f(decoded["procedure_states"].float(), batch["procedure_states"][:, None].expand_as(decoded["procedure_states"]))) / 3


@torch.no_grad()
def batch_metrics(model, teacher, batch, out):
    bi = torch.arange(len(batch["anchors"]), device=batch["anchors"].device)[:, None]
    current = batch["summary"][bi, batch["anchors"]]
    decoded = teacher.decode(out["tokens"], current)
    target = normalized_target(model, batch)
    chosen = (out["tokens"].float() - model.latent_mean) / model.latent_std
    joint = mixture_log_prob(out, target)
    error = (decoded["regression"].float() - batch["actions"]).square()
    flat = model.config["tokens"] * model.config["token_dim"]
    metrics = dict(nll=-torch.logsumexp(joint, -1) / flat,
        latent_mse=(chosen - target).square().mean((-1, -2)),
        oracle_latent_mse=(out["means"].float() - target[:, :, None]).square().mean((-1, -2)).min(-1).values,
        terminal=(decoded["terminal"].float() - batch["terminal"][:, None]).square().flatten(2).mean(2),
        procedure=(decoded["procedure"].float() - batch["procedure"][:, None]).square().flatten(2).mean(2),
        procedure_state=(decoded["procedure_states"].float() - batch["procedure_states"][:, None]).square().flatten(2).mean(2),
        action_normalized=(error / teacher.action_std.square()).mean((-1, -2)),
        position=error[..., :3].mean((-1, -2)), rotation=error[..., 3:6].mean((-1, -2)),
        gripper=error[..., 6].mean(-1),
        entropy=-(out["logits"].log_softmax(-1) * out["logits"].softmax(-1)).sum(-1))
    for hi, h in enumerate(model.config["horizons"]):
        valid = batch["anchors"] + h < batch["lengths"][:, None]
        idx = torch.minimum(batch["anchors"] + h, batch["lengths"][:, None] - 1)
        visual = batch["cls"][bi, idx] - batch["cls"][bi, batch["anchors"]]
        state = batch["states"][bi, idx] - batch["states"][bi, batch["anchors"]]
        pv, ps = out["future"][:, :, hi].float().split([model.config["visual_width"], 10], -1)
        err = ((pv - visual).square().mean(-1) + (ps - state).square().mean(-1)) / 2
        persistence = (visual.square().mean(-1) + state.square().mean(-1)) / 2
        metrics[f"future_h{h}"] = err.masked_fill(~valid, float("nan"))
        metrics[f"persistence_h{h}"] = persistence.masked_fill(~valid, float("nan"))
    return metrics


@torch.no_grad()
def evaluate(model, teacher, loader, device):
    model.eval()
    rows = []
    modes = np.zeros(model.config["modes"], dtype=int)
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["summary"], batch["lengths"], batch["anchors"])
        metrics = batch_metrics(model, teacher, batch, out)
        modes += np.bincount(out["mode"].cpu().numpy().ravel(), minlength=len(modes))
        for i, key in enumerate(batch["keys"]):
            rows.append(dict(key=key, **{name: np.nanmean(v[i].cpu().numpy()).item() for name, v in metrics.items()}))
    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "key"}
    # Exclude oracle selection, action targets, and confirmation from checkpoint choice.
    means["selection"] = float(np.mean([means[k] for k in ("latent_mse", "terminal", "procedure", "procedure_state")]))
    means["top_mode_counts"] = modes.tolist()
    return means, rows


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TaskContextEncoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from data.training_split import PIPER_PREFIXES, PIPER_TEACHER, validate_goal_training_manifest, validation_partition
    p.add_argument('--cache',default=str(PIPER_PREFIXES))
    p.add_argument('--teacher',default=str(PIPER_TEACHER))
    p.add_argument('--output',required=True)
    p.add_argument("--history", choices=["full", "current", "short"], default="full")
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--anchors", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--future-weight", type=float, default=.2)
    p.add_argument("--outcome-weight", type=float, default=.2)
    p.add_argument("--selector-weight", type=float, default=.2)
    p.add_argument("--evaluate-every", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-fraction", type=float, default=.24)
    p.add_argument("--final-refit", action=argparse.BooleanOptionalAction, default=True,
        help="Fixed-budget training without validation or checkpoint selection; requires empty development split")
    p.add_argument('--allow-legacy-split',action='store_true',help='Explicitly reproduce a non-Piper historical Goal cache')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.anchors, a.evaluate_every) < 1 or min(a.future_weight, a.outcome_weight, a.selector_weight) < 0:
        p.error("Invalid training configuration")
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f"Use a new run directory: {output}")
    device = torch.device(a.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_cudnn_sdp(False)
    seed_all(a.seed)
    train = TaskContextPrefixes(a.cache, "train", anchors=a.anchors, seed=a.seed, verify=True)
    if not a.allow_legacy_split:validate_goal_training_manifest(train.manifest)
    if a.final_refit and train.manifest["split"]["development"]:
        raise ValueError("Final refit requires an empty development split")
    dev = None if a.final_refit else TaskContextPrefixes(a.cache, validation_partition(train.manifest['split']), verify=True)
    if file_hash(a.teacher) != train.manifest["teacher_sha256"]:
        raise ValueError("Teacher changed after prefix caching")
    teacher, _ = load_teacher(a.teacher, device)
    teacher.requires_grad_(False).eval()
    model = TaskContextEncoder(width=train.manifest["width"], tokens=train.manifest["shape"][0],
        token_dim=train.manifest["shape"][1], visual_width=train.manifest["visual_width"], history=a.history, window=a.window).to(device)
    with torch.no_grad():
        model.latent_mean.copy_(torch.from_numpy(train.norm["latent_mean"]))
        model.latent_std.copy_(torch.from_numpy(train.norm["latent_std"]))
    optimizer = torch.optim.AdamW([dict(params=[v for v in model.parameters() if not getattr(v, "_no_weight_decay", False)], weight_decay=1e-4),
        dict(params=[v for v in model.parameters() if getattr(v, "_no_weight_decay", False)], weight_decay=0.)], lr=a.lr)
    config = dict(vars(a), model_config=model.config, manifest_sha256=train.manifest["manifest_sha256"],
        checkpoint_selection="fixed_budget_final" if a.final_refit else "development",
        teacher_sha256=file_hash(a.teacher), source_hashes={name: file_hash(ROOT / name) for name in SOURCES},
        parameters=sum(v.numel() for v in model.parameters()), evaluation_fractions=list(EVAL_FRACTIONS),
        contract="Causal observation-only prefix. Frozen teacher labels; top-weight evaluation; no policy fine-tuning.")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in SOURCES:
        (snapshot / Path(source).name).write_bytes((ROOT / source).read_bytes())
    print(json.dumps(config), flush=True)
    best, step, start = float("inf"), 0, time.perf_counter()
    train_loader = make_loader(train, a.batch_size, True, a.seed)
    dev_loader = None if a.final_refit else make_loader(dev, a.batch_size)
    for epoch in range(a.epochs):
        train.epoch = epoch
        model.train()
        sums, count = {}, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch["summary"], batch["lengths"], batch["anchors"])
                nll = belief_loss(model, out, batch["task_tokens"])
                selector = selector_loss(model, out, batch["task_tokens"])
                future, _ = execution_loss(model, out, batch)
                outcome = outcome_loss(teacher, assigned_tokens(model, out, batch), batch) if a.outcome_weight else nll * 0
                loss = nll + a.future_weight * future + a.outcome_weight * outcome + a.selector_weight * selector
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
            step += 1
            count += len(batch["keys"])
            for k, v in dict(loss=loss, nll=nll, future=future, outcome=outcome, selector=selector).items():
                sums[k] = sums.get(k, 0.) + float(v.detach()) * len(batch["keys"])
            if a.max_steps and step >= a.max_steps:
                break
        row = dict(epoch=epoch + 1, steps=step, elapsed_s=time.perf_counter() - start, train={k: v / count for k, v in sums.items()})
        if not a.final_refit and (epoch == 0 or (epoch + 1) % a.evaluate_every == 0 or epoch + 1 == a.epochs or (a.max_steps and step >= a.max_steps)):
            metrics, rows = evaluate(model, teacher, dev_loader, device)
            row["development"] = metrics
            if metrics["selection"] < best:
                best = metrics["selection"]
                checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    model_config=model.config, config=config, epoch=epoch + 1, steps=step, development=metrics)
                temporary = output / "student.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "student_best.pt")
                (output / "best_development.json").write_text(json.dumps(rows, indent=2) + "\n")
        if device.type == "cuda":
            row["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(json.dumps(row), flush=True)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if a.max_steps and step >= a.max_steps:
            break
    if any(v.grad is not None or v.requires_grad for v in teacher.parameters()):
        raise AssertionError("Teacher was not frozen")
    if a.final_refit:
        checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            model_config=model.config, config=config, epoch=epoch+1, steps=step,
            checkpoint_selection="fixed_budget_final")
        temporary = output / "final.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "student_final.pt")
    print(json.dumps(dict(completed=True, best_selection=None if a.final_refit else best, steps=step, elapsed_s=time.perf_counter() - start)), flush=True)


if __name__ == "__main__":
    main()
