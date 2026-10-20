"""Prefix information, history interventions, execution prediction, and streaming.

All student metrics use the top-weight hypothesis. The frozen teacher is an
explicitly privileged diagnostic. Fractions select assessment anchors only.
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

from data.task_context_dataset import file_hash
from data.task_context_prefix_dataset import TaskContextPrefixes, EVAL_FRACTIONS
from training.train_task_context_encoder import (load_checkpoint, make_loader, batch_metrics)
from training.train_task_context_teacher import load_checkpoint as load_teacher, to_device
from tools.assess_task_context_teacher import paired_difference, donor_indices

MIDDLE = [2, 3, 4, 5]  # 25%, 40%, 50%, 70% of valid action-anchor range.
METRICS = ("terminal", "procedure", "procedure_state", "action_normalized", "position", "rotation", "gripper")


@torch.no_grad()
def decoded_metrics(teacher, tokens, current, batch):
    out = teacher.decode(tokens, current)
    err = (out["regression"] - batch["actions"]).square()
    return dict(terminal=(out["terminal"] - batch["terminal"][:, None]).square().flatten(2).mean(2),
        procedure=(out["procedure"] - batch["procedure"][:, None]).square().flatten(2).mean(2),
        procedure_state=(out["procedure_states"] - batch["procedure_states"][:, None]).square().flatten(2).mean(2),
        action_normalized=(err / teacher.action_std.square()).mean((-1, -2)),
        position=err[..., :3].mean((-1, -2)), rotation=err[..., 3:6].mean((-1, -2)), gripper=err[..., 6].mean(-1))


@torch.no_grad()
def extract(model, teacher, ds, device):
    arrays = {}
    tokens, weights, keys = [], [], []
    for batch in make_loader(ds):
        keys += batch["keys"]
        batch = to_device(batch, device)
        out = model(batch["summary"], batch["lengths"], batch["anchors"])
        for name, value in batch_metrics(model, teacher, batch, out).items():
            arrays.setdefault(name, []).append(value.cpu().numpy())
        tokens.append(out["tokens"].cpu().numpy())
        weights.append(out["logits"].softmax(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in arrays.items()}, dict(keys=keys, tokens=np.concatenate(tokens), weights=np.concatenate(weights))


@torch.no_grad()
def reference_metrics(teacher, ds, device, mean):
    result = {"teacher_oracle": {}, "train_mean": {}}
    for batch in make_loader(ds):
        batch = to_device(batch, device)
        bi = torch.arange(len(batch["keys"]), device=device)[:, None]
        current = batch["summary"][bi, batch["anchors"]]
        shape = (len(batch["keys"]), batch["anchors"].shape[1], *mean.shape)
        for name, z in [("teacher_oracle", batch["task_tokens"][:, None].expand(shape)),
                        ("train_mean", mean.expand(shape))]:
            for metric, v in decoded_metrics(teacher, z, current, batch).items():
                result[name].setdefault(metric, []).append(v.cpu().numpy())
    return {name: {k: np.concatenate(v) for k, v in metrics.items()} for name, metrics in result.items()}


@torch.no_grad()
def history_interventions(model, teacher, ds, device):
    items = [ds[i] for i in range(len(ds))]
    same, wrong = donor_indices(np.asarray([x["group"] for x in items]))
    metrics = {name: {} for name in ("same_task_history", "wrong_task_history", "reset_history", "recent_history")}
    for start in range(0, len(items), 8):
        rows = items[start:start + 8]
        batch = next(iter(make_loader_subset(rows)))
        batch = to_device(batch, device)
        bi = torch.arange(len(rows), device=device)[:, None]
        current = batch["summary"][bi, batch["anchors"]]
        for condition in metrics:
            sequences = []
            for j, row in enumerate(rows):
                for a, anchor in enumerate(row["anchors"]):
                    if condition == "reset_history":
                        prefix = row["summary"][anchor:anchor + 1]
                    elif condition == "recent_history":
                        prefix = row["summary"][max(0, anchor - 15):anchor + 1]
                    else:
                        donor = items[int((same if condition == "same_task_history" else wrong)[start + j])]
                        donor_anchor = donor["anchors"][a]
                        # Preserve the recipient's exact current summary. This is
                        # a synthetic history intervention, not a matched-scene rollout.
                        prefix = np.concatenate([donor["summary"][:donor_anchor], row["summary"][anchor:anchor + 1]])
                    sequences.append(prefix)
            lengths = torch.tensor([len(x) for x in sequences], device=device)
            x = torch.zeros(len(sequences), int(lengths.max()), model.config["width"], device=device)
            for j, prefix in enumerate(sequences):
                x[j, :len(prefix)] = torch.from_numpy(prefix).to(device)
            out = model(x, lengths, (lengths - 1)[:, None])
            tokens = out["tokens"][:, 0].reshape(len(rows), len(EVAL_FRACTIONS), *ds.manifest["shape"])
            for name, value in decoded_metrics(teacher, tokens, current, batch).items():
                metrics[condition].setdefault(name, []).append(value.cpu().numpy())
    return {name: {k: np.concatenate(v) for k, v in values.items()} for name, values in metrics.items()}


def make_loader_subset(rows):
    from data.task_context_prefix_dataset import collate_prefixes
    return [collate_prefixes(rows)]


@torch.no_grad()
def streaming_audit(model, ds, device):
    item = ds[0]
    x = torch.from_numpy(item["summary"]).to(device)[None]
    count = x.shape[1]
    anchors = torch.arange(count, device=device)[None]
    expected = model(x, torch.tensor([count], device=device), anchors)
    state = model.new_stream()
    maximum = 0.
    durations = []
    for t in range(count):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.step(x[:, t], state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if t >= 10:
            durations.append((time.perf_counter() - start) * 1000)
        maximum = max(maximum, float((out["tokens"] - expected["tokens"][:, t]).abs().max()))
    anchor = count // 2
    prefix = model(x[:, :anchor + 1], torch.tensor([anchor + 1], device=device), torch.tensor([[anchor]], device=device))["tokens"]
    changed = x.clone()
    changed[:, anchor + 1:] = float("nan")
    mutated = model(changed, torch.tensor([count], device=device), torch.tensor([[anchor]], device=device))["tokens"]
    mutation = float((prefix - mutated).abs().max())
    reset = model.step(x[:, 0], model.new_stream())["tokens"]
    reset_error = float((reset - expected["tokens"][:, 0]).abs().max())
    return dict(episode=item["key"], frames=count, stream_max_absolute_error=maximum,
        future_mutation_max_absolute_error=mutation, reset_max_absolute_error=reset_error,
        passed=bool(maximum < 2e-4 and mutation < 2e-4 and reset_error < 2e-4),
        cached_summary_step_ms=dict(p50=float(np.median(durations)), p95=float(np.quantile(durations, .95))),
        latency_scope="Encoder/readout after cached frozen summary; excludes vision, summary extraction, action generation; shared GPU load.")


def summarize(values):
    result = {}
    for name, x in values.items():
        def finite_mean(v):
            v = np.asarray(v)
            return float(v[np.isfinite(v)].mean()) if np.isfinite(v).any() else None
        result[name] = dict(all_prefixes=finite_mean(x), middle=finite_mean(x[:, MIDDLE]),
            by_prefix=[finite_mean(x[:, a]) for a in range(len(EVAL_FRACTIONS))])
    return result


def evaluate_gates(values, audit):
    comparisons = {}
    def compare(a, b, metric):
        av, bv = values[a][metric][:, MIDDLE].mean(1), values[b][metric][:, MIDDLE].mean(1)
        return dict(relative_improvement=float(1 - av.mean() / bv.mean()),
            benefit=paired_difference(bv, av))
    for baseline in ("train_mean", "current", "short"):
        comparisons[baseline] = {k: compare("full", baseline, k) for k in ("terminal", "procedure", "action_normalized")}
    retained = {}
    for metric in ("terminal", "procedure"):
        mean, full, oracle = [values[name][metric][:, MIDDLE].mean() for name in ("train_mean", "full", "teacher_oracle")]
        retained[metric] = float((mean - full) / (mean - oracle)) if mean > oracle else None
    history = {}
    for condition in ("same_task_history", "wrong_task_history", "reset_history", "recent_history"):
        history[condition] = {k: paired_difference(values[condition][k][:, MIDDLE].mean(1), values["full"][k][:, MIDDLE].mean(1))
            for k in ("terminal", "procedure", "action_normalized")}
    wrong_same = paired_difference(values["wrong_task_history"]["action_normalized"][:, MIDDLE].mean(1),
        values["same_task_history"]["action_normalized"][:, MIDDLE].mean(1))
    information = audit["passed"] and all(comparisons["train_mean"][k]["relative_improvement"] >= .10
        and comparisons["train_mean"][k]["benefit"]["ci95"][0] > 0
        and retained[k] is not None and retained[k] >= .5 for k in ("terminal", "procedure"))
    memory = information and all(comparisons[b][k]["relative_improvement"] >= .05 and comparisons[b][k]["benefit"]["ci95"][0] > 0
        for b in ("current", "short") for k in ("terminal", "procedure")) and wrong_same["ci95"][0] > 0
    return dict(causal_information=dict(passed=bool(information),
        rule="Causal audit; middle-prefix terminal/procedure >=10% below train-mean context, paired benefit CI>0, >=50% of teacher-over-mean benefit retained."),
        history_advantage=dict(passed=bool(memory),
        rule="Information gate plus >=5% terminal AND procedure gain over both independently trained current/16-frame controls, paired CI>0; wrong-history continuation worse than same-task history."),
        comparisons=comparisons, teacher_advantage_retained=retained,
        fixed_current_history_harm=history, wrong_minus_same_history_action=wrong_same)


def validate_confirmation(report, paths, manifest):
    if report.get("split") != "development" or not report.get("gates", {}).get("causal_information", {}).get("passed"):
        raise ValueError("Confirmation requires a passing development information gate")
    if report.get("manifest_sha256") != manifest:
        raise ValueError("Prefix manifest changed after development")
    for name, path in paths.items():
        if report["checkpoints"][name]["sha256"] != file_hash(path):
            raise ValueError("Selected checkpoint changed after development")


def execution_assessment(values):
    """Use only real future pairs; cluster uncertainty by episode, not by frame."""
    report = {}
    for h in (16, 64):
        key = f"future_h{h}"
        def per_episode(v):
            count = np.isfinite(v).sum(1)
            if (count == 0).any():
                raise ValueError("Execution comparison has an episode with no real future pair")
            return np.nansum(v, axis=1) / count
        full = per_episode(values["full"][key])
        comparisons = {}
        for name in ("persistence", "current", "short"):
            baseline = values["full"][f"persistence_h{h}"] if name == "persistence" else values[name][key]
            other = per_episode(baseline)
            comparisons[name] = dict(relative_improvement=float(1 - full.mean() / other.mean()), benefit=paired_difference(other, full))
        report[str(h)] = dict(valid_pairs=int(np.isfinite(values["full"][key]).sum()), comparisons=comparisons)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "teacher", "full", "current", "short", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", choices=["development", "confirmation"], default="development")
    p.add_argument("--confirm-development-report")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f"Use a new assessment path: {output}")
    device = torch.device(a.device)
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.24)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.enable_cudnn_sdp(False)
    ds = TaskContextPrefixes(a.cache, a.split, verify=True)
    paths = {name: getattr(a, name) for name in ("full", "current", "short", "teacher")}
    if a.split == "confirmation":
        if not a.confirm_development_report:
            p.error("Confirmation requires --confirm-development-report")
        validate_confirmation(json.loads(Path(a.confirm_development_report).read_text()), paths, ds.manifest["manifest_sha256"])
    if file_hash(a.teacher) != ds.manifest["teacher_sha256"]:
        raise ValueError("Frozen teacher changed")
    teacher, _ = load_teacher(a.teacher, device)
    teacher.requires_grad_(False)
    values, artifacts, checkpoints = {}, {}, {}
    full_model = None
    reference_config = None
    for name in ("full", "current", "short"):
        model, ckpt = load_checkpoint(paths[name], device)
        if ckpt["config"]["manifest_sha256"] != ds.manifest["manifest_sha256"] or ckpt["config"]["teacher_sha256"] != file_hash(a.teacher):
            raise ValueError("Student teacher/cache binding mismatch")
        if model.config["history"] != name:
            raise ValueError("Student supplied for the wrong history condition")
        cfg = {k: v for k, v in ckpt["config"].items() if k not in ("output", "history", "model_config")}
        model_cfg = {k: v for k, v in model.config.items() if k != "history"}
        comparable = (cfg, model_cfg)
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise ValueError("Student controls do not have matched architecture/training budgets")
        values[name], artifacts[name] = extract(model, teacher, ds, device)
        checkpoints[name] = dict(sha256=file_hash(paths[name]), epoch=ckpt["epoch"], steps=ckpt["steps"])
        if name == "full":
            full_model = model
        else:
            del model
    checkpoints["teacher"] = dict(sha256=file_hash(a.teacher))
    values.update(reference_metrics(teacher, ds, device, full_model.latent_mean))
    values.update(history_interventions(full_model, teacher, ds, device))
    audit = streaming_audit(full_model, ds, device)
    report = dict(split=a.split, episodes=len(ds), manifest_sha256=ds.manifest["manifest_sha256"],
        checkpoints=checkpoints, evaluation_fractions=list(EVAL_FRACTIONS), middle_indices=MIDDLE,
        selection="Student top-weight mode only; oracle_latent_mse is explicitly a privileged coverage diagnostic.",
        metrics={name: summarize(v) for name, v in values.items()},
        mode_counts={name: np.bincount(v["weights"].argmax(-1).ravel(), minlength=4).tolist() for name, v in artifacts.items()},
        streaming=audit, gates=evaluate_gates(values, audit), execution=execution_assessment(values),
        limitations=ds.manifest["limitations"] + ["No physical handover, action-head fine-tuning, or closed-loop success evaluation.",
            "Known-task episode split; current scenes often identify task. Task/procedure constraints not verified.",
            "Donor histories preserve current summaries but are not physically matched histories; transitions may be unnatural.",
            "Mixture weights are not calibrated task correctness or takeover probabilities.",
            "One training seed; four components can collapse; confirmation remains the teacher's previously characterized episode split."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    arrays = {f"{name}__{metric}": x for name, v in values.items() for metric, x in v.items()}
    arrays.update({f"{name}__{field}": v[field] for name, v in artifacts.items() for field in ("tokens", "weights")})
    np.savez(output.with_suffix(".npz"), keys=np.asarray(ds.keys), **arrays)
    print(json.dumps(dict(split=a.split, gates=report["gates"], streaming=audit), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
