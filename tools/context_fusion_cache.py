"""Pack frozen, checkpoint-bound expert features for context-fusion experiments.

Only training and development observations are read. The compact cache contains
the exact pretrained actor's visual/state inputs, native causal context tokens,
and expert continuation targets. It is an offline screen, not rollout evidence.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.align_dataset import ALIGNDataset
from data.task_context_dataset import file_hash, json_hash
from eval.eval_intention import load_intention_model
from eval.task_estimation import context_fingerprint
from models.task_context_policy import TaskContextPolicyBranch


SCHEMA = "align-context-fusion-feature-cache-v1"
FRACTIONS = (.4, .5, .7, .9)


def _resolve_saved_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _check_partitions(cfg, manifest, probe):
    partitions = manifest["split"]
    names = ("train", "development", "confirmation")
    all_keys = [key for name in names for key in partitions[name]]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Prefix-cache episode partitions overlap")
    for name in names:
        if set(cfg["task_context_split"][name + "_episodes"]) != set(partitions[name]):
            raise ValueError(f"Policy and prefix-cache {name} episodes differ")
    for name in ("train", "development"):
        if set(probe["splits"][name]) != set(partitions[name]):
            raise ValueError(f"Probe and prefix-cache {name} episodes differ")
    if probe.get("confirmation_opened"):
        raise ValueError("This screen requires a training/development-only probe")
    return partitions


@torch.inference_mode()
def build_cache(checkpoint, data, cache, token_probe, prefix_cache, output, device):
    """Build and save a weights-only-loadable packed dict; return that dict.

    ``output`` is a new .pt file. ``token_probe`` and ``prefix_cache`` are the
    existing separability and observation-summary directories. Cached normalized
    task tokens can be reused across policy checkpoints only when the complete
    frozen observer, normalization, and causal query definitions are identical.
    All learned actor feature extraction is FP32 with autocast disabled.
    """
    checkpoint, data, cache = map(Path, (checkpoint, data, cache))
    token_probe, prefix_cache, output = map(Path, (token_probe, prefix_cache, output))
    device = torch.device(device)
    if output.exists():
        raise FileExistsError(output)
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    manifest_path = prefix_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if json_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("Invalid prefix-cache manifest hash")
    if file_hash(prefix_cache / "normalization.npz") != manifest["normalization_sha256"]:
        raise ValueError("Prefix-cache normalization hash differs")
    probe_path = token_probe / "protocol.json"
    probe = json.loads(probe_path.read_text())
    if probe.get("schema") != "align-task-token-separability-v1":
        raise ValueError("Expected the causal task-token separability cache")
    if probe["prefix_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Probe and prefix cache are not bound")
    query_indices = []
    for fraction in FRACTIONS:
        matches = [i for i, query in enumerate(probe["queries"]) if query.get("fraction") == fraction]
        if len(matches) != 1:
            raise ValueError(f"Probe must contain exactly one {fraction:g} causal query")
        query_indices.append(matches[0])

    model, cfg = load_intention_model(str(checkpoint), device)
    model.requires_grad_(False).eval()
    if (model.task_context is None or model.use_memory_bank or model.text_condition is not None
            or model.history_size != 1 or model.visual_context_mode != "none"):
        raise ValueError("Feature screen requires context_no_memory, no text, history 1, and no extra visual pathway")
    if cfg.get("task_context_trainable") or cfg.get("task_context_initialization", "pretrained") != "pretrained":
        raise ValueError("Only a frozen pretrained context branch can reuse these tokens")
    partitions = _check_partitions(cfg, manifest, probe)
    branch = model.task_context
    if branch.config["metadata"]["prefix_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Policy observer is not bound to the prefix cache")
    observer_path = _resolve_saved_path(cfg["task_context_checkpoint"])
    observer_hash = file_hash(observer_path)
    if observer_hash != probe["observer_sha256"] or observer_hash != branch.config["bundle_sha256"]:
        raise ValueError("Probe and policy original observer identities differ")
    reference = TaskContextPolicyBranch(checkpoint=str(observer_path)).requires_grad_(False).eval()
    fingerprint = context_fingerprint(branch)
    if fingerprint != context_fingerprint(reference):
        raise ValueError("Policy observation/context weights differ from the original probe observer")
    del reference
    encoder = branch.observer.encoder
    dataset = ALIGNDataset(str(data), mode="head", cameras=cfg["task_context_split"]["cameras"],
        traj_window=model.chunk_size, dinov2_path=str(cache))
    expected_keys = set().union(*(set(partitions[name]) for name in partitions))
    if set(dataset._episode_keys) != expected_keys:
        dataset.close()
        raise ValueError("Dataset episodes differ from the protected split")
    index = {key: i for i, key in enumerate(dataset._episode_keys)}
    metadata = dict(schema=SCHEMA, checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_hash(checkpoint),
        context_fingerprint=fingerprint, original_observer_sha256=observer_hash,
        probe_checkpoint_sha256=probe["policy_sha256"],
        reused_across_policy_checkpoint=probe["policy_sha256"] != file_hash(checkpoint),
        prefix_manifest_sha256=manifest["manifest_sha256"], prefix_manifest_file_sha256=file_hash(manifest_path),
        probe_protocol_sha256=file_hash(probe_path),
        train_features_sha256=file_hash(token_probe / "train_features.npz"),
        development_features_sha256=file_hash(token_probe / "development_features.npz"),
        data=str(data.resolve()), cache=str(cache.resolve()), cameras=list(dataset.cameras),
        fractions=list(FRACTIONS), chunk_size=model.chunk_size,
        precision="FP32 frozen features; autocast disabled; checkpoint TF32 setting retained",
        task_context_matmul_tf32=cfg.get("task_context_matmul_tf32"),
        split={name: list(partitions[name]) for name in ("train", "development")},
        confirmation_opened=False, weights_updated=False,
        source_sha256={name: file_hash(ROOT / name) for name in (
            "tools/context_fusion_cache.py", "models/task_context_policy.py", "models/align_intention.py")},
        contract="Cached expert observations and continuation targets. Labels are metadata only. No simulator success claim.")
    packed = {"metadata": metadata}
    started = time.monotonic()
    max_summary_error = 0.
    audits = []
    try:
        for split in ("train", "development"):
            fields = {name: [] for name in ("visual", "states", "task", "execution", "frozen_task40", "actions")}
            fields.update(episodes=[], tasks=[], anchors=[], fractions=[])
            with np.load(token_probe / f"{split}_features.npz", allow_pickle=False) as source:
                keys = source["keys"].tolist()
                if len(keys) != len(set(keys)) or set(keys) != set(partitions[split]):
                    raise ValueError(f"Probe {split} feature episodes differ from the protected split")
                feature_index = {key: i for i, key in enumerate(keys)}
                for ei, key in enumerate(partitions[split]):
                    ep = index[key]
                    fi = feature_index[key]
                    length = dataset._get_episode_length(ep)
                    if length != int(source["lengths"][fi]):
                        raise ValueError(f"{key}: episode length differs from cached context")
                    anchors = np.asarray([int(length * f) for f in FRACTIONS], dtype=np.int64)
                    if not np.array_equal(source["anchors"][fi, query_indices], anchors):
                        raise ValueError(f"{key}: context query indices differ")
                    if np.any(anchors + model.chunk_size > length):
                        raise ValueError(f"{key}: a query lacks a full continuation target")
                    prefix_path = prefix_cache / f"{key}.npz"
                    if file_hash(prefix_path) != manifest["files"][key]:
                        raise ValueError(f"{key}: prefix-cache file hash differs")
                    with np.load(prefix_path, allow_pickle=False) as prefix:
                        if len(prefix["summary"]) != length:
                            raise ValueError(f"{key}: prefix-cache length differs")
                        expected_summary = torch.from_numpy(prefix["summary"][anchors].copy()).to(device)
                        if not np.array_equal(prefix["summary"][anchors], source["summary"][fi, query_indices]):
                            raise ValueError(f"{key}: probe observations differ from the bound prefix cache")
                        native = torch.from_numpy(source["tokens"][fi, query_indices].copy()).to(device)
                        native = native * encoder.latent_std + encoder.latent_mean
                        execution = torch.from_numpy(source["execution"][fi, query_indices].copy()).to(device)
                        if ei == 0:
                            # Audit actual causal execution rather than trusting metadata alone.
                            observed = int(anchors[-1]) + 1
                            sequence = torch.from_numpy(prefix["summary"][:observed].copy()).to(device)[None]
                            direct = encoder(sequence, torch.tensor([observed], device=device),
                                torch.from_numpy(anchors).to(device)[None])
                            task_error = float((direct["tokens"][0] - native).abs().max())
                            exec_error = float((direct["execution_token"][0, :, 0] - execution).abs().max())
                            if task_error > 2e-4 or exec_error > 2e-4:
                                raise ValueError(f"{key}: cached causal readout differs ({task_error}, {exec_error})")
                            audits.append(dict(split=split, episode=key,
                                task_max_error=task_error, execution_max_error=exec_error))
                    frames = np.stack([np.array(dataset._read_frames_dinov2_view(ep, int(t), 1), copy=True)[0]
                        for t in anchors])
                    robot = np.stack([np.concatenate((dataset._read_poses(ep, int(t), 1)[0, :6],
                        np.asarray([dataset._read_poses_gripper(ep, int(t), 1)[0]]))) for t in anchors]).astype(np.float32)
                    raw = torch.from_numpy(frames).to(device=device, dtype=torch.float32).reshape(
                        len(anchors), model.num_cameras, 257, 768)
                    patches = raw[:, :, :-1].reshape(len(anchors), -1, 768)
                    cls = raw[:, :, -1]
                    robot_tensor = torch.from_numpy(robot).to(device)
                    with torch.autocast(device_type=device.type, enabled=False):
                        fresh = branch.summaries(patches[:, None], cls[:, None], robot_tensor[:, None],
                            torch.from_numpy(anchors).to(device=device, dtype=torch.float32)[:, None] / branch.fps)[:, 0]
                        error = float((fresh - expected_summary).abs().max())
                        max_summary_error = max(max_summary_error, error)
                        if error > 2e-4:
                            raise ValueError(f"{key}: cached observer summary differs from actor inputs ({error})")
                        state = model.state_encoder(robot_tensor)[:, None]
                        visual = model.encode_visual_features(patches, state[:, 0]).flatten(1)[:, None]
                        payload = torch.cat((native, execution[:, None]), dim=1)
                        visual, state, payload = model.condition_actions(visual, state, payload)
                    targets = torch.from_numpy(np.stack([dataset._read_actions(ep, int(t), model.chunk_size)
                        for t in anchors]).astype(np.float32))
                    values = dict(visual=visual, states=state, task=payload[:, :-1],
                        execution=payload[:, -1], frozen_task40=payload[0:1, :-1].expand(len(anchors), -1, -1),
                        actions=targets)
                    for name, value in values.items():
                        value = value.detach().float().cpu().clone()
                        if not torch.isfinite(value).all():
                            raise ValueError(f"{key}: nonfinite {name} features")
                        fields[name].append(value)
                    fields["episodes"].extend([key] * len(anchors))
                    fields["tasks"].extend([dataset._read_text(ep)] * len(anchors))
                    fields["anchors"].extend(anchors.tolist())
                    fields["fractions"].extend(FRACTIONS)
                    if getattr(dataset, "_dinov2_memmaps", None) is not None:
                        dataset._dinov2_memmaps.pop(ep, None)
                    if (ei + 1) % 20 == 0 or ei + 1 == len(partitions[split]):
                        print(f"CACHE {split} {ei+1}/{len(partitions[split])} elapsed={time.monotonic()-started:.1f}s", flush=True)
            packed[split] = {name: torch.cat(fields[name]) for name in
                ("visual", "states", "task", "execution", "frozen_task40", "actions")}
            packed[split].update(episodes=fields["episodes"], tasks=fields["tasks"],
                anchors=torch.tensor(fields["anchors"], dtype=torch.int64),
                fractions=torch.tensor(fields["fractions"], dtype=torch.float32))
    finally:
        dataset.close()
    metadata.update(summary_max_error=max_summary_error, causal_readout_audits=audits,
        elapsed_seconds=time.monotonic()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(packed, temporary)
    temporary.replace(output)
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--token-probe", required=True)
    parser.add_argument("--prefix-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.24)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    packed = build_cache(args.checkpoint, args.data, args.cache, args.token_probe,
        args.prefix_cache, args.output, args.device)
    print(json.dumps(packed["metadata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
