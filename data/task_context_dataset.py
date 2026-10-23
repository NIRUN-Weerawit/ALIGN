"""Versioned, instruction-free whole-demonstration teacher data.

Task strings are used only to hash offline groups. They never enter features.
Unknown success/command/event provenance remains explicit in the manifest.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.gripper_state import previous_gripper_commands


SCHEMA = "align-task-context-targets-v1"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_visual(x):
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-5)


def spatial_summary(features, cameras):
    """Camera-major [CLS, top-left, top-right, bottom-left, bottom-right]."""
    if features.ndim != 3 or features.shape[1] != cameras * 257:
        raise ValueError("Expected per-camera 16x16 patches followed by CLS")
    f = features.reshape(len(features), cameras, 257, features.shape[-1])
    grid = f[:, :, :-1].reshape(len(f), cameras, 2, 8, 2, 8, f.shape[-1])
    cells = grid.mean(axis=(3, 5)).reshape(len(f), cameras, 4, f.shape[-1])
    return normalized_visual(np.concatenate([f[:, :, -1:], cells], axis=2))


def geometric_state(poses, gripper):
    """Translation, first two rotation-matrix columns, causal gripper (10D)."""
    from scipy.spatial.transform import Rotation
    poses = np.asarray(poses, dtype=np.float32)
    rotation = Rotation.from_rotvec(poses[:, 3:6]).as_matrix()[:, :, :2]
    return np.concatenate([poses[:, :3], rotation.reshape(len(poses), 6),
                           np.asarray(gripper)[:, None]], -1).astype(np.float32)


def grouped_split(groups, seed, development_fraction=.1, confirmation_fraction=.1):
    if not 0 < development_fraction < 1 or not 0 < confirmation_fraction < 1:
        raise ValueError("Held-out fractions must be positive")
    if development_fraction + confirmation_fraction >= 1:
        raise ValueError("No training fraction remains")
    result = {k: [] for k in ("train", "development", "confirmation")}
    by_group = defaultdict(list)
    for key, group in groups.items():
        by_group[group].append(key)
    rng = np.random.default_rng(seed)
    for group in sorted(by_group):
        keys = rng.permutation(sorted(by_group[group])).tolist()
        if len(keys) < 3:
            raise ValueError("Every offline group needs at least three episodes")
        ndev = max(1, round(len(keys) * development_fraction))
        ntest = max(1, round(len(keys) * confirmation_fraction))
        if ndev + ntest >= len(keys):
            raise ValueError("Group too small for requested split")
        result["development"].extend(keys[:ndev])
        result["confirmation"].extend(keys[ndev:ndev + ntest])
        result["train"].extend(keys[ndev + ntest:])
    return {k: sorted(v) for k, v in result.items()}


def build_targets(data, cache, output, *, cameras, fps, seed=20261020,
                  terminal_tail=8, action_horizon=8, procedure_points=8,
                  annotations=None, block_size=16, split_mode='auto'):
    data, cache, output = Path(data).resolve(), Path(cache).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Target directory already exists: {output}")
    if min(fps, terminal_tail, action_horizon, procedure_points, block_size) <= 0:
        raise ValueError("Target sizes and fps must be positive")
    index = json.loads((cache / "index.json").read_text())
    meta = index.get("__meta__", {})
    if (meta.get("format") != "align-dinov2-per-camera-v2" or
            meta.get("cameras") != list(cameras) or meta.get("source") != str(data)):
        raise ValueError("Cache format, source, or camera order mismatch")
    annotation = json.loads(Path(annotations).read_text()) if annotations else {}
    groups, episodes = {}, {}
    with h5py.File(data, "r") as source:
        keys = sorted(k for k in source if k.startswith("ep_"))
        for key in keys:
            group = source[key]
            raw = group["texts"][()]
            descriptions = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            label = descriptions[0] if isinstance(descriptions, list) else str(descriptions)
            group_id = hashlib.sha256(label.strip().encode()).hexdigest()[:16]
            spec = annotation.get(key, {})
            if spec.get("success") is False:
                continue  # Failed endpoints are not desired-goal targets.
            n = len(group["poses"])
            if key not in index or index[key]["length"] != n or n < action_horizon:
                raise ValueError(f"Episode/cache length mismatch or short episode: {key}")
            groups[key] = spec.get("objective_group_id", group_id)
            interval = spec.get("terminal_interval", [max(0, n-terminal_tail), n])
            if not 0 <= interval[0] < interval[1] <= n:
                raise ValueError(f"Invalid terminal interval for {key}")
            if "actions" not in group or group["actions"].shape != (n, 7):
                raise ValueError(f"Seven-dimensional expert action targets required: {key}")
            episodes[key] = dict(length=n, group_id=groups[key], terminal_interval=interval,
                success=spec.get("success"), terminal_approved=bool(spec.get("success") is True and "terminal_interval" in spec),
                events=spec.get("events", []), input_commands="unavailable_unverified",
                state_gripper="measured" if "gripper" in group else "previous_recorded_action",
                timestamp_source="recorded" if "timestamps" in group else "index_over_declared_fps",
                annotations=spec)
    from data.training_split import piper_partition
    fixed = piper_partition(list(episodes), {k:v['length'] for k,v in episodes.items()}, str(data)) if split_mode == 'auto' else None
    split = (dict(train=fixed[0], development=[], confirmation=fixed[1])
             if fixed is not None else grouped_split(groups, seed))
    group_ids = {g: i for i, g in enumerate(sorted(set(groups.values())))}
    output.mkdir(parents=True)
    files = {}
    with h5py.File(data, "r") as source:
        for number, (key, ep) in enumerate(episodes.items()):
            n = ep["length"]
            raw = np.load(cache / f"{key}.npy", mmap_mode="r")
            if raw.shape != (n, len(cameras)*257, 768) or raw.dtype != np.float32:
                raise ValueError(f"Invalid raw cached features: {key}")
            # Sequential blocks keep both host memory and mmap reads bounded.
            feature_path = output / f"{key}.npy"
            summary = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32,
                                               shape=(n, len(cameras), 5, 768))
            for start in range(0, n, block_size):
                summary[start:start+block_size] = spatial_summary(raw[start:start+block_size], len(cameras))
            summary.flush()
            group = source[key]
            actions = np.asarray(group["actions"], dtype=np.float32)
            if "gripper" in group:
                gripper = np.asarray(group["gripper"], dtype=np.float32).reshape(n)
            else:
                gripper = previous_gripper_commands(actions, initial_gripper=float(group.attrs.get("initial_gripper", 0.)))
            states = geometric_state(group["poses"][()], gripper)
            times = (np.asarray(group["timestamps"], dtype=np.float32) if "timestamps" in group
                     else np.arange(n, dtype=np.float32) / fps)
            if times.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f"Invalid observation timestamps: {key}")
            times -= times[0]
            proc = np.rint(np.linspace(0, n-1, procedure_points)).astype(np.int64)
            begin, end = ep["terminal_interval"]
            terminal = np.array(summary[begin:end].mean(0), copy=True)
            initial = np.array(summary[0], copy=True)
            procedure = np.array(summary[proc], copy=True)
            if not all(np.isfinite(x).all() for x in [summary, states, actions]):
                raise ValueError(f"Nonfinite real observation/target: {key}")
            target_path = output / f"{key}.npz"
            np.savez(target_path, states=states, actions=actions, times=times,
                initial=initial, terminal=terminal, procedure=procedure,
                terminal_delta=terminal-initial, procedure_delta=procedure-initial,
                procedure_indices=proc, procedure_states=states[proc],
                group=np.int64(group_ids[ep["group_id"]]))
            files[key] = {"features_sha256": file_hash(feature_path), "targets_sha256": file_hash(target_path)}
            del summary, raw
            if number % 20 == 0 or number+1 == len(episodes):
                print(f"targets {number+1}/{len(episodes)}: {key}", flush=True)
    # No held-out data contributes to fitted normalization or constant targets.
    train_states, terminal_deltas, procedure_deltas = [], [], []
    for key in split["train"]:
        with np.load(output / f"{key}.npz") as values:
            train_states.append(values["states"])
            terminal_deltas.append(values["terminal_delta"])
            procedure_deltas.append(values["procedure_delta"])
    states = np.concatenate(train_states)
    terminal_deltas = np.stack(terminal_deltas)
    procedure_deltas = np.stack(procedure_deltas)
    # A scalar scale per camera/slot preserves within-slot feature geometry.
    visual_scale = np.maximum(terminal_deltas.std(axis=(0, 3)), .05)[..., None]
    np.savez(output / "normalization.npz", state_mean=states.mean(0),
        state_std=np.maximum(states.std(0), .05), visual_scale=visual_scale,
        terminal_mean=terminal_deltas.mean(0), procedure_mean=procedure_deltas.mean(0))
    manifest = dict(schema=SCHEMA, source=str(data), source_stat={"size":data.stat().st_size,"mtime_ns":data.stat().st_mtime_ns},
        raw_cache=str(cache), cache_index_sha256=file_hash(cache / "index.json"), cameras=list(cameras), fps=float(fps),
        seed=seed, action_horizon=action_horizon, procedure_points=procedure_points,
        terminal_tail=terminal_tail, feature_shape=[len(cameras),5,768], state_width=10,
        split=split, groups=group_ids, episodes=episodes, files=files,
        normalization_sha256=file_hash(output / "normalization.npz"),
        limitations=["No verified raw human commands are encoder inputs.",
            "Unannotated terminal intervals are observed outcomes, not approved goals.",
            "Episode split is not a verified scene/operator-disjoint split.",
            "Task grouping uses hashed descriptions as offline labels only.",
            "Procedure anchors supervise realized traces, not hidden constraints."])
    manifest["manifest_sha256"] = json_hash(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print({k:len(v) for k,v in split.items()}, flush=True)
    return manifest


class TaskContextEpisodes(Dataset):
    def __init__(self, root, split, *, anchors=8, seed=0, verify=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        expected = self.manifest["manifest_sha256"]
        if self.manifest["schema"] != SCHEMA or json_hash({k:v for k,v in self.manifest.items() if k != "manifest_sha256"}) != expected:
            raise ValueError("Invalid target manifest")
        all_keys = [k for values in self.manifest["split"].values() for k in values]
        if len(all_keys) != len(set(all_keys)) or set(all_keys) != set(self.manifest["episodes"]):
            raise ValueError("Episode split overlap or omission")
        if file_hash(self.root/"normalization.npz") != self.manifest["normalization_sha256"]:
            raise ValueError("Normalization hash mismatch")
        self.keys = self.manifest["split"][split]
        self.split, self.anchors, self.seed, self.epoch = split, anchors, seed, 0
        self.norm = dict(np.load(self.root / "normalization.npz"))
        self._cache = {}
        if verify:
            for key in self.keys:
                for suffix, field in [("npy","features_sha256"),("npz","targets_sha256")]:
                    if file_hash(self.root / f"{key}.{suffix}") != self.manifest["files"][key][field]:
                        raise ValueError(f"Target cache hash mismatch: {key}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        if key not in self._cache:
            self._cache[key] = (np.load(self.root/f"{key}.npy", mmap_mode="r"),
                               dict(np.load(self.root/f"{key}.npz")))
        features, values = self._cache[key]
        n, horizon = len(features), self.manifest["action_horizon"]
        if self.split == "train":
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            times = np.sort(rng.choice(n-horizon+1, min(self.anchors,n-horizon+1), replace=False))
        else:
            times = np.rint(np.linspace(.1,.7,self.anchors) * (n-horizon)).astype(np.int64)
        return dict(key=key, features=np.asarray(features),
            states=(values["states"]-self.norm["state_mean"])/self.norm["state_std"],
            times=values["times"], anchors=times,
            actions=np.stack([values["actions"][t:t+horizon] for t in times]),
            terminal=values["terminal_delta"]/self.norm["visual_scale"],
            procedure=values["procedure_delta"]/self.norm["visual_scale"],
            procedure_states=(values["procedure_states"]-self.norm["state_mean"])/self.norm["state_std"],
            group=int(values["group"]))


def collate_task_context(items):
    length = max(len(item["features"]) for item in items)
    batch = dict(keys=[x["key"] for x in items], lengths=torch.tensor([len(x["features"]) for x in items]))
    for key in ("features", "states", "times"):
        shape = (len(items),length,*items[0][key].shape[1:])
        value = np.zeros(shape,dtype=np.float32)
        for i,item in enumerate(items):
            value[i,:len(item[key])] = item[key]
        batch[key] = torch.from_numpy(value)
    for key in ("anchors","actions","terminal","procedure","procedure_states","group"):
        batch[key] = torch.as_tensor(np.stack([x[key] for x in items]))
    return batch
