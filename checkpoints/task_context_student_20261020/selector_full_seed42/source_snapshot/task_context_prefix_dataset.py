"""Hash-bound frozen summaries, privileged labels, and causal prefix sampling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.task_context_dataset import TaskContextEpisodes, file_hash, json_hash

SCHEMA = "align-task-context-prefix-v1"
EVAL_FRACTIONS = (0., .1, .25, .4, .5, .7, .9)


@torch.no_grad()
def build_prefix_cache(targets, latents, checkpoint_path, output, device="cpu"):
    from training.train_task_context_teacher import load_checkpoint
    targets, latents, output = Path(targets), Path(latents), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new prefix cache: {output}")
    metadata = json.loads((latents / "manifest.json").read_text())
    if metadata["schema"] != "align-privileged-task-tokens-v1" or file_hash(checkpoint_path) != metadata["teacher_sha256"]:
        raise ValueError("Frozen teacher hash mismatch")
    train = TaskContextEpisodes(targets, "train", verify=True)
    if train.manifest["manifest_sha256"] != metadata["target_manifest_sha256"]:
        raise ValueError("Teacher labels and source targets differ")
    teacher, checkpoint = load_checkpoint(checkpoint_path, device)
    teacher.requires_grad_(False).eval()
    if checkpoint["config"]["manifest_sha256"] != metadata["target_manifest_sha256"]:
        raise ValueError("Teacher was trained on a different target manifest")
    output.mkdir(parents=True)
    files = {}
    all_train = None
    for split, info in metadata["splits"].items():
        source = latents / f"{split}.npz"
        if file_hash(source) != info["sha256"]:
            raise ValueError("Privileged latent file hash mismatch")
        values = dict(np.load(source, allow_pickle=False))
        keys = values["keys"].tolist()
        ds = train if split == "train" else TaskContextEpisodes(targets, split, verify=True)
        if keys != ds.keys or len(set(keys)) != len(keys):
            raise ValueError("Latent split keys differ from the observation split")
        if values["task_tokens"].shape != (len(keys), *metadata["shape"]) or not np.isfinite(values["task_tokens"]).all():
            raise ValueError("Invalid teacher tokens")
        if split == "train":
            all_train = values["task_tokens"]
        for i, key in enumerate(keys):
            item = ds[i]
            tensor = lambda x: torch.from_numpy(np.array(x, dtype=np.float32, copy=True)).to(device)
            # The frozen summary is frame-local; encoding never visits another frame.
            summary = teacher.summary(tensor(item["features"])[None], tensor(item["states"])[None], tensor(item["times"])[None])[0]
            raw = dict(np.load(targets / f"{key}.npz", allow_pickle=False))
            cls = item["features"][:, :, 0] / ds.norm["visual_scale"][:, 0]
            path = output / f"{key}.npz"
            np.savez(path, summary=summary.float().cpu().numpy(), states=item["states"],
                cls=cls.reshape(len(cls), -1), times=item["times"], actions=raw["actions"],
                terminal=item["terminal"], procedure=item["procedure"], procedure_states=item["procedure_states"],
                task_tokens=values["task_tokens"][i], group=item["group"])
            files[key] = file_hash(path)
        print(json.dumps(dict(cached_split=split, episodes=len(keys))), flush=True)
    np.savez(output / "normalization.npz", latent_mean=all_train.mean(0), latent_std=np.maximum(all_train.std(0), .1))
    manifest = dict(schema=SCHEMA, teacher_sha256=metadata["teacher_sha256"],
        target_manifest_sha256=metadata["target_manifest_sha256"], latent_manifest_sha256=file_hash(latents / "manifest.json"),
        source_hashes={name: file_hash(Path(__file__).resolve().parents[1] / name) for name in
            ["data/task_context_prefix_dataset.py", "models/task_context_teacher.py"]},
        normalization_sha256=file_hash(output / "normalization.npz"),
        split=train.manifest["split"], files=files, action_horizon=train.manifest["action_horizon"],
        fps=train.manifest["fps"], width=teacher.config["width"], shape=metadata["shape"],
        visual_width=len(train.manifest["cameras"]) * teacher.config["raw_dim"],
        supervision="Whole-episode tokens/targets are labels only. Frozen summaries are frame-local observations.",
        anchor_policy="16 uniform training anchors: event/handover/disambiguation annotations unavailable.",
        limitations=train.manifest["limitations"] + ["Confirmation split previously characterized for teacher; not a fresh final task benchmark."])
    manifest["manifest_sha256"] = json_hash(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class TaskContextPrefixes(Dataset):
    def __init__(self, root, split, *, anchors=16, seed=42, verify=False):
        self.root, self.split, self.anchors, self.seed, self.epoch = Path(root), split, anchors, seed, 0
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if m["schema"] != SCHEMA or json_hash({k: v for k, v in m.items() if k != "manifest_sha256"}) != m["manifest_sha256"]:
            raise ValueError("Invalid prefix manifest")
        keys = [k for rows in m["split"].values() for k in rows]
        if len(set(keys)) != len(keys) or set(keys) != set(m["files"]):
            raise ValueError("Prefix split overlap or omission")
        if file_hash(self.root / "normalization.npz") != m["normalization_sha256"]:
            raise ValueError("Prefix normalization hash mismatch")
        self.keys = m["split"][split]
        self.norm = dict(np.load(self.root / "normalization.npz", allow_pickle=False))
        self._cache = {}
        if verify:
            for key in self.keys:
                if file_hash(self.root / f"{key}.npz") != m["files"][key]:
                    raise ValueError(f"Prefix cache hash mismatch: {key}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        if key not in self._cache:
            self._cache[key] = dict(np.load(self.root / f"{key}.npz", allow_pickle=False))
        v = self._cache[key]
        n, horizon = len(v["summary"]), self.manifest["action_horizon"]
        if self.split == "train":
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            anchors = np.sort(rng.choice(n - horizon + 1, self.anchors, replace=n - horizon + 1 < self.anchors))
        else:
            anchors = np.rint(np.asarray(EVAL_FRACTIONS) * (n - horizon)).astype(np.int64)
        return dict(key=key, **{name: v[name] for name in ["summary", "states", "cls", "times", "terminal", "procedure", "procedure_states", "task_tokens", "group"]},
            anchors=anchors, actions=np.stack([v["actions"][t:t + horizon] for t in anchors]))


def collate_prefixes(items):
    lengths = np.asarray([len(x["summary"]) for x in items])
    batch = dict(keys=[x["key"] for x in items], lengths=torch.from_numpy(lengths))
    for name in ("summary", "states", "cls", "times"):
        value = np.zeros((len(items), max(lengths), *items[0][name].shape[1:]), dtype=np.float32)
        for i, x in enumerate(items):
            value[i, :lengths[i]] = x[name]
        batch[name] = torch.from_numpy(value)
    for name in ("anchors", "actions", "terminal", "procedure", "procedure_states", "task_tokens", "group"):
        batch[name] = torch.as_tensor(np.stack([x[name] for x in items]))
    return batch
