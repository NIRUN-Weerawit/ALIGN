"""Fine-tune xvla-libero on the retained single-task Piper replay set.

TRAINING CONTRACT
INPUT: two 480x640 RGB views, one injected black 224x224 view, 8-D Piper
proprioception (xyz, quaternion, normalized gripper), and a 64-token task.
TARGET: one measured future 20-D EE6D action at 20 Hz. The inactive arm is zero.
LOSS: checkpoint EE6D position + rotation + gripper loss. This run deliberately
uses chunk_size=1 because the source labels are one-step measured future states.
METRICS: train/loss and val/loss; best.pt is selected by val/loss.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from piper_xvla.xvla_dataset import PiperXVLAConversion, PiperXVLALiberoAdapterDataset


CONFIG_KEYS = {"manifest", "checkpoint", "output", "epochs", "batch_size", "val_episodes", "lr", "device"}


def load_config(path: str | Path) -> dict:
    """Load the complete, explicit training configuration."""
    config = json.loads(Path(path).read_text())
    missing = CONFIG_KEYS - set(config)
    unknown = set(config) - CONFIG_KEYS
    if missing or unknown:
        raise ValueError(f"config keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return config


def split_episode_ids(episode_ids: list[int], val_count: int) -> tuple[list[int], list[int]]:
    if val_count <= 0 or val_count >= len(episode_ids):
        raise ValueError("val_count must be between 1 and len(episode_ids)-1")
    return episode_ids[:-val_count], episode_ids[-val_count:]


def _dataset(root: str, episode_ids: list[int], gripper: dict):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    kwargs = dict(repo_id="local/piper-replay", root=root, episodes=episode_ids)
    # LeRobot 0.5.x exposes return_uint8; older releases do not.  The
    # checkpoint processor accepts either image representation, so keep the
    # training entrypoint usable with the older environment as well.
    try:
        source = LeRobotDataset(**kwargs, return_uint8=True)
    except TypeError as exc:
        if "return_uint8" not in str(exc):
            raise
        source = LeRobotDataset(**kwargs)
    conversion = PiperXVLAConversion(gripper["raw_meters_min"], gripper["raw_meters_max"])
    return PiperXVLALiberoAdapterDataset(source, conversion)


def _prepare(batch: dict, tokenizer, device: str) -> dict:
    tasks = batch.pop("task")
    tokens = tokenizer(list(tasks), max_length=64, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    batch["observation.language.tokens"] = tokens
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mean_loss(policy, loader, tokenizer, device: str, *, train: bool, optimizer=None) -> float:
    values = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = _prepare(batch, tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss, _ = policy(batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
            values.append(float(loss.detach().cpu()))
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="piper_xvla/config/piper_xvla_single_task.json")
    args = parser.parse_args()
    config_values = load_config(args.config)

    manifest = json.loads(Path(config_values["manifest"]).read_text())
    train_ids, val_ids = split_episode_ids(manifest["source_episode_ids"], config_values["val_episodes"])
    train_loader = DataLoader(_dataset(manifest["source_dataset"], train_ids, manifest["gripper_normalization"]), batch_size=config_values["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(_dataset(manifest["source_dataset"], val_ids, manifest["gripper_normalization"]), batch_size=config_values["batch_size"], shuffle=False, num_workers=0)

    from transformers import AutoTokenizer
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    config = PreTrainedConfig.from_pretrained(config_values["checkpoint"])
    config.device, config.dtype = config_values["device"], "bfloat16"
    config.tokenizer_max_length, config.chunk_size, config.n_action_steps = 64, 1, 1
    policy = XVLAPolicy.from_pretrained(config_values["checkpoint"], config=config, local_files_only=True, strict=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config_values["lr"], betas=(0.9, 0.99))

    output = Path(config_values["output"]); output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps({"train_episode_ids": train_ids, "val_episode_ids": val_ids}, indent=2) + "\n")
    best = float("inf")
    for epoch in range(1, config_values["epochs"] + 1):
        policy.train(); train_loss = _mean_loss(policy, train_loader, tokenizer, config_values["device"], train=True, optimizer=optimizer)
        policy.eval(); val_loss = _mean_loss(policy, val_loader, tokenizer, config_values["device"], train=False)
        print(json.dumps({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss}))
        if val_loss < best:
            best = val_loss
            torch.save({"epoch": epoch, "model": policy.state_dict(), "optimizer": optimizer.state_dict(), "val_loss": val_loss, "split": {"train": train_ids, "val": val_ids}}, output / "best.pt")


if __name__ == "__main__":
    main()
