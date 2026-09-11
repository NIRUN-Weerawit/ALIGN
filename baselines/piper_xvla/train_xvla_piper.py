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
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from piper_xvla.xvla_dataset import CachedPiperXVLALiberoDataset, load_prepared_numeric_cache


CONFIG_KEYS = {
    "manifest", "prepared_cache", "checkpoint", "output", "steps", "batch_size", "val_episodes", "device",
    "action_mode", "freeze_vision_encoder", "freeze_language_encoder",
    "train_policy_transformer", "train_soft_prompts", "optimizer_lr",
    "scheduler_warmup_steps", "scheduler_decay_steps", "scheduler_decay_lr",
    "validate_every_steps", "save_every_steps",
}


def status(stage: int, total: int, message: str) -> None:
    """Emit clear, flushed lifecycle markers for long-running remote training."""
    print(f"[train stage {stage}/{total}] {message}", flush=True)


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


def configure_cuda_attention(device: str) -> bool:
    """Avoid cuDNN SDPA plan failures while retaining Flash/efficient attention."""
    if not device.startswith("cuda"):
        return False
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if not callable(enable_cudnn_sdp):
        return False
    enable_cudnn_sdp(False)
    return True


def load_xvla_config(checkpoint: str | Path):
    """Load a typed X-VLA config across LeRobot public-API layouts."""
    # Importing the concrete class registers the checkpoint's `type: xvla`
    # discriminator with LeRobot's generic configuration loader.
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: F401
    try:
        from lerobot.configs import PreTrainedConfig
    except ImportError:
        from lerobot.configs.policies import PreTrainedConfig
    return PreTrainedConfig.from_pretrained(checkpoint)


def apply_xvla_finetuning_config(config, values: dict) -> None:
    """Apply the official X-VLA new-embodiment fine-tuning settings explicitly."""
    config.device = values["device"]
    config.dtype = "bfloat16"
    config.action_mode = values["action_mode"]
    config.tokenizer_max_length = 64
    # Piper pendant replay supplies one measured future-state label per frame.
    config.chunk_size = config.n_action_steps = 1
    for key in (
        "freeze_vision_encoder", "freeze_language_encoder",
        "train_policy_transformer", "train_soft_prompts", "optimizer_lr",
        "scheduler_warmup_steps", "scheduler_decay_steps", "scheduler_decay_lr",
    ):
        setattr(config, key, values[key])


def _open_source_dataset(root: str, episode_ids: list[int]):
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
    return source


def _prepare(batch: dict, tokenizer, device: str) -> dict:
    tasks = batch.pop("task")
    tokens = tokenizer(list(tasks), max_length=64, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    batch["observation.language.tokens"] = tokens
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mean_loss(policy, loader, tokenizer, device: str) -> float:
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = _prepare(batch, tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss, _ = policy(batch)
            values.append(float(loss.detach().cpu()))
    return sum(values) / len(values)


def _train_step(policy, batch: dict, tokenizer, device: str, optimizer, scheduler, grad_clip_norm: float) -> float:
    batch = _prepare(batch, tokenizer, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        loss, _ = policy(batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    optimizer.step()
    scheduler.step()
    return float(loss.detach().cpu())


def serialize_policy_config(config) -> dict:
    """Serialize both modern and legacy LeRobot configuration objects."""
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _save_checkpoint(path: Path, *, step: int, policy, optimizer, scheduler, val_loss: float | None, split: dict) -> None:
    torch.save(
        {
            "step": step,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "split": split,
            "xvla_config": serialize_policy_config(policy.config),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="piper_xvla/config/piper_xvla_single_task.json")
    args = parser.parse_args()
    total_stages = 6
    status(1, total_stages, "loading explicit training configuration")
    config_values = load_config(args.config)

    status(2, total_stages, "loading manifest and validating prepared numeric cache")
    manifest = json.loads(Path(config_values["manifest"]).read_text())
    prepared_cache = load_prepared_numeric_cache(config_values["prepared_cache"], manifest)
    train_ids, val_ids = split_episode_ids(manifest["source_episode_ids"], config_values["val_episodes"])

    status(3, total_stages, "opening train/validation image sources and attaching shared prepared labels")
    train_dataset = CachedPiperXVLALiberoDataset(
        _open_source_dataset(manifest["source_dataset"], train_ids), prepared_cache,
    )
    val_dataset = CachedPiperXVLALiberoDataset(
        _open_source_dataset(manifest["source_dataset"], val_ids), prepared_cache,
    )
    if len(train_dataset) + len(val_dataset) != manifest["frame_count"]:
        raise ValueError("source frame split does not cover the manifest frame count")
    train_loader = DataLoader(train_dataset, batch_size=config_values["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config_values["batch_size"], shuffle=False, num_workers=0)

    status(4, total_stages, "loading tokenizer, X-VLA checkpoint, optimizer, and scheduler")
    if configure_cuda_attention(config_values["device"]):
        print("[train stage 4/6] disabled cuDNN SDPA; retaining Flash/efficient SDPA backends", flush=True)
    from transformers import AutoTokenizer
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    config = load_xvla_config(config_values["checkpoint"])
    apply_xvla_finetuning_config(config, config_values)
    policy = XVLAPolicy.from_pretrained(config_values["checkpoint"], config=config, local_files_only=True, strict=True)
    optimizer_preset = config.get_optimizer_preset()
    optimizer = optimizer_preset.build(policy.get_optim_params())
    scheduler = config.get_scheduler_preset().build(optimizer, config_values["steps"])

    status(5, total_stages, "writing reproducibility artifacts")
    output = Path(config_values["output"]); output.mkdir(parents=True, exist_ok=True)
    split = {"train_episode_ids": train_ids, "val_episode_ids": val_ids}
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    best = float("inf")
    train_iter = iter(train_loader)
    recent_train_losses: list[float] = []
    status(6, total_stages, f"starting {config_values['steps']:,} optimizer steps; validation/save every {config_values['validate_every_steps']:,}/{config_values['save_every_steps']:,} steps")
    for step in range(1, config_values["steps"] + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        policy.train()
        recent_train_losses.append(_train_step(
            policy, batch, tokenizer, config_values["device"], optimizer, scheduler,
            optimizer_preset.grad_clip_norm,
        ))

        validate_now = step % config_values["validate_every_steps"] == 0 or step == config_values["steps"]
        save_now = step % config_values["save_every_steps"] == 0 or step == config_values["steps"]
        val_loss = None
        if validate_now:
            policy.eval()
            val_loss = _mean_loss(policy, val_loader, tokenizer, config_values["device"])
            print(json.dumps({"step": step, "train/loss": sum(recent_train_losses) / len(recent_train_losses), "val/loss": val_loss}))
            recent_train_losses.clear()
            if val_loss < best:
                best = val_loss
                _save_checkpoint(output / "best.pt", step=step, policy=policy, optimizer=optimizer, scheduler=scheduler, val_loss=val_loss, split=split)
        if save_now:
            _save_checkpoint(output / "last.pt", step=step, policy=policy, optimizer=optimizer, scheduler=scheduler, val_loss=val_loss, split=split)


if __name__ == "__main__":
    main()
