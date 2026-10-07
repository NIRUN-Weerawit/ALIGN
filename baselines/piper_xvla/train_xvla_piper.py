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
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

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


def setup_distributed(device: str) -> tuple[int, int, int, str]:
    """Use torchrun's rank and one local device per training process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0, device
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed support is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    if device.startswith("cuda"):
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {local_rank} is unavailable for this torchrun worker")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    elif device == "cpu":
        backend = "gloo"
    else:
        raise ValueError(f"distributed training requires CUDA or CPU, got {device}")
    dist.init_process_group(backend=backend, init_method="env://")
    return dist.get_rank(), world_size, local_rank, device


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


def build_xvla_optimizer(optimizer_preset, parameters: dict, device: str) -> torch.optim.Optimizer:
    """Retain the X-VLA preset's parameter groups and fuse AdamW on CUDA."""
    optimizer = optimizer_preset.build(parameters)
    if not device.startswith("cuda"):
        return optimizer
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("fused X-VLA training requires the AdamW optimizer preset")
    # The preset's groups hold the differential VLM/soft-prompt learning rates.
    # Rebuild with fused=True rather than mutating an already initialized AdamW.
    groups = [
        {key: value for key, value in group.items() if key not in ("fused", "foreach")}
        for group in optimizer.param_groups
    ]
    return torch.optim.AdamW(groups, fused=True)


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


def _mean_loss(policy, loader, tokenizer, device: str, *, distributed: bool = False) -> float:
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch["action"])
            batch = _prepare(batch, tokenizer, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss, _ = policy(batch)
            totals[0] += loss.detach() * batch_size
            totals[1] += batch_size
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[1].item() == 0:
        raise ValueError("validation dataset is empty")
    return (totals[0] / totals[1]).item()


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
    config_values = load_config(args.config)
    rank, world_size, local_rank, device = setup_distributed(config_values["device"])
    config_values["device"] = device
    try:
        run_training(config_values, rank=rank, world_size=world_size, local_rank=local_rank)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def run_training(config_values: dict, *, rank: int, world_size: int, local_rank: int) -> None:
    distributed = world_size > 1
    device = config_values["device"]
    total_stages = 6
    if rank == 0:
        status(1, total_stages, "loading explicit training configuration")

    if rank == 0:
        status(2, total_stages, "loading manifest and validating prepared numeric cache")
    manifest = json.loads(Path(config_values["manifest"]).read_text())
    prepared_cache = load_prepared_numeric_cache(config_values["prepared_cache"], manifest)
    train_ids, val_ids = split_episode_ids(manifest["source_episode_ids"], config_values["val_episodes"])

    if rank == 0:
        status(3, total_stages, "opening train/validation image sources and attaching shared prepared labels")
    train_dataset = CachedPiperXVLALiberoDataset(
        _open_source_dataset(manifest["source_dataset"], train_ids), prepared_cache,
    )
    val_dataset = CachedPiperXVLALiberoDataset(
        _open_source_dataset(manifest["source_dataset"], val_ids), prepared_cache,
    )
    if len(train_dataset) + len(val_dataset) != manifest["frame_count"]:
        raise ValueError("source frame split does not cover the manifest frame count")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    # DistributedSampler pads uneven datasets, which would count some validation
    # frames twice. Partition validation by index instead for an exact mean.
    local_val_dataset = Subset(val_dataset, range(rank, len(val_dataset), world_size)) if distributed else val_dataset
    train_loader = DataLoader(
        train_dataset, batch_size=config_values["batch_size"], shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=0,
    )
    val_loader = DataLoader(local_val_dataset, batch_size=config_values["batch_size"], shuffle=False, num_workers=0)

    if rank == 0:
        status(4, total_stages, "loading tokenizer, X-VLA checkpoint, optimizer, and scheduler")
    if configure_cuda_attention(config_values["device"]):
        if rank == 0:
            print("[train stage 4/6] disabled cuDNN SDPA; retaining Flash/efficient SDPA backends", flush=True)
    from transformers import AutoTokenizer
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    config = load_xvla_config(config_values["checkpoint"])
    apply_xvla_finetuning_config(config, config_values)
    policy = XVLAPolicy.from_pretrained(config_values["checkpoint"], config=config, local_files_only=True, strict=True)
    optimizer_preset = config.get_optimizer_preset()
    optimizer = build_xvla_optimizer(optimizer_preset, policy.get_optim_params(), device)
    scheduler = config.get_scheduler_preset().build(optimizer, config_values["steps"])
    training_policy = DistributedDataParallel(
        policy, device_ids=[local_rank] if device.startswith("cuda") else None,
        find_unused_parameters=True,
    ) if distributed else policy

    if rank == 0:
        status(5, total_stages, "writing reproducibility artifacts")
    output = Path(config_values["output"])
    split = {"train_episode_ids": train_ids, "val_episode_ids": val_ids}
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    best = float("inf")
    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    recent_train_loss_total = 0.0
    recent_train_examples = 0
    if rank == 0:
        status(6, total_stages, f"starting {config_values['steps']:,} optimizer steps on {world_size} process(es); batch size {config_values['batch_size']} per process; validation/save every {config_values['validate_every_steps']:,}/{config_values['save_every_steps']:,} steps")
    for step in range(1, config_values["steps"] + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch_size = len(batch["action"])
        training_policy.train()
        loss = _train_step(training_policy, batch, tokenizer, device, optimizer, scheduler, optimizer_preset.grad_clip_norm)
        recent_train_loss_total += loss * batch_size
        recent_train_examples += batch_size

        validate_now = step % config_values["validate_every_steps"] == 0 or step == config_values["steps"]
        save_now = step % config_values["save_every_steps"] == 0 or step == config_values["steps"]
        val_loss = None
        if validate_now:
            policy.eval()
            val_loss = _mean_loss(policy, val_loader, tokenizer, device, distributed=distributed)
            train_totals = torch.tensor([recent_train_loss_total, recent_train_examples], dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(json.dumps({"step": step, "train/loss": (train_totals[0] / train_totals[1]).item(), "val/loss": val_loss}), flush=True)
            recent_train_loss_total = 0.0
            recent_train_examples = 0
            if rank == 0 and val_loss < best:
                best = val_loss
                _save_checkpoint(output / "best.pt", step=step, policy=policy, optimizer=optimizer, scheduler=scheduler, val_loss=val_loss, split=split)
        if save_now and rank == 0:
            _save_checkpoint(output / "last.pt", step=step, policy=policy, optimizer=optimizer, scheduler=scheduler, val_loss=val_loss, split=split)
        if distributed and (validate_now or save_now):
            dist.barrier()


if __name__ == "__main__":
    main()
