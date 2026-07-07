#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALIGN Task Identification Head — training script.

Trains the TaskHead to predict which LIBERO task the user is performing
from (z_v, z_t). The head's softmax is the source of both the gating
signal α and the language conditioning for the assistant head
(see docs/TASK_IDENTIFICATION_HEAD.md).

The assistant head is trained separately — the task head only needs to
produce good task beliefs. E_clip is built from the same CLIP text
encoder the assistant head was trained on, so z_text = p @ E_clip is
already in the right semantic space.

OOD training:
  - --num-ood-tasks N reserves N randomly chosen task descriptions as
    "held-out OOD" — they are never used as K known classes, only as
    OOD training samples. Default 0 (no held-out tasks; uses augmented
    OOD only).
  - On-the-fly Gaussian-noise augmentation produces synthetic OOD
    frames from in-distribution data, controlled by
    --ood-augment-prob (default 0.2, i.e. ~20% of in-distribution
    samples get an OOD label).

Usage:
    PYTHONNOUSERSITE=1 python training/train_task_head.py \\
        --data h5_data/libero_10.h5 \\
        --pretrained checkpoints/pretrain/best.pt \\
        --epochs 10

    # With held-out real OOD tasks
    PYTHONNOUSERSITE=1 python training/train_task_head.py \\
        --data h5_data/libero_10.h5 \\
        --pretrained checkpoints/pretrain/best.pt \\
        --num-ood-tasks 2 \\
        --epochs 10

IMPORTANT: Run with `PYTHONNOUSERSITE=1` to prevent the user-site
`~/.local/` pip packages from shadowing the conda-installed cuDNN
(9.10.2 → 9.2.0), which causes `CUDNN_STATUS_NOT_INITIALIZED` on
torch 2.10.0+cu128. This matches the convention used by all other
ALIGN training scripts and is already in the project README.
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.align_model import ALIGNModel
from models.task_head import (
    TaskHead, TaskHeadBundle, create_task_head, task_head_loss,
)
from data.align_dataset import (
    ALIGNDataset, MultiALIGNDataset, head_collate, TRAJ_WINDOW,
)
from training.wandb_utils import init_wandb, log_metrics


# ================================================================
# Task vocabulary
# ================================================================

def build_task_vocabulary(h5_path: str) -> List[str]:
    """Walk the HDF5 and extract the unique task_description from each
    episode's meta. Returns a sorted list of unique strings."""
    tasks: set = set()
    with h5py.File(h5_path, "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("ep_"):
                continue
            ep = f[key]
            # Prefer the canonical task_description from meta
            try:
                meta = json.loads(ep["meta"][()])
                if "task_description" in meta:
                    tasks.add(meta["task_description"])
                    continue
            except (KeyError, ValueError):
                pass
            # Fallback: take the first text from `texts`
            try:
                raw = ep["texts"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                texts = json.loads(raw)
                if isinstance(texts, list) and texts:
                    tasks.add(str(texts[0]))
            except (KeyError, ValueError):
                pass
    return sorted(tasks)


def split_vocabulary(
    all_tasks: List[str],
    num_ood: int,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split a task list into (known, ood) by random sample."""
    if num_ood < 0:
        raise ValueError(f"num_ood must be >= 0, got {num_ood}")
    if num_ood >= len(all_tasks):
        raise ValueError(
            f"num_ood ({num_ood}) >= total tasks ({len(all_tasks)}). "
            f"Need at least 1 known class."
        )
    rng = random.Random(seed)
    shuffled = list(all_tasks)
    rng.shuffle(shuffled)
    return sorted(shuffled[num_ood:]), sorted(shuffled[:num_ood])


# ================================================================
# Collate: emits (z_v-ready frame, z_t-ready traj, text, task_id, ood_label)
# ================================================================

def task_collate(
    batch: list,
    chunk_size: int,
    task_to_id: Dict[str, int],
    ood_set: set,
    augment_ood_prob: float,
    rng: np.random.Generator,
) -> dict:
    """Collate batch for task-head training.

    Reuses head_collate for the vision/trajectory samples, then attaches:
      - task_id:    (B,) int64 in [0, K), or -100 if OOD (ignored in CE)
      - ood_label:  (B,) float32 in {0, 1} — 1 if OOD task OR augmented

    Augmentation: with probability `augment_ood_prob`, add Gaussian noise
    to the frame and label the sample OOD. The model must learn that a
    corrupted frame is not a recognized task.

    Returns: dict compatible with head_collate outputs plus task_id and
             ood_label.
    """
    # First build the standard head batch (frames, trajectory, text, ...)
    head_batch = head_collate(batch, chunk_size=chunk_size)
    B = head_batch["frames"].shape[0]

    task_ids = np.full(B, -100, dtype=np.int64)
    ood_labels = np.zeros(B, dtype=np.float32)

    for i, item in enumerate(batch):
        # The dataset gives a list of text variants; the standard collate
        # already picked one. Re-derive the canonical task description
        # for this episode via the same fallback the dataset uses.
        text_i = head_batch["texts"][i]
        # text_i may be a list of variants or a single string; normalize
        if isinstance(text_i, list):
            canonical = str(text_i[0])
        else:
            canonical = str(text_i)

        if canonical in ood_set:
            # Held-out OOD task
            ood_labels[i] = 1.0
            # task_id stays -100 (ignored by CE)
        else:
            # In-distribution
            if canonical in task_to_id:
                task_ids[i] = task_to_id[canonical]
            else:
                # Shouldn't happen if vocab was built from this dataset,
                # but be defensive: treat unknown strings as OOD.
                ood_labels[i] = 1.0

            # Augmentation: with prob p, corrupt the frame and label OOD
            if augment_ood_prob > 0 and rng.random() < augment_ood_prob:
                # Replace the frame with Gaussian noise. We mutate in place
                # because the head collate has already stacked frames.
                # head_batch["frames"] is (B, H, W, 3) uint8.
                noise = rng.integers(0, 256, size=head_batch["frames"][i].shape,
                                     dtype=np.uint8)
                head_batch["frames"][i] = noise
                ood_labels[i] = 1.0
                # Re-set task_id to -100 (OOD) so CE skips it
                task_ids[i] = -100

    head_batch["task_id"] = torch.from_numpy(task_ids)
    head_batch["ood_label"] = torch.from_numpy(ood_labels)
    return head_batch


# ================================================================
# Eval helpers
# ================================================================

@torch.no_grad()
def evaluate(
    model: ALIGNModel,
    task_head: TaskHead,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, float]:
    """Evaluate task-head accuracy on the validation set.

    Returns:
        dict with accuracy, top-3 accuracy, mean_entropy, ood_recall
        (held-out real OOD), ood_false_positive_rate (in-distribution
        mis-flagged as OOD).
    """
    model.eval()
    task_head.eval()

    correct = 0
    correct_top3 = 0
    total_known = 0
    total_ood = 0
    ood_correct = 0   # OOD sample flagged OOD (p_ood > 0.5)
    id_correct = 0    # In-distribution sample NOT flagged OOD (p_ood <= 0.5)
    entropy_sum = 0.0
    n = 0

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16
        else torch.amp.autocast("cuda", enabled=False)
    )

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        traj = batch["trajectory"].to(device, non_blocking=True)
        task_ids = batch["task_id"].to(device)
        ood_labels = batch["ood_label"].to(device)

        with autocast_ctx:
            # Run encoders + mixer (text not used — task head only sees z_v, z_t)
            z_v, z_t = _encode_zv_zt(model, frames, traj)
            logits = task_head(z_v, z_t)         # (B, K+1)
        logits = logits.float()
        p = F.softmax(logits, dim=-1)

        # Known-class accuracy
        known_mask = task_ids != -100
        if known_mask.any():
            preds = logits[known_mask, :-1].argmax(dim=-1)
            correct += (preds == task_ids[known_mask]).sum().item()
            # Top-3
            top3 = logits[known_mask, :-1].topk(min(3, logits.shape[1] - 1), dim=-1).indices
            correct_top3 += (top3 == task_ids[known_mask].unsqueeze(-1)).any(dim=-1).sum().item()
            total_known += int(known_mask.sum().item())

        # OOD detection
        p_ood = p[:, -1]
        ood_pred = (p_ood > 0.5).float()
        ood_mask = ood_labels == 1.0
        id_mask = ood_labels == 0.0
        if ood_mask.any():
            ood_correct += (ood_pred[ood_mask] == 1.0).sum().item()
            total_ood += int(ood_mask.sum().item())
        if id_mask.any():
            id_correct += (ood_pred[id_mask] == 0.0).sum().item()

        # Entropy
        K = p.shape[-1]
        log_p = torch.log(p.clamp_min(1e-12))
        H = -(p * log_p).sum(dim=-1) / float(np.log(K))
        entropy_sum += H.sum().item()

        n += frames.shape[0]

    metrics = {
        "val_accuracy": correct / max(total_known, 1),
        "val_top3_accuracy": correct_top3 / max(total_known, 1),
        "val_mean_entropy": entropy_sum / max(n, 1),
    }
    if total_ood > 0:
        metrics["val_ood_recall"] = ood_correct / total_ood
    id_total = n - total_ood
    if id_total > 0:
        metrics["val_ood_false_positive_rate"] = 1.0 - (id_correct / id_total)

    return metrics


def _encode_zv_zt(
    model: ALIGNModel,
    frames,                 # (B, H, W, 3) uint8 — np.ndarray or torch.Tensor
    traj,                   # (B, K, 6) float32 — np.ndarray or torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch through vision, trajectory, and the cross-attention
    mixer, returning (z_v, z_t) — both (B, 256) post-mixer.

    The TaskHead doesn't consume text, so we skip the text path entirely:
      1. Encode raw vision and trajectory to (B, 256) and (B, K, 256)
      2. Run through the cross-attention mixer with a zero text vector
         (no information, but keeps the mixer's K/V contract valid)
      3. Mean-pool the trajectory tokens to (B, 256)

    At training time the mixer is frozen, so the fact that text is
    always zero means z_v and z_t are deterministic functions of
    (vision, trajectory) alone. At inference, the assistant head will
    use the *task head's* z_text in its forward pass, not this one.

    Accepts numpy arrays (the default from `head_collate`) and torch
    tensors. If numpy, converts to float/uint8 tensors and moves them
    to the same device as the model.
    """
    device = next(model.parameters()).device
    if not torch.is_tensor(frames):
        frames = torch.as_tensor(frames, dtype=torch.uint8, device=device)
    else:
        frames = frames.to(device)
    if not torch.is_tensor(traj):
        traj = torch.as_tensor(traj, dtype=torch.float32, device=device)
    else:
        traj = traj.to(device).float()

    # Raw encoders (no mixer yet)
    z_v_raw = model.encode_vision(frames)                              # (B, 256)
    z_t_tokens_raw = model.encode_trajectory_tokens(traj)              # (B, K, 256)
    # Cross-attention mixer with a zero text vector
    z_v_mixed, z_t_tokens_mixed, _ = model.cross_attention_mixer(
        z_v_raw, z_t_tokens_raw, torch.zeros_like(z_v_raw),
    )
    # Mean-pool trajectory tokens — this is what the assistant head
    # and TaskHead both consume
    z_t = z_t_tokens_mixed.mean(dim=1)                                # (B, 256)
    return z_v_mixed, z_t


# ================================================================
# Training
# ================================================================

def train_task_head(
    data_paths: List[str],
    pretrained_checkpoint: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    cameras: Optional[List[str]] = None,
    chunk_size: int = 5,
    traj_window: int = 20,
    device: Optional[str] = None,
    max_steps_per_epoch: int = 2000,
    num_ood_tasks: int = 0,
    ood_augment_prob: float = 0.2,
    augment_noise_std: float = 0.0,    # if > 0, mild Gaussian on top of pure-noise aug
    seed: int = 0,
    enable_wandb: bool = False,
    wandb_project: str = "align-task-head",
    wandb_run: Optional[str] = None,
    num_workers: int = 0,
    use_bf16: bool = True,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    dropout: float = 0.0,
) -> str:
    """Train the TaskHead on top of frozen encoders.

    Returns:
        Path to the best checkpoint.
    """
    # Detect the user-site cuDNN shadow early with a clear error message.
    # Without PYTHONNOUSERSITE=1, the system pip at ~/.local/ installs
    # cuDNN 9.2.0 which is incompatible with the conda torch 2.10.0+cu128
    # stack and causes CUDNN_STATUS_NOT_INITIALIZED on the first conv op.
    # The conda-installed cuDNN is 9.10.2, which works.
    _cudnn_ver = torch.backends.cudnn.version()
    if _cudnn_ver == 92000:
        raise RuntimeError(
            f"Detected user-site cuDNN shadow (version {_cudnn_ver}). "
            f"This causes CUDNN_STATUS_NOT_INITIALIZED on this torch stack. "
            f"Re-run with: PYTHONNOUSERSITE=1 {sys.argv[0]} ..."
        )

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- Build task vocabulary from the dataset ----
    # We use the FIRST data path's vocabulary (all paths in a single
    # training run are expected to share the same task set; MultiALIGNDataset
    # would need per-file vocab merging, out of scope here).
    all_tasks = build_task_vocabulary(data_paths[0])
    if not all_tasks:
        raise RuntimeError(
            f"No task descriptions found in {data_paths[0]}. "
            f"Check that the HDF5 has /ep_XXX/meta with 'task_description'."
        )
    known_tasks, ood_tasks = split_vocabulary(all_tasks, num_ood_tasks, seed=seed)
    K = len(known_tasks)
    task_to_id = {t: i for i, t in enumerate(known_tasks)}
    ood_set = set(ood_tasks)

    # ---- Run directory ----
    if len(data_paths) == 1:
        ds_name = Path(data_paths[0]).stem
    else:
        ds_name = "+".join(Path(p).stem for p in data_paths)
    base_dir = Path(output_dir) / f"{ds_name}_ood{num_ood_tasks}"
    existing = sorted(base_dir.glob("run_*")) if base_dir.exists() else []
    next_run = max([int(d.name.split("_")[-1]) for d in existing
                    if d.name.split("_")[-1].isdigit()] + [0]) + 1
    out_dir = base_dir / f"run_{next_run}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ALIGN Task Head Training ===")
    print(f"  Run:           {out_dir}")
    print(f"  Data:          {data_paths}")
    print(f"  Pretrained:    {pretrained_checkpoint}")
    print(f"  Device:        {device}")
    print(f"  Epochs:        {epochs}")
    print(f"  LR:            {lr}")
    print(f"  Vocab size:    K = {K} known + {len(ood_tasks)} held-out OOD")
    print(f"  Augment OOD p: {ood_augment_prob}")

    # ---- Save vocab and config ----
    config = {
        "model": "align-task-head",
        "data": [str(p) for p in data_paths],
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_split": val_split,
        "chunk_size": chunk_size,
        "traj_window": traj_window,
        "num_ood_tasks": num_ood_tasks,
        "ood_augment_prob": ood_augment_prob,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "use_bf16": use_bf16,
        "device": str(device),
        "K": K,
        "vocab_known": known_tasks,
        "vocab_ood": ood_tasks,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---- W&B ----
    wandb_trainer = init_wandb(
        project=wandb_project,
        name=wandb_run,
        config=config,
    ) if enable_wandb else init_wandb(project=wandb_project, name=wandb_run, config={})
    print(f"  W&B:           {'enabled' if wandb_trainer.enabled else 'disabled'}")

    # ---- Load pretrained ALIGNModel (encoders + mixer + existing heads) ----
    num_cameras = len(cameras) if cameras else 1
    model = ALIGNModel(
        embed_dim=embed_dim,
        chunk_size=chunk_size,
        use_text=True,
        device=str(device),
        num_cameras=num_cameras,
    ).to(device)
    ckpt = torch.load(pretrained_checkpoint, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    model.freeze_backbone()
    model.freeze_all_encoders()
    print(f"  Loaded pretrained backbone, all encoders + mixer frozen.")

    # ---- Build the TaskHead + CLIP basis (E_clip) ----
    task_head = create_task_head(
        num_known_classes=K,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    # Pre-compute E_clip using the frozen text encoder
    with torch.no_grad():
        E_clip_raw = model.encode_raw_text(known_tasks)        # (K, 256)
        if E_clip_raw is None:
            raise RuntimeError("Text encoder is not available; cannot build E_clip.")
        ood_anchor = E_clip_raw.mean(dim=0, keepdim=True)        # (1, 256)
        E_clip = torch.cat([E_clip_raw, ood_anchor], dim=0).float()  # (K+1, 256)

    bundle = TaskHeadBundle(task_head, E_clip).to(device)
    print(f"  E_clip shape:  {tuple(E_clip.shape)}")

    # ---- Optimizer & loss ----
    params = list(task_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # ---- Dataset ----
    if len(data_paths) == 1:
        full_ds = ALIGNDataset(
            data_paths[0], mode="head", traj_window=traj_window, cameras=cameras,
        )
    else:
        full_ds = MultiALIGNDataset(
            data_paths, mode="head", traj_window=traj_window, cameras=cameras,
        )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed),
    )
    print(f"  {n_train} train, {n_val} val samples")

    train_rng = np.random.default_rng(seed)

    def _train_collate(batch):
        return task_collate(
            batch, chunk_size=chunk_size, task_to_id=task_to_id,
            ood_set=ood_set, augment_ood_prob=ood_augment_prob, rng=train_rng,
        )

    def _val_collate(batch):
        # No augmentation at val time
        return task_collate(
            batch, chunk_size=chunk_size, task_to_id=task_to_id,
            ood_set=ood_set, augment_ood_prob=0.0,
            rng=np.random.default_rng(seed + 1),
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=_train_collate, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=_val_collate, num_workers=num_workers,
    )

    # ---- Training loop ----
    best_val_acc = 0.0
    best_ckpt = out_dir / "task_head_best.pt"
    last_ckpt = out_dir / "task_head_last.pt"
    log_path = out_dir / "task_head_log.jsonl"
    log_fp = open(log_path, "w")
    val_acc = 0.0
    val_metrics: Dict[str, float] = {"val_accuracy": 0.0}

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16
        else torch.amp.autocast("cuda", enabled=False)
    )

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        task_head.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_ood = 0.0
        n_batches = 0
        t_epoch = time.time()

        pbar = tqdm(train_loader, desc=f"ep {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break

            frames = batch["frames"]
            traj = batch["trajectory"]
            task_ids = batch["task_id"].to(device)
            ood_labels = batch["ood_label"].to(device)

            with autocast_ctx:
                z_v, z_t = _encode_zv_zt(model, frames, traj)
                logits = task_head(z_v, z_t)
                loss, comps = task_head_loss(logits, task_ids, ood_labels)
            loss = loss.float()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_ce += comps["task_ce"]
            epoch_ood += comps["ood_bce"]
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ce=f"{comps['task_ce']:.3f}",
                ood=f"{comps['ood_bce']:.3f}",
            )

        train_loss = epoch_loss / max(n_batches, 1)
        train_ce = epoch_ce / max(n_batches, 1)
        train_ood = epoch_ood / max(n_batches, 1)

        # Validate
        val_metrics = evaluate(model, task_head, val_loader, device, use_bf16=use_bf16)
        val_acc = val_metrics["val_accuracy"]

        # Logging
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_ood_bce": train_ood,
            **{k: float(v) for k, v in val_metrics.items()},
            "epoch_time_sec": time.time() - t_epoch,
        }
        log_fp.write(json.dumps(log_entry) + "\n")
        log_fp.flush()
        log_metrics(wandb_trainer, log_entry, step=epoch)

        print(
            f"  ep {epoch:3d}  loss={train_loss:.4f}  "
            f"val_acc={val_acc:.3f}  val_top3={val_metrics['val_top3_accuracy']:.3f}  "
            f"val_H={val_metrics['val_mean_entropy']:.3f}  "
            f"({time.time() - t_epoch:.1f}s)"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(
                path=best_ckpt,
                model=model, task_head=task_head, bundle=bundle,
                K=K, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout,
                known_tasks=known_tasks, ood_tasks=ood_tasks,
                val_acc=val_acc, epoch=epoch,
            )
            print(f"    ↳ new best (val_acc={val_acc:.3f}), saved to {best_ckpt}")

    # Save last
    final_val_acc = val_metrics["val_accuracy"]
    _save_checkpoint(
        path=last_ckpt,
        model=model, task_head=task_head, bundle=bundle,
        K=K, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout,
        known_tasks=known_tasks, ood_tasks=ood_tasks,
        val_acc=final_val_acc, epoch=epochs,
    )

    log_fp.close()
    print(f"\nDone. Best val_acc={best_val_acc:.3f}. Best checkpoint: {best_ckpt}")
    return str(best_ckpt)


def _save_checkpoint(
    path: Path,
    model: ALIGNModel,
    task_head: TaskHead,
    bundle: TaskHeadBundle,
    K: int,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
    known_tasks: List[str],
    ood_tasks: List[str],
    val_acc: float,
    epoch: int,
) -> None:
    """Save TaskHead + E_clip + config in a single .pt file."""
    state = {
        "task_head_state_dict": task_head.state_dict(),
        "E_clip": bundle.E_clip.detach().cpu(),
        "config": {
            "K": K,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "vocab_known": known_tasks,
            "vocab_ood": ood_tasks,
            "val_acc": val_acc,
            "epoch": epoch,
        },
    }
    torch.save(state, path)


# ================================================================
# CLI
# ================================================================

def main():
    p = argparse.ArgumentParser(
        description="ALIGN Task Identification Head — training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, nargs="+",
                   help="Path(s) to HDF5 file(s) for training")
    p.add_argument("--pretrained", required=True,
                   help="Pretrained ALIGNModel checkpoint (.pt) with trainable_state_dict")
    p.add_argument("--output-dir", default="./checkpoints/task_head",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--cameras", nargs="+", default=None,
                   help="Camera keys for multi-camera training (e.g. wrist agent)")
    p.add_argument("--chunk-size", type=int, default=5)
    p.add_argument("--traj-window", type=int, default=20)
    p.add_argument("--max-steps-per-epoch", type=int, default=2000)
    p.add_argument("--num-ood-tasks", type=int, default=0,
                   help="Number of held-out real OOD tasks (rest are known classes)")
    p.add_argument("--ood-augment-prob", type=float, default=0.2,
                   help="Probability of in-distribution sample being augmented → OOD")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--use-bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="use_bf16", action="store_false")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--enable-wandb", action="store_true")
    p.add_argument("--wandb-project", default="align-task-head")
    p.add_argument("--wandb-run", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    train_task_head(
        data_paths=args.data,
        pretrained_checkpoint=args.pretrained,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        cameras=args.cameras,
        chunk_size=args.chunk_size,
        traj_window=args.traj_window,
        max_steps_per_epoch=args.max_steps_per_epoch,
        num_ood_tasks=args.num_ood_tasks,
        ood_augment_prob=args.ood_augment_prob,
        seed=args.seed,
        num_workers=args.num_workers,
        use_bf16=args.use_bf16,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        device=args.device,
    )


if __name__ == "__main__":
    main()
