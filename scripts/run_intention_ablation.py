#!/usr/bin/env python3
"""Train four cached-feature intention/memory ablations sequentially on CUDA.

Uses the production model, sequential trainer, and validation functions.
Unlike the standard trainer, splits whole episodes and fixes validation crops.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.align_dataset import ALIGNDataset
from models.align_intention import ALIGNIntentionModel
from training.train_intention import train_v4_epoch, validate


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def episode_split(dataset, seed, fraction):
    tasks = defaultdict(list)
    for ep, key in enumerate(dataset._episode_keys):
        text = dataset._h5[f"{key}/texts"][()]
        if isinstance(text, bytes):
            text = text.decode()
        tasks[str(text)].append(ep)
    train, val = [], []
    rng = np.random.default_rng(seed)
    for task in sorted(tasks):
        eps = rng.permutation(tasks[task]).tolist()
        if len(eps) < 2:
            raise ValueError(f"Task has fewer than two episodes: {task}")
        count = min(len(eps) - 1, max(1, round(len(eps) * fraction)))
        val.extend(eps[:count])
        train.extend(eps[count:])
    assert not set(train) & set(val)
    return sorted(train), sorted(val)


class CachedEpisodes(Dataset):
    def __init__(self, dataset, episodes, length, seed, training, temporal_sampling="crop", supervision_points=16, chunk_size=8, observation_dropout_prob=0., drop_state_with_all_views=False):
        self.dataset, self.episodes = dataset, episodes
        self.length, self.seed, self.training = length, seed, training
        self.epoch = 0
        self.temporal_sampling, self.supervision_points, self.chunk_size = temporal_sampling, supervision_points, chunk_size
        self.observation_dropout_prob = observation_dropout_prob
        self.drop_state_with_all_views = drop_state_with_all_views

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        ep = self.episodes[index]
        n = self.dataset._get_episode_length(ep)
        length = min(n, self.length)
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, ep, self.epoch if self.training else 0]))
        start = int(rng.integers(0, n - length + 1))
        anchors = None
        if self.temporal_sampling == "episode":
            start,length = 0,n
            available = n-self.chunk_size+1
            if available<1:raise ValueError('Episode shorter than action horizon')
            anchors = np.zeros(n,dtype=bool)
            anchors[rng.choice(available,min(available,self.supervision_points),replace=False)] = True
        frames = self.dataset._read_frames_dinov2(ep, start, length)
        poses = self.dataset._read_poses(ep, start, length)
        gripper = self.dataset._read_poses_gripper(ep, start, length)
        states = np.concatenate([poses[:, :6], gripper[:, None]], axis=1)
        actions = self.dataset._read_actions(ep, start, length)
        result = dict(frames_segment=frames, states_segment=states.astype(np.float32),
                    actions_segment=actions.astype(np.float32), segment_len=length)
        result['text'] = self.dataset._read_text(ep) if hasattr(self.dataset, '_read_text') else ''
        if anchors is not None:
            result.update(loss_anchor_mask=anchors,observation_timesteps=np.arange(n,dtype=np.float32))
        if self.training and self.observation_dropout_prob > 0:
            cameras = len(self.dataset.cameras)
            visible = np.ones((length,cameras),dtype=bool)
            eligible = np.flatnonzero(anchors) if anchors is not None else np.arange(length-self.chunk_size+1)
            for t in eligible:
                if t == 0 or rng.random() >= self.observation_dropout_prob:continue
                if rng.random() < .5:visible[t] = False
                else:visible[t,int(rng.integers(cameras))] = False
            result['observation_camera_mask'] = visible
            if self.drop_state_with_all_views:result['observation_state_mask'] = visible.any(1)
        return result


def collate_segments(items):
    length = max(item["segment_len"] for item in items)
    result = {"segment_len": np.array([x["segment_len"] for x in items])}
    result["texts"] = [x.get("text", "") for x in items]
    for key in ("frames_segment", "states_segment", "actions_segment"):
        rows = []
        for item in items:
            arr = item[key]
            pad = [(0, length - len(arr))] + [(0, 0)] * (arr.ndim - 1)
            rows.append(np.pad(arr, pad))
        result[key] = torch.from_numpy(np.stack(rows))
    for key in ("loss_anchor_mask","observation_timesteps","observation_camera_mask","observation_state_mask"):
        if key in items[0]:
            result[key] = torch.from_numpy(np.stack([np.pad(x[key],[(0,length-len(x[key]))]+[(0,0)]*(x[key].ndim-1)) for x in items]))
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(out, records, manifest):
    atomic_json(out / "summary.json", records)
    rows = ["# Intention/memory ablation", "",
            f"Dataset: `{manifest['data']}`. Held-out prediction metrics, not simulator success rates.", "",
            f"Epoch budget: {manifest['epochs']}; optimizer steps/epoch: {manifest['max_steps'] or 'full loader'}.",
            f"Provisional checkpoint selection: {manifest.get('selection_metric','val/loss')}; closed-loop selection is reported separately.", "",
            "| Variant | Epoch | Val loss | Position MSE | Rotation MSE | Gripper MSE | Gripper accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, record in records.items():
        r = record["best"]
        rows.append(f"| {name} | {r['epoch']} | {r['val/loss']:.6f} | {r['pos_mse']:.6f} | {r['rot_mse']:.6f} | {r['grip_mse']:.6f} | {r['grip_acc']:.3f} |")
    rows += ["", "All variants use the same episode split, crop schedule, observation history,",
             "batch order, optimizer budget, and validation RNG seeds. Intent-disabled variants",
             "omit the intention encoder; memory-only retrieves perceptual and state streams.",
             "The cached per-camera features bypass raw-image/cross-camera vision computation.",
             "Results use one training seed; model dimensions and optimizer budget are recorded in manifest.json.",
             "Fixed validation seeds make comparisons repeatable but do not remove seed uncertainty.",
             "Comparative policy-quality conclusions require multiple seeds and simulator rollouts."]
    (out / "comparison.md").write_text("\n".join(rows) + "\n")


def evaluate_existing(out):
    """Reevaluate best checkpoints using their saved shared configuration."""
    manifest = json.loads((out / "manifest.json").read_text())
    records = json.loads((out / "summary.json").read_text())
    cameras = manifest.get("cameras", ["image", "wrist_image"])
    threshold = manifest.get("gripper_threshold", 0.5)
    dataset = ALIGNDataset(manifest["data"], mode="head", cameras=cameras,
        traj_window=manifest["segment_length"], dinov2_path=manifest["cache"])
    val_eps = [dataset._episode_keys.index(key) for key in manifest["val_episodes"]]
    data = CachedEpisodes(dataset, val_eps, manifest["segment_length"], manifest["seed"] + 1000, False, manifest.get("temporal_sampling","crop"),manifest.get("supervision_points",16),manifest["chunk_size"])
    loader = DataLoader(data, batch_size=manifest["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_segments)
    args = SimpleNamespace(history_size=manifest["history_size"], chunk_size=manifest["chunk_size"],
        action_dim=7, head_type=manifest["head_type"], skip_nan=False, gripper_threshold=threshold,
        gripper_loss_weight=manifest.get("gripper_loss_weight", 0.01))
    device = torch.device("cuda")
    torch.set_num_threads(manifest.get("cpu_threads", 2))
    torch.cuda.set_per_process_memory_fraction(manifest.get("gpu_memory_fraction", 0.40))
    for name, record in records.items():
        with torch.serialization.safe_globals([type(torch.__version__)]):
            checkpoint = torch.load(out / name / "intention_best.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        model = ALIGNIntentionModel(state_dim=config["state_dim"], mamba_output_dim=config["mamba_output_dim"],
            action_dim=7, chunk_size=config["chunk_size"], history_size=config["history_size"],
            num_cameras=len(cameras), compressed_dim=config["compressed_dim"], head_type=config["head_type"],
            head_d_model=config["head_d_model"], use_intent_tokens=config["use_intent_tokens"],
            use_text=config.get("use_text",False),text_dim=config.get("text_dim",256),
            text_encoder_type=config.get("text_encoder_type","clip"),text_vocab=config.get("text_vocab"),
            num_intent_tokens=config.get("num_intent_tokens", 1), intent_dim=config["intent_dim"],
            mamba_d_state=config.get("mamba_d_state", 16), mamba_d_conv=config.get("mamba_d_conv", 4),
            mamba_expand=config.get("mamba_expand", 2),
            use_memory_bank=config["use_memory_bank"], memory_bank_len=config["memory_bank_len"],
            memory_mode=config.get("memory_mode","legacy"),memory_detach_writes=config.get("memory_detach_writes",False),
            memory_perceptual_recency=config.get("memory_perceptual_recency",0.),memory_field_masks=config.get("memory_field_masks",False),memory_pre_state_visual=config.get("memory_pre_state_visual",False),memory_value_preserving=config.get("memory_value_preserving",False),memory_patch_temporal=config.get("memory_patch_temporal",False),memory_context_only=config.get("memory_context_only",False),memory_write_fused=config.get("memory_write_fused",True),memory_patch_retrieval=config.get("memory_patch_retrieval",False),
            diffusion_train_steps=config.get("diffusion_train_steps",10),diffusion_loss_repeats=config.get("diffusion_loss_repeats",1),
            visual_token_attention=config.get("visual_token_attention",False),diffusion_clip_sample=config.get("diffusion_clip_sample",False))
        model._build_head_and_bank(config.get("pool_out_dim", 256 * len(cameras) * config["compressed_dim"]))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        vision = model.vision_encoder
        model.vision_encoder = torch.nn.Identity()
        model.to(device)
        model.vision_encoder = vision
        seed_everything(manifest["seed"] + 10000)
        val_loss, _, metrics = validate(model, loader, device, args)
        record["best"].update(metrics)
        record["best"]["val/loss"] = val_loss
        record["gripper_threshold"] = threshold
        atomic_json(out / name / "best_evaluation.json", dict(val_loss=val_loss, **metrics,
            gripper_threshold=threshold, checkpoint_epoch=checkpoint["epoch"]))
        print(f"REEVALUATED {name}: val={val_loss:.6f} grip_acc={metrics['grip_acc']:.3f} threshold={threshold:g}", flush=True)
        del model, checkpoint, vision
        gc.collect()
        torch.cuda.empty_cache()
    summarize(out, records, manifest)
    with (out / "comparison.md").open("a") as f:
        f.write(f"\nBest checkpoints reevaluated with gripper prediction threshold {threshold:g}.\n")
        baseline_path = out / "trivial_baseline.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            f.write(f"\nReference: zero motion plus last observed gripper — position MSE "
                    f"{baseline['pos_mse']:.6f}, rotation MSE {baseline['rot_mse']:.6f}, "
                    f"gripper MSE {baseline['grip_mse']:.6f}, accuracy {baseline['grip_acc']:.3f}.\n")
        f.write("\nAction errors are in dataset action units, not Cartesian meters/degrees. "
                "GPU elapsed times include contention with another training job.\n")
    dataset.close()
    (out / "EVALUATION_COMPLETE").write_text(f"All best checkpoints reevaluated with threshold {threshold:g}.\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/libero_goal.h5"))
    parser.add_argument("--cache", default=str(ROOT / "data/libero_goal.dinov2"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from epoch checkpoints; preserve/restart attempts without optimizer state.")
    parser.add_argument("--temporal-sampling",choices=["crop","episode"],default="episode")
    parser.add_argument("--supervision-points",type=int,default=16)
    parser.add_argument("--observation-dropout-prob",type=float,default=0.,help="Training-only camera/all-view feature loss at supervised anchors; same masks across ablations")
    parser.add_argument("--drop-state-with-all-views",action=argparse.BooleanOptionalAction,default=False,help="Complete observation-packet outages when all cameras are dropped")
    parser.add_argument("--memory-mode",choices=["legacy","episodic"],default="episodic")
    parser.add_argument("--memory-detach-writes",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--memory-write-fused",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--memory-perceptual-recency",type=float,default=0.,help="Experimental visual attention logit penalty per frame of age; zero preserves checkpoint behavior")
    parser.add_argument("--memory-field-masks",action=argparse.BooleanOptionalAction,default=False,help="Exclude zero-marked missing fields from retrieval and preserve valid fields during consolidation")
    parser.add_argument("--memory-pre-state-visual",action=argparse.BooleanOptionalAction,default=False,help="Store state-independent compressed visual features, then apply current state after retrieval")
    parser.add_argument("--memory-value-preserving",action=argparse.BooleanOptionalAction,default=False,help="Experimental learned Q/K selection with raw historical feature values, bypassing value projections and FFN")
    parser.add_argument("--memory-patch-temporal",action=argparse.BooleanOptionalAction,default=False,help="Experimental per-camera/grid-slot temporal retrieval instead of global historical patch attention")
    parser.add_argument("--memory-context-only",action=argparse.BooleanOptionalAction,default=False,help="Experimental retrieved branch without a direct current-query residual; current features remain in the fusion gate")
    parser.add_argument("--memory-patch-retrieval",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--visual-token-attention",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--diffusion-clip-sample",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--diffusion-train-steps",type=int,default=100)
    parser.add_argument("--diffusion-loss-repeats",type=int,default=4)
    parser.add_argument("--warm-start",type=Path,help="Existing four-variant run; copy matching parameters, retain fresh diffusion schedule")
    parser.add_argument("--keep-epoch-checkpoints",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--selection-metric",choices=["val/loss","pos_mse"],default="pos_mse",
                        help="Prediction-based provisional selection; closed-loop selection is separate")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optimizer steps per epoch; 0 uses the full training loader.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--head-window-batch-size", type=int, default=2,
                        help="Group this many supervised anchors per diffusion/flow loss call after ordered memory retrieval; 1 restores one loss call per anchor.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cameras", nargs="+", default=["image", "wrist_image"])
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--head-type", choices=["diffusion", "flow_matching"], default="diffusion")
    parser.add_argument("--use-task-text", action="store_true", help="Condition the generative head on the recorded task instruction")
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--compressed-dim", type=int, default=4)
    parser.add_argument("--head-d-model", type=int, default=64)
    parser.add_argument("--mamba-output-dim", type=int, default=128)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--num-intent-tokens", type=int, default=1)
    parser.add_argument("--intent-dim", type=int, default=128)
    parser.add_argument("--memory-bank-len", type=int, default=16)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--segment-length", type=int, default=20)
    parser.add_argument("--variants", nargs="+", default=["no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory"])
    parser.add_argument("--worker-variant", choices=["no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory"],
                        help="Train only this variant, retaining the ordered common initialization.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.40)
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "history_size", "chunk_size", "segment_length",
                 "state_dim", "compressed_dim", "head_d_model", "mamba_output_dim",
                 "mamba_d_state", "mamba_d_conv", "mamba_expand", "num_intent_tokens",
                 "intent_dim", "memory_bank_len", "cpu_threads", "head_window_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.segment_length < args.history_size + args.chunk_size:
        parser.error("Segment must contain both history and action chunk")
    if args.chunk_size < 4:
        parser.error("--chunk-size must be at least 4 for the generative U-Net")
    if args.head_d_model % 8:
        parser.error("--head-d-model must be divisible by 8 for GroupNorm")
    if args.compressed_dim % 4 or args.state_dim % 4 or (args.intent_dim * args.num_intent_tokens) % 4:
        parser.error("Compressed/state/total intent dimensions must be divisible by 4 for attention")
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("Camera names must be unique")
    if not 0 < args.validation_fraction < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Validation fraction must be in (0, 1); GPU memory fraction in (0, 1]")
    if args.lr <= 0 or min(args.weight_decay, args.grad_clip, args.gripper_loss_weight, args.max_steps) < 0:
        parser.error("Learning rate must be positive; decay, clipping, loss weight and max steps must be nonnegative")
    if min(args.supervision_points,args.diffusion_train_steps,args.diffusion_loss_repeats)<1 or args.diffusion_train_steps<10:
        parser.error('Supervision points/repeats must be positive; diffusion schedule needs at least 10 steps')
    if not 0 <= args.observation_dropout_prob <= 1:
        parser.error("Observation dropout probability must be in [0,1]")
    if args.temporal_sampling=="episode" and args.history_size!=1:
        parser.error('Episode supervision currently requires history size 1')
    if args.memory_perceptual_recency<0 or (args.memory_perceptual_recency>0 and (args.memory_mode!="episodic" or not args.memory_value_preserving)):
        parser.error("Nonnegative perceptual recency requires episodic raw-value retrieval")
    if args.memory_field_masks and (args.memory_mode != "episodic" or args.memory_write_fused):
        parser.error("Field validity requires episodic raw writes")
    if args.memory_pre_state_visual and (args.memory_mode != "episodic" or args.memory_write_fused):
        parser.error("Pre-state visual memory requires episodic raw writes")
    if args.memory_value_preserving and (args.memory_context_only or args.memory_mode != "episodic" or (args.memory_patch_retrieval and not args.memory_patch_temporal)):
        parser.error("Value-preserving retrieval requires episodic mode, no context FFN, and temporal alignment for patch banks")
    if args.memory_patch_temporal and (not args.memory_patch_retrieval or args.memory_mode != "episodic"):
        parser.error("Temporal patch retrieval requires episodic patch memory")
    if args.memory_context_only and args.memory_mode != "episodic":
        parser.error("Context-only retrieval requires episodic memory")
    if args.memory_patch_retrieval and (args.compressed_dim%2 or args.memory_mode!="episodic"):
        parser.error('Patch retrieval needs even compressed width and episodic memory')
    if args.use_task_text and args.text_dim < 1:
        parser.error('--text-dim must be positive')
    if len(set(args.variants)) != len(args.variants):
        parser.error("Each variant may appear only once")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("These ablations require a CUDA GPU")
    if args.evaluate_existing:
        evaluate_existing(Path(args.output).resolve())
        return
    if args.segment_length < args.history_size + args.chunk_size:
        raise ValueError("Segment must contain both history and action chunk")
    args.warm_start = str(args.warm_start.resolve()) if args.warm_start else None
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError("Use a new output directory; existing runs are preserved")
    os.environ["WANDB_MODE"] = "disabled"
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda")
    # Bound this process's allocator while another training job shares the GPU.
    if not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0, 1]")
    if args.worker_variant and args.worker_variant not in args.variants:
        raise ValueError("Worker variant must be in the requested variants")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    dataset = ALIGNDataset(args.data, mode="head", cameras=args.cameras,
                           traj_window=args.segment_length, dinov2_path=args.cache)
    sample = dataset._read_frames_dinov2(0, 0, 1)
    # The production cached trainer expects 256 patches + CLS per camera.
    if sample.shape[-2:] != (257 * len(args.cameras), 768):
        raise ValueError("Expected 257 DINOv2 tokens of width 768 per selected camera")
    pool_out_dim = (sample.shape[-2] - len(args.cameras)) * args.compressed_dim
    train_eps, val_eps = episode_split(dataset, args.seed, args.validation_fraction)
    text_vocab = (sorted({word for ep in train_eps for word in
                          re.findall(r"[a-z0-9]+", dataset._read_text(ep).lower())})
                  if args.use_task_text else [])
    if len(train_eps) < args.batch_size:
        raise ValueError("Batch size exceeds the training episode count")
    if any(dataset._get_episode_length(ep) < args.history_size + args.chunk_size for ep in train_eps + val_eps):
        raise ValueError("An episode is too short for this history/chunk configuration")
    manifest = dict(vars(args), commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), split="task-stratified whole episodes",
        train_episodes=[dataset._episode_keys[ep] for ep in train_eps],
        val_episodes=[dataset._episode_keys[ep] for ep in val_eps],
        num_cameras=len(args.cameras), pool_out_dim=pool_out_dim, frozen_cached_vision=True,
        use_text=args.use_task_text, text_encoder_type="bag" if args.use_task_text else "clip",
        text_vocab=text_vocab,
        intention_recurrence="causal observation state across each segment")
    if args.resume and (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        legacy_defaults = dict(cameras=["image", "wrist_image"], validation_fraction=0.1,
                               weight_decay=1e-4, grad_clip=1.0, mamba_output_dim=128,
                               mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, temporal_sampling="crop",supervision_points=16,
                               memory_perceptual_recency=0.,memory_field_masks=False,memory_pre_state_visual=False,memory_value_preserving=False,memory_patch_temporal=False,memory_context_only=False,memory_mode="legacy",memory_detach_writes=False,memory_write_fused=True,memory_patch_retrieval=False,
                               diffusion_train_steps=10,diffusion_loss_repeats=1,warm_start=None,selection_metric="val/loss",visual_token_attention=False,diffusion_clip_sample=False,observation_dropout_prob=0.,drop_state_with_all_views=False,
                               use_task_text=False,text_dim=128,text_vocab=[],head_window_batch_size=2)
        for key in ("epochs", "max_steps", "history_size", "chunk_size", "segment_length", "batch_size",
                    "seed", "variants", "data", "cache", "cameras", "validation_fraction", "lr",
                    "weight_decay", "grad_clip", "head_type", "state_dim", "compressed_dim",
                    "head_d_model", "mamba_output_dim", "mamba_d_state", "mamba_d_conv", "mamba_expand",
                    "use_task_text", "text_dim", "text_vocab",
                    "intent_dim", "num_intent_tokens", "memory_bank_len", "gripper_loss_weight", "gripper_threshold",
                    "temporal_sampling","supervision_points","memory_mode","memory_detach_writes","memory_write_fused",
                    "memory_perceptual_recency","memory_field_masks","memory_pre_state_visual","memory_value_preserving","memory_patch_temporal","memory_context_only","memory_patch_retrieval","diffusion_train_steps","diffusion_loss_repeats","warm_start","selection_metric","visual_token_attention","diffusion_clip_sample","observation_dropout_prob","drop_state_with_all_views","head_window_batch_size"):
            if previous.get(key, legacy_defaults.get(key)) != manifest[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        manifest = dict(manifest, **previous)
    else:
        atomic_json(out / "manifest.json", manifest)
    print(f"Split: {len(train_eps)} train episodes / {len(val_eps)} held-out episodes", flush=True)
    train_data = CachedEpisodes(dataset, train_eps, args.segment_length, args.seed, True, args.temporal_sampling,args.supervision_points,args.chunk_size,args.observation_dropout_prob,args.drop_state_with_all_views)
    val_data = CachedEpisodes(dataset, val_eps, args.segment_length, args.seed + 1000, False, args.temporal_sampling,args.supervision_points,args.chunk_size)
    records = (json.loads((out / "summary.json").read_text())
               if args.resume and (out / "summary.json").exists() else {})
    report_out = out
    if args.worker_variant:
        report_out = out / args.worker_variant
        if (report_out / "summary.json").exists():
            records = json.loads((report_out / "summary.json").read_text())
        records = {name: record for name, record in records.items() if name == args.worker_variant}
    shared_initialization = {}
    for name in args.variants:
        if name not in {"no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory"}:
            raise ValueError(name)
        intent = name.startswith("intent_")
        memory = name.endswith("_memory") and not name.endswith("no_memory")
        seed_everything(args.seed)
        model = ALIGNIntentionModel(state_dim=args.state_dim, mamba_output_dim=args.mamba_output_dim,
            action_dim=7, chunk_size=args.chunk_size, history_size=args.history_size,
            num_cameras=len(args.cameras), compressed_dim=args.compressed_dim,
            head_type=args.head_type, head_d_model=args.head_d_model,
            mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv, mamba_expand=args.mamba_expand,
            use_intent_tokens=intent, num_intent_tokens=args.num_intent_tokens, intent_dim=args.intent_dim,
            use_memory_bank=memory, memory_bank_len=args.memory_bank_len,
            memory_mode=args.memory_mode,memory_detach_writes=args.memory_detach_writes,
            memory_perceptual_recency=args.memory_perceptual_recency,memory_field_masks=args.memory_field_masks,memory_pre_state_visual=args.memory_pre_state_visual,memory_value_preserving=args.memory_value_preserving,memory_patch_temporal=args.memory_patch_temporal,memory_context_only=args.memory_context_only,memory_write_fused=args.memory_write_fused,memory_patch_retrieval=args.memory_patch_retrieval,
            diffusion_train_steps=args.diffusion_train_steps,diffusion_loss_repeats=args.diffusion_loss_repeats,
            visual_token_attention=args.visual_token_attention,diffusion_clip_sample=args.diffusion_clip_sample,
            use_text=args.use_task_text,text_dim=args.text_dim,
            text_encoder_type="bag" if args.use_task_text else "clip",text_vocab=text_vocab)
        # Cached training never uses the raw-image modules. Keep them off GPU.
        model.vision_encoder.requires_grad_(False)
        vision = model.vision_encoder
        model.vision_encoder = torch.nn.Identity()
        model.to(device)
        model.vision_encoder = vision
        seed_everything(args.seed + 1)
        model._build_head_and_bank(pool_out_dim)
        model.intention_head.to(device)
        if model.text_condition is not None:
            model.text_condition.to(device)
        if model.memory_module is not None:
            model.memory_module.to(device)
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key.startswith("vision_encoder."):
                    continue
                if key in shared_initialization and shared_initialization[key].shape == value.shape:
                    value.copy_(shared_initialization[key].to(value.device))
                else:
                    shared_initialization.setdefault(key, value.detach().cpu().clone())
        # The encoder's legacy patch encoder and hidden projection have no consumers.
        if model.intention_encoder is not None:
            model.intention_encoder.vision_patch_encoder.requires_grad_(False)
            model.intention_encoder.mamba_to_hidden.requires_grad_(False)
        if args.worker_variant and name != args.worker_variant:
            del model, vision
            torch.cuda.empty_cache()
            continue
        if args.warm_start:
            with torch.serialization.safe_globals([type(torch.__version__)]):
                saved = torch.load(Path(args.warm_start)/name/"intention_best.pt",map_location="cpu",weights_only=True)
            source = saved["model_state_dict"]
            own = model.state_dict()
            matched = {k:v for k,v in source.items() if k in own and own[k].shape==v.shape
                       and not k.startswith("vision_encoder.") and not k.endswith(("alpha_bar","sigma"))}
            model.load_state_dict(matched,strict=False)
            print(f"WARM START {name}: {len(matched)} parameters/buffers, fresh noise schedule",flush=True)
            del saved,source,own,matched
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
        loop_args = SimpleNamespace(history_size=args.history_size, chunk_size=args.chunk_size,
            action_dim=7, head_type=args.head_type, skip_nan=False, grad_clip=args.grad_clip,
            no_sample_during_train=True, debug=False, gripper_threshold=args.gripper_threshold,
            gripper_loss_weight=args.gripper_loss_weight,
            head_window_batch_size=args.head_window_batch_size)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=collate_segments)
        run = out / name
        start_epoch = 1
        best = float("inf")
        if args.resume and run.exists():
            if records.get(name, {}).get("completed_epochs", 0) >= args.epochs:
                print(f"SKIP completed {name}", flush=True)
                del model, optimizer, params, vision, val_loader
                torch.cuda.empty_cache()
                continue
            last_checkpoint = run / "training_last.pt"
            if last_checkpoint.exists():
                saved = torch.load(last_checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(saved["model_state_dict"], strict=False)
                optimizer.load_state_dict(saved["optimizer_state_dict"])
                start_epoch = saved["epoch"] + 1
                best = saved["best_loss"]
                records[name] = saved["record"]
                # A metric written before an interrupted checkpoint save is replayed.
                metric_file = run / "metrics.jsonl"
                if metric_file.exists():
                    completed = [line for line in metric_file.read_text().splitlines()
                                 if json.loads(line)["epoch"] < start_epoch]
                    metric_file.write_text("\n".join(completed) + ("\n" if completed else ""))
                del saved
                print(f"RESUME {name} epoch={start_epoch}", flush=True)
            else:
                archive = out / f"{name}_interrupted_{int(time.time())}"
                run.rename(archive)
                records.pop(name, None)
                print(f"RESTART {name}: no optimizer checkpoint; preserved {archive.name}", flush=True)
        run.mkdir(exist_ok=True)
        config = dict(manifest, use_intent_tokens=intent, use_memory_bank=memory,
                      use_history=True, mamba_output_dim=args.mamba_output_dim, action_dim=7,
                      num_cameras=len(args.cameras), model_class="ALIGNIntentionModel",
                      trainable_parameters=sum(p.numel() for p in params))
        atomic_json(run / "config.json", config)
        torch.cuda.reset_peak_memory_stats()
        print(f"START {name}: {config['trainable_parameters']:,} trainable parameters", flush=True)
        with (run / "metrics.jsonl").open("a" if start_epoch > 1 else "w", buffering=1) as log:
            for epoch in range(start_epoch, args.epochs + 1):
                train_data.epoch = epoch
                generator = torch.Generator().manual_seed(args.seed + epoch)
                train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                    generator=generator, num_workers=0, pin_memory=True,
                    collate_fn=collate_segments, drop_last=True)
                seed_everything(args.seed + epoch)
                start = time.monotonic()
                train_loss, _ = train_v4_epoch(model, train_loader, optimizer, device,
                    loop_args, max_steps=args.max_steps)
                seed_everything(args.seed + 10000)
                val_loss, _, metrics = validate(model, val_loader, device, loop_args)
                row = dict(epoch=epoch, **{"train/loss": train_loss, "val/loss": val_loss},
                    **metrics, elapsed_s=time.monotonic() - start,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
                if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                    raise FloatingPointError(f"Nonfinite loss in {name}: {row}")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                selection_score = row[args.selection_metric]
                if selection_score < best:
                    best = selection_score
                    records[name] = dict(best=row, completed_epochs=epoch)
                    torch.save(dict(model_state_dict=model.state_dict(), config=config,
                        epoch=epoch, val_loss=val_loss), run / "intention_best.pt")
                if args.keep_epoch_checkpoints:
                    torch.save(dict(model_state_dict={k:v for k,v in model.state_dict().items() if not k.startswith("vision_encoder.")},
                                    config=dict(config,frozen_vision_omitted=True),epoch=epoch,val_loss=val_loss),run/f"epoch_{epoch:03d}.pt")
                records[name]["completed_epochs"] = epoch
                # Frozen raw vision is reconstructible and excluded to keep
                # epoch recovery saves small. Seed/crop/order reset every epoch.
                latest = dict(model_state_dict={key: value for key, value in model.state_dict().items()
                                                if not key.startswith("vision_encoder.")},
                              optimizer_state_dict=optimizer.state_dict(), epoch=epoch,
                              best_loss=best, record=records[name])
                temporary = run / "training_last.pt.tmp"
                torch.save(latest, temporary)
                temporary.replace(run / "training_last.pt")
                del latest
                summarize(report_out, records, manifest)
                print(f"RESULT {name} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f} "
                      f"pos={metrics['pos_mse']:.6f} rot={metrics['rot_mse']:.6f} "
                      f"grip_acc={metrics['grip_acc']:.3f} seconds={row['elapsed_s']:.1f}", flush=True)
        del model, optimizer, params, vision, val_loader
        torch.cuda.empty_cache()
    dataset.close()
    summarize(report_out, records, manifest)
    (report_out / "COMPLETE").write_text("Requested worker variants completed.\n")
    print(f"COMPLETE: {report_out / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
