#!/usr/bin/env python3
"""Train four cached-feature intention/memory ablations sequentially on CUDA.

Uses the production model, sequential trainer, and validation functions.
Unlike the standard trainer, splits whole episodes and fixes validation crops.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
from itertools import islice
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.align_dataset import ALIGNDataset
from models.flow_policy_config import (add_flow_policy_arguments, flow_policy_options,
    validate_flow_policy_arguments, LEGACY_FLOW_DEFAULTS)
from models.align_intention import ALIGNIntentionModel
from models.intention_future import add_future_arguments, validate_future_arguments, freeze_intent_backbone
from training.checkpoint_io import CheckpointWriter, cpu_snapshot
from training.memory_resume import memory_patch_transition, migrate_memory_weights, archive_memory_transition
from training.train_intention import train_v4_epoch, validate
from training.cached_loading import collate_cached_segments
from training.plain_history_training import (SparseHistoryEpisodes,
    collate_sparse_histories, plain_history_epoch)
from data.task_context_policy_dataset import (policy_episode_split, add_context_policy_arguments,
    context_policy_options, policy_parameter_groups, warm_start_policy)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LimitedBatches:
    """Bound smoke-test validation without fetching an extra batch."""
    def __init__(self, loader, limit=0):
        self.loader, self.limit = loader, limit

    def __len__(self):
        return min(len(self.loader), self.limit) if self.limit else len(self.loader)

    def __iter__(self):
        return islice(self.loader, len(self))


def source_provenance(root=ROOT):
    """Support a documented source snapshot deployed without a Git checkout."""
    snapshot = root / 'source_snapshot.json'
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    import hashlib
    names = ['models/context_aci_head.py','models/policy_temporal_encoder.py','models/flow_policy_config.py','models/task_text_encoder.py','models/vision_config.py','models/align_model.py','models/align_intention.py','models/intention_head.py','models/memory_bank.py',
        'data/training_split.py','configs/splits/libero_goal_piper_384_44.json',
        'models/task_context_policy.py','models/task_context_observer.py','models/task_context_encoder.py',
        'data/task_context_policy_dataset.py','training/train_intention.py','training/memory_resume.py','scripts/run_intention_ablation.py']
    return {'commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in names if (root/name).is_file()}}


def episode_split(dataset, seed, fraction):
    from data.training_split import piper_episode_indices
    fixed = piper_episode_indices(dataset)
    if fixed is not None:
        return fixed
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
    def __init__(self, dataset, episodes, length, seed, training, temporal_sampling="crop", supervision_points=16, chunk_size=8, observation_dropout_prob=0., drop_state_with_all_views=False, profile=False, gripper_transition_fraction=0.0, gripper_state_source="causal"):
        self.dataset, self.episodes = dataset, episodes
        self.length, self.seed, self.training = length, seed, training
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.temporal_sampling, self.supervision_points, self.chunk_size = temporal_sampling, supervision_points, chunk_size
        self.observation_dropout_prob = observation_dropout_prob
        self.drop_state_with_all_views = drop_state_with_all_views
        self.profile = profile
        self.gripper_transition_fraction = gripper_transition_fraction
        self.gripper_state_source = gripper_state_source

    @property
    def epoch(self):
        return int(self._epoch.item())

    @epoch.setter
    def epoch(self, value):
        self._epoch.fill_(int(value))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        before = time.perf_counter() if self.profile else 0.
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
        read_features = getattr(self.dataset, "_read_frames_dinov2_view",
                                self.dataset._read_frames_dinov2)
        frames = read_features(ep, start, length)
        poses = self.dataset._read_poses(ep, start, length)
        gripper = self.dataset._read_poses_gripper(ep, start, length)
        if self.gripper_state_source == "legacy_zero":
            gripper = np.zeros_like(gripper)
        states = np.concatenate([poses[:, :6], gripper[:, None]], axis=1)
        actions = self.dataset._read_actions(ep, start, length)
        if anchors is not None and self.training and self.gripper_transition_fraction > 0:
            # Target changes identify supervision locations only; never enter inputs.
            switches = np.flatnonzero(np.diff(actions[:, 6] > .5)) + 1
            switches = switches[switches < available]
            count = min(len(switches), int(round(anchors.sum() * self.gripper_transition_fraction)))
            if count:
                selected = rng.choice(switches, count, replace=False)
                remaining = np.setdiff1d(np.arange(available), selected)
                uniform = rng.choice(remaining, int(anchors.sum()) - count, replace=False)
                anchors[:] = False
                anchors[np.concatenate([selected, uniform])] = True
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
        handles = getattr(self.dataset, '_dinov2_memmaps', None)
        if handles is not None:
            handles.pop(ep, None)
        if self.profile:
            result['data_stage_cpu_s'] = dict(episode_metadata_and_mmap_wall_s=time.perf_counter()-before)
        return result


def collate_segments(items):
    length = max(item["segment_len"] for item in items)
    result = {"segment_len": np.array([x["segment_len"] for x in items])}
    result["texts"] = [x.get("text", "") for x in items]
    for key in ("frames_segment", "states_segment", "actions_segment"):
        first = items[0][key]
        batch = np.zeros((len(items), length, *first.shape[1:]), dtype=first.dtype)
        for index, item in enumerate(items):
            arr = item[key]
            batch[index, :len(arr)] = arr
        result[key] = torch.from_numpy(batch)
    for key in ("loss_anchor_mask","observation_timesteps","observation_camera_mask","observation_state_mask"):
        if key in items[0]:
            first = items[0][key]
            batch = np.zeros((len(items), length, *first.shape[1:]), dtype=first.dtype)
            for index, item in enumerate(items):
                arr = item[key]
                batch[index, :len(arr)] = arr
            result[key] = torch.from_numpy(batch)
    return result


class LengthBucketBatchSampler(Sampler):
    """Randomize inside length-sorted pools to limit padding between batches."""
    def __init__(self, lengths, batch_size, seed, bucket_multiplier=20, drop_last=True,
                 micro_batch_size=None):
        self.lengths = list(map(int, lengths))
        self.batch_size = int(batch_size)
        self.micro_batch_size = int(micro_batch_size or batch_size)
        if self.batch_size % self.micro_batch_size:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_multiplier))
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        randomized = torch.randperm(len(self.lengths), generator=generator).tolist()
        random_rank = {index: rank for rank, index in enumerate(randomized)}
        if self.drop_last:
            keep = len(randomized) - len(randomized) % self.batch_size
            randomized = randomized[:keep]
        order = sorted(randomized, key=lambda index: (self.lengths[index], random_rank[index]))
        batches = []
        for start in range(0, len(order), self.bucket_size):
            pool = order[start:start + self.bucket_size]
            pool_order = torch.randperm(len(pool), generator=generator).tolist()
            shuffled_pool = [pool[index] for index in pool_order]
            batches.extend(shuffled_pool[i:i + self.batch_size]
                           for i in range(0, len(shuffled_pool), self.batch_size)
                           if not self.drop_last or len(pool[i:i + self.batch_size]) == self.batch_size)
        if batches:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            for index in permutation:
                group = batches[index]
                for start in range(0, len(group), self.micro_batch_size):
                    yield group[start:start + self.micro_batch_size]

    def __len__(self):
        if self.drop_last:
            return (len(self.lengths) // self.batch_size) * (self.batch_size // self.micro_batch_size)
        return (len(self.lengths) + self.micro_batch_size - 1) // self.micro_batch_size


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def legacy_resume_defaults(previous):
    """Historical defaults shared by config restoration and compatibility checks."""
    legacy_defaults = dict(max_val_batches=0, gripper_state_source="legacy_zero", gripper_transition_fraction=0.0, cameras=["image", "wrist_image"], validation_fraction=0.1,
                           plain_history_size=4,
                           weight_decay=1e-4, grad_clip=1.0, mamba_output_dim=128,
                           mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, temporal_sampling="crop",supervision_points=16,
                           memory_perceptual_recency=0.,memory_field_masks=False,memory_pre_state_visual=False,memory_value_preserving=False,memory_patch_temporal=False,memory_context_only=False,memory_mode="legacy",memory_detach_writes=False,memory_write_fused=True,memory_patch_retrieval=False,
                           visual_context_mode="none",memory_retrieval_depth=1,memory_intent_query=False,
                           diffusion_train_steps=10,diffusion_loss_repeats=1,warm_start=None,selection_metric="val/loss",visual_token_attention=False,diffusion_clip_sample=False,observation_dropout_prob=0.,drop_state_with_all_views=False,
                           use_task_text=False,text_dim=128,text_vocab=[],head_window_batch_size=2,
                           length_bucket_multiplier=20)
    legacy_defaults.update(LEGACY_FLOW_DEFAULTS)
    legacy_defaults.update(text_encoder_type="bag" if previous.get("use_task_text",False) else "clip",text_model_name="distilbert/distilbert-base-uncased",text_model_revision="main",text_max_length=128,vision_backbone="dinov2_vitb14",raw_dim=768,intent_future_weight=0.,intent_future_horizons=[16,64],intent_future_hidden_dim=256,freeze_intent_backbone=False,
                           task_context_fusion='additive',task_context_fusion_dim=128)
    return legacy_defaults


def extend_resume_manifest(previous, current):
    """Preserve model/split provenance while updating allowed runtime settings."""
    if current["epochs"] < previous["epochs"]:
        raise ValueError("Resume epoch target cannot decrease; --epochs is the total target")
    merged = dict(current, **previous)
    for key in ("epochs", "num_workers", "prefetch_factor", "cpu_threads",
                "gpu_memory_fraction", "micro_batch_size", "pin_memory",
                "plain_history_loader", "plain_head_batch_size", "profile_data_stages",
                "plain_history_copy_mode", "checkpoint_staging_dir", "memory_patch_retrieval"):
        if key in current:
            merged[key] = current[key]
    return merged


def summarize(out, records, manifest):
    atomic_json(out / "summary.json", records)
    rows = ["# Intention/memory ablation", "",
            f"Dataset: `{manifest['data']}`. Held-out prediction metrics, not simulator success rates.", "",
            f"Epoch budget: {manifest['epochs']}; optimizer steps/epoch: {manifest['max_steps'] or 'full loader'}.",
            f"Validation batches: {manifest.get('max_val_batches', 0) or 'full loader'}; limited runs are smoke tests only.",
            f"Provisional checkpoint selection: {manifest.get('selection_metric','val/loss')}; closed-loop selection is reported separately.", "",
            "| Variant | Epoch | Val loss | Position MSE | Rotation MSE | Gripper MSE | Gripper accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name, record in records.items():
        r = record["best"]
        rows.append(f"| {name} | {r['epoch']} | {r['val/loss']:.6f} | {r['pos_mse']:.6f} | {r['rot_mse']:.6f} | {r['grip_mse']:.6f} | {r['grip_acc']:.3f} |")
    rows += ["", "All variants use the same episode split, crop schedule,",
             "batch order, optimizer budget, and validation RNG seeds. Intent-disabled variants",
             "omit the intention encoder; memory-only retrieves perceptual and state streams.",
            "The plain_history and context_history variants use their configured ordered feature/state window.",
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
    data.gripper_state_source = manifest.get("gripper_state_source", "legacy_zero")
    loader = DataLoader(data, batch_size=manifest.get('micro_batch_size') or manifest["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_cached_segments)
    loader = LimitedBatches(loader, manifest.get('max_val_batches', 0))
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
        model = ALIGNIntentionModel(vision_backbone=config.get("vision_backbone","dinov2_vitb14"), raw_dim=config.get("raw_dim",768), state_dim=config["state_dim"], mamba_output_dim=config["mamba_output_dim"],
            action_dim=7, chunk_size=config["chunk_size"], history_size=config["history_size"],
            num_cameras=len(cameras), compressed_dim=config["compressed_dim"], head_type=config["head_type"],
            head_history_mode=config.get("head_history_mode", "mean"),
            task_context_config=config.get('task_context_config'),
            task_context_tokens=config.get('task_context_tokens',config.get('num_task_tokens')),
            task_context_trainable=config.get('task_context_trainable',False),
            task_context_initialization=config.get('task_context_initialization','pretrained'),
            task_context_fusion=config.get('task_context_fusion','additive'),
            task_context_fusion_dim=config.get('task_context_fusion_dim',128),
            policy_temporal_mamba=config.get('policy_temporal_mamba',False),
            policy_temporal_width=config.get('policy_temporal_width',128),
            policy_temporal_layers=config.get('policy_temporal_layers',1),
            head_d_model=config["head_d_model"], use_intent_tokens=config["use_intent_tokens"],
            use_text=config.get("use_text",False),text_dim=config.get("text_dim",256),
            text_encoder_type=config.get("text_encoder_type","clip"),text_vocab=config.get("text_vocab"),
            text_model_name=config.get("text_model_name","distilbert/distilbert-base-uncased"),text_model_revision=config.get("text_model_revision","main"),text_max_length=config.get("text_max_length",128),
            intent_future_weight=config.get('intent_future_weight',0.),
            intent_future_horizons=config.get('intent_future_horizons',[16,64]),
            intent_future_hidden_dim=config.get('intent_future_hidden_dim',256),
            num_intent_tokens=config.get("num_intent_tokens", 1), intent_dim=config["intent_dim"],
            mamba_d_state=config.get("mamba_d_state", 16), mamba_d_conv=config.get("mamba_d_conv", 4),
            mamba_expand=config.get("mamba_expand", 2),
            use_memory_bank=config["use_memory_bank"], memory_bank_len=config["memory_bank_len"],
            memory_mode=config.get("memory_mode","legacy"),memory_detach_writes=config.get("memory_detach_writes",False),
            memory_perceptual_recency=config.get("memory_perceptual_recency",0.),memory_field_masks=config.get("memory_field_masks",False),memory_pre_state_visual=config.get("memory_pre_state_visual",False),memory_value_preserving=config.get("memory_value_preserving",False),memory_patch_temporal=config.get("memory_patch_temporal",False),memory_context_only=config.get("memory_context_only",False),memory_write_fused=config.get("memory_write_fused",True),memory_patch_retrieval=config.get("memory_patch_retrieval",False),
            **flow_policy_options(config),
            diffusion_train_steps=config.get("diffusion_train_steps",10),diffusion_loss_repeats=config.get("diffusion_loss_repeats",1),
            visual_context_mode=config.get("visual_context_mode","none"),
            memory_retrieval_depth=config.get("memory_retrieval_depth",1),memory_intent_query=config.get("memory_intent_query",False),
            visual_token_attention=config.get("visual_token_attention",False),diffusion_clip_sample=config.get("diffusion_clip_sample",False))
        model._build_head_and_bank(config.get("pool_out_dim", 256 * len(cameras) * config["compressed_dim"]))
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if incompatible.unexpected_keys or any(not key.startswith("vision_encoder.") for key in incompatible.missing_keys):
            raise ValueError("Checkpoint is missing trained parameters; evaluation aborted")
        vision = model.vision_encoder
        model.vision_encoder = torch.nn.Identity()
        model.to(device)
        model.vision_encoder = vision
        seed_everything(manifest["seed"] + 10000)
        args.history_size = config["history_size"]
        if model.task_context is None and (config.get("head_history_mode") == "full" or
            (model.intention_encoder is None and not model.use_memory_bank and args.history_size == 1 and manifest.get('temporal_sampling') == 'episode')):
            args.plain_head_batch_size = manifest.get("plain_head_batch_size", 32)
            sparse_loader = DataLoader(SparseHistoryEpisodes(data, args.history_size),
                batch_size=manifest.get("micro_batch_size") or manifest["batch_size"],
                num_workers=0, pin_memory=False, collate_fn=collate_sparse_histories)
            sparse_loader = LimitedBatches(sparse_loader, manifest.get('max_val_batches', 0))
            val_loss, _, metrics = plain_history_epoch(model, sparse_loader, device, args)
        else:
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


def restore_resume_arguments(parser, argv, initial):
    """Apply the original run's defaults before normal argument validation.

    CLI arguments still win. Immutable settings are checked against the manifest
    before training; runtime controls and an increased epoch target may change.
    A variant directory uses its parent's authoritative common manifest because
    variant config.json files contain derived/overridden architecture fields.
    """
    out = Path(initial.output).expanduser().resolve()
    manifest_path = out / 'manifest.json'
    selected_variant = None
    if not manifest_path.is_file() and (out / 'config.json').is_file():
        manifest_path = out.parent / 'manifest.json'
        selected_variant = out.name
    try:
        previous = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        parser.error(f'Cannot resume: could not read saved configuration {manifest_path}: {error}')
    if not isinstance(previous, dict) or not all(key in previous for key in ('epochs', 'variants', 'data', 'cache')):
        parser.error(f'Cannot resume: incomplete run manifest {manifest_path}')
    if selected_variant is not None:
        if selected_variant not in previous['variants']:
            parser.error(f'Variant directory {selected_variant} is not in {manifest_path}')
        if initial.worker_variant not in (None, selected_variant):
            parser.error('--worker-variant conflicts with the selected output variant directory')
    saved = dict(legacy_resume_defaults(previous), **previous)
    invocation_only = {'help', 'output', 'resume', 'evaluate_existing', 'worker_variant'}
    defaults = {action.dest: saved[action.dest] for action in parser._actions
                if action.dest in saved and action.dest not in invocation_only}
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.output = str(manifest_path.parent)
    if selected_variant is not None:
        args.worker_variant = selected_variant
    if args.epochs < previous['epochs']:
        parser.error('Resume epoch target cannot decrease; --epochs is the total target')
    print(f'RESUME configuration: {manifest_path}', flush=True)
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/libero_goal.h5"))
    parser.add_argument("--cache", default=str(ROOT / "data/libero_goal.dinov2_vitl14"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Load saved settings from --output and continue from the latest checkpoint. Explicit runtime options may override settings; --epochs is the total target.")
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
    parser.add_argument("--memory-patch-retrieval",action=argparse.BooleanOptionalAction,default=False,
                        help="Retrieve per-patch memory. May change on resume: retain compatible weights, reset changed memory projections and optimizer state.")
    parser.add_argument("--visual-token-attention",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--visual-context-mode",choices=["none","multiscale"],default="none",help="Explicit patch tokens at three action U-Net scales; enables patch memory when used")
    parser.add_argument("--memory-retrieval-depth",type=int,default=2,help="Sequential reads of the same past bank (legacy checkpoints use 1)")
    parser.add_argument("--memory-intent-query",action=argparse.BooleanOptionalAction,default=False,help="Use inferred intention to guide visual retrieval; no-intent variants retain visual-only queries")
    add_flow_policy_arguments(parser)
    parser.add_argument("--diffusion-clip-sample",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--diffusion-train-steps",type=int,default=100)
    parser.add_argument("--diffusion-loss-repeats",type=int,default=4)
    parser.add_argument("--warm-start",type=Path,help="Existing four-variant run; copy matching parameters, retain fresh diffusion schedule")
    parser.add_argument("--keep-epoch-checkpoints",action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--checkpoint-staging-dir",default=None,help="Fast local checkpoint directory; mirror immutable files to output asynchronously. Resume prefers the highest saved epoch from local/output checkpoints.")
    parser.add_argument("--selection-metric",choices=["val/loss","pos_mse"],default="pos_mse",
                        help="Prediction-based provisional selection; closed-loop selection is separate")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optimizer steps per epoch; 0 uses the full training loader.")
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="Smoke-test validation batch limit; 0 preserves full validation. Must stay fixed on resume.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Episodes loaded together; gradients accumulate to --batch-size before each optimizer step.")
    parser.add_argument("--plain-history-loader", choices=["sparse", "dense"], default="sparse",
                        help="Sparse loads selected anchors for plain_history and the episode-sampled stateless baseline with history 1. Other configurations retain full sequences.")
    parser.add_argument("--plain-head-batch-size", type=int, default=32,
                        help="Maximum supervised points per sparse diffusion loss/sampling call.")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None,
                        help="Pin CPU batches; default off to bound host staging for every variant.")
    parser.add_argument("--profile-data-stages", action="store_true",
                        help="Record cache materialization, packing and CUDA-stage timings for every variant.")
    parser.add_argument("--cached-copy-mode", "--plain-history-copy-mode", dest='plain_history_copy_mode', choices=['direct', 'materialized'], default='direct',
                        help="Direct reads into final batch storage; materialized retains the extra-copy reference.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="DataLoader processes; default 1 with bounded prefetch. HDF5 handles reopen per worker.")
    parser.add_argument("--prefetch-factor", type=int, default=1,
                        help="Batches prefetched per worker when --num-workers is positive.")
    parser.add_argument("--length-bucket-multiplier", type=int, default=20,
                        help="Randomize and batch inside length-sorted pools of this many batches.")
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
    parser.add_argument("--gripper-state-source", choices=["causal", "legacy_zero"], default="causal",
                        help="Measured state or previous command; legacy_zero reproduces old ablation inputs")
    parser.add_argument("--gripper-transition-fraction", type=float, default=0.0,
                        help="Fraction of episode training anchors reserved for actual gripper switches")
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--plain-history-size", type=int, default=4,
                        help="Ordered visual/state window for the plain_history and context_history generative variants")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--segment-length", type=int, default=20)
    parser.add_argument("--variants", nargs="+", default=["no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory"])
    parser.add_argument("--worker-variant", choices=["no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory", "plain_history", 'context_history', 'context_no_memory', 'context_memory', 'context_scratch_no_memory', 'context_scratch_memory'],
                        help="Train only this variant, retaining the ordered common initialization.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.40)
    add_future_arguments(parser)
    add_context_policy_arguments(parser)
    from models.vision_config import add_vision_arguments, feature_width
    add_vision_arguments(parser)
    from models.task_text_encoder import add_task_text_arguments
    add_task_text_arguments(parser)
    args = parser.parse_args(argv)
    if args.resume:
        args = restore_resume_arguments(parser, argv, args)
    args.raw_dim = feature_width(args.vision_backbone,args.raw_dim)
    try:validate_flow_policy_arguments(args)
    except ValueError as error:parser.error(str(error))
    if args.num_task_tokens is not None and (args.num_task_tokens<1 or not args.task_context_checkpoint):
        parser.error("--num-task-tokens must be positive and requires --task-context-checkpoint")
    from data.training_split import resolve_context_manifest
    try:resolve_context_manifest(args)
    except ValueError as error:parser.error(str(error))
    allowed = {'no_intent_no_memory','intent_no_memory','no_intent_memory','intent_memory','plain_history',
        'context_history','context_no_memory','context_memory','context_scratch_no_memory','context_scratch_memory'}
    if any(v not in allowed for v in args.variants):
        parser.error('Unknown ablation variant')
    context_variants = [v for v in args.variants if v.startswith('context_')]
    from data.task_context_policy_dataset import validate_policy_temporal_arguments
    try:validate_policy_temporal_arguments(args)
    except ValueError as error:parser.error(str(error))
    if args.task_context_fusion != 'additive' and not context_variants:
        parser.error('Non-additive task-context fusion requires a context variant')
    if args.task_context_fusion == 'aci' and ('context_history' in args.variants or args.history_size != 1 or
            args.visual_token_attention or args.visual_context_mode != 'none'):
        parser.error('ACI requires history-size 1 and no additional visual-attention mode')
    if args.task_context_fusion_dim < 1 or (args.task_context_fusion in ('cross_attention', 'aci') and args.task_context_fusion_dim % 4):
        parser.error('Task-context fusion width must be positive and divisible by 4 for cross-attention')
    if args.policy_warm_start and args.warm_start:
        parser.error('Choose shared --policy-warm-start or per-variant --warm-start')
    if context_variants:
        if not args.task_context_checkpoint or not args.task_context_split_manifest:
            parser.error('Context variants require --task-context-checkpoint and --task-context-split-manifest')
        if args.temporal_sampling != 'episode' or args.history_size != 1 or args.head_type not in ('diffusion', 'flow_matching') or args.use_task_text or args.memory_intent_query or args.intent_future_weight:
            parser.error('Context variants require full episodes, history size 1, diffusion/flow matching, no task text/legacy future loss/cognitive memory query')
        if args.gripper_state_source != 'causal':
            parser.error('Context variants require causal gripper state')
    try:validate_future_arguments(args)
    except ValueError as error:parser.error(str(error))
    if args.freeze_intent_backbone and not (args.warm_start or args.resume):
        parser.error('Freezing the backbone requires --warm-start or --resume; do not freeze a random encoder')
    if args.intent_future_weight:
        if any(not name.startswith('intent_') for name in args.variants):
            parser.error('Future prediction requires intent variants: --variants intent_no_memory and/or intent_memory')
        if args.temporal_sampling=='crop' and args.segment_length<args.history_size+max(args.intent_future_horizons):
            parser.error('Cropped segments must include the longest future horizon after history; increase --segment-length or use episode sampling')
    if args.plain_head_batch_size < 1:
        parser.error("--plain-head-batch-size must be positive")
    if args.micro_batch_size is not None and (args.micro_batch_size < 1 or
                                             args.batch_size % args.micro_batch_size):
        parser.error("--micro-batch-size must be positive and divide --batch-size")
    for name in ("epochs", "batch_size", "history_size", "plain_history_size", "chunk_size", "segment_length",
                 "state_dim", "compressed_dim", "head_d_model", "mamba_output_dim",
                 "mamba_d_state", "mamba_d_conv", "mamba_expand", "num_intent_tokens",
                 "intent_dim", "memory_bank_len", "cpu_threads", "head_window_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.prefetch_factor < 1 or args.length_bucket_multiplier < 1:
        parser.error("--num-workers must be nonnegative; prefetch and bucket multipliers must be positive")
    if args.segment_length < args.history_size + args.chunk_size:
        parser.error("Segment must contain both history and action chunk")
    if any(name in args.variants for name in ("plain_history", "context_history")):
        if args.head_type not in ("diffusion", "flow_matching"):
            parser.error("History variants require diffusion or flow matching")
        if args.temporal_sampling != "episode":
            parser.error("History variants require episode sampling for causal start padding")
    if args.chunk_size < 4:
        parser.error("--chunk-size must be at least 4 for the generative U-Net")
    if args.head_d_model % 8:
        parser.error("--head-d-model must be divisible by 8 for GroupNorm")
    if not 0 <= args.gripper_transition_fraction <= 1:
        parser.error("gripper-transition-fraction must be in [0, 1]")
    if args.temporal_sampling != "episode" and args.gripper_transition_fraction:
        parser.error("Transition sampling requires episode sampling; use --gripper-transition-fraction 0 for crops")
    if args.compressed_dim % 4 or args.state_dim % 4 or (args.intent_dim * args.num_intent_tokens) % 4:
        parser.error("Compressed/state/total intent dimensions must be divisible by 4 for attention")
    if len(set(args.cameras)) != len(args.cameras):
        parser.error("Camera names must be unique")
    if not 0 < args.validation_fraction < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("Validation fraction must be in (0, 1); GPU memory fraction in (0, 1]")
    if args.lr <= 0 or min(args.weight_decay, args.grad_clip, args.gripper_loss_weight, args.max_steps, args.max_val_batches) < 0:
        parser.error("Learning rate must be positive; decay, clipping, loss weight and batch/step limits must be nonnegative")
    if min(args.supervision_points,args.diffusion_train_steps,args.diffusion_loss_repeats)<1 or args.diffusion_train_steps<10:
        parser.error('Supervision points/repeats must be positive; diffusion schedule needs at least 10 steps')
    if not 0 <= args.observation_dropout_prob <= 1:
        parser.error("Observation dropout probability must be in [0,1]")
    if args.temporal_sampling=="episode" and args.history_size!=1:
        parser.error('Episode supervision currently requires history size 1')
    if args.visual_context_mode == 'multiscale':
        args.memory_patch_retrieval = True
    if args.memory_retrieval_depth < 1:
        parser.error('--memory-retrieval-depth must be positive')
    if args.memory_mode == 'legacy' and (args.memory_retrieval_depth != 1 or args.memory_intent_query):
        parser.error('Legacy memory requires depth 1 and no intention-guided query')
    if args.visual_context_mode != 'none' and (args.visual_token_attention or args.head_type not in ('diffusion','flow_matching')):
        parser.error('Multiscale context requires a generative head and replaces --visual-token-attention')
    if args.visual_context_mode != 'none' and args.memory_mode != 'episodic':
        parser.error('Multiscale visual memory requires episodic mode')
    if args.visual_context_mode != 'none' and args.memory_value_preserving and not args.memory_patch_temporal:
        parser.error('Multiscale raw-value memory requires --memory-patch-temporal')
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
    if args.checkpoint_staging_dir:
        args.checkpoint_staging_dir = str(Path(args.checkpoint_staging_dir).expanduser().resolve())
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
    from models.vision_config import validate_cache
    validate_cache(dataset._dinov2_index["__meta__"], args.vision_backbone, args.raw_dim)
    if sample.shape[-2:] != (257 * len(args.cameras), args.raw_dim):
        raise ValueError("Expected 257 DINOv2 tokens of configured width per selected camera")
    pool_out_dim = (sample.shape[-2] - len(args.cameras)) * args.compressed_dim
    split_record = None
    if args.task_context_checkpoint or args.task_context_split_manifest:
        if not args.task_context_split_manifest:
            raise ValueError('Pretrained context requires its exact split manifest for every arm')
        from models.task_context_observer import load_observer_bundle
        metadata = (load_observer_bundle(args.task_context_checkpoint)['metadata']
                    if args.task_context_checkpoint else None)
        train_eps, val_eps, split_record = policy_episode_split(dataset, args.task_context_split_manifest, metadata,
            allow_unbound_manifest=args.task_context_split_override)
    else:
        train_eps, val_eps = episode_split(dataset, args.seed, args.validation_fraction)
    if args.intent_future_weight:
        longest=max(dataset._get_episode_length(ep) for ep in train_eps)
        if max(args.intent_future_horizons)+args.history_size>longest:
            raise ValueError('The longest future horizon has no eligible training episode')
    text_vocab = (sorted({word for ep in train_eps for word in
                          re.findall(r"[a-z0-9]+", dataset._read_text(ep).lower())})
                  if args.use_task_text and args.text_encoder_type == "bag" else [])
    if len(train_eps) < args.batch_size:
        raise ValueError("Batch size exceeds the training episode count")
    if any(dataset._get_episode_length(ep) < args.history_size + args.chunk_size for ep in train_eps + val_eps):
        raise ValueError("An episode is too short for this history/chunk configuration")
    provenance = source_provenance()
    manifest = dict(vars(args), commit=provenance['commit'], source_provenance=provenance,
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), split="task-stratified whole episodes",
        train_episodes=[dataset._episode_keys[ep] for ep in train_eps],
        val_episodes=[dataset._episode_keys[ep] for ep in val_eps],
        num_cameras=len(args.cameras), pool_out_dim=pool_out_dim, frozen_cached_vision=True,
        use_text=args.use_task_text, text_encoder_type=args.text_encoder_type if args.use_task_text else "clip",
        text_vocab=text_vocab,
        intention_recurrence="causal observation state across each segment")
    manifest['task_context_split'] = split_record
    if split_record is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
    manifest['task_context_matmul_tf32'] = False if split_record is not None else None
    from data.task_context_dataset import file_hash
    manifest['task_context_bundle_sha256'] = file_hash(args.task_context_checkpoint) if args.task_context_checkpoint else None
    manifest['policy_warm_start_sha256'] = file_hash(args.policy_warm_start) if args.policy_warm_start else None
    if args.resume and (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        legacy_defaults = legacy_resume_defaults(previous)
        for key in (*LEGACY_FLOW_DEFAULTS, "text_encoder_type","text_model_name","text_model_revision","text_max_length","vision_backbone","raw_dim","intent_future_weight","intent_future_horizons","intent_future_hidden_dim","freeze_intent_backbone",
                    "gripper_state_source", "gripper_transition_fraction", "max_steps", "max_val_batches", "history_size", "plain_history_size", "chunk_size", "segment_length", "batch_size",
                    "seed", "variants", "data", "cache", "cameras", "validation_fraction", "lr",
                    "weight_decay", "grad_clip", "head_type", "state_dim", "compressed_dim",
                    "head_d_model", "mamba_output_dim", "mamba_d_state", "mamba_d_conv", "mamba_expand",
                    "use_task_text", "text_dim", "text_vocab", "task_context_fusion", "task_context_fusion_dim",
                    "intent_dim", "num_intent_tokens", "memory_bank_len", "gripper_loss_weight", "gripper_threshold",
                    "temporal_sampling","supervision_points","memory_mode","memory_detach_writes","memory_write_fused",
                    "visual_context_mode","memory_retrieval_depth","memory_intent_query","memory_perceptual_recency","memory_field_masks","memory_pre_state_visual","memory_value_preserving","memory_patch_temporal","memory_context_only","memory_patch_retrieval","diffusion_train_steps","diffusion_loss_repeats","warm_start","selection_metric","visual_token_attention","diffusion_clip_sample","observation_dropout_prob","drop_state_with_all_views","head_window_batch_size","length_bucket_multiplier"):
            if key == 'memory_patch_retrieval':
                continue  # Per-checkpoint architecture migration is handled below.
            if key in LEGACY_FLOW_DEFAULTS and args.head_type != "flow_matching":
                continue
            if previous.get(key, legacy_defaults.get(key)) != manifest[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        for key, default in dict(policy_temporal_mamba=False, policy_temporal_width=128, policy_temporal_layers=1).items():
            if previous.get(key, default) != manifest[key]:
                raise ValueError(f'Resume policy temporal configuration mismatch: {key}')
        for key in ('task_context_checkpoint','task_context_split_manifest','task_context_trainable',
                    'task_context_bundle_sha256','task_context_split','task_context_matmul_tf32'):
            if previous.get(key, False if key == 'task_context_trainable' else None) != manifest[key]:
                raise ValueError(f'Resume context configuration mismatch: {key}')
        if previous.get('policy_warm_start_sha256') != manifest['policy_warm_start_sha256']:
            raise ValueError('Policy warm-start weights changed on resume')
        patch_changed = previous.get('memory_patch_retrieval', False) != args.memory_patch_retrieval
        if patch_changed:
            atomic_json(out / f'manifest_before_memory_patch_{time.time_ns()}.json', previous)
        manifest = extend_resume_manifest(previous, manifest)
        atomic_json(out / "manifest.json", manifest)
        if args.epochs > previous["epochs"] or patch_changed:
            # Completion markers from the previous target must not describe an
            # extended run as finished while its new epochs are in progress.
            (out / "COMPLETE").unlink(missing_ok=True)
            for name in args.variants:
                (out / name / "COMPLETE").unlink(missing_ok=True)
    else:
        atomic_json(out / "manifest.json", manifest)
    print(f"Split: {len(train_eps)} train episodes / {len(val_eps)} held-out episodes", flush=True)
    train_data = CachedEpisodes(dataset, train_eps, args.segment_length, args.seed, True, args.temporal_sampling,args.supervision_points,args.chunk_size,args.observation_dropout_prob,args.drop_state_with_all_views)
    val_data = CachedEpisodes(dataset, val_eps, args.segment_length, args.seed + 1000, False, args.temporal_sampling,args.supervision_points,args.chunk_size)
    train_data.gripper_transition_fraction = args.gripper_transition_fraction
    train_data.gripper_state_source = val_data.gripper_state_source = args.gripper_state_source
    train_data.profile = val_data.profile = args.profile_data_stages
    train_lengths = [dataset._get_episode_length(ep) if args.temporal_sampling == "episode"
                     else min(dataset._get_episode_length(ep), args.segment_length)
                     for ep in train_eps]
    micro_batch_size = args.micro_batch_size or args.batch_size
    train_sampler = LengthBucketBatchSampler(
        train_lengths, args.batch_size, args.seed, args.length_bucket_multiplier,
        drop_last=True, micro_batch_size=micro_batch_size)
    loader_options = dict(
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_options.update(prefetch_factor=args.prefetch_factor, multiprocessing_context="spawn")
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
        if name not in {"no_intent_no_memory", "intent_no_memory", "no_intent_memory", "intent_memory", "plain_history",
                        'context_history','context_no_memory','context_memory','context_scratch_no_memory','context_scratch_memory'}:
            raise ValueError(name)
        intent = name.startswith("intent_")
        memory = name.endswith("_memory") and not name.endswith("no_memory")
        full_history = name in ("plain_history", "context_history")
        context = name.startswith('context_')
        scratch = name.startswith('context_scratch_')
        variant_history_size = args.plain_history_size if full_history else args.history_size
        # Context encoders must scan every real observation, including frames
        # between supervised windows. Sparse history loading would skip them.
        sparse_history = ((full_history and not context) or (not intent and not context and not memory and variant_history_size == 1 and args.temporal_sampling == 'episode')) and args.plain_history_loader == "sparse"
        variant_options = dict(loader_options, pin_memory=(args.pin_memory
            if args.pin_memory is not None else False))
        variant_train_data = SparseHistoryEpisodes(train_data, variant_history_size,
            profile=args.profile_data_stages, copy_mode=args.plain_history_copy_mode) if sparse_history else train_data
        variant_val_data = SparseHistoryEpisodes(val_data, variant_history_size,
            profile=args.profile_data_stages, copy_mode=args.plain_history_copy_mode) if sparse_history else val_data
        collator = (collate_sparse_histories if sparse_history else
                    collate_cached_segments if args.plain_history_copy_mode == 'direct' else collate_segments)
        train_loader = DataLoader(variant_train_data, batch_sampler=train_sampler,
                                  collate_fn=collator, **variant_options)
        val_loader = DataLoader(variant_val_data, batch_size=micro_batch_size,
                                shuffle=False, collate_fn=collator, **variant_options)
        val_loader = LimitedBatches(val_loader, args.max_val_batches)
        seed_everything(args.seed)
        model = ALIGNIntentionModel(vision_backbone=args.vision_backbone,raw_dim=args.raw_dim,state_dim=args.state_dim,
            mamba_output_dim=0 if full_history else args.mamba_output_dim,
            action_dim=7, chunk_size=args.chunk_size, history_size=variant_history_size,
            num_cameras=len(args.cameras), compressed_dim=args.compressed_dim,
            head_type=args.head_type, head_history_mode="full" if full_history else "mean",
            head_d_model=args.head_d_model,
            mamba_d_state=args.mamba_d_state, mamba_d_conv=args.mamba_d_conv, mamba_expand=args.mamba_expand,
            use_intent_tokens=intent, num_intent_tokens=args.num_intent_tokens, intent_dim=args.intent_dim,
            use_memory_bank=memory, memory_bank_len=args.memory_bank_len,
            memory_mode=args.memory_mode,memory_detach_writes=args.memory_detach_writes,
            memory_perceptual_recency=args.memory_perceptual_recency,memory_field_masks=args.memory_field_masks,memory_pre_state_visual=args.memory_pre_state_visual,memory_value_preserving=args.memory_value_preserving,memory_patch_temporal=args.memory_patch_temporal,memory_context_only=args.memory_context_only,memory_write_fused=args.memory_write_fused,memory_patch_retrieval=args.memory_patch_retrieval,
            **flow_policy_options(vars(args)),
            diffusion_train_steps=args.diffusion_train_steps,diffusion_loss_repeats=args.diffusion_loss_repeats,
            visual_context_mode=getattr(args,"visual_context_mode","none"),
            memory_retrieval_depth=getattr(args,"memory_retrieval_depth",1),memory_intent_query=getattr(args,"memory_intent_query",False),
            visual_token_attention=args.visual_token_attention,diffusion_clip_sample=args.diffusion_clip_sample,
            use_text=args.use_task_text,text_dim=args.text_dim,
            text_model_name=args.text_model_name,text_model_revision=args.text_model_revision,text_max_length=args.text_max_length,
            text_encoder_type=args.text_encoder_type if args.use_task_text else "clip",text_vocab=text_vocab,
            intent_future_weight=args.intent_future_weight,intent_future_horizons=args.intent_future_horizons,
            intent_future_hidden_dim=args.intent_future_hidden_dim,
            task_context_checkpoint=args.task_context_checkpoint if context else None,
            task_context_tokens=args.num_task_tokens if context else None,
            task_context_trainable=args.task_context_trainable,
            task_context_initialization='scratch' if scratch else 'pretrained',
            task_context_fusion=args.task_context_fusion if context else 'additive',
            task_context_fusion_dim=args.task_context_fusion_dim,
            policy_temporal_mamba=args.policy_temporal_mamba,
            policy_temporal_width=args.policy_temporal_width,
            policy_temporal_layers=args.policy_temporal_layers)
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
                if key.startswith('task_context.'):
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
            future_compatible=(saved['config'].get('intent_future_horizons',[16,64])==args.intent_future_horizons)
            matched = {k:v for k,v in source.items() if k in own and own[k].shape==v.shape
                       and not k.startswith("vision_encoder.") and not k.endswith(("alpha_bar","sigma"))
                       and (future_compatible or not k.startswith('intention_future_predictor.'))}
            model.load_state_dict(matched,strict=False)
            print(f"WARM START {name}: {len(matched)} parameters/buffers, fresh noise schedule",flush=True)
            del saved,source,own,matched
        if args.policy_warm_start:
            print(f'POLICY WARM START {name}: {warm_start_policy(model,args.policy_warm_start)}',flush=True)
        if args.freeze_intent_backbone:freeze_intent_backbone(model)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(policy_parameter_groups(model,args.lr), lr=args.lr, weight_decay=args.weight_decay, fused=True)
        loop_args = SimpleNamespace(history_size=variant_history_size, chunk_size=args.chunk_size,
            action_dim=7, head_type=args.head_type, skip_nan=False, grad_clip=args.grad_clip,
            no_sample_during_train=True, debug=False, gripper_threshold=args.gripper_threshold,
            gripper_loss_weight=args.gripper_loss_weight,
            head_window_batch_size=args.head_window_batch_size,
            plain_head_batch_size=args.plain_head_batch_size,
            profile_data_stages=args.profile_data_stages,
            gradient_accum_steps=args.batch_size // micro_batch_size)
        run = out / name
        checkpoints = CheckpointWriter(run,args.checkpoint_staging_dir)
        start_epoch = 1
        best = float("inf")
        memory_migration = None
        if args.resume and run.exists():
            last_checkpoint = checkpoints.resume_path()
            if last_checkpoint is not None:
                saved = torch.load(last_checkpoint, map_location="cpu", weights_only=True, mmap=True)
                # With asynchronous mirroring, summary.json may be ahead of NFS
                # weights after interruption. Only a complete checkpoint proves
                # the requested epoch budget has been saved.
                if saved["epoch"] >= args.epochs:
                    print(f"SKIP completed {name} (checkpoint epoch {saved['epoch']})",flush=True)
                    if memory_patch_transition(model, saved):
                        print(f"  Memory architecture unchanged in completed checkpoint; increase --epochs beyond {saved['epoch']} to train the new memory.",flush=True)
                    del model,optimizer,params,vision,saved
                    torch.cuda.empty_cache()
                    continue
                if memory_patch_transition(model, saved):
                    memory_migration = migrate_memory_weights(model, saved)
                    memory_migration.update(source_checkpoint=str(last_checkpoint), source_provenance=provenance)
                    memory_migration = archive_memory_transition(run,
                        [last_checkpoint, checkpoints.best_path(), run/'training_last.pt', run/'intention_best.pt'],
                        memory_migration)
                    # The optimizer is newly constructed and has no moments yet.
                    # Never put the old flattened-memory moments onto the GPU.
                    print(f"MIGRATE {name}: patch retrieval {memory_migration['from_memory_patch_retrieval']} -> {args.memory_patch_retrieval}; compatible weights retained; perceptual memory, optimizer and best selection reset. Archive: {memory_migration['archive']}",flush=True)
                else:
                    incompatible = model.load_state_dict(saved["model_state_dict"], strict=False)
                    if model.task_context is not None and (incompatible.unexpected_keys or
                            any(not k.startswith('vision_encoder.') for k in incompatible.missing_keys)):
                        raise ValueError('Context resume checkpoint is missing trained parameters')
                    optimizer.load_state_dict(saved["optimizer_state_dict"])
                    memory_migration = saved.get('config', {}).get('memory_patch_migration')
                start_epoch = saved["epoch"] + 1
                best = float('inf') if memory_patch_transition(model, saved) else saved["best_loss"]
                records[name] = saved["record"]
                best_checkpoint = checkpoints.best_path()
                if not best_checkpoint.exists() or not zipfile.is_zipfile(best_checkpoint):
                    print(f"RESUME {name}: best checkpoint missing or incomplete; "
                          "next epoch will establish a new best", flush=True)
                    best = float("inf")
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
                      use_history=not full_history, mamba_output_dim=0 if full_history else args.mamba_output_dim,
                      history_size=variant_history_size,
                      head_history_mode="full" if full_history else "mean", action_dim=7,
                      num_cameras=len(args.cameras), model_class="ALIGNIntentionModel",
                      trainable_parameters=sum(p.numel() for p in params))
        if memory_migration is not None:
            config['memory_patch_migration'] = memory_migration
        config.update(context_policy_options(model))
        atomic_json(run / "config.json", config)
        torch.cuda.reset_peak_memory_stats()
        print(f"START {name}: {config['trainable_parameters']:,} trainable parameters", flush=True)
        with checkpoints, (run / "metrics.jsonl").open("a" if start_epoch > 1 else "w", buffering=1) as log:
            for epoch in range(start_epoch, args.epochs + 1):
                checkpoints.raise_if_failed()
                train_data.epoch = epoch
                train_sampler.set_epoch(epoch)
                seed_everything(args.seed + epoch)
                start = time.monotonic()
                train_start = time.perf_counter()
                if sparse_history:
                    train_loss, _ = plain_history_epoch(model, train_loader, device,
                        loop_args, optimizer=optimizer, max_steps=args.max_steps)
                else:
                    train_loss, _ = train_v4_epoch(model, train_loader, optimizer, device,
                        loop_args, max_steps=args.max_steps)
                train_wall_s = time.perf_counter() - train_start
                seed_everything(args.seed + 10000)
                val_start = time.perf_counter()
                val_loss, _, metrics = (plain_history_epoch(model, val_loader, device, loop_args)
                                       if sparse_history else validate(model, val_loader, device, loop_args))
                val_wall_s = time.perf_counter() - val_start
                row = dict(epoch=epoch, **{"train/loss": train_loss, "val/loss": val_loss},
                    **{'train/'+k:v for k,v in getattr(loop_args,'_last_future_metrics',{}).items()},
                    **metrics, elapsed_s=time.monotonic() - start,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                    timing={**getattr(loop_args, "_last_stage_times", {}),
                            "train_wall_s": train_wall_s,
                            "val_wall_s": val_wall_s})
                if not np.isfinite(train_loss) or not np.isfinite(val_loss):
                    raise FloatingPointError(f"Nonfinite loss in {name}: {row}")
                checkpoint_start = time.perf_counter()
                selection_score = row[args.selection_metric]
                improved = selection_score < best
                if improved:
                    best = selection_score
                    records[name] = dict(best=row, completed_epochs=epoch)
                records[name]["completed_epochs"] = epoch
                # One immutable CPU snapshot shared by best/archive/recovery saves.
                # Frozen DINO weights are reconstructible and omitted from all files.
                snapshot = cpu_snapshot(dict(
                    model_state_dict={key:value for key,value in model.state_dict().items()
                                      if not key.startswith("vision_encoder.")},
                    optimizer_state_dict=optimizer.state_dict(), record=records[name],
                    config=dict(config,frozen_vision_omitted=True)))
                evaluation = dict(model_state_dict=snapshot["model_state_dict"],
                                  config=snapshot["config"],epoch=epoch,val_loss=val_loss)
                if improved:
                    checkpoints.save("intention_best.pt",evaluation)
                if args.keep_epoch_checkpoints:
                    checkpoints.save(f"epoch_{epoch:03d}.pt",evaluation)
                latest = dict(snapshot,epoch=epoch,best_loss=best)
                checkpoints.save("training_last.pt",latest)
                del latest,snapshot,evaluation
                row["checkpoint_io"] = checkpoints.status()
                row["timing"]["checkpoint_save_s"] = time.perf_counter() - checkpoint_start
                row["timing"]["total_wall_s"] = time.monotonic() - start
                log.write(json.dumps(row, allow_nan=False) + "\n")
                summarize(report_out, records, manifest)
                print(f"RESULT {name} epoch={epoch} train={train_loss:.6f} val={val_loss:.6f} "
                      f"pos={metrics['pos_mse']:.6f} rot={metrics['rot_mse']:.6f} "
                      f"grip_acc={metrics['grip_acc']:.3f} seconds={row['timing']['total_wall_s']:.1f} "
                      f"checkpoint={row['timing']['checkpoint_save_s']:.1f}s "
                      f"mirror_pending={row['checkpoint_io']['pending_files']}", flush=True)
        del model, optimizer, params, vision
        torch.cuda.empty_cache()
    del train_loader, val_loader
    gc.collect()
    dataset.close()
    summarize(report_out, records, manifest)
    (report_out / "COMPLETE").write_text("Requested worker variants completed.\n")
    print(f"COMPLETE: {report_out / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
