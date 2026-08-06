#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Async-inference LIBERO evaluation with real-time live view.

Architecture:
  - Sim thread (main thread): runs env.step() at a target FPS (default 20 Hz).
    Uses the latest available model action (or expert action in Phase 1).
    Renders each frame and pushes it to the display queue.
  - Inference thread: reads the latest sim state from a shared buffer,
    runs the model, and publishes new actions to a shared action queue.
    Runs at whatever speed the GPU allows — decoupled from sim FPS.
  - Display: cv2.imshow in the main thread, showing each rendered frame
    at the sim's natural rate.

This means rendering FPS is constant (controlled by --sim-fps) regardless
of inference latency. The sim uses the most recent model-predicted action
chunk, falling back to the last known action if inference hasn't finished
yet for the current step.

Usage:
    python eval/eval_libero_v4_async.py \
        --data data/libero_object.h5 \
        --checkpoint checkpoints/v4/libero_object/run_1/intention_best.pt \
        --n-episodes 1 --live-view --sim-fps 20

Requires: opencv-python (pip install opencv-python) for --live-view.
"""

import argparse
import json
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.backends.cudnn.enabled = False
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from eval.eval_intention import load_intention_model

# MuJoCo / LIBERO imports
try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    OffScreenRenderEnv = None
    get_libero_path = None

try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:
    _Rotation = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ================================================================
# Shared helpers (copied from eval_libero_v4_trajectory.py for
# standalone operation — no import dependency on that script)
# ================================================================

def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    if _Rotation is not None:
        return _Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def get_sim_frame(env, key: str = "agentview_image",
                  render_size: int = 256,
                  flip_vertical: bool = True,
                  flip_horizontal: bool = False) -> np.ndarray:
    sim_key_map = {
        "image": "agentview_image",
        "agentview_image": "agentview_image",
        "wrist_image": "robot0_eye_in_hand_image",
        "robot0_eye_in_hand_image": "robot0_eye_in_hand_image",
    }
    sim_key = sim_key_map.get(key, key)
    try:
        obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    except Exception:
        img = np.zeros((render_size, render_size, 3), dtype=np.uint8)
    else:
        img = obs.get(sim_key)
        if img is None:
            for k in ["agentview_image", "robot0_eye_in_hand_image"]:
                img = obs.get(k)
                if img is not None:
                    break
        if img is None:
            img = np.zeros((render_size, render_size, 3), dtype=np.uint8)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 4:
            img = img[0]
        img = img.astype(np.uint8)
    if flip_vertical:
        img = np.flipud(img).copy()
    if flip_horizontal:
        img = np.fliplr(img).copy()
    return img


def get_sim_eef_pose(obs: dict) -> np.ndarray:
    pos = obs.get("robot0_eef_pos", np.zeros(3))
    quat = obs.get("robot0_eef_quat", np.zeros(4))
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    if isinstance(quat, torch.Tensor):
        quat = quat.cpu().numpy()
    aa = quat_to_axisangle(quat)
    return np.concatenate([pos, aa]).astype(np.float32)


def get_bddl_path(suite_name: str, task_name: str) -> str:
    if get_libero_path is None:
        raise ImportError("libero not installed")
    safe_name = "".join(
        c if c.isalnum() else "_" for c in task_name.lower()
    ).strip("_")
    bddl_dir = os.path.join(get_libero_path("bddl_files"), suite_name)
    if not os.path.isdir(bddl_dir):
        return os.path.join(bddl_dir, f"{safe_name}.bddl")
    for fname in os.listdir(bddl_dir):
        if safe_name in fname and fname.endswith(".bddl"):
            return os.path.join(bddl_dir, fname)
    return os.path.join(bddl_dir, f"{safe_name}.bddl")


def load_trajectory(h5_path: str, episode_key: str,
                    cameras: List[str]) -> Optional[Dict]:
    with h5py.File(h5_path, "r") as h5:
        if episode_key not in h5:
            return None
        group = h5[episode_key]
        frames_group = group.get("frames", None)
        if frames_group is None:
            return None
        available = list(frames_group.keys()) if hasattr(frames_group, "keys") else []
        cam_list = [c for c in cameras if c in available]
        if not cam_list:
            cam_list = [available[0]] if available else None
            if cam_list is None:
                return None
        if len(cam_list) == 1:
            frames = frames_group[cam_list[0]][:]
        else:
            per_cam = [frames_group[c][:] for c in cam_list]
            frames = np.stack(per_cam, axis=1)
        poses = None
        if "poses" in group:
            poses = group["poses"][:]
        elif "noisy_poses" in group:
            poses = group["noisy_poses"][:]
        actions = group["actions"][:]
        if poses is not None:
            gripper = actions[:, -1:]
            states = np.concatenate([poses, gripper], axis=1).astype(np.float32)
        else:
            return None
        text = ""
        if "texts" in group:
            try:
                text = json.loads(group["texts"][()])[0]
            except Exception:
                text = ""
        return {
            "frames": frames,
            "states": states,
            "actions": actions,
            "poses": poses,
            "text": text,
            "cam_name": cam_list[0] if len(cam_list) == 1 else cam_list,
        }


def list_episodes(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as h5:
        return sorted([k for k in h5.keys() if k.startswith("ep_")])


def _try_load_libero_task_list(suite_name: str) -> List[str]:
    try:
        from libero.libero.benchmark import get_benchmark
        benchmark = get_benchmark(suite_name)
        task_list = []
        for i in range(benchmark.n_tasks):
            task = benchmark.get_task(i)
            task_list.append(task.name)
        return task_list
    except Exception:
        return []


# ================================================================
# Inference thread
# ================================================================

class InferenceWorker(threading.Thread):
    """Background thread that runs model inference asynchronously.

    Reads the latest sim state from `state_queue` (a single-slot queue,
    always keeping only the newest), runs the model, and puts action
    chunks into `action_queue`. The sim thread consumes actions from
    `action_queue` at its own pace.

    Thread safety:
    - state_queue: maxsize=1, only newest state kept (sim overwrites)
    - action_queue: maxsize=1, only newest action chunk kept (inference overwrites)
    """

    def __init__(self, model, device, chunk_size: int,
                 state_queue: queue.Queue, action_queue: queue.Queue,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.stop_event = stop_event
        self.n_calls = 0
        self.total_inference_ms = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Get latest state (block up to 0.5s so we can check stop_event)
                state_data = self.state_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            f_t, s_t = state_data  # (1, K, V, H, W, 3) and (1, K, 7)

            t0 = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=self.device.type == "cuda"):
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        out = self.model(f_t, s_t)
                        h_current = out["h_seq"][:, -1]
                        intent_emb = out.get("intent_emb", None)

                        if getattr(self.model, 'use_memory_bank', False) and intent_emb is not None:
                            z_v_current = out["z_v_pooled_seq"][:, -1]
                            z_s_current = out["z_s_seq"][:, -1]
                            z_v_fused, z_s_fused, intent_fused = self.model.memory_module(
                                z_v_current, z_s_current, intent_emb,
                            )
                            h_for_head = intent_fused
                        else:
                            h_for_head = intent_emb if intent_emb is not None else None

                        if self.model.head_type == "diffusion":
                            a_model_full = self.model.sample_actions(
                                out["z_v_pooled_seq"], out["z_s_seq"], h_for_head,
                            )
                        else:
                            a_model_full = self.model.predict_actions(
                                out["z_v_pooled_seq"], out["z_s_seq"], h_for_head,
                            )

            inference_ms = (time.perf_counter() - t0) * 1000.0
            self.n_calls += 1
            self.total_inference_ms += inference_ms

            # chunk_np: (chunk_size, action_dim)
            chunk_np = a_model_full[0, :self.chunk_size, :].float().cpu().numpy()

            # Publish to action queue (overwrite stale if needed)
            try:
                self.action_queue.get_nowait()  # drain stale
            except queue.Empty:
                pass
            self.action_queue.put((chunk_np, self.n_calls, inference_ms))


# ================================================================
# Async sim rollout
# ================================================================

def run_async_episode(
    env,
    model: torch.nn.Module,
    device: torch.device,
    expert_actions: np.ndarray,
    expert_poses: Optional[np.ndarray],
    chunk_size: int,
    max_steps: int,
    alpha: float,
    action_scale: float,
    switch_at: float,
    render_size: int,
    cameras: List[str],
    flip_vertical: bool,
    flip_horizontal: bool,
    noise_std: float,
    sim_fps: float,
    live_view: bool,
    debug: bool,
    action_horizon: int = 1,
    ensemble: str = "none",
    ensemble_decay: float = 0.9,
) -> Dict:
    """Run one episode with async inference and constant-FPS rendering.

    The sim thread (main) runs at `sim_fps` Hz. The inference thread
    runs in the background at whatever speed the GPU allows.

    Actions:
    - Phase 1 (step < switch_step): expert action (with optional noise)
    - Phase 2 (step >= switch_step): latest model action from inference
      thread. If no new action is available, reuse the last known action.

    The model is fed the current K-window of (frames, states) every step
    by the sim thread (pushed to state_queue). The inference thread picks
    up the newest one whenever it's ready.

    Action horizon and ensemble:
    - action_horizon=H: the first H actions from each chunk are consumed
      sequentially before the sim considers the chunk exhausted.
    - ensemble="none": each chunk's first H actions are used as-is.
    - ensemble="uniform": all K actions from each chunk are pushed to a
      buffer; at each step, all entries whose target_step == step are
      averaged with equal weight.
    - ensemble="decay": like "uniform" but weights decay as
      ensemble_decay ** (current_call_idx - source_call_idx).
    """
    # Pad expert actions
    actions = expert_actions[:max_steps].copy()
    if actions.shape[1] < 7:
        pad = np.zeros((actions.shape[0], 7 - actions.shape[1]), dtype=actions.dtype)
        actions = np.concatenate([actions, pad], axis=1)
    ep_len = len(actions)
    if len(actions) < max_steps:
        pad = np.tile(actions[-1:], (max_steps - len(actions), 1))
        actions = np.concatenate([actions, pad], axis=0)

    obs = env.reset()
    frames = []
    sim_positions = []
    errors = []
    stored_actions = []
    stored_inference_flags = []  # True if this step used a fresh model action
    stale_repeat_counts = []     # consecutive repeats of the same stale action
    success = 0

    # Reset memory bank
    if getattr(model, 'use_memory_bank', False) and model.memory_module is not None:
        model.memory_module.reset(batch_size=1, device=device)

    # Buffers (K-window)
    pose_buffer = []
    frame_buffer = []

    def _normalize_frame(f):
        if f is None:
            return None
        if f.ndim == 4:
            f = f[0]
        return f.astype(np.uint8)

    def _render_all_cameras():
        per_cam = []
        for camera_view in cameras:
            f = _normalize_frame(get_sim_frame(env, key=camera_view,
                                               render_size=render_size,
                                               flip_vertical=flip_vertical,
                                               flip_horizontal=flip_horizontal))
            per_cam.append(f)
        return np.stack(per_cam, axis=0)  # (V, H, W, 3)

    # Initial state
    init_eef = get_sim_eef_pose(obs)
    init_state = np.concatenate([init_eef, [0.0]]).astype(np.float32)
    init_frame_stack = _render_all_cameras()

    for k in range(chunk_size):
        pose_buffer.append(init_state.copy())
        frame_buffer.append(init_frame_stack.copy())

    n_steps = max_steps
    switch_step = int(ep_len * switch_at)

    # Async inference setup
    state_queue: queue.Queue = queue.Queue(maxsize=1)
    action_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = InferenceWorker(
        model=model, device=device, chunk_size=chunk_size,
        state_queue=state_queue, action_queue=action_queue,
        stop_event=stop_event,
    )
    worker.start()

    # Latest model action chunk (for Phase 2)
    latest_chunk: Optional[np.ndarray] = None
    chunk_consumed_idx = 0  # how many actions from current chunk have been used
    latest_call_idx = 0

    # Ensemble buffer: list of (target_step, weight, action_vector)
    # and parallel list of source_call_idx for decay weighting.
    pending_actions: List[Tuple[int, float, np.ndarray]] = []
    pending_source_call_idx: List[int] = []
    n_model_calls_so_far = 0  # tracks total inference calls for decay weighting

    # Validate action_horizon
    if action_horizon < 1:
        raise ValueError(f"action_horizon must be >= 1 (got {action_horizon})")
    if action_horizon > chunk_size:
        raise ValueError(
            f"action_horizon ({action_horizon}) cannot exceed chunk_size "
            f"({chunk_size})"
        )
    if ensemble not in ("none", "uniform", "decay"):
        raise ValueError(
            f"ensemble must be one of 'none', 'uniform', 'decay' (got {ensemble!r})"
        )
    if not (0.0 < ensemble_decay <= 1.0):
        raise ValueError(
            f"ensemble_decay must be in (0, 1] (got {ensemble_decay})"
        )

    target_dt = 1.0 / sim_fps if sim_fps > 0 else 0.0

    for step in range(n_steps):
        step_t0 = time.perf_counter()

        # 1. Render current sim frame
        current_frame_stack = _render_all_cameras()
        frames.append(current_frame_stack[0].copy())

        if live_view and CV2_AVAILABLE:
            bgr = cv2.cvtColor(current_frame_stack[0], cv2.COLOR_RGB2BGR)
            phase_str = "EXPERT" if step < switch_step else "MODEL"
            inf_str = "fresh" if (latest_chunk is not None and chunk_consumed_idx == 0) else "reuse"
            # Show stale streak if in model phase and reusing
            stale_n = stale_repeat_counts[-1] if stale_repeat_counts else 0
            stale_str = f" STALE×{stale_n}" if stale_n > 0 else ""
            cv2.putText(bgr, f"Step {step} [{phase_str}]", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(bgr, f"Inf: {worker.n_calls} calls, {inf_str}{stale_str}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("ALIGN Async Sim", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyWindow("ALIGN Async Sim")
                live_view = False

        # 2. Get sim state
        sim_eef = get_sim_eef_pose(obs)
        sim_positions.append(sim_eef)

        # 3. Update sliding windows
        sim_state = np.concatenate([sim_eef, [0.0]]).astype(np.float32)
        pose_buffer.append(sim_state)
        pose_buffer.pop(0)
        frame_buffer.append(current_frame_stack)
        frame_buffer.pop(0)

        # 4. Push state to inference thread (always — so inference always
        #    has the freshest sim state to work with)
        win_states = np.stack(pose_buffer, axis=0).astype(np.float32)  # (K, 7)
        win_frames = np.stack(frame_buffer, axis=0)  # (K, V, H, W, 3)
        f_t = torch.from_numpy(win_frames).unsqueeze(0).to(device)
        s_t = torch.from_numpy(win_states).float().unsqueeze(0).to(device)

        # Overwrite stale state in queue (keep only newest)
        try:
            state_queue.get_nowait()
        except queue.Empty:
            pass
        state_queue.put((f_t, s_t))

        # 5. Check for new action from inference thread
        got_fresh = False
        try:
            chunk_np, call_idx, inf_ms = action_queue.get_nowait()
            latest_chunk = chunk_np
            chunk_consumed_idx = 0
            latest_call_idx = call_idx
            n_model_calls_so_far += 1
            got_fresh = True

            # Push chunk actions into pending buffer based on ensemble mode
            source_call_idx = n_model_calls_so_far - 1
            if ensemble == "none":
                # Push actions 1..H-1 for future steps; action 0 is consumed now
                for k in range(1, action_horizon):
                    target = step + k
                    weight = 1.0
                    pending_actions.append((target, weight, chunk_np[k]))
                    pending_source_call_idx.append(source_call_idx)
            else:
                # "uniform" or "decay": push all K predictions
                for k in range(chunk_size):
                    target = step + k
                    weight = 1.0
                    pending_actions.append((target, weight, chunk_np[k]))
                    pending_source_call_idx.append(source_call_idx)
        except queue.Empty:
            pass  # no fresh chunk — use pending buffer or reuse

        # 6. Build the final action
        if step < switch_step:
            # Phase 1: expert controls, model observes
            final_action = actions[step].copy()
            if noise_std > 0.0 and final_action.shape[0] >= 6:
                final_action[:6] += np.random.normal(
                    0.0, noise_std, size=6
                ).astype(np.float32)
            stored_actions.append(final_action.copy())
            stored_inference_flags.append(False)
            stale_repeat_counts.append(0)
        else:
            # Phase 2: model controls
            a_model = None

            if ensemble == "none":
                # Try to get action from pending buffer (FIFO from previous chunks)
                if pending_actions and pending_actions[0][0] == step:
                    a_model = pending_actions.pop(0)[2]
                    pending_source_call_idx.pop(0)
                    stale_repeat_counts.append(0)
                elif got_fresh and latest_chunk is not None:
                    # Consume first action of fresh chunk
                    a_model = latest_chunk[0]
                    chunk_consumed_idx = 1
                    stale_repeat_counts.append(0)
                elif latest_chunk is not None and chunk_consumed_idx < min(action_horizon, len(latest_chunk)):
                    # Consume next action from current chunk
                    a_model = latest_chunk[chunk_consumed_idx]
                    chunk_consumed_idx += 1
                    stale_repeat_counts.append(0)
                elif latest_chunk is not None:
                    # All H actions consumed, no fresh chunk — reuse last action
                    a_model = latest_chunk[min(action_horizon - 1, len(latest_chunk) - 1)]
                    if stale_repeat_counts and step > 0 and stale_repeat_counts[-1] > 0:
                        stale_repeat_counts.append(stale_repeat_counts[-1] + 1)
                    else:
                        stale_repeat_counts.append(1)
                else:
                    # No model action yet — zero fallback
                    a_model = np.zeros(7, dtype=np.float32)
                    stale_repeat_counts.append(0)
            else:
                # Ensemble averaging (uniform or decay)
                # Prune stale entries first (target <= step or target > step + chunk_size)
                new_pending = []
                new_sources = []
                for (t, w, a), sc in zip(pending_actions, pending_source_call_idx):
                    if t > step and t <= step + chunk_size:
                        new_pending.append((t, w, a))
                        new_sources.append(sc)
                pending_actions = new_pending
                pending_source_call_idx = new_sources

                # Find all entries whose target_step == step
                matching = [
                    (w, a, sc) for (t, w, a), sc in zip(pending_actions, pending_source_call_idx)
                    if t == step
                ]
                if matching:
                    current_call_idx = n_model_calls_so_far
                    if ensemble == "decay":
                        weights_arr = np.array([
                            w * (ensemble_decay ** (current_call_idx - sc))
                            for w, _, sc in matching
                        ])
                    else:  # "uniform"
                        weights_arr = np.array([w for w, _, _ in matching])
                    actions_arr = np.array([a for _, a, _ in matching])  # (E, A)
                    weights_arr /= weights_arr.sum()
                    a_model = (weights_arr[:, None] * actions_arr).sum(axis=0)
                    stale_repeat_counts.append(0)
                elif latest_chunk is not None:
                    # No matching entries — reuse last action
                    a_model = latest_chunk[-1]
                    if stale_repeat_counts and step > 0 and stale_repeat_counts[-1] > 0:
                        stale_repeat_counts.append(stale_repeat_counts[-1] + 1)
                    else:
                        stale_repeat_counts.append(1)
                else:
                    a_model = np.zeros(7, dtype=np.float32)
                    stale_repeat_counts.append(0)

            a_model_scaled = a_model * action_scale
            final_action = a_model_scaled.copy()
            stored_actions.append(a_model_scaled.copy())
            stored_inference_flags.append(got_fresh)

        # Gripper
        if final_action.shape[0] >= 7:
            final_action[6] = 1.0 if final_action[6] <= 0.5 else -1.0
        else:
            final_action[6] = -1.0

        if debug:
            phase = "expert" if step < switch_step else "model"
            fresh_tag = " [FRESH]" if got_fresh else ""
            print(f"Step {step}: phase={phase} action={final_action[:6]}{fresh_tag}")

        # 7. Step sim
        obs, reward, done, info = env.step(final_action)
        sim_eef_after = get_sim_eef_pose(obs)
        sim_positions[-1] = sim_eef_after

        # 8. EEF error
        if expert_poses is not None and step < len(expert_poses):
            expert_eef = expert_poses[step]
            err = float(np.linalg.norm(sim_eef_after[:3] - expert_eef[:3]))
            errors.append(err)
        else:
            errors.append(0.0)

        if done:
            success += 1

        # 9. Throttle to target FPS
        if target_dt > 0:
            elapsed = time.perf_counter() - step_t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    # Stop inference thread
    stop_event.set()
    worker.join(timeout=5.0)

    avg_inf_ms = worker.total_inference_ms / max(worker.n_calls, 1)

    # Stale action statistics (model phase only)
    model_phase_stale = [s for s in stale_repeat_counts if s > 0]
    n_stale_steps = len(model_phase_stale)
    max_stale_streak = max(model_phase_stale) if model_phase_stale else 0
    avg_stale_streak = float(np.mean(model_phase_stale)) if model_phase_stale else 0.0

    return {
        "frames": frames,
        "sim_positions": np.array(sim_positions),
        "errors": np.array(errors),
        "stored_actions": np.array(stored_actions) if stored_actions else np.zeros((0, 7)),
        "inference_flags": stored_inference_flags,
        "stale_repeat_counts": stale_repeat_counts,
        "n_steps": len(frames),
        "switch_step": switch_step,
        "n_inference_calls": worker.n_calls,
        "avg_inference_ms": avg_inf_ms,
        "n_stale_steps": n_stale_steps,
        "max_stale_streak": max_stale_streak,
        "avg_stale_streak": avg_stale_streak,
        "action_magnitude_model": float(np.mean(np.linalg.norm(
            np.array(stored_actions), axis=1
        ))) if stored_actions else 0.0,
        "action_magnitude_dataset": float(np.mean(np.linalg.norm(
            actions[:, :6], axis=1
        ))),
        "success": True if success >= 1 else False,
    }


# ================================================================
# Video saving
# ================================================================

def save_video(frames: List[np.ndarray], path: str, fps: int = 20) -> None:
    """Save a list of (H, W, 3) uint8 frames to an MP4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Async-inference LIBERO eval with real-time live view."
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to intention_best.pt")
    parser.add_argument("--cameras", nargs="+", default=["image", "wrist_image"],
                        help="Camera names (default: wrist_image). "
                             "MUST match training cameras.")
    parser.add_argument("--n-episodes", type=int, default=1,
                        help="Number of episodes to evaluate.")
    parser.add_argument("--val-episodes", type=str, default=None,
                        help="Path to a text file with episode keys to evaluate.")
    parser.add_argument("--noise-std", type=float, default=0.00001,
                        help="Gaussian noise std for noised actions.")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode.")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for results (default: alongside checkpoint).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save MP4 video of episodes (default on).")
    parser.add_argument("--no-video", dest="save_video", action="store_false",
                        help="Skip video saving.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Blend factor (kept for compat, not used in async).")
    parser.add_argument("--switch-at", type=float, default=0.5,
                        help="Fraction of episode after which model takes over. "
                             "0.0 = model from start. 1.0 = expert only.")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Scale factor applied to model actions.")
    parser.add_argument("--action-horizon", type=int, default=1,
                        help="Number of consecutive actions taken from each model "
                             "chunk before re-inferring. With horizon=H and chunk=K, "
                             "the model is called every H steps (each call costs K/H "
                             "effective). Must satisfy 1 <= horizon <= chunk_size. "
                             "Default: 1 (re-infer every step, original behavior).")
    parser.add_argument("--ensemble", type=str, default="none",
                        choices=["none", "uniform", "decay"],
                        help="Temporal ensembling mode for overlapping predictions. "
                             "When 'none' (default), only the first `action_horizon` "
                             "predictions from each chunk are used. When 'uniform' or "
                             "'decay', all `chunk_size` predictions are kept and averaged "
                             "with weights at each step. Average ensemble size is K/H. "
                             "Compatible with --action-horizon.")
    parser.add_argument("--ensemble-decay", type=float, default=0.9,
                        help="Decay factor for ensemble='decay' mode. Weight of each "
                             "entry is (ensemble_decay ** age_in_calls). Newer predictions "
                             "dominate. Default: 0.9.")
    parser.add_argument("--sim-fps", type=float, default=20.0,
                        help="Target sim/render FPS. 0 = no throttle (run as "
                             "fast as possible, like synchronous eval). "
                             "Default: 20.0 (matches control_freq).")
    parser.add_argument("--live-view", action="store_true",
                        help="Open an OpenCV window showing each sim frame in "
                             "real-time. Requires opencv-python and a display.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step action values and inference status.")
    parser.add_argument("--libero-suite", default=None,
                        choices=["libero_spatial", "libero_object",
                                 "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO benchmark suite. Auto-detected from --data.")
    parser.add_argument("--render-size", type=int, default=256,
                        help="Frame size for MuJoCo rendering.")
    parser.add_argument("--no-flip-vertical", action="store_true",
                        help="Skip vertical flip on sim frames.")
    parser.add_argument("--no-flip-horizontal", action="store_true",
                        help="Skip horizontal flip on sim frames.")
    args = parser.parse_args()

    if args.live_view and not CV2_AVAILABLE:
        print("  ⚠️  --live-view requested but opencv-python is not installed.")
        print("      Install with: pip install opencv-python")
        print("      Continuing without live view.")
        args.live_view = False

    if not LIBERO_AVAILABLE:
        print("  ❌ libero not installed. Cannot run sim eval.")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n=== ALIGN v4 Async Inference Evaluation ===")
    print(f"  Data:        {args.data}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Device:      {device}")
    print(f"  Sim FPS:     {args.sim_fps if args.sim_fps > 0 else 'unthrottled'}")
    print(f"  Cameras:     {args.cameras}")
    print(f"  Live view:   {args.live_view}")

    # Load model
    model, cfg = load_intention_model(args.checkpoint, device)
    chunk_size = cfg["chunk_size"]
    print(f"  Chunk (K):   {chunk_size}")
    print(f"  Head:        {cfg.get('head_type', 'mamba')}")

    model_cameras = cfg.get("num_cameras", 1)
    if model_cameras != len(args.cameras):
        print(f"  ⚠️  Camera mismatch: model expects {model_cameras}, got {len(args.cameras)}")

    # Auto-detect suite
    if args.libero_suite is None:
        data_name = Path(args.data).stem
        for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
            if suite in data_name:
                args.libero_suite = suite
                break
        if args.libero_suite is None:
            args.libero_suite = "libero_spatial"
        print(f"  Auto-detected suite: {args.libero_suite}")

    # Load task list
    task_list = _try_load_libero_task_list(args.libero_suite)
    if task_list:
        print(f"  Suite: {args.libero_suite} ({len(task_list)} tasks)")

    # List episodes
    episodes = list_episodes(args.data)
    if not episodes:
        print(f"  No episodes found in {args.data}")
        return

    if args.val_episodes is not None:
        val_keys = set()
        with open(args.val_episodes) as f:
            for line in f:
                val_keys.add(line.strip())
        episodes = [e for e in episodes if e in val_keys]
        print(f"  Using {len(episodes)} val episodes (from {args.val_episodes})")
    else:
        print(f"  Episodes:    {len(episodes)} (all)")

    episodes = episodes[:args.n_episodes]

    # Output dir
    if args.out_dir is None:
        args.out_dir = str(Path(args.checkpoint).parent / "async_eval")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    flip_vertical = not args.no_flip_vertical
    flip_horizontal = not args.no_flip_horizontal

    all_results = []
    for ep_idx, ep_key in enumerate(episodes):
        traj = load_trajectory(args.data, ep_key, args.cameras)
        if traj is None:
            continue

        task_name = traj.get("text", ep_key)
        if task_list and task_name not in task_list:
            best_match = None
            for t in task_list:
                if task_name.lower() in t.lower() or t.lower() in task_name.lower():
                    best_match = t
                    break
            if best_match:
                task_name = best_match

        bddl_path = get_bddl_path(args.libero_suite, task_name)
        if not os.path.exists(bddl_path):
            print(f"\n  [{ep_idx+1}/{len(episodes)}] {ep_key}: BDDL not found: {bddl_path}")
            continue

        print(f"\n  [{ep_idx+1}/{len(episodes)}] {ep_key} → task='{task_name}'")

        # Map dataset camera names to robosuite env camera names.
        # The env always uses robosuite names; get_sim_frame handles the
        # reverse mapping (dataset name → obs key) during rendering.
        ENV_CAMERA_MAP = {
            "image": "agentview",
            "agentview_image": "agentview",
            "agentview": "agentview",
            "wrist_image": "robot0_eye_in_hand",
            "robot0_eye_in_hand": "robot0_eye_in_hand",
            "robot0_eye_in_hand_image": "robot0_eye_in_hand",
        }
        env_cameras = [ENV_CAMERA_MAP.get(c, c) for c in args.cameras]

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                use_camera_obs=True,
                camera_names=env_cameras,
                camera_widths=args.render_size,
                camera_heights=args.render_size,
                reward_shaping=False,
                control_freq=20,
                initialization_noise=None,
            )
        except Exception as e:
            print(f"    ⚠️  Failed to create env: {e}")
            continue

        t0 = time.time()
        result = run_async_episode(
            env=env,
            model=model,
            device=device,
            expert_actions=traj["actions"],
            expert_poses=traj["poses"] if traj["poses"] is not None else None,
            chunk_size=chunk_size,
            max_steps=args.max_steps,
            alpha=args.alpha,
            action_scale=args.action_scale,
            switch_at=args.switch_at,
            render_size=args.render_size,
            cameras=args.cameras,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            noise_std=args.noise_std,
            sim_fps=args.sim_fps,
            live_view=args.live_view,
            debug=args.debug,
            action_horizon=args.action_horizon,
            ensemble=args.ensemble,
            ensemble_decay=args.ensemble_decay,
        )
        wall_time = time.time() - t0

        # Report
        eef_err = float(np.mean(result["errors"])) if len(result["errors"]) > 0 else float("nan")
        n_fresh = sum(result["inference_flags"])
        n_model_steps = result["n_steps"] - result["switch_step"]
        print(f"    Steps:      {result['n_steps']:3d}  ({wall_time:.1f}s wall)")
        print(f"    Switch at:  step {result['switch_step']}")
        print(f"    EEF err:    {eef_err:.4f} m")
        print(f"    Success:    {result['success']}")
        print(f"    Inference:  {result['n_inference_calls']} calls, "
              f"avg {result['avg_inference_ms']:.1f} ms/call")
        print(f"    Fresh actions: {n_fresh}/{n_model_steps} model-phase steps "
              f"({n_fresh/max(n_model_steps,1):.0%})")
        n_stale = result["n_stale_steps"]
        max_stale = result["max_stale_streak"]
        avg_stale = result["avg_stale_streak"]
        print(f"    Stale actions: {n_stale}/{n_model_steps} model-phase steps "
              f"({n_stale/max(n_model_steps,1):.0%})  "
              f"max streak: {max_stale}  avg streak: {avg_stale:.1f}")

        # Save video
        if args.save_video and result["frames"]:
            video_path = str(Path(args.out_dir) / f"{ep_key}_async.mp4")
            save_video(result["frames"], video_path, fps=int(args.sim_fps) if args.sim_fps > 0 else 20)
            print(f"    Video:      {video_path}")

        all_results.append({
            "episode": ep_key,
            "task": task_name,
            "n_steps": result["n_steps"],
            "switch_step": result["switch_step"],
            "eef_error": eef_err,
            "success": result["success"],
            "n_inference_calls": result["n_inference_calls"],
            "avg_inference_ms": result["avg_inference_ms"],
            "fresh_action_ratio": n_fresh / max(n_model_steps, 1),
            "n_stale_steps": result["n_stale_steps"],
            "max_stale_streak": result["max_stale_streak"],
            "avg_stale_streak": result["avg_stale_streak"],
            "stale_action_ratio": n_stale / max(n_model_steps, 1),
            "wall_time_s": wall_time,
            "action_magnitude_model": result["action_magnitude_model"],
            "action_magnitude_dataset": result["action_magnitude_dataset"],
        })

    if CV2_AVAILABLE:
        cv2.destroyAllWindows()

    # Save summary
    if all_results:
        summary_path = str(Path(args.out_dir) / "async_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "config": {
                    "sim_fps": args.sim_fps,
                    "switch_at": args.switch_at,
                    "action_scale": args.action_scale,
                    "action_horizon": args.action_horizon,
                    "ensemble": args.ensemble,
                    "ensemble_decay": args.ensemble_decay,
                    "max_steps": args.max_steps,
                    "noise_std": args.noise_std,
                    "chunk_size": chunk_size,
                },
                "episodes": all_results,
            }, f, indent=2)
        print(f"\n  Summary: {summary_path}")

        # Aggregate
        successes = sum(1 for r in all_results if r["success"])
        avg_err = np.mean([r["eef_error"] for r in all_results if not np.isnan(r["eef_error"])])
        avg_inf = np.mean([r["avg_inference_ms"] for r in all_results])
        avg_fresh = np.mean([r["fresh_action_ratio"] for r in all_results])
        avg_stale = np.mean([r["stale_action_ratio"] for r in all_results])
        worst_streak = max(r["max_stale_streak"] for r in all_results)
        print(f"\n{'='*60}")
        print(f"  Episodes:    {len(all_results)}")
        print(f"  Success:     {successes}/{len(all_results)} ({successes/len(all_results):.0%})")
        print(f"  Avg EEF err: {avg_err:.4f} m")
        print(f"  Avg inf ms:  {avg_inf:.1f} ms/call")
        print(f"  Fresh ratio: {avg_fresh:.0%} (model-phase steps with fresh action)")
        print(f"  Stale ratio: {avg_stale:.0%} (model-phase steps reusing exhausted action)")
        print(f"  Worst stale streak: {worst_streak} consecutive steps")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()