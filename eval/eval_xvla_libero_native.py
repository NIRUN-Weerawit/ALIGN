#!/usr/bin/env python3
"""Native absolute-control XVLA evaluation on ALIGN's saved episode selection.

Uses the author client's control, chunk, settling and proprioception contracts,
but keeps ALIGN's episode/task selection and deterministic reset seeds. This is
not the official fixed-initial-state benchmark. No demonstration prefix is used.
See scripts/README_xvla_libero.md for runtime setup and a complete command.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.xvla_libero_adapter import XVLALiberoAdapter, rotation6d_matrix


def episode_seed(base_seed, key):
    """Same episode-order-independent seed as eval_libero_v4_trajectory.py."""
    offset = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
    return (int(base_seed) + offset) % (2**31)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_instruction(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def load_cases(data, episode_file, tasks, seed=42, limit=None, reference=None):
    keys = sorted(line.strip() for line in Path(episode_file).read_text().splitlines() if line.strip())
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Episode file must contain nonempty, unique episode keys')
    if limit is not None and limit < 1:
        raise ValueError('--n-episodes must be positive')
    task_map = {}
    for i, task in enumerate(tasks):
        name = normalized_instruction(task.language)
        if name in task_map:
            raise ValueError(f'Ambiguous benchmark task language: {task.language}')
        task_map[name] = (i, task)
    prior = {}
    if reference:
        rows = json.loads(Path(reference).read_text())['episodes']
        prior = {row['episode']: row for row in rows}
        if len(prior) != len(rows) or set(prior) != set(keys):
            raise ValueError('Reference result must contain exactly the selected episode set')
    cases = []
    with h5py.File(data, 'r') as h5:
        for key in keys:
            if key not in h5:
                raise ValueError(f'Dataset is missing selected episode {key}')
            texts = json.loads(h5[key]['texts'][()])
            if not isinstance(texts, list) or not texts or not isinstance(texts[0], str):
                raise ValueError(f'{key}: expected a nonempty JSON instruction list')
            match = task_map.get(normalized_instruction(texts[0]))
            if match is None:
                raise ValueError(f'{key}: no exact benchmark task match for {texts[0]!r}')
            task_id, task = match
            row = prior.get(key)
            if row and normalized_instruction(row['task_name']) != normalized_instruction(task.language):
                raise ValueError(f'{key}: reference task differs from dataset task')
            case_seed = int(row['episode_seed']) if row else episode_seed(seed, key)
            cases.append(dict(episode=key, task_id=task_id, task_name=task.language,
                              dataset_instruction=texts[0], episode_seed=case_seed))
    return cases[:limit] if limit is not None else cases


def absolute_sim_action(action):
    """20-D predicted absolute EE6D -> native OSC XYZ/axis-angle/gripper.

    No subtraction of the measured pose, delta scaling, or translation clipping.
    Author client's gripper rule is strictly > 0.5 for closed (+1).
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (20,) or not np.isfinite(action).all():
        raise ValueError('Expected a finite 20-D XVLA action')
    rotvec = Rotation.from_matrix(rotation6d_matrix(action[3:9])).as_rotvec()
    return np.r_[action[:3], rotvec, 1. if action[9] > .5 else -1.].astype(np.float32)


def set_absolute_control(env, enabled):
    for robot in env.robots:
        controller = robot.controller
        if controller.control_dim != 6 or not hasattr(controller, 'use_delta'):
            raise ValueError('Native XVLA requires a 6-D OSC pose controller')
        controller.use_delta = not enabled


def reset_case(env, seed, *, initial_state=None, settle_steps=10):
    """Settle in delta mode, then enable native absolute control like the author."""
    set_absolute_control(env, False)
    seed_all(seed)
    env.seed(seed)
    obs = env.reset()
    if initial_state is not None:
        initial_state = np.asarray(initial_state)
        if initial_state.ndim != 1 or not initial_state.size or not np.isfinite(initial_state).all():
            raise ValueError('Initial simulator state must be a finite flat vector')
        obs = env.set_init_state(initial_state.copy())
    reset_state = np.asarray(env.get_sim_state()).copy()
    for _ in range(settle_steps):
        obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    set_absolute_control(env, True)
    return obs, reset_state


class NativeChunkPolicy:
    def __init__(self, adapter, proprio_mode='author'):
        if proprio_mode not in ('author', 'measured'):
            raise ValueError('Unknown proprioception mode')
        self.adapter, self.proprio_mode = adapter, proprio_mode
        self.reset()

    def reset(self):
        self.adapter.reset_episode()
        self.queue = deque()
        self.proprio = None
        self.chunks = []
        self.calls = 0

    def step(self, obs, task_text, env):
        if not self.queue:
            # The first call uses the measured pose. Subsequent author-client
            # calls use the last predicted target, with both gripper slots zero.
            if self.proprio_mode == 'author' and self.proprio is None:
                controller = env.robots[0].controller
                controller.update()
                matrix = np.asarray(controller.ee_ori_mat)
                self.proprio = np.r_[controller.ee_pos, matrix[:, 0], matrix[:, 1],
                                     0., np.zeros(10)].astype(np.float32)
            actions = self.adapter.predict_sim_action_chunk(
                obs, task_text=task_text, env=env,
                proprio=self.proprio if self.proprio_mode == 'author' else None)
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            actions = np.asarray(actions)
            expected = (1, self.adapter.chunk_size, 20)
            if actions.shape != expected or not np.isfinite(actions).all():
                raise ValueError(f'Invalid action chunk: expected {expected}, got {actions.shape}')
            chunk = actions[0].copy()
            self.chunks.append(chunk)
            self.calls += 1
            self.proprio = np.zeros(20, dtype=np.float32)
            self.proprio[:9] = chunk[-1, :9]
            self.queue.extend(absolute_sim_action(action) for action in chunk)
        return self.queue.popleft()


def rollout(env, policy, obs, task_text, max_steps, frame_callback=None):
    actions, positions = [], []
    success = bool(env.check_success())
    initial_success = success
    done = False
    started = time.perf_counter()
    if frame_callback:
        frame_callback(obs)
    for _ in range(max_steps):
        if success:
            break
        positions.append(np.asarray(obs['robot0_eef_pos']).copy())
        action = policy.step(obs, task_text, env)
        obs, reward, done, _ = env.step(action)
        actions.append(action.copy())
        if frame_callback:
            frame_callback(obs)
        success = bool(env.check_success())  # timeout/done alone is never success
        if success or done:
            break
    outcome = dict(success=success, success_at_start=initial_success, n_steps=len(actions),
                   policy_calls=policy.calls, wall_seconds=time.perf_counter() - started,
                   termination=('success' if success else 'environment_done' if done else 'step_limit'))
    trace = dict(actions=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
                 pre_action_positions=np.asarray(positions, dtype=np.float32).reshape(-1, 3))
    return outcome, trace


def summarize(rows):
    tasks = {}
    for row in rows:
        counts = tasks.setdefault(row['task_name'], dict(successes=0, total=0))
        counts['total'] += 1
        counts['successes'] += int(row['success'])
    for counts in tasks.values():
        counts['success_rate'] = counts['successes'] / counts['total']
    return dict(successes=sum(int(row['success']) for row in rows), total=len(rows),
                per_task=dict(sorted(tasks.items())))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='lerobot/xvla-libero')
    p.add_argument('--hf-cache-dir', default=os.environ.get('HF_HUB_CACHE'))
    p.add_argument('--tokenizer', help='Offline BART tokenizer snapshot')
    p.add_argument('--data', default='data/libero_goal.h5')
    p.add_argument('--val-episodes', default='checkpoints/piper_v4_ablation_val_episodes.txt')
    p.add_argument('--libero-suite', default='libero_goal',
                   choices=['libero_goal', 'libero_spatial', 'libero_object', 'libero_10'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--reference-results', help='Use/check episode seeds and task labels from a prior 44-case JSON')
    p.add_argument('--initial-states', help='Episode-keyed NPZ of pre-settling reset states; otherwise HDF state or seeded reset')
    p.add_argument('--n-episodes', type=int, help='Optional smoke-test limit; default evaluates the entire saved list')
    p.add_argument('--max-steps', type=int, help='Default: 800, or 900 for libero_10; excludes settling')
    p.add_argument('--settle-steps', type=int, default=10)
    p.add_argument('--proprio-mode', choices=['author', 'measured'], default='author',
                   help='Author: previous chunk final target; measured: LeRobot closed-loop pose')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'])
    p.add_argument('--out-dir', default='eval/libero_results/xvla_native_goal')
    p.add_argument('--save-video', action='store_true')
    p.add_argument('--validate-only', action='store_true', help='Validate episode selection and task mapping without loading weights')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_steps is None:
        args.max_steps = 900 if args.libero_suite == 'libero_10' else 800
    if args.max_steps < 1 or args.settle_steps < 0:
        raise ValueError('max-steps must be positive and settle-steps nonnegative')
    # Set before importing MuJoCo/robosuite. Respect explicitly supplied settings.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[args.libero_suite](task_order_index=0)
    cases = load_cases(args.data, args.val_episodes, suite.tasks, args.seed,
                      args.n_episodes, args.reference_results)
    plan = dict(protocol='XVLA native absolute control on ALIGN saved episodes',
                official_benchmark=False, evaluation_args=vars(args), cases=cases,
                episode_file_sha256=hashlib.sha256(Path(args.val_episodes).read_bytes()).hexdigest())
    print(f'Selected {len(cases)} episodes across {len(set(c["task_id"] for c in cases))} tasks', flush=True)
    for name, counts in summarize([dict(c, success=False) for c in cases])['per_task'].items():
        print(f'  {name}: {counts["total"]} episodes', flush=True)
    if args.validate_only:
        print(json.dumps(plan, indent=2))
        return
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'xvla_native_eval.json'
    if result_path.exists():
        raise FileExistsError(f'{result_path} already exists; choose a new --out-dir')
    # Validate optional reset files before loading the large model.
    initial_states = {}
    with h5py.File(args.data, 'r') as h5:
        for case in cases:
            key = case['episode']
            if 'initial_sim_state' in h5[key]:
                initial_states[key] = np.asarray(h5[key]['initial_sim_state'][()])
    state_source = 'HDF demonstration state when present, otherwise seeded random reset'
    if args.initial_states:
        with np.load(args.initial_states, allow_pickle=False) as states:
            for case in cases:
                key = case['episode']
                if key not in states:
                    raise ValueError(f'Initial-state file is missing {key}')
                initial_states[key] = states[key].copy()
        state_source = 'episode-keyed pre-settling initial-state file'
    adapter = XVLALiberoAdapter(args.checkpoint, torch.device(args.device),
        cache_dir=args.hf_cache_dir, tokenizer_path=args.tokenizer, dtype=args.dtype)
    metadata = dict(adapter.metadata)
    metadata.update(action_contract='native absolute XYZ/axis-angle OSC; gripper +1 closed/-1 open',
                    proprio_mode=args.proprio_mode, executed_chunk_size=adapter.chunk_size)
    report = dict(plan, policy_metadata=metadata, initial_state_source=state_source,
                  status='running', episodes=[], summary=summarize([]))
    write_json(result_path, report)
    from libero.libero.envs import OffScreenRenderEnv
    saved_states = {}
    for index, case in enumerate(cases):
        key = case['episode']
        env = writer = None
        try:
            seed_all(case['episode_seed'])
            task = suite.get_task(case['task_id'])
            env = OffScreenRenderEnv(
                bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                camera_heights=256, camera_widths=256)
            obs, reset_state = reset_case(env, case['episode_seed'],
                initial_state=initial_states.get(key), settle_steps=args.settle_steps)
            saved_states[key] = reset_state
            np.savez_compressed(output / 'initial_states.npz', **saved_states)
            policy = NativeChunkPolicy(adapter, args.proprio_mode)
            if args.save_video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(output / f'{key}.mp4', fps=20)
            def frame_callback(observation):
                writer.append_data(np.concatenate([
                    observation['agentview_image'][::-1, ::-1],
                    observation['robot0_eye_in_hand_image']], axis=1))
            started = time.perf_counter()
            outcome, trace = rollout(env, policy, obs, task.language, args.max_steps,
                                     frame_callback if writer else None)
            # Save predictions as absolute EE6D for action/control audits.
            np.savez_compressed(output / f'{key}_chunks.npz',
                chunks=np.asarray(policy.chunks, dtype=np.float32).reshape(-1, adapter.chunk_size, 20),
                **trace)
            row = dict(case, **outcome, episode_wall_seconds=time.perf_counter() - started,
                       initial_state_source=('provided state' if key in initial_states else 'seeded random reset'),
                       initial_state_sha256=hashlib.sha256(reset_state.tobytes()).hexdigest(),
                       controller='OSC_POSE absolute', control_freq=20)
            report['episodes'].append(row)
            report['summary'] = summarize(report['episodes'])
            report['success_rate'] = f'{report["summary"]["successes"]}/{report["summary"]["total"]}'
            write_json(result_path, report)
            print(f'[{index+1}/{len(cases)}] {key} {task.language}: '
                  f'{"SUCCESS" if row["success"] else "FAIL"}, {row["n_steps"]} steps, '
                  f'{row["policy_calls"]} chunks; total {report["success_rate"]}', flush=True)
        except BaseException as exc:
            report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                          error=dict(episode=key, type=type(exc).__name__, message=str(exc)))
            write_json(result_path, report)
            raise
        finally:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
    report['status'] = 'complete'
    write_json(result_path, report)
    print('\nSuccess by task:', flush=True)
    for name, counts in report['summary']['per_task'].items():
        print(f'  {name}: {counts["successes"]}/{counts["total"]}', flush=True)
    print(f'TOTAL: {report["success_rate"]}\nResults: {result_path}', flush=True)


if __name__ == '__main__':
    main()
