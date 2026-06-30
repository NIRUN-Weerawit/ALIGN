#!/usr/bin/env python3
"""Verify how OSC_POSE actions map to EEF movements in the simulator.

Sends specific action commands and measures the actual EEF displacement.
This confirms:
  1. The units of the action (m vs scaled)
  2. The scaling factor between action and actual movement
  3. Whether position and rotation are scaled differently

Usage:
    python eval/verify_action_to_pose_mapping.py \
        --suite libero_spatial \
        --task "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate" \
        --action "[1.0, 0, 0, 0, 0, 0, 1.0]" \
        --steps 5

The script:
  - Creates a LIBERO env for the given task
  - Resets the env to get an initial pose
  - Sends the given action N times (or zero actions for baseline)
  - Records EEF pose before and after each step
  - Reports actual displacement per axis

The expected behavior (from robosuite OSC_POSE defaults):
  - Position action scaled by 0.05 (max 5cm per step)
  - Rotation action scaled by 0.5 (max 0.5 rad per step)
  - action = [1, 0, 0, 0, 0, 0] should move EEF ~5cm in x
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MuJoCo EGL corrupts PyTorch's cuDNN state.
os.environ.setdefault("MUJOCO_GPU_RENDERING", "0")

try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
except ImportError:
    raise ImportError("libero not installed. Run: pip install libero")

from scipy.spatial.transform import Rotation


# ================================================================
# Task mapping (loaded from libero's task map)
# ================================================================

LIBERO_TASK_MAP = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
    "libero_90": "libero_90",
}

SUITE_TASK_LISTS = {}

_benchmark_file = os.path.join(
    os.path.dirname(__import__("libero").__file__),
    "libero", "benchmark", "libero_suite_task_map.py"
)
if os.path.exists(_benchmark_file):
    import importlib.util as _util
    _spec = _util.spec_from_file_location("_libero_task_map", _benchmark_file)
    if _spec and _spec.loader:
        _mod = _util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for suite_key in LIBERO_TASK_MAP:
            if hasattr(_mod, 'libero_task_map') and suite_key in _mod.libero_task_map:
                SUITE_TASK_LISTS[suite_key] = _mod.libero_task_map[suite_key]


def get_bddl_path(suite_name: str, task_name: str) -> str:
    return os.path.join(get_libero_path("bddl_files"), suite_name, f"{task_name}.bddl")


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat).as_rotvec()


def axisangle_to_quat(axisangle: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(axisangle).as_quat()


# ================================================================
# EEF measurement helpers
# ================================================================

def get_eef_state(env) -> dict:
    """Return the current EEF state: position (xyz) and orientation (quat)."""
    obs = env.env._get_observations() if hasattr(env, "env") else env._get_observations()
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0]))
    if isinstance(eef_pos, torch.Tensor):
        eef_pos = eef_pos.cpu().numpy()
    if isinstance(eef_quat, torch.Tensor):
        eef_quat = eef_quat.cpu().numpy()
    return {"pos": eef_pos.copy(), "quat": eef_quat.copy()}


def compute_displacement(p_before: dict, p_after: dict) -> dict:
    """Compute the displacement between two EEF states.

    Returns:
      pos_disp: (3,) xyz displacement in meters (world frame)
      pos_disp_local: (3,) xyz displacement in EEF-local frame
      ori_disp: (3,) axis-angle rotation displacement in radians
    """
    # Position: world frame
    pos_disp = p_after["pos"] - p_before["pos"]

    # Position: local frame (rotate world displacement into EEF frame)
    # The EEF frame is defined by the quaternion. We need the inverse
    # rotation that maps world → EEF-local.
    q = p_after["quat"]  # [w, x, y, z] or [x, y, z, w] depending on convention
    # Use scipy which expects (x, y, z, w)
    if q[0] > q[3] if abs(q[0]) < 0.5 else False:
        # If first element is small, probably (x, y, z, w) order
        q_xyzw = q
    else:
        # Assume (w, x, y, z) order, convert to (x, y, z, w)
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    rot_world_to_local = Rotation.from_quat(q_xyzw).inv()
    pos_disp_local = rot_world_to_local.apply(pos_disp)

    # Orientation: convert quaternions to axis-angle and compute difference
    rot_before = Rotation.from_quat(q_xyzw)  # NOTE: this is the AFTER rotation
    # Better: use both quaternions
    # Actually let's use scipy's proper difference
    q_before_xyzw = p_before["quat"]
    if abs(q_before_xyzw[0]) < 0.5:
        q_before_xyzw = np.array([q_before_xyzw[1], q_before_xyzw[2], q_before_xyzw[3], q_before_xyzw[0]])
    rot_before = Rotation.from_quat(q_before_xyzw)
    rot_after = Rotation.from_quat(q_xyzw)

    # Relative rotation: after * before^-1
    rot_relative = rot_after * rot_before.inv()
    ori_disp = rot_relative.as_rotvec()

    return {
        "pos_world": pos_disp,
        "pos_local": pos_disp_local,
        "ori": ori_disp,
    }


# ================================================================
# Main verification
# ================================================================

def verify_action_to_pose_mapping(
    suite_name: str,
    task_name: str,
    action: np.ndarray,
    n_steps: int = 1,
    n_trials: int = 1,
    render_size: int = 256,
):
    """Send `action` to the env `n_steps` times and measure EEF displacement.

    Args:
        suite_name: e.g., "libero_spatial"
        task_name: BDDL task name
        action: 7-dim action vector (xyz + axis-angle + gripper)
        n_steps: number of timesteps to apply the same action
        n_trials: number of independent resets (averages over trials)
    """
    print(f"\n{'='*60}")
    print(f"=== Verifying Action → EEF Pose Mapping ===")
    print(f"{'='*60}")
    print(f"  Suite:     {suite_name}")
    print(f"  Task:      {task_name[:60]}...")
    print(f"  Action:    {action}")
    print(f"  Steps:     {n_steps}")
    print(f"  Trials:    {n_trials}")
    print(f"  Action xyz magnitude:    {np.linalg.norm(action[:3]):.4f}")
    print(f"  Action rot magnitude:    {np.linalg.norm(action[3:6]):.4f}")

    # Setup env
    bddl_path = get_bddl_path(suite_name, task_name)
    if not os.path.exists(bddl_path):
        print(f"  ERROR: BDDL not found: {bddl_path}")
        return None

    print(f"  BDDL:      {bddl_path}")

    # Run multiple trials for robustness
    all_pos_disp_world = []
    all_pos_disp_local = []
    all_ori_disp = []

    for trial in range(n_trials):
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            use_camera_obs=False,  # Don't need cameras for EEF measurement
            camera_names=["agentview"],
            camera_widths=render_size,
            camera_heights=render_size,
            reward_shaping=True,
            control_freq=20,
            initialization_noise=None,
        )

        # Reset env
        env.reset()
        # Take a few zero-action steps to settle
        zero_action = np.zeros(7)
        zero_action[6] = 1.0  # gripper open
        for _ in range(3):
            obs, _, _, _ = env.step(zero_action)

        # Measure initial EEF pose
        p_before = get_eef_state(env)

        # Apply the action `n_steps` times
        for step in range(n_steps):
            obs, _, _, _ = env.step(action)
            p_after = get_eef_state(env)
            disp = compute_displacement(p_before, p_after)
            all_pos_disp_world.append(disp["pos_world"])
            all_pos_disp_local.append(disp["pos_local"])
            all_ori_disp.append(disp["ori"])
            # Update for next step (so we can see step-by-step displacement)
            p_before = p_after

        env.close()

    # Aggregate results
    pos_world = np.array(all_pos_disp_world)
    pos_local = np.array(all_pos_disp_local)
    ori = np.array(all_ori_disp)

    # Per-step average
    avg_pos_world = pos_world.mean(axis=0)
    avg_pos_local = pos_local.mean(axis=0)
    avg_ori = ori.mean(axis=0)

    print(f"\n{'='*60}")
    print(f"=== RESULTS (averaged over {n_trials} trials, {n_steps} steps each) ===")
    print(f"{'='*60}")

    print(f"\nPosition displacement per step (world frame):")
    print(f"  World xyz: dx={avg_pos_world[0]:+.6f}, dy={avg_pos_world[1]:+.6f}, dz={avg_pos_world[2]:+.6f}")
    print(f"  World magnitude: {np.linalg.norm(avg_pos_world):.6f} m")

    print(f"\nPosition displacement per step (EEF-local frame):")
    print(f"  Local xyz: dx={avg_pos_local[0]:+.6f}, dy={avg_pos_local[1]:+.6f}, dz={avg_pos_local[2]:+.6f}")
    print(f"  Local magnitude: {np.linalg.norm(avg_pos_local):.6f} m")

    print(f"\nRotation displacement per step (axis-angle):")
    print(f"  Axis-angle: dax={avg_ori[0]:+.6f}, day={avg_ori[1]:+.6f}, daz={avg_ori[2]:+.6f}")
    print(f"  Magnitude: {np.linalg.norm(avg_ori):.6f} rad")

    print(f"\n=== Inferred scaling factor ===")
    print(f"  Position: action[xyz] = {action[:3]}, displacement = {avg_pos_local[:3]}")
    for i, axis in enumerate(["x", "y", "z"]):
        if abs(action[i]) > 1e-6:
            inferred_scale = avg_pos_local[i] / action[i]
            print(f"    {axis}: scale = displacement/action = {inferred_scale:+.4f} m/unit")
    print(f"  Rotation: action[ori] = {action[3:6]}, displacement = {avg_ori}")
    for i, axis in enumerate(["ax", "ay", "az"]):
        if abs(action[3 + i]) > 1e-6:
            inferred_scale = avg_ori[i] / action[3 + i]
            print(f"    {axis}: scale = displacement/action = {inferred_scale:+.4f} rad/unit")

    return {
        "action": action.tolist(),
        "pos_world_avg": avg_pos_world.tolist(),
        "pos_local_avg": avg_pos_local.tolist(),
        "ori_avg": avg_ori.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify how OSC_POSE actions map to EEF movements in the simulator"
    )
    parser.add_argument("--suite", default="libero_spatial",
                        choices=list(LIBERO_TASK_MAP.keys()))
    parser.add_argument("--task", default=None,
                        help="BDDL task name (default: first task in suite)")
    parser.add_argument("--action", default="[1.0, 0, 0, 0, 0, 0, 1.0]",
                        help="Action to send as JSON list, e.g. '[1.0, 0, 0, 0, 0, 0, 1.0]'")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of timesteps to apply the action")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of independent trials")
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--all-axes", action="store_true",
                        help="Test all 6 axes (xyz position + axis-angle rotation)")
    args = parser.parse_args()

    # Pick task
    if args.task is None:
        if args.suite not in SUITE_TASK_LISTS:
            print(f"ERROR: No task list for {args.suite}")
            return
        task_name = SUITE_TASK_LISTS[args.suite][0]
        print(f"Using first task in {args.suite}: {task_name}")
    else:
        task_name = args.task

    # Run tests
    if args.all_axes:
        # Test each axis individually with magnitude 1.0
        test_actions = [
            ("x+", np.array([1.0, 0, 0, 0, 0, 0, 1.0])),
            ("x-", np.array([-1.0, 0, 0, 0, 0, 0, 1.0])),
            ("y+", np.array([0, 1.0, 0, 0, 0, 0, 1.0])),
            ("y-", np.array([0, -1.0, 0, 0, 0, 0, 1.0])),
            ("z+", np.array([0, 0, 1.0, 0, 0, 0, 1.0])),
            ("z-", np.array([0, 0, -1.0, 0, 0, 0, 1.0])),
            ("ax+", np.array([0, 0, 0, 1.0, 0, 0, 1.0])),
            ("ay+", np.array([0, 0, 0, 0, 1.0, 0, 1.0])),
            ("az+", np.array([0, 0, 0, 0, 0, 1.0, 1.0])),
        ]
        results = []
        for name, action in test_actions:
            print(f"\n{'#'*60}")
            print(f"# Testing axis: {name}, action = {action}")
            print(f"{'#'*60}")
            r = verify_action_to_pose_mapping(
                suite_name=args.suite,
                task_name=task_name,
                action=action,
                n_steps=args.steps,
                n_trials=args.trials,
                render_size=args.render_size,
            )
            if r is not None:
                r["axis"] = name
                results.append(r)

        # Summary
        print(f"\n{'='*60}")
        print(f"=== SUMMARY: action → EEF displacement per unit action ===")
        print(f"{'='*60}")
        for r in results:
            if r["axis"].endswith("+") or r["axis"].endswith("-"):
                axis_idx = {"x": 0, "y": 1, "z": 2, "ax": 0, "ay": 1, "az": 2}[r["axis"][:-1]]
                if r["axis"][-1] == "-":
                    pass
                if "x" in r["axis"] or "y" in r["axis"] or "z" in r["axis"]:
                    pos = r["pos_local_avg"][axis_idx]
                    print(f"  {r['axis']}: pos_local[{axis_idx}] = {pos:+.6f} m per unit action")
                else:
                    ori = r["ori_avg"][axis_idx]
                    print(f"  {r['axis']}: ori[{axis_idx}] = {ori:+.6f} rad per unit action")
    else:
        # Custom action
        action = np.array(json.loads(args.action), dtype=np.float64)
        if len(action) == 6:
            action = np.append(action, 1.0)  # add gripper
        verify_action_to_pose_mapping(
            suite_name=args.suite,
            task_name=task_name,
            action=action,
            n_steps=args.steps,
            n_trials=args.trials,
            render_size=args.render_size,
        )


if __name__ == "__main__":
    main()