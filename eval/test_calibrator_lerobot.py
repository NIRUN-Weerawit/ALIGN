#!/usr/bin/env python3
"""
Smoke test for DeploymentCalibrator using LIBERO simulation.

The real test of the calibrator's math is in deployment_calibrator.py
(an end-to-end pass with synthetic data). This script verifies the
integration:

  1. The LiberoRobot wrapper correctly implements the get_current_pose
     and set_target_pose interface that DeploymentCalibrator expects
  2. The calibrator can run all 8 cardinal motions in the sim without
     crashing
  3. The calibration completes and produces a sensible result

Quantitative verification of the recovery logic is done in
deployment_calibrator.py (the `if __name__ == '__main__'` block) using
synthetic data. Replicating the *exact* perturbation in LIBERO is
non-trivial because the OSC_POSE controller has its own internal
gain and we can't precisely command an EEF delta without IK. So we
keep the LIBERO test as a smoke/integration check.

Usage:
    PYTHONNOUSERSITE=1 /home/ucluser/miniconda3/envs/align/bin/python \\
        eval/test_calibrator_lerobot.py [--suite libero_10]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("MUJOCO_GPU_RENDERING", "0")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ALIGN_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ALIGN_ROOT))
sys.path.insert(0, str(ALIGN_ROOT / "inference"))

try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
except ImportError:
    print("ERROR: libero not installed. Run: pip install libero")
    sys.exit(1)

from deployment_calibrator import (
    DeploymentCalibrator,
    CalibrationResult,
    CARDINAL_MOTIONS,
)


class LiberoRobot:
    """Thin adapter that wraps LIBERO env as a calibrator-compatible robot.

    The pose stream is reported in the env's native frame (no perturbation).
    The calibrator will compute scale=1, offset=0, rotation=identity, which
    is the expected passthrough case when the env matches the training frame.
    """

    def __init__(self, env):
        self.env = env
        self._latest_obs: Optional[dict] = None
        try:
            self._latest_obs = env.env._get_observations()
        except AttributeError:
            self._latest_obs = env._get_observations()

    def _get_canonical_pose(self) -> np.ndarray:
        try:
            obs = self.env.env._get_observations()
        except AttributeError:
            obs = self.env._get_observations()
        self._latest_obs = obs
        eef_pos = np.array(obs["robot0_eef_pos"]).flatten()[:3]
        eef_quat = np.array(obs["robot0_eef_quat"]).flatten()[:4]
        from scipy.spatial.transform import Rotation
        eef_aa = Rotation.from_quat(eef_quat).as_rotvec()
        return np.concatenate([eef_pos, eef_aa]).astype(np.float32)

    def get_current_pose(self) -> np.ndarray:
        return self._get_canonical_pose()

    def set_target_pose(self, target_pose: np.ndarray) -> None:
        """Move the EEF toward target_pose using repeated OSC_POSE steps.
        Uses an empirical gain of ~0.04 (action unit → meters)."""
        eef_action_gain = 0.04
        action = np.zeros(7)
        for _ in range(30):
            obs = self._latest_obs
            if obs is None:
                try:
                    obs = self.env.env._get_observations()
                except AttributeError:
                    obs = self.env._get_observations()
                self._latest_obs = obs
            current_pos = np.array(obs["robot0_eef_pos"]).flatten()[:3]
            remaining = target_pose[:3] - current_pos
            if np.linalg.norm(remaining) < 0.001:
                break
            action[:3] = remaining / eef_action_gain
            action[6] = -1.0
            self.env.step(action)
            try:
                obs = self.env.env._get_observations()
            except AttributeError:
                obs = self.env._get_observations()
            self._latest_obs = obs

    def get_camera_frame(self) -> np.ndarray:
        if self._latest_obs is None:
            try:
                obs = self.env.env._get_observations()
            except AttributeError:
                obs = self.env._get_observations()
        else:
            obs = self._latest_obs
        return obs["robot0_eye_in_hand"][::-1].copy()


def find_first_libero_task(suite_name: str = "libero_10"):
    try:
        bddl_dir = os.path.join(get_libero_path("bddl_files"), suite_name)
        if not os.path.isdir(bddl_dir):
            return None, None
        for f in sorted(os.listdir(bddl_dir)):
            if f.endswith(".bddl"):
                return suite_name, f[:-5]
        return None, None
    except Exception as e:
        print(f"Could not find BDDL tasks: {e}")
        return None, None


def run_smoke_test(suite_name: str = "libero_10", expected_amplitude_pos: float = 0.05):
    print("=" * 60)
    print("ALIGN DeploymentCalibrator — LIBERO Sim Smoke Test")
    print("=" * 60)

    suite, task_name = find_first_libero_task(suite_name)
    if task_name is None:
        print(f"ERROR: No tasks found for suite '{suite_name}'")
        return False
    print(f"Using task: {task_name}")

    bddl_path = os.path.join(get_libero_path("bddl_files"), suite, f"{task_name}.bddl")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_widths=224,
        camera_heights=224,
        reward_shaping=True,
        control_freq=20,
        initialization_noise=None,
    )
    env.reset()
    print("✓ LIBERO env initialized")

    robot = LiberoRobot(env)
    print("✓ LiberoRobot wrapper created")
    print()

    # Run calibration
    cal = DeploymentCalibrator(
        expected_amplitude_pos=expected_amplitude_pos,
        expected_amplitude_rot=0.30,
        settle_time_s=0.4,
    )
    cal.start(robot, source_label=f"libero_{task_name}_smoke")

    # Use a sim-stepping run_motion instead of time.sleep
    def run_motion_sim(self, direction):
        label, pos_dir, rot_dir = direction
        target = self.start_pose.copy().astype(np.float64)
        target[:3] += np.array(pos_dir) * self.expected_pos_amp
        target[3:6] += np.array(rot_dir) * self.expected_rot_amp
        for _ in range(8):  # 8 sim steps per motion
            self.robot_controller.set_target_pose(target)
        final_pose = self.robot_controller.get_current_pose()
        observed_delta = final_pose - self.start_pose
        self.observations.append((label, observed_delta))
        for _ in range(8):
            self.robot_controller.set_target_pose(self.start_pose)

    DeploymentCalibrator.run_motion = run_motion_sim

    print("Running 8 cardinal motions in LIBERO sim (no perturbation, expect passthrough)...")
    t0 = time.time()
    result = cal.run_full_calibration()
    elapsed = time.time() - t0
    print(f"✓ Calibration complete in {elapsed:.1f}s")
    print()

    # Verify passthrough: with no perturbation, recovered factors should be
    # near-identity. Some drift is expected because of OSC_POSE controller
    # nonlinearity.
    print("=" * 60)
    print("Passthrough Verification (no perturbation → should be near-identity)")
    print("=" * 60)
    print(f"position_scale  expected: [1.0, 1.0, 1.0]")
    print(f"position_scale  got:      {result.position_scale}")
    print(f"position_offset got:      {result.position_offset.round(4).tolist()}")
    print(f"position_rotation:\n{result.position_rotation}")
    print(f"orientation_scale: {result.orientation_scale}")
    print(f"confidence:        {result.confidence_per_axis}")
    print()

    scale_err = np.abs(result.position_scale - 1.0).max()
    offset_err = np.abs(result.position_offset).max()
    # Off-diagonal entries of rotation should be near 0
    rot_err = (np.abs(result.position_rotation - np.eye(3)) *
               (1.0 - np.eye(3))).max()
    print(f"Max errors: scale={scale_err:.4f}, offset={offset_err:.4f}, rotation={rot_err:.4f}")

    # Tolerances: scale within 0.2, offset within 0.02m, rotation off-diag within 0.2
    scale_ok = scale_err < 0.2
    offset_ok = offset_err < 0.02
    rot_ok = rot_err < 0.2

    print()
    print(f"Scale near 1.0:   {'✓ PASS' if scale_ok else '✗ FAIL'}")
    print(f"Offset near 0:    {'✓ PASS' if offset_ok else '✗ FAIL'}")
    print(f"Rotation identity:{' ✓ PASS' if rot_ok else ' ✗ FAIL'}")

    # Save
    out_path = ALIGN_ROOT / "checkpoints" / "calibration_smoke.json"
    result.save(str(out_path))
    print(f"\n✓ Saved calibration to {out_path}")

    try:
        env.close()
    except Exception:
        pass

    overall = scale_ok and offset_ok and rot_ok
    print()
    print("=" * 60)
    print(f"Overall: {'✓ PASS' if overall else '✗ FAIL'}")
    print("=" * 60)
    return overall


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="libero_10", help="LIBERO suite")
    p.add_argument("--amplitude", type=float, default=0.05, help="Calibration motion amplitude (m)")
    args = p.parse_args()
    ok = run_smoke_test(suite_name=args.suite, expected_amplitude_pos=args.amplitude)
    sys.exit(0 if ok else 1)
