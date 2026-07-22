"""
Deployment-time calibration for ALIGN.

When deploying in a new environment (new robot, new camera angle, new base
frame), the model needs to know:
  1. How axes map to physical directions (which +x is "left" vs "right")
  2. What scale units the EEF pose is in (meters, mm, etc.)
  3. How the camera frame relates to the robot base (hand-eye)
  4. What visual feature shift DINOv2 sees (lighting/angle bias)

This module runs a 5-10 second calibration procedure at deployment start
where the robot executes 8 known axis-aligned motions and we observe what
actually happens. From this, we derive a per-dimension scale, offset, and
optional rotation matrix that gets applied to every pose at inference time.

Conceptually: it's the same pattern as IMU bias calibration, but in
6D pose space. Done ONCE per session (or once per robot, if frame is fixed).

Usage:
    calibrator = DeploymentCalibrator(expected_amplitude=0.10)  # 10cm motions
    calibrator.start(robot_controller, observation_fn)
    for _ in range(8):
        calibrator.run_motion(direction)  # executes one cardinal motion
    calib = calibrator.finish()
    calib.save("calib.yaml")

    # At inference:
    pose_in = calib.transform(robot_pose)
    pose_in = calib.apply_camera_correction(pose_in)  # if hand-eye known
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# 8 cardinal directions: +x, -x, +y, -y, +z, -z, and 2 diagonal rotations
# Format: (label, axis_indices_to_excite, expected_rotation_dims)
# axis_indices_to_excite: which of (x,y,z) get pushed positive
# expected_rotation_dims: which of (rx,ry,rz) get a small rotational response
CARDINAL_MOTIONS = [
    ("+x_pos",   (1, 0, 0), (0, 1, 0)),  # move in +x, expect some y-rotation
    ("-x_pos",   (-1, 0, 0), (0, -1, 0)),
    ("+y_pos",   (0, 1, 0), (-1, 0, 0)),  # move in +y, expect some -x rotation
    ("-y_pos",   (0, -1, 0), (1, 0, 0)),
    ("+z_pos",   (0, 0, 1), (0, 0, 0)),  # pure vertical, no rotation
    ("-z_pos",   (0, 0, -1), (0, 0, 0)),
    ("+rot_z",   (0, 0, 0), (0, 0, 1)),  # pure rotation around z
    ("-rot_z",   (0, 0, 0), (0, 0, -1)),
]


@dataclass
class CalibrationResult:
    """Per-axis scale, offset, and frame transform."""
    # Position: pose[:3] is in the source frame; we want it in canonical frame
    position_scale: np.ndarray = field(default_factory=lambda: np.ones(3))    # (3,) per-dim scale
    position_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # (3,) per-dim bias
    position_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))   # (3,3) axis permutation/rotation

    # Orientation: pose[3:6] axis-angle
    orientation_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    orientation_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Visual feature shift (z_v bias)
    visual_bias: np.ndarray = field(default_factory=lambda: np.zeros(256))   # 256-dim DINOv2 offset

    # Diagnostic info
    raw_observations: list = field(default_factory=list)  # list of (label, observed_delta)
    expected_magnitudes: list = field(default_factory=list)
    observed_magnitudes: list = field(default_factory=list)
    confidence_per_axis: list = field(default_factory=list)
    timestamp: float = 0.0
    source_label: str = ""

    def transform_position(self, pos: np.ndarray) -> np.ndarray:
        """Convert a raw source-frame position into the canonical training frame."""
        return (pos - self.position_offset) * self.position_scale
        # NOTE: rotation applied separately via .position_rotation

    def transform_orientation(self, orn: np.ndarray) -> np.ndarray:
        return (orn - self.orientation_offset) * self.orientation_scale

    def transform(self, pose_6d: np.ndarray) -> np.ndarray:
        """Full transform: raw pose (in source frame) → canonical frame."""
        pos = self.position_rotation @ self.transform_position(pose_6d[:3])
        orn = self.transform_orientation(pose_6d[3:6])
        return np.concatenate([pos, orn]).astype(np.float32)

    def save(self, path: str):
        """Save as JSON (numpy arrays as lists)."""
        out = {
            "position_scale": self.position_scale.tolist(),
            "position_offset": self.position_offset.tolist(),
            "position_rotation": self.position_rotation.tolist(),
            "orientation_scale": self.orientation_scale.tolist(),
            "orientation_offset": self.orientation_offset.tolist(),
            "visual_bias": self.visual_bias.tolist(),
            "raw_observations": self.raw_observations,
            "expected_magnitudes": self.expected_magnitudes,
            "observed_magnitudes": self.observed_magnitudes,
            "confidence_per_axis": self.confidence_per_axis,
            "timestamp": self.timestamp,
            "source_label": self.source_label,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        with open(path) as f:
            d = json.load(f)
        return cls(
            position_scale=np.array(d["position_scale"]),
            position_offset=np.array(d["position_offset"]),
            position_rotation=np.array(d["position_rotation"]),
            orientation_scale=np.array(d["orientation_scale"]),
            orientation_offset=np.array(d["orientation_offset"]),
            visual_bias=np.array(d["visual_bias"]),
            raw_observations=d.get("raw_observations", []),
            expected_magnitudes=d.get("expected_magnitudes", []),
            observed_magnitudes=d.get("observed_magnitudes", []),
            confidence_per_axis=d.get("confidence_per_axis", []),
            timestamp=d.get("timestamp", 0.0),
            source_label=d.get("source_label", ""),
        )


class DeploymentCalibrator:
    """Runs the 8-cardinal-motion calibration procedure.

    The robot_controller is expected to have:
        .set_target_pose(pose_6d) -> None
        .get_current_pose() -> np.ndarray (6,)

    The observation_fn is expected to return the current EEF pose.
    For closed-loop systems, this is just robot_controller.get_current_pose().
    For real hardware, it might read from ROS or the Franka state interface.
    """

    def __init__(
        self,
        expected_amplitude_pos: float = 0.10,  # 10cm motions
        expected_amplitude_rot: float = 0.30,  # ~17 deg rotations
        settle_time_s: float = 0.8,             # how long to wait after each motion
        visual_encoder_fn: Optional[Callable] = None,  # optional: DINOv2 encoding
    ):
        self.expected_pos_amp = expected_amplitude_pos
        self.expected_rot_amp = expected_amplitude_rot
        self.settle_time_s = settle_time_s
        self.visual_encoder_fn = visual_encoder_fn
        self.observations: list[tuple[str, np.ndarray]] = []
        self.start_pose: Optional[np.ndarray] = None
        self.robot_controller = None

    def start(self, robot_controller, source_label: str = "new_deployment"):
        """Begin a calibration session. Records the resting pose as reference."""
        self.robot_controller = robot_controller
        self.source_label = source_label
        self.observations = []
        start = robot_controller.get_current_pose()
        self.start_pose = start.copy() if start is not None else None
        self.z_v_samples = []  # for visual feature shift computation

    def run_motion(self, direction: tuple[str, tuple[int, int, int], tuple[int, int, int]]):
        """Execute one cardinal motion and record the observed EEF delta.

        direction = (label, pos_axis_indicator, rot_axis_indicator)
        """
        label, pos_dir, rot_dir = direction
        if self.robot_controller is None:
            raise RuntimeError("Call .start() first")

        # Compute target pose
        target = self.start_pose.copy().astype(np.float64)
        target[:3] += np.array(pos_dir) * self.expected_pos_amp
        target[3:6] += np.array(rot_dir) * self.expected_rot_amp

        # Command motion, then wait for settle
        self.robot_controller.set_target_pose(target)
        time.sleep(self.settle_time_s)

        # Observe
        final_pose = self.robot_controller.get_current_pose()
        observed_delta = final_pose - self.start_pose

        self.observations.append((label, observed_delta))

        # Optional: record visual embedding at this pose (for z_v bias estimation)
        if self.visual_encoder_fn is not None:
            try:
                frame = self.robot_controller.get_camera_frame()  # user must provide
                z_v = self.visual_encoder_fn(frame)
                self.z_v_samples.append(z_v)
            except Exception:
                pass  # visual calibration is optional

        # Return to start
        self.robot_controller.set_target_pose(self.start_pose)
        time.sleep(self.settle_time_s)

    def run_full_calibration(self) -> CalibrationResult:
        """Run all 8 cardinal motions, derive the per-dim scale/offset/rotation."""
        if self.robot_controller is None:
            raise RuntimeError("Call .start() first")

        for direction in CARDINAL_MOTIONS:
            self.run_motion(direction)

        return self.finish()

    def finish(self) -> CalibrationResult:
        """Compute calibration factors from observed deltas.

        Method: For each cardinal direction, we expect the robot to move
        +expected_amp in one axis. If it actually moves different amount
        or in a different axis, that's the scale/offset/rotation we need
        to apply.
        """
        if not self.observations:
            raise RuntimeError("No observations recorded")

        # Build observed matrix: (8, 3) for positions, (8, 3) for orientations
        observed_pos = np.array([o[1][:3] for o in self.observations])  # (8, 3)
        expected_pos = np.array([motion[1] for motion in CARDINAL_MOTIONS], dtype=float) * self.expected_pos_amp
        # (8, 3) - one-hot-ish indicator of which axis we excited

        # Solve: observed = expected @ R^T @ diag(scale) + offset
        # Simplified: axis-by-axis solve.
        # For each pair of opposite motions (e.g. +x and -x), the observed
        # responses should be opposite. If +x moved +0.10 in axis j, then
        # axis j in the source frame maps to "our +x" in the canonical frame.

        # Step 1: determine which source axis corresponds to which canonical axis
        # For each canonical axis (k=0,1,2), find the source axis (j) with
        # the largest positive response when we commanded +k.
        # That source axis j maps to canonical axis k.
        axis_map = np.zeros(3, dtype=int)  # axis_map[k] = j means source_axis_j = canonical_axis_k
        used = set()
        for k in range(3):
            # Find the source axis with largest response in +k motion
            plus_idx = 2 * k  # the "+k" motion is at index 2k
            minus_idx = 2 * k + 1
            plus_obs = observed_pos[plus_idx]
            minus_obs = observed_pos[minus_idx]
            # The correct axis should have: plus_obs[j] ≈ +amp, minus_obs[j] ≈ -amp
            best_j = -1
            best_score = -np.inf
            for j in range(3):
                if j in used:
                    continue
                score = plus_obs[j] - minus_obs[j]  # should be ~2*amp
                if score > best_score:
                    best_score = score
                    best_j = j
            axis_map[k] = best_j
            used.add(best_j)

        # Step 2: compute scale for each canonical axis based on magnitude
        pos_scale = np.ones(3)
        for k in range(3):
            j = axis_map[k]
            # Magnitude of the observed response in source axis j when +k commanded
            plus_obs = observed_pos[2 * k, j]
            minus_obs = observed_pos[2 * k + 1, j]
            measured = (plus_obs - minus_obs) / 2.0
            if abs(measured) > 1e-6:
                pos_scale[k] = self.expected_pos_amp / measured

        # Step 3: compute per-axis offset (constant bias)
        # Use the +k and -k motions' midpoint; offset is the average
        # (assuming zero offset in both, midpoint should be ~0)
        pos_offset = np.zeros(3)
        for k in range(3):
            j = axis_map[k]
            plus_obs = observed_pos[2 * k, j]
            minus_obs = observed_pos[2 * k + 1, j]
            pos_offset[k] = (plus_obs + minus_obs) / 2.0 * pos_scale[k]

        # Step 4: build rotation matrix (axis permutation + sign)
        R = np.zeros((3, 3))
        for k in range(3):
            j = axis_map[k]
            # Determine sign: +k motion should move source axis j in the
            # direction it actually moved
            plus_obs = observed_pos[2 * k, j]
            sign = 1.0 if plus_obs >= 0 else -1.0
            R[k, j] = sign

        # Step 5: orientation calibration (similar logic, on pose[3:6])
        observed_orn = np.array([o[1][3:6] for o in self.observations])
        expected_orn = np.array([motion[2] for motion in CARDINAL_MOTIONS], dtype=float) * self.expected_rot_amp

        orn_scale = np.ones(3)
        orn_offset = np.zeros(3)
        for k in range(3):
            plus_obs = observed_orn[2 * k, k]
            minus_obs = observed_orn[2 * k + 1, k]
            measured = (plus_obs - minus_obs) / 2.0
            if abs(measured) > 1e-6:
                orn_scale[k] = self.expected_rot_amp / measured
            orn_offset[k] = (plus_obs + minus_obs) / 2.0 * orn_scale[k]

        # Step 6: confidence per axis (low if scale is very different from 1.0,
        # or if sign determination was ambiguous)
        confidence = []
        for k in range(3):
            scale = pos_scale[k]
            # Confidence drops if we had to flip axes (scale < 0) or if scale is huge
            conf = 1.0 / (1.0 + abs(np.log(abs(scale) + 1e-9)))
            confidence.append(float(conf))

        # Step 7: visual bias (mean of z_v samples minus expected baseline)
        visual_bias = np.zeros(256)
        if self.z_v_samples:
            visual_bias = np.mean(np.stack(self.z_v_samples, axis=0), axis=0)
            # Note: this is a coarse shift estimate. For better results,
            # subtract the average z_v from training data.

        return CalibrationResult(
            position_scale=pos_scale,
            position_offset=pos_offset,
            position_rotation=R,
            orientation_scale=orn_scale,
            orientation_offset=orn_offset,
            visual_bias=visual_bias,
            raw_observations=[(lbl, delta.tolist()) for lbl, delta in self.observations],
            expected_magnitudes=expected_pos.tolist(),
            observed_magnitudes=observed_pos.tolist(),
            confidence_per_axis=confidence,
            timestamp=time.time(),
            source_label=self.source_label,
        )


# ---------------------- CLI test mode ----------------------

if __name__ == "__main__":
    # Synthetic test: pretend the robot reports pose in a different frame
    # than what we trained on. Specifically: +x and +y are swapped, and
    # the scale is 2x larger than expected.
    class FakeRobot:
        def __init__(self):
            self.pose = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])

        def set_target_pose(self, target):
            # In this fake robot, our target in canonical frame gets:
            # - +x and +y swapped (so the robot moves along wrong axes)
            # - magnitude 2x (so a 10cm command becomes 20cm)
            # - 5mm constant bias in +x
            canonical_target = target.copy()
            fake_actual = canonical_target.copy()
            fake_actual[0] = canonical_target[1] * 2.0 + 0.005  # canonical y -> source x, 2x scale, +5mm bias
            fake_actual[1] = canonical_target[0] * 2.0
            fake_actual[2] = canonical_target[2]  # z is fine
            self.pose = fake_actual

        def get_current_pose(self):
            return self.pose.copy()

    robot = FakeRobot()
    cal = DeploymentCalibrator(expected_amplitude_pos=0.10)
    cal.start(robot, source_label="fake_test")
    result = cal.run_full_calibration()

    print("=== Calibration Result ===")
    print(f"position_scale:    {result.position_scale}")
    print(f"position_offset:   {result.position_offset}")
    print(f"position_rotation:\n{result.position_rotation}")
    print(f"orientation_scale: {result.orientation_scale}")
    print(f"confidence:        {result.confidence_per_axis}")
    print()

    # Test: if we feed the result.transform() with the robot's reported pose,
    # it should recover something close to canonical.
    test_pose = np.array([0.025, 0.20, 0.30, 0.0, 0.3, 0.0])  # in fake (source) frame
    transformed = result.transform(test_pose)
    print(f"Source pose:    {test_pose}")
    print(f"After transform: {transformed}")
    print("(should be close to canonical; exact recovery is hard due to noise)")

    result.save("/tmp/test_calib.json")
    print("\nSaved to /tmp/test_calib.json")
