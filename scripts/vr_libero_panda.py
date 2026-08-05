#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vr_libero_panda.py -- VR Teleoperation for Franka Panda in LIBERO
=================================================================

OVERVIEW
--------
Drives a Franka Panda robot arm inside a LIBERO (MuJoCo) simulation
using a Meta Quest 3 VR controller. Pose data is received over MQTT
from the same WebXR browser client used by
`sim_vr_panda_single.py` (sora2.uclab.jp:1883, topic control/piper-wee),
so the **same browser page works unchanged**.

Key design choice: the existing Isaac Sim script runs a heavy USD-based
simulator; this script targets LIBERO's MuJoCo backend so the same VR
hardware can be used for data collection on tasks that are evaluated in
`eval/eval_libero_v4_trajectory.py` (libero_spatial / libero_goal).

WORKFLOW
--------
1. Connect to MQTT broker, subscribe to VR topic.
2. Initialize LIBERO environment for a chosen task (`--suite`, `--task`).
3. Reset the env. On the **first** trigger-down:
   - Record the controller's current world pose as `pos_start_ctrl`,
     `quat_start_ctrl`.
   - Read the robot's current EEF world pose as `pos_start_ee`,
     `quat_start_ee`.
4. While trigger held: dead-reckon the controller delta into a target
   EEF pose using the world-frame quaternion arithmetic from the
   existing script (no Euler).
5. Convert the absolute target EEF pose into a **LIBERO OSC_POSE
   action** (XYZ delta, axis-angle delta, gripper).
   - LIBERO's default controller is OSC_POSE, action_dim=7, input in
     [-1, 1] mapped to position deltas up to 0.05 m and rotation
     deltas up to 0.5 rad per step.
6. Step the env; capture `agentview` and `robot0_eye_in_hand` images.
7. Button A → reset to the same task.
8. Button B → cycle to the next task in the suite.

MQTT EXPECTED PAYLOAD
---------------------
Same payload as `sim_vr_panda_single.py`:
  {
    "controller_object": {
      "_x": float, "_y": float, "_z": float,
      "_qx": float, "_qy": float, "_qz": float, "_qw": float
    },
    "sending":    bool,
    "grip":       float,
    "trigger":    float,
    "buttonA":    bool,        # reset episode
    "buttonB":    bool,        # next task
    "thumbstick": {x, y} | null
  }

The browser is configured for the Isaac Sim's Z-up world; we reuse the
exact same payload format. VRState.update() converts the position+rot
into the LIBERO world frame (which is also Z-up, but with a different
axis alignment). See `libero_to_js_offset` below.

CONTROLS
---------
  Trigger (hold)        : Enable VR teleoperation (dead reckoning)
  Trigger (release)     : Hold current pose
  Grip button           : Close gripper
  Thumbstick UP/DOWN    : Increase/decrease alpha_pos (live-vs-saved pose blend)
  Thumbstick LEFT/RIGHT : Increase/decrease alpha_rot (live-vs-saved pose blend)
  Button A              : Reset episode (same task)
  Button B              : Cycle to next task in the suite

KEY CONFIGURATION
-----------------
  MQTT broker    : sora2.uclab.jp:1883
  MQTT topic     : control/piper-wee
  Camera outputs : agentview, robot0_eye_in_hand (128 x 128 default)
  Controller     : OSC_POSE (LIBERO default)
  Position tol   : input_max * 0.05 m = 0.05 m per step
  Orientation tol: input_max * 0.5 rad = 0.5 rad per step

RENDERING MODES
---------------
  --gui           On-screen OpenGL viewer (agentview render_camera).
                  Builds ControlEnv with has_renderer=True; calls
                  env.render() every step. Requires a display (X/Wayland).
  --headless      Force offscreen-only rendering (default).
  --save-video    Always capture offscreen cameras; works in both modes.

DEPENDENCIES
------------
  libero (libero.libero.envs)
  NumPy, SciPy
  OpenCV (for video saving)
  paho-mqtt
  libero.libero.envs.env_wrapper.ControlEnv (for --gui)
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", message=".*paho.*")
import paho.mqtt.client as mqtt_client

# LIBERO imports (delayed to allow --help without libero installed)
LIBERO_AVAILABLE = False
OffScreenRenderEnv = None
ControlEnv = None
get_benchmark = None
get_libero_path = None
try:
    from libero.libero.envs import OffScreenRenderEnv
    try:
        from libero.libero.envs.env_wrapper import ControlEnv
    except ImportError:
        ControlEnv = None  # older LIBERO versions
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    LIBERO_AVAILABLE = True
except ImportError as e:
    print(f"[warn] LIBERO not available: {e}", file=sys.stderr)

# --- VR coordinate frame conversion (Three.js -> LIBERO/MuJoCo world) ---
# The browser client (A-Frame oculus-touch-controls) sends the controller's
# grip pose as (position, quaternion). The position is the **controller
# grip anchor's world position** (the point on the controller where the
# user's hand grips it), NOT the wrist position. This means rotating the
# wrist while keeping the hand still will cause the reported position to
# sweep in an arc around the wrist. The Isaac script uses
# R.from_euler('zy', [90,90]) for the position remap, which converts
# Three.js (Y-up) to LIBERO's MuJoCo world (Z-up). We use the same
# convention so that "JS up" maps to "LIBERO up". A user-tunable
# --js-to-libero-rot-deg adds an extra Y-rotation offset on top.
#
# Most importantly, the position remap is applied via R.apply() and the
# quaternion remap via R * q * R.inv() so axes stay consistent.

DEFAULT_Z_DEG = 90.0
DEFAULT_Y_DEG = 90.0


# ---------------------------------------------------------------
# VR state container (same shape as the Isaac script for easy comparison)
# ---------------------------------------------------------------
class VRState:
    """Holds the latest MQTT message, accessible from the main sim loop."""

    def __init__(self, z_deg: float = DEFAULT_Z_DEG, y_deg: float = DEFAULT_Y_DEG,
                 pos_ema_alpha: float = 0.3):
        """
        Args:
            z_deg: rotation around Z (degrees) applied first.
            y_deg: rotation around Y (degrees) applied second.
                The combined rotation R = R_y(y_deg) * R_z(z_deg) converts
                JS controller (pos, quat) to LIBERO world frame.
            pos_ema_alpha: EMA smoothing factor for the position signal,
                in [0, 1]. Lower = more smoothing. The browser sends the
                controller's grip anchor world position, which moves with
                wrist rotation (the anchor sweeps an arc around the wrist).
                A low-pass filter on position removes the high-frequency
                rotation-bleeds-into-position artefact while preserving
                the user's translational intent (which is low-frequency).
                0.3 means ~70% new value, ~30% old -- a moderate smoothing.
                Set to 1.0 for no smoothing.
        """
        self.lock = Lock()
        # Pose (in LIBERO world frame, after axis remap)
        self.goal_pos = np.zeros(3)
        self.goal_pos_filtered = np.zeros(3)
        self.goal_rot = np.array([0, 0, 0, 1])  # xyzw quaternion
        # Buttons
        self.sending    = False
        self.grip       = False
        self.buttonA    = False
        self.buttonB    = False
        self.thumbstick = None
        # Combined Z-then-Y rotation. Same matrix is applied to position
        # and quaternion so axes stay consistent.
        self.z_deg = z_deg
        self.y_deg = y_deg
        self._rot_remap = R.from_euler("zy", [z_deg, y_deg], degrees=True)
        # EMA filter state
        self.pos_ema_alpha = pos_ema_alpha
        self._initialized = False

    def update(self, data: dict):
        with self.lock:
            co = data.get("controller_object", {})
            # Position: apply the rotation matrix (consistent with rotation).
            # Replaces the old sign-flip transform that was a mirror reflection
            # (changing handedness) and not a true rotation.
            raw_pos = np.array([
                co.get("_x", 0.0),
                co.get("_y", 0.0),
                co.get("_z", 0.0),
            ])
            self.goal_pos = self._rot_remap.apply(raw_pos)

            # EMA filter to dampen the wrist-rotation-induced position drift
            # (the controller's grip anchor sweeps an arc when the user
            # rotates their wrist). The filter is initialized lazily on
            # the first message.
            if not self._initialized:
                self.goal_pos_filtered = self.goal_pos.copy()
                self._initialized = True
            else:
                a = self.pos_ema_alpha
                self.goal_pos_filtered = (
                    a * self.goal_pos + (1 - a) * self.goal_pos_filtered
                )

            # Rotation: apply the same rotation via conjugation so that
            # pose composition (q_now_ctrl * q_start_ctrl^-1) remains
            # consistent with the rotated position frame.
            q_raw = np.array([
                co.get("_qx", 0.0),
                co.get("_qy", 0.0),
                co.get("_qz", 0.0),
                co.get("_qw", 1.0),
            ])
            nrm = np.linalg.norm(q_raw)
            if nrm > 1e-8:
                q_raw = q_raw / nrm
            q_remap = (self._rot_remap * R.from_quat(q_raw) *
                       self._rot_remap.inv()).as_quat()
            self.goal_rot = q_remap
            nrm2 = np.linalg.norm(self.goal_rot)
            if nrm2 > 1e-8:
                self.goal_rot = self.goal_rot / nrm2

            self.sending = bool(data.get("sending", False))
            self.grip    = bool(data.get("grip", False))
            self.buttonA = bool(data.get("buttonA", False))
            self.buttonB = bool(data.get("buttonB", False))
            ts = data.get("thumbstick")
            if ts is not None and isinstance(ts, dict):
                tx, ty = ts.get("x", 0), ts.get("y", 0)
                if abs(ty) > abs(tx):
                    self.thumbstick = 1 if ty > 0 else 3
                elif abs(tx) > abs(ty):
                    self.thumbstick = 0 if tx < 0 else 2
                else:
                    self.thumbstick = None
            else:
                self.thumbstick = None

    def snapshot(self) -> dict:
        """Return a point-in-time copy of all fields.

        Returns the RAW (unfiltered) position. The rotation-compensation
        logic in the main loop requires the unfiltered position to
        correctly estimate grip_offset at trigger-down.

        `goal_pos_filtered` is still available for callers that want it
        (e.g. for plotting the raw vs. smoothed trace), but the main loop
        should always use `goal_pos`.
        """
        with self.lock:
            return {
                "goal_pos":   self.goal_pos.copy(),
                "goal_pos_filtered": self.goal_pos_filtered.copy(),
                "goal_rot":   self.goal_rot.copy(),
                "sending":    self.sending,
                "grip":       self.grip,
                "buttonA":    self.buttonA,
                "buttonB":    self.buttonB,
                "thumbstick": self.thumbstick,
            }

    def reset_filter(self):
        """Reset the EMA filter so the next update re-initializes from the
        raw signal. Call this at the start of each episode to avoid
        propagating the previous episode's filter state."""
        with self.lock:
            self._initialized = False


# ---------------------------------------------------------------
# MQTT setup (same broker/topic as the Isaac script)
# ---------------------------------------------------------------
BROKER      = "sora2.uclab.jp"
PORT        = 1883
CLIENT_ID   = "LIBERO-vr-wee"
TOPIC       = "control/piper-wee"


def mqtt_on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("[MQTT] Connected to broker")
    else:
        print(f"[MQTT] Connect failed: rc={rc}")


def setup_mqtt(vr_state_arg: VRState) -> mqtt_client.Client:
    inst = mqtt_client.Client(
        client_id=CLIENT_ID,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
    )
    inst.on_connect = mqtt_on_connect

    def _on_message(client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            vr_state_arg.update(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[MQTT] Bad message: {e}")

    inst.on_message = _on_message
    inst.connect(BROKER, PORT)
    inst.subscribe(TOPIC)
    inst.loop_start()
    print(f"[MQTT] Subscribed: '{TOPIC}' on {BROKER}:{PORT}")
    return inst


# ---------------------------------------------------------------
# LIBERO action conversion
# ---------------------------------------------------------------
def world_pose_to_action(
    current_ee_pos: np.ndarray,
    current_ee_rot: np.ndarray,   # (3, 3) rotation matrix
    target_ee_pos: np.ndarray,
    target_ee_rot: np.ndarray,    # (3, 3) rotation matrix
    grip_closed: bool,
    pos_scale: float = 1.0,
    rot_scale: float = 1.0,
    pos_max: float = 0.05,
    rot_max: float = 0.5,
) -> np.ndarray:
    """Convert absolute target EEF pose into a LIBERO OSC_POSE action.

    Args:
        current_ee_pos / target_ee_pos: (3,) world-frame position.
        current_ee_rot / target_ee_rot: (3, 3) world-frame rotation matrix.
        grip_closed: bool — whether to send gripper close signal.
        pos_scale: multiplier applied to the position delta before
            clipping (lets the operator trade off speed vs. accuracy).
        rot_scale: same for rotation.
        pos_max: LIBERO's OSC_POSE position delta cap (default 0.05 m).
        rot_max: LIBERO's OSC_POSE rotation delta cap (default 0.5 rad).

    Returns:
        action: (7,) np.ndarray in OSC_POSE format:
            [dx, dy, dz, drx, dry, drz, gripper]
            where the first 6 dims are in [-1, 1] and gripper is 1
            (close) / -1 (open).
    """
    # Position delta
    dp = (target_ee_pos - current_ee_pos) * pos_scale
    dp_norm = np.linalg.norm(dp)
    if dp_norm > pos_max and dp_norm > 0:
        dp = dp * (pos_max / dp_norm)
    dp_action = dp / pos_max  # map back to [-1, 1]
    # Rotation delta: target_rot * current_rot^-1 -> axis-angle
    delta_R = target_ee_rot @ current_ee_rot.T
    delta_rotvec = R.from_matrix(delta_R).as_rotvec() * rot_scale
    rot_norm = np.linalg.norm(delta_rotvec)
    if rot_norm > rot_max and rot_norm > 0:
        delta_rotvec = delta_rotvec * (rot_max / rot_norm)
    rot_action = delta_rotvec / rot_max
    # Gripper
    grip_action = -1.0 if grip_closed else 1.0
    return np.concatenate([dp_action, rot_action, [grip_action]]).astype(np.float32)


# ---------------------------------------------------------------
# SLERP helper
# ---------------------------------------------------------------
def slerp_quat(q0_xyzw, q1_xyzw, t):
    from scipy.spatial.transform import Slerp
    t = np.clip(t, 0.0, 1.0)
    if t <= 0.0:
        return q0_xyzw.copy()
    if t >= 1.0:
        return q1_xyzw.copy()
    interp = Slerp([0.0, 1.0], R.from_quat(np.array([q0_xyzw, q1_xyzw])))
    return interp(t).as_quat()


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", default="libero_spatial",
                        choices=["libero_spatial", "libero_goal", "libero_object"],
                        help="LIBERO benchmark suite (default: libero_spatial)")
    parser.add_argument("--task", default=None,
                        help="Task name to run. If omitted, start at the first "
                             "task in the suite; use button B to cycle.")
    parser.add_argument("--render-size", type=int, default=128,
                        help="Camera render size (default: 128)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save rendered frames to ./videos/")
    parser.add_argument("--video-fps", type=int, default=20,
                        help="Output video frame rate")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max env steps per episode")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless rendering (no on-screen viewer)")
    parser.add_argument("--gui", action="store_true",
                        help="Show an on-screen OpenGL viewer (the agentview "
                             "render-camera is displayed in a window). Requires "
                             "a display (X server / Wayland) — don't combine with "
                             "--headless. The offscreen cameras are still "
                             "captured for --save-video.")
    parser.add_argument("--js-to-libero-y-deg", type=float,
                        default=DEFAULT_Y_DEG,
                        help="Additional Y-axis rotation (degrees) applied "
                             "on top of the default zy=[90,90] remap. Useful "
                             "for tuning the LIBERO scene's right/left "
                             "axis to match your physical setup.")
    parser.add_argument("--pos-scale", type=float, default=2.0,
                        help="Position-delta multiplier before clipping "
                             "(1.0 = match LIBERO's natural 0.05 m/step cap; "
                             "2.0 = move twice as fast).")
    parser.add_argument("--pos-ema-alpha", type=float, default=0.3,
                        help="[legacy] EMA smoothing factor. No longer used "
                             "by the main loop -- the rotation-compensation "
                             "logic requires raw positions. Kept as a "
                             "placeholder in case future work adds filter "
                             "back as an optional post-processing step.")
    parser.add_argument("--print-action", action="store_true",
                        help="Print the 7-dim LIBERO action sent at every step "
                             "(pos_delta[0:3], rot_delta[3:6], gripper[6]). "
                             "Useful for debugging axis mapping and "
                             "rotation translation issues.")
    parser.add_argument("--rot-scale", type=float, default=2.0,
                        help="Rotation-delta multiplier before clipping "
                             "(1.0 = match LIBERO's natural 0.5 rad/step cap; "
                             "2.0 = rotate twice as fast).")
    parser.add_argument("--save-dataset", default=None,
                        help="If set, save each successful episode as an HDF5 "
                             "file compatible with ALIGN's data loader. Path is "
                             "the output directory.")
    args = parser.parse_args()

    if not LIBERO_AVAILABLE:
        print("[error] LIBERO is not installed in this Python environment.")
        print("        Activate the ALIGN conda env: conda activate align")
        sys.exit(1)

    # Build the VRState with the rotation-only transform. The position
    # and rotation are now both remapped through the same rotation matrix,
    # so axes stay consistent and handedness is preserved.
    vr_state = VRState(z_deg=DEFAULT_Z_DEG, y_deg=args.js_to_libero_y_deg,
                       pos_ema_alpha=args.pos_ema_alpha)

    mqtt_inst = setup_mqtt(vr_state)

    # --- Initialize LIBERO benchmark + task list ---
    benchmark = get_benchmark(args.suite)()
    task_list = [t.name for t in benchmark.tasks]
    if args.task is not None:
        if args.task not in task_list:
            print(f"[error] --task '{args.task}' not in {args.suite}; choices: "
                  f"{task_list}")
            sys.exit(1)
        task_idx = task_list.index(args.task)
    else:
        task_idx = 0
    print(f"[LIBERO] Suite: {args.suite}, {len(task_list)} tasks available.")
    print(f"[LIBERO] Starting at task {task_idx}: {task_list[task_idx]}")

    # --- Pre-build env (one per task; rebuilt when cycling) ---
    def build_env(task_name):
        bddl_dir = os.path.join(
            get_libero_path("bddl_files"), args.suite, "",
        )
        # Find the BDDL file matching task_name (it may have a hash suffix)
        candidates = [
            f for f in os.listdir(bddl_dir)
            if f.startswith(task_name) and f.endswith(".bddl")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No BDDL file matching '{task_name}' in {bddl_dir}"
            )
        bddl_file = os.path.join(bddl_dir, candidates[0])

        # Render-mode dispatch:
        #   --gui          : on-screen OpenGL viewer (ControlEnv).
        #                    The OffScreenRenderEnv subclass forces
        #                    has_renderer=False, so we use the parent
        #                    ControlEnv class for the GUI case.
        #   --save-video (no --gui): offscreen only (OffScreenRenderEnv).
        #   headless default : offscreen only (OffScreenRenderEnv).
        use_gui = bool(getattr(args, "gui", False))

        if use_gui:
            if ControlEnv is None:
                raise ImportError(
                    "GUI mode requested but libero.libero.envs.env_wrapper."
                    "ControlEnv is not importable. Update LIBERO or run "
                    "without --gui."
                )
            print(f"[ENV] Building GUI env for task: {task_name}")
            return ControlEnv(
                bddl_file_name=bddl_file,
                use_camera_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_widths=args.render_size,
                camera_heights=args.render_size,
                reward_shaping=False,
                control_freq=20,
                initialization_noise=None,
                has_renderer=True,
                has_offscreen_renderer=True,  # keep offscreen cams for video
                render_camera="agentview",     # what the on-screen viewer shows
            )

        # Default: offscreen-only (faster, headless-safe).
        print(f"[ENV] Building offscreen env for task: {task_name}")
        return OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_widths=args.render_size,
            camera_heights=args.render_size,
            reward_shaping=False,
            control_freq=20,
            initialization_noise=None,
        )

    # --- VR teleoperation state ---
    pos_start_ee   = None
    pos_start_ctrl = None
    quat_start_ctrl = None
    quat_start_ee   = None
    grip_offset     = None  # estimated controller local-frame offset to EE
    pos_save  = None  # last target pos (held pose)
    quat_save = None
    alpha_pos = 0.0
    alpha_rot = 0.0
    prev_trigger = False
    saved_videos = []
    saved_episodes = []

    # --- Episode loop ---
    try:
        while task_idx < len(task_list):
            task_name = task_list[task_idx]
            print(f"\n{'=' * 70}")
            print(f"[TASK {task_idx + 1}/{len(task_list)}] {task_name}")
            print(f"{'=' * 70}")

            # Build (or rebuild) env for the current task
            try:
                env = build_env(task_name)
            except Exception as e:
                print(f"[error] failed to build env: {e}")
                task_idx += 1
                continue

            obs = env.reset()
            task_desc = env.language_instruction
            print(f"[TASK] Description: {task_desc}")

            # Capture initial EEF pose (MuJoCo world frame)
            current_ee_pos = env.env._eef_xpos.copy()
            current_ee_rot = env.env._eef_xmat.copy()
            print(f"[INIT] EE pos: {current_ee_pos}")
            print(f"[INIT] EE rot:\n{current_ee_rot}")

            # Reset VR anchors
            pos_start_ee   = None
            pos_start_ctrl = None
            quat_start_ctrl = None
            quat_start_ee   = None
            grip_offset     = None
            pos_save  = current_ee_pos.copy()
            quat_save = R.from_matrix(current_ee_rot).as_quat()
            alpha_pos = 0.0
            alpha_rot = 0.0
            prev_trigger = False

            # Reset the EMA filter so the new episode doesn't inherit the
            # filter state from the previous one (would cause a 1-frame
            # bias at the start of the episode).
            vr_state.reset_filter()

            episode_frames = []  # list of (agentview_rgb, wrist_rgb) tuples

            for step in range(args.max_steps):
                d = vr_state.snapshot()
                goal_pos = d["goal_pos"]
                goal_rot = d["goal_rot"]
                sending  = d["sending"]
                grip     = d["grip"]
                buttonA  = d["buttonA"]
                buttonB  = d["buttonB"]
                thumbstick = d["thumbstick"]

                # --- Trigger edge: OFF -> ON ---
                if sending and not prev_trigger:
                    # Capture anchors (current EE pose + current VR pose).
                    # We assume the user's hand is at "rest" at trigger-down,
                    # so the controller's reported pose approximates the
                    # EE pose plus a fixed grip_offset in the controller's
                    # local frame:
                    #   pos_start_ctrl ≈ pos_start_ee + R_ctrl_start * grip_offset
                    # We can solve for grip_offset:
                    #   grip_offset ≈ R_ctrl_start^-1 * (pos_start_ctrl - pos_start_ee)
                    # At trigger-down, R_ctrl_start is whatever the controller
                    # is currently rotated to -- we use its inverse to undo.
                    pos_start_ee   = current_ee_pos.copy()
                    pos_start_ctrl = goal_pos.copy()
                    quat_start_ctrl = R.from_quat(goal_rot)
                    quat_start_ee   = R.from_matrix(current_ee_rot)
                    # Estimate the grip offset (controller's local-frame
                    # offset from EE to controller grip anchor).
                    grip_offset = quat_start_ctrl.inv().apply(
                        pos_start_ctrl - pos_start_ee
                    )
                    print(f"[VR] Trigger pressed — anchors recorded. "
                          f"grip_offset={grip_offset.round(3).tolist()}")

                # --- If never calibrated yet, hold default pose ---
                if quat_start_ee is None:
                    target_pos = pos_save.copy()
                    target_rot = quat_save.copy()
                    prev_trigger = sending
                else:
                    if sending:
                        # Position: subtract the rotation-induced grip
                        # anchor sweep before computing the delta.
                        #
                        # The browser reports the controller's grip anchor
                        # position, which is wrist_pos + R_ctrl * grip_offset.
                        # When the user rotates the wrist, R_ctrl rotates and
                        # the anchor sweeps an arc around the wrist. We
                        # compensate by removing this rotation-induced offset:
                        #   wrist_pos_now ≈ goal_pos - R_ctrl_now * grip_offset
                        # The wrist translation since trigger-down is:
                        #   wrist_delta = (goal_pos - R_ctrl_now * grip_offset)
                        #               - pos_start_ctrl_start
                        # and we apply it to the EE:
                        #   target_pos = pos_start_ee + wrist_delta
                        #
                        # When wrist rotation is unchanged from trigger-down
                        # (R_ctrl_now = R_ctrl_start), this reduces to the
                        # original naive delta: target_pos = pos_start_ee +
                        # (goal_pos - pos_start_ctrl), so the math is a strict
                        # superset of the original behavior.
                        q_now_ctrl = R.from_quat(goal_rot)
                        wrist_pos_now = goal_pos - q_now_ctrl.apply(grip_offset)
                        wrist_pos_start = (
                            pos_start_ctrl
                            - quat_start_ctrl.apply(grip_offset)
                        )
                        wrist_delta = wrist_pos_now - wrist_pos_start
                        target_pos = (pos_start_ee + wrist_delta).copy()

                        # Rotation: world-frame delta in the remapped
                        # (LIBERO) frame. The same R_remap matrix that maps
                        # JS positions to LIBERO positions also maps JS
                        # rotations to LIBERO rotations:
                        #   R_ctrl_LIBERO = R_remap @ R_js @ R_remap^-1
                        # Since both q_now_ctrl and quat_start_ctrl are
                        # already in LIBERO frame (they were remapped in
                        # VRState.update()), the delta is just:
                        rot_delta_world = q_now_ctrl * quat_start_ctrl.inv()
                        target_rot = (quat_start_ee * rot_delta_world).as_quat()
                    else:
                        target_pos = pos_save.copy()
                        target_rot = quat_save.copy()

                    # --- Trigger edge: ON -> OFF -> save FK pose ---
                    if prev_trigger and not sending:
                        pos_save  = target_pos.copy()
                        quat_save = target_rot.copy()
                        print(f"[VR] Trigger released — pose held.")

                    # --- Thumbstick: alpha blending with saved pose ---
                    if thumbstick == 1:
                        alpha_pos = min(alpha_pos + 0.1, 1.0)
                    elif thumbstick == 3:
                        alpha_pos = max(alpha_pos - 0.1, 0.0)
                    elif thumbstick == 0:
                        alpha_rot = min(alpha_rot + 0.1, 1.0)
                    elif thumbstick == 2:
                        alpha_rot = max(alpha_rot - 0.1, 0.0)
                    alpha_pos = round(alpha_pos, 1)
                    alpha_rot = round(alpha_rot, 1)

                    # Lerp position, slerp rotation toward saved pose
                    target_pos_live = target_pos.copy()
                    target_pos = target_pos * (1 - alpha_pos) + pos_save * alpha_pos

                    target_rot_live = target_rot.copy()
                    target_rot = slerp_quat(target_rot, quat_save, alpha_rot)

                    prev_trigger = sending

                # --- Convert target pose -> LIBERO action ---
                target_R = R.from_quat(target_rot).as_matrix()
                action = world_pose_to_action(
                    current_ee_pos=current_ee_pos,
                    current_ee_rot=current_ee_rot,
                    target_ee_pos=target_pos,
                    target_ee_rot=target_R,
                    grip_closed=grip,
                    pos_scale=args.pos_scale,
                    rot_scale=args.rot_scale,
                )

                # --- Step the env ---
                if args.print_action and step < 50:  # only first 50 steps
                    print(f"  [action] step {step:3d}  dp={action[:3].round(2).tolist()}  "
                          f"dr={action[3:6].round(2).tolist()}  grip={action[6]:.1f}")
                obs, reward, done, info = env.step(action)

                # --- On-screen GUI viewer (no-op if not enabled) ---
                # Both ControlEnv and OffScreenRenderEnv are wrappers around
                # the underlying robosuite env; the render() method lives on
                # env.env. The call is a no-op when has_renderer=False.
                if args.gui:
                    env.env.render()

                # --- Read new EEF pose ---
                current_ee_pos = env.env._eef_xpos.copy()
                current_ee_rot = env.env._eef_xmat.copy()

                # --- Capture images (if saving video) ---
                if args.save_video:
                    agentview_rgb = obs.get("agentview_image")
                    wrist_rgb = obs.get("robot0_eye_in_hand_image")
                    if agentview_rgb is not None:
                        episode_frames.append((agentview_rgb, wrist_rgb))

                # --- Status ---
                if step % 20 == 0:
                    print(f"  step {step:4d}  EE_pos={current_ee_pos.round(3).tolist()}  "
                          f"reward={reward:.2f}  done={done}")

                # --- Check success ---
                if env.check_success():
                    print(f"\n[SUCCESS] Task completed at step {step}!")
                    done = True

                # --- Episode end conditions ---
                if done or buttonA:
                    reason = "success" if env.check_success() else \
                             ("buttonA reset" if buttonA else "done flag")
                    print(f"[EPISODE] End at step {step} ({reason}).")

                    # Save video if requested
                    if args.save_video and episode_frames:
                        out_dir = Path("videos") / args.suite
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{task_name}.mp4"
                        save_video(episode_frames, str(out_path), args.video_fps)
                        saved_videos.append(str(out_path))
                        print(f"[VIDEO] Saved: {out_path}")

                    # Save dataset if requested
                    if args.save_dataset and done:
                        out_path = Path(args.save_dataset) / f"{task_name}.h5"
                        save_episode(obs, action, str(out_path))
                        saved_episodes.append(str(out_path))
                        print(f"[DATASET] Saved: {out_path}")

                    if buttonA:
                        # Reset same task (don't advance)
                        obs = env.reset()
                        current_ee_pos = env.env._eef_xpos.copy()
                        current_ee_rot = env.env._eef_xmat.copy()
                        pos_start_ee = None
                        pos_start_ctrl = None
                        quat_start_ctrl = None
                        quat_start_ee = None
                        grip_offset = None
                        pos_save = current_ee_pos.copy()
                        quat_save = R.from_matrix(current_ee_rot).as_quat()
                        alpha_pos = 0.0
                        alpha_rot = 0.0
                        prev_trigger = False
                        episode_frames = []
                        print(f"[RESET] Re-running task: {task_name}")
                        continue
                    break

            # Cycle to next task on normal completion
            task_idx += 1
            if task_idx < len(task_list):
                print(f"\n[BTN B hint] Press button B to advance to: "
                      f"{task_list[task_idx]}")
                print("[BTN B hint] Or Ctrl-C to exit.")

            # Wait for button B or timeout
            print("[WAIT] Press button B to advance to next task "
                  "(or Ctrl-C to exit).")
            while True:
                d = vr_state.snapshot()
                if d["buttonB"]:
                    # Drain the press so it doesn't double-fire
                    time.sleep(0.3)
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        mqtt_inst.loop_stop()
        mqtt_inst.disconnect()
        if saved_videos:
            print(f"\n[INFO] Saved {len(saved_videos)} videos to ./videos/{args.suite}/")
        if saved_episodes:
            print(f"[INFO] Saved {len(saved_episodes)} dataset files to "
                  f"{args.save_dataset}/")


def save_video(frames, output_path, fps):
    """Save a list of (agentview_rgb, wrist_rgb) tuples as a side-by-side MP4."""
    import cv2
    if not frames:
        return
    h, w, _ = frames[0][0].shape
    # Side-by-side layout
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h)
    )
    for agent, wrist in frames:
        if wrist is None:
            wrist = np.zeros_like(agent)
        # obs returns (H, W, 3) uint8
        combined = np.concatenate([agent, wrist], axis=1)
        # cv2 expects BGR; flip RGB -> BGR
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    out.release()


def save_episode(obs, action, output_path):
    """Save a successful episode as an HDF5 file (lightweight wrapper).

    For full alignment with ALIGN's data loader you'd want to save
    the full per-step trajectory, but this hook captures only the
    final observation + action to validate the format."""
    import h5py
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                try:
                    f.create_dataset(k, data=v)
                except Exception:
                    pass
        f.create_dataset("action", data=action)


if __name__ == "__main__":
    main()