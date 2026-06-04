#!/usr/bin/env python3
"""ALIGN Data Collection — standalone script for recording teleop episodes.

Runs Franka Panda in Isaac Sim with VR teleoperation (same MQTT setup as
sim_vr_panda_single.py) and records data using the ALIGN DataRecorder.

Can optionally inject synthetic noise into the VR pose before sending to IK,
so the collected data contains both the noisy teleop (for training) and
the smooth reference (the actual VR input before noise).

Usage:
    # Record 50 episodes with default settings
    python collect_episodes.py --num-episodes 50

    # Record with noise injection + specific objects
    python collect_episodes.py --num-episodes 30 --noise moderate --objects mug bowl can

    # Record expert smooth demonstrations (no noise)
    python collect_episodes.py --num-episodes 10 --smooth-only
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np

# ── ALIGN modules ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_data_recorder import DataRecorder
# Note: align_noise is available for evaluation/stress-testing but NOT used in default collection

# ── Isaac Sim (must come first) ───────────────────────────────────────
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import VisualCuboid
from omni.isaac.motion_generation import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaKinematicsSolver
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from pxr import UsdPhysics

# ── MQTT ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", message=".*paho.*")
from paho.mqtt import client as mqtt_client

# ============================================================
# Constants
# ============================================================
BROKER = "sora2.uclab.jp"
PORT = 1883
CLIENT_ID = "ALIGN-data-collector"
TOPIC = "control/piper-wee"

# Camera config
WIDTH = 720
HEIGHT = 480
FREQUENCY = 20

# Initial joint positions (matches sim_vr_panda_single.py)
FRANKA_HOME = np.array([
    0.0,      # panda_joint1
    -0.93,    # panda_joint2
    0.0,      # panda_joint3
    -1.43,    # panda_joint4
    0.0,      # panda_joint5
    1.25,     # panda_joint6
    0.86,     # panda_joint7
    0.0,      # panda_finger_joint1
    -0.0,     # panda_finger_joint2
])

# Rotation conversion: Three.js → Isaac Sim (from sim_vr_panda_single.py)
POSITION_OFFSET = None  # lazy init
ROTATION_OFFSET = None  # lazy init

# Whether recording (set by CLI args)
RECORD_SMOOTH_ONLY = False

# ============================================================
# VR State (copied from sim_vr_panda_single.py — standalone)
# ============================================================
class VRState:
    """Holds the latest MQTT message, accessible from main sim loop."""

    def __init__(self):
        self.lock = Lock()
        self.goal_pos = np.zeros(3)
        self.goal_rot = np.array([0, 0, 0, 1])
        self.sending = False
        self.grip = False
        self.buttonA = False
        self.buttonB = False
        self.thumbstick = None

    def update(self, data: dict):
        from scipy.spatial.transform import Rotation as R

        with self.lock:
            global POSITION_OFFSET, ROTATION_OFFSET
            if POSITION_OFFSET is None:
                POSITION_OFFSET = R.from_euler('zy', [90, 90], degrees=True)
                ROTATION_OFFSET = R.from_euler('zy', [90, 180], degrees=True)

            co = data.get("controller_object", {})
            self.goal_pos = np.array([
                co.get("_x", 0),
                co.get("_y", 0),
                co.get("_z", 0),
            ])
            self.goal_pos = POSITION_OFFSET.apply(self.goal_pos)

            q_js_raw = np.array([
                co.get("_qx", 0),
                co.get("_qy", 0),
                co.get("_qz", 0),
                co.get("_qw", 1),
            ])
            nrm = np.linalg.norm(q_js_raw)
            if nrm > 1e-8:
                q_js_raw /= nrm
            q_offset = (ROTATION_OFFSET * R.from_quat(q_js_raw) * ROTATION_OFFSET.inv()).as_quat()
            self.goal_rot = q_offset
            nrm2 = np.linalg.norm(self.goal_rot)
            if nrm2 > 1e-8:
                self.goal_rot /= nrm2

            self.sending = bool(data.get("sending", False))
            self.grip = bool(data.get("grip", False))
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
        with self.lock:
            return {
                "goal_pos": self.goal_pos.copy(),
                "goal_rot": self.goal_rot.copy(),
                "sending": self.sending,
                "grip": self.grip,
                "buttonA": self.buttonA,
                "buttonB": self.buttonB,
                "thumbstick": self.thumbstick,
            }


vr_state = VRState()


# ============================================================
# MQTT callbacks
# ============================================================
def mqtt_on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("[MQTT] Connected to broker")
    else:
        print(f"[MQTT] Connect failed: rc={rc}")


def mqtt_on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        vr_state.update(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[MQTT] Bad message: {e}")


def setup_mqtt():
    mqtt_inst = mqtt_client.Client(
        client_id=CLIENT_ID,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
    )
    mqtt_inst.on_connect = mqtt_on_connect
    mqtt_inst.on_message = mqtt_on_message
    mqtt_inst.connect(BROKER, PORT)
    mqtt_inst.subscribe(TOPIC)
    mqtt_inst.loop_start()
    print(f"[MQTT] Subscribed: '{TOPIC}' on {BROKER}:{PORT}")
    return mqtt_inst


# ============================================================
# Object spawning
# ============================================================
# Pre-defined object templates (name, prim_path_suffix)
OBJECT_TEMPLATES = [
    {"name": "cube_1",      "prim": "/World/Xform_dex_cube"},
    {"name": "cube_2",    "prim": "/World/Xform_dex_cube_01"},
    {"name": "rubik_1",     "prim": "/World/Xform_rubik"},
    {"name": "rubik_2",    "prim": "/World/Xform_rubik_01"},
    {"name": "nvidia_cube", "prim": "/World/Xform_nvidia_cube"},
]

def random_pose(is_tray=False):
    z = 2.8 if is_tray else 2.45
    if is_tray:
        regions = [((0.1, 0.5), (-0.6, -0.3)), ((0.1, 0.5), (0.3, 0.6)),
                   ((0.5, 0.6), (-0.6, 0.6))]
    else:
        regions = [((0.0, 0.5), (-1.0, -0.3)), ((0.0, 0.5), (0.3, 1.0)),
                   ((0.5, 1.15), (-1.0, 1.0))]
    areas = [(x1 - x0) * (y1 - y0) for (x0, x1), (y0, y1) in regions]
    p = np.array(areas) / sum(areas)
    region = regions[np.random.choice(len(regions), p=p)]
    (x0, x1), (y0, y1) = region
    x = np.random.uniform(x0, x1)
    y = np.random.uniform(y0, y1)
    return x,y

def spawn_objects(object_names: list[str] | None = None):
    """Place objects randomly on the table. Returns dict of {name: SingleXFormPrim}."""
    from isaacsim.core.utils.prims import get_prim_at_path

    if object_names is None:
        object_names = [t["name"] for t in OBJECT_TEMPLATES[:5]]

    objects = {}
    for t in OBJECT_TEMPLATES:
        if t["name"] not in object_names:
            continue
        prim_path = t["prim"]
        prim = SingleXFormPrim(prim_path)
        prim.initialize()
        
        # Random position on table
        x, y = random_pose()
        z = 2.45  # table height
        prim.set_world_pose(position=np.array([x, y, z]))
        
        objects[t["name"]] = {
            "prim": prim,
            "position": np.array([x, y, z]),
        }
        print(f"  Spawned '{t['name']}' at ({x:.2f}, {y:.2f}, {z:.2f})")

    return objects


# ============================================================
# Episode configuration
# ============================================================
def get_episode_config(
    objects: dict,
    episode_num: int,
    total_episodes: int,
) -> tuple[str, str, list[str]]:
    """Determine task description and target for an episode.

    Returns:
        (task_description, target_object_name, active_object_names)
    """
    object_names = list(objects.keys())
    target = object_names[episode_num % len(object_names)]

    # Generate text variants based on episode number
    variants = [
        f"pick up the {target}",
        f"grasp the {target}",
        f"reach for the {target}",
    ]
    task_desc = variants[episode_num % len(variants)]
    
    return task_desc, target, object_names


# ============================================================
# Main collection loop
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="ALIGN Data Collection — record teleop episodes for training"
    )
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to collect (default: 10)")
    parser.add_argument("--output-dir", type=str, default="/home/ucluser/VRWIT/ALIGN/align_data",
                        help="Output directory (default: ./align_data)")
    parser.add_argument("--noise", type=str, default="none",
                        choices=["none", "mild", "moderate", "heavy"],
                        help="Noise injection preset (NOT used in collection — for evaluation only)")
    parser.add_argument("--smooth-only", action="store_true",
                        help="Record clean expert demos")
    parser.add_argument("--operator", type=str, default="anonymous",
                        help="Operator ID for metadata")
    parser.add_argument("--episode-timeout", type=float, default=60.0,
                        help="Max seconds per episode before auto-finalize")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for object placement")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Note on noise ──
    global RECORD_SMOOTH_ONLY
    RECORD_SMOOTH_ONLY = args.smooth_only
    print(f"[Config] Mode: {'smooth-only (expert demos)' if RECORD_SMOOTH_ONLY else 'normal (real human noise)'}")
    print("[Config] Synthetic noise injection disabled — real human teleop provides sufficient noise")

    # ── Isaac Sim setup ──
    print("\n=== Isaac Sim Setup ===")
    open_stage(
        "/home/ucluser/isaacgym/assets/urdf/piper_description/urdf/"
        "piper_description/franka_simple_1.usd"
    )
    sim = SimulationContext()
    dt = sim.get_physics_dt()
    sim.reset()
    sim.play()
    set_camera_view(eye=[2.0, 0.0, 4.0], target=[0.0, 0.0, 2.5])

    robot = SingleArticulation("/World/franka")
    robot.initialize()
    print(f"  Robot DOF: {robot.dof_names}")

    base = SingleRigidPrim("/World/franka/panda_link0")
    franka_hand = SingleRigidPrim("/World/franka/panda_hand")
    base.initialize()
    franka_hand.initialize()

    # Cameras
    wrist_cam = Camera(
        prim_path="/World/franka/panda_hand/Realsense/RSD455/Camera_OmniVision_OV9782_Color",
        frequency=FREQUENCY, resolution=(WIDTH, HEIGHT),
    )
    mid_cam = Camera(
        prim_path="/World/Realsense_mid/RSD455/Camera_OmniVision_OV9782_Color",
        frequency=FREQUENCY, resolution=(WIDTH, HEIGHT),
    )
    wrist_cam.initialize()
    mid_cam.initialize()

    # PD drives
    robot_prim = get_prim_at_path("/World/franka")
    stage = robot_prim.GetStage()
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(robot_prim.GetPath()):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.GetStiffnessAttr().Set(1e4)
            drive.GetDampingAttr().Set(1e2)

    # IK solver
    ik_solver = LulaKinematicsSolver(
        robot_description_path=(
            "/home/ucluser/isaacgym/assets/urdf/franka_description/"
            "config/franka_robot.yaml"
        ),
        urdf_path=(
            "/home/ucluser/isaacgym/assets/urdf/franka_description/"
            "robots/franka_panda.urdf"
        ),
    )
    ik_solver.set_default_position_tolerance(0.02)
    ik_solver.set_default_orientation_tolerance(0.02)
    kin_solver = ArticulationKinematicsSolver(
        robot_articulation=robot,
        kinematics_solver=ik_solver,
        end_effector_frame_name="panda_hand",
    )

    # Base pose for IK
    base_pos_world, _ = base.get_world_pose()

    # Visual marker
    target_marker = VisualCuboid(
        prim_path="/World/IK_Target",
        position=[0.0, 0.0, 2.5],
        scale=[0.03, 0.03, 0.03],
        color=np.array([0.0, 1.0, 0.0]),  # Green = recording active
    )

    # ── MQTT ──
    mqtt_inst = setup_mqtt()

    # ── VR teleop state ──
    pos_start_ee = None
    pos_start_ctrl = None
    quat_start_ctrl = None
    quat_start_ee = None
    pos_save = np.array([0.2, 0.0, 0.25])
    quat_save = np.array([-0.70710678, 0.0, 0.0, 0.70710678])
    prev_trigger = False

    # ── Objects ──
    objects = spawn_objects()
    obj_positions = {name: info["position"].tolist() for name, info in objects.items()}
    print(f"  Objects spawned: {list(objects.keys())}")

    # ── Output dir ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Set initial pose ──
    robot.set_joint_positions(FRANKA_HOME)

    # ── Main collection loop ──
    print(f"\n{'='*60}")
    print(f"ALIGN Data Collection: {args.num_episodes} episodes")
    print(f"{'='*60}")
    print("Controls:")
    print("  Trigger (hold):   Enable VR teleoperation")
    print("  Trigger (release): Hold current pose")
    print("  Grip button:      Close gripper")
    print("  Button A:         Cancel current episode / reset")
    print("  Button B:         Force-finalize current episode")
    print(f"{'='*60}\n")

    t = 0
    episode_num = 0
    recorder = None
    episode_start_time = 0.0
    episode_recording = False  # True while trigger is held during recording
    waiting_for_trigger = True
    elapsed = 0.0

    while simulation_app.is_running() and episode_num < args.num_episodes:
        sim.step(render=True)

        # ── 1. Read VR state ──
        d = vr_state.snapshot()
        goal_pos = d["goal_pos"]
        goal_rot = d["goal_rot"]
        sending = d["sending"]
        grip = d["grip"]
        buttonA = d["buttonA"]
        buttonB = d["buttonB"]

        # ── 2. Gripper ──
        joint_efforts = robot.get_measured_joint_efforts()
        if grip:
            joint_efforts[7] = -500.0
        else:
            joint_efforts[7] = 1000.0
        robot.apply_action(ArticulationAction(joint_efforts=joint_efforts))

        # ── 3. VR teleoperation ──
        ee_pos_world, ee_rot_world = franka_hand.get_world_pose()
        ee_rel = ee_pos_world - base_pos_world

        ctrl_norm = np.linalg.norm(goal_rot)
        if ctrl_norm < 1e-8:
            goal_rot = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            goal_rot = goal_rot / ctrl_norm

        from scipy.spatial.transform import Rotation as R

        # Trigger edge: first press
        if sending and not prev_trigger:
            ee_pos_world, ee_rot_world = franka_hand.get_world_pose()
            ee_rel = ee_pos_world - base_pos_world
            pos_start_ee = pos_save.copy()
            pos_start_ctrl = goal_pos.copy()
            quat_start_ee = R.from_quat(ee_rot_world)
            quat_start_ctrl = R.from_quat(goal_rot)

        # Save pose on trigger release
        if prev_trigger and not sending:
            pos_save = target_pos_VR.copy()
            quat_save = target_rot_VR.copy()
            
        # Compute target pose from VR
        if quat_start_ee is not None and sending:
            ctrl_pos_delta = goal_pos - pos_start_ctrl
            target_pos_VR = pos_start_ee + ctrl_pos_delta

            q_now_ctrl = R.from_quat(goal_rot)
            rot_delta_world = q_now_ctrl * quat_start_ctrl.inv()
            dq = rot_delta_world.as_quat().copy()
            dq[0] = -dq[0]
            rot_delta_world = R.from_quat(dq)
            target_rot_VR = (quat_start_ee * rot_delta_world).as_quat()
        else:
            target_pos_VR = pos_save
            target_rot_VR = quat_save


        prev_trigger = sending

        target_pos_IK = target_pos_VR.copy()
        target_rot_IK = target_rot_VR.copy()

        # ── 4. Camera capture ──
        wrist_rgba = wrist_cam.get_rgba()
        wrist_rgb = (
            wrist_rgba.copy().reshape((HEIGHT, WIDTH, 4))[:, :, :3].astype(np.uint8)
            if len(wrist_rgba) > 0
            else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        )

        # ── 5. Data recording ──
        if recorder is not None and episode_recording:
            
            # print(f"\r[Recording] Episode {episode_num + 1} — Time: {elapsed:.1f}s — Frames: {recorder.num_frames}", end="")
            # Auto-finalize if timeout reached
            if elapsed > args.episode_timeout:
                print(f"\n[Auto-finalize] Timeout ({args.episode_timeout}s) reached")
                recorder.set_notes("auto-finalized (timeout)")
                recorder.finalize()
                recorder.save()
                recorder = None
                episode_recording = False
                waiting_for_trigger = True

            # Button B: force-finalize current episode
            if buttonB and not prev_buttonB and episode_recording:
                print("\n[Force finalize] Button B pressed")
                recorder.set_notes("force-finalized (button B)")
                recorder.finalize()
                recorder.save()
                recorder = None
                episode_recording = False
                waiting_for_trigger = True

            # Keep recording
            if episode_recording:
                print(f"\r[Recording] Episode {episode_num + 1} — Time: {elapsed:.1f}s — Frames: {recorder.num_frames}", end="")
                # The noisy pose = the VR pose (real human teleop noise)
                noisy_pose = np.concatenate([target_pos_VR, target_rot_VR])

                # Compute gripper state (0=open, 1=closed)
                gripper_state = 1.0 if grip else 0.0

                recorder.step(
                    frame=wrist_rgb,
                    noisy_pose=noisy_pose,
                    gripper_state=gripper_state,
                    absolute_pose=None,  # same as noisy — real teleop is what we train on
                )
                elapsed = time.time() - recorder._start_time
            

        # ── 6. Episode start / end logic ──
        if waiting_for_trigger and buttonB and not prev_buttonB:
            # Start a new episode on first trigger press
            task_desc, target_name, active_objects = get_episode_config(
                objects, episode_num, args.num_episodes
            )
            episode_name = f"ep_{episode_num:04d}_{target_name}"

            recorder = DataRecorder(
                output_dir=str(output_dir),
                episode_name=episode_name,
                camera_label="wrist",
            )
            recorder.set_task_description(task_desc)
            recorder.set_target_object(target_name)
            recorder.set_object_poses(obj_positions)
            recorder.set_operator(args.operator)

            print(f"\n{'─'*50}")
            print(f"[Episode {episode_num + 1}/{args.num_episodes}] {episode_name}")
            print(f"  Task: {task_desc}")
            print(f"  Target: {target_name}")
            print(f"{'─'*50}")

            episode_recording = True
            waiting_for_trigger = False
            
        prev_buttonB = buttonB
        prev_buttonA = buttonA
        # Button A: reset (cancel current episode)
        if buttonA and not prev_buttonA:
            if recorder is not None and episode_recording:
                print("\n[Cancel] Button A pressed — discarding current episode")
                # Delete the episode directory
                import shutil
                ep_dir = output_dir / recorder.episode_name
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)

            recorder = None
            episode_recording = False
            waiting_for_trigger = True

            # Reset sim
            sim.stop()
            sim.reset()
            robot.initialize()
            base.initialize()
            franka_hand.initialize()
            robot.set_joint_positions(FRANKA_HOME)

            # Reset VR state
            pos_start_ee = None
            pos_start_ctrl = None
            quat_start_ctrl = None
            quat_start_ee = None
            pos_save = np.array([0.2, 0.0, 0.25])
            _, reset_ee_rot = franka_hand.get_world_pose()
            quat_save = reset_ee_rot
            prev_trigger = False
            prev_buttonA = False
            prev_buttonB = False
            sim.play()
            sim.step(render=True)

            # Re-spawn objects
            objects = spawn_objects()
            obj_positions = {name: info["position"].tolist() for name, info in objects.items()}

            t = 0
            continue

        # ── 7. IK solve and apply ──
        # No noise injection needed — real human teleop is already noisy enough
        action_target, success = kin_solver.compute_inverse_kinematics(
            target_position=target_pos_IK,
            target_orientation=target_rot_IK,
        )
        if success:
            robot.apply_action(action_target)

        # Update marker
        # target_pos_world = target_pos_IK + base_pos_world
        # target_marker.set_world_pose(
        #     position=target_pos_world,
        #     orientation=target_rot_IK,
        # )

        # ── 8. Episode completion ──
        # Episode ends when: trigger is released after having been held
        if episode_recording and recorder is not None and buttonB and not prev_buttonB and recorder.num_frames > 10:
            print(f"\n[Complete] Episode {episode_num + 1} finished — "
                  f"{recorder.num_frames} frames recorded")
            recorder.finalize()
            recorder.save()
            recorder = None
            episode_recording = False
            waiting_for_trigger = True
            episode_num += 1

            # Reset VR held pose for next episode
            pos_save = target_pos_VR.copy()
            quat_save = target_rot_VR.copy()

            print(f"\nReady for episode {episode_num + 1}/{args.num_episodes}")
            print("  Hold trigger to start...\n")

        t += 1

    # ── Cleanup ──
    print(f"\n{'='*60}")
    print(f"Collection complete: {episode_num} episodes saved")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"{'='*60}")

    mqtt_inst.loop_stop()
    simulation_app.close()


if __name__ == "__main__":
    main()