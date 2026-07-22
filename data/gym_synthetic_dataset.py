#!/usr/bin/env python3
"""Synthetic datasets from Gym environments for ablation testing.

Supports:
  - CartPole-v1: 4-D state, 1-D action, visual of cart + pole
  - Pendulum-v1: 3-D state, 1-D action, visual of pendulum

Both produce batches compatible with the v3 ALIGNIntentionModel:
  - frames_window: (K, H, W, 3) uint8
  - robot_state_window: (K, 7) float32
  - actions_window: (K, 6) float32
"""
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


# ================================================================
# Image rendering (using simple shapes, no external deps)
# ================================================================

def render_cartpole(cart_pos, cart_vel, pole_angle, pole_vel, image_size=64):
    """Render CartPole as a simple image.

    Layout:
    - Cart: white rectangle at bottom
    - Pole: line from cart top, angle determines direction
    - Track: horizontal line
    """
    img = np.full((image_size, image_size, 3), 200, dtype=np.uint8)  # light gray bg

    # Map state to image coords
    # cart_pos in [-2.4, 2.4] -> [0, image_size]
    cx = int((cart_pos + 2.4) / 4.8 * image_size)
    cx = max(5, min(image_size - 5, cx))
    cy = image_size - 10  # cart sits near bottom

    # Track
    img[cy + 5:cy + 7, 0:image_size] = [50, 50, 50]

    # Cart (white rectangle)
    cart_w = 10
    cart_h = 5
    img[cy:cy + cart_h, max(0, cx - cart_w):min(image_size, cx + cart_w)] = [255, 255, 255]

    # Pole: line from top-center of cart, length scales with angle
    pole_len = 30
    angle_rad = pole_angle
    # Pole points up by default; rotate by angle
    dx = int(pole_len * np.sin(angle_rad))
    dy = -int(pole_len * np.cos(angle_rad))
    pole_top_x = cx + dx
    pole_top_y = cy + dy
    # Draw line via Bresenham-ish (simple: just plot pixels along the line)
    steps = max(abs(dx), abs(dy), 1)
    for s in range(steps + 1):
        t = s / steps
        px = int(cx * (1 - t) + pole_top_x * t)
        py = int(cy * (1 - t) + pole_top_y * t)
        if 0 <= px < image_size and 0 <= py < image_size:
            img[py, px] = [255, 0, 0]  # red pole

    return img


def render_pendulum(theta, theta_dot, image_size=64):
    """Render Pendulum as a simple image.

    Layout:
    - Pivot: black dot at center
    - Arm: line from pivot, length fixed, angle = theta
    - Tip: small circle at end of arm
    """
    img = np.full((image_size, image_size, 3), 220, dtype=np.uint8)  # light gray bg

    cx, cy = image_size // 2, image_size // 2
    arm_len = image_size // 2 - 5

    # theta=0 points up; we render as if theta is angle from "up"
    # Pendulum convention: theta=0 is up (north), positive is counter-clockwise
    dx = int(arm_len * np.sin(theta))
    dy = -int(arm_len * np.cos(theta))
    tip_x = cx + dx
    tip_y = cy + dy

    # Draw arm (line)
    steps = max(abs(dx), abs(dy), 1)
    for s in range(steps + 1):
        t = s / steps
        px = int(cx * (1 - t) + tip_x * t)
        py = int(cy * (1 - t) + tip_y * t)
        if 0 <= px < image_size and 0 <= py < image_size:
            img[py, px] = [0, 0, 200]  # blue arm

    # Pivot
    img[cy - 2:cy + 3, cx - 2:cx + 3] = [0, 0, 0]

    # Tip
    img[max(0, tip_y - 2):min(image_size, tip_y + 3),
        max(0, tip_x - 2):min(image_size, tip_x + 3)] = [255, 0, 0]

    return img


# ================================================================
# Episode generation
# ================================================================

def collect_episode_cartpole(K, seed=None, max_steps=200):
    """Collect one CartPole episode with K timesteps.

    Uses a slightly-better-than-random policy: push in the direction
    that would slow the pole's fall. This creates a learnable signal.
    """
    if not HAS_GYM:
        raise RuntimeError("gymnasium not installed")
    env = gym.make("CartPole-v1", render_mode=None)
    obs, _ = env.reset(seed=seed)
    frames = []
    states = []
    actions = []

    for t in range(K):
        # Render
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        img = render_cartpole(cart_pos, cart_vel, pole_angle, pole_vel)
        frames.append(img)
        states.append(obs)

        # Semi-expert policy: push in direction of pole_angle
        # If pole leans right (positive angle), push right (action=1)
        # If pole leans left (negative angle), push left (action=0)
        # Add some noise
        noise = np.random.uniform(-0.5, 0.5)
        a_score = pole_angle * 1.0 + pole_vel * 0.5 + noise
        action = 1 if a_score > 0 else 0
        actions.append([float(action), 0, 0, 0, 0, 0])

        obs, _, done, _, _ = env.step(action)
        if done and t < K - 1:
            # Reset and continue
            obs, _ = env.reset()
    env.close()

    return (
        np.stack(frames),        # (K, H, W, 3) uint8
        np.array(states),        # (K, 4) float
        np.array(actions),       # (K, 6) float
    )


def collect_episode_pendulum(K, seed=None, max_steps=200):
    """Collect one Pendulum episode with K timesteps.

    Uses a simple policy: torque proportional to angle (proportional control).
    """
    if not HAS_GYM:
        raise RuntimeError("gymnasium not installed")
    env = gym.make("Pendulum-v1", render_mode=None)
    obs, _ = env.reset(seed=seed)
    frames = []
    states = []
    actions = []

    for t in range(K):
        # Pendulum obs: [cos(theta), sin(theta), theta_dot]
        cos_t, sin_t, theta_dot = obs
        theta = np.arctan2(sin_t, cos_t)
        img = render_pendulum(theta, theta_dot)
        frames.append(img)
        states.append(obs)

        # Proportional control: torque = -2.0 * theta - 0.5 * theta_dot
        # action is in [-2, 2], applied as continuous torque
        action_val = -2.0 * theta - 0.5 * theta_dot
        action_val = np.clip(action_val, -2.0, 2.0)
        # Add small noise
        action_val += np.random.uniform(-0.1, 0.1)
        actions.append([action_val, 0, 0, 0, 0, 0])

        obs, _, done, _, _ = env.step([action_val])
        if done and t < K - 1:
            obs, _ = env.reset()
    env.close()

    return (
        np.stack(frames),        # (K, H, W, 3) uint8
        np.array(states),        # (K, 3) float
        np.array(actions),       # (K, 6) float
    )


# ================================================================
# Dataset
# ================================================================

class GymSyntheticDataset(Dataset):
    """Dataset of Gym episodes, formatted for v3 ALIGNIntentionModel.

    Output per item:
      - frames_window: (K, 64, 64, 3) uint8 — K past frames
      - robot_state_window: (K, 7) float32 — K past states (padded to 7)
      - actions_window: (K, 6) float32 — K past actions (padded to 6)

    Args:
        env_name: "CartPole-v1" or "Pendulum-v1"
        n_samples: number of episodes to generate
        K: window size (chunk size)
        image_size: 64
        seed: random seed
    """
    def __init__(self, env_name="CartPole-v1", n_samples=200, K=10,
                 image_size=64, seed=0):
        self.env_name = env_name
        self.n_samples = n_samples
        self.K = K
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)  # for gym's internal randomness

        # Determine state dim
        if "CartPole" in env_name:
            self.state_dim = 4
            self.collect_fn = collect_episode_cartpole
        elif "Pendulum" in env_name:
            self.state_dim = 3
            self.collect_fn = collect_episode_pendulum
        else:
            raise ValueError(f"Unknown env: {env_name}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seed = (self.rng.integers(0, 1_000_000)).item()
        frames, states, actions = self.collect_fn(self.K, seed=seed)

        # Pad state to 7-D
        state_padded = np.zeros((self.K, 7), dtype=np.float32)
        state_padded[:, :self.state_dim] = states.astype(np.float32)

        return {
            'frames_window': frames.astype(np.uint8),
            'robot_state_window': state_padded,
            'actions_window': actions.astype(np.float32),
        }
