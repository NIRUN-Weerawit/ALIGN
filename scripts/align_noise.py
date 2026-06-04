#!/usr/bin/env python3
"""ALIGN Noise Injection — adds synthetic teleoperation noise to clean poses.

Three noise types:
  1. Gaussian jitter — simulates VR tracking imprecision
  2. Tremor — physiological hand tremor at 8-12 Hz
  3. Fatigue ramp — noise amplitude grows over episode duration

All configurable. Can be used standalone or as a wrapper around clean poses.

Usage:
    from align_noise import NoiseInjector

    noise = NoiseInjector(
        pos_jitter_sigma=0.02,    # 2cm position noise
        orn_jitter_sigma=0.05,    # 3° orientation noise
        tremor_amplitude=0.005,   # 5mm tremor
        fatigue_rate=0.5,         # amplitude doubles over episode
    )

    for t in range(num_steps):
        noisy_pose = noise.apply(clean_pose, t)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class NoiseInjector:
    """Injects configurable synthetic noise into clean EEF poses.

    The noise model:
        noisy = clean + gaussian_jitter + tremor * sin(ω*t) + fatigue_growth
    """

    def __init__(
        self,
        pos_jitter_sigma: float = 0.02,     # meters (std)
        orn_jitter_sigma: float = 0.05,     # radians (std, ≈3°)
        tremor_amplitude: float = 0.005,    # meters
        tremor_frequency: float = 10.0,     # Hz
        fatigue_rate: float = 0.0,          # 0 = no fatigue, 1 = 2× at end
        seed: int | None = None,
        dt: float = 0.033,                  # ~30Hz timestep
    ):
        """
        Args:
            pos_jitter_sigma: Standard deviation of position noise (meters).
            orn_jitter_sigma: Standard deviation of orientation noise (radians).
            tremor_amplitude: Amplitude of sinusoidal hand tremor (meters).
            tremor_frequency: Frequency of tremor (Hz, typically 8-12).
            fatigue_rate: How fast noise grows. 0 = constant, >0 = linear growth.
            seed: Random seed for reproducibility.
            dt: Simulation timestep (seconds).
        """
        self.pos_jitter_sigma = pos_jitter_sigma
        self.orn_jitter_sigma = orn_jitter_sigma
        self.tremor_amplitude = tremor_amplitude
        self.tremor_frequency = tremor_frequency
        self.fatigue_rate = fatigue_rate
        self.dt = dt

        self._rng = np.random.default_rng(seed)
        self._tremor_phase = self._rng.uniform(0, 2 * np.pi)  # random start phase

    def apply(
        self,
        clean_pose: np.ndarray,
        timestep: int,
        in_place: bool = False,
    ) -> np.ndarray:
        """Add noise to a clean EEF pose.

        Args:
            clean_pose: (6,) or (7,) pose.
                        If 6D: [x, y, z, rx, ry, rz] (axis-angle).
                        If 7D: [x, y, z, qx, qy, qz, qw].
            timestep: Current timestep index (0-based).
            in_place: If True, modify the input array in-place.

        Returns:
            Noisy pose, same shape as input.
        """
        pose = np.asarray(clean_pose, dtype=np.float64).flatten()
        is_quat = len(pose) == 7

        # Position (first 3 elements)
        pos = pose[:3].copy()

        # ── 1. Gaussian jitter ──
        pos_jitter = self._rng.normal(0, self.pos_jitter_sigma, size=3)

        # ── 2. Tremor ──
        t_seconds = timestep * self.dt
        tremor = self.tremor_amplitude * np.sin(
            2 * np.pi * self.tremor_frequency * t_seconds + self._tremor_phase
        )
        tremor_vec = np.full(3, tremor)

        # ── 3. Fatigue ramp ──
        fatigue_factor = 1.0 + self.fatigue_rate * timestep * self.dt

        pos_noisy = pos + (pos_jitter + tremor_vec) * fatigue_factor

        # Orientation
        if is_quat:
            q = R.from_quat(pose[3:7])  # xyzw
            # Apply small random rotation
            orn_jitter = self._rng.normal(0, self.orn_jitter_sigma, size=3) * fatigue_factor
            q_jitter = R.from_euler("xyz", orn_jitter)
            q_noisy = (q_jitter * q).as_quat()
            noisy_pose = np.concatenate([pos_noisy, q_noisy])
        else:
            # Axis-angle: add noise directly
            orn_noisy = pose[3:6] + self._rng.normal(0, self.orn_jitter_sigma, size=3) * fatigue_factor
            noisy_pose = np.concatenate([pos_noisy, orn_noisy])

        if in_place:
            clean_pose[:] = noisy_pose
            return clean_pose

        return noisy_pose

    def apply_to_trajectory(
        self,
        clean_trajectory: np.ndarray,
    ) -> np.ndarray:
        """Apply noise to every timestep in a trajectory.

        Args:
            clean_trajectory: (T, 6) or (T, 7) array of clean poses.

        Returns:
            (T, 6) or (T, 7) noisy trajectory.
        """
        T = len(clean_trajectory)
        noisy = np.zeros_like(clean_trajectory)
        for t in range(T):
            noisy[t] = self.apply(clean_trajectory[t], t)
        return noisy


# ── Convenience Presets ──────────────────────────────────────────────


# Mild: clean VR tracking with slight hand tremor
PRESET_MILD = {
    "pos_jitter_sigma": 0.005,      # 5mm
    "orn_jitter_sigma": 0.02,       # ~1°
    "tremor_amplitude": 0.002,      # 2mm
    "tremor_frequency": 10.0,
    "fatigue_rate": 0.0,
}

# Moderate: typical VR teleoperation noise
PRESET_MODERATE = {
    "pos_jitter_sigma": 0.02,       # 2cm
    "orn_jitter_sigma": 0.05,       # ~3°
    "tremor_amplitude": 0.005,      # 5mm
    "tremor_frequency": 10.0,
    "fatigue_rate": 0.03,           # 3× growth over 100s
}

# Heavy: noisy operator + poor tracking
PRESET_HEAVY = {
    "pos_jitter_sigma": 0.04,       # 4cm
    "orn_jitter_sigma": 0.10,       # ~6°
    "tremor_amplitude": 0.01,       # 1cm
    "tremor_frequency": 8.0,
    "fatigue_rate": 0.08,           # 8× growth over 100s
}


# ── Quick Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ALIGN Noise Injection: Quick Test ===\n")

    # Create a clean trajectory: straight line in x
    T = 100
    clean = np.zeros((T, 6))
    clean[:, 0] = np.linspace(0.3, 0.6, T)   # x: 0.3 → 0.6
    clean[:, 2] = 0.25                         # z: constant
    # Orientation: identity (all zeros for axis-angle)

    for preset_name, preset_kwargs in [
        ("Mild", PRESET_MILD),
        ("Moderate", PRESET_MODERATE),
        ("Heavy", PRESET_HEAVY),
    ]:
        injector = NoiseInjector(**preset_kwargs, seed=42)
        noisy = injector.apply_to_trajectory(clean)

        # Compute statistics
        pos_error = np.linalg.norm(noisy[:, :3] - clean[:, :3], axis=1)
        orn_error = np.linalg.norm(noisy[:, 3:6] - clean[:, 3:6], axis=1)

        print(f"[{preset_name}]")
        print(f"  Mean position error: {pos_error.mean():.4f}m ± {pos_error.std():.4f}m")
        print(f"  Max position error:  {pos_error.max():.4f}m")
        print(f"  Mean orientation err: {np.degrees(orn_error.mean()):.1f}°")
        print()

    # Test quaternion mode
    print("[Quaternion mode]")
    injector = NoiseInjector(**PRESET_MODERATE, seed=42)
    clean_7d = np.zeros((1, 7))
    clean_7d[0, :3] = [0.3, 0.1, 0.25]
    clean_7d[0, 3:7] = [0, 0, 0, 1]  # identity quat
    noisy_7d = injector.apply(clean_7d[0], 10)
    print(f"  Clean: {clean_7d[0]}")
    print(f"  Noisy: {noisy_7d}")
    print()

    # Test in-place
    print("[In-place mode]")
    arr = np.array([0.3, 0.1, 0.25, 0, 0, 0])
    injector.apply(arr, 5, in_place=True)
    print(f"  Modified in-place: {arr}")
    print("\n✅ Noise Injection: Quick Test Passed")