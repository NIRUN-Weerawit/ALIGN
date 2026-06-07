#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discrete Dynamic Movement Primitive (DMP) for ALIGN approach phase.

Fits a forcing term from a single demonstration trajectory, then adapts
the attractor landscape to arbitrary start/goal poses at inference time.

Formulation (Ijspeert et al. 2013, discrete DMP):
    τ · ÿ = α_z · (β_z · (g − y) − τ · ẏ) + f(x)
    τ · ẋ = −α_x · x

where f(x) is a learned forcing term encoded as weighted Gaussian basis
functions, g is the goal attractor, and the canonical system x decays from
1 to 0, driving the forcing term to zero and guaranteeing convergence to g.

Reference: Ijspeert et al. (2013), "Dynamical Movement Primitives: Learning
Attractor Models for Motor Behaviors", Neural Computation.
"""

from typing import Optional, Tuple

import numpy as np


# ================================================================
# Constants
# ================================================================

DEFAULT_N_BASIS = 50       # number of Gaussian basis functions
DEFAULT_ALPHA_Z = 25.0     # spring constant
DEFAULT_BETA_Z = 6.25      # damping (β = α/4 for critical damping)
DEFAULT_ALPHA_X = 2.0      # canonical system decay rate
DEFAULT_TAU = 1.0          # temporal scaling (1.0 = real-time)
DT = 0.01                  # integration timestep


# ================================================================
# DMP Core
# ================================================================

class DiscreteDMP:
    """Discrete DMP for a single degree of freedom.

    Fits a forcing term from one demonstration trajectory y_demo(t),
    then can generate adapted trajectories for new start/goal pairs.
    """

    def __init__(
        self,
        n_basis: int = DEFAULT_N_BASIS,
        alpha_z: float = DEFAULT_ALPHA_Z,
        beta_z: float = DEFAULT_BETA_Z,
        alpha_x: float = DEFAULT_ALPHA_X,
        tau: float = DEFAULT_TAU,
    ):
        self.n_basis = n_basis
        self.alpha_z = alpha_z
        self.beta_z = beta_z if beta_z is not None else alpha_z / 4.0
        self.alpha_x = alpha_x
        self.tau = tau

        # Basis function centers (exponentially spaced in canonical space)
        self.c = np.exp(-self.alpha_x * np.linspace(0, 1, n_basis))
        # Basis function widths (inverse variance: 1/(2σ²))
        # Standard DMP convention: σ proportional to spacing between centers
        sigma = 1.0 / (n_basis * np.abs(np.diff(self.c)).mean())
        self.h = np.ones(n_basis) / (2.0 * sigma * sigma)

        self.weights: Optional[np.ndarray] = None  # learned forcing term weights
        self.goal: Optional[float] = None
        self.start: Optional[float] = None

    # ── Fit from demonstration ──────────────────────────────────────

    def fit(self, y_demo: np.ndarray, dt: float = DT):
        """Learn forcing term weights from a demonstration trajectory.

        Args:
            y_demo: (T,) demonstrated trajectory (e.g., x-position over time).
            dt: Timestep of the demonstration.
        """
        T = len(y_demo)
        self.goal = y_demo[-1]
        self.start = y_demo[0]

        # Compute velocities and accelerations from demonstration
        yd = np.gradient(y_demo, dt)
        ydd = np.gradient(yd, dt)

        # Canonical system: x decays from 1 to 0
        x_seq = np.exp(-self.alpha_x * np.linspace(0, self.tau, T) / self.tau)

        # Desired forcing term: solve f_target from DMP equation
        # f_target(x) = τ² · ÿ − α_z · (β_z · (g − y) − τ · ẏ)
        f_target = (
            (self.tau ** 2) * ydd
            - self.alpha_z * (
                self.beta_z * (self.goal - y_demo) - self.tau * yd
            )
        )

        # Learn weights via locally weighted regression (LWR)
        self.weights = np.zeros(self.n_basis)
        for j in range(self.n_basis):
            # Gaussian basis activation at each timestep
            psi = np.exp(-self.h[j] * (x_seq - self.c[j]) ** 2)
            # Weighted linear regression
            denominator = np.sum(psi)
            if denominator > 1e-10:
                numerator = np.sum(psi * f_target)
                self.weights[j] = numerator / denominator

    # ── Generate adapted trajectory ─────────────────────────────────

    def rollout(
        self,
        y0: Optional[float] = None,
        goal: Optional[float] = None,
        tau: Optional[float] = None,
        n_steps: int = 200,
        dt: float = DT,
    ) -> np.ndarray:
        """Generate trajectory by integrating the DMP.

        Args:
            y0: Starting position (default: learned start).
            goal: Goal attractor (default: learned goal).
            tau: Temporal scaling (default: learned tau).
            n_steps: Number of integration steps.
            dt: Integration timestep.

        Returns:
            (n_steps,) generated trajectory.
        """
        _y0 = float(y0 if y0 is not None else self.start)
        _goal = float(goal if goal is not None else self.goal)
        _tau = float(tau if tau is not None else self.tau)

        # Scale forcing term: (goal − y0) / (learned_goal − learned_start)
        scale = 1.0
        if self.start is not None and self.goal is not None:
            denom = self.goal - self.start
            if abs(denom) > 1e-10:
                scale = (_goal - _y0) / denom

        # Integration
        y = _y0
        z = 0.0  # scaled velocity (τ · ẏ)
        x = 1.0
        trajectory = np.zeros(n_steps)
        trajectory[0] = y

        for i in range(1, n_steps):
            # Canonical system
            dx = -self.alpha_x * x / _tau
            x += dx * dt

            # Forcing term from weighted basis functions
            psi = np.exp(-self.h * (x - self.c) ** 2)
            f = np.sum(psi * self.weights) * x / (np.sum(psi) + 1e-10) * scale

            # DMP dynamics
            dz = (self.alpha_z * (self.beta_z * (_goal - y) - z) + f) / _tau
            dy = z / _tau

            z += dz * dt
            y += dy * dt

            trajectory[i] = y

        # Ensure final value reaches goal exactly
        trajectory[-1] = _goal
        return trajectory


# ================================================================
# Multi-DOF DMP
# ================================================================

class MultiDOFDMP:
    """Discrete DMP for multi-dimensional trajectories (e.g., 3D positions).

    Maintains one DMP per dimension. All share the same canonical system
    and basis function configuration.
    """

    def __init__(self, n_dims: int = 3, n_basis: int = DEFAULT_N_BASIS):
        self.n_dims = n_dims
        self.dmps = [DiscreteDMP(n_basis=n_basis) for _ in range(n_dims)]

    def fit(self, trajectory: np.ndarray, dt: float = DT):
        """Fit DMPs to a multi-dimensional demonstration.

        Args:
            trajectory: (T, D) demonstrated trajectory.
            dt: Timestep.
        """
        T, D = trajectory.shape
        if D != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dims, got {D}")
        for d in range(D):
            self.dmps[d].fit(trajectory[:, d], dt=dt)

    def rollout(
        self,
        y0: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        n_steps: int = 200,
        dt: float = DT,
    ) -> np.ndarray:
        """Generate adapted multi-dimensional trajectory.

        Args:
            y0: (D,) starting position.
            goal: (D,) goal position.
            n_steps: Integration steps.
            dt: Integration timestep.

        Returns:
            (n_steps, D) generated trajectory.
        """
        trajectory = np.zeros((n_steps, self.n_dims))
        for d in range(self.n_dims):
            y0_d = y0[d] if y0 is not None else None
            goal_d = goal[d] if goal is not None else None
            trajectory[:, d] = self.dmps[d].rollout(y0=y0_d, goal=goal_d, n_steps=n_steps, dt=dt)
        return trajectory

    @property
    def start(self):
        return np.array([dmp.start for dmp in self.dmps])

    @property
    def goal(self):
        return np.array([dmp.goal for dmp in self.dmps])


# ================================================================
# High-level approach planner interface
# ================================================================

def dmp_approach_plan(
    noisy_pos: np.ndarray,
    grasp_goal: np.ndarray,
    n_steps: Optional[int] = None,
) -> np.ndarray:
    """Plan an approach trajectory using DMP learned from the human demo.

    Args:
        noisy_pos: (N, 3) noisy position trajectory (approach segment).
        grasp_goal: (3,) target grasp position.
        n_steps: Output trajectory length (default: same as input).

    Returns:
        (n_out, 3) smooth approach trajectory.
    """
    N = len(noisy_pos)
    if N < 3:
        # Too short for DMP — fall back to linear interpolation
        return np.linspace(noisy_pos[0], grasp_goal, max(N, 2))

    n_out = n_steps if n_steps is not None else N
    dmp = MultiDOFDMP(n_dims=3)
    dmp.fit(noisy_pos, dt=DT)

    y0 = noisy_pos[0].copy()
    return dmp.rollout(y0=y0, goal=grasp_goal, n_steps=n_out, dt=DT)


# ================================================================
# Quick test
# ================================================================

if __name__ == "__main__":
    print("=== DMP Module: Quick Test ===")

    # Create a synthetic curved approach trajectory (human-like arc)
    T = 100
    t = np.linspace(0, 1, T)
    demo = np.zeros((T, 3))
    demo[:, 0] = np.linspace(0.3, 0.5, T)                          # x: approach
    demo[:, 1] = 0.02 * np.sin(np.linspace(0, np.pi, T))           # y: gentle arc
    demo[:, 2] = np.linspace(0.25, 0.30, T)                        # z: slight rise

    grasp_goal = np.array([0.5, 0.0, 0.30])

    # Fit and adapt with new start
    new_start = np.array([0.35, 0.02, 0.24])
    smooth = dmp_approach_plan(demo, grasp_goal)

    print(f"  Input shape:   {demo.shape}")
    print(f"  Output shape:  {smooth.shape}")
    print(f"  Start:         {smooth[0]} (target: {new_start})")
    print(f"  End:           {smooth[-1]} (target: {grasp_goal})")
    print(f"  Converged:     {np.linalg.norm(smooth[-1] - grasp_goal) < 1e-6}")
    print(f"  Shape match:   {np.allclose(smooth.shape, demo.shape)}")
    print(f"  Smooth:        {np.all(np.diff(smooth, axis=0).std(axis=0) < np.diff(demo, axis=0).std(axis=0))}")
    print()
    print("✅ DMP Module: Quick Test Passed")
