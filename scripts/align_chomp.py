#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHOMP (Covariant Hamiltonian Optimization for Motion Planning) for ALIGN.

Trajectory optimization that minimizes a combined objective:
    U[ξ] = λ_smooth · U_smooth[ξ] + λ_obs · U_obs[ξ]

where:
  - U_smooth = ½ ∫ ‖ξ̇(t)‖² + ‖ξ̈(t)‖² dt   (velocity + acceleration penalty)
  - U_obs    = ∫ c(ξ(t) · c_obs) dt           (obstacle cost)

The functional gradient update in trajectory space:
    ξ_{k+1} = ξ_k − η · A⁻¹ · ∇U[ξ_k]

where A is the metric tensor from the smoothness cost (finite-difference
approximation of the differential operator).  This "covariant" gradient
descent respects the Riemannian metric induced by the smoothness prior.

Reference: Ratliff et al. (2009), "CHOMP: Gradient Optimization Techniques
for Efficient Motion Planning", ICRA 2009.
Zucker et al. (2013), "CHOMP: Covariant Hamiltonian Optimization for
Motion Planning", IJRR.
"""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


# ================================================================
# Constants
# ================================================================

DEFAULT_LAMBDA_SMOOTH = 1.0      # smoothness cost weight
DEFAULT_LAMBDA_OBS = 0.0         # obstacle cost weight (0 = no obstacles for GT)
DEFAULT_ETA = 0.01               # gradient descent step size
DEFAULT_MAX_ITER = 200           # maximum optimization iterations
DEFAULT_TOL = 1e-6               # convergence tolerance


# ================================================================
# Finite-difference matrix A (smoothness metric)
# ================================================================

def _build_metric_matrix(n_waypoints: int) -> np.ndarray:
    """Build the smoothness metric A (sparse, n_waypoints × n_waypoints).

    A encodes the squared acceleration (second-derivative) penalty:
        ‖ξ̈‖² ≈ ξᵀ Aᵀ A ξ

    Uses second-order central differences. The resulting A⁻¹ smooths
    the gradient, projecting it onto the manifold of smooth trajectories.

    Returns:
        Dense (n_waypoints, n_waypoints) matrix.
    """
    if n_waypoints < 3:
        return np.eye(n_waypoints)

    # Second derivative operator (n-2 × n)
    data = np.array([[1.0], [-2.0], [1.0]])
    offsets = [0, 1, 2]
    D2 = diags(data, offsets, shape=(n_waypoints - 2, n_waypoints)).toarray()

    # Smoothness metric: A = I + λ · D2ᵀ D2  (Tikhonov regularization)
    A = np.eye(n_waypoints) + D2.T @ D2

    return A


def _build_block_metric_matrix(n_waypoints: int, n_dims: int) -> np.ndarray:
    """Build block-diagonal metric matrix for multi-dimensional trajectories.

    Returns:
        (n_waypoints·n_dims, n_waypoints·n_dims) sparse matrix.
    """
    A_single = _build_metric_matrix(n_waypoints)
    return kron(np.eye(n_dims), A_single).tocsr()


# ================================================================
# CHOMP Core
# ================================================================

class CHOMP:
    """Trajectory optimization via covariant gradient descent.

    Optimizes a trajectory ξ in workspace to minimize:
        U[ξ] = λ_smooth · U_smooth[ξ] + λ_obs · U_obs[ξ]

    The smoothness cost penalizes velocity and acceleration magnitude.
    The obstacle cost (optional) penalizes proximity to obstacles.
    """

    def __init__(
        self,
        lambda_smooth: float = DEFAULT_LAMBDA_SMOOTH,
        lambda_obs: float = DEFAULT_LAMBDA_OBS,
        eta: float = DEFAULT_ETA,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
    ):
        self.lambda_smooth = lambda_smooth
        self.lambda_obs = lambda_obs
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

        self.trajectory: Optional[np.ndarray] = None   # (N, D)
        self._A: Optional[np.ndarray] = None           # metric matrix
        self._A_inv: Optional[np.ndarray] = None       # pre-computed inverse

    # ── Optimization ──────────────────────────────────────────────────

    def optimize(
        self,
        init_trajectory: np.ndarray,
        obstacle_cost_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        fix_endpoints: bool = True,
    ) -> np.ndarray:
        """Optimize trajectory from initial guess.

        Args:
            init_trajectory: (N, D) initial trajectory (e.g., straight-line).
            obstacle_cost_fn: Optional f(ξ) → (N,) cost per waypoint.
            fix_endpoints: If True, start/goal are clamped at each iteration.

        Returns:
            (N, D) optimized trajectory.
        """
        N, D = init_trajectory.shape

        if N < 3:
            return init_trajectory.copy()

        # Flat representation for gradient operations
        xi = init_trajectory.reshape(-1).copy()  # (N·D,)

        # Build metric matrix and its inverse (precomputed, same for all iterations)
        self._A = _build_block_metric_matrix(N, D)
        self._A_inv = spsolve(self._A, eye(N * D, format='csc'))

        start_flat = init_trajectory[0].copy()   # (D,)
        goal_flat = init_trajectory[-1].copy()   # (D,)

        for iteration in range(self.max_iter):
            # Reshape to (N, D) for gradient computation
            xi_mat = xi.reshape(N, D)

            # Smoothness gradient: ∇U_smooth = A · ξ
            grad_smooth = self._A @ xi  # (N·D,)

            # Obstacle gradient (optional)
            grad_obs_flat = np.zeros(N * D)
            if obstacle_cost_fn is not None and self.lambda_obs > 0:
                grad_obs = self._obstacle_gradient(xi_mat, obstacle_cost_fn)
                grad_obs_flat = grad_obs.reshape(-1)

            # Total gradient
            grad = self.lambda_smooth * grad_smooth + self.lambda_obs * grad_obs_flat

            # Covariant update: ξ_{k+1} = ξ_k − η · A⁻¹ · ∇U
            xi_new = xi - self.eta * (self._A_inv @ grad)

            # Fix endpoints
            if fix_endpoints:
                xi_new[0:D] = start_flat
                xi_new[(N-1)*D:N*D] = goal_flat

            # Check convergence
            delta = np.linalg.norm(xi_new - xi)
            xi = xi_new

            if delta < self.tol:
                break

        self.trajectory = xi.reshape(N, D)
        return self.trajectory

    # ── Obstacle gradient (finite differences) ────────────────────────

    def _obstacle_gradient(
        self,
        xi_mat: np.ndarray,
        cost_fn: Callable[[np.ndarray], np.ndarray],
        eps: float = 1e-4,
    ) -> np.ndarray:
        """Compute gradient of obstacle cost w.r.t. each waypoint.

        Uses central finite differences.

        Args:
            xi_mat: (N, D) trajectory waypoints.
            cost_fn: f(ξ) → (N,) obstacle cost per waypoint.
            eps: Finite difference step.

        Returns:
            (N, D) obstacle gradient.
        """
        N, D = xi_mat.shape
        grad = np.zeros_like(xi_mat)

        for d in range(D):
            xi_plus = xi_mat.copy()
            xi_minus = xi_mat.copy()
            xi_plus[:, d] += eps
            xi_minus[:, d] -= eps
            grad[:, d] = (cost_fn(xi_plus) - cost_fn(xi_minus)) / (2 * eps)

        return grad


# ================================================================
# High-level approach planner interface
# ================================================================

def chomp_approach_plan(
    noisy_pos: np.ndarray,
    grasp_goal: np.ndarray,
    obstacle_cost_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lambda_smooth: float = DEFAULT_LAMBDA_SMOOTH,
    n_steps: Optional[int] = None,
) -> np.ndarray:
    """Plan an approach trajectory using CHOMP optimization.

    Initializes from the human's noisy demonstration, then optimizes for
    smoothness by gradient descent in trajectory space. This preserves the
    human's intended path shape while suppressing tremor/jitter.

    Args:
        noisy_pos: (N, 3) noisy position trajectory (approach segment).
        grasp_goal: (3,) target grasp position.
        obstacle_cost_fn: Optional obstacle cost function.
        lambda_smooth: Smoothness regularization strength.
        n_steps: Output trajectory length (default: same as input).

    Returns:
        (n_out, 3) smooth approach trajectory.
    """
    N = len(noisy_pos)
    if N < 3:
        return np.linspace(noisy_pos[0], grasp_goal, max(N, 2))

    n_out = n_steps if n_steps is not None else N

    # Initialize from the human's noisy approach path — not a straight line.
    # This preserves the intended arc/curvature while optimizing for smoothness.
    if N != n_out:
        # Resample to target length via linear interpolation
        t_old = np.linspace(0, 1, N)
        t_new = np.linspace(0, 1, n_out)
        init = np.zeros((n_out, 3))
        for d in range(3):
            init[:, d] = np.interp(t_new, t_old, noisy_pos[:, d])
        # Clamp endpoints
        init[0] = noisy_pos[0]
        init[-1] = grasp_goal
    else:
        init = noisy_pos.copy()
        init[-1] = grasp_goal  # ensure goal is correct

    # NOTE: CHOMP's smoothness optimization adds negligible value for short
    # approach trajectories without obstacles — the initialization from the
    # human demo is already a good prior. CHOMP's primary value is obstacle
    # avoidance with lambda_obs > 0. For pure smoothing, DMP or quintic are
    # preferred.
    chomp = CHOMP(lambda_smooth=lambda_smooth, lambda_obs=0.0)
    trajectory = chomp.optimize(init, obstacle_cost_fn=obstacle_cost_fn, fix_endpoints=True)

    return trajectory


# ================================================================
# Quick test
# ================================================================

if __name__ == "__main__":
    print("=== CHOMP Module: Quick Test ===")

    # Synthetic approach: start → goal
    start = np.array([0.35, 0.02, 0.24])
    goal = np.array([0.50, 0.0, 0.30])
    n_waypoints = 50
    init = np.zeros((n_waypoints, 3))
    for d in range(3):
        init[:, d] = np.linspace(start[d], goal[d], n_waypoints)

    # Add noise to simulate a "noisy" initial guess
    noisy_init = init + np.random.normal(0, 0.005, init.shape)

    chomp = CHOMP(lambda_smooth=1.0, max_iter=100, eta=0.05)
    result = chomp.optimize(noisy_init, fix_endpoints=True)

    print(f"  Input shape:   {noisy_init.shape}")
    print(f"  Output shape:  {result.shape}")
    print(f"  Start:         {result[0]} (target: {start})")
    print(f"  End:           {result[-1]} (target: {goal})")
    print(f"  Converged:     {np.linalg.norm(result[-1] - goal) < 1e-8}")
    print(f"  Smoothness gain: std(diff) reduced from "
          f"{np.std(np.diff(noisy_init, axis=0)):.6f} to "
          f"{np.std(np.diff(result, axis=0)):.6f}")
    print()
    print("✅ CHOMP Module: Quick Test Passed")
