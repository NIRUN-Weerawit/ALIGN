#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable cubic B-spline trajectory interpolation (pure PyTorch).

Interpolates a small number of control points C (default 5) into smooth, dense steps S at runtime. B-splines replace discrete chunk outputs with continuous differentiable splines so you can speed/slow execution without re-predicting the policy head.

Reference:
    Han et al., "B-spline Policy", arXiv 2607.09648

Shapes in/out:
     input knots (B, C, D)
   output traj (B, steps, D)
"""
import torch
import torch.nn as nn


class BsplineInterpolator(nn.Module):
    """Cubic B-spline evaluator.

    Args:
        num_knots : Number of control points C  (default 5; paper uses C=5 for K=10 horizon)

    Usage:
        interp = BsplineInterpolator(num_knots=5)
        knots  = head.predict(z_v, z_s, h)   # (B, C, D)
        traj   = interp(knots, steps=K)       # (B, K, D)
    """

    def __init__(self, num_knots: int = 5):
        super().__init__()
        assert num_knots >= 4, "Need at least 4 control points for a cubic B-spline"
        self.num_knots = num_knots
        self._p = 3   # cubic degree

    def forward(self, knots: torch.Tensor, steps: int) -> torch.Tensor:
        """Evaluate cubic B-spline at ``steps`` evenly spaced t-in-domain values.

        Args:
            knots: (B, C, D) — control points predicted by head
            steps: int       — dense evaluation resolution
        Returns:
            traj: (B, steps, D) — smooth interpolated trajectory
        """
        B, C, D = knots.shape
        assert C == self.num_knots

        u  = _make_clamped_knot_vector(C, device=knots.device)
        if steps <= 1:
            t_param = torch.zeros(B, 1, device=knots.device)
        else:
            line   = torch.linspace(0.0, float(C - 1), steps, device=knots.device)
            t_param = line.unsqueeze(0).expand(B, -1)

        wts = _cox_de_boor(t_param, u, self._p)         # (B, S, C)
        traj = torch.einsum('bsc,bcd->bsd', wts.to(knots.dtype), knots)
        return traj


# ============================================================================
# Internal helpers                                                           #
# ============================================================================

def _make_clamped_knot_vector(C: int, device):
    """Clamped uniform knot vector of length M = C+p+1 with p=3.

    Domain spans 0 to n where n=C-1 (index of last control point).
      Left clamp : U[0..3]   = 0.0
      Middle     : U[4 .. n-1] = evenly spaced integers
      Right clamp: U[n .. M-1] = float(n)
    """
    n = C - 1          # index of last control point
    M = C + 4          # p=3 -> length is C+4
    i = torch.arange(M, device=device).float()
    return torch.clamp(i - 3.0, min=0.0, max=float(n))


def _safe_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Stable division that returns zero whenever |b| < eps."""
    mask = (torch.abs(b) > 1e-8).to(dtype=a.dtype)
    return a / torch.clamp(torch.abs(b), min=1.0e-8) * mask


def _pad_right(x: torch.Tensor, new_width: int) -> torch.Tensor:
    """Pad last-axis of x with zeros until it has length ``new_width``."""
    if x.shape[-1] >= new_width:
        return x[..., :new_width]
    sz = [0]*2*x.dim()
    sz[-2] = 0; sz[-1] = new_width - x.shape[-1]
    return torch.nn.functional.pad(x, tuple([sz[-1], sz[-2]]))


def _cox_de_boor(t: torch.Tensor, u: torch.Tensor, p: int) -> torch.Tensor:
    """Cox-de Boor basis-weight recursion.

    Args:
        t : (B, S) — parameter values in [0.0, C-1]
        u : (M,)   — clamped knot vector, M = C+p+1
        p : int    — degree of spline (3 for cubic splines)

    Returns:
        N : (B, S, C) — per-control-point basis weights whose rows sum to 1.0.
    
    Recurrence from the standard textbook / Wikipedia:
       B_i,r(t) = w * B_(i,r-1)(t)  + v * B_(i+1,r-1)(t)

       where w = (t - U[i])        / (U[i+r] - U[i])           and 
             v = (U[i+r+1] - t)    / (U[i+r+1] - U[i+1])
    """ 
    B, S  = t.shape
    C     = len(u) - p - 1

    # --- Degree zero --------------------------------------------------------
    uL   = u[:C].view(1, 1, C)
    uR   = u[1:C+1].view(1, 1, C)
    tc   = t.view(B, S, 1).expand(-1, -1, C)

    N = ((tc >= uL) & (tc < uR)).to(t.dtype)          # B_{i,0}
    N[:, :, -1] = ((t >= u[-2]) & (t <= u[-1]+1e-6)).to(t.dtype)  # close last interval

    # --- Recursive steps r = 1 .. p ---------------------------------------- 
    for r in range(1, p+1):
        nr = C - r                        # surviving basis functions at level r

        tr = t.view(B, S, 1).expand(-1, -1, nr)   # (B,S,nr) broadcast-safe 

        w_i = _safe_div(tr - u[:nr].view(1,1,nr), 
                      (u[r:r+nr] - u[:nr]).view(1,1,nr))

        v_i = _safe_div((u[r+1:nr+2]).expand_as(tr)  - tr,
                      (u[r+1:nr+2] - u[1:1+nr]).view(1,1,nr))

        N = _pad_right(w_i * N[:, :, :nr].to(t.dtype) + v_i * N[:, :, 1:].to(t.dtype), new_width=C)

    return N
