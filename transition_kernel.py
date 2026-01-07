# transition_kernel.py
"""
Markov transition kernel for the capital stock K.

This module constructs a discrete approximation to the one-step transition kernel
P(K' | K) implied by the model. The kernel is represented as a row-stochastic
matrix over a user-supplied grid K_grid.

Design principles (for reproducible prototypes):
- No plotting, no printing, no side effects on import.
- Pure functions: inputs -> outputs.
- Minimal dependencies (NumPy only).
"""

from __future__ import annotations

from typing import Literal
import numpy as np


# -----------------------------
# Core economic objects
# -----------------------------
def hat_r_K(K: np.ndarray, sigma_0: float, delta: float) -> np.ndarray:
    """
    Compute the implied interest rate r(K) (the 'hat r_K' mapping).

    Notes:
    - Uses a safe guard for very small sigma_0 * K to avoid division blow-ups.
    - Clips the discriminant at zero to avoid complex values due to round-off.
    """
    K = np.asarray(K, dtype=float)
    safe = np.where(sigma_0 * K < 1e-12, 1e-12, sigma_0 * K)
    disc = ((1.0 / safe) ** 2 - (1.0 + delta)) ** 2 - 4.0 * delta
    disc = np.maximum(disc, 0.0)
    return ((1.0 / safe) ** 2 - (1.0 + delta) - np.sqrt(disc)) / 2.0


def Kd_K(alpha: float, eta: float, delta: float, rK: np.ndarray, c: float, K: np.ndarray) -> np.ndarray:
    """Capital demand K^d as a function of K (as implemented in your current code)."""
    return alpha * (eta - 1.0) / eta / (rK + delta) * (c + delta) * K


def compute_zeta(K: np.ndarray, sigma_0: float, c: float, alpha: float, eta: float, delta: float) -> np.ndarray:
    """Compute zeta(K) = (util(K) - 1)/delta, where util = K^d / K."""
    K = np.asarray(K, dtype=float)
    rK = hat_r_K(K, sigma_0, delta)
    Kd = Kd_K(alpha, eta, delta, rK, c, K)
    util = Kd / K
    return (util - 1.0) / delta


# -----------------------------
# Branch probabilities
# -----------------------------
def p_func(b: float, phi: float, zeta: np.ndarray) -> np.ndarray:
    """Up probability p_+(K)."""
    return np.clip(0.5 - b * np.maximum(phi - zeta, 0.0), 0.0, 0.5)


def q_func(b: float, phi: float, cb: float, zeta: np.ndarray) -> np.ndarray:
    """Down probability p_-(K)."""
    return np.clip(0.5 - b * np.maximum(zeta + 1.0 + cb - phi, 0.0), 0.0, 0.5)


def s_func(b: float, phi: float, cb: float, zeta: np.ndarray) -> np.ndarray:
    """Stay probability p_0(K) = 1 - p_+(K) - p_-(K)."""
    return 1.0 - p_func(b, phi, zeta) - q_func(b, phi, cb, zeta)


# -----------------------------
# Discrete transition kernel
# -----------------------------
def build_transition_matrix(
    K_grid: np.ndarray,
    sigma_0: float,
    b: float,
    phi: float,
    c: float,
    alpha: float,
    eta: float,
    delta: float,
    cb: float,
    J: int = 40,
    u_mode: Literal["midpoint", "grid"] = "midpoint",
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Build a discrete transition matrix P over the supplied K_grid.

    The construction follows your current discrete-jump approximation:
    - Stay mass: p_0(K) at K' = K.
    - Up branch: K' = K * (1 + ((1 - p*u) * delta/(2b))) with u in [0,1] discretized.
    - Down branch: K' = K * (1 - ((1 - q*u) * delta/(2b))) with u in [0,1] discretized.
    - Each branch mass is allocated to neighboring grid points via linear interpolation.

    Parameters
    ----------
    K_grid : ndarray
        Monotone increasing grid for K (and K').
    J : int
        Number of quadrature points for u-discretization.
    u_mode : {"midpoint","grid"}
        Midpoint rule is recommended for stability.
    normalize_rows : bool
        If True, renormalize each row to sum to 1 (robust to round-off).

    Returns
    -------
    P : ndarray (nK, nK)
        Row-stochastic transition matrix.
    """
    K_grid = np.asarray(K_grid, dtype=float)
    if K_grid.ndim != 1 or len(K_grid) < 2:
        raise ValueError("K_grid must be a 1D array with length >= 2.")
    if not np.all(np.diff(K_grid) > 0):
        raise ValueError("K_grid must be strictly increasing.")
    if J <= 0:
        raise ValueError("J must be a positive integer.")

    n = len(K_grid)
    P = np.zeros((n, n), dtype=float)

    zeta = compute_zeta(K_grid, sigma_0, c, alpha, eta, delta)
    p_vals = p_func(b, phi, zeta)
    q_vals = q_func(b, phi, cb, zeta)
    s_vals = 1.0 - p_vals - q_vals

    # quadrature points on [0,1]
    if u_mode == "midpoint":
        U = (np.arange(J) + 0.5) / J
    elif u_mode == "grid":
        U = np.array([0.5]) if J == 1 else np.linspace(0.0, 1.0, J)
    else:
        raise ValueError("u_mode must be 'midpoint' or 'grid'.")

    def add_mass(i: int, Kprime: float, mass: float) -> None:
        # clamp to support
        if Kprime <= K_grid[0]:
            P[i, 0] += mass
            return
        if Kprime >= K_grid[-1]:
            P[i, -1] += mass
            return

        j = int(np.searchsorted(K_grid, Kprime))
        K_lo, K_hi = K_grid[j - 1], K_grid[j]
        w_hi = (Kprime - K_lo) / (K_hi - K_lo)
        w_lo = 1.0 - w_hi
        P[i, j - 1] += mass * w_lo
        P[i, j] += mass * w_hi

    for i, K in enumerate(K_grid):
        p = float(p_vals[i])
        q = float(q_vals[i])
        s = float(s_vals[i])

        # stay-put
        P[i, i] += s

        # up
        if p > 0.0:
            mass_each = p / J
            for u in U:
                Kp = (1.0 + ((1.0 - p * u) * delta / (2.0 * b))) * K
                add_mass(i, float(Kp), mass_each)

        # down
        if q > 0.0:
            mass_each = q / J
            for u in U:
                Km = (1.0 - ((1.0 - q * u) * delta / (2.0 * b))) * K
                add_mass(i, float(Km), mass_each)

    if normalize_rows:
        rs = P.sum(axis=1, keepdims=True)
        rs[rs == 0.0] = 1.0
        P = P / rs

    return P
