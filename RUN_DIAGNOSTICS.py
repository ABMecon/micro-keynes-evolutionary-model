# RUN_DIAGNOSTICS.py
"""
Minimal entry point for sanity-checking the transition kernel.

This script:
1) builds a discrete Markov transition matrix P(K'|K) on a user-supplied grid,
2) checks that each row sums to (approximately) one, and
3) produces diagnostic plots that overlay a few representative rows of P(K'|K).

Usage:
  python RUN_DIAGNOSTICS.py

Outputs:
  - kernel_rows_overlay.png
  - kernel_rows_overlay_zoom.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

import transition_kernel as tk
from diagnostics import check_row_sums, normalize_rows, plot_kernel_rows, plot_kernel_rows_zoom


@dataclass(frozen=True)
class Params:
    # Grid
    K_min: float = 1.0
    K_max: float = 10.0
    nK: int = 120

    # Model parameters (edit as needed)
    c: float = 0.18
    sigma_0: float = 0.005
    b: float = 0.7
    phi: float = 0.0
    cb: float = 0.0
    alpha: float = 0.33
    eta: float = 7.0
    delta: float = 0.06

    # Discretization of within-branch variation
    J: int = 50
    u_mode: str = "midpoint"  # "midpoint" (recommended) or "grid"

    # Plot controls
    zoom_halfwidth: float = 0.8  # set None to disable zoom windowing


def main() -> None:
    p = Params()

    outdir = Path(".")
    K_grid = np.linspace(p.K_min, p.K_max, p.nK)

    # Build transition matrix P(K'|K)
    P = tk.build_transition_matrix(
        K_grid=K_grid,
        sigma_0=p.sigma_0,
        b=p.b,
        phi=p.phi,
        c=p.c,
        alpha=p.alpha,
        eta=p.eta,
        delta=p.delta,
        cb=p.cb,
        J=p.J,
        u_mode=p.u_mode,
        normalize_rows=False,  # we'll normalize explicitly after checking
    )

    # Diagnostics + robust normalization
    check_row_sums(P)
    P = normalize_rows(P)
    check_row_sums(P)

    # Diagnostic plots (representative conditional distributions)
    f1 = outdir / "kernel_rows_overlay.png"
    plot_kernel_rows(
        P, K_grid,
        outpath=str(f1),
        title=rf"Representative rows of $P(K'\mid K)$ (c={p.c}, $\sigma_0$={p.sigma_0})",
    )

    f2 = outdir / "kernel_rows_overlay_zoom.png"
    plot_kernel_rows_zoom(
        P, K_grid,
        outpath=str(f2),
        window_halfwidth=p.zoom_halfwidth,
        title=rf"Zoomed rows of $P(K'\mid K)$ (c={p.c}, $\sigma_0$={p.sigma_0})",
    )

    print(f"Saved: {f1.name}, {f2.name}")


if __name__ == "__main__":
    main()
