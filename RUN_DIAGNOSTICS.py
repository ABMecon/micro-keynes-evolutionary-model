"""
Minimal entry point for sanity-checking the transition kernel.

This script:
1) builds a discrete Markov transition matrix P(K'|K) on a user-supplied grid,
2) checks that each row sums to (approximately) one, and
3) produces diagnostic plots that overlay a few representative rows of P(K'|K).
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

    # Deep model parameters
    c: float = 0.18
    sigma_0: float = 0.005
    V: float = 0.1        # deep sentiment range (baseline)
    cb: float = 0.05      # deep inertia parameter
    alpha: float = 0.33
    eta: float = 7.0
    delta: float = 0.06

    # Discretization of within-branch variation
    J: int = 50
    u_mode: str = "midpoint"  # "midpoint" (recommended) or "grid"

    # Plot controls
    zoom_halfwidth: float = 0.8  # set None to disable zoom windowing

    @property
    def phi(self) -> float:
        """Derived neutral point parameter."""
        den = self.alpha * (self.eta - 1.0) + 1.0
        if den <= 0.0:
            raise ValueError("Invalid (alpha, eta): denominator for phi is nonpositive.")
        return self.alpha * (self.eta - 1.0) / den

    @property
    def b(self) -> float:
        """Derived slope parameter (for reporting only)."""
        if self.V <= 0.0:
            raise ValueError("V must be positive.")
        if self.cb <= 0.0:
            raise ValueError("cb must be positive.")
        return self.cb / self.V


def main() -> None:
    p = Params()

    outdir = Path(".")
    K_grid = np.linspace(p.K_min, p.K_max, p.nK)

    # Build transition matrix P(K'|K)
    P = tk.build_transition_matrix(
        K_grid=K_grid,
        sigma_0=p.sigma_0,
        V=p.V,                 # deep parameter
        phi=p.phi,             # derived
        c=p.c,
        alpha=p.alpha,
        eta=p.eta,
        delta=p.delta,
        cb=p.cb,               # deep parameter
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
        title=(
            rf"Representative rows of $P(K'\mid K)$ "
            rf"(c={p.c}, $\sigma_0$={p.sigma_0}, V={p.V}, c_b={p.cb}, b={p.b:.2f})"
        ),
    )

    f2 = outdir / "kernel_rows_overlay_zoom.png"
    plot_kernel_rows_zoom(
        P, K_grid,
        outpath=str(f2),
        window_halfwidth=p.zoom_halfwidth,
        title=(
            rf"Zoomed rows of $P(K'\mid K)$ "
            rf"(c={p.c}, $\sigma_0$={p.sigma_0}, V={p.V}, c_b={p.cb}, b={p.b:.2f})"
        ),
    )

    print(f"Saved: {f1.name}, {f2.name}")


if __name__ == "__main__":
    main()
