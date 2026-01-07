# diagnostics.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def check_row_sums(P: np.ndarray, tol: float = 1e-12) -> None:
    """Print diagnostic statistics about row sums of a transition matrix."""
    row_sums = P.sum(axis=1)
    max_abs_err = float(np.max(np.abs(row_sums - 1.0)))
    min_sum = float(np.min(row_sums))
    max_sum = float(np.max(row_sums))

    print(f"[diagnostic] row sums: min={min_sum:.16g}, max={max_sum:.16g}, max|sum-1|={max_abs_err:.3g}")
    if max_abs_err > tol:
        print(f"[warning] Row sums deviate from 1 beyond tol={tol}. Consider renormalization or checking kernel construction.")


def normalize_rows(P: np.ndarray) -> np.ndarray:
    """Robustly normalize rows to sum to 1 (in case of small numerical drift)."""
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return P / row_sums


def choose_representative_rows(K_grid: np.ndarray, mode: str = "quantiles") -> list[int]:
    """
    Choose representative K indices for row-plot diagnostics.

    mode:
      - "quantiles": ~10%, 50%, 90% of grid
      - "levels": fixed K values (edit below)
    """
    n = len(K_grid)
    if n < 3:
        return list(range(n))

    if mode == "quantiles":
        # 10%, 50%, 90% indices
        idxs = [int(round(0.10 * (n - 1))), int(round(0.50 * (n - 1))), int(round(0.90 * (n - 1)))]
        # ensure uniqueness and within bounds
        idxs = sorted(set(max(0, min(n - 1, i)) for i in idxs))
        return idxs

    if mode == "levels":
        # If you prefer specific K-values, edit these:
        K_targets = [K_grid[0] + 0.2 * (K_grid[-1] - K_grid[0]),
                     K_grid[0] + 0.5 * (K_grid[-1] - K_grid[0]),
                     K_grid[0] + 0.8 * (K_grid[-1] - K_grid[0])]
        idxs = [int(np.argmin(np.abs(K_grid - Kt))) for Kt in K_targets]
        idxs = sorted(set(idxs))
        return idxs

    raise ValueError("mode must be 'quantiles' or 'levels'.")


def plot_kernel_rows(
    P: np.ndarray,
    K_grid: np.ndarray,
    outpath: str,
    row_indices: list[int] | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    show: bool = False,
) -> None:
    """
    Overlay a few conditional distributions (rows) of P(K'|K) as curves over K'.

    - P: transition matrix, shape (nK, nK)
    - K_grid: grid for K and K'
    - row_indices: which rows (K values) to plot; default chooses 3 representative rows
    """
    P = np.asarray(P)
    K_grid = np.asarray(K_grid)

    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be square.")
    if P.shape[0] != len(K_grid):
        raise ValueError("P dimension must match len(K_grid).")

    if row_indices is None:
        row_indices = choose_representative_rows(K_grid, mode="quantiles")

    fig, ax = plt.subplots()

    # Plot each chosen row
    for i in row_indices:
        Ki = float(K_grid[i])
        ax.plot(K_grid, P[i, :], label=rf"$K={Ki:.3g}$")

    ax.set_xlabel(r"$K'$")
    ax.set_ylabel(r"$P(K'\mid K)$")
    if title:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_kernel_rows_zoom(
    P: np.ndarray,
    K_grid: np.ndarray,
    outpath: str,
    row_indices: list[int] | None = None,
    window_halfwidth: float | None = None,
    title: str | None = None,
    show: bool = False,
) -> None:
    """
    Same as plot_kernel_rows, but zoom each row around its own K value if window_halfwidth is provided.
    Useful when the kernel is very concentrated near the diagonal.
    """
    P = np.asarray(P)
    K_grid = np.asarray(K_grid)

    if row_indices is None:
        row_indices = choose_representative_rows(K_grid, mode="quantiles")

    fig, ax = plt.subplots()

    for i in row_indices:
        Ki = float(K_grid[i])

        if window_halfwidth is None:
            mask = slice(None)
        else:
            lo = Ki - window_halfwidth
            hi = Ki + window_halfwidth
            mask = (K_grid >= lo) & (K_grid <= hi)

        ax.plot(K_grid[mask], P[i, :][mask], label=rf"$K={Ki:.3g}$")

    ax.set_xlabel(r"$K'$")
    ax.set_ylabel(r"$P(K'\mid K)$")
    if title:
        ax.set_title(title)

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
