"""Differentiable soft-DTW distance for voltage-trace matching.

WHY (see docs / the voltage-only plan): pointwise voltage-MSE on a spiking trace
is the source of the "cliffy" loss landscape — a sub-millisecond spike-timing
shift produces a large MSE with a gradient that points sideways off the cliff, so
the spike-shaping conductances (gbar_na3, gkdrbar_kdr) get a noise-dominated
signal even though they are identifiable.  Soft-DTW (Cuturi & Blondel 2017)
*aligns* the two traces in time before penalising, turning a spike-timing cliff
into a smooth amplitude/shape gradient.

This is a self-contained PyTorch implementation (no external dependency), so it
runs inside the HybridLoss/jaxley training loop and backprops into `pred_unit`.

Design notes:
  * Batched, gamma-smoothed soft-DTW via the anti-diagonal recurrence, so the DP
    is O(N+M) sequential Python steps (each vectorised over the diagonal and the
    batch) instead of O(N*M).
  * A Sakoe-Chiba `band` bounds the warp (and the cost) — spikes may drift a few
    ms, not reorder — set it from `band_ms` at the compared resolution.
  * Traces are downsampled to `n_points` before the DP: full 5001-bin traces make
    an N*M cost matrix of 25M cells; ~200 points keeps it cheap and the coarse
    envelope is what carries the smooth alignment gradient.
  * The result is normalised by the aligned path length so `dtw_weight` is on a
    comparable scale to the z-MSE term regardless of `n_points`.
"""

from typing import Optional

import torch
import torch.nn.functional as F

_INF = 1.0e10


def _softmin3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """gamma-softmin of three tensors, numerically stable.

    softmin_gamma(x) = -gamma * logsumexp(-x / gamma).  As gamma -> 0 this
    approaches the hard min (classic DTW); larger gamma = smoother / more convex.
    """
    stacked = torch.stack([a, b, c], dim=0) * (-1.0 / gamma)   # (3, ...)
    return -gamma * torch.logsumexp(stacked, dim=0)


def soft_dtw(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0, band: int = 0) -> torch.Tensor:
    """Batched soft-DTW between two 1-D sequence batches.

    Args:
      x: (B, N) sequence, y: (B, M) sequence (same B).
      gamma: smoothing temperature (>0).  Smaller = closer to hard DTW.
      band: Sakoe-Chiba half-width in *points* at this resolution; 0 = no band
            (full warp).  Cells with |i - j| > band are forbidden.

    Returns:
      (B,) soft-DTW value (sum of squared distances along the soft-aligned path).
    """
    B, N = x.shape
    M = y.shape[1]
    D = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2)              # (B, N, M) squared cost
    # R[:, i, j] = soft-DTW of x[:i] vs y[:j]; padded with an INF border and R[0,0]=0.
    R = x.new_full((B, N + 1, M + 1), _INF)
    R[:, 0, 0] = 0.0
    for d in range(2, N + M + 1):
        i_lo = max(1, d - M)
        i_hi = min(N, d - 1)
        if band > 0:
            i_lo = max(i_lo, (d - band + 1) // 2)             # |2i - d| <= band
            i_hi = min(i_hi, (d + band) // 2)
        if i_lo > i_hi:
            continue
        ii = torch.arange(i_lo, i_hi + 1, device=x.device)    # (K,) rows on this diagonal
        jj = d - ii                                           # (K,) cols
        d_ij  = D[:, ii - 1, jj - 1]                          # (B, K) local cost
        r_diag = R[:, ii - 1, jj - 1]                         # (B, K) match  (diag d-2)
        r_up   = R[:, ii - 1, jj]                             # (B, K) insert (diag d-1)
        r_left = R[:, ii,     jj - 1]                         # (B, K) delete (diag d-1)
        R[:, ii, jj] = d_ij + _softmin3(r_diag, r_up, r_left, gamma)
    return R[:, N, M]


def _downsample(v: torch.Tensor, n_points: int) -> torch.Tensor:
    """Downsample (B, T) -> (B, n_points) by average pooling (anti-aliased)."""
    T = v.shape[-1]
    if n_points <= 0 or n_points >= T:
        return v
    return F.adaptive_avg_pool1d(v.unsqueeze(1), n_points).squeeze(1)


def soft_dtw_loss(
    v_sim: torch.Tensor,
    v_true: torch.Tensor,
    gamma: float = 0.1,
    n_points: int = 200,
    band_ms: float = 8.0,
    dt_ms: float = 0.1,
    orig_T: Optional[int] = None,
) -> torch.Tensor:
    """Mean, length-normalised soft-DTW distance between two (B, T) trace batches.

    Both traces are downsampled to `n_points` first; the Sakoe-Chiba band is
    `band_ms` converted to points at the *downsampled* resolution.  The raw
    soft-DTW (a sum over ~n_points aligned cells) is divided by `n_points` so the
    magnitude is comparable to a per-sample MSE and `dtw_weight` transfers across
    resolutions.
    """
    xs = _downsample(v_sim, n_points)
    ys = _downsample(v_true, n_points)
    N = xs.shape[-1]
    T = orig_T if orig_T is not None else v_true.shape[-1]
    # points-per-ms at the downsampled grid = N / (T * dt_ms)
    pts_per_ms = N / max(T * dt_ms, 1e-6)
    band = int(round(band_ms * pts_per_ms)) if band_ms and band_ms > 0 else 0
    band = max(band, 1) if band else 0
    d = soft_dtw(xs, ys, gamma=gamma, band=band)              # (B,)
    return d.mean() / float(N)


# ─────────────────────────────────────────────────────────────────────────────
# self-test: verify the anti-diagonal DP matches a brute-force reference and that
# gradients flow.  Run:  python -m toolbox.soft_dtw
# ─────────────────────────────────────────────────────────────────────────────
def _reference_soft_dtw(x, y, gamma, band=0):
    """Straightforward (slow) double-loop soft-DTW, for test cross-checking."""
    B, N = x.shape
    M = y.shape[1]
    D = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2)
    R = x.new_full((B, N + 1, M + 1), _INF)
    R[:, 0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if band > 0 and abs(i - j) > band:
                continue
            R[:, i, j] = D[:, i - 1, j - 1] + _softmin3(
                R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1], gamma)
    return R[:, N, M]


if __name__ == "__main__":
    torch.manual_seed(0)
    for (N, M, band) in [(6, 6, 0), (8, 5, 0), (10, 10, 3), (7, 9, 2)]:
        x = torch.randn(4, N, dtype=torch.float64)
        y = torch.randn(4, M, dtype=torch.float64)
        fast = soft_dtw(x, y, gamma=0.5, band=band)
        ref  = _reference_soft_dtw(x, y, gamma=0.5, band=band)
        err = (fast - ref).abs().max().item()
        print(f"N={N} M={M} band={band}: max|fast-ref|={err:.2e}  "
              f"{'OK' if err < 1e-8 else 'FAIL'}")
        assert err < 1e-8, "anti-diagonal DP disagrees with reference"

    # gradient sanity: identical traces -> ~0 loss and finite grad; a shifted
    # copy should cost far less under DTW than under MSE.
    x = torch.randn(3, 500, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone()
    L = soft_dtw_loss(x, y, gamma=0.1, n_points=100, band_ms=8.0, dt_ms=0.1, orig_T=500)
    L.backward()
    print(f"identical-trace dtw loss = {L.item():.3e}  grad_finite={torch.isfinite(x.grad).all().item()}")

    base = torch.randn(3, 500, dtype=torch.float64)
    shifted = torch.roll(base, shifts=5, dims=-1)              # 5-bin time shift
    dtw = soft_dtw_loss(base, shifted, gamma=0.1, n_points=250, band_ms=8.0, dt_ms=0.1, orig_T=500).item()
    mse = ((base - shifted) ** 2).mean().item()
    print(f"5-bin shift:  dtw={dtw:.4f}  mse={mse:.4f}  (dtw should be << mse)")
    print("soft_dtw self-test passed.")
