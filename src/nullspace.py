"""Null-space / weakly-determined-subspace analysis of the sparse-view operator A (m x n).

Why: the measurements y = A x + n only constrain the components of x along the right singular
vectors of A with non-negligible singular values. The remaining directions (the exact null space and
the weakly-determined directions) are filled in by the PRIOR. Error and energy in that subspace is
therefore a direct, prior-attributable measure of "hallucination" (cf. Bhadra et al., 2021).

Notes on interpretation
- PnP-PGD's gradient step x - eta*A^T(Ax - y) never changes the exact-null-space component, so
  null-space content of a PnP output comes from the denoiser (and the FBP initialisation) only.
  This is checked in tests/test_nullspace.py.
- All metrics are computed on the reconstructions CLAMPED to [0, 1], exactly like PSNR/SSIM in
  src/metrics.py. Clamping is non-linear and can itself add null-space content; it is applied to
  every method identically.
- The subspace is defined by a threshold on singular values relative to the largest one, so it
  depends on the operator (angles, geometry) and on the threshold. Report several thresholds.
"""
import numpy as np
import torch

# name -> relative singular-value threshold tau. Subspace = {right singular vectors with s_i < tau * s_max}.
# "exact_null" is the numerically exact null space; the others add weakly-determined directions.
# NOTE: for the 28x28 / 20-angle operator the spectrum has 79 singular values at ~1e-16 (exact null), then a
# CONTINUUM from ~5e-9 to 1e-2 (relative) with no gap. A cut of 1e-10 isolates the 79 exact ones with a ~1e7 gap;
# a cut such as 1e-8 or 1e-6 would land inside the continuum and give a different (arbitrary) dimension.
SUBSPACES = {"exact_null": 1e-10, "tau0.05": 0.05, "tau0.10": 0.10}


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


class OperatorSubspaces:
    """SVD of the dense operator matrix A (m x n) and projections onto its weakly-determined subspaces."""

    def __init__(self, A):
        A = _np(A)
        m, n = A.shape
        _, S, Vt = np.linalg.svd(A, full_matrices=True)   # Vt: (n, n)
        self.n = n
        self.S = np.zeros(n)
        self.S[:len(S)] = S                                # if m < n the remaining singular values are exactly 0
        self.V = Vt.T                                      # columns: right singular vectors, decreasing s
        self.smax = float(S[0])

    def mask(self, tau):
        return self.S < tau * self.smax

    def dim(self, tau):
        return int(self.mask(tau).sum())

    def basis(self, tau):
        return self.V[:, self.mask(tau)]                   # (n, d)

    def project(self, X, tau):
        """Orthogonal projection of each row of X (B, n) onto the subspace."""
        Vw = self.basis(tau)
        X = _np(X)
        return (X @ Vw) @ Vw.T

    def gap_report(self, tau=SUBSPACES["exact_null"]):
        """Singular values just above/below the exact_null cut, to check the cut sits in a clear gap."""
        below = self.S[self.S < tau * self.smax]
        above = self.S[self.S >= tau * self.smax]
        return dict(n_null=len(below), largest_below=float(below.max()) if len(below) else 0.0,
                    smallest_above=float(above.min()) if len(above) else 0.0)


def subspace_metrics(subs, x_hat, x_true, tau):
    """Per-image arrays (length B) describing what happens inside the subspace.

    null_err_energy : ||P (x_hat - x)||^2        error the data cannot see (prior-driven)
    range_err_energy: ||(I-P) (x_hat - x)||^2    error in the directions the data DO constrain
    total_err_energy: ||x_hat - x||^2            (= null + range, orthogonal decomposition)
    null_err_frac   : null_err_energy / total_err_energy
    null_err_rel    : ||P (x_hat - x)|| / ||x||
    recon_energy    : ||P x_hat||^2              energy the method puts into the subspace
    true_energy     : ||P x||^2                  energy the true image actually has there
    inner           : <P x_hat, P x>             (so null_err = recon + true - 2*inner, exactly)
    cos_sub         : inner / (||P x_hat|| ||P x||)   alignment of the subspace content with the truth (NaN if undefined)
    inject_ratio    : recon_energy / true_energy      >1: method puts MORE energy there than the truth has (NaN if true=0)

    "Hallucination signature" = high inject_ratio together with low cos_sub: loud but uncorrelated content.
    Aggregate over images with POOLED quantities (ratio of means), not means of per-image ratios.
    """
    xh, xt = _np(x_hat), _np(x_true)
    e = xh - xt
    Pe, Pxh, Pxt = subs.project(e, tau), subs.project(xh, tau), subs.project(xt, tau)
    null_e = (Pe ** 2).sum(1)
    tot_e = (e ** 2).sum(1)
    range_e = ((e - Pe) ** 2).sum(1)
    recon_e, true_e = (Pxh ** 2).sum(1), (Pxt ** 2).sum(1)
    inner = (Pxh * Pxt).sum(1)
    denom = np.sqrt(recon_e * true_e)
    return dict(
        null_err_energy=null_e,
        range_err_energy=range_e,
        total_err_energy=tot_e,
        null_err_frac=null_e / np.maximum(tot_e, 1e-12),
        null_err_rel=np.sqrt(null_e) / np.maximum(np.sqrt((xt ** 2).sum(1)), 1e-12),
        recon_energy=recon_e,
        true_energy=true_e,
        inner=inner,
        cos_sub=np.where(denom > 1e-12, inner / np.maximum(denom, 1e-12), np.nan),
        inject_ratio=np.where(true_e > 1e-12, recon_e / np.maximum(true_e, 1e-12), np.nan),
    )


# ---------------------------------------------------------------------------------------------------------------
# Spectral band decomposition (step 2c): where along the singular-value spectrum does the error live?
# Bands are defined on s_i / s_max and partition ALL n directions: [0,1e-10) is the exact null space, the rest are
# increasingly well-measured directions. (Bands are disjoint; tau-subspaces above are unions of the first bands.)
# ---------------------------------------------------------------------------------------------------------------
BAND_EDGES = [0.0, 1e-10, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 + 1e-9]


def band_labels(edges=BAND_EDGES):
    labs = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        labs.append("exact null" if i == 0 else (f"<{hi:g}" if i == 1 else (f">={lo:g}" if i == len(edges) - 2 else f"{lo:g}-{hi:g}")))
    return labs


def band_masks(subs, edges=BAND_EDGES):
    r = subs.S / subs.smax
    return [(r >= lo) & (r < hi) for lo, hi in zip(edges[:-1], edges[1:])]


def band_metrics(subs, x_hat, x_true, edges=BAND_EDGES):
    """Per-image energies in each singular-value band (arrays of shape (B, n_bands)):
    err = ||band part of (x_hat - x)||^2, recon = ||band part of x_hat||^2, true = ||band part of x||^2,
    inner = <band part of x_hat, band part of x>.  dims: number of directions per band.
    Sum over bands of err equals the total error energy (V is orthogonal)."""
    Ch, Ct = _np(x_hat) @ subs.V, _np(x_true) @ subs.V
    masks = band_masks(subs, edges)
    err = np.stack([((Ch[:, m] - Ct[:, m]) ** 2).sum(1) for m in masks], 1)
    recon = np.stack([(Ch[:, m] ** 2).sum(1) for m in masks], 1)
    true = np.stack([(Ct[:, m] ** 2).sum(1) for m in masks], 1)
    inner = np.stack([(Ch[:, m] * Ct[:, m]).sum(1) for m in masks], 1)
    return dict(err=err, recon=recon, true=true, inner=inner, dims=np.array([int(m.sum()) for m in masks]))