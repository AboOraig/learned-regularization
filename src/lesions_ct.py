"""Lesion-insertion tools for 362x362 CT slices (extends src/lesions.py, which is written for 28x28).
Several well separated lesions of the SAME type are inserted in one slice (more detection samples per reconstruction);
all statistics are therefore clustered by slice (= patient) and bootstrapped over slices."""
import numpy as np
import torch
from scipy import ndimage
from . import lesions as L

SIZE = 362


def insert_multi(x, centers, valid, sigma_l, contrast, size=SIZE):
    """x: (B, size*size); centers (B,M,2) int tensor; valid (B,M) bool. Additive Gaussian blobs, clipped to [0,1]."""
    add = torch.zeros_like(x)
    for m in range(centers.shape[1]):
        blob = L.gaussian_lesion(centers[:, m], sigma_l, size).to(x.device)
        add = add + blob * valid[:, m:m + 1].to(x.device).float()
    return (x + contrast * add).clamp(0, 1)


def responses(x, rho, centers, size=SIZE):
    """Local-contrast response at each of the M centres of each image -> (B, M)."""
    m = L.contrast_map(x, rho, size)
    h = L.margin(rho)
    c = torch.as_tensor(centers, device=x.device)
    b = torch.arange(len(x), device=x.device)[:, None]
    return m[b, c[..., 0] - h, c[..., 1] - h]


def body_mask(img, margin=22):
    """Boolean mask of the interior of the body (holes such as lungs filled), eroded by `margin` px. img: (H,W) numpy."""
    m = ndimage.binary_fill_holes(ndimage.gaussian_filter(img, 2) > 0.1)
    return ndimage.binary_erosion(m, iterations=margin)


def square_mask(size=SIZE, lo=60, hi=301):
    m = np.zeros((size, size), bool); m[lo:hi + 1, lo:hi + 1] = True
    return m


def sample_centers(masks, M, min_dist, seed=0, tries=400):
    """masks: list of (H,W) bool arrays (one per image). Returns centers (B,M,2) int64 and valid (B,M) bool; lesions of one
    image are at least min_dist px apart; if fewer than M fit, the rest are marked invalid."""
    rng = np.random.default_rng(seed)
    B = len(masks); C = np.zeros((B, M, 2), np.int64); V = np.zeros((B, M), bool)
    for b, mk in enumerate(masks):
        pts = np.argwhere(mk)
        if len(pts) == 0:
            continue
        chosen = []
        for _ in range(tries):
            p = pts[rng.integers(len(pts))]
            if all(np.hypot(*(p - q)) >= min_dist for q in chosen):
                chosen.append(p)
            if len(chosen) == M:
                break
        for j, q in enumerate(chosen):
            C[b, j], V[b, j] = q, True
    return torch.tensor(C), torch.tensor(V)


def cluster_paired_auc(t1, t0, valid, idx=None):
    """t1,t0,valid: (S,M) arrays; pooled paired AUC over the valid lesions of the (resampled) slices idx."""
    if idx is not None:
        t1, t0, valid = t1[idx], t0[idx], valid[idx]
    v = valid.astype(bool)
    if v.sum() == 0:
        return float("nan")
    a, b = t1[v], t0[v]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def cluster_boot(fn, S, n_boot=2000, seed=0):
    """95% interval of fn(idx) over bootstrap resamples of the S slices."""
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, S, S)) for _ in range(n_boot)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


# ---------------------------------------------------------------- step 4b helpers
def unsupported_map(recon, truth, rho, nom, nms=5, support_radius=2, support_frac=0.5, size=SIZE):
    """Local-contrast map R of `recon`, mask of its local maxima (NMS window) that have NO support in the truth, where support is
    defined as in L.count_false_lesions at the base threshold 0.5*nom (kept fixed while the candidate threshold is swept)."""
    import torch.nn.functional as F
    R, T = L.contrast_map(recon, rho, size), L.contrast_map(truth, rho, size)
    pool = lambda M, k: F.max_pool2d(M[:, None], k, 1, k // 2)[:, 0]
    localmax = R == pool(R, nms)
    support = pool(T, 2 * support_radius + 1) >= support_frac * 0.5 * nom
    return R, localmax & ~support


def _stat(R, s, c, m, kind, idx):
    if kind == "auc":
        return cluster_paired_auc(R[f"t1|{s:g}|{c:g}|{m}"], R[f"t0|{s:g}|{c:g}|{m}"], R[f"valid|{s:g}|{c:g}"], idx)
    return float(R[f"fl|{s:g}|{c:g}|{m}"][idx].mean())


def paired_diff(R, types, mA, mB, kind, n_boot=2000, seed=0):
    """mA - mB on the SAME bootstrap resamples of slices (paired). `types` = list of (sigma, contrast); several types are averaged with
    equal weight. Returns (difference, lo, hi) of the 95% percentile interval."""
    S = len(R[f"psnr|{mA}"])
    stat = lambda idx, m: float(np.nanmean([_stat(R, s, c, m, kind, idx) for s, c in types]))
    fn = lambda idx: stat(idx, mA) - stat(idx, mB)
    lo, hi = cluster_boot(fn, S, n_boot, seed)
    return fn(np.arange(S)), lo, hi
