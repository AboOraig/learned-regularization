"""Lesion-insertion test (Stage 1, step 5): tools for inserting a known lesion and measuring what a reconstruction does with it.

Terminology
  lesion         : additive Gaussian blob (peak 1, std sigma_l px) times a contrast c, inserted into a 28x28 base image and clipped to [0,1]
  local contrast : mean(image in disc of radius rho) - mean(image in the annulus rho < r <= rho + W_OUT); rho = ceil(1.5 sigma_l)
                   (a zero-sum kernel: a constant image gives exactly 0)
  detection      : lesion-known-exactly / location-known observer; statistic = local contrast at the (known) lesion centre.
                   paired AUC   = P(stat_present_i > stat_absent_i) over base images i (same background, independent noise): isolates what
                                  the reconstruction does with the lesion itself (primary score);
                   unpaired AUC = Mann-Whitney over all present vs all absent (background varies too, so it is lower for every method)
  false lesion   : in a LESION-ABSENT reconstruction, a local maximum of the contrast map with response >= theta (half the response of
                   the nominal lesion) that has no support in the ground truth (truth contrast >= theta/2 within support_radius px)
  observability  : fraction of the lesion's energy that lies OUTSIDE the weakly-determined subspace of the operator (1 = fully measured)
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

SIZE = 28
W_OUT = 3


def lesion_rho(sigma_l):
    return int(math.ceil(1.5 * sigma_l))


def margin(rho):
    return rho + W_OUT


def centre_range(sigmas, size=SIZE):
    """Allowed integer centre coordinates [lo, hi] so that the measurement kernel fits for every lesion size."""
    h = max(margin(lesion_rho(s)) for s in sigmas)
    return h, size - 1 - h


def gaussian_lesion(centers, sigma_l, size=SIZE):
    """centers: (B,2) integer tensor (row, col) -> (B, size*size), peak value 1 at the centre."""
    c = torch.as_tensor(centers, dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32, device=c.device), torch.arange(size, dtype=torch.float32, device=c.device), indexing="ij")
    d2 = (yy[None] - c[:, 0, None, None]) ** 2 + (xx[None] - c[:, 1, None, None]) ** 2
    return torch.exp(-d2 / (2 * sigma_l ** 2)).reshape(len(c), -1)


def insert_lesion(x, centers, sigma_l, contrast):
    """x: (B, 784) in [0,1]. Returns clip(x + contrast * lesion, 0, 1)."""
    return (x + contrast * gaussian_lesion(centers, sigma_l).to(x.device)).clamp(0, 1)


def contrast_kernel(rho, w_out=W_OUT):
    K = 2 * (rho + w_out) + 1
    ax = torch.arange(K, dtype=torch.float32) - (rho + w_out)
    r = torch.sqrt(ax[:, None] ** 2 + ax[None, :] ** 2)
    inside, ring = (r <= rho).float(), ((r > rho) & (r <= rho + w_out)).float()
    return inside / inside.sum() - ring / ring.sum()


def contrast_map(x, rho, size=SIZE):
    """(B, 784) -> (B, H', H') local-contrast response at every centre whose kernel fits ('valid' positions only)."""
    k = contrast_kernel(rho).to(x.device)[None, None]
    return F.conv2d(x.reshape(-1, 1, size, size), k).squeeze(1)


def contrast_at(x, rho, centers, size=SIZE):
    """Local contrast of each image at its own centre (row, col). Centres must lie within the valid range."""
    m = contrast_map(x, rho, size)
    h = margin(rho)
    c = torch.as_tensor(centers, device=x.device)
    return m[torch.arange(len(x), device=x.device), c[:, 0] - h, c[:, 1] - h]


def nominal_response(sigma_l, contrast, size=SIZE):
    """Response of the contrast kernel to a lesion alone, on an empty background (used to set the false-lesion threshold)."""
    rho = lesion_rho(sigma_l); c = size // 2
    les = contrast * gaussian_lesion(torch.tensor([[c, c]]), sigma_l, size)
    return contrast_at(les, rho, torch.tensor([[c, c]])).item()


def count_false_lesions(recon, truth, rho, theta, nms=5, support_radius=2, support_frac=0.5, size=SIZE):
    """Per image: (#unsupported candidate blobs, #candidate blobs) in `recon` (see module docstring)."""
    R, T = contrast_map(recon, rho, size), contrast_map(truth, rho, size)
    pool = lambda M, k: F.max_pool2d(M[:, None], k, 1, k // 2)[:, 0]
    cand = (R >= theta) & (R == pool(R, nms))
    support = pool(T, 2 * support_radius + 1) >= support_frac * theta
    return (cand & ~support).sum((1, 2)), cand.sum((1, 2))


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie)  (Mann-Whitney), arrays of statistics."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def bootstrap_auc_ci(pos, neg, n_boot=500, seed=0):
    """95% interval of the AUC, resampling base images (pos[i], neg[i] come from the same base image)."""
    rng = np.random.default_rng(seed); n = len(pos); pos, neg = np.asarray(pos), np.asarray(neg)
    v = [auc(pos[i], neg[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def observability(subs, e, tau):
    """1 - ||P_weak e||^2 / ||e||^2 per row of e (lesion = inserted-minus-base image); nan if e == 0."""
    e = np.asarray(e.detach().cpu().double() if torch.is_tensor(e) else e, float)
    w = subs.project(e, tau)
    en = (e ** 2).sum(1)
    return np.where(en > 0, 1.0 - (w ** 2).sum(1) / np.where(en > 0, en, 1), np.nan)
