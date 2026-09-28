import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def per_image_metrics(x_hat, x, size=28):
    """x_hat, x: torch (B, n) in [0,1] scale. Returns arrays of PSNR, SSIM, rel. error."""
    xh = x_hat.clamp(0, 1).cpu().numpy().reshape(-1, size, size)
    xt = x.cpu().numpy().reshape(-1, size, size)
    psnr = np.array([peak_signal_noise_ratio(a, b, data_range=1.0) for a, b in zip(xt, xh)])
    ssim = np.array([structural_similarity(a, b, data_range=1.0, win_size=7) for a, b in zip(xt, xh)])
    rel = np.linalg.norm((xh - xt).reshape(len(xt), -1), axis=1) / (np.linalg.norm(xt.reshape(len(xt), -1), axis=1) + 1e-12)
    return psnr, ssim, rel
