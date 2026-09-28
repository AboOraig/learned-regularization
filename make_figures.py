"""Qualitative figures: reconstruction grid + error map, shift grid, PnP convergence.
Usage: python make_figures.py"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, reconstruct_all
from src.pnp import pnp_pgd
from src.data import load_mnist, load_fashion, load_emnist_letters, shepp_logan

dev = "cuda" if torch.cuda.is_available() else "cpu"
_, val, test = load_mnist(); val, test = val[:100].to(dev), test[:200].to(dev)
den = load_denoiser("results/dncnn.pt", dev)
op = RadonOperator(28, 20, device=dev); sig = 0.05
params = tune_all(op, den, val, add_noise(op.forward(val), sig, seed=0))
im = lambda t: t.detach().cpu().numpy().reshape(28, 28)


def show(ax, img, title=None, **kw):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, **kw) if "cmap" not in kw else ax.imshow(img, **kw)
    ax.axis("off"); ax.set_title(title, fontsize=9) if title else None


# Fig 1: in-distribution grid + error maps
idx = [0, 1, 2, 3]; x = test[idx]; y = add_noise(op.forward(x), sig, seed=5)
xs = reconstruct_all(op, den, y, params)
cols = ["GT"] + list(xs) + ["|PnP - GT|"]
fig, axs = plt.subplots(len(idx), len(cols), figsize=(2 * len(cols), 2 * len(idx)))
for r in range(len(idx)):
    show(axs[r, 0], im(x[r]), cols[0] if r == 0 else None)
    for c, (n, xr) in enumerate(xs.items(), 1):
        show(axs[r, c], im(xr[r].clamp(0, 1)), n if r == 0 else None)
    axs[r, -1].imshow(np.abs(im(xs["PnP-DnCNN"][r].clamp(0, 1)) - im(x[r])), cmap="hot", vmin=0, vmax=1)
    axs[r, -1].axis("off"); axs[r, -1].set_title(cols[-1] if r == 0 else None, fontsize=9)
plt.tight_layout(); plt.savefig("results/fig_recon_grid.png", dpi=150); plt.close()

# Fig 2: distribution shift (one example per dataset): GT | TV | PnP
sets = {"MNIST": test[:1], "Fashion-MNIST": load_fashion()[:1].to(dev),
        "EMNIST": load_emnist_letters()[:1].to(dev), "Shepp-Logan": shepp_logan().to(dev)}
fig, axs = plt.subplots(len(sets), 3, figsize=(6, 2 * len(sets)))
for r, (n, xx) in enumerate(sets.items()):
    rec = reconstruct_all(op, den, add_noise(op.forward(xx), sig, seed=9), params)
    for c, (t, img) in enumerate([("GT", xx), ("TV", rec["TV"]), ("PnP-DnCNN", rec["PnP-DnCNN"])]):
        show(axs[r, c], im(img.clamp(0, 1)), t if r == 0 else None)
    axs[r, 0].text(-0.1, 0.5, n, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("results/fig_shift_grid.png", dpi=150); plt.close()

# Fig 3: PnP convergence for several sigma_d (50 test images)
xt = test[:50]; yt = add_noise(op.forward(xt), sig, seed=7); eta = 1.0 / op.lipschitz()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for sd in [0.02, 0.05, 0.1, 0.2]:
    _, h = pnp_pgd(op, yt, den, sd, eta, (400,), x_true=xt)
    ax[0].plot(h["psnr"], label=f"sigma_d={sd}"); ax[1].plot(h["residual"], label=f"sigma_d={sd}")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("PSNR (dB)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("||Ax_k - y||"); ax[1].set_yscale("log"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/fig_convergence.png", dpi=150)
print("saved figures to results/")
