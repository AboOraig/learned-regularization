"""Merge baseline + PnP results: comparison plots (mean, 95% CI) and paired Wilcoxon tests.
Usage: python make_plots.py --exp noise"""
import argparse, pandas as pd, numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("--exp", default="noise"); a = p.parse_args()
key = "sigma" if a.exp == "noise" else "n_angles"
df = pd.concat([pd.read_csv(f"results/baselines_{a.exp}.csv"), pd.read_csv(f"results/pnp_{a.exp}.csv")])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, m in enumerate(["psnr", "ssim"]):
    for name, d in df.groupby("method"):
        g = d.groupby(key)[m].agg(["mean", "sem"])
        ax[k].errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", capsize=3, label=name)
    ax[k].set_xlabel(key); ax[k].set_ylabel(m.upper()); ax[k].grid(alpha=.3)
    if a.exp == "angles": ax[k].set_xscale("log")
ax[0].legend(); plt.tight_layout(); plt.savefig(f"results/comparison_{a.exp}.png", dpi=150)

print("Paired Wilcoxon, PnP vs TV (PSNR, same test images & noise):")
for v, d in df.groupby(key):
    pv = d[d.method == "PnP-DnCNN"].sort_values("img").psnr.values
    tv = d[d.method == "TV"].sort_values("img").psnr.values
    print(f"  {key}={v}: mean diff={np.mean(pv-tv):+.2f} dB, p={wilcoxon(pv, tv).pvalue:.2e}")
