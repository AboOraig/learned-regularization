"""Paper figure for E5 (Fig. convergence): PnP iterates against the iteration count k (MNIST, 200 test digits).
Reads  results/iteration_control.csv and results/spectral_bands.csv   (both written by run_spectral.py)
Writes results/fig_e5_iterations.png
Usage: python make_fig_e5_iterations.py [--results DIR] [--out PATH] [--dataset NAME] [--sigma_d 0.1]"""
import argparse, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--results", default="results")
p.add_argument("--out", default="results/fig_e5_iterations.png")
p.add_argument("--dataset", default="MNIST (in-dist)")
p.add_argument("--sigma_d", type=float, default=0.1, help="denoiser strength shown in the band panel and marked as tuned")
a = p.parse_args()

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
d = pd.read_csv(f"{a.results}/iteration_control.csv"); b = pd.read_csv(f"{a.results}/spectral_bands.csv")
m = d[(d.dataset == a.dataset) & (d.method == "PnP-DnCNN")]
tv = d[(d.dataset == a.dataset) & (d.method == "TV")].psnr.iloc[0]
tuned = m[m.is_tuned]
cols = {0.02: "#9ecae1", 0.05: "#4292c6", 0.1: "#d95f02", 0.2: "#6a3d9a"}

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
for sd, c in cols.items():
    s = m[m.sigma_d == sd].sort_values("k")
    lw = 1.8 if sd == a.sigma_d else 1.2
    ax[0].plot(s.k, s.psnr, "-o", ms=3.5, color=c, label=f"sigma_d={sd}", lw=lw)
    ax[1].plot(s.k, s.resid_over_noise, "-o", ms=3.5, color=c, lw=lw)
ax[0].axhline(tv, color="k", ls="--", lw=1); ax[0].text(300, tv + 0.15, "TV-ADMM", fontsize=8, ha="right")
if len(tuned):
    t = tuned.iloc[0]; ax[0].plot([t.k], [t.psnr], "o", ms=10, mfc="none", mec="k"); ax[0].text(t.k * 1.15, t.psnr + 0.28, "tuned", fontsize=8)
ax[0].set_ylim(17, 24.8); ax[0].set_ylabel("PSNR (dB)"); ax[0].set_title("PSNR", fontsize=10)
ax[0].legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
ax[1].axhline(1, color="k", ls=":", lw=1); ax[1].set_ylabel("||Ax_k - y|| / (sigma sqrt(m))")
ax[1].set_title("Data residual relative to the noise norm", fontsize=10)

groups = [("< 0.05 (incl. null)", ("exact null", "<0.05"), "#bdbdbd"), ("0.05-0.1", ("0.05-0.1",), "#9ecae1"),
          ("0.1-0.2", ("0.1-0.2",), "#d95f02"), ("0.2-0.3", ("0.2-0.3",), "#e6ab02"), (">= 0.3", ("0.3-0.5", ">=0.5"), "#4d4d4d")]
pb = b[(b.dataset == a.dataset) & (b.method == "PnP-DnCNN") & (b.sigma_d == a.sigma_d)]
for lab, bands, c in groups:
    s = pb[pb.band.isin(bands)].groupby("k").err.sum()
    ax[2].plot(s.index, s.values, "-o", ms=3.5, color=c, label=lab, lw=1.6)
ax[2].set_ylim(0, 2.9); ax[2].set_ylabel("squared error in the band")
ax[2].set_title(f"Error by singular-value band (sigma_d={a.sigma_d:g})", fontsize=10)
ax[2].legend(frameon=False, fontsize=7.5, title="s_i / s_1", title_fontsize=8, loc="upper center", ncol=3)
for x in ax:
    x.set_xscale("log"); x.set_xlabel("iteration k"); x.set_xticks([1, 2, 5, 10, 25, 50, 100, 400]); x.set_xticklabels([1, 2, 5, 10, 25, 50, 100, 400]); x.grid(alpha=0.25)
plt.tight_layout(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); plt.savefig(a.out, dpi=200)
print("saved", a.out)
