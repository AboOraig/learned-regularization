"""Paper figure for E1-E2 (Fig. e1e2gap): PnP - TV PSNR gap against TV-ADMM and TV-box, crime-free data.
Reads  results/control/tv_box_fine5c_summary.csv   (written by run_tv_box_control.py --data fine5c)
Writes results/fig_e1e2_gap.png
Usage: python make_fig_e1e2_gap.py [--csv PATH] [--out PATH]"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--csv", default="results/control/tv_box_fine5c_summary.csv")
p.add_argument("--out", default="results/fig_e1e2_gap.png")
a = p.parse_args()

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
f = pd.read_csv(a.csv)
noise = f[f.n_angles == 20].sort_values("sigma")
ang = f[f.sigma == 0.05].sort_values("n_angles")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))

def panel(axis, d, labels, xlabel):
    x = np.arange(len(d))
    for k, c, lab, o in [("TVADMM", "#1f77b4", "vs TV-ADMM (unconstrained)", -0.06),
                         ("TVbox", "#d95f02", "vs TV-box (box-constrained)", 0.06)]:
        y, lo, hi = d["d_PnP-" + k].values, d["lo_PnP-" + k].values, d["hi_PnP-" + k].values
        axis.errorbar(x + o, y, yerr=[y - lo, hi - y], color=c, marker="o", ms=4, capsize=2.5, lw=1.4, label=lab)
    axis.axhline(0, color="k", lw=0.8)
    axis.set_xticks(x); axis.set_xticklabels(labels)
    axis.set_xlabel(xlabel); axis.set_ylabel("PnP - TV  PSNR gap (dB)"); axis.grid(axis="y", alpha=0.25)

panel(ax[0], noise, [f"{s:g}" for s in noise.sigma], "noise level sigma (20 angles)")
panel(ax[1], ang, [str(int(k)) for k in ang.n_angles], "number of angles (sigma = 0.05)")
ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
ax[0].set_title("Noise sweep", fontsize=10); ax[1].set_title("Angle sweep", fontsize=10)
plt.tight_layout(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); plt.savefig(a.out, dpi=200)
print("saved", a.out)
