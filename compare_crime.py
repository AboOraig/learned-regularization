"""Stage 1 / step 3: paired comparison  inverse crime (v1)  vs  crime-free (fine-grid data).

Reads the outputs of run_nullspace.py from two runs with identical settings:
    python run_nullspace.py                                      -> nullspace_*.csv            (crime)
    python run_nullspace.py --fine_factor 5 --suffix _fine5      -> nullspace_*_fine5.csv      (no crime)
and writes results/crime_comparison.csv, results/crime_comparison.png and prints the tables that answer:
  (a) does PnP keep its advantage over TV (PSNR, SSIM)?   (b) how big is the model error relative to the noise?
  (c) do the null-space conclusions (error split, injection, alignment) survive?
Usage:  python compare_crime.py [--suffix _fine5] [--out results]
"""
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--suffix", default="_fine5")
p.add_argument("--out", default="results")
a = p.parse_args()
H = "tau0.05"
KEY = ["experiment", "dataset", "sigma", "n_angles"]

mm0 = pd.read_csv(f"{a.out}/nullspace_method_means.csv"); mm1 = pd.read_csv(f"{a.out}/nullspace_method_means{a.suffix}.csv")
sm0 = pd.read_csv(f"{a.out}/nullspace_pnp_vs_tv.csv"); sm1 = pd.read_csv(f"{a.out}/nullspace_pnp_vs_tv{a.suffix}.csv")
mm0, mm1 = mm0[mm0.subspace == H], mm1[mm1.subspace == H]
sm0, sm1 = sm0[sm0.subspace == H], sm1[sm1.subspace == H]
pd.set_option("display.width", 250, "display.max_columns", 40)

# ---- (a) PSNR / SSIM per method, crime vs no crime ------------------------------------------------------------------
def wide(mm, col):
    return mm.pivot_table(index=KEY, columns="method", values=col)

rows = []
for col in ("psnr", "ssim"):
    w0, w1 = wide(mm0, col), wide(mm1, col)
    for m in ("FBP", "Tikhonov", "TV", "PnP-DnCNN"):
        for k in w0.index:
            if k in w1.index:
                rows.append(dict(zip(KEY, k), metric=col, method=m, crime=w0.loc[k, m], no_crime=w1.loc[k, m]))
cmp_ = pd.DataFrame(rows); cmp_["delta"] = cmp_.no_crime - cmp_.crime
cmp_.to_csv(f"{a.out}/crime_comparison.csv", index=False)

ps = cmp_[cmp_.metric == "psnr"].pivot_table(index=KEY, columns="method", values=["crime", "no_crime"])
tab = pd.DataFrame({"TV crime": ps[("crime", "TV")], "TV fine": ps[("no_crime", "TV")],
                    "PnP crime": ps[("crime", "PnP-DnCNN")], "PnP fine": ps[("no_crime", "PnP-DnCNN")]})
tab["PnP-TV crime"] = tab["PnP crime"] - tab["TV crime"]; tab["PnP-TV fine"] = tab["PnP fine"] - tab["TV fine"]
tab["gap change"] = tab["PnP-TV fine"] - tab["PnP-TV crime"]
me = sm1.set_index(KEY).model_err_norm
tab["model err"] = me.reindex(tab.index).values
tab["noise norm"] = [s * np.sqrt(40 * n) for (_, _, s, n) in tab.index]        # m = 40 detector bins x n_angles for the 28x28 operator
tab["model/noise"] = tab["model err"] / tab["noise norm"].replace(0, np.nan)
print("=== PSNR (dB): inverse crime vs crime-free; PnP-TV gap and its change ===")
print(tab.round(2).to_string())

# ---- (b) hyperparameters re-tuned on the new data ---------------------------------------------------------------------
par = sm0.set_index(KEY)[["pnp_sd", "pnp_k", "tv_lam"]].join(sm1.set_index(KEY)[["pnp_sd", "pnp_k", "tv_lam"]], lsuffix="_crime", rsuffix="_fine")
print("\n=== tuned hyperparameters (crime -> crime-free) ===")
print(par.round(4).to_string())

# ---- (c) null-space conclusions ----------------------------------------------------------------------------------------
keep = ["null_err_ratio_PnP_over_TV", "range_err_ratio_PnP_over_TV", "total_err_ratio_PnP_over_TV", "inject_TV", "inject_PnP", "cos_pooled_TV", "cos_pooled_PnP"]
ns = sm0.set_index(KEY)[keep].join(sm1.set_index(KEY)[keep], lsuffix=" crime", rsuffix=" fine")
order = [c for k in keep for c in (f"{k} crime", f"{k} fine")]
print(f"\n=== null-space metrics at '{H}': crime vs crime-free (ratios are PnP/TV) ===")
print(ns[order].round(3).to_string())

# ---- figure: PSNR vs sigma / angles, solid = crime-free, dashed = crime -------------------------------------------------
sweeps = [(e, c, l) for e, c, l in (("noise", "sigma", "noise sigma"), ("angles", "n_angles", "number of angles")) if (cmp_.experiment == e).any()]
if sweeps:
    fig, axs = plt.subplots(1, len(sweeps), figsize=(6 * len(sweeps), 4), squeeze=False)
    COL = {"FBP": "tab:gray", "Tikhonov": "tab:green", "TV": "tab:orange", "PnP-DnCNN": "tab:blue"}
    for ax, (e, xc, xl) in zip(axs[0], sweeps):
        for m, c in COL.items():
            d = cmp_[(cmp_.experiment == e) & (cmp_.metric == "psnr") & (cmp_.method == m)].sort_values(xc)
            ax.plot(d[xc], d.crime, "--o", color=c, ms=3, alpha=.6); ax.plot(d[xc], d.no_crime, "-o", color=c, ms=4, label=m)
        ax.set_xlabel(xl); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=8); fig.text(0.5, -0.02, "solid = crime-free (fine-grid data), dashed = inverse crime (v1)", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{a.out}/crime_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"\nSaved: {a.out}/crime_comparison.csv, crime_comparison.png")
