"""Stage 1 / step 5: aggregate run_lesions.py outputs.
Prints (1) detection / retention / false-lesion table per condition and lesion type (TV vs PnP, mean over denoiser seeds),
(2) AUC by lesion observability tercile, (3) paired differences PnP - TV; writes step5_summary.csv and step5_summary.png.
Usage: python summarize_step5.py [--dir results/step5]
"""
import argparse, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("--dir", default="results/step5"); a = p.parse_args()
pd.set_option("display.width", 250, "display.max_columns", 40)
M = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.dir}/lesion_*_metrics.csv"))], ignore_index=True)
FAMN = {"m": "MNIST-trained prior", "f": "Fashion-trained prior"}
M["prior"] = M.family.str[0].map(FAMN); M["cond"] = M.prior + " on " + M.test_set + " (" + M.role + ")"
M["ltype"] = "s" + M.sigma_l.astype(str) + "/c" + M.contrast.astype(str)
g = M.groupby(["cond", "ltype", "method"]).agg(pauc=("paired_auc", "mean"), pauc_sd=("paired_auc", "std"), auc=("auc", "mean"), auc_sd=("auc", "std"), retention=("retention", "mean"),
                                                 false_per_img=("false_per_img", "mean"), obs=("obs_mean", "mean"), n_valid=("n_valid", "mean"), n_models=("model", "nunique")).reset_index()
g.to_csv(f"{a.dir}/step5_summary.csv", index=False)

print("=== (1) lesion detection (paired AUC = same base image, location known; unpaired AUC pools backgrounds), contrast retention, false lesions per lesion-ABSENT image; mean over denoiser seeds ===")
print("    ltype = s<lesion std px>/c<contrast>;  n = images with a valid lesion (of those inserted);  retention 1 = lesion contrast preserved, <1 = suppressed;  obs = lesion observability (1 = fully measured)")
for cond, d in g.groupby("cond", sort=False):
    w = d.pivot(index="ltype", columns="method", values=["pauc", "auc", "retention", "false_per_img"])
    t = pd.DataFrame({"n": d[d.method == "TV"].set_index("ltype").n_valid, "obs": d[d.method == "TV"].set_index("ltype").obs,
                      "pAUC TV": w[("pauc", "TV")], "pAUC PnP": w[("pauc", "PnP-DnCNN")], "d_pAUC": w[("pauc", "PnP-DnCNN")] - w[("pauc", "TV")],
                      "AUC TV": w[("auc", "TV")], "AUC PnP": w[("auc", "PnP-DnCNN")],
                      "ret TV": w[("retention", "TV")], "ret PnP": w[("retention", "PnP-DnCNN")],
                      "false/img TV": w[("false_per_img", "TV")], "false/img PnP": w[("false_per_img", "PnP-DnCNN")]})
    print(f"\n-- {cond}  (n_models={int(d.n_models.max())})"); print(t.round(3).to_string())

print("\n=== (2) paired AUC by observability tercile of the lesion (terciles within each lesion type; same base images/locations in both arms), mean over models ===")
I = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.dir}/lesion_*_per_image.csv"))], ignore_index=True)
I = I[I.valid]
rows = []
for (mod, ts, sl, c, meth), d in I[I.method.isin(["TV", "PnP-DnCNN"])].groupby(["model", "test_set", "sigma_l", "contrast", "method"]):
    q = np.nanquantile(d.obs, [1 / 3, 2 / 3]); b = np.digitize(d.obs, q)
    for k, nm in enumerate(["low", "mid", "high"]):
        s = b == k
        if s.sum() > 5 and not d.obs[s].isna().all():
            rows.append(dict(model=mod, test_set=ts, sigma_l=sl, contrast=c, method=meth, tercile=nm, auc=float(np.mean((d.t_present[s].values > d.t_absent[s].values) + 0.5 * (d.t_present[s].values == d.t_absent[s].values))), obs=d.obs[s].mean()))
T = pd.DataFrame(rows); T["prior"] = T.model.str[0].map(FAMN)
tab = T.pivot_table(index=["prior", "test_set", "tercile"], columns="method", values=["auc", "obs"], aggfunc="mean")
print(tab.reindex(["low", "mid", "high"], level=2).round(3).to_string())

# figure: AUC and false lesions per image, TV vs PnP, per condition
conds = list(dict.fromkeys(g.cond)); lt = list(dict.fromkeys(g.ltype))
fig, axs = plt.subplots(2, len(conds), figsize=(3.6 * len(conds), 6.2), squeeze=False)
for j, cnd in enumerate(conds):
    d = g[g.cond == cnd]
    for r, (col, yl) in enumerate([("pauc", "paired detection AUC"), ("false_per_img", "false lesions / absent image")]):
        ax = axs[r, j]
        for k, (meth, colr) in enumerate([("TV", "tab:orange"), ("PnP-DnCNN", "tab:blue")]):
            v = d[d.method == meth].set_index("ltype").reindex(lt)
            ax.bar(np.arange(len(lt)) + (k - .5) * 0.38, v[col], 0.38, color=colr, label=meth, yerr=v.pauc_sd if col == "pauc" else None, capsize=2)
        ax.set_xticks(range(len(lt))); ax.set_xticklabels(lt, rotation=30, fontsize=7); ax.set_ylabel(yl, fontsize=8); ax.grid(alpha=.3, axis="y")
        if r == 0: ax.set_title(cnd.replace(" (", "\n("), fontsize=8); ax.set_ylim(0.4, 1.02)
axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{a.dir}/step5_summary.png", dpi=150); plt.close()
print(f"\nSaved: {a.dir}/step5_summary.csv, step5_summary.png")
