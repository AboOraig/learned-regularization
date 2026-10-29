"""Stage 1 / step 5b: can PnP be re-tuned to keep small lesions without losing its PSNR advantage?

For each denoiser: sweep a grid of PnP operating points (sigma_d x iterations k) and of TV lambdas on
  * a SELECTION set  = held-out in-distribution validation images (disjoint from the images used for PSNR tuning), and
  * the TEST sets    = in-distribution test images and shifted test images (the same images, centres and noise seeds as step 5).
Operating points are CHOSEN on the in-distribution selection set only (as a practitioner would) and then reported on both test sets:
  PnP-psnr / TV-psnr  PSNR-tuned (the v1 / step-5 points)      PnP-lesion / TV-lesion  best mean paired AUC on the selection set
  PnP-constrained     best paired AUC among PnP points whose selection-set PSNR >= that of TV-psnr
Metrics per point: PSNR of the lesion-absent reconstruction, paired AUC, retention, false lesions/image (see src/lesions.py).
Outputs (results/step5b): oppoint_<model>_grid.csv (every point), oppoint_<model>_rules.csv (selected points on the test sets),
oppoint_<model>.png (PSNR vs paired-AUC frontier).  Usage: python run_oppoint.py [--models m03_s0,f03_s0]
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, set_sd_grid
from src.baselines import TVADMM
from src.pnp import pnp_pgd
from src.metrics import per_image_metrics
from src.finegrid import make_simulator
from src.data import load_mnist, load_fashion_train, load_fashion
from src import lesions as L
from src.oppoint import select_rules

p = argparse.ArgumentParser()
p.add_argument("--models", default="m03_s0,f03_s0")
p.add_argument("--ckpt_dir", default="results/step4")
p.add_argument("--out", default="results/step5b")
p.add_argument("--types", default="0.75:0.5,1.25:0.25", help="lesion types 'sigma_l:contrast' (must be among the step-5 types)")
p.add_argument("--pnp_sds", default="0.02,0.05,0.1,0.15,0.2")
p.add_argument("--pnp_ks", default="1,2,3,5,10,25,50,100,200")
p.add_argument("--tv_lams", default="0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05")
p.add_argument("--n_val", type=int, default=100, help="validation images used for PSNR tuning (as in step 5)")
p.add_argument("--n_sel", type=int, default=200, help="held-out validation images used for operating-point selection")
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--n_angles", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fine_factor", type=int, default=5)
p.add_argument("--up_mode", default="bicubic")
p.add_argument("--sd_grid", default="0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5", help="grid used by the PSNR tuning (as in step 5)")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
set_sd_grid([float(v) for v in a.sd_grid.split(",")])
ALL_SIZES, ALL_CONTR = [0.5, 0.75, 1.25, 2.0], [0.25, 0.5]
ALL_TYPES = [(s, c) for s in ALL_SIZES for c in ALL_CONTR]                 # same ordering as run_lesions.py (seeds depend on the index)
TYPES = [tuple(float(v) for v in t.split(":")) for t in a.types.split(",")]
TIDX = [ALL_TYPES.index(t) for t in TYPES]
LO, HI = L.centre_range(ALL_SIZES)
op = RadonOperator(28, a.n_angles, device=dev); sim = make_simulator(op, a.fine_factor, a.up_mode); eta = 1.0 / op.lipschitz()
mn_val, mn_test = load_mnist()[1], load_mnist()[2]; fa_val = load_fashion_train()[1]
TEST = {"mnist": ("MNIST", mn_test[:a.n_test]), "fashion": ("Fashion-MNIST", load_fashion()[:a.n_test])}
SDS = [float(v) for v in a.pnp_sds.split(",")]; KS = [int(v) for v in a.pnp_ks.split(",")]; LAMS = [float(v) for v in a.tv_lams.split(",")]


def make_arms(x, centers, seed_abs, seed_les):
    """Noisy data for the lesion-absent arm and for every lesion type (stacked: [absent, type1, type2, ...])."""
    xs = [x] + [L.insert_lesion(x, centers, s, c) for s, c in TYPES]
    ys = [add_noise(sim(x), a.sigma, seed=seed_abs)] + [add_noise(sim(xl), a.sigma, seed=seed_les + li) for xl, li in zip(xs[1:], TIDX)]
    return xs, torch.cat(ys)


def evaluate(den, x, centers, seed_abs, seed_les, tuned_sd, tuned_k, tuned_lam, set_label):
    """Metrics of every operating point on one image set. Returns a long DataFrame."""
    xs, Y = make_arms(x, centers, seed_abs, seed_les); n = len(x); rows = []

    def score(rec, meta):
        rec = rec.clamp(0, 1); parts = [rec[i * n:(i + 1) * n] for i in range(len(xs))]
        psnr = float(per_image_metrics(parts[0], x)[0].mean())
        for t, (s, c) in enumerate(TYPES):
            rows.append(dict(**meta, set=set_label, ltype=f"s{s}/c{c}", psnr=psnr, **L.arm_stats(x, xs[t + 1], centers, s, c, parts[0], parts[t + 1])))

    for sd in sorted(set(SDS + [tuned_sd])):
        outs, _ = pnp_pgd(op, Y, den, sd, eta, tuple(sorted(set(KS + [tuned_k]))))
        for k, rec in outs.items():
            score(rec, dict(method="PnP", sd=sd, k=k, lam=np.nan))
    tv = TVADMM(op)
    for lam in sorted(set(LAMS + [tuned_lam])):
        score(tv(Y, lam), dict(method="TV", sd=np.nan, k=0, lam=lam))
    return pd.DataFrame(rows)


for name in a.models.split(","):
    tstart = time.time(); fam, seed = name.rsplit("_s", 1); tuned_on = "fashion" if fam.startswith("f") else "mnist"
    den = load_denoiser(f"{a.ckpt_dir}/dn_{name}.pt", dev)
    val_all = fa_val if tuned_on == "fashion" else mn_val
    params = tune_all(op, den, val_all[:a.n_val].to(dev), add_noise(sim(val_all[:a.n_val].to(dev)), a.sigma, seed=a.seed))
    tsd, tk = params["PnP"]; tlam = float(params["TV"])
    print(f"\n[{name}] PSNR-tuned: PnP (sigma_d={tsd}, k={tk}), TV lambda={tlam:.4g}", flush=True)
    # selection set: in-distribution validation images disjoint from the tuning images
    xsel = val_all[a.n_val:a.n_val + a.n_sel].to(dev)
    csel = torch.tensor(np.random.default_rng(a.seed + 77).integers(LO, HI + 1, size=(len(xsel), 2)), device=dev)
    sel = evaluate(den, xsel, csel, a.seed + 1100, a.seed + 1200, tsd, tk, tlam, "selection")
    pts = sel.groupby(["method", "sd", "k", "lam"], dropna=False).agg(psnr=("psnr", "mean"), pauc=("pauc", "mean")).reset_index()
    rules = select_rules(pts, (tsd, tk), tlam)
    # test sets (same images / centres / noise seeds as run_lesions.py)
    grids = []
    for si, key in enumerate(["mnist", "fashion"]):
        label, xb = TEST[key]; x = xb.to(dev)
        centers = torch.tensor(np.random.default_rng(a.seed + 7 + si).integers(LO, HI + 1, size=(len(x), 2)), device=dev)
        role = "in-dist" if key == tuned_on else "shifted"
        g = evaluate(den, x, centers, a.seed + 100 + si, a.seed + 200 + 10 * si, tsd, tk, tlam, f"{label} ({role})"); g["role"] = role
        grids.append(g)
    G = pd.concat([sel.assign(role="selection")] + grids, ignore_index=True); G["model"] = name
    G.to_csv(f"{a.out}/oppoint_{name}_grid.csv", index=False)
    # rules on the test sets
    R = []
    for rule, r in rules.items():
        for g in grids:
            m = (g.method == r["method"]) & (g.k == r["k"]) & (np.isclose(g.sd.fillna(-1), -1 if np.isnan(r["sd"]) else r["sd"])) & (np.isclose(g.lam.fillna(-1), -1 if np.isnan(r["lam"]) else r["lam"]))
            R.append(g[m].assign(rule=rule, feasible=r["feasible"], model=name))
    R = pd.concat(R, ignore_index=True); R.to_csv(f"{a.out}/oppoint_{name}_rules.csv", index=False)
    print(f"\n=== {name}: selected operating points (chosen on the in-distribution selection set; selection-set mean paired AUC in brackets) ===")
    for rule, r in rules.items():
        ps = pts[(pts.method == r["method"]) & (pts.k == r["k"]) & np.isclose(pts.sd.fillna(-1), -1 if np.isnan(r["sd"]) else r["sd"]) & np.isclose(pts.lam.fillna(-1), -1 if np.isnan(r["lam"]) else r["lam"])].iloc[0]
        desc = f"sigma_d={r['sd']}, k={int(r['k'])}" if r["method"] == "PnP" else f"lambda={r['lam']:.4g}"
        print(f"  {rule:16s} {desc:24s} sel PSNR {ps.psnr:.2f} dB, mean pAUC {ps.pauc:.3f}" + ("" if r["feasible"] else "   (no PnP point keeps TV's PSNR; fell back to PnP-psnr)"))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for setl, d in R.groupby("set", sort=False):
        t = d.pivot_table(index="rule", columns="ltype", values=["psnr", "pauc", "retention", "false_per_img"], aggfunc="first")
        t = t.reindex(["PnP-psnr", "PnP-constrained", "PnP-lesion", "TV-psnr", "TV-lesion"])
        print(f"\n-- TEST: {setl}"); print(t.round(3).to_string())
    # frontier figure on the test sets
    fig, axs = plt.subplots(len(TYPES), 2, figsize=(11, 4 * len(TYPES)), squeeze=False)
    for j, g in enumerate(grids):
        for i, (s, c) in enumerate(TYPES):
            ax = axs[i, j]; d = g[g.ltype == f"s{s}/c{c}"]; pn, tvv = d[d.method == "PnP"], d[d.method == "TV"].sort_values("lam")
            sc = ax.scatter(pn.psnr, pn.pauc, c=np.log10(pn.sd), s=12 + 6 * np.log2(pn.k + 1), cmap="viridis", alpha=.8, label="PnP (colour = sigma_d, size = k)")
            ax.plot(tvv.psnr, tvv.pauc, "-o", color="tab:orange", ms=4, label="TV (lambda sweep)")
            for rule, mk, col in [("PnP-psnr", "*", "k"), ("PnP-constrained", "s", "tab:red"), ("PnP-lesion", "^", "tab:red"), ("TV-psnr", "*", "tab:orange"), ("TV-lesion", "^", "tab:orange")]:
                r = R[(R.rule == rule) & (R.set == g.set.iloc[0]) & (R.ltype == f"s{s}/c{c}")]
                ax.scatter(r.psnr, r.pauc, marker=mk, s=140, facecolor="none" if mk != "*" else col, edgecolor=col, lw=1.5, zorder=5, label=rule)
            ax.set_xlabel("PSNR of lesion-absent reconstruction (dB)"); ax.set_ylabel("paired detection AUC"); ax.grid(alpha=.3)
            ax.set_title(f"{g.set.iloc[0]}  |  lesion s{s}/c{c}", fontsize=9)
            if i == 0 and j == 0: ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(f"{name}: PSNR vs lesion detectability over operating points (test sets)", fontsize=10); plt.tight_layout()
    plt.savefig(f"{a.out}/oppoint_{name}.png", dpi=140); plt.close()
    print(f"\n[{name}] done in {time.time() - tstart:.0f}s", flush=True)
