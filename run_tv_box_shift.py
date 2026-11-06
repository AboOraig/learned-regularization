"""Control 2 (E4 / E8): does a box-constrained TV change the distribution-shift and matched-prior conclusions?

Same protocol as run_step4.py -> run_nullspace.py --exp shift --fine_factor 5 --up_mode bicubic (sigma 0.05, 20 angles):
for every denoiser of results/step4 (m03_s0..2 = MNIST prior, f03_s0..2 = Fashion prior), all methods are tuned on the model's OWN validation
distribution (crime-free data), then frozen and applied to MNIST / Fashion-MNIST / EMNIST letters / Shepp-Logan, with the same noise seeds.
New here: TV-box = TV with 0 <= x <= 1 (Chambolle-Pock, src/tv_pdhg.py, started from the clipped FBP image, lambda tuned on the same
validation data). The script reports, per model and test set,
    gap_ADMM = PnP - TV-ADMM   (the paper's E4/E8 number; reproduced and checked against results/step4/nullspace_method_means_shift_<name>.csv)
    gap_box  = PnP - TV-box
and, aggregated over the three seeds of each family, the prior x test-set matrix for both comparisons.

Usage (repository root):
    python run_tv_box_shift.py                     # all six models, four test sets (about 20-40 min on a GPU; resumable)
    python run_tv_box_shift.py --quick             # one model (m03_s0), MNIST + Shepp-Logan
    python run_tv_box_shift.py --summarize_only    # re-print the aggregated tables from the per-model CSVs
Outputs (results/control/): tv_box_shift_<model>.csv (per model and test set), tv_box_shift_images_<model>.csv (per image), tv_box_shift_seeds.csv
"""
import argparse, os, time, numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, reconstruct_all, set_sd_grid
from src.finegrid import make_simulator
from src.metrics import per_image_metrics
from src.tv_pdhg import tv_pdhg
from src.tune_ct import climb
from src.data import load_mnist, load_fashion_train, load_fashion, load_emnist_letters, shepp_logan

p = argparse.ArgumentParser()
p.add_argument("--models", default="m03_s0,m03_s1,m03_s2,f03_s0,f03_s1,f03_s2")
p.add_argument("--ckpt_dir", default="results/step4")
p.add_argument("--sets", default="mnist,fashion,emnist,shepp")
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tv_iters", type=int, default=1000)
p.add_argument("--boot", type=int, default=5000)
p.add_argument("--sd_grid", default="0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
p.add_argument("--out", default="results/control")
p.add_argument("--quick", action="store_true")
p.add_argument("--force", action="store_true")
p.add_argument("--summarize_only", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.quick:
    a.models, a.sets = "m03_s0", "mnist,shepp"

NAMES = {"mnist": "MNIST (in-dist)", "fashion": "Fashion-MNIST", "emnist": "EMNIST-letters", "shepp": "Shepp-Logan (x20 noise)"}
ORDER = ["mnist", "fashion", "emnist", "shepp"]          # position fixes the noise seed (seed + 10 + i), as in run_nullspace.py


def boot_ci(d, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilc(x, y):
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return float("nan")


def summarize():
    import glob
    fs = sorted(glob.glob(f"{a.out}/tv_box_shift_m*_s*.csv") + glob.glob(f"{a.out}/tv_box_shift_f*_s*.csv"))
    fs = [f for f in fs if "images" not in f and "seeds" not in f]
    if not fs:
        print("no per-model outputs found"); return
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    g = D.groupby(["family", "dataset"], sort=False).agg(
        n_seeds=("seed", "nunique"), TV_ADMM=("psnr_TV-ADMM", "mean"), TV_box=("psnr_TV-box", "mean"), PnP=("psnr_PnP", "mean"),
        gap_ADMM=("gap_ADMM", "mean"), gap_ADMM_sd=("gap_ADMM", "std"), gap_box=("gap_box", "mean"), gap_box_sd=("gap_box", "std"),
        box_gain=("box_gain", "mean")).reset_index()
    g.to_csv(f"{a.out}/tv_box_shift_seeds.csv", index=False)
    print("\n=== PnP - TV (dB), mean over seeds (sd over seeds); <0 means TV is better ===")
    for _, r in g.iterrows():
        print(f"  {r.family:4s} {r.dataset:26s} n={int(r.n_seeds)}  vs TV-ADMM {r.gap_ADMM:+.2f} ({r.gap_ADMM_sd:.2f})   vs TV-box {r.gap_box:+.2f} ({r.gap_box_sd:.2f})   box gain for TV {r.box_gain:+.2f}")
    print("\n  prior x test set, PnP - TV-ADMM:\n" + g.pivot(index="family", columns="dataset", values="gap_ADMM").round(2).to_string())
    print("\n  prior x test set, PnP - TV-box:\n" + g.pivot(index="family", columns="dataset", values="gap_box").round(2).to_string())


if a.summarize_only:
    summarize(); raise SystemExit

dev = "cuda" if torch.cuda.is_available() else "cpu"
set_sd_grid([float(v) for v in a.sd_grid.split(",")])
mn_val = load_mnist()[1]; fa_val = load_fashion_train()[1] if any(m.startswith("f") for m in a.models.split(",")) else None
mn_test = load_mnist()[2][:a.n_test].to(dev)
GET = {"mnist": lambda: mn_test, "fashion": lambda: load_fashion()[:a.n_test].to(dev),
       "emnist": lambda: load_emnist_letters()[:a.n_test].to(dev), "shepp": lambda: shepp_logan().repeat(20, 1).to(dev)}
op = RadonOperator(28, 20, device=dev); L = op.lipschitz()
sim = make_simulator(op, 5, "bicubic")

for name in a.models.split(","):
    f_out = f"{a.out}/tv_box_shift_{name}.csv"
    if os.path.exists(f_out) and not a.force:
        print(f"skip (exists): {f_out}"); continue
    t0 = time.time(); fam, seed = name.rsplit("_s", 1)
    den = load_denoiser(f"{a.ckpt_dir}/dn_{name}.pt", dev)
    val = (fa_val if fam.startswith("f") else mn_val)[:a.n_val].to(dev)
    yv = add_noise(sim(val), a.sigma, seed=a.seed)
    params = tune_all(op, den, val, yv)
    x0v = op.fbp(yv).clamp(0, 1)
    lam_box, sc = climb([float(v) for v in np.logspace(-5, -1, 17)],
                        lambda l: per_image_metrics(tv_pdhg(op, yv, l, a.tv_iters, x0v, L), val)[0].mean(),
                        lambda b: b / 3, lambda b: b * 3, 3, "TV-box", log=lambda s: None)
    edge = " [EDGE of lambda grid]" if lam_box in (min(sc), max(sc)) else ""
    print(f"\n[{name}] tuned on {'Fashion' if fam.startswith('f') else 'MNIST'} validation: {params}; TV-box lambda_norm={lam_box:.3g}{edge}", flush=True)
    rows, irows = [], []
    for i, key in enumerate(ORDER):
        if key not in a.sets.split(","):
            continue
        x = GET[key](); y = add_noise(sim(x), a.sigma, seed=a.seed + 10 + i)
        R = reconstruct_all(op, den, y, params)
        box = tv_pdhg(op, y, lam_box, a.tv_iters, op.fbp(y).clamp(0, 1), L)
        P = {"TV-ADMM": per_image_metrics(R["TV"], x)[0], "TV-box": per_image_metrics(box, x)[0], "PnP": per_image_metrics(R["PnP-DnCNN"], x)[0]}
        d_a, d_b = P["PnP"] - P["TV-ADMM"], P["PnP"] - P["TV-box"]
        lo_a, hi_a = boot_ci(d_a, a.boot); lo_b, hi_b = boot_ci(d_b, a.boot)
        rows.append(dict(model=name, family=fam, seed=int(seed), dataset=NAMES[key], n=len(x), sigma=a.sigma,
                         **{f"psnr_{k}": v.mean() for k, v in P.items()},
                         gap_ADMM=d_a.mean(), gap_ADMM_lo=lo_a, gap_ADMM_hi=hi_a, p_ADMM=wilc(P["PnP"], P["TV-ADMM"]),
                         gap_box=d_b.mean(), gap_box_lo=lo_b, gap_box_hi=hi_b, p_box=wilc(P["PnP"], P["TV-box"]),
                         box_gain=(P["TV-box"] - P["TV-ADMM"]).mean(), lam_ADMM=float(params["TV"]), lam_box=lam_box,
                         pnp_sd=params["PnP"][0], pnp_k=params["PnP"][1]))
        irows.append(pd.DataFrame(dict(model=name, dataset=NAMES[key], img=np.arange(len(x)), **{f"psnr_{k}": v for k, v in P.items()})))
        print(f"  {NAMES[key]:26s} TV-ADMM {P['TV-ADMM'].mean():.2f}  TV-box {P['TV-box'].mean():.2f}  PnP {P['PnP'].mean():.2f} | "
              f"PnP-TV-ADMM {d_a.mean():+.2f} [{lo_a:+.2f},{hi_a:+.2f}]  PnP-TV-box {d_b.mean():+.2f} [{lo_b:+.2f},{hi_b:+.2f}]", flush=True)
    S = pd.DataFrame(rows); S.to_csv(f_out, index=False)
    pd.concat(irows).to_csv(f"{a.out}/tv_box_shift_images_{name}.csv", index=False)
    # reproduction check against the stored E8 outputs
    f_old = f"{a.ckpt_dir}/nullspace_method_means_shift_{name}.csv"
    if os.path.exists(f_old):
        mm = pd.read_csv(f_old); mm = mm[mm.subspace == "tau0.05"]
        ps = mm.pivot_table(index="dataset", columns="method", values="psnr")
        for _, r in S.iterrows():
            if r.dataset in ps.index:
                ok = abs(ps.loc[r.dataset, "TV"] - r["psnr_TV-ADMM"]) < 0.05 and abs(ps.loc[r.dataset, "PnP-DnCNN"] - r["psnr_PnP"]) < 0.05
                print(f"  check vs stored {name} / {r.dataset}: TV {ps.loc[r.dataset, 'TV']:.2f} PnP {ps.loc[r.dataset, 'PnP-DnCNN']:.2f}; here {r['psnr_TV-ADMM']:.2f} / {r['psnr_PnP']:.2f} -> {'OK' if ok else 'DIFFERS'}")
    print(f"  [{name}] done in {time.time() - t0:.0f}s", flush=True)
summarize()
