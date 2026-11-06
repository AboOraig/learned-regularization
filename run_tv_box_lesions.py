"""Control 3 (E9): does a box-constrained TV change the lesion-insertion results (detection, retention, false lesions)?

Same protocol as run_lesions.py (crime-free bicubic f=5 data, sigma 0.05, 20 angles, eight lesion types, same lesion centres and noise seeds,
all methods tuned on the model's own validation distribution), with TV-box added: TV with 0 <= x <= 1 (Chambolle-Pock, src/tv_pdhg.py, started from
the clipped FBP image, lambda tuned on the validation PSNR). Methods reported: FBP, TV (ADMM, as in the paper), TV-box, PnP-DnCNN.
The TV and PnP rows are checked against results/step5/lesion_<model>_metrics.csv (they should reproduce exactly).

Usage (repository root):
    python run_tv_box_lesions.py                    # six models x {MNIST, Fashion-MNIST}, 8 lesion types (resumable; roughly 30-60 min on a GPU)
    python run_tv_box_lesions.py --quick            # m03_s0 on MNIST, two lesion types
    python run_tv_box_lesions.py --summarize_only
Outputs (results/control/): tv_box_lesions_<model>.csv (same columns as lesion_<model>_metrics.csv), printed summary tables.
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, torch
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, reconstruct_all, set_sd_grid
from src.finegrid import make_simulator
from src.metrics import per_image_metrics
from src.tv_pdhg import tv_pdhg
from src.tune_ct import climb
from src.data import load_mnist, load_fashion_train, load_fashion
from src import lesions as L

p = argparse.ArgumentParser()
p.add_argument("--models", default="m03_s0,m03_s1,m03_s2,f03_s0,f03_s1,f03_s2")
p.add_argument("--ckpt_dir", default="results/step4")
p.add_argument("--stored_dir", default="results/step5")
p.add_argument("--out", default="results/control")
p.add_argument("--sets", default="mnist,fashion")
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--n_angles", type=int, default=20)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sizes", default="0.5,0.75,1.25,2.0")
p.add_argument("--contrasts", default="0.25,0.5")
p.add_argument("--tv_iters", type=int, default=1000)
p.add_argument("--sd_grid", default="0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5")
p.add_argument("--n_boot", type=int, default=500)
p.add_argument("--quick", action="store_true")
p.add_argument("--force", action="store_true")
p.add_argument("--summarize_only", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.quick:
    a.models, a.sets, a.sizes, a.contrasts = "m03_s0", "mnist", "0.75,1.25", "0.5"
METHODS = ["FBP", "TV", "TV-box", "PnP-DnCNN"]


def summarize():
    fs = sorted(glob.glob(f"{a.out}/tv_box_lesions_*.csv"))
    if not fs:
        print("no outputs found"); return
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n=== paired AUC / retention / false lesions per image: mean over seeds and the 8 lesion types ===")
    g = D.groupby(["family", "test_set", "role", "method"], sort=False)[["paired_auc", "retention", "false_per_img"]].mean().round(3)
    print(g.unstack("method").to_string())
    print("\n=== paired AUC: PnP - TV and PnP - TV-box (mean over seeds and lesion types; positive = PnP detects better) ===")
    P = D.pivot_table(index=["family", "test_set", "role", "model", "sigma_l", "contrast"], columns="method", values="paired_auc").reset_index()
    P["PnP-TV"] = P["PnP-DnCNN"] - P["TV"]; P["PnP-TVbox"] = P["PnP-DnCNN"] - P["TV-box"]; P["TVbox-TV"] = P["TV-box"] - P["TV"]
    print(P.groupby(["family", "test_set", "role"], sort=False)[["PnP-TV", "PnP-TVbox", "TVbox-TV"]].mean().round(3).to_string())
    print("\n  (model, lesion type) cells where PnP has the higher paired AUC: vs TV, vs TV-box, of n")
    for (fam_, ts_), d in P.groupby(["family", "test_set"], sort=False):
        print(f"  {fam_:4s} {ts_:14s} {int((d['PnP-DnCNN'] > d['TV']).sum()):3d} / {int((d['PnP-DnCNN'] > d['TV-box']).sum()):3d} of {len(d)}")
    print("\n=== false lesions per lesion-absent image, mean over seeds and lesion types (by lesion size) ===")
    F = D.groupby(["family", "test_set", "sigma_l", "method"])["false_per_img"].mean().unstack("method").round(3)
    print(F[["TV", "TV-box", "PnP-DnCNN"]].to_string())
    print("\n=== retention by lesion size (mean over seeds and contrasts) ===")
    R = D.groupby(["family", "test_set", "sigma_l", "method"])["retention"].mean().unstack("method").round(2)
    print(R[["TV", "TV-box", "PnP-DnCNN"]].to_string())


if a.summarize_only:
    summarize(); raise SystemExit

dev = "cuda" if torch.cuda.is_available() else "cpu"
set_sd_grid([float(v) for v in a.sd_grid.split(",")])
SIZES = [float(s) for s in a.sizes.split(",")]; CONTR = [float(c) for c in a.contrasts.split(",")]
TYPES = [(s, c) for s in SIZES for c in CONTR]
LO, HI = L.centre_range(SIZES)
op = RadonOperator(28, a.n_angles, device=dev); Lip = op.lipschitz()
sim = make_simulator(op, 5, "bicubic")
mn_val, mn_test = load_mnist()[1], load_mnist()[2]
fa_val = load_fashion_train()[1] if any(m.startswith("f") for m in a.models.split(",")) else None
BASES = {"mnist": ("MNIST", lambda: mn_test[:a.n_test]), "fashion": ("Fashion-MNIST", lambda: load_fashion()[:a.n_test])}
print(f"device {dev} | lesion types {TYPES} | centre range [{LO},{HI}]", flush=True)

for name in a.models.split(","):
    f_out = f"{a.out}/tv_box_lesions_{name}.csv"
    if os.path.exists(f_out) and not a.force:
        print(f"skip (exists): {f_out}"); continue
    tstart = time.time(); fam, seed = name.rsplit("_s", 1); tuned_on = "fashion" if fam.startswith("f") else "mnist"
    den = load_denoiser(f"{a.ckpt_dir}/dn_{name}.pt", dev)
    val = (fa_val if tuned_on == "fashion" else mn_val)[:a.n_val].to(dev)
    yv = add_noise(sim(val), a.sigma, seed=a.seed)
    params = tune_all(op, den, val, yv)
    x0v = op.fbp(yv).clamp(0, 1)
    lam_box, sc = climb([float(v) for v in np.logspace(-5, -1, 17)],
                        lambda l: per_image_metrics(tv_pdhg(op, yv, l, a.tv_iters, x0v, Lip), val)[0].mean(),
                        lambda b: b / 3, lambda b: b * 3, 3, "TV-box", log=lambda s: None)
    edge = " [EDGE of lambda grid]" if lam_box in (min(sc), max(sc)) else ""
    print(f"\n[{name}] tuned on {tuned_on}: {params}; TV-box lambda_norm={lam_box:.3g}{edge}", flush=True)

    def rec(y):
        out = {k: v.clamp(0, 1) for k, v in reconstruct_all(op, den, y, params).items()}
        out["TV-box"] = tv_pdhg(op, y, lam_box, a.tv_iters, op.fbp(y).clamp(0, 1), Lip).clamp(0, 1)
        return out

    rows = []
    for si, key in enumerate(a.sets.split(",")):
        set_name, getx = BASES[key]; x = getx().to(dev); n = len(x)
        role = "in-dist" if key == tuned_on else "shifted"
        rng = np.random.default_rng(a.seed + 7 + si)
        centers = torch.tensor(rng.integers(LO, HI + 1, size=(n, 2)), device=dev)
        rec0 = rec(add_noise(sim(x), a.sigma, seed=a.seed + 100 + si))
        for li, (sl, c) in enumerate(TYPES):
            rho = L.lesion_rho(sl); nom = L.nominal_response(sl, c); theta = 0.5 * nom
            xl = L.insert_lesion(x, centers, sl, c)
            recl = rec(add_noise(sim(xl), a.sigma, seed=a.seed + 200 + 10 * si + li))
            tt1, tt0 = L.contrast_at(xl, rho, centers).cpu().numpy(), L.contrast_at(x, rho, centers).cpu().numpy()
            valid = (tt1 - tt0) >= 0.5 * nom
            for m in METHODS:
                t1, t0 = L.contrast_at(recl[m], rho, centers).cpu().numpy(), L.contrast_at(rec0[m], rho, centers).cpu().numpy()
                fl, cand = L.count_false_lesions(rec0[m], x, rho, theta)
                lo, hi = L.bootstrap_auc_ci(t1[valid], t0[valid], a.n_boot, seed=a.seed)
                rows.append(dict(model=name, family=fam, seed=int(seed), tuned_on=tuned_on, test_set=set_name, role=role, sigma_l=sl, contrast=c, rho=rho,
                                 method=m, n_valid=int(valid.sum()), auc=L.auc(t1[valid], t0[valid]), auc_lo=lo, auc_hi=hi,
                                 paired_auc=L.paired_auc(t1[valid], t0[valid]), retention=float((t1 - t0)[valid].mean() / (tt1 - tt0)[valid].mean()),
                                 false_per_img=float(fl.float().mean()), cand_per_img=float(cand.float().mean()),
                                 pnp_sd=params["PnP"][0], pnp_k=params["PnP"][1], tv_lam=float(params["TV"]), tv_box_lam=lam_box))
        print(f"  {set_name} ({role}) done", flush=True)
    S = pd.DataFrame(rows); S.to_csv(f_out, index=False)
    f_old = f"{a.stored_dir}/lesion_{name}_metrics.csv"
    if os.path.exists(f_old):                       # reproduction check on the rows both files have
        O = pd.read_csv(f_old)
        M = S[S.method.isin(["TV", "PnP-DnCNN"])].merge(O[["test_set", "sigma_l", "contrast", "method", "paired_auc", "false_per_img", "retention"]],
                                                         on=["test_set", "sigma_l", "contrast", "method"], suffixes=("", "_stored"))
        if len(M):
            dauc = (M.paired_auc - M.paired_auc_stored).abs().max(); dfl = (M.false_per_img - M.false_per_img_stored).abs().max()
            print(f"  check vs {f_old}: {len(M)} rows, max |d paired AUC| = {dauc:.4f}, max |d false/img| = {dfl:.4f} -> {'OK' if dauc < 0.011 and dfl < 0.011 else 'DIFFERS'}")
    print(f"  [{name}] done in {time.time() - tstart:.0f}s", flush=True)
summarize()
