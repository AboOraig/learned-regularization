"""Control experiment: is the PnP advantage over TV on digit images due to the [0,1] clipping inside the PnP iteration?

In the original digit experiments (E1-E10) the PnP iterates are clipped to [0,1] at every step (src/pnp.py, src/denoiser.py),
whereas ADMM-TV (src/baselines.py) is unconstrained inside the iteration and clipped only when the metrics are computed.
This script compares, on exactly the same data, noise seeds and images as run_baselines.py / run_pnp.py:

    TV-ADMM    the TV baseline of the paper (unconstrained, tuned lambda)
    TV-box     TV with the box constraint 0 <= x <= 1 (Chambolle-Pock, src/tv_pdhg.py), own tuned lambda, started from FBP
    PnP        the PnP-DnCNN of the paper (iterates clipped to [0,1])
    PnP-noclip PnP with NO clipping inside the iteration (clipped only when the metrics are computed, like TV-ADMM)

and reports paired differences (mean, 95% bootstrap interval over images, Wilcoxon p) for
    PnP - TV-ADMM          (the paper's comparison; should reproduce the E1/E2 numbers)
    PnP - TV-box           (comparison against a TV that has the same constraint)
    PnP-noclip - TV-ADMM   (both unconstrained)
    TV-box - TV-ADMM       (what the box alone gives TV)

Usage (from the repository root):
    python run_tv_box_control.py --data crime  --exp both          # E1/E2 conditions (data generated with A, as in the paper)
    python run_tv_box_control.py --data fine5c --exp both          # crime-free data (bicubic, factor 5), as in E6-E10
    python run_tv_box_control.py --quick                           # 2 conditions, a few minutes
    python run_tv_box_control.py --synthetic --quick --n_val 10 --n_test 20   # smoke test without MNIST

Outputs (results/control/): tv_box_<data>_images.csv (per image), tv_box_<data>_summary.csv (means and paired differences).
"""
import argparse, os, time, numpy as np, pandas as pd, torch
from scipy.stats import wilcoxon
from src.operators import RadonOperator, add_noise
from src.pnp import pnp_pgd
from src.baselines import TVADMM
from src.tv_pdhg import tv_pdhg
from src.tune_ct import climb
from src.denoiser import load_denoiser
from src.finegrid import make_simulator
from src.metrics import per_image_metrics
from src.data import load_mnist
from src.experiment import LAM_TV, SD, CK

p = argparse.ArgumentParser()
p.add_argument("--data", default="crime", choices=["crime", "fine5c"], help="crime: y = A x + n (E1/E2); fine5c: crime-free, bicubic factor 5")
p.add_argument("--exp", default="both", choices=["noise", "angles", "both"])
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--model", default="results/dncnn.pt")
p.add_argument("--tv_iters", type=int, default=1000, help="Chambolle-Pock iterations for TV-box")
p.add_argument("--boot", type=int, default=5000)
p.add_argument("--quick", action="store_true", help="only (20 angles, sigma 0.05) and (5 angles, sigma 0.05)")
p.add_argument("--synthetic", action="store_true", help="smoke test: synthetic discs instead of MNIST")
p.add_argument("--out", default="results/control")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
if a.synthetic:
    g = torch.Generator().manual_seed(1)
    yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
    xs = torch.zeros(a.n_val + a.n_test, 28, 28)
    for i in range(len(xs)):
        c = torch.randint(8, 20, (2,), generator=g); r = torch.randint(3, 8, (1,), generator=g)
        xs[i] = (((yy - c[0]) ** 2 + (xx - c[1]) ** 2) < r ** 2).float()
    xs = xs.reshape(len(xs), -1); val, test = xs[:a.n_val].to(dev), xs[a.n_val:].to(dev)
else:
    _, val, test = load_mnist(); val, test = val[:a.n_val].to(dev), test[:a.n_test].to(dev)
den = load_denoiser(a.model, dev)

conds = []
if a.exp in ("noise", "both"):
    conds += [dict(n_angles=20, sigma=s) for s in [0, 0.01, 0.05, 0.1, 0.2]]
if a.exp in ("angles", "both"):
    conds += [dict(n_angles=n, sigma=0.05) for n in [5, 10, 40, 90]] + ([dict(n_angles=20, sigma=0.05)] if a.exp == "angles" else [])
if a.quick:
    conds = [dict(n_angles=20, sigma=0.05), dict(n_angles=5, sigma=0.05)]


@torch.no_grad()
def pnp_noclip(op, y, den, sigma_d, eta, checkpoints):
    """PnP-PGD without any clipping inside the iteration (the metrics clip at evaluation, as for TV-ADMM)."""
    x = op.fbp(y)
    out = {}
    s = torch.full((len(y),), float(sigma_d), device=y.device)
    if 0 in checkpoints:
        out[0] = den(x.reshape(-1, 1, 28, 28), s).reshape(len(y), -1)
    for k in range(1, max(checkpoints) + 1):
        z = x - eta * op.adjoint(op.forward(x) - y)
        x = den(z.reshape(-1, 1, 28, 28), s).reshape(len(y), -1)
        if k in checkpoints:
            out[k] = x.clone()
    return out


def tune_pnp(runner, op, eta, yv, label):
    best, best_ps = None, -1.0
    for sd in SD:
        outs = runner(op, yv, den, sd, eta, CK)
        for k, x in outs.items():
            ps = per_image_metrics(x, val)[0].mean()
            if ps > best_ps:
                best, best_ps = (sd, k), ps
    if best[0] in (SD[0], SD[-1]) or best[1] in (CK[0], CK[-1]):
        print(f"  WARNING: {label} hyperparameter at grid edge {best}")
    return best


def boot_ci(d, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def wilc(x, y):
    try:
        return wilcoxon(x, y).pvalue
    except ValueError:
        return float("nan")


rows, summ = [], []
os.makedirs(a.out, exist_ok=True)
for c in conds:
    t0 = time.time()
    op = RadonOperator(28, c["n_angles"], device=dev)
    L = op.lipschitz(); eta = 1.0 / L
    sim = make_simulator(op, 5, "bicubic") if a.data == "fine5c" else op.forward
    yv = add_noise(sim(val), c["sigma"], seed=a.seed)
    yt = add_noise(sim(test), c["sigma"], seed=a.seed + 1)
    res = {}
    # --- TV-ADMM (the paper's TV): lambda tuned on the validation PSNR
    tv = TVADMM(op)
    lam = max(LAM_TV, key=lambda l: per_image_metrics(tv(yv, l), val)[0].mean())
    res["TV-ADMM"] = (tv(yt, lam), f"lam={lam:.3g}")
    # --- TV-box: Chambolle-Pock with 0<=x<=1, started from the clipped FBP image, lambda in ||A||^2-normalised units
    x0v, x0t = op.fbp(yv).clamp(0, 1), op.fbp(yt).clamp(0, 1)
    grid = [float(v) for v in np.logspace(-5, -1, 17)]
    best_l, _ = climb(grid, lambda l: per_image_metrics(tv_pdhg(op, yv, l, a.tv_iters, x0v, L), val)[0].mean(),
                      lambda b: b / 3, lambda b: b * 3, 3, f"[{c}] TV-box lambda -> val PSNR", log=lambda s: None)
    res["TV-box"] = (tv_pdhg(op, yt, best_l, a.tv_iters, x0t, L), f"lam_norm={best_l:.3g}")
    # --- PnP (clipped, as in the paper) and PnP without clipping inside the iteration
    sd, k = tune_pnp(lambda o, y, d, s, e, ck: pnp_pgd(o, y, d, s, e, ck)[0], op, eta, yv, "PnP")
    res["PnP"] = (pnp_pgd(op, yt, den, sd, eta, (k,))[0][k], f"sd={sd},k={k}")
    sd, k = tune_pnp(pnp_noclip, op, eta, yv, "PnP-noclip")
    res["PnP-noclip"] = (pnp_noclip(op, yt, den, sd, eta, (k,))[k], f"sd={sd},k={k}")

    P = {}
    for name, (x, par) in res.items():
        ps, ss, rel = per_image_metrics(x, test)
        P[name] = ps
        for i in range(len(ps)):
            rows.append(dict(method=name, **c, data=a.data, param=par, img=i, psnr=ps[i], ssim=ss[i], rel=rel[i]))
    line = dict(**c, data=a.data, n_test=len(test))
    for name in res:
        line[f"psnr_{name}"] = P[name].mean()
        line[f"param_{name}"] = res[name][1]
    for nm, (u, v) in {"PnP-TVADMM": ("PnP", "TV-ADMM"), "PnP-TVbox": ("PnP", "TV-box"),
                       "PnPnoclip-TVADMM": ("PnP-noclip", "TV-ADMM"), "TVbox-TVADMM": ("TV-box", "TV-ADMM")}.items():
        d = P[u] - P[v]; lo, hi = boot_ci(d, a.boot)
        line[f"d_{nm}"], line[f"lo_{nm}"], line[f"hi_{nm}"], line[f"p_{nm}"] = d.mean(), lo, hi, wilc(P[u], P[v])
    summ.append(line)
    print(f"\n{c} [{a.data}] ({time.time() - t0:.0f}s)")
    print("  PSNR  " + "  ".join(f"{n}={P[n].mean():.2f} ({res[n][1]})" for n in res))
    for nm in ("PnP-TVADMM", "PnP-TVbox", "PnPnoclip-TVADMM", "TVbox-TVADMM"):
        print(f"  {nm:17s} {line['d_' + nm]:+.2f} dB  [{line['lo_' + nm]:+.2f}, {line['hi_' + nm]:+.2f}]  Wilcoxon p={line['p_' + nm]:.1e}")

img = pd.DataFrame(rows); S = pd.DataFrame(summ)
img.to_csv(f"{a.out}/tv_box_{a.data}_images.csv", index=False); S.to_csv(f"{a.out}/tv_box_{a.data}_summary.csv", index=False)

# consistency check with the stored E1/E2 results (crime data only): should reproduce TV-ADMM and PnP to ~0.01 dB
if a.data == "crime" and not a.synthetic:
    for exp, key in (("noise", "sigma"), ("angles", "n_angles")):
        fb, fp = f"results/baselines_{exp}.csv", f"results/pnp_{exp}.csv"
        if not (os.path.exists(fb) and os.path.exists(fp)):
            continue
        B, Pn = pd.read_csv(fb), pd.read_csv(fp)
        for _, r in S.iterrows():
            sel = (B.n_angles == r.n_angles) & (np.isclose(B.sigma, r.sigma))
            tvm = B[sel & (B.method == "TV")].psnr.mean()
            pnm = Pn[(Pn.n_angles == r.n_angles) & np.isclose(Pn.sigma, r.sigma)].psnr.mean()
            if np.isnan(tvm) or np.isnan(pnm):
                continue
            ok = abs(tvm - r["psnr_TV-ADMM"]) < 0.05 and abs(pnm - r["psnr_PnP"]) < 0.05
            print(f"check vs {exp} CSV, {int(r.n_angles)} angles sigma={r.sigma}: stored TV {tvm:.2f} / PnP {pnm:.2f}; here {r['psnr_TV-ADMM']:.2f} / {r['psnr_PnP']:.2f}  -> {'OK' if ok else 'DIFFERS'}")
print(f"\nwrote {a.out}/tv_box_{a.data}_images.csv and tv_box_{a.data}_summary.csv")
