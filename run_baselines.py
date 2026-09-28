"""E1 (noise) and E2 (number of angles) for classical baselines.
Hyperparameters are tuned per condition on the VALIDATION set only, then evaluated on TEST.
Usage: python run_baselines.py --exp noise   |   --exp angles
"""
import argparse, os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.operators import RadonOperator, add_noise
from src.baselines import Tikhonov, TVADMM
from src.metrics import per_image_metrics
from src.data import load_mnist

p = argparse.ArgumentParser()
p.add_argument("--exp", default="noise", choices=["noise", "angles"])
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--synthetic", action="store_true", help="smoke test without downloading MNIST")
args = p.parse_args()
os.makedirs("results", exist_ok=True)

if args.synthetic:
    g = torch.Generator().manual_seed(1)
    xs = torch.zeros(400, 28, 28)
    yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing="ij")
    for i in range(400):
        c = torch.randint(8, 20, (2,), generator=g); r = torch.randint(3, 8, (1,), generator=g)
        xs[i] = (((yy - c[0]) ** 2 + (xx - c[1]) ** 2) < r ** 2).float()
    xs = xs.reshape(400, -1); val, test = xs[:args.n_val], xs[args.n_val:args.n_val + args.n_test]
else:
    _, val, test = load_mnist()
    val, test = val[:args.n_val], test[:args.n_test]

conds = ([dict(n_angles=20, sigma=s) for s in [0, 0.01, 0.05, 0.1, 0.2]] if args.exp == "noise"
         else [dict(n_angles=a, sigma=0.05) for a in [5, 10, 20, 40, 90]])
key = "sigma" if args.exp == "noise" else "n_angles"
from src.experiment import LAM_TIK, LAM_TV
GRIDS = {"Tikhonov": LAM_TIK, "TV": LAM_TV}

rows = []
for c in conds:
    op = RadonOperator(28, c["n_angles"])
    tik, tv = Tikhonov(op), TVADMM(op)
    solvers = {"FBP": lambda y, _: op.fbp(y), "Tikhonov": lambda y, l: tik(y, l), "TV": lambda y, l: tv(y, l)}
    yv = add_noise(op.forward(val), c["sigma"], seed=args.seed)
    yt = add_noise(op.forward(test), c["sigma"], seed=args.seed + 1)
    for name, f in solvers.items():
        grid = GRIDS.get(name, [None])
        best = max(grid, key=lambda l: per_image_metrics(f(yv, l), val)[0].mean())  # tune on val PSNR
        ps, ss, rel = per_image_metrics(f(yt, best), test)
        for i in range(len(ps)):
            rows.append(dict(method=name, **c, param=best, img=i, psnr=ps[i], ssim=ss[i], rel=rel[i]))
        print(f"{c} {name:9s} param={best} PSNR={ps.mean():.2f} SSIM={ss.mean():.3f}")

df = pd.DataFrame(rows); df.to_csv(f"results/baselines_{args.exp}.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, metric in enumerate(["psnr", "ssim"]):
    for name, d in df.groupby("method"):
        g = d.groupby(key)[metric].agg(["mean", "sem"])
        ax[k].errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", capsize=3, label=name)
    ax[k].set_xlabel(key); ax[k].set_ylabel(metric.upper()); ax[k].grid(alpha=.3)
    if args.exp == "angles": ax[k].set_xscale("log")
ax[0].legend(); plt.tight_layout(); plt.savefig(f"results/baselines_{args.exp}.png", dpi=150)
