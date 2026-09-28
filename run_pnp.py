"""PnP-PGD for E1 (noise) / E2 (angles). (sigma_d, #iters) tuned on VALIDATION only.
Also saves the full validation grid (results/pnp_ablation_*.csv) = E5 ablation data.
Usage: python run_pnp.py --exp noise | --exp angles"""
import argparse, itertools, os, numpy as np, pandas as pd, torch
from src.operators import RadonOperator, add_noise
from src.pnp import pnp_pgd
from src.denoiser import load_denoiser
from src.metrics import per_image_metrics
from src.data import load_mnist

p = argparse.ArgumentParser()
p.add_argument("--exp", default="noise", choices=["noise", "angles"])
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--model", default="results/dncnn.pt")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
_, val, test = load_mnist(); val, test = val[:a.n_val].to(dev), test[:a.n_test].to(dev)
den = load_denoiser(a.model, dev)
conds = ([dict(n_angles=20, sigma=s) for s in [0, 0.01, 0.05, 0.1, 0.2]] if a.exp == "noise"
         else [dict(n_angles=n, sigma=0.05) for n in [5, 10, 20, 40, 90]])
SD, CK = [0.02, 0.05, 0.1, 0.15, 0.2], (25, 50, 100, 200)

rows, abl = [], []
for c in conds:
    op = RadonOperator(28, c["n_angles"], device=dev); eta = 1.0 / op.lipschitz()
    yv = add_noise(op.forward(val), c["sigma"], seed=a.seed)
    yt = add_noise(op.forward(test), c["sigma"], seed=a.seed + 1)
    best, best_ps = None, -1
    for sd in SD:
        outs, _ = pnp_pgd(op, yv, den, sd, eta, CK)
        for k, x in outs.items():
            ps = per_image_metrics(x, val)[0].mean(); abl.append(dict(**c, sigma_d=sd, iters=k, val_psnr=ps))
            if ps > best_ps: best, best_ps = (sd, k), ps
    sd, k = best
    outs, _ = pnp_pgd(op, yt, den, sd, eta, (k,))
    ps, ss, rel = per_image_metrics(outs[k], test)
    for i in range(len(ps)):
        rows.append(dict(method="PnP-DnCNN", **c, param=f"sd={sd},k={k}", img=i, psnr=ps[i], ssim=ss[i], rel=rel[i]))
    print(f"{c} PnP sigma_d={sd} iters={k} PSNR={ps.mean():.2f} SSIM={ss.mean():.3f}")

os.makedirs("results", exist_ok=True)
pd.DataFrame(rows).to_csv(f"results/pnp_{a.exp}.csv", index=False)
pd.DataFrame(abl).to_csv(f"results/pnp_ablation_{a.exp}.csv", index=False)
