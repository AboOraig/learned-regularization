"""E3 (forward-model mismatch) and E4 (distribution shift). Params tuned ONCE on MNIST val at the
nominal condition (20 angles, sigma=0.05) and frozen. Usage: python run_robustness.py --exp mismatch|shift"""
import argparse, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, reconstruct_all
from src.metrics import per_image_metrics
from src.data import load_mnist, load_fashion, load_emnist_letters, shepp_logan

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True, choices=["mismatch", "shift"])
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--model", default="results/dncnn.pt")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
_, val, test = load_mnist(); val, test = val[:a.n_val].to(dev), test[:a.n_test].to(dev)
den = load_denoiser(a.model, dev)
true_op = RadonOperator(28, 20, device=dev)
yv = add_noise(true_op.forward(val), a.sigma, seed=a.seed)
params = tune_all(true_op, den, val, yv)
print("Frozen params:", params)


def collect(xs, x_true, tag, rows):
    for name, x in xs.items():
        ps, ss, rel = per_image_metrics(x, x_true)
        for i in range(len(ps)):
            rows.append(dict(method=name, img=i, psnr=ps[i], ssim=ss[i], rel=rel[i], **tag))
        print(f"  {tag} {name:10s} PSNR={ps.mean():.2f} SSIM={ss.mean():.3f}")


rows = []
if a.exp == "mismatch":
    yt = add_noise(true_op.forward(test), a.sigma, seed=a.seed + 1)  # data from the TRUE operator
    rng = np.random.RandomState(0)
    base = np.linspace(0, 180, 20, endpoint=False)
    for kind in ["offset", "jitter"]:
        for delta in [0, 0.5, 1, 2, 4]:
            ang = base + delta if kind == "offset" else base + delta * rng.randn(20)
            mop = RadonOperator(28, angles=ang, device=dev)  # WRONG operator used for reconstruction
            collect(reconstruct_all(mop, den, yt, params), test, dict(kind=kind, delta=delta), rows)
    df = pd.DataFrame(rows); df.to_csv("results/robust_mismatch.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for k, kind in enumerate(["offset", "jitter"]):
        for name, d in df[df.kind == kind].groupby("method"):
            g = d.groupby("delta").psnr.agg(["mean", "sem"])
            ax[k].errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", capsize=3, label=name)
        ax[k].set_title(f"Angle {kind} (degrees)"); ax[k].set_xlabel("delta (deg)"); ax[k].grid(alpha=.3)
    ax[0].set_ylabel("PSNR (dB)"); ax[0].legend(); plt.tight_layout()
    plt.savefig("results/robust_mismatch.png", dpi=150)
else:
    sets = {"MNIST (in-dist)": test,
            "Fashion-MNIST": load_fashion()[:a.n_test].to(dev),
            "EMNIST-letters": load_emnist_letters()[:a.n_test].to(dev),
            "Shepp-Logan (x20 noise)": shepp_logan().repeat(20, 1).to(dev)}
    for i, (name, x) in enumerate(sets.items()):
        y = add_noise(true_op.forward(x), a.sigma, seed=a.seed + 10 + i)
        collect(reconstruct_all(true_op, den, y, params), x, dict(dataset=name), rows)
    df = pd.DataFrame(rows); df.to_csv("results/robust_shift.csv", index=False)
    g = df.groupby(["dataset", "method"]).psnr.agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4)); methods = list(df.method.unique()); w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        d = g[g.method == m].set_index("dataset").reindex(list(sets))
        ax.bar(np.arange(len(sets)) + j * w, d["mean"], w, yerr=1.96 * d["sem"], capsize=2, label=m)
    ax.set_xticks(np.arange(len(sets)) + 0.4 - w / 2); ax.set_xticklabels(list(sets), rotation=10)
    ax.set_ylabel("PSNR (dB)"); ax.legend(); ax.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig("results/robust_shift.png", dpi=150)
